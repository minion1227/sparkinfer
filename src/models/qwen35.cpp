// Qwen3.5-35B-A3B single-sequence greedy decoder.
//
// Per token: embed -> [40x Qwen layer] -> final RMSNorm -> LM head -> argmax.
// Qwen layer: RMSNorm -> Q/K/V -> per-head QK-norm -> RoPE -> KV append ->
//             GQA flash decode -> O-proj -> residual -> RMSNorm ->
//             routed top-8 MoE (+ shared expert) -> residual.
// All steps run on one stream; only the sampled id is copied to the host, which
// autoregressive greedy decoding fundamentally requires.

#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/kv_ops.h"
#include "sparkinfer/kernels/attention.h"
#include "sparkinfer/kernels/gemm.h"
#include "sparkinfer/kernels/fused.h"
#include "sparkinfer/kernels/moe.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

namespace sparkinfer {

namespace {
inline void cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fprintf(stderr, "[qwen35] %s: %s\n", what, cudaGetErrorString(e));
}
using bf16 = unsigned short;
}

struct Qwen35Model::Impl {
    Qwen35Config cfg;
    KVCacheManager* kv;
    moe::MoEEngine* engine;
    Qwen35Weights w;
    cudaStream_t stream{};
    uint64_t seq_id = 0;
    int qdim, kvdim;

    // scratch (bf16)
    bf16 *x, *xn, *q, *k, *v, *attn, *ao, *h, *hn, *routed, *shared;
    float* logits;
    int *d_tok, *d_out_id, *d_pos, *d_seqlen, *d_writepos, *d_shared_ids;
    float* d_shared_w;
    std::vector<void*> owned;   // device buffers from load_weights

    template <class T> T* alloc(size_t n) { void* p=nullptr; cu(cudaMalloc(&p, n*sizeof(T)), "malloc"); return (T*)p; }
};

Qwen35Model::Qwen35Model(const Qwen35Config& cfg, KVCacheManager* kv, moe::MoEEngine* engine)
    : p_(new Impl()) {
    p_->cfg = cfg; p_->kv = kv; p_->engine = engine;
    p_->qdim = cfg.n_q_heads * cfg.head_dim;
    p_->kvdim = cfg.n_kv_heads * cfg.head_dim;
    cudaStreamCreate(&p_->stream);
    const int H = cfg.hidden;
    p_->x=p_->alloc<bf16>(H); p_->xn=p_->alloc<bf16>(H);
    p_->q=p_->alloc<bf16>(p_->qdim); p_->k=p_->alloc<bf16>(p_->kvdim); p_->v=p_->alloc<bf16>(p_->kvdim);
    p_->attn=p_->alloc<bf16>(p_->qdim); p_->ao=p_->alloc<bf16>(H);
    p_->h=p_->alloc<bf16>(H); p_->hn=p_->alloc<bf16>(H);
    p_->routed=p_->alloc<bf16>(H); p_->shared=p_->alloc<bf16>(H);
    p_->logits=p_->alloc<float>(cfg.vocab);
    p_->d_tok=p_->alloc<int>(1); p_->d_out_id=p_->alloc<int>(1);
    p_->d_pos=p_->alloc<int>(1); p_->d_seqlen=p_->alloc<int>(1); p_->d_writepos=p_->alloc<int>(1);
    p_->d_shared_ids=p_->alloc<int>(1); p_->d_shared_w=p_->alloc<float>(1);
    int zero=0; float one=1.f;
    cu(cudaMemcpy(p_->d_shared_ids,&zero,sizeof(int),cudaMemcpyHostToDevice),"shared ids");
    cu(cudaMemcpy(p_->d_shared_w,&one,sizeof(float),cudaMemcpyHostToDevice),"shared w");
}

Qwen35Model::~Qwen35Model() {
    for (void* b : p_->owned) cudaFree(b);
    cudaFree(p_->x); cudaFree(p_->xn); cudaFree(p_->q); cudaFree(p_->k); cudaFree(p_->v);
    cudaFree(p_->attn); cudaFree(p_->ao); cudaFree(p_->h); cudaFree(p_->hn);
    cudaFree(p_->routed); cudaFree(p_->shared); cudaFree(p_->logits);
    cudaFree(p_->d_tok); cudaFree(p_->d_out_id); cudaFree(p_->d_pos);
    cudaFree(p_->d_seqlen); cudaFree(p_->d_writepos); cudaFree(p_->d_shared_ids); cudaFree(p_->d_shared_w);
    cudaStreamDestroy(p_->stream);
    delete p_;
}

void Qwen35Model::set_weights(const Qwen35Weights& w) { p_->w = w; }
const Qwen35Config& Qwen35Model::config() const { return p_->cfg; }

int Qwen35Model::forward_token(int token_id, int position) {
    Impl& s = *p_;
    const Qwen35Config& c = s.cfg;
    const int H = c.hidden;
    kernels::GemmConfig gc{};
    int seqlen = position + 1;
    cudaStream_t st = s.stream;

    cu(cudaMemcpyAsync(s.d_tok, &token_id, sizeof(int), cudaMemcpyHostToDevice, st), "tok");
    cu(cudaMemcpyAsync(s.d_pos, &position, sizeof(int), cudaMemcpyHostToDevice, st), "pos");
    cu(cudaMemcpyAsync(s.d_writepos, &position, sizeof(int), cudaMemcpyHostToDevice, st), "wpos");
    cu(cudaMemcpyAsync(s.d_seqlen, &seqlen, sizeof(int), cudaMemcpyHostToDevice, st), "slen");

    kernels::launch_embedding(s.d_tok, s.w.embed_tokens, s.x, 1, H, st);

    int* btable = s.kv->block_table(s.seq_id);
    for (int L = 0; L < c.n_layers; L++) {
        const Qwen35LayerWeights& w = s.w.layers[L];
        kernels::launch_rmsnorm(s.x, w.input_norm, s.xn, 1, H, c.rms_eps, st);
        kernels::launch_gemm(s.xn, w.wq, s.q, 1, s.qdim,  H, 1.f, 0.f, gc, st);
        kernels::launch_gemm(s.xn, w.wk, s.k, 1, s.kvdim, H, 1.f, 0.f, gc, st);
        kernels::launch_gemm(s.xn, w.wv, s.v, 1, s.kvdim, H, 1.f, 0.f, gc, st);
        // per-head QK-norm (rows = heads, cols = head_dim)
        kernels::launch_rmsnorm(s.q, w.q_norm, s.q, c.n_q_heads,  c.head_dim, c.rms_eps, st);
        kernels::launch_rmsnorm(s.k, w.k_norm, s.k, c.n_kv_heads, c.head_dim, c.rms_eps, st);
        kernels::launch_rope(s.q, s.k, s.d_pos, 1, c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_theta, st);

        bf16* kpool = (bf16*)s.kv->k_pool() + (size_t)L * s.kv->layer_stride_elems();
        bf16* vpool = (bf16*)s.kv->v_pool() + (size_t)L * s.kv->layer_stride_elems();
        launch_kv_append(kpool, vpool, s.k, s.v, btable, s.d_writepos, 1,
                         c.n_kv_heads, c.head_dim, s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
        kernels::launch_flash_decode_gqa8(s.q, kpool, vpool, btable, s.d_seqlen, s.attn,
                                          1, c.n_kv_heads, c.head_dim, s.kv->block_size(),
                                          s.kv->max_blocks_per_seq(), 1.f / sqrtf((float)c.head_dim), st);

        kernels::launch_gemm(s.attn, w.wo, s.ao, 1, H, s.qdim, 1.f, 0.f, gc, st);
        launch_residual_add(s.x, s.ao, s.h, H, st);
        kernels::launch_rmsnorm(s.h, w.post_attn_norm, s.hn, 1, H, c.rms_eps, st);

        s.engine->set_layer_weights(L, {w.router_w, w.gate, w.up, w.down});
        s.engine->forward(s.hn, s.routed, 1, L, st);
        if (c.n_shared > 0) {
            kernels::launch_moe_expert_ffn(s.hn, w.shared_gate, w.shared_up, w.shared_down,
                                           s.d_shared_ids, s.d_shared_w, s.shared,
                                           1, 1, 1, H, c.moe_ffn, st);
            launch_residual_add(s.routed, s.shared, s.routed, H, st);
        }
        launch_residual_add(s.h, s.routed, s.x, H, st);
    }

    kernels::launch_rmsnorm(s.x, s.w.final_norm, s.xn, 1, H, c.rms_eps, st);
    kernels::launch_linear_f32(s.xn, s.w.lm_head, s.logits, 1, c.vocab, H, st);
    kernels::launch_argmax(s.logits, s.d_out_id, 1, c.vocab, st);

    int out_id = 0;
    cu(cudaMemcpyAsync(&out_id, s.d_out_id, sizeof(int), cudaMemcpyDeviceToHost, st), "out_id");
    cu(cudaStreamSynchronize(st), "sync");
    return out_id;
}

std::vector<int> Qwen35Model::generate(const std::vector<int>& prompt, int max_new) {
    Impl& s = *p_;
    std::vector<int> out;
    if (prompt.empty()) return out;
    if (!s.kv->allocate(s.seq_id, s.cfg.max_seq)) {
        fprintf(stderr, "[qwen35] KV allocate failed (pool too small for max_seq=%d)\n", s.cfg.max_seq);
        return out;
    }
    int next = -1;
    for (size_t i = 0; i < prompt.size(); i++) next = forward_token(prompt[i], (int)i);
    for (int i = 0; i < max_new; i++) {
        out.push_back(next);
        if (next == s.cfg.eos_id) break;
        next = forward_token(next, (int)prompt.size() + i);
    }
    s.kv->free(s.seq_id);
    return out;
}

// ----- weight loading from a sparkinfer weight directory -----
namespace {
void* load_bin(const std::string& path, std::vector<void*>& owned) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "[qwen35] missing weight: %s\n", path.c_str()); return nullptr; }
    std::streamsize n = f.tellg(); f.seekg(0);
    std::vector<char> host(n);
    f.read(host.data(), n);
    void* d = nullptr;
    if (cudaMalloc(&d, n) != cudaSuccess) return nullptr;
    cudaMemcpy(d, host.data(), n, cudaMemcpyHostToDevice);
    owned.push_back(d);
    return d;
}
}

bool Qwen35Model::load_weights(const std::string& dir) {
    Impl& s = *p_;
    auto L = [&](const std::string& n) { return load_bin(dir + "/" + n + ".bin", s.owned); };
    s.w.embed_tokens = L("embed_tokens");
    s.w.final_norm   = L("final_norm");
    s.w.lm_head      = L("lm_head");
    if (!s.w.embed_tokens || !s.w.final_norm || !s.w.lm_head) return false;
    s.w.layers.resize(s.cfg.n_layers);
    for (int i = 0; i < s.cfg.n_layers; i++) {
        std::string pfx = "layer_" + std::to_string(i) + ".";
        Qwen35LayerWeights& w = s.w.layers[i];
        w.input_norm     = L(pfx + "input_norm");
        w.wq = L(pfx + "wq"); w.wk = L(pfx + "wk"); w.wv = L(pfx + "wv"); w.wo = L(pfx + "wo");
        w.q_norm = L(pfx + "q_norm"); w.k_norm = L(pfx + "k_norm");
        w.post_attn_norm = L(pfx + "post_attn_norm");
        w.router_w = L(pfx + "router_w");
        w.gate = L(pfx + "gate"); w.up = L(pfx + "up"); w.down = L(pfx + "down");
        if (s.cfg.n_shared > 0) {
            w.shared_gate = L(pfx + "shared_gate"); w.shared_up = L(pfx + "shared_up"); w.shared_down = L(pfx + "shared_down");
        }
        if (!w.wq || !w.gate || !w.router_w) return false;
    }
    return true;
}

} // namespace sparkinfer
