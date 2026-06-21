#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/moe/engine.h"

namespace sparkinfer {

// Qwen3.5-35B-A3B architecture.
//   40 layers, hidden 2048, 16 Q / 2 KV heads (8:1 GQA), head_dim 128,
//   256 routed experts (top-8) + 1 shared expert, moe ffn 512,
//   RoPE + per-head QK-norm, RMSNorm, SwiGLU.
struct Qwen35Config {
    int   vocab       = 151936;
    int   hidden      = 2048;
    int   n_layers    = 40;
    int   n_q_heads   = 16;
    int   n_kv_heads  = 2;
    int   head_dim    = 128;
    int   n_experts   = 256;
    int   top_k       = 8;
    int   n_shared    = 1;
    int   moe_ffn     = 512;
    float rope_theta  = 1000000.f;
    float rms_eps     = 1e-6f;
    int   max_seq     = 4096;   // KV-cache cap for a sequence
    int   eos_id      = 151645;
};

// Device (bf16) weight pointers for one layer.
struct Qwen35LayerWeights {
    const void* input_norm   = nullptr;  // [hidden]
    const void* wq = nullptr;            // [hidden, n_q_heads*head_dim]
    const void* wk = nullptr;            // [hidden, n_kv_heads*head_dim]
    const void* wv = nullptr;            // [hidden, n_kv_heads*head_dim]
    const void* wo = nullptr;            // [n_q_heads*head_dim, hidden]
    const void* q_norm = nullptr;        // [head_dim]
    const void* k_norm = nullptr;        // [head_dim]
    const void* post_attn_norm = nullptr;// [hidden]
    const void* router_w = nullptr;      // [hidden, n_experts]
    const void* gate = nullptr;          // [n_experts, hidden, moe_ffn]
    const void* up   = nullptr;          // [n_experts, hidden, moe_ffn]
    const void* down = nullptr;          // [n_experts, moe_ffn, hidden]
    const void* shared_gate = nullptr;   // [hidden, moe_ffn]
    const void* shared_up   = nullptr;   // [hidden, moe_ffn]
    const void* shared_down = nullptr;   // [moe_ffn, hidden]
};

struct Qwen35Weights {
    const void* embed_tokens = nullptr;  // [vocab, hidden]
    const void* final_norm   = nullptr;  // [hidden]
    const void* lm_head      = nullptr;  // [hidden, vocab]  (pre-transposed)
    std::vector<Qwen35LayerWeights> layers;
};

// Single-sequence (batch=1) greedy decoder for Qwen3.5. Owns scratch buffers and
// drives embed -> N layers -> final norm -> LM head -> argmax per token.
class Qwen35Model {
public:
    Qwen35Model(const Qwen35Config& cfg, KVCacheManager* kv, moe::MoEEngine* engine);
    ~Qwen35Model();

    void set_weights(const Qwen35Weights& w);

    // Load weights from a sparkinfer weight directory (see tools/convert_qwen35.py).
    // Returns false on failure. Allocates device buffers it owns.
    bool load_weights(const std::string& dir);

    // Greedy generate: prompt token ids -> generated token ids (host).
    std::vector<int> generate(const std::vector<int>& prompt_ids, int max_new_tokens);

    // Run one token at `position`, return the argmax next-token id.
    int forward_token(int token_id, int position);

    const Qwen35Config& config() const;

private:
    struct Impl;
    Impl* p_;
};

} // namespace sparkinfer
