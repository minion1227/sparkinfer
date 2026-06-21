// Run a real Qwen3-MoE model from GGUF-converted weights (see tools/convert_gguf.py).
// Reads arch from <weight_dir>/config.txt, loads bf16 weights, greedily generates
// from prompt token ids. Tokenize/detokenize with tools/run_qwen3.py.
//
// Usage: qwen3_gguf_generate <weight_dir> <max_new> <id0> <id1> ...

#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <fstream>
#include <unordered_map>
#include <vector>

static std::unordered_map<std::string, std::string> read_config(const std::string& path) {
    std::unordered_map<std::string, std::string> m;
    std::ifstream f(path); std::string line;
    while (std::getline(f, line)) {
        auto eq = line.find('=');
        if (eq != std::string::npos) m[line.substr(0, eq)] = line.substr(eq + 1);
    }
    return m;
}

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: %s <weight_dir> <max_new> <id0> [id1 ...]\n", argv[0]); return 2; }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("[SKIP] no GPU\n"); return 0; }

    const std::string dir = argv[1];
    const int max_new = atoi(argv[2]);
    std::vector<int> prompt;
    for (int i = 3; i < argc; i++) prompt.push_back(atoi(argv[i]));

    auto cf = read_config(dir + "/config.txt");
    if (cf.empty()) { printf("[FAIL] no config.txt in %s\n", dir.c_str()); return 1; }
    auto gi = [&](const char* k, int d){ auto it=cf.find(k); return it==cf.end()?d:atoi(it->second.c_str()); };
    auto gf = [&](const char* k, float d){ auto it=cf.find(k); return it==cf.end()?d:(float)atof(it->second.c_str()); };

    sparkinfer::Qwen35Config cfg;
    cfg.vocab=gi("vocab",151936); cfg.hidden=gi("hidden",2048); cfg.n_layers=gi("n_layers",48);
    cfg.n_q_heads=gi("n_q_heads",32); cfg.n_kv_heads=gi("n_kv_heads",4); cfg.head_dim=gi("head_dim",128);
    cfg.n_experts=gi("n_experts",128); cfg.top_k=gi("top_k",8); cfg.n_shared=gi("n_shared",0);
    cfg.moe_ffn=gi("moe_ffn",768); cfg.rope_theta=gf("rope_theta",1e6f); cfg.rms_eps=gf("rms_eps",1e-6f);
    cfg.eos_id=gi("eos_id",151645); cfg.max_seq=2048;

    auto rt = sparkinfer::Runtime::create({}); rt->initialize();

    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers=cfg.n_layers; kvc.num_kv_heads=cfg.n_kv_heads; kvc.head_dim=cfg.head_dim; kvc.block_size=16;
    const size_t elems_per_block=(size_t)16*cfg.n_kv_heads*cfg.head_dim;
    const size_t blocks_needed=(cfg.max_seq+15)/16 + 8;
    const size_t pool_bytes=(size_t)cfg.n_layers*2*elems_per_block*2*blocks_needed;
    sparkinfer::KVCacheManager kv(kvc, pool_bytes);

    sparkinfer::moe::MoEConfig mc;
    mc.num_experts=cfg.n_experts; mc.top_k=cfg.top_k; mc.hidden_dim=cfg.hidden;
    mc.ffn_dim=cfg.moe_ffn; mc.num_layers=cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);

    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    printf("loading weights from %s ...\n", dir.c_str());
    if (!model.load_weights(dir)) { printf("[FAIL] load_weights\n"); return 1; }
    printf("loaded. generating %d tokens from %zu prompt tokens\n", max_new, prompt.size());

    auto out = model.generate(prompt, max_new);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("[FAIL] cuda: %s\n", cudaGetErrorString(e)); return 1; }

    printf("OUTPUT_IDS:");
    for (int id : out) printf(" %d", id);
    printf("\n");
    return 0;
}
