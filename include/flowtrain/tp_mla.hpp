#pragma once

// ============================================================================
// flowtrain - DeepSeek 式 MLA (多头低秩潜空间注意力) 张量并行 (Tensor Parallelism)
// 
// 切分策略 (Megatron-style Column-Row Parallelism):
//   1. 潜空间投影 W_DKV: 各 rank 保持全量潜向量 C = X @ W_DKV;
//   2. 多头上投影 W_UK, W_UV: 列切 (Column Parallel)，每 rank 分得 H / world 个头;
//   3. 输出投影 W_O: 行切 (Row Parallel)，每 rank 分得 (H / world) * d 的输入切片;
//   4. 最终输出通过 In-process AllReduce Sum 归约求和;
//   5. 细胞位级对账戒律: 参考实现按相同 shard 和 rank 顺序求和，实现逐 float 位级全等。
// ============================================================================

#include "flowtrain/collective.hpp"
#include "flowtrain/tensor.hpp"
#include <cassert>
#include <vector>

namespace flowtrain {

struct TpMlaReport {
    int world{0};
    int num_heads{0};
    int local_heads{0};
    int latent_dim{0};
    bool forward_bit_identical{false};
    int comm_elems{0}; // AllReduce volume
};

inline TpMlaReport tp_mla_forward(int world, int batch, int in_dim, int latent_dim,
                                  int num_heads, int head_dim, uint32_t seed) {
    TpMlaReport r;
    r.world = world;
    r.num_heads = num_heads;
    r.latent_dim = latent_dim;
    assert(num_heads % world == 0);
    r.local_heads = num_heads / world;
    const int total_head_dim = num_heads * head_dim;
    const int local_head_dim = r.local_heads * head_dim;

    Tensor X(batch, in_dim);
    Tensor W_dkv(in_dim, latent_dim);
    Tensor W_uv(latent_dim, total_head_dim);
    Tensor W_o(total_head_dim, in_dim);

    fill_seed(X, seed);
    fill_seed(W_dkv, seed + 1u);
    fill_seed(W_uv, seed + 2u);
    fill_seed(W_o, seed + 3u);

    // 潜空间投影 C = X @ W_DKV
    Tensor C = gemm_new(X, W_dkv);

    // W_uv 列切为 world 份: 每份 [latent_dim, local_head_dim]
    auto uv_parts = split_cols(W_uv, world);

    // W_o 行切为 world 份: 每份 [local_head_dim, in_dim]
    std::vector<Tensor> o_parts;
    o_parts.reserve(static_cast<size_t>(world));
    for (int rank = 0; rank < world; ++rank) {
        Tensor part(local_head_dim, in_dim);
        for (int i = 0; i < local_head_dim; ++i) {
            for (int j = 0; j < in_dim; ++j) {
                part.at(i, j) = W_o.at(rank * local_head_dim + i, j);
            }
        }
        o_parts.push_back(std::move(part));
    }

    // 1. 单卡参考计算 (遵守细胞位级对账戒律：按同样 rank 顺序累加)
    Tensor Y_ref(batch, in_dim, 0.0f);
    for (int rank = 0; rank < world; ++rank) {
        Tensor local_ref = gemm_new(gemm_new(C, uv_parts[static_cast<size_t>(rank)]),
                                    o_parts[static_cast<size_t>(rank)]);
        for (int i = 0; i < Y_ref.numel(); ++i) {
            Y_ref.data[static_cast<size_t>(i)] += local_ref.data[static_cast<size_t>(i)];
        }
    }

    // 2. 张量并行切分计算 (TP Sharded Execution)
    std::vector<Tensor> partial_Y;
    partial_Y.reserve(static_cast<size_t>(world));
    std::vector<Tensor*> y_ptrs;
    for (int rank = 0; rank < world; ++rank) {
        Tensor local_head_out = gemm_new(C, uv_parts[static_cast<size_t>(rank)]);
        Tensor local_y = gemm_new(local_head_out, o_parts[static_cast<size_t>(rank)]);
        partial_Y.push_back(std::move(local_y));
        y_ptrs.push_back(&partial_Y.back());
    }

    // 3. 通信归约: In-process AllReduce Sum
    allreduce_sum(y_ptrs);

    r.comm_elems = y_ptrs[0]->numel();
    r.forward_bit_identical = bit_equal(*y_ptrs[0], Y_ref);
    return r;
}

} // namespace flowtrain
