#pragma once

#include "flowtrain/collective.hpp"
#include "flowtrain/tensor.hpp"
#include <vector>

namespace flowtrain {

struct DpReport {
    int world{0};
    int steps{0};
    bool weight_bit_identical{false};
    bool matches_reference{false};
};

// Linear y = x @ W, MSE vs target, mean over global batch.
// Each rank owns a contiguous shard of rows. Allreduce sums dW in rank order,
// then SGD. Reference folds the same shards in the same order — bit identity
// is the cellular discipline applied to DP.
inline DpReport train_dp_linear(int world, int batch_per_rank, int in_dim, int out_dim,
                                int steps, float lr, uint32_t seed) {
    DpReport r;
    r.world = world;
    r.steps = steps;
    const int B = batch_per_rank;

    std::vector<Tensor> W(static_cast<size_t>(world));
    std::vector<Tensor> X(static_cast<size_t>(world));
    std::vector<Tensor> Tgt(static_cast<size_t>(world));
    for (int rank = 0; rank < world; ++rank) {
        W[static_cast<size_t>(rank)] = Tensor(in_dim, out_dim);
        fill_seed(W[static_cast<size_t>(rank)], seed);
        X[static_cast<size_t>(rank)] = Tensor(B, in_dim);
        fill_seed(X[static_cast<size_t>(rank)], seed + 17u * static_cast<uint32_t>(rank + 1));
        Tgt[static_cast<size_t>(rank)] = Tensor(B, out_dim);
        fill_seed(Tgt[static_cast<size_t>(rank)], seed + 101u * static_cast<uint32_t>(rank + 1));
    }

    Tensor W_ref = W[0];

    for (int step = 0; step < steps; ++step) {
        std::vector<Tensor> dW(static_cast<size_t>(world));
        for (int rank = 0; rank < world; ++rank) {
            Tensor Y = gemm_new(X[static_cast<size_t>(rank)], W[static_cast<size_t>(rank)]);
            Tensor dY(B, out_dim);
            const float scale = 2.0f / static_cast<float>(world * B);
            for (int i = 0; i < dY.numel(); ++i) {
                dY.data[static_cast<size_t>(i)] =
                    scale * (Y.data[static_cast<size_t>(i)] - Tgt[static_cast<size_t>(rank)].data[static_cast<size_t>(i)]);
            }
            dW[static_cast<size_t>(rank)] = gemm_new(transpose(X[static_cast<size_t>(rank)]), dY);
        }

        Tensor dW_ref(in_dim, out_dim, 0.0f);
        for (int rank = 0; rank < world; ++rank) {
            for (int i = 0; i < dW_ref.numel(); ++i) {
                dW_ref.data[static_cast<size_t>(i)] += dW[static_cast<size_t>(rank)].data[static_cast<size_t>(i)];
            }
        }

        std::vector<Tensor*> ptrs;
        ptrs.reserve(static_cast<size_t>(world));
        for (int rank = 0; rank < world; ++rank) ptrs.push_back(&dW[static_cast<size_t>(rank)]);
        allreduce_sum(ptrs);

        for (int rank = 0; rank < world; ++rank) axpy(W[static_cast<size_t>(rank)], dW[static_cast<size_t>(rank)], lr);
        axpy(W_ref, dW_ref, lr);
    }

    r.weight_bit_identical = true;
    for (int rank = 1; rank < world; ++rank) {
        if (!bit_equal(W[0], W[static_cast<size_t>(rank)])) r.weight_bit_identical = false;
    }
    r.matches_reference = bit_equal(W[0], W_ref);
    return r;
}

} // namespace flowtrain
