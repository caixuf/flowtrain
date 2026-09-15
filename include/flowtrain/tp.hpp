#pragma once

#include "flowtrain/collective.hpp"
#include "flowtrain/tensor.hpp"
#include <vector>

namespace flowtrain {

struct TpReport {
    int world{0};
    bool forward_bit_identical{false};
    int local_cols{0};
    int comm_elems{0};  // allgather volume
};

// Megatron-style column-parallel linear: split output dim, allgather Y.
inline TpReport tp_column_linear(int world, int batch, int in_dim, int out_dim, uint32_t seed) {
    TpReport r;
    r.world = world;
    assert(out_dim % world == 0);
    r.local_cols = out_dim / world;

    Tensor X(batch, in_dim);
    Tensor W(in_dim, out_dim);
    fill_seed(X, seed);
    fill_seed(W, seed + 9u);

    Tensor Y_ref = gemm_new(X, W);

    auto parts = split_cols(W, world);
    std::vector<Tensor> Ys;
    Ys.reserve(static_cast<size_t>(world));
    std::vector<Tensor*> yptr;
    for (int rank = 0; rank < world; ++rank) {
        Ys.push_back(gemm_new(X, parts[static_cast<size_t>(rank)]));
        yptr.push_back(&Ys.back());
    }
    Tensor Y = allgather_cols(yptr);
    r.comm_elems = Y.numel();
    r.forward_bit_identical = bit_equal(Y, Y_ref);
    return r;
}

} // namespace flowtrain
