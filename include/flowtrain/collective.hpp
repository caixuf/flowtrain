#pragma once

#include "flowtrain/tensor.hpp"
#include <cassert>
#include <vector>

namespace flowtrain {

// In-process stand-in for NCCL. Reduction order is always rank 0..world-1
// so DP can bit-match a single-process reference that sums shards the same way.
inline void allreduce_sum(std::vector<Tensor*>& shards) {
    if (shards.empty()) return;
    const int n = shards[0]->numel();
    for (auto* t : shards) assert(t->numel() == n);
    for (int i = 0; i < n; ++i) {
        float acc = 0.0f;
        for (auto* t : shards) acc += t->data[static_cast<size_t>(i)];
        for (auto* t : shards) t->data[static_cast<size_t>(i)] = acc;
    }
}

// Column-concat allgather: each rank holds [rows, local_cols], result is [rows, sum cols].
inline Tensor allgather_cols(const std::vector<Tensor*>& parts) {
    assert(!parts.empty());
    const int rows = parts[0]->rows;
    int cols = 0;
    for (auto* p : parts) {
        assert(p->rows == rows);
        cols += p->cols;
    }
    Tensor out(rows, cols);
    int off = 0;
    for (auto* p : parts) {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < p->cols; ++j)
                out.at(i, off + j) = p->at(i, j);
        off += p->cols;
    }
    return out;
}

inline std::vector<Tensor> split_cols(const Tensor& W, int world) {
    assert(world > 0 && W.cols % world == 0);
    const int local = W.cols / world;
    std::vector<Tensor> parts;
    parts.reserve(static_cast<size_t>(world));
    for (int r = 0; r < world; ++r) {
        Tensor p(W.rows, local);
        for (int i = 0; i < W.rows; ++i)
            for (int j = 0; j < local; ++j)
                p.at(i, j) = W.at(i, r * local + j);
        parts.push_back(std::move(p));
    }
    return parts;
}

} // namespace flowtrain
