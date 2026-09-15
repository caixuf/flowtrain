#pragma once

#include <cassert>
#include <cstdint>
#include <vector>

namespace flowtrain {

// Contiguous row-major tensor. Hot path is pointer + shape, not nested heap.
struct Tensor {
    int rows{0};
    int cols{0};
    std::vector<float> data;

    Tensor() = default;
    Tensor(int r, int c, float fill = 0.0f) : rows(r), cols(c), data(static_cast<size_t>(r) * static_cast<size_t>(c), fill) {}

    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
    int numel() const { return rows * cols; }

    float& at(int i, int j) { return data[static_cast<size_t>(i) * static_cast<size_t>(cols) + static_cast<size_t>(j)]; }
    float at(int i, int j) const { return data[static_cast<size_t>(i) * static_cast<size_t>(cols) + static_cast<size_t>(j)]; }
};

inline bool bit_equal(const Tensor& a, const Tensor& b) {
    if (a.rows != b.rows || a.cols != b.cols || a.data.size() != b.data.size()) return false;
    return a.data == b.data;
}

// C[M,N] = A[M,K] @ B[K,N]. Float MAC, fixed i-j-k order (cellular-style deterministic kernel).
inline void gemm(const Tensor& A, const Tensor& B, Tensor& C) {
    assert(A.cols == B.rows);
    assert(C.rows == A.rows && C.cols == B.cols);
    const int M = A.rows, N = B.cols, K = A.cols;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A.at(i, k) * B.at(k, j);
            }
            C.at(i, j) = acc;
        }
    }
}

inline Tensor gemm_new(const Tensor& A, const Tensor& B) {
    Tensor C(A.rows, B.cols);
    gemm(A, B, C);
    return C;
}

inline Tensor transpose(const Tensor& A) {
    Tensor T(A.cols, A.rows);
    for (int i = 0; i < A.rows; ++i)
        for (int j = 0; j < A.cols; ++j)
            T.at(j, i) = A.at(i, j);
    return T;
}

inline void axpy(Tensor& w, const Tensor& g, float lr) {
    assert(w.numel() == g.numel());
    for (int i = 0; i < w.numel(); ++i) w.data[static_cast<size_t>(i)] -= lr * g.data[static_cast<size_t>(i)];
}

inline void fill_seed(Tensor& t, uint32_t seed) {
    // LCG, same as a tiny cellular reset: no std::mt19937 variance across platforms.
    uint32_t s = seed ? seed : 1u;
    for (float& x : t.data) {
        s = s * 1664525u + 1013904223u;
        x = static_cast<float>(s & 0xFFFFu) / 65536.0f - 0.5f;
    }
}

} // namespace flowtrain
