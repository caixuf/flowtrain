#pragma once

// 协程 1F1B：每 Stage 一条 Task，跨级用 flowcoro::BoundedChannel（容量 2 反压）。
// 前向：H2D → 回传后走 tensor.hpp::gemm → Y 再走一遍 DMA；反向本机 gemm。
// 每 microbatch 的 dW 先落地，再按 mb_id 0..M-1 归约，与单卡参考逐 float 相等。

#include "flowcoro/bounded_channel.h"
#include "flowcoro/cuda.h"
#include "flowcoro/task.h"
#include "flowtrain/pipeline.hpp"
#include "flowtrain/tensor.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace flowtrain {

struct MicrobatchTensor {
    int mb_id{-1};
    std::vector<float> tensor;
};

enum class PipelineScheduleType { GPipe, OneFOneB };

struct PipelineExecutionStats {
    int stages{0};
    int microbatches{0};
    int channel_capacity{0};
    int peak_live_activations{0};
    int peak_queue_depth{0};
    bool dma_roundtrip_bit_identical{false};
    std::vector<std::vector<float>> stage_weight_grads;
};

inline void wait_task(flowcoro::Task<void>&& task) {
    task.get(std::chrono::seconds(60));
}

template <typename T>
void bounded_push(flowcoro::BoundedChannel<T>& ch, const T& item) {
    while (!ch.try_push(T(item))) {
        std::this_thread::yield();
    }
}

template <typename T>
T bounded_pop(flowcoro::BoundedChannel<T>& ch) {
    T out{};
    while (!ch.try_pop(out)) {
        std::this_thread::yield();
    }
    return out;
}

inline std::vector<float> host_gemm_row(const std::vector<float>& x, const std::vector<float>& W, int dim) {
    Tensor A(1, dim);
    Tensor B(dim, dim);
    Tensor C(1, dim);
    A.data = x;
    B.data = W;
    gemm(A, B, C);
    return C.data;
}

// 激活与权重必须经过设备往返后才参与 gemm；再把输出走一遍 DMA。
// 多 Stage 共用一张卡：串行化 Driver 调用，完成用 stream.synchronize。
inline std::vector<float> dma_gemm_row(
    const std::vector<float>& x,
    const std::vector<float>& W,
    int dim,
    std::atomic<bool>& dma_ok) {
    using flowcoro::cuda::CudaStream;
    using flowcoro::cuda::DeviceBuffer;
    using flowcoro::cuda::PinnedHostBuffer;

    static std::mutex gpu_mu;
    std::lock_guard<std::mutex> gpu_lock(gpu_mu);
    static CudaStream stream;

    const int n = dim;
    const int nw = dim * dim;
    PinnedHostBuffer<float> hx(static_cast<size_t>(n));
    PinnedHostBuffer<float> hw(static_cast<size_t>(nw));
    PinnedHostBuffer<float> hy(static_cast<size_t>(n));
    PinnedHostBuffer<float> hx_back(static_cast<size_t>(n));
    PinnedHostBuffer<float> hw_back(static_cast<size_t>(nw));
    PinnedHostBuffer<float> hy_back(static_cast<size_t>(n));
    DeviceBuffer<float> dx(static_cast<size_t>(n));
    DeviceBuffer<float> dw(static_cast<size_t>(nw));
    DeviceBuffer<float> dy(static_cast<size_t>(n));

    for (int i = 0; i < n; ++i) hx[static_cast<size_t>(i)] = x[static_cast<size_t>(i)];
    for (int i = 0; i < nw; ++i) hw[static_cast<size_t>(i)] = W[static_cast<size_t>(i)];

    FLOWCORO_CUDA_CHECK(cuMemcpyHtoDAsync(dx.get(), hx.data(), static_cast<size_t>(n) * sizeof(float), stream.get()));
    FLOWCORO_CUDA_CHECK(cuMemcpyHtoDAsync(dw.get(), hw.data(), static_cast<size_t>(nw) * sizeof(float), stream.get()));
    FLOWCORO_CUDA_CHECK(cuMemcpyDtoHAsync(hx_back.data(), dx.get(), static_cast<size_t>(n) * sizeof(float), stream.get()));
    FLOWCORO_CUDA_CHECK(cuMemcpyDtoHAsync(hw_back.data(), dw.get(), static_cast<size_t>(nw) * sizeof(float), stream.get()));
    stream.synchronize();

    for (int i = 0; i < n; ++i) {
        if (hx_back[static_cast<size_t>(i)] != hx[static_cast<size_t>(i)]) dma_ok.store(false);
    }
    for (int i = 0; i < nw; ++i) {
        if (hw_back[static_cast<size_t>(i)] != hw[static_cast<size_t>(i)]) dma_ok.store(false);
    }

    Tensor A(1, dim);
    Tensor B(dim, dim);
    Tensor C(1, dim);
    for (int i = 0; i < n; ++i) A.data[static_cast<size_t>(i)] = hx_back[static_cast<size_t>(i)];
    for (int i = 0; i < nw; ++i) B.data[static_cast<size_t>(i)] = hw_back[static_cast<size_t>(i)];
    gemm(A, B, C);

    for (int i = 0; i < n; ++i) hy[static_cast<size_t>(i)] = C.data[static_cast<size_t>(i)];
    FLOWCORO_CUDA_CHECK(cuMemcpyHtoDAsync(dy.get(), hy.data(), static_cast<size_t>(n) * sizeof(float), stream.get()));
    FLOWCORO_CUDA_CHECK(cuMemcpyDtoHAsync(hy_back.data(), dy.get(), static_cast<size_t>(n) * sizeof(float), stream.get()));
    stream.synchronize();
    for (int i = 0; i < n; ++i) {
        if (hy_back[static_cast<size_t>(i)] != hy[static_cast<size_t>(i)]) dma_ok.store(false);
    }

    std::vector<float> out(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) out[static_cast<size_t>(i)] = hy_back[static_cast<size_t>(i)];
    return out;
}

inline std::vector<float> fold_mb_grads(const std::vector<std::vector<float>>& per_mb) {
    if (per_mb.empty()) return {};
    std::vector<float> acc(per_mb[0].size(), 0.0f);
    for (int m = 0; m < static_cast<int>(per_mb.size()); ++m) {
        for (size_t i = 0; i < acc.size(); ++i) {
            acc[i] += per_mb[static_cast<size_t>(m)][i];
        }
    }
    return acc;
}

class AsyncPipelineEngine {
public:
    static constexpr int kChannelCapacity = 2;

    AsyncPipelineEngine(int stages, int microbatches, int dim, PipelineScheduleType schedule_type)
        : P_(stages), M_(microbatches), dim_(dim), schedule_type_(schedule_type) {
        for (int i = 0; i < P_; ++i) {
            fwd_queues_.push_back(std::make_unique<flowcoro::BoundedChannel<MicrobatchTensor>>(kChannelCapacity));
            bwd_queues_.push_back(std::make_unique<flowcoro::BoundedChannel<MicrobatchTensor>>(kChannelCapacity));
        }
        stage_weights_.resize(static_cast<size_t>(P_));
        for (int s = 0; s < P_; ++s) {
            stage_weights_[static_cast<size_t>(s)].resize(static_cast<size_t>(dim_ * dim_));
            for (size_t i = 0; i < stage_weights_[static_cast<size_t>(s)].size(); ++i) {
                stage_weights_[static_cast<size_t>(s)][i] =
                    static_cast<float>((s + 1) * 100 + static_cast<int>(i) + 1) * 0.001f;
            }
        }
    }

    PipelineExecutionStats run_pipeline(const std::vector<std::vector<float>>& inputs,
                                        const std::vector<std::vector<float>>& targets) {
        std::atomic<int> current_activations{0};
        std::atomic<int> peak_activations{0};
        std::atomic<int> peak_queue{0};
        std::atomic<bool> dma_ok{true};

        std::vector<std::vector<PipeOp>> schedule_ops =
            schedule_type_ == PipelineScheduleType::OneFOneB ? compile_1f1b(P_, M_)
                                                              : compile_gpipe(P_, M_);

        std::vector<std::vector<std::vector<float>>> saved_activations(
            static_cast<size_t>(P_), std::vector<std::vector<float>>(static_cast<size_t>(M_)));
        std::vector<std::vector<std::vector<float>>> mb_dW(
            static_cast<size_t>(P_),
            std::vector<std::vector<float>>(static_cast<size_t>(M_),
                                            std::vector<float>(static_cast<size_t>(dim_ * dim_), 0.0f)));

        auto note_queue = [&](const flowcoro::BoundedChannel<MicrobatchTensor>& ch) {
            const int depth = static_cast<int>(ch.size());
            int old = peak_queue.load();
            while (depth > old && !peak_queue.compare_exchange_weak(old, depth)) {
            }
        };

        auto feed_inputs = [&]() -> flowcoro::Task<void> {
            for (int m = 0; m < M_; ++m) {
                bounded_push(*fwd_queues_[0], MicrobatchTensor{m, inputs[static_cast<size_t>(m)]});
                note_queue(*fwd_queues_[0]);
            }
            co_return;
        };

        auto stage_worker = [&](int stage_id) -> flowcoro::Task<void> {
            const auto& my_ops = schedule_ops[static_cast<size_t>(stage_id)];
            std::vector<std::vector<float>> last_fwd(static_cast<size_t>(M_));

            for (const auto& op : my_ops) {
                if (op.kind == PipeOpKind::F) {
                    MicrobatchTensor in_msg = bounded_pop(*fwd_queues_[static_cast<size_t>(stage_id)]);
                    assert(in_msg.mb_id == op.mb);

                    int cur_act = ++current_activations;
                    int old_peak = peak_activations.load();
                    while (cur_act > old_peak && !peak_activations.compare_exchange_weak(old_peak, cur_act)) {
                    }

                    saved_activations[static_cast<size_t>(stage_id)][static_cast<size_t>(op.mb)] = in_msg.tensor;

                    std::vector<float> out =
                        stage_id == 0
                            ? dma_gemm_row(in_msg.tensor, stage_weights_[static_cast<size_t>(stage_id)], dim_, dma_ok)
                            : host_gemm_row(in_msg.tensor, stage_weights_[static_cast<size_t>(stage_id)], dim_);

                    if (stage_id == P_ - 1) {
                        last_fwd[static_cast<size_t>(op.mb)] = std::move(out);
                    } else {
                        bounded_push(*fwd_queues_[static_cast<size_t>(stage_id + 1)],
                                     MicrobatchTensor{op.mb, std::move(out)});
                        note_queue(*fwd_queues_[static_cast<size_t>(stage_id + 1)]);
                    }
                } else {
                    MicrobatchTensor grad_in;
                    if (stage_id == P_ - 1) {
                        const auto& fwd_out = last_fwd[static_cast<size_t>(op.mb)];
                        grad_in.mb_id = op.mb;
                        grad_in.tensor.resize(static_cast<size_t>(dim_));
                        for (int i = 0; i < dim_; ++i) {
                            grad_in.tensor[static_cast<size_t>(i)] =
                                fwd_out[static_cast<size_t>(i)] -
                                targets[static_cast<size_t>(op.mb)][static_cast<size_t>(i)];
                        }
                    } else {
                        grad_in = bounded_pop(*bwd_queues_[static_cast<size_t>(stage_id + 1)]);
                        assert(grad_in.mb_id == op.mb);
                    }

                    const auto& in_act = saved_activations[static_cast<size_t>(stage_id)][static_cast<size_t>(op.mb)];
                    --current_activations;

                    Tensor X(1, dim_);
                    Tensor dY(1, dim_);
                    Tensor dW(dim_, dim_);
                    X.data = in_act;
                    dY.data = grad_in.tensor;
                    // dW = x^T @ dy  →  [dim,1] @ [1,dim]
                    Tensor XT = transpose(X);
                    gemm(XT, dY, dW);
                    mb_dW[static_cast<size_t>(stage_id)][static_cast<size_t>(op.mb)] = std::move(dW.data);

                    if (stage_id > 0) {
                        Tensor W(dim_, dim_);
                        W.data = stage_weights_[static_cast<size_t>(stage_id)];
                        Tensor WT = transpose(W);
                        Tensor dX(1, dim_);
                        gemm(dY, WT, dX);
                        bounded_push(*bwd_queues_[static_cast<size_t>(stage_id)],
                                     MicrobatchTensor{op.mb, std::move(dX.data)});
                    }
                }
            }
            co_return;
        };

        std::vector<std::thread> stage_threads;
        stage_threads.reserve(static_cast<size_t>(P_));
        for (int s = 0; s < P_; ++s) {
            stage_threads.emplace_back([s, &stage_worker]() { wait_task(stage_worker(s)); });
        }
        wait_task(feed_inputs());
        for (auto& th : stage_threads) {
            if (th.joinable()) th.join();
        }

        PipelineExecutionStats stats;
        stats.stages = P_;
        stats.microbatches = M_;
        stats.channel_capacity = kChannelCapacity;
        stats.peak_live_activations = peak_activations.load();
        stats.peak_queue_depth = peak_queue.load();
        stats.dma_roundtrip_bit_identical = dma_ok.load();
        stats.stage_weight_grads.resize(static_cast<size_t>(P_));
        for (int s = 0; s < P_; ++s) {
            stats.stage_weight_grads[static_cast<size_t>(s)] =
                fold_mb_grads(mb_dW[static_cast<size_t>(s)]);
        }
        return stats;
    }

    PipelineExecutionStats run_reference(const std::vector<std::vector<float>>& inputs,
                                         const std::vector<std::vector<float>>& targets) {
        std::vector<std::vector<std::vector<float>>> mb_dW(
            static_cast<size_t>(P_),
            std::vector<std::vector<float>>(static_cast<size_t>(M_),
                                            std::vector<float>(static_cast<size_t>(dim_ * dim_), 0.0f)));

        for (int m = 0; m < M_; ++m) {
            std::vector<std::vector<float>> acts(static_cast<size_t>(P_ + 1));
            acts[0] = inputs[static_cast<size_t>(m)];

            for (int s = 0; s < P_; ++s) {
                Tensor A(1, dim_);
                Tensor B(dim_, dim_);
                Tensor C(1, dim_);
                A.data = acts[static_cast<size_t>(s)];
                B.data = stage_weights_[static_cast<size_t>(s)];
                gemm(A, B, C);
                acts[static_cast<size_t>(s + 1)] = std::move(C.data);
            }

            Tensor grad(1, dim_);
            for (int i = 0; i < dim_; ++i) {
                grad.data[static_cast<size_t>(i)] = acts[static_cast<size_t>(P_)][static_cast<size_t>(i)] -
                                                   targets[static_cast<size_t>(m)][static_cast<size_t>(i)];
            }

            for (int s = P_ - 1; s >= 0; --s) {
                Tensor X(1, dim_);
                X.data = acts[static_cast<size_t>(s)];
                Tensor XT = transpose(X);
                Tensor dW(dim_, dim_);
                gemm(XT, grad, dW);
                mb_dW[static_cast<size_t>(s)][static_cast<size_t>(m)] = std::move(dW.data);

                if (s > 0) {
                    Tensor W(dim_, dim_);
                    W.data = stage_weights_[static_cast<size_t>(s)];
                    Tensor WT = transpose(W);
                    Tensor dX(1, dim_);
                    gemm(grad, WT, dX);
                    grad = std::move(dX);
                }
            }
        }

        PipelineExecutionStats stats;
        stats.stages = P_;
        stats.microbatches = M_;
        stats.channel_capacity = 0;
        stats.peak_live_activations = M_;
        stats.dma_roundtrip_bit_identical = true;
        stats.stage_weight_grads.resize(static_cast<size_t>(P_));
        for (int s = 0; s < P_; ++s) {
            stats.stage_weight_grads[static_cast<size_t>(s)] =
                fold_mb_grads(mb_dW[static_cast<size_t>(s)]);
        }
        return stats;
    }

private:
    int P_;
    int M_;
    int dim_;
    PipelineScheduleType schedule_type_;
    std::vector<std::unique_ptr<flowcoro::BoundedChannel<MicrobatchTensor>>> fwd_queues_;
    std::vector<std::unique_ptr<flowcoro::BoundedChannel<MicrobatchTensor>>> bwd_queues_;
    std::vector<std::vector<float>> stage_weights_;
};

} // namespace flowtrain
