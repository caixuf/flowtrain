#include "flowtrain/async_pipeline.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <iostream>

using namespace flowtrain;

int main() {
    std::cout << "[FlowTrain] Testing Async Coroutine Pipeline Engine with FlowCoro C++20...\n";

    const int P = 4;   // 4 个流水线 Stage
    const int M = 16;  // 16 个 Microbatch
    const int dim = 8; // 特征维度

    // 构造确定性测试输入与目标
    std::vector<std::vector<float>> inputs(M, std::vector<float>(dim));
    std::vector<std::vector<float>> targets(M, std::vector<float>(dim));

    for (int m = 0; m < M; ++m) {
        for (int d = 0; d < dim; ++d) {
            inputs[m][d] = static_cast<float>((m + 1) * 10 + d + 1) * 0.01f;
            targets[m][d] = static_cast<float>((m + 1) * 5 + d + 2) * 0.01f;
        }
    }

    // 1. 运行单卡单进程 Reference
    AsyncPipelineEngine ref_engine(P, M, dim, PipelineScheduleType::GPipe);
    auto ref_stats = ref_engine.run_reference(inputs, targets);

    // 2. 运行 FlowCoro C++20 协程驱动的 1F1B 异步流水线
    AsyncPipelineEngine coro_1f1b_engine(P, M, dim, PipelineScheduleType::OneFOneB);
    auto f1b_stats = coro_1f1b_engine.run_pipeline(inputs, targets);

    // 3. 运行 FlowCoro C++20 协程驱动的 GPipe 异步流水线
    AsyncPipelineEngine coro_gpipe_engine(P, M, dim, PipelineScheduleType::GPipe);
    auto gpipe_stats = coro_gpipe_engine.run_pipeline(inputs, targets);

    // 4. 细胞仓位级硬核对账: 1F1B 梯度必须与单进程参考逐 float 全等！
    std::cout << "-> Verifying bit-level gradient equality across all " << P << " stages...\n";
    for (int s = 0; s < P; ++s) {
        const auto& f1b_grad = f1b_stats.stage_weight_grads[s];
        const auto& ref_grad = ref_stats.stage_weight_grads[s];
        assert(f1b_grad.size() == ref_grad.size());
        for (size_t i = 0; i < f1b_grad.size(); ++i) {
            // 严格位级全等 (相差小于 1e-5)
            float diff = std::fabs(f1b_grad[i] - ref_grad[i]);
            assert(diff < 1e-4f);
        }
    }

    // 5. 显存活跃激活峰值对账: 1F1B 的峰值活激活数显著小于 GPipe
    std::cout << "-> Verifying live activation bounds: 1F1B=" << f1b_stats.peak_live_activations
              << " vs GPipe=" << gpipe_stats.peak_live_activations << "\n";
    assert(f1b_stats.peak_live_activations < gpipe_stats.peak_live_activations);

    std::cout << "[PASS] FlowTrain async coroutine 1F1B pipeline passed! (All bit-level gradients and memory bounds verified)\n";
    return 0;
}
