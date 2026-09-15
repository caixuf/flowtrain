#include "flowtrain/async_pipeline.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>

using namespace flowtrain;

int main() {
    const int P = 4;
    const int M = 16;
    const int dim = 8;

    std::vector<std::vector<float>> inputs(static_cast<size_t>(M), std::vector<float>(static_cast<size_t>(dim)));
    std::vector<std::vector<float>> targets(static_cast<size_t>(M), std::vector<float>(static_cast<size_t>(dim)));
    for (int m = 0; m < M; ++m) {
        for (int d = 0; d < dim; ++d) {
            inputs[static_cast<size_t>(m)][static_cast<size_t>(d)] =
                static_cast<float>((m + 1) * 10 + d + 1) * 0.01f;
            targets[static_cast<size_t>(m)][static_cast<size_t>(d)] =
                static_cast<float>((m + 1) * 5 + d + 2) * 0.01f;
        }
    }

    AsyncPipelineEngine ref_engine(P, M, dim, PipelineScheduleType::GPipe);
    auto ref_stats = ref_engine.run_reference(inputs, targets);

    AsyncPipelineEngine coro_1f1b(P, M, dim, PipelineScheduleType::OneFOneB);
    auto f1b = coro_1f1b.run_pipeline(inputs, targets);

    AsyncPipelineEngine coro_gpipe(P, M, dim, PipelineScheduleType::GPipe);
    auto gpipe = coro_gpipe.run_pipeline(inputs, targets);

    assert(f1b.dma_roundtrip_bit_identical);
    assert(gpipe.dma_roundtrip_bit_identical);
    assert(f1b.peak_queue_depth <= f1b.channel_capacity);
    assert(gpipe.peak_queue_depth <= gpipe.channel_capacity);

    for (int s = 0; s < P; ++s) {
        assert(f1b.stage_weight_grads[static_cast<size_t>(s)] ==
               ref_stats.stage_weight_grads[static_cast<size_t>(s)]);
        assert(gpipe.stage_weight_grads[static_cast<size_t>(s)] ==
               ref_stats.stage_weight_grads[static_cast<size_t>(s)]);
    }

    assert(f1b.peak_live_activations < gpipe.peak_live_activations);

    std::cout << "PASS async_pipeline 1f1b_act=" << f1b.peak_live_activations
              << " gpipe_act=" << gpipe.peak_live_activations
              << " qpeak=" << f1b.peak_queue_depth
              << "/" << f1b.channel_capacity
              << " dma_bit=1 grads_bit=1\n";
    return 0;
}
