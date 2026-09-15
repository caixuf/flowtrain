#include "flowtrain/pipeline.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>

int main() {
    const int P = 4, M = 8;
    const auto gpipe = flowtrain::simulate_pipeline(flowtrain::compile_gpipe(P, M), 10, 1);
    const auto f1b = flowtrain::simulate_pipeline(flowtrain::compile_1f1b(P, M), 10, 1);
    const auto f1b2 = flowtrain::simulate_pipeline(flowtrain::compile_1f1b(P, M), 10, 1);

    assert(f1b.makespan == f1b2.makespan);
    assert(f1b.n_ops == gpipe.n_ops);
    assert(f1b.n_ops == P * M * 2);
    // 1F1B is Megatron's memory schedule: fewer live activations. Equal Tf/Tb
    // and small M do not give a free makespan win over GPipe — don't claim one.
    assert(f1b.peak_activations < gpipe.peak_activations);

    std::cout << "PASS pp 1f1b_act=" << f1b.peak_activations
              << " gpipe_act=" << gpipe.peak_activations
              << " 1f1b_span=" << f1b.makespan
              << " gpipe_span=" << gpipe.makespan
              << " 1f1b_bubble=" << f1b.bubble
              << " gpipe_bubble=" << gpipe.bubble << "\n";
    return 0;
}
