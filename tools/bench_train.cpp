#include "flowtrain/flowtrain.hpp"
#include <iomanip>
#include <iostream>

int main() {
    using namespace flowtrain;
    const auto dp = train_dp_linear(4, 8, 32, 16, 8, 0.02f, 2026);
    const auto tp = tp_column_linear(4, 16, 32, 32, 11);
    const auto gpipe = simulate_pipeline(compile_gpipe(4, 8), 10, 1);
    const auto f1b = simulate_pipeline(compile_1f1b(4, 8), 10, 1);

    std::cout << "flowtrain bench (CPU collectives, fake layers)\n";
    std::cout << std::left << std::setw(22) << "dp" << " bit_id=" << dp.weight_bit_identical
              << " vs_ref=" << dp.matches_reference << " world=" << dp.world << "\n";
    std::cout << std::left << std::setw(22) << "tp column" << " bit_id=" << tp.forward_bit_identical
              << " comm_elems=" << tp.comm_elems << "\n";
    std::cout << std::left << std::setw(22) << "pp gpipe" << " span=" << gpipe.makespan
              << " util=" << std::fixed << std::setprecision(3) << gpipe.utilization
              << " bubble=" << gpipe.bubble
              << " peak_act=" << gpipe.peak_activations << "\n";
    std::cout << std::left << std::setw(22) << "pp 1f1b" << " span=" << f1b.makespan
              << " util=" << f1b.utilization << " bubble=" << f1b.bubble
              << " peak_act=" << f1b.peak_activations << "\n";
    std::cout << "not Megatron, not NCCL, not a real LLM trainer.\n";
    return 0;
}
