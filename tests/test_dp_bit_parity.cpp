#include "flowtrain/dp.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>

int main() {
    const auto r = flowtrain::train_dp_linear(4, 8, 16, 8, 5, 0.05f, 2026);
    assert(r.weight_bit_identical);
    assert(r.matches_reference);
    std::cout << "PASS dp bit-identical world=" << r.world << " steps=" << r.steps << "\n";
    return 0;
}
