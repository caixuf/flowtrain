#include "flowtrain/tp_mla.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>

int main() {
    // world=4 ranks, batch=8, in_dim=16, latent_dim=4, num_heads=4, head_dim=4, seed=42
    const auto r = flowtrain::tp_mla_forward(4, 8, 16, 4, 4, 4, 42);
    assert(r.forward_bit_identical);
    assert(r.local_heads == 1);
    assert(r.comm_elems == 8 * 16);
    std::cout << "PASS tp MLA column-row parallel forward bit_id=1 comm_elems=" << r.comm_elems << "\n";
    return 0;
}
