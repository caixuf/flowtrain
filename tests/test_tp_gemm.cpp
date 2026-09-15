#include "flowtrain/tp.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>

int main() {
    const auto r = flowtrain::tp_column_linear(4, 12, 16, 16, 7);
    assert(r.forward_bit_identical);
    assert(r.local_cols == 4);
    std::cout << "PASS tp column-parallel gemm comm_elems=" << r.comm_elems << "\n";
    return 0;
}
