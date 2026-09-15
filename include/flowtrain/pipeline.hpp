#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

namespace flowtrain {

enum class PipeOpKind : uint8_t { F = 0, B = 1 };

struct PipeOp {
    int stage{0};
    int mb{0};
    PipeOpKind kind{PipeOpKind::F};
};

struct TimedPipeOp {
    PipeOp op;
    int start{0};
    int end{0};
};

struct PipeReport {
    int stages{0};
    int microbatches{0};
    int makespan{0};
    int busy{0};
    double utilization{0};
    double bubble{0};
    int n_ops{0};
    int peak_activations{0};
    std::vector<TimedPipeOp> trace;
};

// Compile a per-stage program. Kahn-style: the sequence is the topology;
// simulate() is the only clock. Borrowed from cellular graph compile-then-run.
inline std::vector<std::vector<PipeOp>> compile_gpipe(int stages, int microbatches) {
    std::vector<std::vector<PipeOp>> seq(static_cast<size_t>(stages));
    for (int s = 0; s < stages; ++s) {
        for (int m = 0; m < microbatches; ++m) seq[static_cast<size_t>(s)].push_back({s, m, PipeOpKind::F});
        for (int m = 0; m < microbatches; ++m) seq[static_cast<size_t>(s)].push_back({s, m, PipeOpKind::B});
    }
    return seq;
}

// Megatron 1F1B: warmup = P - rank - 1 forwards, then (F,B) pairs, then cooldown B.
inline std::vector<std::vector<PipeOp>> compile_1f1b(int stages, int microbatches) {
    std::vector<std::vector<PipeOp>> seq(static_cast<size_t>(stages));
    for (int s = 0; s < stages; ++s) {
        const int warmup = stages - s - 1;
        const int warm = std::min(warmup, microbatches);
        for (int i = 0; i < warm; ++i) seq[static_cast<size_t>(s)].push_back({s, i, PipeOpKind::F});
        const int rest = microbatches - warm;
        for (int i = 0; i < rest; ++i) {
            seq[static_cast<size_t>(s)].push_back({s, warm + i, PipeOpKind::F});
            seq[static_cast<size_t>(s)].push_back({s, i, PipeOpKind::B});
        }
        for (int i = 0; i < warm; ++i) {
            seq[static_cast<size_t>(s)].push_back({s, rest + i, PipeOpKind::B});
        }
    }
    return seq;
}

inline PipeReport simulate_pipeline(const std::vector<std::vector<PipeOp>>& seq, int compute = 1, int comm = 0) {
    const int P = static_cast<int>(seq.size());
    assert(P > 0);
    int M = 0;
    for (const auto& st : seq)
        for (const auto& op : st) M = std::max(M, op.mb + 1);

    std::vector<int> idx(static_cast<size_t>(P), 0);
    std::vector<int> free_at(static_cast<size_t>(P), 0);
    std::vector<std::vector<int>> f_end(static_cast<size_t>(P), std::vector<int>(static_cast<size_t>(M), -1));
    std::vector<std::vector<int>> b_end(static_cast<size_t>(P), std::vector<int>(static_cast<size_t>(M), -1));

    auto ready = [&](const PipeOp& op) -> int {
        int t = free_at[static_cast<size_t>(op.stage)];
        if (op.kind == PipeOpKind::F) {
            if (op.stage > 0) {
                const int dep = f_end[static_cast<size_t>(op.stage - 1)][static_cast<size_t>(op.mb)];
                if (dep < 0) return -1;
                t = std::max(t, dep + comm);
            }
        } else {
            const int fdep = f_end[static_cast<size_t>(op.stage)][static_cast<size_t>(op.mb)];
            if (fdep < 0) return -1;
            t = std::max(t, fdep);
            if (op.stage + 1 < P) {
                const int bdep = b_end[static_cast<size_t>(op.stage + 1)][static_cast<size_t>(op.mb)];
                if (bdep < 0) return -1;
                t = std::max(t, bdep + comm);
            }
        }
        return t;
    };

    int n_ops = 0;
    for (const auto& st : seq) n_ops += static_cast<int>(st.size());
    int done = 0;
    std::vector<TimedPipeOp> trace;
    trace.reserve(static_cast<size_t>(n_ops));
    while (done < n_ops) {
        int best_s = -1;
        int best_t = std::numeric_limits<int>::max();
        for (int s = 0; s < P; ++s) {
            if (idx[static_cast<size_t>(s)] >= static_cast<int>(seq[static_cast<size_t>(s)].size())) continue;
            const int t = ready(seq[static_cast<size_t>(s)][static_cast<size_t>(idx[static_cast<size_t>(s)])]);
            if (t >= 0 && (t < best_t || (t == best_t && s > best_s))) {
                best_t = t;
                best_s = s;
            }
        }
        assert(best_s >= 0 && "pipeline deadlock");
        const PipeOp op = seq[static_cast<size_t>(best_s)][static_cast<size_t>(idx[static_cast<size_t>(best_s)])];
        const int end = best_t + compute;
        if (op.kind == PipeOpKind::F) f_end[static_cast<size_t>(op.stage)][static_cast<size_t>(op.mb)] = end;
        else b_end[static_cast<size_t>(op.stage)][static_cast<size_t>(op.mb)] = end;
        free_at[static_cast<size_t>(best_s)] = end;
        idx[static_cast<size_t>(best_s)] += 1;
        done += 1;
        trace.push_back(TimedPipeOp{op, best_t, end});
    }

    PipeReport r;
    r.stages = P;
    r.microbatches = M;
    r.n_ops = n_ops;
    r.trace = std::move(trace);
    r.makespan = 0;
    for (int t : free_at) r.makespan = std::max(r.makespan, t);
    r.busy = n_ops * compute;
    r.utilization = r.makespan > 0 ? static_cast<double>(r.busy) / (static_cast<double>(P) * r.makespan) : 0;
    r.bubble = 1.0 - r.utilization;
    {
        auto ordered = r.trace;
        std::sort(ordered.begin(), ordered.end(), [](const TimedPipeOp& a, const TimedPipeOp& b) {
            if (a.end != b.end) return a.end < b.end;
            return a.op.stage < b.op.stage;
        });
        std::vector<int> inflight(static_cast<size_t>(P), 0);
        int peak = 0, live = 0;
        for (const auto& e : ordered) {
            if (e.op.kind == PipeOpKind::F) {
                inflight[static_cast<size_t>(e.op.stage)] += 1;
                live += 1;
            } else {
                inflight[static_cast<size_t>(e.op.stage)] -= 1;
                live -= 1;
            }
            peak = std::max(peak, live);
        }
        r.peak_activations = peak;
    }
    return r;
}

} // namespace flowtrain
