// src/ds4/ds4_arslab.hpp - DS4_AR_SLAB's run cutting: the prompt attention output's all-reduce on a tensor-parallel
// group in runs of `rows` rows cut from row 0, each issued once the slabs before its end are done - pure functions, CPU
// only, shared by ds4_model.cu and tests/ds4_pool_test.cpp
#pragma once

#include <algorithm>
#include <vector>

namespace strata::ds4 {

// rows [r0, r1) of a chunk's all-reduce
struct ArRun {
    int r0 = 0, r1 = 0;
};

// the runs a slab's end completes: `done` rows of a T-row chunk have their wo_b, runs of `rows` rows cut from 0 (the last
// one shorter); ar0, the first row not yet issued, advances past them.  The cut depends on T and rows alone, so every
// member of a group issues the same runs whatever its own slabs (each member's all-reduces must pair with the other's)
inline std::vector<ArRun> ar_runs_done(int done, int T, int rows, int& ar0) {
    std::vector<ArRun> runs;
    while (rows > 0 && ar0 < T && done >= std::min(T, ar0 + rows)) {
        const int r1 = std::min(T, ar0 + rows);
        runs.push_back({ar0, r1});
        ar0 = r1;
    }
    return runs;
}

}  // namespace strata::ds4
