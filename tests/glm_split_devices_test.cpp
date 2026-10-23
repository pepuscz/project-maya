// The devices of a GLM layer split beside an expert helper (glm_split_devices): the helper's device left out of the
// default list, a named list (MAYA_GLM_DEVS) refused when it names the helper or repeats a device, pinned boundaries
// taking the first n + 1 devices beside the helper, `auto` taking up to the part limit - and no helper: the lists as
// before.  And the background moves' candidates beside a promoting helper (glm_helper_moves.hpp).  Host only, no GPU.
#include "strata/core/glm_helper_moves.hpp"
#include "strata/core/glm_split_devices.hpp"

#include <cstdio>
#include <string>
#include <vector>

using strata::core::glm_split_devices;

static int fails = 0;

static void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++fails;
    }
}

static std::vector<int> run(const std::vector<int>* named, int n_dev, int helper, int n_parts, int max_parts,
                            bool want_ok, const char* what) {
    std::vector<int> out;
    std::string err;
    const bool ok = glm_split_devices(named, n_dev, helper, n_parts, max_parts, out, err);
    expect(ok == want_ok, what);
    if (!ok && want_ok) std::fprintf(stderr, "  %s\n", err.c_str());
    if (ok && !want_ok) std::fprintf(stderr, "  accepted\n");
    return out;
}

int main() {
    // no helper: every visible device in order, up to the part limit (auto) or the boundaries' count
    expect(run(nullptr, 3, -1, 0, 8, true, "auto, no helper") == std::vector<int>{0, 1, 2}, "auto lists every device");
    expect(run(nullptr, 6, -1, 0, 4, true, "auto, the part limit") == std::vector<int>{0, 1, 2, 3}, "auto stops at the limit");
    expect(run(nullptr, 3, -1, 2, 8, true, "one boundary") == std::vector<int>{0, 1}, "one boundary takes two devices");
    // the helper (the APU, visible last) beside two cards
    expect(run(nullptr, 3, 2, 0, 8, true, "auto beside the helper") == std::vector<int>{0, 1}, "the helper is left out");
    expect(run(nullptr, 3, 2, 2, 8, true, "one boundary beside the helper") == std::vector<int>{0, 1}, "pinned, helper out");
    expect(run(nullptr, 3, 0, 0, 8, true, "the helper visible first") == std::vector<int>{1, 2}, "a first helper is left out");
    run(nullptr, 3, 2, 3, 8, false, "two boundaries need three devices beside the helper");
    run(nullptr, 2, 1, 2, 8, false, "one card beside the helper is not a split");
    // a named list
    const std::vector<int> cards{0, 1}, with_helper{0, 1, 2}, twice{0, 0};
    expect(run(&cards, 3, 2, 0, 8, true, "named cards beside the helper") == cards, "a named list is kept");
    run(&with_helper, 3, 2, 0, 8, false, "a named list naming the helper is refused");
    run(&twice, 3, -1, 0, 8, false, "a named list naming a device twice is refused");
    run(&cards, 3, -1, 3, 8, false, "a named list shorter than the boundaries need is refused");

    // the background moves' candidates (bg_move_candidate): not in VRAM, no promotion under way, held as the owner holds it
    using strata::core::RamHold;
    using strata::core::bg_move_candidate;
    expect(!bg_move_candidate(true, false, false, false, RamHold::tier), "in VRAM: no candidate");
    expect(bg_move_candidate(false, false, false, false, RamHold::tier), "a RAM-tier expert is a candidate");
    expect(!bg_move_candidate(false, false, false, false, RamHold::other), "a pinned (moving) RAM copy is not");
    expect(!bg_move_candidate(false, false, true, false, RamHold::helper), "helper-owned, promote off: not a candidate");
    expect(bg_move_candidate(false, false, true, true, RamHold::helper), "helper-owned, promote on: a candidate");
    expect(!bg_move_candidate(false, false, true, true, RamHold::tier), "helper-owned but held by the tier: not");
    expect(!bg_move_candidate(false, false, false, true, RamHold::helper), "not owned but held as the helper's: not");
    expect(!bg_move_candidate(false, true, true, true, RamHold::helper), "a helper expert's promotion under way: not");
    expect(!bg_move_candidate(false, true, false, false, RamHold::tier), "a tier expert's promotion under way: not");
    {
        // Codex's ecb37b3 repro: a helper expert's promotion enqueued at boundary 0, its copy still landing at
        // boundaries 1 and 2 (NotReady), landed at 3 - one VRAM slot, never a second enqueue
        int slots = 0;
        bool in_vram = false, pending = false;
        for (int boundary = 0; boundary < 4; ++boundary) {
            if (pending && boundary == 3) {   // step 5 (a): the landed move goes live
                pending = false;
                in_vram = true;
            }
            if (bg_move_candidate(in_vram, pending, true, true, RamHold::helper)) {   // step 5 (c): enqueue
                ++slots;
                pending = true;
            }
        }
        expect(slots == 1 && in_vram && !pending, "a promotion under way is never enqueued twice; landed, one slot");
    }
    if (fails == 0) std::puts("glm_split_devices: the helper's device is never a part of the split; a promoting helper's moves take one slot an expert");
    return fails == 0 ? 0 : 1;
}
