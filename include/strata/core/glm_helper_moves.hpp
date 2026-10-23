#pragma once

namespace strata::core {

// How a background move's candidate is held in RAM (glm_fast_path.cu fast_boundary, step 5): by the RAM tier
// (kRHold), by the expert helper (kRHelper), or otherwise (pinned by a move under way, free, landing...).
enum class RamHold { other, tier, helper };

// A background promotion candidate: not resident in VRAM, no promotion of it already under way, and held in RAM as
// its owner holds it - by the RAM tier, or (promote mode, STRATA_GLM_HELPER_PROMOTE) a helper-owned expert by the
// helper.  A helper's copy is never pinned while it is promoted (the helper keeps reading it), so the pending up
// move alone keeps the expert from a second VRAM slot (ecb37b3, Codex's review: two boundaries, the first copy
// still landing, enqueued the same expert twice).
inline bool bg_move_candidate(bool in_vram, bool pending_up, bool helper_owned, bool promote, RamHold hold) {
    if (in_vram || pending_up) return false;
    if (helper_owned) return promote && hold == RamHold::helper;
    return hold == RamHold::tier;
}

} // namespace strata::core
