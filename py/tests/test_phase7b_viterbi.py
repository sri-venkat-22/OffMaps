"""Phase-7b regressions: Viterbi/HMM map matching.

  1. native mm_match_seq == oracle: identical decoded way indices + feet;
  2. on a Y-fork, Viterbi beats greedy nearest-road and stops the wrong-snap.

Skipped cleanly when torch is absent (core_bridge imports it) or the native
core isn't built.
"""
import numpy as np
import pytest

pytest.importorskip("core_bridge")
from core_bridge import MapMatcher
from map_match_ref import MapMatcherRef
from phase7b_gate import build_scene, _build_matcher, B

try:
    MapMatcher()
    _BUILT = True
except FileNotFoundError:
    _BUILT = False

pytestmark = pytest.mark.skipif(not _BUILT, reason="native core not built (sh core/build.sh)")


@pytest.mark.parametrize("hmm", [None, (6.0, 15.0, 3.0)])
def test_viterbi_native_matches_oracle(hmm):
    ways, e, n, psi, *_ = build_scene(seed=2)
    def run(cls):
        m = _build_matcher(cls, ways)
        if hmm is not None: m.set_hmm(*hmm)
        return m.match_seq(e, n, psi)
    ref, cpp = run(MapMatcherRef), run(MapMatcher)
    assert np.array_equal(ref["way"], cpp["way"]), "way-index sequences differ"
    assert np.max(np.abs(ref["foot_e"] - cpp["foot_e"])) < 1e-9
    assert np.max(np.abs(ref["foot_n"] - cpp["foot_n"])) < 1e-9
    assert np.array_equal(ref["corridor"], cpp["corridor"])


def test_viterbi_fixes_junction_wrong_snap():
    ways, e, n, psi, te, tn, true_way = build_scene(seed=1)
    m = _build_matcher(MapMatcher, ways)
    mref = _build_matcher(MapMatcherRef, ways)     # oracle exposes the greedy baseline
    g = mref.greedy_seq(e, n, psi)
    v = m.match_seq(e, n, psi)["way"]
    valid = g >= 0
    g_ok = np.mean(g[valid] == true_way[valid])
    v_ok = np.mean(v[valid] == true_way[valid])
    zone = (tn >= 0) & (tn <= 40)
    assert np.sum(g[zone] == B) > 0, "scenario should make greedy wrong-snap"
    assert np.sum(v[zone] == B) == 0, "Viterbi should not wrong-snap to B"
    assert v_ok >= 0.98 and v_ok > g_ok + 0.05
