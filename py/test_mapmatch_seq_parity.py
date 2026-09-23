"""Phase-7b parity gate: native Viterbi map matcher (mm_match_seq) == Python
oracle (MapMatcherRef.match_seq). The decoded way-index sequence must be
bit-identical (a discrete argmin -- any FP divergence in the cost DP would flip a
snap), and the projected feet must match to numeric noise.

Run: PYTHONPATH=. python3 test_mapmatch_seq_parity.py
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_match_ref import MapMatcherRef
from phase7b_gate import build_scene, _build_matcher, TRUNK, A, B


def _run(cls, hmm=None):
    ways, e, n, psi, *_ = build_scene(seed=2)
    m = _build_matcher(cls, ways)
    if hmm is not None:
        m.set_hmm(*hmm)
    return m.match_seq(e, n, psi)


def seq_parity(hmm=None):
    from core_bridge import MapMatcher
    ref = _run(MapMatcherRef, hmm)
    cpp = _run(MapMatcher, hmm)
    same_way = int(np.sum(ref["way"] != cpp["way"]))
    foot_err = float(np.max(np.abs(ref["foot_e"] - cpp["foot_e"])) +
                     np.max(np.abs(ref["foot_n"] - cpp["foot_n"])))
    cor_err = int(np.sum(ref["corridor"] != cpp["corridor"]))
    tag = "default" if hmm is None else f"hmm={hmm}"
    print(f"mm_seq {tag:22} way mismatches={same_way}  foot max|diff|={foot_err:.2e}  "
          f"corridor mismatches={cor_err}")
    assert same_way == 0, "VITERBI WAY-INDEX PARITY FAIL"
    assert foot_err < 1e-9, "VITERBI FOOT PARITY FAIL"
    assert cor_err == 0, "VITERBI CORRIDOR PARITY FAIL"


if __name__ == "__main__":
    seq_parity()                       # default HMM params
    seq_parity(hmm=(6.0, 15.0, 3.0))   # custom emission/transition/bearing weights
    print("PHASE 7b PARITY GATES PASS: native Viterbi decode matches the oracle")
