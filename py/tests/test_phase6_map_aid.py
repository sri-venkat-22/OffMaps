"""Road-snapping regression for the nav app (FusionEngine.mapMatchUpdate), run
off-device through phase6_check's line-for-line mirror of the live loop.

Pins what the map is allowed to do, and what it isn't:
  1. never consulted while GNSS is healthy -- a road centreline isn't your lane, so
     the fused track must be bit-identical with or without roads loaded;
  2. while dead-reckoning it pins the CROSS-track to the road -- outage cross-track
     error collapses;
  3. it never touches distance travelled -- that stays the speed model's job. With
     the shipped profile's mm_keep_speed the road updates cannot move v at all, so
     the dead-reckoned speed must be BIT-identical with or without roads (v's
     variance never mixes with the cross-terms a road update changes), and
     along-track moves only by re-projection onto the road's curves.

Map = the drive's true centreline (the vehicle was on the road), same validation map
as Phase 5's eskf_map.

The scenario needs cross-track drift to pin. With Doppler owning speed while GNSS is
healthy, the loop now learns the gyro bias well enough that a clean synthetic outage
drifts only ~1 m sideways, so the gyro bias steps by 0.05 deg/s when the outage
starts (a thermal shift the filter cannot see) -- ~20 m of cross-track by its end.
"""
import dataclasses
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("core_bridge")
from phase6_check import run                       # noqa: E402
from core_bridge import MapMatcher                 # noqa: E402
from mapdata import map_from_path                  # noqa: E402
from data.io_vnbd import synth_drive               # noqa: E402

OUTAGE = (150.0, 210.0)
I_END = int(OUTAGE[1] * 10) - 1
_clean = synth_drive("phase6", duration=300, seed=11)
DRIVE = dataclasses.replace(                       # unmodelled gyro-bias step at outage start
    _clean, gyro_z=_clean.gyro_z + np.where(_clean.t >= OUTAGE[0], np.radians(0.05), 0.0))


def _matcher():
    m = MapMatcher()
    for e, n, tunnel, oneway in map_from_path(DRIVE.e, DRIVE.n):
        m.add_way(e, n, tunnel, oneway)
    return m


def _along_cross(fused, i):
    de, dn = fused[i, 0] - DRIVE.e[i], fused[i, 1] - DRIVE.n[i]
    psi = DRIVE.heading[i]
    return de * np.sin(psi) + dn * np.cos(psi), de * np.cos(psi) - dn * np.sin(psi)


@pytest.fixture(scope="module")
def runs():
    return dict(
        gnss=run(DRIVE)[0],
        gnss_map=run(DRIVE, matcher=_matcher())[0],
        dr=run(DRIVE, outage=OUTAGE)[1],
        dr_map=run(DRIVE, outage=OUTAGE, matcher=_matcher())[1],
    )


def test_map_unused_while_gnss_healthy(runs):
    assert np.array_equal(runs["gnss"], runs["gnss_map"]), "road snapping ran while GNSS was healthy"


def test_map_pins_crosstrack_during_outage(runs):
    _, cross_dr = _along_cross(runs["dr"], I_END)
    _, cross_map = _along_cross(runs["dr_map"], I_END)
    assert abs(cross_dr) > 3.0, f"scenario too easy: no-map cross-track only {cross_dr:.1f} m"
    assert abs(cross_map) < 1.0, f"map did not pin cross-track ({cross_map:.1f} m)"


def test_map_never_changes_speed():
    tr_off, tr_on = {}, {}
    run(DRIVE, outage=OUTAGE, trace=tr_off)
    run(DRIVE, outage=OUTAGE, matcher=_matcher(), trace=tr_on)
    # Through the whole outage (the only time the map acts). Once GNSS returns, its
    # position update legitimately reads the cross-terms the road updates changed.
    assert np.array_equal(np.asarray(tr_off["v"])[:I_END + 1], np.asarray(tr_on["v"])[:I_END + 1]), \
        "road snapping changed the dead-reckoned speed"


def test_map_leaves_along_track_alone(runs):
    along_dr, _ = _along_cross(runs["dr"], I_END)
    along_map, _ = _along_cross(runs["dr_map"], I_END)
    # Pinning heading to the road's curves re-projects the same distance travelled,
    # so along-track may move by a small fraction of the drift -- but the map must not
    # correct distance (that is the speed model's job): within 1 m or 5 % of the drift.
    tol = max(1.0, 0.05 * abs(along_dr))
    assert abs(along_map - along_dr) < tol, \
        f"map changed along-track ({along_dr:.1f} -> {along_map:.1f} m); it must only constrain the lane"
