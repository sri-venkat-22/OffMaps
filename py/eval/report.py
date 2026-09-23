"""Phase-0 exit gate.

    python -m eval.report --model zero          # synth drive, no dataset needed
    python -m eval.report --model cv,zero
    python -m eval.report --data DIR --lat-col Latitude ... --model cv
    python -m eval.report --data DIR --describe            # parse + QC only
    python -m eval.report --data DIR --manifest out/manifest.json --model cv

Runs each model over the frozen outage set, prints the per-duration table, and
writes plots + the outage manifest to --out. Every later phase is graded by
re-running this with a new --model.

--describe parses and QCs a dataset without scoring anything, so a column-flag
or unit mistake surfaces in a second instead of as a plausible-looking table.
--manifest replays a committed window set instead of resampling one, and
refuses to run if the data underneath it changed.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.io_vnbd import synth_drive, load_dir                        # noqa: E402
from data.outage import (make_outages, dump_manifest, load_manifest,  # noqa: E402
                         verify_manifest, DURATIONS)
from eval.models import REGISTRY                            # noqa: E402
from eval import metrics as M                               # noqa: E402
import baseline.physics  # noqa: E402,F401  (import registers the 'physics' model)

_OPTIONAL_ERR = None
try:  # optional: needs torch (nn) + built core (eskf)
    import model.nn_model  # noqa: F401  registers "nn"
    import core_bridge     # noqa: F401  registers "eskf"
except Exception as _e:  # keep baselines usable without torch/native lib
    _OPTIONAL_ERR = f"{_e.__class__.__name__}: {_e}"
    print(f"[report] nn/eskf unavailable: {_OPTIONAL_ERR}")

# IO-VNBD-ish header sets seen in the wild. A preset is a starting point that
# --*-col flags still override; it is not a claim about your particular export.
PRESETS = {
    "io-vnbd": dict(time_col="time", lat_col="latitude", lon_col="longitude",
                    speed_col="speed", ecu_speed_col="wheel_speed"),
    "phone":   dict(time_col="t_ns", lat_col="lat", lon_col="lon",
                    speed_col="speed", heading_col="bearing", time_unit="ns"),
}


def load_drives(args):
    if args.phone:
        from data.phone_log import load_phone
        return [load_phone(args.phone, target_hz=args.hz, yaw_sign=args.yaw_sign)]
    if args.data:
        cols = dict(time_col=args.time_col, lat_col=args.lat_col, lon_col=args.lon_col,
                    speed_col=args.speed_col, heading_col=args.heading_col,
                    ecu_speed_col=args.ecu_speed_col, speed_is_kmph=args.kmph,
                    ecu_is_kmph=args.ecu_kmph, time_unit=args.time_unit,
                    max_gap_s=args.max_gap)
        return load_dir(args.data, pattern=args.glob, **cols)
    return [synth_drive(f"synth{i}", duration=900, seed=i) for i in range(args.n_synth)]


def speed_vs_track(d, *, min_speed=5.0, min_n=50):
    """Median (GNSS step distance) / (speed * dt), over fast samples only.

    Cross-checks the drift denominator (the speed column) against its numerator
    (the ENU track). Comparing TOTAL path lengths does not work on real data:
    per-sample position noise adds length that never cancels, so a clean 1 Hz
    log with 3 m noise reads ~12% long and trips a naive check. Restricting to
    fast samples shrinks the relative inflation (4 m of jitter on a 20 m step is
    2%, on a 1 m step is 300%), and the median kills the rest.
    """
    dt = np.diff(d.t)
    step = np.hypot(np.diff(d.e), np.diff(d.n))
    v = 0.5 * (d.speed[:-1] + d.speed[1:])
    m = (v > min_speed) & (dt > 0)
    if m.sum() < min_n:
        return None
    return float(np.median(step[m] / (v[m] * dt[m]))), int(m.sum())


def describe(drives):
    print(f"\n{len(drives)} drive(s)")
    bad = 0
    for d in drives:
        qc = d.qc
        print(f"  {d.vehicle_id}")
        print(f"    {qc.summary() if qc else 'no QC'}")
        for w in (qc.warnings if qc else []):
            print(f"      ! {w}")
        r = speed_vs_track(d)
        if r is not None:
            ratio, nfast = r
            flag = "" if 0.85 < ratio < 1.15 else "   <-- SUSPECT (wrong speed units or wheel-speed scale?)"
            print(f"    speed vs GNSS track: ratio {ratio:.3f} over {nfast} fast samples{flag}")
            if not 0.85 < ratio < 1.15:
                bad += 1
    return bad


def _plot_drift(results, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, (rows, table) in results.items():
        xs = sorted(table); ys = [table[d]["drift_med"] for d in xs]
        ax.plot(xs, ys, "o-", label=name)
    ax.axhline(10, ls="--", c="r", label="ISRO 10% limit")
    ax.set(xlabel="outage duration (s)", ylabel="median drift (% of distance)",
           title="Dead-reckoning drift vs outage duration")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_examples(drive, outs, name, model, path):
    """Longest available duration, up to 4 windows. Real logs are often too
    short to contain a 60 s window, and subplots(1, 0) raises."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if not outs:
        return
    prefer = [o for o in outs if o.duration_s == 60]
    if not prefer:
        best = max(o.duration_s for o in outs)
        prefer = [o for o in outs if o.duration_s == best]
    show = prefer[:4]
    fig, axs = plt.subplots(1, len(show), figsize=(4 * len(show), 4), squeeze=False)
    axs = axs.ravel()
    for ax, o in zip(axs, show):
        pe, pn, _, _ = model(drive, o)
        ax.plot(drive.e[o.i0:o.i1], drive.n[o.i0:o.i1], "g-", label="truth")
        ax.plot(pe, pn, "r--", label=name)
        ax.plot(drive.e[o.i0], drive.n[o.i0], "ko", ms=4)
        ax.set(title=f"{o.duration_s:.0f}s", aspect="equal"); ax.grid(alpha=.3)
    axs[0].legend(); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="zero", help="comma list from REGISTRY")
    ap.add_argument("--data", help="dir of drive CSVs (default: synthetic)")
    ap.add_argument("--glob", default="*.csv", help="filename pattern under --data")
    ap.add_argument("--phone", help="dir with a phone log (imu.csv + gnss.csv)")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="column defaults for a known layout")
    ap.add_argument("--describe", action="store_true", help="parse + QC only, no scoring")
    ap.add_argument("--manifest", help="replay a committed window set instead of resampling")
    ap.add_argument("--allow-manifest-drift", action="store_true",
                    help="run even if --manifest no longer matches the data (NOT reportable)")
    ap.add_argument("--hz", type=float, default=10.0, help="resample grid for --phone")
    ap.add_argument("--yaw-sign", type=float, default=-1.0, help="gyro->heading sign knob")
    ap.add_argument("--out", default="out"); ap.add_argument("--n-synth", type=int, default=3)
    ap.add_argument("--n-per-duration", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-dist", type=float, default=20.0, help="reject windows covering less ground (m)")
    ap.add_argument("--max-gap", type=float, default=None, help="reject drives/windows with a bigger sample gap (s)")
    ap.add_argument("--dist-source", choices=["speed", "track"], default="speed",
                    help="drift%% denominator: speed integral (ISRO) or GNSS track length")
    ap.add_argument("--time-unit", default="auto", choices=["auto", "s", "ms", "us", "ns"])
    for c, dflt in [("time", "time"), ("lat", "latitude"), ("lon", "longitude"),
                    ("speed", "speed"), ("heading", None), ("ecu-speed", None)]:
        ap.add_argument(f"--{c}-col", default=dflt)
    ap.add_argument("--kmph", action="store_true", help="--speed-col is km/h")
    ap.add_argument("--ecu-kmph", action="store_true", default=None,
                    help="--ecu-speed-col is km/h (defaults to --kmph; set when units differ)")
    args = ap.parse_args()

    if args.preset:
        for k, v in PRESETS[args.preset].items():
            # only fill what the user left at its default
            if getattr(args, k, None) in (None, ap.get_default(k)):
                setattr(args, k, v)

    names = [n for n in args.model.split(",") if n]
    missing = [n for n in names if n not in REGISTRY]
    if missing and not args.describe:
        extra = f"\n  (nn/eskf failed to import: {_OPTIONAL_ERR})" if _OPTIONAL_ERR else ""
        sys.exit(f"unknown model(s) {missing}; have {sorted(REGISTRY)}{extra}")

    os.makedirs(args.out, exist_ok=True)
    drives = load_drives(args)
    suspect = describe(drives)
    if args.describe:
        print("\n--describe: parsed only, nothing scored.")
        return 1 if suspect else 0
    if suspect:
        print(f"\n[report] {suspect} drive(s) have a speed/track mismatch -- drift%% may be "
              f"systematically off. Check --kmph / --speed-col.")

    params = dict(seed=args.seed, n_per_duration=args.n_per_duration,
                  durations=list(DURATIONS), min_dist_m=args.min_dist,
                  max_gap_s=args.max_gap, dist_source=args.dist_source,
                  n_synth=args.n_synth if not args.data else None,
                  argv=sys.argv[1:])

    if args.manifest:
        outs_all, doc = load_manifest(args.manifest)
        problems = verify_manifest(doc, drives)
        if problems:
            msg = "manifest does not match the loaded data:\n  " + "\n  ".join(problems)
            if not args.allow_manifest_drift:
                sys.exit(msg + "\n(pass --allow-manifest-drift to run anyway; results are NOT reportable)")
            print(f"[report] WARNING {msg}")
        outs_by_drive = {}
        for o in outs_all:
            outs_by_drive.setdefault(o.drive_id, []).append(o)
        print(f"replaying {len(outs_all)} window(s) from {args.manifest} "
              f"(manifest v{doc.get('version')}, params={doc.get('params', {}).get('seed', '?')})")
    else:
        outs_by_drive, rejected = {}, {}
        for d in drives:
            outs_by_drive[d.vehicle_id] = make_outages(
                d, DURATIONS, args.n_per_duration, args.seed,
                min_dist_m=args.min_dist, max_gap_s=args.max_gap)
            for k, v in getattr(make_outages, "last_rejected", {}).items():
                rejected[k] = rejected.get(k, 0) + v
        outs_all = [o for outs in outs_by_drive.values() for o in outs]
        dump_manifest(outs_all, os.path.join(args.out, "manifest.json"),
                      drives=drives, params=params)
        want = len(drives) * len(DURATIONS) * args.n_per_duration
        print(f"\n{len(drives)} drive(s), {len(outs_all)}/{want} frozen outage windows "
              f"(seed={args.seed}) -> {args.out}/manifest.json")
        if sum(rejected.values()):
            print(f"  rejected candidates: " +
                  ", ".join(f"{k}={v}" for k, v in rejected.items() if v))

    if not outs_all:
        sys.exit("no usable outage windows -- drives too short, too gappy, or all stationary")

    results = {}
    for name in names:
        model = REGISTRY[name]
        rows = [M.score_outage(d, o, model(d, o), dist_source=args.dist_source)
                for d in drives for o in outs_by_drive.get(d.vehicle_id, [])]
        if not rows:
            print(f"\n== {name} ==\n   no windows matched the loaded drives"); continue
        table = M.per_duration_table(rows)
        M.print_table(table, name)
        de = [r["end_err"] for r in rows]
        print(f"   overall CEP50={M.cep(de,50):.1f}m CEP95={M.cep(de,95):.1f}m  (n={len(rows)})")
        zs = [r["_z"] for r in rows if "_z" in r]
        if zs:
            z = np.concatenate(zs)
            cal = M.sigma_calibration(z, np.zeros_like(z), np.ones_like(z))
            print(f"   sigma-calibration: z_var={cal['z_var']:.2f} z_mean={cal['z_mean']:+.2f} "
                  f"n={cal['n']} -> {'PASS' if cal['passed'] else 'FAIL: ' + cal['why']}")
        results[name] = (rows, table)
        _plot_examples(drives[0], outs_by_drive.get(drives[0].vehicle_id, []), name, model,
                       os.path.join(args.out, f"examples_{name}.png"))

    if results:
        _plot_drift(results, os.path.join(args.out, "drift_vs_duration.png"))
        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump({"params": params,
                       "tables": {k: v[1] for k, v in results.items()}}, f, indent=2)
    print(f"\nplots + manifest in {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
