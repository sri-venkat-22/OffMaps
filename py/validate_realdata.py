"""One-command REAL-DATA validation -> a committed, auditable report.

Every headline number in the repo is on synthetic data (clean speed cue by
construction -- pipeline validation, not a competition claim). This driver is the
push-button path from an actual drive to a *reportable* number: point it at a
real IO-VNBD export (--data) or a phone log (--phone) and it

  1. QC-gates the drive (speed-vs-GNSS-track ratio, unit sanity) and refuses to
     call anything reportable that fails;
  2. freezes a deterministic outage manifest pinned to the data's sha256 (so the
     number can be reproduced and audited, never silently re-cut);
  3. scores every available model (physics / cv / nn / eskf / eskf_map);
  4. writes out/<name>/REPORT.md + results.json + manifest.json + plots, with full
     provenance (file hashes, columns, units, env, exact command) and a REPORTABLE
     / NOT REPORTABLE verdict at the top.

It never fabricates: synthetic or --smoke input is always stamped "PIPELINE SMOKE
-- NOT a competition claim", and QC failure / manifest drift / too-few-windows
each block the reportable stamp with the reason printed.

    python -m validate_realdata --data DRIVES_DIR --preset io-vnbd --kmph
    python -m validate_realdata --phone drive_1699_ --yaw-sign -1
    python -m validate_realdata --data IOVNBD_SYNC_DIR --layout iovnbd-sync   # phone IMU vs vehicle GNSS
    python -m validate_realdata --data D --manifest out/realdata/D/manifest.json  # reproduce
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.io_vnbd import load_dir                                    # noqa: E402
from data.phone_log import load_phone                               # noqa: E402
from data.iovnbd_sync import load_sync_dir                          # noqa: E402
from data.outage import (make_outages, dump_manifest, load_manifest,  # noqa: E402
                         verify_manifest, DURATIONS, _env, tilde_home)
from eval import metrics as M                                        # noqa: E402
from eval.models import REGISTRY                                     # noqa: E402
import baseline.physics  # noqa: E402,F401  registers "physics"
from eval.report import PRESETS, speed_vs_track, _plot_drift, _plot_examples  # noqa: E402

_OPTIONAL_ERR = None
try:  # optional: needs torch (nn) + built native core (eskf/eskf_map)
    import model.nn_model  # noqa: F401
    import core_bridge      # noqa: F401
except Exception as _e:
    _OPTIONAL_ERR = f"{_e.__class__.__name__}: {_e}"

# The models we report, best first. Filtered to what actually registered, so the
# report degrades gracefully to baselines-only when torch/native core are absent.
REPORTABLE_MODELS = ["eskf_map", "eskf", "nn", "physics", "cv"]
MIN_WINDOWS = 5                 # below this a "number" is noise, not a result


# ------------------------------------------------------------------ provenance
def _sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _input_files(args):
    """Hash the exact bytes that fed the run, for both loader paths."""
    files = []
    if args.phone:
        for name in ("imu.csv", "gnss.csv", "meas.csv"):
            p = os.path.join(args.phone, name)
            if os.path.exists(p):
                files.append((name, _sha256(p), os.path.getsize(p)))
    elif args.data:
        import glob
        for p in sorted(glob.glob(os.path.join(args.data, "**", args.glob), recursive=True)):
            if os.path.isfile(p):
                files.append((os.path.relpath(p, args.data), _sha256(p), os.path.getsize(p)))
    return files


# --------------------------------------------------------------------- loading
def _apply_preset(args, ap):
    if not args.preset:
        return
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k, None) in (None, ap.get_default(k)):
            setattr(args, k, v)


def _load(args):
    if args.phone:
        return [load_phone(args.phone, target_hz=args.hz, yaw_sign=args.yaw_sign)]
    if args.data and args.layout == "iovnbd-sync":
        return load_sync_dir(args.data, min_corr=args.min_sync_corr)
    if args.data:
        cols = dict(time_col=args.time_col, lat_col=args.lat_col, lon_col=args.lon_col,
                    speed_col=args.speed_col, heading_col=args.heading_col,
                    ecu_speed_col=args.ecu_speed_col, speed_is_kmph=args.kmph,
                    ecu_is_kmph=args.ecu_kmph, time_unit=args.time_unit, max_gap_s=args.max_gap)
        return load_dir(args.data, pattern=args.glob, **cols)
    sys.exit("give --data DIR or --phone DIR (this driver validates REAL drives)")


# ------------------------------------------------------------------------- QC
def _qc(drives):
    """Per-drive QC verdict. ratio ~1 means the speed column and the GNSS track
    agree; a mismatch means wrong units / wheel-speed scale and every drift% is
    off, so it blocks the reportable stamp."""
    rows = []
    for d in drives:
        r = speed_vs_track(d)
        ratio, nfast = (r if r else (None, 0))
        ok = (ratio is not None) and (0.85 < ratio < 1.15)
        rows.append(dict(id=d.vehicle_id, ratio=ratio, nfast=nfast, ok=ok,
                         n=len(d), synthetic=bool(d.meta.get("synthetic")),
                         warnings=list(d.qc.warnings) if getattr(d, "qc", None) else [],
                         qc_summary=d.qc.summary() if getattr(d, "qc", None) else None))
    return rows


# ---------------------------------------------------------------------- report
def _md_table(table):
    has_mae = any("speed_mae" in r for r in table.values())
    head = "| dur (s) | n | drift % | CEP50 (m) | CEP95 (m) | along (m) | cross (m) |"
    sep = "|--:|--:|--:|--:|--:|--:|--:|"
    if has_mae:
        head += " vMAE (m/s) |"; sep += "--:|"
    lines = [head, sep]
    for dur, r in table.items():
        cells = (f"| {dur:.0f} | {r['n']} | {r['drift_med']:.1f} | {r['cep50']:.1f} | "
                 f"{r['cep95']:.1f} | {r['along_med']:.1f} | {r['cross_med']:.1f} |")
        if has_mae:
            cells += f" {r.get('speed_mae', float('nan')):.2f} |"
        lines.append(cells)
    return "\n".join(lines)


def _write_report(path, *, verdict, reasons, drives, qc_rows, files, results, sig,
                  args, manifest_path, replayed, env):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cmd = "python -m validate_realdata " + " ".join(tilde_home(sys.argv[1:]))
    stamp = ("**REPORTABLE**" if verdict else "**NOT REPORTABLE**")
    lines = [
        f"# Real-data validation — {', '.join(d.vehicle_id for d in drives)}",
        "",
        f"{stamp}  ·  {now}",
        "",
    ]
    if not verdict:
        lines += ["> " + "  \n> ".join(reasons or ["(no reason recorded)"]), ""]
    lines += [
        "Dead-reckoning numbers on the **loaded drive**, distinct from the repo's "
        "synthetic pipeline-validation numbers. Drift % is against the ISRO metric "
        f"(`--dist-source {args.dist_source}`); the 10 % line is the PS limit.",
        "",
        "## Provenance",
        "",
        f"- source: `{tilde_home(args.phone or args.data)}`  ({'phone log' if args.phone else 'dataset dir'})",
        f"- manifest: `{manifest_path}`  ({'replayed (frozen)' if replayed else 'freshly frozen this run'})",
        f"- seed: {args.seed}  ·  windows/duration requested: {args.n_per_duration}  ·  min-dist: {args.min_dist} m",
        f"- reproduce: `{cmd}`",
        f"- env: python {env.get('python')} · numpy {env.get('numpy')} · "
        f"pandas {env.get('pandas', 'n/a')} · {env.get('platform')}",
        "",
        "| input file | sha256 (first 16) | bytes |",
        "|---|---|--:|",
    ]
    for name, sha, nb in files:
        lines.append(f"| `{name}` | `{sha[:16]}` | {nb:,} |")
    if not files:
        lines.append("| _(no input files hashed)_ | | |")

    lines += ["", "## QC gate", "",
              "| drive | rows | speed/track ratio | fast n | verdict |",
              "|---|--:|--:|--:|---|"]
    for q in qc_rows:
        ratio = "—" if q["ratio"] is None else f"{q['ratio']:.3f}"
        v = "SYNTHETIC" if q["synthetic"] else ("PASS" if q["ok"] else "SUSPECT (units?)")
        lines.append(f"| {q['id']} | {q['n']} | {ratio} | {q['nfast']} | {v} |")
    for q in qc_rows:
        for w in q["warnings"]:
            lines.append(f"- ⚠ {q['id']}: {w}")
    rejected = drives[0].meta.get("rejected_segments") if drives else None
    if rejected:
        lines += ["", f"<details><summary>{len(rejected)} input segment(s) rejected by the "
                  "loader (not scored)</summary>", ""]
        lines += [f"- {r}" for r in rejected] + ["", "</details>"]

    lines += ["", "## Results", ""]
    if not results:
        lines.append("_No models scored (no usable outage windows, or none registered)._")
    for name, (rows, table) in results.items():
        de = [r["end_err"] for r in rows]
        lines += [f"### `{name}`  (n={len(rows)} windows)", "", _md_table(table), "",
                  f"overall **CEP50 {M.cep(de,50):.1f} m · CEP95 {M.cep(de,95):.1f} m**", ""]
        if name in sig and sig[name] is not None:
            c = sig[name]
            lines.append(f"σ-calibration: z_var={c['z_var']:.2f} z_mean={c['z_mean']:+.2f} "
                         f"(n={c['n']}) → {'PASS' if c['passed'] else 'FAIL: ' + c['why']}")
            lines.append("")

    if _OPTIONAL_ERR:
        lines += [f"> nn/eskf unavailable this run: `{_OPTIONAL_ERR}` — baselines only.", ""]
    lines += ["## Plots", "",
              "- `drift_vs_duration.png` — median drift % vs outage length (10 % ISRO line)",
              "- `examples_<model>.png` — dead-reckoned vs truth for the longest windows", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------- driver
def _build_argparser():
    ap = argparse.ArgumentParser(description="Validate a REAL drive and emit a committed report.")
    ap.add_argument("--data", help="dir of drive CSVs (IO-VNBD export)")
    ap.add_argument("--phone", help="dir with a phone log (imu.csv + gnss.csv)")
    ap.add_argument("--glob", default="*.csv", help="filename pattern under --data")
    ap.add_argument("--layout", choices=["csv", "iovnbd-sync"], default="csv",
                    help="csv: one drive per CSV (column flags). iovnbd-sync: IO-VNBD S-*/V-* "
                         "pairs -- phone IMU scored against vehicle GNSS")
    ap.add_argument("--min-sync-corr", type=float, default=0.8,
                    help="iovnbd-sync: drop phone segments whose yaw gyro tracks the ECU yaw "
                         "rate worse than this after time sync")
    ap.add_argument("--include", default=None,
                    help="regex: score only drives whose id matches (e.g. a held-out test split)")
    ap.add_argument("--nn-ckpt", default=None,
                    help="SpeedNet checkpoint for nn/eskf/eskf_map (default model/nn.pt)")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="column defaults for a known layout")
    ap.add_argument("--models", default=None, help="comma list (default: all reportable that registered)")
    ap.add_argument("--manifest", help="replay a committed window set (reproduce a past run)")
    ap.add_argument("--allow-manifest-drift", action="store_true",
                    help="run even if the data no longer matches the manifest (NOT reportable)")
    ap.add_argument("--out", default=None, help="output dir (default out/realdata/<name>)")
    ap.add_argument("--hz", type=float, default=10.0, help="resample grid for --phone")
    ap.add_argument("--yaw-sign", type=float, default=-1.0, help="gyro->heading sign knob (--phone)")
    ap.add_argument("--n-per-duration", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-dist", type=float, default=20.0, help="reject windows covering less ground (m)")
    ap.add_argument("--max-gap", type=float, default=None, help="reject windows with a bigger sample gap (s)")
    ap.add_argument("--dist-source", choices=["speed", "track"], default="speed")
    ap.add_argument("--time-unit", default="auto", choices=["auto", "s", "ms", "us", "ns"])
    for c, dflt in [("time", "time"), ("lat", "latitude"), ("lon", "longitude"),
                    ("speed", "speed"), ("heading", None), ("ecu-speed", None)]:
        ap.add_argument(f"--{c}-col", default=dflt)
    ap.add_argument("--kmph", action="store_true", help="--speed-col is km/h")
    ap.add_argument("--ecu-kmph", action="store_true", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="dev/pipeline run: never stamp reportable (use on synthetic input)")
    return ap


def run(args) -> dict:
    """Execute the workflow. Returns a summary dict (also written to results.json)."""
    name = os.path.basename(os.path.normpath(args.phone or args.data or "drive"))
    out = args.out or os.path.join("out", "realdata", name)
    os.makedirs(out, exist_ok=True)

    if args.nn_ckpt:
        import model.nn_model as _nnm
        _nnm.DEFAULT = os.path.abspath(args.nn_ckpt)
    drives = _load(args)
    if args.include:
        import re
        drives = [d for d in drives if re.search(args.include, d.vehicle_id)]
        if not drives:
            sys.exit(f"--include {args.include!r} matched no drive")
    files = _input_files(args)
    qc_rows = _qc(drives)
    synthetic = any(q["synthetic"] for q in qc_rows)
    qc_ok = all(q["ok"] for q in qc_rows)

    # --- freeze or replay the outage manifest ---
    manifest_path = args.manifest or os.path.join(out, "manifest.json")
    replayed = bool(args.manifest)
    manifest_drift = False
    if replayed:
        outs_all, doc = load_manifest(args.manifest)
        problems = verify_manifest(doc, drives)
        manifest_drift = bool(problems)
        outs_by_drive = {}
        for o in outs_all:
            outs_by_drive.setdefault(o.drive_id, []).append(o)
    else:
        outs_by_drive = {d.vehicle_id: make_outages(
            d, DURATIONS, args.n_per_duration, args.seed,
            min_dist_m=args.min_dist, max_gap_s=args.max_gap) for d in drives}
        outs_all = [o for outs in outs_by_drive.values() for o in outs]
        dump_manifest(outs_all, manifest_path, drives=drives,
                      params=dict(seed=args.seed, n_per_duration=args.n_per_duration,
                                  durations=list(DURATIONS), min_dist_m=args.min_dist,
                                  dist_source=args.dist_source, argv=sys.argv[1:]))

    # --- score every requested/available model ---
    if args.models:
        names = [n for n in args.models.split(",") if n]
    else:
        names = [n for n in REPORTABLE_MODELS if n in REGISTRY]
    results, sig = {}, {}
    for nm in names:
        if nm not in REGISTRY:
            print(f"[validate] skip unknown model {nm}"); continue
        model = REGISTRY[nm]
        rows = [M.score_outage(d, o, model(d, o), dist_source=args.dist_source)
                for d in drives for o in outs_by_drive.get(d.vehicle_id, [])]
        if not rows:
            continue
        results[nm] = (rows, M.per_duration_table(rows))
        zs = [r["_z"] for r in rows if "_z" in r]
        sig[nm] = (M.sigma_calibration(np.concatenate(zs), np.zeros(sum(len(z) for z in zs)),
                                       np.ones(sum(len(z) for z in zs))) if zs else None)

    # --- reportable verdict (the anti-fabrication gate) ---
    reasons = []
    if synthetic:
        reasons.append("input is SYNTHETIC — pipeline smoke, not a competition claim.")
    if args.smoke:
        reasons.append("run with --smoke — dev/pipeline check, not stamped reportable.")
    if not qc_ok:
        bad = [q["id"] for q in qc_rows if not q["ok"]]
        reasons.append(f"QC failed on {bad} — speed/track mismatch (check units / --kmph).")
    if len(outs_all) < MIN_WINDOWS:
        reasons.append(f"only {len(outs_all)} outage window(s) (< {MIN_WINDOWS}) — too few to report.")
    if manifest_drift:
        reasons.append("manifest no longer matches the data — frozen number cannot be reproduced.")
    verdict = not reasons

    # --- artifacts ---
    if results:
        _plot_drift({k: v for k, v in results.items()}, os.path.join(out, "drift_vs_duration.png"))
        d0 = drives[0]
        for nm, (rows, table) in results.items():
            _plot_examples(d0, outs_by_drive.get(d0.vehicle_id, []), nm, REGISTRY[nm],
                           os.path.join(out, f"examples_{nm}.png"))
    env = _env()
    _write_report(os.path.join(out, "REPORT.md"), verdict=verdict, reasons=reasons,
                  drives=drives, qc_rows=qc_rows, files=files, results=results, sig=sig,
                  args=args, manifest_path=manifest_path, replayed=replayed, env=env)
    summary = dict(
        name=name, out=out, reportable=verdict, reasons=reasons,
        synthetic=synthetic, qc_ok=qc_ok, n_windows=len(outs_all),
        manifest=manifest_path, manifest_drift=manifest_drift, env=env,
        input_files=[dict(name=n, sha256=s, bytes=b) for n, s, b in files],
        qc=[{k: q[k] for k in ("id", "ratio", "nfast", "ok", "synthetic", "n")} for q in qc_rows],
        models={nm: tbl for nm, (rows, tbl) in results.items()},
        argv=sys.argv[1:],
    )
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(tilde_home(summary), f, indent=2)
    return summary


def main():
    ap = _build_argparser()
    args = ap.parse_args()
    _apply_preset(args, ap)
    s = run(args)
    print(f"\n{'='*64}")
    banner = "REPORTABLE" if s["reportable"] else "NOT REPORTABLE"
    print(f"  {banner}: {s['name']}  ({s['n_windows']} windows, {len(s['models'])} models)")
    for r in s["reasons"]:
        print(f"    - {r}")
    print(f"  report -> {s['out']}/REPORT.md")
    print("=" * 64)
    return 0 if s["reportable"] or args.smoke else 2


if __name__ == "__main__":
    sys.exit(main())
