"""Named-stage ablation of the live loop: what each component buys, on real drives.

    PYTHONPATH=. python3 ablation.py --set val  --out ../out/ablation/val.json
    PYTHONPATH=. python3 ablation.py --set lodo --stages physics,speednet --out ../out/ablation/lodo_a.json
    PYTHONPATH=. python3 ablation.py --report ../out/ablation/val.json ../out/ablation/lodo.json   # table + plot

Each stage adds ONE component to the previous one; everything else is the shipped
live loop (edge_engine = the phone's FusionEngine), same protocol and metric as
phase8_eval / phase9_map_eval (outages 10/30/60/120 s after 3 min of GNSS, median
drift % of distance; identical windows in every stage, so stages pair outage by outage).

  physics    gyro heading + the speed at outage entry, held (no AI while dead-reckoning)
  speednet   + SpeedNet pseudo-odometer, Doppler self-calibrated (hold 10 s, then SpeedNet)
  head       + learned fusion head: speed AND its sigma from a GRU (the shipped app)
  map        + HMM road matching on the real OSM network (edge_engine.MAP_HMM, Phase 9)

Sets: `val` (2 drives, ~1 h) and `lodo` (train, leave-one-drive-out: each drive runs with a
SpeedNet and head that never saw it, ~9.5 h). The test split is not used here.
"""
from __future__ import annotations
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import numpy as np

from edge_engine import EdgeEngine
from phase8_eval import live, DURATIONS

HERE = os.path.dirname(os.path.abspath(__file__))
ROADS = os.path.join(HERE, "..", "map", "coventry", "roads.bin")
STAGES = ("physics", "speednet", "head", "map")
LABEL = {"physics": "Gyro heading + entry speed (no AI)", "speednet": "+ SpeedNet (AI speed)",
         "head": "+ learned fusion head (shipped)", "map": "+ HMM road matching"}


def factory(stage, head, roads, net=None):
    return {
        "physics": lambda: EdgeEngine(head=None, speed_net=net, dr_speed=False),
        "speednet": lambda: EdgeEngine(head=None, speed_net=net),
        "head": lambda: EdgeEngine(head=head, speed_net=net),
        "map": lambda: EdgeEngine(head=head, speed_net=net, roads=roads, map_mode="hmm"),
    }[stage]


def run_set(which, stages, data, roads):
    from data.iovnbd_sync import load_sync_dir
    from model.train_real import split_drives
    from model.fusion_head import load_head
    split = split_drives(load_sync_dir(data, verbose=False))
    raw = {}
    for s in stages:
        if which == "val":
            head = load_head(os.path.join(HERE, "model", "fusion_head.pt"))
            raw[s] = live(split["val"], factory(s, head, roads), raw=True)
        else:
            from model.fusion_head import FOLDS
            from edge_engine import TorchSpeedNet
            oof = os.path.join(HERE, "model", "oof")
            fold_of = lambda d: [k for k, rx in FOLDS.items() if re.search(rx, d.vehicle_id.split("#")[0])][0]
            per = {D: [] for D in DURATIONS}
            for fk in FOLDS:
                drv = [d for d in split["train"] if fold_of(d) == fk]
                net = TorchSpeedNet(os.path.join(oof, f"nn_oof_{fk}.pt"))
                head = load_head(os.path.join(oof, f"fusion_head_oof_{fk}.pt"))
                r = live(drv, factory(s, head, roads, net), raw=True)
                for D in DURATIONS:
                    per[D] += r[D]
            raw[s] = per
        m = [np.median(raw[s][D]) for D in DURATIONS]
        print(f"{which:4s} {s:9s} " + "  ".join(f"{D}s {x:5.1f}%" for D, x in zip(DURATIONS, m))
              + f"  | mean {np.mean(m):5.1f}", flush=True)
    return raw


def summary(raw):
    out = {}
    for s, per in raw.items():
        out[s] = {str(D): dict(median=float(np.median(p)), p75=float(np.percentile(p, 75)),
                               under10=float((np.asarray(p) < 10).mean()), n=len(p))
                  for D, p in ((int(k), v) for k, v in per.items())}
    return out


def paired_step(raw, a, b, rng):
    """Median per-outage change from stage a to b (points) + bootstrap 90 % CI, per duration."""
    out = {}
    for D in raw[a]:
        x = np.asarray(raw[b][D]) - np.asarray(raw[a][D])
        bs = [np.median(rng.choice(x, len(x))) for _ in range(2000)]
        out[str(D)] = dict(median=float(np.median(x)), lo=float(np.percentile(bs, 5)),
                           hi=float(np.percentile(bs, 95)), better=float((x < 0).mean()))
    return out


def report(paths, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    lines, sets = [], []
    for p in paths:
        J = json.load(open(p)); raw = J["per_outage"]; S = J["summary"]
        name = {"val": "Validation drives (~1 h)", "lodo": "Train drives, leave-one-drive-out (~9.5 h)"}[J["set"]]
        sets.append((name, S))
        stages = [s for s in STAGES if s in S]
        lines += [f"**{J['title']}**", "",
                  "| stage | " + " | ".join(f"{D} s" for D in DURATIONS) + " | mean | < 10 % (30 / 60 s) |",
                  "|---|" + "--:|" * len(DURATIONS) + "--:|--:|"]
        for s in stages:
            m = [S[s][str(D)]["median"] for D in DURATIONS]
            lines.append(f"| {LABEL[s]} | " + " | ".join(f"{x:.1f}" for x in m) + f" | **{np.mean(m):.1f}** | "
                         f"{100 * S[s]['30']['under10']:.0f} / {100 * S[s]['60']['under10']:.0f} % |")
        n = [S[stages[0]][str(D)]["n"] for D in DURATIONS]
        lines += ["", f"Outages per duration: {' / '.join(map(str, n))}. Median drift % of distance.", "",
                  "Paired step, outage by outage (median change in points [90 % bootstrap CI], share of outages better):", ""]
        for a, b in zip(stages, stages[1:]):
            st = paired_step(raw, a, b, rng)
            lines.append(f"- {LABEL[b]}: " + "; ".join(
                f"{D} s {st[str(D)]['median']:+.1f} [{st[str(D)]['lo']:+.1f}, {st[str(D)]['hi']:+.1f}], "
                f"{100 * st[str(D)]['better']:.0f} %" for D in DURATIONS))
        lines.append("")
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "ablation.md"), "w").write("\n".join(lines))
    print("\n".join(lines))

    fig, axes = plt.subplots(1, len(sets), figsize=(6.2 * len(sets), 4.2), squeeze=False, sharey=True)
    colors = {"physics": "#9aa3ad", "speednet": "#5b8def", "head": "#2459c9", "map": "#0f9d74"}
    for ax, (name, S) in zip(axes[0], sets):
        stages = [s for s in STAGES if s in S]
        w = 0.8 / len(stages); x = np.arange(len(DURATIONS))
        for i, s in enumerate(stages):
            m = [S[s][str(D)]["median"] for D in DURATIONS]
            ax.bar(x + (i - (len(stages) - 1) / 2) * w, m, w, label=LABEL[s], color=colors[s])
        ax.axhline(10, color="#c0392b", ls="--", lw=1.2, label="PS target: 10 %")
        ax.set_xticks(x, [f"{D} s" for D in DURATIONS]); ax.set_xlabel("GNSS outage length")
        ax.set_ylabel("median drift, % of distance"); ax.set_title(name)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
    axes[0][0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "ablation.png"), dpi=150)
    print("->", os.path.join(out_dir, "ablation.md"), os.path.join(out_dir, "ablation.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.path.expanduser("~/OffMaps-data/IO-VNBD-sync"))
    ap.add_argument("--set", choices=["val", "lodo"])
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--out")
    ap.add_argument("--merge", nargs="+", help="partial result jsons of one set -> --out")
    ap.add_argument("--report", nargs="+", help="result jsons -> ablation.md + ablation.png next to the first")
    a = ap.parse_args()
    if a.report:
        return report(a.report, os.path.dirname(os.path.abspath(a.report[0])))
    if a.merge:
        J = [json.load(open(p)) for p in a.merge]
        raw = {k: v for j in J for k, v in j["per_outage"].items()}
        raw = {s: raw[s] for s in STAGES if s in raw}
        out = {**J[0], "per_outage": raw, "summary": summary({s: {int(D): p for D, p in r.items()} for s, r in raw.items()})}
        json.dump(out, open(a.out, "w"), indent=1)
        return print("merged ->", a.out, list(raw))
    from osm_layers import read_roads_bin
    roads = read_roads_bin(ROADS, with_class=True)
    raw = run_set(a.set, a.stages.split(","), a.data, roads)
    title = {"val": "Validation drives (2 drives, ~1 h)",
             "lodo": "Train drives, leave-one-drive-out (~9.5 h; every drive run with models that never saw it)"}[a.set]
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(dict(set=a.set, title=title, summary=summary(raw),
                       per_outage={s: {str(D): p for D, p in r.items()} for s, r in raw.items()}),
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
