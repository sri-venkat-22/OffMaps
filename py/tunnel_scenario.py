"""Hyderabad tunnel scenario: the Mindspace Underpass (HITEC City) with REAL phone data.

    PYTHONPATH=. python3 tunnel_scenario.py [--sets val,lodo] [--out ../out/tunnel]

A purely synthetic IMU cannot test this system: SpeedNet reads real phone vibration, which
no simulator here reproduces (on data/synth_rig.py drives it is 25-35 % off). So each
realization is a ROUTE TRANSPLANT of a real IO-VNBD drive segment:

  kept, as recorded   the phone's accelerometer + gyroscope (vibration, bias, noise, mount),
                      and the car's real speed profile;
  replaced            the geometry only: the car's true turn rate is taken out of the yaw
                      gyro and the Hyderabad route's turn rate put in (the centripetal term
                      of the lateral accelerometer likewise), and the reference position
                      runs along a real OSM route through the underpass at the real speed.

GNSS (1 Hz, reference position + Doppler speed + course, as in the IO-VNBD evaluation) is
removed over (a) the underpass itself and (b) the PS benchmark: 1 km centred on it. The
engine is the edge engine (the phone's loop) in the four ablation stages, with road
matching on the real Hyderabad network. Realizations: distinct stretches of the val
drives (shipped models) and of the train drives run leave-one-drive-out (models that
never saw them) where the car cruises through the GNSS-denied span (mean 45-75 km/h, never
below 30 km/h from 10 s before the entrance), after >= 3 min of GNSS.

CONTROL: every realization is also run untransplanted -- the same real stretch and outage
times on the drive's own road (and the Coventry map). If the transplant were distorting
the problem, the two drift distributions would differ; they are reported side by side. Both carriageways. The test split is not used.

The Mindspace Underpass is the longest road tunnel in the Hyderabad OSM extract (~360 m
per carriageway); Hyderabad has no 1 km road tunnel, so the 1 km case denies GNSS over
the underpass and the high-rise corridor either side of it.
"""
from __future__ import annotations
import argparse, heapq, json, math, os, re, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HYD = os.path.join(HERE, "..", "map", "hyderabad", "roads.bin")
K = math.pi / 180 * 6_371_000.0
# the underpass carriageways (OSM, name "Mindspace Underpass", tunnel=yes, oneway, 3 lanes)
UNDERPASS = {"northbound": ((17.4383991, 78.3760509), (17.4411629, 78.3771336)),
             "southbound": ((17.4411647, 78.3772629), (17.4382950, 78.3761805))}
LAT0, LON0 = 17.4397, 78.3766                  # scenario frame origin (underpass centre)
BBOX_M = 7000.0
WARM, AFTER = 190.0, 30.0                      # s of GNSS before the outage / kept after it
V_MIN, V_MAX, V_FLOOR = 45 / 3.6, 75 / 3.6, 30 / 3.6   # cruising: mean 45-75, never < 30 km/h
LEAD = 10.0                                    # s before the entrance that must also be >= V_FLOOR
CORRIDOR_M = 1000.0
ROAD_COST = {"motorway": 0.8, "trunk": 0.8, "primary": 0.9, "secondary": 1.0, "tertiary": 1.2}


def en(lat, lon):
    return (np.asarray(lon) - LON0) * math.cos(math.radians(LAT0)) * K, (np.asarray(lat) - LAT0) * K


def ll(e, n):
    return LAT0 + np.asarray(n) / K, LON0 + np.asarray(e) / (math.cos(math.radians(LAT0)) * K)


# ---------------- the route ----------------
def load_ways():
    from osm_layers import read_roads_bin
    out = []
    for la, lo, tun, one, cls in read_roads_bin(HYD, with_class=True):
        e, n = en(la, lo)
        if np.abs(e).min() > BBOX_M or np.abs(n).min() > BBOX_M or cls in (None, "service"):
            continue
        out.append((la, lo, e, n, tun, one, cls))
    return out


def graph(ways):
    """Directed node graph: node key = exact 1e-7 deg coordinates; edges carry geometry."""
    adj, xy = defaultdict(list), {}
    for la, lo, e, n, tun, one, cls in ways:
        keys = [(int(round(a * 1e7)), int(round(b * 1e7))) for a, b in zip(la, lo)]
        for i in range(len(keys) - 1):
            L = math.hypot(e[i + 1] - e[i], n[i + 1] - n[i])
            if L == 0:
                continue
            w = L * ROAD_COST.get((cls or "").replace("_link", ""), 1.6)
            xy[keys[i]] = (e[i], n[i]); xy[keys[i + 1]] = (e[i + 1], n[i + 1])
            adj[keys[i]].append((keys[i + 1], w, L, tun))
            if not one:
                adj[keys[i + 1]].append((keys[i], w, L, tun))
    return adj, xy


def dijkstra(adj, src, reverse=False):
    if reverse:
        radj = defaultdict(list)
        for u, es in adj.items():
            for v, w, L, t in es:
                radj[v].append((u, w, L, t))
        adj = radj
    dist, prev, length = {src: 0.0}, {}, {src: 0.0}
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w, L, t in adj.get(u, ()):
            if d + w < dist.get(v, math.inf):
                dist[v] = d + w; prev[v] = u; length[v] = length[u] + L
                heapq.heappush(pq, (d + w, v))
    return dist, prev, length


def nearest(xy, e, n):
    return min(xy, key=lambda k: (xy[k][0] - e) ** 2 + (xy[k][1] - n) ** 2)


def build_route(direction, adj, xy, before_m=5200.0, after_m=1800.0):
    """Main-road route: ~before_m of driving into the underpass, through it, ~after_m out."""
    (la0, lo0), (la1, lo1) = UNDERPASS[direction]
    a = nearest(xy, *en(la0, lo0)); b = nearest(xy, *en(la1, lo1))
    u = np.subtract(xy[b], xy[a]); u = u / np.linalg.norm(u)          # direction of travel
    # never through the other carriageway: drop tunnel edges near the underpass that run against u
    against = lambda k, e: e[3] and np.dot(np.subtract(xy[e[0]], xy[k]), u) < 0 and np.hypot(*xy[k]) < 600
    adj = {k: [e for e in es if not against(k, e)] for k, es in adj.items()}
    du, pu, lu = dijkstra(adj, a, reverse=True)       # upstream tree: prev points downstream
    behind = lambda k: np.dot(np.subtract(xy[k], xy[a]), u) < -0.5 * np.hypot(*np.subtract(xy[k], xy[a]))
    # start: behind the entry, ~before_m of driving away, reached by the cheapest (main) roads
    start = min((k for k in lu if behind(k)), key=lambda k: abs(lu[k] - before_m) + 0.15 * du[k])
    up = [start]
    while up[-1] != a:
        up.append(pu[up[-1]])
    _, pm, _ = dijkstra(adj, a)
    mid = [b]
    while mid[-1] != a:
        mid.append(pm[mid[-1]])
    mid = mid[::-1]
    dd, pd, ld = dijkstra(adj, b)
    ahead = lambda k: np.dot(np.subtract(xy[k], xy[b]), u) > 0.5 * np.hypot(*np.subtract(xy[k], xy[b]))
    cand = {k: v for k, v in ld.items() if abs(v - after_m) < 400 and ahead(k)}
    end = min(cand, key=lambda k: dd[k]) if cand else max(ld, key=ld.get)
    down = [end]
    while down[-1] != b:
        down.append(pd[down[-1]])
    nodes = up + mid[1:] + down[::-1][1:]
    P = np.array([xy[k] for k in nodes])
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    s_in = float(s[len(up) - 1]); s_out = float(s[len(up) + len(mid) - 2])
    return P, s, s_in, s_out


class Route:
    def __init__(self, P, s, s_in, s_out):
        self.P, self.s, self.s_in, self.s_out = P, s, s_in, s_out
        # heading vs arc length, smoothed over ~8 m so polyline vertices are not turn impulses
        ds = 1.0
        self.g = np.arange(0, s[-1], ds)
        e = np.interp(self.g, s, P[:, 0]); n = np.interp(self.g, s, P[:, 1])
        h = np.unwrap(np.arctan2(np.gradient(e), np.gradient(n)))
        ker = np.exp(-0.5 * (np.arange(-24, 25) / 8.0) ** 2); ker /= ker.sum()
        pad = np.r_[np.full(24, h[0]), h, np.full(24, h[-1])]
        self.h = np.convolve(pad, ker, mode="valid")
        self.k = np.gradient(self.h, ds)                   # curvature (rad/m), compass sense

    def at(self, sq):
        return (np.interp(sq, self.s, self.P[:, 0]), np.interp(sq, self.s, self.P[:, 1]),
                np.interp(sq, self.g, self.h), np.interp(sq, self.g, self.k))


# ---------------- transplant ----------------
def true_yaw_rate(d):
    h = np.unwrap(d.heading)
    r = np.gradient(h, d.t)
    ker = np.ones(10) / 10
    r = np.convolve(r, ker, mode="same")
    r[d.speed < 1.0] = 0.0
    return r


def windows(d, route, c0, c1):
    """Real sample indices where a GNSS-denied traverse of [c0, c1] (route metres) can start:
    45-75 km/h mean, never below 30 km/h from LEAD s before the entrance to the exit (cruising
    traffic, as in the PS example), >= WARM s of real driving before it that fits on
    the route before c0, >= AFTER s after it that fits after c1; distinct traverses do not
    overlap in real time."""
    span = c1 - c0
    cum = np.r_[0, np.cumsum(0.5 * (d.speed[1:] + d.speed[:-1]) * np.diff(d.t))]
    out, last_end = [], -1.0
    for i0 in range(0, len(d.t), 10):
        t0 = d.t[i0]
        if t0 - d.t[0] < WARM or t0 < last_end:
            continue
        j = np.searchsorted(cum, cum[i0] + span)
        ja = np.searchsorted(d.t, d.t[min(j, len(d.t) - 1)] + AFTER)
        if ja >= len(d.t):
            break
        seg = d.speed[np.searchsorted(d.t, t0 - LEAD):j]
        T = d.t[j] - t0
        if not (len(seg) and V_MIN <= span / T <= V_MAX and seg.min() >= V_FLOOR):
            continue
        before = cum[i0] - cum[np.searchsorted(d.t, t0 - WARM)]
        after = cum[ja] - cum[j]
        if before > c0 - 5 or after > route.s[-1] - c1 - 5:
            continue                                        # the route is too short for this stretch
        out.append(i0); last_end = d.t[j]
    return out, cum


def transplant(d, route, cum, i0, s_start):
    """Streams (edge_engine.run inputs) for real samples [w0, w1) mapped onto the route so
    that sample i0 is at arc length s_start."""
    w0 = np.searchsorted(d.t, d.t[i0] - WARM)
    w1 = min(len(d.t), int(np.searchsorted(cum, cum[i0] + route.s[-1] - s_start)))
    sl = slice(w0, w1)
    s = s_start + cum[sl] - cum[i0]
    ok = (s >= 0) & (s <= route.s[-1])
    t = d.t[sl][ok]; s = s[ok]; v = d.speed[sl][ok]
    e, n, h, kap = route.at(s)
    r_route = kap * v                                         # compass yaw rate on the route
    r_true = true_yaw_rate(d)[sl][ok]
    acc = d.acc[sl][ok].copy(); gyr = d.gyro[sl][ok].copy()
    gyr[:, 2] += r_route - r_true                             # the car's turns out, the route's in
    acc[:, 1] += -v * (r_route - r_true)                      # centripetal, vehicle-left axis
    gyr[:, 2] = -gyr[:, 2]                                    # compass yaw -> device CCW (phase8_eval.streams)
    lat, lon = ll(e, n)
    imu = dict(t=t, acc=acc, gyro=gyr, mag=np.full((len(t), 3), np.nan))
    i = np.arange(0, len(t), 10); m = len(i)
    gn = dict(t=t[i], lat=lat[i], lon=lon[i], speed=v[i], bearing=np.degrees(h[i]) % 360,
              cn0=np.full(m, 42.0), sv=np.full(m, 9), navic=np.full(m, 3), masked=np.zeros(m))
    return imu, gn, dict(t=t, e=e, n=n, s=s, v=v)


# ---------------- run ----------------
def run_set(which, stages, data, routes, hyd_ways, cases, folds=None):
    from data.iovnbd_sync import load_sync_dir
    from model.train_real import split_drives
    from model.fusion_head import load_head, FOLDS
    from edge_engine import run, TorchSpeedNet
    from ablation import factory, ROADS
    from osm_layers import read_roads_bin
    from phase8_eval import streams
    cov = read_roads_bin(ROADS, with_class=True)            # the control runs on the drive's own roads
    split = split_drives(load_sync_dir(data, verbose=False))
    if which == "val":
        head = load_head(os.path.join(HERE, "model", "fusion_head.pt"))
        groups = [(split["val"], head, None, "val")]
    else:
        oof = os.path.join(HERE, "model", "oof")
        fold_of = lambda d: [k for k, rx in FOLDS.items() if re.search(rx, d.vehicle_id.split("#")[0])][0]
        groups = [([d for d in split["train"] if fold_of(d) == fk], load_head(os.path.join(oof, f"fusion_head_oof_{fk}.pt")),
                   TorchSpeedNet(os.path.join(oof, f"nn_oof_{fk}.pt")), f"lodo:{fk}") for fk in FOLDS
                  if folds is None or fk in folds]
    rows = []
    for drives, head, net, tag in groups:
        for d in drives:
            imu0, gn0 = streams(d)                               # the untouched drive, for the control
            for case, (lo_off, hi_off) in cases.items():
                for direction, route in routes.items():
                    c0 = route.s_in + lo_off(route); c1 = route.s_in + hi_off(route)
                    idx, cum = windows(d, route, c0, c1)
                    for i0 in idx:
                        imu, gn, tr = transplant(d, route, cum, i0, c0)
                        a = float(np.interp(c0, tr["s"], tr["t"])); b = float(np.interp(c1, tr["s"], tr["t"]))
                        row = dict(set=tag, drive=d.vehicle_id, case=case, direction=direction, t_real=float(d.t[i0] - d.t[0]),
                                   span_m=float(c1 - c0), secs=b - a, kmh=3.6 * (c1 - c0) / (b - a), stages={})
                        for sname in stages:
                            o = run(factory(sname, head, hyd_ways, net)(), imu, gn, [(a, b)])
                            j = np.searchsorted(tr["t"], b) - 1
                            # the engine's ENU origin is the first fix: compare in lat/lon -> scenario frame
                            lat_o, lon_o = ll_engine(o, gn)
                            eo, no = en(lat_o, lon_o)
                            ex = np.hypot(eo - tr["e"], no - tr["n"])
                            ia = np.searchsorted(tr["t"], a)
                            # control: the same real stretch and outage times on the drive's ORIGINAL road
                            ci, cg = crop(imu0, gn0, tr["t"][0], tr["t"][-1])
                            oc = run(factory(sname, head, cov, net)(), ci, cg, [(a, b)])
                            # the engine's frame starts at the crop's first fix: shift into the drive's frame
                            f0 = np.searchsorted(d.t, cg["t"][0])
                            jc = np.searchsorted(d.t, b) - 1
                            kc = np.searchsorted(ci["t"], d.t[jc])
                            ec = float(np.hypot(oc[kc, 1] + d.e[f0] - d.e[jc], oc[kc, 2] + d.n[f0] - d.n[jc]))
                            dc = float(np.trapezoid(d.speed[np.searchsorted(d.t, a):jc], d.t[np.searchsorted(d.t, a):jc]))
                            row["stages"][sname] = dict(end_err=float(ex[j]), max_err=float(np.nanmax(ex[ia:j + 1])),
                                                        drift=float(100 * ex[j] / (c1 - c0)),
                                                        control_err=ec, control_drift=float(100 * ec / dc))
                        rows.append(row)
                        print(f"{tag:8s} {case:9s} {direction:10s} {row['kmh']:4.0f} km/h  " +
                              "  ".join(f"{s} {row['stages'][s]['end_err']:6.1f} m" for s in stages), flush=True)
    return rows


def crop(imu, gn, t0, t1):
    """The original streams between t0 and t1 (inclusive): the control starts cold exactly when
    the transplanted run does."""
    mi = (imu["t"] >= t0) & (imu["t"] <= t1); mg = (gn["t"] >= t0) & (gn["t"] <= t1)
    return {k: v[mi] for k, v in imu.items()}, {k: np.asarray(v)[mg] for k, v in gn.items()}


def ll_engine(o, gn):
    """Engine ENU (origin = its first GNSS fix) -> lat/lon."""
    lat0, lon0 = gn["lat"][0], gn["lon"][0]
    return lat0 + o[:, 2] / K, lon0 + o[:, 1] / (math.cos(math.radians(lat0)) * K)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.environ.get("OFFMAPS_IOVNBD", os.path.expanduser("~/OffMaps-data/IO-VNBD-sync")))
    ap.add_argument("--sets", default="val,lodo")
    ap.add_argument("--stages", default="physics,speednet,head,map")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "out", "tunnel"))
    ap.add_argument("--cases", default="underpass,1km")
    ap.add_argument("--folds", help="lodo only: comma-separated fold keys (to run folds in parallel)")
    ap.add_argument("--tag", default="", help="suffix for the output file")
    ap.add_argument("--merge", nargs="+", help="result jsons of one set -> --out/<set>.json, then summarise")
    ap.add_argument("--report", action="store_true", help="--out/*.json -> --out/summary.md")
    a = ap.parse_args()
    if a.report:
        return report(a.out, a.stages.split(","))
    if a.merge:
        J = [json.load(open(p)) for p in a.merge]
        rows = [r for j in J for r in j["rows"]]
        out = {**J[0], "rows": rows}
        with open(os.path.join(a.out, f"{J[0]['set']}.json"), "w") as f:
            json.dump(out, f, indent=1)
        return summarise(rows, a.stages.split(","))
    ways = load_ways()
    adj, xy = graph(ways)
    routes = {k: Route(*build_route(k, adj, xy)) for k in UNDERPASS}
    for k, r in routes.items():
        print(f"route {k}: {r.s[-1] / 1000:.1f} km, underpass {r.s_out - r.s_in:.0f} m at {r.s_in / 1000:.2f} km", flush=True)
    # the matcher's roads (edge engine form): every class incl. service (the preset drops those
    # itself), within 800 m of either route -- the phone would hold a 1 km window anyway
    from osm_layers import read_roads_bin
    RP = np.vstack([r.P for r in routes.values()])
    cell = {(int(x // 400), int(y // 400)) for x, y in RP}
    near = {(cx + i, cy + j) for cx, cy in cell for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2)}
    hyd = []
    for la, lo, t, o, c in read_roads_bin(HYD, with_class=True):
        e, n = en(la, lo)
        if np.abs(e).min() < BBOX_M and np.abs(n).min() < BBOX_M and \
                any((int(x // 400), int(y // 400)) in near for x, y in zip(e, n)):
            hyd.append((la, lo, t, o, c))
    print(f"matcher roads: {len(hyd)} pieces", flush=True)
    cases = {"underpass": (lambda r: 0.0, lambda r: r.s_out - r.s_in),
             "1km": (lambda r: (r.s_out - r.s_in) / 2 - CORRIDOR_M / 2, lambda r: (r.s_out - r.s_in) / 2 + CORRIDOR_M / 2)}
    cases = {k: v for k, v in cases.items() if k in a.cases.split(",")}
    os.makedirs(a.out, exist_ok=True)
    geo = {k: dict(e=[round(float(x), 1) for x in r.P[:, 0]], n=[round(float(x), 1) for x in r.P[:, 1]],
                   s_in=r.s_in, s_out=r.s_out) for k, r in routes.items()}
    for which in a.sets.split(","):
        rows = run_set(which, a.stages.split(","), a.data, routes, hyd, cases,
                       a.folds.split(",") if a.folds else None)
        with open(os.path.join(a.out, f"{which}{a.tag}.json"), "w") as f:
            json.dump(dict(set=which, frame=dict(lat0=LAT0, lon0=LON0), routes=geo, rows=rows), f, indent=1)
        summarise(rows, a.stages.split(","))


def summarise(rows, stages):
    for case in ("underpass", "1km"):
        R = [r for r in rows if r["case"] == case]
        if not R:
            continue
        print(f"\n{case}: {len(R)} realizations, {np.mean([r['span_m'] for r in R]):.0f} m at "
              f"{np.median([r['kmh'] for r in R]):.0f} km/h median")
        for s in stages:
            e = np.array([r["stages"][s]["end_err"] for r in R]); p = np.array([r["stages"][s]["drift"] for r in R])
            pc = np.array([r["stages"][s]["control_drift"] for r in R])
            print(f"  {s:9s} exit error median {np.median(e):6.1f} m  p90 {np.percentile(e, 90):6.1f} m  "
                  f"drift median {np.median(p):5.1f} %  <10 %: {100 * np.mean(p < 10):3.0f} %"
                  + (f"  <100 m: {100 * np.mean(e < 100):3.0f} %" if case == "1km" else "")
                  + f"   | control (original road) drift median {np.median(pc):5.1f} %")



def report(out, stages):
    from ablation import LABEL
    lines = []
    for which, title in (("val", "Validation drives, the shipped models"),
                         ("lodo", "Train drives, each run with models that never saw it")):
        p = os.path.join(out, f"{which}.json")
        if not os.path.exists(p):
            continue
        rows = json.load(open(p))["rows"]
        for case, name in (("1km", "1 km, the PS benchmark"), ("underpass", "the underpass itself")):
            R = [r for r in rows if r["case"] == case]
            if not R:
                continue
            lines += [f"**{title}: {name}** ({len(R)} runs, {np.mean([r['span_m'] for r in R]):.0f} m without GNSS at "
                      f"{np.median([r['kmh'] for r in R]):.0f} km/h median)", "",
                      "| stage | exit error, median | p90 | drift, median | under 10 % | under 100 m | same stretches on their own road: drift, median |",
                      "|---|--:|--:|--:|--:|--:|--:|"]
            for s in stages:
                e = np.array([r["stages"][s]["end_err"] for r in R]); d = np.array([r["stages"][s]["drift"] for r in R])
                c = np.array([r["stages"][s]["control_drift"] for r in R])
                lines.append(f"| {LABEL[s]} | {np.median(e):.0f} m | {np.percentile(e, 90):.0f} m | {np.median(d):.1f} % | "
                             f"{100 * np.mean(d < 10):.0f} % | {100 * np.mean(e < 100):.0f} % | {np.median(c):.1f} % |")
            lines.append("")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
