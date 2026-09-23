// Offline map matching. Roads are polylines (from OSM, loaded once). Given the
// current estimate, find the best road, project onto it, and hand back a
// CROSS-TRACK-ONLY pseudo-measurement + the road bearing.
//
// Why cross-track only: the foot of the perpendicular has zero along-track
// residual by construction, so the plan's anisotropic (sigma_cross=1.5,
// sigma_along=50) update reduces to one cross-track scalar. A road pins the
// lane, never the distance travelled -- exactly what we want.
//
// Corridor: a road tagged tunnel, or the sole candidate with no junction in the
// horizon, switches to 1-D mode (tighter cross-track, bearing locked).
#include "idr.h"
#include <cmath>
#include <vector>
#include <algorithm>

struct Way { std::vector<double> e, n, cum; int tunnel, oneway; };  // cum = arc length to each vertex
struct mm {
    std::vector<Way> ways;
    double bearing_gate = M_PI / 4;
    // --- Phase 7b: HMM/Viterbi ---
    std::vector<std::vector<char>> adj;    // symmetric way adjacency; empty => all-connected
    int has_edges = 0;
    double sigma_emit = 8.0;               // emission: cross-track distance scale (m)
    double sigma_trans = 12.0;             // transition: GPS-vs-route distance-consistency scale (m)
    double beta_bearing = 4.0;             // emission: weight on (1 - cos bearing mismatch)
    bool adjacent(int a, int b) const {
        if (a == b) return true;
        if (!has_edges) return true;
        if (a < 0 || b < 0 || a >= (int)adj.size() || b >= (int)adj.size()) return false;
        return adj[a][b] != 0;
    }
};

namespace {
// closest point on segment AB to P; returns foot + segment tangent bearing + dist2
void proj_seg(double px, double py, double ax, double ay, double bx, double by,
              double& fx, double& fy, double& brg, double& d2) {
    double dx = bx - ax, dy = by - ay, L2 = dx*dx + dy*dy + 1e-12;
    double t = ((px-ax)*dx + (py-ay)*dy) / L2; t = t < 0 ? 0 : t > 1 ? 1 : t;
    fx = ax + t*dx; fy = ay + t*dy;
    brg = std::atan2(dx, dy);                 // 0 = north, +east (matches state psi)
    d2 = (px-fx)*(px-fx) + (py-fy)*(py-fy);
}
double wrap(double a){ while(a>M_PI)a-=2*M_PI; while(a<=-M_PI)a+=2*M_PI; return a; }

// Project P onto a whole way: best foot + tangent bearing + dist^2 + arc length to foot.
void proj_way(const Way& w, double px, double py,
              double& fe, double& fn, double& brg, double& d2, double& arclen) {
    d2 = 1e18; fe = px; fn = py; brg = 0; arclen = 0;
    for (size_t i = 0; i + 1 < w.e.size(); i++) {
        double dx = w.e[i+1]-w.e[i], dy = w.n[i+1]-w.n[i], seg = std::sqrt(dx*dx+dy*dy);
        double L2 = dx*dx + dy*dy + 1e-12;
        double t = ((px-w.e[i])*dx + (py-w.n[i])*dy) / L2; t = t < 0 ? 0 : t > 1 ? 1 : t;
        double fx = w.e[i] + t*dx, fy = w.n[i] + t*dy;
        double dd = (px-fx)*(px-fx) + (py-fy)*(py-fy);
        if (dd < d2) { d2 = dd; fe = fx; fn = fy; brg = std::atan2(dx, dy); arclen = w.cum[i] + t*seg; }
    }
}

struct Cand { int way; double fe, fn, brg, cross, arclen; int corridor; };
constexpr double SEQ_GATE2 = 2500.0;   // candidate if within 50 m of a way
constexpr double SWITCH_PEN = 1.0;     // cost to change way across a junction (adjacent)
constexpr double FAR_PEN = 50.0;       // extra cost to hop between non-adjacent ways
}

extern "C" {
mm* mm_create(void) { return new mm(); }
void mm_destroy(mm* m) { delete m; }
void mm_add_way(mm* m, const double* e, const double* n, int npts, int tunnel, int oneway) {
    Way w; w.tunnel = tunnel; w.oneway = oneway;
    w.e.assign(e, e + npts); w.n.assign(n, n + npts);
    w.cum.resize(npts);
    w.cum[0] = 0.0;
    for (int i = 1; i < npts; i++)
        w.cum[i] = w.cum[i-1] + std::hypot(w.e[i]-w.e[i-1], w.n[i]-w.n[i-1]);
    m->ways.push_back(std::move(w));
}

// Match (e,n,psi). Fills out[6] = {matched, foot_e, foot_n, road_bearing, cross_dist, corridor}
// where cross_dist is signed (left-positive) distance from estimate to the road.
void mm_match(const mm* m, double e, double n, double psi, double out[6]) {
    double best_d2 = 1e18, fe = 0, fn = 0, brg = 0; int hits = 0, tunnel = 0;
    for (const auto& w : m->ways) {
        for (size_t i = 0; i + 1 < w.e.size(); i++) {
            double f_x, f_y, b, d2;
            proj_seg(e, n, w.e[i], w.n[i], w.e[i+1], w.n[i+1], f_x, f_y, b, d2);
            // bearing gate: reject roads not aligned with travel (opposite lane, cross streets)
            double db = std::fabs(wrap(b - psi));
            if (db > m->bearing_gate && db < M_PI - m->bearing_gate) continue;  // allow +/-180 (oneway sign)
            if (d2 < 900.0) hits++;                    // candidates within 30 m
            if (d2 < best_d2) { best_d2 = d2; fe = f_x; fn = f_y; brg = b; tunnel = w.tunnel; }
        }
    }
    if (best_d2 > 1e17) { for (int i = 0; i < 6; i++) out[i] = 0; return; }
    // signed cross-track: project (foot-est) onto the road normal (left = +)
    double nx = -std::cos(brg), ny = std::sin(brg);   // left-normal of bearing
    double cross = (fe - e) * nx + (fn - n) * ny;
    int corridor = (tunnel || hits <= 1) ? 1 : 0;
    out[0] = 1; out[1] = fe; out[2] = fn; out[3] = brg; out[4] = cross; out[5] = corridor;
}

void mm_add_edge(mm* m, int a, int b) {
    int W = (int)m->ways.size();
    if (a < 0 || b < 0 || a >= W || b >= W) return;
    if ((int)m->adj.size() != W) m->adj.assign(W, std::vector<char>(W, 0));
    m->adj[a][b] = m->adj[b][a] = 1;
    m->has_edges = 1;
}
void mm_set_hmm(mm* m, double sigma_emit, double sigma_trans, double beta_bearing) {
    if (sigma_emit > 0) m->sigma_emit = sigma_emit;
    if (sigma_trans > 0) m->sigma_trans = sigma_trans;
    if (beta_bearing >= 0) m->beta_bearing = beta_bearing;
}

// Per-step candidates: one per way whose nearest point is within the gate and
// whose bearing agrees with travel (allowing +/-180 for the opposite lane).
static std::vector<Cand> seq_cands(const mm* m, double e, double n, double psi) {
    std::vector<Cand> cs;
    for (int j = 0; j < (int)m->ways.size(); j++) {
        const Way& w = m->ways[j];
        double fe, fn, brg, d2, arclen;
        proj_way(w, e, n, fe, fn, brg, d2, arclen);
        if (d2 > SEQ_GATE2) continue;
        double db = std::fabs(wrap(brg - psi));
        if (db > m->bearing_gate && db < M_PI - m->bearing_gate) continue;
        double nx = -std::cos(brg), ny = std::sin(brg);            // left-normal (signed cross)
        double cross = (fe - e) * nx + (fn - n) * ny;
        cs.push_back(Cand{j, fe, fn, brg, cross, arclen, w.tunnel ? 1 : 0});
    }
    return cs;
}

static double emit_cost(const mm* m, const Cand& c, double psi) {
    double db = wrap(c.brg - psi);
    double align = m->ways[c.way].oneway ? std::cos(db) : std::fabs(std::cos(db));
    double dcross = c.cross / m->sigma_emit;
    return 0.5 * dcross * dcross + m->beta_bearing * (1.0 - align);
}

// Viterbi over the whole trajectory. Returns #steps decoded; way_out[i] is the
// chosen way index (-1 if no candidate that step). Optional arrays get the foot.
int mm_match_seq(const mm* m, const double* e, const double* n, const double* psi,
                 int nsteps, int* way_out, double* foot_e, double* foot_n,
                 double* bearing_out, int* corridor_out) {
    if (nsteps <= 0) return 0;
    const double NULL_EMIT = 0.5 * (SEQ_GATE2 / (m->sigma_emit * m->sigma_emit))
                             + m->beta_bearing;                    // cost of "off any road"
    std::vector<std::vector<Cand>> cand(nsteps);
    std::vector<std::vector<double>> cost(nsteps);
    std::vector<std::vector<int>> back(nsteps);
    for (int i = 0; i < nsteps; i++) {
        cand[i] = seq_cands(m, e[i], n[i], psi[i]);
        cand[i].push_back(Cand{-1, e[i], n[i], psi[i], 0.0, 0.0, 0});   // null state (off-road)
        int C = (int)cand[i].size();
        cost[i].assign(C, 0.0); back[i].assign(C, -1);
        for (int c = 0; c < C; c++) {
            double em = cand[i][c].way < 0 ? NULL_EMIT : emit_cost(m, cand[i][c], psi[i]);
            if (i == 0) { cost[i][c] = em; continue; }
            double dgps = std::hypot(e[i]-e[i-1], n[i]-n[i-1]);
            double best = 1e300; int bp = -1;
            for (int p = 0; p < (int)cand[i-1].size(); p++) {
                const Cand& P = cand[i-1][p]; const Cand& Cc = cand[i][c];
                double base, droute;
                if (P.way < 0 || Cc.way < 0) { base = SWITCH_PEN; droute = dgps; }
                else if (P.way == Cc.way)    { base = 0.0; droute = std::fabs(Cc.arclen - P.arclen); }
                else if (m->adjacent(P.way, Cc.way)) { base = SWITCH_PEN; droute = std::hypot(Cc.fe-P.fe, Cc.fn-P.fn); }
                else { base = SWITCH_PEN + FAR_PEN; droute = std::hypot(Cc.fe-P.fe, Cc.fn-P.fn); }
                double dc = (dgps - droute) / m->sigma_trans;
                double tc = base + 0.5 * dc * dc;
                double tot = cost[i-1][p] + tc;
                if (tot < best) { best = tot; bp = p; }               // strict: lowest index wins ties
            }
            cost[i][c] = em + best; back[i][c] = bp;
        }
    }
    // backtrack from the best terminal state
    int c = 0; double best = 1e300;
    for (int k = 0; k < (int)cost[nsteps-1].size(); k++)
        if (cost[nsteps-1][k] < best) { best = cost[nsteps-1][k]; c = k; }
    std::vector<int> chosen(nsteps);
    for (int i = nsteps - 1; i >= 0; i--) { chosen[i] = c; if (i) c = back[i][c]; }
    for (int i = 0; i < nsteps; i++) {
        const Cand& s = cand[i][chosen[i]];
        int ncand = (int)cand[i].size() - 1;                          // minus the null state
        if (way_out)      way_out[i] = s.way;
        if (foot_e)       foot_e[i] = s.fe;
        if (foot_n)       foot_n[i] = s.fn;
        if (bearing_out)  bearing_out[i] = s.brg;
        if (corridor_out) corridor_out[i] = (s.way >= 0 && (s.corridor || ncand <= 1)) ? 1 : 0;
    }
    return nsteps;
}
}
