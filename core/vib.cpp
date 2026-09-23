// libidr high-rate vibration / pothole front-end. See idr.h (Phase 7c).
//
// Streaming, O(1) per sample, no buffers -> runs at 200-400 Hz on a phone.
//   1. 1st-order high-pass per axis removes gravity + vehicle dynamics, leaving
//      the road-vibration band whose RMS is the speed cue.
//   2. an adaptive threshold (EMA baseline + k*scale, frozen during a shock so a
//      pothole can't poison its own baseline) flags shock transients; a
//      refractory window covers the ring-down.
//   3. per-window RMS is accumulated twice: rms_raw over all samples and
//      rms_clean with shock samples excised -- the latter is the pothole-robust
//      speed cue. Held in lockstep with py/vib_ref.py (the parity oracle).
#include "idr.h"
#include <cmath>

struct vib {
    double hz;
    double hp_fc, alpha;                 // high-pass cutoff + coefficient
    double ema_tau, beta;               // baseline EMA time constant + rate
    double shock_k;                     // threshold = m + shock_k * s
    int refractory_set, refractory;     // ring-down suppression (samples)
    double px[3], py[3];                // HPF state (prev in / prev out) per axis
    int have_prev;
    double m, s;                        // EMA mean + scale of AC magnitude
    int have_base;
    int warm;                           // warmup samples: build the baseline before detecting
    double clean_ss, raw_ss;            // window sums of squares
    long clean_cnt, cnt, shock_cnt;
    int events;

    void recompute() {
        double dt = 1.0 / hz;
        double tau_hp = 1.0 / (2 * M_PI * hp_fc);
        alpha = tau_hp / (tau_hp + dt);
        beta = 1.0 - std::exp(-dt / ema_tau);
        refractory_set = (int)std::lround(0.15 * hz);   // 150 ms ring-down
        warm = (int)std::lround(0.4 * hz);              // 400 ms baseline warmup
    }
    void defaults(double h) {
        hz = h; hp_fc = 3.0; ema_tau = 0.5; shock_k = 8.0;
        px[0]=px[1]=px[2]=py[0]=py[1]=py[2]=0; have_prev = 0;
        m = s = 0; have_base = 0; refractory = 0;
        clean_ss = raw_ss = 0; clean_cnt = cnt = shock_cnt = 0; events = 0;
        recompute();
    }
};

extern "C" {
vib* vib_create(double hz) { auto* v = new vib(); v->defaults(hz > 0 ? hz : 250.0); return v; }
void vib_destroy(vib* v) { delete v; }

void vib_set_params(vib* v, double shock_k, double refractory_s, double hp_fc, double ema_tau) {
    if (shock_k > 0) v->shock_k = shock_k;
    if (hp_fc > 0) v->hp_fc = hp_fc;
    if (ema_tau > 0) v->ema_tau = ema_tau;
    v->recompute();
    if (refractory_s > 0) v->refractory_set = (int)std::lround(refractory_s * v->hz);
}

int vib_push(vib* v, double ax, double ay, double az, double /*dt*/) {
    double x[3] = {ax, ay, az}, y[3];
    if (!v->have_prev) {
        for (int i = 0; i < 3; i++) { v->px[i] = x[i]; v->py[i] = 0.0; y[i] = 0.0; }
        v->have_prev = 1;
    } else {
        for (int i = 0; i < 3; i++) {
            y[i] = v->alpha * (v->py[i] + x[i] - v->px[i]);
            v->px[i] = x[i]; v->py[i] = y[i];
        }
    }
    double mag = std::sqrt(y[0]*y[0] + y[1]*y[1] + y[2]*y[2]);
    if (!v->have_base) { v->m = mag; v->s = 0.0; v->have_base = 1; }

    if (v->warm > 0) {                        // warmup: build the baseline, never fire
        v->warm--;
        double dev = std::fabs(mag - v->m);
        v->m += v->beta * (mag - v->m);
        v->s += v->beta * (dev - v->s);
        v->raw_ss += mag * mag; v->cnt++;
        v->clean_ss += mag * mag; v->clean_cnt++;
        return 0;
    }

    double s_floor = v->s > 0.05 ? v->s : 0.05;
    double thresh = v->m + v->shock_k * s_floor;
    int over = mag > thresh;
    int edge = 0, shock_sample;
    if (v->refractory > 0) {                 // inside ring-down: shock sample, baseline frozen
        shock_sample = 1; v->refractory--;
    } else if (over) {                        // leading edge of a new shock
        edge = 1; shock_sample = 1; v->events++; v->refractory = v->refractory_set;
    } else {                                  // quiet sample: update the baseline EMA
        shock_sample = 0;
        double dev = std::fabs(mag - v->m);
        v->m += v->beta * (mag - v->m);
        v->s += v->beta * (dev - v->s);
    }
    v->raw_ss += mag * mag; v->cnt++;
    if (shock_sample) v->shock_cnt++;
    else { v->clean_ss += mag * mag; v->clean_cnt++; }
    return edge;
}

void vib_window(vib* v, double out[4]) {
    out[0] = v->clean_cnt > 0 ? std::sqrt(v->clean_ss / v->clean_cnt) : 0.0;
    out[1] = v->cnt > 0 ? std::sqrt(v->raw_ss / v->cnt) : 0.0;
    out[2] = v->cnt > 0 ? (double)v->shock_cnt / v->cnt : 0.0;
    out[3] = v->events;
    v->clean_ss = v->raw_ss = 0; v->clean_cnt = v->cnt = v->shock_cnt = 0; v->events = 0;
}
}
