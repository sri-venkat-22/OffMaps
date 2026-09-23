// Doppler speed-scale self-calibration. Improvement #1 -- the mechanism that
// kills the IO-VNBD->phone + vehicle-to-vehicle + tyre/load domain gaps online,
// with no retraining. Deliberately OUTSIDE the EKF (a rolling-buffer LS, not
// filter states): k,c are badly correlated at constant speed, so we gate on
// speed excitation and only fit the offset when the buffer actually varies.
//
// Errors-in-variables: the regressor here is the NN speed v_nn, which is itself
// noisy (~1-2 m/s per second). Ordinary LS of Doppler-on-v_nn assumes v_nn is
// exact, so its slope k attenuates toward the noise floor -- it under-recovers
// the true scale by exactly the amount of NN noise (regression dilution). With
// an honest NN sigma we can do better: Deming regression accounts for noise in
// BOTH variables via lambda = var(err_doppler) / var(err_v_nn). Set lambda from
// the NN sigma (spc_set_lambda) and k is recovered without dilution. lambda left
// at its default +inf reproduces the old OLS fit bit-for-bit.
#include "idr.h"
#include <cmath>
#include <cstring>

struct spc {
    static constexpr int CAP = 600;      // 60 s at 10 Hz
    double vn[CAP], vd[CAP], w[CAP];
    int n = 0, head = 0;
    double k = 1.0, c = 0.0;             // frozen fit: v_true ~= k*v_nn + c
    double lam = INFINITY;               // Deming noise-variance ratio; +inf = OLS
    int excited = 0;
};

// Deming slope for y = k*x, given (weighted) second moments Sxx, Syy, Sxy and
// lambda = var(err_y)/var(err_x). lambda -> +inf recovers OLS slope Sxy/Sxx.
static double deming_k(double Sxx, double Syy, double Sxy, double lam) {
    if (!std::isfinite(lam)) return Sxy / Sxx;          // OLS limit
    if (std::fabs(Sxy) < 1e-12) return Sxy / Sxx;       // degenerate: no covariance
    double t = Syy - lam * Sxx;
    return (t + std::sqrt(t * t + 4.0 * lam * Sxy * Sxy)) / (2.0 * Sxy);
}

extern "C" {
spc* spc_create(void) { return new spc(); }
void spc_destroy(spc* s) { delete s; }

// Push a paired sample while GNSS is healthy (v_nn = model speed, v_dop = Doppler).
void spc_push(spc* s, double v_nn, double v_dop, double weight) {
    s->vn[s->head] = v_nn; s->vd[s->head] = v_dop; s->w[s->head] = weight;
    s->head = (s->head + 1) % spc::CAP;
    if (s->n < spc::CAP) s->n++;
}

// Errors-in-variables ratio for the next fit: lambda = var(err_doppler)/var(err_v_nn).
// Doppler noise is small and roughly constant; err_v_nn is the calibrated NN sigma.
// Pass a non-finite value (or never call this) to keep the plain OLS fit.
void spc_set_lambda(spc* s, double lambda) { s->lam = lambda; }

// Weighted LS/Deming fit over the buffer. Excitation test decides 2-param vs
// scale-only; lambda decides OLS vs Deming (errors-in-variables).
int spc_fit(spc* s, double* k_out, double* c_out) {
    if (s->n < 30) { *k_out = s->k; *c_out = s->c; return 0; }
    double sw = 0, swx = 0, swy = 0, swxx = 0, swyy = 0, swxy = 0;
    for (int i = 0; i < s->n; i++) {
        double x = s->vn[i], y = s->vd[i], w = s->w[i];
        sw += w; swx += w * x; swy += w * y;
        swxx += w * x * x; swyy += w * y * y; swxy += w * x * y;
    }
    if (!(sw > 1e-12)) { *k_out = s->k; *c_out = s->c; return 0; }   // all weights 0: keep the fit
    double mean = swx / sw, var = swxx / sw - mean * mean;
    s->excited = (var > 4.0) ? 1 : 0;               // std(v_nn) > 2 m/s
    if (!std::isfinite(s->lam)) {                   // --- OLS (default) ---
        if (s->excited) {                            // full affine fit
            double d = sw * swxx - swx * swx;
            s->k = (sw * swxy - swx * swy) / d;
            s->c = (swxx * swy - swx * swxy) / d;
        } else {                                     // scale-only through origin
            s->k = swxy / swxx;
            s->c = 0.0;
        }
    } else {                                        // --- Deming (errors-in-variables) ---
        if (s->excited) {                            // centered second moments
            double xb = swx / sw, yb = swy / sw;
            double sxx = swxx / sw - xb * xb;
            double syy = swyy / sw - yb * yb;
            double sxy = swxy / sw - xb * yb;
            s->k = deming_k(sxx, syy, sxy, s->lam);
            s->c = yb - s->k * xb;
        } else {                                     // through the origin
            s->k = deming_k(swxx, swyy, swxy, s->lam);
            s->c = 0.0;
        }
    }
    *k_out = s->k; *c_out = s->c;
    return s->excited;
}

double spc_apply(const spc* s, double v_nn) { return s->k * v_nn + s->c; }  // uses frozen fit
void spc_get(const spc* s, double* k, double* c, int* excited) { *k = s->k; *c = s->c; *excited = s->excited; }
}
