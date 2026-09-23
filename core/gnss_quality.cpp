// Continuous GNSS trust -> there is no mode switch. GNSS is just a measurement
// whose R scales from ~3 m (healthy) to ~1e6 (useless) via one weight, so the
// tunnel transition is one filter cycle and cannot produce a position jump.
// NavIC-weighted (ISRO). The spoof flag is the same signal, second threshold:
// strong C/N0 but the INS violently disagrees = spoofing/meaconing.
#include "idr.h"
#include <cmath>

namespace {
double clamp01(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
}

extern "C" {
// trust in [0,1]. innov_chi2 is the GNSS-position innovation normalised by the
// filter's predicted covariance (dimensionless); >~9 means INS disagrees badly.
double gq_trust(double cn0_mean, int sv_used, int navic_sv, double dop, double innov_chi2) {
    double s_cn0 = clamp01((cn0_mean - 25.0) / (40.0 - 25.0));
    double s_sv  = clamp01((sv_used - 4.0) / (8.0 - 4.0));
    double s_dop = clamp01((6.0 - dop) / (6.0 - 1.0));
    double s_innov = std::exp(-innov_chi2 / 9.0);          // chi2 gate ~9 (3-sigma, 2 DoF)
    double navic = navic_sv > 0 ? 1.0 + 0.05 * std::fmin(navic_sv, 4) : 1.0;
    return clamp01(s_cn0 * s_sv * s_dop * s_innov * navic);
}

// measurement R (m^2) for GNSS position from trust: 3 m at full trust -> 1e6.
double gq_R(double trust) {
    double sigma = 3.0 / std::fmax(trust, 3e-3);           // caps ~1000 m
    return sigma * sigma;
}

// spoof/jam: healthy-looking signal (high C/N0, enough SVs) that the INS
// rejects (large innovation), or a Doppler-vs-INS velocity mismatch.
int gq_spoof(double cn0_mean, int sv_used, double innov_chi2, double doppler_resid) {
    int looks_strong = (cn0_mean > 35.0 && sv_used >= 5);
    int ins_rejects  = (innov_chi2 > 25.0) || (std::fabs(doppler_resid) > 5.0);
    return (looks_strong && ins_rejects) ? 1 : 0;
}
}
