// Mount alignment: device frame -> vehicle frame, online, with auto re-align.
//  - roll/pitch from a low-passed gravity vector (fast EMA).
//  - yaw from PCA of horizontal accel during LONGITUDINAL events only (small
//    yaw-rate + large horizontal accel) -> forward axis. Turns are excluded
//    (their accel is lateral, which would give a 90-deg-wrong axis).
//  - Re-mount detector: CUSUM on the angle between a medium (5 s) and a slow
//    (30 s) gravity estimate. Braking, hills and curves separate them only
//    briefly; a re-mount separates them until the slow one catches up. On alarm
//    the slow estimate snaps to the medium one and the yaw PCA restarts.
//    (The first version compared the 0.5 s gravity with a SNAPSHOT reference at
//    2 deg slack: on real rigid-mount IO-VNBD drives it fired ~1800x/hour. Tuned
//    on train+val drives: ~1 false alarm/hour, a 30 deg re-mount found in 3-8 s.)
#include "idr.h"
#include <cmath>
#include <cstring>

namespace {
constexpr double DEG = M_PI / 180.0;
constexpr double REMOUNT_TAU_MED = 5.0;      // s
constexpr double REMOUNT_TAU_SLOW = 30.0;    // s (== the app's leveling EMA, Level.kt)
constexpr double REMOUNT_SLACK = 10.0 * DEG; // per step
constexpr double REMOUNT_H = 200.0 * DEG;    // CUSUM alarm threshold
}

struct aln {
    double g[3] = {0, 0, 9.81};          // low-pass gravity (device frame)
    double gm[3] = {0, 0, 9.81};         // medium (5 s) gravity, detector input
    double gs[3] = {0, 0, 9.81};         // slow (30 s) gravity, detector reference
    double cusum = 0;
    double Sxx = 0, Sxy = 0, Syy = 0;    // horizontal-accel covariance for PCA
    double roll = 0, pitch = 0, yaw = 0;
    int have_ref = 0, changed = 0;
};

extern "C" {
aln* aln_create(void) { return new aln(); }
void aln_destroy(aln* a) { delete a; }

void aln_update(aln* a, double ax, double ay, double az,
                double gx, double gy, double gz, double dt) {
    a->changed = 0;
    double beta = 1.0 - std::exp(-dt / 0.5);         // ~0.5 s gravity time constant
    a->g[0] += beta * (ax - a->g[0]);
    a->g[1] += beta * (ay - a->g[1]);
    a->g[2] += beta * (az - a->g[2]);
    double gn = std::sqrt(a->g[0]*a->g[0] + a->g[1]*a->g[1] + a->g[2]*a->g[2]) + 1e-9;
    double u[3] = {a->g[0]/gn, a->g[1]/gn, a->g[2]/gn};

    a->roll = std::atan2(u[1], u[2]);
    a->pitch = std::atan2(-u[0], std::sqrt(u[1]*u[1] + u[2]*u[2]));

    // re-mount detector: CUSUM on angle(medium gravity, slow gravity)
    double acc3[3] = {ax, ay, az};
    if (!a->have_ref) {
        std::memcpy(a->gm, acc3, sizeof acc3); std::memcpy(a->gs, acc3, sizeof acc3); a->have_ref = 1;
    }
    double bm = 1.0 - std::exp(-dt / REMOUNT_TAU_MED), bs = 1.0 - std::exp(-dt / REMOUNT_TAU_SLOW);
    for (int i = 0; i < 3; i++) { a->gm[i] += bm * (acc3[i] - a->gm[i]); a->gs[i] += bs * (acc3[i] - a->gs[i]); }
    double nm = std::sqrt(a->gm[0]*a->gm[0] + a->gm[1]*a->gm[1] + a->gm[2]*a->gm[2]) + 1e-9;
    double ns = std::sqrt(a->gs[0]*a->gs[0] + a->gs[1]*a->gs[1] + a->gs[2]*a->gs[2]) + 1e-9;
    double cosang = (a->gm[0]*a->gs[0] + a->gm[1]*a->gs[1] + a->gm[2]*a->gs[2]) / (nm * ns);
    double dev = std::acos(cosang > 1 ? 1 : cosang < -1 ? -1 : cosang);
    a->cusum = std::fmax(0.0, a->cusum + dev - REMOUNT_SLACK);
    if (a->cusum > REMOUNT_H) {                             // re-mount
        std::memcpy(a->gs, a->gm, sizeof a->gm);            // slow reference jumps to the new mount
        a->cusum = 0; a->Sxx = a->Sxy = a->Syy = 0;        // restart yaw estimate
        a->changed = 1;
    }

    // yaw PCA: only during longitudinal events (not turning, strong horiz accel)
    double ah0 = ax - u[0]*9.81, ah1 = ay - u[1]*9.81;     // remove gravity (approx level)
    double amag = std::sqrt(ah0*ah0 + ah1*ah1);
    if (std::fabs(gz) < 5*DEG && amag > 1.0) {
        double f = 0.98;                                   // forgetting factor
        a->Sxx = f*a->Sxx + ah0*ah0; a->Sxy = f*a->Sxy + ah0*ah1; a->Syy = f*a->Syy + ah1*ah1;
        a->yaw = 0.5 * std::atan2(2*a->Sxy, a->Sxx - a->Syy);   // principal axis = forward
    }
}

void aln_get(const aln* a, double* roll, double* pitch, double* yaw, int* changed) {
    *roll = a->roll; *pitch = a->pitch; *yaw = a->yaw; *changed = a->changed;
}
}
