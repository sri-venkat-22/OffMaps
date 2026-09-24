// libidr planar 5-state EKF. See idr.h for the design rationale.
// State x = [e, n, psi, v, b_g]. Covariance P is 5x5. No external deps.
#include "idr.h"
#include <cmath>
#include <cstring>

namespace {
constexpr int N = 5;
enum { E, PN, PSI, V, BG };
constexpr double DEG = M_PI / 180.0;
constexpr double CURV_MIN_RATE = 10.0 * DEG;   // gate curvature obs at |psidot|>10 deg/s

inline double wrap(double a) {                  // -> (-pi, pi]
    while (a > M_PI) a -= 2 * M_PI;
    while (a <= -M_PI) a += 2 * M_PI;
    return a;
}
} // namespace

struct idr_filter {
    double x[N];
    double P[N][N];
    double q_psi, q_bg, q_v;                    // per-second process variances
    int map_keep_v = 0;                         // road updates may not move v (idr_set_map_keep_speed)

    void set_defaults() {
        std::memset(x, 0, sizeof x);
        std::memset(P, 0, sizeof P);
        P[E][E] = P[PN][PN] = 1.0;
        P[PSI][PSI] = (1 * DEG) * (1 * DEG);
        P[V][V] = 0.5 * 0.5;
        P[BG][BG] = (0.05 * DEG) * (0.05 * DEG);
        set_noise(0.3 * DEG, 0.01 * DEG, 0.7);  // gyro ARW, bias RW, speed RW
    }
    void set_noise(double arw, double brw, double srw) {
        q_psi = arw * arw; q_bg = brw * brw; q_v = srw * srw;
    }

    // Sequential scalar update: measurement z = H.x + noise(R). y = z - H.x already formed.
    void update(const double H[N], double innov, double R) {
        double PHt[N] = {0};                    // P H^T
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) PHt[i] += P[i][j] * H[j];
        double S = R;
        for (int j = 0; j < N; j++) S += H[j] * PHt[j];
        double K[N];
        for (int i = 0; i < N; i++) K[i] = PHt[i] / S;
        for (int i = 0; i < N; i++) x[i] += K[i] * innov;
        // P = (I - K H) P  == P - K (H P);  H P row = PHt^T (P symmetric)
        double HP[N] = {0};
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++) HP[j] += H[i] * P[i][j];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) P[i][j] -= K[i] * HP[j];
    }

    // Same measurement, but the gain's row `skip` is forced to 0 (that state is not
    // corrected). The gain is then sub-optimal, so P uses the Joseph form
    // P = (I-KH) P (I-KH)^T + K R K^T, which is valid for any gain.
    void update_skip(const double H[N], double innov, double R, int skip) {
        double PHt[N] = {0};
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) PHt[i] += P[i][j] * H[j];
        double S = R;
        for (int j = 0; j < N; j++) S += H[j] * PHt[j];
        double K[N];
        for (int i = 0; i < N; i++) K[i] = PHt[i] / S;
        K[skip] = 0.0;
        for (int i = 0; i < N; i++) x[i] += K[i] * innov;
        double A[N][N];                          // A = I - K H
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) A[i][j] = (i == j ? 1.0 : 0.0) - K[i] * H[j];
        double AP[N][N] = {{0}};
        for (int i = 0; i < N; i++)
            for (int k = 0; k < N; k++)
                for (int j = 0; j < N; j++) AP[i][j] += A[i][k] * P[k][j];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                double s = K[i] * K[j] * R;
                for (int k = 0; k < N; k++) s += AP[i][k] * A[j][k];
                P[i][j] = s;
            }
    }
    void update_map(const double H[N], double innov, double R) {
        if (map_keep_v) update_skip(H, innov, R, V); else update(H, innov, R);
    }
};

extern "C" {

idr_filter* idr_create(void) { auto* f = new idr_filter(); f->set_defaults(); return f; }
void idr_destroy(idr_filter* f) { delete f; }

void idr_init(idr_filter* f, double e, double n, double psi, double v) {
    f->x[E] = e; f->x[PN] = n; f->x[PSI] = psi; f->x[V] = v; f->x[BG] = 0.0;
}
void idr_set_process_noise(idr_filter* f, double arw, double brw, double srw) {
    f->set_noise(arw, brw, srw);
}

void idr_predict(idr_filter* f, double dt, double gyro_z) {
    double* x = f->x;
    double psi = x[PSI], v = x[V];
    double s = std::sin(psi), c = std::cos(psi);
    // nominal
    x[E]  += v * s * dt;
    x[PN] += v * c * dt;
    x[PSI] = wrap(psi + (gyro_z - x[BG]) * dt);
    // F (Jacobian of the above wrt x)
    double F[N][N] = {{0}};
    for (int i = 0; i < N; i++) F[i][i] = 1.0;
    F[E][PSI] = v * c * dt;  F[E][V] = s * dt;
    F[PN][PSI] = -v * s * dt; F[PN][V] = c * dt;
    F[PSI][BG] = -dt;
    // P = F P F^T + Q
    double FP[N][N] = {{0}};
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            double a = 0;
            for (int j = 0; j < N; j++) a += F[i][j] * f->P[j][k];
            FP[i][k] = a;
        }
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            double a = 0;
            for (int j = 0; j < N; j++) a += FP[i][j] * F[k][j];
            f->P[i][k] = a;
        }
    f->P[PSI][PSI] += f->q_psi * dt;
    f->P[V][V]     += f->q_v   * dt;
    f->P[BG][BG]   += f->q_bg  * dt;
}

void idr_update_speed(idr_filter* f, double v_meas, double sigma) {
    double H[N] = {0}; H[V] = 1.0;
    f->update(H, v_meas - f->x[V], sigma * sigma);
}
void idr_update_zupt(idr_filter* f, double gyro_z) {
    double Hv[N] = {0}; Hv[V] = 1.0;
    f->update(Hv, 0.0 - f->x[V], 0.02 * 0.02);          // v = 0
    double Hb[N] = {0}; Hb[BG] = 1.0;
    f->update(Hb, gyro_z - f->x[BG], (0.02 * DEG) * (0.02 * DEG)); // ZARU: b_g = gyro
}
void idr_update_curvature(idr_filter* f, double a_lat, double psidot, double base_sigma) {
    if (std::fabs(psidot) < CURV_MIN_RATE) return;          // gravity-leakage swamps it below this
    double v_curv = a_lat / psidot;
    double sigma = base_sigma * CURV_MIN_RATE / std::fabs(psidot);   // R inflates as ~1/psidot
    double H[N] = {0}; H[V] = 1.0;
    f->update(H, v_curv - f->x[V], sigma * sigma);
}
void idr_update_gnss_pos(idr_filter* f, double e, double n, double sigma) {
    double He[N] = {0}; He[E] = 1.0;  f->update(He, e - f->x[E], sigma * sigma);
    double Hn[N] = {0}; Hn[PN] = 1.0; f->update(Hn, n - f->x[PN], sigma * sigma);
}
void idr_update_gnss_vel(idr_filter* f, double v_meas, double bearing, double sv, double sp) {
    double Hv[N] = {0}; Hv[V] = 1.0;   f->update(Hv, v_meas - f->x[V], sv * sv);
    double Hp[N] = {0}; Hp[PSI] = 1.0; f->update(Hp, wrap(bearing - f->x[PSI]), sp * sp);
}
void idr_update_crosstrack(idr_filter* f, double ne, double nn, double cross_innov, double sigma) {
    double H[N] = {0}; H[E] = ne; H[PN] = nn;   // measure position along the road normal only
    f->update_map(H, cross_innov, sigma * sigma);   // along-track is left free (the anisotropy)
}
void idr_update_heading(idr_filter* f, double bearing, double sigma) {
    double H[N] = {0}; H[PSI] = 1.0;
    f->update_map(H, wrap(bearing - f->x[PSI]), sigma * sigma);
}
void idr_set_map_keep_speed(idr_filter* f, int keep) { f->map_keep_v = keep ? 1 : 0; }
void idr_set_heading_sigma(idr_filter* f, double sigma) {
    for (int i = 0; i < N; i++) { f->P[PSI][i] = 0.0; f->P[i][PSI] = 0.0; }
    f->P[PSI][PSI] = sigma * sigma;
}
void idr_get_state(const idr_filter* f, double out[5]) {
    for (int i = 0; i < 5; i++) out[i] = f->x[i];
}
void idr_get_cov(const idr_filter* f, double out[3]) {
    out[0] = f->P[E][E]; out[1] = f->P[PN][PN]; out[2] = f->P[PSI][PSI];
}
} // extern "C"
