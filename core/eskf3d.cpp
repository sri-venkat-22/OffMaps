// libidr full 3D 16-state error-state KF. See idr.h (Phase 7a) for rationale.
//
// Nominal x = [p(3), v(3), q(4), b_a(3), b_g(3)]  (16).
// Error   dx = [dp(3), dv(3), dtheta(3), db_a(3), db_g(3)]  (15); P is 15x15.
// Frames: ENU world (z up), body x-fwd/y-left/z-up. q = Hamilton [w,x,y,z],
// body->world (v_world = R(q) v_body). Local (right) attitude error:
// q_true = q_nom (x) Exp(dtheta).
//
// Design choices held in lockstep with py/eskf3d_ref.py (the parity oracle):
//  - first-order discrete error transition F (the classic 15-state INS model);
//  - nominal attitude integrated with the exact quaternion exponential;
//  - sequential scalar updates with immediate inject-and-reset (G ~= I).
// No Eigen, no matrix inverse -- same discipline as the planar core.
#include "idr.h"
#include <cmath>
#include <cstring>

namespace {
constexpr int N = 15;                       // error-state dimension
constexpr double G0 = 9.81;                 // gravity magnitude (ENU: g = -G0 * z_hat)

// v_world = R(q) v_body, Hamilton q = [w,x,y,z].
void rotmat(const double q[4], double R[3][3]) {
    double w = q[0], x = q[1], y = q[2], z = q[3];
    R[0][0] = 1 - 2*(y*y + z*z); R[0][1] = 2*(x*y - w*z);   R[0][2] = 2*(x*z + w*y);
    R[1][0] = 2*(x*y + w*z);     R[1][1] = 1 - 2*(x*x + z*z); R[1][2] = 2*(y*z - w*x);
    R[2][0] = 2*(x*z - w*y);     R[2][1] = 2*(y*z + w*x);   R[2][2] = 1 - 2*(x*x + y*y);
}

void quatmul(const double a[4], const double b[4], double o[4]) {
    o[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    o[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    o[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    o[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
}

void normq(double q[4]) {
    double n = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (n > 0) for (int i = 0; i < 4; i++) q[i] /= n;
}

// Exp: rotation vector phi -> unit quaternion.
void expq(const double phi[3], double q[4]) {
    double th = std::sqrt(phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2]);
    if (th < 1e-12) {                       // small-angle: q ~= [1, phi/2], then renormalised
        q[0] = 1.0; q[1] = 0.5*phi[0]; q[2] = 0.5*phi[1]; q[3] = 0.5*phi[2];
        normq(q); return;
    }
    double s = std::sin(0.5*th) / th;
    q[0] = std::cos(0.5*th); q[1] = s*phi[0]; q[2] = s*phi[1]; q[3] = s*phi[2];
}
} // namespace

struct idr3d {
    double p[3], v[3], q[4], ba[3], bg[3];
    double P[N][N];
    double q_a, q_g, q_ba, q_bg;            // per-second process variances

    void set_defaults() {
        std::memset(p, 0, sizeof p); std::memset(v, 0, sizeof v);
        q[0] = 1; q[1] = q[2] = q[3] = 0;
        std::memset(ba, 0, sizeof ba); std::memset(bg, 0, sizeof bg);
        std::memset(P, 0, sizeof P);
        for (int i = 0; i < 3; i++)  P[i][i] = 4.0;            // dp  (2 m)
        for (int i = 3; i < 6; i++)  P[i][i] = 0.25;           // dv  (0.5 m/s)
        for (int i = 6; i < 9; i++)  P[i][i] = (2*M_PI/180)*(2*M_PI/180); // dtheta (2 deg)
        for (int i = 9; i < 12; i++) P[i][i] = 0.1*0.1;        // b_a
        for (int i = 12; i < 15; i++) P[i][i] = (0.2*M_PI/180)*(0.2*M_PI/180); // b_g
        set_noise(0.05, 0.3*M_PI/180, 0.001, 0.01*M_PI/180);
    }
    void set_noise(double vrw, double arw, double barw, double bgrw) {
        q_a = vrw*vrw; q_g = arw*arw; q_ba = barw*barw; q_bg = bgrw*bgrw;
    }

    // Inject error dx into the nominal state, then (implicitly) reset dx to 0.
    void inject(const double dx[N]) {
        for (int i = 0; i < 3; i++) { p[i] += dx[i]; v[i] += dx[3+i]; }
        double dq[4], qn[4]; expq(dx+6, dq); quatmul(q, dq, qn);
        std::memcpy(q, qn, sizeof q); normq(q);
        for (int i = 0; i < 3; i++) { ba[i] += dx[9+i]; bg[i] += dx[12+i]; }
    }

    // Sequential scalar update: measurement Jacobian H (1x15), innovation, noise R.
    void update(const double H[N], double innov, double R) {
        double PHt[N] = {0};
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) PHt[i] += P[i][j] * H[j];
        double S = R;
        for (int j = 0; j < N; j++) S += H[j] * PHt[j];
        double K[N], dx[N];
        for (int i = 0; i < N; i++) { K[i] = PHt[i] / S; dx[i] = K[i] * innov; }
        inject(dx);
        double HP[N] = {0};
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++) HP[j] += H[i] * P[i][j];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) P[i][j] -= K[i] * HP[j];
    }
};

extern "C" {

idr3d* idr3d_create(void) { auto* f = new idr3d(); f->set_defaults(); return f; }
void idr3d_destroy(idr3d* f) { delete f; }

void idr3d_init(idr3d* f, const double p[3], const double v[3], const double q[4]) {
    for (int i = 0; i < 3; i++) { f->p[i] = p[i]; f->v[i] = v[i]; }
    for (int i = 0; i < 4; i++) f->q[i] = q[i];
    normq(f->q);
    std::memset(f->ba, 0, sizeof f->ba); std::memset(f->bg, 0, sizeof f->bg);
}
void idr3d_set_noise(idr3d* f, double vrw, double arw, double barw, double bgrw) {
    f->set_noise(vrw, arw, barw, bgrw);
}

void idr3d_predict(idr3d* f, double dt, const double a_m[3], const double w_m[3]) {
    double R[3][3]; rotmat(f->q, R);
    double ab[3] = {a_m[0]-f->ba[0], a_m[1]-f->ba[1], a_m[2]-f->ba[2]};  // specific force, body
    double wb[3] = {w_m[0]-f->bg[0], w_m[1]-f->bg[1], w_m[2]-f->bg[2]};  // angular rate, body
    // world acceleration = R*a_body + g
    double aw[3] = {0,0,0};
    for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) aw[i] += R[i][j]*ab[j]; }
    aw[2] -= G0;
    // nominal p, v
    for (int i = 0; i < 3; i++) {
        f->p[i] += f->v[i]*dt + 0.5*aw[i]*dt*dt;
        f->v[i] += aw[i]*dt;
    }
    // nominal attitude: q <- q (x) Exp(wb*dt)
    double phi[3] = {wb[0]*dt, wb[1]*dt, wb[2]*dt};
    double dq[4], qn[4]; expq(phi, dq); quatmul(f->q, dq, qn);
    std::memcpy(f->q, qn, sizeof f->q); normq(f->q);

    // error-state transition F (first order). Blocks in [dp dv dth dba dbg].
    double F[N][N] = {{0}};
    for (int i = 0; i < N; i++) F[i][i] = 1.0;
    for (int i = 0; i < 3; i++) F[i][3+i] = dt;               // dp <- dv
    // dv <- -R [ab]x dtheta dt - R dba dt
    double Rax[3][3];                                          // R * [ab]x
    // [ab]x = [[0,-abz,aby],[abz,0,-abx],[-aby,abx,0]]
    double ax[3][3] = {{0,-ab[2],ab[1]},{ab[2],0,-ab[0]},{-ab[1],ab[0],0}};
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
        double s = 0; for (int k = 0; k < 3; k++) s += R[i][k]*ax[k][j]; Rax[i][j] = s;
    }
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
        F[3+i][6+j] = -Rax[i][j]*dt;
        F[3+i][9+j] = -R[i][j]*dt;
    }
    // dtheta <- (I - [wb*dt]x) dtheta - dbg dt
    double wx[3][3] = {{0,-phi[2],phi[1]},{phi[2],0,-phi[0]},{-phi[1],phi[0],0}};
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) F[6+i][6+j] = (i==j?1.0:0.0) - wx[i][j];
    for (int i = 0; i < 3; i++) F[6+i][12+i] = -dt;

    // P = F P F^T + Q
    double FP[N][N] = {{0}};
    for (int i = 0; i < N; i++) for (int k = 0; k < N; k++) {
        double a = 0; for (int j = 0; j < N; j++) a += F[i][j]*f->P[j][k]; FP[i][k] = a;
    }
    for (int i = 0; i < N; i++) for (int k = 0; k < N; k++) {
        double a = 0; for (int j = 0; j < N; j++) a += FP[i][j]*F[k][j]; f->P[i][k] = a;
    }
    for (int i = 0; i < 3; i++) {
        f->P[3+i][3+i]  += f->q_a  * dt;
        f->P[6+i][6+i]  += f->q_g  * dt;
        f->P[9+i][9+i]  += f->q_ba * dt;
        f->P[12+i][12+i]+= f->q_bg * dt;
    }
}

void idr3d_update_gnss_pos(idr3d* f, const double p[3], double sigma) {
    for (int k = 0; k < 3; k++) {
        double H[N] = {0}; H[k] = 1.0;
        f->update(H, p[k] - f->p[k], sigma*sigma);
    }
}
void idr3d_update_gnss_vel(idr3d* f, const double v[3], double sigma) {
    for (int k = 0; k < 3; k++) {
        double H[N] = {0}; H[3+k] = 1.0;
        f->update(H, v[k] - f->v[k], sigma*sigma);
    }
}
void idr3d_update_zupt(idr3d* f, double sigma) {
    for (int k = 0; k < 3; k++) {
        double H[N] = {0}; H[3+k] = 1.0;
        f->update(H, 0.0 - f->v[k], sigma*sigma);
    }
}
void idr3d_update_zaru(idr3d* f, const double w_m[3], double sigma) {
    for (int k = 0; k < 3; k++) {
        double H[N] = {0}; H[12+k] = 1.0;
        f->update(H, w_m[k] - f->bg[k], sigma*sigma);
    }
}
void idr3d_update_nhc(idr3d* f, double sigma) {
    // body velocity u = R^T v; constrain lateral (y) and vertical (z) to 0.
    double R[3][3]; rotmat(f->q, R);
    double u[3] = {0,0,0};
    for (int i = 0; i < 3; i++) for (int k = 0; k < 3; k++) u[i] += R[k][i]*f->v[k]; // R^T v
    // y-constraint: h = u_y ; H_dv = R[:,1], H_dtheta = (u_z, 0, -u_x)
    { double H[N] = {0};
      for (int k = 0; k < 3; k++) H[3+k] = R[k][1];
      H[6+0] = u[2]; H[6+1] = 0.0; H[6+2] = -u[0];
      f->update(H, 0.0 - u[1], sigma*sigma); }
    // z-constraint: recompute u after the y-update changed the state
    for (int i = 0; i < 3; i++) u[i] = 0;
    rotmat(f->q, R);
    for (int i = 0; i < 3; i++) for (int k = 0; k < 3; k++) u[i] += R[k][i]*f->v[k];
    { double H[N] = {0};
      for (int k = 0; k < 3; k++) H[3+k] = R[k][2];
      H[6+0] = -u[1]; H[6+1] = u[0]; H[6+2] = 0.0;
      f->update(H, 0.0 - u[2], sigma*sigma); }
}
void idr3d_update_odo(idr3d* f, double v_fwd, double sigma) {
    // body forward velocity u_x = (R^T v)_x = wheel speed.
    double R[3][3]; rotmat(f->q, R);
    double u[3] = {0,0,0};
    for (int i = 0; i < 3; i++) for (int k = 0; k < 3; k++) u[i] += R[k][i]*f->v[k];
    double H[N] = {0};
    for (int k = 0; k < 3; k++) H[3+k] = R[k][0];             // H_dv = R[:,0]
    H[6+0] = 0.0; H[6+1] = -u[2]; H[6+2] = u[1];             // H_dtheta = (0,-u_z,u_y)
    f->update(H, v_fwd - u[0], sigma*sigma);
}
void idr3d_update_baro(idr3d* f, double up, double sigma) {
    double H[N] = {0}; H[2] = 1.0;
    f->update(H, up - f->p[2], sigma*sigma);
}

void idr3d_get_state(const idr3d* f, double out[16]) {
    for (int i = 0; i < 3; i++) out[i] = f->p[i];
    for (int i = 0; i < 3; i++) out[3+i] = f->v[i];
    for (int i = 0; i < 4; i++) out[6+i] = f->q[i];
    for (int i = 0; i < 3; i++) out[10+i] = f->ba[i];
    for (int i = 0; i < 3; i++) out[13+i] = f->bg[i];
}
void idr3d_get_cov(const idr3d* f, double out[15]) {
    for (int i = 0; i < N; i++) out[i] = f->P[i][i];
}
} // extern "C"
