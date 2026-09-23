/* libidr -- inertial dead-reckoning core, C ABI.
 *
 * One core, two frontends: Android NDK binds this header; the Python harness
 * loads the same .dylib via ctypes; tools/replay.cpp links it directly.
 *
 * Planar 5-state EKF: state = [e, n, psi, v, b_g_yaw].
 *   - position integrates SPEED along HEADING (never accelerometer) -> drift
 *     linear in distance, the whole design thesis.
 *   - NHC (lateral velocity = 0) is structural in the unicycle model, so there
 *     is no separate NHC update.
 * All measurement updates are sequential scalar (no matrix inverse, no Eigen).
 */
#ifndef IDR_H
#define IDR_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct idr_filter idr_filter;

idr_filter* idr_create(void);
void        idr_destroy(idr_filter*);

/* Initialise nominal state (typically from the last GNSS fix). */
void idr_init(idr_filter*, double e, double n, double psi, double v);

/* Process-noise knobs (per your Allan-variance fit). Units: SI, per sqrt(s).
 * gyro_arw rad/sqrt(s); gyro_bias_rw rad/s/sqrt(s); speed_rw m/s/sqrt(s). */
void idr_set_process_noise(idr_filter*, double gyro_arw, double gyro_bias_rw, double speed_rw);

/* Propagate one IMU step. gyro_z = measured yaw rate about vertical (rad/s). */
void idr_predict(idr_filter*, double dt, double gyro_z);

/* Measurement updates (each may be called or skipped per sample). */
void idr_update_speed(idr_filter*, double v_meas, double sigma);   /* NN / Doppler speed */
void idr_update_zupt(idr_filter*, double gyro_z);                  /* stopped: v=0 & ZARU on b_g */
void idr_update_curvature(idr_filter*, double a_lat, double psidot, double base_sigma); /* v=a_lat/psidot, gated |psidot|>10deg/s */
void idr_update_gnss_pos(idr_filter*, double e, double n, double sigma);
void idr_update_gnss_vel(idr_filter*, double v_meas, double bearing, double sigma_v, double sigma_psi);

/* Read state into out[5] = {e, n, psi, v, b_g}. */
void idr_update_crosstrack(idr_filter*, double ne, double nn, double cross_innov, double sigma);
void idr_update_heading(idr_filter*, double bearing, double sigma);
/* Road updates (crosstrack + heading) may not change speed: with keep != 0 their
 * gain's speed row is zeroed and P takes the Joseph form (exact for that gain).
 * A road fixes the lane and the direction, never the distance travelled; with a
 * loose speed random walk (the real-data srw=24) an unrestricted crosstrack
 * innovation leaked into v through the heading/position correlation and made
 * real outages WORSE with the map (README_PHASE6 6d). Default 0 = the original
 * update, bit-exact. */
void idr_set_map_keep_speed(idr_filter*, int keep);
void idr_get_state(const idr_filter*, double out[5]);

/* Read the diagonal covariance the live loop needs for GNSS innovation gating:
 * out[3] = {P_ee, P_nn, P_psipsi}. Pure accessor -- no state change, so parity
 * with the Python oracle is untouched. (Phase 4's gq_trust chi2 is the position
 * innovation normalised by P_ee/P_nn; the on-device fusion computes it here.) */
void idr_get_cov(const idr_filter*, double out[3]);

/* --- Phase 4: pre-filter estimators (all on-device, same lib) --- */
typedef struct spc spc;            /* Doppler speed-scale self-calibration */
spc*   spc_create(void);
void   spc_destroy(spc*);
void   spc_push(spc*, double v_nn, double v_dop, double weight);
void   spc_set_lambda(spc*, double lambda);   /* Deming noise ratio; +inf = OLS */
int    spc_fit(spc*, double* k_out, double* c_out);   /* returns excited flag */
double spc_apply(const spc*, double v_nn);
void   spc_get(const spc*, double* k, double* c, int* excited);

typedef struct aln aln;            /* mount alignment + re-align */
aln*   aln_create(void);
void   aln_destroy(aln*);
void   aln_update(aln*, double ax, double ay, double az,
                  double gx, double gy, double gz, double dt);
void   aln_get(const aln*, double* roll, double* pitch, double* yaw, int* changed);

double gq_trust(double cn0_mean, int sv_used, int navic_sv, double dop, double innov_chi2);
double gq_R(double trust);
int    gq_spoof(double cn0_mean, int sv_used, double innov_chi2, double doppler_resid);

typedef struct mm mm;              /* offline map matching */
mm*  mm_create(void);
void mm_destroy(mm*);
void mm_add_way(mm*, const double* e, const double* n, int npts, int tunnel, int oneway);
void mm_match(const mm*, double e, double n, double psi, double out[6]);

/* Viterbi/HMM map matching (Phase 7b). mm_match is greedy nearest-road; at a
 * junction where two roads are momentarily near-parallel it can wrong-snap for a
 * stretch. mm_match_seq decodes the WHOLE trajectory at once: HMM states are the
 * road ways, emission = perpendicular distance + bearing agreement, transition =
 * road-network adjacency + a distance-consistency term (how far along the roads
 * you'd have to travel vs how far the trajectory moved). Viterbi returns the
 * globally most-consistent way index per step, killing the greedy wrong-snaps.
 * way_out[i] = matched way index for step i (-1 = no candidate). Optional
 * foot_e/foot_n/bearing/corridor arrays (each length nsteps, or NULL) receive the
 * projection onto the chosen way. Returns the number of steps matched. */
int mm_match_seq(const mm*, const double* e, const double* n, const double* psi,
                 int nsteps, int* way_out, double* foot_e, double* foot_n,
                 double* bearing_out, int* corridor_out);
/* Declare which ways are connected (share a junction). Adjacency is symmetric;
 * a way is always adjacent to itself. Without any edges every way is treated as
 * connected to every other (transition falls back to distance-consistency only). */
void mm_add_edge(mm*, int way_a, int way_b);
void mm_set_hmm(mm*, double sigma_emit, double sigma_trans, double beta_bearing);

/* --- Phase 7c: high-rate (200-400 Hz) vibration / pothole front-end --- */
/* The 10 Hz feature path (model/features.py) cannot see road vibration above its
 * 5 Hz Nyquist -- it aliases -- and a pothole shock read at 10 Hz looks like a
 * huge acceleration that corrupts the speed cue. This streaming DSP runs on the
 * raw high-rate accelerometer: a high-pass isolates the vibration band, an
 * adaptive threshold flags pothole/shock transients, and the per-window RMS is
 * computed with those transients EXCISED, so a pothole no longer masquerades as
 * speed. The learned high-rate speed map trains on real phone logs; this is the
 * deterministic front-end it consumes. */
typedef struct vib vib;
vib*  vib_create(double hz);                /* high-rate sample rate, e.g. 250 */
void  vib_destroy(vib*);
void  vib_set_params(vib*, double shock_k, double refractory_s, double hp_fc, double ema_tau);
/* Push one tri-axial accel sample; returns 1 on the leading edge of a shock. */
int   vib_push(vib*, double ax, double ay, double az, double dt);
/* Per-window features since the last call; out[4] = {rms_clean (pothole-rejected),
 * rms_raw (all samples), shock_frac, n_events}. Resets the window accumulators. */
void  vib_window(vib*, double out[4]);

/* --- Phase 7a: full 3D 16-state error-state KF (idr3d) --- */
/* The planar 5-state core above is exact for the road/corridor case. This is the
 * documented 3D extension for genuinely-3D motion (multi-level parking ramps):
 * nominal state = position(3) + velocity(3) + attitude quaternion(4) +
 * accel-bias(3) + gyro-bias(3) = 16; error-state covariance is 15x15
 * [dp, dv, dtheta, db_a, db_g]. Frames: ENU world (z up), body x-fwd y-left z-up,
 * quaternion Hamilton [w,x,y,z] rotating body->world. Gravity is modelled, so
 * pitch/grade no longer leak into horizontal position the way the planar core's
 * "all speed is horizontal" assumption does. Updates are sequential scalar with
 * immediate error injection + reset (G~=I), matching the planar core's style. */
typedef struct idr3d idr3d;
idr3d* idr3d_create(void);
void   idr3d_destroy(idr3d*);
void   idr3d_init(idr3d*, const double p[3], const double v[3], const double q[4]);
/* per-sqrt(s): accel VRW (m/s/sqrt(s)), gyro ARW (rad/sqrt(s)),
 * accel-bias RW (m/s^2/sqrt(s)), gyro-bias RW (rad/s/sqrt(s)). */
void idr3d_set_noise(idr3d*, double accel_vrw, double gyro_arw,
                     double accel_bias_rw, double gyro_bias_rw);
/* a_m[3] = measured specific force (body), w_m[3] = measured angular rate (body). */
void idr3d_predict(idr3d*, double dt, const double a_m[3], const double w_m[3]);
void idr3d_update_gnss_pos(idr3d*, const double p[3], double sigma);
void idr3d_update_gnss_vel(idr3d*, const double v[3], double sigma);
void idr3d_update_zupt(idr3d*, double sigma);           /* world velocity = 0 (3 scalars) */
void idr3d_update_zaru(idr3d*, const double w_m[3], double sigma); /* b_g = w_m (stopped) */
void idr3d_update_nhc(idr3d*, double sigma);            /* body lateral & vertical vel = 0 */
void idr3d_update_odo(idr3d*, double v_fwd, double sigma); /* body forward vel = wheel speed */
void idr3d_update_baro(idr3d*, double up, double sigma);/* height (ENU up) */
/* out[16] = {p(3), v(3), q(4 w,x,y,z), b_a(3), b_g(3)}. */
void idr3d_get_state(const idr3d*, double out[16]);
void idr3d_get_cov(const idr3d*, double out[15]);       /* diag of the 15x15 error cov */

#ifdef __cplusplus
}
#endif
#endif
