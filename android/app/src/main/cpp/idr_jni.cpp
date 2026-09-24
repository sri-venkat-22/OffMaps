// JNI bridge -- the Android NDK frontend to libidr. This is the "embed via
// NDK/JNI" deliverable of Phase 6: it exposes the SAME C ABI that py/core_bridge.py
// reaches through ctypes and tools/replay.cpp links directly, so the phone runs
// the identical, parity-proven core object.
//
// Every function here is a thin marshaller: unwrap a jlong handle, call one
// idr_*/spc_*/aln_*/gq_*/mm_*/vib_*/idr3d_* function, box any multi-value result into a
// double[]. No filtering logic lives here -- the live fusion loop is Kotlin
// (com.offmaps.nav.FusionEngine), mirroring core_bridge.py::_run, which is the
// validated orchestration. Native names match Kotlin `object IdrNative`.
#include <jni.h>
#include <vector>
#include "idr.h"

// Handles are opaque native pointers carried across the boundary as jlong.
#define H(T, h) reinterpret_cast<T*>(static_cast<intptr_t>(h))
#define AS_LONG(p) static_cast<jlong>(reinterpret_cast<intptr_t>(p))

static jdoubleArray box(JNIEnv* env, const double* v, int n) {
    jdoubleArray a = env->NewDoubleArray(n);
    env->SetDoubleArrayRegion(a, 0, n, v);
    return a;
}

extern "C" {

// ---------------- idr_filter: the planar 5-state ESKF ----------------
JNIEXPORT jlong JNICALL
Java_com_offmaps_nav_IdrNative_idrCreate(JNIEnv*, jobject) { return AS_LONG(idr_create()); }

JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idrDestroy(JNIEnv*, jobject, jlong h) { idr_destroy(H(idr_filter, h)); }

JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idrInit(JNIEnv*, jobject, jlong h, jdouble e, jdouble n, jdouble psi, jdouble v) {
    idr_init(H(idr_filter, h), e, n, psi, v);
}
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idrSetHeadingSigma(JNIEnv*, jobject, jlong h, jdouble sigma) {
    idr_set_heading_sigma(H(idr_filter, h), sigma);
}
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idrSetProcessNoise(JNIEnv*, jobject, jlong h, jdouble arw, jdouble brw, jdouble srw) {
    idr_set_process_noise(H(idr_filter, h), arw, brw, srw);
}
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idrPredict(JNIEnv*, jobject, jlong h, jdouble dt, jdouble gyroZ) {
    idr_predict(H(idr_filter, h), dt, gyroZ);
}
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idrUpdateSpeed(JNIEnv*, jobject, jlong h, jdouble v, jdouble sigma) {
    idr_update_speed(H(idr_filter, h), v, sigma);
}
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idrUpdateZupt(JNIEnv*, jobject, jlong h, jdouble gyroZ) {
    idr_update_zupt(H(idr_filter, h), gyroZ);
}
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idrUpdateCurvature(JNIEnv*, jobject, jlong h, jdouble aLat, jdouble psidot, jdouble baseSigma) {
    idr_update_curvature(H(idr_filter, h), aLat, psidot, baseSigma);
}
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idrUpdateGnssPos(JNIEnv*, jobject, jlong h, jdouble e, jdouble n, jdouble sigma) {
    idr_update_gnss_pos(H(idr_filter, h), e, n, sigma);
}
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idrUpdateGnssVel(JNIEnv*, jobject, jlong h, jdouble v, jdouble bearing, jdouble sigmaV, jdouble sigmaPsi) {
    idr_update_gnss_vel(H(idr_filter, h), v, bearing, sigmaV, sigmaPsi);
}
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idrUpdateCrosstrack(JNIEnv*, jobject, jlong h, jdouble ne, jdouble nn, jdouble crossInnov, jdouble sigma) {
    idr_update_crosstrack(H(idr_filter, h), ne, nn, crossInnov, sigma);
}
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idrUpdateHeading(JNIEnv*, jobject, jlong h, jdouble bearing, jdouble sigma) {
    idr_update_heading(H(idr_filter, h), bearing, sigma);
}
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idrSetMapKeepSpeed(JNIEnv*, jobject, jlong h, jboolean keep) {
    idr_set_map_keep_speed(H(idr_filter, h), keep ? 1 : 0);
}
JNIEXPORT jdoubleArray JNICALL
Java_com_offmaps_nav_IdrNative_idrGetState(JNIEnv* env, jobject, jlong h) {
    double out[5]; idr_get_state(H(idr_filter, h), out); return box(env, out, 5);
}
JNIEXPORT jdoubleArray JNICALL
Java_com_offmaps_nav_IdrNative_idrGetCov(JNIEnv* env, jobject, jlong h) {
    double out[3]; idr_get_cov(H(idr_filter, h), out); return box(env, out, 3);
}

// ---------------- spc: Doppler speed-scale self-calibration ----------------
JNIEXPORT jlong JNICALL
Java_com_offmaps_nav_IdrNative_spcCreate(JNIEnv*, jobject) { return AS_LONG(spc_create()); }
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_spcDestroy(JNIEnv*, jobject, jlong h) { spc_destroy(H(spc, h)); }
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_spcPush(JNIEnv*, jobject, jlong h, jdouble vNn, jdouble vDop, jdouble w) {
    spc_push(H(spc, h), vNn, vDop, w);
}
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_spcSetLambda(JNIEnv*, jobject, jlong h, jdouble lam) { spc_set_lambda(H(spc, h), lam); }
JNIEXPORT jdoubleArray JNICALL
Java_com_offmaps_nav_IdrNative_spcFit(JNIEnv* env, jobject, jlong h) {
    double k, c; int ex = spc_fit(H(spc, h), &k, &c);
    double out[3] = {k, c, (double) ex}; return box(env, out, 3);
}
JNIEXPORT jdouble JNICALL
Java_com_offmaps_nav_IdrNative_spcApply(JNIEnv*, jobject, jlong h, jdouble vNn) { return spc_apply(H(spc, h), vNn); }
JNIEXPORT jdoubleArray JNICALL
Java_com_offmaps_nav_IdrNative_spcGet(JNIEnv* env, jobject, jlong h) {
    double k, c; int ex; spc_get(H(spc, h), &k, &c, &ex);
    double out[3] = {k, c, (double) ex}; return box(env, out, 3);
}

// ---------------- aln: mount alignment + re-align ----------------
JNIEXPORT jlong JNICALL
Java_com_offmaps_nav_IdrNative_alnCreate(JNIEnv*, jobject) { return AS_LONG(aln_create()); }
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_alnDestroy(JNIEnv*, jobject, jlong h) { aln_destroy(H(aln, h)); }
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_alnUpdate(JNIEnv*, jobject, jlong h,
        jdouble ax, jdouble ay, jdouble az, jdouble gx, jdouble gy, jdouble gz, jdouble dt) {
    aln_update(H(aln, h), ax, ay, az, gx, gy, gz, dt);
}
JNIEXPORT jdoubleArray JNICALL
Java_com_offmaps_nav_IdrNative_alnGet(JNIEnv* env, jobject, jlong h) {
    double r, p, y; int ch; aln_get(H(aln, h), &r, &p, &y, &ch);
    double out[4] = {r, p, y, (double) ch}; return box(env, out, 4);
}

// ---------------- gq: GNSS quality (stateless pure functions) ----------------
JNIEXPORT jdouble JNICALL
Java_com_offmaps_nav_IdrNative_gqTrust(JNIEnv*, jobject, jdouble cn0, jint sv, jint navic, jdouble dop, jdouble chi2) {
    return gq_trust(cn0, sv, navic, dop, chi2);
}
JNIEXPORT jdouble JNICALL
Java_com_offmaps_nav_IdrNative_gqR(JNIEnv*, jobject, jdouble trust) { return gq_R(trust); }
JNIEXPORT jint JNICALL
Java_com_offmaps_nav_IdrNative_gqSpoof(JNIEnv*, jobject, jdouble cn0, jint sv, jdouble chi2, jdouble dopResid) {
    return gq_spoof(cn0, sv, chi2, dopResid);
}

// ---------------- mm: offline map matching (fed by RoadMatcher.kt) ----------------
JNIEXPORT jlong JNICALL
Java_com_offmaps_nav_IdrNative_mmCreate(JNIEnv*, jobject) { return AS_LONG(mm_create()); }
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_mmDestroy(JNIEnv*, jobject, jlong h) { mm_destroy(H(mm, h)); }
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_mmAddWay(JNIEnv* env, jobject, jlong h,
        jdoubleArray e, jdoubleArray n, jint npts, jint tunnel, jint oneway) {
    std::vector<double> ce(npts), cn(npts);
    env->GetDoubleArrayRegion(e, 0, npts, ce.data());
    env->GetDoubleArrayRegion(n, 0, npts, cn.data());
    mm_add_way(H(mm, h), ce.data(), cn.data(), npts, tunnel, oneway);
}
JNIEXPORT jdoubleArray JNICALL
Java_com_offmaps_nav_IdrNative_mmMatch(JNIEnv* env, jobject, jlong h, jdouble e, jdouble n, jdouble psi) {
    double out[6]; mm_match(H(mm, h), e, n, psi, out); return box(env, out, 6);
}

// ---------------- mm (Phase 7b): Viterbi/HMM sequence decode ----------------
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_mmAddEdge(JNIEnv*, jobject, jlong h, jint a, jint b) { mm_add_edge(H(mm, h), a, b); }
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_mmSetHmm(JNIEnv*, jobject, jlong h, jdouble sEmit, jdouble sTrans, jdouble beta) {
    mm_set_hmm(H(mm, h), sEmit, sTrans, beta);
}
// Returns [way(0..n), footE(n), footN(n), bearing(n), corridor(n)] flattened (ways as doubles).
JNIEXPORT jdoubleArray JNICALL
Java_com_offmaps_nav_IdrNative_mmMatchSeq(JNIEnv* env, jobject, jlong h,
        jdoubleArray e, jdoubleArray n, jdoubleArray psi) {
    const int ns = env->GetArrayLength(e);
    std::vector<double> ce(ns), cn(ns), cp(ns), fe(ns), fn(ns), br(ns), out(5 * ns);
    std::vector<int> way(ns), cor(ns);
    env->GetDoubleArrayRegion(e, 0, ns, ce.data());
    env->GetDoubleArrayRegion(n, 0, ns, cn.data());
    env->GetDoubleArrayRegion(psi, 0, ns, cp.data());
    mm_match_seq(H(mm, h), ce.data(), cn.data(), cp.data(), ns, way.data(), fe.data(), fn.data(), br.data(), cor.data());
    for (int i = 0; i < ns; i++) {
        out[i] = way[i]; out[ns + i] = fe[i]; out[2 * ns + i] = fn[i]; out[3 * ns + i] = br[i]; out[4 * ns + i] = cor[i];
    }
    return box(env, out.data(), 5 * ns);
}

// ---------------- vib (Phase 7c): high-rate vibration / pothole front-end ----------------
JNIEXPORT jlong JNICALL
Java_com_offmaps_nav_IdrNative_vibCreate(JNIEnv*, jobject, jdouble hz) { return AS_LONG(vib_create(hz)); }
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_vibDestroy(JNIEnv*, jobject, jlong h) { vib_destroy(H(vib, h)); }
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_vibSetParams(JNIEnv*, jobject, jlong h, jdouble shockK, jdouble refrS, jdouble hpFc, jdouble emaTau) {
    vib_set_params(H(vib, h), shockK, refrS, hpFc, emaTau);
}
JNIEXPORT jint JNICALL
Java_com_offmaps_nav_IdrNative_vibPush(JNIEnv*, jobject, jlong h, jdouble ax, jdouble ay, jdouble az, jdouble dt) {
    return vib_push(H(vib, h), ax, ay, az, dt);
}
JNIEXPORT jdoubleArray JNICALL
Java_com_offmaps_nav_IdrNative_vibWindow(JNIEnv* env, jobject, jlong h) {
    double out[4]; vib_window(H(vib, h), out); return box(env, out, 4);
}

// ---------------- idr3d (Phase 7a): 3D 16-state ESKF ----------------
static void get3(JNIEnv* env, jdoubleArray a, double* o, int k) { env->GetDoubleArrayRegion(a, 0, k, o); }
JNIEXPORT jlong JNICALL
Java_com_offmaps_nav_IdrNative_idr3dCreate(JNIEnv*, jobject) { return AS_LONG(idr3d_create()); }
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idr3dDestroy(JNIEnv*, jobject, jlong h) { idr3d_destroy(H(idr3d, h)); }
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idr3dInit(JNIEnv* env, jobject, jlong h, jdoubleArray p, jdoubleArray v, jdoubleArray q) {
    double cp[3], cv[3], cq[4]; get3(env, p, cp, 3); get3(env, v, cv, 3); get3(env, q, cq, 4);
    idr3d_init(H(idr3d, h), cp, cv, cq);
}
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idr3dSetNoise(JNIEnv*, jobject, jlong h, jdouble vrw, jdouble arw, jdouble abrw, jdouble gbrw) {
    idr3d_set_noise(H(idr3d, h), vrw, arw, abrw, gbrw);
}
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idr3dPredict(JNIEnv* env, jobject, jlong h, jdouble dt, jdoubleArray a, jdoubleArray w) {
    double ca[3], cw[3]; get3(env, a, ca, 3); get3(env, w, cw, 3);
    idr3d_predict(H(idr3d, h), dt, ca, cw);
}
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idr3dUpdateGnssPos(JNIEnv* env, jobject, jlong h, jdoubleArray p, jdouble s) {
    double c[3]; get3(env, p, c, 3); idr3d_update_gnss_pos(H(idr3d, h), c, s);
}
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idr3dUpdateGnssVel(JNIEnv* env, jobject, jlong h, jdoubleArray v, jdouble s) {
    double c[3]; get3(env, v, c, 3); idr3d_update_gnss_vel(H(idr3d, h), c, s);
}
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idr3dUpdateZupt(JNIEnv*, jobject, jlong h, jdouble s) { idr3d_update_zupt(H(idr3d, h), s); }
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idr3dUpdateZaru(JNIEnv* env, jobject, jlong h, jdoubleArray w, jdouble s) {
    double c[3]; get3(env, w, c, 3); idr3d_update_zaru(H(idr3d, h), c, s);
}
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idr3dUpdateNhc(JNIEnv*, jobject, jlong h, jdouble s) { idr3d_update_nhc(H(idr3d, h), s); }
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idr3dUpdateOdo(JNIEnv*, jobject, jlong h, jdouble v, jdouble s) { idr3d_update_odo(H(idr3d, h), v, s); }
JNIEXPORT void JNICALL
Java_com_offmaps_nav_IdrNative_idr3dUpdateBaro(JNIEnv*, jobject, jlong h, jdouble up, jdouble s) { idr3d_update_baro(H(idr3d, h), up, s); }
JNIEXPORT jdoubleArray JNICALL
Java_com_offmaps_nav_IdrNative_idr3dGetState(JNIEnv* env, jobject, jlong h) {
    double out[16]; idr3d_get_state(H(idr3d, h), out); return box(env, out, 16);
}
JNIEXPORT jdoubleArray JNICALL
Java_com_offmaps_nav_IdrNative_idr3dGetCov(JNIEnv* env, jobject, jlong h) {
    double out[15]; idr3d_get_cov(H(idr3d, h), out); return box(env, out, 15);
}

} // extern "C"
