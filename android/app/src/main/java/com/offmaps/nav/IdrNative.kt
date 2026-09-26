package com.offmaps.nav

/**
 * The Kotlin face of libidr, embedded via NDK/JNI (Phase 6). Every `external fun`
 * here binds one symbol in `app/src/main/cpp/idr_jni.cpp`, which calls the
 * unmodified `core/` C++ sources -- so these are the SAME parity-proven functions that
 * `py/core_bridge.py` reaches through ctypes and `tools/replay.cpp` links.
 *
 * Handles are opaque native pointers carried as `Long`. Multi-value returns come
 * back as `DoubleArray` (see the JNI bridge for each layout). The thin handle
 * classes below (Filter/SpeedCal/Align/MapMatcher/Gq) mirror core_bridge.py so
 * the on-device fusion reads like the harness loop it was validated against.
 */
object IdrNative {
    init { System.loadLibrary("idrjni") }

    // idr_filter -- planar 5-state ESKF [e, n, psi, v, b_g]
    external fun idrCreate(): Long
    external fun idrDestroy(h: Long)
    external fun idrInit(h: Long, e: Double, n: Double, psi: Double, v: Double)
    external fun idrSetProcessNoise(h: Long, arw: Double, brw: Double, srw: Double)
    external fun idrPredict(h: Long, dt: Double, gyroZ: Double)
    external fun idrUpdateSpeed(h: Long, v: Double, sigma: Double)
    external fun idrUpdateZupt(h: Long, gyroZ: Double)
    external fun idrUpdateCurvature(h: Long, aLat: Double, psidot: Double, baseSigma: Double)
    external fun idrUpdateGnssPos(h: Long, e: Double, n: Double, sigma: Double)
    external fun idrUpdateGnssVel(h: Long, v: Double, bearing: Double, sigmaV: Double, sigmaPsi: Double)
    external fun idrUpdateCrosstrack(h: Long, ne: Double, nn: Double, crossInnov: Double, sigma: Double)
    external fun idrUpdateHeading(h: Long, bearing: Double, sigma: Double)
    external fun idrSetMapKeepSpeed(h: Long, keep: Int)   // mask: 1 speed, 2 gyro bias, 3 both
    external fun idrSetHeadingSigma(h: Long, sigma: Double)
    external fun idrGetState(h: Long): DoubleArray   // [e, n, psi, v, b_g]
    external fun idrGetCov(h: Long): DoubleArray      // [P_ee, P_nn, P_psipsi]

    // spc -- Doppler speed-scale self-calibration
    external fun spcCreate(): Long
    external fun spcDestroy(h: Long)
    external fun spcPush(h: Long, vNn: Double, vDop: Double, w: Double)
    external fun spcSetLambda(h: Long, lam: Double)
    external fun spcFit(h: Long): DoubleArray         // [k, c, excited]
    external fun spcApply(h: Long, vNn: Double): Double
    external fun spcGet(h: Long): DoubleArray         // [k, c, excited]

    // aln -- mount alignment + re-align
    external fun alnCreate(): Long
    external fun alnDestroy(h: Long)
    external fun alnUpdate(h: Long, ax: Double, ay: Double, az: Double,
                           gx: Double, gy: Double, gz: Double, dt: Double)
    external fun alnGet(h: Long): DoubleArray         // [roll, pitch, yaw, changed]

    // gq -- GNSS quality (stateless)
    external fun gqTrust(cn0: Double, sv: Int, navic: Int, dop: Double, chi2: Double): Double
    external fun gqR(trust: Double): Double
    external fun gqSpoof(cn0: Double, sv: Int, chi2: Double, dopResid: Double): Int

    // mm -- offline map matching (fed by RoadMatcher from the bundled roads.bin)
    external fun mmCreate(): Long
    external fun mmDestroy(h: Long)
    external fun mmAddWay(h: Long, e: DoubleArray, n: DoubleArray, npts: Int, tunnel: Int, oneway: Int)
    external fun mmMatch(h: Long, e: Double, n: Double, psi: Double): DoubleArray  // [matched,footE,footN,bearing,cross,corridor]
    // mm (Phase 7b) -- Viterbi/HMM sequence decode
    external fun mmAddEdge(h: Long, a: Int, b: Int)
    external fun mmSetHmm(h: Long, sigmaEmit: Double, sigmaTrans: Double, betaBearing: Double)
    external fun mmMatchSeq(h: Long, e: DoubleArray, n: DoubleArray, psi: DoubleArray): DoubleArray  // [way|footE|footN|bearing|corridor] x n

    // vib (Phase 7c) -- high-rate vibration / pothole front-end
    external fun vibCreate(hz: Double): Long
    external fun vibDestroy(h: Long)
    external fun vibSetParams(h: Long, shockK: Double, refractoryS: Double, hpFc: Double, emaTau: Double)
    external fun vibSetShockFilter(h: Long, minPeak: Double, gapS: Double)
    external fun vibPush(h: Long, ax: Double, ay: Double, az: Double, dt: Double): Int
    external fun vibWindow(h: Long): DoubleArray      // [rms_clean, rms_raw, shock_frac, n_events]

    // idr3d (Phase 7a) -- 3D 16-state ESKF (bound, not in the live loop: Phase 6 write-up §6f, git show ec89099:README_PHASE6.md)
    external fun idr3dCreate(): Long
    external fun idr3dDestroy(h: Long)
    external fun idr3dInit(h: Long, p: DoubleArray, v: DoubleArray, q: DoubleArray)
    external fun idr3dSetNoise(h: Long, accelVrw: Double, gyroArw: Double, accelBiasRw: Double, gyroBiasRw: Double)
    external fun idr3dPredict(h: Long, dt: Double, a: DoubleArray, w: DoubleArray)
    external fun idr3dUpdateGnssPos(h: Long, p: DoubleArray, sigma: Double)
    external fun idr3dUpdateGnssVel(h: Long, v: DoubleArray, sigma: Double)
    external fun idr3dUpdateZupt(h: Long, sigma: Double)
    external fun idr3dUpdateZaru(h: Long, w: DoubleArray, sigma: Double)
    external fun idr3dUpdateNhc(h: Long, sigma: Double)
    external fun idr3dUpdateOdo(h: Long, vFwd: Double, sigma: Double)
    external fun idr3dUpdateBaro(h: Long, up: Double, sigma: Double)
    external fun idr3dGetState(h: Long): DoubleArray  // [p(3), v(3), q(4 w,x,y,z), b_a(3), b_g(3)]
    external fun idr3dGetCov(h: Long): DoubleArray    // diag of the 15x15 error cov
}

/** Native ESKF handle; method names mirror core_bridge.py::Filter. */
class Filter : AutoCloseable {
    private val h = IdrNative.idrCreate()
    fun init(e: Double, n: Double, psi: Double, v: Double) = IdrNative.idrInit(h, e, n, psi, v)
    fun setNoise(arw: Double, brw: Double, srw: Double) = IdrNative.idrSetProcessNoise(h, arw, brw, srw)
    fun predict(dt: Double, gz: Double) = IdrNative.idrPredict(h, dt, gz)
    fun updateSpeed(v: Double, s: Double) = IdrNative.idrUpdateSpeed(h, v, s)
    fun updateZupt(gz: Double) = IdrNative.idrUpdateZupt(h, gz)
    fun updateCurvature(aLat: Double, psidot: Double, baseSigma: Double) =
        IdrNative.idrUpdateCurvature(h, aLat, psidot, baseSigma)
    fun updateGnssPos(e: Double, n: Double, s: Double) = IdrNative.idrUpdateGnssPos(h, e, n, s)
    fun updateGnssVel(v: Double, brg: Double, sv: Double, sp: Double) =
        IdrNative.idrUpdateGnssVel(h, v, brg, sv, sp)
    fun updateCrosstrack(ne: Double, nn: Double, ci: Double, s: Double) =
        IdrNative.idrUpdateCrosstrack(h, ne, nn, ci, s)
    fun updateHeading(brg: Double, s: Double) = IdrNative.idrUpdateHeading(h, brg, s)
    fun setMapKeepSpeed(keep: Int) = IdrNative.idrSetMapKeepSpeed(h, keep)
    fun setHeadingSigma(sigma: Double) = IdrNative.idrSetHeadingSigma(h, sigma)
    fun state(): DoubleArray = IdrNative.idrGetState(h)
    fun cov(): DoubleArray = IdrNative.idrGetCov(h)
    override fun close() = IdrNative.idrDestroy(h)
}

/** Rolling-buffer Doppler self-calibration; mirrors core_bridge.py::SpeedCal. */
class SpeedCal : AutoCloseable {
    private val h = IdrNative.spcCreate()
    fun push(vNn: Double, vDop: Double, w: Double = 1.0) = IdrNative.spcPush(h, vNn, vDop, w)
    fun setLambda(lam: Double) = IdrNative.spcSetLambda(h, lam)
    /** @return Triple(k, c, excited) */
    fun fit(): Triple<Double, Double, Boolean> {
        val o = IdrNative.spcFit(h); return Triple(o[0], o[1], o[2] != 0.0)
    }
    fun apply(vNn: Double): Double = IdrNative.spcApply(h, vNn)
    fun get(): Triple<Double, Double, Boolean> {
        val o = IdrNative.spcGet(h); return Triple(o[0], o[1], o[2] != 0.0)
    }
    override fun close() = IdrNative.spcDestroy(h)
}

/** Online mount alignment; mirrors core_bridge.py::Align. */
class Align : AutoCloseable {
    private val h = IdrNative.alnCreate()
    fun update(ax: Double, ay: Double, az: Double, gx: Double, gy: Double, gz: Double, dt: Double) =
        IdrNative.alnUpdate(h, ax, ay, az, gx, gy, gz, dt)
    /** @return roll, pitch, yaw, changed */
    fun get(): AlignState {
        val o = IdrNative.alnGet(h); return AlignState(o[0], o[1], o[2], o[3] != 0.0)
    }
    override fun close() = IdrNative.alnDestroy(h)
}
data class AlignState(val roll: Double, val pitch: Double, val yaw: Double, val changed: Boolean)

/** Native greedy map matcher; mirrors core_bridge.py::MapMatcher (add_way / match). */
class MapMatcher : AutoCloseable {
    private val h = IdrNative.mmCreate()
    fun addWay(e: DoubleArray, n: DoubleArray, tunnel: Int, oneway: Int) =
        IdrNative.mmAddWay(h, e, n, e.size, tunnel, oneway)
    fun match(e: Double, n: Double, psi: Double): MatchResult {
        val o = IdrNative.mmMatch(h, e, n, psi)
        return MatchResult(o[0] != 0.0, o[1], o[2], o[3], o[4], o[5] != 0.0)
    }
    fun addEdge(a: Int, b: Int) = IdrNative.mmAddEdge(h, a, b)
    fun setHmm(sigmaEmit: Double, sigmaTrans: Double, betaBearing: Double) =
        IdrNative.mmSetHmm(h, sigmaEmit, sigmaTrans, betaBearing)
    /** Viterbi decode of a whole track; way index per step (-1 = no candidate). */
    fun matchSeq(e: DoubleArray, n: DoubleArray, psi: DoubleArray): SeqMatch {
        val o = IdrNative.mmMatchSeq(h, e, n, psi); val k = e.size
        return SeqMatch(IntArray(k) { o[it].toInt() }, o.copyOfRange(k, 2 * k), o.copyOfRange(2 * k, 3 * k),
                        o.copyOfRange(3 * k, 4 * k), BooleanArray(k) { o[4 * k + it] != 0.0 })
    }
    override fun close() = IdrNative.mmDestroy(h)
}
class SeqMatch(val way: IntArray, val footE: DoubleArray, val footN: DoubleArray,
               val bearing: DoubleArray, val corridor: BooleanArray)
/** foot = projection onto the road; cross = signed (left +) distance estimate -> road. */
data class MatchResult(val matched: Boolean, val footE: Double, val footN: Double,
                       val bearing: Double, val cross: Double, val corridor: Boolean)

/** Stateless GNSS-quality functions; mirror core_bridge.py::gq_*. */
object Gq {
    fun trust(cn0: Double, sv: Int, navic: Int, dop: Double, chi2: Double) =
        IdrNative.gqTrust(cn0, sv, navic, dop, chi2)
    fun r(trust: Double) = IdrNative.gqR(trust)
    fun spoof(cn0: Double, sv: Int, chi2: Double, dopResid: Double) =
        IdrNative.gqSpoof(cn0, sv, chi2, dopResid) != 0
}

/** High-rate vibration / pothole front-end (Phase 7c); mirrors core_bridge.py::Vib. */
class Vib(hz: Double) : AutoCloseable {
    private val h = IdrNative.vibCreate(hz)
    fun setParams(shockK: Double = 0.0, refractoryS: Double = 0.0, hpFc: Double = 0.0, emaTau: Double = 0.0) =
        IdrNative.vibSetParams(h, shockK, refractoryS, hpFc, emaTau)
    /** Road-hazard filter: an event needs a vertical jolt >= minPeak (m/s^2) and gapS since the last. */
    fun setShockFilter(minPeak: Double, gapS: Double) = IdrNative.vibSetShockFilter(h, minPeak, gapS)
    /** @return true when a shock event (pothole / bump) fires. */
    fun push(ax: Double, ay: Double, az: Double, dt: Double) = IdrNative.vibPush(h, ax, ay, az, dt) != 0
    /** [rms_clean (pothole-rejected), rms_raw, shock_frac, n_events] since the last call. */
    fun window(): DoubleArray = IdrNative.vibWindow(h)
    override fun close() = IdrNative.vibDestroy(h)
}

/** 3D 16-state ESKF (Phase 7a); mirrors core_bridge.py::Filter3D. Bound for replay /
 *  experiments; the live loop runs the planar filter (the one validated on real drives). */
class Filter3D : AutoCloseable {
    private val h = IdrNative.idr3dCreate()
    fun init(p: DoubleArray, v: DoubleArray, q: DoubleArray) = IdrNative.idr3dInit(h, p, v, q)
    fun setNoise(accelVrw: Double, gyroArw: Double, accelBiasRw: Double, gyroBiasRw: Double) =
        IdrNative.idr3dSetNoise(h, accelVrw, gyroArw, accelBiasRw, gyroBiasRw)
    fun predict(dt: Double, a: DoubleArray, w: DoubleArray) = IdrNative.idr3dPredict(h, dt, a, w)
    fun updateGnssPos(p: DoubleArray, sigma: Double) = IdrNative.idr3dUpdateGnssPos(h, p, sigma)
    fun updateGnssVel(v: DoubleArray, sigma: Double) = IdrNative.idr3dUpdateGnssVel(h, v, sigma)
    fun updateZupt(sigma: Double) = IdrNative.idr3dUpdateZupt(h, sigma)
    fun updateZaru(w: DoubleArray, sigma: Double) = IdrNative.idr3dUpdateZaru(h, w, sigma)
    fun updateNhc(sigma: Double) = IdrNative.idr3dUpdateNhc(h, sigma)
    fun updateOdo(vFwd: Double, sigma: Double) = IdrNative.idr3dUpdateOdo(h, vFwd, sigma)
    fun updateBaro(up: Double, sigma: Double) = IdrNative.idr3dUpdateBaro(h, up, sigma)
    fun state(): DoubleArray = IdrNative.idr3dGetState(h)
    fun cov(): DoubleArray = IdrNative.idr3dGetCov(h)
    override fun close() = IdrNative.idr3dDestroy(h)
}
