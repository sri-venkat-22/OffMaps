package com.offmaps.nav

import android.content.Context
import android.hardware.GeomagneticField
import android.os.Handler
import android.os.Looper
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * The live on-device fusion loop -- the piece Phase 6 exists to build. It runs
 * the SAME cadence as the validated harness loop (py/core_bridge.py::_run, the
 * `eskf` model) and layers on live GNSS fusion gated by the outage flag exactly
 * like tools/replay.cpp (`mask==0 => apply GNSS`).
 *
 * Per 10 Hz IMU step:      predict(dt, gyro_z), then (if the profile enables it)
 *                          curvature update when |psidot|>10deg/s.
 * Per 1 s:                 SpeedNet -> (v, sigma), always (it is the self-cal regressor).
 *                          Only while dead-reckoning, and only after the profile's
 *                          handover_s (before that the held Doppler speed IS the
 *                          `physics` baseline, which wins short outages), is it
 *                          FUSED: v < zupt_v -> ZUPT, else
 *                          speed update with sigma * sig_scale. Curvature likewise.
 * Per GNSS fix (unmasked): gq trust -> R, GNSS pos+vel updates, Doppler self-cal push.
 *                          Spoof check = position chi2 only; after REACQ_FIXES strong fixes
 *                          rejected in a row, re-seed the filter from GNSS.
 * Per GNSS fix (masked):   NOTHING fed to the filter -> pure dead-reckoning. This is
 *                          "honour the outage flag against the LIVE filter" (README android/5).
 * Per IMU step while dead-reckoning (masked, or no GNSS applied for 2 s) and roads are
 *                          loaded: map match -> cross-track + road-heading update (sigmas and
 *                          "may not change speed" from the profile, tuned on real val), the
 *                          Phase-5 `eskf_map` step verbatim (core_bridge.py::_run matcher
 *                          branch). Never while GNSS is healthy: a road centreline is not
 *                          your lane, so it only earns its place once GNSS is gone.
 *                          With the profile's mm_hmm (Phase 9) the road comes from RoadHmm
 *                          instead: it runs at 1 Hz whenever road snapping is on (so it is
 *                          locked on when the outage starts), and a road update is applied
 *                          only while dead-reckoning, on a confident match, while the filter
 *                          sigma <= 25 m, if it passes a chi2 gate -- edge_engine.MAP_HMM.
 *
 * Why speed aids are dead-reckoning only: with GNSS healthy, Doppler (~0.1 m/s) owns
 * speed. Fusing a net that is metres/second off on real roads pulled the filter away
 * from Doppler before every fix, the spoof check rejected the fix, and GNSS was locked
 * out for good -- measured on real IO-VNBD val drives with the old nn profile: 99% of
 * fixes rejected, 12.6 km median error WITH GNSS on (Phase 6 write-up §6c, git show ec89099:README_PHASE6.md).
 *
 * The filter settings (noise, zupt_v, curv, sig_scale) come from the model's
 * SpeedProfile -- the ones core_bridge.eskf_config gives that checkpoint on host.
 * Default is nn_real: retrained on real drives, settings tuned on validation drives.
 *
 * gyro_z is the gravity-projected yaw rate and a_lat is device-y accel -- the same
 * inputs phone_log.py feeds the validated model. The NN window is LEVELED (Level.kt,
 * z = up from a 30 s gravity EMA) to match the frame it was trained in.
 *
 * Everything mutating filter state runs on one background thread, so the native
 * core needs no locking; snapshots are posted to the main thread for the UI.
 */
class FusionEngine(
    private val context: Context,
    private val profileName: String = SpeedProfile.DEFAULT,
    private val listener: (NavState) -> Unit,
) {
    // ---- validated constants (mirror core_bridge.py::_run + phase4_gates.py) ----
    // Noise / ZUPT / curvature / sigma scale are per-model: see SpeedProfile.
    private val YAW_SIGN = -1.0                 // phone_log default gyro->heading sign
    private val CURV_GATE = Math.toRadians(10.0)
    private val DOPPLER_SIGMA = 0.1             // chip Doppler ground-speed noise (phase4_gates)
    private val NOMINAL_DOP = 1.5               // DOP proxy: not in the logging contract
    private val GNSS_BASE_SIGMA = 3.0           // healthy GNSS position sigma (m), for chi2
    private val DECIMATE_NS = 100_000_000L      // 10 Hz IMU grid
    private val GRAV_A = 1.0 - exp(-1.0 / (0.5 * Features.HZ))  // gravity EMA (phone_log _gravity_up)
    private val NN_EVERY_STEPS = Features.HZ.toInt()            // 10 steps == 1 s
    private val GNSS_STALE_STEPS = 2 * Features.HZ.toInt()      // no GNSS applied for 2 s -> dead-reckoning
    private val MM_MAX_CROSS = 25.0             // ignore a "nearest road" further than this (m): off-road / wrong street
    private val TRUST_APPLIED = 0.1             // a fix counts as "GNSS healthy" only at trust >= this (sigma <= 30 m)
    private val REACQ_FIXES = 5                 // consecutive rejected strong fixes -> re-seed from GNSS
    private val K_MIN = 1.0 / 3.0               // self-cal scale outside [K_MIN,K_MAX] is a degenerate
    private val SHOCK_MIN_SPEED = 3.0           // m/s: below this a "shock" is the phone being handled, not a road
    // pothole markers (py/tests/test_pothole_filter.py): the adaptive detector alone marked ~27/km on the
    // first Redmi drives (road texture); a 1 g vertical jolt, one mark per 1 s and none within 30 m of
    // another leaves ~2/km
    private val SHOCK_MIN_PEAK = 10.0           // m/s^2 vertical (high-passed, along gravity)
    private val SHOCK_GAP_S = 1.0
    private val SHOCK_MERGE_M = 30.0
    private val K_MAX = 3.0                     //   fit (NN not tracking speed), not a calibration
    // strict stop detector (== edge_engine.STOP_*): SpeedNet < STOP_V for STOP_S s AND max |gyro| < STOP_GYRO
    private val STOP_V = 0.3; private val STOP_S = 3; private val STOP_GYRO = 0.03
    private val PRE_KEEP = 600                  // trusted (Doppler, NN) pairs kept for the fusion head's context
    // HMM road matching (== edge_engine.MAP_HMM, py/phase9_map_eval.py): confident match, filter sigma
    // gate, chi2 innovation gate; the road update moves neither speed nor gyro bias (keep mask 3)
    private val HMM_CONF = 0.9
    private val HMM_SIGMA_MAX = 25.0
    private val HMM_CHI2 = 6.63                 // 1 dof, 99 %
    private val HMM_CROSS_SCALE = 0.6           // cross-track sigma = this x road half-width (>= 1 m)
    private val HMM_HEADING_SIGMA = Math.toRadians(6.0)

    private val main = Handler(Looper.getMainLooper())

    // ---- native + model handles (created in start, freed in stop) ----
    private var filter: Filter? = null
    private var speedCal: SpeedCal? = null
    private var align: Align? = null
    private var vib: Vib? = null                // Phase 7c pothole/bump detector, raw sensor rate
    private var net: SpeedNet? = null
    private var profile: SpeedProfile? = null
    private var enu = Enu()
    private var roadMatcher: RoadMatcher? = null
    private var roadHmm: RoadHmm? = null
    private var hmmSteps = 0
    private var hmmSnapped = false              // the last 1 Hz HMM decision applied a road update
    @Volatile private var roads: RoadNetwork? = null

    // ---- 10 Hz decimated IMU ring buffer for the NN window ----
    private val accBuf = Array(Features.MAX_WIN) { DoubleArray(3) }
    private val gyrBuf = Array(Features.MAX_WIN) { DoubleArray(3) }
    private var win = Features.WIN              // the profile's NN window (samples); set in start()
    private var bufFill = 0
    private var bufHead = 0

    // ---- streaming state (fusion thread only) ----
    private val grav = DoubleArray(3)
    private val level = Level()                 // slow-gravity leveling for the NN inputs (Level.kt)
    private val accL = DoubleArray(3); private val gyrL = DoubleArray(3)
    private var haveGrav = false
    private var lastStepNs = 0L
    private val decimator = Decimator(DECIMATE_NS)
    private var initialized = false
    private var stepsSinceNn = 0
    private var steps = 0L
    private var vHeld = 0.0                     // last NN speed after Doppler self-cal
    private var sigHeld = 1.0
    private var pStopHeld = Double.NaN          // SpeedNet's calibrated p(stopped), latest 1 s step
    private var vNnRaw = 0.0                    // last NN speed (affine only) -- regressor for self-cal
    private var lastAlat = 0.0
    private var sumSigSq = 0.0
    private var calPushes = 0
    private var k = 1.0; private var c = 0.0
    // last published scalars carried across 10 Hz frames
    private var lastGnssE = 0.0; private var lastGnssN = 0.0; private var haveGnss = false
    private var lastTrust = 0.0; private var lastSpoof = false
    private var lastCn0 = 0.0; private var lastSv = 0; private var lastNavic = 0
    private var lastDrift = 0.0; private var lastAlignChanged = false
    private var stepsSinceGnss = 0              // IMU steps since GNSS was last fed to the filter
    private var rejects = 0                     // consecutive strong fixes rejected by the spoof check
    private var drSteps = 0                     // IMU steps since dead-reckoning began
    private var snapped = false                 // map match applied on the latest step

    // raw-rate measurement for Vib (its DSP assumes a fixed rate; phones differ, 100-500 Hz)
    private var rawT0Ns = 0L
    private var rawCount = 0
    private var shockCount = 0
    private val pendingShocks = ArrayList<Double>()   // lat, lon pairs not yet published
    private val shockEn = ArrayList<DoubleArray>()    // ENU of every mark this session (SHOCK_MERGE_M)

    // ---- Phase 8: learned fusion head, heading aids, strict ZUPT ----
    private var head: FusionHead? = null
    private var headState: FusionHead.State? = null
    private val pre = ArrayDeque<DoubleArray>()        // (Doppler, NN raw) per trusted fix
    private val yawAlign = YawAlign()
    private val mag = DoubleArray(3); private var haveMag = false; private var lastMagNs = 0L
    private var stillS = 0
    private var lastYaw = 0.0
    var headingSeed = "none"; private set
    private val vitHist = ArrayDeque<DoubleArray>(); private var vitWay = -1; private var vitCor = false
    @Volatile private var vehicle = "car"          // "car" | "two_wheeler" (two-wheeler: lean-compensated yaw)

    @Volatile private var masked = false
    // initial road-snap state from the profile (nn_real: on since the Phase 9 HMM matcher)
    @Volatile private var mapAid = try { SpeedProfile.fromAssets(context, profileName).mapDefaultOn } catch (e: Exception) { true }
    @Volatile private var running = false

    /** Outage Simulator toggle. Volatile: set from the UI thread, read on the fusion thread. */
    fun setMasked(m: Boolean) { masked = m }
    fun isMasked() = masked

    /** Road-snapping toggle (only acts while dead-reckoning). */
    fun setMapAid(on: Boolean) { mapAid = on }
    fun isMapAid() = mapAid

    /** Vehicle type: "two_wheeler" switches the yaw rate to the lean-compensated one. */
    fun setVehicle(v: String) { vehicle = v }
    fun getVehicle() = vehicle

    /** Hand over the offline road network once it has loaded (any thread). */
    fun setRoads(r: RoadNetwork?) { roads = r }

    fun start() {
        if (running) return
        val p = SpeedProfile.fromAssets(context, profileName)
        require(p.win in 1..Features.MAX_WIN) { "profile window ${p.win} > ${Features.MAX_WIN}" }
        profile = p
        win = p.win
        filter = Filter().also {
            it.setNoise(p.arw, p.brw, p.srw)
            it.setMapKeepSpeed(if (p.mmHmm) 3 else if (p.mmKeepSpeed) 1 else 0)   // HMM: keep speed + gyro bias
        }
        speedCal = SpeedCal()
        align = Align()
        net = SpeedNet.fromAssets(context, p)
        head = p.fusionHead?.let { FusionHeadAsset.load(context, it) }
        mapAid = p.mapDefaultOn
        resetStreaming()
        running = true
    }

    fun stop() {
        running = false
        filter?.close(); speedCal?.close(); align?.close(); net?.close(); roadMatcher?.close(); vib?.close()
        filter = null; speedCal = null; align = null; net = null; roadMatcher = null; roadHmm = null; profile = null; vib = null
    }

    private fun resetStreaming() {
        bufFill = 0; bufHead = 0; haveGrav = false; initialized = false; level.reset()
        decimator.reset(); lastStepNs = 0L
        stepsSinceNn = 0; steps = 0; vHeld = 0.0; sigHeld = 1.0; vNnRaw = 0.0; pStopHeld = Double.NaN
        sumSigSq = 0.0; calPushes = 0; k = 1.0; c = 0.0; haveGnss = false
        lastTrust = 0.0; lastSpoof = false; lastDrift = 0.0; lastAlignChanged = false
        stepsSinceGnss = 0; rejects = 0; drSteps = 0; snapped = false
        rawT0Ns = 0L; rawCount = 0; shockCount = 0; pendingShocks.clear(); shockEn.clear()
        enu = Enu()                                      // new session -> new origin at its first fix
        headState = null; pre.clear(); yawAlign.reset(); haveMag = false; stillS = 0; lastYaw = 0.0
        headingSeed = "none"
        roadHmm = null; hmmSteps = 0; hmmSnapped = false  // its ENU frame is the session's
    }

    private fun deadReckoning() = masked || stepsSinceGnss > GNSS_STALE_STEPS

    /** Doppler self-cal (frozen k,c), used only when the fitted scale is physical. */
    private fun calApply(v: Double) = if (k in K_MIN..K_MAX) k * v + c else v

    /**
     * Phase-5 map aiding (core_bridge.py::_run, matcher branch): pin the cross-track
     * to the matched road and pull heading onto its bearing. The road never constrains
     * distance travelled -- that stays the speed model's job.
     */
    private fun mapMatchUpdate(f: Filter, dt: Double) {
        snapped = false
        if (!deadReckoning()) { vitHist.clear(); vitWay = -1 }
        val p = profile ?: return
        if (p.mmHmm) { hmmUpdate(f, dt); return }
        if (!mapAid || !deadReckoning()) return
        val r = roads ?: return
        val rm = roadMatcher ?: RoadMatcher(r, enu, p.mmViterbi).also { roadMatcher = it }
        val st = f.state()
        val m = if (p.mmViterbi) {                                    // live fixed-lag Viterbi (Phase 7b online)
            if (drSteps % NN_EVERY_STEPS == 1) {
                vitHist.addLast(doubleArrayOf(st[0], st[1], st[2]))
                if (vitHist.size > RoadMatcher.LAG) vitHist.removeFirst()
                val d = rm.decode(vitHist.toList()); vitWay = d.first; vitCor = d.second
            }
            rm.project(vitWay, st[0], st[1], st[2], vitCor)
        } else rm.match(st[0], st[1], st[2])
        if (!m.matched || abs(m.cross) > MM_MAX_CROSS) return
        // Phase 8 safeguards (edge_engine._map_step): on real OSM roads a snap to the wrong street
        // at a junction, plus its heading pull, made outages far worse (Phase 8 §8d; re-measured by py/phase9_map_eval.py)
        if (p.mmUnique && !m.corridor) return
        val db = abs(wrapPi(m.bearing - st[2]))
        if (minOf(db, PI - db) > Math.toRadians(p.mmGateDeg)) return
        val nx = -cos(m.bearing); val ny = sin(m.bearing)            // left-normal (matches C++ cross sign)
        f.updateCrosstrack(nx, ny, m.cross, if (m.corridor) 0.3 else p.mmCrossSigma)
        if (p.mmHeading) {
            val tgt = if (db < PI / 2) m.bearing else m.bearing + PI
            f.updateHeading(tgt, Math.toRadians(if (m.corridor) 1.0 else p.mmHeadingSigmaDeg))
        }
        snapped = true
    }

    /** Phase 9 road matching, the edge engine's _map_step_hmm: the HMM runs at 1 Hz while
     *  road snapping is on, GNSS or not; its road is applied only while dead-reckoning. */
    private fun hmmUpdate(f: Filter, dt: Double) {
        if (!mapAid) { roadHmm?.reset(); hmmSnapped = false; return }
        val r = roads ?: return
        if (!enu.hasOrigin) return
        val hmm = roadHmm ?: RoadHmm(r, enu).also { roadHmm = it }
        val st = f.state()
        hmm.addTravel(abs(st[3]) * dt)
        // a decision holds for its second, so the published flag doesn't blink at 10 Hz
        if (++hmmSteps % NN_EVERY_STEPS != 0) { snapped = hmmSnapped && deadReckoning(); return }
        hmmSnapped = false
        var cov = f.cov()
        val ps = sqrt(maxOf(cov[0] + cov[1], 0.0) / 2)
        val m = hmm.update(st[0], st[1], st[2], abs(st[3]), ps) ?: return
        if (!deadReckoning() || m.conf < HMM_CONF || abs(st[3]) < 1.0 || ps > HMM_SIGMA_MAX) return
        val nE = -cos(m.bearing); val nN = sin(m.bearing)            // left-normal (matches C++ cross sign)
        val cross = (m.footE - st[0]) * nE + (m.footN - st[1]) * nN
        val sc = maxOf(1.0, HMM_CROSS_SCALE * m.halfWidth)
        if (cross * cross / (nE * nE * cov[0] + nN * nN * cov[1] + sc * sc) > HMM_CHI2) return   // wrong road
        f.updateCrosstrack(nE, nN, cross, sc)
        if (m.segLen >= 25.0 && m.endDist >= 8.0) {                  // not at a vertex / on a short piece
            val dpsi = wrapPi(m.bearing - st[2]); cov = f.cov()
            if (dpsi * dpsi / (cov[2] + HMM_HEADING_SIGMA * HMM_HEADING_SIGMA) <= HMM_CHI2)
                f.updateHeading(m.bearing, HMM_HEADING_SIGMA)
        }
        snapped = true; hmmSnapped = true
    }

    private fun wrapPi(a: Double): Double {
        val x = (a + PI) % (2 * PI)
        return (if (x < 0) x + 2 * PI else x) - PI
    }

    /** Magnetometer (device frame, uT): EMA only; used once, to seed heading at a course-less first fix. */
    fun onMag(tNs: Long, mx: Double, my: Double, mz: Double) {
        if (!running) return
        if (!haveMag) { mag[0] = mx; mag[1] = my; mag[2] = mz; haveMag = true }
        else {
            val a = 1.0 - exp(-((tNs - lastMagNs).coerceAtLeast(0L) * 1e-9) / 1.0)
            mag[0] += a * (mx - mag[0]); mag[1] += a * (my - mag[1]); mag[2] += a * (mz - mag[2])
        }
        lastMagNs = tNs
    }

    /** Vehicle forward axis (device frame): GNSS-aided yaw alignment once converged, else the mount guess. */
    private fun forward(): DoubleArray =
        if (yawAlign.valid) yawAlign.forward(Level.basis(level.up())) else HeadingAids.defaultForward(level.up())

    // ---------------- IMU: high-rate, decimated to 10 Hz ----------------
    fun onImu(tNs: Long, ax: Double, ay: Double, az: Double, gx: Double, gy: Double, gz: Double) {
        if (!running) return
        highRate(tNs, ax, ay, az)                        // every raw sample: pothole / bump front-end
        if (!decimator.accept(tNs)) return               // keep the 10 Hz grid

        // fast (0.5 s) gravity EMA: the "fast"/"coord" yaw projections use it (phone_log convention)
        if (!haveGrav) { grav[0] = ax; grav[1] = ay; grav[2] = az; haveGrav = true }
        else { grav[0] += GRAV_A * (ax - grav[0]); grav[1] += GRAV_A * (ay - grav[1]); grav[2] += GRAV_A * (az - grav[2]) }
        lastAlat = ay                                    // device-y accel = curvature lateral accel

        // mount alignment / re-mount detector: a re-mount re-levels the NN inputs at once
        // (medium vs slow gravity CUSUM, core/align.cpp: ~1 false alarm/hour on real drives)
        val dtStep = if (lastStepNs == 0L) 1.0 / Features.HZ else (tNs - lastStepNs) * 1e-9
        align?.let { a ->
            a.update(ax, ay, az, gx, gy, gz, dtStep)
            if (a.get().changed) { lastAlignChanged = true; level.onRemount(); yawAlign.reset() }
        }

        // push the LEVELED sample into the NN ring buffer: SpeedNet was trained on a
        // z-up frame, and a dash-mounted phone has y up (py/model/mount.py)
        level.step(doubleArrayOf(ax, ay, az), doubleArrayOf(gx, gy, gz), accL, gyrL)
        yawAlign.addAcc(accL[0], accL[1])
        // yaw rate about TRUE vertical (HeadingAids): "slow" = 30 s gravity, exact for a car; the old
        // 0.5 s projection ("fast") under-reads turns by cos(atan(v*psidot/g)); a two-wheeler leans,
        // so it needs the coordinated-turn tilt-back ("coord")
        val mode = if (vehicle == "two_wheeler") "coord" else (profile?.yawMode ?: "fast")
        val gyroZ = HeadingAids.yawRate(mode, doubleArrayOf(gx, gy, gz), grav.copyOf(), level.up(),
            if (mode == "coord") forward() else null, filter?.state()?.get(3) ?: 0.0)
        lastYaw = gyroZ
        accBuf[bufHead][0] = accL[0]; accBuf[bufHead][1] = accL[1]; accBuf[bufHead][2] = accL[2]
        gyrBuf[bufHead][0] = gyrL[0]; gyrBuf[bufHead][1] = gyrL[1]; gyrBuf[bufHead][2] = gyrL[2]
        bufHead = (bufHead + 1) % win
        if (bufFill < win) bufFill++

        if (!initialized) { lastStepNs = tNs; return }   // wait for first GNSS fix to seed the filter
        val f = filter ?: return
        val p = profile ?: return

        f.predict(dtStep, gyroZ)
        lastStepNs = tNs
        drSteps = if (deadReckoning()) drSteps + 1 else 0
        if (drSteps == 1) headState = head?.start(f.state()[3], k, c, pre.toList())   // outage begins: freeze context
        else if (drSteps == 0) headState = null

        // 1 s NN cadence: speed / ZUPT update
        stepsSinceNn++
        if (stepsSinceNn >= NN_EVERY_STEPS && bufFill >= win) {
            stepsSinceNn = 0
            val (accW, gyrW) = orderedWindow()
            val pred = net!!.predict(accW, gyrW)          // [v_calibrated, sigma]
            vNnRaw = pred[0]; sigHeld = pred[1]           // always: the self-cal regressor
            pStopHeld = pred[2]                           // display + edge output only (no ZUPT from it)
            vHeld = calApply(vNnRaw)                      // Doppler self-cal (frozen k,c)
            var gMax = 0.0; var yawSum = 0.0
            for (i in win - NN_EVERY_STEPS until win) {
                gMax = maxOf(gMax, sqrt(gyrW[i][0] * gyrW[i][0] + gyrW[i][1] * gyrW[i][1] + gyrW[i][2] * gyrW[i][2]))
                yawSum += gyrW[i][2]
            }
            stillS = if (vNnRaw < STOP_V && gMax < STOP_GYRO) stillS + 1 else 0
            val hd = head; val hs = headState
            if (deadReckoning() && p.zuptStrict && stillS >= STOP_S) {
                f.updateZupt(yawSum / NN_EVERY_STEPS)     // strict stop: v = 0 and gyro bias (ZARU)
            } else if (deadReckoning() && hd != null && hs != null) {
                // learned fusion head (FusionHead.kt): the speed measurement AND its sigma
                val last = win - NN_EVERY_STEPS
                val r = hd.step(hs, vNnRaw, sigHeld, drSteps / Features.HZ,
                                accW.copyOfRange(last, win), gyrW.copyOfRange(last, win))
                f.updateSpeed(r[0], r[1])
            } else if (deadReckoning() && drSteps > (p.handoverS * Features.HZ).toInt()) {   // else Doppler speed held
                val zv = p.zuptV
                if (zv != null && vHeld < zv) f.updateZupt(gyroZ) else f.updateSpeed(vHeld, sigHeld * p.sigScale)
            }
        }

        // curvature update (a speed aid too, so dead-reckoning only), gated at |psidot|>10deg/s; off for nn_real
        if (p.curv && deadReckoning()) {
            val st = f.state()
            val psidot = gyroZ - st[4]
            if (abs(psidot) > CURV_GATE) f.updateCurvature(lastAlat, psidot, p.curvSigma)
        }

        stepsSinceGnss++
        mapMatchUpdate(f, dtStep)                         // no-op unless roads are loaded (and, except HMM, dead-reckoning)

        steps++
        publish(newGnss = false)
    }

    // ---------------- GNSS: ~1 Hz, gated by the outage flag ----------------
    fun onGnss(lat: Double, lon: Double, doppSpeed: Double, bearingDeg: Double,
               cn0: Double, svUsed: Int, navicSv: Int) {
        if (!running) return
        enu.setOrigin(lat, lon)
        val en = enu.toEn(lat, lon)
        lastGnssE = en[0]; lastGnssN = en[1]; haveGnss = true
        lastCn0 = cn0; lastSv = svUsed; lastNavic = navicSv
        val bearingRad = Math.toRadians(bearingDeg)
        val hasBearing = !bearingDeg.isNaN()
        val hasSpeed = !doppSpeed.isNaN()

        if (!initialized) {                               // first fix seeds the filter at the origin
            val f0 = filter!!
            val v0 = if (hasSpeed) doppSpeed else 0.0
            if (hasBearing && (!hasSpeed || doppSpeed >= 1.0)) {          // a real course over ground
                f0.init(0.0, 0.0, bearingRad, v0); headingSeed = "gnss"
            } else {                                                      // parked: GNSS bearing is noise
                val decl = Math.toRadians(GeomagneticField(lat.toFloat(), lon.toFloat(), 0f,
                                                           System.currentTimeMillis()).declination.toDouble())
                val h = if (haveMag) HeadingAids.magHeading(mag, level.up(), forward(), decl) else null
                if (h != null) { f0.init(0.0, 0.0, h, v0); f0.setHeadingSigma(HeadingAids.MAG_SIGMA); headingSeed = "magnetometer" }
                else { f0.init(0.0, 0.0, 0.0, v0); f0.setHeadingSigma(HeadingAids.UNKNOWN_SIGMA); headingSeed = "unknown" }
            }
            initialized = true; stepsSinceGnss = 0
            publish(newGnss = true)
            return
        }
        val f = filter!!

        // Honour the outage flag: masked -> feed the filter NOTHING (pure dead-reckoning).
        // GNSS is still received and drawn as ground truth (same as the logger's Outage Simulator).
        if (masked) {
            lastTrust = 0.0
            lastDrift = hypot(f.state()[0] - lastGnssE, f.state()[1] - lastGnssN)  // live DR error
            publish(newGnss = true)
            return
        }

        // --- healthy GNSS: continuous-trust fusion (no mode switch) ---
        var st = f.state(); val cov = f.cov()
        val de = lastGnssE - st[0]; val dn = lastGnssN - st[1]
        val chi2 = de * de / (cov[0] + GNSS_BASE_SIGMA * GNSS_BASE_SIGMA) +
                   dn * dn / (cov[1] + GNSS_BASE_SIGMA * GNSS_BASE_SIGMA)
        var trust = Gq.trust(cn0, svUsed, navicSv, NOMINAL_DOP, chi2)
        // Position chi2 only -- the configuration Phase 4's spoof gate validated. Its fixed
        // 5 m/s Doppler-residual test was never gated and cannot work with a speed net that
        // is ~5 m/s off on real roads (it rejected nearly every real fix).
        lastSpoof = Gq.spoof(cn0, svUsed, chi2, 0.0)
        if (lastSpoof) {
            rejects++
            if (rejects >= REACQ_FIXES) {
                // Strong GNSS has disagreed for REACQ_FIXES s straight: after a long outage
                // with an off-distribution net, our dead-reckoning is what's wrong, and its
                // covariance is too tight to ever admit the truth. Re-seed from GNSS.
                // (Trade-off: a spoof sustained this long is then followed.)
                f.init(lastGnssE, lastGnssN, if (hasBearing) bearingRad else st[2], if (hasSpeed) doppSpeed else st[3])
                st = f.state()
                rejects = 0; lastSpoof = false
                trust = Gq.trust(cn0, svUsed, navicSv, NOMINAL_DOP, 0.0)
            }
        } else rejects = 0
        lastTrust = trust

        if (!lastSpoof) {
            // Weak fixes (few satellites, low C/N0) are still fused with their big R, but
            // they don't end dead-reckoning: otherwise the NN and road aids switch off
            // while GNSS contributes almost nothing (seen on the emulator: 0 satellites).
            if (trust >= TRUST_APPLIED) stepsSinceGnss = 0
            val sigmaPos = sqrt(Gq.r(trust))
            f.updateGnssPos(lastGnssE, lastGnssN, sigmaPos)
            if (hasSpeed && hasBearing) {
                val sigmaV = (DOPPLER_SIGMA / trust.coerceAtLeast(3e-3)).coerceIn(0.1, 50.0)
                val sigmaPsi = (Math.toRadians(3.0) / trust.coerceAtLeast(3e-3)).coerceIn(Math.toRadians(1.0), Math.toRadians(90.0))
                f.updateGnssVel(doppSpeed, bearingRad, sigmaV, sigmaPsi)
            }
            // Doppler self-calibration: regress Doppler on the (noisy) NN speed, Deming-weighted.
            if (hasSpeed && trust >= TRUST_APPLIED) {                 // yaw alignment + fusion-head context
                yawAlign.onFix(lastStepNs * 1e-9, doppSpeed)
                pre.addLast(doubleArrayOf(doppSpeed, vNnRaw)); if (pre.size > PRE_KEEP) pre.removeFirst()
            }
            if (hasSpeed && vNnRaw > 0.0 && trust >= TRUST_APPLIED) {   // trusted Doppler only
                sumSigSq += sigHeld * sigHeld; calPushes++
                val meanSigSq = (sumSigSq / calPushes).coerceAtLeast(1e-9)
                speedCal!!.setLambda(DOPPLER_SIGMA * DOPPLER_SIGMA / meanSigSq)
                speedCal!!.push(vNnRaw, doppSpeed, trust)      // weight healthy fixes by trust
                val fit = speedCal!!.fit(); k = fit.first; c = fit.second
            }
        }
        lastDrift = hypot(st[0] - lastGnssE, st[1] - lastGnssN)
        publish(newGnss = true)
    }

    /**
     * Phase 7c on the phone: the raw accelerometer at full sensor rate through the
     * core's vibration front-end (high-pass + adaptive shock detector, rotation
     * invariant). The first 2 s measure this phone's actual rate, since the DSP is
     * built for a fixed one. Each shock's leading edge is placed at the fused position.
     */
    private fun highRate(tNs: Long, ax: Double, ay: Double, az: Double) {
        val v = vib
        if (v == null) {
            if (rawCount == 0) rawT0Ns = tNs
            rawCount++
            val span = (tNs - rawT0Ns) * 1e-9
            if (span >= 2.0) vib = Vib((rawCount - 1) / span).also { it.setShockFilter(SHOCK_MIN_PEAK, SHOCK_GAP_S) }
            return
        }
        if (v.push(ax, ay, az, 0.0) && initialized) {
            val st = filter?.state() ?: return
            if (st[3] < SHOCK_MIN_SPEED) return          // first desk test: 20 "potholes" from handling the phone
            if (!enu.hasOrigin) return
            if (shockEn.any { hypot(it[0] - st[0], it[1] - st[1]) < SHOCK_MERGE_M }) return   // already marked
            shockEn.add(doubleArrayOf(st[0], st[1]))
            val ll = enu.toLatLon(st[0], st[1])
            shockCount++; pendingShocks.add(ll[0]); pendingShocks.add(ll[1])
        }
    }

    /** Copy the ring buffer into oldest->newest order for the NN. */
    private fun orderedWindow(): Pair<Array<DoubleArray>, Array<DoubleArray>> {
        val a = Array(win) { DoubleArray(3) }
        val g = Array(win) { DoubleArray(3) }
        for (i in 0 until win) {
            val idx = (bufHead + i) % win              // bufHead points at the oldest slot
            a[i][0] = accBuf[idx][0]; a[i][1] = accBuf[idx][1]; a[i][2] = accBuf[idx][2]
            g[i][0] = gyrBuf[idx][0]; g[i][1] = gyrBuf[idx][1]; g[i][2] = gyrBuf[idx][2]
        }
        return a to g
    }

    private fun publish(newGnss: Boolean) {
        val f = filter ?: return
        if (!enu.hasOrigin) return
        val st = f.state()
        val cv = f.cov()
        val ll = enu.toLatLon(st[0], st[1])
        val gll = enu.toLatLon(lastGnssE, lastGnssN)
        val s = NavState(
            e = st[0], n = st[1], psi = st[2], v = st[3], bg = st[4],
            lat = ll[0], lon = ll[1], gnssLat = gll[0], gnssLon = gll[1],
            gnssE = lastGnssE, gnssN = lastGnssN, hasGnss = haveGnss, newGnss = newGnss,
            masked = masked, trust = lastTrust, spoof = lastSpoof,
            k = k, c = c, cn0 = lastCn0, svUsed = lastSv, navicSv = lastNavic,
            driftM = lastDrift, alignChanged = lastAlignChanged, steps = steps,
            outageActive = deadReckoning(), snapped = snapped,
            roadsLoaded = roads != null, windowWays = roadMatcher?.windowWays ?: 0,
            shocks = shockCount, newShocks = pendingShocks.toDoubleArray(),
            calUsed = k in K_MIN..K_MAX,
            posSigma = sqrt(maxOf(cv[0] + cv[1], 0.0) / 2),
            pStop = pStopHeld,
        )
        pendingShocks.clear()
        lastAlignChanged = false
        main.post { listener(s) }
    }
}

/** One published fusion snapshot (immutable; safe to hand to the main thread). */
data class NavState(
    val e: Double, val n: Double, val psi: Double, val v: Double, val bg: Double,
    val lat: Double, val lon: Double, val gnssLat: Double, val gnssLon: Double,
    val gnssE: Double, val gnssN: Double, val hasGnss: Boolean, val newGnss: Boolean,
    val masked: Boolean, val trust: Double, val spoof: Boolean,
    val k: Double, val c: Double,
    val cn0: Double, val svUsed: Int, val navicSv: Int,
    val driftM: Double, val alignChanged: Boolean, val steps: Long,
    val outageActive: Boolean,       // dead-reckoning: Outage Simulator on, or no GNSS for 2 s
    val snapped: Boolean,            // road match applied this step
    val roadsLoaded: Boolean, val windowWays: Int,
    val shocks: Int,                 // potholes / bumps detected this session (Phase 7c)
    val newShocks: DoubleArray,      // lat, lon pairs detected since the previous snapshot
    val calUsed: Boolean,            // Doppler self-cal fit is physical (1/3 <= k <= 3) and applied
    val posSigma: Double,            // filter's 1-sigma position uncertainty (m): the map's accuracy circle
    val pStop: Double,               // SpeedNet's calibrated p(stopped) (NaN until the first 1 s step)
)
