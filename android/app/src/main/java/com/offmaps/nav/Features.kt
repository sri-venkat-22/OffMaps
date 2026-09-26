package com.offmaps.nav

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * The IMU-window -> feature-tensor contract, ported verbatim from
 * py/model/features.py. Training and Python inference share that one
 * implementation; this is the on-device third copy, and it must agree to the
 * bit or the shipped net silently rots (same warning as the Python docstring).
 *
 * Input:  a window of leveled acc/gyro at 10 Hz (z up); its length is the
 *         profile's window (SpeedProfile.win), 20 samples for version 1.
 * Output: (C, T) float, flattened channel-major (index = channel*T + t) -- the ONNX layout.
 *   version 1: [ax,ay,az, gx,gy,gz, |a|,|g|,|jerk|] / SCALE
 *   version 2: heading-free physics channels (see windowFeaturesV2) / SCALE2
 *
 * applyCalib mirrors nn_model.apply_calib: the affine (a,b,s) recalibration
 * stored in the checkpoint. (a,b,s) are no longer hard-coded here -- they come
 * from the shipped model's profile (SpeedProfile, <model>.profile.json).
 */
object Features {
    const val C = 9
    const val WIN = 20
    const val HZ = 10.0
    private val SCALE = doubleArrayOf(10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 10.0, 1.0, 10.0)

    const val C2 = 10
    const val WIN2 = 200
    const val MAX_WIN = 200            // the ring buffer FusionEngine keeps
    private const val LP_N = 5         // 0.5 s moving mean
    private const val VIB_N = 10       // 1 s moving mean
    private const val TURN_MIN = 0.05  // rad/s
    private const val TURN_MAX = 40.0  // m/s
    private val SCALE2 = doubleArrayOf(1.0, 1.0, 0.2, 0.1, 1.0, 1.0, 0.1, 10.0, 10.0, 1.0)

    fun channels(version: Int): Int = if (version == 1) C else C2

    fun features(version: Int, acc: Array<DoubleArray>, gyro: Array<DoubleArray>): FloatArray =
        if (version == 1) windowFeatures(acc, gyro) else windowFeaturesV2(acc, gyro)

    /**
     * Build the version-1 (9,T) feature tensor from the most recent window.
     * @param acc  T rows of [ax,ay,az]
     * @param gyro T rows of [gx,gy,gz]
     * @return FloatArray(C*T), channel-major, ready as the ONNX "imu" input.
     */
    fun windowFeatures(acc: Array<DoubleArray>, gyro: Array<DoubleArray>): FloatArray {
        val t = acc.size
        val out = FloatArray(C * t)
        var prevAx = acc[0][0]; var prevAy = acc[0][1]; var prevAz = acc[0][2] // prepend acc[:1] -> jerk[0]=0
        for (i in 0 until t) {
            val ax = acc[i][0]; val ay = acc[i][1]; val az = acc[i][2]
            val gx = gyro[i][0]; val gy = gyro[i][1]; val gz = gyro[i][2]
            val an = sqrt(ax * ax + ay * ay + az * az)
            val gn = sqrt(gx * gx + gy * gy + gz * gz)
            val dx = ax - prevAx; val dy = ay - prevAy; val dz = az - prevAz
            val jerk = sqrt(dx * dx + dy * dy + dz * dz) * HZ
            prevAx = ax; prevAy = ay; prevAz = az
            // channel c at time i -> out[c*t + i], divided by SCALE[c]
            out[0 * t + i] = (ax / SCALE[0]).toFloat()
            out[1 * t + i] = (ay / SCALE[1]).toFloat()
            out[2 * t + i] = (az / SCALE[2]).toFloat()
            out[3 * t + i] = (gx / SCALE[3]).toFloat()
            out[4 * t + i] = (gy / SCALE[4]).toFloat()
            out[5 * t + i] = (gz / SCALE[5]).toFloat()
            out[6 * t + i] = (an / SCALE[6]).toFloat()
            out[7 * t + i] = (gn / SCALE[7]).toFloat()
            out[8 * t + i] = (jerk / SCALE[8]).toFloat()
        }
        return out
    }

    /** Centred n-sample moving mean of each column, edges held (features.box). */
    private fun box(x: Array<DoubleArray>, n: Int): Array<DoubleArray> {
        val t = x.size; val k = x[0].size
        val lo = n / 2; val hi = n - 1 - lo
        return Array(t) { i ->
            DoubleArray(k) { c ->
                var s = 0.0
                for (j in i - lo..i + hi) s += x[j.coerceIn(0, t - 1)][c]
                s / n
            }
        }
    }

    /**
     * Version 2: channels that do not depend on the heading of the horizontal axes
     * (|a_h|, a_up, yaw, |w_h|, vib_a, vib_up, vib_g, jerk, turn_v, turning), the
     * leveled-frame port of features.window_features_v2. Inputs are rounded to
     * float32 first, as numpy does.
     */
    fun windowFeaturesV2(acc: Array<DoubleArray>, gyro: Array<DoubleArray>): FloatArray {
        val t = acc.size
        val a = Array(t) { i -> DoubleArray(3) { acc[i][it].toFloat().toDouble() } }
        val g = Array(t) { i -> DoubleArray(3) { gyro[i][it].toFloat().toDouble() } }
        val aLp = box(a, LP_N); val wLp = box(g, LP_N)
        var zMean = 0.0
        for (i in 0 until t) zMean += a[i][2]
        zMean /= t
        val raw = Array(t) { i ->                              // vib_a, vib_up, vib_g, jerk (before the 1 s mean)
            val dx = a[i][0] - aLp[i][0]; val dy = a[i][1] - aLp[i][1]; val dz = a[i][2] - aLp[i][2]
            val ex = g[i][0] - wLp[i][0]; val ey = g[i][1] - wLp[i][1]; val ez = g[i][2] - wLp[i][2]
            val p = if (i == 0) a[0] else a[i - 1]
            val jx = a[i][0] - p[0]; val jy = a[i][1] - p[1]; val jz = a[i][2] - p[2]
            doubleArrayOf(sqrt(dx * dx + dy * dy + dz * dz), abs(dz), sqrt(ex * ex + ey * ey + ez * ez),
                          sqrt(jx * jx + jy * jy + jz * jz) * HZ)
        }
        val vib = box(raw, VIB_N)
        val out = FloatArray(C2 * t)
        for (i in 0 until t) {
            val ah = sqrt(aLp[i][0] * aLp[i][0] + aLp[i][1] * aLp[i][1])
            val yaw = wLp[i][2]
            val turning = abs(yaw) > TURN_MIN
            val turnV = if (turning) (ah / max(abs(yaw), TURN_MIN)).coerceIn(0.0, TURN_MAX) else 0.0
            val ch = doubleArrayOf(ah, aLp[i][2] - zMean, yaw, sqrt(wLp[i][0] * wLp[i][0] + wLp[i][1] * wLp[i][1]),
                                   vib[i][0], vib[i][1], vib[i][2], vib[i][3], turnV, if (turning) 1.0 else 0.0)
            for (c in 0 until C2) out[c * t + i] = (ch[c] / SCALE2[c]).toFloat()
        }
        return out
    }

    /**
     * Affine mean + variance recalibration: v = (mu - a)/b, sigma = clip(sigma/b*s).
     * @param mu    raw net displacement over 1 s (== speed in m/s at 1 Hz steps)
     * @param sigma raw net sigma = exp(0.5*logvar)
     * @param a,b,s the checkpoint's calibration; identity is (0,1,1)
     * @return [v, sigma]
     */
    fun applyCalib(mu: Double, sigma: Double, a: Double, b: Double, s: Double): DoubleArray {
        val v = (mu - a) / b
        val sig = (sigma / b * s).coerceAtLeast(1e-6)
        return doubleArrayOf(v, sig)
    }

}
