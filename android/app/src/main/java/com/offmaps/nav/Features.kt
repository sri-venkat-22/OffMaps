package com.offmaps.nav

import kotlin.math.sqrt

/**
 * The IMU-window -> feature-tensor contract, ported verbatim from
 * py/model/features.py. Training and Python inference share that one
 * implementation; this is the on-device third copy, and it must agree to the
 * bit or the shipped net silently rots (same warning as the Python docstring).
 *
 * Input:  WIN=20 samples of acc/gyro at 10 Hz, vehicle frame.
 * Output: (9, 20) float row-major = [ax,ay,az, gx,gy,gz, |a|,|g|,|jerk|] / SCALE,
 *         flattened channel-major (index = channel*WIN + t) -- the ONNX layout.
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

    /**
     * Build the (9,20) feature tensor from the most recent window.
     * @param acc  WIN rows of [ax,ay,az]
     * @param gyro WIN rows of [gx,gy,gz]
     * @return FloatArray(C*WIN), channel-major, ready as the ONNX "imu" input.
     */
    fun windowFeatures(acc: Array<DoubleArray>, gyro: Array<DoubleArray>): FloatArray {
        val t = WIN
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
            // channel c at time i -> out[c*WIN + i], divided by SCALE[c]
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
