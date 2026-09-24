package com.offmaps.nav

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * Levels device-frame IMU into the frame SpeedNet was trained in: z = up (+9.81 at
 * rest), x/y horizontal. Port of py/model/mount.py (level_stream), checked against
 * it by app/src/test/.../LevelTest.kt. The real-data loader trained nn_real on a
 * vehicle frame; a phone in a dash mount has y up, which moved nn_real's bias by
 * 5 m/s on val drives. The net is insensitive to the horizontal heading, so up
 * alone (a slow 30 s gravity EMA) is enough.
 */
class Level(hz: Double = Features.HZ, tauS: Double = TAU_S) {
    private val a = 1.0 - exp(-1.0 / (tauS * hz))
    private val am = 1.0 - exp(-1.0 / (TAU_MED_S * hz))
    private val g = DoubleArray(3)              // slow gravity: defines up
    private val gm = DoubleArray(3)             // medium gravity: where a re-mount snaps to
    private var have = false

    fun reset() { have = false }

    /** The core's re-mount detector fired (aln changed): level to the new mount now. */
    fun onRemount() { if (have) for (i in 0..2) g[i] = gm[i] }

    /** The slow (30 s) gravity: device-frame up, as the leveling uses it. */
    fun up(): DoubleArray = g.copyOf()

    /** Advance one sample; writes leveled acc/gyro (gyro z = compass yaw rate). */
    fun step(acc: DoubleArray, gyro: DoubleArray, accOut: DoubleArray, gyroOut: DoubleArray) {
        if (!have) { for (i in 0..2) { g[i] = acc[i]; gm[i] = acc[i] }; have = true }
        else for (i in 0..2) { g[i] += a * (acc[i] - g[i]); gm[i] += am * (acc[i] - gm[i]) }
        val r = basis(g)
        for (row in 0..2) {
            accOut[row] = r[row][0] * acc[0] + r[row][1] * acc[1] + r[row][2] * acc[2]
            gyroOut[row] = r[row][0] * gyro[0] + r[row][1] * gyro[1] + r[row][2] * gyro[2]
        }
        gyroOut[2] *= YAW_SIGN
    }

    companion object {
        const val TAU_S = 30.0
        const val TAU_MED_S = 5.0               // == core/align.cpp REMOUNT_TAU_MED
        const val YAW_SIGN = -1.0

        /** Rows (h1, h2, u): right-handed, u = unit up in device coordinates. */
        fun basis(up: DoubleArray): Array<DoubleArray> {
            val n = sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2]).coerceAtLeast(1e-6)
            val u = doubleArrayOf(up[0] / n, up[1] / n, up[2] / n)
            val ref = if (abs(u[0]) < 0.9) doubleArrayOf(1.0, 0.0, 0.0) else doubleArrayOf(0.0, 1.0, 0.0)
            val d = ref[0] * u[0] + ref[1] * u[1] + ref[2] * u[2]
            val h1 = doubleArrayOf(ref[0] - u[0] * d, ref[1] - u[1] * d, ref[2] - u[2] * d)
            val hn = sqrt(h1[0] * h1[0] + h1[1] * h1[1] + h1[2] * h1[2])
            for (i in 0..2) h1[i] /= hn
            val h2 = doubleArrayOf(u[1] * h1[2] - u[2] * h1[1], u[2] * h1[0] - u[0] * h1[2], u[0] * h1[1] - u[1] * h1[0])
            return arrayOf(h1, h2, u)
        }
    }
}
