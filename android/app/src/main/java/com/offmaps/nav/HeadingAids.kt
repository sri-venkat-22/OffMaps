package com.offmaps.nav

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Heading aids: port of py/heading_aids.py (checked by py/tests/test_kotlin_ports.py).
 *  - [yawRate]: yaw rate about TRUE vertical. "slow" (30 s gravity; exact for a car) or
 *    "coord" (fast gravity tilted back by the coordinated-turn angle atan(v*psidot/g);
 *    needed on a two-wheeler, which leans). The old app used "fast" (0.5 s gravity),
 *    which under-reads every turn by cos(phi).
 *  - [magHeading]: tilt-compensated magnetometer heading of the vehicle's forward axis,
 *    used once to seed the filter when the first GNSS fix has no course (parked).
 *  - [YawAlign]: the vehicle's forward axis in the leveled frame from Doppler dv/dt.
 */
object HeadingAids {
    const val YAW_SIGN = -1.0
    const val G = 9.81
    val MAG_SIGMA = Math.toRadians(20.0)
    const val UNKNOWN_SIGMA = PI

    fun dot(a: DoubleArray, b: DoubleArray) = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    fun cross(a: DoubleArray, b: DoubleArray) =
        doubleArrayOf(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    fun unit(a: DoubleArray): DoubleArray {
        val n = sqrt(dot(a, a))
        return if (n > 1e-9) doubleArrayOf(a[0] / n, a[1] / n, a[2] / n) else doubleArrayOf(0.0, 0.0, 0.0)
    }
    fun horiz(a: DoubleArray, u: DoubleArray): DoubleArray {
        val d = dot(a, u)
        return unit(doubleArrayOf(a[0] - d * u[0], a[1] - d * u[1], a[2] - d * u[2]))
    }
    fun wrap(a: Double): Double { val x = (a + PI) % (2 * PI); return (if (x < 0) x + 2 * PI else x) - PI }

    /** Mount guess until YawAlign converges: flat phone -> top (+y) forward; upright -> back (-z) forward. */
    fun defaultForward(up: DoubleArray): DoubleArray {
        val u = unit(up)
        return if (abs(u[2]) >= 0.7) horiz(doubleArrayOf(0.0, 1.0, 0.0), u) else horiz(doubleArrayOf(0.0, 0.0, -1.0), u)
    }

    /** Compass heading of fwd from a device-frame magnetometer reading; null if no horizontal field. */
    fun magHeading(mag: DoubleArray, up: DoubleArray, fwd: DoubleArray, declination: Double = 0.0): Double? {
        val u = unit(up)
        val north = horiz(mag, u)
        if (north[0] == 0.0 && north[1] == 0.0 && north[2] == 0.0) return null
        val east = cross(north, u)
        val f = horiz(fwd, u)
        return wrap(atan2(dot(f, east), dot(f, north)) + declination)
    }

    fun yawRate(mode: String, gyro: DoubleArray, upFast: DoubleArray, upSlow: DoubleArray,
                fwd: DoubleArray?, v: Double): Double {
        if (mode == "slow") return YAW_SIGN * dot(gyro, unit(upSlow))
        val z = unit(upFast)
        var r = YAW_SIGN * dot(gyro, z)
        if (mode == "fast" || fwd == null) return r
        val x = horiz(fwd, z)
        val y = cross(z, x)
        repeat(3) {
            val phi = atan(v * r / G)
            val c = cos(phi); val s = sin(phi)
            r = YAW_SIGN * dot(gyro, doubleArrayOf(c * z[0] + s * y[0], c * z[1] + s * y[1], c * z[2] + s * y[2]))
        }
        return r
    }
}

/** GNSS-aided forward axis in the leveled frame (heading_aids.YawAlign). */
class YawAlign {
    private var s11 = 0.0; private var s12 = 0.0; private var s22 = 0.0
    private var b1 = 0.0; private var b2 = 0.0; private var exc = 0.0
    private var sum1 = 0.0; private var sum2 = 0.0; private var cnt = 0
    private var tPrev = Double.NaN; private var vPrev = 0.0
    var theta = 0.0; private set
    var valid = false; private set

    fun reset() {
        s11 = 0.0; s12 = 0.0; s22 = 0.0; b1 = 0.0; b2 = 0.0; exc = 0.0
        sum1 = 0.0; sum2 = 0.0; cnt = 0; tPrev = Double.NaN; vPrev = 0.0; theta = 0.0; valid = false
    }

    fun addAcc(a1: Double, a2: Double) { sum1 += a1; sum2 += a2; cnt++ }

    fun onFix(t: Double, v: Double): Boolean {
        var upd = false
        if (!tPrev.isNaN() && cnt > 0 && t - tPrev in 0.5..2.5) {
            val dv = (v - vPrev) / (t - tPrev)
            val a1 = sum1 / cnt; val a2 = sum2 / cnt
            s11 = FORGET * s11 + a1 * a1; s12 = FORGET * s12 + a1 * a2; s22 = FORGET * s22 + a2 * a2
            b1 = FORGET * b1 + a1 * dv; b2 = FORGET * b2 + a2 * dv; exc = FORGET * exc + dv * dv
            val det = s11 * s22 - s12 * s12
            if (det > 1e-9) {
                val c1 = (s22 * b1 - s12 * b2) / det; val c2 = (s11 * b2 - s12 * b1) / det
                val gain = hypot(c1, c2)
                theta = atan2(c2, c1)
                valid = exc >= MIN_EXC && gain in GAIN_MIN..GAIN_MAX
                upd = true
            }
        }
        tPrev = t; vPrev = v; sum1 = 0.0; sum2 = 0.0; cnt = 0
        return upd
    }

    /** Forward axis in device coords from the leveled basis rows (h1, h2, up). */
    fun forward(basis: Array<DoubleArray>): DoubleArray {
        val c = cos(theta); val s = sin(theta)
        return DoubleArray(3) { c * basis[0][it] + s * basis[1][it] }
    }

    companion object {
        const val FORGET = 0.995
        const val MIN_EXC = 4.0
        const val GAIN_MIN = 0.2
        const val GAIN_MAX = 5.0
    }
}
