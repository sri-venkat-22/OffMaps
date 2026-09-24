package com.offmaps.nav

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.math.tanh

/**
 * The learned fusion head on the phone: port of py/model/fusion_head.py
 * (context / step_features / imu_stats / FusionHead / HeadRunner), checked against
 * it by py/tests/test_kotlin_ports.py. While dead-reckoning it runs once per second
 * and returns the speed measurement AND its sigma that FusionEngine fuses with
 * updateSpeed -- replacing the fixed "hold Doppler 10 s, then SpeedNet * 0.25" rule.
 *
 * A 32-unit GRU is ~5k weights, so it runs as plain Kotlin (no ONNX state plumbing);
 * weights come from assets/fusion_head.json (py/model/fusion_head.export_json) via
 * [FusionHeadAsset]. Pure math here, no Android imports, so the host can compile it.
 */
class FusionHead(private val members: List<Member>) {
    /** One GRU member; arrays row-major exactly as torch stores them. */
    class Member(
        val hidden: Int,
        val ctxW: Array<DoubleArray>, val ctxB: DoubleArray,     // (H, N_CTX), (H)
        val wIh: Array<DoubleArray>, val wHh: Array<DoubleArray>, // (3H, N_STEP+N_CTX), (3H, H)
        val bIh: DoubleArray, val bHh: DoubleArray,               // (3H), (3H)
        val outW: Array<DoubleArray>, val outB: DoubleArray,      // (2, H), (2)
    )

    /** Outage state: context + one hidden vector per member. */
    class State(val ctx: DoubleArray, val h: Array<DoubleArray>, val k: Double, val c: Double)

    fun start(v0: Double, k: Double, c: Double, pre: List<DoubleArray>): State {
        val ctx = context(v0, k, c, pre)
        val h = Array(members.size) { m ->
            val mb = members[m]
            DoubleArray(mb.hidden) { i -> tanh(dot(mb.ctxW[i], ctx) + mb.ctxB[i]) }
        }
        return State(ctx, h, k, c)
    }

    /** One second of dead-reckoning -> [v, sigma]. acc/gyro: that second's leveled 10 Hz samples. */
    fun step(st: State, vnn: Double, sig: Double, tau: Double, acc: Array<DoubleArray>, gyro: Array<DoubleArray>): DoubleArray {
        val x = stepFeatures(vnn, sig, tau, imuStats(acc, gyro), st.k, st.c)
        val z = x + st.ctx
        val vs = DoubleArray(members.size); val vars = DoubleArray(members.size)
        for (m in members.indices) {
            val mb = members[m]; val H = mb.hidden; val h = st.h[m]
            val hn = DoubleArray(H)
            for (i in 0 until H) {
                val r = sigm(dot(mb.wIh[i], z) + mb.bIh[i] + dot(mb.wHh[i], h) + mb.bHh[i])
                val u = sigm(dot(mb.wIh[H + i], z) + mb.bIh[H + i] + dot(mb.wHh[H + i], h) + mb.bHh[H + i])
                val n = tanh(dot(mb.wIh[2 * H + i], z) + mb.bIh[2 * H + i] + r * (dot(mb.wHh[2 * H + i], h) + mb.bHh[2 * H + i]))
                hn[i] = (1 - u) * n + u * h[i]
            }
            st.h[m] = hn
            val o0 = dot(mb.outW[0], hn) + mb.outB[0]
            val o1 = (dot(mb.outW[1], hn) + mb.outB[1]).coerceIn(-6.0, 6.0)
            vs[m] = softplus(5.0 * o0); vars[m] = exp(o1)
        }
        val v = vs.average()
        var spread = 0.0
        for (x2 in vs) spread += (x2 - v) * (x2 - v)
        return doubleArrayOf(v, sqrt(vars.average() + spread / vs.size))
    }

    companion object {
        const val N_CTX = 9
        const val N_STEP = 12
        const val PRE_S = 120
        const val K_MIN = 1.0 / 3.0
        const val K_MAX = 3.0

        private fun dot(w: DoubleArray, x: DoubleArray): Double { var s = 0.0; for (i in w.indices) s += w[i] * x[i]; return s }
        private fun sigm(x: Double) = 1.0 / (1.0 + exp(-x))
        private fun softplus(x: Double) = if (x > 20) x else ln(1.0 + exp(x))

        /** pre: (doppler, nn_raw) per trusted fix before the outage, oldest first. */
        fun context(v0: Double, k: Double, c: Double, pre: List<DoubleArray>): DoubleArray {
            val calOk = k in K_MIN..K_MAX
            val kk = if (calOk) k else 1.0; val cc = if (calOk) c else 0.0
            val p = if (pre.size > PRE_S) pre.subList(pre.size - PRE_S, pre.size) else pre
            var vbar = v0; var stop = 0.0; var rm = 0.0; var rs = 3.0
            if (p.isNotEmpty()) {
                var sd = 0.0; var st = 0; var sr = 0.0
                val res = DoubleArray(p.size)
                for ((i, r) in p.withIndex()) {
                    sd += r[0]; if (r[0] < 0.5) st++
                    res[i] = r[0] - max(kk * r[1] + cc, 0.0); sr += res[i]
                }
                vbar = sd / p.size; stop = st.toDouble() / p.size; rm = sr / p.size
                var v2 = 0.0; for (x in res) v2 += (x - rm) * (x - rm); rs = sqrt(v2 / p.size)
            }
            return doubleArrayOf(v0 / 10, vbar / 10, stop, rm / 5, rs / 5, if (calOk) 1.0 else 0.0, kk, cc / 5,
                                 min(p.size, PRE_S).toDouble() / PRE_S)
        }

        fun stepFeatures(vnn: Double, sig: Double, tau: Double, s: DoubleArray, k: Double, c: Double): DoubleArray {
            val calOk = k in K_MIN..K_MAX
            val vcal = if (calOk) max(k * vnn + c, 0.0) else vnn
            return doubleArrayOf(vnn / 10, vcal / 10, ln(max(sig, 1e-3)), min(tau, 180.0) / 60, min(tau, 10.0) / 10,
                                 s[0], s[1] / 3, s[2] / 3, s[3], s[4] * 5, s[5] * 5, s[6] / 10)
        }

        /** Rotation-invariant stats of one second of leveled 10 Hz IMU (z = up). */
        fun imuStats(acc: Array<DoubleArray>, gyro: Array<DoubleArray>): DoubleArray {
            val n = acc.size
            val an = DoubleArray(n) { sqrt(acc[it][0] * acc[it][0] + acc[it][1] * acc[it][1] + acc[it][2] * acc[it][2]) }
            val anM = an.average()
            var anV = 0.0; for (x in an) anV += (x - anM) * (x - anM)
            var ah = 0.0; var up = 0.0; var gn = 0.0; var yaw = 0.0
            for (i in 0 until n) {
                ah += sqrt(acc[i][0] * acc[i][0] + acc[i][1] * acc[i][1]); up += acc[i][2]
                gn += sqrt(gyro[i][0] * gyro[i][0] + gyro[i][1] * gyro[i][1] + gyro[i][2] * gyro[i][2]); yaw += abs(gyro[i][2])
            }
            val upM = up / n
            var upV = 0.0; for (i in 0 until n) upV += (acc[i][2] - upM) * (acc[i][2] - upM)
            var jerk = 0.0
            for (i in 1 until n) {
                val dx = acc[i][0] - acc[i - 1][0]; val dy = acc[i][1] - acc[i - 1][1]; val dz = acc[i][2] - acc[i - 1][2]
                jerk += sqrt(dx * dx + dy * dy + dz * dz)
            }
            jerk = if (n > 1) jerk / (n - 1) * 10.0 else 0.0
            return doubleArrayOf(sqrt(anV / n), ah / n, upM - 9.81, sqrt(upV / n), gn / n, yaw / n, jerk)
        }
    }
}
