// Host harness for the app's pure-Kotlin math (no Android deps): compiled together
// with nav/Level.kt + nav/Features.kt by py/tests/test_kotlin_ports.py and driven
// from Python so the on-device ports are checked against their Python references.
// stdin: lines "ax ay az gx gy gz remount(0|1)" at 10 Hz. stdout: per line the
// leveled "ax ay az gx gy gz", then one final line with windowFeatures of the last
// Features.WIN leveled samples (C*WIN floats, channel-major).
// With arg "decim": stdin sensor timestamps (ns) -> number kept by Decimator(100 ms).
// With arg "head <weights.txt>": FusionHead.kt on an outage (see headMode).
// With arg "aids": HeadingAids.kt / YawAlign lines (see aidsMode).
package com.offmaps.nav

import java.io.File

fun main(args: Array<String>) {
    if (args.isNotEmpty() && args[0] == "head") { headMode(args[1]); return }
    if (args.isNotEmpty() && args[0] == "aids") { aidsMode(); return }
    if (args.isNotEmpty() && args[0] == "decim") {           // stdin: sensor timestamps (ns); stdout: kept count
        val d = Decimator(100_000_000L)
        println(generateSequence(::readLine).filter { it.isNotBlank() }.count { d.accept(it.trim().toLong()) })
        return
    }
    val lv = Level()
    val accOut = DoubleArray(3); val gyrOut = DoubleArray(3)
    val accW = ArrayList<DoubleArray>(); val gyrW = ArrayList<DoubleArray>()
    val sb = StringBuilder()
    generateSequence(::readLine).filter { it.isNotBlank() }.forEach { line ->
        val v = line.trim().split(" ").map { it.toDouble() }
        if (v[6] != 0.0) lv.onRemount()
        lv.step(doubleArrayOf(v[0], v[1], v[2]), doubleArrayOf(v[3], v[4], v[5]), accOut, gyrOut)
        sb.append((accOut.toList() + gyrOut.toList()).joinToString(" ") { it.toString() }).append('\n')
        accW.add(accOut.copyOf()); gyrW.add(gyrOut.copyOf())
    }
    val n = accW.size
    val f = Features.windowFeatures(accW.subList(n - Features.WIN, n).toTypedArray(),
                                    gyrW.subList(n - Features.WIN, n).toTypedArray())
    sb.append(f.joinToString(" ") { it.toString() }).append('\n')
    print(sb)
}

/** weights.txt: "members H" then per member 8 tensors "rows cols v..." in the order
 *  ctxW ctxB wIh wHh bIh bHh outW outB (a vector has cols = 1).
 *  stdin: "v0 k c npre", npre lines "dop vn", then per second
 *  "vnn sig tau" + 10x3 acc + 10x3 gyro. stdout per second: "v sigma". */
fun headMode(path: String) {
    val tok = File(path).readText().trim().split(Regex("\\s+")).iterator()
    val nm = tok.next().toInt(); val h = tok.next().toInt()
    fun tensor(): Array<DoubleArray> {
        val r = tok.next().toInt(); val c = tok.next().toInt()
        return Array(r) { DoubleArray(c) { tok.next().toDouble() } }
    }
    fun vec(): DoubleArray = tensor().map { it[0] }.toDoubleArray()
    val members = (0 until nm).map {
        FusionHead.Member(h, tensor(), vec(), tensor(), tensor(), vec(), vec(), tensor(), vec())
    }
    val head = FusionHead(members)
    val lines = generateSequence(::readLine).filter { it.isNotBlank() }.iterator()
    val hdr = lines.next().trim().split(" ").map { it.toDouble() }
    val pre = (0 until hdr[3].toInt()).map { lines.next().trim().split(" ").map { x -> x.toDouble() }.toDoubleArray() }
    val st = head.start(hdr[0], hdr[1], hdr[2], pre)
    val sb = StringBuilder()
    while (lines.hasNext()) {
        val v = lines.next().trim().split(" ").map { it.toDouble() }
        val acc = Array(10) { i -> doubleArrayOf(v[3 + 3 * i], v[4 + 3 * i], v[5 + 3 * i]) }
        val gyr = Array(10) { i -> doubleArrayOf(v[33 + 3 * i], v[34 + 3 * i], v[35 + 3 * i]) }
        val r = head.step(st, v[0], v[1], v[2], acc, gyr)
        sb.append("${r[0]} ${r[1]}\n")
    }
    print(sb)
}

/** stdin lines: "yaw mode gx gy gz ufx ufy ufz usx usy usz fx fy fz v" -> yaw rate
 *  "mag mx my mz ux uy uz fx fy fz decl" -> heading (or NaN); "fwd ux uy uz" -> default forward;
 *  "acc a1 a2" -> (nothing); "fix t v" -> "theta valid" of YawAlign. */
fun aidsMode() {
    val ya = YawAlign()
    val sb = StringBuilder()
    generateSequence(::readLine).filter { it.isNotBlank() }.forEach { line ->
        val p = line.trim().split(" ")
        val d = p.drop(1).mapNotNull { it.toDoubleOrNull() }
        when (p[0]) {
            "yaw" -> {
                val x = p.drop(2).map { it.toDouble() }
                val fwd = doubleArrayOf(x[9], x[10], x[11])
                sb.append(HeadingAids.yawRate(p[1], doubleArrayOf(x[0], x[1], x[2]), doubleArrayOf(x[3], x[4], x[5]),
                    doubleArrayOf(x[6], x[7], x[8]), fwd, x[12])).append('\n')
            }
            "mag" -> sb.append(HeadingAids.magHeading(doubleArrayOf(d[0], d[1], d[2]), doubleArrayOf(d[3], d[4], d[5]),
                    doubleArrayOf(d[6], d[7], d[8]), d[9]) ?: Double.NaN).append('\n')
            "fwd" -> sb.append(HeadingAids.defaultForward(doubleArrayOf(d[0], d[1], d[2])).joinToString(" ")).append('\n')
            "acc" -> ya.addAcc(d[0], d[1])
            "fix" -> { ya.onFix(d[0], d[1]); sb.append("${ya.theta} ${if (ya.valid) 1 else 0}\n") }
        }
    }
    print(sb)
}
