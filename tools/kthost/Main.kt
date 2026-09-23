// Host harness for the app's pure-Kotlin math (no Android deps): compiled together
// with nav/Level.kt + nav/Features.kt by py/tests/test_kotlin_ports.py and driven
// from Python so the on-device ports are checked against their Python references.
// stdin: lines "ax ay az gx gy gz remount(0|1)" at 10 Hz. stdout: per line the
// leveled "ax ay az gx gy gz", then one final line with windowFeatures of the last
// Features.WIN leveled samples (C*WIN floats, channel-major).
// With arg "decim": stdin sensor timestamps (ns) -> number kept by Decimator(100 ms).
package com.offmaps.nav

fun main(args: Array<String>) {
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
