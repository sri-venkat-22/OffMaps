package com.offmaps.nav

/**
 * Keeps one sample per [periodNs] (the 10 Hz fusion grid) from a faster sensor stream.
 *
 * The first version kept `last = Long.MIN_VALUE` and tested `t - last < period`.
 * For any positive timestamp that subtraction overflows to a negative Long, so
 * EVERY sample was dropped: the Phase-6 app never ran predict or the NN on a phone.
 * Found on the emulator (position frozen while GNSS speed read 10 m/s). Checked on
 * the host by py/tests/test_kotlin_ports.py.
 */
class Decimator(private val periodNs: Long) {
    private var last = 0L
    private var have = false

    fun reset() { have = false }

    /** @return true if this sample starts a new grid slot (and is kept). */
    fun accept(tNs: Long): Boolean {
        if (have && tNs - last < periodNs) return false
        have = true; last = tNs
        return true
    }
}
