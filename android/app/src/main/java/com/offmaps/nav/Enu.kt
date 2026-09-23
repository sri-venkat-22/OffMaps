package com.offmaps.nav

import kotlin.math.PI
import kotlin.math.cos

/**
 * Local ENU projection, byte-for-byte the harness convention
 * (py/data/io_vnbd.py::_lla_to_enu): equirectangular about the FIRST fix.
 *   e = (lon - lon0) * cos(lat0) * k ,  n = (lat - lat0) * k ,  k = pi/180 * R.
 * The first GNSS fix defines the origin, so the ESKF starts at (0, 0) -- the
 * same seeding core_bridge.py uses (drive.e[i0], drive.n[i0]).
 */
class Enu {
    companion object { const val R_EARTH = 6_371_000.0 }
    private val k = PI / 180.0 * R_EARTH
    private var lat0 = Double.NaN
    private var lon0 = 0.0
    private var cosLat0 = 1.0
    val hasOrigin: Boolean get() = !lat0.isNaN()

    /** Set the origin from the first fix. Idempotent: only the first call sticks. */
    fun setOrigin(lat: Double, lon: Double) {
        if (hasOrigin) return
        lat0 = lat; lon0 = lon; cosLat0 = cos(lat0 * PI / 180.0)
    }

    /** (lat, lon) degrees -> (e, n) metres in the local frame. */
    fun toEn(lat: Double, lon: Double): DoubleArray =
        doubleArrayOf((lon - lon0) * cosLat0 * k, (lat - lat0) * k)

    /** Exact inverse of [toEn]: (e, n) metres -> (lat, lon) degrees, for drawing on the map. */
    fun toLatLon(e: Double, n: Double): DoubleArray =
        doubleArrayOf(lat0 + n / k, lon0 + e / (cosLat0 * k))
}
