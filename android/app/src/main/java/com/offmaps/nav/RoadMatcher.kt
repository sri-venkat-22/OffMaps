package com.offmaps.nav

import kotlin.math.hypot

/**
 * Feeds libidr's greedy matcher (mm_match) a moving window of the city's roads.
 * mm_match scans every segment it holds, so instead of loading all of Hyderabad
 * it keeps only the ways within [WINDOW_M] of where the window was last centred,
 * projected into the live ENU frame, and rebuilds once the estimate has moved
 * [RECENTER_M] -- leaving >= WINDOW_M - RECENTER_M of road around the car.
 *
 * Fusion thread only (like every other native handle in FusionEngine).
 */
class RoadMatcher(private val roads: RoadNetwork, private val enu: Enu) : AutoCloseable {
    private var mm: MapMatcher? = null
    private var cE = 0.0; private var cN = 0.0
    var windowWays = 0; private set

    fun match(e: Double, n: Double, psi: Double): MatchResult {
        if (mm == null || hypot(e - cE, n - cN) > RECENTER_M) rebuild(e, n)
        return mm!!.match(e, n, psi)
    }

    private fun rebuild(e: Double, n: Double) {
        mm?.close()
        val ll = enu.toLatLon(e, n)
        val m = MapMatcher()
        val ids = roads.waysNear(ll[0], ll[1], WINDOW_M)
        for (w in ids) {
            val k = roads.size(w)
            val es = DoubleArray(k); val ns = DoubleArray(k)
            for (i in 0 until k) {
                val p = enu.toEn(roads.lat(w, i), roads.lon(w, i)); es[i] = p[0]; ns[i] = p[1]
            }
            m.addWay(es, ns, roads.tunnel(w), roads.oneway(w))
        }
        mm = m; cE = e; cN = n; windowWays = ids.size
    }

    override fun close() { mm?.close(); mm = null }

    companion object {
        const val WINDOW_M = 1000.0
        const val RECENTER_M = 400.0
    }
}
