package com.offmaps.nav

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/**
 * Feeds libidr's map matchers a moving window of the city's roads.
 * mm_match scans every segment it holds, so instead of loading all of Hyderabad
 * it keeps only the ways within [WINDOW_M] of where the window was last centred,
 * projected into the live ENU frame, and rebuilds once the estimate has moved
 * [RECENTER_M] -- leaving >= WINDOW_M - RECENTER_M of road around the car.
 *
 * Two ways to pick the road (py/road_window.py is the host twin):
 *  - [match]: greedy nearest road (Phase 5, core mm_match), every 10 Hz step;
 *  - [decode] + [project]: LIVE Viterbi (Phase 7b's mm_match_seq run online). Once a
 *    second while dead-reckoning the last [LAG] one-second positions are decoded
 *    (HMM: cross-track + bearing emission, road adjacency + distance-consistency
 *    transition) and the way chosen for the newest step is the road the next second
 *    snaps to. Adjacency comes from the map: ways that share a vertex are connected
 *    (roads.bin pieces share endpoints; ways meeting at a junction share that node).
 *
 * [withEdges]: build the junction adjacency (Viterbi mode only; the core stores a
 * W x W matrix, wasted on the greedy matcher).
 *
 * Fusion thread only (like every other native handle in FusionEngine).
 */
class RoadMatcher(private val roads: RoadNetwork, private val enu: Enu,
                  private val withEdges: Boolean = false) : AutoCloseable {
    private var mm: MapMatcher? = null
    private var cE = 0.0; private var cN = 0.0
    private var geoE: Array<DoubleArray> = emptyArray()
    private var geoN: Array<DoubleArray> = emptyArray()
    var windowWays = 0; private set

    fun match(e: Double, n: Double, psi: Double): MatchResult {
        if (mm == null || hypot(e - cE, n - cN) > RECENTER_M) rebuild(e, n)
        return mm!!.match(e, n, psi)
    }

    /** hist: (e, n, psi) at 1 Hz, oldest first -> window way chosen for the newest step
     *  (-1 = off-road) and its corridor flag. */
    fun decode(hist: List<DoubleArray>): Pair<Int, Boolean> {
        val last = hist.last()
        if (mm == null || hypot(last[0] - cE, last[1] - cN) > RECENTER_M) rebuild(last[0], last[1])
        val r = mm!!.matchSeq(DoubleArray(hist.size) { hist[it][0] }, DoubleArray(hist.size) { hist[it][1] },
                              DoubleArray(hist.size) { hist[it][2] })
        return r.way[hist.size - 1] to r.corridor[hist.size - 1]
    }

    /** Snap (e, n) onto one window way; unmatched if gone, or its bearing disagrees (45 deg gate). */
    fun project(way: Int, e: Double, n: Double, psi: Double, corridor: Boolean): MatchResult {
        val none = MatchResult(false, 0.0, 0.0, 0.0, 0.0, false)
        if (way < 0 || way >= geoE.size) return none
        val we = geoE[way]; val wn = geoN[way]
        var best = Double.MAX_VALUE; var fx = 0.0; var fy = 0.0; var brg = 0.0
        for (i in 0 until we.size - 1) {
            val dx = we[i + 1] - we[i]; val dy = wn[i + 1] - wn[i]
            val t = (((e - we[i]) * dx + (n - wn[i]) * dy) / (dx * dx + dy * dy + 1e-12)).coerceIn(0.0, 1.0)
            val px = we[i] + t * dx; val py = wn[i] + t * dy
            val d2 = (e - px) * (e - px) + (n - py) * (n - py)
            if (d2 < best) { best = d2; fx = px; fy = py; brg = atan2(dx, dy) }
        }
        val db = abs(HeadingAids.wrap(brg - psi))
        if (db > PI / 4 && db < PI - PI / 4) return none
        val cross = (fx - e) * -cos(brg) + (fy - n) * sin(brg)
        return MatchResult(true, fx, fy, brg, cross, corridor)
    }

    private fun rebuild(e: Double, n: Double) {
        mm?.close()
        val ll = enu.toLatLon(e, n)
        val m = MapMatcher()
        val ids = roads.waysNear(ll[0], ll[1], WINDOW_M)
        val nodes = HashMap<Long, MutableList<Int>>()
        geoE = Array(ids.size) { DoubleArray(0) }; geoN = Array(ids.size) { DoubleArray(0) }
        for ((j, w) in ids.withIndex()) {
            val k = roads.size(w)
            val es = DoubleArray(k); val ns = DoubleArray(k)
            for (i in 0 until k) {
                val p = enu.toEn(roads.lat(w, i), roads.lon(w, i)); es[i] = p[0]; ns[i] = p[1]
                val key = (roads.lat7(w, i).toLong() shl 32) or (roads.lon7(w, i).toLong() and 0xffffffffL)
                nodes.getOrPut(key) { ArrayList(2) }.add(j)
            }
            m.addWay(es, ns, roads.tunnel(w), roads.oneway(w))
            geoE[j] = es; geoN[j] = ns
        }
        // junction adjacency for the Viterbi decoder only (the core keeps a W x W matrix)
        if (withEdges) for (s in nodes.values) if (s.size > 1) {
            val u = s.distinct()
            for (a in u.indices) for (b in a + 1 until u.size) m.addEdge(u[a], u[b])
        }
        mm = m; cE = e; cN = n; windowWays = ids.size
    }

    override fun close() { mm?.close(); mm = null }

    companion object {
        const val WINDOW_M = 1000.0
        const val RECENTER_M = 400.0
        const val LAG = 20                     // seconds of history the live decoder re-reads
    }
}
