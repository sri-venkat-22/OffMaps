package com.offmaps.nav

import java.util.PriorityQueue
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/**
 * Online HMM road matching (Newson & Krumm 2009 as a causal forward filter): the port of
 * py/road_hmm.py, checked against it by py/tests/test_kotlin_ports.py (tools/kthost).
 * Pure Kotlin, no Android deps. Phase 9 measured it on real IO-VNBD roads: train drives
 * 13.6 -> 11.7 % mean median drift, where the greedy RoadMatcher made outages worse
 * (py/phase9_map_eval.py, out/phase9/).
 *
 * States are directed road segments (consecutive vertices of a way) near the filter's
 * position; once a second:
 *   emission    N(distance; 0, sigma) * N(heading error; 0, 25 deg), sigma = max(4 m, filter sigma)
 *   transition  exp(-|route distance - distance driven| / beta), route distance by bounded
 *               Dijkstra on the directed (one-way aware) road graph
 * [update] returns the best segment with a confidence per ROAD (posterior of the candidates
 * on the same line), since OSM splits one road into many short segments.
 *
 * The phone cannot hold a city's graph, so a [WINDOW_M] window of ways is rebuilt whenever
 * the car has moved [RECENTER_M]. Candidates carry stable keys (way, vertex) and graph
 * nodes are keyed by their exact 1e-7 degree coordinates, so the HMM state survives a
 * rebuild and the result equals the Python reference on the whole network: candidates
 * lie within RADIUS_MAX of the car and routes within (distance driven + 60 m) of them,
 * both well inside the window.
 */
interface RoadSource {
    fun size(w: Int): Int
    fun lat7(w: Int, i: Int): Int
    fun lon7(w: Int, i: Int): Int
    fun tunnel(w: Int): Int
    fun oneway(w: Int): Int
    fun hwCode(w: Int): Int                    // tools/osm_layers.py HW_CODE, 0 = unknown
    fun waysNear(lat: Double, lon: Double, radiusM: Double): IntArray
}

class RoadHmm(private val roads: RoadSource, private val enu: Enu,
              private val exclude: Set<Int> = setOf(HW_SERVICE)) {

    class Match(val way: Int, val vtx: Int, val footE: Double, val footN: Double, val bearing: Double,
                val dist: Double, val conf: Double, val segConf: Double, val segLen: Double,
                val endDist: Double, val halfWidth: Double, val tunnel: Boolean)

    // ---- the window graph ----
    private var nSeg = 0
    private var segKey = LongArray(0)          // (way shl 8) or vtx: ascending == Python segment id order
    private var aNode = LongArray(0); private var bNode = LongArray(0)
    private var px = DoubleArray(0); private var py = DoubleArray(0)
    private var dx = DoubleArray(0); private var dy = DoubleArray(0)
    private var len = DoubleArray(0); private var brg = DoubleArray(0)
    private var oneway = BooleanArray(0); private var tun = BooleanArray(0); private var hw = DoubleArray(0)
    private var out = HashMap<Long, MutableList<IntArray>>()   // node -> [(seg, dirn)]; next node from seg
    private var grid = HashMap<Long, IntArray>()
    private var cE = 0.0; private var cN = 0.0; private var built = false
    var windowSegments = 0; private set

    // ---- HMM state: previous candidates (by value, so a rebuild doesn't break them) ----
    private class Cands(val key: LongArray, val dirn: IntArray, val t: DoubleArray, val exitNode: LongArray,
                        val entryNode: LongArray, val len: DoubleArray, val foot: Array<DoubleArray>,
                        val dist: DoubleArray, val head: DoubleArray, val seg: IntArray)
    private var prev: Cands? = null
    private var logp = DoubleArray(0)
    private var travel = 0.0
    var last: Match? = null; private set

    fun reset() { prev = null; logp = DoubleArray(0); travel = 0.0; last = null }
    fun addTravel(d: Double) { travel += d }

    private fun rebuild(e: Double, n: Double) {
        val ll = enu.toLatLon(e, n)
        val ids = roads.waysNear(ll[0], ll[1], WINDOW_M).filter { roads.hwCode(it) !in exclude }.sorted()
        var cap = 0
        for (w in ids) cap += roads.size(w) - 1
        val sk = LongArray(cap); val an = LongArray(cap); val bn = LongArray(cap)
        val sx = DoubleArray(cap); val sy = DoubleArray(cap); val ddx = DoubleArray(cap); val ddy = DoubleArray(cap)
        val ow = BooleanArray(cap); val tn = BooleanArray(cap); val hwd = DoubleArray(cap)
        var s = 0
        for (w in ids) {
            val k = roads.size(w)
            var prevKey = 0L; var pe = 0.0; var pn = 0.0
            for (i in 0 until k) {
                val la7 = roads.lat7(w, i); val lo7 = roads.lon7(w, i)
                val p = enu.toEn(la7 / SCALE, lo7 / SCALE)
                val key = (la7.toLong() shl 32) or (lo7.toLong() and 0xffffffffL)
                if (i > 0 && (p[0] != pe || p[1] != pn)) {
                    sk[s] = (w.toLong() shl 8) or (i - 1).toLong(); an[s] = prevKey; bn[s] = key
                    sx[s] = pe; sy[s] = pn; ddx[s] = p[0] - pe; ddy[s] = p[1] - pn
                    ow[s] = roads.oneway(w) == 1; tn[s] = roads.tunnel(w) == 1
                    hwd[s] = halfWidth(roads.hwCode(w)); s++
                }
                prevKey = key; pe = p[0]; pn = p[1]
            }
        }
        nSeg = s; segKey = sk.copyOf(s); aNode = an.copyOf(s); bNode = bn.copyOf(s)
        px = sx.copyOf(s); py = sy.copyOf(s); dx = ddx.copyOf(s); dy = ddy.copyOf(s)
        oneway = ow.copyOf(s); tun = tn.copyOf(s); hw = hwd.copyOf(s)
        len = DoubleArray(s) { hypot(dx[it], dy[it]) }
        brg = DoubleArray(s) { atan2(dx[it], dy[it]) }
        out = HashMap(2 * s)
        for (i in 0 until s) {
            out.getOrPut(aNode[i]) { ArrayList(3) }.add(intArrayOf(i, 1))
            if (!oneway[i]) out.getOrPut(bNode[i]) { ArrayList(3) }.add(intArrayOf(i, -1))
        }
        val g = HashMap<Long, MutableList<Int>>()
        for (i in 0 until s) {
            val x0 = floor(min(px[i], px[i] + dx[i]) / CELL_M).toInt(); val x1 = floor(max(px[i], px[i] + dx[i]) / CELL_M).toInt()
            val y0 = floor(min(py[i], py[i] + dy[i]) / CELL_M).toInt(); val y1 = floor(max(py[i], py[i] + dy[i]) / CELL_M).toInt()
            for (cx in x0..x1) for (cy in y0..y1) g.getOrPut(cell(cx, cy)) { ArrayList(4) }.add(i)
        }
        grid = HashMap(g.size * 2); for ((kk, v) in g) grid[kk] = v.toIntArray()
        cE = e; cN = n; built = true; windowSegments = s
    }

    private fun candidates(x: Double, y: Double, psi: Double, moving: Boolean, r: Double): Cands? {
        val ids = HashSet<Int>()
        for (cx in floor((x - r) / CELL_M).toInt()..floor((x + r) / CELL_M).toInt())
            for (cy in floor((y - r) / CELL_M).toInt()..floor((y + r) / CELL_M).toInt())
                grid[cell(cx, cy)]?.let { for (i in it) ids.add(i) }
        val c = ArrayList<DoubleArray>()                     // [seg, t, footE, footN, dist]
        for (i in ids) {
            val t = (((x - px[i]) * dx[i] + (y - py[i]) * dy[i]) / (len[i] * len[i])).coerceIn(0.0, 1.0)
            val fe = px[i] + t * dx[i]; val fn = py[i] + t * dy[i]
            val d = hypot(fe - x, fn - y)
            if (d <= r) c.add(doubleArrayOf(i.toDouble(), t, fe, fn, d))
        }
        if (c.isEmpty()) return null
        // by distance, ties by segment key (== np.lexsort((ids, dist)) in road_hmm.py)
        c.sortWith(compareBy<DoubleArray>({ it[4] }, { segKey[it[0].toInt()] }))
        val m = min(c.size, MAX_CANDS)
        val seg = IntArray(m) { c[it][0].toInt() }
        val dirn = IntArray(m) { 1 }; val head = DoubleArray(m) { brg[seg[it]] }
        if (moving) for (j in 0 until m) if (!oneway[seg[j]] && abs(wrap(psi - head[j])) > PI / 2) {
            dirn[j] = -1; head[j] = wrap(head[j] + PI)
        }
        return Cands(LongArray(m) { segKey[seg[it]] }, dirn, DoubleArray(m) { c[it][1] },
                     LongArray(m) { if (dirn[it] > 0) bNode[seg[it]] else aNode[seg[it]] },
                     LongArray(m) { if (dirn[it] > 0) aNode[seg[it]] else bNode[seg[it]] },
                     DoubleArray(m) { len[seg[it]] }, Array(m) { doubleArrayOf(c[it][2], c[it][3]) },
                     DoubleArray(m) { c[it][4] }, head, seg)
    }

    /** Bounded Dijkstra on the directed window graph: node -> driving distance from src. */
    private fun reach(src: Long, maxD: Double): HashMap<Long, Double> {
        val dist = HashMap<Long, Double>(); dist[src] = 0.0
        val pq = PriorityQueue<Pair<Double, Long>>(compareBy { it.first }); pq.add(0.0 to src)
        while (pq.isNotEmpty()) {
            val (dd, u) = pq.poll()
            if (dd > (dist[u] ?: Double.MAX_VALUE) || dd > maxD) continue
            for (edge in out[u] ?: continue) {
                val sg = edge[0]
                val v = if (edge[1] > 0) bNode[sg] else aNode[sg]
                val nd = dd + len[sg]
                if (nd < (dist[v] ?: Double.MAX_VALUE) && nd <= maxD) { dist[v] = nd; pq.add(nd to v) }
            }
        }
        return dist
    }

    private fun route(pv: Cands, i: Int, cu: Cands, j: Int, cache: HashMap<Int, HashMap<Long, Double>>): Double {
        val d0 = pv.dirn[i]; val t0 = pv.t[i]; val l0 = pv.len[i]
        val d1 = cu.dirn[j]; val t1 = cu.t[j]; val l1 = cu.len[j]
        if (pv.key[i] == cu.key[j] && d0 == d1) {
            val along = (t1 - t0) * l0 * d0
            return if (along >= -3.0) along else Double.POSITIVE_INFINITY
        }
        val r = cache.getOrPut(i) { reach(pv.exitNode[i], travel + 60.0) }
        val sp = r[cu.entryNode[j]] ?: return Double.POSITIVE_INFINITY
        return (if (d0 > 0) (1 - t0) * l0 else t0 * l0) + sp + (if (d1 > 0) t1 * l1 else (1 - t1) * l1)
    }

    /** One 1 Hz step at the filter's (e, n, psi); call [addTravel] with the distance driven in between. */
    fun update(e: Double, n: Double, psi: Double, speed: Double, posSigma: Double): Match? {
        if (!built || hypot(e - cE, n - cN) > RECENTER_M) rebuild(e, n)
        val moving = speed > 2.0
        val sig = max(SIGMA_MIN, posSigma)
        val cu = candidates(e, n, psi, moving, min(max(RADIUS, 3 * sig), RADIUS_MAX))
        if (cu == null) { reset(); return null }
        val m = cu.key.size
        val em = DoubleArray(m) {
            var v = -0.5 * (cu.dist[it] / sig) * (cu.dist[it] / sig)
            if (moving) { val dh = wrap(psi - cu.head[it]); v -= 0.5 * (dh / HEADING_SIGMA) * (dh / HEADING_SIGMA) }
            v
        }
        var lp: DoubleArray
        val pv = prev
        if (pv == null) lp = em
        else {
            val cache = HashMap<Int, HashMap<Long, Double>>()
            lp = DoubleArray(m) { Double.NEGATIVE_INFINITY }
            for (j in 0 until m) {
                var best = Double.NEGATIVE_INFINITY
                for (i in logp.indices) {
                    if (logp[i] == Double.NEGATIVE_INFINITY) continue
                    val rd = route(pv, i, cu, j, cache)
                    if (rd < Double.POSITIVE_INFINITY) best = max(best, logp[i] - abs(rd - travel) / BETA)
                }
                lp[j] = best + em[j]
            }
            if (lp.none { it.isFinite() }) lp = em              // broken chain (off-map, big jump): restart
        }
        val mx = lp.max()
        val post = DoubleArray(m) { exp(lp[it] - mx) }
        val sum = post.sum()
        for (i in 0 until m) post[i] /= sum
        prev = cu; logp = DoubleArray(m) { ln(max(post[it], 1e-300)) }; travel = 0.0
        var k = 0
        for (i in 1 until m) if (post[i] > post[k]) k = i         // first maximum, like np.argmax
        var conf = 0.0
        for (i in 0 until m) {
            val df = hypot(cu.foot[i][0] - cu.foot[k][0], cu.foot[i][1] - cu.foot[k][1])
            if (df < 5.0 && abs(wrap(cu.head[i] - cu.head[k])) < SAME_ROAD_DEG * PI / 180) conf += post[i]
        }
        val s = cu.seg[k]; val tk = cu.t[k]; val l = len[s]
        return Match((cu.key[k] shr 8).toInt(), (cu.key[k] and 0xff).toInt(), cu.foot[k][0], cu.foot[k][1],
                     cu.head[k], cu.dist[k], conf, post[k], l, min(tk, 1 - tk) * l, hw[s], tun[s]).also { last = it }
    }

    companion object {
        const val WINDOW_M = 1000.0
        const val RECENTER_M = 400.0
        const val CELL_M = 100.0
        const val SIGMA_MIN = 4.0
        const val BETA = 8.0
        const val RADIUS = 45.0
        const val RADIUS_MAX = 150.0
        const val MAX_CANDS = 10
        const val SAME_ROAD_DEG = 30.0
        val HEADING_SIGMA = 25.0 * PI / 180
        const val HW_SERVICE = 14                               // osm_layers.HW_CODE["service"]
        private const val SCALE = 1e7
        // osm_layers.ROAD_CLASS order (HW_CODE = index + 1); "_link" uses its road's width
        private val HW_NAMES = arrayOf("", "motorway", "motorway", "trunk", "trunk", "primary", "primary",
            "secondary", "secondary", "tertiary", "tertiary", "unclassified", "residential", "living_street", "service")
        private val HALF_WIDTH = mapOf("motorway" to 7.0, "trunk" to 7.0, "primary" to 6.0, "secondary" to 5.0,
            "tertiary" to 4.5, "unclassified" to 3.5, "residential" to 3.5, "living_street" to 3.0, "service" to 3.0)

        fun halfWidth(code: Int): Double = HALF_WIDTH[HW_NAMES.getOrElse(code) { "" }] ?: 3.5
        private fun cell(cx: Int, cy: Int) = (cx.toLong() shl 32) or (cy.toLong() and 0xffffffffL)

        /** Python's (a + pi) % 2pi - pi (floor modulo), so wraps agree to the last bit. */
        fun wrap(a: Double): Double {
            val m = (a + PI) % (2 * PI)
            return (if (m < 0) m + 2 * PI else m) - PI
        }
    }
}
