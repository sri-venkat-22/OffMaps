package com.offmaps.nav

import android.content.Context
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor

/**
 * The bundled offline road network (assets/map/roads.bin, written by
 * tools/osm_layers.py -- see its docstring for the byte layout; read_roads_bin()
 * there is this reader's Python twin). Ways are stored in lat/lon so they can be
 * projected into whatever ENU frame the first GNSS fix defines.
 *
 * A whole city is far too much to hand the native matcher (mm_match scans every
 * segment), so ways are bucketed in a coarse lat/lon grid and [waysNear] returns
 * only the ones around the car. RoadMatcher keeps a ~1 km window of those loaded.
 */
class RoadNetwork private constructor(
    private val start: IntArray,     // point offset of each way; start[nWays] == nPts
    private val flags: ByteArray,    // bit0 tunnel, bit1 oneway, bits2-5 highway class (osm_layers.HW_CODE)
    private val lat7: IntArray,      // degrees * 1e7
    private val lon7: IntArray,
) : RoadSource {
    val nWays: Int get() = flags.size
    private val cells = HashMap<Long, IntArray>()
    private val stamp = IntArray(flags.size)   // dedupe marker for waysNear
    private var gen = 0

    init {
        val tmp = HashMap<Long, IntList>()
        for (w in 0 until nWays) {
            var la0 = Int.MAX_VALUE; var la1 = Int.MIN_VALUE
            var lo0 = Int.MAX_VALUE; var lo1 = Int.MIN_VALUE
            for (p in start[w] until start[w + 1]) {
                la0 = minOf(la0, lat7[p]); la1 = maxOf(la1, lat7[p])
                lo0 = minOf(lo0, lon7[p]); lo1 = maxOf(lo1, lon7[p])
            }
            for (iy in cell(la0 / SCALE)..cell(la1 / SCALE))
                for (ix in cell(lo0 / SCALE)..cell(lo1 / SCALE))
                    tmp.getOrPut(key(iy, ix)) { IntList() }.add(w)
        }
        for ((k, v) in tmp) cells[k] = v.toArray()
    }

    override fun tunnel(w: Int) = flags[w].toInt() and 1
    override fun oneway(w: Int) = (flags[w].toInt() shr 1) and 1
    override fun hwCode(w: Int) = (flags[w].toInt() shr 2) and 15   // 0 = unknown (roads.bin built before Phase 9)
    override fun size(w: Int) = start[w + 1] - start[w]
    fun lat(w: Int, i: Int) = lat7[start[w] + i] / SCALE
    fun lon(w: Int, i: Int) = lon7[start[w] + i] / SCALE
    override fun lat7(w: Int, i: Int) = lat7[start[w] + i]      // raw 1e-7 deg: exact vertex identity (junctions)
    override fun lon7(w: Int, i: Int) = lon7[start[w] + i]

    /** Ids of every way whose grid cells touch the box of +-radiusM around (lat, lon). */
    override fun waysNear(lat: Double, lon: Double, radiusM: Double): IntArray {
        val dLat = radiusM / M_PER_DEG
        val dLon = radiusM / (M_PER_DEG * cos(lat * PI / 180.0).coerceAtLeast(0.1))
        val out = IntList()
        gen++
        for (iy in cell(lat - dLat)..cell(lat + dLat))
            for (ix in cell(lon - dLon)..cell(lon + dLon)) {
                val ws = cells[key(iy, ix)] ?: continue
                for (w in ws) if (stamp[w] != gen) { stamp[w] = gen; out.add(w) }
            }
        return out.toArray()
    }

    companion object {
        const val ASSET = "map/roads.bin"
        private const val SCALE = 1e7
        private const val CELL_DEG = 0.005                  // ~550 m cells
        private const val M_PER_DEG = PI / 180.0 * Enu.R_EARTH
        private fun cell(deg: Double) = floor(deg / CELL_DEG).toInt()
        private fun key(iy: Int, ix: Int) = (iy.toLong() shl 32) or (ix.toLong() and 0xffffffffL)

        /** Load the bundled network, or null if no roads.bin was shipped. Call off the UI thread. */
        fun fromAssets(ctx: Context): RoadNetwork? = try {
            // stored uncompressed (build.gradle noCompress) so it can be memory-mapped in place
            ctx.assets.openFd(ASSET).use { fd ->
                FileInputStream(fd.fileDescriptor).channel.use { ch ->
                    parse(ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.length))
                }
            }
        } catch (e: IOException) {
            null
        }

        fun parse(buf: ByteBuffer): RoadNetwork {
            val b = buf.order(ByteOrder.LITTLE_ENDIAN)
            val magic = ByteArray(4).also { b.get(it) }
            require(String(magic, Charsets.US_ASCII) == "OMRD") { "not a roads.bin" }
            require(b.int == 1) { "unsupported roads.bin version" }
            val nw = b.int; val np = b.int
            val start = IntArray(nw + 1).also { b.asIntBuffer().get(it) }
            b.position(b.position() + 4 * (nw + 1))
            val flags = ByteArray(nw).also { b.get(it) }
            b.position(b.position() + (4 - nw % 4) % 4)
            val lat = IntArray(np).also { b.asIntBuffer().get(it) }
            b.position(b.position() + 4 * np)
            val lon = IntArray(np).also { b.asIntBuffer().get(it) }
            return RoadNetwork(start, flags, lat, lon)
        }
    }
}

/** Growable int list (avoids boxing ~10^5 way ids into List<Int>). */
private class IntList {
    private var a = IntArray(4); private var n = 0
    fun add(v: Int) { if (n == a.size) a = a.copyOf(n * 2); a[n++] = v }
    fun toArray(): IntArray = a.copyOf(n)
}
