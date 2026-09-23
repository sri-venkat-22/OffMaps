package com.offmaps.nav

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * Dependency-free position renderer. Draws two tracks in the local ENU frame,
 * auto-scaled to fit:
 *   - GNSS ground truth (grey) -- keeps logging even during a simulated outage,
 *   - the fused ESKF estimate (blue), with segments produced DURING an outage in
 *     orange so the dead-reckoning divergence is the visible story.
 * Plus the current position marker with a heading tick. No tiles/basemap: Phase 6
 * is about the live filter, and a self-contained canvas makes the drift legible
 * without a map dependency (offline OSM corridor mode is wired in libidr's mm_*).
 */
class TrackView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) :
    View(context, attrs) {

    private val CAP = 8000
    private val fx = ArrayList<Float>(); private val fy = ArrayList<Float>()
    private val fOut = ArrayList<Boolean>()
    private val gx = ArrayList<Float>(); private val gy = ArrayList<Float>()
    private var curE = 0f; private var curN = 0f; private var curPsi = 0.0
    private var hasCur = false

    private val gnssPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#9E9E9E"); style = Paint.Style.STROKE; strokeWidth = 3f
    }
    private val fusedPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#1E88E5"); style = Paint.Style.STROKE; strokeWidth = 5f
    }
    private val outagePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FB8C00"); style = Paint.Style.STROKE; strokeWidth = 5f
    }
    private val markerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#0D47A1") }
    private val headingPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#0D47A1"); style = Paint.Style.STROKE; strokeWidth = 4f
    }
    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#22888888"); strokeWidth = 1f
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.GRAY; textSize = 28f }

    fun addFused(e: Double, n: Double, psi: Double, outage: Boolean) {
        fx.add(e.toFloat()); fy.add(n.toFloat()); fOut.add(outage)
        curE = e.toFloat(); curN = n.toFloat(); curPsi = psi; hasCur = true
        if (fx.size > CAP) decimate(fx, fy, fOut)
        invalidate()
    }

    fun addGnss(e: Double, n: Double) {
        gx.add(e.toFloat()); gy.add(n.toFloat())
        if (gx.size > CAP) decimate(gx, gy, null)
    }

    fun clearTracks() {
        fx.clear(); fy.clear(); fOut.clear(); gx.clear(); gy.clear(); hasCur = false; invalidate()
    }

    /** Halve a track in place (keep even indices) once it exceeds CAP. */
    private fun decimate(xs: ArrayList<Float>, ys: ArrayList<Float>, flags: ArrayList<Boolean>?) {
        var w = 0
        for (r in xs.indices step 2) { xs[w] = xs[r]; ys[w] = ys[r]; flags?.set(w, flags[r]); w++ }
        while (xs.size > w) { val last = xs.size - 1; xs.removeAt(last); ys.removeAt(last); flags?.removeAt(last) }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val n = fx.size + gx.size
        if (n == 0) {
            canvas.drawText("Waiting for GNSS fix…", 32f, height / 2f, textPaint)
            return
        }
        // fit bounds over both tracks
        var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
        var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
        fun acc(x: Float, y: Float) { minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y) }
        for (i in fx.indices) acc(fx[i], fy[i])
        for (i in gx.indices) acc(gx[i], gy[i])
        val pad = 60f
        val spanX = max(maxX - minX, 1f); val spanY = max(maxY - minY, 1f)
        val sc = min((width - 2 * pad) / spanX, (height - 2 * pad) / spanY)
        val cx = (minX + maxX) / 2f; val cy = (minY + maxY) / 2f
        // world (e east, n north) -> screen: x right = e, y down = -n
        fun sx(e: Float) = width / 2f + (e - cx) * sc
        fun sy(nn: Float) = height / 2f - (nn - cy) * sc

        canvas.drawLine(0f, height / 2f, width.toFloat(), height / 2f, gridPaint)
        canvas.drawLine(width / 2f, 0f, width / 2f, height.toFloat(), gridPaint)

        // GNSS ground truth
        for (i in 1 until gx.size)
            canvas.drawLine(sx(gx[i - 1]), sy(gy[i - 1]), sx(gx[i]), sy(gy[i]), gnssPaint)

        // fused track, segment colour tracks outage state
        for (i in 1 until fx.size) {
            val p = if (fOut[i]) outagePaint else fusedPaint
            canvas.drawLine(sx(fx[i - 1]), sy(fy[i - 1]), sx(fx[i]), sy(fy[i]), p)
        }

        // current position + heading tick
        if (hasCur) {
            val px = sx(curE); val py = sy(curN)
            canvas.drawCircle(px, py, 12f, markerPaint)
            val len = 34f
            canvas.drawLine(px, py, px + (sin(curPsi) * len).toFloat(), py - (cos(curPsi) * len).toFloat(), headingPaint)
        }

        // scale bar
        val meters = niceMeters(spanX / 4f)
        val barPx = meters * sc
        val bx = pad; val by = height - pad / 2f
        canvas.drawLine(bx, by, bx + barPx, by, textPaint)
        canvas.drawText("${meters.toInt()} m", bx, by - 10f, textPaint)
    }

    private fun niceMeters(approx: Float): Float {
        val pow = Math.pow(10.0, Math.floor(Math.log10(approx.toDouble().coerceAtLeast(1.0))).toDouble())
        val f = approx / pow
        val mult = if (f < 1.5) 1.0 else if (f < 3.5) 2.0 else if (f < 7.5) 5.0 else 10.0
        return (mult * pow).toFloat()
    }
}
