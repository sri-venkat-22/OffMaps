package com.offmaps.nav

import android.content.Context
import android.graphics.Canvas
import android.graphics.DashPathEffect
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Shader
import android.view.View
import java.util.Locale
import kotlin.math.max

/**
 * Live chart of one simulated outage: position error vs GNSS (drift) against the
 * ISRO budget, 10 % of the distance travelled since GNSS was cut. The drift line
 * is green while it is inside the budget and amber once it leaves it.
 * Only meaningful with the Outage Simulator on, where GNSS keeps arriving as truth.
 */
class DriftChart(context: Context) : View(context) {
    private val ts = ArrayList<Float>()
    private val drift = ArrayList<Float>()
    private val budget = ArrayList<Float>()

    private val axis = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Ui.STROKE; strokeWidth = Ui.dp(context, 1f)
    }
    private val budgetPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; color = Ui.MUTED; strokeWidth = Ui.dp(context, 1.5f)
        pathEffect = DashPathEffect(floatArrayOf(Ui.dp(context, 5f), Ui.dp(context, 4f)), 0f)
    }
    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = Ui.dp(context, 2.5f)
        strokeJoin = Paint.Join.ROUND; strokeCap = Paint.Cap.ROUND
    }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val small = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Ui.MUTED; textSize = Ui.dp(context, 9.5f); typeface = Ui.BOLD
    }
    private val p = Path()

    fun clear() { ts.clear(); drift.clear(); budget.clear(); invalidate() }

    fun add(tS: Double, driftM: Double, distM: Double) {
        if (!driftM.isFinite()) return
        ts.add(tS.toFloat()); drift.add(driftM.toFloat()); budget.add((0.10 * distM).toFloat())
        if (ts.size > 1200) { for (l in listOf(ts, drift, budget)) l.subList(0, 600).clear() }
        invalidate()
    }

    val isEmpty get() = ts.isEmpty()

    override fun onDraw(c: Canvas) {
        val padL = Ui.dp(context, 4f); val padB = Ui.dp(context, 14f); val padT = Ui.dp(context, 4f)
        val w = width - padL; val h = height - padB - padT
        c.drawLine(padL, padT + h, width.toFloat(), padT + h, axis)
        if (ts.size < 2) {
            c.drawText("waiting for the outage to start…", padL, padT + h / 2f, small)
            return
        }
        val t0 = ts.first(); val tSpan = max(ts.last() - t0, 10f)
        var yMax = 5f
        for (i in ts.indices) yMax = max(yMax, max(drift[i], budget[i]))
        yMax *= 1.15f
        fun x(i: Int) = padL + (ts[i] - t0) / tSpan * w
        fun y(v: Float) = padT + h - v / yMax * h

        // budget (dashed)
        p.reset(); p.moveTo(x(0), y(budget[0]))
        for (i in 1 until ts.size) p.lineTo(x(i), y(budget[i]))
        c.drawPath(p, budgetPaint)

        // drift: gradient fill + line, coloured by whether we are inside budget now
        val inside = drift.last() <= budget.last()
        val col = if (inside) Ui.OK else Ui.DR
        p.reset(); p.moveTo(x(0), y(drift[0]))
        for (i in 1 until ts.size) p.lineTo(x(i), y(drift[i]))
        val area = Path(p).apply { lineTo(x(ts.size - 1), padT + h); lineTo(x(0), padT + h); close() }
        fill.shader = LinearGradient(0f, padT, 0f, padT + h,
            (col and 0x00FFFFFF) or 0x55000000, col and 0x00FFFFFF, Shader.TileMode.CLAMP)
        c.drawPath(area, fill)
        line.color = col
        c.drawPath(p, line)
        c.drawCircle(x(ts.size - 1), y(drift.last()), Ui.dp(context, 3.5f), line.apply { style = Paint.Style.FILL })
        line.style = Paint.Style.STROKE

        c.drawText(String.format(Locale.US, "0–%.0f s", tSpan), padL, height - Ui.dp(context, 2f), small)
        val lbl = "- - ISRO 10 % budget"
        c.drawText(lbl, width - small.measureText(lbl), height - Ui.dp(context, 2f), small)
    }
}
