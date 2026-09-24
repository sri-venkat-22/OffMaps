package com.offmaps.nav

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.view.View
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

/**
 * Heading dial: a fixed ring (north up, red tick; ticks every 30 degrees) and an arrow at the
 * fused heading psi (0 = north, clockwise). Arrow colour follows the nav mode.
 */
class CompassView(context: Context) : View(context) {
    private var psi = 0.0
    private var arrowColor = Ui.ACCENT
    private var live = false

    private val ring = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; color = Ui.STROKE; strokeWidth = Ui.dp(context, 2f)
    }
    private val tick = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Ui.MUTED; strokeWidth = Ui.dp(context, 1.5f); strokeCap = Paint.Cap.ROUND
    }
    private val north = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Ui.DANGER; strokeWidth = Ui.dp(context, 3f); strokeCap = Paint.Cap.ROUND
    }
    private val arrow = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val path = Path()

    fun set(psiRad: Double, color: Int) {
        psi = psiRad; arrowColor = color; live = true; invalidate()
    }

    fun reset() { live = false; psi = 0.0; invalidate() }

    override fun onDraw(c: Canvas) {
        val cx = width / 2f; val cy = height / 2f
        val r = min(cx, cy) - ring.strokeWidth
        c.drawCircle(cx, cy, r, ring)
        for (i in 0 until 12) {
            val a = Math.toRadians(i * 30.0)
            val inner = if (i % 3 == 0) r * 0.78f else r * 0.86f
            c.drawLine(cx + (sin(a) * inner).toFloat(), cy - (cos(a) * inner).toFloat(),
                       cx + (sin(a) * r * 0.95f).toFloat(), cy - (cos(a) * r * 0.95f).toFloat(),
                       if (i == 0) north else tick)                  // north tick in red
        }

        // arrow: a slim chevron pointing along psi
        arrow.color = if (live) arrowColor else Ui.STROKE
        c.save()
        c.rotate(Math.toDegrees(psi).toFloat(), cx, cy)
        path.reset()
        path.moveTo(cx, cy - r * 0.62f)
        path.lineTo(cx + r * 0.26f, cy + r * 0.34f)
        path.lineTo(cx, cy + r * 0.18f)
        path.lineTo(cx - r * 0.26f, cy + r * 0.34f)
        path.close()
        c.drawPath(path, arrow)
        c.restore()
    }
}
