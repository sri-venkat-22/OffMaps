package com.offmaps.nav

import android.content.Context
import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.RippleDrawable
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView

/**
 * The app's visual language, in one place: a dark "mission control" palette over
 * a night map, plus small builders for the rounded cards, pills and buttons that
 * NavActivity assembles in code (the app has no XML layouts).
 */
object Ui {
    // palette
    val BG = Color.parseColor("#0B1220")
    val SURFACE = Color.parseColor("#EE111A2B")      // header card: near-opaque over the map
    val SHEET = Color.parseColor("#111A2B")          // bottom sheet: opaque so labels never bleed through
    val SURFACE_2 = Color.parseColor("#1A2438")
    val STROKE = Color.parseColor("#26344D")
    val TEXT = Color.parseColor("#F3F6FB")
    val MUTED = Color.parseColor("#8FA0BA")
    val ACCENT = Color.parseColor("#22D3EE")         // cyan: brand + primary action
    val FUSED = Color.parseColor("#3B82F6")          // blue: fused ESKF track
    val DR = Color.parseColor("#F59E0B")             // amber: dead-reckoning
    val GNSS = Color.parseColor("#CBD5E1")           // light grey: GNSS fixes
    val OK = Color.parseColor("#10B981")
    val DANGER = Color.parseColor("#EF4444")
    val SHOCK = Color.parseColor("#F97316")

    val MONO: Typeface = Typeface.create("monospace", Typeface.BOLD)
    val BOLD: Typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
    val BLACK: Typeface = Typeface.create("sans-serif-black", Typeface.NORMAL)
    val CONDENSED: Typeface = Typeface.create("sans-serif-condensed", Typeface.BOLD)

    fun dp(c: Context, v: Float) =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, c.resources.displayMetrics)
    fun dpi(c: Context, v: Float) = dp(c, v).toInt()

    fun rounded(c: Context, fill: Int, radiusDp: Float, stroke: Int = 0, strokeDp: Float = 1f) =
        GradientDrawable().apply {
            setColor(fill); cornerRadius = dp(c, radiusDp)
            if (stroke != 0) setStroke(dpi(c, strokeDp), stroke)
        }

    /** A card with only its top corners rounded: the bottom sheet. */
    fun sheet(c: Context, fill: Int, radiusDp: Float) = GradientDrawable().apply {
        setColor(fill)
        val r = dp(c, radiusDp)
        cornerRadii = floatArrayOf(r, r, r, r, 0f, 0f, 0f, 0f)
        setStroke(dpi(c, 1f), STROKE)
    }

    fun text(c: Context, s: String, sp: Float, color: Int = TEXT, face: Typeface? = null) =
        TextView(c).apply {
            text = s; textSize = sp; setTextColor(color)
            face?.let { typeface = it }
            includeFontPadding = false
        }

    /** Small caps label above a value ("SPEED", "DRIFT", ...). */
    fun label(c: Context, s: String) = text(c, s, 10f, MUTED, BOLD).apply { letterSpacing = 0.12f }

    /** Status pill: dot + text on a tinted background. */
    class Pill(c: Context) : LinearLayout(c) {
        private val dot = View(c)
        val txt = text(c, "", 11f, TEXT, BOLD).apply { letterSpacing = 0.08f }
        init {
            orientation = HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            val p = dpi(c, 10f); setPadding(p, dpi(c, 6f), p + dpi(c, 2f), dpi(c, 6f))
            val d = dpi(c, 8f)
            addView(dot, LayoutParams(d, d).apply { marginEnd = dpi(c, 7f) })
            addView(txt)
        }
        fun set(s: String, color: Int) {
            txt.text = s; txt.setTextColor(color)
            dot.background = GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(color) }
            background = rounded(context, (color and 0x00FFFFFF) or 0x26000000, 999f, (color and 0x00FFFFFF) or 0x66000000)
        }
    }

    /** Metric chip for the header strip: LABEL value. */
    class Chip(c: Context, lbl: String) : LinearLayout(c) {
        val value = text(c, "–", 13f, TEXT, MONO)
        init {
            orientation = VERTICAL
            val p = dpi(c, 10f); setPadding(p, dpi(c, 6f), p, dpi(c, 6f))
            background = rounded(c, SURFACE_2, 12f)
            addView(label(c, lbl).apply { textSize = 9f })
            addView(value, LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT).apply { topMargin = dpi(c, 2f) })
        }
    }

    /** Full-width pill button. [primary] = filled accent, else outlined surface. */
    fun button(c: Context, s: String, primary: Boolean) = TextView(c).apply {
        text = s; textSize = 15f; typeface = BOLD; gravity = Gravity.CENTER
        letterSpacing = 0.02f; isClickable = true; isFocusable = true
        minHeight = dpi(c, 52f)
        style(this, primary, if (primary) ACCENT else TEXT)
    }

    fun style(b: TextView, filled: Boolean, color: Int) {
        val c = b.context
        val base = if (filled) rounded(c, color, 999f)
                   else rounded(c, SURFACE_2, 999f, (color and 0x00FFFFFF) or 0x80000000.toInt(), 1.5f)
        b.setTextColor(if (filled) BG else color)
        b.background = RippleDrawable(ColorStateList.valueOf(Color.parseColor("#33FFFFFF")), base, null)
    }
}
