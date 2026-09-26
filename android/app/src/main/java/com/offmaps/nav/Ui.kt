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
 * The app's visual language, in one place: a light, map-first palette (white cards
 * over a daytime map, one blue for "you" and the fused track, amber only while
 * dead-reckoning), plus small builders for the cards, chips and buttons that
 * NavActivity assembles in code (the app has no XML layouts).
 */
object Ui {
    // palette
    val BG = Color.parseColor("#F3F1ED")             // map land; shown while the style loads
    val SURFACE = Color.parseColor("#FFFFFF")        // floating cards over the map
    val SHEET = Color.parseColor("#FFFFFF")          // bottom sheet
    val SURFACE_2 = Color.parseColor("#F1F3F4")      // quiet fills inside a card
    val STROKE = Color.parseColor("#DADCE0")
    val TEXT = Color.parseColor("#202124")
    val MUTED = Color.parseColor("#5F6368")
    val ACCENT = Color.parseColor("#1A73E8")         // blue: primary action + you are here
    val FUSED = Color.parseColor("#1A73E8")          // fused ESKF track
    val DR = Color.parseColor("#E37400")             // amber: dead-reckoning
    val GNSS = Color.parseColor("#9AA0A6")           // grey: raw GNSS fixes
    val OK = Color.parseColor("#188038")
    val DANGER = Color.parseColor("#D93025")
    val SHOCK = Color.parseColor("#C5221F")          // pothole / bump marker
    val PUCK = Color.parseColor("#4285F4")           // the location dot (Google Maps blue)

    val REGULAR: Typeface = Typeface.create("sans-serif", Typeface.NORMAL)
    val BOLD: Typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
    val MONO: Typeface = BOLD                        // numbers: medium weight with tabular figures (num())

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
    }

    fun text(c: Context, s: String, sp: Float, color: Int = TEXT, face: Typeface? = null) =
        TextView(c).apply {
            text = s; textSize = sp; setTextColor(color)
            face?.let { typeface = it }
            includeFontPadding = false
        }

    /** Figures that do not jitter as they change (tabular numerals). */
    fun num(c: Context, s: String, sp: Float, color: Int = TEXT, face: Typeface = BOLD) =
        text(c, s, sp, color, face).apply { fontFeatureSettings = "tnum" }

    /** Small label above a value ("Speed", "Drift", ...). */
    fun label(c: Context, s: String) = text(c, s, 12f, MUTED, REGULAR)

    /** Status chip: dot + text on a pale tint of the colour. */
    class Pill(c: Context) : LinearLayout(c) {
        private val dot = View(c)
        val txt = text(c, "", 13f, TEXT, BOLD)
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
            background = rounded(context, (color and 0x00FFFFFF) or 0x1A000000, 999f)
        }
    }

    /** Header metric: small label over a value, no box. */
    class Chip(c: Context, lbl: String) : LinearLayout(c) {
        val value = num(c, "–", 15f)
        init {
            orientation = VERTICAL
            addView(label(c, lbl).apply { textSize = 11f })
            addView(value, LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT).apply { topMargin = dpi(c, 3f) })
        }
    }

    /** Full-width button. [primary] = filled blue, else outlined. */
    fun button(c: Context, s: String, primary: Boolean) = TextView(c).apply {
        text = s; textSize = 15f; typeface = BOLD; gravity = Gravity.CENTER
        isClickable = true; isFocusable = true
        minHeight = dpi(c, 48f)
        style(this, primary, if (primary) ACCENT else TEXT)
    }

    fun style(b: TextView, filled: Boolean, color: Int) {
        val c = b.context
        val base = if (filled) rounded(c, color, 999f)
                   else rounded(c, SURFACE, 999f, STROKE, 1f)
        b.setTextColor(if (filled) Color.WHITE else color)
        val ripple = if (filled) "#33FFFFFF" else "#1F1A73E8"
        b.background = RippleDrawable(ColorStateList.valueOf(Color.parseColor(ripple)), base, null)
    }
}
