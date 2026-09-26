package com.offmaps.nav

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RadialGradient
import android.graphics.RectF
import android.graphics.Shader
import android.view.Gravity
import android.os.SystemClock
import org.maplibre.android.camera.CameraPosition
import org.maplibre.android.camera.CameraUpdateFactory
import org.maplibre.android.geometry.LatLng
import org.maplibre.android.maps.MapLibreMap
import org.maplibre.android.maps.MapView
import org.maplibre.android.maps.Style
import org.maplibre.android.style.expressions.Expression.eq
import org.maplibre.android.style.expressions.Expression.exponential
import org.maplibre.android.style.expressions.Expression.get
import org.maplibre.android.style.expressions.Expression.interpolate
import org.maplibre.android.style.expressions.Expression.literal
import org.maplibre.android.style.expressions.Expression.product
import org.maplibre.android.style.expressions.Expression.stop
import org.maplibre.android.style.expressions.Expression.zoom
import org.maplibre.android.style.layers.CircleLayer
import org.maplibre.android.style.layers.LineLayer
import org.maplibre.android.style.layers.Property
import org.maplibre.android.style.layers.SymbolLayer
import org.maplibre.android.style.layers.PropertyFactory.circleColor
import org.maplibre.android.style.layers.PropertyFactory.circleOpacity
import org.maplibre.android.style.layers.PropertyFactory.circlePitchAlignment
import org.maplibre.android.style.layers.PropertyFactory.circleRadius
import org.maplibre.android.style.layers.PropertyFactory.circleStrokeColor
import org.maplibre.android.style.layers.PropertyFactory.circleStrokeOpacity
import org.maplibre.android.style.layers.PropertyFactory.circleStrokeWidth
import org.maplibre.android.style.layers.PropertyFactory.iconAllowOverlap
import org.maplibre.android.style.layers.PropertyFactory.iconIgnorePlacement
import org.maplibre.android.style.layers.PropertyFactory.iconImage
import org.maplibre.android.style.layers.PropertyFactory.iconPitchAlignment
import org.maplibre.android.style.layers.PropertyFactory.iconRotate
import org.maplibre.android.style.layers.PropertyFactory.iconRotationAlignment
import org.maplibre.android.style.layers.PropertyFactory.lineCap
import org.maplibre.android.style.layers.PropertyFactory.lineColor
import org.maplibre.android.style.layers.PropertyFactory.lineJoin
import org.maplibre.android.style.layers.PropertyFactory.lineOpacity
import org.maplibre.android.style.layers.PropertyFactory.lineWidth
import org.maplibre.android.style.sources.GeoJsonSource
import org.maplibre.geojson.Feature
import org.maplibre.geojson.FeatureCollection
import org.maplibre.geojson.LineString
import org.maplibre.geojson.Point
import java.io.File
import kotlin.math.cos

/**
 * The offline map. MapLibre renders the bundled Hyderabad vector tiles
 * (assets/map/tiles.mbtiles, from tools/build_map.sh) with assets/map/style.json,
 * and the live tracks are drawn on top -- the same three stories TrackView told on
 * a blank canvas, now on real streets:
 *   grey  = GNSS fixes (keeps logging during a simulated outage),
 *   blue  = fused ESKF estimate,
 *   amber = fused estimate while dead-reckoning (outage),
 * over a daytime basemap. "You are here" is drawn the way Google Maps draws it: a
 * blue dot in a white ring, a fading beam along the heading, and a pale accuracy
 * circle -- here the filter's own 1-sigma position uncertainty, so it visibly
 * grows while dead-reckoning and shrinks when GNSS returns. The camera follows
 * the car until the user pans; [follow] re-arms it. Colours come from [Ui].
 *
 * Main thread only.
 */
class NavMap(private val ctx: Context, private val view: MapView) {
    private var map: MapLibreMap? = null
    private var style: Style? = null
    private var following = true
    private var zoomedIn = false
    private var lastTrackPush = 0L
    private var insetTop = 0
    private var insetBottom = 0
    private var dr = false

    // tracks in lon/lat. Fused track is a list of runs, each with one outage state.
    private val gnss = ArrayList<Point>()
    private val runs = ArrayList<Pair<Boolean, ArrayList<Point>>>()
    private val shocks = ArrayList<Point>()                 // pothole / bump markers (Phase 7c)
    private var pos: Point? = null
    private var psi = 0.0
    private var sigma = 0.0

    /** tiles = the mbtiles file on disk, or null to draw tracks over a blank background. */
    fun load(tiles: File?, onReady: () -> Unit = {}) {
        view.getMapAsync { m ->
            map = m
            m.cameraPosition = CameraPosition.Builder()
                .target(HYDERABAD).zoom(11.5).build()
            m.uiSettings.apply {
                isCompassEnabled = false                 // the sheet has its own heading dial
                isLogoEnabled = false
                isAttributionEnabled = true              // "(c) OpenStreetMap contributors" (ODbL)
                attributionGravity = Gravity.BOTTOM or Gravity.START
                setAttributionTintColor(Ui.MUTED)
            }
            applyInsets()
            m.addOnCameraMoveStartedListener { reason ->
                if (reason == MapLibreMap.OnCameraMoveStartedListener.REASON_API_GESTURE) following = false
            }
            m.setStyle(Style.Builder().fromJson(styleJson(tiles))) { s ->
                style = s
                addOverlays(s)
                pushTracks(); pushPos()
                onReady()
            }
        }
    }

    private fun styleJson(tiles: File?): String {
        val base = ctx.assets.open(STYLE_ASSET).bufferedReader().use { it.readText() }
        if (tiles != null) return base.replace("{MBTILES}", "mbtiles://" + tiles.absolutePath)
        // no basemap shipped: keep only the background layer so the tracks still draw
        return """{"version":8,"sources":{},"layers":[{"id":"bg","type":"background","paint":{"background-color":"#F3F1ED"}}]}"""
    }

    private fun addOverlays(s: Style) {
        s.addSource(GeoJsonSource(SRC_GNSS))
        s.addSource(GeoJsonSource(SRC_FUSED))
        s.addSource(GeoJsonSource(SRC_POS))
        s.addSource(GeoJsonSource(SRC_SHOCK))
        s.addImage(IMG_BEAM, beamBitmap())
        s.addImage(IMG_DOT, dotBitmap())
        val round = arrayOf(lineJoin(Property.LINE_JOIN_ROUND), lineCap(Property.LINE_CAP_ROUND))
        val isDr = eq(get("dr"), literal(true))
        s.addLayer(LineLayer("gnss-line", SRC_GNSS).withProperties(
            lineColor(Ui.GNSS), lineWidth(3f), lineOpacity(0.9f), *round))
        // fused track: a white casing under the line, like a route on a light map
        s.addLayer(LineLayer("fused-casing", SRC_FUSED).withProperties(
            lineColor(Color.WHITE), lineWidth(8f), *round))
        s.addLayer(LineLayer("fused-line", SRC_FUSED).withProperties(
            lineColor(Ui.FUSED), lineWidth(5f), *round)
            .withFilter(eq(get("dr"), literal(false))))
        s.addLayer(LineLayer("dr-line", SRC_FUSED).withProperties(
            lineColor(Ui.DR), lineWidth(5f), *round)
            .withFilter(isDr))
        s.addLayer(CircleLayer("shock-dot", SRC_SHOCK).withProperties(      // potholes / bumps (Phase 7c)
            circleRadius(5f), circleColor(Ui.SHOCK),
            circleStrokeColor(Color.WHITE), circleStrokeWidth(2f)))
        // you are here: accuracy circle (metres -> px at every zoom), heading beam, dot
        s.addLayer(CircleLayer("pos-accuracy", SRC_POS).withProperties(
            circleRadius(interpolate(exponential(2), zoom(),
                stop(0, get("r0")), stop(22, product(literal(Z22), get("r0"))))),
            circleColor(Ui.PUCK), circleOpacity(0.15f),
            circleStrokeColor(Ui.PUCK), circleStrokeOpacity(0.35f), circleStrokeWidth(1f),
            circlePitchAlignment(Property.CIRCLE_PITCH_ALIGNMENT_MAP)))
        s.addLayer(SymbolLayer("pos-beam", SRC_POS).withProperties(
            iconImage(IMG_BEAM), iconRotate(get("bearing")),
            iconRotationAlignment(Property.ICON_ROTATION_ALIGNMENT_MAP),
            iconPitchAlignment(Property.ICON_PITCH_ALIGNMENT_MAP),
            iconAllowOverlap(true), iconIgnorePlacement(true)))
        s.addLayer(SymbolLayer("pos-dot", SRC_POS).withProperties(
            iconImage(IMG_DOT), iconAllowOverlap(true), iconIgnorePlacement(true)))
    }

    /** Heading beam: a 70 degree fan pointing up, fading out from the dot. */
    private fun beamBitmap(): Bitmap {
        val d = ctx.resources.displayMetrics.density
        val r = BEAM_DP * d
        val bmp = Bitmap.createBitmap((2 * r).toInt(), (2 * r).toInt(), Bitmap.Config.ARGB_8888)
        val c = Canvas(bmp); val cx = r; val cy = r
        val rgb = Ui.PUCK and 0x00FFFFFF
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            shader = RadialGradient(cx, cy, r, intArrayOf(rgb or 0x8C000000.toInt(), rgb or 0x40000000, rgb),
                                    floatArrayOf(0f, 0.45f, 1f), Shader.TileMode.CLAMP)
        }
        val path = Path().apply {
            moveTo(cx, cy)
            arcTo(RectF(cx - r, cy - r, cx + r, cy + r), -90f - BEAM_DEG / 2, BEAM_DEG)
            close()
        }
        c.drawPath(path, paint)
        return bmp
    }

    /** The dot: blue in a white ring, with a soft shadow. */
    private fun dotBitmap(): Bitmap {
        val d = ctx.resources.displayMetrics.density
        val rOut = 11f * d; val rIn = 8f * d; val pad = 4f * d
        val size = (2 * (rOut + pad)).toInt()
        val bmp = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmp); val cx = size / 2f
        c.drawCircle(cx, cx, rOut, Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE; setShadowLayer(3f * d, 0f, 1f * d, 0x55000000)
        })
        c.drawCircle(cx, cx, rIn, Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Ui.PUCK })
        return bmp
    }

    /** Keep the attribution and the followed car clear of the header and bottom sheet. */
    fun setInsets(topPx: Int, bottomPx: Int) {
        insetTop = topPx; insetBottom = bottomPx; applyInsets()
    }

    private fun applyInsets() {
        val m = map ?: return
        val gap = (8 * ctx.resources.displayMetrics.density).toInt()
        m.uiSettings.setAttributionMargins(gap, 0, 0, insetBottom + gap)
        @Suppress("DEPRECATION")
        m.setPadding(0, insetTop, 0, insetBottom)
    }

    fun clear() {
        gnss.clear(); runs.clear(); shocks.clear(); pos = null; zoomedIn = false; following = true
        pushTracks(); pushPos(); pushShocks()
    }

    /** Re-centre on the car and keep following it. */
    fun follow() {
        following = true
        pos?.let { map?.animateCamera(CameraUpdateFactory.newLatLng(LatLng(it.latitude(), it.longitude())), 400) }
    }

    fun update(s: NavState) {
        val p = Point.fromLngLat(s.lon, s.lat)
        pos = p; psi = s.psi; dr = s.outageActive; sigma = s.posSigma
        val run = runs.lastOrNull()
        if (run == null || run.first != s.outageActive) {
            // new run starts at the previous run's last point so the line stays continuous
            runs.add(s.outageActive to arrayListOf<Point>().also { l -> run?.second?.lastOrNull()?.let(l::add); l.add(p) })
        } else {
            run.second.add(p)
        }
        if (s.newGnss && s.hasGnss) gnss.add(Point.fromLngLat(s.gnssLon, s.gnssLat))
        if (s.newShocks.isNotEmpty()) {
            for (i in s.newShocks.indices step 2) shocks.add(Point.fromLngLat(s.newShocks[i + 1], s.newShocks[i]))
            if (shocks.size > CAP) shocks.subList(0, shocks.size - CAP).clear()
            pushShocks()
        }
        if (runs.sumOf { it.second.size } > CAP) runs.forEach { decimate(it.second) }
        if (gnss.size > CAP) decimate(gnss)

        pushPos()
        val now = SystemClock.uptimeMillis()
        if (now - lastTrackPush > TRACK_PUSH_MS) { lastTrackPush = now; pushTracks() }

        val m = map ?: return
        val ll = LatLng(s.lat, s.lon)
        if (!zoomedIn) {                     // first fix: jump from the city view to street level
            zoomedIn = true
            m.moveCamera(CameraUpdateFactory.newLatLngZoom(ll, 16.5))
        } else if (following) {
            m.moveCamera(CameraUpdateFactory.newLatLng(ll))
        }
    }

    private fun pushTracks() {
        val st = style ?: return
        st.getSourceAs<GeoJsonSource>(SRC_GNSS)?.setGeoJson(
            if (gnss.size >= 2) FeatureCollection.fromFeature(Feature.fromGeometry(LineString.fromLngLats(ArrayList(gnss))))
            else FeatureCollection.fromFeatures(emptyList()))
        st.getSourceAs<GeoJsonSource>(SRC_FUSED)?.setGeoJson(FeatureCollection.fromFeatures(
            runs.filter { it.second.size >= 2 }.map { (dr, pts) ->
                Feature.fromGeometry(LineString.fromLngLats(ArrayList(pts))).also { it.addBooleanProperty("dr", dr) }
            }))
    }

    private fun pushShocks() {
        val st = style ?: return
        st.getSourceAs<GeoJsonSource>(SRC_SHOCK)?.setGeoJson(
            FeatureCollection.fromFeatures(shocks.map { Feature.fromGeometry(it) }))
    }

    private fun pushPos() {
        val st = style ?: return
        val p = pos
        val fc = if (p == null) FeatureCollection.fromFeatures(emptyList()) else {
            // accuracy radius in map px at zoom 0 (512 px tiles); the layer scales it by 2^zoom
            val mPerPx0 = 2 * Math.PI * WEB_MERCATOR_R * cos(Math.toRadians(p.latitude())) / 512.0
            FeatureCollection.fromFeature(Feature.fromGeometry(p).also {
                it.addNumberProperty("bearing", Math.toDegrees(psi))
                it.addNumberProperty("r0", sigma.coerceIn(0.0, MAX_ACCURACY_M) / mPerPx0)
                it.addBooleanProperty("dr", dr)
            })
        }
        st.getSourceAs<GeoJsonSource>(SRC_POS)?.setGeoJson(fc)
    }

    /** Halve a track in place (keep even indices and the last point). */
    private fun decimate(pts: ArrayList<Point>) {
        if (pts.size < 4) return
        val last = pts.last()
        val kept = pts.filterIndexed { i, _ -> i % 2 == 0 }
        pts.clear(); pts.addAll(kept); if (pts.last() != last) pts.add(last)
    }

    companion object {
        const val STYLE_ASSET = "map/style.json"
        const val TILES_ASSET = "map/tiles.mbtiles"
        private const val SRC_GNSS = "trk-gnss"
        private const val SRC_FUSED = "trk-fused"
        private const val SRC_POS = "trk-pos"
        private const val SRC_SHOCK = "trk-shock"
        private const val CAP = 6000                 // points per track before halving
        private const val TRACK_PUSH_MS = 500L       // re-upload the long tracks at 2 Hz; the dot at 10 Hz
        private const val IMG_BEAM = "puck-beam"
        private const val IMG_DOT = "puck-dot"
        private const val BEAM_DP = 56f              // beam length
        private const val BEAM_DEG = 70f             // beam opening
        private const val MAX_ACCURACY_M = 2000.0
        private const val Z22 = 4194304.0            // 2^22: the accuracy circle's zoom-22 scale
        private const val WEB_MERCATOR_R = 6378137.0
        private val HYDERABAD = LatLng(17.385, 78.4867)

        /**
         * MapLibre reads mbtiles from a real file, so copy the bundled asset out once
         * (again only if the shipped size changed). Returns null if none is bundled.
         * Call off the UI thread.
         */
        fun installTiles(ctx: Context): File? {
            val fd = try { ctx.assets.openFd(TILES_ASSET) } catch (e: java.io.IOException) { return null }
            val len = fd.use { it.length }
            val out = File(ctx.filesDir, TILES_ASSET)
            if (out.length() != len) {
                out.parentFile?.mkdirs()
                val tmp = File(out.path + ".tmp")
                ctx.assets.open(TILES_ASSET).use { i -> tmp.outputStream().use { o -> i.copyTo(o, 1 shl 16) } }
                tmp.renameTo(out)
            }
            return out
        }
    }
}
