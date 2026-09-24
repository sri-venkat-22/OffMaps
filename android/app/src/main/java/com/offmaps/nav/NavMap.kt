package com.offmaps.nav

import android.content.Context
import android.view.Gravity
import android.os.SystemClock
import org.maplibre.android.camera.CameraPosition
import org.maplibre.android.camera.CameraUpdateFactory
import org.maplibre.android.geometry.LatLng
import org.maplibre.android.maps.MapLibreMap
import org.maplibre.android.maps.MapView
import org.maplibre.android.maps.Style
import org.maplibre.android.style.expressions.Expression.color
import org.maplibre.android.style.expressions.Expression.eq
import org.maplibre.android.style.expressions.Expression.switchCase
import org.maplibre.android.style.expressions.Expression.geometryType
import org.maplibre.android.style.expressions.Expression.get
import org.maplibre.android.style.expressions.Expression.literal
import org.maplibre.android.style.layers.CircleLayer
import org.maplibre.android.style.layers.LineLayer
import org.maplibre.android.style.layers.Property
import org.maplibre.android.style.layers.PropertyFactory.circleBlur
import org.maplibre.android.style.layers.PropertyFactory.circleColor
import org.maplibre.android.style.layers.PropertyFactory.circleOpacity
import org.maplibre.android.style.layers.PropertyFactory.circleRadius
import org.maplibre.android.style.layers.PropertyFactory.circleStrokeColor
import org.maplibre.android.style.layers.PropertyFactory.circleStrokeWidth
import org.maplibre.android.style.layers.PropertyFactory.lineCap
import org.maplibre.android.style.layers.PropertyFactory.lineColor
import org.maplibre.android.style.layers.PropertyFactory.lineBlur
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
import kotlin.math.sin

/**
 * The offline map. MapLibre renders the bundled Hyderabad vector tiles
 * (assets/map/tiles.mbtiles, from tools/build_map.sh) with assets/map/style.json,
 * and the live tracks are drawn on top -- the same three stories TrackView told on
 * a blank canvas, now on real streets:
 *   light grey = GNSS fixes (keeps logging during a simulated outage),
 *   blue       = fused ESKF estimate,
 *   amber      = fused estimate while dead-reckoning (outage),
 * plus a glowing position puck (cyan, amber while dead-reckoning) with a heading
 * tick, over a night-style basemap. The camera follows the car until the user
 * pans; [follow] re-arms it. Colours come from [Ui].
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
        return """{"version":8,"sources":{},"layers":[{"id":"bg","type":"background","paint":{"background-color":"#0E1626"}}]}"""
    }

    private fun addOverlays(s: Style) {
        s.addSource(GeoJsonSource(SRC_GNSS))
        s.addSource(GeoJsonSource(SRC_FUSED))
        s.addSource(GeoJsonSource(SRC_POS))
        s.addSource(GeoJsonSource(SRC_SHOCK))
        val round = arrayOf(lineJoin(Property.LINE_JOIN_ROUND), lineCap(Property.LINE_CAP_ROUND))
        val isDr = eq(get("dr"), literal(true))
        val modeColor = switchCase(isDr, color(Ui.DR), color(Ui.ACCENT))
        s.addLayer(LineLayer("gnss-line", SRC_GNSS).withProperties(
            lineColor(Ui.GNSS), lineWidth(3f), lineOpacity(0.75f), *round))
        // fused track: a soft glow under a crisp line, blue on GNSS, amber while dead-reckoning
        s.addLayer(LineLayer("fused-glow", SRC_FUSED).withProperties(
            lineColor(switchCase(isDr, color(Ui.DR), color(Ui.FUSED))),
            lineWidth(14f), lineOpacity(0.28f), lineBlur(6f), *round))
        s.addLayer(LineLayer("fused-line", SRC_FUSED).withProperties(
            lineColor(Ui.FUSED), lineWidth(5f), *round)
            .withFilter(eq(get("dr"), literal(false))))
        s.addLayer(LineLayer("dr-line", SRC_FUSED).withProperties(
            lineColor(Ui.DR), lineWidth(5f), *round)
            .withFilter(isDr))
        s.addLayer(CircleLayer("shock-dot", SRC_SHOCK).withProperties(      // potholes / bumps (Phase 7c)
            circleRadius(6f), circleColor(Ui.SHOCK),
            circleStrokeColor(Ui.BG), circleStrokeWidth(2f)))
        val isPoint = eq(geometryType(), literal("Point"))
        s.addLayer(CircleLayer("pos-halo", SRC_POS).withProperties(
            circleRadius(26f), circleColor(modeColor), circleOpacity(0.22f), circleBlur(0.6f))
            .withFilter(isPoint))
        s.addLayer(LineLayer("pos-heading", SRC_POS).withProperties(
            lineColor(modeColor), lineWidth(4f), lineCap(Property.LINE_CAP_ROUND)))
        s.addLayer(CircleLayer("pos-dot", SRC_POS).withProperties(
            circleRadius(9f), circleColor(modeColor),
            circleStrokeColor(Ui.TEXT), circleStrokeWidth(3f))
            .withFilter(isPoint))
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
        pos = p; psi = s.psi; dr = s.outageActive
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
            // heading tick: HEADING_M ahead along psi (0 = north, +east), in local metres
            val dLat = HEADING_M * cos(psi) / M_PER_DEG
            val dLon = HEADING_M * sin(psi) / (M_PER_DEG * cos(Math.toRadians(p.latitude())))
            val tip = Point.fromLngLat(p.longitude() + dLon, p.latitude() + dLat)
            FeatureCollection.fromFeatures(listOf(
                Feature.fromGeometry(LineString.fromLngLats(listOf(p, tip))),
                Feature.fromGeometry(p)).onEach { it.addBooleanProperty("dr", dr) })
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
        private const val HEADING_M = 25.0
        private const val M_PER_DEG = Math.PI / 180.0 * Enu.R_EARTH
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
