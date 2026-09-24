package com.offmaps.nav

import android.Manifest
import android.animation.ObjectAnimator
import android.animation.ValueAnimator
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.GnssStatus
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import org.maplibre.android.MapLibre
import org.maplibre.android.maps.MapLibreMapOptions
import org.maplibre.android.maps.MapView
import java.util.Locale

/**
 * Phase 6 -- the real on-device nav app. Unlike the Phase-1 logger (which only
 * records CSVs), this runs the live fusion: IMU + GNSS stream straight into
 * FusionEngine, which drives the embedded libidr ESKF and renders position in
 * real time. The "Outage Simulator" toggle is honoured against the LIVE filter
 * -- GNSS keeps arriving and is drawn as ground truth, but while the toggle is ON
 * the filter is fed no GNSS and dead-reckons, so the blue track visibly diverges
 * from grey. That divergence, and the metres of drift, is the whole demo.
 *
 * Everything is drawn on the offline Hyderabad map (NavMap: MapLibre + bundled
 * mbtiles), and while dead-reckoning the bundled road network (RoadNetwork ->
 * RoadMatcher) snaps the estimate's cross-track and heading to the road. The
 * "Road snapping" toggle turns that off so the two outage tracks can be compared.
 *
 * Sensor + GNSS callbacks run on a dedicated background thread so the native core
 * is only ever touched from one thread; the engine posts UI snapshots to main.
 *
 * Screen: the map fills the display; a header card carries the nav mode and the
 * satellite picture, and a bottom sheet the live readout (speed, heading dial,
 * drift vs GNSS as % of distance, the ISRO 10 % budget chart) and the controls.
 * Built in code with the palette in [Ui].
 */
class NavActivity : AppCompatActivity(), SensorEventListener, LocationListener {

    private lateinit var sm: SensorManager
    private lateinit var lm: LocationManager
    private lateinit var engine: FusionEngine
    private lateinit var mapView: MapView
    private lateinit var navMap: NavMap

    // header
    private lateinit var modePill: Ui.Pill
    private lateinit var chipSat: Ui.Chip
    private lateinit var chipNavic: Ui.Chip
    private lateinit var chipCn0: Ui.Chip
    private lateinit var chipTrust: Ui.Chip
    private var pulse: ObjectAnimator? = null
    private var pillState = ""

    // bottom sheet
    private lateinit var sheet: LinearLayout
    private lateinit var readyPanel: LinearLayout
    private lateinit var livePanel: LinearLayout
    private lateinit var rowMap: TextView
    private lateinit var rowRoads: TextView
    private lateinit var rowModel: TextView
    private lateinit var rowSensors: TextView
    private lateinit var speedVal: TextView
    private lateinit var speedSrc: TextView
    private lateinit var compass: CompassView
    private lateinit var hdgVal: TextView
    private lateinit var driftLbl: TextView
    private lateinit var driftVal: TextView
    private lateinit var driftSub: TextView
    private lateinit var chartBox: LinearLayout
    private lateinit var chart: DriftChart
    private lateinit var statCal: TextView
    private lateinit var statRoad: TextView
    private lateinit var statBumps: TextView
    private lateinit var statMount: TextView
    private lateinit var note: TextView
    private lateinit var startBtn: TextView
    private lateinit var outageBtn: TextView
    private lateinit var snapBtn: TextView
    private lateinit var vehicleBtn: TextView
    private var recorder: DriveRecorder? = null
    private lateinit var followBtn: ImageView

    // outage bookkeeping for the readout (UI only; the engine owns the filter)
    private var wasOutage = false
    private var wasMasked = false
    private var lastGE = Double.NaN
    private var lastGN = Double.NaN
    private var outStartMs = 0L
    private var lastSnapMs = 0L
    private var outDistM = 0.0
    private var lastOutage: String? = null

    private lateinit var fusionThread: HandlerThread
    private lateinit var fusionHandler: Handler

    private var running = false
    // latest gyro, emitted together with each accelerometer sample (the fastest stream)
    private val gyro = FloatArray(3)
    private val magv = FloatArray(3)
    @Volatile private var cn0Mean = 0.0
    @Volatile private var svUsed = 0
    @Volatile private var navicSv = 0

    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        MapLibre.getInstance(this)                       // offline: tiles come from the bundled mbtiles
        sm = getSystemService(SENSOR_SERVICE) as SensorManager
        lm = getSystemService(LOCATION_SERVICE) as LocationManager
        engine = FusionEngine(applicationContext) { s -> onNav(s) }

        // dark placeholder while the style loads (MapLibre's default is a light beige flash)
        val opts = MapLibreMapOptions.createFromAttributes(this).foregroundLoadColor(Ui.BG)
        mapView = MapView(this, opts).apply { onCreate(b) }
        navMap = NavMap(this, mapView)

        val root = FrameLayout(this).apply { setBackgroundColor(Ui.BG) }
        root.addView(mapView, FrameLayout.LayoutParams(MATCH, MATCH))
        root.addView(buildHeader(), FrameLayout.LayoutParams(MATCH, WRAP, Gravity.TOP).apply {
            val m = dp(12f); setMargins(m, m, m, 0)
        })
        followBtn = ImageView(this).apply {
            setImageResource(com.offmaps.R.drawable.ic_my_location)
            val p = dp(12f); setPadding(p, p, p, p)
            background = Ui.rounded(this@NavActivity, Ui.SURFACE, 999f, Ui.STROKE)
            elevation = Ui.dp(this@NavActivity, 6f)
            contentDescription = "Re-centre on the car"
            setOnClickListener { navMap.follow() }
        }
        root.addView(followBtn, FrameLayout.LayoutParams(dp(48f), dp(48f), Gravity.BOTTOM or Gravity.END).apply {
            setMargins(0, 0, dp(16f), dp(16f))
        })
        sheet = buildSheet()
        root.addView(sheet, FrameLayout.LayoutParams(MATCH, WRAP, Gravity.BOTTOM))
        // keep the re-centre button, the OSM attribution and the followed car above the sheet
        sheet.addOnLayoutChangeListener { _, _, top, _, bottom, _, oldTop, _, oldBottom ->
            val h = bottom - top
            if (h == oldBottom - oldTop) return@addOnLayoutChangeListener
            followBtn.post {                              // not during this layout pass
                (followBtn.layoutParams as FrameLayout.LayoutParams).bottomMargin = h + dp(12f)
                followBtn.requestLayout()
                navMap.setInsets(dp(150f), h)
            }
        }

        startBtn.setOnClickListener {
            if (running) stopNav() else if (ensurePerms()) startNav()
        }
        outageBtn.setOnClickListener {
            if (!running) return@setOnClickListener
            engine.setMasked(!engine.isMasked()); refreshButtons()
        }
        snapBtn.setOnClickListener {
            engine.setMapAid(!engine.isMapAid()); refreshButtons()
        }
        vehicleBtn.setOnClickListener {                   // two-wheeler: lean-compensated yaw rate
            engine.setVehicle(if (engine.getVehicle() == "car") "two_wheeler" else "car"); refreshButtons()
        }
        refreshButtons()
        setMode("STANDBY", Ui.MUTED)
        setContentView(root)
        loadOfflineMap()
    }

    // ------------------------------------------------------------------ layout

    private fun buildHeader(): View {
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            val p = dp(14f); setPadding(p, dp(12f), p, p)
            background = Ui.rounded(this@NavActivity, Ui.SURFACE, 22f, Ui.STROKE)
            elevation = Ui.dp(this@NavActivity, 8f)
        }
        val top = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        val brand = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        val title = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        title.addView(Ui.text(this, "Off", 21f, Ui.TEXT, Ui.BLACK))
        title.addView(Ui.text(this, "Maps", 21f, Ui.ACCENT, Ui.BLACK))
        brand.addView(title)
        brand.addView(Ui.text(this, "Navigation that keeps going without GNSS", 11f, Ui.MUTED).apply {
            setPadding(0, dp(3f), 0, 0)
        })
        top.addView(brand, LinearLayout.LayoutParams(0, WRAP, 1f))
        modePill = Ui.Pill(this)
        top.addView(modePill)
        card.addView(top)

        val chips = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        chipSat = Ui.Chip(this, "SATELLITES"); chipNavic = Ui.Chip(this, "NAVIC")
        chipCn0 = Ui.Chip(this, "C/N0 dB-Hz"); chipTrust = Ui.Chip(this, "GNSS TRUST")
        listOf(chipSat, chipNavic, chipCn0, chipTrust).forEachIndexed { i, c ->
            chips.addView(c, LinearLayout.LayoutParams(0, WRAP, 1f).apply { if (i > 0) marginStart = dp(6f) })
        }
        card.addView(chips, LinearLayout.LayoutParams(MATCH, WRAP).apply { topMargin = dp(12f) })
        return card
    }

    private fun buildSheet(): LinearLayout {
        val sh = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            val p = dp(18f); setPadding(p, dp(10f), p, dp(16f))
            background = Ui.sheet(this@NavActivity, Ui.SHEET, 28f)
            elevation = Ui.dp(this@NavActivity, 12f)
            isClickable = true                                   // don't pass touches to the map
        }
        sh.addView(View(this).apply { background = Ui.rounded(this@NavActivity, Ui.STROKE, 999f) },
            LinearLayout.LayoutParams(dp(40f), dp(4f)).apply { gravity = Gravity.CENTER_HORIZONTAL; bottomMargin = dp(14f) })

        // --- idle: system check ---
        readyPanel = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        readyPanel.addView(Ui.label(this, "SYSTEM CHECK"))
        rowMap = readyRow(); rowRoads = readyRow(); rowModel = readyRow(); rowSensors = readyRow()
        listOf(rowMap, rowRoads, rowModel, rowSensors).forEach { readyPanel.addView(it) }
        setRow(rowMap, null, "Offline map", "loading…")
        setRow(rowRoads, null, "Road network", "loading…")
        setRow(rowModel, null, "AI speed model", "loading…")
        val hasImu = sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER) != null && sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE) != null
        setRow(rowSensors, hasImu, "Inertial sensors", if (hasImu) "accelerometer + gyroscope" else "missing")
        sh.addView(readyPanel)

        // --- live readout ---
        livePanel = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; visibility = View.GONE }
        val big = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }

        val speedCol = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        speedCol.addView(Ui.label(this, "SPEED"))
        val speedRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.BOTTOM }
        speedVal = Ui.text(this, "0", 44f, Ui.TEXT, Ui.BLACK)
        speedRow.addView(speedVal)
        speedRow.addView(Ui.text(this, " km/h", 13f, Ui.MUTED, Ui.BOLD).apply { setPadding(0, 0, 0, dp(6f)) })
        speedCol.addView(speedRow, LinearLayout.LayoutParams(WRAP, WRAP).apply { topMargin = dp(4f) })
        speedSrc = Ui.text(this, "", 11f, Ui.MUTED)
        speedCol.addView(speedSrc, LinearLayout.LayoutParams(WRAP, WRAP).apply { topMargin = dp(4f) })
        big.addView(speedCol, LinearLayout.LayoutParams(0, WRAP, 1.1f))

        val hdgCol = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; gravity = Gravity.CENTER_HORIZONTAL }
        compass = CompassView(this)
        hdgCol.addView(compass, LinearLayout.LayoutParams(dp(66f), dp(66f)))
        hdgVal = Ui.text(this, "–", 12f, Ui.TEXT, Ui.MONO)
        hdgCol.addView(hdgVal, LinearLayout.LayoutParams(WRAP, WRAP).apply { topMargin = dp(6f) })
        big.addView(hdgCol, LinearLayout.LayoutParams(0, WRAP, 0.8f))

        val driftCol = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; gravity = Gravity.END }
        driftLbl = Ui.label(this, "FIX AGREEMENT")
        driftVal = Ui.text(this, "–", 28f, Ui.TEXT, Ui.CONDENSED)
        driftSub = Ui.text(this, "", 11f, Ui.MUTED).apply { gravity = Gravity.END }
        driftCol.addView(driftLbl)
        driftCol.addView(driftVal, LinearLayout.LayoutParams(WRAP, WRAP).apply { topMargin = dp(6f) })
        driftCol.addView(driftSub, LinearLayout.LayoutParams(WRAP, WRAP).apply { topMargin = dp(4f) })
        big.addView(driftCol, LinearLayout.LayoutParams(0, WRAP, 1.1f))
        livePanel.addView(big)

        chartBox = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL; visibility = View.GONE
            val p = dp(12f); setPadding(p, dp(10f), p, dp(8f))
            background = Ui.rounded(this@NavActivity, Ui.SURFACE_2, 16f)
        }
        chartBox.addView(Ui.label(this, "DRIFT DURING OUTAGE  vs  ISRO LIMIT"))
        chart = DriftChart(this)
        chartBox.addView(chart, LinearLayout.LayoutParams(MATCH, dp(78f)).apply { topMargin = dp(6f) })
        livePanel.addView(chartBox, LinearLayout.LayoutParams(MATCH, WRAP).apply { topMargin = dp(14f) })

        val stats = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        fun stat(lbl: String): TextView {
            val col = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                val p = dp(10f); setPadding(p, dp(8f), p, dp(8f))
                background = Ui.rounded(this@NavActivity, Ui.SURFACE_2, 12f)
            }
            val v = Ui.text(this, "–", 12f, Ui.TEXT, Ui.BOLD)
            col.addView(Ui.label(this, lbl).apply { textSize = 9f })
            col.addView(v, LinearLayout.LayoutParams(WRAP, WRAP).apply { topMargin = dp(3f) })
            stats.addView(col, LinearLayout.LayoutParams(0, WRAP, 1f).apply { if (stats.childCount > 0) marginStart = dp(6f) })
            return v
        }
        statRoad = stat("ROAD MATCH"); statCal = stat("SELF-CAL"); statBumps = stat("POTHOLES"); statMount = stat("MOUNT")
        livePanel.addView(stats, LinearLayout.LayoutParams(MATCH, WRAP).apply { topMargin = dp(12f) })
        sh.addView(livePanel)

        // legend
        val legend = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER }
        fun key(color: Int, s: String) {
            legend.addView(View(this).apply { background = Ui.rounded(this@NavActivity, color, 999f) },
                LinearLayout.LayoutParams(dp(14f), dp(4f)).apply { marginStart = if (legend.childCount > 0) dp(12f) else 0; marginEnd = dp(5f) })
            legend.addView(Ui.text(this, s, 10.5f, Ui.MUTED))
        }
        key(Ui.GNSS, "GNSS"); key(Ui.FUSED, "Fused"); key(Ui.DR, "Dead-reckoning"); key(Ui.SHOCK, "Pothole")
        sh.addView(legend, LinearLayout.LayoutParams(MATCH, WRAP).apply { topMargin = dp(14f) })

        note = Ui.text(this, "", 11f, Ui.DANGER).apply { gravity = Gravity.CENTER; visibility = View.GONE }
        sh.addView(note, LinearLayout.LayoutParams(MATCH, WRAP).apply { topMargin = dp(8f) })

        // controls
        startBtn = Ui.button(this, "Start navigation", true)
        sh.addView(startBtn, LinearLayout.LayoutParams(MATCH, WRAP).apply { topMargin = dp(14f) })
        val toggles = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        outageBtn = Ui.button(this, "", false).apply { textSize = 13f; minHeight = dp(44f) }
        snapBtn = Ui.button(this, "", false).apply { textSize = 13f; minHeight = dp(44f) }
        toggles.addView(outageBtn, LinearLayout.LayoutParams(0, WRAP, 1.25f))
        toggles.addView(snapBtn, LinearLayout.LayoutParams(0, WRAP, 1f).apply { marginStart = dp(8f) })
        vehicleBtn = Ui.button(this, "", false).apply { textSize = 13f; minHeight = dp(44f) }
        toggles.addView(vehicleBtn, LinearLayout.LayoutParams(0, WRAP, 0.9f).apply { marginStart = dp(8f) })
        sh.addView(toggles, LinearLayout.LayoutParams(MATCH, WRAP).apply { topMargin = dp(8f) })
        return sh
    }

    private fun readyRow() = Ui.text(this, "", 13f, Ui.TEXT).apply { setPadding(0, dp(9f), 0, 0) }

    /** ok: true = check, false = cross, null = pending. */
    private fun setRow(row: TextView, ok: Boolean?, what: String, detail: String) {
        val (mark, col) = when (ok) { true -> "✓" to Ui.OK; false -> "✕" to Ui.DANGER; null -> "•" to Ui.MUTED }
        row.text = android.text.SpannableStringBuilder().apply {
            append(mark, android.text.style.ForegroundColorSpan(col), 0)
            append("   $what   ")
            append(detail, android.text.style.ForegroundColorSpan(Ui.MUTED), 0)
        }
    }

    private fun refreshButtons() {
        startBtn.text = if (running) "Stop navigation" else "Start navigation"
        Ui.style(startBtn, !running, if (running) Ui.DANGER else Ui.ACCENT)
        val masked = running && engine.isMasked()
        outageBtn.text = if (masked) "Restore GNSS" else "Simulate GNSS outage"
        Ui.style(outageBtn, masked, Ui.DR)
        outageBtn.alpha = if (running) 1f else 0.4f
        val snap = engine.isMapAid()
        snapBtn.text = if (snap) "Road snap: ON" else "Road snap: OFF"
        Ui.style(snapBtn, false, if (snap) Ui.ACCENT else Ui.MUTED)
        vehicleBtn.text = if (engine.getVehicle() == "car") "Car" else "Two-wheeler"
        Ui.style(vehicleBtn, false, Ui.ACCENT)
    }

    /** Header pill. Dead-reckoning pulses so the mode change is unmissable. */
    private fun setMode(s: String, color: Int, pulsing: Boolean = false) {
        val key = "$s|$pulsing"
        if (key == pillState) return
        pillState = key
        modePill.set(s, color)
        pulse?.cancel(); modePill.alpha = 1f
        if (pulsing) pulse = ObjectAnimator.ofFloat(modePill, View.ALPHA, 1f, 0.45f).apply {
            duration = 700; repeatMode = ValueAnimator.REVERSE; repeatCount = ValueAnimator.INFINITE; start()
        }
    }

    private fun dp(v: Float) = Ui.dpi(this, v)

    /** Copy the tiles out + index the road network off the UI thread, then show the map. */
    private fun loadOfflineMap() {
        Thread {
            val tiles = NavMap.installTiles(applicationContext)
            val roads = RoadNetwork.fromAssets(applicationContext)
            val model = try { SpeedProfile.fromAssets(applicationContext).name } catch (e: Exception) { null }
            engine.setRoads(roads)
            runOnUiThread {
                if (isDestroyed) return@runOnUiThread
                navMap.load(tiles)
                setRow(rowMap, tiles != null, "Offline map",
                    if (tiles != null) "Hyderabad · no network needed" else "not bundled (tools/build_map.sh)")
                setRow(rowRoads, roads != null, "Road network",
                    if (roads != null) String.format(Locale.US, "%,d road pieces for snapping", roads.nWays) else "no roads.bin")
                setRow(rowModel, model != null, "AI speed model", model?.let { "SpeedNet · $it · on-device" } ?: "profile missing")
            }
        }.start()
    }

    private fun ensurePerms(): Boolean {
        val need = arrayOf(Manifest.permission.ACCESS_FINE_LOCATION)
        if (need.any { ActivityCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED }) {
            ActivityCompat.requestPermissions(this, need, 1); return false
        }
        return true
    }

    override fun onRequestPermissionsResult(rc: Int, p: Array<out String>, r: IntArray) {
        super.onRequestPermissionsResult(rc, p, r)
        if (rc == 1 && r.isNotEmpty() && r[0] == PackageManager.PERMISSION_GRANTED) startNav()
    }

    private fun startNav() {
        if (running) return
        fusionThread = HandlerThread("fusion").apply { start() }
        fusionHandler = Handler(fusionThread.looper)
        navMap.clear()
        engine.start()
        running = true
        wasOutage = false; lastOutage = null; chart.clear(); chartBox.visibility = View.GONE
        readyPanel.visibility = View.GONE; livePanel.visibility = View.VISIBLE
        compass.reset(); note.visibility = View.GONE
        refreshButtons()
        setMode("ACQUIRING GNSS", Ui.ACCENT, pulsing = true)

        recorder = try { DriveRecorder(getExternalFilesDir(null) ?: filesDir) } catch (e: Exception) { null }
        listOf(Sensor.TYPE_ACCELEROMETER, Sensor.TYPE_GYROSCOPE, Sensor.TYPE_MAGNETIC_FIELD).forEach { type ->
            sm.getDefaultSensor(type)?.let {
                sm.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST, fusionHandler)
            }
        }
        try {
            lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, 0L, 0f, this, fusionThread.looper)
            lm.registerGnssStatusCallback(gnssStatus, fusionHandler)
        } catch (e: SecurityException) {
            note.text = "Location permission missing"; note.visibility = View.VISIBLE
        }
    }

    private fun stopNav() {
        running = false
        sm.unregisterListener(this)
        lm.removeUpdates(this)
        lm.unregisterGnssStatusCallback(gnssStatus)
        engine.stop()
        if (::fusionThread.isInitialized) {
            val r = recorder; recorder = null
            fusionHandler.post { r?.close() }              // the recorder belongs to the fusion thread
            fusionThread.quitSafely()
        }
        engine.setMasked(false)
        readyPanel.visibility = View.VISIBLE; livePanel.visibility = View.GONE
        for (c in listOf(chipSat, chipNavic, chipCn0, chipTrust)) { c.value.text = "–"; c.value.setTextColor(Ui.TEXT) }
        refreshButtons()
        setMode("STANDBY", Ui.MUTED)
    }

    // --- IMU: cache gyro, feed engine on each accel sample (event.timestamp = elapsedRealtimeNanos) ---
    override fun onSensorChanged(e: SensorEvent) {
        if (!running) return
        when (e.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> System.arraycopy(e.values, 0, gyro, 0, 3)
            Sensor.TYPE_MAGNETIC_FIELD -> {
                System.arraycopy(e.values, 0, magv, 0, 3)
                engine.onMag(e.timestamp, e.values[0].toDouble(), e.values[1].toDouble(), e.values[2].toDouble())
            }
            Sensor.TYPE_ACCELEROMETER -> {
                recorder?.imu(e.timestamp, e.values, gyro, magv)
                engine.onImu(
                    e.timestamp,
                    e.values[0].toDouble(), e.values[1].toDouble(), e.values[2].toDouble(),
                    gyro[0].toDouble(), gyro[1].toDouble(), gyro[2].toDouble())
            }
        }
    }
    override fun onAccuracyChanged(s: Sensor?, acc: Int) {}

    override fun onLocationChanged(loc: Location) {
        if (!running) return
        recorder?.gnss(SystemClock.elapsedRealtimeNanos(), loc.latitude, loc.longitude,
            if (loc.hasSpeed()) loc.speed.toDouble() else Double.NaN,
            if (loc.hasBearing()) loc.bearing.toDouble() else Double.NaN,
            cn0Mean, svUsed, navicSv, engine.isMasked())
        engine.onGnss(
            loc.latitude, loc.longitude,
            if (loc.hasSpeed()) loc.speed.toDouble() else Double.NaN,
            if (loc.hasBearing()) loc.bearing.toDouble() else Double.NaN,
            cn0Mean, svUsed, navicSv)
    }

    private val gnssStatus = object : GnssStatus.Callback() {
        override fun onSatelliteStatusChanged(s: GnssStatus) {
            var used = 0; var navic = 0; var cn0 = 0.0
            for (i in 0 until s.satelliteCount) {
                if (s.usedInFix(i)) { used++; cn0 += s.getCn0DbHz(i) }
                if (s.getConstellationType(i) == GnssStatus.CONSTELLATION_IRNSS) navic++  // NavIC
            }
            svUsed = used; navicSv = navic; cn0Mean = if (used > 0) cn0 / used else 0.0
        }
    }

    /** Engine snapshot, on the main thread. Append to the map and refresh the readout. */
    private fun onNav(s: NavState) {
        if (!running) return
        navMap.update(s)
        val now = SystemClock.elapsedRealtime()

        // outage bookkeeping: elapsed time + distance travelled since GNSS was lost.
        // A simulated outage restarts the clock even if we were already dead-reckoning
        // (weak GNSS), and measures distance on the GNSS truth it still receives -- the
        // ISRO metric is drift as a % of the distance actually driven.
        val simStart = s.masked && !wasMasked
        if ((s.outageActive && !wasOutage) || simStart) {
            outStartMs = now; outDistM = 0.0; lastGE = Double.NaN; lastGN = Double.NaN
            if (s.masked) { chart.clear(); chartBox.visibility = View.VISIBLE }
        }
        if (s.masked) {
            if (s.newGnss && s.hasGnss) {
                if (lastGE.isFinite()) outDistM += Math.hypot(s.gnssE - lastGE, s.gnssN - lastGN)
                lastGE = s.gnssE; lastGN = s.gnssN
            }
        } else if (s.outageActive && wasOutage) {
            outDistM += s.v * (now - lastSnapMs) / 1000.0          // no truth: our own estimate
        }
        val outS = (now - outStartMs) / 1000.0
        if (!s.masked && wasMasked && !chart.isEmpty) {
            lastOutage = String.format(Locale.US, "last outage %s · %s", clock(outS), pctText(pct(chartLastDrift, outDistM)))
        }
        wasOutage = s.outageActive; wasMasked = s.masked; lastSnapMs = now

        // header: mode + satellite picture
        when {
            s.spoof -> setMode("SPOOF REJECTED", Ui.DANGER, pulsing = true)
            s.outageActive -> setMode("DEAD-RECKONING", Ui.DR, pulsing = true)
            else -> setMode("GNSS LOCK", Ui.OK)
        }
        chipSat.value.text = s.svUsed.toString()
        chipNavic.value.text = s.navicSv.toString()
        chipNavic.value.setTextColor(if (s.navicSv > 0) Ui.ACCENT else Ui.TEXT)
        chipCn0.value.text = if (s.cn0 > 0) String.format(Locale.US, "%.0f", s.cn0) else "–"
        chipTrust.value.text = String.format(Locale.US, "%.0f %%", 100 * s.trust.coerceIn(0.0, 1.0))
        chipTrust.value.setTextColor(when {
            s.masked -> Ui.MUTED; s.trust >= 0.5 -> Ui.OK; s.trust >= 0.1 -> Ui.DR; else -> Ui.DANGER })

        // speed + heading
        speedVal.text = String.format(Locale.US, "%.0f", s.v * 3.6)
        speedSrc.text = if (s.outageActive) "IMU + AI speed model" else "GNSS-aided fusion"
        val modeCol = if (s.outageActive) Ui.DR else Ui.ACCENT
        compass.set(s.psi, modeCol)
        val hdg = ((Math.toDegrees(s.psi) % 360) + 360) % 360
        hdgVal.text = String.format(Locale.US, "%03.0f° %s", hdg, cardinal(hdg))

        // right block: drift vs truth during a simulated outage, else fix agreement
        when {
            s.outageActive && s.masked -> {
                chart.add(outS, s.driftM, outDistM); chartLastDrift = s.driftM
                val p = pct(s.driftM, outDistM)
                driftLbl.text = "DRIFT vs GNSS"
                driftVal.text = String.format(Locale.US, "%.1f m", s.driftM)
                driftVal.setTextColor(if (outDistM > 1.0 && p <= 10.0) Ui.OK else Ui.DR)
                driftSub.text = String.format(Locale.US, "%s of %.0f m · %s", pctText(p), outDistM, clock(outS))
            }
            s.outageActive -> {
                driftLbl.text = "NO GNSS"
                driftVal.text = clock(outS); driftVal.setTextColor(Ui.DR)
                driftSub.text = String.format(Locale.US, "%.0f m dead-reckoned", outDistM)
            }
            else -> {
                driftLbl.text = "FIX AGREEMENT"
                driftVal.text = String.format(Locale.US, "%.1f m", s.driftM); driftVal.setTextColor(Ui.TEXT)
                driftSub.text = lastOutage ?: "fused vs raw GNSS"
            }
        }

        // stats
        statRoad.text = when {
            !s.roadsLoaded -> "no data"
            !engine.isMapAid() -> "off"
            !s.outageActive -> "standby"
            s.snapped -> "snapped"
            else -> "searching"
        }
        statRoad.setTextColor(if (s.outageActive && s.snapped && engine.isMapAid()) Ui.OK else Ui.TEXT)
        statCal.text = if (s.calUsed) String.format(Locale.US, "k %.2f", s.k) else "learning"
        statBumps.text = s.shocks.toString()
        statBumps.setTextColor(if (s.shocks > 0) Ui.SHOCK else Ui.TEXT)
        statMount.text = if (s.alignChanged) "re-mount" else "stable"
        statMount.setTextColor(if (s.alignChanged) Ui.DR else Ui.TEXT)
    }

    private var chartLastDrift = 0.0

    private fun pct(driftM: Double, distM: Double) = if (distM > 1.0) 100.0 * driftM / distM else 0.0

    private fun pctText(p: Double) =
        if (p > 999.0) "> 999 %" else String.format(Locale.US, "%.1f %%", p)

    private fun clock(sec: Double): String {
        val t = sec.toLong().coerceAtLeast(0)
        return String.format(Locale.US, "%d:%02d", t / 60, t % 60)
    }

    private fun cardinal(deg: Double) =
        arrayOf("N", "NE", "E", "SE", "S", "SW", "W", "NW")[(((deg + 22.5) % 360) / 45).toInt()]

    // MapView needs every lifecycle callback forwarded.
    override fun onStart() { super.onStart(); mapView.onStart() }
    override fun onResume() { super.onResume(); mapView.onResume() }
    override fun onPause() { mapView.onPause(); super.onPause() }
    override fun onStop() { mapView.onStop(); super.onStop() }
    override fun onLowMemory() { super.onLowMemory(); mapView.onLowMemory() }
    override fun onSaveInstanceState(out: Bundle) { super.onSaveInstanceState(out); mapView.onSaveInstanceState(out) }

    override fun onDestroy() { if (running) stopNav(); pulse?.cancel(); mapView.onDestroy(); super.onDestroy() }

    private companion object {
        const val MATCH = ViewGroup.LayoutParams.MATCH_PARENT
        const val WRAP = ViewGroup.LayoutParams.WRAP_CONTENT
    }
}
