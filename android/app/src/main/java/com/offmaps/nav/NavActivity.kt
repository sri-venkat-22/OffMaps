package com.offmaps.nav

import android.Manifest
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
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.FrameLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import org.maplibre.android.MapLibre
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
 */
class NavActivity : AppCompatActivity(), SensorEventListener, LocationListener {

    private lateinit var sm: SensorManager
    private lateinit var lm: LocationManager
    private lateinit var engine: FusionEngine
    private lateinit var status: TextView
    private lateinit var startBtn: Button
    private lateinit var outageBtn: Button
    private lateinit var snapBtn: Button
    private lateinit var mapView: MapView
    private lateinit var navMap: NavMap

    private lateinit var fusionThread: HandlerThread
    private lateinit var fusionHandler: Handler

    private var running = false
    // latest gyro, emitted together with each accelerometer sample (the fastest stream)
    private val gyro = FloatArray(3)
    @Volatile private var cn0Mean = 0.0
    @Volatile private var svUsed = 0
    @Volatile private var navicSv = 0

    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        MapLibre.getInstance(this)                       // offline: tiles come from the bundled mbtiles
        sm = getSystemService(SENSOR_SERVICE) as SensorManager
        lm = getSystemService(LOCATION_SERVICE) as LocationManager
        engine = FusionEngine(applicationContext) { s -> onNav(s) }

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL; setPadding(32, 64, 32, 32)
        }
        startBtn = Button(this).apply { text = "Start navigation" }
        outageBtn = Button(this).apply { text = "Outage sim: OFF"; isEnabled = false }
        snapBtn = Button(this).apply { text = "Road snapping: ON" }
        status = TextView(this).apply { textSize = 14f; text = "loading offline map…"; setPadding(0, 12, 0, 12) }
        mapView = MapView(this).apply { onCreate(b) }
        navMap = NavMap(this, mapView)
        val followBtn = Button(this).apply { text = "◎ Follow"; setOnClickListener { navMap.follow() } }
        val mapBox = FrameLayout(this).apply {
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f)
            addView(mapView)
            addView(followBtn, FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM or Gravity.END).apply { setMargins(0, 0, 16, 16) })
        }
        val hint = TextView(this).apply {
            textSize = 12f; gravity = Gravity.CENTER
            text = "grey = GNSS   blue = fused ESKF   orange = dead-reckoning (outage)"
        }
        val toggles = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            val w = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            addView(outageBtn, w); addView(snapBtn, LinearLayout.LayoutParams(w))
        }

        startBtn.setOnClickListener {
            if (running) stopNav() else if (ensurePerms()) startNav()
        }
        outageBtn.setOnClickListener {
            val on = !engine.isMasked(); engine.setMasked(on)
            outageBtn.text = "Outage sim: " + if (on) "ON" else "OFF"
        }
        snapBtn.setOnClickListener {
            val on = !engine.isMapAid(); engine.setMapAid(on)
            snapBtn.text = "Road snapping: " + if (on) "ON" else "OFF"
        }
        root.addView(startBtn); root.addView(toggles); root.addView(status)
        root.addView(mapBox); root.addView(hint)
        setContentView(root)
        loadOfflineMap()
    }

    /** Copy the tiles out + index the road network off the UI thread, then show the map. */
    private fun loadOfflineMap() {
        Thread {
            val tiles = NavMap.installTiles(applicationContext)
            val roads = RoadNetwork.fromAssets(applicationContext)
            engine.setRoads(roads)
            runOnUiThread {
                if (isDestroyed) return@runOnUiThread
                navMap.load(tiles)
                if (!running) status.text = when {
                    tiles == null -> "no offline map bundled (run tools/build_map.sh) — tracks only"
                    roads == null -> "offline map ready (no roads.bin: road snapping unavailable)"
                    else -> "offline map ready · ${roads.nWays} road pieces for snapping"
                }
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
        startBtn.text = "Stop navigation"; outageBtn.isEnabled = true

        listOf(Sensor.TYPE_ACCELEROMETER, Sensor.TYPE_GYROSCOPE).forEach { type ->
            sm.getDefaultSensor(type)?.let {
                sm.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST, fusionHandler)
            }
        }
        try {
            lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, 0L, 0f, this, fusionThread.looper)
            lm.registerGnssStatusCallback(gnssStatus, fusionHandler)
        } catch (e: SecurityException) {
            status.text = "location permission missing"
        }
        status.text = "navigating… waiting for first GNSS fix"
    }

    private fun stopNav() {
        running = false
        sm.unregisterListener(this)
        lm.removeUpdates(this)
        lm.unregisterGnssStatusCallback(gnssStatus)
        engine.stop()
        if (::fusionThread.isInitialized) fusionThread.quitSafely()
        startBtn.text = "Start navigation"; outageBtn.isEnabled = false
        outageBtn.text = "Outage sim: OFF"; engine.setMasked(false)
        status.text = "stopped"
    }

    // --- IMU: cache gyro, feed engine on each accel sample (event.timestamp = elapsedRealtimeNanos) ---
    override fun onSensorChanged(e: SensorEvent) {
        if (!running) return
        when (e.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> System.arraycopy(e.values, 0, gyro, 0, 3)
            Sensor.TYPE_ACCELEROMETER -> engine.onImu(
                e.timestamp,
                e.values[0].toDouble(), e.values[1].toDouble(), e.values[2].toDouble(),
                gyro[0].toDouble(), gyro[1].toDouble(), gyro[2].toDouble())
        }
    }
    override fun onAccuracyChanged(s: Sensor?, acc: Int) {}

    override fun onLocationChanged(loc: Location) {
        if (!running) return
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
        status.text = buildString {
            val hdg = ((Math.toDegrees(s.psi) % 360) + 360) % 360
            append(String.format(Locale.US, "pos  e=%.1f  n=%.1f m    speed=%.1f m/s    hdg=%.0f°\n",
                s.e, s.n, s.v, hdg))
            if (s.outageActive) {
                val road = when {
                    !s.roadsLoaded -> "no road data"
                    !engine.isMapAid() -> "road snapping off"
                    s.snapped -> "snapped to road"
                    else -> "no road within 25 m"
                }
                if (s.masked) append(String.format(Locale.US, "OUTAGE (dead-reckoning)   drift vs GNSS = %.1f m   %s\n", s.driftM, road))
                else append("NO GNSS FIX (dead-reckoning)   $road\n")
            } else {
                append(String.format(Locale.US, "GNSS trust=%.2f   err vs fix=%.1f m%s\n",
                    s.trust, s.driftM, if (s.spoof) "   ⚠ SPOOF" else ""))
            }
            append(String.format(Locale.US, "SV used=%d  (NavIC %d)  C/N0=%.0f dBHz\n", s.svUsed, s.navicSv, s.cn0))
            append(String.format(Locale.US, "Doppler self-cal  k=%.3f  c=%.2f%s   potholes/bumps: %d%s",
                s.k, s.c, if (s.calUsed) "" else " (not used)", s.shocks,
                if (s.alignChanged) "   ↻ re-mount detected" else ""))
        }
    }

    // MapView needs every lifecycle callback forwarded.
    override fun onStart() { super.onStart(); mapView.onStart() }
    override fun onResume() { super.onResume(); mapView.onResume() }
    override fun onPause() { mapView.onPause(); super.onPause() }
    override fun onStop() { mapView.onStop(); super.onStop() }
    override fun onLowMemory() { super.onLowMemory(); mapView.onLowMemory() }
    override fun onSaveInstanceState(out: Bundle) { super.onSaveInstanceState(out); mapView.onSaveInstanceState(out) }

    override fun onDestroy() { if (running) stopNav(); mapView.onDestroy(); super.onDestroy() }
}
