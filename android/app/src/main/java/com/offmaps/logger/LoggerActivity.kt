package com.offmaps.logger

import android.Manifest
import android.content.pm.PackageManager
import android.hardware.*
import android.location.*
import android.os.*
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Phase-1 sensor logger. Records raw IMU + GNSS to CSV so a drive can be
 * replayed offline through the Python eval harness (see py/data/phone_log.py --
 * the column order below IS that contract).
 *
 * The "Outage Simulator" button is the demo-winning control: it never stops
 * logging GNSS (we still need it as ground truth), it only stamps masked=1 on
 * GNSS rows. Replay treats masked segments as outages; the Phase-6 nav app
 * (com.offmaps.nav.NavActivity) now honours the same flag against the LIVE
 * filter -- while masked it feeds the embedded ESKF no GNSS, so it dead-reckons.
 */
class LoggerActivity : AppCompatActivity(), SensorEventListener, LocationListener {

    private lateinit var sm: SensorManager
    private lateinit var lm: LocationManager
    private var imu: CsvWriter? = null
    private var gnss: CsvWriter? = null
    private var meas: CsvWriter? = null
    private val logging = AtomicBoolean(false)
    private val masked = AtomicBoolean(false)

    // latest sensor values, written out on each accelerometer sample (fastest stream)
    private val a = FloatArray(3); private val g = FloatArray(3); private val m = FloatArray(3)
    @Volatile private var pressure = 0f
    @Volatile private var light = 0f
    // GNSS-quality snapshot maintained by the status callback
    @Volatile private var cn0Mean = 0.0
    @Volatile private var svUsed = 0
    @Volatile private var navicSv = 0

    private lateinit var status: TextView

    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        sm = getSystemService(SENSOR_SERVICE) as SensorManager
        lm = getSystemService(LOCATION_SERVICE) as LocationManager

        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(48, 96, 48, 48) }
        status = TextView(this).apply { textSize = 16f; text = "idle" }
        val start = Button(this).apply { text = "Start logging" }
        val outage = Button(this).apply { text = "Outage Simulator: OFF" }
        start.setOnClickListener {
            if (logging.get()) { stop(); start.text = "Start logging" }
            else if (ensurePerms()) { begin(); start.text = "Stop logging" }
        }
        outage.setOnClickListener {
            val on = !masked.get(); masked.set(on)
            outage.text = "Outage Simulator: " + if (on) "ON (GNSS masked)" else "OFF"
        }
        root.addView(start); root.addView(outage); root.addView(status)
        setContentView(root)
    }

    private fun ensurePerms(): Boolean {
        val need = arrayOf(Manifest.permission.ACCESS_FINE_LOCATION)
        if (need.any { ActivityCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED }) {
            ActivityCompat.requestPermissions(this, need, 1); return false
        }
        return true
    }

    private fun begin() {
        val dir = File(getExternalFilesDir(null), "drive_" + System.currentTimeMillis()).apply { mkdirs() }
        imu = CsvWriter(File(dir, "imu.csv"), "t_ns,ax,ay,az,gx,gy,gz,mx,my,mz,pressure,light")
        gnss = CsvWriter(File(dir, "gnss.csv"), "t_ns,lat,lon,speed,bearing,cn0_mean,sv_used,navic_sv,masked")
        meas = CsvWriter(File(dir, "meas.csv"), "t_ns,svid,constellation,prr_mps,cn0")  // raw, for Phase 4
        logging.set(true)

        // FASTEST so the high-rate branch (Phase 3 ESKF/ZUPT) has real 100-400 Hz data
        listOf(Sensor.TYPE_ACCELEROMETER, Sensor.TYPE_GYROSCOPE, Sensor.TYPE_MAGNETIC_FIELD,
               Sensor.TYPE_PRESSURE, Sensor.TYPE_LIGHT).forEach { type ->
            sm.getDefaultSensor(type)?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        }
        try {
            lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, 0L, 0f, this)
            lm.registerGnssStatusCallback(gnssStatus, Handler(Looper.getMainLooper()))
            lm.registerGnssMeasurementsCallback(gnssMeas, Handler(Looper.getMainLooper()))
        } catch (e: SecurityException) { toast("location permission missing") }
        status.text = "logging -> $dir"
    }

    private fun stop() {
        logging.set(false)
        sm.unregisterListener(this)
        lm.removeUpdates(this)
        lm.unregisterGnssStatusCallback(gnssStatus)
        lm.unregisterGnssMeasurementsCallback(gnssMeas)
        listOf(imu, gnss, meas).forEach { it?.close() }
        status.text = "stopped"
    }

    // --- sensors: cache latest, emit an aligned row on each accel sample ---
    override fun onSensorChanged(e: SensorEvent) {
        if (!logging.get()) return
        when (e.sensor.type) {
            Sensor.TYPE_MAGNETIC_FIELD -> System.arraycopy(e.values, 0, m, 0, 3)
            Sensor.TYPE_GYROSCOPE -> System.arraycopy(e.values, 0, g, 0, 3)
            Sensor.TYPE_PRESSURE -> pressure = e.values[0]
            Sensor.TYPE_LIGHT -> light = e.values[0]
            Sensor.TYPE_ACCELEROMETER -> {
                System.arraycopy(e.values, 0, a, 0, 3)
                imu?.row(e.timestamp, a[0], a[1], a[2], g[0], g[1], g[2],
                         m[0], m[1], m[2], pressure, light)   // event.timestamp is elapsedRealtimeNanos
            }
        }
    }
    override fun onAccuracyChanged(s: Sensor?, acc: Int) {}

    // --- location: lat/lon + Doppler speed/bearing from the GNSS chip ---
    override fun onLocationChanged(loc: Location) {
        if (!logging.get()) return
        gnss?.row(SystemClock.elapsedRealtimeNanos(), loc.latitude, loc.longitude,
                  if (loc.hasSpeed()) loc.speed else Float.NaN,
                  if (loc.hasBearing()) loc.bearing else Float.NaN,
                  cn0Mean, svUsed, navicSv, if (masked.get()) 1 else 0)
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

    private val gnssMeas = object : GnssMeasurementsEvent.Callback() {
        override fun onGnssMeasurementsReceived(ev: GnssMeasurementsEvent) {
            if (!logging.get()) return
            val t = SystemClock.elapsedRealtimeNanos()
            for (mm in ev.measurements)
                meas?.row(t, mm.svid, mm.constellationType, mm.pseudorangeRateMetersPerSecond, mm.cn0DbHz)
        }
    }

    private fun toast(s: String) = Toast.makeText(this, s, Toast.LENGTH_SHORT).show()
    override fun onDestroy() { if (logging.get()) stop(); super.onDestroy() }
}

/** Buffered CSV writer; row() joins any args with commas. Not thread-safe by design:
 *  GNSS + IMU callbacks arrive on the main looper thread, so writes are serialized. */
class CsvWriter(file: File, header: String) {
    private val w = file.bufferedWriter().apply { write(header); newLine() }
    fun row(vararg v: Any?) { w.write(v.joinToString(",")); w.newLine() }
    fun close() { w.flush(); w.close() }
}
