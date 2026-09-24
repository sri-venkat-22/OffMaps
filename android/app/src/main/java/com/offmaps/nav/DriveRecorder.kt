package com.offmaps.nav

import java.io.BufferedWriter
import java.io.File

/**
 * Records every navigation session as a raw drive in the Phase-1 logger schema
 * (data/phone_log.py): imu.csv (one row per accelerometer sample, with the latest
 * gyro + magnetometer) and gnss.csv (one row per fix; masked = the Outage Simulator
 * was on). `py/score_drive.py --dir <drive_ms>` replays it through the same loop on
 * the host and scores drift, stops/ZUPT and sensor sanity -- the phone-side numbers
 * that no IO-VNBD result can give. Files live in the app's external files dir:
 *   adb pull /sdcard/Android/data/com.offmaps/files/drive_<ms>
 * Fusion thread only (every sensor/location callback is delivered there).
 */
class DriveRecorder(root: File) : AutoCloseable {
    val dir: File = File(root, "drive_" + System.currentTimeMillis()).apply { mkdirs() }
    private val imu: BufferedWriter = File(dir, "imu.csv").bufferedWriter().apply {
        write("t_ns,ax,ay,az,gx,gy,gz,mx,my,mz"); newLine()
    }
    private val gnss: BufferedWriter = File(dir, "gnss.csv").bufferedWriter().apply {
        write("t_ns,lat,lon,speed,bearing,cn0_mean,sv_used,navic_sv,masked"); newLine(); flush()
    }
    var imuRows = 0L; private set
    var gnssRows = 0L; private set

    fun imu(tNs: Long, a: FloatArray, g: FloatArray, m: FloatArray) {
        imu.write("$tNs,${a[0]},${a[1]},${a[2]},${g[0]},${g[1]},${g[2]},${m[0]},${m[1]},${m[2]}"); imu.newLine()
        if (++imuRows % FLUSH_IMU_ROWS == 0L) imu.flush()
    }

    fun gnss(tNs: Long, lat: Double, lon: Double, speed: Double, bearing: Double,
             cn0: Double, sv: Int, navic: Int, masked: Boolean) {
        gnss.write("$tNs,$lat,$lon,$speed,$bearing,$cn0,$sv,$navic,${if (masked) 1 else 0}"); gnss.newLine()
        gnssRows++
        gnss.flush()                             // 1 Hz: a killed app still leaves a scorable drive
    }

    override fun close() { imu.flush(); imu.close(); gnss.flush(); gnss.close() }

    companion object { const val FLUSH_IMU_ROWS = 500L }   // ~1-5 s at phone IMU rates
}
