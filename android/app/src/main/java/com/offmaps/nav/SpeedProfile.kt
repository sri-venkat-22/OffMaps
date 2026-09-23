package com.offmaps.nav

import android.content.Context
import org.json.JSONObject

/**
 * Everything the phone needs to run one SpeedNet checkpoint the way the host
 * harness validated it: which ONNX graph, its speed calibration, and the fusion
 * settings tuned for it. Read from assets/<name>.profile.json, which
 * py/model/export.py --profile writes straight from the checkpoint
 * (core_bridge.ESKF_DEFAULT overridden by the checkpoint's eskf_cfg). Nothing
 * here is hand-copied, so the phone cannot drift from the checkpoint.
 *
 * The same file drives the host mirror (py/phase6_check.py), and
 * py/tests/test_phase6_shipped_model.py checks each profile against its checkpoint.
 */
data class SpeedProfile(
    val name: String,
    val model: String,          // ONNX asset beside the profile
    val calibA: Double, val calibB: Double, val calibS: Double,
    val arw: Double,            // gyro angle random walk (rad/sqrt s)
    val brw: Double,            // gyro bias random walk
    val srw: Double,            // speed random walk
    val zuptV: Double?,         // NN speed below this -> ZUPT + ZARU; null = never
    val curv: Boolean,          // curvature speed pseudo-measurement in turns
    val curvSigma: Double,
    val sigScale: Double,       // multiplier on the NN sigma fed to the speed update
    val handoverS: Double,      // first seconds of dead-reckoning: hold the Doppler speed (physics)
    val mmCrossSigma: Double,   // road cross-track sigma (m), open road
    val mmHeadingSigmaDeg: Double,  // road heading sigma (deg), open road
    val mmKeepSpeed: Boolean,   // road updates may not change speed (idr_set_map_keep_speed)
) {
    companion object {
        /**
         * The checkpoint the app runs: SpeedNet retrained on real IO-VNBD drives with
         * ESKF settings tuned on validation drives only (REALDATA.md). "nn" (the
         * synthetic-trained net and its settings) still ships, as a one-line fallback.
         */
        const val DEFAULT = "nn_real"

        fun fromAssets(context: Context, name: String = DEFAULT): SpeedProfile {
            val j = JSONObject(context.assets.open("$name.profile.json").bufferedReader().use { it.readText() })
            val c = j.getJSONObject("calib")
            val e = j.getJSONObject("eskf")
            return SpeedProfile(
                name = name,
                model = j.getString("model"),
                calibA = c.getDouble("a"), calibB = c.getDouble("b"), calibS = c.getDouble("s"),
                arw = e.getDouble("arw"), brw = e.getDouble("brw"), srw = e.getDouble("srw"),
                zuptV = if (e.isNull("zupt_v")) null else e.getDouble("zupt_v"),
                curv = e.getBoolean("curv"),
                curvSigma = e.getDouble("curv_sigma"),
                sigScale = e.getDouble("sig_scale"),
                handoverS = e.optDouble("handover_s", 0.0),
                mmCrossSigma = e.optDouble("mm_cross_sigma", 1.5),
                mmHeadingSigmaDeg = e.optDouble("mm_heading_sigma_deg", 3.0),
                mmKeepSpeed = e.optBoolean("mm_keep_speed", false),
            )
        }
    }
}
