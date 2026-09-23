package com.offmaps.nav

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer
import kotlin.math.exp

/**
 * On-device SpeedNet: the ~180K-param TCN exported to ONNX (py/model/export.py)
 * running under ONNX Runtime Mobile. Phase 2's stated deliverable was "run on
 * device via onnxruntime"; this is that path.
 *
 * One 1 s step at a time: feed the 2 s (20-sample, 10 Hz) IMU window ending now,
 * read the displacement head (mu = metres over 1 s = speed in m/s) and its
 * log-variance, then apply the checkpoint's calibration (from its SpeedProfile). Returns the same
 * (v, sigma) that py/model/nn_model.predict_steps produces -- verified equal on
 * host to <1e-6 m/s against the reference feature+ONNX path.
 */
class SpeedNet private constructor(modelBytes: ByteArray, private val profile: SpeedProfile) : AutoCloseable {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(modelBytes, OrtSession.SessionOptions())
    private val shape = longArrayOf(1L, Features.C.toLong(), Features.WIN.toLong())

    /** @return [v (m/s), sigma (m/s)] for the window; sigma is the fusable heteroscedastic uncertainty. */
    fun predict(acc: Array<DoubleArray>, gyro: Array<DoubleArray>): DoubleArray {
        val feat = Features.windowFeatures(acc, gyro)          // FloatArray(9*20), channel-major
        OnnxTensor.createTensor(env, FloatBuffer.wrap(feat), shape).use { input ->
            session.run(mapOf("imu" to input)).use { res ->
                val out = HashMap<String, Any>()
                for (e in res) out[e.key] = (e.value as OnnxTensor).value
                val mu = (out["mu"] as FloatArray)[0].toDouble()          // softplus displacement >= 0
                val logvar = (out["logvar"] as FloatArray)[0].toDouble()
                val sigmaRaw = exp(0.5 * logvar)
                return Features.applyCalib(mu, sigmaRaw, profile.calibA, profile.calibB, profile.calibS)
            }
        }
    }

    override fun close() { session.close() }

    companion object {
        /** Load the profile's ONNX graph from assets; its calibration comes with it. */
        fun fromAssets(context: Context, profile: SpeedProfile): SpeedNet =
            SpeedNet(context.assets.open(profile.model).use { it.readBytes() }, profile)
    }
}
