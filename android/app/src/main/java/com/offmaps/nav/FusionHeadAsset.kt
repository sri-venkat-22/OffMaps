package com.offmaps.nav

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/**
 * Loads assets/fusion_head.json (py/model/fusion_head.export_json) into a [FusionHead].
 * Returns null when the asset is absent, so the app falls back to the profile's
 * hand-over rule (hold Doppler, then SpeedNet) exactly as before.
 */
object FusionHeadAsset {
    const val ASSET = "fusion_head.json"

    fun load(context: Context, asset: String = ASSET): FusionHead? = try {
        val j = JSONObject(context.assets.open(asset).bufferedReader().use { it.readText() })
        require(j.getString("format") == "offmaps-fusion-head/1") { "unknown fusion head format" }
        val h = j.getInt("hidden")
        val ms = j.getJSONArray("members")
        FusionHead((0 until ms.length()).map { i ->
            val m = ms.getJSONObject(i)
            FusionHead.Member(
                hidden = h,
                ctxW = mat(m.getJSONArray("ctx.0.weight")), ctxB = vec(m.getJSONArray("ctx.0.bias")),
                wIh = mat(m.getJSONArray("gru.weight_ih_l0")), wHh = mat(m.getJSONArray("gru.weight_hh_l0")),
                bIh = vec(m.getJSONArray("gru.bias_ih_l0")), bHh = vec(m.getJSONArray("gru.bias_hh_l0")),
                outW = mat(m.getJSONArray("out.weight")), outB = vec(m.getJSONArray("out.bias")),
            )
        })
    } catch (e: java.io.IOException) {
        null
    }

    private fun vec(a: JSONArray) = DoubleArray(a.length()) { a.getDouble(it) }
    private fun mat(a: JSONArray) = Array(a.length()) { vec(a.getJSONArray(it)) }
}
