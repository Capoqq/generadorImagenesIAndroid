package ai.localdiffusion

import android.content.Context
import android.util.AtomicFile
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

data class BenchmarkRecord(
    val timestamp: Long,
    val modelId: String,
    val modelHash: String,
    val backend: Backend,
    val steps: Int,
    val seed: Int,
    val initializationMs: Long,
    val generationMs: Long,
    val outputBytes: Long,
    val thermalBefore: Int,
    val thermalAfter: Int
)

/** Small append-only history for verified generations. */
class BenchmarkStore(context: Context) {
    private val file = AtomicFile(File(context.filesDir, "benchmarks.json"))

    @Synchronized fun all(): List<BenchmarkRecord> = runCatching {
        if (!file.baseFile.isFile) return emptyList()
        val array = JSONArray(String(file.openRead().use { it.readBytes() }, Charsets.UTF_8))
        (0 until array.length()).map { index ->
            val row = array.getJSONObject(index)
            BenchmarkRecord(
                row.getLong("timestamp"), row.getString("modelId"), row.getString("modelHash"),
                Backend.valueOf(row.getString("backend")), row.getInt("steps"), row.getInt("seed"),
                row.getLong("initializationMs"), row.getLong("generationMs"), row.getLong("outputBytes"),
                row.getInt("thermalBefore"), row.getInt("thermalAfter")
            )
        }
    }.getOrDefault(emptyList())

    @Synchronized fun add(record: BenchmarkRecord) {
        val records = (all() + record).takeLast(100)
        val array = JSONArray()
        records.forEach { row -> array.put(JSONObject().apply {
            put("timestamp", row.timestamp)
            put("modelId", row.modelId)
            put("modelHash", row.modelHash)
            put("backend", row.backend.name)
            put("steps", row.steps)
            put("seed", row.seed)
            put("initializationMs", row.initializationMs)
            put("generationMs", row.generationMs)
            put("outputBytes", row.outputBytes)
            put("thermalBefore", row.thermalBefore)
            put("thermalAfter", row.thermalAfter)
        }) }
        val stream = file.startWrite()
        try {
            stream.write(array.toString().toByteArray(Charsets.UTF_8))
            file.finishWrite(stream)
        } catch (e: Exception) {
            file.failWrite(stream)
            throw e
        }
    }
}
