package ai.localdiffusion

import android.app.Activity
import android.app.Instrumentation
import android.os.Bundle
import android.os.Debug
import android.os.PowerManager
import android.content.Context
import android.os.SystemClock
import kotlinx.coroutines.runBlocking
import java.io.File

/** Run with `adb shell am instrument -w -e backend CPU ai.localdiffusion.test/ai.localdiffusion.GenerationInstrumentation`. */
class GenerationInstrumentation : Instrumentation() {
    private var options: Bundle? = null
    override fun onCreate(arguments: Bundle?) {
        super.onCreate(arguments)
        options = arguments
        start()
    }

    override fun onStart() {
        val data = Bundle()
        val backend = if (options?.getString("backend") == "OPENCL") Backend.OPENCL else Backend.CPU
        val steps = options?.getString("steps")?.toIntOrNull()?.coerceIn(1, 50) ?: 10
        val memoryMode = options?.getString("memory_mode")?.toIntOrNull()?.coerceIn(0, 2) ?: 0
        val seed = options?.getString("seed")?.toIntOrNull() ?: 42
        val prompt = options?.getString("prompt") ?: "a cute cat"
        val engine = MnnImageGenerationEngine()
        try {
            val context = targetContext
            if (options?.getString("npuProbe") == "true") {
                data.putString("npu_probe", NpuProbe().run())
                finish(Activity.RESULT_OK, data)
                return
            }
            val requestedModel = options?.getString("model_id")
            val model = ModelStore(context).installed().firstOrNull { requestedModel == null || it.id == requestedModel }
                ?: error("No validated SD1.5 model installed in app private storage")
            if (options?.getString("validateOnly") == "true") {
                data.putString("result", "VALIDATED")
                data.putString("model_id", model.id)
                data.putString("model_sha256", model.sha256)
                data.putString("backends", model.backends.joinToString())
                finish(Activity.RESULT_OK, data)
                return
            }
            if (options?.getString("initializeOnly") == "true") {
                val start = SystemClock.elapsedRealtime()
                runBlocking {
                    engine.initialize(model, EngineConfiguration(backend, memoryMode)).getOrThrow()
                    data.putLong("initialization_ms", SystemClock.elapsedRealtime() - start)
                    engine.release()
                }
                data.putString("result", "INITIALIZED")
                data.putString("model_id", model.id)
                data.putInt("memory_mode", memoryMode)
                finish(Activity.RESULT_OK, data)
                return
            }
            if (options?.getString("publishExisting") == "true") {
                val existing = context.filesDir.listFiles()?.filter { it.name.startsWith("benchmark-opencl-") && it.extension == "jpg" }
                    ?.maxByOrNull { it.lastModified() } ?: error("No existing benchmark image")
                check(ImagePublisher(context).publish(existing)) { "MediaStore publish failed" }
                data.putString("result", "PUBLISHED")
                data.putString("image", existing.name)
                finish(Activity.RESULT_OK, data)
                return
            }
            val output = File(context.filesDir, "benchmark-${backend.name.lowercase()}-${System.currentTimeMillis()}.jpg")
            val memoryBefore = Debug.getNativeHeapAllocatedSize()
            val thermal = context.getSystemService(Context.POWER_SERVICE) as PowerManager
            val thermalBefore = thermal.currentThermalStatus
            val start = SystemClock.elapsedRealtime()
            runBlocking {
                engine.initialize(model, EngineConfiguration(backend, memoryMode)).getOrThrow()
                val initializedAt = SystemClock.elapsedRealtime()
                data.putLong("initialization_ms", initializedAt - start)
                val generated = engine.generate(GenerationRequest(prompt, steps, seed, output)) {}.getOrThrow()
                data.putLong("generation_ms", generated.elapsedMs)
                engine.release()
                BenchmarkStore(context).add(BenchmarkRecord(
                    System.currentTimeMillis(), model.id, model.sha256, backend, steps, seed,
                    initializedAt - start, generated.elapsedMs, output.length(), thermalBefore,
                    thermal.currentThermalStatus
                ))
            }
            data.putString("result", "SUCCESS")
            data.putString("backend", backend.name)
            data.putString("model_id", model.id)
            data.putString("model_sha256", model.sha256)
            data.putString("prompt", prompt)
            data.putInt("seed", seed)
            data.putInt("steps", steps)
            data.putInt("memory_mode", memoryMode)
            data.putString("output", output.absolutePath)
            data.putLong("output_bytes", output.length())
            data.putLong("native_heap_delta_bytes", Debug.getNativeHeapAllocatedSize() - memoryBefore)
            data.putInt("thermal_before", thermalBefore)
            data.putInt("thermal_after", thermal.currentThermalStatus)
            finish(Activity.RESULT_OK, data)
        } catch (e: Throwable) {
            runBlocking { engine.release() }
            data.putString("result", "FAILED")
            data.putString("error", e.stackTraceToString())
            finish(Activity.RESULT_CANCELED, data)
        }
    }
}
