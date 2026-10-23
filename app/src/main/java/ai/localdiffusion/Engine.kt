package ai.localdiffusion

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

enum class Backend { CPU, OPENCL }
data class ModelDescriptor(val id: String, val directory: File, val sha256: String, val backends: Set<Backend>)
data class EngineConfiguration(val backend: Backend, val memoryMode: Int = 0)
data class GenerationRequest(val prompt: String, val steps: Int, val seed: Int, val output: File)
data class GenerationProgress(val percent: Int)
data class GenerationResult(val image: File, val elapsedMs: Long, val backend: Backend)
data class EngineCapabilities(val available: Boolean, val backends: Set<Backend>, val reason: String?)

interface ImageGenerationEngine {
    suspend fun initialize(model: ModelDescriptor, config: EngineConfiguration): Result<Unit>
    suspend fun generate(request: GenerationRequest, progress: (GenerationProgress) -> Unit): Result<GenerationResult>
    suspend fun cancel()
    fun capabilities(): EngineCapabilities
    suspend fun release()
}

/** Calls upstream MNN Diffusion on a worker thread. The optional JNI library is never silently replaced. */
class MnnImageGenerationEngine : ImageGenerationEngine {
    private val loaded = try {
        System.loadLibrary("localdiffusion")
        true
    } catch (_: UnsatisfiedLinkError) { false }
    private var handle = 0L
    private var backend = Backend.CPU
    private val openClAvailable = loaded && nativeHasOpenClPlatform()

    override fun capabilities() = EngineCapabilities(
        loaded, if (loaded) setOf(Backend.CPU) + (if (openClAvailable) setOf(Backend.OPENCL) else emptySet()) else emptySet(),
        if (loaded) null else "MNN nativo no integrado en este APK"
    )

    override suspend fun initialize(model: ModelDescriptor, config: EngineConfiguration): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            check(loaded) { "MNN nativo no integrado" }
            check(config.backend in capabilities().backends) { "Backend no disponible en este dispositivo" }
            check(config.backend in model.backends) { "El paquete no admite este backend" }
            check(handle == 0L) { "Libera el modelo anterior antes de cargar otro" }
            val created = nativeCreate(model.directory.absolutePath, if (config.backend == Backend.CPU) 0 else 3, config.memoryMode)
            check(created != 0L) { "MNN no pudo cargar el modelo" }
            handle = created
            backend = config.backend
        }
    }

    override suspend fun generate(request: GenerationRequest, progress: (GenerationProgress) -> Unit): Result<GenerationResult> = withContext(Dispatchers.IO) {
        runCatching {
            check(handle != 0L) { "No hay modelo cargado" }
            require(request.prompt.isNotBlank())
            require(request.steps in 1..50)
            val start = android.os.SystemClock.elapsedRealtime()
            val ok = nativeGenerate(handle, request.prompt, request.output.absolutePath, request.steps, request.seed) {
                progress(GenerationProgress(it.coerceIn(0, 100)))
            }
            check(ok && request.output.isFile && request.output.length() > 0) { "MNN no produjo una imagen" }
            GenerationResult(request.output, android.os.SystemClock.elapsedRealtime() - start, backend)
        }
    }

    // Upstream MNN's synchronous run() does not expose a safe interrupt. The UI disables cancellation
    // while the native call is active instead of destroying a session used by another thread.
    override suspend fun cancel() { }
    override suspend fun release() = withContext(Dispatchers.IO) {
        if (handle != 0L) { nativeRelease(handle); handle = 0L }
    }

    private external fun nativeCreate(directory: String, backend: Int, memoryMode: Int): Long
    private external fun nativeGenerate(handle: Long, prompt: String, output: String, steps: Int, seed: Int, progress: (Int) -> Unit): Boolean
    private external fun nativeRelease(handle: Long)
    private external fun nativeHasOpenClPlatform(): Boolean
}
