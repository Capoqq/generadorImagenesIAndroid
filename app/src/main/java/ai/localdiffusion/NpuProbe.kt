package ai.localdiffusion

/** Explicitly compiles and executes a tiny NNAPI graph on the MediaTek driver. */
class NpuProbe {
    private external fun nativeRun(): String
    fun run(): String = runCatching {
        System.loadLibrary("localdiffusion")
        nativeRun()
    }.getOrElse { "NPU probe unavailable: ${it.message}" }
}
