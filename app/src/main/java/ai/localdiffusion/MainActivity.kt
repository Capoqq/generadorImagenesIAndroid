package ai.localdiffusion

import android.app.ActivityManager
import android.content.Context
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.os.SystemClock
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.lifecycleScope
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : ComponentActivity() {
    private val engine = MnnImageGenerationEngine()
    private lateinit var store: ModelStore
    private lateinit var benchmarks: BenchmarkStore
    private val models = mutableStateListOf<ModelDescriptor>()
    private val benchmarkRecords = mutableStateListOf<BenchmarkRecord>()
    private var message by mutableStateOf("Listo")
    private var busy by mutableStateOf(false)
    private var generationPercent by mutableIntStateOf(0)
    private var result by mutableStateOf<GenerationResult?>(null)

    private val importLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            lifecycleScope.launch {
                busy = true; message = "Validando paquete..."
                runCatching { withContext(Dispatchers.IO) { store.importZip(uri) } }
                    .onSuccess { models.add(it); message = "Modelo importado: ${it.id}" }
                    .onFailure { message = "Importación fallida: ${it.message}" }
                busy = false
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = ModelStore(this)
        benchmarks = BenchmarkStore(this)
        message = "Validando modelos instalados..."
        lifecycleScope.launch {
            val (installed, history) = withContext(Dispatchers.IO) { store.installed() to benchmarks.all() }
            models.addAll(installed)
            benchmarkRecords.addAll(history)
            message = if (installed.isEmpty()) "No hay modelos instalados" else "${installed.size} modelo(s) validado(s)"
        }
        setContent { MaterialTheme { AppContent() } }
    }

    @Composable private fun AppContent() {
        var tab by remember { mutableIntStateOf(0) }
        var prompt by remember { mutableStateOf("professional photo of a beautiful fluffy domestic kitten, natural fur, soft daylight, sharp focus, realistic eyes") }
        var selected by remember { mutableStateOf<String?>(null) }
        var backend by remember { mutableStateOf(Backend.CPU) }
        var seed by remember { mutableStateOf("42") }
        var steps by remember { mutableStateOf("20") }
        val scope = rememberCoroutineScope()
        LaunchedEffect(models.size) {
            if (selected == null && models.isNotEmpty()) {
                val available = engine.capabilities().backends
                val preferred = models.firstOrNull { it.id == "realistic-vision-v5.1-mnn" && Backend.OPENCL in available }
                    ?: models.firstOrNull { Backend.OPENCL in it.backends && Backend.OPENCL in available }
                    ?: models.first()
                selected = preferred.id
                backend = preferred.backends.firstOrNull { it in available } ?: preferred.backends.first()
            }
        }
        Column(Modifier.fillMaxSize().safeDrawingPadding().padding(16.dp)) {
            Text("LocalDiffusion AI", style = MaterialTheme.typography.headlineMedium)
            Row { listOf("Generar", "Modelos", "Hardware", "Benchmarks").forEachIndexed { i, title ->
                TextButton(onClick = { tab = i }) { Text(title) }
            } }
            HorizontalDivider()
            Column(Modifier.verticalScroll(rememberScrollState()).weight(1f)) {
                when (tab) {
                    0 -> {
                        Text("Motor: ${if (engine.capabilities().available) "MNN nativo integrado" else "No disponible"}")
                        engine.capabilities().reason?.let { Text(it) }
                        OutlinedTextField(prompt, { prompt = it }, label = { Text("Prompt") }, modifier = Modifier.fillMaxWidth())
                        OutlinedTextField(steps, { steps = it }, label = { Text("Pasos (1–50)") })
                        OutlinedTextField(seed, { seed = it }, label = { Text("Semilla") })
                        Text("Modelo")
                        models.forEach { model ->
                            val label = when (model.id) {
                                "realistic-vision-v5.1-mnn" -> "Realistic Vision 5.1 (fotos)"
                                "sd15-mnn-opencl" -> "Stable Diffusion 1.5 (base)"
                                else -> model.id
                            }
                            Row { RadioButton(selected == model.id, { selected = model.id; backend = model.backends.first() }); Text("$label (${model.backends.joinToString()})") }
                        }
                        Text("Backend")
                        val chosenModel = models.firstOrNull { it.id == selected }
                        Row {
                            FilterChip(backend == Backend.CPU, { backend = Backend.CPU }, enabled = Backend.CPU in engine.capabilities().backends && Backend.CPU in (chosenModel?.backends ?: emptySet()), label = { Text("CPU") })
                            Spacer(Modifier.width(8.dp))
                            FilterChip(backend == Backend.OPENCL, { backend = Backend.OPENCL }, enabled = Backend.OPENCL in engine.capabilities().backends && Backend.OPENCL in (chosenModel?.backends ?: emptySet()), label = { Text("OpenCL") })
                        }
                        Button(enabled = !busy && engine.capabilities().available && backend in (chosenModel?.backends ?: emptySet()) && backend in engine.capabilities().backends && prompt.isNotBlank(), onClick = {
                            val model = models.firstOrNull { it.id == selected } ?: return@Button
                            val count = steps.toIntOrNull(); val value = seed.toIntOrNull()
                            if (count == null || count !in 1..50 || value == null) { message = "Pasos o semilla inválidos"; return@Button }
                            scope.launch {
                                busy = true; generationPercent = 0; result = null; message = "Cargando modelo..."
                                try {
                                    val thermal = getSystemService(Context.POWER_SERVICE) as PowerManager
                                    val thermalBefore = if (Build.VERSION.SDK_INT >= 29) thermal.currentThermalStatus else -1
                                    val started = SystemClock.elapsedRealtime()
                                    engine.initialize(model, EngineConfiguration(backend)).getOrThrow()
                                    val initializationMs = SystemClock.elapsedRealtime() - started
                                    message = "Generando..."
                                    val output = File(filesDir, "image-${System.currentTimeMillis()}.jpg")
                                    val generated = engine.generate(GenerationRequest(prompt, count, value, output)) { generationPercent = it.percent }.getOrThrow()
                                    result = generated
                                    val inGallery = withContext(Dispatchers.IO) { ImagePublisher(this@MainActivity).publish(generated.image) }
                                    val record = BenchmarkRecord(
                                        System.currentTimeMillis(), model.id, model.sha256, backend, count, value,
                                        initializationMs, generated.elapsedMs, output.length(), thermalBefore,
                                        if (Build.VERSION.SDK_INT >= 29) thermal.currentThermalStatus else -1
                                    )
                                    runCatching { withContext(Dispatchers.IO) { benchmarks.add(record) } }
                                        .onSuccess { benchmarkRecords.add(record) }
                                    message = "Imagen generada en ${generated.elapsedMs} ms" + if (inGallery) " · Guardada en Galería" else " · Guardada en la app"
                                } catch (e: Exception) { message = "Error: ${e.message}" }
                                finally { engine.release(); busy = false }
                            }
                        }) { Text("Generar") }
                        if (busy) LinearProgressIndicator(progress = { generationPercent / 100f }, modifier = Modifier.fillMaxWidth())
                        result?.let { generated ->
                            val bitmap = remember(generated.image.path) { BitmapFactory.decodeFile(generated.image.path) }
                            if (bitmap != null) Image(bitmap.asImageBitmap(), "Imagen generada", Modifier.fillMaxWidth())
                            Text("${generated.backend} · ${generated.elapsedMs} ms · ${generated.image.name}")
                        }
                    }
                    1 -> {
                        Text("Importa un ZIP SD 1.5 convertido a MNN con manifiesto y SHA-256.")
                        Button(enabled = !busy, onClick = { importLauncher.launch(arrayOf("application/zip", "application/octet-stream")) }) { Text("Importar modelo") }
                        models.forEach { Text("${it.id} · validado") }
                    }
                    2 -> HardwarePanel()
                    else -> {
                        Text("Generaciones completadas: ${benchmarkRecords.size}")
                        benchmarkRecords.asReversed().take(20).forEach { record ->
                            Text("${record.modelId} · ${record.backend} · ${record.steps} pasos · carga ${record.initializationMs} ms · generación ${record.generationMs} ms · térmico ${record.thermalBefore}→${record.thermalAfter}")
                            HorizontalDivider()
                        }
                        if (benchmarkRecords.isEmpty()) Text("Los resultados de las generaciones iniciadas desde la app aparecerán aquí.")
                    }
                }
            }
            HorizontalDivider()
            Text(message)
        }
    }

    @Composable private fun HardwarePanel() {
        val context = LocalContext.current
        var npuStatus by remember { mutableStateOf("Comprobando MediaTek Neuron...") }
        LaunchedEffect(Unit) {
            npuStatus = withContext(Dispatchers.IO) { NpuProbe().run() }
        }
        val memory = remember { ActivityManager.MemoryInfo().also { (getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(it) } }
        val thermal = if (Build.VERSION.SDK_INT >= 29) (context.getSystemService(Context.POWER_SERVICE) as PowerManager).currentThermalStatus.toString() else "No disponible"
        Text("ABI: ${Build.SUPPORTED_ABIS.joinToString()}")
        Text("Núcleos visibles: ${Runtime.getRuntime().availableProcessors()}")
        Text("Dispositivo: ${Build.MANUFACTURER} ${Build.MODEL}")
        Text("Android: ${Build.VERSION.RELEASE}")
        Text("RAM total: ${memory.totalMem / (1024 * 1024)} MiB")
        Text("RAM libre: ${memory.availMem / (1024 * 1024)} MiB")
        Text("Estado térmico: $thermal")
        Text("OpenCL: ${if (Backend.OPENCL in engine.capabilities().backends) "controlador disponible" else "no disponible"}.")
        Text("NPU: $npuStatus")
        Text("La prueba NPU verifica una operación; la generación de imágenes sigue usando OpenCL.")
    }

}
