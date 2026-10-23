package ai.localdiffusion

import android.content.Context
import android.net.Uri
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.util.zip.ZipInputStream

/** Imports the flat MNN SD1.5 files used by the pinned runtime. */
class ModelStore(private val context: Context) {
    private val root = File(context.filesDir, "models")
    private val required = setOf(
        "text_encoder.mnn", "text_encoder.mnn.weight",
        "unet.mnn", "unet.mnn.weight",
        "vae_decoder.mnn", "vae_decoder.mnn.weight", "vocab.json", "merges.txt"
    )
    private val allowed = required + setOf("manifest.json", "alphas.txt", "tokenizer.mtok")

    fun installed(): List<ModelDescriptor> = root.listFiles()?.filter { it.isDirectory }?.mapNotNull {
        runCatching { validate(it) }.getOrNull()
    } ?: emptyList()

    fun importZip(uri: Uri): ModelDescriptor {
        root.mkdirs()
        val temp = File(root, "incoming-${System.nanoTime()}")
        check(temp.mkdir())
        try {
            var total = 0L
            val seen = mutableSetOf<String>()
            context.contentResolver.openInputStream(uri).use { input ->
                requireNotNull(input) { "No se pudo abrir el archivo" }
                ZipInputStream(input).use { zip ->
                    while (true) {
                        val entry = zip.nextEntry ?: break
                        require(!entry.isDirectory && entry.name in allowed && seen.add(entry.name)) { "Entrada ZIP no permitida: ${entry.name}" }
                        val output = File(temp, entry.name)
                        output.outputStream().use { stream ->
                            val buffer = ByteArray(64 * 1024)
                            var entrySize = 0L
                            while (true) {
                                val n = zip.read(buffer)
                                if (n < 0) break
                                entrySize += n; total += n
                                require(entrySize <= 2L * 1024 * 1024 * 1024 && total <= 4L * 1024 * 1024 * 1024) { "Modelo demasiado grande" }
                                stream.write(buffer, 0, n)
                            }
                        }
                        zip.closeEntry()
                    }
                }
            }
            require(seen.containsAll(required) && "manifest.json" in seen) { "Faltan componentes del modelo" }
            val checked = validate(temp)
            val target = File(root, checked.id)
            require(!target.exists()) { "Ya existe un modelo con este ID" }
            check(temp.renameTo(target)) { "No se pudo activar el modelo" }
            return checked.copy(directory = target)
        } catch (e: Exception) {
            temp.deleteRecursively()
            throw e
        }
    }

    fun validate(dir: File): ModelDescriptor {
        val manifestBytes = File(dir, "manifest.json").readBytes()
        val manifest = JSONObject(String(manifestBytes, Charsets.UTF_8))
        require(manifest.getInt("schemaVersion") == 1)
        val id = manifest.getString("id")
        require(id.matches(Regex("[a-z0-9][a-z0-9._-]{0,63}")))
        require(manifest.getString("architecture") == "stable-diffusion-1.5")
        require(manifest.getString("runtime") == "mnn")
        require(manifest.getString("license").isNotBlank())
        require(manifest.getString("source").startsWith("https://"))
        val backendNames = manifest.getJSONArray("supportedBackends")
        val backends = (0 until backendNames.length()).map { Backend.valueOf(backendNames.getString(it).uppercase()) }.toSet()
        require(backends.isNotEmpty())
        val hashes = manifest.getJSONObject("sha256")
        val digest = MessageDigest.getInstance("SHA-256")
        for (name in required + setOf("alphas.txt", "tokenizer.mtok").filter { File(dir, it).isFile }) {
            val file = File(dir, name)
            require(file.isFile && file.length() > 0)
            digest.reset()
            file.inputStream().use { input ->
                val buffer = ByteArray(64 * 1024)
                while (true) { val n = input.read(buffer); if (n < 0) break; digest.update(buffer, 0, n) }
            }
            val actual = digest.digest().joinToString("") { "%02x".format(it) }
            require(hashes.getString(name).equals(actual, ignoreCase = true)) { "SHA-256 incorrecto: $name" }
        }
        digest.reset()
        return ModelDescriptor(id, dir, digest.digest(manifestBytes).joinToString("") { "%02x".format(it) }, backends)
    }
}
