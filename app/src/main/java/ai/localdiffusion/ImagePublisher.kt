package ai.localdiffusion

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.provider.MediaStore
import java.io.File

class ImagePublisher(private val context: Context) {
    fun publish(image: File): Boolean {
        if (Build.VERSION.SDK_INT < 29) return false
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, image.name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/LocalDiffusion AI")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return false
        return try {
            context.contentResolver.openOutputStream(uri)?.use { output -> image.inputStream().use { it.copyTo(output) } }
                ?: error("No se pudo abrir la galería")
            context.contentResolver.update(uri, ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }, null, null)
            true
        } catch (_: Exception) {
            context.contentResolver.delete(uri, null, null)
            false
        }
    }
}
