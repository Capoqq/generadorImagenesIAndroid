# LocalDiffusion AI

Proyecto Android ARM64 para generación local con MNN Diffusion. El APK integra MNN 3.5.0 y produjo una imagen reconocible de 512 × 512 en un teléfono físico con OpenCL. No usa servidores ni claves API. Los resultados medidos están en [docs/benchmark-results.md](docs/benchmark-results.md).

## Compilación

Requiere Android SDK 35, NDK `27.1.12297006`, JDK 21, Gradle Wrapper 8.10.2 y acceso a Google Maven/Maven Central. Las versiones del proyecto están fijadas en `build.gradle.kts`. Clona [alibaba/MNN](https://github.com/alibaba/MNN) en `third_party/MNN-3.5.0` y fija el tag `3.5.0`. Compila con `.\gradlew.bat :app:assembleDebug :app:assembleDebugAndroidTest "-PmnnRoot=C:/ruta/absoluta/MNN-3.5.0"`. Sin `-PmnnRoot`, el APK muestra explícitamente que el motor nativo no está integrado. MNN 3.6.0 produjo una imagen con artefactos graves en este teléfono; consulta el benchmark.

El `targetSdk` actual es 35 para esta base local. Antes de publicar en Google Play, verifica el nivel exigido por la [política vigente de Play](https://developer.android.com/google/play/requirements/target-sdk) y los cambios de comportamiento correspondientes.

El ZIP de modelo debe contener `manifest.json`, `text_encoder.mnn`, `text_encoder.mnn.weight`, `unet.mnn`, `unet.mnn.weight`, `vae_decoder.mnn`, `vae_decoder.mnn.weight`, `vocab.json` y `merges.txt`. `alphas.txt` y `tokenizer.mtok` son opcionales. MNN 3.5.0 utiliza `vocab.json` y `merges.txt` para CLIP. Convierte el modelo siguiendo la [guía de MNN](https://github.com/alibaba/MNN/blob/master/docs/transformers/diffusion.md) y verifica los archivos exigidos por la revisión que compiles. El paquete debe declarar el backend compatible. Manifiesto:

Después de convertir, `python tools/package_model.py ./modelo ./sd15.zip --id sd15-local --license creativeml-openrail-m --source https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 --backend cpu` calcula los hashes y crea el ZIP importable. Usa `--backend opencl` para un modelo específico de OpenCL. Los modelos descargados en `models/` están excluidos del repositorio.

```json
{
  "schemaVersion": 1,
  "id": "sd15-local",
  "architecture": "stable-diffusion-1.5",
  "runtime": "mnn",
  "license": "creativeml-openrail-m",
  "source": "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5",
  "supportedBackends": ["cpu"],
  "sha256": {
    "text_encoder.mnn": "<64 hex chars>",
    "text_encoder.mnn.weight": "<64 hex chars>",
    "unet.mnn": "<64 hex chars>",
    "unet.mnn.weight": "<64 hex chars>",
    "vae_decoder.mnn": "<64 hex chars>",
    "vae_decoder.mnn.weight": "<64 hex chars>",
    "vocab.json": "<64 hex chars>",
    "merges.txt": "<64 hex chars>"
  }
}
```

## Referencias de modelos

Los pesos, las conversiones y los ZIP importables se guardan en `models/`, excluido de Git. Los archivos `.mnn`, `.mnn.weight`, `.onnx`, `.safetensors` y `.tflite` también se ignoran fuera de esa carpeta. Guarda allí los paquetes generados con `tools/package_model.py`.

Además del modelo base Stable Diffusion 1.5 enlazado arriba, estas son las fuentes utilizadas en el proyecto:

- SD 1.5 convertido a MNN: [general](https://huggingface.co/taobao-mnn/stable-diffusion-v1-5-mnn-general) y [OpenCL](https://huggingface.co/taobao-mnn/stable-diffusion-v1-5-mnn-opencl). Se descargan con `tools/download_mnn_sd15.py`, seleccionando el repositorio mediante `--repo`.
- [Realistic Vision v5.1 MNN INT8](https://huggingface.co/Zevas1993/nexus-sd15-inpaint-mnn-int8): componentes de texto a imagen usados en `models/realistic-vision-v5.1-mnn/`.
- [Realistic Vision v5.1 noVAE](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE): checkpoint de origen para la exportación experimental a LiteRT; revisión utilizada `1e9f017a7b1eaefb63a1900ea6c5953d2739fd21`.
- [SD 1.5 LiteRT](https://huggingface.co/SearchingMan/stable-diffusion-v1-5-litert): modelo de referencia del [laboratorio NPU](npu_lab/README.md).

## Prueba en teléfono

`adb devices -l`; instala ambos APK y usa la app para importar el ZIP. El paquete probado está en `models/sd15-mnn35-import.zip` de este espacio de trabajo y en `Descargas/LocalDiffusion-sd15.zip` del teléfono; ambos SHA-256 coinciden (`9ec5fe5185921e198505fb4059d31067f034fd3a7f97eb55b78d2009b1164145`). Una prueba instrumentada se ejecuta con `adb shell am instrument -w -e backend OPENCL -e steps 10 ai.localdiffusion.test/ai.localdiffusion.GenerationInstrumentation`. El modelo OpenCL fusionado no es compatible con CPU. La imagen verificada está también en `Descargas/LocalDiffusion-cat-10steps.jpg` y en `Imágenes/LocalDiffusion AI` de la Galería. Las generaciones iniciadas desde la interfaz se publican automáticamente en Galería en Android 10 o superior. Consulta [evaluación de runtimes](docs/runtime-evaluation.md), [arquitectura](docs/architecture.md) y [resultados](docs/benchmark-results.md).

La pestaña **Benchmarks** muestra tiempos de carga y generación, pasos, backend y estado térmico de las ejecuciones completadas. El historial se guarda como JSON privado con escritura atómica y conserva los 100 registros más recientes.
