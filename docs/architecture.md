# Arquitectura y fases

`MainActivity` presenta generación, importación, diagnóstico y un historial de mediciones. `ModelStore` importa ZIP mediante SAF a almacenamiento privado, admite únicamente archivos planos conocidos y exige SHA-256 por componente. `MnnImageGenerationEngine` encapsula la llamada JNI. `BenchmarkStore` guarda tiempos y estado térmico de las generaciones completadas mediante escritura atómica de un JSON privado (máximo 100 registros). La inferencia y E/S se ejecutan en `Dispatchers.IO`; la pantalla permite una sola generación. JNI transfiere únicamente prompt, progreso porcentual y ruta del JPEG, sin tensores intermedios.

El puente C++ usa la API documentada por el [demo upstream](https://github.com/alibaba/MNN/blob/master/transformers/diffusion/engine/diffusion_demo.cpp). La integración se activa con `-PmnnRoot=...`; sin ella el APK muestra explícitamente que el motor está ausente. OpenCL se probó en el teléfono `25080RABDG` con MNN 3.5.0. La API síncrona `run()` usada aquí no ofrece cancelación cooperativa segura, por lo que esa función queda pendiente de una implementación dentro del pipeline nativo.

## Estado

| Fase | Estado |
|---|---|
| 0: factibilidad | MNN SD1.5 y teléfono ARM64 verificados |
| 1: base Android | APK Compose compilado e instalado |
| 2: runtime nativo | MNN 3.5.0 compilado y cargado mediante JNI |
| 3: primera imagen | JPEG 512 × 512 de un gato generado en OpenCL con 10 pasos |
| 4: GPU | OpenCL ejecutó el pipeline; falta comparación y optimización |
| 5–8 | NPU, selección adaptativa, persistencia y preparación de producción pendientes |

La prueba instrumentada en teléfono produjo una imagen reconocible de 512 × 512 con 10 pasos y midió la duración. La prueba de CPU del modelo general agotó memoria antes de producir archivo.

## Verificación local

- `python -m unittest discover -s tests -v`: 2 pruebas correctas (empaquetado y rechazo de componente ausente).
- `.\gradlew.bat :app:assembleDebug :app:assembleDebugAndroidTest -PmnnRoot=...`: `BUILD SUCCESSFUL` con MNN 3.5.0.
- `adb install --no-streaming -r` de ambos APK: correcto en el dispositivo `25080RABDG`.
- `adb shell am instrument -w -e backend OPENCL -e steps 10 ...`: `SUCCESS`, 318.669 s de generación, JPEG de 143,770 bytes y gato reconocible.
- `adb shell am instrument -w -e validateOnly true ...`: `VALIDATED`, paquete y hashes aceptados por la app final.
- `adb shell am instrument -w -e publishExisting true ...`: `PUBLISHED`, JPEG exportado a MediaStore y visible en `Pictures/LocalDiffusion AI`.
- Flujo de interfaz en teléfono: prompt, selección OpenCL, botón **Generar**, vista previa, exportación a Galería e historial Benchmarks verificados; duración 317.595 s y mismo SHA-256 que las pruebas instrumentadas.

Persistencia Room, cálculo estadístico dentro de la app, selección automática por medidas, pantalla de galería propia, cancelación cooperativa, gestión térmica avanzada y adaptadores ONNX/QNN permanecen pendientes. La publicación de imágenes mediante MediaStore está implementada y probada en Android API 36; en API 26–28 el archivo queda en almacenamiento privado de la app.
