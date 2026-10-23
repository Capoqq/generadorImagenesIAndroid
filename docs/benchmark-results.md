# Resultados de benchmark

Fecha: 19 octubre 2026. Dispositivo físico `25080RABDG` (Android API 36, ARM64, 7.6 GB RAM, GPU Mali). Modelo `taobao-mnn/stable-diffusion-v1-5-mnn-opencl`. Prompt `a cute cat`, seed 42, 512 × 512. Se compararon MNN 3.6.0 y 3.5.0 con el mismo teléfono y los mismos pesos. El JPEG de MNN 3.5.0 se extrajo del almacenamiento privado y mostró un gato reconocible. SHA-256: `4fe116495992060b57615e84c93265289cd510220dfb6acd69a75383a1abe6ef`.

| Configuración | Tiempo | RAM pico | Estado |
|---|---|---|---|
| CPU 2 hilos | — | — | NOT_TESTED |
| CPU 4 hilos, modelo general | >239 s, sin imagen | PSS ~12 GB incluyendo swap, observado en log del sistema | FAILED: presión de memoria y desconexión ADB |
| CPU 8 hilos | — | — | NOT_TESTED |
| OpenCL GPU, MNN 3.6.0, 1 paso | 316.811 s + 7.935 s carga | PSS 1.84 GB durante la ejecución (muestra, no pico) | JPEG de 327,140 bytes; solo ruido; térmico 0 |
| OpenCL GPU, MNN 3.6.0, 10 pasos | 474.252 s + 6.196 s carga | PSS 1.85 GB durante la ejecución (muestra, no pico) | JPEG de 178,691 bytes; artefactos graves, imagen semánticamente incorrecta |
| OpenCL GPU, MNN 3.5.0, 10 pasos | 318.669 s + 4.515 s carga | PSS 1.75 GB durante la ejecución (muestra, no pico) | SUCCESS, gato reconocible, JPEG de 143,770 bytes; térmico 0 |
| OpenCL GPU, MNN 3.5.0, 10 pasos, repetición | 314.663 s + 3.720 s carga | PSS 1.10 GB durante VAE (muestra, no pico) | SUCCESS, mismo SHA-256 y tamaño; térmico 0 |
| OpenCL GPU, MNN 3.5.0, 10 pasos, botón de la app | 317.595 s + 4.758 s carga | No medida | SUCCESS, vista previa y MediaStore; mismo SHA-256; térmico 0 |
| QNN NPU | — | — | NOT_TESTED |
| Híbrido | — | — | NOT_TESTED |

El binario MNN de línea de comandos, lanzado fuera del contexto de la app, no pudo enumerar plataformas OpenCL (`clGetPlatformIDs=-1001`). Dentro de la app sí cargó el controlador porque el manifiesto declara `libOpenCL.so` como biblioteca nativa opcional. El modelo OpenCL fusionado falló en CPU por una aserción `CPURaster`; no debe usarse para comparar ambos backends. El modelo general de CPU llegó al encoder, pero el teléfono agotó memoria antes de completar la imagen. Ninguno de esos fallos se cuenta como generación exitosa.

`NOT_TESTED` no implica `UNSUPPORTED`. El adaptador aún no expone selección de hilos por experimento ni persistencia Room. La imagen de MNN 3.5.0 se copió a `Descargas/LocalDiffusion-cat-10steps.jpg` y se verificó visualmente. El A/B aisló la versión del runtime como causa probable de los artefactos observados; [MNN documenta otra regresión OpenCL en 3.6.0](https://github.com/alibaba/MNN/issues/4653), aunque en un GPU distinto. El modelo específico OpenCL no se ejecutó en CPU.

Las tres generaciones correctas con MNN 3.5.0 produjeron el mismo SHA-256 `4fe116495992060b57615e84c93265289cd510220dfb6acd69a75383a1abe6ef`. La mediana de generación es 317.595 s (n=3; rango 314.663–318.669 s). Las dos ejecuciones del 20 de octubre quedaron persistidas en el historial de la app. La tercera se inició con el botón **Generar**, mostró el JPEG en pantalla y publicó `image-1792502104476.jpg` en MediaStore. `PSS` corresponde a muestras puntuales de `dumpsys meminfo`, no a un pico monitorizado continuamente.

## Realistic Vision 5.1

El 20 de octubre se descargó y verificó mediante SHA-256 el paquete [Nexus Realistic Vision v5.1 MNN INT8](https://huggingface.co/Zevas1993/nexus-sd15-inpaint-mnn-int8), usando únicamente los componentes de texto a imagen. Se instaló como segundo modelo, `realistic-vision-v5.1-mnn`, sin retirar SD1.5 base. El paquete importable local es `models/realistic-vision-v5.1-mnn-import.zip` (1,153,108,614 bytes, SHA-256 `197c1d6cf6c420e7db05c1264097ee4bde3b45d63f73d665e9b4053c1248ef24`); también está en `Descargas/LocalDiffusion-RealisticVision-v5.1.zip` del teléfono.

| Configuración | Carga | Generación | Observación visual |
|---|---:|---:|---|
| Realistic Vision 5.1, OpenCL, 10 pasos, `a cute cat`, semilla 42 | 93.858 s | 246.921 s | Gato reconocible, pero artefactos fuertes en la cara; no recomendado. |
| Realistic Vision 5.1, OpenCL, 20 pasos, prompt fotográfico descriptivo, semilla 42 | 99.208 s | 361.462 s | Gatito fotográfico nítido y coherente; guardado en Galería. |

Ambas pruebas terminaron sin alerta térmica. La segunda imagen se extrajo como `models/realistic-vision-cat-20steps.jpg`; el resultado depende tanto del prompt y los pasos como de los pesos, así que las dos filas no aíslan el efecto del modelo. Por esa razón la app ofrece Realistic Vision con 20 pasos y el prompt probado como valores iniciales; el usuario puede cambiar ambos. El modelo base permanece seleccionable. La ficha del modelo indica que la conversión se hizo para MNN 3.4.1; aquí se verificó generación real con MNN 3.5.0 y GPU Mali.

## Mejora de carga OpenCL

El 20 de octubre se corrigió la ruta de caché del runtime MNN. La ruta relativa original `.tempcache` no dejaba un archivo utilizable en el almacenamiento de la app; ahora cada modelo guarda `.opencl-cache` en su directorio privado. [MNN documenta](https://github.com/alibaba/MNN/blob/master/docs/inference/module.md) esta caché para reutilizar programas compilados y resultados de autotuning. La primera inicialización con el cambio tardó 100.716 s y creó un archivo de 916 KB. La segunda inicialización tardó 8.424 s. Una generación completa posterior usó 8.395 s de carga y 363.484 s de generación, con el mismo prompt, semilla y 20 pasos. El JPEG fue idéntico byte por byte al resultado previo (SHA-256 `690c030dafcf668df5bdef3a393afde3d76f8a7db811ad5ed0efac660ca3a29f`). El tiempo total bajó de 460.670 s a 371.879 s, un ahorro de 88.791 s (19.3%) en esa comparación. La primera ejecución tras instalar el modelo sigue pagando la compilación inicial.

También se probó el prompt fotográfico con 10 pasos: carga 101.221 s, generación 250.740 s. La imagen mostró artefactos marcados en la cara, por lo que se conservaron 20 pasos como valor inicial de calidad.
