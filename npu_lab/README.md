# Laboratorio NPU MediaTek

Este módulo Android es una prueba de compilación aislada. No forma parte del APK de LocalDiffusion AI y no genera imágenes. Usa LiteRT 2.3.0 con el plugin MediaTek para comprobar la cobertura de un modelo `.tflite` en el teléfono.

## Reproducir

El modelo de referencia se obtuvo de [SearchingMan/stable-diffusion-v1-5-litert](https://huggingface.co/SearchingMan/stable-diffusion-v1-5-litert). Colocar `unet_sd15_int8.tflite` en `models/`; no se redistribuyen aquí sus pesos.

```powershell
python tools/rewrite_identity_gather.py models/unet_sd15_int8.tflite models/unet_sd15_int8_reshape.tflite
.\gradlew.bat :npu_lab:assembleDebug :npu_lab:assembleDebugAndroidTest
adb install -r npu_lab/build/outputs/apk/debug/npu_lab-debug.apk
adb install -r npu_lab/build/outputs/apk/androidTest/debug/npu_lab-debug-androidTest.apk
adb push models/unet_sd15_int8_reshape.tflite /data/local/tmp/unet_sd15_int8_reshape.tflite
adb shell run-as ai.localdiffusion.npulab cp /data/local/tmp/unet_sd15_int8_reshape.tflite files/unet_sd15_int8_reshape.tflite
adb logcat -c
adb shell am instrument -w -e model unet_sd15_int8_reshape.tflite ai.localdiffusion.npulab.test/ai.localdiffusion.npulab.UnetProbeInstrumentation
adb logcat -d -s litert neuron
```

`CREATED_CHECK_LOGCAT` significa que se creó un `CompiledModel`, **no** que se usó la NPU. Buscar `Partitioned subgraph<0>, selected X ops` y `resulted in Y partitions` en logcat. Solo X e Y mayores que cero justifican una prueba de ejecución; también hay que medir salida y latencia. En el MT6878 probado, este modelo dio X=0 e Y=0 y volvió a CPU/XNNPACK. Véase [la evaluación](../docs/runtime-evaluation.md).

En modelos grandes hay que vigilar `VmRSS` del proceso `ai.localdiffusion.npulab` durante la compilación. El VAE experimental llegó a ~6.7 GB y Android lo abortó. No ejecutar varias compilaciones grandes a la vez.

### U-Net Realistic Vision

El U-Net exacto exportado a LiteRT está en `models/realistic-vision-source/realvision_unet_wf16_afp32.tflite` (1 721 580 704 bytes, SHA-256 `dca2bcfcce1177ed4873d7db25c5394d96a7e456d36b7d68654043d8b8e69202`). Este modelo usa pesos FP16 y activaciones FP32. Para repetir la prueba:

```sh
adb push models/realistic-vision-source/realvision_unet_wf16_afp32.tflite /data/local/tmp/realvision_unet_wf16_afp32.tflite
adb shell run-as ai.localdiffusion.npulab cp /data/local/tmp/realvision_unet_wf16_afp32.tflite files/realvision_unet_wf16_afp32.tflite
adb logcat -c
adb shell am instrument -w -e model realvision_unet_wf16_afp32.tflite ai.localdiffusion.npulab.test/ai.localdiffusion.npulab.UnetProbeInstrumentation
```

La prueba del 20 de octubre de 2026 reportó **3211/3211 operaciones seleccionadas, una partición NPU**, pero la compilación no terminó antes de que Android empezara a cerrar otras apps por presión de memoria. Se detuvo el proceso de prueba. No usar este resultado como medición de inferencia ni habilitar aún la NPU para generar imágenes. Véase [la evaluación](../docs/runtime-evaluation.md).
