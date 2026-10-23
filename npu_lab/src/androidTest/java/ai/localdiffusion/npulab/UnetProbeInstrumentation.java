package ai.localdiffusion.npulab;

import android.app.Activity;
import android.app.Instrumentation;
import android.os.Bundle;
import android.os.SystemClock;
import com.google.ai.edge.litert.Accelerator;
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider;
import com.google.ai.edge.litert.CompiledModel;
import com.google.ai.edge.litert.Environment;
import java.io.File;
import java.util.Collections;

/** Compile-only probe; the host watches RSS and terminates if memory grows too large. */
public final class UnetProbeInstrumentation extends Instrumentation {
    private Bundle arguments;

    @Override public void onCreate(Bundle args) {
        super.onCreate(args);
        arguments = args;
        start();
    }

    @Override public void onStart() {
        Bundle result = new Bundle();
        try {
            if (arguments.containsKey("split")) {
                result = SplitUnetProbe.run(getTargetContext(), arguments);
                finish(Activity.RESULT_OK, result);
                return;
            }
            String name = arguments.getString("model", "unet_sd15_int8.tflite");
            if (name.contains("/") || name.contains("\\")) throw new IllegalArgumentException("Invalid model name");
            File file = new File(getTargetContext().getFilesDir(), name);
            if (!file.isFile()) throw new IllegalStateException("Missing model: " + file);
            BuiltinNpuAcceleratorProvider provider = new BuiltinNpuAcceleratorProvider(getTargetContext());
            result.putBoolean("npu_supported", provider.isDeviceSupported());
            long started = SystemClock.elapsedRealtime();
            try (Environment environment = Environment.create(getTargetContext(), provider, Collections.emptyMap(), true);
                 CompiledModel model = CompiledModel.create(file.getAbsolutePath(),
                     new CompiledModel.Options(Accelerator.NPU), environment)) {
                result.putLong("compile_ms", SystemClock.elapsedRealtime() - started);
                // LiteRT may silently fall back to CPU. The host must inspect
                // compiler_plugin.cc partition counts in logcat before claiming NPU use.
                result.putString("result", "CREATED_CHECK_LOGCAT");
            }
            finish(Activity.RESULT_OK, result);
        } catch (Throwable failure) {
            result.putString("result", "FAILED");
            result.putString("error", failure.toString());
            finish(Activity.RESULT_CANCELED, result);
        }
    }
}
