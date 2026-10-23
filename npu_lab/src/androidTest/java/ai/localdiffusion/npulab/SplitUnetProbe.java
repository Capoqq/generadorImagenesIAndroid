package ai.localdiffusion.npulab;

import android.content.Context;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import com.google.ai.edge.litert.Accelerator;
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider;
import com.google.ai.edge.litert.CompiledModel;
import com.google.ai.edge.litert.Environment;
import com.google.ai.edge.litert.TensorBuffer;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.json.JSONArray;
import org.json.JSONObject;

/** Executes precompiled graph partitions and preserves their explicit skip tensors. */
final class SplitUnetProbe {
    private static final String SIGNATURE = "serving_default";

    static Bundle run(Context context, Bundle args) throws Exception {
        String directory = args.getString("split");
        if (directory == null || !directory.matches("[a-zA-Z0-9_-]+")) {
            throw new IllegalArgumentException("Invalid split directory");
        }
        File root = new File(context.getFilesDir(), directory);
        JSONObject manifest = new JSONObject(new String(Files.readAllBytes(new File(root, "manifest.json").toPath()), StandardCharsets.UTF_8));
        JSONArray parts = manifest.getJSONArray("partitions");
        JSONObject tensors = manifest.getJSONObject("tensors");
        int limit = Math.min(parts.length(), Integer.parseInt(args.getString("max_parts", "1000")));
        Map<Integer, float[]> values = new HashMap<>();
        JSONArray inputs = manifest.getJSONArray("inputs");
        for (int i = 0; i < inputs.length(); i++) {
            int id = inputs.getInt(i);
            byte[] bytes = Files.readAllBytes(new File(root, "input_t" + id + ".bin").toPath());
            if (bytes.length != tensors.getJSONObject(Integer.toString(id)).getInt("bytes")) {
                throw new IllegalArgumentException("Input length mismatch: " + id);
            }
            float[] array = new float[bytes.length / 4];
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(array);
            values.put(id, array);
        }
        JSONArray measurements = new JSONArray();
        long totalStarted = SystemClock.elapsedRealtime();
        BuiltinNpuAcceleratorProvider provider = new BuiltinNpuAcceleratorProvider(context);
        try (Environment environment = Environment.create(context, provider, Collections.emptyMap(), true)) {
            for (int partIndex = 0; partIndex < limit; partIndex++) {
                JSONObject part = parts.getJSONObject(partIndex);
                String filename = part.getString("file");
                if (filename.contains("/") || filename.contains("\\")) throw new IllegalArgumentException("Invalid file");
                long started = SystemClock.elapsedRealtime();
                JSONObject measurement = new JSONObject().put("partition", partIndex);
                Map<String, TensorBuffer> in = new HashMap<>();
                Map<String, TensorBuffer> out = new HashMap<>();
                Log.i("SplitUnetProbe", "Loading partition " + partIndex);
                try (CompiledModel model = CompiledModel.create(new File(root, filename).getAbsolutePath(),
                        new CompiledModel.Options(Accelerator.NPU), environment)) {
                    measurement.put("load_ms", SystemClock.elapsedRealtime() - started);
                    JSONArray ids = part.getJSONArray("inputs");
                    for (int i = 0; i < ids.length(); i++) {
                        int id = ids.getInt(i);
                        float[] data = values.get(id);
                        if (data == null) throw new IllegalStateException("Missing tensor " + id);
                        TensorBuffer buffer = model.createInputBuffer("t" + id, SIGNATURE);
                        in.put("t" + id, buffer);
                        buffer.writeFloat(data);
                    }
                    JSONArray outputIds = part.getJSONArray("outputs");
                    for (int i = 0; i < outputIds.length(); i++) {
                        String name = "t" + outputIds.getInt(i);
                        out.put(name, model.createOutputBuffer(name, SIGNATURE));
                    }
                    long runStarted = SystemClock.elapsedRealtime();
                    model.run(in, out, SIGNATURE);
                    measurement.put("invoke_ms", SystemClock.elapsedRealtime() - runStarted);
                    for (int i = 0; i < outputIds.length(); i++) {
                        int id = outputIds.getInt(i);
                        float[] data = out.get("t" + id).readFloat();
                        if (data.length * 4 != tensors.getJSONObject(Integer.toString(id)).getInt("bytes")) {
                            throw new IllegalStateException("Output length mismatch: " + id);
                        }
                        for (float value : data) {
                            if (!Float.isFinite(value)) throw new IllegalStateException("Nonfinite tensor " + id);
                        }
                        values.put(id, data);
                    }
                } finally {
                    for (TensorBuffer buffer : in.values()) buffer.close();
                    for (TensorBuffer buffer : out.values()) buffer.close();
                }
                measurement.put("total_ms", SystemClock.elapsedRealtime() - started);
                measurements.put(measurement);
                Log.i("SplitUnetProbe", measurement.toString());
                Files.write(new File(root, "measurements.json").toPath(), measurements.toString(2).getBytes(StandardCharsets.UTF_8));
                Set<Integer> needed = new HashSet<>();
                for (int j = partIndex + 1; j < limit; j++) {
                    JSONArray future = parts.getJSONObject(j).getJSONArray("inputs");
                    for (int k = 0; k < future.length(); k++) needed.add(future.getInt(k));
                }
                JSONArray finalIds = parts.getJSONObject(limit - 1).getJSONArray("outputs");
                for (int j = 0; j < finalIds.length(); j++) needed.add(finalIds.getInt(j));
                values.keySet().retainAll(needed);
            }
        }
        JSONArray outputIds = parts.getJSONObject(limit - 1).getJSONArray("outputs");
        for (int i = 0; i < outputIds.length(); i++) {
            int id = outputIds.getInt(i);
            float[] data = values.get(id);
            ByteBuffer buffer = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
            buffer.asFloatBuffer().put(data);
            Files.write(new File(root, "output_t" + id + ".bin").toPath(), buffer.array());
        }
        Bundle result = new Bundle();
        result.putString("result", "INFERENCE_FINISHED_CHECK_OUTPUTS_AND_DISPATCH");
        result.putInt("partitions", limit);
        result.putLong("total_ms", SystemClock.elapsedRealtime() - totalStarted);
        result.putString("measurements", measurements.toString());
        return result;
    }
}
