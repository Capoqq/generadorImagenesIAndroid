#include <jni.h>
#include <android/NeuralNetworks.h>
#include <cmath>
#include <chrono>
#include <string>

namespace {
std::string error(const char* stage, int status) {
    return std::string("NPU test failed at ") + stage + " (NNAPI " + std::to_string(status) + ")";
}
}

__attribute__((availability(android, introduced=29)))
static jstring runNpuProbe(JNIEnv* env) {
    uint32_t count = 0;
    int status = ANeuralNetworks_getDeviceCount(&count);
    if (status != ANEURALNETWORKS_NO_ERROR) return env->NewStringUTF(error("enumeration", status).c_str());
    const ANeuralNetworksDevice* selected = nullptr;
    std::string name;
    for (uint32_t i = 0; i < count; ++i) {
        ANeuralNetworksDevice* device = nullptr;
        if (ANeuralNetworks_getDevice(i, &device) != ANEURALNETWORKS_NO_ERROR) continue;
        const char* candidate = nullptr;
        if (ANeuralNetworksDevice_getName(device, &candidate) != ANEURALNETWORKS_NO_ERROR || !candidate) continue;
        std::string candidateName(candidate);
        if (candidateName.find("neuron") != std::string::npos) {
            selected = device;
            name = candidateName;
            break;
        }
    }
    if (!selected) return env->NewStringUTF("No MediaTek NNAPI device exposed to this app");

    ANeuralNetworksModel* model = nullptr;
    ANeuralNetworksCompilation* compilation = nullptr;
    ANeuralNetworksExecution* execution = nullptr;
    auto cleanup = [&] {
        if (execution) ANeuralNetworksExecution_free(execution);
        if (compilation) ANeuralNetworksCompilation_free(compilation);
        if (model) ANeuralNetworksModel_free(model);
    };
    auto fail = [&](const char* stage, int code) {
        cleanup();
        return env->NewStringUTF(error(stage, code).c_str());
    };
    status = ANeuralNetworksModel_create(&model);
    if (status) return fail("model create", status);
    uint32_t dimensions[] = {1};
    ANeuralNetworksOperandType tensorType{ANEURALNETWORKS_TENSOR_FLOAT32, 1, dimensions, 0.0f, 0};
    ANeuralNetworksOperandType scalarType{ANEURALNETWORKS_INT32, 0, nullptr, 0.0f, 0};
    for (int i = 0; i < 4; ++i) {
        status = ANeuralNetworksModel_addOperand(model, i == 2 ? &scalarType : &tensorType);
        if (status) return fail("operand", status);
    }
    int32_t activation = ANEURALNETWORKS_FUSED_NONE;
    status = ANeuralNetworksModel_setOperandValue(model, 2, &activation, sizeof(activation));
    if (status) return fail("activation", status);
    uint32_t inputs[] = {0, 1, 2};
    uint32_t outputs[] = {3};
    status = ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_ADD, 3, inputs, 1, outputs);
    if (status) return fail("operation", status);
    uint32_t modelInputs[] = {0, 1};
    status = ANeuralNetworksModel_identifyInputsAndOutputs(model, 2, modelInputs, 1, outputs);
    if (status) return fail("inputs/outputs", status);
    status = ANeuralNetworksModel_finish(model);
    if (status) return fail("model finish", status);

    bool supported = false;
    status = ANeuralNetworksModel_getSupportedOperationsForDevices(model, &selected, 1, &supported);
    if (status) return fail("support query", status);
    if (!supported) {
        cleanup();
        std::string message = name + ": float32 ADD unsupported by driver";
        return env->NewStringUTF(message.c_str());
    }

    status = ANeuralNetworksCompilation_createForDevices(model, &selected, 1, &compilation);
    if (status) return fail("device compilation", status);
    status = ANeuralNetworksCompilation_finish(compilation);
    if (status) return fail("compile", status);
    status = ANeuralNetworksExecution_create(compilation, &execution);
    if (status) return fail("execution create", status);
    float a = 1.25f, b = 2.5f, result = NAN;
    status = ANeuralNetworksExecution_setInput(execution, 0, nullptr, &a, sizeof(a));
    if (status) return fail("input 0", status);
    status = ANeuralNetworksExecution_setInput(execution, 1, nullptr, &b, sizeof(b));
    if (status) return fail("input 1", status);
    status = ANeuralNetworksExecution_setOutput(execution, 0, nullptr, &result, sizeof(result));
    if (status) return fail("output", status);
    auto started = std::chrono::steady_clock::now();
    status = ANeuralNetworksExecution_compute(execution);
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count();
    if (status) return fail("compute", status);
    cleanup();
    if (std::fabs(result - 3.75f) > 0.0001f) return env->NewStringUTF("NPU result mismatch");
    std::string message = name + ": ADD verified (" + std::to_string(elapsed) + " us)";
    return env->NewStringUTF(message.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_localdiffusion_NpuProbe_nativeRun(JNIEnv* env, jobject) {
    if (__builtin_available(android 29, *)) return runNpuProbe(env);
    return env->NewStringUTF("NPU probe requires Android 10 or newer");
}
