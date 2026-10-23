#include <jni.h>
#include <dlfcn.h>
#include <memory>
#include <string>
#include "diffusion/stable_diffusion.hpp"

using MNN::DIFFUSION::Diffusion;
using MNN::DIFFUSION::STABLE_DIFFUSION_1_5;

namespace {
struct NativeDiffusionSession { std::unique_ptr<Diffusion> diffusion; };
void fail(JNIEnv* env, const char* message) {
    auto cls = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(cls, message);
}
std::string utf(JNIEnv* env, jstring value) {
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (!chars) return {};
    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_localdiffusion_MnnImageGenerationEngine_nativeCreate(JNIEnv* env, jobject, jstring directory, jint backend, jint memoryMode) {
    if (!directory || (backend != 0 && backend != 3) || memoryMode < 0 || memoryMode > 2) {
        fail(env, "Invalid MNN configuration"); return 0;
    }
    try {
        auto session = std::make_unique<NativeDiffusionSession>();
        session->diffusion.reset(Diffusion::createDiffusion(utf(env, directory), STABLE_DIFFUSION_1_5,
            static_cast<MNNForwardType>(backend), memoryMode));
        if (!session->diffusion || !session->diffusion->load()) { fail(env, "MNN load failed"); return 0; }
        return reinterpret_cast<jlong>(session.release());
    } catch (const std::exception& e) { fail(env, e.what()); return 0; }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_ai_localdiffusion_MnnImageGenerationEngine_nativeGenerate(JNIEnv* env, jobject, jlong handle, jstring prompt,
    jstring output, jint steps, jint seed, jobject progress) {
    if (!handle || !prompt || !output || !progress || steps < 1 || steps > 50) {
        fail(env, "Invalid generation request"); return JNI_FALSE;
    }
    auto* session = reinterpret_cast<NativeDiffusionSession*>(handle);
    auto integer = env->FindClass("java/lang/Integer");
    auto integerCtor = env->GetMethodID(integer, "<init>", "(I)V");
    auto function = env->FindClass("kotlin/jvm/functions/Function1");
    auto invoke = env->GetMethodID(function, "invoke", "(Ljava/lang/Object;)Ljava/lang/Object;");
    try {
        bool ok = session->diffusion->run(utf(env, prompt), utf(env, output), steps, seed, [&](int percent) {
            auto boxed = env->NewObject(integer, integerCtor, percent);
            auto ignored = env->CallObjectMethod(progress, invoke, boxed);
            if (ignored) env->DeleteLocalRef(ignored);
            env->DeleteLocalRef(boxed);
        });
        return ok && !env->ExceptionCheck() ? JNI_TRUE : JNI_FALSE;
    } catch (const std::exception& e) { fail(env, e.what()); return JNI_FALSE; }
}

extern "C" JNIEXPORT void JNICALL
Java_ai_localdiffusion_MnnImageGenerationEngine_nativeRelease(JNIEnv*, jobject, jlong handle) {
    delete reinterpret_cast<NativeDiffusionSession*>(handle);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_ai_localdiffusion_MnnImageGenerationEngine_nativeHasOpenClPlatform(JNIEnv*, jobject) {
    void* library = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (!library) return JNI_FALSE;
    using GetPlatformIds = int (*)(unsigned int, void*, unsigned int*);
    auto getPlatforms = reinterpret_cast<GetPlatformIds>(dlsym(library, "clGetPlatformIDs"));
    unsigned int count = 0;
    bool available = getPlatforms && getPlatforms(0, nullptr, &count) == 0 && count > 0;
    dlclose(library);
    return available ? JNI_TRUE : JNI_FALSE;
}
