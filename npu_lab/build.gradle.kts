plugins {
    id("com.android.application")
}

android {
    namespace = "ai.localdiffusion.npulab"
    compileSdk = 35
    defaultConfig {
        applicationId = "ai.localdiffusion.npulab"
        minSdk = 29
        targetSdk = 35
        testInstrumentationRunner = "ai.localdiffusion.npulab.UnetProbeInstrumentation"
        ndk { abiFilters += "arm64-v8a" }
    }
    packaging { jniLibs { useLegacyPackaging = true } }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation("com.google.ai.edge.litert:litert:2.3.0")
    implementation("com.google.ai.edge.litert:litert-npu-runtime-mediatek:2.3.0")
}
