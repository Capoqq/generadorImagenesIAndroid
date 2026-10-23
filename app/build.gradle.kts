plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

val mnnRoot = providers.gradleProperty("mnnRoot").orNull

android {
    namespace = "ai.localdiffusion"
    compileSdk = 35
    defaultConfig {
        applicationId = "ai.localdiffusion"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "ai.localdiffusion.GenerationInstrumentation"
        ndk { abiFilters += "arm64-v8a" }
        if (mnnRoot != null) {
            externalNativeBuild { cmake {
                arguments += "-DMNN_ROOT=$mnnRoot"
                targets += "localdiffusion"
            } }
        }
    }
    buildFeatures { compose = true }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    if (mnnRoot != null) {
        externalNativeBuild { cmake { path = file("../native/CMakeLists.txt") } }
        ndkVersion = "27.1.12297006"
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.10.00")
    implementation(composeBom)
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.foundation:foundation")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
    testImplementation("junit:junit:4.13.2")
}
