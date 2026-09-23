# libidr is reached through JNI; keep the binding object and its native methods
# so R8/ProGuard can't rename or strip the symbols the .so resolves against.
-keep class com.offmaps.nav.IdrNative { *; }
-keepclasseswithmembernames class * { native <methods>; }
# ONNX Runtime ships its own consumer rules, but pin the entry points defensively.
-keep class ai.onnxruntime.** { *; }
