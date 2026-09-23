# Builds libidrjni.so = the JNI bridge + the UNMODIFIED libidr core sources.
# We compile core/*.cpp directly (not a prebuilt .so) so the phone binary is the
# same translation units the ctypes harness and tools/replay build -- one core,
# three frontends, no drift.
cmake_minimum_required(VERSION 3.22.1)
project(idrjni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Repo core/ lives five levels up from this file (app/src/main/cpp -> repo root).
get_filename_component(CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../core" ABSOLUTE)

if(NOT EXISTS "${CORE_DIR}/idr.h")
    message(FATAL_ERROR "libidr core not found at ${CORE_DIR}. "
            "The android/ project must stay inside the OffMaps repo so it can "
            "compile the shared core sources.")
endif()

add_library(idrjni SHARED
    idr_jni.cpp
    ${CORE_DIR}/eskf.cpp
    ${CORE_DIR}/speed_cal.cpp
    ${CORE_DIR}/align.cpp
    ${CORE_DIR}/gnss_quality.cpp
    ${CORE_DIR}/map_match.cpp
    ${CORE_DIR}/vib.cpp
    ${CORE_DIR}/eskf3d.cpp)

target_include_directories(idrjni PRIVATE ${CORE_DIR})
target_compile_options(idrjni PRIVATE -O2 -fPIC)

# libidr uses only libc/libm and <jni.h>; link the NDK log lib for optional traces.
find_library(log-lib log)
target_link_libraries(idrjni ${log-lib})
