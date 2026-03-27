# CMakeLists.txt — Anomaly Detection C++ Application
# ─────────────────────────────────────────────────────────────────────────────
# Builds the DOCSIS gateway anomaly detection inference application.
# Requires TFLite + nlohmann/json.
#
# Build (host / x86):
#   mkdir -p build && cd build
#   cmake ..
#   cmake --build . -j$(nproc)
#   ./anomaly_app --config ../../../_SYSTEM_STATS/inference_config.json \
#                 --input  ../../../_SYSTEM_STATS/cleaned_data_50mac_xb10_8_3p5s1.csv \
#                 --verbose
#
# Cross-compile for ARM64 (using the toolchain from BStormGettingStarted):
#   cmake -DCMAKE_TOOLCHAIN_FILE=<path>/bstorm/cmake/stbgcc.cmake \
#         -DCMAKE_BUILD_TYPE=Release ..
# ─────────────────────────────────────────────────────────────────────────────

cmake_minimum_required(VERSION 3.16)
project(anomaly_detection CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Find TensorFlow Lite ──────────────────────────────────────────────────────
# Locate the TFLite source tree two levels above this example directory.
set(TFLITE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.."
    CACHE PATH "Root of the tensorflow source tree")

set(TFLITE_SOURCE_DIR "${TFLITE_ROOT}/tensorflow/lite")

if(NOT EXISTS "${TFLITE_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "TFLite CMakeLists.txt not found at ${TFLITE_SOURCE_DIR}. "
    "Set TFLITE_ROOT to the tensorflow repository root.")
endif()

# Build TFLite as a static library
set(TFLITE_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
add_subdirectory("${TFLITE_SOURCE_DIR}" tflite EXCLUDE_FROM_ALL)

# ── nlohmann/json ─────────────────────────────────────────────────────────────
# Use the vendored copy in anomaly_detection/nlohmann_json/ if available,
# otherwise fall back to the system package or FetchContent.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nlohmann_json/json.hpp")
  add_library(nlohmann_json INTERFACE)
  target_include_directories(nlohmann_json INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}")
else()
  find_package(nlohmann_json QUIET)
  if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      nlohmann_json
      URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
      DOWNLOAD_NO_EXTRACT TRUE
      DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/nlohmann_json"
    )
    FetchContent_MakeAvailable(nlohmann_json)
    add_library(nlohmann_json INTERFACE)
    target_include_directories(nlohmann_json INTERFACE
      "${CMAKE_BINARY_DIR}")
  endif()
endif()

# ── Inference engine library ──────────────────────────────────────────────────
add_library(anomaly_detection_lib STATIC
  anomaly_detection.cc
)

target_include_directories(anomaly_detection_lib PUBLIC
  # Expose the examples directory so includes resolve correctly
  "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(anomaly_detection_lib PUBLIC
  tensorflow-lite
  nlohmann_json
  # Flex delegate — required for LSTM .tflite models that use SELECT_TF_OPS.
  # These models contain UnidirectionalSequenceLSTM which is not a TFLite builtin op.
  # On the target device, libtensorflowlite_flex.so must be present alongside the binary.
  tensorflowlite_flex
)

# ── anomaly_app executable ────────────────────────────────────────────────────
add_executable(anomaly_app
  anomaly_app_main.cc
)

target_include_directories(anomaly_app PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(anomaly_app
  anomaly_detection_lib
)

# ── Optional: Release optimisation flags ─────────────────────────────────────
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(anomaly_detection_lib PRIVATE -O3 -DNDEBUG)
  target_compile_options(anomaly_app           PRIVATE -O3 -DNDEBUG)
endif()
