# CMakeLists.txt — GPT-2 Inference Application
# ─────────────────────────────────────────────────────────────────────────────
# Builds the GPT-2 text generation application for CPE/RDKB devices.
# Requires TFLite + nlohmann/json.
#
# Build (host / x86):
#   mkdir -p build && cd build
#   cmake ..
#   cmake --build . -j$(nproc)
#   ./gpt2_app --config ../gpt2_config.json --prompt "Hello, world"
#
# Cross-compile for ARM64 (using the toolchain from BStormGettingStarted):
#   cmake -DCMAKE_TOOLCHAIN_FILE=<path>/bstorm/cmake/stbgcc.cmake \
#         -DCMAKE_BUILD_TYPE=Release ..
#
# With BStorm NPU delegate:
#   ./gpt2_app --config gpt2_config.json \
#              --delegate-path /usr/lib/libbstorm_external_delegate.so \
#              --delegate-options "bstm:1;dynamic-tensors:1"
# ─────────────────────────────────────────────────────────────────────────────

cmake_minimum_required(VERSION 3.16)
project(gpt2_inference CXX)

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
# Use the vendored copy in gpt2_inference/nlohmann_json/ if available,
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

# ── GPT-2 Inference engine library ────────────────────────────────────────────
add_library(gpt2_inference_lib STATIC
  gpt2_inference.cc
)

target_include_directories(gpt2_inference_lib PUBLIC
  # Expose the examples directory so includes resolve correctly
  "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(gpt2_inference_lib PUBLIC
  tensorflow-lite
  nlohmann_json
)

# ── gpt2_app executable ───────────────────────────────────────────────────────
add_executable(gpt2_app
  gpt2_app_main.cc
)

target_include_directories(gpt2_app PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(gpt2_app
  gpt2_inference_lib
)

# ── Optional: Release optimisation flags ──────────────────────────────────────
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(gpt2_inference_lib PRIVATE -O3 -DNDEBUG)
  target_compile_options(gpt2_app PRIVATE -O3 -DNDEBUG)
endif()

# ── Installation ──────────────────────────────────────────────────────────────
install(TARGETS gpt2_app DESTINATION bin)
install(FILES gpt2_config.json DESTINATION etc/gpt2)
