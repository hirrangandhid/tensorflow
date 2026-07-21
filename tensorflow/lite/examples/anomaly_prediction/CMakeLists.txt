# CMakeLists.txt — Anomaly Prediction C++ Application
# ─────────────────────────────────────────────────────────────────────────────
# Builds the RDKB gateway anomaly prediction inference application.
# TCN autoencoder models with sliding window inference.
#
# Build (host / x86):
#   mkdir -p build && cd build
#   cmake ..
#   cmake --build . -j$(nproc)
#   ./anomaly_prediction_app --config ../anomaly_prediction_config.json \
#                            --input  test_data.csv \
#                            --verbose
#
# Cross-compile for RDKB ARM64:
#   cmake -DCMAKE_TOOLCHAIN_FILE=<path>/bstorm/cmake/stbgcc.cmake \
#         -DCMAKE_BUILD_TYPE=Release ..
#
# With bstorm delegate:
#   ./anomaly_prediction_app --config config.json \
#                            --delegate-path /usr/lib/libbstorm_external_delegate.so \
#                            --delegate-options "bstm:1;bstm-client-mode:0"
# ─────────────────────────────────────────────────────────────────────────────

cmake_minimum_required(VERSION 3.16)
project(anomaly_prediction CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Options ───────────────────────────────────────────────────────────────────
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(ANOMALY_PREDICTION_ENABLE_TESTS "Enable unit tests" OFF)

# ── Find TensorFlow Lite ──────────────────────────────────────────────────────
# Locate the TFLite source tree relative to this example directory.
# Expected path: tensorflow/tensorflow/lite/examples/anomaly_prediction
set(TFLITE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.."
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
# Use the vendored copy in nlohmann_json/ if available
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nlohmann_json/json.hpp")
  add_library(nlohmann_json INTERFACE)
  target_include_directories(nlohmann_json INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}")
else()
  # Try system package
  find_package(nlohmann_json QUIET)
  if(NOT nlohmann_json_FOUND)
    # Fetch from GitHub
    include(FetchContent)
    FetchContent_Declare(
      nlohmann_json
      URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
      DOWNLOAD_NO_EXTRACT TRUE
      DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/nlohmann_json"
    )
    FetchContent_MakeAvailable(nlohmann_json)
    add_library(nlohmann_json INTERFACE)
    target_include_directories(nlohmann_json INTERFACE "${CMAKE_BINARY_DIR}")
  endif()
endif()

# ── Prediction engine library ─────────────────────────────────────────────────
add_library(anomaly_prediction_lib STATIC
  anomaly_prediction.cc
)

target_include_directories(anomaly_prediction_lib PUBLIC
  "${TFLITE_ROOT}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(anomaly_prediction_lib PUBLIC
  tensorflow-lite
  nlohmann_json
)

# ── Main executable ───────────────────────────────────────────────────────────
add_executable(anomaly_prediction_app
  anomaly_prediction_main.cc
)

target_include_directories(anomaly_prediction_app PRIVATE
  "${TFLITE_ROOT}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(anomaly_prediction_app
  anomaly_prediction_lib
)

# ── Release optimization ──────────────────────────────────────────────────────
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(anomaly_prediction_lib PRIVATE -O3 -DNDEBUG)
  target_compile_options(anomaly_prediction_app PRIVATE -O3 -DNDEBUG)
endif()

# ── Install targets ───────────────────────────────────────────────────────────
install(TARGETS anomaly_prediction_app
  RUNTIME DESTINATION bin
)

install(TARGETS anomaly_prediction_lib
  ARCHIVE DESTINATION lib
)

install(FILES anomaly_prediction.h
  DESTINATION include/tensorflow/lite/examples/anomaly_prediction
)

# ── Unit tests (optional) ─────────────────────────────────────────────────────
if(ANOMALY_PREDICTION_ENABLE_TESTS)
  enable_testing()
  find_package(GTest QUIET)
  
  if(GTest_FOUND)
    add_executable(anomaly_prediction_test
      anomaly_prediction_test.cc
    )
    target_link_libraries(anomaly_prediction_test
      anomaly_prediction_lib
      GTest::GTest
      GTest::Main
    )
    add_test(NAME anomaly_prediction_test COMMAND anomaly_prediction_test)
  endif()
endif()

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "Anomaly Prediction Build Configuration")
message(STATUS "======================================")
message(STATUS "  TFLite root:     ${TFLITE_ROOT}")
message(STATUS "  Build type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  Tests enabled:   ${ANOMALY_PREDICTION_ENABLE_TESTS}")
message(STATUS "")
