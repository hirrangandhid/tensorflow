# BUILD — Bazel build rules for the anomaly detection application
# ─────────────────────────────────────────────────────────────────────────────
# Usage (from the tensorflow/ root, after activating the Python virtual env):
#
#   make -C bstorm/core run          # optional: verify core unit tests pass
#   make bstorm_example              # verify Bazel is functional
#
#   # Build for host (x86)
#   bazel build //tensorflow/lite/examples/anomaly_detection:anomaly_app
#
#   # Build for target (ARM64, e.g. BCM7712)
#   bazel build \
#     --config=aarch64 \
#     //tensorflow/lite/examples/anomaly_detection:anomaly_app
# ─────────────────────────────────────────────────────────────────────────────

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

# ── nlohmann/json interface library ──────────────────────────────────────────
# Uses the vendored single-header copy placed at
#   tensorflow/lite/examples/anomaly_detection/nlohmann_json/json.hpp
cc_library(
    name = "nlohmann_json",
    hdrs = ["nlohmann_json/json.hpp"],
    includes = ["."],
    visibility = ["//visibility:public"],
)

# ── Inference engine library ──────────────────────────────────────────────────
cc_library(
    name = "anomaly_detection_lib",
    srcs = ["anomaly_detection.cc"],
    hdrs = ["anomaly_detection.h"],
    deps = [
        ":nlohmann_json",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/core:interpreter_builder",
        # Flex delegate — required for LSTM models (SELECT_TF_OPS / UnidirectionalSequenceLSTM)
        "//tensorflow/lite/delegates/flex:delegate",
    ],
)

# ── Executable ────────────────────────────────────────────────────────────────
cc_binary(
    name = "anomaly_app",
    srcs = ["anomaly_app_main.cc"],
    deps = [
        ":anomaly_detection_lib",
    ],
)
