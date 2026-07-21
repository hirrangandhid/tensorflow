# BUILD — Bazel build rules for the anomaly prediction application
# ─────────────────────────────────────────────────────────────────────────────
# TCN autoencoder inference for RDKB gateway anomaly prediction.
# Supports bstorm delegate for NPU acceleration.
#
# Usage (from the tensorflow/ root):
#
#   # Build for host (x86)
#   bazel build //tensorflow/lite/examples/anomaly_prediction:anomaly_prediction_app
#
#   # Build for RDKB target (ARM64)
#   bazel build \
#     --config=aarch64 \
#     //tensorflow/lite/examples/anomaly_prediction:anomaly_prediction_app
#
#   # Build with bstorm delegate support
#   bazel build \
#     --config=aarch64 \
#     --define=use_bstorm=true \
#     //tensorflow/lite/examples/anomaly_prediction:anomaly_prediction_app
# ─────────────────────────────────────────────────────────────────────────────

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

# ── nlohmann/json interface library ──────────────────────────────────────────
cc_library(
    name = "nlohmann_json",
    hdrs = ["nlohmann_json/json.hpp"],
    includes = ["."],
    visibility = ["//visibility:public"],
)

# ── Prediction engine library ─────────────────────────────────────────────────
cc_library(
    name = "anomaly_prediction_lib",
    srcs = ["anomaly_prediction.cc"],
    hdrs = ["anomaly_prediction.h"],
    deps = [
        ":nlohmann_json",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "//tensorflow/lite/delegates/external:external_delegate",
    ],
    linkopts = ["-ldl"],
)

# ── Main executable ───────────────────────────────────────────────────────────
cc_binary(
    name = "anomaly_prediction_app",
    srcs = ["anomaly_prediction_main.cc"],
    deps = [
        ":anomaly_prediction_lib",
    ],
)

# ── Test target ───────────────────────────────────────────────────────────────
cc_test(
    name = "anomaly_prediction_test",
    srcs = ["anomaly_prediction_test.cc"],
    data = [
        "testdata/test_input.csv",
        "testdata/anomaly_prediction_config.json",
    ],
    deps = [
        ":anomaly_prediction_lib",
        "@com_google_googletest//:gtest_main",
    ],
)
