# BUILD — Bazel build rules for the GPT-2 inference application
# ─────────────────────────────────────────────────────────────────────────────
# Usage (from the tensorflow/ root, after activating the Python virtual env):
#
#   # Build for host (x86)
#   bazel build //tensorflow/lite/examples/gpt2_inference:gpt2_app
#
#   # Build for target (ARM64, e.g. BCM7712)
#   bazel build \
#     --config=aarch64 \
#     //tensorflow/lite/examples/gpt2_inference:gpt2_app
#
#   # Run on device
#   ./gpt2_app --config gpt2_config.json --prompt "Hello, world"
#
#   # With BStorm NPU delegate
#   ./gpt2_app --config gpt2_config.json \
#              --delegate-path /usr/lib/libbstorm_external_delegate.so \
#              --delegate-options "bstm:1;dynamic-tensors:1"
# ─────────────────────────────────────────────────────────────────────────────

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

# ── nlohmann/json interface library ──────────────────────────────────────────
# Uses the vendored single-header copy placed at
#   tensorflow/lite/examples/gpt2_inference/nlohmann_json/json.hpp
cc_library(
    name = "nlohmann_json",
    hdrs = ["nlohmann_json/json.hpp"],
    includes = ["."],
    visibility = ["//visibility:public"],
)

# ── GPT-2 Inference engine library ────────────────────────────────────────────
cc_library(
    name = "gpt2_inference_lib",
    srcs = ["gpt2_inference.cc"],
    hdrs = ["gpt2_inference.h"],
    deps = [
        ":nlohmann_json",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        # Provides TfLiteDelegateUniquePtr without the full TF runtime.
        "//tensorflow/lite/delegates/utils:simple_delegate",
        # External delegate API — for hardware accelerators (e.g. BStorm NPU).
        "//tensorflow/lite/delegates/external:external_delegate",
    ],
    linkopts = ["-ldl"],  # for dlopen / dlsym
)

# ── Executable ────────────────────────────────────────────────────────────────
cc_binary(
    name = "gpt2_app",
    srcs = ["gpt2_app_main.cc"],
    deps = [
        ":gpt2_inference_lib",
    ],
    data = [
        "gpt2_config.json",
    ],
)

# ── Test target (optional) ────────────────────────────────────────────────────
# cc_test(
#     name = "gpt2_inference_test",
#     srcs = ["gpt2_inference_test.cc"],
#     deps = [
#         ":gpt2_inference_lib",
#         "@com_google_googletest//:gtest_main",
#     ],
# )
