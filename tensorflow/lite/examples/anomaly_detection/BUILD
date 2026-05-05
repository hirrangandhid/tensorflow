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

# ── mcp_rbus IPC framework (header-only) ─────────────────────────────────────
# mcp_rbus_framework.h must be a declared Bazel target so Bazel symlinks it
# into the exec root for every action that depends on it.  Without this rule
# the header is absent from the exec root and the compiler cannot find it,
# which cascades into thousands of undefined-symbol errors.
cc_library(
    name = "mcp_rbus_framework",
    hdrs = ["mcp_rbus_framework.h"],
    deps = [":nlohmann_json"],
    # -finput-charset=UTF-8: source files use Unicode box-drawing characters.
    # -DHAVE_RBUS: rbus IPC is always enabled on RDK devices.
    copts = ["-finput-charset=UTF-8", "-DHAVE_RBUS"],
    linkopts = ["-lrbus"],
    visibility = ["//visibility:public"],
)

# ── Inference engine library ──────────────────────────────────────────────────
cc_library(
    name = "anomaly_detection_lib",
    srcs = ["anomaly_detection.cc"],
    hdrs = ["anomaly_detection.h"],
    copts = ["-finput-charset=UTF-8"],
    deps = [
        ":nlohmann_json",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        # Provides TfLiteDelegateUniquePtr without the full TF runtime.
        # The flex delegate itself is loaded at runtime via dlopen.
        "//tensorflow/lite/delegates/utils:simple_delegate",
        # External delegate API — TfLiteExternalDelegateCreate/Delete for hardware accelerators.
        "//tensorflow/lite/delegates/external:external_delegate",
    ],
    linkopts = ["-ldl"],  # for dlopen / dlsym (flex delegate)
)

# ── Executable ────────────────────────────────────────────────────────────────
cc_binary(
    name = "anomaly_app",
    srcs = ["anomaly_app_main.cc"],
    copts = ["-finput-charset=UTF-8", "-DHAVE_RBUS"],
    deps = [
        ":anomaly_detection_lib",
        ":mcp_rbus_framework",
    ],
)

# ── remediation_agent ─────────────────────────────────────────────────────────
# Standalone MQTT-to-rbus bridge.  Subscribes to anomaly alerts from any
# on-device app via rbus and forwards them to the MQTT broker; receives
# corrective actions back and executes them via WebPA / TR-181.
#
# Runtime deps (provided by the Yocto sysroot / recipe DEPENDS):
#   libmosquitto  — Eclipse Mosquitto MQTT client
#   libcurl       — WebPA HTTP calls
#   librbus       — RDK bus (always enabled)
#
# Build:
#   bazel build //tensorflow/lite/examples/anomaly_detection:remediation_agent
cc_binary(
    name = "remediation_agent",
    srcs = ["remediation_agent.cc"],
    copts = ["-finput-charset=UTF-8", "-DHAVE_RBUS"],
    linkopts = [
        "-lmosquitto",
        "-lcurl",
    ],
    deps = [
        ":mcp_rbus_framework",
    ],
)
