# GPT-2 Inference Application for CPE/RDKB Devices

A C++ inference engine for running GPT-2 language models on edge devices like
RDKB gateways with optional NPU acceleration via the BStorm delegate.

## Overview

This application provides:

- **Interactive text generation** from user prompts
- **Single-shot inference** for scripted usage
- **Hardware acceleration** via TFLite external delegates (BStorm NPU)
- **Configurable generation parameters** (temperature, top-k, top-p sampling)
- **Lightweight deployment** optimized for embedded Linux systems

## Directory Structure

```
gpt2_inference/
├── gpt2_inference.h       # Header file with API definitions
├── gpt2_inference.cc      # Core inference engine implementation
├── gpt2_app_main.cc       # Command-line application
├── gpt2_config.json       # Model configuration
├── CMakeLists.txt         # CMake build configuration
├── BUILD                  # Bazel build rules
├── nlohmann_json/         # JSON library (vendored)
│   └── json.hpp
└── README.md              # This file
```

## Prerequisites

- TensorFlow Lite source tree
- C++17 compatible compiler
- CMake 3.16+ or Bazel
- GPT-2 TFLite model (`gpt2_64-8bits.tflite`)

## Building

### Using CMake (Host/x86)

```bash
# From the gpt2_inference directory
mkdir -p build && cd build
cmake ..
cmake --build . -j$(nproc)
```

### Using CMake (Cross-compile for ARM64)

```bash
cmake -DCMAKE_TOOLCHAIN_FILE=<path>/bstorm/cmake/stbgcc.cmake \
      -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
```

### Using Bazel

```bash
# From tensorflow root
bazel build //tensorflow/lite/examples/gpt2_inference:gpt2_app

# For ARM64 target
bazel build --config=aarch64 //tensorflow/lite/examples/gpt2_inference:gpt2_app
```

## Usage

### Interactive Mode

Start an interactive chat session:

```bash
./gpt2_app --config gpt2_config.json
```

Example session:
```
╔═══════════════════════════════════════════════════════════════════════════╗
║                    GPT-2 Text Generation Engine                            ║
║                     For CPE/RDKB Edge Devices                              ║
╚═══════════════════════════════════════════════════════════════════════════╝

Model loaded successfully!
Type your prompts below. Type 'quit' or 'exit' to end.

You: What is machine learning?

GPT-2: Machine learning is a type of artificial intelligence that allows computers
to learn from data without being explicitly programmed...

You: quit
Goodbye!
```

### Single Prompt Mode

Generate a response to a single prompt and exit:

```bash
./gpt2_app --config gpt2_config.json --prompt "Explain neural networks"
```

### With NPU Acceleration (BStorm Delegate)

```bash
./gpt2_app --config gpt2_config.json \
           --delegate-path /usr/lib/libbstorm_external_delegate.so \
           --delegate-options "bstm:1;dynamic-tensors:1"
```

### Device Deployment (model in /nvram or /tmp)

```bash
# Model stored in /nvram (persistent storage)
./gpt2_app --config /etc/gpt2/gpt2_config.json \
           --model /nvram/gpt2_64-8bits.tflite

# Model stored in /tmp (RAM disk)
./gpt2_app -c /etc/gpt2/gpt2_config.json \
           -M /tmp/gpt2_64-8bits.tflite

# Full example with NPU on RDKB device
./gpt2_app --config /etc/gpt2/gpt2_config.json \
           --model /nvram/gpt2_64-8bits.tflite \
           --delegate-path /usr/lib/libbstorm_external_delegate.so \
           --delegate-options "bstm:1;dynamic-tensors:1" \
           --prompt "What is the network status?"
```

## Command-Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--config` | `-c` | Path to model config JSON |
| `--model` | `-M` | Path to TFLite model (overrides config) |
| `--prompt` | `-p` | Single prompt (non-interactive) |
| `--max-tokens` | `-m` | Maximum tokens to generate |
| `--temperature` | `-T` | Sampling temperature (0.0-1.0) |
| `--top-k` | | Top-k sampling (0=disabled) |
| `--top-p` | | Top-p nucleus sampling |
| `--greedy` | | Use greedy decoding |
| `--threads` | `-t` | TFLite thread count |
| `--delegate-path` | `-d` | External delegate library |
| `--delegate-options` | | Delegate configuration |
| `--verbose` | `-v` | Enable debug output |
| `--stats` | | Show statistics on exit |
| `--help` | `-h` | Show help message |

## Configuration File

The `gpt2_config.json` file contains model and generation settings:

```json
{
  "model_file": "gpt2_64-8bits.tflite",
  "vocab_file": "",
  "max_seq_length": 64,
  "vocab_size": 50257,
  
  "generation": {
    "max_new_tokens": 50,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "do_sample": true,
    "repetition_penalty": 1.1,
    "eos_token_id": 50256
  }
}
```

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 50 | Maximum tokens to generate |
| `temperature` | 0.7 | Sampling randomness (0=deterministic) |
| `top_k` | 40 | Keep top-k tokens (0=disabled) |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `do_sample` | true | Enable sampling (false=greedy) |
| `repetition_penalty` | 1.1 | Penalty for repeated tokens |
| `eos_token_id` | 50256 | End-of-sequence token ID |

## API Usage

For integration into other applications:

```cpp
#include "gpt2_inference.h"

// Initialize engine
tflite::gpt2::GPT2InferenceEngine engine(
    "gpt2_config.json",
    "/usr/lib/libbstorm_external_delegate.so",  // optional delegate
    "bstm:1",                                    // delegate options
    2,                                           // threads
    false                                        // verbose
);

// Check initialization
if (!engine.IsReady()) {
    std::cerr << "Error: " << engine.GetLastError() << std::endl;
    return 1;
}

// Generate text
auto result = engine.Generate("What is the weather like?");

if (result.success) {
    std::cout << result.generated_text << std::endl;
    std::cout << "Generated " << result.tokens_generated << " tokens in "
              << result.inference_time_ms << " ms" << std::endl;
}
```

## BStorm NPU Delegate Options

For RDKB devices with Broadcom NPU:

| Option | Description |
|--------|-------------|
| `bstm:1` | Enable BSTM core |
| `bstm-client-mode:0` | Direct mode (vs client) |
| `dynamic-tensors:1` | Enable dynamic tensor support |

Example:
```bash
--delegate-options "bstm:1;bstm-client-mode:0;dynamic-tensors:1"
```

## Performance Notes

- **Quantized models** (8-bit) run faster and use less memory
- **NPU acceleration** can provide 2-10x speedup depending on operations
- **Sequence length** affects memory usage and inference time
- Use `--threads` to leverage multi-core CPUs when NPU is not available

## Troubleshooting

### Model not found
```
Error: Failed to load TFLite model: gpt2_64-8bits.tflite
```
Ensure the model file path in `gpt2_config.json` is correct.

### Delegate initialization failed
```
Warning: Failed to apply external delegate
```
Check that the delegate library exists and is compatible with your platform.

### Out of memory
Reduce `max_seq_length` in the config or use a smaller model.

## License

Apache License 2.0 - See LICENSE file for details.

## See Also

- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [BStorm Delegate Documentation](../bstorm/README.md)
- [Anomaly Detection Example](../anomaly_detection/README.md)
