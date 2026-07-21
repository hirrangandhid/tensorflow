# TensorFlow Lite Anomaly Prediction Application

TCN autoencoder-based inference application for **RDKB gateway routers** that predicts CPU and memory anomalies using sliding window temporal analysis. Supports hardware acceleration via the **bstorm delegate** for NPU offloading.

---

## Features

- **Windowed TCN Inference**: Uses a 30-timestep sliding window for temporal pattern recognition
- **Dual Model Support**: Separate models for CPU (11 features) and memory (20 features) anomaly detection
- **bstorm Delegate**: Hardware acceleration for RDKB devices via external delegate API
- **Batch & Watcher Modes**: Process CSV files or continuously monitor telemetry in real-time
- **Configurable Thresholds**: JSON configuration for model paths, thresholds, and feature scaling

---

## Quick Start

### Prerequisites

- TensorFlow Lite C++ library
- C++17 compiler (GCC 9+ or Clang 10+)
- nlohmann/json (single-header, included or installed)

### Build with CMake

```bash
# From the anomaly_prediction directory
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
```

### Build with Bazel

```bash
# From the tensorflow root
bazel build //tensorflow/lite/examples/anomaly_prediction:anomaly_prediction_app
```

---

## Usage

### Batch Mode (CSV Processing)

```bash
./anomaly_prediction_app \
  --config anomaly_prediction_config.json \
  --input telemetry_data.csv \
  --output predictions.csv \
  --verbose
```

### Watcher Mode (Real-time Monitoring)

```bash
./anomaly_prediction_app \
  --config anomaly_prediction_config.json \
  --watch /tmp/telemetry_input.csv \
  --output /tmp/predictions.csv
```

### With bstorm Delegate (RDKB NPU Acceleration)

```bash
./anomaly_prediction_app \
  --config anomaly_prediction_config.json \
  --input data.csv \
  --delegate-path /usr/lib/libbstorm_external_delegate.so \
  --delegate-options "bstm:1;bstm-client-mode:0"
```

---

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON configuration file (required) |
| `--input PATH` | Input CSV file for batch processing |
| `--output PATH` | Output CSV for predictions (default: stdout) |
| `--watch PATH` | Watch file for continuous inference |
| `--delegate-path PATH` | External delegate shared library |
| `--delegate-options OPTS` | Delegate options (semicolon-separated key:value) |
| `--threads N` | Number of inference threads (default: 1) |
| `--verbose` | Enable verbose logging |
| `--help` | Show usage information |

---

## Configuration File

The JSON configuration specifies model paths, thresholds, and feature scaling:

```json
{
  "cpu_model_file": "tcn_cpu_anomaly_model_v2.tflite",
  "mem_model_file": "tcn_mem_anomaly_model_v2.tflite",
  "cpu_threshold": 0.007692869286984205,
  "mem_threshold": 0.005448348354548216,
  "window_size": 30,
  "warmup_samples": 30,
  "cpu_features": ["USED_CPU_ATOM", "LOAD_AVG_ATOM", ...],
  "mem_features": ["mem_utilization", "avail_to_total", ...],
  "cpu_scaler": {
    "min": [0.0, 0.0, ...],
    "max": [100.0, 10.0, ...]
  },
  "mem_scaler": {
    "min": [0.0, 0.0, ...],
    "max": [1.0, 1.0, ...]
  }
}
```

---

## Input CSV Format

The input CSV must contain a `timestamp` column and the feature columns listed in the configuration. Example:

```csv
timestamp,USED_CPU_ATOM,LOAD_AVG_ATOM,FREE_MEM_ATOM,AVAILABLE_MEM_ATOM,...
2024-01-15 10:00:00,45.2,1.23,512000,600000,...
2024-01-15 10:05:00,48.1,1.45,508000,595000,...
```

### Required Columns

| Column | Description |
|--------|-------------|
| `timestamp` | ISO 8601 datetime string |
| `USED_CPU_ATOM` | CPU usage percentage (0-100) |
| `LOAD_AVG_ATOM` | System load average |
| `FREE_MEM_ATOM` | Free memory in KB |
| `AVAILABLE_MEM_ATOM` | Available memory in KB |
| `TOTAL_MEM_ATOM` | Total memory in KB |
| `total_clients` | Number of connected clients |

Additional columns may be present; unused columns are ignored.

---

## Output Format

The output CSV contains predictions for each input row:

```csv
timestamp,cpu_mse,cpu_anomaly,mem_mse,mem_anomaly
2024-01-15 10:30:00,0.0045,0,0.0032,0
2024-01-15 10:35:00,0.0089,1,0.0041,0
```

| Column | Description |
|--------|-------------|
| `cpu_mse` | CPU model reconstruction error |
| `cpu_anomaly` | 1 if anomaly detected, 0 otherwise |
| `mem_mse` | Memory model reconstruction error |
| `mem_anomaly` | 1 if anomaly detected, 0 otherwise |

---

## Model Architecture

The application uses TCN (Temporal Convolutional Network) autoencoder models trained on normal RDKB gateway telemetry:

- **CPU Model**: Input shape `[1, 30, 11]` → Output shape `[1, 30, 11]`
- **Memory Model**: Input shape `[1, 30, 20]` → Output shape `[1, 30, 20]`

Anomalies are detected when the Mean Squared Error (MSE) between input and reconstructed output exceeds the configured threshold.

---

## Cross-Compilation for RDKB

```bash
# Using CMake with toolchain file
cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/rdkb-toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      ..

# Using Bazel with config
bazel build --config=aarch64 \
  //tensorflow/lite/examples/anomaly_prediction:anomaly_prediction_app
```

---

## Deploying to RDKB Device

1. Copy the binary and models to the device:
   ```bash
   scp anomaly_prediction_app root@<device>:/usr/bin/
   scp *.tflite root@<device>:/etc/anomaly_models/
   scp anomaly_prediction_config.json root@<device>:/etc/anomaly_models/
   ```

2. Run with bstorm delegate:
   ```bash
   /usr/bin/anomaly_prediction_app \
     --config /etc/anomaly_models/anomaly_prediction_config.json \
     --watch /tmp/telemetry.csv \
     --output /tmp/predictions.csv \
     --delegate-path /usr/lib/libbstorm_external_delegate.so
   ```

---

## Files

| File | Description |
|------|-------------|
| `anomaly_prediction.h` | Header with engine class and data structures |
| `anomaly_prediction.cc` | Inference implementation |
| `anomaly_prediction_main.cc` | CLI application |
| `anomaly_prediction_config.json` | Default configuration |
| `BUILD` | Bazel build rules |
| `CMakeLists.txt` | CMake build configuration |

---

## License

Apache 2.0 — see the TensorFlow [LICENSE](../../../../LICENSE) file.
