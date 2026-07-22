# TensorFlow Lite Anomaly Prediction Application

TCN-based inference application for **RDKB gateway routers** that predicts CPU and memory anomalies using sliding window temporal analysis. Supports both **autoencoder** (reconstruction-based) and **forecaster** (prediction-based) models. Hardware acceleration via the **bstorm delegate** is available for NPU offloading.

---

## Features

- **Dual Model Types**: Supports autoencoder (reconstruction error) and forecaster (prediction error) architectures
- **Windowed TCN Inference**: Uses a 30-timestep sliding window for temporal pattern recognition
- **Dual Signal Support**: Separate models for CPU (11 features) and memory (20 features) anomaly detection
- **bstorm Delegate**: Hardware acceleration for RDKB devices via external delegate API (autoencoder only)
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

The application supports two TCN (Temporal Convolutional Network) model architectures:

### Autoencoder Mode (Default)

Reconstruction-based anomaly detection. The model learns to reconstruct normal telemetry patterns.

- **CPU Model**: Input shape `[1, 30, 11]` → Output shape `[1, 30, 11]`
- **Memory Model**: Input shape `[1, 30, 20]` → Output shape `[1, 30, 20]`
- **Anomaly Detection**: MSE between input window and reconstructed window exceeds threshold
- **NPU Compatible**: Yes (bstorm delegate supported)

```bash
# Run with autoencoder config
./anomaly_prediction_app --config anomaly_prediction_config.json --input data.csv
```

### Forecaster Mode

Prediction-based anomaly detection. The model predicts the next timestep from the current window.

- **CPU Model**: Input shape `[1, 30, 11]` → Output shape `[1, 11]` (predicted t+1)
- **Memory Model**: Input shape `[1, 30, 20]` → Output shape `[1, 20]` (predicted t+1)
- **Anomaly Detection**: MSE between predicted t+1 and actual t+1 exceeds threshold
- **NPU Compatible**: No (requires causal padding, CPU only)

```bash
# Run with forecaster config
./anomaly_prediction_app --config forecaster_config.json --input data.csv
```

**Forecaster workflow:**
1. Receive telemetry reading at time t
2. Compare with prediction made at t-1 (if available)
3. If MSE(predicted, actual) > threshold → anomaly
4. Generate prediction for t+1 using window [t-29...t]

---

## Configuration File

The JSON configuration specifies model type, paths, thresholds, and feature scaling:

### Autoencoder Configuration

```json
{
  "model_type": "autoencoder",
  "cpu_model_file": "tcn_cpu_anomaly_model_v2.tflite",
  "mem_model_file": "tcn_mem_anomaly_model_v2.tflite",
  "cpu_threshold": 0.007692869286984205,
  "mem_threshold": 0.005448348354548216,
  "window_size": 30,
  "warmup_samples": 30,
  "bstorm_compatible": true,
  "cpu_features": ["USED_CPU_ATOM", "LOAD_AVG_ATOM", ...],
  "mem_features": ["mem_utilization", "avail_to_total", ...],
  "cpu_scaler": { "min": [...], "max": [...] },
  "mem_scaler": { "min": [...], "max": [...] }
}
```

### Forecaster Configuration

```json
{
  "model_type": "forecaster",
  "cpu_model_file": "tcn_cpu_forecaster.tflite",
  "mem_model_file": "tcn_mem_forecaster.tflite",
  "cpu_threshold": 0.011065048165619373,
  "mem_threshold": 0.017314743250608444,
  "window_size": 30,
  "warmup_samples": 30,
  "bstorm_compatible": false,
  "cpu_features": ["USED_CPU_ATOM", "LOAD_AVG_ATOM", ...],
  "mem_features": ["mem_utilization", "avail_to_total", ...],
  "cpu_scaler": { "min": [...], "max": [...] },
  "mem_scaler": { "min": [...], "max": [...] }
}
```

| Field | Description |
|-------|-------------|
| `model_type` | `"autoencoder"` or `"forecaster"` (default: autoencoder) |
| `cpu_model_file` | Path to CPU TFLite model |
| `mem_model_file` | Path to Memory TFLite model |
| `cpu_threshold` | MSE threshold for CPU anomaly |
| `mem_threshold` | MSE threshold for Memory anomaly |
| `window_size` | Sliding window size (default: 30) |
| `warmup_samples` | Samples before anomaly detection activates |
| `bstorm_compatible` | Whether model supports NPU acceleration |

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
   scp *_config.json root@<device>:/etc/anomaly_models/
   ```

2. Run with autoencoder model (bstorm NPU acceleration):
   ```bash
   /usr/bin/anomaly_prediction_app \
     --config /etc/anomaly_models/anomaly_prediction_config.json \
     --watch /tmp/telemetry.csv \
     --output /tmp/predictions.csv \
     --delegate-path /usr/lib/libbstorm_external_delegate.so
   ```

3. Run with forecaster model (CPU only):
   ```bash
   /usr/bin/anomaly_prediction_app \
     --config /etc/anomaly_models/forecaster_config.json \
     --watch /tmp/telemetry.csv \
     --output /tmp/predictions.csv
   ```

---

## Files

| File | Description |
|------|-------------|
| `anomaly_prediction.h` | Header with engine class and data structures |
| `anomaly_prediction.cc` | Inference implementation (autoencoder + forecaster) |
| `anomaly_prediction_main.cc` | CLI application |
| `anomaly_prediction_config.json` | Autoencoder configuration |
| `forecaster_config.json` | Forecaster configuration |
| `BUILD` | Bazel build rules |
| `CMakeLists.txt` | CMake build configuration |

---

## License

Apache 2.0 — see the TensorFlow [LICENSE](../../../../LICENSE) file.
