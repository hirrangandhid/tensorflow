# DOCSIS Gateway Anomaly Detection — C++ Inference Engine

C++ port of the Python `AnomalyInferenceEngine` from `inference_app.ipynb`.  
Produces **numerically identical** output to the Python notebook.

---

## File Structure

```
anomaly_detection/
├── anomaly_detection.h        # Engine class + data structures
├── anomaly_detection.cc       # Full implementation
├── anomaly_app_main.cc        # CLI entry point (streaming inference)
├── CMakeLists.txt             # CMake build
├── BUILD                      # Bazel build
├── nlohmann_json/
│   └── json.hpp               # vendored single-header JSON library
└── README.md
```

---

## Python → C++ Mapping

| Python | C++ |
|---|---|
| `AnomalyInferenceEngine.__init__()` | `AnomalyInferenceEngine(config_path)` |
| `engine.process_reading(raw, mac)` | `engine.ProcessReading(reading)` |
| `engine.process_batch(df)` | Loop `ProcessReading()` per row |
| `engine.reset_device(mac)` | `engine.ResetDevice(mac)` |
| `apply_minmax_scaler()` | `AnomalyInferenceEngine::ApplyMinMaxScaler()` |
| `run_dense_inference()` | `AnomalyInferenceEngine::RunDenseInference()` |
| `run_lstm_inference()` | `AnomalyInferenceEngine::RunLstmInference()` |

---

## Numerical Equivalence

Every operation matches Python exactly:

| Operation | Python | C++ |
|---|---|---|
| MinMax scaling | `clip((x - data_min) * scale, 0, 1)` | `ClampF((x - data_min[i]) * scale[i], 0, 1)` |
| Rolling std | `np.std(window, ddof=1)` | `StdDdof1()` — Bessel's correction |
| cpu_delta | `cpu_t - cpu_history[-1]` (0 on first) | same |
| Dense MSE | `mean((inp - out)²)` | sum of sq diff / n_features |
| LSTM MSE | `mean((inp - out)²)` over all seq×feat elements | sum / (seq_len × n_features) |
| Anomaly type | Both > CPU > Memory > Normal | identical precedence |

---

## Build

### Prerequisites

- CMake ≥ 3.16 (or Bazel 4.1.0)
- TFLite built from source (in the `tensorflow/` tree above this folder)
- [nlohmann/json](https://github.com/nlohmann/json) single header — place at `nlohmann_json/json.hpp`

### CMake (host / x86)

```bash
cd tensorflow/lite/examples/anomaly_detection
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
```

### CMake (ARM64 target — BCM7712)

```bash
cmake -DCMAKE_TOOLCHAIN_FILE=<bstorm_tree>/bstorm/cmake/stbgcc.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DBSTORM_CHIP=7712 -DBSTORM_CHIP_REV=c0 \
      ..
cmake --build . -j$(nproc)
```

### Bazel

```bash
# From the tensorflow/ repository root
bazel build //tensorflow/lite/examples/anomaly_detection:anomaly_app
```

---

## Usage

```
anomaly_app [options]

General options:
  --config,  -c <path>   inference_config.json  (default: inference_config.json)
  --threads, -t <n>      TFLite thread count     (default: 1)
  --delegate-path,  -d <path>
                         External hardware delegate .so
                         (e.g. /usr/lib/libbstorm_external_delegate.so)
  --delegate-options <key:val;key:val>
                         Options forwarded to the delegate
  --verbose, -v          Print per-reading details to stderr
  --check-libs           Check shared-library availability and exit
  --help,    -h          Print usage

File-watcher mode  (default — no --input; run in bg with '&'):
  --watch-path  <path>   Device CSV file to monitor
                         (default: /rdklogs/logs/system_stats_data.csv)
  --result-path <path>   Output file; result rows are appended
                         (default: /rdklogs/logs/anomaly_results.csv)
  --poll-interval <ms>   inotify safety-timeout on Linux; stat-poll interval
                         on non-Linux (default: 60000 ms)
  --skip-existing        Seek to EOF on startup; skip rows already present

Batch mode  (one-shot; requires --input):
  --input,  -i <path>    Input CSV file
  --output, -o <path>    Output CSV file (default: stdout)
```

### CSV header auto-detection

Both modes detect the CSV format automatically by peeking at the first byte:

| First byte | Interpretation |
|---|---|
| Letter (`t`, `C`, …) | Named-column header row present |
| Digit (`2`, `0`, …) | No header; fixed positional order used |

Positional order (device CSV from `/rdklogs/logs/system_stats_data.csv`):
```
col 0: timestamp          col 1: CMMAC
col 2: USED_CPU_ATOM      col 3: USED_MEM_ATOM_kB
col 4: LOAD_AVG_ATOM      col 5: AvailMem_kB
col 6: FreeMem_kB         col 7: SlabMem_kB
col 8: 2G_Clients_Count   col 9: 5G_Clients_Count
col10: 6G_Clients_Count
```

### Timestamp formats accepted

```
YYYY-MM-DDTHH:MM:SS[.mmm]   ISO 8601
YYYY-MM-DD HH:MM:SS[.mmm]   space separator
YYYY-MM-DD-HH:MM:SS[.mmm]   device /rdklogs format
```

`hour_of_day` and `day_of_week` are derived from the timestamp automatically.

### Output CSV columns

```
timestamp, CMMAC,
dense_cpu_mse, dense_mem_mse, dense_cpu_flag, dense_mem_flag,
dense_cpu_sev, dense_mem_sev,
[lstm_cpu_mse, lstm_mem_mse, lstm_cpu_flag, lstm_mem_flag,
 lstm_cpu_sev, lstm_mem_sev,]    ← only when LSTM models are loaded
anomaly_type
```

`anomaly_type` ∈ `{Normal, CPU, Memory, Both}` — identical to Python output.

### File-watcher (live device data)

```bash
# Start in the background; kill with SIGTERM when done
anomaly_app \
  --config  /etc/anomaly_detection/inference_config.json \
  --watch-path  /rdklogs/logs/system_stats_data.csv \
  --result-path /rdklogs/logs/anomaly_results.csv &

echo "anomaly_app PID: $!"
```

### Batch (one-shot for testing / validation)

```bash
./anomaly_app \
  --config inference_config.json \
  --input  cleaned_data_50mac_xb10_8_3p5s1.csv \
  --output results.csv \
  --verbose
```

### procd service snippet (OpenWrt / RDK)

```sh
start_service() {
    procd_open_instance
    procd_set_param command /usr/bin/anomaly_app \
        --config  /etc/anomaly_detection/inference_config.json \
        --watch-path  /rdklogs/logs/system_stats_data.csv \
        --result-path /rdklogs/logs/anomaly_results.csv
    procd_set_param respawn
    procd_close_instance
}
```

---

## Deploying on Broadcom BSTM Hardware (BSTORM)

Replace the standard TFLite interpreter with the BSTORM external delegate
to run inference on the BSTM hardware core:

```cpp
// In anomaly_detection.cc LoadInterpreter(), after building the interpreter:
auto delegate = TfLiteExternalDelegateCreate(options);
interp_out->ModifyGraphWithDelegate(delegate);
```

Or load the BSTORM delegate via the external delegate API at runtime:
```bash
# The bstorm_example pattern applies directly — see BStormGettingStarted.pdf
env LD_LIBRARY_PATH=./ ./anomaly_app \
    --config inference_config.json \
    --input  telemetry.csv
```

See `BStormGettingStarted.pdf` → *Using TFLite API* for full instructions.
