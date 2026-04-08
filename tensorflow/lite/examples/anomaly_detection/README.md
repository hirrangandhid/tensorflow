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

  --config,   -c <path>   inference_config.json  (default: inference_config.json)
  --input,    -i <path>   Input CSV file          (default: stdin)
  --output,   -o <path>   Output CSV file         (default: stdout)
  --threads,  -t <n>      TFLite thread count     (default: 1)
  --delegate-path,  -d <path>
                          External hardware delegate .so
  --delegate-options <key:val;key:val>
                          Options forwarded to the delegate
  --verbose,  -v          Print per-reading details to stderr
  --check-libs            Check shared-library availability and exit
  --help,     -h          Print usage

Daemon mode (long-running file watcher):
  --daemon                Watch watch-path for new rows; run until SIGTERM/SIGINT
  --daemonize             Fork to background (implies --daemon)
  --watch-path  <path>    CSV produced by the data-collection process
                          (default: /rdklogs/logs/system_stats_data.csv)
  --result-path <path>    Output file; result rows are appended
                          (default: /rdklogs/logs/anomaly_results.csv)
  --poll-interval <ms>    On Linux (inotify mode): safety-timeout — wakes even
                          if an inotify event was missed (default: 60000 = 60 s).
                          On non-Linux: active stat-poll interval in ms.
  --skip-existing         Seek to EOF on startup; only process newly-written rows
  --pid-file    <path>    Write daemon PID here (--daemonize only)
```

### Input CSV columns

Same names as `cleaned_data_50mac_xb10_8_3p5s1.csv`:

```
timestamp, CMMAC, USED_CPU_ATOM, LOAD_AVG_ATOM,
USED_MEM_ATOM_kB, AvailMem_kB, FreeMem_kB, SlabMem_kB,
2G_Clients_Count, 5G_Clients_Count, 6G_Clients_Count
```

`hour_of_day` and `day_of_week` columns are optional — if absent they are
derived from the `timestamp` column automatically.

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

### One-shot batch example

```bash
# Run against the full validation dataset (mirrors notebook Section 8)
./anomaly_app \
  --config inference_config.json \
  --input  cleaned_data_50mac_xb10_8_3p5s1.csv \
  --output results.csv \
  --verbose
```

---

## Daemon Mode — Live File Watcher

`anomaly_app` can run as a persistent daemon that monitors the CSV file
written by the on-device telemetry-collection process and produces anomaly
results in real time.

### How it works

1. On startup the engine loads the TFLite models and config once.
2. It opens `/rdklogs/logs/system_stats_data.csv`, reads the header, and
   processes any rows already present (unless `--skip-existing` is set).
3. **On Linux** it uses `inotify` to watch the file's parent directory.
   The process sleeps in `poll()` and is woken by the kernel the instant
   the collector writes — zero polling overhead regardless of the collection
   interval (5 min, 1 hour, etc.).
   `--poll-interval` is a safety-timeout only (default 60 s), used to catch
   any event that might be missed during log rotation.
   **On non-Linux** (e.g. macOS dev builds) it falls back to stat-based
   polling using `--poll-interval` as the sleep interval.
4. When the file grows (or is re-created after rotation) the daemon reads
   the new rows, runs inference, and **appends** results to the result file.
5. File rotation / truncation and temporary disappearance are handled
   automatically.
6. Send `SIGTERM` or `SIGINT` for a clean shutdown.

### Foreground (supervised by systemd / procd)

```bash
./anomaly_app \
  --config /etc/anomaly_detection/inference_config.json \
  --daemon \
  --watch-path  /rdklogs/logs/system_stats_data.csv \
  --result-path /rdklogs/logs/anomaly_results.csv \
  --poll-interval 60000 \
  --verbose
```

### Background (self-daemonizing)

```bash
./anomaly_app \
  --config /etc/anomaly_detection/inference_config.json \
  --daemonize \
  --watch-path  /rdklogs/logs/system_stats_data.csv \
  --result-path /rdklogs/logs/anomaly_results.csv \
  --pid-file    /var/run/anomaly_app.pid
```

### procd service snippet (OpenWrt / RDK)

```sh
start_service() {
    procd_open_instance
    procd_set_param command /usr/bin/anomaly_app \
        --config  /etc/anomaly_detection/inference_config.json \
        --daemon \
        --watch-path  /rdklogs/logs/system_stats_data.csv \
        --result-path /rdklogs/logs/anomaly_results.csv \
        --poll-interval 60000
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
