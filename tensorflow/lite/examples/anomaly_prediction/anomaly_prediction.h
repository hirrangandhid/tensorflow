/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================

anomaly_prediction.h
────────────────────────────────────────────────────────────────────────────
C++ inference engine for RDKB gateway telemetry anomaly prediction.

TCN (Temporal Convolutional Network) autoencoder models for CPU and memory
anomaly detection using sliding window approach.
Designed for edge deployment on RDKB routers with bstorm delegate support.

Models supported:
  tcn_cpu_anomaly_model_v2.tflite    CPU TCN AE   input (1, 30, 11)
  tcn_mem_anomaly_model_v2.tflite    Mem TCN AE   input (1, 30, 20)

Window-based inference:
  - Maintains sliding window of 30 timesteps per device
  - Inference runs when window is full
  - MSE computed over entire reconstructed window

Config format (anomaly_prediction_config.json):
  - cpu_model_file: path to CPU .tflite
  - mem_model_file: path to Memory .tflite
  - cpu_threshold: CPU anomaly threshold (MSE)
  - mem_threshold: Memory anomaly threshold (MSE)
  - window_size: number of timesteps (default 30)
  - cpu_features: ordered list of CPU feature names
  - mem_features: ordered list of Memory feature names
  - cpu_scaler / mem_scaler: {min: [], max: []} arrays
============================================================================*/

#ifndef TENSORFLOW_LITE_EXAMPLES_ANOMALY_PREDICTION_ANOMALY_PREDICTION_H_
#define TENSORFLOW_LITE_EXAMPLES_ANOMALY_PREDICTION_ANOMALY_PREDICTION_H_

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/delegates/external/external_delegate.h"

namespace tflite {
namespace anomaly_prediction {

// ── Constants ────────────────────────────────────────────────────────────────
constexpr int kDefaultWindowSize = 30;
constexpr int kCpuFeatureCount = 11;
constexpr int kMemFeatureCount = 20;

// ── Scaler parameters (MinMaxScaler) ─────────────────────────────────────────
struct ScalerParams {
  std::vector<std::string> features;      // ordered feature names
  std::vector<float>       data_min;      // per-feature minimum
  std::vector<float>       data_max;      // per-feature maximum
};

// ── Inference configuration ──────────────────────────────────────────────────
// ── Model type enum ──────────────────────────────────────────────────────────
enum class ModelType {
  kAutoencoder,   // Reconstructs window, anomaly = high reconstruction error
  kForecaster     // Predicts next step, anomaly = prediction != actual
};

struct PredictionConfig {
  // Model type: "autoencoder" or "forecaster"
  ModelType model_type = ModelType::kAutoencoder;
  
  // Model paths
  std::string cpu_model_file;
  std::string mem_model_file;
  
  // Thresholds
  float cpu_threshold = 0.00769f;
  float mem_threshold = 0.00545f;
  
  // Window configuration
  int window_size = kDefaultWindowSize;
  int warmup_samples = 30;  // minimum samples before flagging anomalies
  
  // Feature scalers
  ScalerParams cpu_scaler;
  ScalerParams mem_scaler;
  
  // Model version
  std::string model_version;
  
  // NPU compatibility flag
  bool bstorm_compatible = true;
};

// ── Raw telemetry reading ────────────────────────────────────────────────────
struct TelemetryReading {
  std::string mac;
  std::string timestamp;
  
  // CPU metrics
  float used_cpu     = 0.0f;   // USED_CPU_ATOM (%)
  float load_avg     = 0.0f;   // LOAD_AVG_ATOM
  
  // Memory metrics (kB)
  float total_mem_kb = 0.0f;   // TOTAL_MEM_ATOM_kB
  float used_mem_kb  = 0.0f;   // USED_MEM_ATOM_kB
  float avail_mem_kb = 0.0f;   // AvailMem_kB
  float free_mem_kb  = 0.0f;   // FreeMem_kB
  float slab_mem_kb  = 0.0f;   // SlabMem_kB
  
  // Client counts
  int clients_2g = 0;
  int clients_5g = 0;
  int clients_6g = 0;
  
  // Time features
  int hour_of_day = 0;         // 0-23
  int day_of_week = 0;         // 0-6 (Monday=0)
};

// ── Prediction result ────────────────────────────────────────────────────────
struct PredictionResult {
  // Reconstruction errors (MSE)
  float cpu_mse = 0.0f;
  float mem_mse = 0.0f;
  
  // Anomaly flags (1 = anomaly)
  int cpu_anomaly = 0;
  int mem_anomaly = 0;
  
  // Severity scores (MSE / threshold, > 1.0 = anomaly)
  float cpu_severity = 0.0f;
  float mem_severity = 0.0f;
  
  // Combined status
  std::string anomaly_type;    // "Normal" | "CPU" | "Memory" | "Both"
  
  // Timing
  float inference_time_ms = 0.0f;
  
  // Window state
  int window_samples = 0;      // current window fill level
  bool window_ready = false;   // true if window is full
};

// ── Per-device sliding window state ──────────────────────────────────────────
struct DeviceState {
  // Sliding windows for features [window_size][n_features]
  std::deque<std::vector<float>> cpu_window;
  std::deque<std::vector<float>> mem_window;
  
  // Previous values for delta features
  float prev_cpu = -1.0f;
  float prev_load = -1.0f;
  float prev_mem_util = -1.0f;
  float prev_avail_to_total = -1.0f;
  float prev_slab_pressure = -1.0f;
  float prev_free_to_avail = -1.0f;
  
  // Initialization state
  bool cpu_initialized = false;
  bool mem_initialized = false;
  int sample_count = 0;
  
  // Forecaster mode: store last predictions to compare with actual
  std::vector<float> last_cpu_prediction;  // predicted t+1 features
  std::vector<float> last_mem_prediction;  // predicted t+1 features
  bool has_cpu_prediction = false;
  bool has_mem_prediction = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// AnomalyPredictionEngine
//
// Window-based TCN autoencoder inference for CPU and memory anomaly prediction.
// Maintains sliding window per device, runs inference when window is full.
//
// Usage:
//   AnomalyPredictionEngine engine("config.json");
//   PredictionResult r = engine.ProcessReading(reading);
//   if (r.window_ready && r.cpu_anomaly) { ... }
// ─────────────────────────────────────────────────────────────────────────────
class AnomalyPredictionEngine {
 public:
  // config_path     : path to anomaly_prediction_config.json
  // delegate_path   : path to external delegate .so (e.g., bstorm)
  // delegate_options: semicolon-separated key:value pairs
  explicit AnomalyPredictionEngine(const std::string& config_path,
                                    int num_threads = 1,
                                    const std::string& delegate_path = "",
                                    const std::string& delegate_options = "",
                                    bool verbose = false);

  ~AnomalyPredictionEngine() = default;

  // Process a single telemetry reading for a device.
  // Adds features to sliding window; runs inference if window is full.
  PredictionResult ProcessReading(const TelemetryReading& reading);

  // Clear state for a device (e.g., on reconnect).
  void ResetDevice(const std::string& mac);

  // Get current window size for a device
  int GetWindowFillLevel(const std::string& mac) const;

  // Check if models are loaded
  bool IsInitialized() const { return initialized_; }

  // Get configuration
  const PredictionConfig& GetConfig() const { return cfg_; }

 private:
  PredictionConfig cfg_;
  int              num_threads_ = 1;
  std::string      delegate_path_;
  std::string      delegate_options_;
  bool             verbose_ = false;
  bool             initialized_ = false;

  // TFLite model data + interpreters
  std::unique_ptr<tflite::FlatBufferModel> cpu_model_fb_;
  std::unique_ptr<tflite::FlatBufferModel> mem_model_fb_;

  std::unique_ptr<tflite::Interpreter> cpu_interp_;
  std::unique_ptr<tflite::Interpreter> mem_interp_;

  // External delegates (e.g., bstorm NPU)
  TfLiteDelegateUniquePtr cpu_ext_delegate_{nullptr, nullptr};
  TfLiteDelegateUniquePtr mem_ext_delegate_{nullptr, nullptr};

  // Per-device state keyed by MAC
  std::unordered_map<std::string, DeviceState> device_states_;

  // ── Internal helpers ───────────────────────────────────────────────────────
  DeviceState& GetState(const std::string& mac);

  void LoadConfig(const std::string& config_path);

  void LoadInterpreter(const std::string& path,
                       std::unique_ptr<tflite::FlatBufferModel>& fb_out,
                       std::unique_ptr<tflite::Interpreter>& interp_out,
                       TfLiteDelegateUniquePtr* ext_delegate_out,
                       const std::string& model_name);

  // Compute CPU features (11 features) for one timestep
  void ComputeCpuFeatures(const TelemetryReading& r,
                          DeviceState& state,
                          std::vector<float>& features) const;

  // Compute Memory features (20 features) for one timestep
  void ComputeMemFeatures(const TelemetryReading& r,
                          DeviceState& state,
                          std::vector<float>& features) const;

  // Apply MinMaxScaler
  static std::vector<float> ApplyScaler(const std::vector<float>& x,
                                        const ScalerParams& scaler);

  // Flatten window [window_size][n_features] → [window_size * n_features]
  static std::vector<float> FlattenWindow(
      const std::deque<std::vector<float>>& window);

  // Run TCN autoencoder inference, return reconstruction MSE
  float RunInference(tflite::Interpreter* interp,
                     const std::vector<float>& window_data,
                     int window_size,
                     int n_features);
  
  // Run TCN forecaster inference, return predicted next timestep
  std::vector<float> RunForecasterInference(tflite::Interpreter* interp,
                                            const std::vector<float>& window_data,
                                            int n_features);
};

// ── Config loader ────────────────────────────────────────────────────────────
PredictionConfig LoadPredictionConfig(const std::string& config_path);

}  // namespace anomaly_prediction
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXAMPLES_ANOMALY_PREDICTION_ANOMALY_PREDICTION_H_
