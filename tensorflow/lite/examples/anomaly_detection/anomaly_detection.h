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

anomaly_detection.h
────────────────────────────────────────────────────────────────────────────
C++ inference engine for DOCSIS gateway telemetry anomaly detection.

Mirrors the Python AnomalyInferenceEngine in inference_app.ipynb exactly:
  - Dense AutoEncoder  : stateless, processes each row independently
  - LSTM  AutoEncoder  : stateful, maintains per-device rolling window

Four TFLite models are supported:
  cpu_anomaly_model.tflite       Dense AE  input (1, 10)
  memory_anomaly_model.tflite    Dense AE  input (1,  9)
  lstm_cpu_anomaly_model.tflite  LSTM  AE  input (1, 10, 10)
  lstm_memory_anomaly_model.tflite LSTM AE input (1, 10,  9)
============================================================================*/

#ifndef TENSORFLOW_LITE_EXAMPLES_ANOMALY_DETECTION_ANOMALY_DETECTION_H_
#define TENSORFLOW_LITE_EXAMPLES_ANOMALY_DETECTION_ANOMALY_DETECTION_H_

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/delegates/flex/delegate.h"

namespace tflite {
namespace anomaly_detection {

// ── Scaler parameters (MinMaxScaler) ─────────────────────────────────────────
// Stored in the same order as feature_order so that index i in data_min/scale
// maps directly to element i of the feature vector.
struct ScalerParams {
  std::vector<std::string>     feature_order;   // ordered feature names
  std::vector<float>           data_min;        // per-feature minimum
  std::vector<float>           scale;           // per-feature scale factor
};

// ── Per-model configuration ───────────────────────────────────────────────────
struct ModelConfig {
  std::string  tflite_file;
  float        anomaly_threshold = 0.0f;
  int          seq_len           = 1;      // 1 for dense, 10 for LSTM
  bool         flex_delegate     = false;
  ScalerParams scaler;
};

// ── Top-level inference configuration (mirrors inference_config.json) ─────────
struct InferenceConfig {
  int         rolling_window = 10;
  ModelConfig cpu_model;
  ModelConfig memory_model;
  ModelConfig lstm_cpu_model;
  ModelConfig lstm_memory_model;
  bool        has_lstm_cpu = false;
  bool        has_lstm_mem = false;
};

// ── One raw telemetry reading ─────────────────────────────────────────────────
struct TelemetryReading {
  std::string mac;
  std::string timestamp;
  float used_cpu     = 0.0f;   // USED_CPU_ATOM
  float load_avg     = 0.0f;   // LOAD_AVG_ATOM
  float used_mem_kb  = 0.0f;   // USED_MEM_ATOM_kB
  float avail_mem_kb = 0.0f;   // AvailMem_kB
  float free_mem_kb  = 0.0f;   // FreeMem_kB
  float slab_mem_kb  = 0.0f;   // SlabMem_kB
  int   clients_2g   = 0;      // 2G_Clients_Count
  int   clients_5g   = 0;      // 5G_Clients_Count
  int   clients_6g   = 0;      // 6G_Clients_Count
  int   hour_of_day  = 0;
  int   day_of_week  = 0;
};

// ── Inference result (mirrors Python process_reading() return dict) ───────────
struct AnomalyResult {
  // Dense AE
  float dense_cpu_mse  = 0.0f;
  float dense_mem_mse  = 0.0f;
  int   dense_cpu_flag = 0;     // 1 = anomaly
  int   dense_mem_flag = 0;
  float dense_cpu_sev  = 0.0f; // MSE / threshold  (> 1.0 = anomaly)
  float dense_mem_sev  = 0.0f;
  // LSTM AE (only populated when LSTM models are loaded)
  float lstm_cpu_mse   = 0.0f;
  float lstm_mem_mse   = 0.0f;
  int   lstm_cpu_flag  = 0;
  int   lstm_mem_flag  = 0;
  float lstm_cpu_sev   = 0.0f;
  float lstm_mem_sev   = 0.0f;
  bool  has_lstm       = false;
  // Classification
  std::string anomaly_type;    // "Normal" | "CPU" | "Memory" | "Both"
};

// ── Per-device stateful rolling window ────────────────────────────────────────
struct DeviceState {
  std::deque<float>               cpu_history; // raw CPU% values (up to SEQ_LEN)
  std::deque<std::vector<float>>  cpu_seq;     // scaled CPU feature vectors
  std::deque<std::vector<float>>  mem_seq;     // scaled Mem feature vectors
};

// ─────────────────────────────────────────────────────────────────────────────
// AnomalyInferenceEngine
//
// Mirrors Python class AnomalyInferenceEngine exactly.
//
// Usage:
//   AnomalyInferenceEngine engine("inference_config.json");
//   AnomalyResult r = engine.ProcessReading(reading);
// ─────────────────────────────────────────────────────────────────────────────
class AnomalyInferenceEngine {
 public:
  explicit AnomalyInferenceEngine(const std::string& config_path,
                                   int num_threads = 1);

  // Process a single telemetry reading for a device.
  // State (rolling window) is accumulated per MAC address.
  AnomalyResult ProcessReading(const TelemetryReading& reading);

  // Clear rolling state for a device (e.g. on reconnect).
  void ResetDevice(const std::string& mac);

  int seq_len() const { return seq_len_; }

 private:
  InferenceConfig cfg_;
  int             seq_len_;
  int             num_threads_ = 1;

  // TFLite model data + interpreters (dense and optional LSTM)
  std::unique_ptr<tflite::FlatBufferModel> dense_cpu_fb_;
  std::unique_ptr<tflite::FlatBufferModel> dense_mem_fb_;
  std::unique_ptr<tflite::FlatBufferModel> lstm_cpu_fb_;
  std::unique_ptr<tflite::FlatBufferModel> lstm_mem_fb_;

  std::unique_ptr<tflite::Interpreter> dense_cpu_interp_;
  std::unique_ptr<tflite::Interpreter> dense_mem_interp_;
  std::unique_ptr<tflite::Interpreter> lstm_cpu_interp_;
  std::unique_ptr<tflite::Interpreter> lstm_mem_interp_;
  // Flex delegates for LSTM models — must outlive their interpreters.
  // Initialized with {nullptr, nullptr} because TfLiteDelegateUniquePtr uses
  // a raw function pointer deleter which has no default constructor.
  TfLiteDelegateUniquePtr lstm_cpu_delegate_{nullptr, nullptr};
  TfLiteDelegateUniquePtr lstm_mem_delegate_{nullptr, nullptr};

  // Per-device state keyed by MAC string
  std::unordered_map<std::string, DeviceState> device_states_;

  // ── Helpers ────────────────────────────────────────────────────────────────
  DeviceState& GetState(const std::string& mac);

  // use_flex_delegate=true required for LSTM models (flex_delegate: true in config)
  // because UnidirectionalSequenceLSTM is a SELECT_TF_OPS op, not a TFLite builtin.
  void LoadInterpreter(const std::string& path,
                       std::unique_ptr<tflite::FlatBufferModel>& fb_out,
                       std::unique_ptr<tflite::Interpreter>& interp_out,
                       bool use_flex_delegate = false,
                       TfLiteDelegateUniquePtr* delegate_out = nullptr);

  // Compute CPU (10-feature) and Memory (9-feature) vectors from raw reading.
  // Matches Python _compute_features() exactly, including ddof=1 rolling std.
  void ComputeFeatures(const TelemetryReading& r,
                       DeviceState& state,
                       std::vector<float>& cpu_feat,
                       std::vector<float>& mem_feat) const;

  // Apply MinMaxScaler: scaled = clip((x - data_min) * scale, 0, 1)
  static std::vector<float> ApplyMinMaxScaler(const std::vector<float>& x,
                                              const ScalerParams& scaler);

  // Dense AE forward pass → mean reconstruction MSE over all features.
  static float RunDenseInference(tflite::Interpreter* interp,
                                 const std::vector<float>& x_scaled);

  // LSTM AE forward pass → mean reconstruction MSE over all elements.
  // seq_scaled is shape (seq_len, n_features) — zero-padded at the front.
  static float RunLstmInference(
      tflite::Interpreter* interp,
      const std::deque<std::vector<float>>& seq,
      int seq_len, int n_features);
};

// ── Config loader (parses inference_config.json) ──────────────────────────────
InferenceConfig LoadConfig(const std::string& config_path);

}  // namespace anomaly_detection
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXAMPLES_ANOMALY_DETECTION_ANOMALY_DETECTION_H_
