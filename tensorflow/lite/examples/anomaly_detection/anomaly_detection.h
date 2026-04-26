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

Delta-enhanced autoencoder models for CPU and memory anomaly detection.
Designed for edge deployment with minimal state (16 bytes for memory deltas).

Models supported:
  cpu_anomaly_model.tflite          CPU AE   input (1, 11)  [cpu device features]
  memory_anomaly_dynamic.tflite     Mem AE   input (1, 19)  [delta-enhanced features]

Config format (anomaly_config.json):
  - model_file: path to .tflite
  - threshold: anomaly threshold
  - features: ordered list of feature names
  - scaler: {min: [], max: [], range: []} arrays
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
// Provides TfLiteDelegateUniquePtr without pulling in the full TF runtime.
// The flex delegate itself is loaded dynamically via dlopen at runtime.
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
// External delegate API — used to load hardware accelerators (e.g. BStorm NPU)
// via --delegate-path at runtime, matching the label_image --external_delegate_path pattern.
#include "tensorflow/lite/delegates/external/external_delegate.h"

namespace tflite {
namespace anomaly_detection {

// ── Scaler parameters (MinMaxScaler) ─────────────────────────────────────────
// New format: arrays indexed by feature position (same order as features list)
// scaled = clip((x - min) / range, 0, 1)
struct ScalerParams {
  std::vector<std::string>     features;      // ordered feature names
  std::vector<float>           data_min;      // per-feature minimum
  std::vector<float>           data_max;      // per-feature maximum
  std::vector<float>           data_range;    // per-feature range (max - min)
};

// ── Per-model configuration ───────────────────────────────────────────────────
struct ModelConfig {
  std::string  model_file;                   // .tflite file path
  float        threshold = 0.0f;             // anomaly threshold
  ScalerParams scaler;
};

// ── Top-level inference configuration ─────────────────────────────────────────
// Supports both single-config (cpu_config.json) and dual-config formats
struct InferenceConfig {
  ModelConfig cpu_model;
  ModelConfig memory_model;
  bool        has_memory_model = false;      // true if memory model was loaded
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

// ── Inference result ─────────────────────────────────────────────────────────
struct AnomalyResult {
  float dense_cpu_mse  = 0.0f;
  float dense_mem_mse  = 0.0f;
  int   dense_cpu_flag = 0;     // 1 = anomaly
  int   dense_mem_flag = 0;
  float dense_cpu_sev  = 0.0f; // MSE / threshold  (> 1.0 = anomaly)
  float dense_mem_sev  = 0.0f;
  std::string anomaly_type;    // "Normal" | "CPU" | "Memory" | "Both"
};

// ── Per-device stateful data for delta computation ───────────────────────────
struct DeviceState {
  // CPU delta state
  float prev_cpu  = -1.0f;    // previous CPU value (-1 = not set)
  float prev_load = -1.0f;    // previous Load value (-1 = not set)
  bool  cpu_initialized = false;  // true after first CPU reading
  
  // Memory delta state (for delta-enhanced model)
  float prev_mem_utilization = -1.0f;
  float prev_avail_to_total  = -1.0f;
  float prev_slab_pressure   = -1.0f;
  float prev_free_to_avail   = -1.0f;
  bool  mem_initialized = false;  // true after first memory reading
};

// ─────────────────────────────────────────────────────────────────────────────
// AnomalyInferenceEngine
//
// Single-point autoencoder inference for CPU anomaly detection.
// Designed for idle devices (0 clients) with client-independent features.
//
// Usage:
//   AnomalyInferenceEngine engine("cpu_anomaly_config.json");
//   AnomalyResult r = engine.ProcessReading(reading);
// ─────────────────────────────────────────────────────────────────────────────
class AnomalyInferenceEngine {
 public:
  // config_path       : path to cpu_anomaly_config.json (CPU model)
  // memory_config_path: optional path to memory model config (can be empty)
  // delegate_path     : path to a TFLite external delegate .so
  // delegate_options  : semicolon-separated key:value pairs
  explicit AnomalyInferenceEngine(const std::string& config_path,
                                   const std::string& memory_config_path = "",
                                   int num_threads = 1,
                                   const std::string& delegate_path = "",
                                   const std::string& delegate_options = "",
                                   bool verbose = false);

  // Process a single telemetry reading for a device.
  // State (previous CPU/Load) is accumulated per MAC address.
  AnomalyResult ProcessReading(const TelemetryReading& reading);

  // Clear state for a device (e.g. on reconnect).
  void ResetDevice(const std::string& mac);

 private:
  InferenceConfig cfg_;
  int             num_threads_ = 1;
  std::string     delegate_path_;
  std::string     delegate_options_;
  bool            verbose_ = false;

  // TFLite model data + interpreters
  std::unique_ptr<tflite::FlatBufferModel> cpu_model_fb_;
  std::unique_ptr<tflite::FlatBufferModel> mem_model_fb_;

  std::unique_ptr<tflite::Interpreter> cpu_interp_;
  std::unique_ptr<tflite::Interpreter> mem_interp_;

  // External (hardware) delegates — one per interpreter
  TfLiteDelegateUniquePtr cpu_ext_delegate_{nullptr, nullptr};
  TfLiteDelegateUniquePtr mem_ext_delegate_{nullptr, nullptr};

  // Per-device state keyed by MAC string
  std::unordered_map<std::string, DeviceState> device_states_;

  // ── Helpers ────────────────────────────────────────────────────────────────
  DeviceState& GetState(const std::string& mac);

  void LoadInterpreter(const std::string& path,
                       std::unique_ptr<tflite::FlatBufferModel>& fb_out,
                       std::unique_ptr<tflite::Interpreter>& interp_out,
                       TfLiteDelegateUniquePtr* ext_delegate_out = nullptr,
                       bool verbose = false);

  // Compute CPU features for idle device model (11 features)
  // Features: USED_CPU_ATOM, LOAD_AVG_ATOM, slab_ratio, load_cpu_ratio,
  //           hour_sin, hour_cos, dow_sin, dow_cos,
  //           cpu_delta, load_delta, cpu_delta_abs
  void ComputeCpuFeatures(const TelemetryReading& r,
                          DeviceState& state,
                          std::vector<float>& cpu_feat) const;

  // Compute Memory features (19 features) - delta-enhanced model
  // Single-point (13): mem_utilization, avail_to_total, free_to_avail, slab_pressure,
  //                    mem_fragmentation, cache_ratio, slab_to_free,
  //                    cpu_normalized, load_normalized, clients_normalized, mem_per_client,
  //                    hour_sin, hour_cos
  // Delta (6): mem_utilization_delta, avail_to_total_delta, slab_pressure_delta,
  //            free_to_avail_delta, mem_util_delta_abs, slab_delta_abs
  void ComputeMemFeatures(const TelemetryReading& r,
                          DeviceState& state,
                          std::vector<float>& mem_feat) const;

  // Apply MinMaxScaler: scaled = clip((x - min) / range, 0, 1)
  static std::vector<float> ApplyMinMaxScaler(const std::vector<float>& x,
                                              const ScalerParams& scaler);

  // Autoencoder forward pass → mean reconstruction MSE over all features.
  static float RunInference(tflite::Interpreter* interp,
                            const std::vector<float>& x_scaled);

};

// ── Config loader (parses idle_device_config.json format) ─────────────────────
ModelConfig LoadModelConfig(const std::string& config_path);

}  // namespace anomaly_detection
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXAMPLES_ANOMALY_DETECTION_ANOMALY_DETECTION_H_
