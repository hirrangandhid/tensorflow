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

anomaly_detection.cc
────────────────────────────────────────────────────────────────────────────
C++ inference engine for DOCSIS gateway telemetry anomaly detection.

Delta-enhanced autoencoder models for CPU and memory anomaly detection.
Designed for edge deployment with minimal state storage (16 bytes for memory deltas).

Feature computation for CPU model (11 features):
  USED_CPU_ATOM, LOAD_AVG_ATOM, slab_ratio, load_cpu_ratio,
  hour_sin, hour_cos, dow_sin, dow_cos,
  cpu_delta, load_delta, cpu_delta_abs

Feature computation for Memory model (19 features):
  Single-point (13): mem_utilization, avail_to_total, free_to_avail, slab_pressure,
                     mem_fragmentation, cache_ratio, slab_to_free,
                     cpu_normalized, load_normalized, clients_normalized, mem_per_client,
                     hour_sin, hour_cos
  Delta (6): mem_utilization_delta, avail_to_total_delta, slab_pressure_delta,
             free_to_avail_delta, mem_util_delta_abs, slab_delta_abs

MinMaxScaler: clip((x - min) / range, 0.0, 1.0)
MSE: mean((input − reconstructed)²) over all elements
============================================================================*/

#include "tensorflow/lite/examples/anomaly_detection/anomaly_detection.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann_json/json.hpp"
#include "tensorflow/lite/delegates/external/external_delegate.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"

namespace tflite {
namespace anomaly_detection {

using json = nlohmann::json;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

static float ClampF(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// ─────────────────────────────────────────────────────────────────────────────
// LoadModelConfig — parse idle_device_config.json format
// ─────────────────────────────────────────────────────────────────────────────

ModelConfig LoadModelConfig(const std::string& config_path) {
  std::ifstream f(config_path);
  if (!f) throw std::runtime_error("Cannot open config: " + config_path);

  json j;
  f >> j;

  ModelConfig mc;
  mc.model_file = j.value("model_file", "");
  mc.threshold  = static_cast<float>(j.value("threshold", 0.0));
  
  // Use device_specific_threshold if available, otherwise fall back to threshold
  if (j.contains("device_specific_threshold")) {
    mc.device_threshold = static_cast<float>(j["device_specific_threshold"].get<double>());
  } else {
    mc.device_threshold = mc.threshold;
  }
  
  // Warmup samples (skip anomaly flagging for cold-start)
  mc.warmup_samples = j.value("warmup_samples", 3);

  // Parse features list
  if (j.contains("features")) {
    for (const auto& feat : j["features"]) {
      mc.scaler.features.push_back(feat.get<std::string>());
    }
  }

  // Parse scaler arrays
  if (j.contains("scaler")) {
    const auto& scaler_j = j["scaler"];
    if (scaler_j.contains("min")) {
      for (const auto& v : scaler_j["min"]) {
        mc.scaler.data_min.push_back(static_cast<float>(v.get<double>()));
      }
    }
    if (scaler_j.contains("max")) {
      for (const auto& v : scaler_j["max"]) {
        mc.scaler.data_max.push_back(static_cast<float>(v.get<double>()));
      }
    }
    if (scaler_j.contains("range")) {
      for (const auto& v : scaler_j["range"]) {
        mc.scaler.data_range.push_back(static_cast<float>(v.get<double>()));
      }
    }
  }

  return mc;
}

// ─────────────────────────────────────────────────────────────────────────────
// AnomalyInferenceEngine — constructor
// ─────────────────────────────────────────────────────────────────────────────

AnomalyInferenceEngine::AnomalyInferenceEngine(const std::string& config_path,
                                               const std::string& memory_config_path,
                                               int num_threads,
                                               const std::string& delegate_path,
                                               const std::string& delegate_options,
                                               bool verbose) {
  num_threads_      = num_threads;
  delegate_path_    = delegate_path;
  delegate_options_ = delegate_options;
  verbose_          = verbose;

  // Load CPU model config
  cfg_.cpu_model = LoadModelConfig(config_path);
  LoadInterpreter(cfg_.cpu_model.model_file, cpu_model_fb_, cpu_interp_,
                  &cpu_ext_delegate_, verbose_);

  // Load memory model config (optional)
  if (!memory_config_path.empty()) {
    try {
      cfg_.memory_model = LoadModelConfig(memory_config_path);
      LoadInterpreter(cfg_.memory_model.model_file, mem_model_fb_, mem_interp_,
                      &mem_ext_delegate_, verbose_);
      cfg_.has_memory_model = true;
    } catch (const std::exception& e) {
      std::cerr << "[warning] Failed to load memory model: " << e.what() << "\n";
      cfg_.has_memory_model = false;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// LoadInterpreter — mirrors Python load_interpreter()
// ─────────────────────────────────────────────────────────────────────────────

void AnomalyInferenceEngine::LoadInterpreter(
    const std::string& path,
    std::unique_ptr<tflite::FlatBufferModel>& fb_out,
    std::unique_ptr<tflite::Interpreter>& interp_out,
    TfLiteDelegateUniquePtr* ext_delegate_out,
    bool verbose) {

  fb_out = tflite::FlatBufferModel::BuildFromFile(path.c_str());
  if (!fb_out)
    throw std::runtime_error("Failed to load TFLite model: " + path);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*fb_out, resolver)(&interp_out);
  if (!interp_out)
    throw std::runtime_error("Failed to build interpreter for: " + path);

  // Apply external hardware delegate if a path was provided.
  if (!delegate_path_.empty()) {
    TfLiteExternalDelegateOptions opts =
        TfLiteExternalDelegateOptionsDefault(delegate_path_.c_str());

    // IMPORTANT: TfLiteExternalDelegateOptionsInsert stores raw const char* pointers
    // without copying them. The strings MUST outlive the TfLiteExternalDelegateCreate call.
    // Pre-split all options first and reserve the exact count before inserting, so
    // the vectors never reallocate mid-loop and previously stored c_str() pointers
    // remain valid when TfLiteExternalDelegateCreate reads them.
    std::vector<std::string> opt_keys, opt_vals;
    if (!delegate_options_.empty()) {
      // Pre-split to get the final count before any reserve/insert
      std::vector<std::string> option_pairs;
      {
        std::stringstream oss(delegate_options_);
        std::string tok;
        while (std::getline(oss, tok, ';'))
          if (!tok.empty()) option_pairs.push_back(tok);
      }
      opt_keys.reserve(option_pairs.size());
      opt_vals.reserve(option_pairs.size());
      for (const auto& pair : option_pairs) {
        auto colon = pair.find(':');
        if (colon != std::string::npos) {
          opt_keys.emplace_back(pair.substr(0, colon));
          opt_vals.emplace_back(pair.substr(colon + 1));
          TfLiteExternalDelegateOptionsInsert(
              &opts, opt_keys.back().c_str(), opt_vals.back().c_str());
        }
      }
    }

    TfLiteDelegate* raw = TfLiteExternalDelegateCreate(&opts);
    // opt_keys / opt_vals still in scope here — safe for Create() to read them above.
    if (!raw) {
      std::cerr << "[delegate] Warning: TfLiteExternalDelegateCreate returned null for "
                << delegate_path_ << " — running on CPU for: " << path << "\n";
    } else {
      if (interp_out->ModifyGraphWithDelegate(raw) != kTfLiteOk) {
        std::cerr << "[delegate] Warning: ModifyGraphWithDelegate failed for: " << path
                  << " — running on CPU\n";
        TfLiteExternalDelegateDelete(raw);
      } else {
        std::cerr << "[delegate] Hardware delegate applied for: " << path << "\n";
        if (ext_delegate_out)
          *ext_delegate_out = TfLiteDelegateUniquePtr{raw, TfLiteExternalDelegateDelete};
        else
          TfLiteExternalDelegateDelete(raw);
      }
    }
  }

  interp_out->SetNumThreads(num_threads_);

  if (interp_out->AllocateTensors() != kTfLiteOk)
    throw std::runtime_error("AllocateTensors() failed for: " + path);

  if (verbose)
    tflite::PrintInterpreterState(interp_out.get());
}

// ─────────────────────────────────────────────────────────────────────────────
// GetState / ResetDevice
// ─────────────────────────────────────────────────────────────────────────────

DeviceState& AnomalyInferenceEngine::GetState(const std::string& mac) {
  return device_states_[mac];  // default-constructed if not present
}

void AnomalyInferenceEngine::ResetDevice(const std::string& mac) {
  device_states_.erase(mac);
}

// ─────────────────────────────────────────────────────────────────────────────
// ApplyMinMaxScaler — scaled = (x - min) / range  (NO CLIPPING!)
// Values outside training range will exceed [0,1] bounds, causing high
// reconstruction error when the sigmoid-activated model can't match them.
// ─────────────────────────────────────────────────────────────────────────────

std::vector<float> AnomalyInferenceEngine::ApplyMinMaxScaler(
    const std::vector<float>& x, const ScalerParams& scaler) {
  std::vector<float> out(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    float range = scaler.data_range[i];
    if (range < 1e-9f) range = 1.0f;  // avoid divide by zero
    // NO CLIPPING: values outside training range will have high error
    out[i] = (x[i] - scaler.data_min[i]) / range;
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// ComputeCpuFeatures — idle device model (11 features)
//
// Features (in order):
//   USED_CPU_ATOM, LOAD_AVG_ATOM, slab_ratio, load_cpu_ratio,
//   hour_sin, hour_cos, dow_sin, dow_cos,
//   cpu_delta, load_delta, cpu_delta_abs
// ─────────────────────────────────────────────────────────────────────────────

void AnomalyInferenceEngine::ComputeCpuFeatures(const TelemetryReading& r,
                                                DeviceState& state,
                                                std::vector<float>& cpu_feat) const {
  const float eps = 1e-9f;
  const float PI = 3.14159265358979323846f;

  // Derived ratios
  // slab_ratio = slab / total_mem (matches Python training)
  // total_mem ≈ used_mem + avail_mem (approximate total physical RAM)
  float total_mem = r.used_mem_kb + r.avail_mem_kb;
  float slab_ratio = r.slab_mem_kb / (total_mem + eps);
  float load_cpu_ratio = r.load_avg / (r.used_cpu + eps);

  // Temporal features (sin/cos encoding)
  float hour = static_cast<float>(r.hour_of_day);
  float dow  = static_cast<float>(r.day_of_week);
  float hour_sin = std::sin(2.0f * PI * hour / 24.0f);
  float hour_cos = std::cos(2.0f * PI * hour / 24.0f);
  float dow_sin  = std::sin(2.0f * PI * dow / 7.0f);
  float dow_cos  = std::cos(2.0f * PI * dow / 7.0f);

  // Delta features (difference from previous reading)
  float cpu_delta  = state.cpu_initialized ? (r.used_cpu - state.prev_cpu) : 0.0f;
  float load_delta = state.cpu_initialized ? (r.load_avg - state.prev_load) : 0.0f;
  float cpu_delta_abs = std::fabs(cpu_delta);

  // Update state for next reading
  state.prev_cpu = r.used_cpu;
  state.prev_load = r.load_avg;
  state.cpu_initialized = true;

  // Assemble feature vector (11 features)
  cpu_feat = {
    r.used_cpu,       // USED_CPU_ATOM
    r.load_avg,       // LOAD_AVG_ATOM
    slab_ratio,       // slab_ratio
    load_cpu_ratio,   // load_cpu_ratio
    hour_sin,         // hour_sin
    hour_cos,         // hour_cos
    dow_sin,          // dow_sin
    dow_cos,          // dow_cos
    cpu_delta,        // cpu_delta
    load_delta,       // load_delta
    cpu_delta_abs     // cpu_delta_abs
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// ComputeMemFeatures — delta-enhanced memory model (19 features)
//
// Single-point features (13):
//   mem_utilization, avail_to_total, free_to_avail, slab_pressure,
//   mem_fragmentation, cache_ratio, slab_to_free,
//   cpu_normalized, load_normalized, clients_normalized, mem_per_client,
//   hour_sin, hour_cos
//
// Delta features (6):
//   mem_utilization_delta, avail_to_total_delta, slab_pressure_delta,
//   free_to_avail_delta, mem_util_delta_abs, slab_delta_abs
// ───────────────────────────────────────────────────────────────────────────────

void AnomalyInferenceEngine::ComputeMemFeatures(const TelemetryReading& r,
                                                DeviceState& state,
                                                std::vector<float>& mem_feat) const {
  const float eps = 1e-6f;
  const float PI = 3.14159265358979323846f;

  // ===========================================================================
  // CORE MEMORY RATIO FEATURES (device-agnostic)
  // ===========================================================================
  
  // Total memory estimate (used + available)
  float total_mem = r.used_mem_kb + r.avail_mem_kb;
  
  // Memory utilization ratio (primary indicator)
  float mem_utilization = r.used_mem_kb / (total_mem + eps);
  
  // Available memory ratio (inverse of pressure)
  float avail_to_total = r.avail_mem_kb / (total_mem + eps);
  
  // Free to available ratio (fragmentation indicator)
  float free_to_avail = r.free_mem_kb / (r.avail_mem_kb + eps);
  
  // Slab pressure ratio (kernel memory pressure)
  float slab_pressure = r.slab_mem_kb / (total_mem + eps);

  // ===========================================================================
  // DERIVED MEMORY INDICATORS
  // ===========================================================================
  
  // Memory fragmentation: low free/avail ratio with high used memory
  float mem_fragmentation = (1.0f - free_to_avail) * mem_utilization;
  
  // Cache efficiency: difference between available and free (cached/buffer)
  float cache_ratio = (r.avail_mem_kb - r.free_mem_kb) / (total_mem + eps);
  
  // Slab to free ratio (kernel pressure relative to free memory)
  float slab_to_free = r.slab_mem_kb / (r.free_mem_kb + eps);

  // ===========================================================================
  // CONTEXTUAL FEATURES (workload context)
  // ===========================================================================
  
  // CPU and load context (normalized)
  float cpu_normalized = r.used_cpu / 100.0f;
  float load_normalized = r.load_avg / 4.0f;  // Assuming max load ~4
  
  // Total client count (network load indicator)
  int total_clients = r.clients_2g + r.clients_5g + r.clients_6g;
  float clients_normalized = static_cast<float>(total_clients) / 20.0f;
  
  // Memory per client (memory pressure per workload unit)
  float mem_per_client = mem_utilization / (static_cast<float>(total_clients) + 1.0f);

  // ===========================================================================
  // TEMPORAL FEATURES (cyclical encoding)
  // ===========================================================================
  
  float hour = static_cast<float>(r.hour_of_day);
  float hour_sin = std::sin(2.0f * PI * hour / 24.0f);
  float hour_cos = std::cos(2.0f * PI * hour / 24.0f);

  // ===========================================================================
  // DELTA FEATURES (trend detection)
  // ===========================================================================
  
  float mem_util_delta = 0.0f;
  float avail_delta = 0.0f;
  float slab_delta = 0.0f;
  float free_avail_delta = 0.0f;
  
  if (state.mem_initialized) {
    mem_util_delta = mem_utilization - state.prev_mem_utilization;
    avail_delta = avail_to_total - state.prev_avail_to_total;
    slab_delta = slab_pressure - state.prev_slab_pressure;
    free_avail_delta = free_to_avail - state.prev_free_to_avail;
  }
  
  // Update state for next reading
  state.prev_mem_utilization = mem_utilization;
  state.prev_avail_to_total = avail_to_total;
  state.prev_slab_pressure = slab_pressure;
  state.prev_free_to_avail = free_to_avail;
  state.mem_initialized = true;
  
  // Absolute deltas for spike detection
  float mem_util_delta_abs = std::fabs(mem_util_delta);
  float slab_delta_abs = std::fabs(slab_delta);

  // ===========================================================================
  // ASSEMBLE FEATURE VECTOR (19 features, order must match training)
  // ===========================================================================
  
  mem_feat = {
    // Single-point features (13)
    mem_utilization,      // 0: mem_utilization
    avail_to_total,       // 1: avail_to_total
    free_to_avail,        // 2: free_to_avail
    slab_pressure,        // 3: slab_pressure
    mem_fragmentation,    // 4: mem_fragmentation
    cache_ratio,          // 5: cache_ratio
    slab_to_free,         // 6: slab_to_free
    cpu_normalized,       // 7: cpu_normalized
    load_normalized,      // 8: load_normalized
    clients_normalized,   // 9: clients_normalized
    mem_per_client,       // 10: mem_per_client
    hour_sin,             // 11: hour_sin
    hour_cos,             // 12: hour_cos
    // Delta features (6)
    mem_util_delta,       // 13: mem_utilization_delta
    avail_delta,          // 14: avail_to_total_delta
    slab_delta,           // 15: slab_pressure_delta
    free_avail_delta,     // 16: free_to_avail_delta
    mem_util_delta_abs,   // 17: mem_util_delta_abs
    slab_delta_abs        // 18: slab_delta_abs
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// RunInference — autoencoder forward pass → mean reconstruction MSE
//   inp  shape: (1, n_features)
//   out  shape: (1, n_features)
//   return: mean((inp - out)²)
// ─────────────────────────────────────────────────────────────────────────────

float AnomalyInferenceEngine::RunInference(tflite::Interpreter* interp,
                                           const std::vector<float>& x_scaled) {
  const int in_idx  = interp->inputs()[0];
  const int out_idx = interp->outputs()[0];
  const int n       = static_cast<int>(x_scaled.size());

  // Write input tensor: shape (1, n_features)
  float* in_ptr = interp->typed_tensor<float>(in_idx);
  for (int i = 0; i < n; ++i) in_ptr[i] = x_scaled[i];

  interp->Invoke();

  const float* out_ptr = interp->typed_tensor<float>(out_idx);
  float mse = 0.0f;
  for (int i = 0; i < n; ++i) {
    float diff = x_scaled[i] - out_ptr[i];
    mse += diff * diff;
  }
  return mse / static_cast<float>(n);
}

// ─────────────────────────────────────────────────────────────────────────────
// ProcessReading — single-point anomaly detection
// ─────────────────────────────────────────────────────────────────────────────

AnomalyResult AnomalyInferenceEngine::ProcessReading(const TelemetryReading& r) {
  DeviceState& state = GetState(r.mac);
  AnomalyResult result;

  // 1. Compute CPU features and run inference
  std::vector<float> cpu_feat;
  ComputeCpuFeatures(r, state, cpu_feat);
  std::vector<float> cpu_sc = ApplyMinMaxScaler(cpu_feat, cfg_.cpu_model.scaler);
  float cpu_mse = RunInference(cpu_interp_.get(), cpu_sc);
  
  // Increment CPU sample counter and use device threshold
  state.cpu_sample_count++;
  float cpu_thr = cfg_.cpu_model.device_threshold;
  result.dense_cpu_mse  = cpu_mse;
  // Skip flagging during warmup period (cold-start false positive prevention)
  bool cpu_in_warmup = (state.cpu_sample_count <= cfg_.cpu_model.warmup_samples);
  result.dense_cpu_flag = (!cpu_in_warmup && cpu_mse > cpu_thr) ? 1 : 0;
  result.dense_cpu_sev  = cpu_mse / cpu_thr;

  // 2. Compute Memory features and run inference (if memory model loaded)
  if (cfg_.has_memory_model) {
    std::vector<float> mem_feat;
    ComputeMemFeatures(r, state, mem_feat);
    std::vector<float> mem_sc = ApplyMinMaxScaler(mem_feat, cfg_.memory_model.scaler);
    float mem_mse = RunInference(mem_interp_.get(), mem_sc);
    
    // Increment memory sample counter and use device threshold
    state.mem_sample_count++;
    float mem_thr = cfg_.memory_model.device_threshold;
    result.dense_mem_mse  = mem_mse;
    // Skip flagging during warmup period (cold-start false positive prevention)
    bool mem_in_warmup = (state.mem_sample_count <= cfg_.memory_model.warmup_samples);
    result.dense_mem_flag = (!mem_in_warmup && mem_mse > mem_thr) ? 1 : 0;
    result.dense_mem_sev  = mem_mse / mem_thr;
  }

  // 3. Classify anomaly type
  int c = result.dense_cpu_flag;
  int m = result.dense_mem_flag;
  if      (c && m) result.anomaly_type = "Both";
  else if (c)      result.anomaly_type = "CPU";
  else if (m)      result.anomaly_type = "Memory";
  else             result.anomaly_type = "Normal";

  return result;
}

}  // namespace anomaly_detection
}  // namespace tflite
