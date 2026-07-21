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

anomaly_prediction.cc
────────────────────────────────────────────────────────────────────────────
C++ inference engine for RDKB gateway telemetry anomaly prediction.

TCN autoencoder models with sliding window approach.
Supports bstorm delegate for NPU acceleration on RDKB routers.

Feature computation for CPU model (11 features):
  USED_CPU_ATOM, LOAD_AVG_ATOM, slab_ratio, load_cpu_ratio,
  cpu_delta, load_delta, hour_sin, hour_cos, day_sin, day_cos,
  total_clients

Feature computation for Memory model (20 features):
  mem_utilization, avail_to_total, free_to_avail, mem_fragmentation,
  cache_ratio, slab_to_free, USED_CPU_ATOM, LOAD_AVG_ATOM,
  total_clients, mem_per_client, hour_sin, hour_cos, day_sin, day_cos,
  mem_utilization_delta, avail_to_total_delta, slab_pressure_delta,
  free_to_avail_delta, mem_util_delta_abs, slab_delta_abs

MinMaxScaler: clip((x - min) / (max - min), 0.0, 1.0)
MSE: mean((input − reconstructed)²) over entire window
============================================================================*/

#include "tensorflow/lite/examples/anomaly_prediction/anomaly_prediction.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/time.h>

#include "nlohmann_json/json.hpp"
#include "tensorflow/lite/delegates/external/external_delegate.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"

namespace tflite {
namespace anomaly_prediction {

using json = nlohmann::json;

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions
// ─────────────────────────────────────────────────────────────────────────────

static float ClampF(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

static float SafeDiv(float a, float b, float eps = 1e-9f) {
  return (std::fabs(b) > eps) ? (a / b) : 0.0f;
}

static double GetTimeMs() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<double>(tv.tv_sec) * 1000.0 + 
         static_cast<double>(tv.tv_usec) / 1000.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// LoadPredictionConfig — parse config JSON
// ─────────────────────────────────────────────────────────────────────────────

PredictionConfig LoadPredictionConfig(const std::string& config_path) {
  std::ifstream f(config_path);
  if (!f) throw std::runtime_error("Cannot open config: " + config_path);

  json j;
  f >> j;

  PredictionConfig cfg;
  
  // Model files
  cfg.cpu_model_file = j.value("cpu_model_file", "tcn_cpu_anomaly_model_v2.tflite");
  cfg.mem_model_file = j.value("mem_model_file", "tcn_mem_anomaly_model_v2.tflite");
  
  // Thresholds
  cfg.cpu_threshold = static_cast<float>(j.value("cpu_threshold", 0.00769));
  cfg.mem_threshold = static_cast<float>(j.value("mem_threshold", 0.00545));
  
  // Window configuration
  cfg.window_size = j.value("window_size", kDefaultWindowSize);
  cfg.warmup_samples = j.value("warmup_samples", cfg.window_size);
  
  // Model version
  cfg.model_version = j.value("model_version", "v2_enhanced");

  // Parse CPU features
  if (j.contains("cpu_features")) {
    for (const auto& feat : j["cpu_features"]) {
      cfg.cpu_scaler.features.push_back(feat.get<std::string>());
    }
  }
  
  // Parse Memory features
  if (j.contains("mem_features")) {
    for (const auto& feat : j["mem_features"]) {
      cfg.mem_scaler.features.push_back(feat.get<std::string>());
    }
  }

  // Parse CPU scaler
  if (j.contains("cpu_scaler")) {
    const auto& scaler_j = j["cpu_scaler"];
    if (scaler_j.contains("min")) {
      for (const auto& v : scaler_j["min"]) {
        cfg.cpu_scaler.data_min.push_back(static_cast<float>(v.get<double>()));
      }
    }
    if (scaler_j.contains("max")) {
      for (const auto& v : scaler_j["max"]) {
        cfg.cpu_scaler.data_max.push_back(static_cast<float>(v.get<double>()));
      }
    }
  }
  
  // Parse Memory scaler
  if (j.contains("mem_scaler")) {
    const auto& scaler_j = j["mem_scaler"];
    if (scaler_j.contains("min")) {
      for (const auto& v : scaler_j["min"]) {
        cfg.mem_scaler.data_min.push_back(static_cast<float>(v.get<double>()));
      }
    }
    if (scaler_j.contains("max")) {
      for (const auto& v : scaler_j["max"]) {
        cfg.mem_scaler.data_max.push_back(static_cast<float>(v.get<double>()));
      }
    }
  }

  return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// AnomalyPredictionEngine — constructor
// ─────────────────────────────────────────────────────────────────────────────

AnomalyPredictionEngine::AnomalyPredictionEngine(
    const std::string& config_path,
    int num_threads,
    const std::string& delegate_path,
    const std::string& delegate_options,
    bool verbose) {
  
  num_threads_      = num_threads;
  delegate_path_    = delegate_path;
  delegate_options_ = delegate_options;
  verbose_          = verbose;

  try {
    // Load configuration
    LoadConfig(config_path);

    // Load CPU model
    LoadInterpreter(cfg_.cpu_model_file, cpu_model_fb_, cpu_interp_,
                    &cpu_ext_delegate_, "CPU");

    // Load Memory model
    LoadInterpreter(cfg_.mem_model_file, mem_model_fb_, mem_interp_,
                    &mem_ext_delegate_, "Memory");

    initialized_ = true;
    
    if (verbose_) {
      std::cerr << "[AnomalyPrediction] Initialized successfully\n";
      std::cerr << "  CPU threshold:  " << cfg_.cpu_threshold << "\n";
      std::cerr << "  Mem threshold:  " << cfg_.mem_threshold << "\n";
      std::cerr << "  Window size:    " << cfg_.window_size << "\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "[AnomalyPrediction] Initialization failed: " << e.what() << "\n";
    initialized_ = false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// LoadConfig
// ─────────────────────────────────────────────────────────────────────────────

void AnomalyPredictionEngine::LoadConfig(const std::string& config_path) {
  cfg_ = LoadPredictionConfig(config_path);
}

// ─────────────────────────────────────────────────────────────────────────────
// LoadInterpreter — load model and apply delegate
// ─────────────────────────────────────────────────────────────────────────────

void AnomalyPredictionEngine::LoadInterpreter(
    const std::string& path,
    std::unique_ptr<tflite::FlatBufferModel>& fb_out,
    std::unique_ptr<tflite::Interpreter>& interp_out,
    TfLiteDelegateUniquePtr* ext_delegate_out,
    const std::string& model_name) {

  fb_out = tflite::FlatBufferModel::BuildFromFile(path.c_str());
  if (!fb_out)
    throw std::runtime_error("Failed to load TFLite model: " + path);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*fb_out, resolver)(&interp_out);
  if (!interp_out)
    throw std::runtime_error("Failed to build interpreter for: " + path);

  // Apply external hardware delegate (e.g., bstorm) if provided
  if (!delegate_path_.empty()) {
    TfLiteExternalDelegateOptions opts =
        TfLiteExternalDelegateOptionsDefault(delegate_path_.c_str());

    // Parse delegate options (key:value pairs separated by semicolons)
    std::vector<std::string> opt_keys, opt_vals;
    if (!delegate_options_.empty()) {
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
    if (!raw) {
      std::cerr << "[delegate] Warning: TfLiteExternalDelegateCreate failed for "
                << model_name << " — running on CPU\n";
    } else {
      if (interp_out->ModifyGraphWithDelegate(raw) != kTfLiteOk) {
        std::cerr << "[delegate] Warning: ModifyGraphWithDelegate failed for "
                  << model_name << " — running on CPU\n";
        TfLiteExternalDelegateDelete(raw);
      } else {
        std::cerr << "[delegate] Hardware delegate applied for " << model_name << "\n";
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

  if (verbose_) {
    std::cerr << "[" << model_name << "] Model loaded: " << path << "\n";
    // Print input tensor shape
    int input_idx = interp_out->inputs()[0];
    TfLiteIntArray* dims = interp_out->tensor(input_idx)->dims;
    std::cerr << "  Input shape: [";
    for (int i = 0; i < dims->size; ++i) {
      std::cerr << dims->data[i];
      if (i < dims->size - 1) std::cerr << ", ";
    }
    std::cerr << "]\n";
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// GetState / ResetDevice
// ─────────────────────────────────────────────────────────────────────────────

DeviceState& AnomalyPredictionEngine::GetState(const std::string& mac) {
  return device_states_[mac];
}

void AnomalyPredictionEngine::ResetDevice(const std::string& mac) {
  device_states_.erase(mac);
}

int AnomalyPredictionEngine::GetWindowFillLevel(const std::string& mac) const {
  auto it = device_states_.find(mac);
  if (it == device_states_.end()) return 0;
  return static_cast<int>(it->second.cpu_window.size());
}

// ─────────────────────────────────────────────────────────────────────────────
// ApplyScaler — MinMax scaling
// ─────────────────────────────────────────────────────────────────────────────

std::vector<float> AnomalyPredictionEngine::ApplyScaler(
    const std::vector<float>& x, const ScalerParams& scaler) {
  
  std::vector<float> out(x.size());
  
  // If scaler params not set, return input unchanged
  if (scaler.data_min.empty() || scaler.data_max.empty()) {
    return x;
  }
  
  for (size_t i = 0; i < x.size(); ++i) {
    size_t idx = i % scaler.data_min.size();
    float range = scaler.data_max[idx] - scaler.data_min[idx];
    if (range < 1e-9f) range = 1.0f;
    out[i] = ClampF((x[i] - scaler.data_min[idx]) / range, 0.0f, 1.0f);
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// ComputeCpuFeatures — 11 features matching notebook training
// ─────────────────────────────────────────────────────────────────────────────

void AnomalyPredictionEngine::ComputeCpuFeatures(
    const TelemetryReading& r,
    DeviceState& state,
    std::vector<float>& features) const {
  
  const float PI = 3.14159265358979323846f;
  const float eps = 1e-9f;

  // Derive total memory (used + avail as approximation)
  float total_mem = r.used_mem_kb + r.avail_mem_kb;
  if (total_mem < eps) total_mem = r.total_mem_kb;
  if (total_mem < eps) total_mem = 1.0f;
  
  // Derived ratios
  float slab_ratio = SafeDiv(r.slab_mem_kb, total_mem);
  float load_cpu_ratio = SafeDiv(r.load_avg, r.used_cpu + eps);

  // Delta features
  float cpu_delta = state.cpu_initialized ? (r.used_cpu - state.prev_cpu) : 0.0f;
  float load_delta = state.cpu_initialized ? (r.load_avg - state.prev_load) : 0.0f;

  // Cyclic time features
  float hour = static_cast<float>(r.hour_of_day);
  float dow = static_cast<float>(r.day_of_week);
  float hour_sin = std::sin(2.0f * PI * hour / 24.0f);
  float hour_cos = std::cos(2.0f * PI * hour / 24.0f);
  float day_sin = std::sin(2.0f * PI * dow / 7.0f);
  float day_cos = std::cos(2.0f * PI * dow / 7.0f);

  // Total clients
  float total_clients = static_cast<float>(r.clients_2g + r.clients_5g + r.clients_6g);

  // Update state
  state.prev_cpu = r.used_cpu;
  state.prev_load = r.load_avg;
  state.cpu_initialized = true;

  // Assemble feature vector (11 features) - order matches notebook
  features = {
    r.used_cpu,         // USED_CPU_ATOM
    r.load_avg,         // LOAD_AVG_ATOM
    slab_ratio,         // slab_ratio
    load_cpu_ratio,     // load_cpu_ratio
    cpu_delta,          // cpu_delta
    load_delta,         // load_delta
    hour_sin,           // hour_sin
    hour_cos,           // hour_cos
    day_sin,            // day_sin
    day_cos,            // day_cos
    total_clients       // total_clients
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// ComputeMemFeatures — 20 features matching notebook training
// ─────────────────────────────────────────────────────────────────────────────

void AnomalyPredictionEngine::ComputeMemFeatures(
    const TelemetryReading& r,
    DeviceState& state,
    std::vector<float>& features) const {
  
  const float PI = 3.14159265358979323846f;
  const float eps = 1e-9f;

  // Memory calculations
  float total_mem = r.total_mem_kb > 0 ? r.total_mem_kb : 
                    (r.used_mem_kb + r.avail_mem_kb);
  if (total_mem < eps) total_mem = 1.0f;
  
  float mem_utilization = SafeDiv(r.used_mem_kb, total_mem);
  float avail_to_total = SafeDiv(r.avail_mem_kb, total_mem);
  float free_to_avail = SafeDiv(r.free_mem_kb, r.avail_mem_kb + eps);
  float mem_fragmentation = 1.0f - free_to_avail;
  float cache_ratio = SafeDiv(r.slab_mem_kb, r.used_mem_kb + eps);
  float slab_to_free = SafeDiv(r.slab_mem_kb, r.free_mem_kb + eps);
  float slab_pressure = SafeDiv(r.slab_mem_kb, r.avail_mem_kb + eps);

  // Client and cross-signal features
  float total_clients = static_cast<float>(r.clients_2g + r.clients_5g + r.clients_6g);
  float mem_per_client = SafeDiv(r.used_mem_kb, total_clients + 1.0f);

  // Cyclic time features
  float hour = static_cast<float>(r.hour_of_day);
  float dow = static_cast<float>(r.day_of_week);
  float hour_sin = std::sin(2.0f * PI * hour / 24.0f);
  float hour_cos = std::cos(2.0f * PI * hour / 24.0f);
  float day_sin = std::sin(2.0f * PI * dow / 7.0f);
  float day_cos = std::cos(2.0f * PI * dow / 7.0f);

  // Delta features
  float mem_util_delta = state.mem_initialized ? 
                         (mem_utilization - state.prev_mem_util) : 0.0f;
  float avail_delta = state.mem_initialized ? 
                      (avail_to_total - state.prev_avail_to_total) : 0.0f;
  float slab_pressure_delta = state.mem_initialized ? 
                              (slab_pressure - state.prev_slab_pressure) : 0.0f;
  float free_avail_delta = state.mem_initialized ? 
                           (free_to_avail - state.prev_free_to_avail) : 0.0f;

  // Update state
  state.prev_mem_util = mem_utilization;
  state.prev_avail_to_total = avail_to_total;
  state.prev_slab_pressure = slab_pressure;
  state.prev_free_to_avail = free_to_avail;
  state.mem_initialized = true;

  // Assemble feature vector (20 features) - order matches notebook
  features = {
    mem_utilization,              // mem_utilization
    avail_to_total,               // avail_to_total
    free_to_avail,                // free_to_avail
    mem_fragmentation,            // mem_fragmentation
    cache_ratio,                  // cache_ratio
    slab_to_free,                 // slab_to_free
    r.used_cpu / 100.0f,          // USED_CPU_ATOM (normalized)
    r.load_avg,                   // LOAD_AVG_ATOM
    total_clients,                // total_clients
    mem_per_client,               // mem_per_client
    hour_sin,                     // hour_sin
    hour_cos,                     // hour_cos
    day_sin,                      // day_sin
    day_cos,                      // day_cos
    mem_util_delta,               // mem_utilization_delta
    avail_delta,                  // avail_to_total_delta
    slab_pressure_delta,          // slab_pressure_delta
    free_avail_delta,             // free_to_avail_delta
    std::fabs(mem_util_delta),    // mem_util_delta_abs
    std::fabs(slab_pressure_delta) // slab_delta_abs
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// FlattenWindow
// ─────────────────────────────────────────────────────────────────────────────

std::vector<float> AnomalyPredictionEngine::FlattenWindow(
    const std::deque<std::vector<float>>& window) {
  
  std::vector<float> flat;
  for (const auto& row : window) {
    flat.insert(flat.end(), row.begin(), row.end());
  }
  return flat;
}

// ─────────────────────────────────────────────────────────────────────────────
// RunInference — TCN autoencoder forward pass
// ─────────────────────────────────────────────────────────────────────────────

float AnomalyPredictionEngine::RunInference(
    tflite::Interpreter* interp,
    const std::vector<float>& window_data,
    int window_size,
    int n_features) {
  
  int input_idx = interp->inputs()[0];
  int output_idx = interp->outputs()[0];
  
  // Copy input data to tensor
  float* input_tensor = interp->typed_tensor<float>(input_idx);
  std::memcpy(input_tensor, window_data.data(), 
              window_data.size() * sizeof(float));
  
  // Run inference
  if (interp->Invoke() != kTfLiteOk) {
    std::cerr << "[Inference] Invoke failed\n";
    return -1.0f;
  }
  
  // Get output and compute MSE
  float* output_tensor = interp->typed_tensor<float>(output_idx);
  
  float mse = 0.0f;
  int total_elements = window_size * n_features;
  for (int i = 0; i < total_elements; ++i) {
    float diff = window_data[i] - output_tensor[i];
    mse += diff * diff;
  }
  mse /= static_cast<float>(total_elements);
  
  return mse;
}

// ─────────────────────────────────────────────────────────────────────────────
// ProcessReading — main entry point
// ─────────────────────────────────────────────────────────────────────────────

PredictionResult AnomalyPredictionEngine::ProcessReading(
    const TelemetryReading& reading) {
  
  PredictionResult result;
  result.anomaly_type = "Normal";
  
  if (!initialized_) {
    return result;
  }
  
  DeviceState& state = GetState(reading.mac);
  
  // Compute features for this reading
  std::vector<float> cpu_features, mem_features;
  ComputeCpuFeatures(reading, state, cpu_features);
  ComputeMemFeatures(reading, state, mem_features);
  
  // Apply scaling
  std::vector<float> cpu_scaled = ApplyScaler(cpu_features, cfg_.cpu_scaler);
  std::vector<float> mem_scaled = ApplyScaler(mem_features, cfg_.mem_scaler);
  
  // Add to sliding windows
  state.cpu_window.push_back(cpu_scaled);
  state.mem_window.push_back(mem_scaled);
  
  // Trim windows to size
  while (state.cpu_window.size() > static_cast<size_t>(cfg_.window_size)) {
    state.cpu_window.pop_front();
  }
  while (state.mem_window.size() > static_cast<size_t>(cfg_.window_size)) {
    state.mem_window.pop_front();
  }
  
  state.sample_count++;
  result.window_samples = static_cast<int>(state.cpu_window.size());
  result.window_ready = (result.window_samples >= cfg_.window_size);
  
  // Run inference only if window is full
  if (!result.window_ready) {
    if (verbose_) {
      std::cerr << "[" << reading.mac << "] Window filling: " 
                << result.window_samples << "/" << cfg_.window_size << "\n";
    }
    return result;
  }
  
  double start_time = GetTimeMs();
  
  // Flatten windows for inference
  std::vector<float> cpu_window_flat = FlattenWindow(state.cpu_window);
  std::vector<float> mem_window_flat = FlattenWindow(state.mem_window);
  
  // Run CPU model inference
  result.cpu_mse = RunInference(cpu_interp_.get(), cpu_window_flat,
                                cfg_.window_size, kCpuFeatureCount);
  
  // Run Memory model inference
  result.mem_mse = RunInference(mem_interp_.get(), mem_window_flat,
                                cfg_.window_size, kMemFeatureCount);
  
  double end_time = GetTimeMs();
  result.inference_time_ms = static_cast<float>(end_time - start_time);
  
  // Compute severity
  result.cpu_severity = result.cpu_mse / cfg_.cpu_threshold;
  result.mem_severity = result.mem_mse / cfg_.mem_threshold;
  
  // Apply thresholds (only after warmup)
  if (state.sample_count >= cfg_.warmup_samples) {
    result.cpu_anomaly = (result.cpu_mse > cfg_.cpu_threshold) ? 1 : 0;
    result.mem_anomaly = (result.mem_mse > cfg_.mem_threshold) ? 1 : 0;
    
    // Determine anomaly type
    if (result.cpu_anomaly && result.mem_anomaly) {
      result.anomaly_type = "Both";
    } else if (result.cpu_anomaly) {
      result.anomaly_type = "CPU";
    } else if (result.mem_anomaly) {
      result.anomaly_type = "Memory";
    }
  }
  
  if (verbose_) {
    std::cerr << "[" << reading.mac << "] "
              << "cpu_mse=" << result.cpu_mse << " "
              << "mem_mse=" << result.mem_mse << " "
              << "type=" << result.anomaly_type << " "
              << "time=" << result.inference_time_ms << "ms\n";
  }
  
  return result;
}

}  // namespace anomaly_prediction
}  // namespace tflite
