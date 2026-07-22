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
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/time.h>
#include <sys/stat.h>

#include "nlohmann_json/json.hpp"
#include "tensorflow/lite/delegates/external/external_delegate.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"

namespace tflite {
namespace anomaly_prediction {

using json = nlohmann::json;

// ══════════════════════════════════════════════════════════════════════════════
// ENGINE LOGGING - writes to same log file as main
// ══════════════════════════════════════════════════════════════════════════════

static const char* ENGINE_LOG_PATH = "/rdklogs/logs/anomaly_predict_logs.txt";
static std::ofstream g_engine_log;
static std::mutex g_engine_log_mutex;
static bool g_engine_log_init = false;

static std::string EngineTimestamp() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  struct tm* tm_info = localtime(&tv.tv_sec);
  char buf[64];
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm_info);
  char result[80];
  snprintf(result, sizeof(result), "%s.%03d", buf, (int)(tv.tv_usec / 1000));
  return std::string(result);
}

static void InitEngineLog() {
  if (g_engine_log_init) return;
  std::lock_guard<std::mutex> lock(g_engine_log_mutex);
  if (g_engine_log_init) return;
  mkdir("/rdklogs", 0755);
  mkdir("/rdklogs/logs", 0755);
  g_engine_log.open(ENGINE_LOG_PATH, std::ios::app);
  g_engine_log_init = g_engine_log.is_open();
}

static void EngineLog(const char* level, const std::string& comp, const std::string& msg) {
  if (!g_engine_log_init) InitEngineLog();
  if (!g_engine_log_init) return;
  std::lock_guard<std::mutex> lock(g_engine_log_mutex);
  g_engine_log << "[" << EngineTimestamp() << "] [" << level << "] [" 
               << comp << "] " << msg << "\n";
  g_engine_log.flush();
}

#define ELOG_DEBUG(c, m) EngineLog("DEBUG", c, m)
#define ELOG_INFO(c, m)  EngineLog("INFO ", c, m)
#define ELOG_WARN(c, m)  do { EngineLog("WARN ", c, m); std::cerr << "[" << c << "] WARN: " << m << "\n"; } while(0)
#define ELOG_ERROR(c, m) do { EngineLog("ERROR", c, m); std::cerr << "[" << c << "] ERROR: " << m << "\n"; } while(0)

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
  ELOG_INFO("Config", "Loading configuration from: " + config_path);
  
  std::ifstream f(config_path);
  if (!f) {
    ELOG_ERROR("Config", "Cannot open config file: " + config_path);
    throw std::runtime_error("Cannot open config: " + config_path);
  }

  json j;
  try {
    f >> j;
    ELOG_DEBUG("Config", "JSON parsed successfully");
  } catch (const std::exception& e) {
    ELOG_ERROR("Config", "JSON parse error: " + std::string(e.what()));
    throw;
  }

  PredictionConfig cfg;
  
  // Model type: "autoencoder" (default), "forecaster", or "proactive_classifier"
  std::string model_type_str = j.value("model_type", "autoencoder");
  if (model_type_str == "forecaster") {
    cfg.model_type = ModelType::kForecaster;
    ELOG_INFO("Config", "Model type: FORECASTER");
  } else if (model_type_str == "proactive_classifier" || model_type_str == "classifier") {
    cfg.model_type = ModelType::kClassifier;
    ELOG_INFO("Config", "Model type: PROACTIVE CLASSIFIER");
  } else {
    cfg.model_type = ModelType::kAutoencoder;
    ELOG_INFO("Config", "Model type: AUTOENCODER");
  }
  
  // Model files
  cfg.cpu_model_file = j.value("cpu_model_file", "tcn_cpu_anomaly_model_v2.tflite");
  cfg.mem_model_file = j.value("mem_model_file", "tcn_mem_anomaly_model_v2.tflite");
  ELOG_INFO("Config", "CPU model file: " + cfg.cpu_model_file);
  ELOG_INFO("Config", "Memory model file: " + cfg.mem_model_file);
  
  // Thresholds
  cfg.cpu_threshold = static_cast<float>(j.value("cpu_threshold", 0.00769));
  cfg.mem_threshold = static_cast<float>(j.value("mem_threshold", 0.00545));
  ELOG_INFO("Config", "CPU threshold: " + std::to_string(cfg.cpu_threshold));
  ELOG_INFO("Config", "Memory threshold: " + std::to_string(cfg.mem_threshold));
  
  // Classifier-specific settings
  cfg.probability_threshold = static_cast<float>(j.value("probability_threshold", 0.5));
  cfg.prediction_horizon = j.value("prediction_horizon", 5);
  if (cfg.model_type == ModelType::kClassifier) {
    ELOG_INFO("Config", "Probability threshold: " + std::to_string(cfg.probability_threshold));
    ELOG_INFO("Config", "Prediction horizon: " + std::to_string(cfg.prediction_horizon) + " steps");
  }
  
  // Window configuration
  cfg.window_size = j.value("window_size", kDefaultWindowSize);
  cfg.warmup_samples = j.value("warmup_samples", cfg.window_size);
  ELOG_INFO("Config", "Window size: " + std::to_string(cfg.window_size));
  ELOG_INFO("Config", "Warmup samples: " + std::to_string(cfg.warmup_samples));
  
  // Model version
  cfg.model_version = j.value("model_version", "v2_enhanced");
  ELOG_DEBUG("Config", "Model version: " + cfg.model_version);
  
  // NPU compatibility
  cfg.bstorm_compatible = j.value("bstorm_compatible", true);
  ELOG_INFO("Config", "bstorm compatible: " + std::string(cfg.bstorm_compatible ? "yes" : "no"));

  // Parse CPU features
  if (j.contains("cpu_features")) {
    for (const auto& feat : j["cpu_features"]) {
      cfg.cpu_scaler.features.push_back(feat.get<std::string>());
    }
    ELOG_DEBUG("Config", "Loaded " + std::to_string(cfg.cpu_scaler.features.size()) + " CPU features");
  }
  
  // Parse Memory features
  if (j.contains("mem_features")) {
    for (const auto& feat : j["mem_features"]) {
      cfg.mem_scaler.features.push_back(feat.get<std::string>());
    }
    ELOG_DEBUG("Config", "Loaded " + std::to_string(cfg.mem_scaler.features.size()) + " Memory features");
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
    ELOG_DEBUG("Config", "Loaded CPU scaler with " + std::to_string(cfg.cpu_scaler.data_min.size()) + " min/max values");
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
    ELOG_DEBUG("Config", "Loaded Memory scaler with " + std::to_string(cfg.mem_scaler.data_min.size()) + " min/max values");
  }

  ELOG_INFO("Config", "Configuration loaded successfully");
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
  
  ELOG_INFO("Engine", "Initializing AnomalyPredictionEngine");
  ELOG_DEBUG("Engine", "  config_path: " + config_path);
  ELOG_DEBUG("Engine", "  num_threads: " + std::to_string(num_threads));
  ELOG_DEBUG("Engine", "  delegate_path: " + (delegate_path.empty() ? "(none)" : delegate_path));
  ELOG_DEBUG("Engine", "  delegate_options: " + (delegate_options.empty() ? "(none)" : delegate_options));
  ELOG_DEBUG("Engine", "  verbose: " + std::string(verbose ? "yes" : "no"));
  
  num_threads_      = num_threads;
  delegate_path_    = delegate_path;
  delegate_options_ = delegate_options;
  verbose_          = verbose;

  try {
    // Load configuration
    ELOG_INFO("Engine", "Loading configuration...");
    LoadConfig(config_path);
    ELOG_INFO("Engine", "Configuration loaded successfully");

    // Load CPU model
    ELOG_INFO("Engine", "Loading CPU model: " + cfg_.cpu_model_file);
    LoadInterpreter(cfg_.cpu_model_file, cpu_model_fb_, cpu_interp_,
                    &cpu_ext_delegate_, "CPU");
    ELOG_INFO("Engine", "CPU model loaded successfully");

    // Load Memory model
    ELOG_INFO("Engine", "Loading Memory model: " + cfg_.mem_model_file);
    LoadInterpreter(cfg_.mem_model_file, mem_model_fb_, mem_interp_,
                    &mem_ext_delegate_, "Memory");
    ELOG_INFO("Engine", "Memory model loaded successfully");

    initialized_ = true;
    ELOG_INFO("Engine", "Engine initialized successfully");
    
    if (verbose_) {
      std::cerr << "[AnomalyPrediction] Initialized successfully\n";
      std::cerr << "  CPU threshold:  " << cfg_.cpu_threshold << "\n";
      std::cerr << "  Mem threshold:  " << cfg_.mem_threshold << "\n";
      std::cerr << "  Window size:    " << cfg_.window_size << "\n";
    }
  } catch (const std::exception& e) {
    ELOG_ERROR("Engine", "Initialization failed: " + std::string(e.what()));
    std::cerr << "[AnomalyPrediction] Initialization failed: " << e.what() << "\n";
    initialized_ = false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// LoadConfig
// ─────────────────────────────────────────────────────────────────────────────

void AnomalyPredictionEngine::LoadConfig(const std::string& config_path) {
  ELOG_DEBUG("Engine", "LoadConfig called for: " + config_path);
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

  ELOG_INFO("Model", "Loading TFLite model [" + model_name + "]: " + path);
  
  fb_out = tflite::FlatBufferModel::BuildFromFile(path.c_str());
  if (!fb_out) {
    ELOG_ERROR("Model", "Failed to load TFLite model: " + path);
    throw std::runtime_error("Failed to load TFLite model: " + path);
  }
  ELOG_DEBUG("Model", "FlatBufferModel created for " + model_name);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  ELOG_DEBUG("Model", "Building interpreter for " + model_name);
  tflite::InterpreterBuilder(*fb_out, resolver)(&interp_out);
  if (!interp_out) {
    ELOG_ERROR("Model", "Failed to build interpreter for: " + path);
    throw std::runtime_error("Failed to build interpreter for: " + path);
  }
  ELOG_DEBUG("Model", "Interpreter built successfully for " + model_name);

  // Apply external hardware delegate (e.g., bstorm) if provided
  if (!delegate_path_.empty()) {
    ELOG_INFO("Delegate", "Applying external delegate for " + model_name + ": " + delegate_path_);
    TfLiteExternalDelegateOptions opts =
        TfLiteExternalDelegateOptionsDefault(delegate_path_.c_str());

    // Parse delegate options (key:value pairs separated by semicolons)
    std::vector<std::string> opt_keys, opt_vals;
    if (!delegate_options_.empty()) {
      ELOG_DEBUG("Delegate", "Parsing delegate options: " + delegate_options_);
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
          ELOG_DEBUG("Delegate", "  option: " + opt_keys.back() + " = " + opt_vals.back());
          TfLiteExternalDelegateOptionsInsert(
              &opts, opt_keys.back().c_str(), opt_vals.back().c_str());
        }
      }
    }

    ELOG_INFO("Delegate", "Creating external delegate for " + model_name);
    TfLiteDelegate* raw = TfLiteExternalDelegateCreate(&opts);
    if (!raw) {
      ELOG_WARN("Delegate", "TfLiteExternalDelegateCreate failed for " + model_name + " — running on CPU");
      std::cerr << "[delegate] Warning: TfLiteExternalDelegateCreate failed for "
                << model_name << " — running on CPU\n";
    } else {
      ELOG_INFO("Delegate", "Modifying graph with delegate for " + model_name);
      if (interp_out->ModifyGraphWithDelegate(raw) != kTfLiteOk) {
        ELOG_WARN("Delegate", "ModifyGraphWithDelegate failed for " + model_name + " — running on CPU");
        std::cerr << "[delegate] Warning: ModifyGraphWithDelegate failed for "
                  << model_name << " — running on CPU\n";
        TfLiteExternalDelegateDelete(raw);
      } else {
        ELOG_INFO("Delegate", "Hardware delegate applied successfully for " + model_name);
        std::cerr << "[delegate] Hardware delegate applied for " << model_name << "\n";
        if (ext_delegate_out)
          *ext_delegate_out = TfLiteDelegateUniquePtr{raw, TfLiteExternalDelegateDelete};
        else
          TfLiteExternalDelegateDelete(raw);
      }
    }
  } else {
    ELOG_DEBUG("Delegate", "No external delegate specified, using CPU for " + model_name);
  }

  ELOG_DEBUG("Model", "Setting thread count to " + std::to_string(num_threads_) + " for " + model_name);
  interp_out->SetNumThreads(num_threads_);

  ELOG_INFO("Model", "Allocating tensors for " + model_name);
  if (interp_out->AllocateTensors() != kTfLiteOk) {
    ELOG_ERROR("Model", "AllocateTensors() failed for: " + path);
    throw std::runtime_error("AllocateTensors() failed for: " + path);
  }
  ELOG_INFO("Model", "Tensors allocated successfully for " + model_name);

  // Log tensor shape
  int input_idx = interp_out->inputs()[0];
  TfLiteIntArray* dims = interp_out->tensor(input_idx)->dims;
  std::ostringstream shape_ss;
  shape_ss << "[";
  for (int i = 0; i < dims->size; ++i) {
    shape_ss << dims->data[i];
    if (i < dims->size - 1) shape_ss << ", ";
  }
  shape_ss << "]";
  ELOG_INFO("Model", model_name + " input shape: " + shape_ss.str());

  if (verbose_) {
    std::cerr << "[" << model_name << "] Model loaded: " << path << "\n";
    std::cerr << "  Input shape: " << shape_ss.str() << "\n";
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
  
  ELOG_DEBUG("Inference", "RunInference called - window_size=" + std::to_string(window_size) + 
             " n_features=" + std::to_string(n_features) + 
             " data_size=" + std::to_string(window_data.size()));
  
  int input_idx = interp->inputs()[0];
  int output_idx = interp->outputs()[0];
  
  // Copy input data to tensor
  float* input_tensor = interp->typed_tensor<float>(input_idx);
  if (!input_tensor) {
    ELOG_ERROR("Inference", "Failed to get input tensor pointer");
    return -1.0f;
  }
  std::memcpy(input_tensor, window_data.data(), 
              window_data.size() * sizeof(float));
  ELOG_DEBUG("Inference", "Input data copied to tensor");
  
  // Run inference
  ELOG_DEBUG("Inference", "Invoking interpreter...");
  TfLiteStatus status = interp->Invoke();
  if (status != kTfLiteOk) {
    ELOG_ERROR("Inference", "Invoke failed with status: " + std::to_string(status));
    std::cerr << "[Inference] Invoke failed\n";
    return -1.0f;
  }
  ELOG_DEBUG("Inference", "Invoke completed successfully");
  
  // Get output and compute MSE
  float* output_tensor = interp->typed_tensor<float>(output_idx);
  if (!output_tensor) {
    ELOG_ERROR("Inference", "Failed to get output tensor pointer");
    return -1.0f;
  }
  
  float mse = 0.0f;
  int total_elements = window_size * n_features;
  for (int i = 0; i < total_elements; ++i) {
    float diff = window_data[i] - output_tensor[i];
    mse += diff * diff;
  }
  mse /= static_cast<float>(total_elements);
  
  ELOG_DEBUG("Inference", "MSE computed: " + std::to_string(mse));
  return mse;
}

// ─────────────────────────────────────────────────────────────────────────────
// RunForecasterInference — TCN forecaster predicts next timestep
// Returns: predicted next-step features (n_features)
// ─────────────────────────────────────────────────────────────────────────────

std::vector<float> AnomalyPredictionEngine::RunForecasterInference(
    tflite::Interpreter* interp,
    const std::vector<float>& window_data,
    int n_features) {
  
  ELOG_DEBUG("Forecaster", "RunForecasterInference called - n_features=" + std::to_string(n_features));
  
  std::vector<float> prediction(n_features, 0.0f);
  
  int input_idx = interp->inputs()[0];
  int output_idx = interp->outputs()[0];
  
  // Copy input data to tensor
  float* input_tensor = interp->typed_tensor<float>(input_idx);
  if (!input_tensor) {
    ELOG_ERROR("Forecaster", "Failed to get input tensor pointer");
    return prediction;
  }
  std::memcpy(input_tensor, window_data.data(), 
              window_data.size() * sizeof(float));
  
  // Run inference
  ELOG_DEBUG("Forecaster", "Invoking interpreter...");
  TfLiteStatus status = interp->Invoke();
  if (status != kTfLiteOk) {
    ELOG_ERROR("Forecaster", "Invoke failed with status: " + std::to_string(status));
    std::cerr << "[Forecaster] Invoke failed\n";
    return prediction;
  }
  ELOG_DEBUG("Forecaster", "Invoke completed successfully");
  
  // Get output: shape (1, n_features) — predicted next timestep
  float* output_tensor = interp->typed_tensor<float>(output_idx);
  if (!output_tensor) {
    ELOG_ERROR("Forecaster", "Failed to get output tensor pointer");
    return prediction;
  }
  for (int i = 0; i < n_features; ++i) {
    prediction[i] = output_tensor[i];
  }
  
  return prediction;
}

// ─────────────────────────────────────────────────────────────────────────────
// ComputeForecastError — MSE between predicted and actual features
// ─────────────────────────────────────────────────────────────────────────────

float ComputeForecastError(const std::vector<float>& predicted,
                           const std::vector<float>& actual) {
  if (predicted.size() != actual.size() || predicted.empty()) {
    return 0.0f;
  }
  float mse = 0.0f;
  for (size_t i = 0; i < predicted.size(); ++i) {
    float diff = predicted[i] - actual[i];
    mse += diff * diff;
  }
  return mse / static_cast<float>(predicted.size());
}

// ─────────────────────────────────────────────────────────────────────────────
// RunClassifierInference — TCN proactive classifier predicts anomaly probability
// Returns: probability (0-1) that anomaly will occur within prediction_horizon
// ─────────────────────────────────────────────────────────────────────────────

float AnomalyPredictionEngine::RunClassifierInference(
    tflite::Interpreter* interp,
    const std::vector<float>& window_data) {
  
  ELOG_DEBUG("Classifier", "RunClassifierInference called");
  
  int input_idx = interp->inputs()[0];
  int output_idx = interp->outputs()[0];
  
  // Copy input data to tensor
  float* input_tensor = interp->typed_tensor<float>(input_idx);
  if (!input_tensor) {
    ELOG_ERROR("Classifier", "Failed to get input tensor pointer");
    return 0.0f;
  }
  std::memcpy(input_tensor, window_data.data(), 
              window_data.size() * sizeof(float));
  
  // Run inference
  ELOG_DEBUG("Classifier", "Invoking interpreter...");
  TfLiteStatus status = interp->Invoke();
  if (status != kTfLiteOk) {
    ELOG_ERROR("Classifier", "Invoke failed with status: " + std::to_string(status));
    std::cerr << "[Classifier] Invoke failed\n";
    return 0.0f;
  }
  ELOG_DEBUG("Classifier", "Invoke completed successfully");
  
  // Get output: shape (1, 1) — probability
  float* output_tensor = interp->typed_tensor<float>(output_idx);
  if (!output_tensor) {
    ELOG_ERROR("Classifier", "Failed to get output tensor pointer");
    return 0.0f;
  }
  
  float probability = output_tensor[0];
  // Clamp to valid range (sigmoid should already be 0-1, but be safe)
  probability = std::max(0.0f, std::min(1.0f, probability));
  
  ELOG_DEBUG("Classifier", "Output probability: " + std::to_string(probability));
  return probability;
}

// ─────────────────────────────────────────────────────────────────────────────
// ProcessReading — main entry point
// ─────────────────────────────────────────────────────────────────────────────

PredictionResult AnomalyPredictionEngine::ProcessReading(
    const TelemetryReading& reading) {
  
  PredictionResult result;
  result.anomaly_type = "Normal";
  
  if (!initialized_) {
    ELOG_WARN("Process", "ProcessReading called but engine not initialized");
    return result;
  }
  
  ELOG_DEBUG("Process", "ProcessReading - mac=" + reading.mac + " ts=" + reading.timestamp);
  
  DeviceState& state = GetState(reading.mac);
  
  // Compute features for this reading
  ELOG_DEBUG("Process", "Computing features for " + reading.mac);
  std::vector<float> cpu_features, mem_features;
  ComputeCpuFeatures(reading, state, cpu_features);
  ComputeMemFeatures(reading, state, mem_features);
  ELOG_DEBUG("Process", "Features computed - cpu=" + std::to_string(cpu_features.size()) + 
             " mem=" + std::to_string(mem_features.size()));
  
  // Apply scaling
  std::vector<float> cpu_scaled = ApplyScaler(cpu_features, cfg_.cpu_scaler);
  std::vector<float> mem_scaled = ApplyScaler(mem_features, cfg_.mem_scaler);
  ELOG_DEBUG("Process", "Features scaled");
  
  // ═══════════════════════════════════════════════════════════════════════════
  // FORECASTER MODE: Compare previous prediction with current actual
  // ═══════════════════════════════════════════════════════════════════════════
  if (cfg_.model_type == ModelType::kForecaster) {
    ELOG_DEBUG("Process", "Forecaster mode - checking previous predictions");
    // If we have a previous prediction, compute forecast error
    if (state.has_cpu_prediction && state.sample_count >= cfg_.warmup_samples) {
      result.cpu_mse = ComputeForecastError(state.last_cpu_prediction, cpu_scaled);
      result.cpu_severity = result.cpu_mse / cfg_.cpu_threshold;
      result.cpu_anomaly = (result.cpu_mse > cfg_.cpu_threshold) ? 1 : 0;
      ELOG_DEBUG("Process", "CPU forecast error: " + std::to_string(result.cpu_mse));
    }
    if (state.has_mem_prediction && state.sample_count >= cfg_.warmup_samples) {
      result.mem_mse = ComputeForecastError(state.last_mem_prediction, mem_scaled);
      result.mem_severity = result.mem_mse / cfg_.mem_threshold;
      result.mem_anomaly = (result.mem_mse > cfg_.mem_threshold) ? 1 : 0;
      ELOG_DEBUG("Process", "MEM forecast error: " + std::to_string(result.mem_mse));
    }
    
    // Determine anomaly type
    if (result.cpu_anomaly && result.mem_anomaly) {
      result.anomaly_type = "Both";
    } else if (result.cpu_anomaly) {
      result.anomaly_type = "CPU";
    } else if (result.mem_anomaly) {
      result.anomaly_type = "Memory";
    }
    ELOG_DEBUG("Process", "Forecaster anomaly_type: " + result.anomaly_type);
  }
  
  // Add to sliding windows
  state.cpu_window.push_back(cpu_scaled);
  state.mem_window.push_back(mem_scaled);
  ELOG_DEBUG("Process", "Added to windows - cpu_window_size=" + std::to_string(state.cpu_window.size()));
  
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
    ELOG_DEBUG("Process", "Window not ready - " + std::to_string(result.window_samples) + "/" + 
               std::to_string(cfg_.window_size));
    if (verbose_) {
      std::cerr << "[" << reading.mac << "] Window filling: " 
                << result.window_samples << "/" << cfg_.window_size << "\n";
    }
    return result;
  }
  
  ELOG_INFO("Process", "Window ready, running inference for " + reading.mac);
  double start_time = GetTimeMs();
  
  // Flatten windows for inference
  std::vector<float> cpu_window_flat = FlattenWindow(state.cpu_window);
  std::vector<float> mem_window_flat = FlattenWindow(state.mem_window);
  ELOG_DEBUG("Process", "Flattened windows - cpu=" + std::to_string(cpu_window_flat.size()) + 
             " mem=" + std::to_string(mem_window_flat.size()));
  
  // ═══════════════════════════════════════════════════════════════════════════
  // AUTOENCODER MODE: Compute reconstruction error
  // ═══════════════════════════════════════════════════════════════════════════
  if (cfg_.model_type == ModelType::kAutoencoder) {
    ELOG_DEBUG("Process", "Autoencoder mode - running CPU inference");
    // Run CPU model inference
    result.cpu_mse = RunInference(cpu_interp_.get(), cpu_window_flat,
                                  cfg_.window_size, kCpuFeatureCount);
    ELOG_DEBUG("Process", "CPU inference done, MSE=" + std::to_string(result.cpu_mse));
    
    ELOG_DEBUG("Process", "Running MEM inference");
    // Run Memory model inference
    result.mem_mse = RunInference(mem_interp_.get(), mem_window_flat,
                                  cfg_.window_size, kMemFeatureCount);
    ELOG_DEBUG("Process", "MEM inference done, MSE=" + std::to_string(result.mem_mse));
    
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
      ELOG_DEBUG("Process", "Autoencoder anomaly_type: " + result.anomaly_type);
    } else {
      ELOG_DEBUG("Process", "Still in warmup period - sample_count=" + 
                 std::to_string(state.sample_count) + " warmup=" + std::to_string(cfg_.warmup_samples));
    }
  }
  // ═══════════════════════════════════════════════════════════════════════════
  // CLASSIFIER MODE: Proactive anomaly prediction
  // Predicts probability of anomaly within next N steps
  // ═══════════════════════════════════════════════════════════════════════════
  else if (cfg_.model_type == ModelType::kClassifier) {
    ELOG_DEBUG("Process", "Classifier mode - predicting anomaly probability");
    result.is_proactive = true;
    
    // Run CPU classifier
    result.cpu_anomaly_probability = RunClassifierInference(
        cpu_interp_.get(), cpu_window_flat);
    ELOG_DEBUG("Process", "CPU classifier probability: " + 
               std::to_string(result.cpu_anomaly_probability));
    
    // Run Memory classifier
    result.mem_anomaly_probability = RunClassifierInference(
        mem_interp_.get(), mem_window_flat);
    ELOG_DEBUG("Process", "MEM classifier probability: " + 
               std::to_string(result.mem_anomaly_probability));
    
    // Set severity as the probability itself
    result.cpu_severity = result.cpu_anomaly_probability;
    result.mem_severity = result.mem_anomaly_probability;
    
    // Apply probability threshold (only after warmup)
    if (state.sample_count >= cfg_.warmup_samples) {
      result.cpu_anomaly = (result.cpu_anomaly_probability > cfg_.probability_threshold) ? 1 : 0;
      result.mem_anomaly = (result.mem_anomaly_probability > cfg_.probability_threshold) ? 1 : 0;
      
      // Determine anomaly type
      if (result.cpu_anomaly && result.mem_anomaly) {
        result.anomaly_type = "Both";
      } else if (result.cpu_anomaly) {
        result.anomaly_type = "CPU";
      } else if (result.mem_anomaly) {
        result.anomaly_type = "Memory";
      }
      ELOG_DEBUG("Process", "Classifier anomaly_type: " + result.anomaly_type + 
                 " (proactive, predicting " + std::to_string(cfg_.prediction_horizon) + " steps ahead)");
    } else {
      ELOG_DEBUG("Process", "Still in warmup period - sample_count=" + 
                 std::to_string(state.sample_count) + " warmup=" + std::to_string(cfg_.warmup_samples));
    }
  }
  // ═══════════════════════════════════════════════════════════════════════════
  // FORECASTER MODE: Predict next timestep
  // ═══════════════════════════════════════════════════════════════════════════
  else {
    ELOG_DEBUG("Process", "Forecaster mode - predicting next timestep");
    // Run forecaster to predict NEXT timestep
    state.last_cpu_prediction = RunForecasterInference(
        cpu_interp_.get(), cpu_window_flat, kCpuFeatureCount);
    state.has_cpu_prediction = !state.last_cpu_prediction.empty();
    ELOG_DEBUG("Process", "CPU forecaster prediction stored, has_prediction=" + 
               std::to_string(state.has_cpu_prediction));
    
    state.last_mem_prediction = RunForecasterInference(
        mem_interp_.get(), mem_window_flat, kMemFeatureCount);
    state.has_mem_prediction = !state.last_mem_prediction.empty();
    ELOG_DEBUG("Process", "MEM forecaster prediction stored, has_prediction=" + 
               std::to_string(state.has_mem_prediction));
  }
  
  double end_time = GetTimeMs();
  result.inference_time_ms = static_cast<float>(end_time - start_time);
  
  ELOG_INFO("Process", "Inference complete - mac=" + reading.mac + 
            " cpu_mse=" + std::to_string(result.cpu_mse) + 
            " mem_mse=" + std::to_string(result.mem_mse) + 
            " type=" + result.anomaly_type + 
            " time=" + std::to_string(result.inference_time_ms) + "ms");
  
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
