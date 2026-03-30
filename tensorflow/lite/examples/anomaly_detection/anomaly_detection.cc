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

Exact numerical equivalent of the Python AnomalyInferenceEngine in
inference_app.ipynb. Every arithmetic operation matches the Python:

  MinMaxScaler  : clip((x - data_min) * scale, 0, 1)
  cpu_delta     : cpu_t − cpu_{t-1}  (0 on first reading)
  cpu_roll_mean : mean of window (up to SEQ_LEN values)
  cpu_roll_std  : std with ddof=1 (Bessel's correction), 0 if < 2 values
  MSE           : mean((input − reconstructed)²) over all elements
  Severity      : MSE / threshold
  anomaly_type  : Both > CPU > Memory > Normal  (dense flags only)
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

#include <dlfcn.h>  // dlopen / dlsym — for loading libtensorflowlite_flex.so at runtime

namespace tflite {
namespace anomaly_detection {

using json = nlohmann::json;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

static float ClampF(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Rolling std with ddof=1 — matches numpy.std(window, ddof=1).
// Returns 0 when window has fewer than 2 elements.
static float StdDdof1(const std::vector<float>& w) {
  if (w.size() < 2) return 0.0f;
  float mean = std::accumulate(w.begin(), w.end(), 0.0f) / w.size();
  float sq_sum = 0.0f;
  for (float v : w) sq_sum += (v - mean) * (v - mean);
  return std::sqrt(sq_sum / static_cast<float>(w.size() - 1));
}

// ─────────────────────────────────────────────────────────────────────────────
// LoadConfig — parse inference_config.json
// ─────────────────────────────────────────────────────────────────────────────

static ScalerParams ParseScaler(const json& scaler_j,
                                const std::vector<std::string>& feature_order) {
  ScalerParams s;
  s.feature_order = feature_order;
  for (const auto& f : feature_order) {
    s.data_min.push_back(static_cast<float>(scaler_j["data_min"][f].get<double>()));
    s.scale.push_back(static_cast<float>(scaler_j["scale"][f].get<double>()));
  }
  return s;
}

static ModelConfig ParseModelConfig(const json& j) {
  ModelConfig mc;
  mc.tflite_file       = j["tflite_file"].get<std::string>();
  mc.anomaly_threshold = static_cast<float>(j["anomaly_threshold"].get<double>());
  mc.seq_len           = j.value("seq_len", 1);
  mc.flex_delegate     = j.value("flex_delegate", false);

  std::vector<std::string> feat_order;
  for (const auto& f : j["feature_order"]) feat_order.push_back(f.get<std::string>());
  mc.scaler = ParseScaler(j["scaler"], feat_order);
  return mc;
}


InferenceConfig LoadConfig(const std::string& config_path) {
  std::ifstream f(config_path);
  if (!f) throw std::runtime_error("Cannot open config: " + config_path);

  json j;
  f >> j;

  InferenceConfig cfg;
  cfg.rolling_window  = j.value("rolling_window", 10);
  cfg.cpu_model       = ParseModelConfig(j["cpu_model"]);
  cfg.memory_model    = ParseModelConfig(j["memory_model"]);

  if (j.contains("lstm_cpu_model")) {
    cfg.lstm_cpu_model = ParseModelConfig(j["lstm_cpu_model"]);
    cfg.has_lstm_cpu   = true;
  }
  if (j.contains("lstm_memory_model")) {
    cfg.lstm_memory_model = ParseModelConfig(j["lstm_memory_model"]);
    cfg.has_lstm_mem      = true;
  }
  return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// AnomalyInferenceEngine — constructor
// ─────────────────────────────────────────────────────────────────────────────

AnomalyInferenceEngine::AnomalyInferenceEngine(const std::string& config_path,
                                               int num_threads,
                                               const std::string& delegate_path,
                                               const std::string& delegate_options) {
  cfg_              = LoadConfig(config_path);
  seq_len_          = cfg_.rolling_window;
  num_threads_      = num_threads;
  delegate_path_    = delegate_path;
  delegate_options_ = delegate_options;

  // Dense models are always required
  LoadInterpreter(cfg_.cpu_model.tflite_file,    dense_cpu_fb_, dense_cpu_interp_,
                  false, nullptr, &dense_cpu_ext_delegate_);
  LoadInterpreter(cfg_.memory_model.tflite_file, dense_mem_fb_, dense_mem_interp_,
                  false, nullptr, &dense_mem_ext_delegate_);

  // LSTM models are optional — they use SELECT_TF_OPS (Flex delegate)
  if (cfg_.has_lstm_cpu) {
    try {
      LoadInterpreter(cfg_.lstm_cpu_model.tflite_file, lstm_cpu_fb_, lstm_cpu_interp_,
                      cfg_.lstm_cpu_model.flex_delegate, &lstm_cpu_delegate_,
                      &lstm_cpu_ext_delegate_);
    } catch (const std::exception& e) {
      std::cerr << "Warning: could not load lstm_cpu_model — " << e.what() << "\n";
      cfg_.has_lstm_cpu = false;
    }
  }
  if (cfg_.has_lstm_mem) {
    try {
      LoadInterpreter(cfg_.lstm_memory_model.tflite_file, lstm_mem_fb_, lstm_mem_interp_,
                      cfg_.lstm_memory_model.flex_delegate, &lstm_mem_delegate_,
                      &lstm_mem_ext_delegate_);
    } catch (const std::exception& e) {
      std::cerr << "Warning: could not load lstm_memory_model — " << e.what() << "\n";
      cfg_.has_lstm_mem = false;
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
    bool use_flex_delegate,
    TfLiteDelegateUniquePtr* delegate_out,
    TfLiteDelegateUniquePtr* ext_delegate_out) {

  fb_out = tflite::FlatBufferModel::BuildFromFile(path.c_str());
  if (!fb_out)
    throw std::runtime_error("Failed to load TFLite model: " + path);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*fb_out, resolver)(&interp_out);
  if (!interp_out)
    throw std::runtime_error("Failed to build interpreter for: " + path);

  // Apply external hardware delegate if a path was provided.
  // Skipped for Flex-delegate models (LSTM): those use SELECT_TF_OPS which
  // the hardware delegate does not support, and presenting an unsupported op
  // graph to it causes a hard assert in the delegate core rather than a
  // graceful fallback.
  if (!delegate_path_.empty() && !use_flex_delegate) {
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
      // Capture op codes and node count before delegation.
      // After ModifyGraphWithDelegate, ops accepted by the delegate are fused
      // into delegate node(s). When all accepted ops form one contiguous
      // subgraph: delegated_ops = nodes_before - nodes_after + 1.
      const int nodes_before = static_cast<int>(interp_out->nodes_size());
      std::vector<int32_t> op_codes;
      op_codes.reserve(nodes_before);
      for (int i = 0; i < nodes_before; ++i) {
        auto* nr = interp_out->node_and_registration(i);
        op_codes.push_back(nr ? nr->second.builtin_code : -1);
      }
      if (interp_out->ModifyGraphWithDelegate(raw) != kTfLiteOk) {
        std::cerr << "[delegate] Warning: ModifyGraphWithDelegate failed for: " << path
                  << " — running on CPU\n";
        TfLiteExternalDelegateDelete(raw);
      } else {
        const int nodes_after = static_cast<int>(interp_out->nodes_size());
        // When K ops are accepted in one contiguous subgraph:
        //   nodes_after = (nodes_before - K) + 1  =>  K = nodes_before - nodes_after + 1
        // When 0 ops accepted: nodes_after == nodes_before (no change), K = 0.
        const int delegated_ops = (nodes_after < nodes_before)
                                      ? (nodes_before - nodes_after + 1) : 0;
        const int cpu_ops = nodes_before - delegated_ops;
        std::cerr << "[delegate] Hardware delegate applied for: " << path << "\n"
                  << "[delegate]   ops on NPU : " << delegated_ops
                  << " / " << nodes_before << "\n"
                  << "[delegate]   ops on CPU : " << cpu_ops << "\n";
        if (delegated_ops == 0) {
          std::cerr << "[delegate]   TFLite builtin_code for each op:";
          for (auto c : op_codes) std::cerr << " " << c;
          std::cerr << "\n"
                    << "[delegate]   NPU accepted 0 ops — tensor shapes may be below\n"
                    << "[delegate]   the NPU minimum or the JIT compile step rejected them.\n"
                    << "[delegate]   Inference will run on CPU.\n";
        }
        if (ext_delegate_out)
          *ext_delegate_out = TfLiteDelegateUniquePtr{raw, TfLiteExternalDelegateDelete};
        else
          TfLiteExternalDelegateDelete(raw);
      }
    }
  }

  // Apply the Flex delegate for models that use SELECT_TF_OPS
  // (e.g. LSTM models — UnidirectionalSequenceLSTM is not a TFLite builtin).
  // The flex delegate shared library is loaded dynamically at runtime so that
  // anomaly_app does NOT need to link the full TF runtime at build time.
  if (use_flex_delegate) {
    void* flex_lib = dlopen("libtensorflowlite_flex.so", RTLD_NOW | RTLD_GLOBAL);
    if (!flex_lib)
      throw std::runtime_error(
          "dlopen(libtensorflowlite_flex.so) failed: " + std::string(dlerror()) +
          "\nDeploy libtensorflowlite_flex.so on the target and set LD_LIBRARY_PATH.");

    // TF_AcquireFlexDelegate() is the stable C export from libtensorflowlite_flex.so.
    // It returns TfLiteDelegateUniquePtr (== std::unique_ptr<TfLiteDelegate, void(*)(TfLiteDelegate*)>).
    using AcquireFn = TfLiteDelegateUniquePtr (*)();
    auto* acquire = reinterpret_cast<AcquireFn>(dlsym(flex_lib, "TF_AcquireFlexDelegate"));
    if (!acquire) {
      dlclose(flex_lib);
      throw std::runtime_error(
          "Symbol TF_AcquireFlexDelegate not found in libtensorflowlite_flex.so");
    }

    auto flex_delegate = acquire();
    if (!flex_delegate) {
      dlclose(flex_lib);
      throw std::runtime_error("TF_AcquireFlexDelegate() returned null for: " + path);
    }
    if (interp_out->ModifyGraphWithDelegate(flex_delegate.get()) != kTfLiteOk) {
      dlclose(flex_lib);
      throw std::runtime_error("Failed to apply Flex delegate for: " + path);
    }
    if (delegate_out) *delegate_out = std::move(flex_delegate);
    // Do NOT dlclose(flex_lib) — the delegate's deleter fn lives inside the SO;
    // the SO must stay loaded for the process lifetime. The OS reclaims it at exit.
  }

  interp_out->SetNumThreads(num_threads_);

  if (interp_out->AllocateTensors() != kTfLiteOk)
    throw std::runtime_error("AllocateTensors() failed for: " + path);
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
// ApplyMinMaxScaler — mirrors Python apply_minmax_scaler() exactly
//   scaled = clip((x - data_min) * scale, 0.0, 1.0)
// ─────────────────────────────────────────────────────────────────────────────

std::vector<float> AnomalyInferenceEngine::ApplyMinMaxScaler(
    const std::vector<float>& x, const ScalerParams& scaler) {
  std::vector<float> out(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    out[i] = ClampF((x[i] - scaler.data_min[i]) * scaler.scale[i], 0.0f, 1.0f);
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// ComputeFeatures — mirrors Python _compute_features() exactly
//
// CPU feature vector (10 elements, same order as feature_order in config):
//   [USED_CPU_ATOM, LOAD_AVG_ATOM, load_cpu_divergence, cpu_delta,
//    cpu_rolling_mean, cpu_rolling_std, slab_pressure,
//    total_clients, hour_of_day, day_of_week]
//
// Memory feature vector (9 elements):
//   [USED_MEM_ATOM_kB, AvailMem_kB, FreeMem_kB, SlabMem_kB,
//    memory_util_ratio, slab_pressure, free_mem_ratio,
//    hour_of_day, day_of_week]
// ─────────────────────────────────────────────────────────────────────────────

void AnomalyInferenceEngine::ComputeFeatures(const TelemetryReading& r,
                                             DeviceState& state,
                                             std::vector<float>& cpu_feat,
                                             std::vector<float>& mem_feat) const {
  const float eps = 1e-9f;

  // ── Derived memory ratios (avoid divide-by-zero) ──────────────────────────
  float mem_util_ratio = r.used_mem_kb / (r.used_mem_kb + r.avail_mem_kb + eps);
  float slab_pressure  = r.slab_mem_kb / (r.avail_mem_kb + eps);
  float free_mem_ratio = r.free_mem_kb  / (r.avail_mem_kb + eps);
  float load_cpu_div   = r.load_avg - r.used_cpu;
  float total_clients  = static_cast<float>(r.clients_2g + r.clients_5g + r.clients_6g);

  // ── Rolling CPU stats ─────────────────────────────────────────────────────
  // Build the window: last (SEQ_LEN-1) historical values + current cpu value
  std::vector<float> window;
  window.reserve(seq_len_);
  int history_take = static_cast<int>(state.cpu_history.size());
  int start = std::max(0, history_take - (seq_len_ - 1));
  for (int i = start; i < history_take; ++i)
    window.push_back(state.cpu_history[i]);
  window.push_back(r.used_cpu);

  float cpu_delta     = state.cpu_history.empty()
                          ? 0.0f
                          : r.used_cpu - state.cpu_history.back();
  float cpu_roll_mean = std::accumulate(window.begin(), window.end(), 0.0f)
                        / static_cast<float>(window.size());
  float cpu_roll_std  = StdDdof1(window);   // ddof=1 — matches Python

  // ── Update history ────────────────────────────────────────────────────────
  state.cpu_history.push_back(r.used_cpu);
  if (static_cast<int>(state.cpu_history.size()) > seq_len_)
    state.cpu_history.pop_front();

  // ── Assemble feature vectors ──────────────────────────────────────────────
  cpu_feat = {r.used_cpu, r.load_avg, load_cpu_div, cpu_delta,
              cpu_roll_mean, cpu_roll_std, slab_pressure,
              total_clients,
              static_cast<float>(r.hour_of_day),
              static_cast<float>(r.day_of_week)};

  mem_feat = {r.used_mem_kb, r.avail_mem_kb, r.free_mem_kb, r.slab_mem_kb,
              mem_util_ratio, slab_pressure, free_mem_ratio,
              static_cast<float>(r.hour_of_day),
              static_cast<float>(r.day_of_week)};
}

// ─────────────────────────────────────────────────────────────────────────────
// RunDenseInference — mirrors Python run_dense_inference()
//   inp  shape: (1, n_features)
//   out  shape: (1, n_features)
//   return: mean((inp - out)²)
// ─────────────────────────────────────────────────────────────────────────────

float AnomalyInferenceEngine::RunDenseInference(tflite::Interpreter* interp,
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
// RunLstmInference — mirrors Python run_lstm_inference()
//   seq shape: (seq_len, n_features)  — zero-padded at front if < seq_len rows
//   input tensor shape: (1, seq_len, n_features)
//   return: mean((inp - out)²)  over all seq_len * n_features elements
// ─────────────────────────────────────────────────────────────────────────────

float AnomalyInferenceEngine::RunLstmInference(
    tflite::Interpreter* interp,
    const std::deque<std::vector<float>>& seq,
    int seq_len, int n_features) {

  const int in_idx  = interp->inputs()[0];
  const int out_idx = interp->outputs()[0];
  const int total   = seq_len * n_features;

  float* in_ptr = interp->typed_tensor<float>(in_idx);

  // Zero-fill front padding, then copy real sequences
  int pad_rows = seq_len - static_cast<int>(seq.size());
  int offset   = 0;
  for (int r = 0; r < pad_rows; ++r) {
    for (int c = 0; c < n_features; ++c) in_ptr[offset++] = 0.0f;
  }
  for (const auto& row : seq) {
    for (int c = 0; c < n_features; ++c) in_ptr[offset++] = row[c];
  }

  interp->Invoke();

  const float* out_ptr = interp->typed_tensor<float>(out_idx);
  float mse = 0.0f;
  for (int i = 0; i < total; ++i) {
    float diff = in_ptr[i] - out_ptr[i];
    mse += diff * diff;
  }
  return mse / static_cast<float>(total);
}

// ─────────────────────────────────────────────────────────────────────────────
// ProcessReading — mirrors Python process_reading()
// ─────────────────────────────────────────────────────────────────────────────

AnomalyResult AnomalyInferenceEngine::ProcessReading(const TelemetryReading& r) {
  DeviceState& state = GetState(r.mac);

  // 1. Compute raw feature vectors
  std::vector<float> cpu_feat, mem_feat;
  ComputeFeatures(r, state, cpu_feat, mem_feat);

  // 2. Scale features
  std::vector<float> cpu_sc = ApplyMinMaxScaler(cpu_feat, cfg_.cpu_model.scaler);
  std::vector<float> mem_sc = ApplyMinMaxScaler(mem_feat, cfg_.memory_model.scaler);

  // 3. Dense inference
  float d_cpu_mse = RunDenseInference(dense_cpu_interp_.get(), cpu_sc);
  float d_mem_mse = RunDenseInference(dense_mem_interp_.get(), mem_sc);

  float cpu_thr = cfg_.cpu_model.anomaly_threshold;
  float mem_thr = cfg_.memory_model.anomaly_threshold;

  AnomalyResult result;
  result.dense_cpu_mse  = d_cpu_mse;
  result.dense_mem_mse  = d_mem_mse;
  result.dense_cpu_flag = (d_cpu_mse > cpu_thr) ? 1 : 0;
  result.dense_mem_flag = (d_mem_mse > mem_thr) ? 1 : 0;
  result.dense_cpu_sev  = d_cpu_mse / cpu_thr;
  result.dense_mem_sev  = d_mem_mse / mem_thr;

  // 4. LSTM CPU (optional)
  int cpu_n = static_cast<int>(cfg_.cpu_model.scaler.feature_order.size());
  if (cfg_.has_lstm_cpu && lstm_cpu_interp_) {
    state.cpu_seq.push_back(cpu_sc);
    if (static_cast<int>(state.cpu_seq.size()) > seq_len_)
      state.cpu_seq.pop_front();

    float l_cpu_mse = RunLstmInference(lstm_cpu_interp_.get(),
                                       state.cpu_seq, seq_len_, cpu_n);
    float l_cpu_thr = cfg_.lstm_cpu_model.anomaly_threshold;

    result.lstm_cpu_mse  = l_cpu_mse;
    result.lstm_cpu_flag = (l_cpu_mse > l_cpu_thr) ? 1 : 0;
    result.lstm_cpu_sev  = l_cpu_mse / l_cpu_thr;
    result.has_lstm      = true;
  }

  // 5. LSTM Memory (optional)
  int mem_n = static_cast<int>(cfg_.memory_model.scaler.feature_order.size());
  if (cfg_.has_lstm_mem && lstm_mem_interp_) {
    state.mem_seq.push_back(mem_sc);
    if (static_cast<int>(state.mem_seq.size()) > seq_len_)
      state.mem_seq.pop_front();

    float l_mem_mse = RunLstmInference(lstm_mem_interp_.get(),
                                       state.mem_seq, seq_len_, mem_n);
    float l_mem_thr = cfg_.lstm_memory_model.anomaly_threshold;

    result.lstm_mem_mse  = l_mem_mse;
    result.lstm_mem_flag = (l_mem_mse > l_mem_thr) ? 1 : 0;
    result.lstm_mem_sev  = l_mem_mse / l_mem_thr;
    result.has_lstm      = true;
  }

  // 6. Classify — based on dense flags only (matches Python)
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
