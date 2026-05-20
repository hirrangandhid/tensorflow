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

gpt2_inference.h
────────────────────────────────────────────────────────────────────────────
C++ inference engine for GPT-2 language model on CPE/RDKB devices.

Designed for edge deployment with TFLite runtime and optional NPU acceleration
via BStorm delegate for Broadcom platforms.

Features:
  - Interactive text generation from user prompts
  - Configurable generation parameters (temperature, top-k, max tokens)
  - Support for 8-bit quantized GPT-2 models
  - Optional hardware delegate for NPU acceleration

Model supported:
  gpt2_64-8bits.tflite    GPT-2 8-bit quantized, sequence length 64

Config format (gpt2_config.json):
  - model_file: path to .tflite
  - vocab_file: path to vocabulary JSON
  - max_length: maximum sequence length
  - temperature: sampling temperature
  - top_k: top-k sampling parameter
============================================================================*/

#ifndef TENSORFLOW_LITE_EXAMPLES_GPT2_INFERENCE_GPT2_INFERENCE_H_
#define TENSORFLOW_LITE_EXAMPLES_GPT2_INFERENCE_GPT2_INFERENCE_H_

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
namespace gpt2 {

// ── Generation configuration ─────────────────────────────────────────────────
struct GenerationConfig {
  int    max_new_tokens   = 50;       // Maximum tokens to generate
  float  temperature      = 0.7f;     // Sampling temperature (0 = greedy)
  int    top_k            = 40;       // Top-k sampling (0 = disabled)
  float  top_p            = 0.9f;     // Top-p (nucleus) sampling
  int    repetition_penalty_window = 10;  // Window for repetition penalty
  float  repetition_penalty = 1.1f;   // Penalty for repeated tokens
  bool   do_sample        = true;     // Use sampling vs greedy decoding
  int    eos_token_id     = -1;       // End-of-sequence token (-1 = none)
  int    pad_token_id     = 0;        // Padding token ID
};

// ── Model configuration ───────────────────────────────────────────────────────
struct ModelConfig {
  std::string model_file;             // Path to .tflite model
  std::string vocab_file;             // Path to vocabulary JSON file
  int         max_seq_length = 64;    // Maximum sequence length
  int         vocab_size     = 50257; // GPT-2 vocabulary size
  GenerationConfig generation;        // Generation parameters
};

// ── Tokenizer ─────────────────────────────────────────────────────────────────
// Simple word-piece tokenizer for GPT-2
// For production, use SentencePiece or the full BPE tokenizer
class GPT2Tokenizer {
 public:
  GPT2Tokenizer();
  ~GPT2Tokenizer() = default;

  // Initialize from vocabulary file (JSON format: {"token": id, ...})
  bool LoadVocabulary(const std::string& vocab_path);

  // Initialize with basic vocabulary (built-in fallback)
  void InitializeBasicVocab();

  // Encode text to token IDs
  std::vector<int32_t> Encode(const std::string& text) const;

  // Decode token IDs to text
  std::string Decode(const std::vector<int32_t>& token_ids) const;

  // Get special token IDs
  int32_t GetPadTokenId() const { return pad_token_id_; }
  int32_t GetEosTokenId() const { return eos_token_id_; }
  int32_t GetBosTokenId() const { return bos_token_id_; }
  int32_t GetUnkTokenId() const { return unk_token_id_; }

  // Vocabulary size
  size_t VocabSize() const { return token_to_id_.size(); }

 private:
  std::unordered_map<std::string, int32_t> token_to_id_;
  std::unordered_map<int32_t, std::string> id_to_token_;
  
  int32_t pad_token_id_ = 50256;  // GPT-2 uses <|endoftext|> as pad
  int32_t eos_token_id_ = 50256;  // <|endoftext|>
  int32_t bos_token_id_ = 50256;  // GPT-2 doesn't have explicit BOS
  int32_t unk_token_id_ = 50256;  // Unknown maps to endoftext

  // Simple word splitting for basic tokenization
  std::vector<std::string> SplitIntoWords(const std::string& text) const;
};

// ── Inference Result ──────────────────────────────────────────────────────────
struct InferenceResult {
  std::string              generated_text;     // Generated text response
  std::vector<int32_t>     output_tokens;      // Raw token IDs
  float                    inference_time_ms;  // Time for generation
  int                      tokens_generated;   // Number of new tokens
  bool                     success;            // Whether inference succeeded
  std::string              error_message;      // Error if failed
};

// ── Statistics ────────────────────────────────────────────────────────────────
struct InferenceStats {
  int    total_inferences = 0;
  float  avg_time_ms      = 0.0f;
  float  total_time_ms    = 0.0f;
  int    total_tokens     = 0;
  float  tokens_per_sec   = 0.0f;
};

// ─────────────────────────────────────────────────────────────────────────────
// GPT2InferenceEngine
//
// TFLite-based GPT-2 inference engine for text generation.
// Supports hardware acceleration via external delegates (e.g., BStorm NPU).
//
// Usage:
//   GPT2InferenceEngine engine("gpt2_config.json");
//   InferenceResult result = engine.Generate("Hello, how are you?");
//   std::cout << result.generated_text << std::endl;
// ─────────────────────────────────────────────────────────────────────────────
class GPT2InferenceEngine {
 public:
  // Constructor
  // config_path      : Path to gpt2_config.json
  // model_path       : Path to TFLite model (overrides config if not empty)
  // delegate_path    : Path to external delegate .so (e.g., BStorm)
  // delegate_options : Semicolon-separated key:value pairs
  // num_threads      : Number of CPU threads for TFLite
  // verbose          : Enable verbose logging
  explicit GPT2InferenceEngine(const std::string& config_path,
                                const std::string& model_path = "",
                                const std::string& delegate_path = "",
                                const std::string& delegate_options = "",
                                int num_threads = 1,
                                bool verbose = false);

  ~GPT2InferenceEngine() = default;

  // Generate text from a prompt
  InferenceResult Generate(const std::string& prompt);
  
  // Generate with custom configuration
  InferenceResult Generate(const std::string& prompt,
                           const GenerationConfig& config);

  // Get current configuration
  const ModelConfig& GetConfig() const { return config_; }

  // Get inference statistics
  const InferenceStats& GetStats() const { return stats_; }

  // Reset statistics
  void ResetStats() { stats_ = InferenceStats(); }

  // Check if engine is ready
  bool IsReady() const { return is_initialized_; }

  // Get last error
  const std::string& GetLastError() const { return last_error_; }

 private:
  // Load configuration from JSON file
  bool LoadConfig(const std::string& config_path);

  // Initialize TFLite interpreter
  bool LoadInterpreter();

  // Run single forward pass
  bool RunInference(const std::vector<int32_t>& input_ids,
                    std::vector<float>& logits);

  // Sample next token from logits
  int32_t SampleToken(const std::vector<float>& logits,
                      const GenerationConfig& config,
                      const std::vector<int32_t>& generated_tokens);

  // Apply temperature scaling
  void ApplyTemperature(std::vector<float>& logits, float temperature);

  // Apply top-k filtering
  void ApplyTopK(std::vector<float>& logits, int k);

  // Apply top-p (nucleus) filtering
  void ApplyTopP(std::vector<float>& logits, float p);

  // Apply repetition penalty
  void ApplyRepetitionPenalty(std::vector<float>& logits,
                               const std::vector<int32_t>& generated_tokens,
                               float penalty, int window);

  // Softmax
  void Softmax(std::vector<float>& logits);

  // Sample from probability distribution
  int32_t SampleFromDistribution(const std::vector<float>& probs);

  // Configuration
  ModelConfig config_;
  bool        is_initialized_ = false;
  bool        verbose_        = false;
  int         num_threads_    = 1;
  std::string delegate_path_;
  std::string delegate_options_;
  std::string last_error_;

  // TFLite components
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter>     interpreter_;
  TfLiteDelegateUniquePtr                  external_delegate_;

  // Tokenizer
  GPT2Tokenizer tokenizer_;

  // Statistics
  InferenceStats stats_;

  // Input/output tensor info
  int input_tensor_idx_  = -1;
  int output_tensor_idx_ = -1;
};

// ── Utility functions ─────────────────────────────────────────────────────────

// Load model configuration from JSON file
ModelConfig LoadModelConfig(const std::string& config_path);

// Check if TFLite delegate library is available
bool CheckDelegateAvailable(const std::string& delegate_path);

}  // namespace gpt2
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXAMPLES_GPT2_INFERENCE_GPT2_INFERENCE_H_
