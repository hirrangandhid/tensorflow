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

gpt2_inference.cc
────────────────────────────────────────────────────────────────────────────
C++ inference engine for GPT-2 language model on CPE/RDKB devices.
============================================================================*/

#include "gpt2_inference.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

#include "nlohmann_json/json.hpp"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"

namespace tflite {
namespace gpt2 {

using json = nlohmann::json;

// ─────────────────────────────────────────────────────────────────────────────
// Random number generator for sampling
// ─────────────────────────────────────────────────────────────────────────────
static std::mt19937& GetRNG() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  return gen;
}

// ─────────────────────────────────────────────────────────────────────────────
// Float16 to Float32 conversion
// Converts IEEE 754 half-precision (binary16) to single-precision (binary32)
// ─────────────────────────────────────────────────────────────────────────────
static float Fp16ToFloat32(uint16_t fp16) {
  uint32_t sign = (fp16 >> 15) & 0x1;
  uint32_t exponent = (fp16 >> 10) & 0x1F;
  uint32_t mantissa = fp16 & 0x3FF;

  uint32_t fp32;
  if (exponent == 0) {
    if (mantissa == 0) {
      // Zero
      fp32 = sign << 31;
    } else {
      // Denormalized number - convert to normalized
      exponent = 1;
      while ((mantissa & 0x400) == 0) {
        mantissa <<= 1;
        exponent--;
      }
      mantissa &= 0x3FF;
      fp32 = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    // Inf or NaN
    fp32 = (sign << 31) | (0xFF << 23) | (mantissa << 13);
  } else {
    // Normalized number
    fp32 = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }

  float result;
  std::memcpy(&result, &fp32, sizeof(float));
  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// GPT-2 BPE Byte Decoder
// GPT-2 uses byte-level BPE where bytes are encoded as Unicode characters.
// This decoder converts those Unicode characters back to UTF-8 bytes.
// ─────────────────────────────────────────────────────────────────────────────
class BPEByteDecoder {
 public:
  BPEByteDecoder() {
    InitializeMapping();
  }

  // Decode a BPE token string to regular UTF-8 text
  std::string Decode(const std::string& token) const {
    std::string result;
    size_t i = 0;
    while (i < token.size()) {
      // Decode UTF-8 code point
      uint32_t codepoint = 0;
      size_t bytes_consumed = 0;
      
      unsigned char c = token[i];
      if ((c & 0x80) == 0) {
        // Single byte (ASCII)
        codepoint = c;
        bytes_consumed = 1;
      } else if ((c & 0xE0) == 0xC0) {
        // Two bytes
        if (i + 1 < token.size()) {
          codepoint = ((c & 0x1F) << 6) | (token[i + 1] & 0x3F);
          bytes_consumed = 2;
        } else {
          result += c;
          i++;
          continue;
        }
      } else if ((c & 0xF0) == 0xE0) {
        // Three bytes
        if (i + 2 < token.size()) {
          codepoint = ((c & 0x0F) << 12) | 
                      ((token[i + 1] & 0x3F) << 6) | 
                      (token[i + 2] & 0x3F);
          bytes_consumed = 3;
        } else {
          result += c;
          i++;
          continue;
        }
      } else if ((c & 0xF8) == 0xF0) {
        // Four bytes
        if (i + 3 < token.size()) {
          codepoint = ((c & 0x07) << 18) | 
                      ((token[i + 1] & 0x3F) << 12) |
                      ((token[i + 2] & 0x3F) << 6) | 
                      (token[i + 3] & 0x3F);
          bytes_consumed = 4;
        } else {
          result += c;
          i++;
          continue;
        }
      } else {
        // Invalid UTF-8, pass through
        result += c;
        i++;
        continue;
      }

      // Map Unicode codepoint back to byte
      auto it = unicode_to_byte_.find(codepoint);
      if (it != unicode_to_byte_.end()) {
        result += static_cast<char>(it->second);
      } else {
        // Not a BPE-encoded byte, encode the codepoint back to UTF-8
        if (codepoint < 0x80) {
          result += static_cast<char>(codepoint);
        } else if (codepoint < 0x800) {
          result += static_cast<char>(0xC0 | (codepoint >> 6));
          result += static_cast<char>(0x80 | (codepoint & 0x3F));
        } else if (codepoint < 0x10000) {
          result += static_cast<char>(0xE0 | (codepoint >> 12));
          result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
          result += static_cast<char>(0x80 | (codepoint & 0x3F));
        } else {
          result += static_cast<char>(0xF0 | (codepoint >> 18));
          result += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
          result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
          result += static_cast<char>(0x80 | (codepoint & 0x3F));
        }
      }
      
      i += bytes_consumed;
    }
    return result;
  }

 private:
  std::unordered_map<uint32_t, uint8_t> unicode_to_byte_;

  void InitializeMapping() {
    // GPT-2's bytes_to_unicode mapping (reversed)
    // Printable ASCII and extended Latin characters map to themselves
    // Other bytes (0-32, 127-160, 173) map to Unicode starting at U+0100
    
    std::vector<int> bs;
    // ! to ~ (33-126)
    for (int i = 33; i <= 126; ++i) bs.push_back(i);
    // ¡ to ¬ (161-172)
    for (int i = 161; i <= 172; ++i) bs.push_back(i);
    // ® to ÿ (174-255)
    for (int i = 174; i <= 255; ++i) bs.push_back(i);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
        bs.push_back(b);
        cs.push_back(256 + n);
        n++;
      }
    }

    // Build the reverse mapping: unicode -> byte
    for (size_t i = 0; i < bs.size(); ++i) {
      unicode_to_byte_[static_cast<uint32_t>(cs[i])] = static_cast<uint8_t>(bs[i]);
    }
  }
};

// Global BPE decoder instance
static BPEByteDecoder& GetBPEDecoder() {
  static BPEByteDecoder decoder;
  return decoder;
}

// ─────────────────────────────────────────────────────────────────────────────
// GPT2Tokenizer Implementation
// ─────────────────────────────────────────────────────────────────────────────

GPT2Tokenizer::GPT2Tokenizer() {
  InitializeBasicVocab();
}

void GPT2Tokenizer::InitializeBasicVocab() {
  // Initialize with a basic vocabulary
  // In production, load the full GPT-2 BPE vocabulary
  
  // Special tokens
  token_to_id_["<|endoftext|>"] = 50256;
  id_to_token_[50256] = "<|endoftext|>";
  
  // Common words and characters
  const std::vector<std::pair<std::string, int>> basic_vocab = {
    // Punctuation and special
    {" ", 220}, {".", 13}, {",", 11}, {"!", 0}, {"?", 30},
    {"'", 6}, {"\"", 1}, {":", 25}, {";", 26}, {"-", 12},
    {"(", 7}, {")", 8}, {"[", 58}, {"]", 60}, {"\n", 198},
    
    // Common words
    {"the", 262}, {"The", 464}, {"a", 64}, {"A", 32}, {"an", 281},
    {"is", 318}, {"are", 389}, {"was", 373}, {"were", 547},
    {"be", 307}, {"been", 587}, {"being", 852},
    {"have", 423}, {"has", 468}, {"had", 550},
    {"do", 466}, {"does", 857}, {"did", 750},
    {"will", 481}, {"would", 561}, {"could", 714}, {"should", 815},
    {"can", 460}, {"may", 743}, {"might", 1244}, {"must", 1276},
    {"I", 40}, {"you", 345}, {"he", 339}, {"she", 607}, {"it", 340},
    {"we", 356}, {"they", 484}, {"me", 502}, {"him", 683}, {"her", 607},
    {"this", 428}, {"that", 326}, {"these", 777}, {"those", 883},
    {"what", 644}, {"which", 543}, {"who", 508}, {"whom", 4150},
    {"where", 810}, {"when", 618}, {"why", 1521}, {"how", 703},
    {"all", 477}, {"each", 1123}, {"every", 790}, {"both", 1111},
    {"few", 1178}, {"more", 517}, {"most", 749}, {"some", 617},
    {"any", 597}, {"no", 645}, {"not", 407}, {"only", 691},
    {"own", 898}, {"same", 976}, {"so", 523}, {"than", 621},
    {"too", 1165}, {"very", 845}, {"just", 655}, {"also", 635},
    {"and", 290}, {"or", 393}, {"but", 475}, {"if", 611},
    {"then", 788}, {"because", 780}, {"as", 355}, {"until", 1566},
    {"while", 981}, {"of", 286}, {"at", 379}, {"by", 416},
    {"for", 329}, {"with", 351}, {"about", 546}, {"against", 1028},
    {"between", 1022}, {"into", 656}, {"through", 832}, {"during", 1141},
    {"before", 878}, {"after", 706}, {"above", 2029}, {"below", 2174},
    {"to", 284}, {"from", 422}, {"up", 510}, {"down", 866},
    {"in", 287}, {"out", 503}, {"on", 319}, {"off", 572},
    {"over", 625}, {"under", 739}, {"again", 757}, {"further", 2252},
    {"here", 994}, {"there", 612}, {"now", 783}, {"always", 1464},
    {"never", 1239}, {"today", 1909}, {"tomorrow", 9439},
    {"Hello", 15496}, {"hello", 31373}, {"Hi", 17250}, {"hi", 5765},
    {"Yes", 5765}, {"yes", 8505}, {"No", 2949}, {"no", 645},
    {"please", 4222}, {"Please", 5765}, {"thank", 5765}, {"Thank", 9576},
    {"thanks", 5765}, {"Thanks", 9576}, {"sorry", 6038}, {"Sorry", 14207},
    {"help", 1037}, {"Help", 22087}, {"need", 761}, {"want", 765},
    {"like", 588}, {"know", 760}, {"think", 892}, {"see", 766},
    {"come", 1282}, {"go", 467}, {"get", 651}, {"make", 787},
    {"take", 1011}, {"give", 1577}, {"find", 1064}, {"tell", 1560},
    {"ask", 1265}, {"use", 779}, {"work", 670}, {"call", 1414},
    {"try", 1949}, {"leave", 2666}, {"put", 1234}, {"keep", 1394},
    {"let", 1309}, {"begin", 2221}, {"seem", 1283}, {"show", 905},
    {"hear", 3285}, {"play", 711}, {"run", 1057}, {"move", 1445},
    {"live", 2107}, {"believe", 1975}, {"hold", 1745}, {"bring", 2222},
    {"happen", 1645}, {"write", 3551}, {"provide", 2148}, {"sit", 1650},
    {"stand", 1302}, {"lose", 4425}, {"pay", 1414}, {"meet", 1826},
    {"include", 2291}, {"continue", 2555}, {"set", 900}, {"learn", 2193},
    {"change", 1487}, {"lead", 1085}, {"understand", 1833},
    {"watch", 2342}, {"follow", 1061}, {"stop", 2245}, {"create", 2251},
    {"speak", 2740}, {"read", 1100}, {"allow", 1249}, {"add", 751},
    {"spend", 4341}, {"grow", 1663}, {"open", 1280}, {"walk", 2513},
    {"win", 1592}, {"offer", 2897}, {"remember", 3505}, {"love", 1842},
    {"consider", 2074}, {"appear", 1656}, {"buy", 2822}, {"wait", 4043},
    {"serve", 4691}, {"die", 4656}, {"send", 3758}, {"expect", 1607},
    {"build", 1382}, {"stay", 2652}, {"fall", 2121}, {"cut", 2005},
    {"reach", 3151}, {"kill", 1494}, {"remain", 3520},
  };
  
  for (const auto& [token, id] : basic_vocab) {
    token_to_id_[token] = id;
    id_to_token_[id] = token;
  }
  
  // Add digits
  for (int i = 0; i <= 9; ++i) {
    std::string digit = std::to_string(i);
    int id = 15 + i;  // Approximate GPT-2 IDs for digits
    token_to_id_[digit] = id;
    id_to_token_[id] = digit;
  }
  
  // Add lowercase letters as individual tokens
  for (char c = 'a'; c <= 'z'; ++c) {
    std::string letter(1, c);
    int id = 64 + (c - 'a');  // Approximate
    token_to_id_[letter] = id;
    id_to_token_[id] = letter;
  }
  
  // Add uppercase letters
  for (char c = 'A'; c <= 'Z'; ++c) {
    std::string letter(1, c);
    int id = 32 + (c - 'A');  // Approximate
    token_to_id_[letter] = id;
    id_to_token_[id] = letter;
  }
}

bool GPT2Tokenizer::LoadVocabulary(const std::string& vocab_path) {
  std::ifstream f(vocab_path);
  if (!f) {
    std::cerr << "[GPT2Tokenizer] Failed to open vocabulary file: " << vocab_path << "\n";
    return false;
  }

  try {
    json vocab_json;
    f >> vocab_json;

    token_to_id_.clear();
    id_to_token_.clear();

    for (auto& [token, id] : vocab_json.items()) {
      int token_id = id.get<int>();
      token_to_id_[token] = token_id;
      id_to_token_[token_id] = token;
    }

    // Set special tokens if present
    if (token_to_id_.count("<|endoftext|>")) {
      eos_token_id_ = token_to_id_["<|endoftext|>"];
      pad_token_id_ = eos_token_id_;
      bos_token_id_ = eos_token_id_;
    }

    return true;
  } catch (const std::exception& e) {
    std::cerr << "[GPT2Tokenizer] Error parsing vocabulary: " << e.what() << "\n";
    return false;
  }
}

std::vector<std::string> GPT2Tokenizer::SplitIntoWords(const std::string& text) const {
  std::vector<std::string> words;
  std::string current_word;
  
  for (size_t i = 0; i < text.size(); ++i) {
    char c = text[i];
    
    if (std::isspace(c)) {
      if (!current_word.empty()) {
        words.push_back(current_word);
        current_word.clear();
      }
      // Add space as separate token
      words.push_back(" ");
    } else if (std::ispunct(c)) {
      if (!current_word.empty()) {
        words.push_back(current_word);
        current_word.clear();
      }
      words.push_back(std::string(1, c));
    } else {
      current_word += c;
    }
  }
  
  if (!current_word.empty()) {
    words.push_back(current_word);
  }
  
  return words;
}

std::vector<int32_t> GPT2Tokenizer::Encode(const std::string& text) const {
  std::vector<int32_t> token_ids;
  
  std::vector<std::string> words = SplitIntoWords(text);
  
  for (const auto& word : words) {
    auto it = token_to_id_.find(word);
    if (it != token_to_id_.end()) {
      token_ids.push_back(it->second);
    } else {
      // Try character-by-character for unknown words
      for (char c : word) {
        std::string char_str(1, c);
        auto char_it = token_to_id_.find(char_str);
        if (char_it != token_to_id_.end()) {
          token_ids.push_back(char_it->second);
        } else {
          token_ids.push_back(unk_token_id_);
        }
      }
    }
  }
  
  return token_ids;
}

std::string GPT2Tokenizer::Decode(const std::vector<int32_t>& token_ids) const {
  std::string result;
  BPEByteDecoder& decoder = GetBPEDecoder();
  
  for (int32_t id : token_ids) {
    if (id == eos_token_id_) {
      break;  // Stop at end of sequence
    }
    
    auto it = id_to_token_.find(id);
    if (it != id_to_token_.end()) {
      // Decode BPE token to regular UTF-8 text
      result += decoder.Decode(it->second);
    } else {
      result += "[UNK]";
    }
  }
  
  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Load Model Configuration
// ─────────────────────────────────────────────────────────────────────────────

ModelConfig LoadModelConfig(const std::string& config_path) {
  std::ifstream f(config_path);
  if (!f) {
    throw std::runtime_error("Cannot open config: " + config_path);
  }

  json j;
  f >> j;

  ModelConfig config;
  config.model_file     = j.value("model_file", "gpt2_64-8bits.tflite");
  config.vocab_file     = j.value("vocab_file", "");
  config.max_seq_length = j.value("max_seq_length", 64);
  config.vocab_size     = j.value("vocab_size", 50257);

  // Generation parameters
  if (j.contains("generation")) {
    const auto& gen = j["generation"];
    config.generation.max_new_tokens    = gen.value("max_new_tokens", 50);
    config.generation.temperature       = gen.value("temperature", 0.7f);
    config.generation.top_k             = gen.value("top_k", 40);
    config.generation.top_p             = gen.value("top_p", 0.9f);
    config.generation.do_sample         = gen.value("do_sample", true);
    config.generation.repetition_penalty = gen.value("repetition_penalty", 1.1f);
    config.generation.eos_token_id      = gen.value("eos_token_id", 50256);
    config.generation.pad_token_id      = gen.value("pad_token_id", 50256);
  }

  return config;
}

bool CheckDelegateAvailable(const std::string& delegate_path) {
  if (delegate_path.empty()) return false;
  std::ifstream f(delegate_path);
  return f.good();
}

// ─────────────────────────────────────────────────────────────────────────────
// GPT2InferenceEngine Implementation
// ─────────────────────────────────────────────────────────────────────────────

GPT2InferenceEngine::GPT2InferenceEngine(const std::string& config_path,
                                         const std::string& model_path,
                                         const std::string& delegate_path,
                                         const std::string& delegate_options,
                                         int num_threads,
                                         bool verbose)
    : verbose_(verbose),
      num_threads_(num_threads),
      delegate_path_(delegate_path),
      delegate_options_(delegate_options),
      external_delegate_(nullptr, nullptr) {
  
  if (!LoadConfig(config_path)) {
    std::cerr << "[GPT2] Failed to load configuration from: " << config_path << "\n";
    return;
  }

  // Override model path from command line if provided
  if (!model_path.empty()) {
    config_.model_file = model_path;
    if (verbose_) {
      std::cerr << "[GPT2] Model path overridden to: " << model_path << "\n";
    }
  }

  if (!LoadInterpreter()) {
    std::cerr << "[GPT2] Failed to initialize TFLite interpreter\n";
    return;
  }

  // Load vocabulary if specified
  if (!config_.vocab_file.empty()) {
    if (!tokenizer_.LoadVocabulary(config_.vocab_file)) {
      std::cerr << "[GPT2] Warning: Failed to load vocabulary, using basic vocab\n";
    }
  }

  is_initialized_ = true;
  if (verbose_) {
    std::cerr << "[GPT2] Engine initialized successfully\n";
    std::cerr << "[GPT2] Model: " << config_.model_file << "\n";
    std::cerr << "[GPT2] Max sequence length: " << config_.max_seq_length << "\n";
    std::cerr << "[GPT2] Vocabulary size: " << tokenizer_.VocabSize() << "\n";
  }
}

bool GPT2InferenceEngine::LoadConfig(const std::string& config_path) {
  try {
    config_ = LoadModelConfig(config_path);
    return true;
  } catch (const std::exception& e) {
    last_error_ = std::string("Config error: ") + e.what();
    return false;
  }
}

bool GPT2InferenceEngine::LoadInterpreter() {
  // Load the TFLite model
  model_ = tflite::FlatBufferModel::BuildFromFile(config_.model_file.c_str());
  if (!model_) {
    last_error_ = "Failed to load TFLite model: " + config_.model_file;
    return false;
  }

  // Build interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model_, resolver);
  builder(&interpreter_);
  
  if (!interpreter_) {
    last_error_ = "Failed to build TFLite interpreter";
    return false;
  }

  // Configure threads
  interpreter_->SetNumThreads(num_threads_);

  // Apply external delegate if specified
  if (!delegate_path_.empty() && CheckDelegateAvailable(delegate_path_)) {
    if (verbose_) {
      std::cerr << "[GPT2] Loading external delegate: " << delegate_path_ << "\n";
    }

    TfLiteExternalDelegateOptions opts =
        TfLiteExternalDelegateOptionsDefault(delegate_path_.c_str());

    // Parse delegate options
    std::vector<std::string> opt_keys, opt_vals;
    if (!delegate_options_.empty()) {
      std::stringstream ss(delegate_options_);
      std::string pair;
      while (std::getline(ss, pair, ';')) {
        size_t colon = pair.find(':');
        if (colon != std::string::npos) {
          opt_keys.push_back(pair.substr(0, colon));
          opt_vals.push_back(pair.substr(colon + 1));
        }
      }
      
      for (size_t i = 0; i < opt_keys.size(); ++i) {
        TfLiteExternalDelegateOptionsInsert(&opts, 
                                            opt_keys[i].c_str(), 
                                            opt_vals[i].c_str());
      }
    }

    external_delegate_ = TfLiteDelegateUniquePtr(
        TfLiteExternalDelegateCreate(&opts),
        TfLiteExternalDelegateDelete);
    
    if (external_delegate_) {
      if (interpreter_->ModifyGraphWithDelegate(external_delegate_.get()) != kTfLiteOk) {
        std::cerr << "[GPT2] Warning: Failed to apply external delegate\n";
      } else if (verbose_) {
        std::cerr << "[GPT2] External delegate applied successfully\n";
      }
    }
  }

  // Allocate tensors
  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    last_error_ = "Failed to allocate tensors";
    return false;
  }

  // Get input/output tensor indices
  const auto& inputs = interpreter_->inputs();
  const auto& outputs = interpreter_->outputs();
  
  if (inputs.empty() || outputs.empty()) {
    last_error_ = "Model has no inputs or outputs";
    return false;
  }

  input_tensor_idx_ = inputs[0];
  output_tensor_idx_ = outputs[0];

  if (verbose_) {
    TfLiteTensor* input = interpreter_->tensor(input_tensor_idx_);
    TfLiteTensor* output = interpreter_->tensor(output_tensor_idx_);
    
    std::cerr << "[GPT2] Input tensor: " << input->name 
              << " [" << input->dims->data[0];
    for (int i = 1; i < input->dims->size; ++i) {
      std::cerr << ", " << input->dims->data[i];
    }
    std::cerr << "]\n";
    
    std::cerr << "[GPT2] Output tensor: " << output->name
              << " [" << output->dims->data[0];
    for (int i = 1; i < output->dims->size; ++i) {
      std::cerr << ", " << output->dims->data[i];
    }
    std::cerr << "]\n";
  }

  return true;
}

bool GPT2InferenceEngine::RunInference(const std::vector<int32_t>& input_ids,
                                        std::vector<float>& logits) {
  TfLiteTensor* input_tensor = interpreter_->tensor(input_tensor_idx_);
  TfLiteTensor* output_tensor = interpreter_->tensor(output_tensor_idx_);

  // Determine input size
  int input_size = 1;
  for (int i = 0; i < input_tensor->dims->size; ++i) {
    input_size *= input_tensor->dims->data[i];
  }

  // Fill input tensor with token IDs
  // Pad or truncate to match expected input size
  std::vector<int32_t> padded_input(input_size, config_.generation.pad_token_id);
  int copy_len = std::min(static_cast<int>(input_ids.size()), input_size);
  
  // Copy input tokens (right-aligned for causal LM)
  int start_pos = std::max(0, input_size - static_cast<int>(input_ids.size()));
  for (int i = 0; i < copy_len; ++i) {
    padded_input[start_pos + i] = input_ids[i];
  }

  // Copy to input tensor based on type
  if (input_tensor->type == kTfLiteInt32) {
    std::memcpy(input_tensor->data.i32, padded_input.data(),
                input_size * sizeof(int32_t));
  } else if (input_tensor->type == kTfLiteInt64) {
    std::vector<int64_t> input_i64(padded_input.begin(), padded_input.end());
    std::memcpy(input_tensor->data.i64, input_i64.data(),
                input_size * sizeof(int64_t));
  } else if (input_tensor->type == kTfLiteFloat32) {
    std::vector<float> input_f32(padded_input.begin(), padded_input.end());
    std::memcpy(input_tensor->data.f, input_f32.data(),
                input_size * sizeof(float));
  } else {
    last_error_ = "Unsupported input tensor type";
    return false;
  }

  // Run inference
  if (interpreter_->Invoke() != kTfLiteOk) {
    last_error_ = "TFLite Invoke() failed";
    return false;
  }

  // Extract output logits
  int output_size = 1;
  for (int i = 0; i < output_tensor->dims->size; ++i) {
    output_size *= output_tensor->dims->data[i];
  }

  logits.resize(output_size);
  
  if (output_tensor->type == kTfLiteFloat32) {
    std::memcpy(logits.data(), output_tensor->data.f,
                output_size * sizeof(float));
  } else if (output_tensor->type == kTfLiteFloat16) {
    // Convert float16 to float32
    // TFLite stores float16 as uint16_t, need to convert to float32
    const uint16_t* fp16_data = reinterpret_cast<const uint16_t*>(output_tensor->data.data);
    for (int i = 0; i < output_size; ++i) {
      logits[i] = Fp16ToFloat32(fp16_data[i]);
    }
  } else if (output_tensor->type == kTfLiteInt8) {
    // Dequantize int8 output
    const TfLiteQuantizationParams& quant = output_tensor->params;
    for (int i = 0; i < output_size; ++i) {
      logits[i] = (output_tensor->data.int8[i] - quant.zero_point) * quant.scale;
    }
  } else if (output_tensor->type == kTfLiteUInt8) {
    const TfLiteQuantizationParams& quant = output_tensor->params;
    for (int i = 0; i < output_size; ++i) {
      logits[i] = (output_tensor->data.uint8[i] - quant.zero_point) * quant.scale;
    }
  } else {
    last_error_ = "Unsupported output tensor type";
    return false;
  }

  return true;
}

void GPT2InferenceEngine::ApplyTemperature(std::vector<float>& logits, 
                                            float temperature) {
  if (temperature <= 0.0f) temperature = 1.0f;
  for (auto& logit : logits) {
    logit /= temperature;
  }
}

void GPT2InferenceEngine::ApplyTopK(std::vector<float>& logits, int k) {
  if (k <= 0 || k >= static_cast<int>(logits.size())) return;

  // Find k-th largest value
  std::vector<float> sorted_logits = logits;
  std::partial_sort(sorted_logits.begin(), 
                    sorted_logits.begin() + k,
                    sorted_logits.end(),
                    std::greater<float>());
  float threshold = sorted_logits[k - 1];

  // Zero out values below threshold
  for (auto& logit : logits) {
    if (logit < threshold) {
      logit = -std::numeric_limits<float>::infinity();
    }
  }
}

void GPT2InferenceEngine::ApplyTopP(std::vector<float>& logits, float p) {
  if (p >= 1.0f) return;

  // Apply softmax first
  std::vector<float> probs = logits;
  Softmax(probs);

  // Sort by probability
  std::vector<std::pair<float, int>> sorted_probs;
  for (size_t i = 0; i < probs.size(); ++i) {
    sorted_probs.emplace_back(probs[i], i);
  }
  std::sort(sorted_probs.begin(), sorted_probs.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

  // Find cutoff
  float cumsum = 0.0f;
  std::vector<bool> keep(logits.size(), false);
  for (const auto& [prob, idx] : sorted_probs) {
    cumsum += prob;
    keep[idx] = true;
    if (cumsum >= p) break;
  }

  // Zero out non-kept values
  for (size_t i = 0; i < logits.size(); ++i) {
    if (!keep[i]) {
      logits[i] = -std::numeric_limits<float>::infinity();
    }
  }
}

void GPT2InferenceEngine::ApplyRepetitionPenalty(
    std::vector<float>& logits,
    const std::vector<int32_t>& generated_tokens,
    float penalty, int window) {
  
  if (penalty == 1.0f) return;

  // Apply penalty to recently generated tokens
  int start = std::max(0, static_cast<int>(generated_tokens.size()) - window);
  for (int i = start; i < static_cast<int>(generated_tokens.size()); ++i) {
    int token_id = generated_tokens[i];
    if (token_id >= 0 && token_id < static_cast<int>(logits.size())) {
      if (logits[token_id] > 0) {
        logits[token_id] /= penalty;
      } else {
        logits[token_id] *= penalty;
      }
    }
  }
}

void GPT2InferenceEngine::Softmax(std::vector<float>& logits) {
  float max_logit = *std::max_element(logits.begin(), logits.end());
  float sum = 0.0f;
  
  for (auto& logit : logits) {
    logit = std::exp(logit - max_logit);
    sum += logit;
  }
  
  for (auto& logit : logits) {
    logit /= sum;
  }
}

int32_t GPT2InferenceEngine::SampleFromDistribution(const std::vector<float>& probs) {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float r = dist(GetRNG());
  
  float cumsum = 0.0f;
  for (size_t i = 0; i < probs.size(); ++i) {
    cumsum += probs[i];
    if (r <= cumsum) {
      return static_cast<int32_t>(i);
    }
  }
  
  return static_cast<int32_t>(probs.size() - 1);
}

int32_t GPT2InferenceEngine::SampleToken(const std::vector<float>& logits,
                                          const GenerationConfig& config,
                                          const std::vector<int32_t>& generated_tokens) {
  // Copy logits for processing
  std::vector<float> processed_logits = logits;
  
  // Get logits for the last position (next token prediction)
  // For GPT-2, output shape is typically [batch, seq_len, vocab_size]
  int vocab_size = config_.vocab_size;
  int seq_len = static_cast<int>(processed_logits.size()) / vocab_size;
  
  // Extract logits for last position
  std::vector<float> last_logits(vocab_size);
  int last_pos = seq_len - 1;
  std::copy(processed_logits.begin() + last_pos * vocab_size,
            processed_logits.begin() + (last_pos + 1) * vocab_size,
            last_logits.begin());

  // Apply repetition penalty
  ApplyRepetitionPenalty(last_logits, generated_tokens,
                         config.repetition_penalty,
                         config.repetition_penalty_window);

  if (!config.do_sample) {
    // Greedy decoding
    return static_cast<int32_t>(
        std::max_element(last_logits.begin(), last_logits.end()) - 
        last_logits.begin());
  }

  // Apply temperature
  ApplyTemperature(last_logits, config.temperature);

  // Apply top-k
  if (config.top_k > 0) {
    ApplyTopK(last_logits, config.top_k);
  }

  // Apply top-p
  if (config.top_p < 1.0f) {
    ApplyTopP(last_logits, config.top_p);
  }

  // Convert to probabilities
  Softmax(last_logits);

  // Sample
  return SampleFromDistribution(last_logits);
}

InferenceResult GPT2InferenceEngine::Generate(const std::string& prompt) {
  return Generate(prompt, config_.generation);
}

InferenceResult GPT2InferenceEngine::Generate(const std::string& prompt,
                                               const GenerationConfig& config) {
  InferenceResult result;
  result.success = false;

  if (!is_initialized_) {
    result.error_message = "Engine not initialized: " + last_error_;
    return result;
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  // Encode prompt
  std::vector<int32_t> input_ids = tokenizer_.Encode(prompt);
  
  if (verbose_) {
    std::cerr << "[GPT2] Input prompt: \"" << prompt << "\"\n";
    std::cerr << "[GPT2] Input tokens: " << input_ids.size() << "\n";
  }

  // Check sequence length
  if (static_cast<int>(input_ids.size()) >= config_.max_seq_length) {
    // Truncate from the left to keep recent context
    int excess = static_cast<int>(input_ids.size()) - config_.max_seq_length + 1;
    input_ids.erase(input_ids.begin(), input_ids.begin() + excess);
  }

  // Generate tokens
  std::vector<int32_t> generated_tokens;
  std::vector<int32_t> all_tokens = input_ids;

  for (int i = 0; i < config.max_new_tokens; ++i) {
    // Run inference
    std::vector<float> logits;
    if (!RunInference(all_tokens, logits)) {
      result.error_message = last_error_;
      return result;
    }

    // Sample next token
    int32_t next_token = SampleToken(logits, config, all_tokens);

    // Check for EOS
    if (config.eos_token_id >= 0 && next_token == config.eos_token_id) {
      if (verbose_) {
        std::cerr << "[GPT2] EOS token generated after " << i + 1 << " tokens\n";
      }
      break;
    }

    // Add token
    generated_tokens.push_back(next_token);
    all_tokens.push_back(next_token);

    // Truncate if exceeding max length
    if (static_cast<int>(all_tokens.size()) > config_.max_seq_length) {
      all_tokens.erase(all_tokens.begin());
    }

    if (verbose_ && (i + 1) % 10 == 0) {
      std::cerr << "[GPT2] Generated " << i + 1 << " tokens...\n";
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  // Decode generated tokens
  result.generated_text = tokenizer_.Decode(generated_tokens);
  result.output_tokens = generated_tokens;
  result.inference_time_ms = duration.count() / 1000.0f;
  result.tokens_generated = static_cast<int>(generated_tokens.size());
  result.success = true;

  // Update statistics
  stats_.total_inferences++;
  stats_.total_time_ms += result.inference_time_ms;
  stats_.total_tokens += result.tokens_generated;
  stats_.avg_time_ms = stats_.total_time_ms / stats_.total_inferences;
  stats_.tokens_per_sec = (stats_.total_tokens * 1000.0f) / stats_.total_time_ms;

  if (verbose_) {
    std::cerr << "[GPT2] Generated " << result.tokens_generated << " tokens in "
              << result.inference_time_ms << " ms\n";
    std::cerr << "[GPT2] Tokens/sec: " << stats_.tokens_per_sec << "\n";
  }

  return result;
}

}  // namespace gpt2
}  // namespace tflite
