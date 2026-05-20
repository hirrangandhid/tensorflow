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

gpt2_app_main.cc
────────────────────────────────────────────────────────────────────────────
Command-line interface for GPT-2 text generation on CPE/RDKB devices.

Modes
-----
  Interactive (default) — no --prompt given
    Starts an interactive chat session. Type prompts and get responses.
    Type 'quit' or 'exit' to end the session.

  Single-shot            — --prompt <text> given
    Generates a response to the given prompt and exits.

Usage:
  gpt2_app [options]

Options:
  --config,         -c <path>   Model config JSON
                               (default: gpt2_config.json)
  --model,          -M <path>   Path to TFLite model file (overrides config)
                               (e.g. /nvram/gpt2.tflite or /tmp/gpt2.tflite)
  --prompt,         -p <text>   Single prompt (non-interactive mode)
  --max-tokens,     -m <n>      Maximum tokens to generate (default: 50)
  --temperature,    -T <float>  Sampling temperature (default: 0.7)
  --top-k           <n>         Top-k sampling (default: 40, 0=disabled)
  --top-p           <float>     Top-p nucleus sampling (default: 0.9)
  --greedy                      Use greedy decoding (no sampling)
  --threads,        -t <n>      TFLite thread count (default: 1)
  --delegate-path,  -d <path>   External hardware delegate .so
                               (e.g. /usr/lib/libbstorm_external_delegate.so)
  --delegate-options   <str>    Semicolon-separated key:value delegate options
  --verbose,        -v          Print detailed debug info
  --stats                       Show inference statistics on exit
  --help,           -h          Print this message

Examples:
  # Interactive mode
  ./gpt2_app --config gpt2_config.json

  # Single prompt
  ./gpt2_app -c gpt2_config.json -p "What is machine learning?"

  # With NPU acceleration (BStorm delegate)
  ./gpt2_app -c gpt2_config.json \
             -d /usr/lib/libbstorm_external_delegate.so \
             --delegate-options "bstm:1;dynamic-tensors:1"
============================================================================*/

#include "gpt2_inference.h"

#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Global flag for signal handling
volatile sig_atomic_t g_stop = 0;

void SignalHandler(int) { 
  g_stop = 1;
  std::cout << "\n[Interrupted]\n";
}

void PrintUsage(const char* prog) {
  std::cout << R"(
GPT-2 Text Generation for CPE/RDKB Devices
───────────────────────────────────────────────────────────────────────────

Usage: )" << prog << R"( [options]

Options:
  --config,         -c <path>   Model config JSON (default: gpt2_config.json)
  --model,          -M <path>   TFLite model path (overrides config)
  --prompt,         -p <text>   Single prompt (non-interactive mode)
  --max-tokens,     -m <n>      Maximum tokens to generate (default: 50)
  --temperature,    -T <float>  Sampling temperature (default: 0.7)
  --top-k           <n>         Top-k sampling (default: 40, 0=disabled)
  --top-p           <float>     Top-p nucleus sampling (default: 0.9)
  --greedy                      Use greedy decoding (no sampling)
  --threads,        -t <n>      TFLite thread count (default: 1)
  --delegate-path,  -d <path>   External delegate .so (e.g., BStorm NPU)
  --delegate-options   <str>    Semicolon-separated key:value options
  --verbose,        -v          Print detailed debug info
  --stats                       Show inference statistics on exit
  --help,           -h          Print this message

Examples:
  Interactive mode:
    ./gpt2_app --config gpt2_config.json

  Single prompt:
    ./gpt2_app -c gpt2_config.json -p "What is machine learning?"

  With NPU acceleration:
    ./gpt2_app -c gpt2_config.json \
               -d /usr/lib/libbstorm_external_delegate.so \
               --delegate-options "bstm:1;dynamic-tensors:1"

)";
}

void PrintBanner() {
  std::cout << R"(
╔═══════════════════════════════════════════════════════════════════════════╗
║                    GPT-2 Text Generation Engine                            ║
║                     For CPE/RDKB Edge Devices                              ║
╚═══════════════════════════════════════════════════════════════════════════╝
)";
}

void PrintStats(const tflite::gpt2::InferenceStats& stats) {
  std::cout << "\n";
  std::cout << "════════════════════════════════════════════════════════════════\n";
  std::cout << "                     INFERENCE STATISTICS                        \n";
  std::cout << "════════════════════════════════════════════════════════════════\n";
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "  Total inferences:     " << stats.total_inferences << "\n";
  std::cout << "  Total tokens:         " << stats.total_tokens << "\n";
  std::cout << "  Total time:           " << stats.total_time_ms << " ms\n";
  std::cout << "  Average time:         " << stats.avg_time_ms << " ms/inference\n";
  std::cout << "  Throughput:           " << stats.tokens_per_sec << " tokens/sec\n";
  std::cout << "════════════════════════════════════════════════════════════════\n";
}

std::string Trim(const std::string& str) {
  size_t first = str.find_first_not_of(" \t\n\r");
  if (first == std::string::npos) return "";
  size_t last = str.find_last_not_of(" \t\n\r");
  return str.substr(first, (last - first + 1));
}

}  // namespace

int main(int argc, char* argv[]) {
  // Default parameters
  std::string config_path = "gpt2_config.json";
  std::string model_path;  // Model file path (overrides config)
  std::string delegate_path;
  std::string delegate_options;
  std::string single_prompt;
  int num_threads = 1;
  bool verbose = false;
  bool show_stats = false;
  bool greedy = false;
  
  // Generation parameters (can override config)
  int max_tokens = -1;      // -1 = use config default
  float temperature = -1.0f;
  int top_k = -1;
  float top_p = -1.0f;

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    } else if (arg == "--config" || arg == "-c") {
      if (++i < argc) config_path = argv[i];
    } else if (arg == "--model" || arg == "-M") {
      if (++i < argc) model_path = argv[i];
    } else if (arg == "--prompt" || arg == "-p") {
      if (++i < argc) single_prompt = argv[i];
    } else if (arg == "--max-tokens" || arg == "-m") {
      if (++i < argc) max_tokens = std::stoi(argv[i]);
    } else if (arg == "--temperature" || arg == "-T") {
      if (++i < argc) temperature = std::stof(argv[i]);
    } else if (arg == "--top-k") {
      if (++i < argc) top_k = std::stoi(argv[i]);
    } else if (arg == "--top-p") {
      if (++i < argc) top_p = std::stof(argv[i]);
    } else if (arg == "--greedy") {
      greedy = true;
    } else if (arg == "--threads" || arg == "-t") {
      if (++i < argc) num_threads = std::stoi(argv[i]);
    } else if (arg == "--delegate-path" || arg == "-d") {
      if (++i < argc) delegate_path = argv[i];
    } else if (arg == "--delegate-options") {
      if (++i < argc) delegate_options = argv[i];
    } else if (arg == "--verbose" || arg == "-v") {
      verbose = true;
    } else if (arg == "--stats") {
      show_stats = true;
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      std::cerr << "Use --help for usage information.\n";
      return 1;
    }
  }

  // Set up signal handlers
  signal(SIGINT, SignalHandler);
  signal(SIGTERM, SignalHandler);

  // Check config file exists
  std::ifstream config_check(config_path);
  if (!config_check) {
    std::cerr << "Error: Config file not found: " << config_path << "\n";
    std::cerr << "Create a gpt2_config.json or specify path with --config\n";
    return 1;
  }
  config_check.close();

  // Check model file exists if specified
  if (!model_path.empty()) {
    std::ifstream model_check(model_path);
    if (!model_check) {
      std::cerr << "Error: Model file not found: " << model_path << "\n";
      return 1;
    }
    model_check.close();
  }

  // Initialize engine
  if (verbose) {
    std::cerr << "[main] Loading GPT-2 engine...\n";
    std::cerr << "[main] Config: " << config_path << "\n";
    if (!model_path.empty()) {
      std::cerr << "[main] Model: " << model_path << "\n";
    }
    if (!delegate_path.empty()) {
      std::cerr << "[main] Delegate: " << delegate_path << "\n";
    }
  }

  tflite::gpt2::GPT2InferenceEngine engine(
      config_path,
      model_path,
      delegate_path,
      delegate_options,
      num_threads,
      verbose);

  if (!engine.IsReady()) {
    std::cerr << "Error: Failed to initialize GPT-2 engine\n";
    std::cerr << "       " << engine.GetLastError() << "\n";
    return 1;
  }

  // Build generation config with any command-line overrides
  tflite::gpt2::GenerationConfig gen_config = engine.GetConfig().generation;
  
  if (max_tokens > 0) gen_config.max_new_tokens = max_tokens;
  if (temperature >= 0) gen_config.temperature = temperature;
  if (top_k >= 0) gen_config.top_k = top_k;
  if (top_p >= 0) gen_config.top_p = top_p;
  if (greedy) gen_config.do_sample = false;

  // Single prompt mode
  if (!single_prompt.empty()) {
    if (verbose) {
      std::cerr << "[main] Processing single prompt...\n";
    }

    auto result = engine.Generate(single_prompt, gen_config);

    if (result.success) {
      std::cout << result.generated_text << "\n";
      
      if (verbose) {
        std::cerr << "\n[main] Tokens generated: " << result.tokens_generated << "\n";
        std::cerr << "[main] Time: " << result.inference_time_ms << " ms\n";
      }
    } else {
      std::cerr << "Error: " << result.error_message << "\n";
      return 1;
    }

    if (show_stats) {
      PrintStats(engine.GetStats());
    }

    return 0;
  }

  // Interactive mode
  PrintBanner();
  
  std::cout << "Model loaded successfully!\n";
  std::cout << "Type your prompts below. Type 'quit' or 'exit' to end.\n";
  std::cout << "Type 'stats' to show inference statistics.\n";
  std::cout << "Type 'clear' to reset conversation context.\n";
  std::cout << "\n";

  std::string conversation_context;
  
  while (!g_stop) {
    std::cout << "You: ";
    std::cout.flush();
    
    std::string user_input;
    if (!std::getline(std::cin, user_input)) {
      break;  // EOF
    }

    user_input = Trim(user_input);
    
    if (user_input.empty()) {
      continue;
    }

    // Check for commands
    std::string lower_input = user_input;
    std::transform(lower_input.begin(), lower_input.end(), 
                   lower_input.begin(), ::tolower);

    if (lower_input == "quit" || lower_input == "exit") {
      std::cout << "Goodbye!\n";
      break;
    }

    if (lower_input == "stats") {
      PrintStats(engine.GetStats());
      continue;
    }

    if (lower_input == "clear") {
      conversation_context.clear();
      engine.ResetStats();
      std::cout << "[Context cleared]\n\n";
      continue;
    }

    if (lower_input == "help") {
      std::cout << "\nCommands:\n";
      std::cout << "  quit, exit   - End the session\n";
      std::cout << "  stats        - Show inference statistics\n";
      std::cout << "  clear        - Reset conversation context\n";
      std::cout << "  help         - Show this help\n\n";
      continue;
    }

    // Build prompt with optional conversation context
    std::string full_prompt = user_input;
    if (!conversation_context.empty()) {
      full_prompt = conversation_context + "\n" + user_input;
    }

    // Generate response
    std::cout << "\nGPT-2: ";
    std::cout.flush();

    auto result = engine.Generate(full_prompt, gen_config);

    if (result.success) {
      std::cout << result.generated_text << "\n";
      
      // Update conversation context (keep it bounded)
      conversation_context = full_prompt + " " + result.generated_text;
      
      // Trim context if too long (keep last ~500 chars)
      if (conversation_context.size() > 500) {
        size_t trim_pos = conversation_context.size() - 500;
        // Find next space to avoid breaking mid-word
        trim_pos = conversation_context.find(' ', trim_pos);
        if (trim_pos != std::string::npos) {
          conversation_context = conversation_context.substr(trim_pos + 1);
        }
      }

      if (verbose) {
        std::cerr << "\n[" << result.tokens_generated << " tokens, "
                  << std::fixed << std::setprecision(1) 
                  << result.inference_time_ms << " ms]\n";
      }
    } else {
      std::cout << "[Error: " << result.error_message << "]\n";
    }

    std::cout << "\n";
  }

  // Show final stats if requested
  if (show_stats) {
    PrintStats(engine.GetStats());
  }

  return 0;
}
