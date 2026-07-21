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

anomaly_prediction_main.cc
────────────────────────────────────────────────────────────────────────────
Command-line entry point for the RDKB gateway anomaly prediction engine.

Supports TCN autoencoder models with sliding window inference.
Designed for RDKB routers with bstorm delegate for NPU acceleration.

Modes
-----
  File-watcher (default)  — no --input given
    Monitors the device CSV for new rows and outputs anomaly predictions.
    Run in background on RDKB gateway.

  Batch                   — --input <path> given
    One-shot: reads complete CSV, writes results, then exits.
    Useful for testing and evaluation.

Usage:
  anomaly_prediction_app [options]

General options:
  --config,         -c <path>   Prediction config JSON
                               (default: anomaly_prediction_config.json)
  --threads,        -t <n>     TFLite thread count (default: 1)
  --delegate-path,  -d <path>  External hardware delegate .so
                               (e.g., /usr/lib/libbstorm_external_delegate.so)
  --delegate-options   <str>   Semicolon-separated key:value delegate options
                               (e.g., "bstm:1;bstm-client-mode:0")
  --verbose,        -v         Print per-reading details to stderr
  --help,           -h         Print this message

Batch options (used when --input is given):
  --input,  -i <path>    Input CSV file
  --output, -o <path>    Output CSV file (default: stdout)

File-watcher options:
  --watch-path  <path>   Device CSV to monitor
                         (default: /rdklogs/logs/system_stats_data.csv)
  --result-path <path>   Output file for results
                         (default: /rdklogs/logs/anomaly_predictions.csv)
  --poll-interval <ms>   Polling interval (default: 60000 ms)
  --skip-existing        Skip rows already in file on startup

Output CSV columns:
  timestamp, CMMAC, cpu_mse, mem_mse, cpu_anomaly, mem_anomaly,
  cpu_severity, mem_severity, anomaly_type, inference_ms
============================================================================*/

#include "tensorflow/lite/examples/anomaly_prediction/anomaly_prediction.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <signal.h>
#include <vector>

#ifdef __linux__
#include <limits.h>
#include <poll.h>
#include <sys/inotify.h>
#endif

namespace tflite {
namespace anomaly_prediction {

// ── Global state ─────────────────────────────────────────────────────────────
static volatile sig_atomic_t g_running = 1;

void SignalHandler(int sig) {
  (void)sig;
  g_running = 0;
}

// ── CSV parsing helpers ──────────────────────────────────────────────────────

static std::vector<std::string> SplitCsv(const std::string& line) {
  std::vector<std::string> tokens;
  std::stringstream ss(line);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    while (!tok.empty() && (tok.front() == ' ' || tok.front() == '"'))
      tok.erase(tok.begin());
    while (!tok.empty() && (tok.back() == ' ' || tok.back() == '"'))
      tok.pop_back();
    tokens.push_back(tok);
  }
  return tokens;
}

static float SafeFloat(const std::string& s, float def = 0.0f) {
  try { return std::stof(s); } catch (...) { return def; }
}

static int SafeInt(const std::string& s, int def = 0) {
  try { return std::stoi(s); } catch (...) { return def; }
}

// Parse timestamp → hour and day_of_week
static void ParseTimestamp(const std::string& ts, int& hour, int& dow) {
  hour = 0; dow = 0;
  if (ts.size() >= 13) {
    char sep = ts[10];
    if (sep == 'T' || sep == ' ' || sep == '-')
      hour = SafeInt(ts.substr(11, 2));
  }
  if (ts.size() >= 10) {
    int year = SafeInt(ts.substr(0, 4));
    int month = SafeInt(ts.substr(5, 2));
    int day = SafeInt(ts.substr(8, 2));
    // Zeller's formula for day of week
    if (month < 3) { month += 12; year--; }
    int k = year % 100;
    int j = year / 100;
    dow = (day + (13*(month+1))/5 + k + k/4 + j/4 - 2*j) % 7;
    dow = (dow + 6) % 7;  // Convert to Monday=0
  }
}

// ── CSV reading with column mapping ──────────────────────────────────────────

struct ColumnMap {
  int timestamp = -1;
  int mac = -1;
  int used_cpu = -1;
  int load_avg = -1;
  int total_mem = -1;
  int used_mem = -1;
  int avail_mem = -1;
  int free_mem = -1;
  int slab_mem = -1;
  int clients_2g = -1;
  int clients_5g = -1;
  int clients_6g = -1;
};

ColumnMap BuildColumnMap(const std::vector<std::string>& header) {
  ColumnMap m;
  for (size_t i = 0; i < header.size(); ++i) {
    const std::string& h = header[i];
    if (h == "timestamp" || h == "Timestamp") m.timestamp = i;
    else if (h == "CMMAC" || h == "mac") m.mac = i;
    else if (h == "USED_CPU_ATOM" || h == "cpu_percent") m.used_cpu = i;
    else if (h == "LOAD_AVG_ATOM" || h == "load_avg") m.load_avg = i;
    else if (h == "TOTAL_MEM_ATOM_kB" || h == "total_mem") m.total_mem = i;
    else if (h == "USED_MEM_ATOM_kB" || h == "used_mem") m.used_mem = i;
    else if (h == "AvailMem_kB" || h == "avail_mem") m.avail_mem = i;
    else if (h == "FreeMem_kB" || h == "free_mem") m.free_mem = i;
    else if (h == "SlabMem_kB" || h == "slab_mem") m.slab_mem = i;
    else if (h == "2G_Clients_Count" || h == "clients_2g") m.clients_2g = i;
    else if (h == "5G_Clients_Count" || h == "clients_5g") m.clients_5g = i;
    else if (h == "6G_Clients_Count" || h == "clients_6g") m.clients_6g = i;
  }
  return m;
}

TelemetryReading ParseRow(const std::vector<std::string>& row, 
                          const ColumnMap& m) {
  TelemetryReading r;
  
  auto get = [&row](int idx) -> std::string {
    return (idx >= 0 && idx < static_cast<int>(row.size())) ? row[idx] : "";
  };
  
  r.timestamp = get(m.timestamp);
  r.mac = get(m.mac);
  r.used_cpu = SafeFloat(get(m.used_cpu));
  r.load_avg = SafeFloat(get(m.load_avg));
  r.total_mem_kb = SafeFloat(get(m.total_mem));
  r.used_mem_kb = SafeFloat(get(m.used_mem));
  r.avail_mem_kb = SafeFloat(get(m.avail_mem));
  r.free_mem_kb = SafeFloat(get(m.free_mem));
  r.slab_mem_kb = SafeFloat(get(m.slab_mem));
  r.clients_2g = SafeInt(get(m.clients_2g));
  r.clients_5g = SafeInt(get(m.clients_5g));
  r.clients_6g = SafeInt(get(m.clients_6g));
  
  ParseTimestamp(r.timestamp, r.hour_of_day, r.day_of_week);
  
  return r;
}

// ── Output formatting ────────────────────────────────────────────────────────

void WriteResultHeader(std::ostream& out) {
  out << "timestamp,CMMAC,cpu_mse,mem_mse,cpu_anomaly,mem_anomaly,"
      << "cpu_severity,mem_severity,anomaly_type,inference_ms\n";
}

void WriteResult(std::ostream& out, const TelemetryReading& r, 
                 const PredictionResult& res) {
  out << r.timestamp << ","
      << r.mac << ","
      << std::fixed << std::setprecision(8) << res.cpu_mse << ","
      << res.mem_mse << ","
      << res.cpu_anomaly << ","
      << res.mem_anomaly << ","
      << std::setprecision(4) << res.cpu_severity << ","
      << res.mem_severity << ","
      << res.anomaly_type << ","
      << std::setprecision(2) << res.inference_time_ms << "\n";
}

// ── Batch mode ───────────────────────────────────────────────────────────────

int RunBatch(AnomalyPredictionEngine& engine,
             const std::string& input_path,
             const std::string& output_path,
             bool verbose) {
  
  std::ifstream infile(input_path);
  if (!infile) {
    std::cerr << "Error: Cannot open input file: " << input_path << "\n";
    return 1;
  }
  
  std::ostream* out = &std::cout;
  std::ofstream outfile;
  if (!output_path.empty() && output_path != "-") {
    outfile.open(output_path);
    if (!outfile) {
      std::cerr << "Error: Cannot open output file: " << output_path << "\n";
      return 1;
    }
    out = &outfile;
  }
  
  // Read header
  std::string line;
  if (!std::getline(infile, line)) {
    std::cerr << "Error: Empty input file\n";
    return 1;
  }
  
  auto header = SplitCsv(line);
  ColumnMap colmap = BuildColumnMap(header);
  
  // Write output header
  WriteResultHeader(*out);
  
  // Process rows
  int total_rows = 0;
  int anomaly_count = 0;
  
  while (std::getline(infile, line)) {
    if (line.empty()) continue;
    
    auto row = SplitCsv(line);
    TelemetryReading reading = ParseRow(row, colmap);
    
    PredictionResult result = engine.ProcessReading(reading);
    
    if (result.window_ready) {
      WriteResult(*out, reading, result);
      total_rows++;
      
      if (result.cpu_anomaly || result.mem_anomaly) {
        anomaly_count++;
      }
    }
  }
  
  if (verbose) {
    std::cerr << "\n=== Summary ===\n";
    std::cerr << "Total predictions: " << total_rows << "\n";
    std::cerr << "Anomalies found:   " << anomaly_count << "\n";
    std::cerr << "Anomaly rate:      " 
              << std::fixed << std::setprecision(2)
              << (100.0 * anomaly_count / std::max(1, total_rows)) << "%\n";
  }
  
  return 0;
}

// ── File watcher mode ────────────────────────────────────────────────────────

int RunWatcher(AnomalyPredictionEngine& engine,
               const std::string& watch_path,
               const std::string& result_path,
               int poll_interval_ms,
               bool skip_existing,
               bool verbose) {
  
  std::cerr << "[Watcher] Monitoring: " << watch_path << "\n";
  std::cerr << "[Watcher] Output to:  " << result_path << "\n";
  
  // Open result file for appending
  std::ofstream result_file(result_path, std::ios::app);
  if (!result_file) {
    std::cerr << "Error: Cannot open result file: " << result_path << "\n";
    return 1;
  }
  
  // Check if result file is empty, write header if so
  result_file.seekp(0, std::ios::end);
  if (result_file.tellp() == 0) {
    WriteResultHeader(result_file);
    result_file.flush();
  }
  
  // Track file position
  std::ifstream watch_file(watch_path);
  ColumnMap colmap;
  bool has_header = false;
  
  if (watch_file && skip_existing) {
    // Read header
    std::string line;
    if (std::getline(watch_file, line)) {
      auto header = SplitCsv(line);
      colmap = BuildColumnMap(header);
      has_header = true;
    }
    // Seek to end
    watch_file.seekg(0, std::ios::end);
  }
  
  // Install signal handlers
  signal(SIGTERM, SignalHandler);
  signal(SIGINT, SignalHandler);
  
  // Main watch loop
  while (g_running) {
    // Reopen file if needed
    if (!watch_file || !watch_file.is_open()) {
      watch_file.open(watch_path);
      if (!watch_file) {
        usleep(poll_interval_ms * 1000);
        continue;
      }
      has_header = false;
    }
    
    // Read new lines
    std::string line;
    while (std::getline(watch_file, line)) {
      if (line.empty()) continue;
      
      // Parse header if first line
      if (!has_header) {
        auto header = SplitCsv(line);
        colmap = BuildColumnMap(header);
        has_header = true;
        continue;
      }
      
      auto row = SplitCsv(line);
      TelemetryReading reading = ParseRow(row, colmap);
      
      PredictionResult result = engine.ProcessReading(reading);
      
      if (result.window_ready) {
        WriteResult(result_file, reading, result);
        result_file.flush();
        
        if (verbose || result.cpu_anomaly || result.mem_anomaly) {
          std::cerr << "[" << reading.timestamp << "] " << reading.mac << " "
                    << result.anomaly_type;
          if (result.cpu_anomaly || result.mem_anomaly) {
            std::cerr << " (cpu=" << result.cpu_severity 
                      << " mem=" << result.mem_severity << ")";
          }
          std::cerr << "\n";
        }
      }
    }
    
    // Clear EOF flag for next read
    watch_file.clear();
    
    // Sleep before next poll
    usleep(poll_interval_ms * 1000);
  }
  
  std::cerr << "[Watcher] Shutting down\n";
  return 0;
}

// ── Help message ─────────────────────────────────────────────────────────────

void PrintUsage(const char* prog) {
  std::cerr << R"(
RDKB Anomaly Prediction - TCN Autoencoder Inference
====================================================

Usage: )" << prog << R"( [options]

General options:
  --config, -c <path>     Config JSON (default: anomaly_prediction_config.json)
  --threads, -t <n>       TFLite threads (default: 1)
  --delegate-path, -d     External delegate .so (e.g., bstorm)
  --delegate-options      Delegate options (key:value;key:value)
  --verbose, -v           Verbose output
  --help, -h              This help message

Batch mode (--input given):
  --input, -i <path>      Input CSV file
  --output, -o <path>     Output CSV (default: stdout)

Watch mode (default):
  --watch-path <path>     CSV to monitor (default: /rdklogs/logs/system_stats_data.csv)
  --result-path <path>    Output CSV (default: /rdklogs/logs/anomaly_predictions.csv)
  --poll-interval <ms>    Poll interval (default: 60000)
  --skip-existing         Skip existing rows on startup

Examples:
  # Batch mode with bstorm delegate
  )" << prog << R"( -c config.json -i data.csv -o results.csv \
      -d /usr/lib/libbstorm_external_delegate.so

  # Watch mode on RDKB
  )" << prog << R"( -c config.json \
      --delegate-path /usr/lib/libbstorm_external_delegate.so \
      --delegate-options "bstm:1;bstm-client-mode:0" \
      --verbose

)";
}

}  // namespace anomaly_prediction
}  // namespace tflite

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  using namespace tflite::anomaly_prediction;
  
  // Default options
  std::string config_path = "anomaly_prediction_config.json";
  std::string input_path;
  std::string output_path;
  std::string watch_path = "/rdklogs/logs/system_stats_data.csv";
  std::string result_path = "/rdklogs/logs/anomaly_predictions.csv";
  std::string delegate_path;
  std::string delegate_options;
  int num_threads = 1;
  int poll_interval_ms = 60000;
  bool skip_existing = false;
  bool verbose = false;
  
  // Parse command line
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    
    if (arg == "-h" || arg == "--help") {
      PrintUsage(argv[0]);
      return 0;
    } else if ((arg == "-c" || arg == "--config") && i+1 < argc) {
      config_path = argv[++i];
    } else if ((arg == "-i" || arg == "--input") && i+1 < argc) {
      input_path = argv[++i];
    } else if ((arg == "-o" || arg == "--output") && i+1 < argc) {
      output_path = argv[++i];
    } else if ((arg == "-t" || arg == "--threads") && i+1 < argc) {
      num_threads = std::stoi(argv[++i]);
    } else if ((arg == "-d" || arg == "--delegate-path") && i+1 < argc) {
      delegate_path = argv[++i];
    } else if (arg == "--delegate-options" && i+1 < argc) {
      delegate_options = argv[++i];
    } else if (arg == "--watch-path" && i+1 < argc) {
      watch_path = argv[++i];
    } else if (arg == "--result-path" && i+1 < argc) {
      result_path = argv[++i];
    } else if (arg == "--poll-interval" && i+1 < argc) {
      poll_interval_ms = std::stoi(argv[++i]);
    } else if (arg == "--skip-existing") {
      skip_existing = true;
    } else if (arg == "-v" || arg == "--verbose") {
      verbose = true;
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      PrintUsage(argv[0]);
      return 1;
    }
  }
  
  // Initialize engine
  std::cerr << "[Init] Loading config: " << config_path << "\n";
  AnomalyPredictionEngine engine(config_path, num_threads, delegate_path,
                                  delegate_options, verbose);
  
  if (!engine.IsInitialized()) {
    std::cerr << "Error: Failed to initialize prediction engine\n";
    return 1;
  }
  
  // Run in appropriate mode
  if (!input_path.empty()) {
    return RunBatch(engine, input_path, output_path, verbose);
  } else {
    return RunWatcher(engine, watch_path, result_path, poll_interval_ms,
                      skip_existing, verbose);
  }
}
