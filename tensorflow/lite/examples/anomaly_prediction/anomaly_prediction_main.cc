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
#include <mutex>
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

// ══════════════════════════════════════════════════════════════════════════════
// LOGGING SYSTEM
// ══════════════════════════════════════════════════════════════════════════════

static const char* LOG_FILE_PATH = "/rdklogs/logs/anomaly_predict_logs.txt";
static std::ofstream g_log_file;
static std::mutex g_log_mutex;
static bool g_log_initialized = false;

// Log levels
enum class LogLevel { DEBUG, INFO, WARN, ERROR };

static const char* LogLevelStr(LogLevel level) {
  switch (level) {
    case LogLevel::DEBUG: return "DEBUG";
    case LogLevel::INFO:  return "INFO ";
    case LogLevel::WARN:  return "WARN ";
    case LogLevel::ERROR: return "ERROR";
    default: return "?????";
  }
}

static std::string GetTimestamp() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  struct tm* tm_info = localtime(&tv.tv_sec);
  char buf[64];
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm_info);
  char result[80];
  snprintf(result, sizeof(result), "%s.%03d", buf, (int)(tv.tv_usec / 1000));
  return std::string(result);
}

static void InitLogger() {
  if (g_log_initialized) return;
  
  std::lock_guard<std::mutex> lock(g_log_mutex);
  if (g_log_initialized) return;
  
  // Create directory if it doesn't exist
  mkdir("/rdklogs", 0755);
  mkdir("/rdklogs/logs", 0755);
  
  g_log_file.open(LOG_FILE_PATH, std::ios::app);
  if (g_log_file.is_open()) {
    g_log_initialized = true;
    g_log_file << "\n" << std::string(80, '=') << "\n";
    g_log_file << "[" << GetTimestamp() << "] [INFO ] "
               << "=== ANOMALY PREDICTION APP STARTED ===\n";
    g_log_file << std::string(80, '=') << "\n";
    g_log_file.flush();
  }
}

static void Log(LogLevel level, const std::string& component, 
                const std::string& message) {
  if (!g_log_initialized) InitLogger();
  
  std::string log_line = "[" + GetTimestamp() + "] [" + LogLevelStr(level) + "] "
                       + "[" + component + "] " + message;
  
  // Always print to stderr for WARN and ERROR
  if (level == LogLevel::WARN || level == LogLevel::ERROR) {
    std::cerr << log_line << "\n";
  }
  
  // Write to log file
  if (g_log_initialized) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log_file << log_line << "\n";
    g_log_file.flush();
  }
}

#define LOG_DEBUG(comp, msg) Log(LogLevel::DEBUG, comp, msg)
#define LOG_INFO(comp, msg)  Log(LogLevel::INFO, comp, msg)
#define LOG_WARN(comp, msg)  Log(LogLevel::WARN, comp, msg)
#define LOG_ERROR(comp, msg) Log(LogLevel::ERROR, comp, msg)

// ── Global state ─────────────────────────────────────────────────────────────
static volatile sig_atomic_t g_running = 1;

void SignalHandler(int sig) {
  (void)sig;
  g_running = 0;
  LOG_INFO("Signal", "Received shutdown signal");
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

void WriteResultHeader(std::ostream& out, bool is_classifier = false) {
  if (is_classifier) {
    out << "timestamp,CMMAC,cpu_probability,mem_probability,cpu_anomaly,mem_anomaly,"
        << "cpu_severity,mem_severity,anomaly_type,is_proactive,inference_ms\n";
  } else {
    out << "timestamp,CMMAC,cpu_mse,mem_mse,cpu_anomaly,mem_anomaly,"
        << "cpu_severity,mem_severity,anomaly_type,inference_ms\n";
  }
}

void WriteResult(std::ostream& out, const TelemetryReading& r, 
                 const PredictionResult& res) {
  out << r.timestamp << ","
      << r.mac << ",";
  
  // Use probability for classifier mode, MSE for autoencoder/forecaster
  if (res.is_proactive) {
    out << std::fixed << std::setprecision(4) << res.cpu_anomaly_probability << ","
        << res.mem_anomaly_probability << ",";
  } else {
    out << std::fixed << std::setprecision(8) << res.cpu_mse << ","
        << res.mem_mse << ",";
  }
  
  out << res.cpu_anomaly << ","
      << res.mem_anomaly << ","
      << std::setprecision(4) << res.cpu_severity << ","
      << res.mem_severity << ","
      << res.anomaly_type << ",";
  
  if (res.is_proactive) {
    out << "1,";  // is_proactive flag
  }
  
  out << std::setprecision(2) << res.inference_time_ms << "\n";
}

// ── Batch mode ───────────────────────────────────────────────────────────────

int RunBatch(AnomalyPredictionEngine& engine,
             const std::string& input_path,
             const std::string& output_path,
             bool verbose) {
  
  LOG_INFO("Batch", "Starting batch processing mode");
  LOG_INFO("Batch", "Input file: " + input_path);
  LOG_INFO("Batch", "Output file: " + (output_path.empty() ? "stdout" : output_path));
  
  std::ifstream infile(input_path);
  if (!infile) {
    LOG_ERROR("Batch", "Cannot open input file: " + input_path);
    std::cerr << "Error: Cannot open input file: " << input_path << "\n";
    return 1;
  }
  LOG_DEBUG("Batch", "Input file opened successfully");
  
  std::ostream* out = &std::cout;
  std::ofstream outfile;
  if (!output_path.empty() && output_path != "-") {
    outfile.open(output_path);
    if (!outfile) {
      LOG_ERROR("Batch", "Cannot open output file: " + output_path);
      std::cerr << "Error: Cannot open output file: " << output_path << "\n";
      return 1;
    }
    out = &outfile;
    LOG_DEBUG("Batch", "Output file opened successfully");
  }
  
  // Read header
  std::string line;
  if (!std::getline(infile, line)) {
    LOG_ERROR("Batch", "Empty input file - no header found");
    std::cerr << "Error: Empty input file\n";
    return 1;
  }
  
  auto header = SplitCsv(line);
  ColumnMap colmap = BuildColumnMap(header);
  LOG_INFO("Batch", "CSV header parsed - " + std::to_string(header.size()) + " columns");
  LOG_DEBUG("Batch", "Column mappings - timestamp:" + std::to_string(colmap.timestamp) + 
            " mac:" + std::to_string(colmap.mac) + " cpu:" + std::to_string(colmap.used_cpu));
  
  // Write output header (check if classifier mode for different header format)
  bool is_classifier = (engine.GetConfig().model_type == ModelType::kClassifier);
  WriteResultHeader(*out, is_classifier);
  
  // Process rows
  int total_rows = 0;
  int anomaly_count = 0;
  int processed_count = 0;
  
  LOG_INFO("Batch", "Starting row processing...");
  
  while (std::getline(infile, line)) {
    if (line.empty()) continue;
    processed_count++;
    
    auto row = SplitCsv(line);
    TelemetryReading reading = ParseRow(row, colmap);
    
    if (processed_count % 100 == 0) {
      LOG_DEBUG("Batch", "Processing row " + std::to_string(processed_count) + 
                " - timestamp: " + reading.timestamp + " mac: " + reading.mac);
    }
    
    PredictionResult result = engine.ProcessReading(reading);
    
    if (result.window_ready) {
      WriteResult(*out, reading, result);
      total_rows++;
      
      if (result.cpu_anomaly || result.mem_anomaly) {
        anomaly_count++;
        if (result.is_proactive) {
          LOG_INFO("Batch", "PROACTIVE ANOMALY WARNING at " + reading.timestamp + 
                   " mac=" + reading.mac + " type=" + result.anomaly_type +
                   " cpu_prob=" + std::to_string(result.cpu_anomaly_probability) +
                   " mem_prob=" + std::to_string(result.mem_anomaly_probability));
        } else {
          LOG_INFO("Batch", "ANOMALY DETECTED at " + reading.timestamp + 
                   " mac=" + reading.mac + " type=" + result.anomaly_type +
                   " cpu_mse=" + std::to_string(result.cpu_mse) +
                   " mem_mse=" + std::to_string(result.mem_mse));
        }
      }
    }
  }
  
  std::ostringstream summary;
  summary << "Batch complete - Total predictions: " << total_rows 
          << ", Anomalies: " << anomaly_count 
          << " (" << std::fixed << std::setprecision(2) 
          << (100.0 * anomaly_count / std::max(1, total_rows)) << "%)";
  LOG_INFO("Batch", summary.str());
  
  if (verbose) {
    std::cerr << "\n=== Summary ===\n";
    std::cerr << "Total predictions: " << total_rows << "\n";
    std::cerr << "Anomalies found:   " << anomaly_count << "\n";
    std::cerr << "Anomaly rate:      " 
              << std::fixed << std::setprecision(2)
              << (100.0 * anomaly_count / std::max(1, total_rows)) << "%\n";
  }
  
  LOG_INFO("Batch", "Batch processing completed successfully");
  return 0;
}

// ── Helper: Get file inode ───────────────────────────────────────────────────

ino_t GetFileInode(const std::string& path) {
  struct stat st;
  if (stat(path.c_str(), &st) == 0) {
    return st.st_ino;
  }
  return 0;
}

off_t GetFileSize(const std::string& path) {
  struct stat st;
  if (stat(path.c_str(), &st) == 0) {
    return st.st_size;
  }
  return -1;
}

// ── File watcher mode ────────────────────────────────────────────────────────

int RunWatcher(AnomalyPredictionEngine& engine,
               const std::string& watch_path,
               const std::string& result_path,
               int poll_interval_ms,
               bool skip_existing,
               bool verbose) {
  
  LOG_INFO("Watcher", "Starting file watcher mode");
  LOG_INFO("Watcher", "Watch path: " + watch_path);
  LOG_INFO("Watcher", "Result path: " + result_path);
  LOG_INFO("Watcher", "Poll interval: " + std::to_string(poll_interval_ms) + " ms");
  LOG_INFO("Watcher", "Skip existing: " + std::string(skip_existing ? "yes" : "no"));
  
  std::cerr << "[Watcher] Monitoring: " << watch_path << "\n";
  std::cerr << "[Watcher] Output to:  " << result_path << "\n";
  
  // Open result file for appending
  std::ofstream result_file(result_path, std::ios::app);
  if (!result_file) {
    LOG_ERROR("Watcher", "Cannot open result file: " + result_path);
    std::cerr << "Error: Cannot open result file: " << result_path << "\n";
    return 1;
  }
  LOG_DEBUG("Watcher", "Result file opened for appending");
  
  // Check if result file is empty, write header if so
  bool is_classifier = (engine.GetConfig().model_type == ModelType::kClassifier);
  result_file.seekp(0, std::ios::end);
  if (result_file.tellp() == 0) {
    WriteResultHeader(result_file, is_classifier);
    result_file.flush();
    LOG_DEBUG("Watcher", "Wrote header to empty result file");
  }
  
  // Track file position and inode for rotation detection
  std::ifstream watch_file(watch_path);
  ColumnMap colmap;
  bool has_header = false;
  ino_t current_inode = GetFileInode(watch_path);
  std::streampos last_read_pos = 0;
  
  LOG_DEBUG("Watcher", "Initial inode: " + std::to_string(current_inode));
  
  if (watch_file && skip_existing) {
    LOG_DEBUG("Watcher", "Skipping existing content in watch file");
    // Read header
    std::string line;
    if (std::getline(watch_file, line)) {
      auto header = SplitCsv(line);
      colmap = BuildColumnMap(header);
      has_header = true;
      LOG_DEBUG("Watcher", "Header parsed with " + std::to_string(header.size()) + " columns");
    }
    // Seek to end
    watch_file.seekg(0, std::ios::end);
    last_read_pos = watch_file.tellg();
    LOG_INFO("Watcher", "Positioned at end of file (pos=" + std::to_string(last_read_pos) + "), waiting for new data");
  }
  
  // Install signal handlers
  signal(SIGTERM, SignalHandler);
  signal(SIGINT, SignalHandler);
  LOG_DEBUG("Watcher", "Signal handlers installed");
  
  int total_processed = 0;
  int total_anomalies = 0;
  int poll_count = 0;
  int rotation_count = 0;
  
  LOG_INFO("Watcher", "Entering main watch loop");
  
  // Main watch loop
  while (g_running) {
    poll_count++;
    
    // ═══════════════════════════════════════════════════════════════════════
    // FILE ROTATION DETECTION
    // Detect if file was rotated (uploaded to cloud and replaced with new file)
    // Indicators: inode changed, or file size decreased (new file is smaller)
    // ═══════════════════════════════════════════════════════════════════════
    ino_t new_inode = GetFileInode(watch_path);
    off_t current_size = GetFileSize(watch_path);
    
    bool file_rotated = false;
    
    // Check if inode changed (file was replaced)
    if (new_inode != 0 && current_inode != 0 && new_inode != current_inode) {
      LOG_INFO("Watcher", "FILE ROTATION DETECTED: inode changed from " + 
               std::to_string(current_inode) + " to " + std::to_string(new_inode));
      file_rotated = true;
    }
    // Check if file size decreased (file was truncated/replaced)
    else if (watch_file.is_open() && current_size >= 0) {
      std::streampos current_pos = watch_file.tellg();
      if (current_pos != std::streampos(-1) && 
          current_size < static_cast<off_t>(current_pos)) {
        LOG_INFO("Watcher", "FILE ROTATION DETECTED: size decreased from pos " + 
                 std::to_string(current_pos) + " to " + std::to_string(current_size));
        file_rotated = true;
      }
    }
    
    if (file_rotated) {
      rotation_count++;
      LOG_INFO("Watcher", "Handling file rotation #" + std::to_string(rotation_count));
      std::cerr << "[Watcher] File rotated - reopening from start\n";
      
      // Close existing file handle
      if (watch_file.is_open()) {
        watch_file.close();
      }
      
      // Reset state for new file
      current_inode = new_inode;
      has_header = false;
      last_read_pos = 0;
      
      // Reopen the new file
      watch_file.open(watch_path);
      if (watch_file) {
        LOG_INFO("Watcher", "Reopened rotated file successfully, inode=" + std::to_string(current_inode));
      }
    }
    
    // Reopen file if needed (file doesn't exist or was closed)
    if (!watch_file || !watch_file.is_open()) {
      LOG_DEBUG("Watcher", "Attempting to open watch file");
      watch_file.open(watch_path);
      if (!watch_file) {
        if (poll_count % 10 == 0) {  // Log every 10th attempt
          LOG_WARN("Watcher", "Watch file not available: " + watch_path);
        }
        // Reset inode tracking when file is missing
        current_inode = 0;
        usleep(poll_interval_ms * 1000);
        continue;
      }
      // Update inode for newly opened file
      current_inode = GetFileInode(watch_path);
      has_header = false;
      last_read_pos = 0;
      LOG_INFO("Watcher", "Watch file opened, inode=" + std::to_string(current_inode));
    }
    
    // Read new lines
    std::string line;
    int lines_this_poll = 0;
    
    while (std::getline(watch_file, line)) {
      if (line.empty()) continue;
      
      // Parse header if first line
      if (!has_header) {
        auto header = SplitCsv(line);
        colmap = BuildColumnMap(header);
        has_header = true;
        LOG_INFO("Watcher", "CSV header parsed - " + std::to_string(header.size()) + " columns");
        continue;
      }
      
      lines_this_poll++;
      auto row = SplitCsv(line);
      TelemetryReading reading = ParseRow(row, colmap);
      
      LOG_DEBUG("Watcher", "Processing reading - timestamp: " + reading.timestamp + 
                " mac: " + reading.mac + " cpu: " + std::to_string(reading.used_cpu) +
                " mem: " + std::to_string(reading.used_mem_kb));
      
      PredictionResult result = engine.ProcessReading(reading);
      
      if (result.window_ready) {
        WriteResult(result_file, reading, result);
        result_file.flush();
        total_processed++;
        
        if (result.cpu_anomaly || result.mem_anomaly) {
          total_anomalies++;
          if (result.is_proactive) {
            LOG_WARN("Watcher", "PROACTIVE ANOMALY WARNING at " + reading.timestamp + 
                     " mac=" + reading.mac + " type=" + result.anomaly_type +
                     " cpu_prob=" + std::to_string(result.cpu_anomaly_probability) +
                     " mem_prob=" + std::to_string(result.mem_anomaly_probability));
          } else {
            LOG_WARN("Watcher", "ANOMALY DETECTED at " + reading.timestamp + 
                     " mac=" + reading.mac + " type=" + result.anomaly_type +
                     " cpu_mse=" + std::to_string(result.cpu_mse) +
                     " cpu_sev=" + std::to_string(result.cpu_severity) +
                     " mem_mse=" + std::to_string(result.mem_mse) +
                     " mem_sev=" + std::to_string(result.mem_severity));
          }
        } else {
          if (result.is_proactive) {
            LOG_DEBUG("Watcher", "Normal - " + reading.timestamp + " " + reading.mac +
                      " cpu_prob=" + std::to_string(result.cpu_anomaly_probability) +
                      " mem_prob=" + std::to_string(result.mem_anomaly_probability));
          } else {
            LOG_DEBUG("Watcher", "Normal - " + reading.timestamp + " " + reading.mac +
                      " cpu_mse=" + std::to_string(result.cpu_mse) +
                      " mem_mse=" + std::to_string(result.mem_mse));
          }
        }
        
        if (verbose || result.cpu_anomaly || result.mem_anomaly) {
          std::cerr << "[" << reading.timestamp << "] " << reading.mac << " "
                    << result.anomaly_type;
          if (result.cpu_anomaly || result.mem_anomaly) {
            std::cerr << " (cpu=" << result.cpu_severity 
                      << " mem=" << result.mem_severity << ")";
          }
          std::cerr << "\n";
        }
      } else {
        LOG_DEBUG("Watcher", "Window not ready - samples: " + 
                  std::to_string(result.window_samples) + " for mac: " + reading.mac);
      }
    }
    
    if (lines_this_poll > 0) {
      LOG_DEBUG("Watcher", "Processed " + std::to_string(lines_this_poll) + 
                " lines this poll cycle");
    }
    
    // Clear EOF flag for next read
    watch_file.clear();
    
    // Update last read position for rotation detection
    last_read_pos = watch_file.tellg();
    
    // Sleep before next poll
    usleep(poll_interval_ms * 1000);
  }
  
  LOG_INFO("Watcher", "Shutdown requested - Total processed: " + std::to_string(total_processed) +
           ", Total anomalies: " + std::to_string(total_anomalies) +
           ", File rotations: " + std::to_string(rotation_count));
  std::cerr << "[Watcher] Shutting down (rotations=" << rotation_count << ")\n";
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
  
  // Initialize logger first
  InitLogger();
  LOG_INFO("Main", "Application starting");
  LOG_INFO("Main", "Command line: " + std::string(argv[0]));
  
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
  
  LOG_DEBUG("Main", "Parsing command line arguments");
  
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
      LOG_ERROR("Main", "Unknown option: " + arg);
      std::cerr << "Unknown option: " << arg << "\n";
      PrintUsage(argv[0]);
      return 1;
    }
  }
  
  // Log parsed configuration
  LOG_INFO("Main", "Configuration parsed:");
  LOG_INFO("Main", "  config_path: " + config_path);
  LOG_INFO("Main", "  delegate_path: " + (delegate_path.empty() ? "(none)" : delegate_path));
  LOG_INFO("Main", "  delegate_options: " + (delegate_options.empty() ? "(none)" : delegate_options));
  LOG_INFO("Main", "  num_threads: " + std::to_string(num_threads));
  LOG_INFO("Main", "  verbose: " + std::string(verbose ? "yes" : "no"));
  if (!input_path.empty()) {
    LOG_INFO("Main", "  mode: BATCH");
    LOG_INFO("Main", "  input_path: " + input_path);
    LOG_INFO("Main", "  output_path: " + (output_path.empty() ? "stdout" : output_path));
  } else {
    LOG_INFO("Main", "  mode: WATCHER");
    LOG_INFO("Main", "  watch_path: " + watch_path);
    LOG_INFO("Main", "  result_path: " + result_path);
    LOG_INFO("Main", "  poll_interval_ms: " + std::to_string(poll_interval_ms));
  }
  
  // Initialize engine
  LOG_INFO("Main", "Initializing prediction engine...");
  std::cerr << "[Init] Loading config: " << config_path << "\n";
  
  AnomalyPredictionEngine engine(config_path, num_threads, delegate_path,
                                  delegate_options, verbose);
  
  if (!engine.IsInitialized()) {
    LOG_ERROR("Main", "Failed to initialize prediction engine");
    std::cerr << "Error: Failed to initialize prediction engine\n";
    return 1;
  }
  
  LOG_INFO("Main", "Prediction engine initialized successfully");
  
  // Print model configuration
  const auto& cfg = engine.GetConfig();
  std::string model_type_str = (cfg.model_type == ModelType::kForecaster ? "Forecaster" : "Autoencoder");
  
  LOG_INFO("Main", "Model configuration:");
  LOG_INFO("Main", "  model_type: " + model_type_str);
  LOG_INFO("Main", "  window_size: " + std::to_string(cfg.window_size));
  LOG_INFO("Main", "  warmup_samples: " + std::to_string(cfg.warmup_samples));
  LOG_INFO("Main", "  cpu_threshold: " + std::to_string(cfg.cpu_threshold));
  LOG_INFO("Main", "  mem_threshold: " + std::to_string(cfg.mem_threshold));
  LOG_INFO("Main", "  cpu_model: " + cfg.cpu_model_file);
  LOG_INFO("Main", "  mem_model: " + cfg.mem_model_file);
  LOG_INFO("Main", "  bstorm_compatible: " + std::string(cfg.bstorm_compatible ? "yes" : "no"));
  
  std::cerr << "[Init] Model type: " << model_type_str << "\n";
  std::cerr << "[Init] Window size: " << cfg.window_size << "\n";
  std::cerr << "[Init] Thresholds - CPU: " << cfg.cpu_threshold 
            << ", Memory: " << cfg.mem_threshold << "\n";
  if (cfg.model_type == ModelType::kForecaster) {
    std::cerr << "[Init] NOTE: Forecaster mode - anomaly detected when "
              << "predicted t+1 != actual t+1\n";
  }
  
  // Run in appropriate mode
  int result;
  if (!input_path.empty()) {
    LOG_INFO("Main", "Starting batch mode");
    result = RunBatch(engine, input_path, output_path, verbose);
  } else {
    LOG_INFO("Main", "Starting watcher mode");
    result = RunWatcher(engine, watch_path, result_path, poll_interval_ms,
                      skip_existing, verbose);
  }
  
  LOG_INFO("Main", "Application exiting with code: " + std::to_string(result));
  return result;
}
