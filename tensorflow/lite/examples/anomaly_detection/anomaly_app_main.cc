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

anomaly_app_main.cc
────────────────────────────────────────────────────────────────────────────
Command-line entry point for the DOCSIS gateway anomaly detection engine.

Usage:
  anomaly_app [options]

Options:
  --config,     -c  <path>   Path to inference_config.json
                             (default: inference_config.json)
  --input,      -i  <path>   Path to input CSV file
                             (default: reads CSV from stdin)
  --output,     -o  <path>   Path to output CSV  (default: stdout)
  --threads,    -t  <n>      Number of TFLite threads (default: 1)
  --verbose,    -v           Print per-reading details
  --check-libs               Check shared library availability and exit
  --help,       -h           Print this message

CSV input columns (same names as cleaned_data_50mac_xb10_8_3p5s1.csv):
  timestamp, CMMAC, USED_CPU_ATOM, LOAD_AVG_ATOM,
  USED_MEM_ATOM_kB, AvailMem_kB, FreeMem_kB, SlabMem_kB,
  2G_Clients_Count, 5G_Clients_Count, 6G_Clients_Count

Output CSV columns:
  timestamp, CMMAC, dense_cpu_mse, dense_mem_mse,
  dense_cpu_flag, dense_mem_flag, dense_cpu_sev, dense_mem_sev,
  [lstm_cpu_mse, lstm_mem_mse, lstm_cpu_flag, lstm_mem_flag,
   lstm_cpu_sev, lstm_mem_sev,]
  anomaly_type
============================================================================*/

#include "tensorflow/lite/examples/anomaly_detection/anomaly_detection.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/delegates/flex/delegate.h"

namespace tflite {
namespace anomaly_detection {

// ── Minimal CSV parser ────────────────────────────────────────────────────────
static std::vector<std::string> SplitCsv(const std::string& line) {
  std::vector<std::string> tokens;
  std::stringstream ss(line);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    // Trim surrounding whitespace / quotes
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

// Parse "2026-03-27T09:00:00" → hour and day_of_week (0=Mon … 6=Sun)
static void ParseTimestamp(const std::string& ts, int& hour, int& dow) {
  // Format: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD HH:MM:SS
  hour = 0; dow = 0;
  if (ts.size() >= 16) {
    int delim_pos = (ts[10] == 'T' || ts[10] == ' ') ? 11 : -1;
    if (delim_pos > 0) hour = SafeInt(ts.substr(delim_pos, 2));
  }
  // Simple day-of-week from date using Zeller-like formula
  if (ts.size() >= 10) {
    int y = SafeInt(ts.substr(0, 4));
    int m = SafeInt(ts.substr(5, 2));
    int d = SafeInt(ts.substr(8, 2));
    // Tomohiko Sakamoto's algorithm — returns 0=Sun ... 6=Sat
    // Convert to 0=Mon … 6=Sun to match pandas .dayofweek
    static const int t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
    if (m < 3) y--;
    int wd = (y + y/4 - y/100 + y/400 + t[m-1] + d) % 7;  // 0=Sun
    dow = (wd + 6) % 7;  // shift to 0=Mon
  }
}

// ── Flex delegate runtime check ───────────────────────────────────────────────
// Call this before loading any model to verify libtensorflowlite_flex.so is
// present and functional on the current device.
//
// Returns true if the Flex delegate can be created successfully.
// Prints a clear diagnostic message either way.
static bool CheckFlexDelegate() {
  std::cerr << "\n[lib-check] Testing Flex delegate (required for LSTM models)...\n";

  TfLiteDelegate* d = TfLiteFlexDelegateCreate(nullptr);
  if (!d) {
    std::cerr
      << "[lib-check] FAIL: TfLiteFlexDelegateCreate() returned null.\n"
      << "            Likely cause: libtensorflowlite_flex.so is not in LD_LIBRARY_PATH\n"
      << "            or was not deployed alongside the binary.\n"
      << "            → LSTM models (lstm_cpu_anomaly_model.tflite,\n"
      << "                           lstm_memory_anomaly_model.tflite) will NOT work.\n"
      << "            → Dense models (cpu_anomaly_model.tflite,\n"
      << "                            memory_anomaly_model.tflite) are unaffected.\n\n"
      << "  To fix on target:\n"
      << "    export LD_LIBRARY_PATH=/path/to/tflite/libs:$LD_LIBRARY_PATH\n"
      << "    ls $LD_LIBRARY_PATH/libtensorflowlite_flex.so   # must exist\n\n"
      << "  To verify on host before deploying:\n"
      << "    readelf -d anomaly_app | grep NEEDED\n"
      << "    # must show: libtensorflowlite_flex.so\n"
      << "    aarch64-linux-gnu-readelf -d anomaly_app | grep NEEDED\n\n"
      << "  To verify on target after copying:\n"
      << "    ldd ./anomaly_app | grep flex\n"
      << "    # must show: libtensorflowlite_flex.so => /path/to/lib (0x...)\n"
      << "    # 'not found' means the .so was not deployed.\n\n";
    return false;
  }

  // Clean up — we only created it to test availability
  TfLiteFlexDelegateDelete(d);
  std::cerr << "[lib-check] OK : Flex delegate available — LSTM models will work.\n\n";
  return true;
}

// ── Print usage ───────────────────────────────────────────────────────────────
static void PrintUsage(const char* prog) {
  std::cerr
    << "\nDOCSIS Gateway Anomaly Detection — C++ Inference Engine\n\n"
    << "  Reads a CSV of telemetry readings, runs TFLite anomaly models,\n"
    << "  and outputs anomaly classification results.\n\n"
    << "Usage: " << prog << " [options]\n\n"
    << "  --config,   -c <path>   inference_config.json  (default: inference_config.json)\n"
    << "  --input,    -i <path>   Input CSV file          (default: stdin)\n"
    << "  --output,   -o <path>   Output CSV file         (default: stdout)\n"
    << "  --threads,  -t <n>      TFLite thread count     (default: 1)\n"
    << "  --verbose,  -v          Print per-reading details to stderr\n"
    << "  --check-libs            Check shared library availability and exit\n"
    << "  --help,     -h          Print this message\n\n";
}

// ── Main ──────────────────────────────────────────────────────────────────────
int Main(int argc, char** argv) {
  std::string config_path = "inference_config.json";
  std::string input_path;
  std::string output_path;
  int  num_threads = 1;
  bool verbose     = false;
  bool check_libs  = false;

  // ── Parse arguments ───────────────────────────────────────────────────────
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--check-libs") {
      check_libs = true;
    } else if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
      config_path = argv[++i];
    } else if ((arg == "--input" || arg == "-i") && i + 1 < argc) {
      input_path = argv[++i];
    } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
      output_path = argv[++i];
    } else if ((arg == "--threads" || arg == "-t") && i + 1 < argc) {
      num_threads = std::atoi(argv[++i]);
    } else if (arg == "--verbose" || arg == "-v") {
      verbose = true;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return EXIT_SUCCESS;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      PrintUsage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  // ── Library presence check (--check-libs flag) ────────────────────────────
  if (check_libs) {
    std::cerr << "[lib-check] Checking shared library availability...\n";
    std::cerr << "[lib-check] libtensorflowlite.so       : required for all models\n";
    std::cerr << "[lib-check] libtensorflowlite_flex.so  : required for LSTM models\n";
    bool flex_ok = CheckFlexDelegate();
    return flex_ok ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  // ── Initialise engine ─────────────────────────────────────────────────────
  std::cerr << "Loading config: " << config_path << "\n";
  AnomalyInferenceEngine engine(config_path, num_threads);
  std::cerr << "Engine ready.  SEQ_LEN=" << engine.seq_len()
            << "  threads=" << num_threads << "\n";

  // ── Open I/O streams ──────────────────────────────────────────────────────
  std::ifstream fin;
  if (!input_path.empty()) {
    fin.open(input_path);
    if (!fin) { std::cerr << "Cannot open input: " << input_path << "\n"; return EXIT_FAILURE; }
  }
  std::istream& in = fin.is_open() ? fin : std::cin;

  std::ofstream fout;
  if (!output_path.empty()) {
    fout.open(output_path);
    if (!fout) { std::cerr << "Cannot open output: " << output_path << "\n"; return EXIT_FAILURE; }
  }
  std::ostream& out = fout.is_open() ? fout : std::cout;

  // ── Read CSV header ───────────────────────────────────────────────────────
  std::string header_line;
  if (!std::getline(in, header_line)) {
    std::cerr << "Input is empty.\n";
    return EXIT_FAILURE;
  }
  std::vector<std::string> cols = SplitCsv(header_line);

  // Build column index lookup
  std::unordered_map<std::string, int> col_idx;
  for (int i = 0; i < static_cast<int>(cols.size()); ++i)
    col_idx[cols[i]] = i;

  auto GetCol = [&](const std::string& name, int def = -1) -> int {
    auto it = col_idx.find(name);
    return it != col_idx.end() ? it->second : def;
  };

  // Required column indices
  int ci_ts    = GetCol("timestamp");
  int ci_mac   = GetCol("CMMAC");
  int ci_cpu   = GetCol("USED_CPU_ATOM");
  int ci_load  = GetCol("LOAD_AVG_ATOM");
  int ci_umem  = GetCol("USED_MEM_ATOM_kB");
  int ci_amem  = GetCol("AvailMem_kB");
  int ci_fmem  = GetCol("FreeMem_kB");
  int ci_slab  = GetCol("SlabMem_kB");
  int ci_2g    = GetCol("2G_Clients_Count");
  int ci_5g    = GetCol("5G_Clients_Count");
  int ci_6g    = GetCol("6G_Clients_Count");
  int ci_hour  = GetCol("hour_of_day");
  int ci_dow   = GetCol("day_of_week");

  // ── Write output CSV header ───────────────────────────────────────────────
  bool has_lstm = false;  // will be set after first reading
  // Write header after we know whether LSTM models are present
  bool header_written = false;

  auto WriteHeader = [&](bool with_lstm) {
    out << "timestamp,CMMAC"
        << ",dense_cpu_mse,dense_mem_mse"
        << ",dense_cpu_flag,dense_mem_flag"
        << ",dense_cpu_sev,dense_mem_sev";
    if (with_lstm) {
      out << ",lstm_cpu_mse,lstm_mem_mse"
          << ",lstm_cpu_flag,lstm_mem_flag"
          << ",lstm_cpu_sev,lstm_mem_sev";
    }
    out << ",anomaly_type\n";
  };

  // ── Verbose header ────────────────────────────────────────────────────────
  if (verbose) {
    std::cerr << "\n=== Streaming Inference ===\n"
              << std::left
              << std::setw(22) << "CMMAC"
              << std::setw(8)  << "CPU%"
              << std::setw(10) << "CPU_sev"
              << std::setw(10) << "Mem_sev"
              << std::setw(10) << "Type"
              << "Alert\n"
              << std::string(75, '-') << "\n";
  }

  // ── Process rows ──────────────────────────────────────────────────────────
  std::string line;
  int row_num = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::vector<std::string> fields = SplitCsv(line);
    if (static_cast<int>(fields.size()) < static_cast<int>(cols.size())) continue;

    auto F = [&](int idx, float def = 0.0f) -> float {
      return (idx >= 0 && idx < static_cast<int>(fields.size()))
               ? SafeFloat(fields[idx], def) : def;
    };
    auto I = [&](int idx, int def = 0) -> int {
      return (idx >= 0 && idx < static_cast<int>(fields.size()))
               ? SafeInt(fields[idx], def) : def;
    };
    auto S = [&](int idx) -> std::string {
      return (idx >= 0 && idx < static_cast<int>(fields.size())) ? fields[idx] : "";
    };

    // Build reading
    TelemetryReading rdg;
    rdg.timestamp   = S(ci_ts);
    rdg.mac         = S(ci_mac);
    rdg.used_cpu    = F(ci_cpu);
    rdg.load_avg    = F(ci_load);
    rdg.used_mem_kb = F(ci_umem);
    rdg.avail_mem_kb= F(ci_amem);
    rdg.free_mem_kb = F(ci_fmem);
    rdg.slab_mem_kb = F(ci_slab);
    rdg.clients_2g  = I(ci_2g);
    rdg.clients_5g  = I(ci_5g);
    rdg.clients_6g  = I(ci_6g);

    // Use precomputed time columns if present, else parse timestamp
    if (ci_hour >= 0 && ci_dow >= 0) {
      rdg.hour_of_day = I(ci_hour);
      rdg.day_of_week = I(ci_dow);
    } else {
      ParseTimestamp(rdg.timestamp, rdg.hour_of_day, rdg.day_of_week);
    }

    // Run inference
    AnomalyResult res = engine.ProcessReading(rdg);
    has_lstm = res.has_lstm;

    // Write CSV header on first result
    if (!header_written) {
      WriteHeader(has_lstm);
      header_written = true;
    }

    // ── Output CSV row ─────────────────────────────────────────────────────
    out << std::fixed << std::setprecision(6)
        << rdg.timestamp << "," << rdg.mac
        << "," << res.dense_cpu_mse
        << "," << res.dense_mem_mse
        << "," << res.dense_cpu_flag
        << "," << res.dense_mem_flag
        << "," << std::setprecision(4) << res.dense_cpu_sev
        << "," << res.dense_mem_sev;

    if (has_lstm) {
      out << "," << std::setprecision(6) << res.lstm_cpu_mse
          << "," << res.lstm_mem_mse
          << "," << res.lstm_cpu_flag
          << "," << res.lstm_mem_flag
          << "," << std::setprecision(4) << res.lstm_cpu_sev
          << "," << res.lstm_mem_sev;
    }
    out << "," << res.anomaly_type << "\n";

    // ── Verbose streaming output ──────────────────────────────────────────
    if (verbose) {
      std::string alert;
      if (res.anomaly_type != "Normal") {
        alert = "ALERT --";
        if (res.dense_cpu_flag)
          alert += " CPU sev=" + std::to_string(res.dense_cpu_sev).substr(0, 5) + "x";
        if (res.dense_mem_flag)
          alert += " Mem sev=" + std::to_string(res.dense_mem_sev).substr(0, 5) + "x";
      }
      std::cerr << std::left
                << std::setw(22) << rdg.mac
                << std::setw(8)  << std::fixed << std::setprecision(1) << rdg.used_cpu
                << std::setw(10) << std::setprecision(3) << res.dense_cpu_sev
                << std::setw(10) << res.dense_mem_sev
                << std::setw(10) << res.anomaly_type
                << alert << "\n";
    }
    ++row_num;
  }

  if (!header_written) WriteHeader(false);  // empty input edge case

  std::cerr << "\nProcessed " << row_num << " readings.\n";
  return EXIT_SUCCESS;
}

}  // namespace anomaly_detection
}  // namespace tflite

int main(int argc, char** argv) {
  return tflite::anomaly_detection::Main(argc, argv);
}
