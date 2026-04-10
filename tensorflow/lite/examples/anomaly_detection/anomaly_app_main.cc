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

Modes
-----
  File-watcher (default)  — no --input given
    Watches the on-device CSV for new rows via inotify (Linux) or stat-
    polling (non-Linux) and appends anomaly results continuously.
    Run in the background with '&'.

  Batch                   — --input <path> given
    One-shot: reads a complete CSV, writes results, then exits.

Usage:
  anomaly_app [options]

General options:
  --config,  -c <path>   Path to inference_config.json
                         (default: inference_config.json)
  --threads, -t <n>      TFLite thread count (default: 1)
  --delegate-path,  -d <path>
                         External hardware delegate .so
                         (e.g. /usr/lib/libbstorm_external_delegate.so)
  --delegate-options <str>
                         Semicolon-separated key:value delegate options
                         (e.g. "bstm:1;bstm-client-mode:0;dynamic-tensors:1")
  --verbose, -v          Print per-reading details to stderr
  --check-libs           Check shared library availability and exit
  --help,    -h          Print this message

File-watcher options (used when no --input is given):
  --watch-path  <path>   Device CSV file to monitor
                         (default: /rdklogs/logs/system_stats_data.csv)
  --result-path <path>   Output file; result rows are appended
                         (default: /rdklogs/logs/anomaly_results.csv)
  --poll-interval <ms>   inotify safety-timeout on Linux; stat-poll interval
                         on non-Linux (default: 60000 ms)
  --skip-existing        Seek to EOF on startup; skip rows already present

Batch options (used when --input is given):
  --input,  -i <path>    Input CSV file
  --output, -o <path>    Output CSV file (default: stdout)

CSV header is auto-detected in both modes:
  - First field starts with a letter → column-name header row present
  - First field starts with a digit  → no header; positional order assumed:
      col 0: timestamp          col 1: CMMAC
      col 2: USED_CPU_ATOM      col 3: USED_MEM_ATOM_kB
      col 4: LOAD_AVG_ATOM      col 5: AvailMem_kB
      col 6: FreeMem_kB         col 7: SlabMem_kB
      col 8: 2G_Clients_Count   col 9: 5G_Clients_Count
      col10: 6G_Clients_Count

Timestamp formats accepted (hour_of_day / day_of_week are derived automatically):
  YYYY-MM-DDTHH:MM:SS[.mmm]   (ISO 8601)
  YYYY-MM-DD HH:MM:SS[.mmm]   (space separator)
  YYYY-MM-DD-HH:MM:SS[.mmm]   (device /rdklogs format)

Output CSV columns:
  timestamp, CMMAC,
  dense_cpu_mse, dense_mem_mse, dense_cpu_flag, dense_mem_flag,
  dense_cpu_sev, dense_mem_sev,
  anomaly_type   ∈ {Normal, CPU, Memory, Both}
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
#include <sys/time.h>
#include <unordered_map>
#include <vector>

#include <signal.h>         // signal, SIGTERM, SIGINT, sig_atomic_t
#include <sys/stat.h>       // stat, struct stat
#include <unistd.h>         // fork, setsid, dup2, getpid, usleep
#ifdef __linux__
#include <limits.h>         // NAME_MAX — max filename length for inotify event buffer
#include <poll.h>           // poll() — used with inotify fd
#include <sys/inotify.h>    // inotify_init1, inotify_add_watch
#endif

namespace tflite {
namespace anomaly_detection {

static double get_us(struct timeval t) {
  return static_cast<double>(t.tv_sec) * 1e6 + t.tv_usec;
}

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

// Parse timestamp → hour and day_of_week (0=Mon … 6=Sun)
// Accepted formats:
//   YYYY-MM-DDTHH:MM:SS[.mmm]   (ISO 8601)
//   YYYY-MM-DD HH:MM:SS[.mmm]   (space separator)
//   YYYY-MM-DD-HH:MM:SS[.mmm]   (device format from /rdklogs/logs)
static void ParseTimestamp(const std::string& ts, int& hour, int& dow) {
  hour = 0; dow = 0;
  if (ts.size() >= 13) {   // at minimum YYYY-MM-DD-HH
    // Position 10 can be 'T', ' ', or '-' (device format)
    char sep = ts[10];
    if (sep == 'T' || sep == ' ' || sep == '-')
      hour = SafeInt(ts.substr(11, 2));
  }
  // Date from first 10 chars (format is always YYYY-MM-DD in all variants)
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

// ── CSV result header writer ─────────────────────────────────────────────────
static void WriteResultHeader(std::ostream& out) {
  out << "timestamp,CMMAC"
      << ",dense_cpu_mse,dense_mem_mse"
      << ",dense_cpu_flag,dense_mem_flag"
      << ",dense_cpu_sev,dense_mem_sev"
      << ",anomaly_type\n";
}

// ── CSV result row writer ────────────────────────────────────────────────────
static void WriteResultRow(std::ostream& out,
                           const TelemetryReading& rdg,
                           const AnomalyResult& res) {
  out << std::fixed << std::setprecision(6)
      << rdg.timestamp << "," << rdg.mac
      << "," << res.dense_cpu_mse
      << "," << res.dense_mem_mse
      << "," << res.dense_cpu_flag
      << "," << res.dense_mem_flag
      << "," << std::setprecision(4) << res.dense_cpu_sev
      << "," << res.dense_mem_sev
      << "," << res.anomaly_type << "\n";
}

// ── Signal handling ───────────────────────────────────────────────────────────
static volatile sig_atomic_t g_stop = 0;
static void OnSignal(int) { g_stop = 1; }

// ── Column index bundle ───────────────────────────────────────────────────────
// Parsed once from the CSV header; reused for every data row.
struct ColIdx {
  int ts=-1, mac=-1, cpu=-1, load=-1;
  int umem=-1, amem=-1, fmem=-1, slab=-1;
  int c2g=-1,  c5g=-1,  c6g=-1;
  int hour=-1, dow=-1;

  // Named-column mode: match header fields by name.
  // Returns true iff the two mandatory columns (timestamp, CMMAC) were found.
  bool Parse(const std::vector<std::string>& cols) {
    std::unordered_map<std::string, int> m;
    for (int i = 0; i < static_cast<int>(cols.size()); ++i) m[cols[i]] = i;
    auto G = [&](const std::string& n) -> int {
      auto it = m.find(n); return it != m.end() ? it->second : -1;
    };
    ts   = G("timestamp");        mac  = G("CMMAC");
    cpu  = G("USED_CPU_ATOM");    load = G("LOAD_AVG_ATOM");
    umem = G("USED_MEM_ATOM_kB"); amem = G("AvailMem_kB");
    fmem = G("FreeMem_kB");       slab = G("SlabMem_kB");
    c2g  = G("2G_Clients_Count"); c5g  = G("5G_Clients_Count");
    c6g  = G("6G_Clients_Count");
    hour = G("hour_of_day");      dow  = G("day_of_week");
    return ts >= 0 && mac >= 0;
  }

  // Positional mode (--no-header): fixed column order written by the
  // on-device telemetry collector into /rdklogs/logs/system_stats_data.csv.
  //
  // col  0 : timestamp
  // col  1 : CMMAC (MAC address)
  // col  2 : USED_CPU_ATOM
  // col  3 : USED_MEM_ATOM_kB
  // col  4 : LOAD_AVG_ATOM
  // col  5 : AvailMem_kB
  // col  6 : FreeMem_kB
  // col  7 : SlabMem_kB
  // col  8 : 2G_Clients_Count
  // col  9 : 5G_Clients_Count
  // col 10 : 6G_Clients_Count
  //
  // hour_of_day / day_of_week are derived from the timestamp field.
  void ParsePositional() {
    ts = 0;  mac  = 1;  cpu  = 2;  umem = 3;
    load = 4; amem = 5; fmem = 6;  slab = 7;
    c2g  = 8; c5g  = 9; c6g  = 10;
    hour = -1; dow = -1;  // always derived from timestamp
  }
};

// ── Parse one CSV row into a TelemetryReading ─────────────────────────────────
static TelemetryReading ParseRow(const std::vector<std::string>& fields,
                                 const std::vector<std::string>& cols,
                                 const ColIdx& ci) {
  auto F = [&](int idx, float def = 0.f) -> float {
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
  (void)cols;  // unused but kept for symmetry with the header vector
  TelemetryReading rdg;
  rdg.timestamp    = S(ci.ts);
  rdg.mac          = S(ci.mac);
  rdg.used_cpu     = F(ci.cpu);
  rdg.load_avg     = F(ci.load);
  rdg.used_mem_kb  = F(ci.umem);
  rdg.avail_mem_kb = F(ci.amem);
  rdg.free_mem_kb  = F(ci.fmem);
  rdg.slab_mem_kb  = F(ci.slab);
  rdg.clients_2g   = I(ci.c2g);
  rdg.clients_5g   = I(ci.c5g);
  rdg.clients_6g   = I(ci.c6g);
  if (ci.hour >= 0 && ci.dow >= 0) {
    rdg.hour_of_day = I(ci.hour);
    rdg.day_of_week = I(ci.dow);
  } else {
    ParseTimestamp(rdg.timestamp, rdg.hour_of_day, rdg.day_of_week);
  }
  return rdg;
}

// ── Anomaly alert printer ─────────────────────────────────────────────────────
// Prints a human-readable alert to stderr whenever an anomaly is detected.
// Called unconditionally (not gated behind --verbose) so alerts are always visible.
//
// Output format:
//   [ANOMALY] <timestamp>  <mac>  Type=<type>
//     CPU  : MSE=<val>  sev=<val>x  <ANOMALY | normal>
//     Mem  : MSE=<val>  sev=<val>x  <ANOMALY | normal>
//     Why  : <explanation of what caused the flag>
static void PrintAlert(const TelemetryReading& rdg, const AnomalyResult& res) {
  if (res.anomaly_type == "Normal") return;

  const bool cpu_flag = (res.dense_cpu_flag != 0);
  const bool mem_flag = (res.dense_mem_flag != 0);

  std::cerr << "\n[ANOMALY] " << rdg.timestamp
            << "  " << rdg.mac
            << "  Type=" << res.anomaly_type << "\n";

  // CPU sub-system line
  std::cerr << "  CPU  : MSE=" << std::fixed << std::setprecision(6) << res.dense_cpu_mse
            << "  sev=" << std::setprecision(2) << res.dense_cpu_sev << "x";
  if (cpu_flag)
    std::cerr << "  <-- ANOMALY (" << std::setprecision(1) << res.dense_cpu_sev << "x worse than normal)";
  std::cerr << "\n";

  // Memory sub-system line
  std::cerr << "  Mem  : MSE=" << std::fixed << std::setprecision(6) << res.dense_mem_mse
            << "  sev=" << std::setprecision(2) << res.dense_mem_sev << "x";
  if (mem_flag)
    std::cerr << "  <-- ANOMALY (" << std::setprecision(1) << res.dense_mem_sev << "x worse than normal)";
  std::cerr << "\n";

  std::cerr << "\n";
}

// ── File-watcher loop ────────────────────────────────────────────────────────
// Monitors watch_path for newly-appended CSV rows via inotify (Linux) or
// stat-polling (non-Linux). Results are appended to result_path.
// Runs until SIGTERM or SIGINT.
//
// skip_existing=true  → seek to EOF on open; ignore rows already in the file.
// skip_existing=false → process all rows present at startup, then watch.
// Header detection is automatic: if the first byte of the file is a digit
// (timestamp) it has no header row and positional column mapping is used.
static int RunDaemon(AnomalyInferenceEngine& engine,
                     const std::string& watch_path,
                     const std::string& result_path,
                     int poll_ms,
                     bool verbose,
                     bool skip_existing) {
  signal(SIGTERM, OnSignal);
  signal(SIGINT,  OnSignal);

  // ── Open result file in append mode ──────────────────────────────────────
  // Do NOT write a header if the file already has content (daemon may restart).
  bool result_has_data = false;
  {
    struct stat st;
    result_has_data = (stat(result_path.c_str(), &st) == 0 && st.st_size > 0);
  }
  std::ofstream fout(result_path, std::ios::app);
  if (!fout) {
    std::cerr << "[daemon] Cannot open result file: " << result_path << "\n";
    return EXIT_FAILURE;
  }
  bool header_written = result_has_data;

  // ── File tracking ─────────────────────────────────────────────────────────
  ColIdx ci;
  std::vector<std::string> cols;
  std::ifstream fin;
  std::streampos last_pos = 0;

  // Open the watch file, parse header columns, position stream.
  // Returns true on success; called again on file rotation / reappearance.
  auto OpenWatchFile = [&]() -> bool {
    fin.close();
    fin.clear();
    fin.open(watch_path);
    if (!fin.is_open()) return false;

    // Auto-detect header: peek at the first byte.
    // A timestamp starts with a digit (device CSV, no header).
    // A named-column header starts with a letter.
    const bool has_named_header =
        fin.good() && std::isalpha(static_cast<unsigned char>(fin.peek()));

    if (has_named_header) {
      std::string hdr;
      if (!std::getline(fin, hdr)) return false;
      cols = SplitCsv(hdr);
      if (!ci.Parse(cols)) {
        std::cerr << "[daemon] Required columns (timestamp, CMMAC) missing in: "
                  << watch_path << "\n";
        fin.close();
        return false;
      }
      last_pos = fin.tellg();  // position after the header line
    } else {
      ci.ParsePositional();
      cols.clear();             // empty cols signals positional mode to DrainFile
      last_pos = fin.tellg();   // position at start of first data row
    }
    return true;
  };

  // Wait until the file appears (the collector may not have started yet)
  std::cerr << "[daemon] Waiting for: " << watch_path << "\n";
  while (!g_stop && !OpenWatchFile())
    usleep(static_cast<useconds_t>(poll_ms) * 1000u);
  if (g_stop) return EXIT_SUCCESS;
  std::cerr << "[daemon] Watching:  " << watch_path << "\n";

  // Optionally skip rows already present at startup
  if (skip_existing) {
    fin.seekg(0, std::ios::end);
    last_pos = fin.tellg();
    std::cerr << "[daemon] Skipping existing rows (--skip-existing).\n";
  }

  // ── Helper: read from last_pos to EOF, run inference, write results ───────
  int total_rows = 0;
  auto DrainFile = [&]() {
    fin.seekg(last_pos);
    fin.clear();
    std::string line;
    while (std::getline(fin, line)) {
      if (line.empty()) continue;
      std::vector<std::string> fields = SplitCsv(line);
      // cols.empty() means positional mode (auto-detected, no header row).
      const int min_f = cols.empty() ? 9 : static_cast<int>(cols.size());
      if (static_cast<int>(fields.size()) < min_f) continue;
      TelemetryReading rdg = ParseRow(fields, cols, ci);
      AnomalyResult    res = engine.ProcessReading(rdg);
      if (!header_written) {
        WriteResultHeader(fout);
        header_written = true;
      }
      WriteResultRow(fout, rdg, res);
      fout.flush();
      PrintAlert(rdg, res);
      if (verbose && res.anomaly_type != "Normal") {
        std::cerr << "[daemon] ALERT "
                  << rdg.mac << " ts=" << rdg.timestamp
                  << " type=" << res.anomaly_type
                  << " cpu_sev=" << std::fixed << std::setprecision(3) << res.dense_cpu_sev
                  << " mem_sev=" << res.dense_mem_sev << "\n";
      }
      ++total_rows;
    }
    last_pos = fin.tellg();
    fin.clear();
  };

  // Process rows already in the file (unless --skip-existing)
  int startup_rows = total_rows;
  DrainFile();
  std::cerr << "[daemon] Processed " << (total_rows - startup_rows)
            << " existing row(s). Waiting for new data...\n";

  // ── inotify setup (Linux) / polling fallback (non-Linux) ─────────────────
  // On Linux the kernel notifies us the instant the file is modified —
  // no busy-waiting, and latency is essentially zero regardless of how
  // infrequently the collector writes.  poll_ms is used only as a safety-net
  // timeout (catches missed events, e.g. from log rotation races).
  // On non-Linux the original stat-based sleep-and-check loop is used.
#ifdef __linux__
  // Decompose watch_path into directory + filename for inotify_add_watch.
  const std::string watch_dir = [&]() -> std::string {
    auto p = watch_path.rfind('/');
    return p == std::string::npos ? "." : watch_path.substr(0, p);
  }();
  const std::string watch_file = [&]() -> std::string {
    auto p = watch_path.rfind('/');
    return p == std::string::npos ? watch_path : watch_path.substr(p + 1);
  }();

  int ifd = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
  int iwd = -1;
  if (ifd >= 0) {
    // Watch the parent directory so we also receive IN_CREATE / IN_MOVED_TO
    // events when the file is replaced by log rotation.
    iwd = inotify_add_watch(ifd, watch_dir.c_str(),
                            IN_MODIFY      |  // data appended (collector keeps fd open)
                            IN_CLOSE_WRITE |  // file closed after write
                            IN_MOVED_TO    |  // file renamed into the directory
                            IN_CREATE      |  // file created in the directory
                            IN_DELETE);       // file removed
    if (iwd < 0) { close(ifd); ifd = -1; }
  }
  const bool use_inotify = (ifd >= 0);
  if (use_inotify)
    std::cerr << "[daemon] inotify active  — waking immediately on file change.\n";
  else
    std::cerr << "[daemon] inotify unavailable — using stat polling (" << poll_ms << " ms).\n";

  // inotify event buffer — sized for several events in one read()
  alignas(inotify_event) char ev_buf[16 * (sizeof(inotify_event) + NAME_MAX + 1)];
#endif  // __linux__

  // ── Main event loop ───────────────────────────────────────────────────────
  while (!g_stop) {
#ifdef __linux__
    if (use_inotify) {
      // Block until the kernel delivers an event or the safety timeout fires.
      struct pollfd pfd = { ifd, POLLIN, 0 };
      int ret = poll(&pfd, 1, poll_ms);
      if (ret < 0) {
        if (errno == EINTR) continue;
        std::cerr << "[daemon] poll(): " << strerror(errno) << "\n";
        break;
      }

      bool reopen = false;
      if (ret > 0 && (pfd.revents & POLLIN)) {
        // Drain the inotify fd — may contain multiple events
        ssize_t n;
        while ((n = read(ifd, ev_buf, sizeof(ev_buf))) > 0) {
          for (ssize_t off = 0; off < n; ) {
            const inotify_event* ev =
                reinterpret_cast<const inotify_event*>(ev_buf + off);
            off += static_cast<ssize_t>(sizeof(inotify_event)) + ev->len;
            const std::string ev_name(ev->len > 0 ? ev->name : "");
            if (ev_name != watch_file) continue;
            if (ev->mask & (IN_MOVED_TO | IN_CREATE | IN_DELETE)) reopen = true;
          }
        }
        // EAGAIN means buffer drained — expected for IN_NONBLOCK
      }
      // ret==0 → safety timeout; always fall through to the stat check below.

      if (reopen) {
        // The file was replaced or removed.  Wait for it to reappear
        // (the next IN_CREATE / IN_MOVED_TO unblocks the next poll()).
        std::cerr << "[daemon] File replaced/removed: " << watch_path << "\n";
        fin.close(); fin.clear();
        while (!g_stop) {
          struct stat tmp;
          if (stat(watch_path.c_str(), &tmp) == 0) break;
          struct pollfd pfd2 = { ifd, POLLIN, 0 };
          poll(&pfd2, 1, poll_ms);  // sleep until next event or timeout
        }
        if (g_stop) break;
        if (!OpenWatchFile()) continue;
        std::cerr << "[daemon] Watching (new file): " << watch_path << "\n";
      }
    } else
#endif  // __linux__
    {
      // Polling fallback (non-Linux)
      usleep(static_cast<useconds_t>(poll_ms) * 1000u);
    }

    // ── Common: stat + rotation check + drain ──────────────────────────────
    struct stat st;
    if (stat(watch_path.c_str(), &st) != 0) {
      // File disappeared
      std::cerr << "[daemon] " << watch_path << " gone. Waiting...\n";
      fin.close(); fin.clear();
      while (!g_stop && stat(watch_path.c_str(), &st) != 0)
        usleep(static_cast<useconds_t>(poll_ms) * 1000u);
      if (g_stop) break;
      if (!OpenWatchFile()) continue;
      std::cerr << "[daemon] File reappeared: " << watch_path << "\n";
      continue;
    }

    const std::streampos cur_size = static_cast<std::streampos>(st.st_size);
    if (cur_size < last_pos) {
      // File was truncated or replaced (log rotation)
      std::cerr << "[daemon] File rotated. Re-opening: " << watch_path << "\n";
      if (!OpenWatchFile()) continue;
    }

    if (cur_size == last_pos) continue;  // no new data yet

    DrainFile();
  }

#ifdef __linux__
  if (ifd >= 0) { inotify_rm_watch(ifd, iwd); close(ifd); }
#endif

  std::cerr << "[daemon] Stopped. Total rows processed: " << total_rows << "\n";
  return EXIT_SUCCESS;
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
    << "  --delegate-path,  -d <path>\n"
    << "                          External hardware delegate .so\n"
    << "                          (e.g. /usr/lib/libbstorm_external_delegate.so)\n"
    << "  --delegate-options <key:val;key:val>\n"
    << "                          Options passed to the delegate\n"
    << "                          (e.g. \"bstm:1;bstm-client-mode:0;dynamic-tensors:1\")\n"
    << "  --verbose,  -v          Print per-reading details to stderr\n"
    << "  --check-libs            Check shared library availability and exit\n"
    << "  --help,     -h          Print this message\n\n"
    << "File-watcher mode (default; runs until SIGTERM/SIGINT, use '&' to background):\n"
    << "  --watch-path  <path>    CSV produced by the data-collection process.\n"
    << "                          (default: /rdklogs/logs/system_stats_data.csv)\n"
    << "  --result-path <path>    Output file; results are appended.\n"
    << "                          (default: /rdklogs/logs/anomaly_results.csv)\n"
    << "  --poll-interval <ms>    On Linux (inotify mode): safety-timeout in ms; wakes\n"
    << "                          even if an inotify event was missed. (default: 60000)\n"
    << "                          On non-Linux: active stat-poll interval in ms.\n"
    << "  --skip-existing         Start from EOF; ignore rows already in watch-path.\n\n"
    << "  CSV header is auto-detected: if the first field starts with a letter it\n"
    << "  is treated as a column-name header row; if it starts with a digit the\n"
    << "  file has no header and this positional order is assumed:\n"
    << "    0:timestamp  1:CMMAC  2:USED_CPU_ATOM  3:USED_MEM_ATOM_kB  4:LOAD_AVG_ATOM\n"
    << "    5:AvailMem_kB  6:FreeMem_kB  7:SlabMem_kB  8:2G  9:5G  10:6G_Clients_Count\n\n";
}

// ── Main ──────────────────────────────────────────────────────────────────────
int Main(int argc, char** argv) {
  std::string config_path     = "inference_config.json";
  std::string input_path;
  std::string output_path;
  std::string delegate_path;
  std::string delegate_options;
  int  num_threads     = 1;
  bool verbose         = false;
  bool check_libs      = false;
  // ── File-watcher options ──────────────────────────────────────────────────
  bool skip_existing    = false;
  std::string watch_path    = "/rdklogs/logs/system_stats_data.csv";
  std::string result_path   = "/rdklogs/logs/anomaly_results.csv";
  int  poll_interval_ms = 60000;  // inotify safety timeout; non-Linux poll interval

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
    } else if ((arg == "--delegate-path" || arg == "-d") && i + 1 < argc) {
      delegate_path = argv[++i];
    } else if (arg == "--delegate-options" && i + 1 < argc) {
      delegate_options = argv[++i];
    } else if (arg == "--verbose" || arg == "-v") {
      verbose = true;
    } else if (arg == "--skip-existing") {
      skip_existing = true;
    } else if (arg == "--watch-path" && i + 1 < argc) {
      watch_path = argv[++i];
    } else if (arg == "--result-path" && i + 1 < argc) {
      result_path = argv[++i];
    } else if (arg == "--poll-interval" && i + 1 < argc) {
      poll_interval_ms = std::atoi(argv[++i]);
      if (poll_interval_ms <= 0) poll_interval_ms = 60000;
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
    std::cerr << "[lib-check] libtensorflowlite.so is required for all models.\n"
              << "[lib-check] Use: ldd ./anomaly_app | grep tflite\n";
    return EXIT_SUCCESS;
  }

  // ── Initialise engine ─────────────────────────────────────────────────────
  std::cerr << "Loading config: " << config_path << "\n";
  AnomalyInferenceEngine engine(config_path, num_threads, delegate_path, delegate_options, verbose);
  std::cerr << "Engine ready.  SEQ_LEN=" << engine.seq_len()
            << "  threads=" << num_threads << "\n";

  // ── Dispatch: watch mode (default) or batch mode (--input) ───────────────
  if (input_path.empty()) {
    std::cerr << "[watcher] watch-path  : " << watch_path      << "\n"
              << "[watcher] result-path : " << result_path     << "\n"
              << "[watcher] timeout(ms) : " << poll_interval_ms
              << " (inotify safety timeout on Linux; poll interval on non-Linux)\n"
              << "[watcher] skip-existing: " << (skip_existing ? "yes" : "no") << "\n";
    return RunDaemon(engine, watch_path, result_path,
                     poll_interval_ms, verbose, skip_existing);
  }

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

  // ── Parse CSV header / column mapping (auto-detect) ─────────────────────
  // Peek at the first byte: a digit means the first row is data (no header);
  // a letter means a named-column header row is present.
  ColIdx batch_ci;
  std::vector<std::string> cols;

  if (!in.good()) {
    std::cerr << "Input is empty.\n";
    return EXIT_FAILURE;
  }
  if (std::isalpha(static_cast<unsigned char>(in.peek()))) {
    std::string header_line;
    std::getline(in, header_line);
    cols = SplitCsv(header_line);
    if (!batch_ci.Parse(cols)) {
      std::cerr << "Cannot find required columns (timestamp, CMMAC) in header.\n";
      return EXIT_FAILURE;
    }
  } else {
    batch_ci.ParsePositional();
    // Stream stays at the first data row; the loop below will consume it.
  }

  // Convenience aliases so the rest of batch mode is unchanged
  const int ci_ts   = batch_ci.ts;   const int ci_mac  = batch_ci.mac;
  const int ci_cpu  = batch_ci.cpu;  const int ci_load = batch_ci.load;
  const int ci_umem = batch_ci.umem; const int ci_amem = batch_ci.amem;
  const int ci_fmem = batch_ci.fmem; const int ci_slab = batch_ci.slab;
  const int ci_2g   = batch_ci.c2g;  const int ci_5g   = batch_ci.c5g;
  const int ci_6g   = batch_ci.c6g;
  const int ci_hour = batch_ci.hour; const int ci_dow  = batch_ci.dow;

  // ── Write output CSV header ───────────────────────────────────────────────
  bool header_written = false;

  auto WriteHeader = [&]() {
    out << "timestamp,CMMAC"
        << ",dense_cpu_mse,dense_mem_mse"
        << ",dense_cpu_flag,dense_mem_flag"
        << ",dense_cpu_sev,dense_mem_sev"
        << ",anomaly_type\n";
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
              << std::setw(12) << "Time(ms)"
              << "Alert\n"
              << std::string(87, '-') << "\n";
  }

  // ── Process rows ──────────────────────────────────────────────────────────
  std::string line;
  int row_num = 0;
  double total_inference_us = 0.0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::vector<std::string> fields = SplitCsv(line);
    // cols.empty() means positional mode (auto-detected, no header row).
    const int min_fields = cols.empty() ? 9 : static_cast<int>(cols.size());
    if (static_cast<int>(fields.size()) < min_fields) continue;

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
    struct timeval t0, t1;
    gettimeofday(&t0, nullptr);
    AnomalyResult res = engine.ProcessReading(rdg);
    gettimeofday(&t1, nullptr);
    const double elapsed_us = get_us(t1) - get_us(t0);
    total_inference_us += elapsed_us;

    // Write CSV header on first result
    if (!header_written) {
      WriteHeader();
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
        << "," << res.dense_mem_sev
        << "," << res.anomaly_type << "\n";

    // ── Anomaly alert (always printed when anomaly detected) ──────────────
    PrintAlert(rdg, res);

    // ── Verbose streaming table ───────────────────────────────────────────
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
                << std::setw(12) << std::setprecision(3) << elapsed_us / 1000.0 << "ms"
                << alert << "\n";
    }
    ++row_num;
  }

  if (!header_written) WriteHeader();  // empty input edge case

  std::cerr << "\nProcessed " << row_num << " readings.\n";
  // if (row_num > 0) {
  //   std::cerr << std::fixed << std::setprecision(3)
  //             << "Total inference time : " << total_inference_us / 1000.0 << " ms\n"
  //             << "Average time/reading : " << total_inference_us / row_num / 1000.0 << " ms\n";
  // }
  return EXIT_SUCCESS;
}

}  // namespace anomaly_detection
}  // namespace tflite

int main(int argc, char** argv) {
  return tflite::anomaly_detection::Main(argc, argv);
}
