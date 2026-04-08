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
  --delegate-path,  -d <path>  Path to external hardware delegate .so
                             (e.g. /usr/lib/libbstorm_external_delegate.so)
  --delegate-options  <str>  Semicolon-separated key:value options for the delegate
                             (e.g. "bstm:1;bstm-client-mode:0;dynamic-tensors:1")
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
#include <sys/time.h>
#include <unordered_map>
#include <vector>

#include <dlfcn.h>          // dlopen / dlsym — for runtime flex delegate check
#include <fcntl.h>          // open, O_RDWR
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

// ── Flex delegate runtime check ───────────────────────────────────────────────
// Call this before loading any model to verify libtensorflowlite_flex.so is
// present and functional on the current device.
//
// Returns true if the Flex delegate can be created successfully.
// Prints a clear diagnostic message either way.
static bool CheckFlexDelegate() {
  std::cerr << "\n[lib-check] Testing Flex delegate (required for LSTM models)...\n";

  // Load the flex delegate SO dynamically; not required at link time.
  void* lib = dlopen("libtensorflowlite_flex.so", RTLD_NOW | RTLD_GLOBAL);
  bool available = false;
  if (lib) {
    using AcquireFn = TfLiteDelegateUniquePtr (*)();
    auto* acquire = reinterpret_cast<AcquireFn>(dlsym(lib, "TF_AcquireFlexDelegate"));
    if (acquire) {
      auto d = acquire();       // destructs at end of this block
      available = (d != nullptr);
    }   // d destructs here — deleter called while lib is still open
    dlclose(lib);               // safe: delegate already destroyed above
  }

  if (!available) {
    std::cerr
      << "[lib-check] FAIL: TfLiteFlexDelegateCreate() returned null.\n"
      << "            Likely cause: libtensorflowlite_flex.so is not in LD_LIBRARY_PATH\n"
      << "            or was not deployed alongside the binary.\n"
      << "            \u2192 LSTM models (lstm_cpu_anomaly_model.tflite,\n"
      << "                           lstm_memory_anomaly_model.tflite) will NOT work.\n"
      << "            \u2192 Dense models (cpu_anomaly_model.tflite,\n"
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

  std::cerr << "[lib-check] OK : Flex delegate available \u2014 LSTM models will work.\n\n";
  return true;
}

// ── CSV result header writer ─────────────────────────────────────────────────
static void WriteResultHeader(std::ostream& out, bool with_lstm) {
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
      << "," << res.dense_mem_sev;
  if (res.has_lstm) {
    out << "," << std::setprecision(6) << res.lstm_cpu_mse
        << "," << res.lstm_mem_mse
        << "," << res.lstm_cpu_flag
        << "," << res.lstm_mem_flag
        << "," << std::setprecision(4) << res.lstm_cpu_sev
        << "," << res.lstm_mem_sev;
  }
  out << "," << res.anomaly_type << "\n";
}

// ── Signal handling ───────────────────────────────────────────────────────────
static volatile sig_atomic_t g_stop = 0;
static void OnSignal(int) { g_stop = 1; }

// ── Daemonize (POSIX double-fork) ─────────────────────────────────────────────
// Forks the process into the background and redirects stdio to /dev/null.
// Optionally writes the daemon PID to pid_file.
static void Daemonize(const std::string& pid_file) {
  // First fork — detach from the controlling terminal
  pid_t pid = fork();
  if (pid < 0) { perror("[daemon] fork");  exit(1); }
  if (pid > 0) exit(0);  // parent exits immediately

  setsid();  // become a session leader with no controlling terminal

  // Second fork — prevent re-acquiring a terminal
  pid = fork();
  if (pid < 0) { perror("[daemon] fork2"); exit(1); }
  if (pid > 0) exit(0);  // first child exits

  // Redirect stdio to /dev/null
  int devnull = open("/dev/null", O_RDWR);
  if (devnull >= 0) {
    dup2(devnull, STDIN_FILENO);
    dup2(devnull, STDOUT_FILENO);
    dup2(devnull, STDERR_FILENO);
    if (devnull > STDERR_FILENO) close(devnull);
  }

  // Write PID file so the init system / watchdog can manage the process
  if (!pid_file.empty()) {
    FILE* pf = fopen(pid_file.c_str(), "w");
    if (pf) { fprintf(pf, "%d\n", static_cast<int>(getpid())); fclose(pf); }
  }
}

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

// ── Daemon watch loop ─────────────────────────────────────────────────────────
// Monitors watch_path for newly-appended CSV rows using stat()-based polling.
// Results are appended to result_path.  Runs until SIGTERM or SIGINT.
//
// skip_existing=true  → seek to EOF on open; ignore rows already in the file.
// skip_existing=false → process all rows present at startup, then watch.
// no_header=true      → file has no column-name header; use positional mapping.
static int RunDaemon(AnomalyInferenceEngine& engine,
                     const std::string& watch_path,
                     const std::string& result_path,
                     int poll_ms,
                     bool verbose,
                     bool skip_existing,
                     bool no_header) {
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

    if (no_header) {
      // Device CSV has no header row — use fixed positional column mapping.
      ci.ParsePositional();
      cols.clear();  // not used in positional mode
      last_pos = fin.tellg();
      return true;
    }

    std::string hdr;
    if (!std::getline(fin, hdr)) return false;
    cols = SplitCsv(hdr);
    if (!ci.Parse(cols)) {
      std::cerr << "[daemon] Required columns (timestamp, CMMAC) missing in: "
                << watch_path << "\n";
      fin.close();
      return false;
    }
    last_pos = fin.tellg();
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
      // Skip rows shorter than the header (partial writes) or, in
      // positional/no-header mode, rows with fewer than 9 fields.
      const int min_f = no_header ? 9 : static_cast<int>(cols.size());
      if (static_cast<int>(fields.size()) < min_f) continue;
      TelemetryReading rdg = ParseRow(fields, cols, ci);
      AnomalyResult    res = engine.ProcessReading(rdg);
      if (!header_written) {
        WriteResultHeader(fout, res.has_lstm);
        header_written = true;
      }
      WriteResultRow(fout, rdg, res);
      fout.flush();
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
    << "Daemon mode (long-running file watcher):\n"
    << "  --daemon                Watch watch-path for new rows; process continuously\n"
    << "                          until SIGTERM/SIGINT.\n"
    << "  --daemonize             Fork to background (implies --daemon);\n"
    << "                          stdio is redirected to /dev/null.\n"
    << "  --watch-path  <path>    CSV produced by the data-collection process.\n"
    << "                          (default: /rdklogs/logs/system_stats_data.csv)\n"
    << "  --result-path <path>    Output file; results are appended.\n"
    << "                          (default: /rdklogs/logs/anomaly_results.csv)\n"
    << "  --poll-interval <ms>    On Linux (inotify mode): safety-timeout in ms; wakes\n"
    << "                          even if an inotify event was missed. (default: 60000)\n"
    << "                          On non-Linux: active stat-poll interval in ms.\n"
    << "  --skip-existing         Start from EOF; ignore rows already in watch-path.\n"
    << "  --pid-file    <path>    Write daemon PID here (--daemonize only).\n"
    << "  --no-header             watch-path has no column-name header row;\n"
    << "                          use fixed positional column order:\n"
    << "                          timestamp,CMMAC,USED_CPU_ATOM,USED_MEM_ATOM_kB,\n"
    << "                          LOAD_AVG_ATOM,AvailMem_kB,FreeMem_kB,SlabMem_kB,\n"
    << "                          2G_Clients_Count,5G_Clients_Count,6G_Clients_Count\n\n";
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
  // ── Daemon mode options ───────────────────────────────────────────────────
  bool daemon_mode     = false;
  bool daemonize       = false;
  bool skip_existing   = false;
  bool no_header       = false;  // device CSV has no column-name header row
  std::string watch_path    = "/rdklogs/logs/system_stats_data.csv";
  std::string result_path   = "/rdklogs/logs/anomaly_results.csv";
  std::string pid_file;
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
    } else if (arg == "--daemon") {
      daemon_mode = true;
    } else if (arg == "--daemonize") {
      daemonize   = true;
      daemon_mode = true;
    } else if (arg == "--skip-existing") {
      skip_existing = true;
    } else if (arg == "--no-header") {
      no_header = true;
    } else if (arg == "--watch-path" && i + 1 < argc) {
      watch_path = argv[++i];
    } else if (arg == "--result-path" && i + 1 < argc) {
      result_path = argv[++i];
    } else if (arg == "--poll-interval" && i + 1 < argc) {
      poll_interval_ms = std::atoi(argv[++i]);
      if (poll_interval_ms <= 0) poll_interval_ms = 60000;
    } else if (arg == "--pid-file" && i + 1 < argc) {
      pid_file = argv[++i];
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
  AnomalyInferenceEngine engine(config_path, num_threads, delegate_path, delegate_options, verbose);
  std::cerr << "Engine ready.  SEQ_LEN=" << engine.seq_len()
            << "  threads=" << num_threads << "\n";

  // ── Daemon mode ────────────────────────────────────────────────────────────
  if (daemon_mode) {
    std::cerr << "[daemon] watch-path  : " << watch_path      << "\n"
              << "[daemon] result-path : " << result_path     << "\n"
              << "[daemon] timeout(ms) : " << poll_interval_ms
              << " (inotify safety timeout on Linux; poll interval on non-Linux)\n"
              << "[daemon] skip-existing: " << (skip_existing ? "yes" : "no") << "\n"
              << "[daemon] no-header    : " << (no_header     ? "yes" : "no") << "\n";
    if (daemonize) {
      std::cerr << "[daemon] Forking to background...\n";
      Daemonize(pid_file);
      // stderr is now /dev/null; all subsequent diagnostics are silent
    }
    return RunDaemon(engine, watch_path, result_path,
                     poll_interval_ms, verbose, skip_existing, no_header);
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

  // ── Parse CSV header / column mapping ────────────────────────────────────
  ColIdx batch_ci;
  std::vector<std::string> cols;

  if (no_header) {
    // Device CSV has no column-name header; use fixed positional mapping.
    batch_ci.ParsePositional();
  } else {
    std::string header_line;
    if (!std::getline(in, header_line)) {
      std::cerr << "Input is empty.\n";
      return EXIT_FAILURE;
    }
    cols = SplitCsv(header_line);
    if (!batch_ci.Parse(cols)) {
      std::cerr << "Cannot find required columns (timestamp, CMMAC) in header.\n"
                << "If the file has no header, use --no-header.\n";
      return EXIT_FAILURE;
    }
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
    // In named-column mode require at least as many fields as the header.
    // In positional (no-header) mode require at least 9 fields (up to 6G col).
    const int min_fields = no_header ? 9
                         : static_cast<int>(cols.size());
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
                << std::setw(12) << std::setprecision(3) << elapsed_us / 1000.0 << "ms"
                << alert << "\n";
    }
    ++row_num;
  }

  if (!header_written) WriteHeader(false);  // empty input edge case

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
