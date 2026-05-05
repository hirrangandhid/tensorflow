/*
remediation_agent.cc
────────────────────────────────────────────────────────────────────────────
Bidirectional bridge between on-device MCP-connected apps and the MCP
server over MQTT.  Uses the mcp_rbus_framework to receive alerts from any
number of apps via rbus, then forwards them to the MQTT broker.

Architecture
────────────
  Any on-device app includes mcp_rbus_framework.h and publishes to:
    Device.MCPApps.<app_name>.alert!  (rbus event)

  remediation_agent subscribes to those events, reformats each alert as
  JSON, and publishes it to MQTT:
    anomaly/alerts/<device_id>  (QoS 1)

  The MCP server publishes corrective actions back on:
    anomaly/actions/<device_id>  (QoS 1)

  remediation_agent executes them via WebPA / TR-181.

Build dependencies:
  - librbus        (RDK bus — compile with -DHAVE_RBUS)
  - libmosquitto   (Eclipse Mosquitto MQTT client)
  - libcurl        (WebPA HTTP calls)
  - nlohmann/json  (vendored)

Build WITH rbus (on RDK device):
  g++ -std=c++17 -DHAVE_RBUS -o remediation_agent remediation_agent.cc \
      -lrbus -lmosquitto -lcurl -Inlohmann_json

Build WITHOUT rbus (host/CI testing — no alert delivery):
  g++ -std=c++17 -o remediation_agent remediation_agent.cc \
      -lmosquitto -lcurl -Inlohmann_json

Usage:
  remediation_agent [options]
    --app <name>          rbus app to subscribe to (may repeat)
                          default: anomaly_detection
    --discover            Auto-discover all MCPApps via rbus (rescans every 30 s)
    --broker  <host>      MQTT broker host (default: localhost)
    --port    <port>      MQTT broker port (default: 1883)
    --webpa-url   <url>   WebPA API base URL
    --webpa-token <token> WebPA Bearer token
    --min-severity <n>    Minimum severity to forward (default: 1)
    --verbose             Enable verbose logging
============================================================================*/

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <ctime>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <curl/curl.h>
#include <mosquitto.h>

// rbus IPC framework — all app-to-agent communication goes through here.
// Compile with -DHAVE_RBUS to enable rbus; without it all calls are no-ops.
#include "mcp_rbus_framework.h"

using json = nlohmann::json;

// ── Log helper ────────────────────────────────────────────────────────────────
// Writes a timestamped line to /rdklogs/logs/anomaly_app.txt and mirrors to
// stderr.  Format:  YYYY-MM-DD HH:MM:SS [ra]: <message>
static void ra_log(const std::string& msg) {
  time_t now = time(nullptr);
  struct tm tm_info;
  localtime_r(&now, &tm_info);
  char ts[32];
  strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm_info);
  std::string line = std::string(ts) + " [ra]: " + msg;
  FILE* f = fopen("/rdklogs/logs/anomaly_app.txt", "a");
  if (f) { fprintf(f, "%s\n", line.c_str()); fclose(f); }
  fprintf(stderr, "%s\n", line.c_str());
}

// ── Configuration ─────────────────────────────────────────────────────────────

struct Config {
    std::vector<std::string> app_names = {"anomaly_detection"};
    bool        discover        = false;
    std::string broker_host     = "localhost";
    int         broker_port     = 1883;
    std::string webpa_base_url  = "https://api.webpa.comcast.net/api/v2";
    std::string webpa_token;
    int         min_severity    = 1;    // forward anomalies at this severity+
    bool        verbose         = false;
};

// ── Globals ────────────────────────────────────────────────────────────────────

static std::atomic<bool> g_running{true};
static Config             g_cfg;
static struct mosquitto*  g_mosq = nullptr;

// ── Signal handler ─────────────────────────────────────────────────────────────

void handle_signal(int) { g_running = false; }

// ── Timestamp & alert-id helpers ───────────────────────────────────────────────

static std::string make_alert_id(const std::string& device_id,
                                  const std::string& timestamp) {
    std::string mac_clean, ts_clean;
    for (char c : device_id) if (c != ':') mac_clean += c;
    for (char c : timestamp) if (std::isalnum(c)) ts_clean += c;
    return mac_clean + "_" + ts_clean;
}

// ── MQTT topic helpers ─────────────────────────────────────────────────────────

static std::string mqtt_alert_topic(const std::string& device_id) {
    std::string safe;
    for (char c : device_id) safe += (c == ':' ? '_' : c);
    return "anomaly/alerts/" + safe;
}

// ── Publish alert received from rbus to MQTT ──────────────────────────────────
// Called from the rbus callback (mcp_rbus::Agent subscription) on the rbus
// internal thread.  mosquitto_publish() is thread-safe.

static void publish_rbus_alert(const mcp_rbus::Alert& alert) {
    int max_sev = std::max(alert.cpu_severity, alert.mem_severity);
    if (max_sev < g_cfg.min_severity) return;
    if (std::string(alert.anomaly_type) == "Normal") return;

    // Base MQTT payload from the common alert fields
    json payload = {
        {"alert_id",     make_alert_id(alert.device_id, alert.timestamp)},
        {"app_name",     alert.app_name},
        {"device_id",    alert.device_id},
        {"timestamp",    alert.timestamp},
        {"anomaly_type", alert.anomaly_type},
        {"cpu_severity", alert.cpu_severity},
        {"mem_severity", alert.mem_severity},
    };

    // Merge app-specific fields from payload_json (cpu_mse, cpu_flag, …)
    if (alert.payload_json[0] != '\0') {
        try {
            auto extra = json::parse(alert.payload_json);
            payload.merge_patch(extra);
        } catch (...) { /* non-JSON payload_json — ignore */ }
    }

    std::string topic       = mqtt_alert_topic(alert.device_id);
    std::string payload_str = payload.dump();

    int rc = mosquitto_publish(
        g_mosq, nullptr,
        topic.c_str(),
        static_cast<int>(payload_str.size()),
        payload_str.c_str(),
        1,     // QoS 1 — at-least-once
        false  // retain
    );

    if (rc != MOSQ_ERR_SUCCESS) {
    ra_log("MQTT publish failed: " + std::string(mosquitto_strerror(rc)));
    } else if (g_cfg.verbose) {
        std::ostringstream _oss;
        _oss << "Published alert to " << topic << ": " << payload_str;
        ra_log(_oss.str());
    }
}

// ── WebPA TR-181 ExecuteAction ─────────────────────────────────────────────────

static size_t curl_discard(void*, size_t sz, size_t nm, void*) { return sz * nm; }

static void execute_webpa_action(const std::string& mac,
                                  const std::string& action_id,
                                  const json&        params) {
    // Normalise MAC: ensure "mac:" prefix
    std::string norm_mac = mac;
    if (norm_mac.substr(0, 4) != "mac:")
        norm_mac = "mac:" + norm_mac;

    std::string url = g_cfg.webpa_base_url + "/device/" + norm_mac + "/config";

    json body = {
        {"parameters", json::array({
            {
                {"name",     "Device.X_RDK_AnomalyRemediation.ExecuteAction()"},
                {"dataType", 0},
                {"value",    json({{"action_id", action_id}, {"params", params}}).dump()}
            }
        })},
        {"command", "EXECUTE"}
    };

    std::string body_str = body.dump();

    CURL* curl = curl_easy_init();
    if (!curl) {
        ra_log("curl_easy_init failed");
        return;
    }

    struct curl_slist* headers = nullptr;
    std::string auth_header = "Authorization: Bearer " + g_cfg.webpa_token;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, auth_header.c_str());

    curl_easy_setopt(curl, CURLOPT_URL,            url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,     headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS,     body_str.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE,  (long)body_str.size());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,  curl_discard);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,        30L);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        ra_log("WebPA request failed: " + std::string(curl_easy_strerror(res)));
    } else if (g_cfg.verbose) {
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        std::ostringstream _oss;
        _oss << "WebPA ExecuteAction HTTP " << http_code
             << " action=" << action_id << " device=" << mac;
        ra_log(_oss.str());
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
}

// ── MQTT message callback — receives actions from MCP ──────────────────────────

static void on_message(struct mosquitto*, void*, const struct mosquitto_message* msg) {
    if (!msg || !msg->payload) return;

    std::string topic(msg->topic);
    std::string payload((char*)msg->payload, msg->payloadlen);

    if (g_cfg.verbose) {
        std::ostringstream _oss;
        _oss << "Received on " << topic << ": " << payload;
        ra_log(_oss.str());
    }

    // Only handle action topics
    if (topic.find("anomaly/actions/") != 0) return;

    try {
        json action = json::parse(payload);
        std::string device_id = action.value("device_id", "");
        std::string action_id = action.value("action_id", "");
        json        params    = action.value("params", json::object());
        std::string reason    = action.value("reason", "");

        if (action_id.empty() || device_id.empty()) {
            ra_log("Invalid action payload (missing fields)");
            return;
        }

        { std::ostringstream _oss;
          _oss << "Executing action=" << action_id
               << " device=" << device_id
               << " reason=" << reason;
          ra_log(_oss.str()); }

        execute_webpa_action(device_id, action_id, params);

    } catch (const json::exception& e) {
        ra_log("JSON parse error: " + std::string(e.what()));
    }
}

static void on_connect(struct mosquitto*, void*, int rc) {
    if (rc == 0) {
        ra_log("Connected to MQTT broker");
        mosquitto_subscribe(g_mosq, nullptr, "anomaly/actions/#", 1);
    } else {
        std::ostringstream _oss;
        _oss << "MQTT connect failed rc=" << rc;
        ra_log(_oss.str());
    }
}

// ── CLI argument parser ────────────────────────────────────────────────────────

static void parse_args(int argc, char** argv) {
    // If --app is given at all, clear the default list first
    bool app_flag_seen = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--app") { app_flag_seen = true; break; }
    if (app_flag_seen) g_cfg.app_names.clear();

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if ((arg == "--app")          && i+1 < argc) g_cfg.app_names.push_back(argv[++i]);
        else if (arg == "--discover")               g_cfg.discover = true;
        else if ((arg == "--broker")  && i+1 < argc) g_cfg.broker_host  = argv[++i];
        else if ((arg == "--port")    && i+1 < argc) g_cfg.broker_port  = std::stoi(argv[++i]);
        else if ((arg == "--webpa-url")   && i+1 < argc) g_cfg.webpa_base_url = argv[++i];
        else if ((arg == "--webpa-token") && i+1 < argc) g_cfg.webpa_token    = argv[++i];
        else if ((arg == "--min-severity") && i+1 < argc) g_cfg.min_severity  = std::stoi(argv[++i]);
        else if (arg == "--verbose") g_cfg.verbose = true;
        else if (arg == "--help") {
            std::cout <<
                "Usage: remediation_agent [options]\n"
                "  --app <name>          rbus app to subscribe to (may repeat)\n"
                "  --discover            Auto-discover all MCPApps via rbus\n"
                "  --broker  <host>      MQTT broker (default: localhost)\n"
                "  --port    <port>      MQTT port (default: 1883)\n"
                "  --webpa-url <url>     WebPA base URL\n"
                "  --webpa-token <tok>   WebPA Bearer token\n"
                "  --min-severity <n>    Min severity to forward (1-3)\n"
                "  --verbose             Verbose logging\n";
            exit(0);
        }
    }
}

// ── Main ───────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    parse_args(argc, argv);

    std::signal(SIGINT,  handle_signal);
    std::signal(SIGTERM, handle_signal);

    curl_global_init(CURL_GLOBAL_DEFAULT);
    mosquitto_lib_init();

    // Create MQTT client
    g_mosq = mosquitto_new("remediation_agent", true, nullptr);
    if (!g_mosq) {
    ra_log("Failed to create MQTT client");
    return 1;
    }

    mosquitto_connect_callback_set(g_mosq, on_connect);
    mosquitto_message_callback_set(g_mosq, on_message);

    // Connect to broker
    int rc = mosquitto_connect(g_mosq, g_cfg.broker_host.c_str(),
                                g_cfg.broker_port, 60);
    if (rc != MOSQ_ERR_SUCCESS) {
        std::ostringstream _oss;
        _oss << "Cannot connect to MQTT broker "
             << g_cfg.broker_host << ":" << g_cfg.broker_port
             << " - " << mosquitto_strerror(rc);
        ra_log(_oss.str());
        mosquitto_destroy(g_mosq);
        return 1;
    }

    // Start MQTT network loop in background thread
    mosquitto_loop_start(g_mosq);

    // ── rbus subscription ──────────────────────────────────────────────────────
    // Subscribe to Device.MCPApps.<app_name>.alert! for each configured app.
    // Callbacks arrive on the rbus internal thread and forward to MQTT.
    mcp_rbus::Agent rbus_agent;

    if (rbus_agent.init()) {
        auto alert_handler = [](const mcp_rbus::Alert& alert) {
            publish_rbus_alert(alert);
        };

        if (g_cfg.discover) {
            ra_log("rbus auto-discovery enabled");
            rbus_agent.discover_and_subscribe(alert_handler);
        } else {
            for (const auto& app : g_cfg.app_names)
                rbus_agent.subscribe(app, alert_handler);
        }
    } else {
        ra_log("rbus unavailable - no app alerts will be received via rbus.");
    }

    // ── Main loop ──────────────────────────────────────────────────────────────
    // rbus and MQTT each manage their own background threads.
    ra_log("Running. Press Ctrl-C to stop.");
    while (g_running)
        std::this_thread::sleep_for(std::chrono::seconds(1));

    // ── Cleanup ────────────────────────────────────────────────────────────────
    rbus_agent.close();

    mosquitto_loop_stop(g_mosq, true);
    mosquitto_disconnect(g_mosq);
    mosquitto_destroy(g_mosq);
    mosquitto_lib_cleanup();
    curl_global_cleanup();

    ra_log("Shutdown complete");
    return 0;
}
