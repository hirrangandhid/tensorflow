/*
mcp_rbus_framework.h — MCP rbus IPC Framework for RDK CPE Apps
══════════════════════════════════════════════════════════════════════════════

A lightweight adapter layer between any on-device telemetry/detection app and
the remediation_agent MQTT bridge.  Any app becomes an MCP-connected app by
including this header and calling three functions.

──────────────────────────────────────────────────────────────────────────────
PUBLISHER SIDE — used by each detection/telemetry app
──────────────────────────────────────────────────────────────────────────────

  #include "mcp_rbus_framework.h"

  mcp_rbus::App rbus("anomaly_detection");
  if (!rbus.init()) { } // optional: log warning, continue without rbus

  mcp_rbus::Alert alert = mcp_rbus::Alert::build(
      "anomaly_detection",
      mac.c_str(), timestamp.c_str(),
      "CPU",
      cpu_severity, mem_severity,
      payload_json_str
  );
  rbus.publish(alert);   // no-op if init() failed — safe to call unconditionally
  rbus.close();

──────────────────────────────────────────────────────────────────────────────
SUBSCRIBER SIDE — used by remediation_agent
──────────────────────────────────────────────────────────────────────────────

  #include "mcp_rbus_framework.h"

  mcp_rbus::Agent agent;
  agent.init();

  // Subscribe to a specific app by name
  agent.subscribe("anomaly_detection", [](const mcp_rbus::Alert& a) {
      // forward a to MQTT...
  });

  // OR: discover all registered MCPApps and subscribe dynamically
  agent.discover_and_subscribe([](const mcp_rbus::Alert& a) {
      // unified handler for all apps
  });

  // rbus callbacks arrive on an internal thread — no polling loop needed.
  while (running) std::this_thread::sleep_for(std::chrono::seconds(1));
  agent.close();

──────────────────────────────────────────────────────────────────────────────
rbus TOPIC CONVENTION
──────────────────────────────────────────────────────────────────────────────

  Each app registers and publishes to:
    Device.MCPApps.<app_name>.alert!

  Examples:
    Device.MCPApps.anomaly_detection.alert!
    Device.MCPApps.docsis_monitor.alert!
    Device.MCPApps.wifi_health.alert!

  remediation_agent subscribes to each registered app's event.

──────────────────────────────────────────────────────────────────────────────
BUILD REQUIREMENTS
──────────────────────────────────────────────────────────────────────────────

  Define -DHAVE_RBUS and link against -lrbus to enable rbus IPC.
  Without HAVE_RBUS the framework compiles cleanly but all calls are no-ops.

  CMake:
    find_package(rbus QUIET)
    if(rbus_FOUND)
      target_compile_definitions(my_target PRIVATE HAVE_RBUS)
      target_link_libraries(my_target PRIVATE rbus)
    endif()

══════════════════════════════════════════════════════════════════════════════
*/

#pragma once

// These headers are used in all configurations.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

// <atomic>, <mutex>, <thread> are only needed when rbus is compiled in.
// Pulling them in unconditionally can break cross-compilation sysroots that
// require -pthread to be set explicitly for the threading runtime.
#ifdef HAVE_RBUS
#  include <atomic>
#  include <mutex>
#  include <thread>
#endif

#include "nlohmann_json/json.hpp"

#ifdef HAVE_RBUS
#  include <rbus/rbus.h>
#endif

namespace mcp_rbus {

// ── rbus element name helper ──────────────────────────────────────────────────

inline std::string event_name_for(const std::string& app_name) {
    return "Device.MCPApps." + app_name + ".alert!";
}

// ─────────────────────────────────────────────────────────────────────────────
// Alert — the common data envelope carried over rbus between apps and
// the remediation_agent.
//
// Fixed-size fields make the struct usable as a plain C struct as well.
// The payload_json field carries app-specific key/value pairs for the
// receiving side to forward verbatim to the MQTT broker.
// ─────────────────────────────────────────────────────────────────────────────

struct Alert {
    char app_name[64];       // e.g. "anomaly_detection"
    char device_id[64];      // CMMAC e.g. "5C:22:DA:3C:84:63"
    char timestamp[32];      // ISO-8601 or /rdklogs/ format
    char anomaly_type[64];   // "Normal" | "CPU" | "Memory" | "Both"
    int  cpu_severity;       // integer severity 0–3 (0 = none)
    int  mem_severity;
    char payload_json[4096]; // app-specific JSON string

    // Convenience factory — safer than direct struct initialisation.
    static Alert build(const char* app,
                       const char* device_id,
                       const char* timestamp,
                       const char* anomaly_type,
                       int         cpu_severity,
                       int         mem_severity,
                       const char* payload_json = nullptr) {
        Alert a{};
        std::strncpy(a.app_name,    app,          sizeof(a.app_name) - 1);
        std::strncpy(a.device_id,   device_id,    sizeof(a.device_id) - 1);
        std::strncpy(a.timestamp,   timestamp,    sizeof(a.timestamp) - 1);
        std::strncpy(a.anomaly_type, anomaly_type, sizeof(a.anomaly_type) - 1);
        a.cpu_severity = cpu_severity;
        a.mem_severity = mem_severity;
        if (payload_json)
            std::strncpy(a.payload_json, payload_json, sizeof(a.payload_json) - 1);
        return a;
    }

    // Serialise to a JSON string for rbus transport.
    std::string to_json() const {
        nlohmann::json j = {
            {"app_name",    app_name},
            {"device_id",   device_id},
            {"timestamp",   timestamp},
            {"anomaly_type", anomaly_type},
            {"cpu_severity", cpu_severity},
            {"mem_severity", mem_severity},
        };
        // Merge payload_json as a sub-object if it is valid JSON,
        // otherwise store it as a plain string.
        if (payload_json[0] != '\0') {
            try {
                j["payload"] = nlohmann::json::parse(payload_json);
            } catch (...) {
                j["payload"] = payload_json;
            }
        }
        return j.dump();
    }

    // Deserialise from a JSON string (used by remediation_agent).
    static Alert from_json(const std::string& s) {
        Alert a{};
        try {
            auto j = nlohmann::json::parse(s);
            auto cpy = [](char* dst, size_t n, const std::string& src) {
                std::strncpy(dst, src.c_str(), n - 1);
            };
            cpy(a.app_name,    sizeof(a.app_name),    j.value("app_name",    ""));
            cpy(a.device_id,   sizeof(a.device_id),   j.value("device_id",   ""));
            cpy(a.timestamp,   sizeof(a.timestamp),   j.value("timestamp",   ""));
            cpy(a.anomaly_type, sizeof(a.anomaly_type), j.value("anomaly_type", "Normal"));
            a.cpu_severity = j.value("cpu_severity", 0);
            a.mem_severity = j.value("mem_severity", 0);
            // Re-serialise the payload sub-object back to a string.
            if (j.contains("payload")) {
                std::string p = j["payload"].is_string()
                    ? j["payload"].get<std::string>()
                    : j["payload"].dump();
                std::strncpy(a.payload_json, p.c_str(), sizeof(a.payload_json) - 1);
            }
        } catch (...) { /* return zeroed alert */ }
        return a;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// AlertCallback — function type passed to Agent::subscribe()
// ─────────────────────────────────────────────────────────────────────────────

using AlertCallback = std::function<void(const Alert&)>;

// ─────────────────────────────────────────────────────────────────────────────
// App — publisher side
//
// One instance per process.  Create in main(), call init(), publish alerts,
// call close() on shutdown (or let RAII handle it).
// ─────────────────────────────────────────────────────────────────────────────

class App {
public:
    explicit App(std::string app_name)
        : _app_name(std::move(app_name)),
          _event_name(event_name_for(_app_name)) {}

    // Returns true if rbus is available and the event element was registered.
    // Returns false if rbus is unavailable (no-op mode) — caller continues
    // normally without rbus.
    bool init() {
#ifdef HAVE_RBUS
        rbusError_t rc = rbus_open(&_handle, _app_name.c_str());
        if (rc != RBUS_ERROR_SUCCESS) {
            std::fprintf(stderr,
                "[mcp_rbus] App(%s): rbus_open failed rc=%d\n",
                _app_name.c_str(), rc);
            return false;
        }

        rbusDataElement_t el;
        std::memset(&el, 0, sizeof(el));
        // rbusDataElement_t::name is char* (non-const) in the RDK rbus headers.
        el.name = const_cast<char*>(_event_name.c_str());
        el.type = RBUS_ELEMENT_TYPE_EVENT;
        rc = rbus_regDataElements(_handle, 1, &el);
        if (rc != RBUS_ERROR_SUCCESS) {
            std::fprintf(stderr,
                "[mcp_rbus] App(%s): rbus_regDataElements failed rc=%d\n",
                _app_name.c_str(), rc);
            rbus_close(_handle);
            return false;
        }

        _ready = true;
        std::fprintf(stdout,
            "[mcp_rbus] App(%s): registered event %s\n",
            _app_name.c_str(), _event_name.c_str());
        return true;
#else
        std::fprintf(stderr,
            "[mcp_rbus] App(%s): rbus not available (HAVE_RBUS not defined). "
            "Alerts will not be forwarded to remediation_agent.\n",
            _app_name.c_str());
        return false;
#endif
    }

    // Publish an alert.  Safe to call even if init() returned false — it
    // will be silently discarded.
    bool publish(const Alert& alert) {
#ifdef HAVE_RBUS
        if (!_ready) return false;

        std::string payload = alert.to_json();

        rbusValue_t val;
        rbusValue_Init(&val);
        rbusValue_SetString(val, payload.c_str());

        rbusObject_t data;
        rbusObject_Init(&data, nullptr);
        rbusObject_SetValue(data, "payload", val);

        rbusEvent_t ev;
        std::memset(&ev, 0, sizeof(ev));
        ev.name = _event_name.c_str();
        ev.data = data;
        ev.type = RBUS_EVENT_GENERAL;

        rbusError_t rc = rbusEvent_Publish(_handle, &ev);

        rbusValue_Release(val);
        rbusObject_Release(data);

        if (rc != RBUS_ERROR_SUCCESS) {
            std::fprintf(stderr,
                "[mcp_rbus] App(%s): publish failed rc=%d\n",
                _app_name.c_str(), rc);
            return false;
        }
        return true;
#else
        (void)alert;
        return false;
#endif
    }

    void close() {
#ifdef HAVE_RBUS
        if (_ready) {
            rbus_close(_handle);
            _ready = false;
        }
#endif
    }

    ~App() { close(); }

    const std::string& app_name()   const { return _app_name;   }
    const std::string& event_name() const { return _event_name; }

private:
    std::string  _app_name;
    std::string  _event_name;
    bool         _ready = false;
#ifdef HAVE_RBUS
    rbusHandle_t _handle{};
#endif
};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers for the Agent subscriber — only compiled when rbus is
// available.
// ─────────────────────────────────────────────────────────────────────────────

#ifdef HAVE_RBUS

struct _SubCtx {
    AlertCallback callback;
};

// rbus delivers subscription callbacks on its own internal thread.
inline void _rbus_event_handler(rbusHandle_t,
                                rbusEvent_t const* event,
                                rbusEventSubscription_t* sub) {
    if (!event || !sub || !sub->userData) return;
    auto* ctx = static_cast<_SubCtx*>(sub->userData);

    rbusValue_t val = rbusObject_GetValue(event->data, "payload");
    if (!val) return;

    int len = 0;
    const char* json_str = rbusValue_GetString(val, &len);
    if (!json_str || len == 0) return;

    Alert alert = Alert::from_json(std::string(json_str, len));
    ctx->callback(alert);
}

#endif  // HAVE_RBUS

// ─────────────────────────────────────────────────────────────────────────────
// Agent — subscriber side
//
// One instance in remediation_agent.  Call init(), then subscribe() for each
// known app (or discover_and_subscribe() for automatic discovery).
// rbus delivers callbacks on an internal thread — no polling loop is needed.
// ─────────────────────────────────────────────────────────────────────────────

class Agent {
public:
    Agent() = default;

    bool init() {
#ifdef HAVE_RBUS
        rbusError_t rc = rbus_open(&_handle, "remediation_agent");
        if (rc != RBUS_ERROR_SUCCESS) {
            std::fprintf(stderr,
                "[mcp_rbus] Agent: rbus_open failed rc=%d\n", rc);
            return false;
        }
        _ready = true;
        std::fprintf(stdout, "[mcp_rbus] Agent: rbus connected\n");
        return true;
#else
        std::fprintf(stderr,
            "[mcp_rbus] Agent: rbus not available (HAVE_RBUS not defined).\n");
        return false;
#endif
    }

    // Subscribe to alerts from a specific named app.
    // The callback is invoked on the rbus internal thread.
    bool subscribe(const std::string& app_name, AlertCallback cb) {
#ifdef HAVE_RBUS
        if (!_ready) return false;

        std::string ev = event_name_for(app_name);
        auto* ctx = new _SubCtx{std::move(cb)};

        rbusError_t rc = rbusEvent_Subscribe(
            _handle, ev.c_str(), _rbus_event_handler, ctx,
            0  // timeout: 0 = wait indefinitely until provider registers
        );
        if (rc != RBUS_ERROR_SUCCESS) {
            std::fprintf(stderr,
                "[mcp_rbus] Agent: subscribe(%s) failed rc=%d\n",
                ev.c_str(), rc);
            delete ctx;
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(_mu);
            _subs[ev] = ctx;
        }
        std::fprintf(stdout,
            "[mcp_rbus] Agent: subscribed to %s\n", ev.c_str());
        return true;
#else
        (void)app_name; (void)cb;
        return false;
#endif
    }

    // Discover all currently registered MCPApps via rbus component discovery
    // and subscribe to each one.  Runs the discovery in a background thread
    // that re-scans every `interval` seconds to pick up apps that start later.
    void discover_and_subscribe(AlertCallback cb,
                                std::chrono::seconds interval = std::chrono::seconds(30)) {
#ifdef HAVE_RBUS
        if (!_ready) return;
        _discover_cb = std::move(cb);
        _discover_interval = interval;
        _discover_stop = false;
        _discover_thread = std::thread([this]() {
            while (!_discover_stop) {
                _run_discovery();
                for (int i = 0; i < _discover_interval.count() && !_discover_stop; ++i)
                    std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });
#else
        (void)cb; (void)interval;
#endif
    }

    void close() {
#ifdef HAVE_RBUS
        // Stop discovery thread first
        _discover_stop = true;
        if (_discover_thread.joinable())
            _discover_thread.join();

        if (_ready) {
            std::lock_guard<std::mutex> lock(_mu);
            for (auto& [ev, ctx] : _subs) {
                rbusEvent_Unsubscribe(_handle, ev.c_str());
                delete ctx;
            }
            _subs.clear();
            rbus_close(_handle);
            _ready = false;
        }
#endif
    }

    ~Agent() { close(); }

private:
#ifdef HAVE_RBUS
    void _run_discovery() {
        // rbus_discoverComponentDataElements with partial path "Device.MCPApps."
        // returns all registered elements under that subtree (nextLevel=false),
        // e.g. "Device.MCPApps.anomaly_detection.alert!".
        // This is the correct RDK rbus API per rbus.h @Discovery group.
        int    num_elements  = 0;
        char** element_names = nullptr;

        rbusError_t rc = rbus_discoverComponentDataElements(
            _handle, "Device.MCPApps.", /*nextLevel=*/false,
            &num_elements, &element_names);
        if (rc != RBUS_ERROR_SUCCESS || !element_names) return;

        const std::string prefix = "Device.MCPApps.";
        const std::string suffix = ".alert!";

        for (int i = 0; i < num_elements; ++i) {
            if (!element_names[i]) continue;
            std::string name(element_names[i]);
            free(element_names[i]);

            // Only handle elements that end with ".alert!"
            if (name.size() <= prefix.size() + suffix.size()) continue;
            if (name.substr(name.size() - suffix.size()) != suffix) continue;

            // Extract app_name from "Device.MCPApps.<app_name>.alert!"
            std::string inner = name.substr(prefix.size(),
                name.size() - prefix.size() - suffix.size());

            // Subscribe if not already subscribed
            std::lock_guard<std::mutex> lock(_mu);
            std::string ev = event_name_for(inner);
            if (_subs.find(ev) == _subs.end()) {
                AlertCallback cb_copy = _discover_cb;
                auto* ctx = new _SubCtx{std::move(cb_copy)};
                rbusError_t src = rbusEvent_Subscribe(
                    _handle, ev.c_str(), _rbus_event_handler, ctx, 0);
                if (src == RBUS_ERROR_SUCCESS) {
                    _subs[ev] = ctx;
                    std::fprintf(stdout,
                        "[mcp_rbus] Agent: auto-subscribed to %s\n", ev.c_str());
                } else {
                    delete ctx;
                }
            }
        }
        free(element_names);
    }

    rbusHandle_t _handle{};
    bool         _ready = false;
    std::mutex   _mu;
    std::unordered_map<std::string, _SubCtx*> _subs;

    // Discovery fields
    AlertCallback          _discover_cb;
    std::chrono::seconds   _discover_interval{30};
    std::atomic<bool>      _discover_stop{false};
    std::thread            _discover_thread;
#endif  // HAVE_RBUS
};

}  // namespace mcp_rbus
