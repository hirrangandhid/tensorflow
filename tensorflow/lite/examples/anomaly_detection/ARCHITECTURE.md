# Anomaly Detection & Remediation — Bidirectional Architecture

## Overview

End-to-end automated anomaly detection and remediation pipeline between
RDK CPE devices and the MCP server.

**On-device IPC**: Apps communicate with `remediation_agent` via **rbus**
(RDK Bus) using a shared header-only framework (`mcp_rbus_framework.h`).
Any app joins the system by including one header and calling three methods.

**Cloud transport**: `remediation_agent` bridges rbus events to **MQTT**,
carrying alerts to the MCP server and actions back to the device.

---

## Architecture Diagram

```
+-----------------------------------------------------------------------------+
|  LAYER 1 — On-Device Apps (CPE)                                             |
|                                                                             |
|  +--------------------+  +--------------------+  +--------------------+   |
|  |  anomaly_detection  |  |  docsis_monitor     |  |  wifi_health       |   |
|  |  (C++ / TFLite)    |  |  (C++)              |  |  (C++)             |   |
|  |                    |  |                    |  |                    |   |
|  |  #include           |  |  #include           |  |  #include           |   |
|  |  mcp_rbus_          |  |  mcp_rbus_          |  |  mcp_rbus_          |   |
|  |  framework.h        |  |  framework.h        |  |  framework.h        |   |
|  |                    |  |                    |  |                    |   |
|  |  App rbus(          |  |  App rbus(          |  |  App rbus(          |   |
|  |   "anomaly_         |  |   "docsis_          |  |   "wifi_health");  |   |
|  |    detection");    |  |    monitor");       |  |  rbus.init();      |   |
|  |  rbus.init();      |  |  rbus.init();      |  |  rbus.publish(a);  |   |
|  |  rbus.publish(a);  |  |  rbus.publish(a);  |  |                    |   |
|  +--------+-----------+  +--------+-----------+  +--------+-----------+   |
|           | rbus event             | rbus event             | rbus event    |
|           | Device.MCPApps.        | Device.MCPApps.        | Device.MCPApps|
|           |  anomaly_detection.    |  docsis_monitor.       |  wifi_health. |
|           |  alert!                |  alert!                |  alert!       |
|           +------------------------+------------------------+               |
|                                            |                                |
|                                            v                                |
|  +---------------------------------------------------------------------+   |
|  |  remediation_agent   (C++ — one instance per gateway)               |   |
|  |                                                                     |   |
|  |  mcp_rbus::Agent agent;                                             |   |
|  |  agent.init();                                                      |   |
|  |  agent.discover_and_subscribe(handler);  <- auto-discovers all apps |   |
|  |    OR                                                               |   |
|  |  agent.subscribe("anomaly_detection", handler);  <- explicit        |   |
|  |  agent.subscribe("docsis_monitor",    handler);                     |   |
|  |                                                                     |   |
|  |  on each alert --> publish to MQTT anomaly/alerts/{device_id}       |   |
|  |  on MQTT action --> WebPA TR-181 ExecuteAction()                    |   |
|  +----------------------------------+----------------------------------+   |
+-------------------------------------|---------------------------------------+
                                      | MQTT  (WAN)
                                      | anomaly/alerts/{device_id}   QoS 1  ^
                                      | anomaly/actions/{device_id}  QoS 1  v
                         +------------+-------------+
                         |      MQTT BROKER          |
                         |  (Mosquitto / AWS IoT)    |
                         +------------+-------------+
                                      | MQTT
+-------------------------------------v---------------------------------------+
|  LAYER 3 — MCP Server  (Python / asyncio)                                   |
|                                                                             |
|  +----------------------------------------------------------------------+  |
|  |  anomaly.py  —  AnomalyToolSet                                       |  |
|  |                                                                      |  |
|  |  MqttAnomalyListener  (background asyncio task)                      |  |
|  |    +- subscribes to  anomaly/alerts/#                                |  |
|  |    +- stores alerts in AlertStore (per-device ring buffer, 100 max)  |  |
|  |    +- calls _decide_action(alert) -> action_id + params              |  |
|  |    +- if ANOMALY_AUTO_REMEDIATE=true                                 |  |
|  |         publishes to anomaly/actions/{device_id}                    |  |
|  |                                                                      |  |
|  |  MCP Tools exposed to LLM / user:                                    |  |
|  |    anomaly_list_alerts           anomaly_get_alert_detail            |  |
|  |    anomaly_get_system_status     anomaly_list_available_actions      |  |
|  |    anomaly_execute_action        anomaly_get_mqtt_alerts  [NEW]      |  |
|  |    anomaly_send_action  [NEW]                                        |  |
|  +----------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------+
```

---

## The Core Problem — Why rbus?

The original design had a single hard-coded path:

```
anomaly_detection  -->  anomaly_results.csv  -->  (inotify)  -->  remediation_agent
```

This had three fundamental limits:

| Problem | Impact |
|---|---|
| File-based IPC | Fragile — race conditions on log rotation, ghost-fd on file deletion |
| One producer only | Only `anomaly_detection` could ever send alerts |
| Tight coupling | Adding any new app required changing `remediation_agent` |

The rbus framework solves all three: it is event-driven, supports any number
of producers, and the agent can auto-discover new apps at runtime.

---

## File-by-File Changes

### 1. `mcp_rbus_framework.h` — NEW (header-only framework)

The shared contract that decouples all apps from `remediation_agent`.

#### `Alert` struct — common alert envelope

```cpp
struct Alert {
    char app_name[64];       // "anomaly_detection", "docsis_monitor", etc.
    char device_id[64];      // MAC: "5C:22:DA:3C:84:63"
    char timestamp[32];      // ISO-8601 or /rdklogs/ format
    char anomaly_type[64];   // "Normal" | "CPU" | "Memory" | "Both"
    int  cpu_severity;       // 0 (none) to 3 (high)
    int  mem_severity;
    char payload_json[4096]; // app-specific fields (e.g. cpu_mse, signal_level)
};
```

`Alert::to_json()` and `Alert::from_json()` serialise/deserialise across the
rbus boundary. The `payload_json` field carries app-specific data that gets
**merged** into the final MQTT JSON, so each app can include custom fields
without changing the agent.

#### `mcp_rbus::App` — publisher side (used by each detection app)

```cpp
mcp_rbus::App rbus("anomaly_detection");
rbus.init();           // registers Device.MCPApps.anomaly_detection.alert!
rbus.publish(alert);   // fires the rbus event; silent no-op if rbus unavailable
```

If compiled without `-DHAVE_RBUS`, every call is a **no-op** — the app
continues working (writing CSV, printing to stderr) with no rbus dependency.

#### `mcp_rbus::Agent` — subscriber side (used only by `remediation_agent`)

```cpp
mcp_rbus::Agent agent;
agent.init();

// Option A — explicit per-app subscription:
agent.subscribe("anomaly_detection", handler);
agent.subscribe("docsis_monitor",    handler);

// Option B — auto-discover all registered MCPApps:
agent.discover_and_subscribe(handler);   // rescans every 30 s
```

`discover_and_subscribe` runs a background thread that calls
`rbus_discoverRegisteredComponents()` every 30 seconds. Any app that starts
**after** the agent is already running is picked up automatically —
**zero changes to `remediation_agent`** for new apps.

**rbus topic naming convention:**

```
Device.MCPApps.<app_name>.alert!
```

| App | rbus Event |
|---|---|
| `anomaly_detection` | `Device.MCPApps.anomaly_detection.alert!` |
| `docsis_monitor` | `Device.MCPApps.docsis_monitor.alert!` |
| `wifi_health` | `Device.MCPApps.wifi_health.alert!` |
| _(any future app)_ | `Device.MCPApps.<name>.alert!` |

**Build flags:**

```cmake
find_package(rbus QUIET)
if(rbus_FOUND)
    target_compile_definitions(my_target PRIVATE HAVE_RBUS)
    target_link_libraries(my_target PRIVATE rbus)
endif()
```

---

### 2. `anomaly_app_main.cc` — MODIFIED (two additions only)

The existing app is unchanged in behaviour. Two minimal additions were made.

**At daemon startup** — initialise rbus (non-fatal if unavailable):

```cpp
// ── rbus integration ─────────────────────────────────────────────────────
mcp_rbus::App rbus_app("anomaly_detection");
rbus_app.init();   // non-fatal — daemon continues normally if rbus unavailable
```

**Inside `DrainFile` lambda** — publish after each non-Normal inference result:

```cpp
if (res.anomaly_type != "Normal") {
    nlohmann::json rbus_payload = {
        {"cpu_mse",  res.dense_cpu_mse},
        {"mem_mse",  res.dense_mem_mse},
        {"cpu_flag", res.dense_cpu_flag != 0},
        {"mem_flag", res.dense_mem_flag != 0},
    };
    mcp_rbus::Alert alert = mcp_rbus::Alert::build(
        "anomaly_detection",
        rdg.mac.c_str(),
        rdg.timestamp.c_str(),
        res.anomaly_type.c_str(),
        static_cast<int>(res.dense_cpu_sev),
        static_cast<int>(res.dense_mem_sev),
        rbus_payload.dump().c_str()
    );
    rbus_app.publish(alert);
}
```

> **CSV writing is unchanged.** `anomaly_results.csv` still gets every row as
> an on-device audit log. rbus is a parallel event path, not a replacement.

---

### 3. `remediation_agent.cc` — REWRITTEN

**Old design** (removed):
`watch_csv()` → inotify on `anomaly_results.csv` → parse CSV rows → publish
to MQTT. Only worked for `anomaly_detection`. Hard-coded path.

**New design**:
`mcp_rbus::Agent` subscribes to rbus events from any number of apps, then
`publish_rbus_alert()` forwards each to MQTT. The main loop does nothing —
rbus and MQTT each manage their own background threads.

#### Main flow

```
main()
  +-- parse_args()                        (--app, --discover, --broker, ...)
  +-- mosquitto_connect() + loop_start()  (MQTT background thread)
  +-- mcp_rbus::Agent agent; agent.init()
  |    +-- --discover: agent.discover_and_subscribe(alert_handler)
  |    +-- default:   agent.subscribe("anomaly_detection", alert_handler)
  |                   (repeat for each --app argument)
  +-- while(g_running) sleep(1)           (main thread waits for SIGINT/SIGTERM)
```

#### rbus --> MQTT bridge: `publish_rbus_alert()`

```
Alert received from rbus
  |
  +-- filter: skip Normal, skip below --min-severity
  |
  +-- build JSON from common Alert fields:
  |    alert_id, app_name, device_id, timestamp,
  |    anomaly_type, cpu_severity, mem_severity
  |
  +-- merge Alert.payload_json into JSON
  |    (adds app-specific fields: cpu_mse, mem_mse, cpu_flag, ...)
  |
  +-- mosquitto_publish( topic = "anomaly/alerts/{device_id}", QoS 1 )
```

#### MQTT --> WebPA bridge: `on_message()` + `execute_webpa_action()`

```
MQTT message received on anomaly/actions/{device_id}
  |
  +-- parse JSON: device_id, action_id, params, reason
  |
  +-- libcurl POST to WebPA:
       URL:  {WEBPA_BASE_URL}/device/mac:{device_id}/config
       Body: { "command": "EXECUTE",
               "parameters": [{
                 "name":  "Device.X_RDK_AnomalyRemediation.ExecuteAction()",
                 "value": "{action_id, params}"
               }] }
```

#### CLI usage

```bash
# Subscribe to specific apps (default: anomaly_detection only)
remediation_agent \
    --app anomaly_detection \
    --app docsis_monitor \
    --broker mqtt.example.com \
    --webpa-url https://api.webpa.comcast.net/api/v2 \
    --webpa-token <token>

# Auto-discover all registered MCPApps (recommended for production)
remediation_agent \
    --discover \
    --broker mqtt.example.com \
    --webpa-url https://api.webpa.comcast.net/api/v2 \
    --webpa-token <token>
```

#### Build commands

```bash
# On RDK device (rbus available):
g++ -std=c++17 -DHAVE_RBUS -o remediation_agent remediation_agent.cc \
    -lrbus -lmosquitto -lcurl -Inlohmann_json

# Host / CI testing (no rbus — MQTT action path still works):
g++ -std=c++17 -o remediation_agent remediation_agent.cc \
    -lmosquitto -lcurl -Inlohmann_json
```

---

### 4. `anomaly.py` — EXTENDED (MCP server side)

The existing 5 WebPA-based tools are unchanged. Three new components were added.

#### `_ACTION_RULES` + `_decide_action()` — rule engine

| anomaly_type | Severity | Action | Params |
|---|---|---|---|
| CPU | 1 (low) | _(no action — below `ANOMALY_MIN_SEVERITY` default of 2)_ | — |
| CPU | 2 (medium) | `drop_caches` | `{}` |
| CPU | 3 (high) | `kill_process` | `{"process": "top_cpu_consumer"}` |
| Memory | 1 | _(no action)_ | — |
| Memory | 2 | `drop_caches` | `{}` |
| Memory | 3 | `restart_service` | `{"service": "memAgent"}` |
| Both | 2 | `drop_caches` | `{}` |
| Both | 3 | `restart_service` | `{"service": "memAgent"}` |

#### `AlertStore` — in-memory per-device ring buffer

```python
class AlertStore:
    _store: dict[str, deque]   # device_id -> deque(maxlen=100)

    def add(alert)             # prepend newest alert
    def get(device_id, limit)  # alerts for one device
    def get_all()              # all devices, sorted by timestamp desc
    def devices()              # list of known device MACs
```

Alerts from all apps land in the same store under the device's MAC.
The `app_name` field in every alert distinguishes the source.

#### `MqttAnomalyListener` — background asyncio task

```
startup
  +-- asyncio task: _listen()
       +-- aiomqtt.Client.connect(broker_host, broker_port)
       +-- subscribe("anomaly/alerts/#", qos=1)
       +-- for each message:
            +-- parse JSON payload
            +-- _alert_store.add(payload)        <- store for MCP tool queries
            +-- if ANOMALY_AUTO_REMEDIATE=true
                 action = _decide_action(payload)
                 publish to anomaly/actions/{device_id}  (QoS 1)

shutdown (FastAPI lifespan)
  +-- task.cancel() -> aiomqtt client disconnects
```

Reconnects with **exponential backoff** (5 s -> 10 s -> ... -> 60 s max) on
broker disconnection.

#### New MCP tools

**`anomaly_get_mqtt_alerts`** — query push-received alerts from `AlertStore`:

```python
# All devices, last 20 alerts
result = await toolset.anomaly_get_mqtt_alerts(limit=20)

# Specific device
result = await toolset.anomaly_get_mqtt_alerts(
    device_id="5C:22:DA:3C:84:63", limit=10
)
# Returns: { alerts: [...], monitored_devices: [...], alert_count: N }
```

**`anomaly_send_action`** — manually dispatch an action via MQTT:

```python
# Use when ANOMALY_AUTO_REMEDIATE=false and user/LLM approves the action
result = await toolset.anomaly_send_action(
    device_id="5C:22:DA:3C:84:63",
    action_id="drop_caches",
    reason="User approved after reviewing CPU alert",
)
# Publishes to: anomaly/actions/5C_22_DA_3C_84_63
```

---

## Data Flow (End-to-End)

### Step 1 — Anomaly Detection (On Device)

```
system_stats_data.csv
        |
        v  (inotify file-watcher)
anomaly_detection (C++ / TFLite)
        |
        +---------> anomaly_results.csv          (audit log — unchanged)
        |
        +---------> rbus_app.publish(alert)
                       |
                       v
             Device.MCPApps.anomaly_detection.alert!   (rbus event)
```

### Step 2 — Alert Forwarding (remediation_agent)

```
Device.MCPApps.anomaly_detection.alert!  -+
Device.MCPApps.docsis_monitor.alert!     -+-- rbus --> remediation_agent
Device.MCPApps.wifi_health.alert!        -+               |
                                                          |  merge payload_json
                                                          |  into MQTT JSON
                                                          v
                                            anomaly/alerts/{device_id}  (MQTT QoS 1)
```

**MQTT Alert Payload:**
```json
{
  "alert_id":    "5C22DA3C8463_20260504T100000",
  "app_name":    "anomaly_detection",
  "device_id":   "5C:22:DA:3C:84:63",
  "timestamp":   "2026-05-04T10:00:00",
  "anomaly_type": "CPU",
  "cpu_severity": 2,
  "mem_severity": 0,
  "cpu_mse":     0.185,
  "mem_mse":     0.042,
  "cpu_flag":    true,
  "mem_flag":    false
}
```

### Step 3 — Action Decision (MCP Server)

```
anomaly/alerts/#  (MQTT)
        |
        v  MqttAnomalyListener._handle_alert()
AlertStore.add(alert)                <- stored for MCP tool queries
        |
        v  _decide_action(alert)
   action = { action_id, params, reason }
        |
        +-- if ANOMALY_AUTO_REMEDIATE=true
               v
     anomaly/actions/{device_id}  (MQTT QoS 1)
```

**MQTT Action Payload:**
```json
{
  "alert_id":  "5C22DA3C8463_20260504T100000",
  "device_id": "5C:22:DA:3C:84:63",
  "action_id": "drop_caches",
  "params":    {},
  "reason":    "CPU anomaly severity 2 — clearing memory caches",
  "timestamp": "2026-05-04T10:00:05Z"
}
```

### Step 4 — Action Execution (remediation_agent -> WebPA)

```
anomaly/actions/{device_id}  (MQTT)
        |
        v  on_message() callback
execute_webpa_action(device_id, action_id, params)
        |
        v  libcurl POST
WebPA /device/mac:{device_id}/config
  -> Device.X_RDK_AnomalyRemediation.ExecuteAction()
```

---

## MQTT Topic Schema

| Topic | Direction | Publisher | Subscriber | QoS |
|---|---|---|---|---|
| `anomaly/alerts/{device_id}` | Device -> Cloud | `remediation_agent` | `anomaly.py` | 1 |
| `anomaly/actions/{device_id}` | Cloud -> Device | `anomaly.py` | `remediation_agent` | 1 |

`{device_id}` uses underscores in place of colons for MQTT topic safety
(e.g. MAC `5C:22:DA:3C:84:63` -> topic suffix `5C_22_DA_3C_84_63`).

---

## Adding a New App — Zero Agent Changes

**Step 1 — In the app's C++ source (one header, three calls):**

```cpp
#include "mcp_rbus_framework.h"

// At startup:
mcp_rbus::App rbus("docsis_monitor");
rbus.init();

// When an anomaly is detected:
mcp_rbus::Alert alert = mcp_rbus::Alert::build(
    "docsis_monitor",
    mac.c_str(), timestamp.c_str(),
    "Signal",                                // anomaly type label
    0,                                       // cpu_severity (N/A for DOCSIS)
    signal_severity,                         // repurpose mem_severity field
    R"({"snr": 12.5, "power": -7.3})"       // app-specific payload
);
rbus.publish(alert);
```

**Step 2 — No changes to `remediation_agent` if using `--discover`:**

The agent's `discover_and_subscribe` thread finds the new
`Device.MCPApps.docsis_monitor.alert!` event within 30 seconds of the app
starting. If using explicit mode, add one flag:

```bash
remediation_agent --app anomaly_detection --app docsis_monitor ...
```

**Step 3 — No changes to `anomaly.py`:**

Alerts from `docsis_monitor` land in the same `AlertStore` under the device's
MAC, distinguished by the `app_name` field. All existing and new MCP tools
work immediately with no code changes.

---

## Environment Variables

```bash
# MQTT Broker
MQTT_BROKER_HOST=localhost          # broker hostname or IP
MQTT_BROKER_PORT=1883               # default port (use 8883 for TLS)
MQTT_USERNAME=                      # optional broker credentials
MQTT_PASSWORD=

# WebPA (existing)
TOOL_WEBPA_BASE_URL=https://api.webpa.comcast.net/api/v2
TOOL_WEBPA_AUTH_TOKEN=<your-token>

# Anomaly tool behaviour
ANOMALY_AUTO_REMEDIATE=false        # true: auto-execute; false: manual via tool
ANOMALY_MIN_SEVERITY=2              # 1=low, 2=medium, 3=high (actions below ignored)
TOOL_ANOMALY_TIMEOUT=30             # WebPA request timeout in seconds
```

---

## File Locations

```
anomaly_detection/
  mcp_rbus_framework.h        <- shared rbus IPC framework (header-only)
  anomaly_app_main.cc         <- detection app (minimal rbus addition)
  remediation_agent.cc        <- rbus subscriber + MQTT + WebPA bridge
  anomaly_detection.cc/.h     <- TFLite inference engine (unchanged)
  ARCHITECTURE.md             <- this document

rdk-mcp-server/
  gateway/tools/anomaly.py    <- MCP toolset (MQTT listener + 2 new tools)
  pyproject.toml              <- aiomqtt added as runtime dependency

/rdklogs/logs/
  system_stats_data.csv       <- input telemetry (unchanged)
  anomaly_results.csv         <- detection output / audit log (unchanged)

/usr/bin/
  anomaly_detection           <- detection binary (existing)
  remediation_agent           <- MQTT bridge binary (new)
```
