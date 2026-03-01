# Vision Labs — Architecture Reference

> **Last updated:** Feb 28, 2026

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Service Map](#service-map)
3. [Data Flow Pipelines](#data-flow-pipelines)
4. [Redis Schema](#redis-schema)
5. [Dashboard Backend](#dashboard-backend)
6. [Dashboard Frontend](#dashboard-frontend)
7. [Tracker Service](#tracker-service)
8. [Face Recognition](#face-recognition)
9. [AI Assistant](#ai-assistant)
10. [Image Generation](#image-generation)
11. [Notification System](#notification-system)
12. [Telegram Bot](#telegram-bot)
13. [DVR Recording](#dvr-recording)
14. [Zone System](#zone-system)
15. [Monitoring Stack](#monitoring-stack)
16. [Shared Contracts](#shared-contracts)
17. [Authentication](#authentication)
18. [NAS Storage Layout](#nas-storage-layout)
19. [Docker Infrastructure](#docker-infrastructure)
20. [File Index](#file-index)

---

## System Overview

Vision Labs is an **event-driven microservice system** running on a single machine (RTX 3090) via Docker Compose. A Reolink PoE camera provides an RTSP video feed that flows through a pipeline of AI models, with results displayed in a web dashboard and sent as Telegram notifications.

```
┌──────────────────────────────────────────────────────────────────────┐
│  PC (RTX 3090) — Everything runs here via Docker Compose             │
│                                                                      │
│  ┌─────────────────┐                                                 │
│  │  Camera (RTSP)  │──(sub-stream 640×480)──▶ Ingester ──▶ Redis     │
│  │  Reolink PoE    │──(main-stream HD)──────▶ Ingester ──▶ Redis     │
│  └─────────────────┘                                                 │
│                                                                      │
│  Redis Streams ──▶ Pose Detector (YOLOv8s-pose, GPU)                 │
│                ──▶ Vehicle Detector (YOLOv8s, GPU)                   │
│                ──▶ Face Recognizer (InsightFace, GPU)                 │
│                         │                                            │
│                         ▼                                            │
│                    Tracker (CPU) ──▶ Events Stream                   │
│                                                                      │
│           ┌───────────────────────────────┐                          │
│           │         Dashboard             │                          │
│           │  FastAPI :8080                 │                          │
│           │  WebSocket /ws (live frames)   │                          │
│           │  REST API /api/*              │                          │
│           │  Static frontend              │──▶ Telegram Bot API      │
│           │  Background pollers           │                          │
│           └───────────────────────────────┘                          │
│                    │              │                                   │
│            Ollama (LLM)    ComfyUI (SDXL)                           │
│            Qwen 3 14B      Image generation                         │
│            MiniCPM-V                                                 │
│                                                                      │
│  Recorder ──(ffmpeg copy)──▶ QNAP NAS (DVR segments)                │
│  Prometheus + Grafana ──▶ System monitoring                          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Service Map

| Service | Container | Port | GPU | Description |
|---------|-----------|------|:---:|-------------|
| **redis** | redis:7-alpine | 6379 | — | Central message bus, AOF persistence, 2GB maxmemory |
| **camera-ingester** | custom | host | — | RTSP→JPEG frames, publishes sub-stream + HD to Redis |
| **pose-detector** | custom | — | ✅ | YOLOv8s-pose on GPU, publishes person bboxes + keypoints |
| **vehicle-detector** | custom | — | ✅ | YOLOv8s on GPU, publishes vehicle bboxes |
| **tracker** | custom | — | — | IoU-based person/vehicle tracking, event publishing |
| **face-recognizer** | custom | 8081 | ✅ | InsightFace embedding, SQLite DB, REST API |
| **dashboard** | custom | 8080 | — | FastAPI + WebSocket + static files + background tasks |
| **ollama** | ollama/ollama | 11434 | ✅ | LLM inference (Qwen 3 14B, MiniCPM-V) |
| **comfyui** | custom | 8188 | ✅ | Stable Diffusion image generation |
| **recorder** | custom | host | — | RTSP→MP4 ffmpeg copy, 1-hour segments |
| **prometheus** | prom/prometheus | 9090 | — | Metrics collection, 30d retention |
| **grafana** | grafana/grafana-oss | 3000 | — | Monitoring dashboards |
| **redis-exporter** | redis_exporter | host | — | Redis metrics → Prometheus |
| **dcgm-exporter** | dcgm-exporter | host | ✅ | NVIDIA GPU metrics → Prometheus |

---

## Data Flow Pipelines

### Detection Pipeline (real-time, ~15 FPS)

```
1. camera-ingester:
   - OpenCV reads RTSP sub-stream (640×480, 15 FPS)
   - JPEG encode each frame
   - XADD to frames:front_door (capped at 1000 entries)
   - Separate thread: reads main stream, SET to frame_hd:front_door

2. pose-detector (consumer group "detectors"):
   - XREADGROUP from frames stream
   - YOLOv8s-pose inference (~34ms on RTX 3090)
   - Filter: person class only, confidence > threshold
   - SET detection_frame:pose:front_door = the frame JPEG used
   - XADD detections:pose:front_door {detections: JSON, inference_ms}

3. vehicle-detector (consumer group "vehicle_detectors"):
   - Same pattern as pose-detector
   - Classes: car, truck, bus, motorcycle
   - SET detection_frame:vehicle:front_door = frame used
   - XADD detections:vehicle:front_door

4. tracker (consumer group "trackers"):
   - XREADGROUP from detections:pose + detections:vehicle
   - IoU matching against tracked persons/vehicles
   - If new person: debounce, then emit person_appeared
   - If person gone > lost_timeout: emit person_left
   - If face recognized: emit person_identified
   - If vehicle stationary > idle_timeout: emit vehicle_idle
   - HSET state:front_door with current scene snapshot
   - XADD events:front_door {event_type, person_id, bbox, zone, ...}
   - SET person_snapshot:{camera}:{ts} = frame JPEG at detection time (2h TTL)

5. face-recognizer:
   - XREADGROUP from frames stream
   - InsightFace detection + embedding extraction
   - Compare against SQLite DB of enrolled faces
   - HSET identity_state:front_door with matches
   - XADD identities:front_door

6. dashboard WebSocket (/ws):
   - Read latest frame from detection_frame:pose:front_door
   - Read state:front_door for current persons/vehicles
   - Read identity_state:front_door for face labels
   - Read zones:front_door for zone overlays
   - Draw bounding boxes, keypoints, face labels, zone overlays
   - JPEG encode → base64 → send JSON via WebSocket
   - Target: 10 FPS to browser
```

### Event Notification Flow

```
1. tracker emits event to events:front_door
2. dashboard _event_notification_poller (background thread):
   - XREAD events stream (blocking, in threadpool executor)
   - For each event:
     a. Save snapshot JPEG to /data/snapshots/ (always)
     b. Journal event to /data/events/YYYY-MM-DD.jsonl (always)
     c. If Telegram configured and not rate-limited:
        - Draw bbox on HD frame
        - Run MiniCPM-V scene description
        - Build caption (event type, time, zone, AI description)
        - Broadcast photo to all approved Telegram users
```

### Face Recognition Flow

```
1. face-recognizer runs InsightFace on each frame
2. For each detected face: extract 512-dim embedding
3. Compare against all enrolled faces in SQLite (cosine similarity)
4. If similarity > 0.5: publish identity to identity_state:{camera}
5. Tracker reads identity_state, links name to tracked person_id
6. Dashboard reads identity_state, draws name labels on bboxes
7. Sticky identity: once recognized, name persists even when face turns away
   (10-frame vote buffer with 2x bias for current identity prevents flicker)
```

### Face Enrollment Flow

```
1. User types name in dashboard wizard, clicks Capture
2. Browser → POST /api/faces/enroll {name}
3. Dashboard proxies to face-recognizer:8081/api/faces/enroll
4. Face-recognizer:
   - XREVRANGE frames (latest 1) → full frame
   - Pick largest person bbox from current detections
   - Crop upper 50% → InsightFace → embedding + portrait thumbnail
   - INSERT into SQLite: {name, embedding_blob, portrait_jpeg}
   - Return success + face_id
```

---

## Redis Schema

### Streams

| Key Pattern | Producer | Consumer | Payload |
|-------------|----------|----------|---------|
| `frames:{camera_id}` | camera-ingester | pose-detector, vehicle-detector, face-recognizer | `{frame, timestamp, frame_number, width, height}` |
| `detections:pose:{camera_id}` | pose-detector | tracker | `{detections: JSON, inference_ms, timestamp}` |
| `detections:vehicle:{camera_id}` | vehicle-detector | tracker | `{detections: JSON, inference_ms, timestamp}` |
| `events:{camera_id}` | tracker | dashboard poller | `{event_type, person_id, bbox, zone, alert_level, ...}` |
| `identities:{camera_id}` | face-recognizer | dashboard | `{identities: JSON}` |
| `telegram:access_log` | bot_commands | dashboard (Telegram page) | `{user_id, username, action, authorized, timestamp}` |

### Keys (state)

| Key Pattern | Writer | Reader | Content |
|-------------|--------|--------|---------|
| `state:{camera_id}` | tracker | dashboard WebSocket | `{num_people, people: JSON[{person_id, bbox, action}]}` |
| `identity_state:{camera_id}` | face-recognizer | dashboard WebSocket | `{face_id: {name, confidence, bbox}}` |
| `config:{camera_id}` | dashboard settings | pose-detector, tracker, vehicle-detector | `{confidence_thresh, iou_threshold, lost_timeout, ...}` |
| `zones:{camera_id}` | dashboard zone editor | tracker, dashboard overlay | `{zone_id: JSON{name, points, alert_level}}` |
| `frame_hd:{camera_id}` | camera-ingester | dashboard (snapshots) | Raw JPEG bytes (HD frame) |
| `detection_frame:{type}:{camera_id}` | pose/vehicle detector | dashboard WebSocket | Raw JPEG bytes (the frame bboxes were computed from) |
| `person_snapshot:{camera_id}:{ts}` | tracker | dashboard event feed | Raw JPEG bytes (2-hour TTL) |
| `vehicle_snapshot:{camera_id}:{ts}` | tracker | dashboard browse/events | Raw JPEG bytes (24-hour TTL) |
| `gpu:generation_active` | image_gen | pose-detector, vehicle-detector | Lock flag — detectors pause GPU when present |
| `telegram:users` | dashboard (Telegram Access Manager) | bot_commands | `{user_id: JSON{chat_id, name, role, approved_at}}` |

### Config Keys (in `config:{camera_id}`)

| Field | Default | Description |
|-------|---------|-------------|
| `confidence_thresh` | `0.5` | YOLO detection confidence minimum |
| `iou_threshold` | `0.3` | IoU overlap threshold for tracking |
| `lost_timeout` | `5.0` | Seconds before marking person as left |
| `target_fps` | `5` | Target FPS for WebSocket streaming |
| `notify_person` | `1` | Enable person detection notifications |
| `notify_vehicle` | `1` | Enable vehicle detection notifications |
| `suppress_known` | `0` | Suppress notifications for recognized people |
| `notify_cooldown` | `60` | Seconds between person notifications |
| `vehicle_cooldown` | `60` | Seconds between vehicle notifications |
| `vehicle_confidence_thresh` | `0.35` | Vehicle detection confidence minimum |
| `vehicle_idle_timeout` | `90` | Seconds before vehicle idle alert |

---

## Dashboard Backend

### `server.py` (1178 lines)

The main FastAPI application:

| Component | Lines | Purpose |
|-----------|-------|---------|
| `auth_middleware` | 240-261 | Session-based auth, redirects unauthenticated to login |
| `login_background` | 267-299 | Heavily blurred camera snapshot for login page (no auth) |
| `startup` | 305-346 | Init auth DB, write default config, start background tasks |
| `_reminder_poller` | 349-386 | Check due reminders every 60s, send via Telegram |
| `_ensure_ollama_model` | 389-448 | Pull Qwen 3 14B on first startup, warm-up GPU load |
| `_clear_comfyui_queue_on_startup` | 451-486 | Clear stale GPU locks from previous session |
| `_event_notification_poller` | 488-766 | Poll events, save snapshots, journal, send Telegram |
| `websocket_live` | 772-1162 | Stream frames with overlays at 10 FPS |

### Route Modules (20 files in `routes/`)

| Module | Prefix | Purpose |
|--------|--------|---------|
| `ai.py` | `/api/ai` | Chat, vision analysis, history, reminders, model status |
| `ai_tools.py` | — | 18 LLM tool definitions + executor functions |
| `ai_prompts.py` | — | System prompt builder with live context |
| `ai_state.py` | — | Shared AI state (DB refs, GPU flag, pending media) |
| `notifications.py` | `/api` | Telegram API helpers, scene analysis, snapshot drawing |
| `bot_commands.py` | — | Telegram polling loop, 11 command handlers, audit logging |
| `image_gen.py` | `/api/generate` | ComfyUI proxy, txt2img, img2img, gallery, prompt history |
| `recordings.py` | `/api/recordings` | DVR playback — list dates, segments, stream video |
| `events.py` | `/api/events` | Event feed retrieval from Redis stream |
| `config.py` | `/api/config` | Read/write detection config to Redis |
| `zones.py` | `/api/zones` | Zone CRUD (create, update, delete, list) |
| `faces.py` | `/api/faces` | Face enrollment proxy to face-recognizer service |
| `unknowns.py` | `/api/unknowns` | Unknown face management (list, label, delete) |
| `browse.py` | `/api/browse` | Vehicle snapshot browser, enrolled faces gallery |
| `clips.py` | `/api/clips` | Video clip listing, serving, deletion |
| `conditions.py` | `/api/conditions` | Time period, sunrise/sunset, weather data |
| `metrics.py` | `/api/metrics` | Prometheus metrics endpoint for dashboard stats |
| `auth.py` | `/api/auth` | Login, logout, session management |
| `telegram_access.py` | `/api/telegram` | Telegram user approval, role management, access log |
| `__init__.py` | — | Shared state (Redis clients, key names, defaults) |

---

## Dashboard Frontend

### Pages

| File | URL | Purpose |
|------|-----|---------|
| `index.html` | `/` | Live camera view + settings + events + zones + faces + browse |
| `ai.html` | `/ai.html` | AI chat + vision + DVR + image generation |
| `monitoring.html` | `/monitoring.html` | System health + embedded Grafana |
| `telegram.html` | `/telegram.html` | Telegram user management + access log |
| `login.html` | `/login.html` | Authentication page |

### JavaScript Modules

| File | Lines | Purpose |
|------|-------|---------|
| `app.js` | 362 | WebSocket connection, settings sliders, module init |
| `ai.js` | 962 | AI chat, vision tab, DVR tab, onboarding wizard |
| `generate.js` | 1570 | Image generation, gallery, sweep, img2img, prompt history |
| `events.js` | 345 | Event feed polling, rendering, face cache, photo lightbox |
| `zones.js` | 500+ | Zone drawing canvas, CRUD operations, alert level config |
| `faces.js` | 430+ | Face enrollment wizard, multi-angle capture |
| `browse.js` | 260+ | Vehicle snapshot browser, face gallery |
| `conditions.js` | 200+ | Time period display, weather fetch |
| `unknowns.js` | 190+ | Unknown face grid, label/delete operations |
| `monitoring.js` | 180 | Health cards, Grafana iframe, fullscreen toggle |
| `telegram_access.js` | 240+ | User approval/revoke, access log viewer |
| `auth.js` | 110+ | Login form, session management |

### CSS

| File | Purpose |
|------|---------|
| `style.css` | Main dashboard styles (live view, events, settings, zones) |
| `ai.css` | AI chat interface, DVR player |
| `generate.css` | Image generation UI, gallery, sweep, lightbox |
| `monitoring.css` | System monitor cards, Grafana embed |

---

## Tracker Service

### Person Tracking
- **IoU matching**: for each new detection, compute overlap with every tracked person's last bbox
- **Threshold**: if IoU > 0.3, same person → update state
- **Debounce**: new person must persist 15 frames (~1s) before `person_appeared` event
- **Lost timeout**: if person not seen for `lost_timeout` seconds → `person_left` event
- **Action classification**: keypoint geometry → standing, sitting, crouching, lying (from `contracts/actions.py`)
- **Direction estimation**: bbox center history → left, right, stationary
- **Snapshot at detection**: tracker grabs the frame at event emission time (stored with 2h TTL)

### Vehicle Tracking
- Same IoU pattern, separate `TrackedVehicle` class
- **Idle detection**: if vehicle stationary > `vehicle_idle_timeout` (default 90s) → `vehicle_idle` event
- **Stationarity check**: max displacement from first center < 30px
- Vehicle snapshots stored with 24h TTL

### Zone Checks
- Each detection is checked against configured zones using ray-casting point-in-polygon
- Zone alert levels: `always`, `night_only`, `day_only`, `log_only`, `ignore`
- Alert decision uses `contracts/time_rules.py` `should_alert(zone_level, current_period)`
- Dead zones: detections inside dead zones are completely ignored

---

## Face Recognition

### Service: `face-recognizer`
- **Model**: InsightFace (buffalo_l) running on GPU
- **Database**: SQLite at `/data/faces.db`
- **Match threshold**: cosine similarity > 0.5
- **API port**: 8081 (proxied through dashboard)

### Endpoints (via dashboard proxy)
- `POST /api/faces/enroll` — capture frame, extract embedding, save to DB
- `GET /api/faces` — list all enrolled faces
- `GET /api/faces/{id}/photo` — serve portrait JPEG
- `DELETE /api/faces/{id}` — delete enrollment

### Sticky Identity
Once a face is recognized, the name stays on the bounding box even when the person turns away. A 10-frame vote buffer with 2× bias for the current identity prevents flicker.

### Unknown Face Management
Unrecognized faces are auto-captured and can be labeled later via the dashboard (`/api/unknowns`).

---

## AI Assistant

### Models
| Model | Purpose | Size |
|-------|---------|------|
| **Qwen 3 14B** | Chat + tool calling | ~9.3 GB |
| **MiniCPM-V** | Vision analysis (image description) | ~5 GB |

Both run via Ollama with 5-minute keep-alive. VRAM is shared with ComfyUI via a GPU lock flag.

### Chat Flow
```
User message → build system prompt with live context
→ Qwen 3 14B with TOOLS schema
→ if tool_calls: execute tool → feed result back → re-prompt (up to 5 rounds)
→ final text reply (with embedded media if tools produced any)
```

### System Context (injected each message)
- Current date/time, location, weather
- People currently in frame (from state key)
- Known faces list
- Active zones
- Recent events summary
- Notification status
- System health

### 18 Tool Functions
See `routes/ai_tools.py` — each returns a JSON string the LLM uses to formulate its response. Tools can stash media (snapshots, clips, images) via `ai_state` for embedding in the reply.

---

## Image Generation

### Service: ComfyUI
- Mounts `./models/comfyui/` for checkpoints, LoRAs, VAE
- Dashboard proxies all requests to `http://comfyui:8188`

### Features
- **txt2img**: prompt → ComfyUI workflow → poll for result
- **img2img**: upload source image + denoise strength
- **Batch generation**: queue multiple seeds
- **Parameter sweep**: steps × CFG × LoRA strength grid
- **Gallery**: browse generated images with metadata, lightbox preview, "Use These Settings"
- **Prompt history**: server-side storage with revision tracking (tracks changes during generation)
- **VRAM management**: `gpu:generation_active` flag pauses detectors during generation, auto-unloads Ollama models

---

## Notification System

### Alerts
When Telegram is configured, the dashboard background poller sends photo alerts:

| Event | Photo | Caption Contains |
|-------|:-----:|-----------------|
| `person_appeared` | HD snapshot with bbox | Time, zone, action, AI scene description |
| `person_identified` | HD snapshot with bbox | Name, time, zone |
| `vehicle_idle` | Vehicle snapshot with bbox | Vehicle class, duration, zone |

### Rate Limiting
- Per-event-type cooldowns (configurable via dashboard settings)
- Default: 60s person cooldown, 60s vehicle cooldown
- `suppress_known` toggle: skip notifications for recognized people

### Broadcasting
All alerts are sent to **every approved Telegram user** (multi-user support).

---

## Telegram Bot

### Polling Architecture
`bot_commands.py` runs a long-polling loop as a background task:
1. `getUpdates` from Telegram API (30s timeout)
2. Validate user via `telegram:users` Redis hash
3. Route to command handler
4. Log to `telegram:access_log` stream + per-user audit files on NAS

### Commands
`/snapshot`, `/clip [N]`, `/status`, `/arm`, `/disarm`, `/who`, `/events [N]`, `/analyze`, `/help`, plus photo analysis (send any photo to get MiniCPM-V description).

### Access Control
- Users managed via dashboard Telegram Access Manager page
- Roles: `admin` (full access) and `user` (limited)
- Bootstrap: `TELEGRAM_ALLOWED_USERS` env var seeds initial users

---

## DVR Recording

### Service: `recorder`
- **Method**: ffmpeg RTSP→MP4 copy (no transcode — very low CPU)
- **Segments**: 1-hour MP4 files
- **Retention**: 28-day rolling cleanup
- **Storage**: QNAP NAS via CIFS mount at `/recordings/`
- **Naming**: `{camera_id}/YYYY-MM-DD/HH-MM-SS.mp4`

### Playback API (`recordings.py`)
- `GET /api/recordings/dates` — list available recording dates
- `GET /api/recordings/segments?date=YYYY-MM-DD` — list segments for a date
- `GET /api/recordings/stream/{date}/{segment}` — stream MP4 with range support

---

## Zone System

### Zone Types
| Alert Level | Behavior |
|-------------|----------|
| `always` | Alert on any detection, any time |
| `night_only` | Alert only during night/late-night periods |
| `day_only` | Alert only during daytime |
| `log_only` | Log to event feed but no Telegram notification |
| `ignore` | Completely ignore detections (dead zone) |

### Time Periods
| Period | Window |
|--------|--------|
| **Daytime** | Sunrise + 30min → Sunset − 30min |
| **Twilight** | ±30 min around sunrise and sunset |
| **Night** | Sunset + 30min → Midnight |
| **Late Night** | Midnight → Sunrise − 30min |

### Zone Drawing
Browser-side canvas drawing tool with polygon support. Zones stored in `zones:{camera_id}` Redis hash, read by tracker for alert decisions and by dashboard for overlay rendering.

---

## Monitoring Stack

- **Prometheus** scrapes redis-exporter (Redis stats) and dcgm-exporter (GPU stats)
- **Grafana** serves dashboards at `:3000`, embedded in the System Monitor page via iframe
- **Dashboard metrics** (`routes/metrics.py`) exposes custom Prometheus metrics: inference timing, detection counts, event rates

---

## Shared Contracts

The `contracts/` directory is mounted read-only into every service container:

| File | Exports | Used By |
|------|---------|---------|
| `streams.py` | Redis key templates (`FRAME_STREAM`, `EVENT_STREAM`, etc.) + `stream_key()` resolver + data schema documentation | All services |
| `actions.py` | `classify_action(keypoints)` — keypoint geometry → action label | tracker |
| `time_rules.py` | `get_time_period(dt)`, `should_alert(level, period)`, `point_in_polygon()` | tracker, dashboard |

---

## Authentication

- **Session-based**: cookie + server-side session store in SQLite (`/data/auth.db`)
- **Middleware**: `auth_middleware` in `server.py` intercepts all requests except login, static assets, and API auth endpoints
- **Login page**: blurred camera snapshot background (no auth required for the blurred image)

---

## NAS Storage Layout

All persistent storage beyond Redis uses QNAP NAS CIFS mounts:

```
/data/
├── snapshots/              ← Person detection snapshots
│   └── vehicles/           ← Vehicle snapshots organized by YYYY-MM-DD/
│       └── 2026-02-28/
│           └── 14-30-45_car.jpg
├── events/                 ← Event journal (daily JSONL)
│   └── 2026-02-28.jsonl
├── recordings/             ← DVR segments (read-only in dashboard)
│   └── front_door/
│       └── 2026-02-28/
│           └── 14-00-00.mp4
├── telegram/               ← Per-user bot command audit trail
│   └── username_12345/
│       ├── commands.log
│       └── media/
├── generations/            ← ComfyUI output images
├── clips/                  ← Captured video clips
└── auth.db                 ← Session/auth SQLite database
```

---

## Docker Infrastructure

### Networking
- **camera-ingester** and **recorder**: `network_mode: host` (direct RTSP access to camera on LAN)
- **All other services**: Docker bridge network, communicate via DNS names (e.g., `redis`, `ollama`, `comfyui`)
- **Monitoring stack** (prometheus, grafana, redis-exporter, dcgm-exporter): `network_mode: host` for metric scraping

### Volumes
| Volume | Type | Purpose |
|--------|------|---------|
| `redis-data` | Docker | Redis AOF persistence |
| `face-data` | Docker | InsightFace SQLite DB + portraits |
| `yolo-models` | Docker | YOLO model weights cache |
| `insightface-models` | Docker | InsightFace model weights cache |
| `auth-data` | Docker | Auth SQLite DB |
| `ollama-models` | Docker | LLM model weights (~15 GB) |
| `comfyui-data` | Docker | ComfyUI output images |
| `prometheus-data` | Docker | Prometheus TSDB |
| `grafana-data` | Docker | Grafana state |
| `qnap-*` | CIFS | 7 NAS mounts (snapshots, recordings, events, telegram, generations, videos, clips) |

### GPU Sharing
Five services share the RTX 3090:
1. **pose-detector** — always running (~34ms/frame)
2. **vehicle-detector** — always running
3. **face-recognizer** — always running
4. **ollama** — on-demand (5-min keep-alive), auto-unloaded during image generation
5. **comfyui** — on-demand, sets `gpu:generation_active` flag to pause detectors

---

## File Index

### Services
```
services/
├── camera-ingester/     # RTSP → Redis frames
├── pose-detector/       # YOLOv8s-pose → person bboxes
├── vehicle-detector/    # YOLOv8s → vehicle bboxes
├── tracker/             # IoU tracking → semantic events
├── face-recognizer/     # InsightFace → face identities
├── dashboard/           # FastAPI backend + web frontend
│   ├── server.py        # Main app (WebSocket, event poller, startup)
│   ├── ai_db.py         # AI chat history SQLite
│   ├── routes/          # 20 API route modules
│   └── static/          # 22 frontend files (HTML, JS, CSS)
├── recorder/            # ffmpeg RTSP → MP4 DVR
├── comfyui/             # Stable Diffusion inference
├── prometheus/          # Metrics config
└── grafana/             # Dashboard provisioning
```

### Contracts
```
contracts/
├── streams.py           # Redis key templates + data schemas
├── actions.py           # Keypoint action classification
└── time_rules.py        # Time periods, zone alert rules, PIP test
```
