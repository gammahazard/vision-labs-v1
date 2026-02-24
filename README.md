# Vision Labs — AI-Powered Security Camera System

> **Real hardware. Real-time inference. Self-learning alerts.**

An event-driven, microservices-based security camera system that detects people, tracks them across frames, recognizes known faces, and learns which alerts matter to you over time. Continuous DVR recording to a QNAP NAS with 28-day rolling retention. Runs entirely on a single PC with a GPU, orchestrated via Docker Compose.

---

## What It Does

| Capability | How |
|---|---|
| **Live video feed** | RTSP camera -> Redis -> WebSocket -> browser, annotated with bounding boxes and keypoints |
| **Person detection** | YOLOv8s-pose on GPU (~34ms per frame at 640x480) |
| **Person tracking** | IoU-based matching assigns persistent IDs across frames |
| **Action classification** | Standing, sitting, crouching, lying down, arms raised -- pure geometry on pose keypoints |
| **Face recognition** | InsightFace `buffalo_l` generates 512-dim embeddings, cosine-matched against enrolled faces |
| **Zone-based alerting** | Draw polygonal zones on the camera view, set time-based alert rules (always, night only, dead zone) |
| **Telegram notifications** | Real-time photo alerts broadcast to all approved users, per-user bot commands, dashboard-managed access control |
| **Vehicle detection** | YOLOv8s detects cars, trucks, buses, motorcycles — snapshot + event feed |
| **Self-learning** | User feedback trains suppression rules — fewer false alarms over time |
| **AI assistant** | Local Qwen 3 14B via Ollama — 21 tools: chat about events, query faces/weather/patterns, capture live snapshots and 5-second video clips in chat, send Telegram messages, schedule reminders, retrain suppression rules, activity heatmaps, record verdicts conversationally, show enrolled face photos |

---

## Architecture

```
Camera (RTSP)
    |
    v
Ingester --> Redis Streams --> YOLO Pose Detector --> Tracker --> Events
                           --> YOLO Vehicle Detector -------^
                           --> InsightFace --> Face Identity
                           --> Dashboard (WebSocket --> Browser)
                           --> Telegram Notifications
                           --> Ollama (Qwen 3 14B) --> AI Chat
    |
    v
Recorder --> ffmpeg (copy) --> QNAP NAS (MP4 segments, 28-day rolling)
```

Nine containerized services communicate through Redis Streams -- no direct inter-service calls (except dashboard -> face-recognizer REST proxy for enrollment).

| Service | Purpose | GPU? |
|---------|---------|------|
| **redis** | Central message bus (streams + hashes) | No |
| **camera-ingester** | RTSP decode -> JPEG -> Redis | No |
| **pose-detector** | YOLOv8s-pose inference | Yes |
| **tracker** | Person ID assignment + event generation | No |
| **vehicle-detector** | YOLOv8s vehicle detection (car/truck/bus/motorcycle) | Yes |
| **face-recognizer** | InsightFace embedding + enrollment API | Yes |
| **dashboard** | FastAPI + WebSocket + static frontend | No |
| **ollama** | Local LLM inference (Qwen 3 14B) | Yes |
| **recorder** | Continuous DVR recording to QNAP NAS (ffmpeg) | No |

> See [ARCHITECTURE.md](ARCHITECTURE.md) for the full deep dive -- service internals, Redis key map, data flow diagrams, and design decisions.

### Why Redis Streams?

- **Messaging + data storage** in one service
- **Consumer groups** for horizontal scaling (spin up a second detector, they auto-share load)
- **Persistence** -- if a service crashes, it catches up from stream history
- **< 0.5ms latency** -- invisible to the pipeline
- **Decoupled** -- adding a new worker is just reading from a stream

### Fault Isolation

Any service can fail independently:
- Camera goes offline -> other cameras and pipeline are unaffected
- Face recognizer crashes -> person detection continues, just no face labels
- Redis restarts -> services auto-reconnect and resume from last message ID

---

## Quick Start

### Prerequisites

- **NVIDIA GPU** (tested on RTX 3090, but any CUDA-capable GPU works)
- [Docker Desktop](https://docs.docker.com/desktop/) with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- An **RTSP-capable IP camera** (tested with Reolink RLC-1240A over PoE)

### Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/vision-labs.git
cd vision-labs

# 2. Configure environment
cp .env.example .env
# Edit .env with your camera IP, password, and location coordinates
# (location is used for sunrise/sunset-based alert timing)

# 3. Launch everything
docker compose up -d --build

# 4. Open dashboard
# http://localhost:8080
# Default login: admin / admin (change password on first login)
```

### Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `CAMERA_IP` | Yes | Your IP camera's address on the local network |
| `CAMERA_USER` | Yes | RTSP username (usually `admin`) |
| `CAMERA_PASSWORD` | Yes | RTSP password |
| `TELEGRAM_BOT_TOKEN` | No | Telegram bot for mobile push notifications |
| `TELEGRAM_CHAT_ID` | No | Your Telegram chat ID (seeds the first admin user) |
| `TELEGRAM_ALLOWED_USERS` | No | Comma-separated Telegram user IDs to seed on first startup |
| `OPENWEATHER_API_KEY` | No | Weather data for the conditions panel |
| `LOCATION_NAME` | No | Location label (e.g. "Home") |
| `LOCATION_LAT` | No | Latitude for sunrise/sunset calculations |
| `LOCATION_LON` | No | Longitude for sunrise/sunset calculations |
| `LOCATION_TIMEZONE` | No | IANA timezone (default: `America/Toronto`) |

---

## Dashboard

The web dashboard is accessible from any device on your LAN at port 8080. No app installation -- works on desktop and mobile browsers.

### Features

- **Live camera feed** -- real-time JPEG streaming via WebSocket with detection overlays
- **Bounding boxes** -- green for unknown, cyan for identified people, orange for vehicles, with action labels
- **Keypoint overlay** -- 17-point COCO pose skeleton rendered on each person
- **Event feed** -- scrolling list of detection events with inline face photo thumbnails
- **Zone editor** -- draw polygonal zones directly on the camera feed with drag-to-edit vertices, set alert levels per zone
- **Face enrollment wizard** -- guided multi-angle capture (front, left, right, up, down) with live oval face guide
- **Telegram Access Manager** -- approve/revoke bot users from dashboard, access audit log, one-click enrollment from denied attempts
- **Unknown faces gallery** -- auto-captured unknowns with one-click label/dismiss
- **Conditions panel** -- current time period (daytime/twilight/night), sunrise/sunset, live weather
- **Settings** -- adjustable confidence, IoU threshold, lost timeout -- all hot-reload, no container restart needed
- **Notifications** -- Telegram integration status and test button
- **Authentication** -- session-based login with cookie auth, change password support

---

## How People Are Tracked and Identified

### Detection Pipeline

```
Frame arrives from camera (15 FPS)
  -> YOLO detects person bounding boxes + 17 keypoints
  -> Tracker matches boxes across frames via IoU overlap
  -> Person confirmed after 3 stable frames (debounce)
  -> Action classified from keypoint geometry (no ML -- pure math)
  -> Events emitted: person_appeared, person_left, action_changed
```

### Face Recognition Pipeline

```
Simultaneously for each detected person:
  -> Upper body cropped from frame
  -> InsightFace detects face and generates 512-dim embedding
  -> Embedding compared against all enrolled faces (cosine similarity)
  -> If similarity > threshold: identity published to Redis
  -> Tracker links name to person ID
  -> If no match: face auto-saved as "unknown" for later labeling
```

### Sticky Identity

Once a face is recognized, the name stays on the bounding box even when the person turns away from the camera. A 10-frame vote buffer with 2x bias for the current identity prevents flicker.

### Unknown Face Management

Faces that don't match anyone enrolled are automatically saved with deduplication -- the system groups repeated sightings of the same unknown (cosine similarity > 0.6). From the dashboard you can:
- **Label** an unknown to promote it to a known face (triggers re-enrollment + retroactive cleanup)
- **Dismiss** individual unknowns or clear all

---

## Zone-Based Alerting

Draw polygonal zones on the camera view and assign alert behaviors:

| Alert Level | Behavior |
|-------------|----------|
| **Always** | Alert in all time periods |
| **Night Only** | Alert only during night and late night |
| **Log Only** | Record events, never send notifications |
| **Ignore** | Skip alerting, still track people |
| **Dead Zone** | Completely suppress -- no tracking, no events, no bounding boxes |

### Time-Based Rules

Sunrise and sunset times are calculated daily using the [`astral`](https://github.com/sffjunkie/astral) library based on your configured coordinates in `.env`. Four time periods:

| Period | Window |
|--------|--------|
| **Daytime** | Sunrise + 30min -> Sunset - 30min |
| **Twilight** | +/- 30 min around sunrise and sunset |
| **Night** | Sunset + 30min -> Midnight |
| **Late Night** | Midnight -> Sunrise - 30min |

Dead zones are useful for suppressing detections on neighbor property or public sidewalks where you want zero tracking.

---

## Notification System

When configured with a Telegram bot token, the system sends real-time alerts with camera snapshots:

| Event | Rate Limited? | Includes |
|-------|---------------|----------|
| Person detected | Yes (1/min) | Camera snapshot |
| Person identified | No | Camera snapshot + name |
| Vehicle idle | Yes (1/min) | Camera snapshot |
| Face enrolled | No | Face thumbnail |

---

## Self-Learning Feedback Loop (Phase 6.5)

**The system starts by alerting on everything, then learns from your feedback to filter noise.** No ML retraining — a deterministic rule engine builds suppression rules from your verdict patterns.

### How It Works

Every Telegram notification includes three inline buttons:
- **✅ Real Alert** — confirms this was a genuine detection
- **❌ Not Needed** — marks as false alarm (drives rule creation)
- **🏷️ Tag** — identify the person (delivery, neighbor, shadow, etc.)

You can also classify events from the **dashboard review queue** with thumbnails, or ask the AI assistant to retrain rules at any time.

### Auto-Suppression Rules

The engine creates rules automatically from accumulated `false_alarm` verdicts:

| Rule Type | Threshold | Example | Effect |
|-----------|-----------|---------|--------|
| **Identity** | 3 false alarms | "Mail Carrier" marked false 3× | All future alerts for that person suppressed |
| **Zone + Time** | 5 false alarms | "Driveway" + "daytime" marked false 5× | All alerts in that zone during that time period suppressed |

Suppression is checked **before** the rate-limit timer — suppressed events don't burn your notification cooldown window. Rules can be toggled on/off or deleted from the dashboard at any time, and `retrain` wipes and rebuilds all rules from scratch.

> **No YOLO or InsightFace retraining.** Those are frozen foundation models. The suppression engine is a thin decision layer on top that learns your personal alert preferences.

### Known Limitations

- **Night override** — all suppression rules are bypassed during **night and late_night** time periods. Any person detected at those hours always triggers a notification, even if they're a known suppressed identity. Dead zones still apply 24/7 (enforced at the tracker level, not the notification layer).
- **No action awareness** — zone+time rules don't consider what the person is doing (crouching vs walking). Rules suppress all actions equally.
- **No negative feedback loop** — marking "Real Threat" confirms the alert was valid but doesn't strengthen future alerts for similar patterns. Only false alarms drive rule creation.
- **Progression is emergent, not explicit** — there are no formal "stages." The system naturally sends fewer notifications as rules accumulate, but there's no stage-tracking or mode-switching logic.

---

## Full Roadmap

| Phase | What | Status |
|-------|------|--------|
| Phase 0 | Hardware setup -- camera, network, NAS | Done |
| Phase 1 | Camera ingester + Redis Streams (15 FPS) | Done |
| Phase 2 | YOLOv8s-pose detection + IoU tracker | Done |
| Phase 3 | Live dashboard + WebSocket streaming | Done |
| Phase 4 | Action classification from pose keypoints | Done |
| Phase 5 | Face recognition + enrollment wizard + unknown gallery | Done |
| Phase 6 | Zone editor + time-based alert rules + Telegram notifications | Done |
| Phase 6.1 | Dashboard authentication (login, sessions, password change) | Done |
| Phase 6.2 | Vehicle detection (car/truck/bus/motorcycle) + event feed | Done |
| Phase 6.5 | Self-learning feedback loop (Telegram buttons, review queue, suppression rules) | Done |
| Phase 7 | AI assistant -- Ollama + Qwen 3 14B, onboarding wizard, chat UI, 21 tools | Done |
| Phase 7.5 | Telegram Access Manager -- per-user auth, enrollment flow, access log, dashboard page | Done |
| Phase 8 | Extended bot commands + AI tools -- /zones, /rules, /night, /faces, /timelapse, activity heatmap, record_verdict, show_faces | Done |
| Phase 9 | QNAP NAS storage -- DVR recording, event journal, Telegram audit trail | Done |

### Phase 7: AI Assistant

A local Qwen 3 14B model (via Ollama, ~9.3 GB) adds natural language capabilities:
- **Chat interface** -- dark-themed UI with onboarding wizard, suggestion chips, markdown + inline image rendering
- **21 tools** -- query events/events by date/patterns/activity heatmap, faces/unknowns/show face photos, live scene, capture snapshot (with weather + scene description), capture 5-second video clip in chat, get weather, browse vehicles, zones, notification history, feedback stats, review feedback, retrain rules, send Telegram (text/snapshot/clip), schedule reminders, system status, record verdicts conversationally
- **Runs entirely on-device** -- no cloud APIs, no data leaves the machine
- **Extensibility roadmap** -- see ARCHITECTURE.md for 3-tier plan (smarter context → proactive intelligence → autonomous operation)

### Phase 8: Extended Bot Commands & AI Tools

Expanded Telegram bot and AI assistant capabilities:
- **6 new bot commands** -- `/zones` (snapshot with zone overlays), `/rules` (suppression rules + stats), `/night` (night override status), `/faces` (enrolled people), `/timelapse [YYYY-MM-DD]` (MP4 from day's snapshots), `/events` (with snapshot thumbnails)
- **3 new AI tools** -- `query_activity_heatmap` (day×hour cross-tabulation), `record_verdict` (conversational event classification), `show_faces` (display enrolled photos in chat)
- **Night override awareness** -- `/night` command shows current period and active override status
- **AI learning loop** -- `record_verdict` allows conversational feedback ("mark that as false alarm") while keeping the AI read-only on rule modification

### QNAP NAS Storage (Phase 9)

All persistent data stored on the QNAP TS-431X2 NAS (5.2 TB) via Docker CIFS/SMB volumes:

| Data | Location | Retention |
|------|----------|-----------|
| DVR recordings | `/recordings/front_door/YYYY-MM-DD/HH-MM.mp4` | 28 days (auto-cleanup) |
| Event snapshots | `/snapshots/{event_id}.jpg` | Indefinite |
| Event journal | `/events/YYYY-MM-DD.jsonl` | Indefinite |
| Telegram audit trail | `/telegram/@username/commands.jsonl` | Indefinite |
| Telegram snapshots | `/telegram/@username/snapshots/*.jpg` | Indefinite |
| Telegram clips | `/telegram/@username/clips/*.mp4` | Indefinite |

The DVR recorder uses `ffmpeg -c copy` (zero transcode) to remux the RTSP sub-stream H.264 into 1-hour MP4 segments. Very low CPU usage.

### Future (All Just Redis Workers)

- License plate reader (YOLO + PaddleOCR)
- Vehicle re-identification (same car tracking across visits)
- Cross-camera person tracking
- TensorRT optimization for higher throughput
- Additional cameras (zero code changes -- just add a docker-compose entry)

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Message bus | Redis Streams | Pub/sub + persistence + cache in one |
| Backend API | FastAPI + Uvicorn | Async, WebSocket support, fast |
| Frontend | Vanilla HTML/JS/CSS + WebSocket | Lightweight, no build step |
| Pose detection | YOLOv8s-pose (Ultralytics) | Keypoints + bounding box in one model |
| Face recognition | InsightFace buffalo_l | 512-dim embeddings, cosine matching |
| Time rules | astral | Sunrise/sunset from coordinates, no internet |
| Database | SQLite (faces + auth) | Embedded, zero-config |
| Containers | Docker Compose | One service per container |
| GPU runtime | NVIDIA Container Toolkit + CUDA 11.8 + cuDNN 8 | Required for ONNX Runtime GPU acceleration |

### GPU Memory Budget

| Model | VRAM |
|-------|------|
| YOLOv8s-pose | ~500 MB |
| YOLOv8s (vehicles) | ~500 MB |
| InsightFace buffalo_l | ~600 MB |
| Qwen 3 14B (Ollama) | ~9,300 MB |
| **Current total** | **~10.9 GB** |

Fits on any 12GB+ NVIDIA GPU. The RTX 3090 (24 GB) has ~13 GB headroom.

---

## Project Structure

```
vision-labs/
|-- .env.example                  # Environment template (copy to .env)
|-- docker-compose.yml            # All 9 services orchestrated
|-- ARCHITECTURE.md               # Full technical deep dive
|-- v2.md                         # Phased build plan with design rationale
|
|-- contracts/                    # Shared API contracts (mounted read-only into services)
|   |-- __init__.py               # Package docstring
|   |-- streams.py                # Redis key templates + data schemas
|   |-- actions.py                # Pose-based action classifier (geometry, no ML)
|   +-- time_rules.py             # Sunrise/sunset + zone alert evaluation
|
|-- services/
|   |-- camera-ingester/          # RTSP decode -> JPEG -> Redis
|   |-- pose-detector/            # YOLOv8s-pose inference (GPU)
|   |-- tracker/                  # Person ID assignment + event publishing
|   |-- face-recognizer/          # InsightFace + enrollment REST API (GPU)
|   |-- vehicle-detector/         # YOLOv8s vehicle detection (GPU)
|   |-- recorder/                 # DVR recording to QNAP NAS (ffmpeg)
|   +-- dashboard/                # FastAPI backend + static frontend
|       |-- server.py             # App factory, WebSocket handler, middleware
|       |-- feedback_db.py        # Feedback + suppression rules (SQLite)
|       |-- ai_db.py              # AI config + reminders + chat history (SQLite)
|       |-- routes/               # Modular API endpoints (events, faces, zones, auth, feedback, ai, telegram)
|       +-- static/               # HTML/JS/CSS (no build step, no framework)
|           |-- index.html         # Dashboard (live feed, sidebar panels)
|           |-- ai.html            # AI assistant (onboarding wizard + chat)
|           |-- telegram.html      # Telegram Access Manager (users + access log)
|           +-- login.html         # Authentication page
|
+-- tests/                        # Unit + integration tests (no GPU/Redis required)
    |-- test_actions.py           # Action classification from keypoints
    |-- test_time_rules.py        # Sunrise/sunset + time period logic
    |-- test_face_db.py           # Face database operations
    |-- test_feedback_db.py       # Feedback + suppression rule tests
    |-- test_tracker.py           # IoU matching + person tracking
    |-- test_routes.py            # Dashboard API endpoint tests
    +-- test_vehicles.py          # Vehicle pipeline tests
```

---

## Design Principles

1. **Single source of truth** -- all Redis keys and data schemas live in `contracts/`, shared across services via read-only volume mount
2. **Loose coupling** -- services communicate only through Redis Streams, never direct HTTP calls (except enrollment proxy)
3. **Hot-reload config** -- change detection confidence, IoU threshold, or lost timeout from the dashboard; services pick up changes from Redis without restarting
4. **Fault isolation** -- any service can crash, restart, or be rebuilt without affecting the rest of the pipeline
5. **Privacy first** -- all processing is 100% local, no cloud APIs for detection/recognition, dead zones for neighbor property
6. **Modular scaling** -- adding a camera = one docker-compose entry, adding a new detector = one new file that reads from a stream

---

## Running Tests

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest tests/ -v
```

Tests cover action classification, time rules, face database operations, tracker algorithms, and API routes -- all without requiring GPU or Redis.

---

## Adding a Camera

Zero code changes required:

1. Connect camera, assign static IP
2. Add a new ingester service in `docker-compose.yml` with `CAMERA_ID` and `RTSP_URL`
3. `docker compose up -d camera-ingester-2`
4. Dashboard auto-discovers the new camera via Redis

---

## License

MIT
