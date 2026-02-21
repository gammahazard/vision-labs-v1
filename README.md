# Vision Labs — AI-Powered Security Camera System

> **Real hardware. Real-time inference. Self-learning alerts.**

An event-driven, microservices-based security camera system that detects people, tracks them across frames, recognizes known faces, and learns which alerts matter to you over time. Runs entirely on a single PC with a GPU, orchestrated via Docker Compose.

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
| **Telegram notifications** | Real-time photo alerts when people are detected or identified |
| **Vehicle detection** | YOLOv8s detects cars, trucks, buses, motorcycles — snapshot + event feed |
| **Self-learning** *(coming soon)* | User feedback trains an alert scoring model -- fewer false alarms over time |

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
```

Seven containerized services communicate through Redis Streams -- no direct inter-service calls (except dashboard -> face-recognizer REST proxy for enrollment).

| Service | Purpose | GPU? |
|---------|---------|------|
| **redis** | Central message bus (streams + hashes) | No |
| **camera-ingester** | RTSP decode -> JPEG -> Redis | No |
| **pose-detector** | YOLOv8s-pose inference | Yes |
| **tracker** | Person ID assignment + event generation | No |
| **vehicle-detector** | YOLOv8s vehicle detection (car/truck/bus/motorcycle) | Yes |
| **face-recognizer** | InsightFace embedding + enrollment API | Yes |
| **dashboard** | FastAPI + WebSocket + static frontend | No |

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
| `TELEGRAM_CHAT_ID` | No | Your Telegram chat ID for receiving alerts |
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

## Self-Learning Feedback Loop (Phase 6.5 -- Coming Next)

This is the core differentiator: **the system starts by alerting on everything, then learns from your feedback to filter noise.**

### The Three Stages

| Stage | When | Behavior | Notifications |
|-------|------|----------|---------------|
| **Observer** | Week 1-2 | Alerts on everything, asks for feedback | ~15-20/day |
| **Suggest** | Week 3-4 | Auto-suppresses obvious noise, asks about ambiguous | ~3-5/day |
| **Autonomous** | Month 2+ | Only alerts on events you've historically cared about | ~0-2/day |

### What It Learns

A lightweight scoring model (logistic regression -- no GPU needed) trains on ~12 features already available in the pipeline:

- YOLO confidence, identity (known vs unknown), zone name, time period, action type
- Duration on screen, hour of day, day of week, bounding box area, number of people

### How You Teach It

- **Telegram inline buttons** on every notification: Real Alert / Not Needed / Tag (delivery, neighbor, shadow)
- **Dashboard review queue** for batch approve/reject with thumbnails
- The system tracks its own accuracy ("of the last 50 alerts I sent, user confirmed 45") and advances stages when accuracy > 90%

> **No YOLO or InsightFace retraining.** Those are frozen foundation models. The scoring model is a thin decision layer on top that learns your personal alert preferences.

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
| **Phase 6.5** | **Self-learning feedback loop (Telegram buttons, review queue, scoring)** | **Next** |
| Phase 7 | LLM brain -- Ollama + Mistral 7B for event narration, daily summaries, chat | Planned |
| Phase 8 | OpenPLC integration -- Modbus TCP bridge, ladder logic decisions | Planned |

### Phase 7: LLM Brain (Planned)

A local Mistral 7B model (via Ollama) adds natural language capabilities:
- **Event narration** -- turns raw JSON into "An unrecognized person approached your front door at 2:15 PM and stood there for 30 seconds"
- **Daily/weekly summaries** -- "Tuesday: 24 people detected, 1 package delivery, 0 overnight events"
- **Chat interface** -- "Did anyone come to my door while I was at work?" queries the event database and answers conversationally
- **Review assistant** -- pre-labels events: "Likely mail carrier -- timing matches daily pattern"

### Phase 8: OpenPLC (Planned)

Bridges AI detections to industrial PLC ladder logic:
- AI serves as the sensor (input signal)
- PLC makes decisions via ladder diagrams
- Outputs drive actions (alerts now, physical relays/lights in the future)
- PLC rules are editable without touching AI code

### Additional Features (All Just Redis Workers)

- Event clip recording (10s clips around detections, archived to NAS)
- Vehicle speed estimation
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
| **Current total** | **~1.6 GB** |
| + Mistral 7B Q4 (Phase 7) | +4,500 MB |
| **Future total** | **~6.1 GB** |

Fits comfortably on any modern NVIDIA GPU (4GB+ for current, 8GB+ with LLM).

---

## Project Structure

```
vision-labs/
|-- .env.example                  # Environment template (copy to .env)
|-- docker-compose.yml            # All 7 services orchestrated
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
|   +-- dashboard/                # FastAPI backend + static frontend
|       |-- server.py             # App factory, WebSocket handler, middleware
|       |-- routes/               # Modular API endpoints (events, faces, zones, auth, etc.)
|       +-- static/               # HTML/JS/CSS (no build step, no framework)
|
+-- tests/                        # Unit + integration tests (no GPU/Redis required)
    |-- test_actions.py           # Action classification from keypoints
    |-- test_time_rules.py        # Sunrise/sunset + time period logic
    |-- test_face_db.py           # Face database operations
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
