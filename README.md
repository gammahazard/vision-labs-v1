# Vision Labs — AI-Powered Security Camera System

> **Real hardware. Real-time inference. Fully self-hosted.**

A single-machine security platform built on an RTX 3090 that processes a live RTSP camera feed through multiple AI models in real-time. Person detection, face recognition, vehicle tracking, and intelligent notifications — all running locally via Docker Compose with zero cloud dependencies.

---

## What It Does

| Feature | Details |
|---------|---------|
| **Person detection** | YOLOv8s-pose detects people with keypoint-based action classification (standing, sitting, crouching, lying) |
| **Face recognition** | InsightFace identifies known people — names stick to bounding boxes even when they turn away |
| **Vehicle tracking** | YOLOv8s detects cars, trucks, buses, motorcycles with idle timer alerts |
| **Telegram notifications** | Real-time photo alerts with AI scene descriptions, broadcast to all approved users |
| **AI assistant** | Qwen 3 14B local LLM with 18 tool functions — query events, send alerts, capture snapshots, set reminders |
| **Vision analysis** | MiniCPM-V multimodal model analyzes camera snapshots and user-uploaded images |
| **Image generation** | ComfyUI + SDXL for on-device txt2img/img2img with LoRA support, batch generation, gallery |
| **DVR recording** | Continuous 1-hour MP4 segments to QNAP NAS with 28-day rolling retention and browser playback |
| **Zone management** | Draw detection/alert/dead zones on the camera view — configurable per time-of-day |
| **System monitoring** | Prometheus + Grafana dashboards with GPU, Redis, and inference metrics |

---

## Architecture

```
Camera (RTSP)
    │
    ▼
Ingester ──▶ Redis Streams ──▶ YOLO Pose Detector ──▶ Tracker ──▶ Events
                            ──▶ YOLO Vehicle Detector ────────────^
                            ──▶ InsightFace ──▶ Face Identity
                            ──▶ Dashboard (WebSocket ──▶ Browser)

Events ──▶ Notification Poller ──▶ Telegram Bot API
       ──▶ Event Journal (JSONL on QNAP NAS)

Ollama (Qwen 3 14B + MiniCPM-V) ◀──▶ Dashboard AI Chat
ComfyUI (SDXL) ◀──▶ Dashboard Image Generation
Recorder ──▶ DVR segments on QNAP NAS ──▶ Dashboard Playback
```

### Services

| Service | GPU | Purpose |
|---------|:---:|---------|
| **redis** | — | Central message bus — all inter-service communication via Redis Streams |
| **camera-ingester** | — | Reads RTSP sub-stream (640×480) + main stream (HD), publishes JPEG frames to Redis |
| **pose-detector** | ✅ | YOLOv8s-pose inference (~34ms), publishes person bounding boxes + keypoints |
| **vehicle-detector** | ✅ | YOLOv8s inference, publishes vehicle bounding boxes (car/truck/bus/motorcycle) |
| **tracker** | — | IoU matching across frames, assigns persistent IDs, publishes semantic events |
| **face-recognizer** | ✅ | InsightFace embedding + SQLite enrollment DB, publishes identity matches |
| **dashboard** | — | FastAPI backend + static frontend — WebSocket live view, REST APIs, background pollers |
| **ollama** | ✅ | Local LLM server — Qwen 3 14B (chat + tools) and MiniCPM-V (vision) |
| **comfyui** | ✅ | Stable Diffusion inference — SDXL txt2img/img2img with LoRA support |
| **recorder** | — | ffmpeg RTSP→MP4 copy (no transcode), 1-hour segments, 28-day retention |
| **prometheus** | — | Metrics collection (GPU, Redis, inference timing) |
| **grafana** | — | Monitoring dashboards embedded in the system monitor page |
| **redis-exporter** | — | Exports Redis metrics to Prometheus |
| **dcgm-exporter** | — | Exports NVIDIA GPU metrics to Prometheus |

---

## Requirements

- **NVIDIA GPU** (tested on RTX 3090, any CUDA-capable GPU works)
- [Docker](https://docs.docker.com/get-docker/) with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- An **RTSP-capable IP camera** (tested with Reolink RLC-1240A over PoE)
- **QNAP NAS** (optional — for snapshot/recording/event storage via CIFS mounts)

### Setup

```bash
# 1. Clone
git clone https://github.com/gammahazard/vision-labs-v1.git
cd vision-labs-v1

# 2. Configure environment
cp .env.example .env
# Edit .env with your camera IP, credentials, Telegram bot token, etc.

# 3. Build and run
docker compose up --build

# Dashboard at http://localhost:8080
```

### Environment Variables

| Variable | Required | Description |
|----------|:--------:|-------------|
| `CAMERA_IP` | Yes | IP address of the RTSP camera |
| `CAMERA_USER` | Yes | Camera login username |
| `CAMERA_PASSWORD` | Yes | Camera login password |
| `TELEGRAM_BOT_TOKEN` | No | Telegram bot token for notifications and commands |
| `TELEGRAM_CHAT_ID` | No | Default Telegram chat ID for alerts |
| `TELEGRAM_ALLOWED_USERS` | No | Comma-separated Telegram user IDs allowed to use the bot |
| `OPENWEATHER_API_KEY` | No | OpenWeatherMap API key for conditions panel |
| `LOCATION_NAME` | No | Location label shown in dashboard |
| `LOCATION_LAT` | No | Latitude for sunrise/sunset calculations |
| `LOCATION_LON` | No | Longitude for sunrise/sunset calculations |
| `LOCATION_TIMEZONE` | No | IANA timezone (default: `America/Toronto`) |
| `QNAP_IP` | No | QNAP NAS IP for CIFS volume mounts |
| `QNAP_USER` | No | QNAP login username |
| `QNAP_PASSWORD` | No | QNAP login password |

---

## Dashboard Pages

### Live View (`index.html`)
The main page — shows the live camera feed with real-time overlays:
- **Bounding boxes**: cyan for identified people, green for unknown, orange for vehicles
- **Face labels**: recognized names displayed on bounding boxes with sticky identity (persists when face turns away)
- **Action labels**: classified poses (standing, sitting, crouching, lying)
- **Zone overlays**: drawn zones with color-coded alert levels
- **Settings panel**: adjustable confidence threshold, IoU, lost timeout, FPS, notification cooldowns
- **Event feed**: real-time detection events with inline snapshot photos
- **Zone editor**: draw/edit/delete detection zones with per-time-period alert rules
- **Face enrollment**: capture and label known people for recognition
- **Browse panel**: vehicle snapshots organized by day + enrolled faces gallery
- **Conditions panel**: current time period, sunrise/sunset, live weather

### AI Assistant (`ai.html`)
Three-tab interface:
- **Chat tab**: conversational AI assistant (Qwen 3 14B) with 18 tool functions for querying events, managing faces, sending Telegram messages, capturing snapshots/clips, setting reminders, and browsing vehicle history
- **Vision tab**: upload images or capture live frames for MiniCPM-V analysis
- **DVR tab**: browse and play back recorded camera footage by date/segment

### Image Generation (`ai.html` — Generate tab)
ComfyUI-powered image generation:
- Model/LoRA selection from mounted checkpoints
- Prompt history with revision tracking
- Batch generation and parameter sweep (steps × CFG × LoRA strength grid)
- Image-to-image with denoise control
- Gallery with lightbox preview and metadata import
- VRAM management — auto-unloads Ollama models during generation

### System Monitor (`monitoring.html`)
- People count, inference time, GPU status, Redis memory cards
- Embedded Grafana dashboard with adjustable time range
- Fullscreen toggle

### Telegram Access Manager (`telegram.html`)
- Approve/revoke bot users with role-based access (admin/user)
- Access log viewer showing all incoming bot interactions

### Login (`login.html`)
- Session-based authentication with blurred camera background

---

## Telegram Bot Commands

| Command | Description |
|---------|-------------|
| `/snapshot` | Send a live camera photo with AI scene description |
| `/clip [N]` | Capture and send a video clip (5-40 seconds, default 5) with AI analysis |
| `/status` | System health summary — services, GPU, people count, uptime |
| `/arm` | Enable all notifications |
| `/disarm` | Disable all notifications |
| `/who` | Report who/what is currently in the camera frame |
| `/events [N]` | Show recent detection events with snapshot images |
| `/analyze` | Analyze the live camera frame with MiniCPM-V |
| `/help` | List available commands |
| *Send a photo* | Analyze any user-sent photo with MiniCPM-V vision model |

All commands are audit-logged per user to QNAP NAS.

---

## AI Assistant Tools (18)

The Qwen 3 14B model has access to these function-calling tools:

| Tool | What it does |
|------|-------------|
| `query_events` | Search recent security events by type |
| `query_events_by_date` | Search events for a specific date range |
| `query_event_patterns` | Analyze detection patterns and trends |
| `query_faces` | List enrolled people and recognition stats |
| `query_unknowns` | List unidentified face captures |
| `query_zones` | List configured detection zones |
| `query_notification_history` | Recent Telegram notification log |
| `query_activity_heatmap` | Hourly detection frequency breakdown |
| `browse_vehicles` | Browse vehicle snapshots by date |
| `get_live_scene` | Describe what's currently in the camera frame |
| `get_system_status` | System health and resource usage |
| `get_weather` | Current weather conditions |
| `capture_snapshot` | Take and send a camera snapshot |
| `capture_clip` | Record and send a short video clip |
| `send_telegram` | Send a message to Telegram |
| `schedule_reminder` | Set a timed reminder |
| `show_faces` | Display enrolled face photos inline |
| `analyze_image` | Analyze a specific image with MiniCPM-V |

---

## Data Flow

### Redis Streams (real-time pipeline)

```
frames:front_door          ← Ingester publishes JPEG frames (sub-stream)
detections:pose:front_door ← Pose detector publishes person bboxes + keypoints
detections:vehicle:front_door ← Vehicle detector publishes vehicle bboxes
events:front_door          ← Tracker publishes semantic events
identities:front_door      ← Face recognizer publishes identity matches
```

### Redis Keys (state)

```
state:front_door           ← Tracker: current scene snapshot (who's in frame)
identity_state:front_door  ← Face recognizer: current recognized faces
config:front_door          ← Dashboard: live config (thresholds, cooldowns)
zones:front_door           ← Dashboard: zone definitions
frame_hd:front_door        ← Ingester: latest HD frame (for snapshots)
gpu:generation_active      ← Dashboard: GPU lock flag during image generation
telegram:users             ← Dashboard: approved Telegram users
telegram:access_log        ← Bot commands: access audit trail
person_snapshot:*           ← Tracker: detection-time frame captures (2h TTL)
vehicle_snapshot:*          ← Tracker: vehicle detection snapshots (24h TTL)
```

### NAS Storage (QNAP CIFS mounts)

```
/data/snapshots/           ← Person detection snapshots (per-event)
/data/snapshots/vehicles/  ← Vehicle detection snapshots (organized by day)
/data/events/              ← Event journal (daily JSONL files)
/data/telegram/            ← Per-user Telegram command audit trail
/data/recordings/          ← DVR recordings (1-hour MP4 segments)
/data/generations/         ← AI-generated images (ComfyUI output)
/data/clips/               ← Captured video clips
```

---

## Contracts

Shared code lives in `contracts/` and is mounted read-only into every service:

| File | Purpose |
|------|---------|
| `streams.py` | Redis stream/key name templates — single source of truth |
| `actions.py` | Keypoint-based action classification (standing, sitting, crouching, lying) |
| `time_rules.py` | Time period calculation, zone alert rules, point-in-polygon test |

---

## License

This project is for personal/educational use.
