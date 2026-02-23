# Vision Labs — Architecture Reference

> **Last updated:** Feb 23, 2026
> **Status:** Phases 0–8 complete. 21-tool AI assistant. Telegram Access Manager. Extended bot commands.
> **Hardware:** RTX 3090 PC, Reolink RLC-1240A (PoE), Cisco switch, QNAP NAS.

This document is the definitive reference for how the system works. If you lose context, start here.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Service Inventory](#service-inventory)
4. [Redis Key Map](#redis-key-map)
5. [Data Flow — End to End](#data-flow--end-to-end)
6. [Shared Contracts](#shared-contracts)
7. [Service Deep Dives](#service-deep-dives)
8. [Dashboard Deep Dive](#dashboard-deep-dive)
9. [Frontend Deep Dive](#frontend-deep-dive)
10. [Inter-Service Communication](#inter-service-communication)
11. [Hot-Reload Config System](#hot-reload-config-system)
12. [Authentication System](#authentication-system)
13. [Notification System](#notification-system)
14. [Docker Infrastructure](#docker-infrastructure)
15. [Test Suite](#test-suite)
16. [Current Status vs v2.md Plan](#current-status-vs-v2md-plan)
17. [Phase 6.5: Self-Learning](#phase-65-self-learning-feedback-loop-implemented)
18. [Modularity & Security Principles](#modularity--security-principles)
19. [Extensibility Roadmap](#extensibility-roadmap)
20. [File Index](#file-index)

---

## System Overview

Vision Labs is an **AI-powered security camera system** built as event-driven microservices over Redis Streams. A single Reolink PoE camera provides an RTSP video feed that flows through a pipeline:

```
Camera (RTSP) → Ingester → Redis → YOLO Pose → Tracker → Events
                                  → InsightFace → Identities
                                  → Dashboard (WebSocket → Browser)
                                  → Telegram Notifications
```

**Key design principles:**
- **Single source of truth:** All Redis keys and data schemas defined in `contracts/`
- **Loose coupling:** Services communicate only via Redis streams/hashes — no direct calls (except dashboard → face-recognizer HTTP proxy)
- **Hot-reload:** Config changes from the dashboard propagate via Redis — no restarts needed
- **Fault isolation:** Any service can crash without taking down the pipeline
- **GPU budget:** YOLOv8s-pose (~500 MB) + YOLOv8s vehicles (~500 MB) + InsightFace buffalo_l (~600 MB) + Qwen 3 14B (~9.3 GB) = ~10.9 GB of 24 GB VRAM

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│  PC (RTX 3090) — Everything runs here via Docker Compose             │
│                                                                      │
│  ┌─────────────────┐                                                 │
│  │ camera-ingester  │──RTSP──→ Reolink RLC-1240A (192.168.2.10)     │
│  │ (host network)   │                                                │
│  └────────┬────────┘                                                 │
│           │ XADD frames:front_door                                   │
│           ▼                                                          │
│  ┌────────────────┐                                                  │
│  │     Redis       │ (port 6379, bridge network)                     │
│  │  Streams+Hashes │                                                 │
│  └───┬────┬────┬──┘                                                  │
│      │    │    │                                                      │
│      │    │    └──────────────────────────────────┐                   │
│      │    │                                      │                   │
│      ▼    ▼                                      ▼                   │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐        │
│  │pose-detector │  │    tracker       │  │ face-recognizer  │        │
│  │ (GPU, YOLO)  │  │ (CPU, IoU match) │  │ (GPU, InsightFace│        │
│  │              │  │                  │  │  + REST API :8081)│        │
│  └──────┬───────┘  └────────┬─────────┘  └────────┬─────────┘        │
│         │                   │                     │                  │
│         │ detections:pose:  │ events:front_door   │ identity_state:  │
│         │ front_door        │ state:front_door    │ front_door       │
│         │                   │                     │                  │
│         └───────────────────┴─────────────────────┘                  │
│                             │                                        │
│                     ┌───────▼────────┐                               │
│                     │   dashboard    │                                │
│                     │  (FastAPI :8080)│──HTTP proxy──→ face-recognizer│
│                     │  WebSocket /ws │                                │
│                     │  Static files  │                                │
│                     └───────┬────────┘                               │
│                             │                                        │
│                     ┌───────▼────────┐   ┌──────────────┐            │
│                     │    Browser     │   │   Telegram    │            │
│                     │ (any LAN device│   │   Bot API    │            │
│                     │  :8080)        │   │  (HTTPS)     │            │
│                     └────────────────┘   └──────────────┘            │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     ┌──────────────┐
                     │  QNAP NAS    │ (192.168.2.20)
                     │  FTP archive │
                     └──────────────┘
```

---

## Service Inventory

| Service | Container | Port | GPU? | Purpose |
|---------|-----------|------|------|---------|
| **redis** | vision-labsv1-redis-1 | 6379 | No | Central message bus (streams + hashes) |
| **camera-ingester** | vision-labsv1-camera-ingester-1 | — | No | RTSP decode → JPEG → Redis |
| **pose-detector** | vision-labsv1-pose-detector-1 | — | Yes | YOLOv8s-pose inference |
| **tracker** | vision-labsv1-tracker-1 | — | No | Person tracking + event generation |
| **vehicle-detector** | vision-labsv1-vehicle-detector-1 | — | Yes | YOLOv8s vehicle detection |
| **face-recognizer** | vision-labsv1-face-recognizer-1 | 8081 | Yes | InsightFace embedding + REST API |
| **dashboard** | vision-labsv1-dashboard-1 | 8080 | No | FastAPI + WebSocket + static frontend |
| **ollama** | vision-labsv1-ollama-1 | 11434 | Yes | Local LLM inference (Qwen 3 14B) |

---

## Redis Key Map

All keys defined in `contracts/streams.py`. The function `stream_key(template, **kwargs)` resolves placeholders.

| Key | Type | Producer | Consumer(s) | Data Shape |
|-----|------|----------|-------------|------------|
| `frames:{camera_id}` | Stream | camera-ingester | pose-detector, vehicle-detector, face-recognizer, dashboard | `{frame: JPEG bytes, timestamp: float, frame_number: int, resolution: "WxH"}` |
| `detections:pose:{camera_id}` | Stream | pose-detector | tracker, face-recognizer | `{detections: JSON[{bbox, confidence, keypoints}], inference_ms, frame_number}` |
| `events:{camera_id}` | Stream | tracker | dashboard, notification poller | `{event_type, person_id, timestamp, duration, direction, action, zone, alert_level, alert_triggered, identity_name}` |
| `state:{camera_id}` | Hash | tracker | dashboard WebSocket | `{num_people, people: JSON[{person_id, bbox, action, ...}]}` |
| `config:{camera_id}` | Hash | dashboard | pose-detector, tracker | `{confidence_thresh, iou_threshold, lost_timeout, target_fps}` |
| `zones:{camera_id}` | Hash | dashboard | tracker, dashboard overlay | `{zone_id: JSON{name, points, alert_level}}` |
| `identities:{camera_id}` | Stream | face-recognizer | dashboard | `{identities: JSON[{name, bbox, confidence}]}` |
| `identity_state:{camera_id}` | Hash | face-recognizer | tracker, dashboard | `{identities: JSON[{name, bbox, confidence}]}` |
| `detections:vehicle:{camera_id}` | Stream | vehicle-detector | tracker | `{detections: JSON[{bbox, confidence, class_name, class_id}], detector_type: "vehicle", inference_ms, frame_bytes}` |
| `vehicle_snapshot:{camera_id}:{ts}` | String (TTL 24h) | tracker | dashboard | Raw JPEG bytes |
| `frame_hd:{camera_id}` | String (TTL 5s) | camera-ingester (HD thread) | dashboard (HD toggle) | Raw JPEG bytes of main stream frame |
| `telegram:users` | Hash | dashboard (seed + CRUD) | bot_commands, notifications | `{user_id: JSON{chat_id, name, username, approved_at}}` |
| `telegram:access_log` | Stream (maxlen 1000) | bot_commands (poller) | dashboard (Telegram page) | `{user_id, username, first_name, chat_id, action, authorized, timestamp}` |

**Default camera_id:** `front_door`

---

## Data Flow — End to End

### Frame Pipeline (every ~67ms at 15 FPS)

```
1. CAMERA → RTSP H.264 sub-stream (640×480)
2. camera-ingester:
   - cv2.VideoCapture reads frame
   - JPEG encode (quality 80)
   - XADD frames:front_door {frame, timestamp, frame_number, resolution}
   - Stream capped at 1000 entries (MAXLEN)

3. pose-detector (consumer group "pose_detectors"):
   - XREADGROUP from frames stream
   - Decode JPEG → numpy array
   - YOLOv8s-pose inference (~34ms on RTX 3090)
   - Filter: person class only, confidence > threshold
   - XADD detections:pose:front_door {detections: JSON, inference_ms}

4. tracker (consumer group "trackers"):
   - XREADGROUP from detections stream
   - IoU matching against tracked persons
   - If new person: debounce 3 frames, then emit person_appeared
   - If person gone > lost_timeout: emit person_left
   - Action classification via contracts/actions.py (debounced 10 frames, sticky 2x)
   - Zone evaluation: point_in_polygon + should_alert (time-based)
   - Dead zone: completely suppress — no tracking, no events
   - XADD events:front_door
   - HSET state:front_door

5. face-recognizer (consumer group "face_recognizers"):
   - XREADGROUP from detections stream (separate group from tracker)
   - For each detected person bbox: crop upper 50% → InsightFace
   - If face found: generate 512-dim embedding, cosine match against SQLite
   - If match > 0.5: publish identity
   - If unknown with det_score >= 0.85: auto-save to unknowns table
   - XADD identities:front_door
   - HSET identity_state:front_door

6. dashboard (WebSocket /ws/live):
   - XREVRANGE frames (latest 1)
   - XREVRANGE detections (latest 1)
   - HGETALL state + identity_state
   - Draw bounding boxes (cyan=identified, green=unknown, orange=vehicle)
   - Draw keypoints, zone overlays
   - JPEG encode → base64 → send JSON via WebSocket
   - Target: 10 FPS to browser
```

### Event Notification Flow

```
1. tracker emits person_appeared or person_identified to events stream
2. dashboard background poller (_event_notification_poller):
   - XREAD events stream (blocking, in threadpool executor)
   - ALWAYS on person_appeared: grab latest frame, save as JPEG to /data/snapshots/{event_id}.jpg
   - If Telegram configured + person_appeared: rate-limited (1 per 60s), send Telegram photo
   - If Telegram configured + person_identified: NOT rate-limited, always send Telegram photo
   - Old snapshots auto-cleaned every ~200s (files older than 2 hours)
3. Dashboard frontend shows snapshot thumbnails in the event feed
4. Telegram receives photo + HTML-formatted caption (when configured)
```

### Face Enrollment Flow (user-initiated)

```
1. User types name in wizard, clicks Capture
2. Browser → POST /api/faces/enroll {name}
3. Dashboard proxy → face-recognizer:8081/api/faces/enroll
4. face-recognizer:
   - XREVRANGE frames (latest 1) → full frame
   - XREVRANGE detections (latest 1) → person bboxes
   - Pick largest person bbox
   - Crop upper 50% → InsightFace → embedding + portrait thumbnail
   - SQLite INSERT into known_faces (name, embedding, photo)
   - Sweep unknowns for matches → clear any that match
   - Return {success, face_id, name}
5. Dashboard sends Telegram notification with face photo
6. Wizard shows captured photo, auto-advances to next angle
```

---

## Shared Contracts

**Location:** `contracts/` — mounted into every container via Docker Compose volume.

### contracts/streams.py
- All Redis key templates as string constants
- `stream_key(template, **kwargs)` — resolves `{camera_id}`, `{detector_type}` placeholders
- Dataclasses: `FrameMessage`, `DetectionMessage`, `EventMessage` — document expected schemas

### contracts/actions.py
- `classify_action(keypoints: list) → str` — pure math on 17 COCO keypoints
- Actions: `standing`, `sitting`, `crouching`, `lying_down`, `arms_raised`
- Uses hip-ankle ratios, knee angles, torso orientation, wrist-shoulder positions
- No ML model — just geometry

### contracts/time_rules.py
- `get_time_period(dt) → str` — returns `daytime`, `twilight`, `night`, `late_night`
- `should_alert(zone_alert_level, current_period) → bool` — evaluates zone rules
- `point_in_polygon(x, y, polygon) → bool` — ray-casting PIP test
- Uses `astral` library for sunrise/sunset, location configured via `LOCATION_LAT`/`LOCATION_LON` env vars
- Time periods: daytime (sunrise+30min → sunset-30min), twilight (±30min around sunrise/sunset), night (sunset+30min → midnight), late_night (midnight → sunrise-30min)

---

## Service Deep Dives

### camera-ingester (`services/camera-ingester/ingester.py`)

**~297 lines.** Single file, no dependencies beyond OpenCV + Redis.

- **RTSP connection:** Uses sub-stream (640×480) for AI inference; optional HD thread reads main stream and caches latest frame in Redis for a live HD toggle
- **Frame throttling:** `cap.grab()` discards frames between captures to hit TARGET_FPS
- **Stream capping:** `XADD ... MAXLEN 1000` prevents Redis from growing unbounded
- **Reconnect:** Exponential backoff (1s → 2s → 4s → ... → 30s max) on RTSP failure
- **Docker:** `network_mode: host` to reach camera on `192.168.2.10`
- **Graceful shutdown:** SIGTERM/SIGINT handlers release OpenCV capture

### pose-detector (`services/pose-detector/detector.py`)

**~282 lines.** GPU service.

- **Model:** YOLOv8s-pose (auto-downloaded on first run, cached in Docker volume `yolo-models`)
- **Consumer group:** `pose_detectors` — can run multiple instances for load balancing
- **Inference:** ~34ms on RTX 3090 at 640×480
- **Output:** For each person: `{bbox: [x1,y1,x2,y2], confidence: float, keypoints: [[x,y,conf]×17]}`
- **Hot-reload:** Reads `confidence_thresh` from `config:{camera_id}` every 25 frames
- **YOLO clip_boxes bug:** Documented in v2.md — x-coords clamp to height instead of width

### vehicle-detector (`services/vehicle-detector/detector.py`)

**~270 lines.** GPU service. Mirrors pose-detector architecture.

- **Model:** YOLOv8s (general object detection, auto-downloaded, cached in Docker volume `yolo-models`)
- **Consumer group:** `vehicle_detectors` — reads from same frame stream as pose-detector
- **Class filter:** COCO classes 2 (car), 3 (motorcycle), 5 (bus), 7 (truck) — filtered at inference time
- **Confidence threshold:** 0.5 (env `CONFIDENCE_THRESH`, raised from 0.4 to reduce false positives)
- **Min bbox area:** 5000 px² (env `MIN_VEHICLE_BBOX_AREA`) — discards tiny reflections/distant objects
- **Frame skip:** Default 3 (processes every 3rd frame to save GPU for fast-moving vehicles)
- **Output:** For each vehicle: `{bbox: [x1,y1,x2,y2], confidence: float, class_name: str, class_id: int}`
- **Snapshot:** Includes raw frame bytes in detection message for tracker to save as vehicle snapshot
- **VRAM:** ~500 MB on RTX 3090

### tracker (`services/tracker/tracker.py`)

**~747 lines.** CPU-only, most complex service.

**Core algorithm:**
- Maintains a dict of `TrackedPerson` objects, each with: person_id, bbox, first_seen, last_seen, action, action_history, confirmed (bool)
- Every detection frame: compute IoU matrix between all current persons and new detections
- Greedy assignment: highest IoU match > threshold → update person; unmatched detections → new person
- Person confirmed after 3 stable frames (debounce against flickering detections)

**Action classification:**
- Calls `contracts/actions.py` for each person each frame
- Maintains per-person action vote buffer (10 frames)
- Sticky bias: current action needs 2× opposite votes to change (prevents oscillation)
- Emits `action_changed` event with `prev_action`

**Zone evaluation:**
- Loads zones from `zones:{camera_id}` every 10 seconds
- Tests person bbox center against each zone polygon via `point_in_polygon()`
- Dead zones: if person center is inside a dead zone, completely suppress (delete TrackedPerson, no event)
- Alert evaluation: `should_alert(zone.alert_level, current_time_period)` → sets `alert_triggered` on event

**Identity integration:**
- Reads `identity_state:{camera_id}` every 2 seconds
- Matches face-recognizer identity bboxes to tracked persons via IoU
- Once matched: emits `person_identified` event (fires only once per person per identity assignment)
- Identity name propagated to all subsequent events for that person

**Hot-reload config:**
- Reads `iou_threshold`, `lost_timeout` from Redis config every 10 messages

### face-recognizer (`services/face-recognizer/recognizer.py` + `face_db.py`)

**~603 + ~383 lines.** GPU service + SQLite DB + REST API.

**Dual role:**
1. **Background loop** — reads detections, crops faces, matches embeddings, publishes identities
2. **REST API (port 8081)** — enrollment, preview, unknowns management (called by dashboard proxy)

**InsightFace pipeline:**
- Model: `buffalo_l` (RetinaFace detector + ArcFace recognizer)
- Crops upper 50% of person bbox for face detection
- Generates 512-dimensional normalized embedding
- Cosine similarity matching against all enrolled faces
- Match threshold: 0.5 default (constructor parameter on `FaceDB`)

**face_db.py (FaceDB class):**
- SQLite with WAL mode, in-memory embedding cache for fast matching
- Tables: `known_faces` (id, name, embedding 2048 bytes, photo JPEG), `unknown_faces` (auto-captured)
- `enroll(name, embedding, photo)` → INSERT + cache update
- `match(embedding) → (name, face_id, similarity)` — cosine against all cached embeddings
- `save_unknown(embedding, photo)` — dedup: if >0.6 similar to existing unknown, just bump sighting_count
- `label_unknown(uid, name)` → move from unknown to known (promotes embedding)
- `reconcile_unknowns()` — startup sweep: clear unknowns that match any known face
- Max 100 unknowns, oldest pruned when exceeded

**Face thumbnail (enrollment photo):**
- Detects face within upper torso crop
- Applies 120% horizontal + 100% vertical padding around face bbox
- Crops from full frame (not head region) for natural portrait look
- Resizes to 200×200 JPEG

---

## Dashboard Deep Dive

### Backend (`services/dashboard/server.py`)

**~1075 lines.** FastAPI with modular routes.

**Startup sequence:**
1. Initialize auth SQLite database (create default admin/admin if empty)
2. Write default config to Redis if not present
3. Start background event notification poller (async task)
4. Mount static files, include all route modules

**WebSocket `/ws/live`:**
- Reads latest frame + detections + state + identities every ~100ms
- Draws bounding boxes with OpenCV (cyan for identified, green for unknown)
- Draws keypoint dots (orange, confidence > 30%)
- Draws zone overlays (semi-transparent colored polygons)
- Sticky identity cache: once a face is matched to a tracker person_id, the name persists even when face isn't visible
- Encodes annotated frame as JPEG → base64 → JSON → WebSocket

**Auth middleware:**
- Every HTTP request checked for `vl_session` cookie
- Exempt paths: `/login.html`, `/api/auth/login`, `/api/auth/status`, `/style.css`, `/auth.js`
- Invalid session: redirect to `/login.html` (browser) or 401 (API)

### Route Modules (`services/dashboard/routes/`)

| Module | Endpoints | Purpose |
|--------|-----------|---------|
| `auth.py` | `POST /api/auth/login`, `POST /api/auth/logout`, `POST /api/auth/change-password`, `GET /api/auth/status` | SQLite-backed auth with signed cookie sessions |
| `events.py` | `GET /api/events?count=N`, `GET /api/events/{id}/snapshot` | Event feed from Redis stream + camera snapshot JPEGs from disk |
| `config.py` | `GET /api/config`, `POST /api/config`, `GET /api/stats` | Read/write detector+tracker config, system stats |
| `conditions.py` | `GET /api/conditions` | Time period (astral), sunrise/sunset, weather (OpenWeatherMap 15min cache) |
| `faces.py` | `GET /api/faces`, `POST /api/faces/preview`, `POST /api/faces/enroll`, `DELETE /api/faces/{id}`, `GET /api/faces/{id}/photo` | Proxy to face-recognizer :8081 |
| `unknowns.py` | `GET /api/unknowns`, `GET /api/unknowns/{id}/photo`, `POST /api/unknowns/{id}/label`, `DELETE /api/unknowns/clear`, `DELETE /api/unknowns/{id}` | Proxy to face-recognizer :8081, emits event on label |
| `zones.py` | `GET /api/zones`, `POST /api/zones`, `PUT /api/zones/{id}`, `DELETE /api/zones/{id}` | Zone CRUD in Redis hash |
| `notifications.py` | `GET /api/notifications/status`, `POST /api/notifications/test` | Telegram bot integration + feedback inline buttons |
| `feedback.py` | `GET /api/feedback`, `GET /api/feedback/stats`, `GET /api/feedback/rules`, `POST /api/feedback/{event_id}`, `POST /api/feedback/rules/{id}/toggle`, `DELETE /api/feedback/rules/{id}` | Self-learning feedback CRUD + suppression rules |
| `browse.py` | `GET /api/browse/days`, `GET /api/browse/days/{date}`, `GET /api/browse/snapshot/{date}/{filename}`, `GET /api/browse/faces` | Vehicle snapshot browser + enrolled faces gallery |
| `ai.py` | `GET /api/ai/status`, `GET /api/ai/config`, `POST /api/ai/config`, `POST /api/ai/chat`, `GET /api/ai/history`, `DELETE /api/ai/history`, `POST /api/ai/reset`, `GET /api/ai/reminders`, `GET /api/ai/clip/{filename}` | AI assistant: Ollama chat + 18 tools. Uses `think=False` + `keep_alive="30m"` to avoid Qwen3 thinking delay and cold-start. |
| `ai_tools.py` | (internal, called by `ai.py`) | 18 tool schemas + executor functions (events, faces, unknowns, feedback, retrain, live scene, capture snapshot/clip, weather, patterns, vehicles, zones, notifications, Telegram, reminders, status, review, events by date) |
| `ai_prompts.py` | (internal, called by `ai.py`) | Dynamic system prompt builder with live system info |
| `ai_state.py` | (internal) | Per-request media side-channel state (snapshot/clip stash, request UUID) |
| `bot_commands.py` | (internal, background task) | Telegram bot polling loop + command handlers (/snapshot, /clip, /status, /ask, /arm, /disarm, /who, /events, /help) |
| `telegram_access.py` | `GET /api/telegram/users`, `POST /api/telegram/users`, `DELETE /api/telegram/users/{id}`, `GET /api/telegram/access-log` | Telegram user CRUD + access audit log |

**Shared state pattern:** `routes/__init__.py` defines module-level variables (`r`, `r_bin`, `logger`, `FACE_API_URL`, `HD_FRAME_KEY`, all stream keys). `server.py` sets these before importing routers. Each route module does `import routes as ctx` to access them. `r` is the text Redis client (`decode_responses=True`) and `r_bin` is the binary client (`decode_responses=False`) for JPEG frame data.

---

## Frontend Deep Dive

All files in `services/dashboard/static/`. No build step — plain HTML/JS/CSS.

| File | Lines | Purpose |
|------|-------|---------|
| `index.html` | ~554 | Main dashboard: video feed, sidebar panels (events, faces, unknowns, zones, conditions, settings, notifications, auth). Enrollment wizard modal. Label modal. |
| `ai.html` | ~145 | AI assistant page (onboarding wizard + chat interface) |
| `telegram.html` | ~352 | Telegram Access Manager page (approved users + access log) |
| `login.html` | ~403 | Login page with animated pulsing eye icon, dark theme, fade-in form |
| `style.css` | ~2261 | Full dark theme, glassmorphism panels, responsive layout, zone editor styles, wizard overlay styles, event photo lightbox modal |
| `ai.css` | ~682 | AI assistant page styles (chat bubbles, onboarding wizard, tool status) |
| `app.js` | ~341 | Core: WebSocket connect (auto-reconnect 2s), FPS counter, settings sliders (debounced 300ms POST), notification status, `init()` orchestrator |
| `ai.js` | ~484 | AI chat client: onboarding wizard, message rendering (markdown + inline images), tool-call status display |
| `auth.js` | ~103 | Logout, change password/username, auth status display |
| `events.js` | ~356 | Polls `/api/events` every 2s, deduplicates by event ID, renders event cards with icons + clickable photo thumbnails (face photos for known users, camera snapshots for unknowns), lightbox modal for full-size viewing |
| `faces.js` | ~385 | Multi-angle enrollment wizard (5 angles: front/left/right/up/down), face gallery grouped by name, delete all angles for a person |
| `unknowns.js` | ~192 | Unknown faces gallery, label modal (dropdown of known names OR free text), bulk clear |
| `conditions.js` | ~167 | Fetches `/api/conditions` every 5min, renders time periods, sunrise/sunset, weather emoji mapping |
| `zones.js` | ~527 | Canvas overlay zone drawing (click-to-place polygon, double-click to close), drag-to-edit vertices, letterbox-aware coordinate normalization, zone list with color-coded alert levels |
| `feedback.js` | ~374 | Feedback review queue: verdict history, suppression rules, stats panel, quick-resolve actions |
| `browse.js` | ~173 | Vehicle snapshot browser: day picker, thumbnail grid, face gallery tab |
| `telegram_access.js` | ~223 | Telegram Access Manager: user list, approve/revoke, access log rendering |

**Initialization (`app.js init()`):**
1. Connect WebSocket
2. Load config (populate sliders)
3. Load zones, faces, unknowns
4. Start event polling (2s interval)
5. Start face/unknown refresh (30s), zone refresh (15s)
6. Check notification status

---

## Inter-Service Communication

| From | To | Mechanism | What |
|------|----|-----------|------|
| ingester → detector | Redis Stream (consumer group) | JPEG frames |
| detector → tracker | Redis Stream (consumer group) | Bboxes + keypoints |
| detector → recognizer | Redis Stream (separate consumer group) | Same detection stream |
| recognizer → tracker | Redis Hash (identity_state) | Name ↔ bbox mapping (polled every 2s) |
| recognizer → dashboard | Redis Hash (identity_state) | Same identity data |
| tracker → dashboard | Redis Hash (state) | Current tracked persons |
| tracker → dashboard | Redis Stream (events) | Semantic events |
| dashboard → detector | Redis Hash (config) | confidence_thresh via hot-reload |
| dashboard → tracker | Redis Hash (config) | iou_threshold, lost_timeout via hot-reload |
| dashboard → tracker | Redis Hash (zones) | Zone polygons + alert levels |
| dashboard → recognizer | **HTTP proxy** (port 8081) | Enrollment, face CRUD, unknowns |
| dashboard → Telegram | HTTPS API | Photo + caption notifications |
| browser → dashboard | WebSocket `/ws/live` | Live frame stream (downstream only) |
| browser → dashboard | REST `/api/*` | Config, events, faces, zones, auth |

**Important:** The dashboard → face-recognizer HTTP proxy is the only inter-service communication that bypasses Redis. This is because enrollment is a request/response pattern (user expects immediate feedback), not a fire-and-forget stream.

---

## Hot-Reload Config System

The dashboard writes config to `config:{camera_id}` Redis hash. Services poll this key:

| Service | Key(s) | Poll Frequency |
|---------|--------|----------------|
| pose-detector | `confidence_thresh` | Every 25 frames (~half second) |
| tracker | `iou_threshold`, `lost_timeout` | Every 10 messages |
| tracker | zone definitions | Every 10 seconds |
| tracker | identity state | Every 2 seconds |
| dashboard | zone cache for overlay | Every 5 seconds |

**No restart required.** The user drags a slider → 300ms debounce → POST /api/config → Redis HSET → next poll cycle picks it up.

---

## Authentication System

**Backend:** `routes/auth.py` — SQLite `auth.db` on Docker volume `auth-data`.

| Aspect | Implementation |
|--------|---------------|
| Password storage | SHA-256 with per-user random 16-byte salt |
| Session tokens | `username:timestamp:HMAC-SHA256(secret_key, username:timestamp)` |
| Session expiry | 24 hours |
| Cookie | `vl_session`, httponly, samesite=lax, path=/ |
| Secret key | Auto-generated on first boot, stored in `app_config` table, persists across restarts |
| Default credentials | `admin` / `admin` (created if users table is empty) |

**Exempt paths (no auth needed):** `/login.html`, `/api/auth/login`, `/api/auth/status`, `/style.css`, `/auth.js`, `/favicon.ico`

---

## Notification System

**Backend:** `routes/notifications.py` — Telegram Bot API via `httpx`.

| Event | Trigger | Rate Limited? | Photo Source |
|-------|---------|---------------|--------------|
| Person detected | `person_appeared` event | Yes (1 per 60s) | HD frame (fallback: sub-stream) + bbox highlight |
| Person identified | `person_identified` event | No (always important) | HD frame (fallback: sub-stream) + bbox highlight |
| Vehicle idle | `vehicle_idle` event | Yes (1 per 60s) | HD frame (fallback: sub-stream) |
| Face enrolled | Enrollment API success | No | Face thumbnail from face-recognizer |
| Test notification | Manual button click | No | HD frame (fallback: sub-stream) |

`get_latest_frame()` tries `frame_hd:{camera_id}` first for higher resolution, falling back to the sub-stream. `draw_bbox_on_frame()` scales bbox coordinates from sub-stream pixels to HD resolution when the HD frame is used.

**Architecture:** The dashboard runs a background `asyncio` task (`_event_notification_poller`) that does `XREAD` on the event stream in a thread executor (to avoid blocking the event loop). For every `person_appeared` event, it saves a camera snapshot to `/data/snapshots/` (used by the event feed thumbnails). When Telegram is configured and relevant events fire, it calls `send_photo()` which POSTs to Telegram's `sendPhoto` endpoint.

**Config:** `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` from `.env` → docker-compose environment.

---

## Docker Infrastructure

### docker-compose.yml

8 services, 7 named volumes:

| Volume | Mount Point | Purpose |
|--------|-------------|---------|
| `redis-data` | `/data` (redis) | Redis persistence (AOF) |
| `face-data` | `/data` (face-recognizer) | SQLite face DB + unknowns |
| `yolo-models` | `/root/.config/Ultralytics` | YOLOv8 model cache |
| `insightface-models` | `/root/.insightface` | InsightFace buffalo_l cache |
| `auth-data` | `/data` (dashboard) | Auth + feedback + AI SQLite databases |
| `snapshot-data` | `/data/snapshots` (dashboard) | Event + vehicle snapshots (persists across container restarts) |
| `ollama-models` | `/root/.ollama` (ollama) | Qwen 3 14B model weights (~9.3 GB) |

### Shared Contract Mounting

Every service that imports from `contracts/` mounts the project root's `contracts/` directory:

```yaml
volumes:
  - ./contracts:/app/contracts:ro
```

This ensures a single source of truth — change a stream key in `contracts/streams.py` and every service picks it up on next restart.

### GPU Access

`pose-detector`, `face-recognizer`, `vehicle-detector`, and `ollama` use the NVIDIA runtime:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Network

- Default bridge network for inter-container communication
- `camera-ingester` uses `network_mode: host` to reach the camera at 192.168.2.10
- Dashboard exposes port 8080 to host
- Face-recognizer exposes port 8081 (internal, proxied by dashboard)

---

## Test Suite

All tests in `tests/`. Run with: `pytest tests/ -v`

| Test File | What It Tests | Key Coverage |
|-----------|---------------|--------------|
| `test_actions.py` | `contracts/actions.py` | All 5 action classifications, edge cases, missing keypoints |
| `test_time_rules.py` | `contracts/time_rules.py` | All 4 time periods, `should_alert()` matrix, `point_in_polygon()` edge cases |
| `test_face_db.py` | `face_db.py` | Enroll, match, delete, unknowns, dedup, reconciliation, max limit |
| `test_feedback_db.py` | `feedback_db.py` | Feedback records, suppression rules, retrain, auto-rule generation |
| `test_tracker.py` | `tracker.py` | IoU computation, TrackedPerson state, PersonTracker.update(), debounce |
| `test_routes.py` | `routes/*.py` | Dashboard API endpoints: zones, config, events, auth, notifications (mocked Redis) |
| `test_vehicles.py` | Vehicle pipeline | Vehicle events, snapshot storage, idle detection, browse API |

---

## Current Status vs v2.md Plan

| Phase | Status | Notes |
|-------|--------|-------|
| **0: Hardware** | ✅ Complete | Camera, switch, NAS, PoE injector all working |
| **1: Camera + Redis** | ✅ Complete | RTSP → Redis at 15 FPS, reconnect logic |
| **2: YOLO + Tracker** | ✅ Complete | YOLOv8s-pose ~34ms, IoU tracking, events |
| **3: Dashboard** | ✅ Complete | Live feed, overlays, event feed, settings |
| **4: Actions** | ✅ Complete | 5 actions classified, debounce + sticky bias |
| **5: Face ReID** | ✅ Complete | InsightFace buffalo_l, multi-angle enrollment, sticky identity, unknowns |
| **6: Zones + Alerts** | ✅ Complete | Zone drawing, time rules, dead zones, Telegram notifications. Remaining: event clip recording |
| **6.1: Auth** | ✅ Complete | Login page, cookie sessions, change password |
| **6.2: Vehicles** | ✅ Complete | YOLOv8s vehicle detection, snapshots, idle alerts, live overlay bboxes |
| **6.5: Self-Learning** | ✅ Complete | Feedback DB, Telegram inline buttons, suppression rules, review queue, dashboard widget |
| **7: AI Assistant** | ✅ Complete | Ollama + Qwen 3 14B, onboarding wizard, chat UI, 18 tools (query events/faces/unknowns/feedback/patterns, live scene, capture snapshot with weather+scene description, capture 5-second video clip in chat, weather, browse vehicles/zones/notification history, retrain rules, send Telegram, schedule reminders, system status) |
| **7.5: Telegram Access Manager** | ✅ Complete | Web-based user management page. Approve/revoke Telegram users, view access log. Unauthorized bot access emits `unauthorized_access` events to event stream |

**Minor remaining from Phase 6:** Event clip recording (10s clips around detections, saved to QNAP via FTP/NFS).

---

## Phase 6.5: Self-Learning Feedback Loop (Implemented)

### What it adds

An alert suppression engine that learns from user feedback over time, reducing false notifications from ~15/day to ~2/day.

### Components built

1. **`feedback_db.py`** — SQLite database for feedback records + suppression rules
2. **`routes/feedback.py`** — REST API for viewing feedback, managing rules, submitting verdicts
3. **Telegram inline buttons** — ✅/❌/🏷️ on notifications, callbacks store feedback
4. **Suppression rules** — auto-generated when patterns exceed thresholds (3 identity false alarms, 5 zone+time false alarms)
5. **AI retrain tool** — the AI assistant can re-scan all feedback and regenerate rules on demand
6. **`feedback.js`** — dashboard review queue UI

### How it stays modular

- Suppression engine is a **pure function** `should_suppress(identity, zone, time_period)` — no Redis coupling
- Feedback storage is a **separate SQLite DB** (not mixed with face DB or auth DB)
- Review queue is a **new routes module** (`routes/feedback.py`) — follows existing pattern
- Telegram inline buttons use Telegram's callback_query API — existing `notifications.py` extended
- All existing services remain **unchanged** — suppression happens in `notifications.py` before sending Telegram alerts

### How it stays secure

- No retraining of YOLO/InsightFace (those are frozen foundation models)
- Suppression model is deterministic (threshold-based) — trains instantly, no GPU needed
- Review queue protected by existing auth middleware

---

## Modularity & Security Principles

### Adding a new camera

1. Assign static IP (192.168.2.11, etc.)
2. Add new ingester service in docker-compose.yml with `CAMERA_ID=backyard`
3. `docker compose up -d camera-ingester-2`
4. All downstream services auto-discover via Redis key pattern `frames:backyard`

### Adding a new detector

1. Create new service that reads `frames:*`, publishes to `detections:mytype:*`
2. Add docker-compose entry
3. Tracker auto-discovers if it's configured to read the new detection stream
4. Zero changes to existing services

### Fault isolation

- Any service can crash without affecting others
- Redis Streams persist — crashed service catches up from last consumer group cursor
- Dashboard shows "offline" states for disconnected components
- Face recognition failure → person detection still works, just no names

### Security

- All traffic is LAN-only (no port forwarding to internet)
- Auth protects dashboard with cookie sessions
- Telegram uses HTTPS to external API (only outbound connection)
- Face embeddings are 512-dim float vectors — cannot be reversed to reconstruct a face
- No audio recording (Ontario privacy law compliance)
- Dead zones prevent tracking in specific areas (e.g., neighbor's property)

---

## Extensibility Roadmap

### Tier 1 — Make It Smarter (low effort, high impact)

| Feature | Effort | Impact | Description |
|---------|--------|--------|-------------|
| **Weather + time in system prompt** | 🟢 Low | 🟢 High | ✅ Done. Conditions data and current time already injected into AI context. Snapshot tool includes weather. |
| **Recent events in context** | 🟢 Low | 🟢 High | Pre-load last 5 events into system prompt so AI can proactively mention recent activity without tool calls |
| **Daily briefing** | 🟡 Medium | 🟢 High | Scheduled Telegram summary: "Today: 12 events, 3 unknowns, busiest at 2pm, clear weather" |
| **Rule suggestions** | 🟡 Medium | 🟢 High | AI proactively suggests suppression rules after seeing false alarm patterns |

### Tier 2 — Proactive Intelligence

| Feature | Effort | Impact | Description |
|---------|--------|--------|-------------|
| **Anomaly detection** | 🔴 High | 🟢 High | Track "normal" patterns and flag deviations (e.g., "John usually arrives by 5pm — not home yet") |
| **Event correlation** | 🟡 Medium | 🟡 Medium | "Person appeared → 30s later → vehicle" = likely delivery, auto-label as routine |
| **Auto-escalation** | 🟡 Medium | 🟢 High | Multiple unknowns in dead zone during night → high-priority Telegram without waiting for user query |
| **Conversation memory** | 🟡 Medium | 🟡 Medium | AI remembers preferences ("always alert for driveway") persisted in ai_db |

### Tier 3 — Truly Autonomous

| Feature | Effort | Impact | Description |
|---------|--------|--------|-------------|
| **Multi-camera reasoning** | 🔴 High | 🟢 High | Correlate front + back camera: track movement through property |
| **Voice integration** | 🟡 Medium | 🟡 Medium | Expose AI via API so Home Assistant or custom voice assistant can query it |
| **NAS recording** | 🟢 Low | 🟢 High | QNAP FTP/NFS for continuous recording + event clip storage (weeks of history) |
| **Event clip recording** | 🟡 Medium | 🟢 High | 10-second clips around detections, saved alongside snapshots |

### Architecture Scaling Limits

With the current architecture, the system can reasonably scale to:
- **3–4 cameras** (limited by GPU VRAM: YOLO + InsightFace + Qwen compete for 24 GB)
- **30-day event history** (Redis memory; beyond that, offload to PostgreSQL)
- **50+ suppression rules** (deterministic engine scales linearly)
- **20+ AI tools** (no performance degradation with current Ollama setup)

---

## File Index

```
vision-labsv1/
├── .env                          # Secrets (camera password, Telegram tokens)
├── .env.example                  # Template for .env
├── .gitignore                    # Python, Docker, IDE ignores
├── ARCHITECTURE.md               # THIS FILE
├── README.md                     # Project overview
├── docker-compose.yml            # All 8 services + 7 volumes
├── v1.md                         # Original brainstorm
├── v2.md                         # Refined build plan
│
├── contracts/                    # Shared API contract (single source of truth)
│   ├── __init__.py               # Package docstring
│   ├── streams.py                # Redis key templates + data schemas
│   ├── actions.py                # Action classifier (math-only)
│   └── time_rules.py             # Time periods + zone alert rules (astral)
│
├── services/
│   ├── camera-ingester/
│   │   ├── Dockerfile
│   │   ├── ingester.py           # RTSP → Redis
│   │   └── requirements.txt
│   │
│   ├── pose-detector/
│   │   ├── Dockerfile
│   │   ├── detector.py           # YOLOv8s-pose inference
│   │   └── requirements.txt
│   │
│   ├── tracker/
│   │   ├── Dockerfile
│   │   ├── tracker.py            # IoU tracking + events
│   │   └── requirements.txt
│   │
│   ├── face-recognizer/
│   │   ├── Dockerfile
│   │   ├── recognizer.py         # InsightFace + REST API
│   │   ├── face_db.py            # SQLite face database
│   │   └── requirements.txt
│   │
│   ├── vehicle-detector/
│   │   ├── Dockerfile
│   │   ├── detector.py           # YOLOv8s vehicle detection
│   │   └── requirements.txt
│   │
│   └── dashboard/
│       ├── Dockerfile
│       ├── server.py             # FastAPI + WebSocket (~1075 lines)
│       ├── feedback_db.py        # Feedback + suppression rules (SQLite, ~572 lines)
│       ├── ai_db.py              # AI config + reminders + chat history (SQLite, ~236 lines)
│       ├── requirements.txt
│       ├── routes/
│       │   ├── __init__.py       # Shared state container (r, r_bin, logger, keys)
│       │   ├── auth.py           # Login/logout/password (~311 lines)
│       │   ├── events.py         # Event feed + snapshot API (~114 lines)
│       │   ├── config.py         # Config + stats API (~76 lines)
│       │   ├── conditions.py     # Time + weather API (~110 lines)
│       │   ├── faces.py          # Face enrollment proxy (~108 lines)
│       │   ├── unknowns.py       # Unknown faces proxy (~160 lines)
│       │   ├── zones.py          # Zone CRUD API (~110 lines)
│       │   ├── notifications.py  # Telegram integration (~815 lines)
│       │   ├── feedback.py       # Feedback + suppression rules API (~123 lines)
│       │   ├── browse.py         # Vehicle snapshot browser + faces gallery (~158 lines)
│       │   ├── ai.py             # AI assistant chat endpoint (~309 lines)
│       │   ├── ai_tools.py       # 21 AI tool schemas + executors (~1350 lines)
│       │   ├── ai_prompts.py     # Dynamic system prompt builder (~118 lines)
│       │   ├── ai_state.py       # Per-request media side-channel (~94 lines)
│       │   ├── bot_commands.py   # Telegram bot polling + 11 commands (~1170 lines)
│       │   └── telegram_access.py # Telegram user CRUD + access log (~105 lines)
│       └── static/
│           ├── index.html        # Main dashboard layout (~554 lines)
│           ├── ai.html           # AI assistant (onboarding + chat, ~145 lines)
│           ├── telegram.html     # Telegram Access Manager (~352 lines)
│           ├── login.html        # Login page (~403 lines)
│           ├── style.css         # Full CSS (~2261 lines)
│           ├── ai.css            # AI page styles (~682 lines)
│           ├── app.js            # Core + WebSocket + init (~341 lines)
│           ├── ai.js             # AI chat + wizard logic (~484 lines)
│           ├── auth.js           # Auth UI (~103 lines)
│           ├── events.js         # Event feed (~356 lines)
│           ├── faces.js          # Face enrollment wizard (~385 lines)
│           ├── unknowns.js       # Unknown faces gallery (~192 lines)
│           ├── conditions.js     # Conditions panel (~167 lines)
│           ├── zones.js          # Zone editor + canvas (~527 lines)
│           ├── browse.js         # Vehicle snapshot browser (~173 lines)
│           ├── feedback.js       # Feedback review queue (~374 lines)
│           └── telegram_access.js # Telegram Access Manager UI (~223 lines)
│
└── tests/
    ├── test_actions.py           # Action classifier tests
    ├── test_time_rules.py        # Time rules + PIP tests
    ├── test_face_db.py           # Face DB integration tests
    ├── test_feedback_db.py       # Feedback + suppression tests
    ├── test_tracker.py           # Tracker algorithm tests
    ├── test_routes.py            # Dashboard API endpoint tests
    └── test_vehicles.py          # Vehicle pipeline tests
```

