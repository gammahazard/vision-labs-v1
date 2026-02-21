"""
services/dashboard/server.py — FastAPI backend for the Vision Labs dashboard.

PURPOSE:
    Serves the web dashboard and provides real-time data to the browser:
    - WebSocket streaming of live camera frames with detection data
    - REST API routes (modularized into routes/ package)

RELATIONSHIPS:
    - Reads from: Redis streams (frames, detections, events, state)
    - Writes to: Redis config key (when user adjusts settings)
    - Serves: static frontend files (index.html, style.css, *.js)
    - Used by: browser at http://localhost:8080

DATA FLOW:
    Redis → THIS SERVICE (WebSocket) → Browser (renders frames + overlays)
    Browser (settings change) → THIS SERVICE (REST) → Redis config key → Detector reads it

MODULES:
    routes/events.py      — GET /api/events
    routes/config.py      — GET/POST /api/config, GET /api/stats
    routes/conditions.py  — GET /api/conditions (time + weather)
    routes/faces.py       — Face enrollment proxies (5 endpoints)
    routes/unknowns.py    — Unknown face proxies (5 endpoints)
    routes/zones.py       — Zone CRUD (3 endpoints)
"""

import asyncio
import base64
import json
import os
import time
import logging
from datetime import datetime

import cv2
import numpy as np
import redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# Import stream key definitions from contracts (single source of truth)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "contracts"))
from streams import (
    FRAME_STREAM as _FRAME_TMPL,
    DETECTION_STREAM as _DET_TMPL,
    EVENT_STREAM as _EVT_TMPL,
    STATE_KEY as _STATE_TMPL,
    CONFIG_KEY as _CFG_TMPL,
    IDENTITY_KEY as _IDKEY_TMPL,
    ZONE_KEY as _ZONE_TMPL,
    HD_FRAME_KEY as _HD_TMPL,
    VEHICLE_STREAM as _VEH_DET_TMPL,
    stream_key,
)


# ---------------------------------------------------------------------------
# IoU helper — used by WebSocket overlay to match bboxes consistently
# ---------------------------------------------------------------------------
def _bbox_iou(box_a: list, box_b: list) -> float:
    """Compute IoU between two [x1, y1, x2, y2] bounding boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAMERA_ID = os.getenv("CAMERA_ID", "front_door")
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8080"))
FACE_API_URL = os.getenv("FACE_API_URL", "http://127.0.0.1:8081")

# Redis keys — resolved from contracts/streams.py
FRAME_STREAM = stream_key(_FRAME_TMPL, camera_id=CAMERA_ID)
DETECTION_STREAM = stream_key(_DET_TMPL, detector_type="pose", camera_id=CAMERA_ID)
EVENT_STREAM = stream_key(_EVT_TMPL, camera_id=CAMERA_ID)
STATE_KEY = stream_key(_STATE_TMPL, camera_id=CAMERA_ID)
CONFIG_KEY = stream_key(_CFG_TMPL, camera_id=CAMERA_ID)
IDENTITY_KEY = stream_key(_IDKEY_TMPL, camera_id=CAMERA_ID)
ZONE_KEY = stream_key(_ZONE_TMPL, camera_id=CAMERA_ID)
HD_FRAME_KEY = stream_key(_HD_TMPL, camera_id=CAMERA_ID)
VEHICLE_DET_STREAM = stream_key(_VEH_DET_TMPL, camera_id=CAMERA_ID)

# Default config values (written to Redis on first startup if not present)
DEFAULT_CONFIG = {
    "confidence_thresh": "0.5",
    "iou_threshold": "0.3",
    "lost_timeout": "5.0",
    "target_fps": "5",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dashboard")


# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Vision Labs Dashboard")

# Redis connection (sync client for REST endpoints)
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Auth database path (Docker volume for persistence)
AUTH_DB_PATH = os.getenv("AUTH_DB_PATH", "/data/auth.db")


# ---------------------------------------------------------------------------
# Inject shared state into routes package, then include routers
# ---------------------------------------------------------------------------
import routes as route_ctx

route_ctx.r = r
route_ctx.logger = logger
route_ctx.FACE_API_URL = FACE_API_URL
route_ctx.EVENT_STREAM = EVENT_STREAM
route_ctx.FRAME_STREAM = FRAME_STREAM
route_ctx.DETECTION_STREAM = DETECTION_STREAM
route_ctx.STATE_KEY = STATE_KEY
route_ctx.CONFIG_KEY = CONFIG_KEY
route_ctx.IDENTITY_KEY = IDENTITY_KEY
route_ctx.ZONE_KEY = ZONE_KEY
route_ctx.AUTH_DB_PATH = AUTH_DB_PATH

# Vehicle snapshot disk storage (day-organized)
VEHICLE_SNAPSHOT_DIR = os.path.join(os.environ.get("SNAPSHOT_DIR", "/data/snapshots"), "vehicles")
os.makedirs(VEHICLE_SNAPSHOT_DIR, exist_ok=True)
route_ctx.VEHICLE_SNAPSHOT_DIR = VEHICLE_SNAPSHOT_DIR
route_ctx.CAMERA_ID = CAMERA_ID

from routes.events import router as events_router
from routes.config import router as config_router
from routes.conditions import router as conditions_router
from routes.faces import router as faces_router
from routes.unknowns import router as unknowns_router
from routes.zones import router as zones_router
from routes.notifications import router as notifications_router
from routes.auth import router as auth_router, init_auth_db, validate_session
from routes.browse import router as browse_router

app.include_router(events_router)
app.include_router(config_router)
app.include_router(conditions_router)
app.include_router(faces_router)
app.include_router(unknowns_router)
app.include_router(zones_router)
app.include_router(notifications_router)
app.include_router(auth_router)
app.include_router(browse_router)


# ---------------------------------------------------------------------------
# Auth Middleware — Protect all routes except login page and auth API
# ---------------------------------------------------------------------------
# Paths that don't require authentication
_AUTH_EXEMPT = {
    "/login.html", "/api/auth/login", "/api/auth/status",
    "/style.css", "/auth.js", "/favicon.ico",
}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Redirect unauthenticated requests to the login page."""
    path = request.url.path

    # Allow exempt paths through
    if path in _AUTH_EXEMPT:
        return await call_next(request)

    # Check session cookie
    token = request.cookies.get("vl_session")
    username = validate_session(token)

    if username:
        return await call_next(request)

    # Not authenticated — redirect browser requests, 401 for API
    if path.startswith("/api/") or path == "/ws":
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    return RedirectResponse("/login.html")


# ---------------------------------------------------------------------------
# Startup — Initialize default config if not set
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    """Initialize auth DB, write default config to Redis, start background tasks."""
    # Initialize auth database (creates default admin/admin if empty)
    init_auth_db()
    logger.info("Auth database initialized")

    existing = r.hgetall(CONFIG_KEY)
    if not existing:
        r.hset(CONFIG_KEY, mapping=DEFAULT_CONFIG)
        logger.info(f"Initialized default config in {CONFIG_KEY}")
    else:
        logger.info(f"Config already exists in {CONFIG_KEY}: {existing}")

    # Start background event notification poller
    asyncio.create_task(_event_notification_poller())

    logger.info(f"Dashboard ready at http://localhost:{DASHBOARD_PORT}")


async def _event_notification_poller():
    """
    Background task: poll the event stream for new events.
    Two responsibilities:
      1. ALWAYS save a camera snapshot for person_appeared events (for the event feed)
      2. Optionally send Telegram notifications (when configured)

    IMPORTANT: r.xread(block=...) is a synchronous blocking call.
    We run it in a thread executor to avoid blocking the asyncio event loop,
    which would starve the WebSocket frame streaming.
    """
    from routes.notifications import (
        notify_person_detected, notify_person_identified,
        notify_vehicle_idle, is_configured, get_latest_frame,
    )

    # Ensure snapshot directory exists
    SNAPSHOT_DIR = os.path.join(os.environ.get("SNAPSHOT_DIR", "/data/snapshots"))
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    SNAPSHOT_MAX_AGE = 7200  # 2 hours

    last_id = "$"  # Only process new events from this point forward
    logger.info(f"Event poller started — snapshots → {SNAPSHOT_DIR}, vehicles → {VEHICLE_SNAPSHOT_DIR}")

    loop = asyncio.get_event_loop()

    def _save_snapshot(event_id: str):
        """Grab the latest camera frame and save it as a JPEG for this event."""
        try:
            frame = get_latest_frame()
            if frame:
                # Redis event IDs contain ":" — replace for safe filenames
                safe_id = event_id.replace(":", "-")
                path = os.path.join(SNAPSHOT_DIR, f"{safe_id}.jpg")
                with open(path, "wb") as f:
                    f.write(frame)
        except Exception as e:
            logger.debug(f"Snapshot save failed for {event_id}: {e}")

    def _cleanup_old_snapshots():
        """Remove snapshots older than SNAPSHOT_MAX_AGE."""
        try:
            now = time.time()
            for fname in os.listdir(SNAPSHOT_DIR):
                fpath = os.path.join(SNAPSHOT_DIR, fname)
                if os.path.isfile(fpath) and now - os.path.getmtime(fpath) > SNAPSHOT_MAX_AGE:
                    os.remove(fpath)
        except Exception:
            pass

    def _save_vehicle_snapshot(snapshot_key: str, event_data: dict):
        """
        Pull vehicle snapshot JPEG from Redis and save to disk.
        Organized as: vehicles/YYYY-MM-DD/HH-MM-SS_class.jpg
        """
        try:
            # Use a separate raw-bytes Redis client (decode_responses=False)
            r_bin = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
            jpeg_data = r_bin.get(snapshot_key.encode() if isinstance(snapshot_key, str) else snapshot_key)
            if not jpeg_data:
                return

            # Parse timestamp from event data
            ts = float(event_data.get("timestamp", time.time()))
            dt = datetime.fromtimestamp(ts)
            day_str = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H-%M-%S")
            vehicle_class = event_data.get("vehicle_class", "vehicle")

            # Create day folder and write file
            day_dir = os.path.join(VEHICLE_SNAPSHOT_DIR, day_str)
            os.makedirs(day_dir, exist_ok=True)
            path = os.path.join(day_dir, f"{time_str}_{vehicle_class}.jpg")
            with open(path, "wb") as f:
                f.write(jpeg_data)

            logger.debug(f"Vehicle snapshot saved: {path}")
        except Exception as e:
            logger.debug(f"Vehicle snapshot save failed: {e}")

    cleanup_counter = 0

    while True:
        try:
            # Run blocking xread in a thread so we don't block the event loop
            entries = await loop.run_in_executor(
                None, lambda: r.xread({EVENT_STREAM: last_id}, count=10, block=2000)
            )
            if entries:
                for stream_name, messages in entries:
                    for msg_id, data in messages:
                        last_id = msg_id
                        event_type = data.get("event_type", "")

                        if event_type == "person_appeared":
                            # Always save snapshot (for event feed thumbnails)
                            await loop.run_in_executor(None, _save_snapshot, msg_id)
                            # Optionally send Telegram notification
                            if is_configured():
                                await notify_person_detected(data)

                        elif event_type == "person_identified":
                            if is_configured():
                                await notify_person_identified(data)

                        elif event_type == "vehicle_detected":
                            # Save vehicle snapshot to disk in day folder
                            snapshot_key = data.get("snapshot_key", "")
                            if snapshot_key:
                                await loop.run_in_executor(
                                    None, _save_vehicle_snapshot, snapshot_key, data
                                )

                        elif event_type == "vehicle_idle":
                            # Save snapshot to disk + send Telegram notification
                            snapshot_key = data.get("snapshot_key", "")
                            if snapshot_key:
                                await loop.run_in_executor(
                                    None, _save_vehicle_snapshot, snapshot_key, data
                                )
                            if is_configured():
                                await notify_vehicle_idle(data)

            # Periodic cleanup every ~100 iterations (~200s)
            cleanup_counter += 1
            if cleanup_counter >= 100:
                cleanup_counter = 0
                await loop.run_in_executor(None, _cleanup_old_snapshots)

        except Exception as e:
            logger.warning(f"Event notification poller error: {e}")
            await asyncio.sleep(5)

        await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# WebSocket — Live Frame + Detection Streaming
# ---------------------------------------------------------------------------
@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """
    Stream live camera frames with detection overlays to the browser.

    The browser receives:
    - Base64-encoded JPEG frame with bounding boxes drawn on it
    - Detection metadata (person count, person IDs, etc.)
    - Current state (who's in frame right now)

    We read the LATEST frame and its matching detection from Redis,
    draw overlays, encode as JPEG, and send to the browser.
    """
    await ws.accept()
    logger.info("WebSocket client connected")

    # Use a separate Redis connection for binary frame data
    r_bin = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)

    last_frame_id = "$"  # Start from latest
    target_fps = 10  # Dashboard FPS (higher than ingester for smooth playback)
    frame_interval = 1.0 / target_fps

    # Stream mode — "sd" (default, with overlays) or "hd" (raw main stream)
    stream_mode = "sd"

    try:
        while True:
            loop_start = time.time()

            # --- Check for incoming messages (non-blocking) ---
            try:
                msg_raw = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
                try:
                    msg = json.loads(msg_raw)
                    if msg.get("action") == "switch_stream":
                        new_mode = msg.get("stream", "sd")
                        if new_mode in ("sd", "hd"):
                            stream_mode = new_mode
                            logger.info(f"WebSocket stream mode: {stream_mode}")
                            await ws.send_json({"type": "stream_mode", "mode": stream_mode})
                except json.JSONDecodeError:
                    pass
            except asyncio.TimeoutError:
                pass

            try:
                # === HD MODE: serve raw high-res frame from Redis key ===
                if stream_mode == "hd":
                    hd_bytes = r_bin.get(HD_FRAME_KEY)
                    if not hd_bytes:
                        # No HD frame available — fall back briefly
                        await asyncio.sleep(0.1)
                        continue

                    frame_b64 = base64.b64encode(hd_bytes).decode("ascii")
                    await ws.send_json({
                        "type": "frame",
                        "frame": frame_b64,
                        "frame_number": "0",
                        "num_detections": 0,
                        "inference_ms": "--",
                        "num_people": "--",
                        "timestamp": time.time(),
                        "hd": True,
                    })

                    elapsed = time.time() - loop_start
                    await asyncio.sleep(max(0, frame_interval - elapsed))
                    continue

                # === SD MODE: normal frame with detection overlays ===

                # Get the latest frame from the stream
                frames = r_bin.xrevrange(FRAME_STREAM, count=1)
                if not frames:
                    await asyncio.sleep(0.1)
                    continue

                frame_id, frame_data = frames[0]
                frame_bytes = frame_data[b"frame"]
                frame_number = frame_data.get(b"frame_number", b"0").decode()

                # Get the latest detection
                detections_raw = r_bin.xrevrange(
                    DETECTION_STREAM.encode(), count=1
                )
                detections = []
                inference_ms = "0"
                if detections_raw:
                    det_data = detections_raw[0][1]
                    det_json = det_data.get(b"detections", b"[]").decode()
                    detections = json.loads(det_json)
                    inference_ms = det_data.get(b"inference_ms", b"0").decode()

                # Decode the JPEG frame to draw overlays
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    await asyncio.sleep(0.1)
                    continue

                # Get identity labels from face recognizer
                identity_names = []
                try:
                    id_state = r.hgetall(IDENTITY_KEY)
                    if id_state:
                        id_json = id_state.get("identities", "[]")
                        identity_names = json.loads(id_json)
                except Exception:
                    pass

                # Get tracker state for action labels and person IDs
                tracker_persons = []
                try:
                    state = r.hgetall(STATE_KEY)
                    if state:
                        tracker_persons = json.loads(state.get("persons", state.get("people", "[]")))
                except Exception:
                    pass

                # --- Sticky Identity Logic ---
                # Once a face is identified, stick the name to that person's bbox
                # until they leave the frame entirely.
                if not hasattr(websocket_live, '_sticky_identities'):
                    websocket_live._sticky_identities = {}  # person_id → name

                # Update sticky cache with any new identifications this frame
                for ident in identity_names:
                    id_bbox = ident.get("bbox", [])
                    id_name = ident.get("name", "Unknown")
                    if id_name == "Unknown" or len(id_bbox) != 4:
                        continue
                    # Match identity bbox to a tracker person via IoU
                    for tp in tracker_persons:
                        tp_bbox = tp.get("bbox", [])
                        tp_pid = tp.get("person_id", "")
                        if len(tp_bbox) == 4 and tp_pid:
                            iou = _bbox_iou(id_bbox, tp_bbox)
                            if iou > 0.2:
                                websocket_live._sticky_identities[tp_pid] = id_name
                                break

                # Prune sticky identities for persons no longer tracked
                active_pids = {tp.get("person_id", "") for tp in tracker_persons}
                for pid in list(websocket_live._sticky_identities.keys()):
                    if pid not in active_pids:
                        del websocket_live._sticky_identities[pid]

                # Draw bounding boxes and labels on the frame
                for det in detections:
                    bbox = det.get("bbox", [])
                    conf = det.get("confidence", 0)
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = [int(v) for v in bbox]

                        # Match detection bbox to a tracker person for ID + action
                        person_name = None
                        action = ""
                        for tp in tracker_persons:
                            tp_bbox = tp.get("bbox", [])
                            if len(tp_bbox) == 4:
                                iou = _bbox_iou(
                                    [float(v) for v in tp_bbox],
                                    [float(x1), float(y1), float(x2), float(y2)]
                                )
                                if iou > 0.3:
                                    action = tp.get("action", "")
                                    tp_pid = tp.get("person_id", "")
                                    # Check sticky identity cache
                                    if tp_pid in websocket_live._sticky_identities:
                                        person_name = websocket_live._sticky_identities[tp_pid]
                                    break

                        # If no sticky identity, check live identity this frame
                        if not person_name:
                            for ident in identity_names:
                                id_bbox = ident.get("bbox", [])
                                if len(id_bbox) == 4:
                                    iou = _bbox_iou(
                                        [float(v) for v in id_bbox],
                                        [float(x1), float(y1), float(x2), float(y2)]
                                    )
                                    if iou > 0.3:
                                        person_name = ident.get("name", "Unknown")
                                        break

                        # Color: cyan for identified, green for unknown
                        if person_name and person_name != "Unknown":
                            color = (255, 200, 0)  # Cyan (BGR)
                            label = f"{person_name} {conf:.0%}"
                        else:
                            color = (0, 255, 0)  # Green
                            label = f"Person {conf:.0%}"

                        # Append action if available
                        if action and action not in ("unknown", ""):
                            label += f" · {action}"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label_size = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )[0]
                        # Background rectangle for label
                        cv2.rectangle(
                            frame,
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0] + 4, y1),
                            color,
                            -1,
                        )
                        cv2.putText(
                            frame,
                            label,
                            (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 0),
                            2,
                        )

                        # Draw keypoints if available
                        keypoints = det.get("keypoints", [])
                        for kp in keypoints:
                            if len(kp) >= 3 and kp[2] > 0.3:  # Confidence > 30%
                                cx, cy = int(kp[0]), int(kp[1])
                                cv2.circle(frame, (cx, cy), 3, (0, 200, 255), -1)

                # Draw vehicle bounding boxes (orange)
                try:
                    veh_raw = r_bin.xrevrange(
                        VEHICLE_DET_STREAM.encode(), count=1
                    )
                    if veh_raw:
                        veh_data = veh_raw[0][1]
                        veh_json = veh_data.get(b"detections", b"[]").decode()
                        veh_detections = json.loads(veh_json)
                        for vdet in veh_detections:
                            vbbox = vdet.get("bbox", [])
                            vconf = vdet.get("confidence", 0)
                            vclass = vdet.get("class_name", "vehicle")
                            if len(vbbox) == 4:
                                vx1, vy1, vx2, vy2 = [int(v) for v in vbbox]
                                vcolor = (0, 140, 255)  # Orange (BGR)
                                vlabel = f"{vclass} {vconf:.0%}"
                                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), vcolor, 2)
                                vlabel_size = cv2.getTextSize(
                                    vlabel, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                                )[0]
                                cv2.rectangle(
                                    frame,
                                    (vx1, vy1 - vlabel_size[1] - 10),
                                    (vx1 + vlabel_size[0] + 4, vy1),
                                    vcolor,
                                    -1,
                                )
                                cv2.putText(
                                    frame,
                                    vlabel,
                                    (vx1 + 2, vy1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0, 0, 0),
                                    2,
                                )
                except Exception:
                    pass  # Vehicle stream may not be available

                # Draw zone overlays on the frame
                try:
                    if not hasattr(websocket_live, '_zone_cache'):
                        websocket_live._zone_cache = {}
                        websocket_live._zone_cache_time = 0

                    # Refresh zone cache every 5 seconds
                    now_ts = time.time()
                    if now_ts - websocket_live._zone_cache_time > 5:
                        raw = r.hgetall(ZONE_KEY)
                        websocket_live._zone_cache = {
                            k: json.loads(v) for k, v in raw.items()
                        } if raw else {}
                        websocket_live._zone_cache_time = now_ts

                    h, w = frame.shape[:2]
                    for zone_id, zone in websocket_live._zone_cache.items():
                        pts_norm = zone.get("points", [])
                        if len(pts_norm) < 3:
                            continue

                        # Convert normalized coords to pixel coords
                        pts = np.array(
                            [[int(p[0] * w), int(p[1] * h)] for p in pts_norm],
                            dtype=np.int32,
                        )

                        # Zone color by alert level (BGR)
                        alert_level = zone.get("alert_level", "log_only")
                        zone_colors = {
                            "always": (0, 0, 220),       # Red
                            "night_only": (0, 140, 255),  # Orange
                            "log_only": (200, 160, 60),   # Blue
                            "ignore": (100, 100, 100),    # Gray
                            "dead_zone": (40, 40, 40),    # Dark gray/black
                        }
                        color = zone_colors.get(alert_level, (200, 160, 60))

                        # Semi-transparent fill
                        overlay = frame.copy()
                        cv2.fillPoly(overlay, [pts], color)
                        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

                        # Zone border
                        cv2.polylines(frame, [pts], True, color, 2)

                        # Zone name label
                        name = zone.get("name", zone_id)
                        cx = int(np.mean(pts[:, 0]))
                        cy = int(np.mean(pts[:, 1]))
                        label_size = cv2.getTextSize(
                            name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                        )[0]
                        cv2.rectangle(
                            frame,
                            (cx - label_size[0] // 2 - 4, cy - label_size[1] // 2 - 4),
                            (cx + label_size[0] // 2 + 4, cy + label_size[1] // 2 + 4),
                            color,
                            -1,
                        )
                        cv2.putText(
                            frame,
                            name,
                            (cx - label_size[0] // 2, cy + label_size[1] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )
                except Exception as e:
                    logger.debug(f"Zone overlay error: {e}")

                # Encode frame back to JPEG for sending
                _, jpeg_buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )

                # Get current state
                state = r.hgetall(STATE_KEY)

                # Send frame + metadata as JSON
                frame_b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")

                message = {
                    "type": "frame",
                    "frame": frame_b64,
                    "frame_number": frame_number,
                    "num_detections": len(detections),
                    "inference_ms": inference_ms,
                    "num_people": state.get("num_people", "0"),
                    "timestamp": time.time(),
                }

                await ws.send_json(message)

            except redis.ConnectionError:
                logger.warning("Redis connection lost in WebSocket loop")
                await asyncio.sleep(1)
                continue

            # Throttle to target FPS
            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_interval - elapsed)
            await asyncio.sleep(sleep_time)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# ---------------------------------------------------------------------------
# Static Files — Serve frontend
# ---------------------------------------------------------------------------
# Mount AFTER API routes so /api/* takes priority
app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=DASHBOARD_PORT, log_level="info")
