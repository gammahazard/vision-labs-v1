"""
services/tracker/tracker.py — Tracks people across frames and publishes semantic events.

PURPOSE:
    The pose detector sees each frame independently — it doesn't know if the
    person in frame #500 is the same person in frame #501. This service solves
    that by comparing detections across consecutive frames using bounding box
    overlap (IoU — Intersection over Union).

    It turns raw detections into meaningful events:
    - "Person appeared" (new person entered the frame)
    - "Person left" (person hasn't been seen for N seconds)
    - Tracks duration, approximate direction, and assigns persistent IDs

RELATIONSHIPS:
    - Reads from: Redis Stream "detections:pose:{camera_id}" (published by pose-detector)
    - Writes to: Redis Stream "events:{camera_id}" (consumed by dashboard / rule engine)
    - Updates: Redis Key "state:{camera_id}" (current scene snapshot for dashboard)
    - Stream keys defined in: contracts/streams.py

DATA FLOW:
    pose-detector → [detections:pose:front_door] → THIS SERVICE → [events:front_door]
                                                                 → [state:front_door]

TRACKING METHOD:
    Simple IoU (Intersection over Union) matching:
    - For each new detection, compute overlap with every tracked person's last bbox
    - If overlap > threshold (50%), it's the same person → update their state
    - If no match, it's a new person → assign a new ID
    - If a tracked person has no match for N seconds → emit "person_left" event

    This is intentionally simple. Phase 5 adds face-based re-identification
    for recognizing people who leave and come back.

CONFIG (environment variables):
    CAMERA_ID           — Which camera to track (default: "front_door")
    REDIS_HOST          — Redis server hostname (default: "127.0.0.1")
    REDIS_PORT          — Redis server port (default: 6379)
    IOU_THRESHOLD       — Min overlap to consider same person (default: 0.3)
    LOST_TIMEOUT        — Seconds before a lost person triggers "person_left" (default: 5)
"""

import json
import os
import sys
import time
import signal
import logging

import numpy as np
import redis

# Action classifier — classifies posture from keypoints (no new model needed)
# Imported from contracts/ directory (mounted as volume in Docker)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "contracts"))
from actions import classify_action
from time_rules import point_in_polygon, should_alert, get_time_period
from streams import (
    DETECTION_STREAM as _DET_TMPL,
    EVENT_STREAM as _EVT_TMPL,
    STATE_KEY as _STATE_TMPL,
    CONFIG_KEY as _CFG_TMPL,
    ZONE_KEY as _ZONE_TMPL,
    IDENTITY_KEY as _IDKEY_TMPL,
    VEHICLE_STREAM as _VEH_TMPL,
    stream_key,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAMERA_ID = os.getenv("CAMERA_ID", "front_door")
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.3"))
LOST_TIMEOUT = float(os.getenv("LOST_TIMEOUT", "8.0"))
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "trackers")
VEHICLE_CONSUMER_GROUP = os.getenv("VEHICLE_CONSUMER_GROUP", "vehicle_trackers")
CONSUMER_NAME = os.getenv("CONSUMER_NAME", "tracker_1")

# Stream keys — resolved from contracts/streams.py
DETECTION_STREAM = stream_key(_DET_TMPL, detector_type="pose", camera_id=CAMERA_ID)
EVENT_STREAM = stream_key(_EVT_TMPL, camera_id=CAMERA_ID)
STATE_KEY = stream_key(_STATE_TMPL, camera_id=CAMERA_ID)
VEHICLE_STREAM = stream_key(_VEH_TMPL, camera_id=CAMERA_ID)
CONFIG_KEY = stream_key(_CFG_TMPL, camera_id=CAMERA_ID)
ZONE_KEY = stream_key(_ZONE_TMPL, camera_id=CAMERA_ID)
IDENTITY_KEY = stream_key(_IDKEY_TMPL, camera_id=CAMERA_ID)

MAX_EVENT_STREAM_LEN = 5000  # Keep more events than frames (they're small)
VEHICLE_RATE_LIMIT_SEC = 10  # Max 1 vehicle event per 10 seconds
VEHICLE_IDLE_TIMEOUT = float(os.getenv("VEHICLE_IDLE_TIMEOUT", "5.0"))  # Seconds before idle alert
VEHICLE_LOST_TIMEOUT = 10.0  # Seconds before dropping a tracked vehicle
VEHICLE_IOU_THRESHOLD = 0.2  # Lower than person IoU — vehicles don't move much when parked
CONFIG_RELOAD_INTERVAL = 10  # Check config every N detection messages
ACTION_DEBOUNCE_FRAMES = 10  # New action must be stable for N frames before we accept it
ACTION_STICKY_MULTIPLIER = 2 # Once set, require N * multiplier frames to change away
MIN_BBOX_AREA = 3072          # ~1% of 640×480 frame — skip tiny distant detections

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tracker")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    logger.info("Shutdown signal received...")
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ---------------------------------------------------------------------------
# IoU Calculation
# ---------------------------------------------------------------------------
def compute_iou(box_a: list, box_b: list) -> float:
    """
    Compute Intersection over Union between two bounding boxes.

    Each box is [x1, y1, x2, y2].
    Returns a float between 0 (no overlap) and 1 (perfect overlap).

    This is the core of our tracking — if two boxes in consecutive frames
    overlap significantly, we assume they're the same person.
    """
    # Intersection coordinates
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    # Intersection area (0 if no overlap)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Union area
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    if union == 0:
        return 0.0

    return intersection / union


# ---------------------------------------------------------------------------
# Tracked Vehicle State
# ---------------------------------------------------------------------------
class TrackedVehicle:
    """
    Represents a vehicle being tracked across frames.

    Uses IoU matching to determine if a detected vehicle is the same one
    seen in previous frames. Tracks duration for idle detection.
    """

    def __init__(self, vehicle_id: str, bbox: list, class_name: str,
                 confidence: float, timestamp: float):
        self.vehicle_id = vehicle_id
        self.bbox = bbox                    # Current bounding box [x1, y1, x2, y2]
        self.class_name = class_name        # car, truck, bus, motorcycle
        self.confidence = confidence
        self.first_seen = timestamp         # When this vehicle first appeared
        self.last_seen = timestamp          # Last frame it was detected in
        self.frame_count = 1
        self.idle_alerted = False           # Whether idle notification was sent
        self.snapshot_key = ""              # Redis key for stored snapshot

    def update(self, bbox: list, confidence: float, timestamp: float):
        """Update vehicle state with a new detection."""
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = timestamp
        self.frame_count += 1

    @property
    def duration(self) -> float:
        """How long this vehicle has been seen (seconds)."""
        return self.last_seen - self.first_seen

    @property
    def center(self) -> tuple:
        """Center point of the bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )


# ---------------------------------------------------------------------------
# Tracked Person State
# ---------------------------------------------------------------------------
class TrackedPerson:
    """
    Represents a person being tracked across frames.

    Stores their current bounding box, when they first appeared,
    when they were last seen, and movement history for direction estimation.
    """

    def __init__(self, person_id: str, bbox: list, timestamp: float):
        self.person_id = person_id
        self.bbox = bbox                    # Current bounding box [x1, y1, x2, y2]
        self.first_seen = timestamp         # When this person first appeared
        self.last_seen = timestamp          # Last frame they were detected in
        self.frame_count = 1               # How many frames they've been in
        self.bbox_history: list[list] = [bbox]  # For direction estimation
        self.announced = False              # Whether we've emitted "person_appeared"
        self.action = "unknown"            # Current (stable) action
        self.action_confidence = 0.0       # How confident in the action classification
        self._pending_action = "unknown"   # Candidate action being debounced
        self._pending_count = 0            # Consecutive frames with pending action
        self.identity_name = ""            # Name from face recognition (sticky)

    def update(self, bbox: list, timestamp: float, keypoints: list = None):
        """Update this person's state with a new detection. Returns previous action."""
        self.bbox = bbox
        self.last_seen = timestamp
        self.frame_count += 1
        # Keep last 10 positions for direction estimation
        self.bbox_history.append(bbox)
        if len(self.bbox_history) > 10:
            self.bbox_history.pop(0)
        # Classify action from keypoints (with debounce + sticky bias)
        prev_action = self.action
        if keypoints:
            result = classify_action(keypoints)
            raw_action = result["action"]
            # Debounce: only change if new action is stable for N consecutive frames
            if raw_action == self._pending_action:
                self._pending_count += 1
            else:
                self._pending_action = raw_action
                self._pending_count = 1
            # Sticky bias: once we have a real action, require more evidence to change
            threshold = ACTION_DEBOUNCE_FRAMES
            if self.action not in ("unknown", ""):
                threshold = ACTION_DEBOUNCE_FRAMES * ACTION_STICKY_MULTIPLIER
            if self._pending_count >= threshold:
                self.action = raw_action
                self.action_confidence = result["confidence"]
        return prev_action

    @property
    def duration(self) -> float:
        """How long this person has been visible (seconds)."""
        return self.last_seen - self.first_seen

    @property
    def center(self) -> tuple[float, float]:
        """Center point of the current bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    @property
    def direction(self) -> str:
        """
        Estimate movement direction based on bbox center history.
        Returns: "left", "right", "stationary", or "unknown"
        """
        if len(self.bbox_history) < 3:
            return "unknown"

        # Compare first and last center positions
        first_center_x = (self.bbox_history[0][0] + self.bbox_history[0][2]) / 2
        last_center_x = (self.bbox_history[-1][0] + self.bbox_history[-1][2]) / 2
        dx = last_center_x - first_center_x

        if abs(dx) < 20:  # Pixel threshold for "stationary"
            return "stationary"
        elif dx > 0:
            return "right"
        else:
            return "left"

    def to_dict(self) -> dict:
        """Serialize for Redis state snapshot."""
        return {
            "person_id": self.person_id,
            "bbox": self.bbox,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "duration": round(self.duration, 1),
            "direction": self.direction,
            "action": self.action,
            "frame_count": self.frame_count,
            "identity_name": self.identity_name,
        }


# ---------------------------------------------------------------------------
# Person Tracker
# ---------------------------------------------------------------------------
class PersonTracker:
    """
    Tracks people across frames using IoU matching.

    Maintains a dictionary of currently tracked people. On each new set of
    detections, matches them to existing tracks or creates new ones.
    Emits events when people appear or leave.
    """

    def __init__(self, r: redis.Redis, iou_threshold: float, lost_timeout: float):
        self.r = r
        self.iou_threshold = iou_threshold
        self.lost_timeout = lost_timeout
        self.tracked: dict[str, TrackedPerson] = {}  # person_id → TrackedPerson
        self.next_id = 1  # Simple incrementing ID counter
        self.total_events = 0
        self._zones = {}         # zone_id → zone data
        self._zone_load_time = 0  # Timestamp of last zone load
        self._zone_reload_interval = 10  # Reload zones every N seconds
        self.frame_width = 640   # Updated from detection messages
        self.frame_height = 480  # Updated from detection messages
        self._identity_load_time = 0  # Timestamp of last identity load
        self._last_vehicle_event_time = 0.0  # Rate limiting for vehicle events
        self.tracked_vehicles: dict[str, TrackedVehicle] = {}  # vehicle_id → TrackedVehicle
        self._next_vehicle_id = 1  # Simple incrementing ID counter

    def _generate_id(self) -> str:
        """Generate a short, readable person ID."""
        pid = f"person_{self.next_id:04d}"
        self.next_id += 1
        return pid

    def _emit_event(self, event_type: str, person: TrackedPerson, timestamp: float, extra: dict = None):
        """Publish an event to the events stream."""
        # Determine which zone the person is in
        zone_name, alert_level = self._find_zone(person.bbox)

        # Evaluate zone + time-of-day rules to decide if this should trigger an alert
        alert_triggered = should_alert(alert_level) if alert_level else False

        event = {
            "camera_id": CAMERA_ID,
            "event_type": event_type,
            "timestamp": str(timestamp),
            "person_id": person.person_id,
            "identity_name": person.identity_name,
            "duration": str(round(person.duration, 1)),
            "direction": person.direction,
            "action": person.action,
            "bbox": json.dumps(person.bbox),
            "frame_count": str(person.frame_count),
            "zone": zone_name,
            "alert_level": alert_level,
            "alert_triggered": str(alert_triggered),
            "time_period": get_time_period(),
        }
        if extra:
            event.update(extra)

        self.r.xadd(EVENT_STREAM, event, maxlen=MAX_EVENT_STREAM_LEN)
        self.total_events += 1

        zone_str = f" | zone={zone_name}" if zone_name else ""
        name_str = f" ({person.identity_name})" if person.identity_name else ""
        logger.info(
            f"EVENT: {event_type} | {person.person_id}{name_str} | "
            f"action={person.action} | "
            f"duration={person.duration:.1f}s | direction={person.direction}"
            f"{zone_str}"
        )

    def _process_vehicle_detections(self, detections: list, timestamp: float, frame_bytes: bytes = None):
        """
        Track vehicles across frames using IoU matching and emit events.

        For each incoming detection:
        1. Match to existing tracked vehicles using IoU
        2. If matched → update state, check for idle timeout
        3. If new → create TrackedVehicle, emit vehicle_detected
        4. Prune stale vehicles not seen for VEHICLE_LOST_TIMEOUT

        Emits:
        - vehicle_detected: when a new vehicle first appears
        - vehicle_idle: when a vehicle stays in roughly the same spot
                        for > VEHICLE_IDLE_TIMEOUT seconds
        """
        now = time.time()

        # --- Step 1: Match incoming detections to tracked vehicles via IoU ---
        matched_vehicle_ids = set()

        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            class_name = det.get("class_name", "vehicle")
            confidence = det.get("confidence", 0)

            # Skip vehicles in dead zones
            if self._check_in_dead_zone(bbox):
                continue

            # Try to match to existing tracked vehicle
            best_match_id = None
            best_iou = VEHICLE_IOU_THRESHOLD

            for vid, veh in self.tracked_vehicles.items():
                iou = compute_iou(bbox, veh.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = vid

            if best_match_id:
                # --- Existing vehicle: update state ---
                veh = self.tracked_vehicles[best_match_id]
                veh.update(bbox, confidence, timestamp)
                matched_vehicle_ids.add(best_match_id)

                # Store/update snapshot if frame bytes provided
                if frame_bytes and not veh.snapshot_key:
                    snap_key = f"vehicle_snapshot:{CAMERA_ID}:{int(veh.first_seen)}"
                    self.r.setex(snap_key, 86400, frame_bytes)
                    veh.snapshot_key = snap_key

                # Check for idle timeout
                if (veh.duration >= VEHICLE_IDLE_TIMEOUT
                        and not veh.idle_alerted):
                    veh.idle_alerted = True
                    self._emit_vehicle_idle_event(veh, timestamp)

            else:
                # --- New vehicle: create tracker and emit detection event ---
                vid = f"vehicle_{self._next_vehicle_id:04d}"
                self._next_vehicle_id += 1

                veh = TrackedVehicle(vid, bbox, class_name, confidence, timestamp)

                # Store snapshot in Redis with 24h TTL
                if frame_bytes:
                    snap_key = f"vehicle_snapshot:{CAMERA_ID}:{int(timestamp)}"
                    self.r.setex(snap_key, 86400, frame_bytes)
                    veh.snapshot_key = snap_key

                self.tracked_vehicles[vid] = veh

                # Rate-limit vehicle_detected events to avoid flood
                if now - self._last_vehicle_event_time >= VEHICLE_RATE_LIMIT_SEC:
                    self._emit_vehicle_detected_event(veh, timestamp)
                    self._last_vehicle_event_time = now

        # --- Step 2: Prune stale vehicles ---
        stale_ids = [
            vid for vid, veh in self.tracked_vehicles.items()
            if timestamp - veh.last_seen > VEHICLE_LOST_TIMEOUT
        ]
        for vid in stale_ids:
            del self.tracked_vehicles[vid]

    def _emit_vehicle_detected_event(self, veh: 'TrackedVehicle', timestamp: float):
        """Emit a vehicle_detected event to the events stream."""
        zone_name, alert_level = self._find_zone(veh.bbox)
        alert_triggered = should_alert(alert_level) if alert_level else False

        event = {
            "camera_id": CAMERA_ID,
            "event_type": "vehicle_detected",
            "timestamp": str(timestamp),
            "person_id": "",
            "identity_name": "",
            "duration": "0",
            "direction": "",
            "action": "",
            "bbox": json.dumps(veh.bbox),
            "frame_count": str(veh.frame_count),
            "zone": zone_name,
            "alert_level": alert_level,
            "alert_triggered": str(alert_triggered),
            "vehicle_class": veh.class_name,
            "vehicle_confidence": str(round(veh.confidence, 3)),
            "snapshot_key": veh.snapshot_key,
            "time_period": get_time_period(),
        }

        self.r.xadd(EVENT_STREAM, event, maxlen=MAX_EVENT_STREAM_LEN)
        self.total_events += 1

        logger.info(
            f"EVENT: vehicle_detected | {veh.class_name} ({veh.confidence:.2f})"
            f"{f' | zone={zone_name}' if zone_name else ''}"
        )

    def _emit_vehicle_idle_event(self, veh: 'TrackedVehicle', timestamp: float):
        """
        Emit a vehicle_idle event when a vehicle has been stationary
        for longer than VEHICLE_IDLE_TIMEOUT.
        """
        zone_name, alert_level = self._find_zone(veh.bbox)
        alert_triggered = should_alert(alert_level) if alert_level else False

        event = {
            "camera_id": CAMERA_ID,
            "event_type": "vehicle_idle",
            "timestamp": str(timestamp),
            "person_id": "",
            "identity_name": "",
            "duration": str(round(veh.duration, 1)),
            "direction": "",
            "action": "",
            "bbox": json.dumps(veh.bbox),
            "frame_count": str(veh.frame_count),
            "zone": zone_name,
            "alert_level": alert_level,
            "alert_triggered": str(alert_triggered),
            "vehicle_class": veh.class_name,
            "vehicle_confidence": str(round(veh.confidence, 3)),
            "snapshot_key": veh.snapshot_key,
            "time_period": get_time_period(),
        }

        self.r.xadd(EVENT_STREAM, event, maxlen=MAX_EVENT_STREAM_LEN)
        self.total_events += 1

        logger.info(
            f"EVENT: vehicle_idle | {veh.class_name} idling {veh.duration:.1f}s"
            f"{f' | zone={zone_name}' if zone_name else ''}"
        )

    def _load_zones(self):
        """Load zone definitions from Redis (cached)."""
        now = time.time()
        if now - self._zone_load_time < self._zone_reload_interval:
            return

        try:
            raw = self.r.hgetall(ZONE_KEY)
            self._zones = {}
            for k, v in raw.items():
                key = k.decode() if isinstance(k, bytes) else k
                val = v.decode() if isinstance(v, bytes) else v
                self._zones[key] = json.loads(val)
        except Exception as e:
            logger.debug(f"Zone load error: {e}")

        self._zone_load_time = now

    def _find_zone(self, bbox: list) -> tuple:
        """
        Check which zone a person's bbox center falls in.

        Returns (zone_name, alert_level) or ("", "") if no zone.
        """
        self._load_zones()

        if not self._zones or len(bbox) != 4:
            return ("", "")

        # Normalize bbox center to 0-1 using actual frame dimensions
        frame_w = self.frame_width
        frame_h = self.frame_height

        cx = ((bbox[0] + bbox[2]) / 2) / frame_w
        cy = ((bbox[1] + bbox[3]) / 2) / frame_h

        for zone_id, zone in self._zones.items():
            pts = zone.get("points", [])
            if len(pts) >= 3 and point_in_polygon(cx, cy, pts):
                return (zone.get("name", zone_id), zone.get("alert_level", "log_only"))

        return ("", "")

    def _check_in_dead_zone(self, bbox: list) -> bool:
        """Return True if the bbox center falls in a 'dead_zone' — fully ignored area."""
        self._load_zones()
        if not self._zones or len(bbox) != 4:
            return False
        frame_w = self.frame_width
        frame_h = self.frame_height
        cx = ((bbox[0] + bbox[2]) / 2) / frame_w
        cy = ((bbox[1] + bbox[3]) / 2) / frame_h
        for zone_id, zone in self._zones.items():
            if zone.get("alert_level", "") != "dead_zone":
                continue
            pts = zone.get("points", [])
            if len(pts) >= 3 and point_in_polygon(cx, cy, pts):
                return True
        return False

    def _update_identities(self):
        """Read face identity state from Redis and map names to tracked persons."""
        now = time.time()
        if now - self._identity_load_time < 2:  # Check every 2 seconds
            return
        self._identity_load_time = now

        try:
            id_state = self.r.hgetall(IDENTITY_KEY)
            if not id_state:
                return
            id_json = id_state.get(b"identities", id_state.get("identities", b"[]"))
            if isinstance(id_json, bytes):
                id_json = id_json.decode()
            identities = json.loads(id_json)
        except Exception:
            return

        for ident in identities:
            id_name = ident.get("name", "Unknown")
            if id_name == "Unknown":
                continue
            id_bbox = ident.get("bbox", [])
            if len(id_bbox) != 4:
                continue
            # Match identity bbox to a tracked person via IoU
            best_iou = 0.0
            best_person = None
            for person in self.tracked.values():
                iou = compute_iou(id_bbox, person.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_person = person
            if best_iou > 0.2 and best_person:
                if not best_person.identity_name:
                    # First identification — emit event
                    best_person.identity_name = id_name
                    self._emit_event(
                        "person_identified", best_person, now,
                        extra={"identity_name": id_name}
                    )
                else:
                    best_person.identity_name = id_name

    def _update_state(self):
        """
        Update the Redis state key with the current scene snapshot.

        This is a single key (not a stream) that the dashboard reads to show
        who is currently in the frame RIGHT NOW. Overwritten on every update.
        """
        state = {
            "camera_id": CAMERA_ID,
            "timestamp": str(time.time()),
            "num_people": str(len(self.tracked)),
            "people": json.dumps([p.to_dict() for p in self.tracked.values()]),
        }
        self.r.hset(STATE_KEY, mapping=state)

    def update(self, detections: list[dict], timestamp: float):
        """
        Process a new set of detections and update tracked people.

        Algorithm:
        1. For each detection, find the best IoU match among tracked people
        2. If match > threshold → update that tracked person's state
        3. If no match → create a new tracked person
        4. Check for lost people (not seen for LOST_TIMEOUT seconds)
        """
        current_time = timestamp if timestamp > 0 else time.time()

        # --- Step 1: Match detections to existing tracks ---
        matched_track_ids = set()
        unmatched_detections = []

        for det in detections:
            bbox = det["bbox"]

            # Skip tiny detections (distant people, YOLO artifacts)
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if bbox_area < MIN_BBOX_AREA:
                continue

            # Skip detections in dead zones
            if self._check_in_dead_zone(bbox):
                continue

            best_iou = 0.0
            best_track_id = None

            for track_id, person in self.tracked.items():
                iou = compute_iou(bbox, person.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_iou >= self.iou_threshold and best_track_id not in matched_track_ids:
                # Match found — update existing track (pass keypoints for action detection)
                prev_action = self.tracked[best_track_id].update(
                    bbox, current_time, keypoints=det.get("keypoints")
                )
                matched_track_ids.add(best_track_id)

                # Emit "person_appeared" on first stable detection (after ~1 second)
                person = self.tracked[best_track_id]
                if not person.announced and person.frame_count >= 15:
                    self._emit_event("person_appeared", person, current_time)
                    person.announced = True
                elif (person.announced
                      and prev_action != person.action
                      and prev_action not in ("unknown", "")
                      and person.action not in ("unknown", "")):
                    # Action changed — emit transition event
                    self._emit_event("action_changed", person, current_time,
                                     extra={"prev_action": prev_action})
            else:
                # No match — save for new track creation
                unmatched_detections.append(det)

        # --- Step 2: Create new tracks for unmatched detections ---
        for det in unmatched_detections:
            person_id = self._generate_id()
            person = TrackedPerson(person_id, det["bbox"], current_time)
            self.tracked[person_id] = person

        # --- Step 3: Check for lost people ---
        lost_ids = []
        for track_id, person in self.tracked.items():
            time_since_seen = current_time - person.last_seen
            if time_since_seen > self.lost_timeout:
                # Person has left the frame
                if person.announced:
                    self._emit_event("person_left", person, current_time)
                lost_ids.append(track_id)

        for track_id in lost_ids:
            del self.tracked[track_id]

        # --- Step 4: Update identities from face recognizer ---
        self._update_identities()

        # --- Step 5: Update scene state in Redis ---
        self._update_state()


# ---------------------------------------------------------------------------
# Redis Consumer Group Setup
# ---------------------------------------------------------------------------
def setup_consumer_group(r: redis.Redis) -> None:
    """Create consumer groups for detection and vehicle streams."""
    for stream, group in [
        (DETECTION_STREAM, CONSUMER_GROUP),
        (VEHICLE_STREAM, VEHICLE_CONSUMER_GROUP),
    ]:
        try:
            r.xgroup_create(stream, group, id="$", mkstream=True)
            logger.info(f"Created consumer group '{group}' on '{stream}'")
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group '{group}' already exists")
            else:
                raise


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------
def run():
    """
    Main loop: read detections from Redis → update tracker → publish events.

    The tracker is a lightweight CPU service — no GPU needed. It just does
    bounding box math and state management.

    Reads from two streams:
    - DETECTION_STREAM (person detections from pose-detector)
    - VEHICLE_STREAM (vehicle detections from vehicle-detector)
    """
    logger.info(f"Starting tracker for camera '{CAMERA_ID}'")
    logger.info(f"Reading from: {DETECTION_STREAM} + {VEHICLE_STREAM}")
    logger.info(f"Publishing to: {EVENT_STREAM}")
    logger.info(f"State key: {STATE_KEY}")
    logger.info(f"IoU threshold: {IOU_THRESHOLD}, Lost timeout: {LOST_TIMEOUT}s")

    # Connect to Redis
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    r.ping()
    logger.info("Redis connection verified")

    # Setup consumer groups (person + vehicle)
    setup_consumer_group(r)

    # Initialize tracker
    tracker = PersonTracker(r, IOU_THRESHOLD, LOST_TIMEOUT)
    messages_processed = 0

    while not _shutdown:
        try:
            # Read from both person and vehicle detection streams
            messages = r.xreadgroup(
                CONSUMER_GROUP,
                CONSUMER_NAME,
                {DETECTION_STREAM: ">"},
                count=1,
                block=500,  # Shorter block so we can check vehicles too
            )

            # Also check for vehicle detections
            vehicle_messages = r.xreadgroup(
                VEHICLE_CONSUMER_GROUP,
                CONSUMER_NAME,
                {VEHICLE_STREAM: ">"},
                count=1,
                block=0,  # Non-blocking — just check what's available
            )
        except redis.ConnectionError:
            logger.warning("Redis connection lost — retrying...")
            time.sleep(1)
            continue

        # --- Process person detections ---
        if not messages:
            # Even with no new detections, check for lost people
            tracker.update([], time.time())
        else:
            for stream_name, entries in messages:
                for message_id, data in entries:
                    timestamp = float(data.get(b"timestamp", b"0").decode())
                    detections_json = data.get(b"detections", b"[]").decode()
                    detections = json.loads(detections_json)

                    # Hot-reload IoU and lost timeout from Redis config (set by dashboard)
                    messages_processed += 1
                    if messages_processed % CONFIG_RELOAD_INTERVAL == 0:
                        try:
                            cfg_iou = r.hget(CONFIG_KEY, "iou_threshold")
                            cfg_timeout = r.hget(CONFIG_KEY, "lost_timeout")
                            if cfg_iou:
                                new_iou = float(cfg_iou)
                                if new_iou != tracker.iou_threshold:
                                    logger.info(f"Config updated: IoU {tracker.iou_threshold} → {new_iou}")
                                    tracker.iou_threshold = new_iou
                            if cfg_timeout:
                                new_timeout = float(cfg_timeout)
                                if new_timeout != tracker.lost_timeout:
                                    logger.info(f"Config updated: lost_timeout {tracker.lost_timeout} → {new_timeout}")
                                    tracker.lost_timeout = new_timeout
                        except (ValueError, redis.ConnectionError):
                            pass

                    # Update frame dimensions from detection metadata
                    fw = data.get(b"frame_width", b"").decode()
                    fh = data.get(b"frame_height", b"").decode()
                    if fw and fh:
                        tracker.frame_width = int(fw)
                        tracker.frame_height = int(fh)

                    # Update tracker with new detections
                    tracker.update(detections, timestamp)

                    # Acknowledge message
                    r.xack(DETECTION_STREAM, CONSUMER_GROUP, message_id)

        # --- Process vehicle detections ---
        if vehicle_messages:
            for stream_name, entries in vehicle_messages:
                for message_id, data in entries:
                    timestamp = float(data.get(b"timestamp", b"0").decode())
                    detections_json = data.get(b"detections", b"[]").decode()
                    detections = json.loads(detections_json)
                    frame_bytes = data.get(b"frame_bytes", None)

                    if detections:
                        tracker._process_vehicle_detections(detections, timestamp, frame_bytes)

                    r.xack(VEHICLE_STREAM, VEHICLE_CONSUMER_GROUP, message_id)

    logger.info(
        f"Tracker stopped. Total events emitted: {tracker.total_events}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run()
