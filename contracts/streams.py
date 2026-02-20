"""
contracts/streams.py — THE source of truth for all Redis Stream keys and data schemas.

PURPOSE:
    Every service in the Vision Labs pipeline reads from and writes to Redis Streams.
    This file defines the exact stream key names and the shape of data flowing through
    them. Import from here — never hardcode stream keys in service code.

RELATIONSHIPS:
    - camera-ingester reads RTSP, publishes to FRAME_STREAM
    - pose-detector (Phase 2) reads FRAME_STREAM, publishes to DETECTION_STREAM
    - tracker (Phase 2) reads DETECTION_STREAM, publishes to EVENT_STREAM
    - dashboard (Phase 3) reads EVENT_STREAM + STATE_KEY for live display
    - rule engine (Phase 4) reads EVENT_STREAM, publishes to ALERT_STREAM

DATA FLOW:
    Camera → FRAME_STREAM → DETECTION_STREAM → EVENT_STREAM → ALERT_STREAM
                                                     ↓
                                                STATE_KEY (current scene snapshot)
"""

from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# STREAM KEYS
# =============================================================================
# These are Redis Stream key templates. Replace {camera_id} with the actual
# camera identifier (e.g., "front_door", "backyard").

# Raw JPEG frames from cameras — one stream per camera.
# Published by: camera-ingester
# Consumed by: pose-detector, face-detector, dashboard (live view)
FRAME_STREAM = "frames:{camera_id}"

# Detection results from AI models — one stream per detector per camera.
# Published by: pose-detector, face-detector, emotion-detector
# Consumed by: tracker
DETECTION_STREAM = "detections:{detector_type}:{camera_id}"

# High-level events (person appeared, person left, loitering, etc.)
# Published by: tracker
# Consumed by: rule engine, dashboard event feed, archive worker
EVENT_STREAM = "events:{camera_id}"

# Current state of what the camera sees RIGHT NOW (latest detections).
# This is a Redis key (not a stream) — overwritten on each frame.
# Published by: tracker
# Consumed by: dashboard (real-time scene overlay)
STATE_KEY = "state:{camera_id}"

# Alert decisions after rule evaluation (zone + time + confidence).
# Published by: rule engine (Phase 4)
# Consumed by: dashboard notifications, notification workers
ALERT_STREAM = "alerts"

# Human review queue for active learning feedback loop.
# Published by: rule engine (Phase 5)
# Consumed by: review queue UI
REVIEW_STREAM = "reviews"

# Live config hash (confidence, IoU threshold, etc.)
# Written by: dashboard (settings panel)
# Read by: pose-detector, tracker (hot-reload)
CONFIG_KEY = "config:{camera_id}"

# Zone definitions — stored as Redis hash, managed by dashboard
# Written by: dashboard (zone editor)
# Read by: tracker (zone checks), dashboard (zone overlay)
ZONE_KEY = "zones:{camera_id}"

# Face identity results per frame
# Published by: face-recognizer
# Consumed by: dashboard (name overlay)
IDENTITY_STREAM = "identities:{camera_id}"

# Current identity state — latest recognized faces in view
# Written by: face-recognizer (hash, overwritten each frame)
# Read by: dashboard (name overlay)
IDENTITY_KEY = "identity_state:{camera_id}"


def stream_key(template: str, **kwargs) -> str:
    """Resolve a stream key template with actual values.

    Example:
        stream_key(FRAME_STREAM, camera_id="front_door")
        # Returns: "frames:front_door"

        stream_key(DETECTION_STREAM, detector_type="pose", camera_id="front_door")
        # Returns: "detections:pose:front_door"
    """
    return template.format(**kwargs)


# =============================================================================
# DATA SCHEMAS
# =============================================================================
# These dataclasses define the shape of messages on each stream.
# Services serialize to/from these when publishing and consuming.

@dataclass
class FrameMessage:
    """
    A single video frame captured from a camera.

    Published to FRAME_STREAM by the camera-ingester service.
    Contains the raw JPEG bytes so consumers don't need to decode RTSP themselves.
    """
    camera_id: str                   # e.g., "front_door"
    timestamp: float                 # Unix timestamp (time.time())
    frame_bytes: bytes               # JPEG-encoded frame data
    frame_number: int                # Monotonically increasing frame counter
    resolution: tuple[int, int]      # (width, height) of the original frame


@dataclass
class DetectionMessage:
    """
    AI model detection results for a single frame.

    Published to DETECTION_STREAM by detector workers (pose, face, emotion).
    Each detection includes a bounding box, confidence score, and model-specific
    metadata (keypoints for pose, embeddings for face, etc.)
    """
    camera_id: str                   # Which camera this detection came from
    detector_type: str               # "pose", "face", "emotion"
    timestamp: float                 # Matches the source frame's timestamp
    frame_number: int                # Matches the source frame's number
    detections: list[dict] = field(  # List of detected objects
        default_factory=list
    )
    # Each detection dict contains:
    #   bbox: [x1, y1, x2, y2]      — bounding box coordinates
    #   confidence: float            — model confidence (0-1)
    #   class_name: str              — "person", "face", etc.
    #   metadata: dict               — model-specific (keypoints, embeddings, etc.)


@dataclass
class EventMessage:
    """
    A semantic event derived from tracking detections over time.

    Published to EVENT_STREAM by the tracker service.
    These are what the dashboard displays and what the rule engine evaluates.
    """
    camera_id: str                   # Which camera
    event_type: str                  # "person_appeared", "person_left", "loitering"
    timestamp: float                 # When the event occurred
    person_id: Optional[str] = None  # Tracker-assigned ID (for correlating across frames)
    zone: Optional[str] = None       # Which zone the event occurred in (Phase 4)
    alert_level: Optional[str] = None  # Zone alert level: "always", "night_only", "log_only", "ignore"
    alert_triggered: bool = False    # True if zone + time-of-day rules say we should notify
    metadata: dict = field(          # Event-specific details
        default_factory=dict
    )
    # metadata may contain:
    #   duration: float              — how long the person was visible
    #   direction: str               — "entering", "leaving", "stationary"
    #   face_match: str | None       — known face label if recognized
    #   confidence: float            — overall event confidence
