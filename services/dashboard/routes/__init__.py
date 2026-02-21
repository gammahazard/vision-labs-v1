"""
routes/ — FastAPI APIRouter modules for the dashboard.

PURPOSE:
    Split server.py's REST API endpoints into focused modules.
    Each module creates a router via create_router() and receives
    shared dependencies (Redis client, stream keys, logger).

USAGE (in server.py):
    from routes.events import router as events_router
    app.include_router(events_router)

SHARED STATE:
    Each router module accesses the Redis client and key names
    from this package's module-level variables, set by server.py
    at startup before including routers.
"""

import redis
import logging

# Shared state — set by server.py before routers are included
r: redis.Redis = None           # Redis client
logger: logging.Logger = None   # Logger instance
FACE_API_URL: str = ""          # face-recognizer service URL

# Redis key names — set by server.py
EVENT_STREAM: str = ""
FRAME_STREAM: str = ""
DETECTION_STREAM: str = ""
STATE_KEY: str = ""
CONFIG_KEY: str = ""
IDENTITY_KEY: str = ""
ZONE_KEY: str = ""
AUTH_DB_PATH: str = ""
VEHICLE_SNAPSHOT_DIR: str = ""       # Vehicle snapshot disk storage root
CAMERA_ID: str = "front_door"        # Camera identifier (set by server.py)

# Default config values
DEFAULT_CONFIG = {
    "confidence_thresh": "0.5",
    "iou_threshold": "0.3",
    "lost_timeout": "5.0",
    "target_fps": "5",
}
