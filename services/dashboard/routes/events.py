"""
routes/events.py — Event feed endpoint.

PURPOSE:
    GET /api/events — Read recent events from the Redis event stream.
    GET /api/events/{event_id}/snapshot — Serve saved camera snapshot JPEG.
    Used by events.js in the frontend.
"""

import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
import redis

import routes as ctx

router = APIRouter(prefix="/api", tags=["events"])

SNAPSHOT_DIR = os.environ.get("SNAPSHOT_DIR", "/data/snapshots")


@router.get("/events")
async def get_events(count: int = 50):
    """
    Return the most recent events from the event stream.
    Used by the dashboard's event feed panel.
    """
    try:
        events_raw = ctx.r.xrevrange(ctx.EVENT_STREAM, count=count)
        events = []
        for event_id, data in events_raw:
            events.append({
                "id": event_id,
                "event_type": data.get("event_type", ""),
                "person_id": data.get("person_id", ""),
                "timestamp": data.get("timestamp", "0"),
                "duration": data.get("duration", "0"),
                "direction": data.get("direction", ""),
                "action": data.get("action", "unknown"),
                "camera_id": data.get("camera_id", ""),
                "zone": data.get("zone", ""),
                "alert_level": data.get("alert_level", ""),
                "alert_triggered": data.get("alert_triggered", "false"),
                "prev_action": data.get("prev_action", ""),
                "identity_name": data.get("identity_name", ""),
            })
        return {"events": events}
    except redis.ConnectionError:
        return JSONResponse(status_code=503, content={"error": "Redis unavailable"})


@router.get("/events/{event_id}/snapshot")
async def get_event_snapshot(event_id: str):
    """
    Serve the saved camera snapshot for a given event.
    Snapshots are stored as JPEG files by the event poller in server.py.
    """
    safe_id = event_id.replace(":", "-")
    path = os.path.join(SNAPSHOT_DIR, f"{safe_id}.jpg")

    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "Snapshot not found"})

    with open(path, "rb") as f:
        data = f.read()

    return Response(content=data, media_type="image/jpeg")

