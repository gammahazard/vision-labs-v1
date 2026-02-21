"""
routes/ai.py — AI assistant API endpoints.

PURPOSE:
    Chat with a local Qwen 3 14B model via Ollama. Supports tool/function
    calling for querying security data, sending Telegram messages,
    scheduling reminders, and managing system config.

ENDPOINTS:
    GET  /api/ai/config — Get AI assistant configuration (enabled, names)
    POST /api/ai/config — Save/update AI configuration (onboarding)
    POST /api/ai/chat   — Send message, get streamed AI response

LLM:
    Qwen 3 14B running locally via Ollama Docker container.
    Tool calling is used for structured actions (query events, send alerts).
"""

import os
import json
import time
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

import ollama as ollama_lib

import routes as ctx

router = APIRouter(prefix="/api/ai", tags=["ai"])
logger = logging.getLogger("dashboard.ai")

# Set by server.py at startup
_ai_db = None
_feedback_db = None
_model_gpu_ready = False  # Set True by server.py after warm-up chat succeeds

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = "qwen3:14b"
TZ_LOCAL = ZoneInfo(os.getenv("LOCATION_TIMEZONE", "America/Toronto"))


def set_ai_db(db):
    """Called by server.py to inject the AI database instance."""
    global _ai_db
    _ai_db = db


def set_feedback_db(db):
    """Called by server.py to inject the feedback database instance."""
    global _feedback_db
    _feedback_db = db


def set_gpu_ready_flag(ready: bool):
    """Called by server.py once the warm-up chat confirms model is in GPU memory."""
    global _model_gpu_ready
    _model_gpu_ready = ready


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------
class ConfigRequest(BaseModel):
    enabled: bool = True
    user_name: str = ""
    ai_name: str = "Atlas"


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


# ---------------------------------------------------------------------------
# Tool definitions for the LLM
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_events",
            "description": "Search recent security events (person detected, person identified, vehicle idle). Returns the most recent events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of recent events to return (max 50)",
                    },
                    "event_type": {
                        "type": "string",
                        "description": "Filter by event type: person_appeared, person_identified, person_left, vehicle_idle. Leave empty for all.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_faces",
            "description": "List all enrolled/known faces in the system.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_feedback_stats",
            "description": "Get feedback statistics: total verdicts, false alarm rate, accuracy, active suppression rules.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_telegram",
            "description": "Send a message to the user via Telegram right now. Can include a live camera snapshot or a 5-second video clip.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message text to send",
                    },
                    "include_snapshot": {
                        "type": "boolean",
                        "description": "If true, attach the latest live camera frame to the message.",
                    },
                    "include_clip": {
                        "type": "boolean",
                        "description": "If true, capture and attach a 5-second video clip from the camera.",
                    },
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_reminder",
            "description": "Schedule a reminder to be sent via Telegram at a specific time. Can include a snapshot or video clip captured at the scheduled time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The reminder message",
                    },
                    "time_description": {
                        "type": "string",
                        "description": "When to send, e.g. '10:00 PM', 'in 30 minutes', '2026-02-21T22:00:00'",
                    },
                    "media_type": {
                        "type": "string",
                        "enum": ["text", "snapshot", "clip"],
                        "description": "Type of media to include: 'text' (default), 'snapshot' (camera photo), or 'clip' (5-second video).",
                    },
                },
                "required": ["message", "time_description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_status",
            "description": "Get current system status: stream sizes, config settings, notification preferences.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrain_rules",
            "description": "Retrain the alert suppression model. Deletes all existing suppression rules and re-scans every feedback record to regenerate rules based on current patterns. Use when the user asks to retrain, refresh, or update the learning model.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "review_feedback",
            "description": "Get recent user feedback records showing how the user has rated past alerts (real_threat, false_alarm, identified). Useful for understanding patterns and reviewing the learning history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of recent feedback records to return (max 100)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_live_scene",
            "description": "Get what's happening on camera RIGHT NOW — who is in frame, their actions, how long they've been there.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_unknowns",
            "description": "List unknown/unidentified faces that have been auto-captured by the system. Shows how many strangers have been seen.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_events_by_date",
            "description": "Query events filtered by date. Use this to answer questions like 'how many events today' or 'what happened yesterday'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date to query in YYYY-MM-DD format. Use 'today' or 'yesterday' as shortcuts.",
                    },
                    "event_type": {
                        "type": "string",
                        "description": "Optional: filter by event type (person_appeared, person_identified, person_left, vehicle_idle)",
                    },
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_zones",
            "description": "List all security zones defined on the camera, including their names, alert levels, and point coordinates.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse_vehicles",
            "description": "List vehicle detection snapshots for a given day. Shows how many vehicles were detected and their timestamps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format, or 'today'/'yesterday'. Defaults to today.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather conditions at the camera location. Useful for correlating activity with weather.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_event_patterns",
            "description": "Analyze event patterns and trends. Groups events by hour of day, by type, or calculates daily averages. Use for questions like 'what's the busiest time of day' or 'how many people per day this week'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis_type": {
                        "type": "string",
                        "enum": ["hourly", "daily", "type_breakdown"],
                        "description": "Type of analysis: 'hourly' (by hour of day), 'daily' (by day), 'type_breakdown' (events by type)",
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "How many days of history to analyze (default 7, max 30)",
                    },
                },
                "required": ["analysis_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_snapshot",
            "description": "Capture the current camera frame and show it in the chat. Returns context data (weather, scene) for you to describe.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_clip",
            "description": "Record a 5-second video clip from the live camera and show it in the chat. Use when the user asks to see a clip, video, or recording of what's happening now.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_notification_history",
            "description": "Get recent Telegram notifications that were sent by the system. Shows what alerts the user has received.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of recent notifications to return (default 20, max 50)",
                    },
                },
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------
async def _execute_tool(name: str, args: dict) -> str:
    """Execute a tool call and return the result as a string."""
    try:
        if name == "query_events":
            return _tool_query_events(args)
        elif name == "query_faces":
            return await _tool_query_faces()
        elif name == "query_feedback_stats":
            return _tool_query_feedback_stats()
        elif name == "send_telegram":
            return await _tool_send_telegram(args)
        elif name == "schedule_reminder":
            return _tool_schedule_reminder(args)
        elif name == "get_system_status":
            return _tool_get_system_status()
        elif name == "retrain_rules":
            return _tool_retrain_rules()
        elif name == "review_feedback":
            return _tool_review_feedback(args)
        elif name == "get_live_scene":
            return _tool_get_live_scene()
        elif name == "query_unknowns":
            return await _tool_query_unknowns()
        elif name == "query_events_by_date":
            return _tool_query_events_by_date(args)
        elif name == "query_zones":
            return _tool_query_zones()
        elif name == "browse_vehicles":
            return _tool_browse_vehicles(args)
        elif name == "get_weather":
            return await _tool_get_weather()
        elif name == "query_event_patterns":
            return _tool_query_event_patterns(args)
        elif name == "capture_snapshot":
            return await _tool_capture_snapshot()
        elif name == "capture_clip":
            return _tool_capture_clip()
        elif name == "query_notification_history":
            return _tool_query_notification_history(args)
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as e:
        logger.warning(f"Tool {name} error: {e}")
        return json.dumps({"error": str(e)})


def _tool_query_events(args: dict) -> str:
    """Query recent events from Redis."""
    count = min(int(args.get("count", 20)), 50)
    event_type = args.get("event_type", "")

    try:
        events_raw = ctx.r.xrevrange(ctx.EVENT_STREAM, count=count)
        events = []
        for msg_id, data in events_raw:
            evt = {k: v for k, v in data.items()}
            evt["event_id"] = msg_id
            if event_type and evt.get("type") != event_type:
                continue
            events.append(evt)
        return json.dumps({"events": events, "total": len(events)})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_get_live_scene() -> str:
    """Get the current live scene from the tracker state."""
    try:
        state = ctx.r.hgetall(ctx.STATE_KEY)
        if not state:
            return json.dumps({"scene": "No data — tracker may not be running or no activity detected."})

        num_people = state.get("num_people", "0")
        persons_raw = state.get("persons", "[]")
        try:
            persons = json.loads(persons_raw)
        except (json.JSONDecodeError, TypeError):
            persons = []

        # Also check identity state
        identity_state = ctx.r.hgetall(f"identity_state:{ctx.CAMERA_ID}")
        identities = []
        if identity_state:
            try:
                identities = json.loads(identity_state.get("identities", "[]"))
            except (json.JSONDecodeError, TypeError):
                identities = []

        scene_data = {
            "people_in_frame": int(num_people),
            "persons": persons,
            "identified_faces": identities,
            "camera": ctx.CAMERA_ID,
        }
        return json.dumps(scene_data)
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _tool_query_unknowns() -> str:
    """Query unknown/auto-captured faces from the face recognizer."""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{ctx.FACE_API_URL}/api/faces", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            face_list = data.get("faces", data) if isinstance(data, dict) else data
            if not isinstance(face_list, list):
                face_list = []
            # Filter for unknown/auto-captured faces (names starting with 'unknown' or auto-generated)
            unknowns = [f for f in face_list if isinstance(f, dict) and
                        (f.get("name", "").lower().startswith("unknown") or
                         f.get("name", "").startswith("auto_"))]
            enrolled = [f for f in face_list if isinstance(f, dict) and f not in unknowns]
            return json.dumps({
                "total_faces": len(face_list),
                "enrolled_count": len(enrolled),
                "enrolled_names": [f.get("name", "?") for f in enrolled],
                "unknown_count": len(unknowns),
                "unknowns": [{"name": f.get("name", "?"), "id": f.get("id", "?")} for f in unknowns[:20]],
            })
        return json.dumps({"error": f"Face API returned {resp.status_code}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_query_events_by_date(args: dict) -> str:
    """Query events filtered by date."""
    from datetime import datetime, timedelta
    import time as _time

    date_str = args.get("date", "today")
    event_type = args.get("event_type", "")

    # Parse date (use local timezone so "today" = EST, not UTC in Docker)
    now = datetime.now(TZ_LOCAL)
    if date_str == "today":
        target_date = now.date()
    elif date_str == "yesterday":
        target_date = (now - timedelta(days=1)).date()
    else:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return json.dumps({"error": f"Invalid date format: {date_str}. Use YYYY-MM-DD, 'today', or 'yesterday'."})

    # Convert date to Redis stream timestamp range (timezone-aware)
    day_start = datetime.combine(target_date, datetime.min.time(), tzinfo=TZ_LOCAL)
    day_end = datetime.combine(target_date, datetime.max.time(), tzinfo=TZ_LOCAL)
    start_ms = int(day_start.timestamp() * 1000)
    end_ms = int(day_end.timestamp() * 1000)

    try:
        events_raw = ctx.r.xrange(ctx.EVENT_STREAM, min=f"{start_ms}-0", max=f"{end_ms}-0")
        events = []
        for msg_id, data in events_raw:
            evt = {k: v for k, v in data.items()}
            evt["event_id"] = msg_id
            if event_type and evt.get("type") != event_type:
                continue
            events.append(evt)

        # Summarize by type
        type_counts = {}
        for evt in events:
            t = evt.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        return json.dumps({
            "date": str(target_date),
            "total_events": len(events),
            "by_type": type_counts,
            "latest_events": events[-10:] if len(events) > 10 else events,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_query_zones() -> str:
    """List all defined security zones."""
    try:
        zone_data = ctx.r.hgetall(ctx.ZONE_KEY)
        if not zone_data:
            return json.dumps({"zones": [], "count": 0, "message": "No zones defined yet."})

        zones = []
        for zone_id, zone_json in zone_data.items():
            try:
                zone = json.loads(zone_json)
                zone["id"] = zone_id
                zones.append(zone)
            except (json.JSONDecodeError, TypeError):
                zones.append({"id": zone_id, "raw": zone_json})

        return json.dumps({"zones": zones, "count": len(zones)})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_browse_vehicles(args: dict) -> str:
    """List vehicle detection snapshots for a given day."""
    from datetime import datetime, timedelta
    import os
    import glob

    date_str = args.get("date", "today")
    now = datetime.now(TZ_LOCAL)
    if date_str == "today":
        target_date = now.strftime("%Y-%m-%d")
    elif date_str == "yesterday":
        target_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        target_date = date_str

    snapshot_dir = ctx.VEHICLE_SNAPSHOT_DIR
    day_dir = os.path.join(snapshot_dir, target_date) if snapshot_dir else f"/data/vehicle_snapshots/{target_date}"

    try:
        if not os.path.isdir(day_dir):
            return json.dumps({"date": target_date, "count": 0, "snapshots": [], "message": f"No vehicle snapshots for {target_date}"})

        files = sorted(glob.glob(os.path.join(day_dir, "*.jpg")))
        snapshots = []
        for f in files:
            basename = os.path.basename(f)
            # Extract timestamp from filename (e.g. '1708545600_vehicle.jpg')
            snapshots.append({
                "filename": basename,
                "size_kb": round(os.path.getsize(f) / 1024, 1),
            })

        return json.dumps({
            "date": target_date,
            "count": len(snapshots),
            "snapshots": snapshots[-20:],  # Last 20
            "note": f"Showing last {min(20, len(snapshots))} of {len(snapshots)}" if len(snapshots) > 20 else None,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _tool_get_weather() -> str:
    """Get current weather from OpenWeatherMap."""
    import httpx
    api_key = os.getenv("OPENWEATHER_API_KEY", "")
    lat = os.getenv("LOCATION_LAT", "")
    lon = os.getenv("LOCATION_LON", "")

    if not api_key:
        return json.dumps({"error": "OPENWEATHER_API_KEY not configured"})
    if not lat or not lon:
        return json.dumps({"error": "LOCATION_LAT/LON not configured"})

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            weather = {
                "condition": data.get("weather", [{}])[0].get("description", "unknown"),
                "temperature_c": data.get("main", {}).get("temp"),
                "feels_like_c": data.get("main", {}).get("feels_like"),
                "humidity_pct": data.get("main", {}).get("humidity"),
                "wind_speed_ms": data.get("wind", {}).get("speed"),
                "visibility_m": data.get("visibility"),
                "sunrise": data.get("sys", {}).get("sunrise"),
                "sunset": data.get("sys", {}).get("sunset"),
            }
            return json.dumps(weather)
        return json.dumps({"error": f"Weather API returned {resp.status_code}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_query_event_patterns(args: dict) -> str:
    """Analyze event patterns for trends."""
    from datetime import datetime, timedelta
    from collections import defaultdict

    analysis_type = args.get("analysis_type", "hourly")
    days_back = min(int(args.get("days_back", 7)), 30)

    # Calculate time range (timezone-aware for correct local day boundaries)
    now = datetime.now(TZ_LOCAL)
    start_date = now - timedelta(days=days_back)
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    try:
        events_raw = ctx.r.xrange(ctx.EVENT_STREAM, min=f"{start_ms}-0", max=f"{end_ms}-0")

        if analysis_type == "hourly":
            hourly = defaultdict(int)
            for msg_id, data in events_raw:
                ts = data.get("timestamp") or data.get("first_seen", "")
                try:
                    if "." in str(ts):
                        dt = datetime.fromtimestamp(float(ts), tz=TZ_LOCAL)
                    else:
                        dt = datetime.fromisoformat(str(ts))
                    hourly[dt.hour] += 1
                except (ValueError, TypeError, OSError):
                    continue

            # Format as readable hours
            result = {}
            for h in range(24):
                label = f"{h:02d}:00"
                result[label] = hourly.get(h, 0)

            busiest = max(hourly.items(), key=lambda x: x[1]) if hourly else (0, 0)
            return json.dumps({
                "analysis": "hourly",
                "days_analyzed": days_back,
                "total_events": len(events_raw),
                "hourly_breakdown": result,
                "busiest_hour": f"{busiest[0]:02d}:00 ({busiest[1]} events)",
            })

        elif analysis_type == "daily":
            daily = defaultdict(int)
            for msg_id, data in events_raw:
                ts = data.get("timestamp") or data.get("first_seen", "")
                try:
                    if "." in str(ts):
                        dt = datetime.fromtimestamp(float(ts), tz=TZ_LOCAL)
                    else:
                        dt = datetime.fromisoformat(str(ts))
                    daily[dt.strftime("%Y-%m-%d")] += 1
                except (ValueError, TypeError, OSError):
                    continue

            avg = sum(daily.values()) / max(len(daily), 1)
            return json.dumps({
                "analysis": "daily",
                "days_analyzed": days_back,
                "total_events": len(events_raw),
                "daily_breakdown": dict(sorted(daily.items())),
                "daily_average": round(avg, 1),
                "busiest_day": max(daily.items(), key=lambda x: x[1])[0] if daily else "none",
            })

        elif analysis_type == "type_breakdown":
            types = defaultdict(int)
            for msg_id, data in events_raw:
                evt_type = data.get("type", "unknown")
                types[evt_type] += 1

            return json.dumps({
                "analysis": "type_breakdown",
                "days_analyzed": days_back,
                "total_events": len(events_raw),
                "by_type": dict(sorted(types.items(), key=lambda x: x[1], reverse=True)),
            })

        else:
            return json.dumps({"error": f"Unknown analysis type: {analysis_type}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Snapshot/clip side-channel ─────────────────────────────────────────
# Base64 images are WAY too large to send back to the LLM as a tool result
# (a 51 KB JPEG = ~68 KB base64 ≈ 53 000 tokens, vs 8 192 context limit).
# Instead we stash them here and the chat handler injects the media
# into the final reply before returning it to the browser.
_pending_snapshot: str | None = None
_pending_clip: str | None = None  # filename on disk, served via /api/ai/clip/


async def _tool_capture_snapshot() -> str:
    """Capture camera frame with weather + scene context for AI to describe."""
    global _pending_snapshot
    import base64
    import httpx
    from routes.notifications import get_latest_frame

    try:
        frame = get_latest_frame()
        if not frame:
            _pending_snapshot = None
            return json.dumps({"error": "No frame available — camera may be offline"})

        b64 = base64.b64encode(frame).decode("utf-8")

        # Stash the base64 for the chat handler — NOT for the LLM
        _pending_snapshot = b64

        # Gather contextual data so the AI can describe the scene intelligently
        context = {}

        # Weather from conditions endpoint cache or direct fetch
        try:
            api_key = os.getenv("OPENWEATHER_API_KEY", "")
            lat = os.getenv("LOCATION_LAT", "")
            lon = os.getenv("LOCATION_LON", "")
            if api_key and lat and lon:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        "https://api.openweathermap.org/data/2.5/weather",
                        params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
                        timeout=3,
                    )
                if resp.status_code == 200:
                    w = resp.json()
                    context["weather"] = {
                        "temp_c": round(w["main"]["temp"]),
                        "feels_like_c": round(w["main"]["feels_like"]),
                        "description": w["weather"][0]["description"],
                        "humidity": w["main"]["humidity"],
                        "wind_kmh": round(w["wind"]["speed"] * 3.6),
                    }
        except Exception:
            pass

        # Current scene state
        try:
            state = ctx.r.hgetall(ctx.STATE_KEY)
            if state:
                context["scene"] = {
                    "people_in_frame": int(state.get("num_people", 0)),
                }
                persons_raw = state.get("persons", "[]")
                try:
                    persons = json.loads(persons_raw)
                    if persons:
                        context["scene"]["persons"] = persons
                except (json.JSONDecodeError, TypeError):
                    pass
        except Exception:
            pass

        # Current time
        now = datetime.now(TZ_LOCAL)
        context["timestamp"] = now.strftime("%I:%M %p, %B %d %Y")
        context["time_period"] = "night" if now.hour < 6 or now.hour >= 21 else "day" if 8 <= now.hour < 18 else "twilight"

        # Return ONLY the small metadata to the LLM — no base64!
        return json.dumps({
            "snapshot_captured": True,
            "size_kb": round(len(frame) / 1024, 1),
            "context": context,
            "instruction": "A live camera snapshot has been captured and will be shown to the user automatically. Describe what you know from the context data: weather conditions, who is in frame, time of day. Do NOT try to include or reference the image data — it is handled for you.",
        })
    except Exception as e:
        _pending_snapshot = None
        return json.dumps({"error": str(e)})


def _tool_capture_clip() -> str:
    """Capture 5-second MP4 clip from the live camera."""
    global _pending_clip
    from routes.notifications import build_clip
    import uuid as _uuid

    try:
        mp4_bytes = build_clip(duration=5.0, fps=10)
        if not mp4_bytes:
            _pending_clip = None
            return json.dumps({"error": "Clip capture failed — camera may be offline or not enough frames"})

        # Save to disk so we can serve it via API
        clip_dir = os.path.join("/data/snapshots", "clips")
        os.makedirs(clip_dir, exist_ok=True)
        filename = f"{datetime.now(TZ_LOCAL).strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:6]}.mp4"
        filepath = os.path.join(clip_dir, filename)
        with open(filepath, "wb") as f:
            f.write(mp4_bytes)

        _pending_clip = filename

        # Get scene context for the AI to describe
        context = {}
        try:
            state = ctx.r.hgetall(ctx.STATE_KEY)
            if state:
                context["people_in_frame"] = int(state.get("num_people", 0))
                persons_raw = state.get("persons", "[]")
                try:
                    persons = json.loads(persons_raw)
                    if persons:
                        context["persons"] = persons
                except (json.JSONDecodeError, TypeError):
                    pass
        except Exception:
            pass

        now = datetime.now(TZ_LOCAL)
        context["timestamp"] = now.strftime("%I:%M %p, %B %d %Y")
        context["duration_seconds"] = 5
        context["size_kb"] = round(len(mp4_bytes) / 1024, 1)

        return json.dumps({
            "clip_captured": True,
            "context": context,
            "instruction": "A 5-second video clip has been recorded and will be shown to the user automatically. Describe what you know from the scene context. Do NOT try to embed the video data.",
        })
    except Exception as e:
        _pending_clip = None
        return json.dumps({"error": str(e)})


def _tool_query_notification_history(args: dict) -> str:
    """Get recent notification records from the feedback database."""
    count = min(int(args.get("count", 20)), 50)

    try:
        # Check for recent events that triggered notifications
        events_raw = ctx.r.xrevrange(ctx.EVENT_STREAM, count=count * 3)  # Over-fetch to find notified ones
        notified = []
        for msg_id, data in events_raw:
            if data.get("alert_triggered") == "true" or data.get("alert_triggered") == "1":
                notified.append({
                    "event_id": msg_id,
                    "type": data.get("type", "unknown"),
                    "person_id": data.get("person_id", ""),
                    "identity": data.get("identity_name", ""),
                    "zone": data.get("zone", ""),
                    "timestamp": data.get("timestamp", ""),
                    "alert_level": data.get("alert_level", ""),
                })
                if len(notified) >= count:
                    break

        return json.dumps({
            "notifications": notified,
            "count": len(notified),
            "note": "These events triggered Telegram notifications",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _tool_query_faces() -> str:
    """Query enrolled faces via the face recognizer API."""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{ctx.FACE_API_URL}/api/faces", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            # Face-recognizer returns {"faces": [...], "count": N}
            face_list = data.get("faces", data) if isinstance(data, dict) else data
            if not isinstance(face_list, list):
                face_list = []
            names = [f.get("name", "unknown") for f in face_list if isinstance(f, dict)]
            return json.dumps({"enrolled_faces": names, "count": len(names)})
        return json.dumps({"error": f"Face API returned {resp.status_code}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_query_feedback_stats() -> str:
    """Query feedback statistics."""
    if not _feedback_db:
        return json.dumps({"error": "Feedback DB not initialized"})
    stats = _feedback_db.get_stats()
    return json.dumps(stats)


async def _tool_send_telegram(args: dict) -> str:
    """Send a Telegram message, optionally with a live snapshot or video clip."""
    from routes.notifications import (
        send_text, send_photo, send_video, is_configured,
        get_latest_frame, build_clip,
    )

    message = args.get("message", "")
    if not message:
        return json.dumps({"error": "No message provided"})
    if not is_configured():
        return json.dumps({"error": "Telegram not configured"})

    try:
        include_clip = args.get("include_clip", False)
        include_snapshot = args.get("include_snapshot", False)

        if include_clip:
            clip = build_clip(duration=5.0, fps=10)
            if clip:
                msg_id = await send_video(clip, f"🎬 {message}")
                return json.dumps({"status": "sent_with_clip", "message": message, "message_id": msg_id})
            else:
                await send_text(f"{message}\n\n(Video clip unavailable — camera may be offline)")
                return json.dumps({"status": "sent_text_only", "message": message, "note": "Clip capture failed"})

        if include_snapshot:
            frame = get_latest_frame()
            if frame:
                msg_id = await send_photo(frame, message)
                return json.dumps({"status": "sent_with_snapshot", "message": message, "message_id": msg_id})
            else:
                await send_text(f"{message}\n\n(Snapshot unavailable — camera may be offline)")
                return json.dumps({"status": "sent_text_only", "message": message, "note": "No frame available"})

        await send_text(message)
        return json.dumps({"status": "sent", "message": message})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_schedule_reminder(args: dict) -> str:
    """Schedule a future Telegram reminder, optionally with media."""
    if not _ai_db:
        return json.dumps({"error": "AI DB not initialized"})

    message = args.get("message", "")
    time_desc = args.get("time_description", "")
    media_type = args.get("media_type", "text")
    if media_type not in ("text", "snapshot", "clip"):
        media_type = "text"
    if not message or not time_desc:
        return json.dumps({"error": "message and time_description required"})

    # Parse time — try ISO format first, then common patterns
    trigger_time = _parse_time(time_desc)
    if not trigger_time:
        return json.dumps({"error": f"Could not parse time: {time_desc}"})

    reminder_id = _ai_db.add_reminder(message, trigger_time, media_type=media_type)
    dt = datetime.fromtimestamp(trigger_time, tz=TZ_LOCAL)
    media_label = {"text": "text only", "snapshot": "with snapshot", "clip": "with 5s video clip"}
    return json.dumps({
        "status": "scheduled",
        "reminder_id": reminder_id,
        "message": message,
        "media_type": media_type,
        "scheduled_for": dt.strftime("%I:%M %p, %b %d"),
        "note": media_label.get(media_type, media_type),
    })


def _parse_time(time_desc: str) -> float | None:
    """Parse a time description into a Unix timestamp."""
    now = datetime.now(TZ_LOCAL)

    # Try ISO format
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.strptime(time_desc, fmt).replace(tzinfo=TZ_LOCAL)
            return dt.timestamp()
        except ValueError:
            pass

    # Try time-only formats (assume today or next occurrence)
    for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M"):
        try:
            parsed = datetime.strptime(time_desc.strip(), fmt)
            dt = now.replace(hour=parsed.hour, minute=parsed.minute, second=0)
            if dt <= now:
                dt = dt.replace(day=dt.day + 1)  # Next day
            return dt.timestamp()
        except ValueError:
            pass

    # Try relative: "in X minutes/hours"
    import re
    match = re.match(r"in\s+(\d+)\s+(minute|minutes|min|hour|hours|hr)", time_desc.lower())
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        if "hour" in unit or "hr" in unit:
            amount *= 3600
        else:
            amount *= 60
        return (now.timestamp() + amount)

    return None


def _tool_get_system_status() -> str:
    """Get system status from Redis."""
    try:
        stats = {
            "events_in_stream": ctx.r.xlen(ctx.EVENT_STREAM),
            "config": ctx.r.hgetall(ctx.CONFIG_KEY),
            "state": ctx.r.hgetall(ctx.STATE_KEY),
        }
        return json.dumps(stats)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_retrain_rules() -> str:
    """Retrain suppression rules from all feedback data."""
    if not _feedback_db:
        return json.dumps({"error": "Feedback database not available"})
    try:
        result = _feedback_db.retrain_rules()
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_review_feedback(args: dict) -> str:
    """Get recent feedback records."""
    if not _feedback_db:
        return json.dumps({"error": "Feedback database not available"})
    try:
        count = min(args.get("count", 20), 100)
        records = _feedback_db.get_recent_feedback(limit=count)
        return json.dumps({
            "count": len(records),
            "records": records,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Build system prompt
# ---------------------------------------------------------------------------
async def _build_system_context() -> str:
    """Gather a live system snapshot to inject into the system prompt."""
    parts = []
    # Enrolled faces
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{ctx.FACE_API_URL}/api/faces", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            # Face-recognizer returns {"faces": [...], "count": N}
            face_list = data.get("faces", data) if isinstance(data, dict) else data
            if not isinstance(face_list, list):
                face_list = []
            names = [f.get("name", "unknown") for f in face_list if isinstance(f, dict)]
            parts.append(f"Enrolled faces ({len(names)}): {', '.join(names) if names else 'none'}")
    except Exception:
        pass
    # Zones
    try:
        zone_data = ctx.r.hgetall(ctx.ZONE_KEY)
        count = len(zone_data) if zone_data else 0
        parts.append(f"Active zones: {count}")
    except Exception:
        pass
    # Feedback stats
    if _feedback_db:
        try:
            stats = _feedback_db.get_stats()
            parts.append(
                f"Feedback stats: {stats.get('total_feedback', 0)} total, "
                f"{stats.get('false_alarms', 0)} false alarms, "
                f"{stats.get('real_threats', 0)} real threats, "
                f"{stats.get('active_rules', 0)} active suppression rules"
            )
        except Exception:
            pass
    # Event stream size
    try:
        ev_len = ctx.r.xlen(ctx.EVENT_STREAM)
        parts.append(f"Events in stream: {ev_len}")
    except Exception:
        pass
    return "\n".join(parts)


def _build_system_prompt(config: dict, system_context: str = "") -> str:
    """Build the system prompt with personality and context."""
    ai_name = config.get("ai_name", "Atlas")
    user_name = config.get("user_name", "")
    user_ref = user_name if user_name else "the user"
    now = datetime.now(TZ_LOCAL)

    name_line = f"\nThe user's name is {user_name}. Address them by name occasionally." if user_name else ""
    context_block = f"\n\nCURRENT SYSTEM SNAPSHOT:\n{system_context}" if system_context else ""

    return f"""You are {ai_name}, a helpful and friendly AI assistant for a home security system called Vision Labs. You run locally on the user's own hardware — no data ever leaves this machine.{name_line}

Your primary role is helping {user_ref} monitor and manage their security cameras, but you're also happy to help with general questions, reminders, and conversation.

Current time: {now.strftime("%I:%M %p, %A %B %d, %Y")} ({now.tzname()}){context_block}

CAPABILITIES (use tools when relevant):
- Query recent security events (people detected, identified, vehicles)
- Query events by date (today, yesterday, specific date)
- Analyze event patterns — hourly trends, daily averages, busiest times
- Look up enrolled faces and unknown/auto-captured faces
- View the live scene — who's in frame right now, identified faces
- Capture a live camera snapshot and describe it to the user
- Get current weather conditions (temperature, humidity, wind)
- Browse vehicle detection snapshots by day
- Check feedback stats and suppression rules
- Retrain the alert suppression model from all feedback data
- Review recent feedback history (user verdicts on past alerts)
- Send Telegram messages immediately (with optional live camera snapshot or 5-second video clip)
- Schedule timed reminders via Telegram (with optional snapshot or video clip captured at scheduled time)
- Check system status, configuration, and zone definitions
- View notification history — what Telegram alerts were sent

SNAPSHOT & CLIP HANDLING:
When capture_snapshot or capture_clip returns, the media is automatically displayed to the user — you do NOT need to embed or reference it.
Just describe what you know from the context data the tool returns: weather conditions, who is in frame, time of day.
Do NOT try to output any image/video data or base64 strings.

PERSONALITY:
- Conversational and warm, but concise
- Security-aware — flag anything unusual if asked
- When {user_ref} asks about events, use the query_events tool to get real data
- When asked to send a message or set a reminder, use the appropriate tool
- For general questions unrelated to security, answer normally without tools
- Don't use tools unless the question actually requires data lookup

IMPORTANT: Do NOT wrap your response in <think> tags or show your reasoning process. Respond directly and naturally."""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("/status")
async def get_status():
    """Check if Ollama model is downloaded AND loaded into GPU memory."""
    global _model_gpu_ready
    try:
        client = ollama_lib.Client(host=OLLAMA_HOST)
        models = client.list()
        # The ollama library returns objects with .models attribute (list of Model objs)
        model_list = getattr(models, "models", None) or []
        model_names = []
        for m in model_list:
            name = getattr(m, "model", None) or getattr(m, "name", "") or ""
            model_names.append(name)
        target = OLLAMA_MODEL.split(":")[0]
        model_downloaded = any(target in name for name in model_names)

        if not model_downloaded:
            return {"model_ready": False, "model": OLLAMA_MODEL, "status": "not_found"}

        # Model is downloaded — check if it's in GPU memory.
        # First check our flag (set by warm-up chat)
        if _model_gpu_ready:
            return {"model_ready": True, "model": OLLAMA_MODEL, "status": "ready"}

        # Flag not set yet — check Ollama's /api/ps (running models list)
        # This is faster than waiting for the warm-up chat to complete
        try:
            ps = client.ps()
            running_models = getattr(ps, "models", None) or []
            for rm in running_models:
                rm_name = getattr(rm, "model", None) or getattr(rm, "name", "") or ""
                if target in rm_name:
                    # Model is loaded in VRAM — set flag and return ready
                    _model_gpu_ready = True
                    logger.info(f"Model '{OLLAMA_MODEL}' detected in GPU memory via /api/ps")
                    return {"model_ready": True, "model": OLLAMA_MODEL, "status": "ready"}
        except Exception:
            pass  # ps() not available in older ollama versions, fall through

        return {"model_ready": False, "model": OLLAMA_MODEL, "status": "loading"}
    except Exception as e:
        logger.warning(f"Ollama status check failed: {e}")
        return {"model_ready": False, "model": OLLAMA_MODEL, "status": "offline"}


@router.get("/config")
async def get_config():
    """Get AI assistant configuration."""
    if not _ai_db:
        return {"enabled": False, "user_name": "", "ai_name": "Atlas"}
    return _ai_db.get_config()


@router.post("/config")
async def save_config(req: ConfigRequest):
    """Save AI assistant configuration (onboarding)."""
    if not _ai_db:
        return JSONResponse(status_code=503, content={"error": "AI DB not initialized"})
    return _ai_db.save_config(
        enabled=req.enabled,
        user_name=req.user_name,
        ai_name=req.ai_name,
    )


@router.post("/chat")
async def chat(req: ChatRequest):
    """
    Send a message to the AI and get a streamed response.
    Handles tool calls transparently — the user sees only the final answer.
    """
    if not _ai_db:
        return JSONResponse(status_code=503, content={"error": "AI DB not initialized"})

    config = _ai_db.get_config()
    if not config.get("enabled"):
        return JSONResponse(status_code=400, content={"error": "AI assistant not enabled"})

    # Build message list with live system context
    system_context = await _build_system_context()
    system_prompt = _build_system_prompt(config, system_context)
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (from client)
    for msg in req.history[-20:]:  # Last 20 messages for context
        if msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current message
    messages.append({"role": "user", "content": req.message})

    # Save user message server-side
    _ai_db.save_message("user", req.message)

    # Configure Ollama client
    client = ollama_lib.Client(host=OLLAMA_HOST)

    try:
        # First call — may include tool calls
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=TOOLS,
            options={"num_ctx": 8192},
        )

        # Handle tool calls if any
        tool_rounds = 0
        while response.message.tool_calls and tool_rounds < 5:
            tool_rounds += 1
            # Add assistant's tool call message
            messages.append(response.message)

            # Execute each tool call
            for tool_call in response.message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                logger.info(f"Tool call: {tool_name}({tool_args})")

                result = await _execute_tool(tool_name, tool_args)
                messages.append({
                    "role": "tool",
                    "content": result,
                })

            # Get the next response with tool results
            response = client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                tools=TOOLS,
                options={"num_ctx": 8192},
            )

        # Extract final response text
        reply = response.message.content or ""

        # Strip <think> blocks if Qwen includes them
        if "<think>" in reply:
            import re
            reply = re.sub(r"<think>.*?</think>\s*", "", reply, flags=re.DOTALL).strip()

        # Inject snapshot image if one was captured during this request
        global _pending_snapshot
        if _pending_snapshot:
            snapshot_md = f"![Live snapshot](data:image/jpeg;base64,{_pending_snapshot})"
            reply = f"{snapshot_md}\n\n{reply}"
            _pending_snapshot = None

        # Inject video clip if one was captured during this request
        global _pending_clip
        if _pending_clip:
            clip_url = f"/api/ai/clip/{_pending_clip}"
            clip_html = f'<video controls autoplay muted playsinline style="max-width:100%;border-radius:8px;margin:8px 0;"><source src="{clip_url}" type="video/mp4">Your browser does not support video.</video>'
            reply = f"{clip_html}\n\n{reply}"
            _pending_clip = None

        # Save assistant response server-side
        _ai_db.save_message("assistant", reply)

        return {"reply": reply}

    except Exception as e:
        logger.error(f"AI chat error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"AI unavailable: {str(e)}"},
        )


@router.get("/history")
async def get_history(limit: int = 50):
    """Get server-side chat history."""
    if not _ai_db:
        return []
    return _ai_db.get_recent_history(limit=limit)


@router.get("/clip/{filename}")
async def serve_clip(filename: str):
    """Serve a saved AI-captured video clip."""
    from fastapi.responses import FileResponse
    import re as _re
    # Sanitize filename — only allow safe characters
    if not _re.match(r'^[\w\-]+\.mp4$', filename):
        return JSONResponse(status_code=400, content={"error": "Invalid filename"})
    filepath = os.path.join("/data/snapshots", "clips", filename)
    if not os.path.isfile(filepath):
        return JSONResponse(status_code=404, content={"error": "Clip not found"})
    return FileResponse(filepath, media_type="video/mp4")


@router.delete("/history")
async def clear_history():
    """Clear chat history."""
    if not _ai_db:
        return {"status": "ok"}
    _ai_db.clear_history()
    return {"status": "ok"}


@router.post("/reset")
async def reset_assistant():
    """Reset AI assistant — clears config and history, re-shows wizard."""
    if not _ai_db:
        return {"status": "ok"}
    _ai_db.save_config(enabled=False, user_name="", ai_name="Atlas")
    _ai_db.clear_history()
    return {"status": "ok"}


@router.get("/reminders")
async def get_reminders():
    """Get upcoming reminders."""
    if not _ai_db:
        return []
    return _ai_db.get_reminders()
