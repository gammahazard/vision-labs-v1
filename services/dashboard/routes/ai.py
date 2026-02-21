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
            "description": "Send a message to the user via Telegram right now.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message text to send",
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
            "description": "Schedule a reminder to be sent via Telegram at a specific time. Use ISO format or relative descriptions.",
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


async def _tool_query_faces() -> str:
    """Query enrolled faces via the face recognizer API."""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{ctx.FACE_API_URL}/faces", timeout=5)
        if resp.status_code == 200:
            faces = resp.json()
            names = [f.get("name", "unknown") for f in faces]
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
    """Send a Telegram message immediately."""
    from routes.notifications import send_text, is_configured

    message = args.get("message", "")
    if not message:
        return json.dumps({"error": "No message provided"})
    if not is_configured():
        return json.dumps({"error": "Telegram not configured"})

    try:
        await send_text(message)
        return json.dumps({"status": "sent", "message": message})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_schedule_reminder(args: dict) -> str:
    """Schedule a future Telegram reminder."""
    if not _ai_db:
        return json.dumps({"error": "AI DB not initialized"})

    message = args.get("message", "")
    time_desc = args.get("time_description", "")
    if not message or not time_desc:
        return json.dumps({"error": "message and time_description required"})

    # Parse time — try ISO format first, then common patterns
    trigger_time = _parse_time(time_desc)
    if not trigger_time:
        return json.dumps({"error": f"Could not parse time: {time_desc}"})

    reminder_id = _ai_db.add_reminder(message, trigger_time)
    dt = datetime.fromtimestamp(trigger_time, tz=TZ_LOCAL)
    return json.dumps({
        "status": "scheduled",
        "reminder_id": reminder_id,
        "message": message,
        "scheduled_for": dt.strftime("%I:%M %p, %b %d"),
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


# ---------------------------------------------------------------------------
# Build system prompt
# ---------------------------------------------------------------------------
def _build_system_prompt(config: dict) -> str:
    """Build the system prompt with personality and context."""
    ai_name = config.get("ai_name", "Atlas")
    user_name = config.get("user_name", "")
    user_ref = user_name if user_name else "the user"
    now = datetime.now(TZ_LOCAL)

    return f"""You are {ai_name}, a helpful and friendly AI assistant for a home security system called Vision Labs. You run locally on the user's own hardware — no data ever leaves this machine.

Your primary role is helping {user_ref} monitor and manage their security cameras, but you're also happy to help with general questions, reminders, and conversation.

Current time: {now.strftime("%I:%M %p, %A %B %d, %Y")} ({now.tzname()})

CAPABILITIES (use tools when relevant):
- Query recent security events (people detected, identified, vehicles)
- Look up enrolled faces
- Check feedback stats and suppression rules
- Send Telegram messages immediately
- Schedule timed reminders via Telegram
- Check system status and configuration

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

    # Build message list
    system_prompt = _build_system_prompt(config)
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


@router.delete("/history")
async def clear_history():
    """Clear chat history."""
    if not _ai_db:
        return {"status": "ok"}
    _ai_db.clear_history()
    return {"status": "ok"}


@router.get("/reminders")
async def get_reminders():
    """Get upcoming reminders."""
    if not _ai_db:
        return []
    return _ai_db.get_reminders()
