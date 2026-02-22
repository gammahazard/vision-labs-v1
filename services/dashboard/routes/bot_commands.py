"""
routes/bot_commands.py — Telegram bot command handlers and polling loop.

PURPOSE:
    Handles incoming Telegram updates via long-polling:
    1. Bot commands: /snapshot, /clip [N], /status, /arm, /disarm, /who, /events [N], /help
    2. Callback queries: Verdict buttons (✅ Real | ❌ False | 👤 Name)

    All incoming updates are validated via _is_authorized() before
    processing. Unauthorized users are silently ignored.

EXTRACTED FROM:
    notifications.py — to reduce file size and separate concerns.
    Bot commands (read-only + arm/disarm) are distinct from alert-sending
    functions (notify_person_detected, etc.).
"""

import os
import re
import json
import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import redis
import httpx

import routes as ctx
from routes.notifications import (
    is_configured, _is_authorized,
    send_text, send_photo, send_video,
    edit_message_buttons, answer_callback_query,
    get_latest_frame, build_clip, _now_str,
    TELEGRAM_API, TELEGRAM_CHAT_ID, TELEGRAM_ALLOWED_USERS,
    REDIS_HOST, REDIS_PORT,
)

logger = logging.getLogger("dashboard.notifications")

# Timezone
TZ_LOCAL = ZoneInfo(os.getenv("LOCATION_TIMEZONE", "America/Toronto"))

# Telegram update offset — tracks which updates we've processed
_telegram_update_offset = 0


def _log_access(user_id, username, first_name, chat_id, action, authorized,
                last_name="", language_code=""):
    """Log an access attempt to the Redis access log stream."""
    try:
        if ctx.r and ctx.TELEGRAM_ACCESS_LOG:
            ctx.r.xadd(ctx.TELEGRAM_ACCESS_LOG, {
                "user_id": str(user_id or ""),
                "username": username or "",
                "first_name": first_name or "",
                "last_name": last_name or "",
                "language_code": language_code or "",
                "chat_id": str(chat_id or ""),
                "action": action,
                "authorized": "true" if authorized else "false",
                "timestamp": datetime.now(TZ_LOCAL).strftime("%Y-%m-%d %H:%M:%S"),
            }, maxlen=500)
    except Exception as e:
        logger.debug(f"Access log write failed: {e}")


def _seed_users_from_env():
    """On first startup, seed Redis users hash from env vars if empty."""
    if not ctx.r or not ctx.TELEGRAM_USERS_KEY:
        return
    if ctx.r.hlen(ctx.TELEGRAM_USERS_KEY) > 0:
        return  # Already has users
    if not TELEGRAM_ALLOWED_USERS:
        return
    for uid in TELEGRAM_ALLOWED_USERS:
        meta = json.dumps({
            "chat_id": TELEGRAM_CHAT_ID,
            "name": "Admin (seeded)",
            "username": "",
            "role": "admin",
            "approved_at": datetime.now(TZ_LOCAL).strftime("%Y-%m-%d %H:%M"),
        })
        ctx.r.hset(ctx.TELEGRAM_USERS_KEY, str(uid), meta)
    logger.info(f"Seeded {len(TELEGRAM_ALLOWED_USERS)} user(s) from TELEGRAM_ALLOWED_USERS env var")


# ---------------------------------------------------------------------------
# Polling loop — runs as a background task
# ---------------------------------------------------------------------------
async def poll_telegram_callbacks(feedback_db):
    """
    Background task: poll Telegram for updates (callback queries + commands).

    Handles two types of incoming updates:
    1. callback_query — verdict buttons (✅/❌/👤) on notification messages
    2. message — bot commands (/snapshot, /clip, /status, /arm, /disarm, /who)

    Security: ALL incoming updates are validated via _is_authorized() before
    processing. Unauthorized users are silently ignored.
    """
    global _telegram_update_offset

    if not is_configured():
        logger.info("Telegram not configured — callback poller disabled")
        return

    # Seed users from env var on first startup
    _seed_users_from_env()

    if TELEGRAM_ALLOWED_USERS:
        logger.info(f"Telegram poller started — authorized users: {TELEGRAM_ALLOWED_USERS}")
    else:
        logger.warning("Telegram poller started — NO user whitelist set (commands disabled)")

    # Register bot command menu with Telegram
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{TELEGRAM_API}/setMyCommands",
                json={"commands": [
                    {"command": "snapshot", "description": "Live camera photo"},
                    {"command": "clip", "description": "Video clip (5-40s)"},
                    {"command": "status", "description": "System health"},
                    {"command": "ask", "description": "Ask the AI assistant"},
                    {"command": "arm", "description": "Enable notifications (admin)"},
                    {"command": "disarm", "description": "Disable notifications (admin)"},
                    {"command": "who", "description": "Who's in frame now"},
                    {"command": "events", "description": "Recent detections (1-20)"},
                    {"command": "help", "description": "List all commands"},
                ]},
                timeout=10,
            )
        logger.info("Telegram command menu registered")
    except Exception as e:
        logger.warning(f"Failed to register command menu: {e}")

    while True:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{TELEGRAM_API}/getUpdates",
                    params={
                        "offset": _telegram_update_offset,
                        "timeout": 30,
                        "allowed_updates": json.dumps(["callback_query", "message"]),
                    },
                    timeout=40,
                )

            if resp.status_code != 200:
                logger.warning(f"getUpdates failed: {resp.status_code}")
                await asyncio.sleep(5)
                continue

            updates = resp.json().get("result", [])
            for update in updates:
                _telegram_update_offset = update["update_id"] + 1

                # --- Callback queries (verdict buttons) ---
                cb = update.get("callback_query")
                if cb:
                    cb_from = cb.get("from", {})
                    cb_user_id = cb_from.get("id")
                    cb_username = cb_from.get("username", "")
                    cb_first = cb_from.get("first_name", "")
                    cb_last = cb_from.get("last_name", "")
                    cb_lang = cb_from.get("language_code", "")
                    cb_chat_id = cb.get("message", {}).get("chat", {}).get("id")
                    authorized = _is_authorized(cb_user_id, cb_chat_id)
                    _log_access(cb_user_id, cb_username, cb_first,
                                cb_chat_id, "callback", authorized,
                                last_name=cb_last, language_code=cb_lang)
                    if not authorized:
                        logger.warning(f"Unauthorized callback from user {cb_user_id}")
                        # Emit event so it shows in the dashboard events feed
                        try:
                            ctx.r.xadd(ctx.EVENT_STREAM, {
                                "camera_id": ctx.CAMERA_ID,
                                "event_type": "unauthorized_access",
                                "timestamp": str(datetime.now().timestamp()),
                                "person_id": "",
                                "identity_name": f"{cb_first} {cb_last}".strip() or cb_username or str(cb_user_id),
                                "duration": "0",
                                "direction": "",
                                "action": "callback",
                                "bbox": "",
                                "frame_count": "0",
                                "zone": "",
                                "alert_level": "alert",
                                "alert_triggered": "True",
                                "telegram_user_id": str(cb_user_id),
                                "telegram_username": cb_username,
                                "time_period": "",
                            }, maxlen=5000)
                        except Exception:
                            pass
                        continue
                    await _handle_callback(
                        cb.get("data", ""),
                        cb.get("id", ""),
                        cb.get("message", {}).get("message_id", 0),
                        feedback_db,
                        chat_id=str(cb_chat_id),
                    )
                    continue

                # --- Messages (bot commands) ---
                msg = update.get("message")
                if msg:
                    msg_from = msg.get("from", {})
                    msg_user_id = msg_from.get("id")
                    msg_username = msg_from.get("username", "")
                    msg_first = msg_from.get("first_name", "")
                    msg_last = msg_from.get("last_name", "")
                    msg_lang = msg_from.get("language_code", "")
                    msg_chat_id = msg.get("chat", {}).get("id")
                    text = msg.get("text", "").strip()

                    authorized = _is_authorized(msg_user_id, msg_chat_id)
                    _log_access(msg_user_id, msg_username, msg_first,
                                msg_chat_id, text or "(empty)", authorized,
                                last_name=msg_last, language_code=msg_lang)

                    if not authorized:
                        # Silent rejection — don't reveal bot exists
                        logger.warning(f"Unauthorized command from user {msg_user_id}: {text}")
                        # Emit event so it shows in the dashboard events feed
                        try:
                            ctx.r.xadd(ctx.EVENT_STREAM, {
                                "camera_id": ctx.CAMERA_ID,
                                "event_type": "unauthorized_access",
                                "timestamp": str(datetime.now().timestamp()),
                                "person_id": "",
                                "identity_name": f"{msg_first} {msg_last}".strip() or msg_username or str(msg_user_id),
                                "duration": "0",
                                "direction": "",
                                "action": text.split()[0] if text else "(empty)",
                                "bbox": "",
                                "frame_count": "0",
                                "zone": "",
                                "alert_level": "alert",
                                "alert_triggered": "True",
                                "telegram_user_id": str(msg_user_id),
                                "telegram_username": msg_username,
                                "time_period": "",
                            }, maxlen=5000)
                        except Exception:
                            pass
                        continue

                    # Route to command handlers
                    if text.startswith("/"):
                        cmd = text.split()[0].lower().split("@")[0]  # Strip @botname
                        logger.info(f"Command from user {msg_user_id}: {cmd}")
                        await _handle_command(cmd, chat_id=str(msg_chat_id),
                                              text=text, user_id=str(msg_user_id))

        except httpx.ReadTimeout:
            # Normal — long poll timed out with no updates
            pass
        except Exception as e:
            logger.warning(f"Callback poller error: {e}")
            await asyncio.sleep(5)


# ---------------------------------------------------------------------------
# Command router
# ---------------------------------------------------------------------------
def _get_user_role(user_id: str) -> str:
    """Get user role from Redis. Returns 'admin' or 'user'."""
    try:
        if ctx.r and ctx.TELEGRAM_USERS_KEY:
            raw = ctx.r.hget(ctx.TELEGRAM_USERS_KEY, user_id)
            if raw:
                meta_str = raw if isinstance(raw, str) else raw.decode()
                data = json.loads(meta_str)
                return data.get("role", "user")
    except Exception:
        pass
    return "user"


async def _handle_command(cmd: str, chat_id: str = "", text: str = "",
                          user_id: str = ""):
    """Route a bot command to the appropriate handler."""
    # Commands that accept args from the raw text
    args_handlers = {
        "/clip": _cmd_clip,
        "/events": _cmd_events,
        "/ask": _cmd_ask,
    }
    # Admin-only commands
    admin_handlers = {
        "/arm": _cmd_arm,
        "/disarm": _cmd_disarm,
    }
    # Simple commands
    simple_handlers = {
        "/snapshot": _cmd_snapshot,
        "/status": _cmd_status,
        "/who": _cmd_who,
        "/start": _cmd_help,
        "/help": _cmd_help,
    }

    try:
        if cmd in admin_handlers:
            role = _get_user_role(user_id)
            if role != "admin":
                await send_text("🔒 This command is reserved for admins.", chat_id=chat_id)
                return
            await admin_handlers[cmd](chat_id=chat_id)
        elif cmd in args_handlers:
            await args_handlers[cmd](chat_id=chat_id, text=text)
        elif cmd in simple_handlers:
            await simple_handlers[cmd](chat_id=chat_id)
        else:
            await _cmd_help(chat_id=chat_id)
    except Exception as e:
        logger.warning(f"Command {cmd} failed: {e}")
        await send_text(f"⚠️ Command failed: {e}", chat_id=chat_id)


# ---------------------------------------------------------------------------
# Bot command implementations
# ---------------------------------------------------------------------------
async def _cmd_snapshot(chat_id: str = ""):
    """Send a live camera snapshot."""
    frame = get_latest_frame()
    if frame:
        await send_photo(frame, f"📸 Live snapshot — {_now_str()}", chat_id=chat_id)
    else:
        await send_text("⚠️ No camera frame available", chat_id=chat_id)


async def _cmd_clip(chat_id: str = "", text: str = ""):
    """Capture and send a video clip (5-40s, default 5)."""
    # Parse optional duration from text: /clip 15
    duration = 5.0
    parts = text.split()
    if len(parts) >= 2:
        try:
            duration = float(parts[1])
            duration = max(5.0, min(40.0, duration))
        except (ValueError, IndexError):
            pass

    await send_text(f"🎬 Recording {int(duration)}-second clip...", chat_id=chat_id)
    loop = asyncio.get_running_loop()
    clip_bytes = await loop.run_in_executor(
        None, lambda: build_clip(duration=duration, fps=10)
    )
    if clip_bytes:
        await send_video(clip_bytes, f"🎬 {int(duration)}s clip — {_now_str()}", chat_id=chat_id)
    else:
        await send_text("⚠️ Failed to capture clip — not enough frames", chat_id=chat_id)


async def _cmd_status(chat_id: str = ""):
    """Send system health summary."""
    try:
        r = ctx.r  # Use the centralized Redis connection
        info = r.info("memory")
        mem_used = info.get("used_memory_human", "?")

        # Check frame stream health
        frame_len = r.xlen(ctx.FRAME_STREAM) if ctx.FRAME_STREAM else 0
        # HD frame check needs raw bytes connection
        r_raw = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
        hd_exists = bool(r_raw.get(ctx.HD_FRAME_KEY.encode())) if ctx.HD_FRAME_KEY else False

        # Check event stream length
        event_len = r.xlen(ctx.EVENT_STREAM)

        # Read notification preferences from Redis config
        cfg = r.hgetall(ctx.CONFIG_KEY)
        person_on = cfg.get("notify_person", "1") == "1"
        vehicle_on = cfg.get("notify_vehicle", "1") == "1"
        if person_on and vehicle_on:
            alert_str = "🟢 All alerts on"
        elif not person_on and not vehicle_on:
            alert_str = "🔴 All alerts off"
        else:
            parts_a = []
            if person_on: parts_a.append("Person")
            if vehicle_on: parts_a.append("Vehicle")
            alert_str = f"🟡 {', '.join(parts_a)} only"

        status = (
            f"📊 <b>System Status</b>\n"
            f"• Notifications: {alert_str}\n"
            f"• Redis memory: {mem_used}\n"
            f"• Frame buffer: {frame_len} frames\n"
            f"• HD stream: {'✅' if hd_exists else '❌'}\n"
            f"• Events total: {event_len}\n"
            f"• Time: {_now_str()}"
        )
        await send_text(status, chat_id=chat_id)
    except Exception as e:
        await send_text(f"⚠️ Status check failed: {e}", chat_id=chat_id)


async def _cmd_arm(chat_id: str = ""):
    """Enable all notifications by setting Redis config."""
    try:
        ctx.r.hset(ctx.CONFIG_KEY, mapping={
            "notify_person": "1",
            "notify_vehicle": "1",
        })
        await send_text("🟢 Notifications <b>armed</b> — person + vehicle alerts enabled.", chat_id=chat_id)
        logger.info("Notifications armed via Telegram (wrote Redis config)")
    except Exception as e:
        await send_text(f"⚠️ Failed to arm: {e}", chat_id=chat_id)


async def _cmd_disarm(chat_id: str = ""):
    """Disable all notifications by setting Redis config."""
    try:
        ctx.r.hset(ctx.CONFIG_KEY, mapping={
            "notify_person": "0",
            "notify_vehicle": "0",
        })
        await send_text("🔴 Notifications <b>disarmed</b> — all alerts paused until you /arm again.", chat_id=chat_id)
        logger.info("Notifications disarmed via Telegram (wrote Redis config)")
    except Exception as e:
        await send_text(f"⚠️ Failed to disarm: {e}", chat_id=chat_id)


async def _cmd_who(chat_id: str = ""):
    """Report who/what is currently in the camera frame."""
    try:
        state = ctx.r.hgetall(ctx.STATE_KEY)
        if not state:
            await send_text("👀 No detection state available — scene may be clear.", chat_id=chat_id)
            return

        parts = ["👁️ <b>Current Scene</b>"]

        # People
        num_people = int(state.get("num_people", "0"))
        if num_people > 0:
            parts.append(f"• People: {num_people}")
            try:
                people = json.loads(state.get("people", "[]"))
                for p in people[:5]:
                    name = p.get("identity_name", p.get("id", "unknown"))
                    action = p.get("action", "")
                    parts.append(f"  — {name}{f' ({action})' if action else ''}")
            except json.JSONDecodeError:
                pass
        else:
            parts.append("• People: none")

        # Vehicles (check if tracker publishes vehicle info)
        num_vehicles = int(state.get("num_vehicles", "0"))
        if num_vehicles > 0:
            parts.append(f"• Vehicles: {num_vehicles}")
            try:
                vehicles = json.loads(state.get("vehicles", "[]"))
                for v in vehicles[:5]:
                    parts.append(f"  — {v.get('class', 'vehicle')}")
            except json.JSONDecodeError:
                pass
        else:
            parts.append("• Vehicles: none")

        parts.append(f"• Time: {_now_str()}")
        await send_text("\n".join(parts), chat_id=chat_id)
    except Exception as e:
        await send_text(f"⚠️ Failed to read scene state: {e}", chat_id=chat_id)


async def _cmd_help(chat_id: str = ""):
    """Send list of available commands."""
    await send_text(
        "🤖 <b>Vision Labs Bot</b>\n\n"
        "/snapshot — 📸 Live camera photo\n"
        "/clip [5-40] — 🎬 Video clip (default 5s)\n"
        "/status — 📊 System health\n"
        "/who — 👁️ Who's in frame now\n"
        "/events [1-20] — 📋 Recent detections (default 5)\n"
        "/ask [question] — 🧠 Ask the AI assistant\n\n"
        "🔒 <b>Admin Only</b>\n"
        "/arm — 🟢 Enable notifications\n"
        "/disarm — 🔴 Disable notifications",
        chat_id=chat_id,
    )


# Snapshot directory — same as server.py uses
SNAPSHOT_DIR = os.environ.get("SNAPSHOT_DIR", "/data/snapshots")


async def _cmd_events(chat_id: str = "", text: str = ""):
    """Show recent detection events with snapshot images."""
    # Parse optional count from text: /events 10
    count = 5
    parts_args = text.split()
    if len(parts_args) >= 2:
        try:
            count = int(parts_args[1])
            count = max(1, min(20, count))
        except (ValueError, IndexError):
            pass

    try:
        r_ev = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        entries = r_ev.xrevrange(ctx.EVENT_STREAM, count=count)
        if not entries:
            await send_text("📋 No events recorded yet.", chat_id=chat_id)
            return

        await send_text(f"📋 <b>Recent Events</b> (showing {len(entries)})", chat_id=chat_id)

        for msg_id, data in entries:
            etype = data.get("event_type", "unknown")
            identity = data.get("identity_name", "")
            person_id = data.get("person_id", "")
            zone = data.get("zone", "")
            ts_raw = data.get("timestamp", "")

            icons = {
                "person_appeared": "🚨",
                "person_identified": "👤",
                "vehicle_detected": "🚗",
                "vehicle_idle": "🚗",
            }
            icon = icons.get(etype, "📌")
            who = identity if identity else person_id if person_id else "unknown"

            # Format timestamp: convert unix float to readable time
            time_str = ""
            if ts_raw:
                try:
                    ts_float = float(ts_raw)
                    dt = datetime.fromtimestamp(ts_float, tz=TZ_LOCAL)
                    time_str = dt.strftime("%I:%M %p")
                except (ValueError, OSError):
                    time_str = ts_raw  # Fallback to raw if not a float

            caption = f"{icon} <b>{etype.replace('_', ' ').title()}</b>"
            if who and who != "unknown":
                caption += f" — {who}"
            if zone:
                caption += f" ({zone})"
            if time_str:
                caption += f"\n🕐 {time_str}"

            # Try to send event snapshot as photo
            safe_id = msg_id.replace(":", "-") if isinstance(msg_id, str) else msg_id.decode().replace(":", "-")
            snap_path = os.path.join(SNAPSHOT_DIR, f"{safe_id}.jpg")
            sent_photo = False
            if os.path.isfile(snap_path):
                try:
                    with open(snap_path, "rb") as f:
                        snap_bytes = f.read()
                    if snap_bytes:
                        await send_photo(snap_bytes, caption, chat_id=chat_id)
                        sent_photo = True
                except Exception:
                    pass

            if not sent_photo:
                await send_text(caption, chat_id=chat_id)

    except Exception as e:
        await send_text(f"⚠️ Failed to fetch events: {e}", chat_id=chat_id)


# ---------------------------------------------------------------------------
# Callback handler — verdict buttons on notification messages
# ---------------------------------------------------------------------------
async def _handle_callback(callback_data: str, callback_id: str,
                            message_id: int, feedback_db,
                            chat_id: str = ""):
    """
    Process a Telegram callback_query from an inline keyboard button.

    callback_data format: "v:{verdict}:{event_id}"
      - v:real:{event_id}     → Real threat
      - v:false:{event_id}    → False alarm
      - v:identify:{event_id} → User wants to name this person
    """
    if callback_data == "noop":
        await answer_callback_query(callback_id, "Already recorded")
        return

    parts = callback_data.split(":", 2)
    if len(parts) != 3 or parts[0] != "v":
        await answer_callback_query(callback_id, "Unknown action")
        return

    _, verdict_code, event_id = parts

    verdict_map = {
        "real": "real_threat",
        "false": "false_alarm",
        "identify": "identified",
    }
    verdict = verdict_map.get(verdict_code)
    if not verdict:
        await answer_callback_query(callback_id, "Unknown verdict")
        return

    if verdict == "identified":
        feedback_db.resolve_pending(event_id, "identified", identity_label="")
        await answer_callback_query(callback_id, "Marked for identification — name from dashboard")
        await edit_message_buttons(message_id, "👤 Awaiting name (dashboard)", chat_id=chat_id)
        logger.info(f"Event {event_id}: marked for identification via Telegram")
    else:
        feedback_db.resolve_pending(event_id, verdict)
        label = "✅ Real Threat" if verdict == "real_threat" else "❌ False Alarm"
        await answer_callback_query(callback_id, f"Recorded: {label}")
        await edit_message_buttons(message_id, f"{label} — Recorded", chat_id=chat_id)
        logger.info(f"Event {event_id}: verdict={verdict} via Telegram")


# ---------------------------------------------------------------------------
# /ask — AI assistant via Telegram
# ---------------------------------------------------------------------------
# Ollama config (same as ai.py)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = "qwen3:14b"


async def _send_long_text(text: str, chat_id: str = ""):
    """Send text, splitting at 4096 chars if needed (Telegram limit)."""
    MAX = 4000  # Leave some margin for parse_mode overhead
    if len(text) <= MAX:
        await send_text(text, chat_id=chat_id)
        return
    # Split on double-newline or single-newline boundaries
    while text:
        if len(text) <= MAX:
            await send_text(text, chat_id=chat_id)
            break
        # Find a good split point
        split_at = text.rfind("\n\n", 0, MAX)
        if split_at < 200:
            split_at = text.rfind("\n", 0, MAX)
        if split_at < 200:
            split_at = MAX
        await send_text(text[:split_at], chat_id=chat_id)
        text = text[split_at:].lstrip("\n")


async def _cmd_ask(chat_id: str = "", text: str = ""):
    """Ask the local AI assistant a question via Telegram."""
    # Extract the question from the message text
    question = text[len("/ask"):].strip() if text.startswith("/ask") else text.strip()
    if not question:
        await send_text(
            "🧠 <b>Ask the AI</b>\n\n"
            "Usage: /ask [your question]\n\n"
            "Examples:\n"
            "• /ask how many people were detected today?\n"
            "• /ask what's the weather like?\n"
            "• /ask take a snapshot and describe the scene\n"
            "• /ask show me vehicle detections from today",
            chat_id=chat_id,
        )
        return

    # Send "thinking" indicator
    await send_text("🧠 Thinking...", chat_id=chat_id)

    try:
        import ollama as ollama_lib
        import routes.ai_state as ai_state
        from routes.ai_tools import TOOLS, execute_tool
        from routes.ai_prompts import build_system_context, build_system_prompt

        # Check if AI is configured
        if not ai_state._ai_db:
            await send_text("⚠️ AI assistant not initialized. Set up via the dashboard first.", chat_id=chat_id)
            return

        config = ai_state._ai_db.get_config()
        if not config.get("enabled"):
            await send_text("⚠️ AI assistant is disabled. Enable it via the dashboard.", chat_id=chat_id)
            return

        # Build system prompt with live context
        system_context = await build_system_context()
        system_prompt = build_system_prompt(config, system_context)

        # Add Telegram-specific instruction
        system_prompt += (
            "\n\nYou are replying via Telegram. Keep responses concise. "
            "Use plain text or minimal HTML formatting (<b>bold</b>, <i>italic</i>). "
            "Do NOT use markdown. Do NOT include image/video data."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        # Call Ollama
        client = ollama_lib.Client(host=OLLAMA_HOST)
        loop = asyncio.get_running_loop()

        # Per-request media tracking to avoid race conditions with web chat
        import uuid as _uuid
        request_id = _uuid.uuid4().hex
        ai_state.set_request_id(request_id)

        response = await loop.run_in_executor(
            None,
            lambda: client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                tools=TOOLS,
                options={"num_ctx": 8192},
            ),
        )

        # Handle tool calls (up to 5 rounds)
        tool_rounds = 0
        while response.message.tool_calls and tool_rounds < 5:
            tool_rounds += 1
            messages.append(response.message)

            for tool_call in response.message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                logger.info(f"AI tool call (Telegram): {tool_name}({tool_args})")

                result = await execute_tool(tool_name, tool_args)
                messages.append({"role": "tool", "content": result})

            response = await loop.run_in_executor(
                None,
                lambda: client.chat(
                    model=OLLAMA_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    options={"num_ctx": 8192},
                ),
            )

        # Extract reply
        reply = response.message.content or ""

        # Strip <think> blocks
        if "<think>" in reply:
            reply = re.sub(r"<think>.*?</think>\s*", "", reply, flags=re.DOTALL).strip()

        # Collect media stashed by tools during this request
        media = ai_state.collect_media(request_id)

        # Snapshot: send as photo to this user's chat
        if media["snapshot"]:
            try:
                import base64
                snap_bytes = base64.b64decode(media["snapshot"])
                await send_photo(snap_bytes, f"📸 AI snapshot — {_now_str()}", chat_id=chat_id)
            except Exception as e:
                logger.debug(f"Failed to send AI snapshot via Telegram: {e}")

        # Clip: read file and send as video
        if media["clip"]:
            try:
                clip_dir = os.path.join("/data/snapshots", "clips")
                clip_path = os.path.join(clip_dir, media["clip"])
                if os.path.isfile(clip_path):
                    with open(clip_path, "rb") as f:
                        clip_data = f.read()
                    await send_video(clip_data, f"🎬 AI clip — {_now_str()}", chat_id=chat_id)
            except Exception as e:
                logger.debug(f"Failed to send AI clip via Telegram: {e}")

        # Browse images (vehicle snapshots etc.)
        if media["images"]:
            for img_info in media["images"][:5]:  # Cap at 5
                try:
                    url = img_info.get("url", "")
                    caption = img_info.get("caption", "")
                    # Vehicle snapshots: /api/browse/snapshot/{date}/{filename}
                    if url.startswith("/api/browse/snapshot/"):
                        # Extract date/filename from URL path
                        path_parts = url.replace("/api/browse/snapshot/", "").split("/", 1)
                        if len(path_parts) == 2:
                            date_part, fname = path_parts
                            safe_name = os.path.basename(fname)
                            snap_dir = ctx.VEHICLE_SNAPSHOT_DIR or "/data/vehicle_snapshots"
                            snap_path = os.path.join(snap_dir, date_part, safe_name)
                            if os.path.isfile(snap_path):
                                with open(snap_path, "rb") as f:
                                    await send_photo(f.read(), caption or "🚗 Vehicle", chat_id=chat_id)
                    # Event snapshots: /api/events/{event_id}/snapshot
                    elif url.startswith("/api/events/") and url.endswith("/snapshot"):
                        event_id = url.replace("/api/events/", "").replace("/snapshot", "")
                        safe_id = event_id.replace(":", "-")
                        snap_path = os.path.join(SNAPSHOT_DIR, f"{safe_id}.jpg")
                        if os.path.isfile(snap_path):
                            with open(snap_path, "rb") as f:
                                await send_photo(f.read(), caption or "📸 Event", chat_id=chat_id)
                except Exception:
                    pass

        # Send the text reply
        if reply:
            await _send_long_text(reply, chat_id=chat_id)
        elif not media["snapshot"] and not media["clip"]:
            await send_text("🤔 No response from the AI. Try rephrasing.", chat_id=chat_id)

    except Exception as e:
        logger.error(f"AI ask error (Telegram): {e}")
        await send_text(f"⚠️ AI error: {e}", chat_id=chat_id)

