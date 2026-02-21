"""
routes/notifications.py — Telegram notification endpoints.

PURPOSE:
    Send Telegram notifications with photos for:
    - Person detection events (with camera snapshot + inline feedback buttons)
    - Person identification events (with camera snapshot + inline feedback buttons)
    - Vehicle idle events (with camera snapshot + inline feedback buttons)
    - Face enrollment (with face photo)
    - Manual test notifications (with camera snapshot)

    Self-learning feedback loop (Phase 6.5):
    - Every alert includes inline keyboard: ✅ Real | ❌ False | 👤 Name
    - Background poller receives button taps via Telegram getUpdates API
    - Verdicts are stored in feedback_db.py → auto-generates suppression rules

ENDPOINTS:
    POST /api/notifications/test    — Send a test notification
    GET  /api/notifications/status  — Check if Telegram is configured

SECURITY:
    - Bot token + chat ID kept in .env, passed via docker-compose
    - All API calls use HTTPS to Telegram servers
    - Rate-limited to prevent notification spam
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import redis
import httpx
import numpy as np
import cv2
from fastapi import APIRouter
from fastapi.responses import JSONResponse

import routes as ctx

router = APIRouter(prefix="/api", tags=["notifications"])

# ---------------------------------------------------------------------------
# Telegram config — read from environment
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# Timezone — Toronto (handles EST/EDT automatically)
TZ_LOCAL = ZoneInfo("America/Toronto")

# Rate limiting — max 1 person-detected notification per N seconds
RATE_LIMIT_SECONDS = 60
_last_person_notification = 0.0

# Redis config — for binary frame reads
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

logger = logging.getLogger("dashboard.notifications")


def _now_str() -> str:
    """Get the current time formatted in local timezone."""
    return datetime.now(TZ_LOCAL).strftime("%I:%M:%S %p")


def is_configured() -> bool:
    """Check if Telegram bot token and chat ID are both set."""
    return bool(TELEGRAM_BOT_TOKEN) and bool(TELEGRAM_CHAT_ID)


# ---------------------------------------------------------------------------
# Telegram API helpers
# ---------------------------------------------------------------------------
async def send_text(message: str) -> bool:
    """Send a plain text message to Telegram."""
    if not is_configured():
        logger.warning("Telegram not configured — skipping notification")
        return False
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{TELEGRAM_API}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
                timeout=10,
            )
            if resp.status_code != 200:
                logger.warning(f"Telegram sendMessage failed: {resp.status_code} {resp.text}")
                return False
            return True
    except Exception as e:
        logger.warning(f"Telegram sendMessage error: {e}")
        return False


async def send_photo(photo_bytes: bytes, caption: str = "",
                     reply_markup: dict = None) -> int:
    """
    Send a photo with optional caption to Telegram.
    Returns the Telegram message_id (0 on failure).
    """
    if not is_configured():
        logger.warning("Telegram not configured — skipping photo notification")
        return 0
    try:
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
        if reply_markup:
            data["reply_markup"] = json.dumps(reply_markup)
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{TELEGRAM_API}/sendPhoto",
                data=data,
                files={"photo": ("snapshot.jpg", photo_bytes, "image/jpeg")},
                timeout=15,
            )
            if resp.status_code != 200:
                logger.warning(f"Telegram sendPhoto failed: {resp.status_code} {resp.text}")
                return 0
            result = resp.json().get("result", {})
            return result.get("message_id", 0)
    except Exception as e:
        logger.warning(f"Telegram sendPhoto error: {e}")
        return 0


def _make_feedback_buttons(event_id: str) -> dict:
    """
    Build a Telegram inline keyboard with 3 verdict buttons.
    callback_data format: "verdict:{verdict}:{event_id}"
    """
    return {
        "inline_keyboard": [[
            {"text": "✅ Real Threat", "callback_data": f"v:real:{event_id}"},
            {"text": "❌ False Alarm", "callback_data": f"v:false:{event_id}"},
            {"text": "👤 It's...", "callback_data": f"v:identify:{event_id}"},
        ]]
    }


async def edit_message_buttons(message_id: int, text: str) -> bool:
    """
    Replace the inline keyboard on a sent message with a confirmation text.
    Called after the user taps a verdict button.
    """
    if not is_configured():
        return False
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{TELEGRAM_API}/editMessageReplyMarkup",
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "message_id": message_id,
                    "reply_markup": {"inline_keyboard": [
                        [{"text": text, "callback_data": "noop"}]
                    ]},
                },
                timeout=10,
            )
            return resp.status_code == 200
    except Exception as e:
        logger.warning(f"editMessageReplyMarkup error: {e}")
        return False


async def answer_callback_query(callback_query_id: str,
                                 text: str = "Recorded!") -> bool:
    """Acknowledge a Telegram callback query (removes loading spinner)."""
    if not is_configured():
        return False
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{TELEGRAM_API}/answerCallbackQuery",
                json={"callback_query_id": callback_query_id, "text": text},
                timeout=10,
            )
            return resp.status_code == 200
    except Exception as e:
        logger.warning(f"answerCallbackQuery error: {e}")
        return False


# ---------------------------------------------------------------------------
# Snapshot helper — grab latest frame from Redis (BINARY client)
# Prefers HD frame for higher quality, falls back to sub-stream.
# ---------------------------------------------------------------------------
def get_latest_frame() -> bytes | None:
    """
    Get the latest JPEG frame from Redis.
    Tries the HD frame first (frame_hd:{camera_id}), then falls
    back to the sub-stream (frames:{camera_id}).
    Uses a SEPARATE binary Redis client (decode_responses=False)
    because frame data is raw JPEG bytes.
    """
    try:
        r_bin = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)

        # --- Try HD frame first (clearer image) ---
        if ctx.HD_FRAME_KEY:
            hd_bytes = r_bin.get(ctx.HD_FRAME_KEY.encode())
            if hd_bytes and len(hd_bytes) > 100:
                return hd_bytes

        # --- Fall back to sub-stream ---
        entries = r_bin.xrevrange(ctx.FRAME_STREAM.encode(), count=1)
        if entries:
            _, data = entries[0]
            frame = data.get(b"frame")
            if frame and len(frame) > 100:  # Sanity check — real JPEG is >100 bytes
                return frame
            logger.warning(f"Frame data too small or missing: {len(frame) if frame else 0} bytes")
    except Exception as e:
        logger.warning(f"Failed to get latest frame: {e}")
    return None


def get_sd_frame() -> bytes | None:
    """Get the sub-stream (SD) frame only — used for bbox coordinate reference."""
    try:
        r_bin = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
        entries = r_bin.xrevrange(ctx.FRAME_STREAM.encode(), count=1)
        if entries:
            _, data = entries[0]
            frame = data.get(b"frame")
            if frame and len(frame) > 100:
                return frame
    except Exception:
        pass
    return None


def draw_bbox_on_frame(frame_bytes: bytes, bbox_json: str,
                       label: str = "",
                       color: tuple = (0, 255, 0)) -> bytes:
    """
    Draw a bounding box highlight on a JPEG frame.
    If the frame is HD, scales bbox coords from sub-stream dimensions.
    Returns the modified JPEG bytes.
    """
    try:
        bbox = json.loads(bbox_json) if isinstance(bbox_json, str) else bbox_json
        if not bbox or len(bbox) != 4:
            return frame_bytes

        np_arr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return frame_bytes

        x1, y1, x2, y2 = [float(v) for v in bbox]
        snap_h, snap_w = img.shape[:2]

        # If snapshot is HD (>= 1000px wide), scale bbox from SD coords
        if snap_w >= 1000:
            sd_frame = get_sd_frame()
            if sd_frame:
                sd_arr = np.frombuffer(sd_frame, np.uint8)
                sd_img = cv2.imdecode(sd_arr, cv2.IMREAD_COLOR)
                if sd_img is not None:
                    sd_h, sd_w = sd_img.shape[:2]
                    sx = snap_w / sd_w
                    sy = snap_h / sd_h
                    x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy

        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (ix1, iy1), (ix2, iy2), color, 3)
        if label:
            cv2.putText(img, label, (ix1, iy1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return encoded.tobytes()
    except Exception:
        return frame_bytes


async def send_video(video_bytes: bytes, caption: str = "") -> int:
    """
    Send a video (MP4) with optional caption to Telegram.
    Returns the Telegram message_id (0 on failure).
    """
    if not is_configured():
        logger.warning("Telegram not configured — skipping video notification")
        return 0
    try:
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{TELEGRAM_API}/sendVideo",
                data=data,
                files={"video": ("clip.mp4", video_bytes, "video/mp4")},
                timeout=30,
            )
            if resp.status_code != 200:
                logger.warning(f"Telegram sendVideo failed: {resp.status_code} {resp.text}")
                return 0
            result = resp.json().get("result", {})
            return result.get("message_id", 0)
    except Exception as e:
        logger.warning(f"Telegram sendVideo error: {e}")
        return 0


def build_clip(duration: float = 5.0, fps: int = 10) -> bytes | None:
    """
    Capture frames from the Redis stream and encode as MP4 clip.
    Collects frames for `duration` seconds at `fps` rate.
    Returns MP4 bytes or None on failure.
    """
    import cv2
    import numpy as np
    import tempfile
    import time as _time

    try:
        r_bin = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
        frames = []
        target_count = int(duration * fps)
        interval = 1.0 / fps
        start = _time.monotonic()

        for _ in range(target_count):
            entries = r_bin.xrevrange(ctx.FRAME_STREAM.encode(), count=1)
            if entries:
                _, data = entries[0]
                frame_bytes = data.get(b"frame")
                if frame_bytes and len(frame_bytes) > 100:
                    nparr = np.frombuffer(frame_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        frames.append(img)

            # Wait for next frame
            elapsed = _time.monotonic() - start
            expected = len(frames) * interval
            if expected > elapsed:
                _time.sleep(expected - elapsed)

            # Safety timeout
            if _time.monotonic() - start > duration + 2:
                break

        if len(frames) < 5:
            logger.warning(f"build_clip: only captured {len(frames)} frames, need at least 5")
            return None

        # Encode to MP4
        h, w = frames[0].shape[:2]
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()

        import os
        with open(tmp_path, "rb") as f:
            video_bytes = f.read()
        os.unlink(tmp_path)

        if len(video_bytes) < 1000:
            logger.warning(f"build_clip: video too small ({len(video_bytes)} bytes)")
            return None

        return video_bytes

    except Exception as e:
        logger.warning(f"build_clip error: {e}")
        return None


# ---------------------------------------------------------------------------
# Notify on person detection (rate-limited)
# ---------------------------------------------------------------------------
async def notify_person_detected(event_data: dict,
                                  event_id: str = "",
                                  feedback_db=None) -> int:
    """
    Send a Telegram notification when a person is detected.
    Rate-limited to max 1 per RATE_LIMIT_SECONDS.
    Returns the Telegram message ID (0 if not sent).
    """
    global _last_person_notification

    if not is_configured():
        return 0

    now = time.time()
    if now - _last_person_notification < RATE_LIMIT_SECONDS:
        return 0  # Rate limited

    _last_person_notification = now

    # Check suppression before sending
    identity = event_data.get("identity_name", "")
    zone = event_data.get("zone", "")
    action = event_data.get("action", "")
    time_period = event_data.get("time_period", "")
    confidence = float(event_data.get("confidence", "0") or "0")

    if feedback_db and feedback_db.should_suppress(
        identity=identity, zone=zone, time_period=time_period, action=action
    ):
        logger.info(f"Notification suppressed for event {event_id}")
        return 0

    person_id = event_data.get("person_id", "unknown")
    name = identity if identity else person_id
    parts = [f"\U0001f6a8 <b>Person Detected</b>"]
    parts.append(f"\u2022 Who: {name}")
    if zone:
        parts.append(f"\u2022 Zone: {zone}")
    if action:
        parts.append(f"\u2022 Action: {action}")
    parts.append(f"\u2022 Time: {_now_str()}")

    caption = "\n".join(parts)
    buttons = _make_feedback_buttons(event_id) if event_id else None

    frame = get_latest_frame()
    if frame:
        # Draw bbox highlight on the snapshot if available
        bbox_json = event_data.get("bbox", "")
        if bbox_json:
            frame = draw_bbox_on_frame(frame, bbox_json,
                                       label=name, color=(0, 255, 0))
        msg_id = await send_photo(frame, caption, reply_markup=buttons)
    else:
        await send_text(caption)
        msg_id = 0

    # Store pending feedback record
    if feedback_db and msg_id and event_id:
        feedback_db.store_pending_event(
            event_id=event_id,
            event_type="person_appeared",
            telegram_message_id=msg_id,
            zone=zone, time_period=time_period,
            action=action, confidence=confidence,
            identity=identity,
        )

    return msg_id


# ---------------------------------------------------------------------------
# Notify on person identification (NOT rate-limited — always important)
# ---------------------------------------------------------------------------
async def notify_person_identified(event_data: dict,
                                    event_id: str = "",
                                    feedback_db=None) -> int:
    """
    Send a Telegram notification when a person is identified by face recognition.
    This is NOT rate-limited because identification is a significant event.
    Returns the Telegram message ID (0 if not sent).
    """
    if not is_configured():
        return 0

    person_id = event_data.get("person_id", "unknown")
    identity_name = event_data.get("identity_name", "")
    zone = event_data.get("zone", "")
    action = event_data.get("action", "")
    time_period = event_data.get("time_period", "")
    confidence = float(event_data.get("confidence", "0") or "0")

    if not identity_name:
        return 0  # Skip if no name was identified

    # Check suppression for identified persons
    if feedback_db and feedback_db.should_suppress(
        identity=identity_name, zone=zone, time_period=time_period
    ):
        logger.info(f"Notification suppressed for identified '{identity_name}'")
        return 0

    parts = [f"\U0001f464 <b>Person Identified</b>"]
    parts.append(f"\u2022 Name: {identity_name}")
    parts.append(f"\u2022 Tracker ID: {person_id}")
    if zone:
        parts.append(f"\u2022 Zone: {zone}")
    if action:
        parts.append(f"\u2022 Action: {action}")
    parts.append(f"\u2022 Time: {_now_str()}")

    caption = "\n".join(parts)
    buttons = _make_feedback_buttons(event_id) if event_id else None

    frame = get_latest_frame()
    if frame:
        # Draw bbox highlight on the snapshot if available
        bbox_json = event_data.get("bbox", "")
        if bbox_json:
            frame = draw_bbox_on_frame(frame, bbox_json,
                                       label=identity_name,
                                       color=(255, 255, 0))
        msg_id = await send_photo(frame, caption, reply_markup=buttons)
    else:
        await send_text(caption)
        msg_id = 0

    # Store pending feedback
    if feedback_db and msg_id and event_id:
        feedback_db.store_pending_event(
            event_id=event_id,
            event_type="person_identified",
            telegram_message_id=msg_id,
            zone=zone, time_period=time_period,
            action=action, confidence=confidence,
            identity=identity_name,
        )

    return msg_id


# ---------------------------------------------------------------------------
# Notify on vehicle idle (rate-limited separately from person notifications)
# ---------------------------------------------------------------------------
_last_vehicle_idle_notification = 0.0
VEHICLE_IDLE_RATE_LIMIT = 60  # Max 1 vehicle idle notification per 60s


async def notify_vehicle_idle(event_data: dict,
                               event_id: str = "",
                               feedback_db=None) -> int:
    """
    Send a Telegram notification when a vehicle has been idling.
    Sends a photo snapshot immediately, then follows up with a 5-second
    video clip for additional context.
    Rate-limited to max 1 per VEHICLE_IDLE_RATE_LIMIT seconds.
    Returns the Telegram message ID (0 if not sent).
    """
    global _last_vehicle_idle_notification

    if not is_configured():
        return 0

    now = time.time()
    if now - _last_vehicle_idle_notification < VEHICLE_IDLE_RATE_LIMIT:
        return 0  # Rate limited

    _last_vehicle_idle_notification = now

    vehicle_class = event_data.get("vehicle_class", "vehicle")
    zone = event_data.get("zone", "")
    time_period = event_data.get("time_period", "")
    duration = event_data.get("duration", "0")
    confidence = event_data.get("vehicle_confidence", "")

    # Check suppression before sending
    if feedback_db and feedback_db.should_suppress(
        zone=zone, time_period=time_period
    ):
        logger.info(f"Vehicle idle notification suppressed for event {event_id}")
        return 0

    parts = [f"\U0001f697 <b>Vehicle Idling</b>"]
    parts.append(f"\u2022 Type: {vehicle_class}")
    if zone:
        parts.append(f"\u2022 Zone: {zone}")
    parts.append(f"\u2022 Duration: {duration}s")
    if confidence:
        parts.append(f"\u2022 Confidence: {confidence}")
    parts.append(f"\u2022 Time: {_now_str()}")

    caption = "\n".join(parts)
    buttons = _make_feedback_buttons(event_id) if event_id else None

    frame = get_latest_frame()
    if frame:
        msg_id = await send_photo(frame, caption, reply_markup=buttons)
    else:
        await send_text(caption)
        msg_id = 0

    # Store pending feedback
    if feedback_db and msg_id and event_id:
        feedback_db.store_pending_event(
            event_id=event_id,
            event_type="vehicle_idle",
            telegram_message_id=msg_id,
            zone=zone, time_period=time_period,
            confidence=float(confidence) if confidence else 0.0,
        )

    # --- Follow up with a 5-second video clip ---
    try:
        loop = asyncio.get_running_loop()
        clip_bytes = await loop.run_in_executor(
            None, lambda: build_clip(duration=5.0, fps=10)
        )
        if clip_bytes:
            clip_caption = f"\U0001f3ac <b>Vehicle Idle Clip</b> — {vehicle_class}"
            await send_video(clip_bytes, clip_caption)
            logger.info(f"Vehicle idle clip sent ({len(clip_bytes) / 1024:.0f} KB)")
        else:
            logger.debug("build_clip returned None — skipping video")
    except Exception as e:
        logger.warning(f"Vehicle idle clip failed: {e}")

    return msg_id


# ---------------------------------------------------------------------------
# Telegram Callback Polling — receives button taps from users
# ---------------------------------------------------------------------------
_telegram_update_offset = 0


async def poll_telegram_callbacks(feedback_db):
    """
    Background task: poll Telegram for callback_query updates.
    Routes verdicts to the feedback database.

    This is the "receive" side of the inline keyboard conversation.
    When user taps ✅/❌/👤, Telegram sends a callback_query which
    we pick up here via long-polling getUpdates.
    """
    global _telegram_update_offset

    if not is_configured():
        logger.info("Telegram not configured — callback poller disabled")
        return

    logger.info("Telegram callback poller started")

    while True:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{TELEGRAM_API}/getUpdates",
                    params={
                        "offset": _telegram_update_offset,
                        "timeout": 30,
                        "allowed_updates": json.dumps(["callback_query"]),
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
                cb = update.get("callback_query")
                if not cb:
                    continue

                callback_data = cb.get("data", "")
                callback_id = cb.get("id", "")
                message_id = cb.get("message", {}).get("message_id", 0)

                await _handle_callback(
                    callback_data, callback_id, message_id, feedback_db
                )

        except httpx.ReadTimeout:
            # Normal — long poll timed out with no updates
            pass
        except Exception as e:
            logger.warning(f"Callback poller error: {e}")
            await asyncio.sleep(5)


async def _handle_callback(callback_data: str, callback_id: str,
                            message_id: int, feedback_db):
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
        # For "identify" — we record it with a placeholder.
        # The user can name the person from the dashboard review queue.
        # For now, acknowledge and mark as identified-pending.
        feedback_db.resolve_pending(event_id, "identified", identity_label="")
        await answer_callback_query(callback_id, "Marked for identification — name from dashboard")
        await edit_message_buttons(message_id, "👤 Awaiting name (dashboard)")
        logger.info(f"Event {event_id}: marked for identification via Telegram")
    else:
        feedback_db.resolve_pending(event_id, verdict)
        label = "✅ Real Threat" if verdict == "real_threat" else "❌ False Alarm"
        await answer_callback_query(callback_id, f"Recorded: {label}")
        await edit_message_buttons(message_id, f"{label} — Recorded")
        logger.info(f"Event {event_id}: verdict={verdict} via Telegram")


# ---------------------------------------------------------------------------
# Notify on face enrollment
# ---------------------------------------------------------------------------
async def notify_face_enrolled(name: str, photo_bytes: bytes | None = None):
    """Send a Telegram notification when a new face is enrolled."""
    if not is_configured():
        return

    caption = f"\U0001f4f7 <b>New Face Enrolled</b>\n\u2022 Name: {name}\n\u2022 Time: {_now_str()}"

    if photo_bytes:
        await send_photo(photo_bytes, caption)
    else:
        # Fall back to camera snapshot
        frame = get_latest_frame()
        if frame:
            await send_photo(frame, caption)
        else:
            await send_text(caption)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------
@router.get("/notifications/status")
async def notification_status():
    """Check if Telegram notifications are configured."""
    return {
        "configured": is_configured(),
        "has_token": bool(TELEGRAM_BOT_TOKEN),
        "has_chat_id": bool(TELEGRAM_CHAT_ID),
        "rate_limit_seconds": RATE_LIMIT_SECONDS,
        "feedback_enabled": True,
    }


@router.post("/notifications/test")
async def test_notification():
    """Send a test notification to Telegram with a camera snapshot."""
    if not is_configured():
        return JSONResponse(
            status_code=400,
            content={"error": "Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"},
        )

    caption = (
        f"\U0001f9ea <b>Test Notification</b>\n"
        f"\u2022 Source: Vision Labs Dashboard\n"
        f"\u2022 Time: {_now_str()}\n"
        f"\u2022 Status: \u2705 Notifications working!"
    )

    frame = get_latest_frame()
    if frame:
        ok = await send_photo(frame, caption)
    else:
        ok = await send_text(caption + "\n\n(No camera frame available)")

    if ok:
        return {"status": "sent", "message": "Test notification sent to Telegram"}
    else:
        return JSONResponse(status_code=500, content={"error": "Failed to send. Check bot token and chat ID"})
