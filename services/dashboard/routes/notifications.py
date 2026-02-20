"""
routes/notifications.py — Telegram notification endpoints.

PURPOSE:
    Send Telegram notifications with photos for:
    - Person detection events (with camera snapshot)
    - Person identification events (with camera snapshot)
    - Face enrollment (with face photo)
    - Manual test notifications (with camera snapshot)

ENDPOINTS:
    POST /api/notifications/test    — Send a test notification
    GET  /api/notifications/status  — Check if Telegram is configured

SECURITY:
    - Bot token + chat ID kept in .env, passed via docker-compose
    - All API calls use HTTPS to Telegram servers
    - Rate-limited to prevent notification spam
"""

import os
import time
import logging
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import redis
import httpx
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


async def send_photo(photo_bytes: bytes, caption: str = "") -> bool:
    """Send a photo with optional caption to Telegram."""
    if not is_configured():
        logger.warning("Telegram not configured — skipping photo notification")
        return False
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{TELEGRAM_API}/sendPhoto",
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"},
                files={"photo": ("snapshot.jpg", photo_bytes, "image/jpeg")},
                timeout=15,
            )
            if resp.status_code != 200:
                logger.warning(f"Telegram sendPhoto failed: {resp.status_code} {resp.text}")
                return False
            return True
    except Exception as e:
        logger.warning(f"Telegram sendPhoto error: {e}")
        return False


# ---------------------------------------------------------------------------
# Snapshot helper — grab latest frame from Redis (BINARY client)
# ---------------------------------------------------------------------------
def get_latest_frame() -> bytes | None:
    """
    Get the latest JPEG frame from the Redis frame stream.
    Uses a SEPARATE binary Redis client (decode_responses=False)
    because frame data is raw JPEG bytes.
    """
    try:
        r_bin = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
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


# ---------------------------------------------------------------------------
# Notify on person detection (rate-limited)
# ---------------------------------------------------------------------------
async def notify_person_detected(event_data: dict):
    """
    Send a Telegram notification when a person is detected.
    Rate-limited to max 1 per RATE_LIMIT_SECONDS.
    """
    global _last_person_notification

    if not is_configured():
        return

    now = time.time()
    if now - _last_person_notification < RATE_LIMIT_SECONDS:
        return  # Rate limited

    _last_person_notification = now

    person_id = event_data.get("person_id", "unknown")
    identity = event_data.get("identity_name", "")
    zone = event_data.get("zone", "")
    action = event_data.get("action", "")

    name = identity if identity else person_id
    parts = [f"\U0001f6a8 <b>Person Detected</b>"]
    parts.append(f"\u2022 Who: {name}")
    if zone:
        parts.append(f"\u2022 Zone: {zone}")
    if action:
        parts.append(f"\u2022 Action: {action}")
    parts.append(f"\u2022 Time: {_now_str()}")

    caption = "\n".join(parts)

    frame = get_latest_frame()
    if frame:
        await send_photo(frame, caption)
    else:
        await send_text(caption)


# ---------------------------------------------------------------------------
# Notify on person identification (NOT rate-limited — always important)
# ---------------------------------------------------------------------------
async def notify_person_identified(event_data: dict):
    """
    Send a Telegram notification when a person is identified by face recognition.
    This is NOT rate-limited because identification is a significant event.
    """
    if not is_configured():
        return

    person_id = event_data.get("person_id", "unknown")
    identity_name = event_data.get("identity_name", "")
    zone = event_data.get("zone", "")
    action = event_data.get("action", "")

    if not identity_name:
        return  # Skip if no name was identified

    parts = [f"\U0001f464 <b>Person Identified</b>"]
    parts.append(f"\u2022 Name: {identity_name}")
    parts.append(f"\u2022 Tracker ID: {person_id}")
    if zone:
        parts.append(f"\u2022 Zone: {zone}")
    if action:
        parts.append(f"\u2022 Action: {action}")
    parts.append(f"\u2022 Time: {_now_str()}")

    caption = "\n".join(parts)

    frame = get_latest_frame()
    if frame:
        await send_photo(frame, caption)
    else:
        await send_text(caption)


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
