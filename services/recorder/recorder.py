"""
services/recorder/recorder.py — Continuous DVR recording to NAS.

PURPOSE:
    Records the camera's RTSP sub-stream directly to 1-hour MP4 segments
    on the QNAP NAS. Uses ffmpeg for efficient, zero-copy remuxing
    (no decode/re-encode — just copies the H.264 stream into MP4 container).

    Also handles retention: deletes segments older than RETENTION_DAYS.

STORAGE LAYOUT:
    /recordings/{camera_id}/YYYY-MM-DD/HH-MM.mp4

CONFIG (via environment variables):
    CAMERA_ID           — Camera name (default: front_door)
    RTSP_URL            — RTSP sub-stream URL
    RECORDING_DIR       — Base output directory (default: /recordings)
    SEGMENT_DURATION    — Seconds per segment (default: 3600 = 1 hour)
    RETENTION_DAYS      — Days to keep recordings (default: 28)
    CLEANUP_INTERVAL    — Hours between cleanup runs (default: 6)
"""

import os
import sys
import time
import signal
import logging
import subprocess
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAMERA_ID = os.getenv("CAMERA_ID", "front_door")
RTSP_URL = os.getenv("RTSP_URL", "")
RECORDING_DIR = os.getenv("RECORDING_DIR", "/recordings")
SEGMENT_DURATION = int(os.getenv("SEGMENT_DURATION", "3600"))  # 1 hour
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "28"))
CLEANUP_INTERVAL_HOURS = int(os.getenv("CLEANUP_INTERVAL", "6"))
TZ_NAME = os.getenv("LOCATION_TIMEZONE", "America/Toronto")

TZ_LOCAL = ZoneInfo(TZ_NAME)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("recorder")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown = False
_ffmpeg_proc = None


def _handle_signal(signum, frame):
    global _shutdown
    logger.info("Shutdown signal received — stopping recording...")
    _shutdown = True
    if _ffmpeg_proc and _ffmpeg_proc.poll() is None:
        _ffmpeg_proc.terminate()


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ---------------------------------------------------------------------------
# Retention cleanup
# ---------------------------------------------------------------------------
def cleanup_old_recordings():
    """Delete recording segments older than RETENTION_DAYS."""
    camera_dir = os.path.join(RECORDING_DIR, CAMERA_ID)
    if not os.path.isdir(camera_dir):
        return

    cutoff = datetime.now(TZ_LOCAL) - timedelta(days=RETENTION_DAYS)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    removed_count = 0

    for day_folder in sorted(os.listdir(camera_dir)):
        day_path = os.path.join(camera_dir, day_folder)
        if not os.path.isdir(day_path):
            continue

        # Day folder name is YYYY-MM-DD
        if day_folder < cutoff_str:
            try:
                shutil.rmtree(day_path)
                removed_count += 1
                logger.info(f"Deleted old recordings: {day_path}")
            except Exception as e:
                logger.warning(f"Failed to delete {day_path}: {e}")

    if removed_count:
        logger.info(f"Cleanup complete — removed {removed_count} day folder(s)")
    else:
        logger.info("Cleanup complete — no old recordings to remove")


# ---------------------------------------------------------------------------
# ffmpeg recording
# ---------------------------------------------------------------------------
def get_output_path() -> str:
    """Generate the output path for the current segment."""
    now = datetime.now(TZ_LOCAL)
    day_str = now.strftime("%Y-%m-%d")
    hour_str = now.strftime("%H-%M")

    day_dir = os.path.join(RECORDING_DIR, CAMERA_ID, day_str)
    os.makedirs(day_dir, exist_ok=True)

    return os.path.join(day_dir, f"{hour_str}.mp4")


def record_segment() -> bool:
    """
    Record one segment using ffmpeg.

    Uses segment muxer to automatically split at SEGMENT_DURATION boundaries.
    The `-c copy` flag means no transcoding — just remuxes the H.264 stream
    from RTSP into an MP4 container. Very low CPU usage.

    Returns True if recording completed normally, False on error.
    """
    global _ffmpeg_proc

    output_path = get_output_path()
    logger.info(f"Recording segment: {output_path}")

    # Log the RTSP URL without password
    safe_url = RTSP_URL.split("@")[-1] if "@" in RTSP_URL else RTSP_URL
    logger.info(f"RTSP source: {safe_url}")

    cmd = [
        "ffmpeg",
        "-y",                           # Overwrite output (in case of restart)
        "-rtsp_transport", "tcp",       # TCP for reliability
        "-i", RTSP_URL,                 # Input RTSP stream
        "-c", "copy",                   # No transcode — copy stream
        "-movflags", "+faststart",      # MP4 progressive download
        "-t", str(SEGMENT_DURATION),    # Duration of this segment
        "-f", "mp4",                    # Force MP4 container
        output_path,
    ]

    try:
        _ffmpeg_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for completion or shutdown
        while _ffmpeg_proc.poll() is None:
            if _shutdown:
                _ffmpeg_proc.terminate()
                _ffmpeg_proc.wait(timeout=10)
                return False
            time.sleep(1)

        rc = _ffmpeg_proc.returncode
        if rc == 0:
            # Check file size
            try:
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"Segment complete: {output_path} ({size_mb:.1f} MB)")
            except OSError:
                pass
            return True
        else:
            stderr = _ffmpeg_proc.stderr.read().decode(errors="replace")
            logger.warning(f"ffmpeg exited with code {rc}: {stderr[-500:]}")
            return False

    except FileNotFoundError:
        logger.error("ffmpeg not found — install ffmpeg in the container")
        return False
    except Exception as e:
        logger.error(f"Recording error: {e}")
        return False
    finally:
        _ffmpeg_proc = None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run():
    if not RTSP_URL:
        logger.error("RTSP_URL not set — check your .env or docker-compose.yml")
        sys.exit(1)

    logger.info(f"DVR Recorder starting for camera '{CAMERA_ID}'")
    logger.info(f"Segment duration: {SEGMENT_DURATION}s ({SEGMENT_DURATION//3600}h)")
    logger.info(f"Retention: {RETENTION_DAYS} days")
    logger.info(f"Recording directory: {RECORDING_DIR}/{CAMERA_ID}/")

    reconnect_delay = 5
    last_cleanup = 0

    while not _shutdown:
        # Run cleanup periodically
        now = time.time()
        if now - last_cleanup > CLEANUP_INTERVAL_HOURS * 3600:
            cleanup_old_recordings()
            last_cleanup = now

        # Record a segment
        ok = record_segment()

        if not ok and not _shutdown:
            logger.warning(f"Recording failed — retrying in {reconnect_delay}s...")
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 60)
        else:
            reconnect_delay = 5  # Reset on success

    logger.info("DVR Recorder stopped")


if __name__ == "__main__":
    run()
