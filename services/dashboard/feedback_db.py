"""
feedback_db.py — SQLite database for self-learning feedback loop.

PURPOSE:
    Stores user verdicts on detection events (real threat, false alarm, identified)
    and builds suppression rules from accumulated feedback patterns.

USAGE:
    db = FeedbackDB("/data/feedback.db")
    db.record_feedback(event_id, verdict="false_alarm", ...)
    if db.should_suppress(zone="Driveway", identity="Mail Carrier", time_period="morning"):
        # Don't send Telegram notification

TABLES:
    feedback       — one row per user verdict on a detection event
    suppression    — learned rules (auto-generated from feedback patterns)

DESIGN:
    Follows the same pattern as face_db.py — SQLite with WAL mode, simple
    class-based API, no ORM. Suppression rules are deterministic lookups,
    not ML models.
"""

import os
import time
import json
import sqlite3
import logging
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger("dashboard.feedback")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class FeedbackRecord:
    """A single user verdict on a detection event."""
    event_id: str                    # Redis event stream ID (e.g., "1708472312-0")
    verdict: str                     # "real_threat", "false_alarm", "identified"
    event_type: str                  # "person_appeared", "vehicle_detected", etc.
    identity_label: str = ""         # Name if user identified someone (e.g., "Mail Carrier")
    zone: str = ""                   # Zone name where event occurred
    time_period: str = ""            # "morning", "daytime", "twilight", "night", "late_night"
    action: str = ""                 # Pose action (standing, crouching, etc.)
    confidence: float = 0.0          # YOLO detection confidence
    camera_id: str = "front_door"    # Camera identifier
    snapshot_path: str = ""          # Path to saved snapshot JPEG
    timestamp: float = 0.0          # When the user responded
    telegram_message_id: int = 0     # Telegram msg ID (for editing button text after tap)


@dataclass
class SuppressionRule:
    """A learned suppression rule derived from feedback patterns."""
    rule_id: int = 0
    rule_type: str = ""              # "identity", "zone_time", "confidence_floor"
    identity: str = ""               # For identity-based rules (e.g., "Mail Carrier")
    zone: str = ""                   # For zone-based rules
    time_period: str = ""            # For time-based rules
    action: str = ""                 # For action-based rules
    min_false_alarms: int = 0        # How many false alarms triggered this rule
    created_at: float = 0.0
    active: bool = True


class FeedbackDB:
    """SQLite database for feedback records and suppression rules."""

    def __init__(self, db_path: str = "/data/feedback.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
        logger.info(f"FeedbackDB initialized at {db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    event_type TEXT NOT NULL DEFAULT '',
                    identity_label TEXT DEFAULT '',
                    zone TEXT DEFAULT '',
                    time_period TEXT DEFAULT '',
                    action TEXT DEFAULT '',
                    confidence REAL DEFAULT 0.0,
                    camera_id TEXT DEFAULT 'front_door',
                    snapshot_path TEXT DEFAULT '',
                    timestamp REAL NOT NULL,
                    telegram_message_id INTEGER DEFAULT 0,
                    created_at REAL NOT NULL DEFAULT (strftime('%s', 'now')),
                    UNIQUE(event_id)
                );

                CREATE TABLE IF NOT EXISTS suppression (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_type TEXT NOT NULL,
                    identity TEXT DEFAULT '',
                    zone TEXT DEFAULT '',
                    time_period TEXT DEFAULT '',
                    action TEXT DEFAULT '',
                    min_false_alarms INTEGER DEFAULT 0,
                    created_at REAL NOT NULL DEFAULT (strftime('%s', 'now')),
                    active INTEGER DEFAULT 1
                );

                CREATE INDEX IF NOT EXISTS idx_feedback_verdict
                    ON feedback(verdict);
                CREATE INDEX IF NOT EXISTS idx_feedback_identity
                    ON feedback(identity_label);
                CREATE INDEX IF NOT EXISTS idx_feedback_zone_time
                    ON feedback(zone, time_period);
                CREATE INDEX IF NOT EXISTS idx_suppression_active
                    ON suppression(active);
            """)
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Feedback CRUD
    # ------------------------------------------------------------------
    def record_feedback(self, record: FeedbackRecord) -> int:
        """Store a user verdict. Returns the row ID."""
        if not record.timestamp:
            record.timestamp = time.time()

        conn = self._get_conn()
        try:
            cur = conn.execute("""
                INSERT OR REPLACE INTO feedback
                    (event_id, verdict, event_type, identity_label, zone,
                     time_period, action, confidence, camera_id,
                     snapshot_path, timestamp, telegram_message_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.event_id, record.verdict, record.event_type,
                record.identity_label, record.zone, record.time_period,
                record.action, record.confidence, record.camera_id,
                record.snapshot_path, record.timestamp,
                record.telegram_message_id,
            ))
            conn.commit()
            row_id = cur.lastrowid
            logger.info(
                f"Feedback recorded: event={record.event_id} "
                f"verdict={record.verdict} identity={record.identity_label}"
            )

            # Check if we should auto-generate suppression rules
            self._check_and_create_rules(conn, record)
            return row_id
        finally:
            conn.close()

    def get_feedback(self, event_id: str) -> Optional[dict]:
        """Get feedback for a specific event."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM feedback WHERE event_id = ?", (event_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_recent_feedback(self, limit: int = 50) -> list[dict]:
        """Get recent feedback records, newest first."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM feedback ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_stats(self) -> dict:
        """Get feedback statistics for the dashboard."""
        conn = self._get_conn()
        try:
            total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            by_verdict = {}
            for row in conn.execute(
                "SELECT verdict, COUNT(*) as cnt FROM feedback GROUP BY verdict"
            ).fetchall():
                by_verdict[row["verdict"]] = row["cnt"]

            suppressed = conn.execute(
                "SELECT COUNT(*) FROM suppression WHERE active = 1"
            ).fetchone()[0]

            # Accuracy: of events user responded to, how many were real threats?
            real = by_verdict.get("real_threat", 0)
            false = by_verdict.get("false_alarm", 0)
            accuracy = real / (real + false) if (real + false) > 0 else 0.0

            return {
                "total_feedback": total,
                "by_verdict": by_verdict,
                "active_suppression_rules": suppressed,
                "alert_accuracy": round(accuracy, 3),
            }
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Suppression logic
    # ------------------------------------------------------------------
    def should_suppress(
        self,
        identity: str = "",
        zone: str = "",
        time_period: str = "",
        action: str = "",
    ) -> bool:
        """
        Check if this event matches any active suppression rule.
        Returns True if the notification should be suppressed.
        """
        conn = self._get_conn()
        try:
            rules = conn.execute(
                "SELECT * FROM suppression WHERE active = 1"
            ).fetchall()

            for rule in rules:
                rule_type = rule["rule_type"]

                if rule_type == "identity" and identity:
                    # Suppress if known person is in the rule
                    if rule["identity"].lower() == identity.lower():
                        logger.debug(
                            f"Suppressed: identity rule for '{identity}'"
                        )
                        return True

                elif rule_type == "zone_time" and zone and time_period:
                    # Suppress if zone + time combo matches
                    if (rule["zone"].lower() == zone.lower() and
                            rule["time_period"].lower() == time_period.lower()):
                        logger.debug(
                            f"Suppressed: zone_time rule for "
                            f"'{zone}' at '{time_period}'"
                        )
                        return True

            return False
        finally:
            conn.close()

    def get_suppression_rules(self) -> list[dict]:
        """Get all suppression rules."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM suppression ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def toggle_rule(self, rule_id: int, active: bool) -> bool:
        """Enable or disable a suppression rule."""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE suppression SET active = ? WHERE id = ?",
                (1 if active else 0, rule_id)
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def delete_rule(self, rule_id: int) -> bool:
        """Delete a suppression rule."""
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM suppression WHERE id = ?", (rule_id,))
            conn.commit()
            return True
        finally:
            conn.close()

    def retrain_rules(self) -> dict:
        """
        Re-scan ALL feedback records and regenerate suppression rules.

        Deletes all existing auto-generated rules and rebuilds them from
        scratch based on current thresholds. Returns a summary of what
        was created.
        """
        conn = self._get_conn()
        try:
            # Count existing rules before deletion
            old_count = conn.execute(
                "SELECT COUNT(*) FROM suppression"
            ).fetchone()[0]

            # Delete all existing auto-generated rules
            conn.execute("DELETE FROM suppression")
            conn.commit()
            logger.info(f"Retrain: cleared {old_count} existing rules")

            # --- Re-scan identity-based patterns ---
            identity_rows = conn.execute("""
                SELECT identity_label, COUNT(*) as cnt
                FROM feedback
                WHERE verdict = 'false_alarm'
                  AND identity_label != ''
                GROUP BY identity_label
                HAVING cnt >= ?
            """, (self.IDENTITY_THRESHOLD,)).fetchall()

            new_identity = 0
            for row in identity_rows:
                conn.execute("""
                    INSERT INTO suppression
                        (rule_type, identity, min_false_alarms, created_at)
                    VALUES ('identity', ?, ?, ?)
                """, (row["identity_label"], row["cnt"], time.time()))
                new_identity += 1

            # --- Re-scan zone+time patterns ---
            zone_rows = conn.execute("""
                SELECT zone, time_period, COUNT(*) as cnt
                FROM feedback
                WHERE verdict = 'false_alarm'
                  AND zone != '' AND time_period != ''
                GROUP BY zone, time_period
                HAVING cnt >= ?
            """, (self.ZONE_TIME_THRESHOLD,)).fetchall()

            new_zone_time = 0
            for row in zone_rows:
                conn.execute("""
                    INSERT INTO suppression
                        (rule_type, zone, time_period,
                         min_false_alarms, created_at)
                    VALUES ('zone_time', ?, ?, ?, ?)
                """, (row["zone"], row["time_period"],
                      row["cnt"], time.time()))
                new_zone_time += 1

            conn.commit()

            # Get overall stats
            total_feedback = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE verdict != 'pending'"
            ).fetchone()[0]
            false_alarms = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE verdict = 'false_alarm'"
            ).fetchone()[0]
            real_threats = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE verdict = 'real_threat'"
            ).fetchone()[0]

            summary = {
                "rules_cleared": old_count,
                "identity_rules_created": new_identity,
                "zone_time_rules_created": new_zone_time,
                "total_new_rules": new_identity + new_zone_time,
                "total_feedback_records": total_feedback,
                "false_alarms": false_alarms,
                "real_threats": real_threats,
                "identity_threshold": self.IDENTITY_THRESHOLD,
                "zone_time_threshold": self.ZONE_TIME_THRESHOLD,
            }

            logger.info(
                f"Retrain complete: {new_identity} identity rules, "
                f"{new_zone_time} zone+time rules "
                f"(from {total_feedback} feedback records)"
            )
            return summary
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Auto-rule generation
    # ------------------------------------------------------------------
    IDENTITY_THRESHOLD = 3    # Create identity rule after N false alarms
    ZONE_TIME_THRESHOLD = 5   # Create zone+time rule after N false alarms

    def _check_and_create_rules(self, conn: sqlite3.Connection,
                                 record: FeedbackRecord):
        """
        After recording feedback, check if patterns warrant new rules.
        Only creates rules from false_alarm verdicts.
        """
        if record.verdict != "false_alarm":
            return

        # --- Identity-based rule ---
        if record.identity_label:
            count = conn.execute(
                """SELECT COUNT(*) FROM feedback
                   WHERE identity_label = ? AND verdict = 'false_alarm'""",
                (record.identity_label,)
            ).fetchone()[0]

            if count >= self.IDENTITY_THRESHOLD:
                # Check if rule already exists
                existing = conn.execute(
                    """SELECT id FROM suppression
                       WHERE rule_type = 'identity' AND identity = ?""",
                    (record.identity_label,)
                ).fetchone()
                if not existing:
                    conn.execute(
                        """INSERT INTO suppression
                           (rule_type, identity, min_false_alarms, created_at)
                           VALUES ('identity', ?, ?, ?)""",
                        (record.identity_label, count, time.time())
                    )
                    conn.commit()
                    logger.info(
                        f"Auto-created suppression rule: "
                        f"identity='{record.identity_label}' "
                        f"(after {count} false alarms)"
                    )

        # --- Zone + time period rule ---
        if record.zone and record.time_period:
            count = conn.execute(
                """SELECT COUNT(*) FROM feedback
                   WHERE zone = ? AND time_period = ?
                   AND verdict = 'false_alarm'""",
                (record.zone, record.time_period)
            ).fetchone()[0]

            if count >= self.ZONE_TIME_THRESHOLD:
                existing = conn.execute(
                    """SELECT id FROM suppression
                       WHERE rule_type = 'zone_time'
                       AND zone = ? AND time_period = ?""",
                    (record.zone, record.time_period)
                ).fetchone()
                if not existing:
                    conn.execute(
                        """INSERT INTO suppression
                           (rule_type, zone, time_period,
                            min_false_alarms, created_at)
                           VALUES ('zone_time', ?, ?, ?, ?)""",
                        (record.zone, record.time_period, count, time.time())
                    )
                    conn.commit()
                    logger.info(
                        f"Auto-created suppression rule: "
                        f"zone='{record.zone}' time='{record.time_period}' "
                        f"(after {count} false alarms)"
                    )

    # ------------------------------------------------------------------
    # Pending events (awaiting user feedback via Telegram)
    # ------------------------------------------------------------------
    def store_pending_event(self, event_id: str, event_type: str,
                            telegram_message_id: int, zone: str = "",
                            time_period: str = "", action: str = "",
                            confidence: float = 0.0, camera_id: str = "",
                            identity: str = "",
                            snapshot_path: str = "") -> int:
        """
        Store an event that was sent to Telegram and is awaiting feedback.
        Verdict is set to 'pending' until the user taps a button.
        """
        record = FeedbackRecord(
            event_id=event_id,
            verdict="pending",
            event_type=event_type,
            identity_label=identity,
            zone=zone,
            time_period=time_period,
            action=action,
            confidence=confidence,
            camera_id=camera_id,
            snapshot_path=snapshot_path,
            timestamp=time.time(),
            telegram_message_id=telegram_message_id,
        )
        return self.record_feedback(record)

    def resolve_pending(self, event_id: str, verdict: str,
                        identity_label: str = "") -> bool:
        """
        Update a pending event with the user's verdict.
        Called when user taps a Telegram inline button or classifies
        from the dashboard event detail modal.

        If no prior record exists (e.g., Telegram not configured),
        creates a new feedback record so dashboard-only verdict still works.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM feedback WHERE event_id = ?", (event_id,)
            ).fetchone()

            if row:
                # Existing record — update it
                conn.execute(
                    """UPDATE feedback
                       SET verdict = ?, identity_label = ?, timestamp = ?
                       WHERE event_id = ?""",
                    (verdict, identity_label or row["identity_label"],
                     time.time(), event_id)
                )
            else:
                # No prior record (Telegram not configured or non-notified event)
                # Create a new feedback record from dashboard
                conn.execute(
                    """INSERT INTO feedback
                       (event_id, verdict, event_type, identity_label,
                        timestamp, telegram_message_id)
                       VALUES (?, ?, '', ?, ?, 0)""",
                    (event_id, verdict, identity_label or "",
                     time.time())
                )
                logger.info(f"Created new feedback record for {event_id} from dashboard")

            conn.commit()

            # Re-check auto-rules with the resolved record
            record = FeedbackRecord(
                event_id=event_id,
                verdict=verdict,
                event_type=row["event_type"] if row else "",
                identity_label=identity_label or (row["identity_label"] if row else ""),
                zone=row["zone"] if row else "",
                time_period=row["time_period"] if row else "",
                action=row["action"] if row else "",
                confidence=row["confidence"] if row else 0.0,
                camera_id=row["camera_id"] if row else "front_door",
                snapshot_path=row["snapshot_path"] if row else "",
                timestamp=time.time(),
            )
            self._check_and_create_rules(conn, record)

            logger.info(
                f"Resolved pending event {event_id}: "
                f"verdict={verdict} identity={identity_label}"
            )
            return True
        finally:
            conn.close()

    def get_pending_by_message_id(self, message_id: int) -> Optional[dict]:
        """Look up a pending event by its Telegram message ID."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM feedback WHERE telegram_message_id = ?",
                (message_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
