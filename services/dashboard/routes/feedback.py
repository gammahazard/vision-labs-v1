"""
routes/feedback.py — Feedback API for the self-learning loop.

PURPOSE:
    REST endpoints for viewing feedback history, managing suppression rules,
    and resolving pending identifications from the dashboard.

ENDPOINTS:
    GET  /api/feedback            — Recent feedback records
    GET  /api/feedback/stats      — Feedback statistics
    GET  /api/feedback/rules      — Active suppression rules
    POST /api/feedback/{event_id} — Submit/update a verdict from the dashboard
    POST /api/feedback/rules/{id}/toggle — Enable/disable a rule
    DELETE /api/feedback/rules/{id}      — Delete a rule
"""

import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

import routes as ctx

router = APIRouter(prefix="/api/feedback", tags=["feedback"])
logger = logging.getLogger("dashboard.feedback")

# feedback_db reference is set by server.py during startup
_feedback_db = None


def set_feedback_db(db):
    """Called by server.py to inject the feedback database instance."""
    global _feedback_db
    _feedback_db = db


class VerdictRequest(BaseModel):
    verdict: str                  # "real_threat", "false_alarm", "identified"
    identity_label: Optional[str] = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("")
async def get_feedback(limit: int = 50):
    """Get recent feedback records."""
    if not _feedback_db:
        return JSONResponse(status_code=503, content={"error": "Feedback DB not initialized"})
    return _feedback_db.get_recent_feedback(limit=limit)


@router.get("/stats")
async def get_stats():
    """Get feedback statistics for the dashboard."""
    if not _feedback_db:
        return JSONResponse(status_code=503, content={"error": "Feedback DB not initialized"})
    return _feedback_db.get_stats()


@router.get("/rules")
async def get_rules():
    """Get all suppression rules."""
    if not _feedback_db:
        return JSONResponse(status_code=503, content={"error": "Feedback DB not initialized"})
    return _feedback_db.get_suppression_rules()


@router.post("/{event_id}")
async def submit_verdict(event_id: str, body: VerdictRequest):
    """Submit or update a verdict from the dashboard review queue."""
    if not _feedback_db:
        return JSONResponse(status_code=503, content={"error": "Feedback DB not initialized"})

    if body.verdict not in ("real_threat", "false_alarm", "identified"):
        return JSONResponse(status_code=400, content={"error": "Invalid verdict"})

    ok = _feedback_db.resolve_pending(
        event_id, body.verdict, identity_label=body.identity_label or ""
    )
    if ok:
        return {"status": "ok", "event_id": event_id, "verdict": body.verdict}
    else:
        return JSONResponse(
            status_code=404,
            content={"error": f"Event {event_id} not found"},
        )


@router.post("/rules/{rule_id}/toggle")
async def toggle_rule(rule_id: int, active: bool = True):
    """Enable or disable a suppression rule."""
    if not _feedback_db:
        return JSONResponse(status_code=503, content={"error": "Feedback DB not initialized"})
    _feedback_db.toggle_rule(rule_id, active)
    return {"status": "ok", "rule_id": rule_id, "active": active}


@router.delete("/rules/{rule_id}")
async def delete_rule(rule_id: int):
    """Delete a suppression rule."""
    if not _feedback_db:
        return JSONResponse(status_code=503, content={"error": "Feedback DB not initialized"})
    _feedback_db.delete_rule(rule_id)
    return {"status": "ok", "rule_id": rule_id, "deleted": True}
