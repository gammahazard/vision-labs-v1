"""
routes/unknowns.py — Unknown face proxy endpoints.

PURPOSE:
    Proxy all unknown face management requests to the face-recognizer
    service. Supports listing, viewing photos, labeling (promoting
    to known), clearing all, and deleting individual unknowns.

    When an unknown is labeled, emits a person_identified event to
    the event stream so it appears in the event feed.

ENDPOINTS:
    GET    /api/unknowns              — List auto-captured unknowns
    GET    /api/unknowns/{uid}/photo  — Get unknown face thumbnail
    POST   /api/unknowns/{uid}/label  — Label (promote to known face)
    DELETE /api/unknowns/clear        — Clear all unknowns
    DELETE /api/unknowns/{uid}        — Delete single unknown
"""

import json
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
import httpx

import routes as ctx

router = APIRouter(prefix="/api", tags=["unknowns"])


@router.get("/unknowns")
async def list_unknowns():
    """Proxy: list auto-captured unknown faces."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{ctx.FACE_API_URL}/api/unknowns", timeout=5)
            return resp.json()
    except Exception as e:
        ctx.logger.warning(f"Unknown faces API unavailable: {e}")
        return JSONResponse(status_code=503, content={"error": "Face recognizer not available"})


@router.get("/unknowns/{uid}/photo")
async def get_unknown_photo(uid: int):
    """Proxy: get unknown face thumbnail."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{ctx.FACE_API_URL}/api/unknowns/{uid}/photo", timeout=5)
            if resp.status_code == 200:
                return Response(content=resp.content, media_type="image/jpeg")
            return JSONResponse(status_code=404, content={"error": "Unknown face not found"})
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": "Face recognizer not available"})


@router.post("/unknowns/{uid}/label")
async def label_unknown(uid: int, data: dict):
    """Proxy: promote unknown face to known by assigning a name."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{ctx.FACE_API_URL}/api/unknowns/{uid}/label",
                json=data,
                timeout=10,
            )
            result = resp.json()

            # On success, emit a person_identified event to the event feed
            if resp.status_code == 200 and result.get("success"):
                name = data.get("name", "Unknown")
                try:
                    ctx.r.xadd(ctx.EVENT_STREAM, {
                        "event_type": "person_identified",
                        "person_id": f"unknown_{uid}",
                        "identity_name": name,
                        "camera_id": "front_door",
                        "action": "labeled",
                        "timestamp": str(time.time()),
                    })
                    ctx.logger.info(f"Unknown {uid} labeled as '{name}' — event emitted")
                except Exception as e:
                    ctx.logger.warning(f"Failed to emit label event: {e}")

            return JSONResponse(status_code=resp.status_code, content=result)
    except Exception as e:
        ctx.logger.warning(f"Label unknown failed: {e}")
        return JSONResponse(status_code=503, content={"error": "Face recognizer not available"})


@router.delete("/unknowns/clear")
async def clear_all_unknowns():
    """Proxy: remove all auto-captured unknown faces."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.delete(f"{ctx.FACE_API_URL}/api/unknowns", timeout=5)
            return resp.json()
    except Exception as e:
        ctx.logger.warning(f"Clear all unknowns failed: {e}")
        return JSONResponse(status_code=503, content={"error": "Face recognizer not available"})


@router.delete("/unknowns/{uid}")
async def delete_unknown(uid: int):
    """Proxy: remove an auto-captured unknown face."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.delete(f"{ctx.FACE_API_URL}/api/unknowns/{uid}", timeout=5)
            return resp.json()
    except Exception as e:
        ctx.logger.warning(f"Delete unknown failed: {e}")
        return JSONResponse(status_code=503, content={"error": "Face recognizer not available"})
