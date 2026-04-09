"""
FastAPI application entry-point.

Startup sequence:
  1. Init PostgreSQL tables (pgvector).
  2. Start Redis → WebSocket relay background task.
  3. Mount routes and WebSocket endpoint.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from structlog import get_logger

from app.config import settings
from app.database import init_db
from app.routes import health as health_router
from app.routes import recognition as recognition_router
from app.websocket.manager import manager
from app.websocket.notifications import relay_events_task
from utils.logging_utils import configure_logging

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging(settings.worker_log_level)
    log.info("startup", env=settings.app_env)

    await init_db()
    log.info("db_ready")

    relay_task = asyncio.create_task(relay_events_task())
    log.info("ws_relay_started")

    yield

    relay_task.cancel()
    try:
        await relay_task
    except asyncio.CancelledError:
        pass
    log.info("shutdown")


app = FastAPI(
    title="ORIX Face Recognition API",
    description="Real-time facial recognition with InsightFace, pgvector, and WebSockets.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────────
app.include_router(health_router.router)
app.include_router(recognition_router.router)


# ── WebSocket ──────────────────────────────────────────────────────────────────

@app.websocket("/ws/detections")
async def ws_detections(
    websocket: WebSocket,
    cameras: Optional[str] = Query(None, description="Comma-separated camera IDs to filter"),
) -> None:
    """
    WebSocket endpoint.  Frontend connects here to receive real-time detection events.

    Query param ?cameras=cam_01,cam_02 filters to specific cameras.
    Without the param the client receives events from ALL cameras.

    Event format:
    {
        "camera": "cam_01",
        "timestamp": "2026-04-08T12:00:00+00:00",
        "bboxes": [
            {"x": 120, "y": 60, "width": 100, "height": 100, "name": "Carlos", "confidence": 0.87}
        ]
    }
    """
    cam_list: Optional[List[str]] = (
        [c.strip() for c in cameras.split(",") if c.strip()] if cameras else None
    )
    await manager.connect(websocket, cameras=cam_list)
    try:
        while True:
            # Keep connection alive; workers push events via Redis relay
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket)
