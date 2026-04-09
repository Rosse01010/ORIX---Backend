"""
Redis Streams consumer that relays detection events to WebSocket clients.

Runs as a background asyncio task inside the FastAPI process so the API
stays informed of worker results without polling.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

import redis.asyncio as aioredis
from structlog import get_logger

from app.config import settings
from app.websocket.manager import manager

log = get_logger(__name__)

CONSUMER_GROUP = "ws_notifier"
CONSUMER_NAME = "api_0"


async def _ensure_group(client: aioredis.Redis, stream: str) -> None:
    try:
        await client.xgroup_create(stream, CONSUMER_GROUP, id="$", mkstream=True)
    except aioredis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


async def relay_events_task() -> None:
    """
    Continuously reads from stream:events and broadcasts to WS clients.
    Should be started as an asyncio background task on app startup.
    """
    client = aioredis.from_url(settings.redis_url, decode_responses=True)
    stream = settings.stream_events

    await _ensure_group(client, stream)
    log.info("ws_notifier_started", stream=stream)

    while True:
        try:
            results = await client.xreadgroup(
                CONSUMER_GROUP,
                CONSUMER_NAME,
                {stream: ">"},
                count=20,
                block=500,
            )
            if not results:
                await asyncio.sleep(0.01)
                continue

            for _stream, messages in results:
                for msg_id, fields in messages:
                    await _handle_message(fields)
                    await client.xack(stream, CONSUMER_GROUP, msg_id)

        except aioredis.ConnectionError:
            log.warning("ws_notifier_redis_reconnect")
            await asyncio.sleep(2)
        except Exception as exc:
            log.exception("ws_notifier_error", error=str(exc))
            await asyncio.sleep(1)


async def _handle_message(fields: Dict[str, str]) -> None:
    try:
        payload: Dict[str, Any] = json.loads(fields.get("payload", "{}"))
        camera_id: str = payload.get("camera", "")
        if camera_id:
            await manager.broadcast_to_camera(camera_id, payload)
        else:
            await manager.broadcast(payload)
    except Exception as exc:
        log.warning("ws_notifier_bad_msg", error=str(exc))
