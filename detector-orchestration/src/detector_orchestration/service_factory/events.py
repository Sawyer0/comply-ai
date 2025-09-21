"""Event publication utilities for orchestration service factory."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List

import httpx

logger = logging.getLogger(__name__)


class ServiceFactoryEventsMixin:
    """Handles orchestration event queues and webhook notifications."""

    # pylint: disable=too-few-public-methods

    settings: Any
    _event_subscribers: List[asyncio.Queue]
    _incident_history: Deque[Dict[str, Any]]

    def register_event_subscriber(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=128)
        self._event_subscribers.append(queue)
        return queue

    def unregister_event_subscriber(self, queue: asyncio.Queue) -> None:
        try:
            self._event_subscribers.remove(queue)
        except ValueError:
            pass

    def get_recent_incidents(self) -> List[Dict[str, Any]]:
        return list(self._incident_history)

    def publish_event(self, event: Dict[str, Any]) -> None:
        enriched = dict(event)
        enriched.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

        if enriched.get("type") == "incident":
            self._incident_history.append(enriched)
            asyncio.create_task(self.notify_incident_webhooks(enriched))

        stale: List[asyncio.Queue] = []
        for queue in list(self._event_subscribers):
            try:
                queue.put_nowait(enriched)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                    queue.put_nowait(enriched)
                except Exception:  # pylint: disable=broad-exception-caught
                    stale.append(queue)
            except Exception:  # pylint: disable=broad-exception-caught
                stale.append(queue)

        for queue in stale:
            self.unregister_event_subscriber(queue)

    async def notify_incident_webhooks(self, event: Dict[str, Any]) -> None:
        webhooks = getattr(
            getattr(self.settings, "config", None),
            "incident_notification_webhooks",
            [],
        )
        if not webhooks:
            return

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                for url in webhooks:
                    try:
                        await client.post(url, json=event)
                    except Exception:  # pylint: disable=broad-exception-caught
                        logger.warning(
                            "Failed to deliver incident notification", exc_info=True
                        )
        except Exception:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Incident webhook notifications failed to initialize", exc_info=True
            )


__all__ = ["ServiceFactoryEventsMixin"]
