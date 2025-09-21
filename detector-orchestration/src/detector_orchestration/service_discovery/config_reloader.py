"""Simple configuration hot reloader that polls for changes."""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, List, Optional

from detector_orchestration.config import Settings

logger = logging.getLogger(__name__)

ReloadCallback = Callable[[], Awaitable[None]]


class ConfigurationHotReloader:
    """Periodically checks for configuration changes and notifies callbacks."""

    def __init__(
        self,
        *,
        settings: Settings,
        poll_interval_seconds: int = 30,
    ) -> None:
        self._settings = settings
        self._poll_interval = poll_interval_seconds
        self._callbacks: List[ReloadCallback] = []
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()

    def add_reload_callback(self, callback: ReloadCallback) -> None:
        """Register an async callback triggered when a reload occurs."""

        self._callbacks.append(callback)

    async def start_watching(self) -> None:
        """Begin background polling for configuration changes."""

        if self._task and not self._task.done():
            return
        self._stop_event.clear()
        logger.info("Starting configuration hot reloader")
        self._task = asyncio.create_task(self._watch_loop(), name="config-hot-reloader")

    async def stop_watching(self) -> None:
        """Stop background polling for configuration changes."""

        if not self._task:
            return
        logger.info("Stopping configuration hot reloader")
        self._stop_event.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            logger.debug("Configuration hot reloader task cancelled")
        finally:
            self._task = None

    async def reload_detectors(self) -> None:
        """Convenience entry point for detector-specific reload logic."""

        await self._trigger_callbacks()

    async def reload_policies(self) -> None:
        """Convenience entry point for policy-specific reload logic."""

        await self._trigger_callbacks()

    async def _watch_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(self._poll_interval)
                if await self._has_config_changed():
                    await self._trigger_callbacks()
        except asyncio.CancelledError:
            raise
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Error while watching configuration changes")
        finally:
            self._stop_event.set()

    async def _has_config_changed(self) -> bool:
        """Detect configuration changes.

        The current implementation always returns ``False``. A future
        improvement can hook into inotify or checksum comparisons of the
        underlying configuration files.
        """

        _ = self._settings  # Access to appease the linter until implemented.
        return False

    async def _trigger_callbacks(self) -> None:
        for callback in self._callbacks:
            try:
                await callback()
            except Exception:  # pylint: disable=broad-exception-caught
                logger.exception("Configuration reload callback failed")

