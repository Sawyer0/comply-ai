"""Shared plugin base classes to enforce consistent lifecycle behaviour."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict


class PluginInterface(ABC):
    """Common interface for orchestration plugins."""

    @property
    @abstractmethod
    def plugin_name(self) -> str:
        """Return the plugin identifier."""

    @property
    @abstractmethod
    def plugin_version(self) -> str:
        """Return the plugin version string."""

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Perform plugin initialization using the provided configuration."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Release any resources acquired by the plugin."""


class PluginBase(PluginInterface):
    """Base implementation that handles repeated lifecycle plumbing."""

    def __init__(self, name: str, version: str) -> None:
        self._name = name
        self._version = version
        self._config: Dict[str, Any] = {}
        self._initialized = False
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # --- interface implementation -------------------------------------------------
    @property
    def plugin_name(self) -> str:
        return self._name

    @property
    def plugin_version(self) -> str:
        return self._version

    async def initialize(self, config: Dict[str, Any]) -> bool:
        try:
            self._config = config.copy()
            await self._initialize_plugin()
            self._initialized = True
            self._logger.info("Plugin initialized: %s", self._name)
            return True
        except Exception as exc:  # pragma: no cover - logging helper
            self._logger.error(
                "Failed to initialize plugin %s: %s", self._name, exc
            )
            return False

    async def cleanup(self) -> None:
        try:
            if self._initialized:
                await self._cleanup_plugin()
                self._initialized = False
                self._logger.info("Plugin cleaned up: %s", self._name)
        except Exception as exc:  # pragma: no cover - logging helper
            self._logger.error("Failed to cleanup plugin %s: %s", self._name, exc)

    # --- helpers for subclasses ---------------------------------------------------
    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(f"Plugin {self._name} not initialized")

    def get_config(self) -> Dict[str, Any]:
        return self._config.copy()

    async def _initialize_plugin(self) -> None:
        """Hook for plugin specific initialization."""

    async def _cleanup_plugin(self) -> None:
        """Hook for plugin specific cleanup."""


__all__ = ["PluginInterface", "PluginBase"]
