"""
Configuration management for the Mapper Service.

Single responsibility: Configuration loading and validation.
"""

from .settings import MapperSettings

__all__ = [
    "MapperSettings",
]
