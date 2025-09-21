"""Public package interface for detector orchestration."""

from importlib import import_module

__all__ = ["__version__", "create_orchestration_app", "Settings"]
__version__ = "0.1.0"

from .config import Settings


def create_orchestration_app(*args, **kwargs):
    """Lazy import wrapper to avoid package-level import cycles."""

    module = import_module(".service_factory", __name__)
    return module.create_orchestration_app(*args, **kwargs)
