"""Video pipeline package with lazy loading for image-only consumers."""

from importlib import import_module

__all__ = ["Pipeline"]


def __getattr__(name):
    if name == "Pipeline":
        return import_module(".pipeline", __name__).Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
