"""Compatibility import for callers that historically used api:app."""
from backend.main import app

__all__ = ["app"]
