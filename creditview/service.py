"""Compatibility import; all website inference uses the notebook-backed service."""
from creditview.notebook_service import CreditDecisionService

__all__ = ["CreditDecisionService"]
