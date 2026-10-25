"""Context awareness layer for AetherFlow: Foreground App & Creative Domain Detection."""

from aetherflow.config import ContextDomain
from aetherflow.context.app_detector import ActiveAppContext, AppDetector

__all__ = [
    "AppDetector",
    "ActiveAppContext",
    "ContextDomain",
]
