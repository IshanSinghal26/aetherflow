"""Workspace adaptation and actuation layer for AetherFlow."""

from aetherflow.adaptation.executor import (
    AdaptationExecutor,
    ExecutionResult,
    kelvin_to_rgb,
)
from aetherflow.adaptation.profiles import AdaptationIntent, ProfileRegistry

__all__ = [
    "AdaptationExecutor",
    "AdaptationIntent",
    "ProfileRegistry",
    "ExecutionResult",
    "kelvin_to_rgb",
]
