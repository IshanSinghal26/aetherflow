"""AetherFlow - Local Neuroadaptive Workspace Engine.

A local, closed-loop workspace engine that passively monitors user cognitive
states (via webcam & peripheral telemetry) and dynamically adapts the desktop
environment (screen warmth, UI density, notification suppression) based on active
creative context.
"""

__version__ = "0.1.0"
__author__ = "AetherFlow Team"

from aetherflow.config import AetherFlowConfig, CognitiveState, ContextDomain

__all__ = [
    "AetherFlowConfig",
    "CognitiveState",
    "ContextDomain",
    "__version__",
]
