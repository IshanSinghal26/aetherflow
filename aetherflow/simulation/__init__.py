"""Simulation and model training layer for AetherFlow."""

from aetherflow.simulation.train_synthetic import (
    SyntheticTelemetryGenerator,
    train_and_serialize,
)

__all__ = [
    "SyntheticTelemetryGenerator",
    "train_and_serialize",
]
