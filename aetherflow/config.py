"""Centralized configuration, state definitions, and hyperparameter tokens for AetherFlow."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class CognitiveState(str, Enum):
    """Classified user cognitive states."""
    FLOW = "FLOW"
    NORMAL_FOCUS = "NORMAL_FOCUS"
    COGNITIVE_OVERLOAD = "COGNITIVE_OVERLOAD"
    FATIGUE = "FATIGUE"
    IDLE = "IDLE"

    @classmethod
    def list(cls) -> list[str]:
        return [c.value for c in cls]


class ContextDomain(str, Enum):
    """Categorized creative context domains of the foreground window."""
    CODE_IDE = "CODE_IDE"
    CREATIVE_DESIGN = "CREATIVE_DESIGN"
    MEDIA_VIDEO = "MEDIA_VIDEO"
    WRITING_TEXT = "WRITING_TEXT"
    RESEARCH_BROWSING = "RESEARCH_BROWSING"
    SYSTEM_UTILITY = "SYSTEM_UTILITY"
    GENERAL = "GENERAL"

    @classmethod
    def list(cls) -> list[str]:
        return [c.value for c in cls]


class NotificationPolicy(str, Enum):
    """Notification suppression policies."""
    ALLOW_ALL = "ALLOW_ALL"
    PRIORITY_ONLY = "PRIORITY_ONLY"
    TOTAL_SILENCE = "TOTAL_SILENCE"


class UIDensityMode(str, Enum):
    """Workspace UI density modes."""
    COMPACT = "COMPACT"       # Dense information packing for deep flow
    STANDARD = "STANDARD"     # Baseline UI scaling
    COMFORT = "COMFORT"       # Slightly enlarged text/spacing to ease visual load
    RELAXED = "RELAXED"       # High spacing and enlarged elements for fatigue recovery


class BreakRecommendation(str, Enum):
    """Ergonomic break advice."""
    NONE = "NONE"
    GENTLE_PROMPT = "GENTLE_PROMPT"
    MANDATORY_BREAK = "MANDATORY_BREAK"


# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


@dataclass
class VisionConfig:
    """Configuration for facial and ocular biometric tracking."""
    camera_id: int = 0
    target_width: int = 640
    target_height: int = 480
    target_fps: int = 30
    
    # Eye Aspect Ratio (EAR) thresholds
    ear_blink_threshold: float = 0.21
    ear_perclos_threshold: float = 0.20
    min_blink_duration_ms: float = 60.0
    max_blink_duration_ms: float = 500.0
    
    # Path to MediaPipe task model
    model_asset_path: Path = DATA_DIR / "face_landmarker.task"
    synthetic_mode: bool = False


@dataclass
class PeripheralConfig:
    """Configuration for keyboard cadence and mouse dynamics."""
    max_history_events: int = 2000
    idle_timeout_sec: float = 10.0
    mouse_jitter_window_ms: float = 500.0
    synthetic_mode: bool = False


@dataclass
class EngineConfig:
    """Configuration for feature buffer and machine learning engine."""
    sliding_window_sec: float = 60.0
    step_interval_sec: float = 1.0
    min_samples_for_inference: int = 3
    model_path: Path = DATA_DIR / "cognitive_classifier.joblib"
    scaler_path: Path = DATA_DIR / "scaler.joblib"
    random_seed: int = 42


@dataclass
class AdaptationConfig:
    """Configuration for workspace actuation and display warmth."""
    # Specific adaptation interval (minimum cooldown in seconds between environment shifts)
    min_adaptation_interval_sec: float = 30.0

    # Hysteresis: consecutive ticks required before applying environment change
    hysteresis_ticks: int = 4
    
    # Gradual fade duration in seconds (prevents sudden brightness/color pops)
    transition_duration_sec: float = 4.0
    
    # Gamma ramp micro-steps (higher = more imperceptible and smooth)
    gamma_transition_steps: int = 40
    gamma_step_delay_sec: float = 0.10
    
    # Screen warmth boundaries (in Kelvin)
    min_color_temp_kelvin: int = 2700
    max_color_temp_kelvin: int = 6500
    baseline_color_temp_kelvin: int = 6500
    
    # Simulation / dry-run mode (does not alter hardware gamma)
    dry_run: bool = False


@dataclass
class AetherFlowConfig:
    """Master configuration container for AetherFlow."""
    vision: VisionConfig = field(default_factory=VisionConfig)
    peripheral: PeripheralConfig = field(default_factory=PeripheralConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    debug: bool = False

    def ensure_directories(self) -> None:
        """Ensure necessary runtime directories exist."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
