"""Machine learning classifier for cognitive state inference.

Evaluates multi-modal physiological and behavioral features to predict
cognitive state (FLOW, NORMAL_FOCUS, COGNITIVE_OVERLOAD, FATIGUE, IDLE)
with calibrated probabilities and explainability vectors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from aetherflow.config import CognitiveState, EngineConfig
from aetherflow.engine.feature_buffer import FEATURE_NAMES

logger = logging.getLogger(__name__)


@dataclass
class CognitiveInference:
    """Output packet of the cognitive state inference engine."""
    state: CognitiveState
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    dominant_features: List[Tuple[str, float]] = field(default_factory=list)
    timestamp: float = 0.0


class CognitiveClassifier:
    """Supervised classifier predicting real-time cognitive states from multi-modal vectors."""

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()
        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.classes: List[str] = CognitiveState.list()
        
        # Attempt to auto-load pre-trained weights if existing
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        """Attempt loading serialized model and scaler from configured paths."""
        model_path = Path(self.config.model_path)
        scaler_path = Path(self.config.scaler_path)

        if model_path.exists() and scaler_path.exists():
            try:
                self.load(model_path, scaler_path)
                logger.info(f"Loaded trained cognitive classifier from {model_path}.")
            except Exception as e:
                logger.warning(f"Failed loading model from {model_path}: {e}. Fallback rule engine enabled.")

    def load(self, model_path: Path | str, scaler_path: Path | str) -> None:
        """Load serialized model and scaler artifacts from disk."""
        self.model = joblib.load(str(model_path))
        self.scaler = joblib.load(str(scaler_path))
        if hasattr(self.model, "classes_"):
            self.classes = [str(c) for c in self.model.classes_]

    def save(self, model_path: Path | str, scaler_path: Path | str) -> None:
        """Persist trained model and scaler to disk."""
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, str(model_path))
        joblib.dump(self.scaler, str(scaler_path))
        logger.info(f"Saved cognitive classifier to {model_path} and scaler to {scaler_path}.")

    def predict(self, feature_vector: np.ndarray, feature_dict: Dict[str, float], timestamp: float) -> CognitiveInference:
        """Classify feature vector into CognitiveInference packet."""
        # 1. Check for immediate explicit IDLE conditions
        if feature_dict.get("activity_level", 1.0) < 0.05 and (not feature_dict.get("face_present_ratio", 1.0) or feature_dict.get("idle_ratio", 0.0) > 0.85):
            probs = {c: 0.01 for c in self.classes}
            probs[CognitiveState.IDLE.value] = 0.96
            return CognitiveInference(
                state=CognitiveState.IDLE,
                confidence=0.96,
                probabilities=probs,
                dominant_features=[("idle_ratio", feature_dict.get("idle_ratio", 1.0))],
                timestamp=timestamp,
            )

        # 2. Use ML model if trained and available
        if self.model is not None and self.scaler is not None:
            try:
                vec_2d = feature_vector.reshape(1, -1)
                proba = self.model.predict_proba(vec_2d)[0]
                state_idx = int(np.argmax(proba))
                state_name = self.classes[state_idx]
                confidence = float(proba[state_idx])

                prob_dict = {cls_name: float(p) for cls_name, p in zip(self.classes, proba)}

                # Determine dominant explanation signals
                top_features = self._explain_prediction(feature_dict, state_name)

                return CognitiveInference(
                    state=CognitiveState(state_name),
                    confidence=round(confidence, 4),
                    probabilities=prob_dict,
                    dominant_features=top_features,
                    timestamp=timestamp,
                )
            except Exception as e:
                logger.debug(f"ML inference error: {e}. Falling back to heuristic classifier.")

        # 3. Fallback Heuristic Classifier (Zero-downtime cold start)
        return self._heuristic_classify(feature_dict, timestamp)

    def _explain_prediction(self, feature_dict: Dict[str, float], state_name: str) -> List[Tuple[str, float]]:
        """Identify key contributing biometric & peripheral markers for explainability."""
        explanations: List[Tuple[str, float]] = []

        if state_name == CognitiveState.FLOW.value:
            explanations.append(("gaze_stability_mean", round(feature_dict.get("gaze_stability_mean", 0.9), 3)))
            explanations.append(("mouse_jitter_mean", round(feature_dict.get("mouse_jitter_mean", 0.05), 3)))
            explanations.append(("keystrokes_per_min", round(feature_dict.get("keystrokes_per_min", 200.0), 1)))
        elif state_name == CognitiveState.FATIGUE.value:
            explanations.append(("perclos", round(feature_dict.get("perclos", 0.25), 3)))
            explanations.append(("mean_blink_duration_ms", round(feature_dict.get("mean_blink_duration_ms", 320.0), 1)))
            explanations.append(("dwell_time_mean", round(feature_dict.get("dwell_time_mean", 140.0), 1)))
        elif state_name == CognitiveState.COGNITIVE_OVERLOAD.value:
            explanations.append(("mouse_jitter_mean", round(feature_dict.get("mouse_jitter_mean", 0.45), 3)))
            explanations.append(("typing_burstiness", round(feature_dict.get("typing_burstiness", 1.8), 2)))
            explanations.append(("gaze_stability_mean", round(feature_dict.get("gaze_stability_mean", 0.35), 3)))
        elif state_name == CognitiveState.IDLE.value:
            explanations.append(("idle_ratio", round(feature_dict.get("idle_ratio", 0.95), 2)))
        else: # NORMAL_FOCUS
            explanations.append(("blink_rate_bpm", round(feature_dict.get("blink_rate_bpm", 16.0), 1)))
            explanations.append(("mouse_velocity_mean", round(feature_dict.get("mouse_velocity_mean", 120.0), 1)))

        return explanations

    def _heuristic_classify(self, f: Dict[str, float], timestamp: float) -> CognitiveInference:
        """Robust deterministic rule-engine baseline used prior to model training."""
        perclos = f.get("perclos", 0.0)
        ear_mean = f.get("ear_mean", 0.28)
        blink_dur = f.get("mean_blink_duration_ms", 150.0)
        jitter = f.get("mouse_jitter_mean", 0.1)
        gaze_stab = f.get("gaze_stability_mean", 0.8)
        kpm = f.get("keystrokes_per_min", 0.0)
        idle = f.get("idle_ratio", 0.0)

        # 1. Idle rule
        if idle > 0.7:
            state = CognitiveState.IDLE
            conf = 0.90
        # 2. Fatigue rule (high PERCLOS, slow blinks, low EAR)
        elif perclos > 0.15 or blink_dur > 280.0 or ear_mean < 0.19:
            state = CognitiveState.FATIGUE
            conf = 0.85
        # 3. Overload rule (erratic mouse jitter, unstable gaze, high burstiness)
        elif jitter > 0.28 or gaze_stab < 0.45:
            state = CognitiveState.COGNITIVE_OVERLOAD
            conf = 0.82
        # 4. Flow rule (sustained activity, stable gaze, low jitter, rhythmic cadence)
        elif gaze_stab >= 0.75 and jitter < 0.15 and (kpm > 120.0 or f.get("mouse_velocity_mean", 0.0) > 80.0):
            state = CognitiveState.FLOW
            conf = 0.88
        else:
            state = CognitiveState.NORMAL_FOCUS
            conf = 0.75

        probs = {c: 0.05 for c in self.classes}
        probs[state.value] = conf
        # Normalize probabilities
        tot = sum(probs.values())
        norm_probs = {k: round(v / tot, 4) for k, v in probs.items()}

        return CognitiveInference(
            state=state,
            confidence=conf,
            probabilities=norm_probs,
            dominant_features=self._explain_prediction(f, state.value),
            timestamp=timestamp,
        )
