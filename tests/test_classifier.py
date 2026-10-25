"""Unit tests for Layer 3: Cognitive Classifier model and explainability."""

import time
from pathlib import Path

import numpy as np
import pytest

from aetherflow.config import DATA_DIR, CognitiveState, EngineConfig
from aetherflow.engine.classifier import CognitiveClassifier, CognitiveInference
from aetherflow.engine.feature_buffer import FEATURE_NAMES


class TestCognitiveClassifier:
    """Tests for ML inference, heuristic fallback, confidence calibration, and explainability."""

    def test_heuristic_flow_classification(self):
        # Heuristic classifier without pre-trained model loaded
        cfg = EngineConfig(model_path=Path("non_existent_model.joblib"), scaler_path=Path("non_existent_scaler.joblib"))
        classifier = CognitiveClassifier(cfg)
        classifier.model = None
        classifier.scaler = None

        flow_features = {
            "gaze_stability_mean": 0.92,
            "mouse_jitter_mean": 0.05,
            "keystrokes_per_min": 240.0,
            "ear_mean": 0.29,
            "perclos": 0.02,
            "mean_blink_duration_ms": 130.0,
            "idle_ratio": 0.05,
            "activity_level": 0.95,
        }
        vec = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        inf = classifier.predict(vec, flow_features, time.time())
        assert isinstance(inf, CognitiveInference)
        assert inf.state == CognitiveState.FLOW
        assert inf.confidence >= 0.75
        assert len(inf.dominant_features) > 0

    def test_heuristic_fatigue_classification(self):
        cfg = EngineConfig(model_path=Path("non_existent_model.joblib"), scaler_path=Path("non_existent_scaler.joblib"))
        classifier = CognitiveClassifier(cfg)
        classifier.model = None

        fatigue_features = {
            "perclos": 0.35,
            "mean_blink_duration_ms": 380.0,
            "ear_mean": 0.18,
            "idle_ratio": 0.30,
            "activity_level": 0.70,
        }
        vec = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        inf = classifier.predict(vec, fatigue_features, time.time())
        assert inf.state == CognitiveState.FATIGUE
        assert inf.confidence >= 0.80

    def test_heuristic_overload_classification(self):
        cfg = EngineConfig(model_path=Path("non_existent_model.joblib"), scaler_path=Path("non_existent_scaler.joblib"))
        classifier = CognitiveClassifier(cfg)
        classifier.model = None

        overload_features = {
            "mouse_jitter_mean": 0.45,
            "gaze_stability_mean": 0.35,
            "idle_ratio": 0.10,
            "activity_level": 0.90,
        }
        vec = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        inf = classifier.predict(vec, overload_features, time.time())
        assert inf.state == CognitiveState.COGNITIVE_OVERLOAD

    def test_idle_rule_priority(self):
        classifier = CognitiveClassifier()
        idle_features = {
            "idle_ratio": 0.95,
            "activity_level": 0.02,
            "face_present_ratio": 0.0,
        }
        vec = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        inf = classifier.predict(vec, idle_features, time.time())
        assert inf.state == CognitiveState.IDLE

    def test_trained_model_inference_if_exists(self):
        model_path = DATA_DIR / "cognitive_classifier.joblib"
        scaler_path = DATA_DIR / "scaler.joblib"
        if model_path.exists() and scaler_path.exists():
            cfg = EngineConfig(model_path=model_path, scaler_path=scaler_path)
            classifier = CognitiveClassifier(cfg)
            assert classifier.model is not None
            assert classifier.scaler is not None

            # Generate random normalized feature vector
            vec = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
            dummy_dict = {name: 0.0 for name in FEATURE_NAMES}
            dummy_dict["activity_level"] = 0.8
            dummy_dict["idle_ratio"] = 0.2
            inf = classifier.predict(vec, dummy_dict, time.time())
            assert inf.state in CognitiveState
            assert 0.0 <= inf.confidence <= 1.0
            assert pytest.approx(sum(inf.probabilities.values()), abs=0.01) == 1.0
