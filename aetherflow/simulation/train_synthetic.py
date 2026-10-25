"""Synthetic telemetry generation and offline model pre-training.

Generates realistic physiological and behavioral telemetry distributions across
the five cognitive states (FLOW, NORMAL_FOCUS, COGNITIVE_OVERLOAD, FATIGUE, IDLE),
evaluates classification performance, and serializes trained models for deployment.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from aetherflow.config import DATA_DIR, CognitiveState, EngineConfig
from aetherflow.engine.feature_buffer import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SyntheticTelemetryGenerator:
    """Generates biologically and kinematically plausible telemetry distributions."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def generate_dataset(self, samples_per_class: int = 800) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Generate synthetic feature matrix X and label vector y across all states."""
        records: List[Dict[str, float]] = []
        labels: List[str] = []

        for state in CognitiveState:
            for _ in range(samples_per_class):
                feat = self._sample_features_for_state(state)
                records.append(feat)
                labels.append(state.value)

        df = pd.DataFrame(records)
        # Ensure exact column ordering
        df = df[FEATURE_NAMES]
        X = df.values.astype(np.float32)
        y = np.array(labels)
        return X, y, df

    def _sample_features_for_state(self, state: CognitiveState) -> Dict[str, float]:
        """Draw realistic feature vector based on cognitive state profile."""
        rng = self.rng
        f: Dict[str, float] = {}

        if state == CognitiveState.FLOW:
            # Deep focus, rhythmic cadence, smooth mouse, low blink rate
            f["ear_mean"] = rng.normal(0.29, 0.02)
            f["ear_std"] = rng.normal(0.012, 0.003)
            f["ear_min"] = max(0.18, f["ear_mean"] - 0.06)
            f["ear_p25"] = f["ear_mean"] - 0.015
            f["blink_rate_bpm"] = rng.normal(11.0, 2.5)
            f["perclos"] = rng.uniform(0.01, 0.05)
            f["mean_blink_duration_ms"] = rng.normal(130.0, 20.0)
            f["gaze_stability_mean"] = rng.normal(0.92, 0.04)
            f["gaze_stability_std"] = rng.normal(0.04, 0.015)
            f["head_pitch_std"] = rng.normal(1.2, 0.4)
            f["head_yaw_std"] = rng.normal(1.5, 0.5)
            f["face_present_ratio"] = rng.uniform(0.95, 1.0)

            f["dwell_time_mean"] = rng.normal(95.0, 8.0)
            f["dwell_time_std"] = rng.normal(12.0, 3.0)
            f["flight_time_mean"] = rng.normal(160.0, 25.0)
            f["flight_time_std"] = rng.normal(30.0, 8.0)
            f["keystrokes_per_min"] = rng.normal(240.0, 45.0)
            f["typing_pause_ratio"] = rng.uniform(0.05, 0.20)
            f["typing_burstiness"] = rng.normal(0.55, 0.12)

            f["mouse_velocity_mean"] = rng.normal(180.0, 35.0)
            f["mouse_velocity_max"] = rng.normal(650.0, 100.0)
            f["mouse_jitter_mean"] = rng.normal(0.06, 0.02)
            f["mouse_jitter_std"] = rng.normal(0.03, 0.01)
            f["mouse_click_rate"] = rng.normal(0.65, 0.20)
            f["mouse_scroll_rate"] = rng.normal(0.35, 0.15)

            f["idle_ratio"] = rng.uniform(0.02, 0.12)
            f["activity_level"] = 1.0 - f["idle_ratio"]
            f["ear_trend"] = rng.normal(0.0, 0.0002)
            f["jitter_trend"] = rng.normal(0.0, 0.0003)

        elif state == CognitiveState.NORMAL_FOCUS:
            # Baseline productive state
            f["ear_mean"] = rng.normal(0.28, 0.025)
            f["ear_std"] = rng.normal(0.020, 0.005)
            f["ear_min"] = max(0.16, f["ear_mean"] - 0.08)
            f["ear_p25"] = f["ear_mean"] - 0.025
            f["blink_rate_bpm"] = rng.normal(18.0, 3.5)
            f["perclos"] = rng.uniform(0.04, 0.09)
            f["mean_blink_duration_ms"] = rng.normal(170.0, 25.0)
            f["gaze_stability_mean"] = rng.normal(0.78, 0.06)
            f["gaze_stability_std"] = rng.normal(0.08, 0.025)
            f["head_pitch_std"] = rng.normal(2.5, 0.8)
            f["head_yaw_std"] = rng.normal(3.0, 0.9)
            f["face_present_ratio"] = rng.uniform(0.90, 1.0)

            f["dwell_time_mean"] = rng.normal(110.0, 12.0)
            f["dwell_time_std"] = rng.normal(18.0, 5.0)
            f["flight_time_mean"] = rng.normal(240.0, 45.0)
            f["flight_time_std"] = rng.normal(55.0, 15.0)
            f["keystrokes_per_min"] = rng.normal(150.0, 40.0)
            f["typing_pause_ratio"] = rng.uniform(0.20, 0.45)
            f["typing_burstiness"] = rng.normal(0.85, 0.20)

            f["mouse_velocity_mean"] = rng.normal(130.0, 30.0)
            f["mouse_velocity_max"] = rng.normal(500.0, 90.0)
            f["mouse_jitter_mean"] = rng.normal(0.14, 0.04)
            f["mouse_jitter_std"] = rng.normal(0.06, 0.02)
            f["mouse_click_rate"] = rng.normal(0.40, 0.15)
            f["mouse_scroll_rate"] = rng.normal(0.25, 0.10)

            f["idle_ratio"] = rng.uniform(0.15, 0.35)
            f["activity_level"] = 1.0 - f["idle_ratio"]
            f["ear_trend"] = rng.normal(0.0, 0.0004)
            f["jitter_trend"] = rng.normal(0.0, 0.0005)

        elif state == CognitiveState.COGNITIVE_OVERLOAD:
            # High jitter, erratic saccades, bursty fragmented typing
            f["ear_mean"] = rng.normal(0.27, 0.03)
            f["ear_std"] = rng.normal(0.032, 0.008)
            f["ear_min"] = max(0.14, f["ear_mean"] - 0.10)
            f["ear_p25"] = f["ear_mean"] - 0.035
            f["blink_rate_bpm"] = rng.normal(28.0, 6.0) # High or rapid flurries
            f["perclos"] = rng.uniform(0.05, 0.11)
            f["mean_blink_duration_ms"] = rng.normal(150.0, 30.0)
            f["gaze_stability_mean"] = rng.normal(0.42, 0.09) # Low gaze stability
            f["gaze_stability_std"] = rng.normal(0.18, 0.04)
            f["head_pitch_std"] = rng.normal(4.5, 1.2)
            f["head_yaw_std"] = rng.normal(5.5, 1.5)
            f["face_present_ratio"] = rng.uniform(0.85, 1.0)

            f["dwell_time_mean"] = rng.normal(118.0, 18.0)
            f["dwell_time_std"] = rng.normal(32.0, 8.0)
            f["flight_time_mean"] = rng.normal(310.0, 70.0)
            f["flight_time_std"] = rng.normal(120.0, 35.0) # High variance
            f["keystrokes_per_min"] = rng.normal(160.0, 60.0)
            f["typing_pause_ratio"] = rng.uniform(0.25, 0.55)
            f["typing_burstiness"] = rng.normal(1.65, 0.35) # Choppy bursts

            f["mouse_velocity_mean"] = rng.normal(240.0, 50.0)
            f["mouse_velocity_max"] = rng.normal(850.0, 150.0)
            f["mouse_jitter_mean"] = rng.normal(0.48, 0.09) # High jitter
            f["mouse_jitter_std"] = rng.normal(0.16, 0.04)
            f["mouse_click_rate"] = rng.normal(0.85, 0.25)
            f["mouse_scroll_rate"] = rng.normal(0.70, 0.25)

            f["idle_ratio"] = rng.uniform(0.10, 0.28)
            f["activity_level"] = 1.0 - f["idle_ratio"]
            f["ear_trend"] = rng.normal(-0.0003, 0.0004)
            f["jitter_trend"] = rng.normal(0.0025, 0.0008) # Increasing jitter

        elif state == CognitiveState.FATIGUE:
            # Low EAR, high PERCLOS, slow blinks, sluggish typing, drooping trend
            f["ear_mean"] = rng.normal(0.21, 0.02) # Low average EAR
            f["ear_std"] = rng.normal(0.035, 0.009)
            f["ear_min"] = max(0.08, f["ear_mean"] - 0.12)
            f["ear_p25"] = f["ear_mean"] - 0.04
            f["blink_rate_bpm"] = rng.normal(24.0, 5.0)
            f["perclos"] = rng.uniform(0.22, 0.48) # Substantial eye closure
            f["mean_blink_duration_ms"] = rng.normal(360.0, 60.0) # Slow blinks
            f["gaze_stability_mean"] = rng.normal(0.55, 0.08)
            f["gaze_stability_std"] = rng.normal(0.14, 0.03)
            f["head_pitch_std"] = rng.normal(3.8, 1.0)
            f["head_yaw_std"] = rng.normal(2.5, 0.7)
            f["face_present_ratio"] = rng.uniform(0.80, 0.98)

            f["dwell_time_mean"] = rng.normal(155.0, 25.0) # Sluggish key presses
            f["dwell_time_std"] = rng.normal(28.0, 7.0)
            f["flight_time_mean"] = rng.normal(480.0, 110.0)
            f["flight_time_std"] = rng.normal(140.0, 40.0)
            f["keystrokes_per_min"] = rng.normal(55.0, 25.0)
            f["typing_pause_ratio"] = rng.uniform(0.55, 0.85)
            f["typing_burstiness"] = rng.normal(1.20, 0.30)

            f["mouse_velocity_mean"] = rng.normal(45.0, 18.0) # Slow, lethargic mouse
            f["mouse_velocity_max"] = rng.normal(220.0, 60.0)
            f["mouse_jitter_mean"] = rng.normal(0.18, 0.05)
            f["mouse_jitter_std"] = rng.normal(0.07, 0.02)
            f["mouse_click_rate"] = rng.normal(0.15, 0.08)
            f["mouse_scroll_rate"] = rng.normal(0.12, 0.06)

            f["idle_ratio"] = rng.uniform(0.45, 0.75)
            f["activity_level"] = 1.0 - f["idle_ratio"]
            f["ear_trend"] = rng.normal(-0.0028, 0.0006) # Progressive drooping
            f["jitter_trend"] = rng.normal(0.0, 0.0004)

        else: # CognitiveState.IDLE
            # Inactivity, user absent or stationary
            f["ear_mean"] = rng.uniform(0.0, 0.22)
            f["ear_std"] = rng.uniform(0.0, 0.01)
            f["ear_min"] = 0.0
            f["ear_p25"] = 0.0
            f["blink_rate_bpm"] = 0.0
            f["perclos"] = 0.0
            f["mean_blink_duration_ms"] = 0.0
            f["gaze_stability_mean"] = 0.0
            f["gaze_stability_std"] = 0.0
            f["head_pitch_std"] = 0.0
            f["head_yaw_std"] = 0.0
            f["face_present_ratio"] = rng.uniform(0.0, 0.25)

            f["dwell_time_mean"] = 0.0
            f["dwell_time_std"] = 0.0
            f["flight_time_mean"] = 0.0
            f["flight_time_std"] = 0.0
            f["keystrokes_per_min"] = 0.0
            f["typing_pause_ratio"] = 1.0
            f["typing_burstiness"] = 0.0

            f["mouse_velocity_mean"] = 0.0
            f["mouse_velocity_max"] = 0.0
            f["mouse_jitter_mean"] = 0.0
            f["mouse_jitter_std"] = 0.0
            f["mouse_click_rate"] = 0.0
            f["mouse_scroll_rate"] = 0.0

            f["idle_ratio"] = rng.uniform(0.92, 1.0)
            f["activity_level"] = 1.0 - f["idle_ratio"]
            f["ear_trend"] = 0.0
            f["jitter_trend"] = 0.0

        # Clamp bounded features
        f["perclos"] = float(np.clip(f["perclos"], 0.0, 1.0))
        f["gaze_stability_mean"] = float(np.clip(f["gaze_stability_mean"], 0.0, 1.0))
        f["mouse_jitter_mean"] = float(np.clip(f["mouse_jitter_mean"], 0.0, 1.0))
        f["idle_ratio"] = float(np.clip(f["idle_ratio"], 0.0, 1.0))
        f["activity_level"] = float(np.clip(f["activity_level"], 0.0, 1.0))
        f["typing_pause_ratio"] = float(np.clip(f["typing_pause_ratio"], 0.0, 1.0))
        f["face_present_ratio"] = float(np.clip(f["face_present_ratio"], 0.0, 1.0))
        f["keystrokes_per_min"] = float(max(0.0, f["keystrokes_per_min"]))
        f["mouse_velocity_mean"] = float(max(0.0, f["mouse_velocity_mean"]))

        return f


def train_and_serialize(
    samples_per_class: int = 800,
    output_dir: Path | str = DATA_DIR,
    seed: int = 42,
    evaluate: bool = True
) -> Tuple[RandomForestClassifier, StandardScaler]:
    """Train RandomForest cognitive state classifier, evaluate, and save artifacts."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    generator = SyntheticTelemetryGenerator(seed=seed)
    logger.info(f"Generating synthetic telemetry dataset ({samples_per_class} samples/class)...")
    X, y, df = generator.generate_dataset(samples_per_class=samples_per_class)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Fitting RandomForestClassifier (n_estimators=150, max_depth=14)...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    if evaluate:
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, digits=4)
        cm = confusion_matrix(y_test, y_pred, labels=CognitiveState.list())
        logger.info("\n" + "=" * 60 + "\nCLASSIFICATION EVALUATION REPORT:\n" + "=" * 60)
        logger.info("\n" + report)
        logger.info("Confusion Matrix:\n" + str(cm))

    model_path = out_path / "cognitive_classifier.joblib"
    scaler_path = out_path / "scaler.joblib"

    joblib.dump(model, str(model_path))
    joblib.dump(scaler, str(scaler_path))
    logger.info(f"Model successfully saved to {model_path}")
    logger.info(f"Scaler successfully saved to {scaler_path}")

    return model, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train AetherFlow cognitive state classifier.")
    parser.add_argument("--samples", type=int, default=800, help="Samples per cognitive state class")
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR), help="Output directory for joblib models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--evaluate", action="store_true", default=True, help="Print evaluation metrics")
    args = parser.parse_args()

    train_and_serialize(
        samples_per_class=args.samples,
        output_dir=args.output_dir,
        seed=args.seed,
        evaluate=args.evaluate
    )
