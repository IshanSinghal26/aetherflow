"""Sliding window feature buffer and statistical vector aggregator.

Aggregates high-frequency vision and peripheral streams over a 60-second
rolling window, computing statistical distribution moments, trend slopes,
and normalized feature vectors for ML cognitive inference.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aetherflow.config import EngineConfig
from aetherflow.sensing.peripheral_tracker import PeripheralSample
from aetherflow.sensing.vision_tracker import VisionSample

logger = logging.getLogger(__name__)


# Ordered canonical list of all 29 tabular features fed to the classifier
FEATURE_NAMES: List[str] = [
    # Ocular / Facial biometrics (12 features)
    "ear_mean",
    "ear_std",
    "ear_min",
    "ear_p25",
    "blink_rate_bpm",
    "perclos",
    "mean_blink_duration_ms",
    "gaze_stability_mean",
    "gaze_stability_std",
    "head_pitch_std",
    "head_yaw_std",
    "face_present_ratio",
    
    # Keyboard cadence metrics (7 features)
    "dwell_time_mean",
    "dwell_time_std",
    "flight_time_mean",
    "flight_time_std",
    "keystrokes_per_min",
    "typing_pause_ratio",
    "typing_burstiness",
    
    # Mouse kinematics & Jitter (6 features)
    "mouse_velocity_mean",
    "mouse_velocity_max",
    "mouse_jitter_mean",
    "mouse_jitter_std",
    "mouse_click_rate",
    "mouse_scroll_rate",
    
    # Dynamics & Trends (4 features)
    "idle_ratio",
    "activity_level",
    "ear_trend",
    "jitter_trend",
]


@dataclass
class WindowBufferEntry:
    """Synchronized pairing of vision and peripheral samples at a given timestamp."""
    timestamp: float
    vision: VisionSample
    peripheral: PeripheralSample


class FeatureBuffer:
    """Maintains sliding window buffer and extracts normalized feature vectors."""

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()
        self.window_sec = self.config.sliding_window_sec
        self._entries: collections.deque[WindowBufferEntry] = collections.deque(maxlen=3600)
        self._scaler = None

    def set_scaler(self, scaler: Any) -> None:
        """Attach a pre-fitted sklearn StandardScaler or MinMaxScaler."""
        self._scaler = scaler

    def add_sample(self, vision: VisionSample, peripheral: PeripheralSample) -> None:
        """Append latest synchronized telemetry pair into the buffer."""
        t_now = max(vision.timestamp, peripheral.timestamp)
        self._entries.append(WindowBufferEntry(
            timestamp=t_now,
            vision=vision,
            peripheral=peripheral
        ))
        self._prune(t_now)

    def _prune(self, current_time: float) -> None:
        """Discard entries older than the sliding window threshold."""
        cutoff = current_time - self.window_sec
        while self._entries and self._entries[0].timestamp < cutoff:
            self._entries.popleft()

    def count(self) -> int:
        """Return number of samples currently residing in the active window."""
        return len(self._entries)

    def is_ready(self) -> bool:
        """Check if buffer has accumulated sufficient entries for reliable feature extraction."""
        return len(self._entries) >= self.config.min_samples_for_inference

    def extract_features(self) -> Tuple[np.ndarray, Dict[str, float]]:
        """Compute statistical moments and trends across the sliding window.
        
        Returns:
            feature_vector: 1D numpy array of shape (29,)
            feature_dict: Dictionary mapping feature names to their scalar values.
        """
        if not self._entries:
            # Fallback zero-filled vector
            zero_dict = {name: 0.0 for name in FEATURE_NAMES}
            return np.zeros(len(FEATURE_NAMES), dtype=np.float32), zero_dict

        # Collect time-series arrays
        timestamps = np.array([e.timestamp for e in self._entries], dtype=np.float64)
        
        # Vision metrics
        faces = [e.vision.face_detected for e in self._entries]
        ears = np.array([e.vision.ear_mean for e in self._entries], dtype=np.float32)
        blink_rates = np.array([e.vision.blink_rate_bpm for e in self._entries], dtype=np.float32)
        perclos_vals = np.array([e.vision.perclos for e in self._entries], dtype=np.float32)
        blink_durations = np.array([e.vision.mean_blink_duration_ms for e in self._entries], dtype=np.float32)
        gaze_stabs = np.array([e.vision.gaze_stability for e in self._entries], dtype=np.float32)
        pitches = np.array([e.vision.head_pitch for e in self._entries], dtype=np.float32)
        yaws = np.array([e.vision.head_yaw for e in self._entries], dtype=np.float32)
        
        # Peripheral metrics
        dwells = np.array([e.peripheral.mean_dwell_time_ms for e in self._entries], dtype=np.float32)
        flights = np.array([e.peripheral.mean_flight_time_ms for e in self._entries], dtype=np.float32)
        kpms = np.array([e.peripheral.keystrokes_per_min for e in self._entries], dtype=np.float32)
        pause_ratios = np.array([e.peripheral.typing_pause_ratio for e in self._entries], dtype=np.float32)
        burstinesses = np.array([e.peripheral.typing_burstiness for e in self._entries], dtype=np.float32)
        
        mouse_vels = np.array([e.peripheral.mean_mouse_velocity_px_s for e in self._entries], dtype=np.float32)
        max_vels = np.array([e.peripheral.max_mouse_velocity_px_s for e in self._entries], dtype=np.float32)
        jitters = np.array([e.peripheral.mouse_jitter_index for e in self._entries], dtype=np.float32)
        click_rates = np.array([e.peripheral.mouse_click_rate_hz for e in self._entries], dtype=np.float32)
        scroll_rates = np.array([e.peripheral.mouse_scroll_rate_hz for e in self._entries], dtype=np.float32)
        idles = [e.peripheral.is_idle for e in self._entries]

        # 1. Ocular statistics
        face_present_ratio = float(np.mean(faces))
        valid_ears = ears[faces] if any(faces) else np.array([0.25], dtype=np.float32)
        
        ear_mean = float(np.mean(valid_ears))
        ear_std = float(np.std(valid_ears))
        ear_min = float(np.min(valid_ears))
        ear_p25 = float(np.percentile(valid_ears, 25))
        
        blink_rate_bpm = float(np.mean(blink_rates))
        perclos = float(np.mean(perclos_vals))
        mean_blink_duration_ms = float(np.mean(blink_durations))
        gaze_stability_mean = float(np.mean(gaze_stabs))
        gaze_stability_std = float(np.std(gaze_stabs))
        head_pitch_std = float(np.std(pitches))
        head_yaw_std = float(np.std(yaws))

        # 2. Keyboard statistics
        dwell_time_mean = float(np.mean(dwells))
        dwell_time_std = float(np.std(dwells))
        flight_time_mean = float(np.mean(flights))
        flight_time_std = float(np.std(flights))
        keystrokes_per_min = float(np.mean(kpms))
        typing_pause_ratio = float(np.mean(pause_ratios))
        typing_burstiness = float(np.mean(burstinesses))

        # 3. Mouse statistics
        mouse_velocity_mean = float(np.mean(mouse_vels))
        mouse_velocity_max = float(np.max(max_vels)) if len(max_vels) > 0 else 0.0
        mouse_jitter_mean = float(np.mean(jitters))
        mouse_jitter_std = float(np.std(jitters))
        mouse_click_rate = float(np.mean(click_rates))
        mouse_scroll_rate = float(np.mean(scroll_rates))

        # 4. Dynamics & Trends
        idle_ratio = float(np.mean(idles))
        activity_level = float(np.clip(1.0 - idle_ratio, 0.0, 1.0))
        
        # Calculate trends (linear slopes over window)
        ear_trend = self._calculate_slope(timestamps, ears)
        jitter_trend = self._calculate_slope(timestamps, jitters)

        feature_dict = {
            "ear_mean": ear_mean,
            "ear_std": ear_std,
            "ear_min": ear_min,
            "ear_p25": ear_p25,
            "blink_rate_bpm": blink_rate_bpm,
            "perclos": perclos,
            "mean_blink_duration_ms": mean_blink_duration_ms,
            "gaze_stability_mean": gaze_stability_mean,
            "gaze_stability_std": gaze_stability_std,
            "head_pitch_std": head_pitch_std,
            "head_yaw_std": head_yaw_std,
            "face_present_ratio": face_present_ratio,
            "dwell_time_mean": dwell_time_mean,
            "dwell_time_std": dwell_time_std,
            "flight_time_mean": flight_time_mean,
            "flight_time_std": flight_time_std,
            "keystrokes_per_min": keystrokes_per_min,
            "typing_pause_ratio": typing_pause_ratio,
            "typing_burstiness": typing_burstiness,
            "mouse_velocity_mean": mouse_velocity_mean,
            "mouse_velocity_max": mouse_velocity_max,
            "mouse_jitter_mean": mouse_jitter_mean,
            "mouse_jitter_std": mouse_jitter_std,
            "mouse_click_rate": mouse_click_rate,
            "mouse_scroll_rate": mouse_scroll_rate,
            "idle_ratio": idle_ratio,
            "activity_level": activity_level,
            "ear_trend": ear_trend,
            "jitter_trend": jitter_trend,
        }

        # Pack into ordered vector matching FEATURE_NAMES
        raw_vector = np.array([feature_dict[name] for name in FEATURE_NAMES], dtype=np.float32)

        # Apply scaler if present
        if self._scaler is not None:
            normalized_vector = self._scaler.transform(raw_vector.reshape(1, -1))[0]
        else:
            normalized_vector = raw_vector

        return normalized_vector, feature_dict

    def _calculate_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute first-order linear regression slope."""
        if len(x) < 3 or (x[-1] - x[0]) < 1e-3:
            return 0.0
        
        # Center x to avoid numerical precision issues
        x_norm = x - x[0]
        try:
            poly = np.polyfit(x_norm, y, deg=1)
            return float(poly[0])
        except Exception:
            return 0.0
