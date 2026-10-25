"""Unit tests for Layer 1: Passive Ingestion (Vision and Peripheral Trackers)."""

import time
from dataclasses import dataclass

import numpy as np
import pytest

from aetherflow.sensing.peripheral_tracker import (
    PeripheralConfig,
    PeripheralSample,
    PeripheralTracker,
    np_mean,
    np_std,
)
from aetherflow.sensing.vision_tracker import (
    VisionConfig,
    VisionSample,
    VisionTracker,
    calculate_ear,
    euclidean_dist,
)


@dataclass
class MockLandmark:
    x: float
    y: float


class TestVisionTracker:
    """Tests for Eye Aspect Ratio, blink state tracking, and vision metrics."""

    def test_euclidean_distance(self):
        p1 = (0.0, 0.0)
        p2 = (3.0, 4.0)
        assert euclidean_dist(p1, p2) == pytest.approx(5.0)

    def test_calculate_ear_open_vs_closed(self):
        # Open eye: height is ~30% of width
        # corners: p1=(0.1, 0.5), p4=(0.3, 0.5) => d_h = 0.2
        # vertical pairs: p2=(0.15, 0.47), p6=(0.15, 0.53) => d_v1 = 0.06
        #                 p3=(0.25, 0.47), p5=(0.25, 0.53) => d_v2 = 0.06
        # EAR = (0.06 + 0.06) / (2 * 0.2) = 0.12 / 0.4 = 0.30
        open_landmarks = {
            1: MockLandmark(0.1, 0.5),
            2: MockLandmark(0.15, 0.47),
            3: MockLandmark(0.25, 0.47),
            4: MockLandmark(0.3, 0.5),
            5: MockLandmark(0.25, 0.53),
            6: MockLandmark(0.15, 0.53),
        }
        indices = [1, 2, 3, 4, 5, 6]
        ear_open = calculate_ear(open_landmarks, indices, 1000, 1000)
        assert ear_open == pytest.approx(0.30, abs=0.01)

        # Closed eye: height is nearly 0
        closed_landmarks = {
            1: MockLandmark(0.1, 0.5),
            2: MockLandmark(0.15, 0.501),
            3: MockLandmark(0.25, 0.501),
            4: MockLandmark(0.3, 0.5),
            5: MockLandmark(0.25, 0.502),
            6: MockLandmark(0.15, 0.502),
        }
        ear_closed = calculate_ear(closed_landmarks, indices, 1000, 1000)
        assert ear_closed < 0.05
        assert ear_open > ear_closed

    def test_vision_tracker_synthetic_mode(self):
        cfg = VisionConfig(synthetic_mode=True)
        tracker = VisionTracker(cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        sample = tracker.process_frame(frame, timestamp=time.time())
        
        assert isinstance(sample, VisionSample)
        assert sample.face_detected is True
        assert sample.ear_mean > 0.10
        assert 0.0 <= sample.gaze_stability <= 1.0


class TestPeripheralTracker:
    """Tests for keyboard cadence, dwell/flight times, and mouse jitter."""

    def test_np_math_helpers(self):
        vals = [10.0, 20.0, 30.0]
        assert np_mean(vals) == pytest.approx(20.0)
        assert np_std(vals) == pytest.approx(8.1649, abs=0.01)
        assert np_mean([]) == 0.0
        assert np_std([5.0]) == 0.0

    def test_keystroke_dwell_and_flight_injection(self):
        cfg = PeripheralConfig(synthetic_mode=False)
        tracker = PeripheralTracker(cfg)

        # Inject 5 synthetic keystrokes
        for _ in range(5):
            tracker.inject_synthetic_keystroke(dwell_ms=95.0, flight_ms=180.0)

        sample = tracker.sample(window_sec=60.0)
        assert isinstance(sample, PeripheralSample)
        assert sample.mean_dwell_time_ms == pytest.approx(95.0, abs=1.0)
        assert sample.mean_flight_time_ms == pytest.approx(180.0, abs=1.0)
        assert sample.keystrokes_per_min > 0.0
        assert sample.is_idle is False

    def test_mouse_jitter_calculation(self):
        cfg = PeripheralConfig(synthetic_mode=False)
        tracker = PeripheralTracker(cfg)

        # Smooth linear path
        t0 = time.time()
        for i in range(10):
            tracker.inject_synthetic_mouse(x=float(i * 10), y=float(i * 10))

        sample = tracker.sample(window_sec=60.0)
        assert 0.0 <= sample.mouse_jitter_index <= 1.0

    def test_privacy_preservation(self):
        cfg = PeripheralConfig(synthetic_mode=False)
        tracker = PeripheralTracker(cfg)
        
        # Verify internal structures contain NO string keys or text fields
        tracker.inject_synthetic_keystroke(100.0, 200.0)
        sample = tracker.sample()
        sample_dict = sample.__dict__
        for k, v in sample_dict.items():
            assert not isinstance(v, str) or k in ("is_idle",)
