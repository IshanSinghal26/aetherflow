"""Unit tests for Layer 2: Feature Buffer sliding window and normalizer."""

import time

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from aetherflow.config import EngineConfig
from aetherflow.engine.feature_buffer import FEATURE_NAMES, FeatureBuffer
from aetherflow.sensing.peripheral_tracker import PeripheralSample
from aetherflow.sensing.vision_tracker import VisionSample


class TestFeatureBuffer:
    """Tests for sliding window aggregation, moments, and normalization."""

    def test_feature_buffer_empty_fallback(self):
        buf = FeatureBuffer()
        vec, d = buf.extract_features()
        assert len(vec) == len(FEATURE_NAMES)
        assert len(d) == len(FEATURE_NAMES)
        assert np.all(vec == 0.0)

    def test_feature_buffer_window_pruning(self):
        cfg = EngineConfig(sliding_window_sec=10.0, min_samples_for_inference=2)
        buf = FeatureBuffer(cfg)

        t0 = time.time()
        # Add 3 old samples (> 10s ago)
        for i in range(3):
            vis = VisionSample(timestamp=t0 - 20.0 + i, face_detected=True, ear_mean=0.30)
            per = PeripheralSample(timestamp=t0 - 20.0 + i, is_idle=False)
            buf.add_sample(vis, per)

        # Add 2 recent samples
        for i in range(2):
            vis = VisionSample(timestamp=t0 + i, face_detected=True, ear_mean=0.32)
            per = PeripheralSample(timestamp=t0 + i, is_idle=False)
            buf.add_sample(vis, per)

        # Buffer should have pruned the 3 old samples
        assert buf.count() == 2
        assert buf.is_ready() is True

    def test_feature_extraction_statistics(self):
        buf = FeatureBuffer()
        t0 = time.time()
        
        # Add samples with varying EAR: 0.20, 0.25, 0.30
        for ear_val in [0.20, 0.25, 0.30]:
            vis = VisionSample(
                timestamp=t0,
                face_detected=True,
                ear_mean=ear_val,
                blink_rate_bpm=15.0,
                gaze_stability=0.90
            )
            per = PeripheralSample(
                timestamp=t0,
                mean_dwell_time_ms=100.0,
                mean_flight_time_ms=180.0,
                mouse_jitter_index=0.08,
                is_idle=False
            )
            buf.add_sample(vis, per)
            t0 += 1.0

        vec, d = buf.extract_features()
        assert len(vec) == 29
        assert d["ear_mean"] == pytest.approx(0.25, abs=0.01)
        assert d["ear_min"] == pytest.approx(0.20, abs=0.01)
        assert d["dwell_time_mean"] == pytest.approx(100.0, abs=1.0)
        assert d["gaze_stability_mean"] == pytest.approx(0.90, abs=0.01)

    def test_feature_normalization_with_scaler(self):
        buf = FeatureBuffer()
        # Mock scaler
        scaler = StandardScaler()
        X_mock = np.random.randn(20, len(FEATURE_NAMES)).astype(np.float32)
        scaler.fit(X_mock)
        buf.set_scaler(scaler)

        t0 = time.time()
        vis = VisionSample(timestamp=t0, face_detected=True, ear_mean=0.28)
        per = PeripheralSample(timestamp=t0, is_idle=False)
        buf.add_sample(vis, per)

        vec, d = buf.extract_features()
        assert vec.shape == (len(FEATURE_NAMES),)
        assert not np.isnan(vec).any()
