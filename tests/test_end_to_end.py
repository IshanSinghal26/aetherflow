"""End-to-end integration test validating the closed-loop orchestration cycle."""

import time

import pytest

from aetherflow.config import AetherFlowConfig, CognitiveState, ContextDomain
from aetherflow.main import AetherFlowEngine


class TestEndToEndEngine:
    """Verifies end-to-end pipeline: Sensing -> Buffering -> Classification -> Context -> Actuation."""

    def test_engine_closed_loop_execution(self):
        config = AetherFlowConfig()
        config.adaptation.dry_run = True
        config.vision.synthetic_mode = True
        config.peripheral.synthetic_mode = True
        config.adaptation.hysteresis_ticks = 1 # Immediate for test speed

        engine = AetherFlowEngine(config)
        engine.start()

        try:
            # Run 5 synchronous steps through the engine components
            for iteration in range(5):
                t_now = time.time()

                # 1. Sense
                vis = engine.vision_tracker.get_latest_sample()
                per = engine.peripheral_tracker.sample(window_sec=10.0)

                # 2. Buffer
                engine.feature_buffer.add_sample(vis, per)
                vec, feat_dict = engine.feature_buffer.extract_features()

                # 3. Classify
                inf = engine.classifier.predict(vec, feat_dict, t_now)
                assert inf.state in CognitiveState

                # 4. Context
                ctx = engine.app_detector.get_active_app()
                assert ctx.domain in ContextDomain

                # 5. Adapt
                res = engine.adaptation_executor.dispatch(inf.state, ctx.domain)
                assert res.current_kelvin >= 2500

                engine.analytics.record_tick(inf.state, 0.1, res.executed)

            assert engine.analytics.total_samples == 5
        finally:
            engine.stop()

    def test_main_run_loop_with_iteration_limit(self):
        config = AetherFlowConfig()
        config.adaptation.dry_run = True
        config.vision.synthetic_mode = True
        config.peripheral.synthetic_mode = True

        engine = AetherFlowEngine(config)
        # Should gracefully finish after 2 iterations
        engine.run_loop(interval_sec=0.01, max_iterations=2)
        assert engine.analytics.total_samples == 2
