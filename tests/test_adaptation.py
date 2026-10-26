"""Unit tests for Layer 5: Workspace Adaptation Profiles and Actuator."""

import pytest

from aetherflow.adaptation.executor import (
    AdaptationExecutor,
    ExecutionResult,
    kelvin_to_rgb,
)
from aetherflow.adaptation.profiles import AdaptationIntent, ProfileRegistry
from aetherflow.config import (
    AdaptationConfig,
    BreakRecommendation,
    CognitiveState,
    ContextDomain,
    NotificationPolicy,
    UIDensityMode,
)


class TestAdaptationProfiles:
    """Tests for profile matrix mappings and domain constraints."""

    def test_creative_design_preserves_color_fidelity(self):
        # Even during FLOW, Creative Design MUST remain 6500K neutral
        intent_flow = ProfileRegistry.get_intent(CognitiveState.FLOW, ContextDomain.CREATIVE_DESIGN)
        assert intent_flow.target_kelvin == 6500
        assert intent_flow.notification_policy == NotificationPolicy.TOTAL_SILENCE

        intent_norm = ProfileRegistry.get_intent(CognitiveState.NORMAL_FOCUS, ContextDomain.CREATIVE_DESIGN)
        assert intent_norm.target_kelvin == 6500

    def test_code_ide_warmth_and_density(self):
        intent_flow = ProfileRegistry.get_intent(CognitiveState.FLOW, ContextDomain.CODE_IDE)
        assert intent_flow.target_kelvin == 4200
        assert intent_flow.ui_density == UIDensityMode.COMPACT
        assert intent_flow.notification_policy == NotificationPolicy.TOTAL_SILENCE

        intent_fatigue = ProfileRegistry.get_intent(CognitiveState.FATIGUE, ContextDomain.CODE_IDE)
        assert intent_fatigue.target_kelvin == 3000
        assert intent_fatigue.ui_density == UIDensityMode.RELAXED
        assert intent_fatigue.break_recommendation == BreakRecommendation.MANDATORY_BREAK

    def test_writing_text_intent(self):
        intent = ProfileRegistry.get_intent(CognitiveState.FLOW, ContextDomain.WRITING_TEXT)
        assert intent.target_kelvin == 4000
        assert intent.ui_density == UIDensityMode.COMFORT


class TestAdaptationExecutor:
    """Tests for gamma conversion, hysteresis filtering, and safe restoration."""

    def test_kelvin_to_rgb_ranges(self):
        # 6500K daylight should be close to (1.0, 1.0, 1.0)
        r65, g65, b65 = kelvin_to_rgb(6500)
        assert r65 == pytest.approx(1.0, abs=0.01)
        assert g65 == pytest.approx(1.0, abs=0.01)
        assert b65 == pytest.approx(1.0, abs=0.01)

        # 3000K warm candle light: Red is max, Green reduced, Blue heavily attenuated
        r30, g30, b30 = kelvin_to_rgb(3000)
        assert r30 == pytest.approx(1.0)
        assert g30 < 0.75
        assert b30 < 0.50

        # Monotonicity: cooler temperature has more blue
        _, _, b45 = kelvin_to_rgb(4500)
        assert b30 < b45 < b65

    def test_hysteresis_debouncing(self):
        cfg = AdaptationConfig(hysteresis_ticks=3, dry_run=True)
        executor = AdaptationExecutor(cfg)

        # Initial state: 6500K
        # Tick 1: requesting Flow in IDE (4200K) -> should debounce (count=1)
        res1 = executor.dispatch(CognitiveState.FLOW, ContextDomain.CODE_IDE, force_immediate=False)
        assert res1.executed is False
        assert res1.current_kelvin == 6500
        assert res1.hysteresis_count == 1

        # Tick 2: requesting Flow in IDE again -> should debounce (count=2)
        res2 = executor.dispatch(CognitiveState.FLOW, ContextDomain.CODE_IDE, force_immediate=False)
        assert res2.executed is False
        assert res2.hysteresis_count == 2

        # Tick 3: reaches threshold -> should execute transition!
        res3 = executor.dispatch(CognitiveState.FLOW, ContextDomain.CODE_IDE, force_immediate=False)
        assert res3.executed is True
        assert res3.current_kelvin == 4200
        assert res3.ui_density == UIDensityMode.COMPACT

    def test_force_immediate_dispatch(self):
        cfg = AdaptationConfig(hysteresis_ticks=5, dry_run=True)
        executor = AdaptationExecutor(cfg)

        res = executor.dispatch(CognitiveState.FATIGUE, ContextDomain.WRITING_TEXT, force_immediate=True)
        assert res.executed is True
        assert res.current_kelvin == 2900
        assert res.ui_density == UIDensityMode.RELAXED

    def test_adaptation_interval_cooldown(self):
        cfg = AdaptationConfig(hysteresis_ticks=1, min_adaptation_interval_sec=30.0, dry_run=True)
        executor = AdaptationExecutor(cfg)

        # First transition executes
        res1 = executor.dispatch(CognitiveState.FLOW, ContextDomain.CODE_IDE, force_immediate=False)
        assert res1.executed is True
        assert res1.current_kelvin == 4200
        assert res1.cooldown_remaining_sec == pytest.approx(30.0, abs=1.0)

        # Immediate second transition is prevented by the interval cooldown
        res2 = executor.dispatch(CognitiveState.FATIGUE, ContextDomain.CODE_IDE, force_immediate=False)
        assert res2.executed is False
        assert res2.current_kelvin == 4200
        assert res2.cooldown_remaining_sec > 25.0
        assert "Interval locked" in res2.summary

        # Explicit force_immediate overrides cooldown
        res_forced = executor.dispatch(CognitiveState.FATIGUE, ContextDomain.CODE_IDE, force_immediate=True)
        assert res_forced.executed is True
        assert res_forced.current_kelvin == 3000

    def test_creative_design_bypasses_interval_cooldown_for_color_fidelity(self):
        cfg = AdaptationConfig(hysteresis_ticks=1, min_adaptation_interval_sec=30.0, dry_run=True)
        executor = AdaptationExecutor(cfg)

        # Transition into IDE (4200K) locks cooldown for 30s
        res1 = executor.dispatch(CognitiveState.FLOW, ContextDomain.CODE_IDE, force_immediate=False)
        assert res1.executed is True
        assert res1.current_kelvin == 4200

        # Switching to Creative Design (e.g. Figma, Canva) must immediately restore 6500K daylight
        res2 = executor.dispatch(CognitiveState.FLOW, ContextDomain.CREATIVE_DESIGN, force_immediate=False)
        assert res2.executed is True
        assert res2.current_kelvin == 6500

    def test_safe_restore_display(self):
        cfg = AdaptationConfig(dry_run=True)
        executor = AdaptationExecutor(cfg)
        # Should execute cleanly without error
        executor.restore_display()
