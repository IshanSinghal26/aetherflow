"""Adaptive environment profiles and action mapping matrix.

Maps (CognitiveState x ContextDomain) pairs to concrete workspace adaptation intents:
Screen Warmth (Kelvin), UI Density, Notification Policy, and Ergonomic Prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from aetherflow.config import (
    BreakRecommendation,
    CognitiveState,
    ContextDomain,
    NotificationPolicy,
    UIDensityMode,
)


@dataclass
class AdaptationIntent:
    """Declared goal state for workspace environment adaptations."""
    target_kelvin: int
    ui_density: UIDensityMode
    notification_policy: NotificationPolicy
    break_recommendation: BreakRecommendation
    action_summary: str


class ProfileRegistry:
    """Profile matrix mapping (CognitiveState, ContextDomain) -> AdaptationIntent."""

    # Explicit policy mappings
    _MATRIX: Dict[Tuple[CognitiveState, ContextDomain], AdaptationIntent] = {
        # -------------------------------------------------------------
        # 1. CODE_IDE
        # -------------------------------------------------------------
        (CognitiveState.FLOW, ContextDomain.CODE_IDE): AdaptationIntent(
            target_kelvin=4200,
            ui_density=UIDensityMode.COMPACT,
            notification_policy=NotificationPolicy.TOTAL_SILENCE,
            break_recommendation=BreakRecommendation.NONE,
            action_summary="Deep code immersion: compact layout, 4200K focus warmth, total notification suppression."
        ),
        (CognitiveState.NORMAL_FOCUS, ContextDomain.CODE_IDE): AdaptationIntent(
            target_kelvin=5000,
            ui_density=UIDensityMode.STANDARD,
            notification_policy=NotificationPolicy.PRIORITY_ONLY,
            break_recommendation=BreakRecommendation.NONE,
            action_summary="Balanced IDE workspace: standard layout, neutral 5000K warmth, priority notifications only."
        ),
        (CognitiveState.COGNITIVE_OVERLOAD, ContextDomain.CODE_IDE): AdaptationIntent(
            target_kelvin=3500,
            ui_density=UIDensityMode.COMFORT,
            notification_policy=NotificationPolicy.PRIORITY_ONLY,
            break_recommendation=BreakRecommendation.GENTLE_PROMPT,
            action_summary="Syntax/debug overload: comfort spacing, soothing 3500K warmth, gentle pacing prompt."
        ),
        (CognitiveState.FATIGUE, ContextDomain.CODE_IDE): AdaptationIntent(
            target_kelvin=3000,
            ui_density=UIDensityMode.RELAXED,
            notification_policy=NotificationPolicy.ALLOW_ALL,
            break_recommendation=BreakRecommendation.MANDATORY_BREAK,
            action_summary="Visual strain detected in IDE: candle 3000K warmth, relaxed layout, mandatory rest alert."
        ),
        (CognitiveState.IDLE, ContextDomain.CODE_IDE): AdaptationIntent(
            target_kelvin=6500,
            ui_density=UIDensityMode.STANDARD,
            notification_policy=NotificationPolicy.ALLOW_ALL,
            break_recommendation=BreakRecommendation.NONE,
            action_summary="IDE idle standby: standard 6500K daylight, notifications allowed."
        ),

        # -------------------------------------------------------------
        # 2. CREATIVE_DESIGN (Preserve color fidelity sacredly)
        # -------------------------------------------------------------
        (CognitiveState.FLOW, ContextDomain.CREATIVE_DESIGN): AdaptationIntent(
            target_kelvin=6500, # Color fidelity MUST not be compromised
            ui_density=UIDensityMode.STANDARD,
            notification_policy=NotificationPolicy.TOTAL_SILENCE,
            break_recommendation=BreakRecommendation.NONE,
            action_summary="Creative flow: 6500K true color fidelity preserved, total notification silence."
        ),
        (CognitiveState.NORMAL_FOCUS, ContextDomain.CREATIVE_DESIGN): AdaptationIntent(
            target_kelvin=6500,
            ui_density=UIDensityMode.STANDARD,
            notification_policy=NotificationPolicy.PRIORITY_ONLY,
            break_recommendation=BreakRecommendation.NONE,
            action_summary="Creative design active: 6500K true color fidelity preserved, priority notifications."
        ),
        (CognitiveState.COGNITIVE_OVERLOAD, ContextDomain.CREATIVE_DESIGN): AdaptationIntent(
            target_kelvin=6000, # Very subtle softening, preserving palette
            ui_density=UIDensityMode.COMFORT,
            notification_policy=NotificationPolicy.PRIORITY_ONLY,
            break_recommendation=BreakRecommendation.GENTLE_PROMPT,
            action_summary="Design overload: subtle 6000K softening, enlarged UI spacing, gentle prompt."
        ),
        (CognitiveState.FATIGUE, ContextDomain.CREATIVE_DESIGN): AdaptationIntent(
            target_kelvin=5500,
            ui_density=UIDensityMode.RELAXED,
            notification_policy=NotificationPolicy.ALLOW_ALL,
            break_recommendation=BreakRecommendation.MANDATORY_BREAK,
            action_summary="Designer eye fatigue: 5500K gentle warmth, relaxed spacing, mandatory break alert."
        ),
        (CognitiveState.IDLE, ContextDomain.CREATIVE_DESIGN): AdaptationIntent(
            target_kelvin=6500,
            ui_density=UIDensityMode.STANDARD,
            notification_policy=NotificationPolicy.ALLOW_ALL,
            break_recommendation=BreakRecommendation.NONE,
            action_summary="Creative app idle: 6500K neutral daylight."
        ),

        # -------------------------------------------------------------
        # 3. WRITING_TEXT
        # -------------------------------------------------------------
        (CognitiveState.FLOW, ContextDomain.WRITING_TEXT): AdaptationIntent(
            target_kelvin=4000,
            ui_density=UIDensityMode.COMFORT,
            notification_policy=NotificationPolicy.TOTAL_SILENCE,
            break_recommendation=BreakRecommendation.NONE,
            action_summary="Writing flow: parchment 4000K warmth, high-legibility comfort layout, zero distractions."
        ),
        (CognitiveState.NORMAL_FOCUS, ContextDomain.WRITING_TEXT): AdaptationIntent(
            target_kelvin=4800,
            ui_density=UIDensityMode.STANDARD,
            notification_policy=NotificationPolicy.PRIORITY_ONLY,
            break_recommendation=BreakRecommendation.NONE,
            action_summary="Active writing: 4800K paper warmth, standard typography layout."
        ),
        (CognitiveState.COGNITIVE_OVERLOAD, ContextDomain.WRITING_TEXT): AdaptationIntent(
            target_kelvin=3400,
            ui_density=UIDensityMode.COMFORT,
            notification_policy=NotificationPolicy.PRIORITY_ONLY,
            break_recommendation=BreakRecommendation.GENTLE_PROMPT,
            action_summary="Writer block/overload: amber 3400K warmth, comfort typography, gentle pause prompt."
        ),
        (CognitiveState.FATIGUE, ContextDomain.WRITING_TEXT): AdaptationIntent(
            target_kelvin=2900,
            ui_density=UIDensityMode.RELAXED,
            notification_policy=NotificationPolicy.ALLOW_ALL,
            break_recommendation=BreakRecommendation.MANDATORY_BREAK,
            action_summary="Writing fatigue: warm candle 2900K, enlarged text layout, break recommendation."
        ),
        (CognitiveState.IDLE, ContextDomain.WRITING_TEXT): AdaptationIntent(
            target_kelvin=6500,
            ui_density=UIDensityMode.STANDARD,
            notification_policy=NotificationPolicy.ALLOW_ALL,
            break_recommendation=BreakRecommendation.NONE,
            action_summary="Writing idle: neutral baseline 6500K."
        ),

        # -------------------------------------------------------------
        # 4. MEDIA_VIDEO
        # -------------------------------------------------------------
        (CognitiveState.FLOW, ContextDomain.MEDIA_VIDEO): AdaptationIntent(
            target_kelvin=5800,
            ui_density=UIDensityMode.STANDARD,
            notification_policy=NotificationPolicy.TOTAL_SILENCE,
            break_recommendation=BreakRecommendation.NONE,
            action_summary="Media immersion: cinema 5800K tone, total notification silence."
        ),
        (CognitiveState.NORMAL_FOCUS, ContextDomain.MEDIA_VIDEO): AdaptationIntent(
            target_kelvin=6500,
            ui_density=UIDensityMode.STANDARD,
            notification_policy=NotificationPolicy.PRIORITY_ONLY,
            break_recommendation=BreakRecommendation.NONE,
            action_summary="Media playback: standard 6500K."
        ),
        (CognitiveState.COGNITIVE_OVERLOAD, ContextDomain.MEDIA_VIDEO): AdaptationIntent(
            target_kelvin=4500,
            ui_density=UIDensityMode.COMFORT,
            notification_policy=NotificationPolicy.PRIORITY_ONLY,
            break_recommendation=BreakRecommendation.GENTLE_PROMPT,
            action_summary="Media saturation: gentle 4500K softening, comfort layout."
        ),
        (CognitiveState.FATIGUE, ContextDomain.MEDIA_VIDEO): AdaptationIntent(
            target_kelvin=3200,
            ui_density=UIDensityMode.RELAXED,
            notification_policy=NotificationPolicy.ALLOW_ALL,
            break_recommendation=BreakRecommendation.GENTLE_PROMPT,
            action_summary="Late media fatigue: soothing 3200K amber, relaxed layout."
        ),
        (CognitiveState.IDLE, ContextDomain.MEDIA_VIDEO): AdaptationIntent(
            target_kelvin=6500,
            ui_density=UIDensityMode.STANDARD,
            notification_policy=NotificationPolicy.ALLOW_ALL,
            break_recommendation=BreakRecommendation.NONE,
            action_summary="Media idle: 6500K neutral."
        ),

        # -------------------------------------------------------------
        # 5. RESEARCH_BROWSING & GENERAL
        # -------------------------------------------------------------
        (CognitiveState.FLOW, ContextDomain.RESEARCH_BROWSING): AdaptationIntent(
            target_kelvin=4500,
            ui_density=UIDensityMode.COMPACT,
            notification_policy=NotificationPolicy.TOTAL_SILENCE,
            break_recommendation=BreakRecommendation.NONE,
            action_summary="Research flow: 4500K paper warmth, compact layout, silence."
        ),
        (CognitiveState.COGNITIVE_OVERLOAD, ContextDomain.RESEARCH_BROWSING): AdaptationIntent(
            target_kelvin=3500,
            ui_density=UIDensityMode.COMFORT,
            notification_policy=NotificationPolicy.PRIORITY_ONLY,
            break_recommendation=BreakRecommendation.GENTLE_PROMPT,
            action_summary="Tab/information overload: soothing 3500K warmth, comfort layout, pacing prompt."
        ),
        (CognitiveState.FATIGUE, ContextDomain.RESEARCH_BROWSING): AdaptationIntent(
            target_kelvin=3000,
            ui_density=UIDensityMode.RELAXED,
            notification_policy=NotificationPolicy.ALLOW_ALL,
            break_recommendation=BreakRecommendation.MANDATORY_BREAK,
            action_summary="Browsing fatigue: candle 3000K warmth, relaxed layout, rest recommendation."
        ),
    }

    # Generic fallback intent for unspecified pairs
    _FALLBACK_MAP: Dict[CognitiveState, AdaptationIntent] = {
        CognitiveState.FLOW: AdaptationIntent(
            target_kelvin=4500,
            ui_density=UIDensityMode.COMPACT,
            notification_policy=NotificationPolicy.TOTAL_SILENCE,
            break_recommendation=BreakRecommendation.NONE,
            action_summary="Generic flow state: 4500K warmth, compact layout, silence."
        ),
        CognitiveState.NORMAL_FOCUS: AdaptationIntent(
            target_kelvin=5500,
            ui_density=UIDensityMode.STANDARD,
            notification_policy=NotificationPolicy.PRIORITY_ONLY,
            break_recommendation=BreakRecommendation.NONE,
            action_summary="Standard workspace: 5500K warmth, standard UI, priority alerts."
        ),
        CognitiveState.COGNITIVE_OVERLOAD: AdaptationIntent(
            target_kelvin=3600,
            ui_density=UIDensityMode.COMFORT,
            notification_policy=NotificationPolicy.PRIORITY_ONLY,
            break_recommendation=BreakRecommendation.GENTLE_PROMPT,
            action_summary="Cognitive overload: warm 3600K soothing tone, comfort layout."
        ),
        CognitiveState.FATIGUE: AdaptationIntent(
            target_kelvin=3000,
            ui_density=UIDensityMode.RELAXED,
            notification_policy=NotificationPolicy.ALLOW_ALL,
            break_recommendation=BreakRecommendation.MANDATORY_BREAK,
            action_summary="Fatigue state: 3000K candle warmth, relaxed layout, rest alert."
        ),
        CognitiveState.IDLE: AdaptationIntent(
            target_kelvin=6500,
            ui_density=UIDensityMode.STANDARD,
            notification_policy=NotificationPolicy.ALLOW_ALL,
            break_recommendation=BreakRecommendation.NONE,
            action_summary="Idle standby: 6500K baseline daylight."
        ),
    }

    @classmethod
    def get_intent(cls, state: CognitiveState, domain: ContextDomain) -> AdaptationIntent:
        """Query target adaptation profile for given cognitive state and creative domain."""
        pair = (state, domain)
        if pair in cls._MATRIX:
            return cls._MATRIX[pair]
        return cls._FALLBACK_MAP.get(state, cls._FALLBACK_MAP[CognitiveState.NORMAL_FOCUS])
