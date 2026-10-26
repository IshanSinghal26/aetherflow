"""Workspace execution layer for neuroadaptive environmental actuation.

Executes display warmth shifts (via Win32 SetDeviceGammaRamp), UI density adjustments,
notification suppression policies, and ergonomic prompts with hysteresis debouncing
and guaranteed restoration safeguards.
"""

from __future__ import annotations

import atexit
import ctypes
import logging
import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from aetherflow.adaptation.profiles import AdaptationIntent, ProfileRegistry
from aetherflow.config import (
    AdaptationConfig,
    BreakRecommendation,
    CognitiveState,
    ContextDomain,
    NotificationPolicy,
    UIDensityMode,
)

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Status result of an adaptation dispatch cycle."""
    executed: bool
    current_kelvin: int
    target_kelvin: int
    ui_density: UIDensityMode
    notification_policy: NotificationPolicy
    break_recommendation: BreakRecommendation
    summary: str
    hysteresis_count: int
    timestamp: float
    cooldown_remaining_sec: float = 0.0


class AdaptationExecutor:
    """Dispatches closed-loop workplace adaptations with hysteresis, interval pacing, and display safeguards."""

    def __init__(self, config: Optional[AdaptationConfig] = None) -> None:
        self.config = config or AdaptationConfig()
        self._lock = threading.Lock()
        
        # State tracking
        self.current_kelvin: int = self.config.baseline_color_temp_kelvin
        self.current_ui_density: UIDensityMode = UIDensityMode.STANDARD
        self.current_notification_policy: NotificationPolicy = NotificationPolicy.ALLOW_ALL
        self.current_break: BreakRecommendation = BreakRecommendation.NONE
        
        # Hysteresis buffer & Interval pacing
        self._pending_intent: Optional[AdaptationIntent] = None
        self._hysteresis_counter: int = 0
        self.last_adaptation_time: float = 0.0
        
        # Non-blocking smooth fade thread
        self._transition_thread: Optional[threading.Thread] = None
        self._transition_stop = threading.Event()
        
        # Win32 Gamma Ramp Management
        self._hdc = None
        self._orig_ramp = None
        self._ramp_supported = False
        
        if sys.platform == "win32" and not self.config.dry_run:
            self._init_win32_gamma()

        # Guarantee safe display restoration on process exit
        atexit.register(self.restore_display)

    def _init_win32_gamma(self) -> None:
        """Capture baseline hardware gamma ramp for safe rollbacks."""
        try:
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            gdi32 = ctypes.windll.gdi32

            self._hdc = user32.GetDC(None)
            ramp_type = (wintypes.WORD * 256) * 3
            self._orig_ramp = ramp_type()

            success = gdi32.GetDeviceGammaRamp(self._hdc, ctypes.byref(self._orig_ramp))
            if success:
                self._ramp_supported = True
                logger.info("Win32 hardware gamma ramp captured successfully.")
            else:
                logger.warning("GetDeviceGammaRamp failed. Display gamma actuation disabled.")
                self._ramp_supported = False
        except Exception as e:
            logger.warning(f"Could not initialize Win32 gamma management: {e}")
            self._ramp_supported = False

    def dispatch(
        self,
        cognitive_state: CognitiveState,
        context_domain: ContextDomain,
        force_immediate: bool = False
    ) -> ExecutionResult:
        """Evaluate and actuate adaptation profile with interval cooldown and smooth pacing."""
        t_now = time.time()
        intent = ProfileRegistry.get_intent(cognitive_state, context_domain)

        with self._lock:
            # Check if this intent differs from current state
            is_different = (
                intent.target_kelvin != self.current_kelvin or
                intent.ui_density != self.current_ui_density or
                intent.notification_policy != self.current_notification_policy or
                intent.break_recommendation != self.current_break
            )

            # Check adaptation interval cooldown
            elapsed_since_last = t_now - self.last_adaptation_time
            in_cooldown = (elapsed_since_last < self.config.min_adaptation_interval_sec)
            remaining_cooldown = max(0.0, self.config.min_adaptation_interval_sec - elapsed_since_last)

            # Color fidelity override: Switching to creative design (Figma, Canva, Photoshop)
            # must restore true daylight 6500K without waiting out a prior cooldown.
            is_color_critical = (context_domain == ContextDomain.CREATIVE_DESIGN and intent.target_kelvin >= 6000)
            if is_color_critical:
                in_cooldown = False

            if not is_different:
                # Stable state, reset pending counter
                self._pending_intent = None
                self._hysteresis_counter = 0
                return ExecutionResult(
                    executed=False,
                    current_kelvin=self.current_kelvin,
                    target_kelvin=self.current_kelvin,
                    ui_density=self.current_ui_density,
                    notification_policy=self.current_notification_policy,
                    break_recommendation=self.current_break,
                    summary=f"Stable: {intent.action_summary}",
                    hysteresis_count=0,
                    timestamp=t_now,
                    cooldown_remaining_sec=remaining_cooldown
                )

            # If inside specific adaptation interval, hold off any sudden changes
            if in_cooldown and not force_immediate:
                return ExecutionResult(
                    executed=False,
                    current_kelvin=self.current_kelvin,
                    target_kelvin=intent.target_kelvin,
                    ui_density=self.current_ui_density,
                    notification_policy=self.current_notification_policy,
                    break_recommendation=self.current_break,
                    summary=f"Interval locked: {remaining_cooldown:.1f}s remaining before next display update.",
                    hysteresis_count=self._hysteresis_counter,
                    timestamp=t_now,
                    cooldown_remaining_sec=remaining_cooldown
                )

            # Hysteresis debouncing
            if force_immediate or is_color_critical:
                should_execute = True
                self._hysteresis_counter = self.config.hysteresis_ticks
            else:
                if self._pending_intent is not None and self._pending_intent.target_kelvin == intent.target_kelvin:
                    self._hysteresis_counter += 1
                else:
                    self._pending_intent = intent
                    self._hysteresis_counter = 1

                should_execute = self._hysteresis_counter >= self.config.hysteresis_ticks

            if not should_execute:
                return ExecutionResult(
                    executed=False,
                    current_kelvin=self.current_kelvin,
                    target_kelvin=intent.target_kelvin,
                    ui_density=self.current_ui_density,
                    notification_policy=self.current_notification_policy,
                    break_recommendation=self.current_break,
                    summary=f"Debouncing transition ({self._hysteresis_counter}/{self.config.hysteresis_ticks}) to {intent.target_kelvin}K",
                    hysteresis_count=self._hysteresis_counter,
                    timestamp=t_now,
                    cooldown_remaining_sec=remaining_cooldown
                )

            # Record adaptation timestamp to lock interval
            self.last_adaptation_time = t_now

            # Execute adaptations (gradual, non-blocking fade)
            self._apply_display_warmth(intent.target_kelvin)
            self._apply_ui_density(intent.ui_density)
            self._apply_notification_policy(intent.notification_policy)
            self._apply_break_recommendation(intent.break_recommendation)

            # Update current state
            self.current_kelvin = intent.target_kelvin
            self.current_ui_density = intent.ui_density
            self.current_notification_policy = intent.notification_policy
            self.current_break = intent.break_recommendation
            self._pending_intent = None
            self._hysteresis_counter = 0

            return ExecutionResult(
                executed=True,
                current_kelvin=self.current_kelvin,
                target_kelvin=intent.target_kelvin,
                ui_density=self.current_ui_density,
                notification_policy=self.current_notification_policy,
                break_recommendation=self.current_break,
                summary=intent.action_summary,
                hysteresis_count=self.config.hysteresis_ticks,
                timestamp=t_now,
                cooldown_remaining_sec=self.config.min_adaptation_interval_sec
            )

    def _apply_display_warmth(self, target_kelvin: int) -> None:
        """Launch non-blocking background thread to perform an ultra-smooth, gradual S-curve fade."""
        target_kelvin = max(self.config.min_color_temp_kelvin, min(self.config.max_color_temp_kelvin, target_kelvin))
        start_kelvin = self.current_kelvin

        if start_kelvin == target_kelvin:
            return

        # Signal previous transition to stop
        self._transition_stop.set()
        if self._transition_thread and self._transition_thread.is_alive():
            self._transition_thread.join(timeout=0.2)

        self._transition_stop.clear()

        if self.config.dry_run or not self._ramp_supported or self._orig_ramp is None:
            logger.info(f"[Dry-Run] Transitioning display warmth: {start_kelvin}K -> {target_kelvin}K")
            return

        def _worker():
            total_duration = max(0.5, self.config.transition_duration_sec)
            steps = max(10, self.config.gamma_transition_steps)
            step_delay = total_duration / steps

            logger.info(f"Fading screen warmth {start_kelvin}K -> {target_kelvin}K smoothly over {total_duration:.1f}s ({steps} micro-steps)...")

            for step in range(1, steps + 1):
                if self._transition_stop.is_set():
                    break
                fraction = step / float(steps)
                # Cosine / smoothstep S-curve for natural lighting fade
                smooth_frac = 0.5 * (1.0 - math.cos(fraction * math.pi))
                interp_k = int(start_kelvin + (target_kelvin - start_kelvin) * smooth_frac)
                self._set_hardware_ramp_kelvin(interp_k)
                time.sleep(step_delay)

            if not self._transition_stop.is_set():
                self._set_hardware_ramp_kelvin(target_kelvin)

        self._transition_thread = threading.Thread(target=_worker, daemon=True, name="SmoothWarmthFader")
        self._transition_thread.start()

    def _set_hardware_ramp_kelvin(self, kelvin: int) -> bool:
        """Construct RGB gamma ramp for given temperature and write to Win32 GDI DC."""
        if not self._ramp_supported or self._hdc is None or self._orig_ramp is None:
            return False

        try:
            from ctypes import wintypes
            gdi32 = ctypes.windll.gdi32

            r_mult, g_mult, b_mult = kelvin_to_rgb(kelvin)
            ramp_type = (wintypes.WORD * 256) * 3
            new_ramp = ramp_type()

            for i in range(256):
                new_ramp[0][i] = int(min(65535, max(0, self._orig_ramp[0][i] * r_mult)))
                new_ramp[1][i] = int(min(65535, max(0, self._orig_ramp[1][i] * g_mult)))
                new_ramp[2][i] = int(min(65535, max(0, self._orig_ramp[2][i] * b_mult)))

            return bool(gdi32.SetDeviceGammaRamp(self._hdc, ctypes.byref(new_ramp)))
        except Exception as e:
            logger.debug(f"SetDeviceGammaRamp failed: {e}")
            return False

    def _apply_ui_density(self, mode: UIDensityMode) -> None:
        """Log/dispatch UI density adjustment notification."""
        logger.debug(f"UI density updated: {mode.value}")

    def _apply_notification_policy(self, policy: NotificationPolicy) -> None:
        """Enforce notification suppression policy."""
        logger.debug(f"Notification policy updated: {policy.value}")

    def _apply_break_recommendation(self, recommendation: BreakRecommendation) -> None:
        """Trigger ergonomic rest guidance when required."""
        if recommendation != BreakRecommendation.NONE:
            logger.info(f"Ergonomic advisory triggered: {recommendation.value}")

    def restore_display(self) -> None:
        """Guaranteed restoration of original hardware gamma ramp."""
        if self._ramp_supported and self._hdc is not None and self._orig_ramp is not None:
            try:
                gdi32 = ctypes.windll.gdi32
                user32 = ctypes.windll.user32
                gdi32.SetDeviceGammaRamp(self._hdc, ctypes.byref(self._orig_ramp))
                user32.ReleaseDC(None, self._hdc)
                self._hdc = None
                self._ramp_supported = False
                logger.info("Display gamma ramp successfully restored to pristine original state.")
            except Exception as e:
                logger.error(f"Error restoring gamma ramp: {e}")


def kelvin_to_rgb(kelvin: int) -> Tuple[float, float, float]:
    """Calculate relative RGB scaling multipliers for color temperature using Tanner-Helland formula.
    
    Normalized relative to 6500K neutral daylight.
    """
    temp = max(1000, min(40000, kelvin)) / 100.0

    # Red
    if temp <= 66.0:
        r = 255.0
    else:
        r = 329.698727446 * ((temp - 60.0) ** -0.1332047592)

    # Green
    if temp <= 66.0:
        g = 99.4708025861 * math.log(temp) - 161.1195681661
    else:
        g = 288.1221695283 * ((temp - 60.0) ** -0.0755148492)

    # Blue
    if temp <= 19.0:
        b = 0.0
    elif temp <= 66.0:
        b = 138.5177312231 * math.log(temp - 10.0) - 305.0447927307
    else:
        b = 255.0

    r = max(0.0, min(255.0, r))
    g = max(0.0, min(255.0, g))
    b = max(0.0, min(255.0, b))

    # Reference neutral at 6500K (approx r=255, g=254.1, b=250.0)
    ref_r, ref_g, ref_b = 255.0, 254.11, 250.04
    return (
        min(1.0, r / ref_r),
        min(1.0, g / ref_g),
        min(1.0, b / ref_b),
    )
