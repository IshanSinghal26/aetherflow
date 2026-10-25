"""AetherFlow - Local Closed-Loop Neuroadaptive Workspace Engine.

Passively ingests real-time ocular biometrics and peripheral interaction kinematics,
classifies user cognitive state using trained local models, detects foreground
creative context, and dynamically adapts desktop environment (screen warmth,
UI density, notification suppression) in a closed feedback loop.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

# Ensure standard output doesn't crash on Windows cp1252 consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from aetherflow.adaptation.executor import AdaptationExecutor
from aetherflow.config import (
    AetherFlowConfig,
    CognitiveState,
    ContextDomain,
    DATA_DIR,
)
from aetherflow.context.app_detector import AppDetector
from aetherflow.engine.classifier import CognitiveClassifier
from aetherflow.engine.feature_buffer import FeatureBuffer
from aetherflow.sensing.peripheral_tracker import PeripheralTracker
from aetherflow.sensing.vision_tracker import VisionTracker
from aetherflow.simulation.train_synthetic import train_and_serialize

logger = logging.getLogger("aetherflow")


@dataclass
class SessionAnalytics:
    """Tracks time spent across cognitive states during a session."""
    start_time: float
    state_durations: dict[str, float]
    adaptations_count: int = 0
    total_samples: int = 0

    @classmethod
    def create(cls) -> SessionAnalytics:
        return cls(
            start_time=time.time(),
            state_durations={s: 0.0 for s in CognitiveState.list()},
            adaptations_count=0,
            total_samples=0
        )

    def record_tick(self, state: CognitiveState, dt: float, adapted: bool) -> None:
        self.state_durations[state.value] = self.state_durations.get(state.value, 0.0) + dt
        self.total_samples += 1
        if adapted:
            self.adaptations_count += 1


class AetherFlowEngine:
    """Master orchestrator integrating sensing, ML inference, context detection, and actuation."""

    def __init__(self, config: Optional[AetherFlowConfig] = None) -> None:
        self.config = config or AetherFlowConfig()
        self.config.ensure_directories()
        
        self.running = False
        self.analytics = SessionAnalytics.create()

        logger.info("Initializing AetherFlow layers...")

        # Layer 1: Passive Sensing
        self.vision_tracker = VisionTracker(self.config.vision)
        self.peripheral_tracker = PeripheralTracker(self.config.peripheral)

        # Layer 2: Feature Buffer
        self.feature_buffer = FeatureBuffer(self.config.engine)

        # Layer 3: Cognitive Classifier
        self.classifier = CognitiveClassifier(self.config.engine)
        if self.classifier.scaler is not None:
            self.feature_buffer.set_scaler(self.classifier.scaler)

        # Layer 4: Context Awareness
        self.app_detector = AppDetector()

        # Layer 5: Workspace Execution
        self.adaptation_executor = AdaptationExecutor(self.config.adaptation)

        # Register signal handlers for safe rollback
        signal.signal(signal.SIGINT, self._handle_exit_signal)
        signal.signal(signal.SIGTERM, self._handle_exit_signal)

    def _handle_exit_signal(self, signum, frame) -> None:
        """Handle termination signal gracefully."""
        print("\n[AetherFlow] Shutdown requested. Restoring environment...")
        self.stop()
        sys.exit(0)

    def start(self) -> None:
        """Start background sensors and feature intake."""
        self.running = True
        self.analytics = SessionAnalytics.create()
        self.vision_tracker.start()
        self.peripheral_tracker.start()
        logger.info("AetherFlow engine initialized and active.")

    def stop(self) -> None:
        """Stop sensors, restore display gamma ramp, and print session metrics."""
        if not self.running:
            return
        self.running = False
        self.vision_tracker.stop()
        self.peripheral_tracker.stop()
        self.adaptation_executor.restore_display()
        logger.info("AetherFlow stopped successfully.")

    def run_loop(self, interval_sec: float = 1.0, max_iterations: Optional[int] = None) -> None:
        """Main non-blocking closed-loop orchestration cycle."""
        self.start()
        print("\n" + "=" * 70)
        print("  AETHERFLOW -- Local Neuroadaptive Workspace Engine Active")
        print("  100% Local Inference | Privacy Preserving | Closed-Loop Actuation")
        print("=" * 70 + "\n")

        iteration = 0
        last_tick_time = time.time()

        try:
            while self.running:
                t_start = time.time()
                dt = t_start - last_tick_time
                last_tick_time = t_start

                # 1. Sense: acquire latest biometrics & peripheral metrics
                vis_sample = self.vision_tracker.get_latest_sample()
                per_sample = self.peripheral_tracker.sample(window_sec=self.config.engine.sliding_window_sec)

                # 2. Buffer: push to sliding window
                self.feature_buffer.add_sample(vis_sample, per_sample)
                feat_vec, feat_dict = self.feature_buffer.extract_features()

                # 3. Classify: infer cognitive state
                inference = self.classifier.predict(feat_vec, feat_dict, t_start)

                # 4. Context: detect active application domain
                app_context = self.app_detector.get_active_app()

                # 5. Adapt: dispatch closed-loop environment actuation
                exec_result = self.adaptation_executor.dispatch(
                    cognitive_state=inference.state,
                    context_domain=app_context.domain
                )

                # 6. Analytics
                self.analytics.record_tick(inference.state, dt, exec_result.executed)

                # 7. Render telemetry dashboard
                self._render_dashboard(
                    vis_sample=vis_sample,
                    per_sample=per_sample,
                    inference=inference,
                    app_context=app_context,
                    exec_result=exec_result,
                    iteration=iteration
                )

                iteration += 1
                if max_iterations is not None and iteration >= max_iterations:
                    break

                # Sleep to maintain desired loop frequency
                elapsed = time.time() - t_start
                sleep_sec = max(0.05, interval_sec - elapsed)
                time.sleep(sleep_sec)

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            self._print_session_summary()

    def _render_dashboard(
        self,
        vis_sample,
        per_sample,
        inference,
        app_context,
        exec_result,
        iteration: int
    ) -> None:
        """Render rich console dashboard with real-time status."""
        state_colors = {
            CognitiveState.FLOW.value: "\033[92m",              # Green
            CognitiveState.NORMAL_FOCUS.value: "\033[96m",      # Cyan
            CognitiveState.COGNITIVE_OVERLOAD.value: "\033[93m",# Yellow
            CognitiveState.FATIGUE.value: "\033[91m",           # Red
            CognitiveState.IDLE.value: "\033[90m",              # Grey
        }
        reset = "\033[0m"
        bold = "\033[1m"
        dim = "\033[2m"

        state_color = state_colors.get(inference.state.value, reset)
        conf_pct = int(inference.confidence * 100)

        # Build dominant markers string
        dom_str = ", ".join(f"{k}: {v}" for k, v in inference.dominant_features[:2]) if inference.dominant_features else "calibrated baseline"

        # Telemetry summary line
        print(f"{bold}[#{iteration:04d} - {time.strftime('%H:%M:%S')}]{reset} "
              f"Cognitive: {state_color}{bold}{inference.state.value} ({conf_pct}%){reset} | "
              f"Domain: {bold}{app_context.domain.value}{reset} ({app_context.process_name})")

        print(f"  {dim}Biometrics:{reset} EAR={vis_sample.ear_mean:.2f} | "
              f"Blinks={vis_sample.blink_rate_bpm:.1f} BPM | "
              f"GazeStab={vis_sample.gaze_stability:.2f} | "
              f"PERCLOS={vis_sample.perclos:.1%}")

        print(f"  {dim}Peripherals:{reset} Dwell={per_sample.mean_dwell_time_ms:.0f}ms | "
              f"Flight={per_sample.mean_flight_time_ms:.0f}ms | "
              f"Cadence={per_sample.keystrokes_per_min:.0f} KPM | "
              f"Jitter={per_sample.mouse_jitter_index:.3f}")

        adapt_mark = "[*]" if exec_result.executed else "[ ]"
        cd_info = f"Lock: {exec_result.cooldown_remaining_sec:.0f}s" if exec_result.cooldown_remaining_sec > 0 else "Ready"
        print(f"  {dim}Adaptation:{reset} {adapt_mark} Warmth: {bold}{exec_result.current_kelvin}K{reset} | "
              f"UI: {exec_result.ui_density.value} | "
              f"DND: {exec_result.notification_policy.value} | "
              f"Pacing: {cd_info}")

        if exec_result.executed:
            print(f"  {bold}\033[94m-> ACTION:{reset} {exec_result.summary}")
        print("-" * 70)

    def _print_session_summary(self) -> None:
        """Display final analytics report upon shutdown."""
        total_time = sum(self.analytics.state_durations.values())
        print("\n" + "=" * 60)
        print("          AETHERFLOW SESSION SUMMARY REPORT")
        print("=" * 60)
        print(f"Total Monitoring Time : {total_time:.1f} seconds")
        print(f"Closed-Loop Adaptations: {self.analytics.adaptations_count}")
        print("-" * 60)
        print("Cognitive State Distribution:")
        for s in CognitiveState.list():
            dur = self.analytics.state_durations.get(s, 0.0)
            pct = (dur / total_time * 100) if total_time > 0 else 0.0
            bar = "#" * int(pct / 4)
            print(f"  {s:<20} {dur:>6.1f}s  ({pct:>5.1f}%) {bar}")
        print("=" * 60 + "\n")


def build_config(args: argparse.Namespace) -> AetherFlowConfig:
    """Build structured configuration from parsed CLI arguments."""
    config = AetherFlowConfig()
    
    if args.dry_run:
        config.adaptation.dry_run = True
    if args.synthetic_sensing:
        config.vision.synthetic_mode = True
        config.peripheral.synthetic_mode = True
    if args.camera_id is not None:
        config.vision.camera_id = args.camera_id
    if args.window_size is not None:
        config.engine.sliding_window_sec = args.window_size
    if getattr(args, "adaptation_interval", None) is not None:
        config.adaptation.min_adaptation_interval_sec = args.adaptation_interval
    if getattr(args, "transition_duration", None) is not None:
        config.adaptation.transition_duration_sec = args.transition_duration
    if args.debug:
        config.debug = True

    return config


def main() -> None:
    """CLI entrypoint for AetherFlow."""
    parser = argparse.ArgumentParser(
        description="AetherFlow - Local Closed-Loop Neuroadaptive Workspace Engine."
    )
    parser.add_argument("--dry-run", action="store_true", help="Simulate adaptations without altering display gamma")
    parser.add_argument("--synthetic-sensing", action="store_true", help="Generate simulated biometric telemetry")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device ID (default 0)")
    parser.add_argument("--window-size", type=float, default=60.0, help="Sliding feature window in seconds (default 60)")
    parser.add_argument("--interval", type=float, default=1.0, help="Event loop tick interval in seconds (default 1.0)")
    parser.add_argument("--iterations", type=int, default=None, help="Stop after N iterations (for testing)")
    parser.add_argument("--adaptation-interval", type=float, default=None, help="Minimum interval in seconds between display adjustments (default: 30.0)")
    parser.add_argument("--transition-duration", type=float, default=None, help="Gradual fade duration in seconds (default: 4.0)")
    parser.add_argument("--gui", action="store_true", help="Launch dedicated desktop visual window with live camera feed")
    parser.add_argument("--train", action="store_true", help="Pre-train cognitive classifier model before starting")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.train:
        print("[AetherFlow] Training classifier model...")
        train_and_serialize(samples_per_class=800, output_dir=DATA_DIR)

    config = build_config(args)

    if args.gui:
        from aetherflow.gui import run_gui
        run_gui(config)
        return

    engine = AetherFlowEngine(config)
    engine.run_loop(interval_sec=args.interval, max_iterations=args.iterations)


if __name__ == "__main__":
    main()
