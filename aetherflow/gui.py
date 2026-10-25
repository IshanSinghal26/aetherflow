"""Dedicated Desktop GUI Frontend for AetherFlow.

Provides a real-time visual control room featuring:
1. Live camera viewport with eye meshes, iris tracking, and head pose orientation.
2. Cognitive state inference card with confidence meters and probability distributions.
3. Live biometric gauges (EAR, Blink Rate, PERCLOS, Gaze Stability).
4. Peripheral telemetry meters (Dwell time, Flight time, Typing cadence, Mouse jitter).
5. Workspace adaptation display with live Kelvin screen warmth color swatches.
6. Interactive simulation controls to test cognitive states on demand.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

# Ensure Tkinter & PIL imports
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    from PIL import Image, ImageTk
except ImportError as e:
    raise ImportError(f"GUI dependencies missing: {e}. Please install pillow.")

from aetherflow.adaptation.executor import AdaptationExecutor, kelvin_to_rgb
from aetherflow.adaptation.profiles import ProfileRegistry
from aetherflow.config import (
    AetherFlowConfig,
    BreakRecommendation,
    CognitiveState,
    ContextDomain,
    NotificationPolicy,
    UIDensityMode,
)
from aetherflow.context.app_detector import ActiveAppContext, AppDetector
from aetherflow.engine.classifier import CognitiveClassifier, CognitiveInference
from aetherflow.engine.feature_buffer import FeatureBuffer
from aetherflow.sensing.peripheral_tracker import PeripheralSample, PeripheralTracker
from aetherflow.sensing.vision_tracker import VisionSample, VisionTracker

logger = logging.getLogger("aetherflow.gui")

# Visual Palette (Dark Cyberpunk / Glassmorphic Slate)
BG_MAIN = "#0f172a"        # Deep Slate 900
BG_CARD = "#1e293b"        # Slate 800
BG_CARD_LIGHT = "#334155"  # Slate 700
BORDER_COLOR = "#475569"   # Slate 600

TEXT_PRIMARY = "#f8fafc"   # Slate 50
TEXT_SECONDARY = "#94a3b8" # Slate 400
TEXT_MUTED = "#64748b"     # Slate 500

ACCENT_CYAN = "#06b6d4"    # Normal Focus
ACCENT_GREEN = "#10b981"   # Flow
ACCENT_AMBER = "#f59e0b"   # Overload
ACCENT_RED = "#ef4444"     # Fatigue
ACCENT_SLATE = "#64748b"   # Idle


def kelvin_to_hex(kelvin: int) -> str:
    """Convert Kelvin temperature to approximate hex color code for swatch preview."""
    r_mult, g_mult, b_mult = kelvin_to_rgb(kelvin)
    # Baseline daylight white
    r = int(min(255, max(0, 255 * r_mult)))
    g = int(min(255, max(0, 245 * g_mult)))
    b = int(min(255, max(0, 230 * b_mult)))
    return f"#{r:02x}{g:02x}{b:02x}"


class AetherFlowApp(tk.Tk):
    """Main desktop GUI window for AetherFlow."""

    def __init__(self, config: Optional[AetherFlowConfig] = None) -> None:
        super().__init__()
        self.config = config or AetherFlowConfig()
        
        self.title("AetherFlow — Local Neuroadaptive Workspace Engine")
        self.geometry("1260x820")
        self.minsize(1100, 750)
        self.configure(bg=BG_MAIN)

        # Engine Components
        self.vision_tracker = VisionTracker(self.config.vision)
        self.peripheral_tracker = PeripheralTracker(self.config.peripheral)
        self.feature_buffer = FeatureBuffer(self.config.engine)
        self.classifier = CognitiveClassifier(self.config.engine)
        if self.classifier.scaler is not None:
            self.feature_buffer.set_scaler(self.classifier.scaler)
        self.app_detector = AppDetector()
        self.adaptation_executor = AdaptationExecutor(self.config.adaptation)

        # Simulation / Override State
        self.simulated_state: Optional[CognitiveState] = None
        self.camera_enabled = True
        self.show_overlay = True

        # Latest runtime data
        self.latest_inference: Optional[CognitiveInference] = None
        self.latest_app: Optional[ActiveAppContext] = None

        # Build UI layout
        self._setup_styles()
        self._build_header()
        self._build_body()
        self._build_footer()

        # Start background sensors
        self.vision_tracker.start()
        self.peripheral_tracker.start()

        # Bind close handler
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Schedule refresh cycles (Camera at ~30 FPS, Telemetry loop at ~10 FPS)
        self.after(30, self._update_camera_feed)
        self.after(100, self._engine_tick)

    def _setup_styles(self) -> None:
        """Configure ttk styles."""
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure(".", background=BG_MAIN, foreground=TEXT_PRIMARY)
        self.style.configure("TProgressbar", thickness=8, troughcolor=BG_CARD_LIGHT, background=ACCENT_CYAN)

    def _build_header(self) -> None:
        """Top bar with title, status pill, and clock."""
        header_frame = tk.Frame(self, bg=BG_CARD, height=64, padx=20, pady=12)
        header_frame.pack(side=tk.TOP, fill=tk.X)
        header_frame.pack_propagate(False)

        # Logo and Title
        title_box = tk.Frame(header_frame, bg=BG_CARD)
        title_box.pack(side=tk.LEFT)

        logo_lbl = tk.Label(title_box, text="⚡", font=("Segoe UI", 18, "bold"), fg=ACCENT_CYAN, bg=BG_CARD)
        logo_lbl.pack(side=tk.LEFT, padx=(0, 8))

        title_lbl = tk.Label(title_box, text="AETHERFLOW", font=("Segoe UI", 16, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD)
        title_lbl.pack(side=tk.LEFT)

        subtitle_lbl = tk.Label(
            title_box,
            text="Local Neuroadaptive Workspace Engine",
            font=("Segoe UI", 10),
            fg=TEXT_SECONDARY,
            bg=BG_CARD
        )
        subtitle_lbl.pack(side=tk.LEFT, padx=(12, 0))

        # Status badge
        status_box = tk.Frame(header_frame, bg=BG_CARD)
        status_box.pack(side=tk.RIGHT)

        self.status_dot = tk.Label(status_box, text="●", font=("Segoe UI", 12), fg=ACCENT_GREEN, bg=BG_CARD)
        self.status_dot.pack(side=tk.LEFT, padx=(0, 6))

        self.status_lbl = tk.Label(
            status_box,
            text="CLOSED LOOP ACTIVE",
            font=("Segoe UI", 10, "bold"),
            fg=ACCENT_GREEN,
            bg=BG_CARD
        )
        self.status_lbl.pack(side=tk.LEFT, padx=(0, 15))

        # Clock
        self.clock_lbl = tk.Label(status_box, text="", font=("Segoe UI", 10), fg=TEXT_SECONDARY, bg=BG_CARD)
        self.clock_lbl.pack(side=tk.LEFT)

    def _build_body(self) -> None:
        """Main split container (Camera viewport on left, Dashboard cards on right)."""
        body_frame = tk.Frame(self, bg=BG_MAIN, padx=16, pady=16)
        body_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # LEFT PANE: Camera & Vision Viewport
        left_pane = tk.Frame(body_frame, bg=BG_CARD, width=540, relief=tk.FLAT, bd=0)
        left_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 16))
        left_pane.pack_propagate(False)

        cam_header = tk.Frame(left_pane, bg=BG_CARD, padx=16, pady=12)
        cam_header.pack(fill=tk.X)
        tk.Label(cam_header, text="LIVE BIOMETRIC TELEMETRY FEED", font=("Segoe UI", 11, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD).pack(side=tk.LEFT)
        self.cam_fps_lbl = tk.Label(cam_header, text="30 FPS", font=("Segoe UI", 9), fg=TEXT_SECONDARY, bg=BG_CARD)
        self.cam_fps_lbl.pack(side=tk.RIGHT)

        # Video Canvas
        self.video_canvas = tk.Canvas(left_pane, bg="#05070d", width=508, height=381, highlightthickness=1, highlightbackground=BORDER_COLOR)
        self.video_canvas.pack(padx=16, pady=(0, 10))

        # Video controls
        cam_ctrls = tk.Frame(left_pane, bg=BG_CARD, padx=16, pady=6)
        cam_ctrls.pack(fill=tk.X)

        self.btn_cam_toggle = tk.Button(
            cam_ctrls,
            text="Camera: ON",
            font=("Segoe UI", 9, "bold"),
            bg=BG_CARD_LIGHT,
            fg=TEXT_PRIMARY,
            activebackground=BORDER_COLOR,
            relief=tk.FLAT,
            padx=10,
            pady=4,
            command=self._toggle_camera
        )
        self.btn_cam_toggle.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_overlay_toggle = tk.Button(
            cam_ctrls,
            text="Overlay: ON",
            font=("Segoe UI", 9),
            bg=BG_CARD_LIGHT,
            fg=TEXT_PRIMARY,
            activebackground=BORDER_COLOR,
            relief=tk.FLAT,
            padx=10,
            pady=4,
            command=self._toggle_overlay
        )
        self.btn_overlay_toggle.pack(side=tk.LEFT)

        # Vision Telemetry Metrics Sub-card
        vis_box = tk.LabelFrame(left_pane, text=" Ocular Biometrics ", font=("Segoe UI", 9, "bold"), fg=TEXT_SECONDARY, bg=BG_CARD, padx=12, pady=10)
        vis_box.pack(fill=tk.X, padx=16, pady=(12, 10))

        self.metric_ear_val = self._create_metric_row(vis_box, "Eye Aspect Ratio (EAR):", "0.29", "Blink Threshold: 0.21")
        self.metric_blink_val = self._create_metric_row(vis_box, "Blink Frequency:", "14.0 BPM", "Normal: 15-22 BPM")
        self.metric_perclos_val = self._create_metric_row(vis_box, "PERCLOS (% Closed):", "3.2%", "Fatigue: > 15%")
        self.metric_gaze_val = self._create_metric_row(vis_box, "Gaze Stability:", "92%", "Flow: > 85%")

        # RIGHT PANE: Dashboard Cards & Adaptations
        right_pane = tk.Frame(body_frame, bg=BG_MAIN)
        right_pane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 1. Active Context Card
        ctx_card = tk.Frame(right_pane, bg=BG_CARD, padx=16, pady=12)
        ctx_card.pack(fill=tk.X, pady=(0, 12))

        tk.Label(ctx_card, text="ACTIVE FOREGROUND CONTEXT", font=("Segoe UI", 9, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor=tk.W)
        ctx_row = tk.Frame(ctx_card, bg=BG_CARD)
        ctx_row.pack(fill=tk.X, pady=(4, 0))

        self.ctx_domain_badge = tk.Label(
            ctx_row,
            text="CODE_IDE",
            font=("Segoe UI", 11, "bold"),
            bg="#0284c7",
            fg="#ffffff",
            padx=8,
            pady=3
        )
        self.ctx_domain_badge.pack(side=tk.LEFT, padx=(0, 10))

        self.ctx_app_lbl = tk.Label(ctx_row, text="code.exe — Visual Studio Code", font=("Segoe UI", 11), fg=TEXT_PRIMARY, bg=BG_CARD)
        self.ctx_app_lbl.pack(side=tk.LEFT)

        # 2. Cognitive State Card
        state_card = tk.Frame(right_pane, bg=BG_CARD, padx=16, pady=14)
        state_card.pack(fill=tk.X, pady=(0, 12))

        tk.Label(state_card, text="INFERRED COGNITIVE STATE", font=("Segoe UI", 9, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor=tk.W)

        state_top_row = tk.Frame(state_card, bg=BG_CARD)
        state_top_row.pack(fill=tk.X, pady=(6, 8))

        self.state_pill = tk.Label(
            state_top_row,
            text="NORMAL_FOCUS",
            font=("Segoe UI", 16, "bold"),
            bg=ACCENT_CYAN,
            fg="#000000",
            padx=14,
            pady=4
        )
        self.state_pill.pack(side=tk.LEFT, padx=(0, 14))

        self.confidence_lbl = tk.Label(state_top_row, text="Confidence: 88%", font=("Segoe UI", 12, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD)
        self.confidence_lbl.pack(side=tk.LEFT)

        # Probabilities Mini Bars
        prob_container = tk.Frame(state_card, bg=BG_CARD)
        prob_container.pack(fill=tk.X, pady=(4, 4))

        self.prob_bars: dict[str, ttk.Progressbar] = {}
        self.prob_labels: dict[str, tk.Label] = {}
        for state_name in CognitiveState.list():
            row = tk.Frame(prob_container, bg=BG_CARD)
            row.pack(fill=tk.X, pady=1)
            lbl = tk.Label(row, text=f"{state_name[:8]:<8}", font=("Consolas", 8), fg=TEXT_SECONDARY, bg=BG_CARD, width=9, anchor=tk.W)
            lbl.pack(side=tk.LEFT)
            pb = ttk.Progressbar(row, maximum=100, length=180, style="TProgressbar")
            pb.pack(side=tk.LEFT, padx=6)
            val_lbl = tk.Label(row, text="0%", font=("Consolas", 8), fg=TEXT_MUTED, bg=BG_CARD, width=4)
            val_lbl.pack(side=tk.LEFT)
            self.prob_bars[state_name] = pb
            self.prob_labels[state_name] = val_lbl

        self.explain_lbl = tk.Label(state_card, text="Top Markers: Gaze=0.88, Jitter=0.12", font=("Segoe UI", 9, "italic"), fg=TEXT_SECONDARY, bg=BG_CARD)
        self.explain_lbl.pack(anchor=tk.W, pady=(6, 0))

        # 3. Peripheral Dynamics Card
        per_card = tk.LabelFrame(right_pane, text=" Peripheral Kinematics ", font=("Segoe UI", 9, "bold"), fg=TEXT_SECONDARY, bg=BG_CARD, padx=16, pady=10)
        per_card.pack(fill=tk.X, pady=(0, 12))

        self.metric_cadence_val = self._create_metric_row(per_card, "Typing Cadence:", "185 KPM (37 WPM)", "Active Flow")
        self.metric_dwell_val = self._create_metric_row(per_card, "Dwell / Flight Time:", "98 ms / 165 ms", "Crisp Rhythmic")
        self.metric_jitter_val = self._create_metric_row(per_card, "Mouse Jitter Index:", "0.082", "Low (Calm Trajectory)")
        self.metric_idle_val = self._create_metric_row(per_card, "Inactivity / Idle:", "0.8s", "Active")

        # 4. Workspace Adaptation Actuation Card
        act_card = tk.Frame(right_pane, bg=BG_CARD, padx=16, pady=12)
        act_card.pack(fill=tk.X, pady=(0, 12))

        tk.Label(act_card, text="ACTIVE WORKSPACE ENVIRONMENT ADAPTATION", font=("Segoe UI", 9, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor=tk.W)

        act_grid = tk.Frame(act_card, bg=BG_CARD)
        act_grid.pack(fill=tk.X, pady=(8, 0))

        # Kelvin warmth chip & color swatch
        warmth_box = tk.Frame(act_grid, bg=BG_CARD_LIGHT, padx=10, pady=8)
        warmth_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        tk.Label(warmth_box, text="SCREEN WARMTH", font=("Segoe UI", 8, "bold"), fg=TEXT_SECONDARY, bg=BG_CARD_LIGHT).pack(anchor=tk.W)
        warmth_sub = tk.Frame(warmth_box, bg=BG_CARD_LIGHT)
        warmth_sub.pack(anchor=tk.W, pady=(4, 0))
        self.kelvin_swatch = tk.Label(warmth_sub, text="   ", bg="#fff5ea", relief=tk.SOLID, bd=1)
        self.kelvin_swatch.pack(side=tk.LEFT, padx=(0, 8))
        self.kelvin_lbl = tk.Label(warmth_sub, text="5000 K", font=("Segoe UI", 12, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD_LIGHT)
        self.kelvin_lbl.pack(side=tk.LEFT)

        # UI Density
        density_box = tk.Frame(act_grid, bg=BG_CARD_LIGHT, padx=10, pady=8)
        density_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        tk.Label(density_box, text="UI DENSITY", font=("Segoe UI", 8, "bold"), fg=TEXT_SECONDARY, bg=BG_CARD_LIGHT).pack(anchor=tk.W)
        self.density_lbl = tk.Label(density_box, text="STANDARD", font=("Segoe UI", 12, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD_LIGHT)
        self.density_lbl.pack(anchor=tk.W, pady=(4, 0))

        # Notification DND
        dnd_box = tk.Frame(act_grid, bg=BG_CARD_LIGHT, padx=10, pady=8)
        dnd_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        tk.Label(dnd_box, text="NOTIFICATION DND", font=("Segoe UI", 8, "bold"), fg=TEXT_SECONDARY, bg=BG_CARD_LIGHT).pack(anchor=tk.W)
        self.dnd_lbl = tk.Label(dnd_box, text="PRIORITY_ONLY", font=("Segoe UI", 11, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD_LIGHT)
        self.dnd_lbl.pack(anchor=tk.W, pady=(4, 0))

        # Adaptation Interval & Gradual Fade Status
        pacing_box = tk.Frame(act_grid, bg=BG_CARD_LIGHT, padx=10, pady=8)
        pacing_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(pacing_box, text="ADAPTATION PACING", font=("Segoe UI", 8, "bold"), fg=TEXT_SECONDARY, bg=BG_CARD_LIGHT).pack(anchor=tk.W)
        self.pacing_lbl = tk.Label(pacing_box, text="Ready (30s interval)", font=("Segoe UI", 11, "bold"), fg=ACCENT_CYAN, bg=BG_CARD_LIGHT)
        self.pacing_lbl.pack(anchor=tk.W, pady=(4, 0))

        # Interval Quick Selector Toolbar
        interval_row = tk.Frame(act_card, bg=BG_CARD)
        interval_row.pack(fill=tk.X, pady=(6, 0))
        tk.Label(interval_row, text="Change Interval Cooldown:", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD).pack(side=tk.LEFT, padx=(0, 6))
        for sec in [15, 30, 60, 120]:
            btn = tk.Button(
                interval_row,
                text=f"{sec}s",
                font=("Segoe UI", 8),
                bg=BG_CARD_LIGHT,
                fg=TEXT_PRIMARY,
                activebackground=BORDER_COLOR,
                relief=tk.FLAT,
                padx=6,
                pady=1,
                command=lambda s=sec: self._set_adaptation_interval(s)
            )
            btn.pack(side=tk.LEFT, padx=2)

        fade_dur = self.config.adaptation.transition_duration_sec
        tk.Label(interval_row, text=f"• Fade: {fade_dur:.1f}s smooth S-curve", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD).pack(side=tk.RIGHT)

        # Ergonomic Advisory
        self.break_banner = tk.Label(
            act_card,
            text="✓ Ergonomics optimal — no rest required",
            font=("Segoe UI", 9),
            bg=BG_CARD_LIGHT,
            fg=ACCENT_GREEN,
            padx=8,
            pady=4
        )
        self.break_banner.pack(fill=tk.X, pady=(8, 0))

        # 5. Interactive Simulation & Testing Controls
        sim_box = tk.Frame(right_pane, bg=BG_CARD, padx=16, pady=10)
        sim_box.pack(fill=tk.X)

        tk.Label(sim_box, text="INTERACTIVE STATE SIMULATION / OVERRIDE:", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(side=tk.LEFT, padx=(0, 10))

        self.btn_flow = self._create_sim_button(sim_box, "Flow", CognitiveState.FLOW, ACCENT_GREEN)
        self.btn_norm = self._create_sim_button(sim_box, "Normal", CognitiveState.NORMAL_FOCUS, ACCENT_CYAN)
        self.btn_overload = self._create_sim_button(sim_box, "Overload", CognitiveState.COGNITIVE_OVERLOAD, ACCENT_AMBER)
        self.btn_fatigue = self._create_sim_button(sim_box, "Fatigue", CognitiveState.FATIGUE, ACCENT_RED)
        self.btn_idle = self._create_sim_button(sim_box, "Idle", CognitiveState.IDLE, ACCENT_SLATE)

        self.btn_auto = tk.Button(
            sim_box,
            text="Auto-Sense",
            font=("Segoe UI", 8, "bold"),
            bg="#3b82f6",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=8,
            pady=3,
            command=self._reset_to_auto
        )
        self.btn_auto.pack(side=tk.LEFT, padx=(8, 4))

        self.btn_reset_gamma = tk.Button(
            sim_box,
            text="Restore 6500K",
            font=("Segoe UI", 8),
            bg=BG_CARD_LIGHT,
            fg=TEXT_PRIMARY,
            relief=tk.FLAT,
            padx=8,
            pady=3,
            command=self._restore_display_gamma
        )
        self.btn_reset_gamma.pack(side=tk.RIGHT)

    def _build_footer(self) -> None:
        """Bottom status bar with privacy statement and safeguards."""
        footer_frame = tk.Frame(self, bg=BG_MAIN, height=28, padx=16)
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Label(
            footer_frame,
            text="🔒 100% Local Inference — Zero Cloud Transmission | Zero Keylogging | Hardware Gamma Auto-Restoration Safeguard Active",
            font=("Segoe UI", 8),
            fg=TEXT_MUTED,
            bg=BG_MAIN
        ).pack(side=tk.LEFT)

    def _create_metric_row(self, parent: tk.Widget, label_text: str, default_val: str, note_text: str) -> tk.Label:
        """Helper to create standardized 3-column metric row."""
        row = tk.Frame(parent, bg=BG_CARD)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text=label_text, font=("Segoe UI", 9), fg=TEXT_SECONDARY, bg=BG_CARD, width=22, anchor=tk.W).pack(side=tk.LEFT)
        val_lbl = tk.Label(row, text=default_val, font=("Segoe UI", 9, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD, width=16, anchor=tk.W)
        val_lbl.pack(side=tk.LEFT)
        tk.Label(row, text=note_text, font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD).pack(side=tk.LEFT)
        return val_lbl

    def _create_sim_button(self, parent: tk.Widget, title: str, state: CognitiveState, fg_col: str) -> tk.Button:
        """Helper to create a state simulation override button."""
        btn = tk.Button(
            parent,
            text=title,
            font=("Segoe UI", 8, "bold"),
            bg=BG_CARD_LIGHT,
            fg=fg_col,
            activebackground=BORDER_COLOR,
            relief=tk.FLAT,
            padx=6,
            pady=2,
            command=lambda: self._set_simulated_state(state)
        )
        btn.pack(side=tk.LEFT, padx=2)
        return btn

    def _set_simulated_state(self, state: CognitiveState) -> None:
        """Manually force a cognitive state to test workplace adaptations immediately."""
        self.simulated_state = state
        self.status_lbl.configure(text=f"SIMULATION: {state.value}", fg=ACCENT_AMBER)
        self.status_dot.configure(fg=ACCENT_AMBER)

    def _reset_to_auto(self) -> None:
        """Resume real-time biometric and peripheral sensor fusion."""
        self.simulated_state = None
        self.status_lbl.configure(text="CLOSED LOOP ACTIVE", fg=ACCENT_GREEN)
        self.status_dot.configure(fg=ACCENT_GREEN)

    def _restore_display_gamma(self) -> None:
        """Restore screen display to default pristine gamma ramp."""
        self.adaptation_executor.restore_display()
        self.kelvin_lbl.configure(text="6500 K")
        self.kelvin_swatch.configure(bg="#ffffff")

    def _toggle_camera(self) -> None:
        """Toggle camera stream on/off."""
        self.camera_enabled = not self.camera_enabled
        self.btn_cam_toggle.configure(
            text="Camera: ON" if self.camera_enabled else "Camera: OFF",
            fg=ACCENT_GREEN if self.camera_enabled else ACCENT_RED
        )

    def _toggle_overlay(self) -> None:
        """Toggle landmark overlay on video."""
        self.show_overlay = not self.show_overlay
        self.btn_overlay_toggle.configure(
            text="Overlay: ON" if self.show_overlay else "Overlay: OFF"
        )

    # -------------------------------------------------------------
    # Main Engine Tick and Camera Update
    # -------------------------------------------------------------
    def _update_camera_feed(self) -> None:
        """Fetch frame from VisionTracker and blit to Tkinter canvas at ~30 FPS."""
        if self.camera_enabled:
            frame, _ = self.vision_tracker.get_latest_frame()
            if frame is not None:
                # Resize to fit canvas (508 x 381)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (508, 381), interpolation=cv2.INTER_AREA)
                img = Image.fromarray(frame_resized)
                img_tk = ImageTk.PhotoImage(image=img)
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                self.video_canvas.image = img_tk  # keep reference
        self.after(33, self._update_camera_feed)

    def _engine_tick(self) -> None:
        """Core closed-loop iteration executed at 10 Hz."""
        t_now = time.time()
        self.clock_lbl.configure(text=time.strftime("%H:%M:%S"))

        # 1. Sense
        vis_sample = self.vision_tracker.get_latest_sample()
        per_sample = self.peripheral_tracker.sample(window_sec=self.config.engine.sliding_window_sec)

        # 2. Buffer & Extract
        self.feature_buffer.add_sample(vis_sample, per_sample)
        feat_vec, feat_dict = self.feature_buffer.extract_features()

        # 3. Classify (or use manual simulation override)
        if self.simulated_state is not None:
            probs = {c: 0.02 for c in CognitiveState.list()}
            probs[self.simulated_state.value] = 0.92
            inference = CognitiveInference(
                state=self.simulated_state,
                confidence=0.92,
                probabilities=probs,
                dominant_features=[("manual_override", 1.0)],
                timestamp=t_now
            )
        else:
            inference = self.classifier.predict(feat_vec, feat_dict, t_now)

        self.latest_inference = inference

        # 4. Context Detection
        app_context = self.app_detector.get_active_app()
        self.latest_app = app_context

        # 5. Adaptation Actuation
        exec_res = self.adaptation_executor.dispatch(
            cognitive_state=inference.state,
            context_domain=app_context.domain
        )

        # 6. Update GUI Displays
        self._update_dashboard_ui(vis_sample, per_sample, inference, app_context, exec_res)

        self.after(100, self._engine_tick)

    def _update_dashboard_ui(
        self,
        vis: VisionSample,
        per: PeripheralSample,
        inf: CognitiveInference,
        ctx: ActiveAppContext,
        res
    ) -> None:
        """Refresh all telemetry widgets with latest metrics."""
        # 1. Context updates
        self.ctx_domain_badge.configure(text=ctx.domain.value)
        title_str = ctx.window_title[:45] + "..." if len(ctx.window_title) > 45 else ctx.window_title
        self.ctx_app_lbl.configure(text=f"{ctx.process_name} — {title_str or 'Foreground Window'}")

        # 2. State & Confidence updates
        state_colors = {
            CognitiveState.FLOW.value: (ACCENT_GREEN, "#000000"),
            CognitiveState.NORMAL_FOCUS.value: (ACCENT_CYAN, "#000000"),
            CognitiveState.COGNITIVE_OVERLOAD.value: (ACCENT_AMBER, "#000000"),
            CognitiveState.FATIGUE.value: (ACCENT_RED, "#ffffff"),
            CognitiveState.IDLE.value: (ACCENT_SLATE, "#ffffff"),
        }
        bg_col, fg_col = state_colors.get(inf.state.value, (BG_CARD_LIGHT, TEXT_PRIMARY))
        self.state_pill.configure(text=inf.state.value, bg=bg_col, fg=fg_col)
        self.confidence_lbl.configure(text=f"Confidence: {int(inf.confidence * 100)}%")

        # Probability mini bars
        for s_name, pb in self.prob_bars.items():
            prob_pct = int(inf.probabilities.get(s_name, 0.0) * 100)
            pb["value"] = prob_pct
            self.prob_labels[s_name].configure(text=f"{prob_pct}%")

        if inf.dominant_features:
            explain_str = "Top Markers: " + ", ".join(f"{k}: {v}" for k, v in inf.dominant_features[:2])
            self.explain_lbl.configure(text=explain_str)

        # 3. Biometric metrics
        blink_tag = " [BLINK]" if vis.is_blinking else ""
        self.metric_ear_val.configure(
            text=f"{vis.ear_mean:.2f}{blink_tag}",
            fg=ACCENT_RED if vis.ear_mean < 0.21 else TEXT_PRIMARY
        )
        self.metric_blink_val.configure(text=f"{vis.blink_rate_bpm:.1f} BPM")
        self.metric_perclos_val.configure(
            text=f"{vis.perclos:.1%}",
            fg=ACCENT_RED if vis.perclos > 0.15 else TEXT_PRIMARY
        )
        self.metric_gaze_val.configure(text=f"{int(vis.gaze_stability * 100)}%")

        # 4. Peripheral metrics
        wpm = int(per.keystrokes_per_min / 5.0)
        self.metric_cadence_val.configure(text=f"{per.keystrokes_per_min:.0f} KPM ({wpm} WPM)")
        self.metric_dwell_val.configure(text=f"{per.mean_dwell_time_ms:.0f} ms / {per.mean_flight_time_ms:.0f} ms")
        self.metric_jitter_val.configure(
            text=f"{per.mouse_jitter_index:.3f}",
            fg=ACCENT_RED if per.mouse_jitter_index > 0.35 else (ACCENT_AMBER if per.mouse_jitter_index > 0.20 else TEXT_PRIMARY)
        )
        self.metric_idle_val.configure(text=f"{per.idle_seconds:.1f}s ({'IDLE' if per.is_idle else 'ACTIVE'})")

        # 5. Adaptations
        self.kelvin_lbl.configure(text=f"{res.current_kelvin} K")
        self.kelvin_swatch.configure(bg=kelvin_to_hex(res.current_kelvin))
        self.density_lbl.configure(text=res.ui_density.value)
        self.dnd_lbl.configure(text=res.notification_policy.value)

        # Update Pacing Status
        if res.cooldown_remaining_sec > 0:
            self.pacing_lbl.configure(
                text=f"Locked: {res.cooldown_remaining_sec:.0f}s",
                fg=ACCENT_AMBER
            )
        else:
            int_sec = int(self.config.adaptation.min_adaptation_interval_sec)
            self.pacing_lbl.configure(
                text=f"Ready ({int_sec}s interval)",
                fg=ACCENT_GREEN
            )

        if res.break_recommendation == BreakRecommendation.MANDATORY_BREAK:
            self.break_banner.configure(
                text="⚠️ FATIGUE DETECTED — Take an ergonomic 5-minute break now!",
                bg=ACCENT_RED,
                fg="#ffffff"
            )
        elif res.break_recommendation == BreakRecommendation.GENTLE_PROMPT:
            self.break_banner.configure(
                text="⏳ Cognitive strain elevated — gently pace yourself",
                bg=ACCENT_AMBER,
                fg="#000000"
            )
        else:
            self.break_banner.configure(
                text="✓ Ergonomics optimal — environment matched to creative context",
                bg=BG_CARD_LIGHT,
                fg=ACCENT_GREEN
            )

    def _set_adaptation_interval(self, seconds: int) -> None:
        """Dynamically update the adaptation interval / cooldown period."""
        self.config.adaptation.min_adaptation_interval_sec = float(seconds)
        self.adaptation_executor.config.min_adaptation_interval_sec = float(seconds)
        int_sec = int(seconds)
        self.pacing_lbl.configure(text=f"Ready ({int_sec}s interval)", fg=ACCENT_GREEN)
        logger.info(f"Adaptation interval set to {seconds} seconds.")

    def on_closing(self) -> None:
        """Safe shutdown when user closes window."""
        logger.info("GUI window closed. Terminating background threads...")
        self.vision_tracker.stop()
        self.peripheral_tracker.stop()
        self.adaptation_executor.restore_display()
        self.destroy()


def run_gui(config: Optional[AetherFlowConfig] = None) -> None:
    """Entry point to start the dedicated GUI application."""
    app = AetherFlowApp(config)
    app.mainloop()


if __name__ == "__main__":
    run_gui()
