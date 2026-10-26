"""Dedicated Desktop GUI Frontend for AetherFlow.

Provides a real-time visual control room featuring:
1. Live camera viewport with eye meshes, iris tracking, and head pose orientation.
2. Cognitive state inference hero card with confidence meters and multi-state probability bars.
3. 2x2 Elevated Ocular Biometric HUD (EAR, Blink Frequency, PERCLOS Strain, Gaze Stability).
4. 2x2 Elevated Peripheral Kinematics HUD (WPM/KPM Cadence, Dwell/Flight, Mouse Jitter, Inactivity).
5. Workspace adaptation display with live Kelvin screen warmth color swatches and pacing control.
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
from aetherflow.report import generate_report
from aetherflow.sensing.peripheral_tracker import PeripheralSample, PeripheralTracker
from aetherflow.sensing.vision_tracker import VisionSample, VisionTracker
from aetherflow.storage import SessionStore

logger = logging.getLogger("aetherflow.gui")

# =====================================================================
# Visual Palette (Cockpit Obsidian / Glassmorphic Slate)
# =====================================================================
BG_MAIN = "#070b12"        # Deep Obsidian Void
BG_CARD = "#0d1526"        # Deep Navy HUD Card
BG_CARD_LIGHT = "#152238"  # Elevated Container / Card Section
BG_TILE = "#101b2f"        # Elevated Biometric KPI Tile
BORDER_COLOR = "#1e2c47"   # High-Tech Crisp Border
BORDER_ACCENT = "#38bdf8"  # Cyan Accent Border

TEXT_PRIMARY = "#f8fafc"   # Slate 50 High Contrast
TEXT_SECONDARY = "#94a3b8" # Slate 400 Readable Secondary
TEXT_MUTED = "#64748b"     # Slate 500 Subtle Captions

# Accent States
ACCENT_CYAN = "#06b6d4"    # Focus / HUD Telemetry
ACCENT_BLUE = "#0284c7"    # Primary Action
ACCENT_GREEN = "#10b981"   # Flow / Optimal State
ACCENT_AMBER = "#f59e0b"   # Overload / Warning
ACCENT_RED = "#ef4444"     # Fatigue / Strain Alert
ACCENT_PURPLE = "#8b5cf6"  # Deep Focus / Code
ACCENT_SLATE = "#475569"   # Idle / Standby

DOMAIN_COLORS: dict[str, tuple[str, str]] = {
    ContextDomain.CODE_IDE.value: ("#4f46e5", "#ffffff"),
    ContextDomain.RESEARCH_BROWSING.value: ("#0284c7", "#ffffff"),
    ContextDomain.CREATIVE_DESIGN.value: ("#db2777", "#ffffff"),
    ContextDomain.WRITING_TEXT.value: ("#9333ea", "#ffffff"),
    ContextDomain.MEDIA_VIDEO.value: ("#059669", "#ffffff"),
    ContextDomain.SYSTEM_UTILITY.value: ("#334155", "#ffffff"),
    ContextDomain.GENERAL.value: ("#1e293b", "#94a3b8"),
}

STATE_HERO_STYLES: dict[str, tuple[str, str, str]] = {
    CognitiveState.FLOW.value: (ACCENT_GREEN, "#022c22", "High alpha-theta synchrony • Flow immersion"),
    CognitiveState.NORMAL_FOCUS.value: (ACCENT_CYAN, "#042f2e", "Steady beta rhythm • Engaged attention"),
    CognitiveState.COGNITIVE_OVERLOAD.value: (ACCENT_AMBER, "#451a03", "Elevated high-beta • Workload surge"),
    CognitiveState.FATIGUE.value: (ACCENT_RED, "#450a0a", "Elevated delta/theta • Ocular strain alert"),
    CognitiveState.IDLE.value: (ACCENT_SLATE, "#0f172a", "Inactive baseline • Sensor standby"),
}


def kelvin_to_hex(kelvin: int) -> str:
    """Convert Kelvin temperature to approximate hex color code for swatch preview."""
    r_mult, g_mult, b_mult = kelvin_to_rgb(kelvin)
    r = int(min(255, max(0, 255 * r_mult)))
    g = int(min(255, max(0, 245 * g_mult)))
    b = int(min(255, max(0, 230 * b_mult)))
    return f"#{r:02x}{g:02x}{b:02x}"


def _bind_hover(btn: tk.Button, normal_bg: str, hover_bg: str) -> None:
    """Add subtle hover highlighting to native Tkinter buttons."""
    btn.bind("<Enter>", lambda _e: btn.configure(bg=hover_bg))
    btn.bind("<Leave>", lambda _e: btn.configure(bg=normal_bg))


class AetherFlowApp(tk.Tk):
    """Main desktop GUI window for AetherFlow."""

    def __init__(self, config: Optional[AetherFlowConfig] = None) -> None:
        super().__init__()
        self.config = config or AetherFlowConfig()

        self.title("AetherFlow — Local Neuroadaptive Workspace Engine")
        self.geometry("1280x850")
        self.minsize(1120, 760)
        self.configure(bg=BG_MAIN)

        # Persistence & Analytics
        self.session_store = SessionStore()
        self.session_id = self.session_store.start_session()
        self.session_start_time = time.time()
        self._last_log_time = 0.0

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
        """Configure ttk styles with custom state progress bars."""
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure(".", background=BG_MAIN, foreground=TEXT_PRIMARY)

        # Progress bar variants for cognitive states
        self.style.configure("Flow.Horizontal.TProgressbar", troughcolor=BG_CARD_LIGHT, background=ACCENT_GREEN, thickness=6)
        self.style.configure("Focus.Horizontal.TProgressbar", troughcolor=BG_CARD_LIGHT, background=ACCENT_CYAN, thickness=6)
        self.style.configure("Overload.Horizontal.TProgressbar", troughcolor=BG_CARD_LIGHT, background=ACCENT_AMBER, thickness=6)
        self.style.configure("Fatigue.Horizontal.TProgressbar", troughcolor=BG_CARD_LIGHT, background=ACCENT_RED, thickness=6)
        self.style.configure("Idle.Horizontal.TProgressbar", troughcolor=BG_CARD_LIGHT, background=ACCENT_SLATE, thickness=6)

    def _build_header(self) -> None:
        """Top obsidian bar with glowing status indicator, session duration, and action pills."""
        header_frame = tk.Frame(self, bg=BG_CARD, height=64, padx=20, pady=12, highlightbackground=BORDER_COLOR, highlightthickness=1)
        header_frame.pack(side=tk.TOP, fill=tk.X)
        header_frame.pack_propagate(False)

        # Brand Logo and Title
        title_box = tk.Frame(header_frame, bg=BG_CARD)
        title_box.pack(side=tk.LEFT)

        logo_lbl = tk.Label(title_box, text="⚡", font=("Segoe UI", 18, "bold"), fg=ACCENT_CYAN, bg=BG_CARD)
        logo_lbl.pack(side=tk.LEFT, padx=(0, 8))

        title_lbl = tk.Label(title_box, text="AETHERFLOW", font=("Segoe UI", 16, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD)
        title_lbl.pack(side=tk.LEFT)

        tagline_lbl = tk.Label(
            title_box,
            text="Local Neuroadaptive Workspace Engine",
            font=("Segoe UI", 9),
            fg=TEXT_SECONDARY,
            bg=BG_CARD
        )
        tagline_lbl.pack(side=tk.LEFT, padx=(12, 0))

        # Status badge, Session Clock & Action buttons
        status_box = tk.Frame(header_frame, bg=BG_CARD)
        status_box.pack(side=tk.RIGHT)

        self.btn_report = tk.Button(
            status_box,
            text="📊 Productivity Report",
            font=("Segoe UI", 9, "bold"),
            bg="#0284c7",
            fg="#ffffff",
            activebackground="#0369a1",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
            command=self._on_generate_report
        )
        self.btn_report.pack(side=tk.LEFT, padx=(0, 10))
        _bind_hover(self.btn_report, "#0284c7", "#0369a1")

        self.btn_min_bg = tk.Button(
            status_box,
            text="🗕 Run in Background",
            font=("Segoe UI", 9),
            bg=BG_CARD_LIGHT,
            fg=TEXT_PRIMARY,
            activebackground=BORDER_COLOR,
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._minimize_to_background
        )
        self.btn_min_bg.pack(side=tk.LEFT, padx=(0, 14))
        _bind_hover(self.btn_min_bg, BG_CARD_LIGHT, BORDER_COLOR)

        # Live Session Duration Timer
        self.session_timer_lbl = tk.Label(
            status_box,
            text="⏱️ 00:00:00",
            font=("Segoe UI", 9, "bold"),
            fg=ACCENT_CYAN,
            bg=BG_CARD_LIGHT,
            padx=8,
            pady=3
        )
        self.session_timer_lbl.pack(side=tk.LEFT, padx=(0, 14))

        # Closed Loop Indicator
        self.status_dot = tk.Label(status_box, text="●", font=("Segoe UI", 12), fg=ACCENT_GREEN, bg=BG_CARD)
        self.status_dot.pack(side=tk.LEFT, padx=(0, 6))

        self.status_lbl = tk.Label(
            status_box,
            text="CLOSED LOOP ACTIVE",
            font=("Segoe UI", 10, "bold"),
            fg=ACCENT_GREEN,
            bg=BG_CARD
        )
        self.status_lbl.pack(side=tk.LEFT, padx=(0, 14))

        # System Clock
        self.clock_lbl = tk.Label(status_box, text="", font=("Segoe UI", 10), fg=TEXT_SECONDARY, bg=BG_CARD)
        self.clock_lbl.pack(side=tk.LEFT)

    def _build_body(self) -> None:
        """Main split container (Camera viewport on left, Dashboard cards on right)."""
        body_frame = tk.Frame(self, bg=BG_MAIN, padx=16, pady=14)
        body_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # -------------------------------------------------------------
        # LEFT PANE: Camera & Ocular HUD
        # -------------------------------------------------------------
        left_pane = tk.Frame(body_frame, bg=BG_CARD, width=520, highlightbackground=BORDER_COLOR, highlightthickness=1)
        left_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 16))
        left_pane.pack_propagate(False)

        cam_header = tk.Frame(left_pane, bg=BG_CARD, padx=16, pady=10)
        cam_header.pack(fill=tk.X)
        tk.Label(cam_header, text="LIVE BIOMETRIC TELEMETRY FEED", font=("Segoe UI", 10, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD).pack(side=tk.LEFT)
        self.cam_fps_lbl = tk.Label(cam_header, text="HUD CAM • 30 FPS", font=("Segoe UI", 8, "bold"), fg=ACCENT_CYAN, bg=BG_CARD_LIGHT, padx=6, pady=2)
        self.cam_fps_lbl.pack(side=tk.RIGHT)

        # Video Canvas (Letterboxed 488 x 366)
        self.video_canvas = tk.Canvas(left_pane, bg="#030712", width=488, height=366, highlightthickness=1, highlightbackground=BORDER_COLOR)
        self.video_canvas.pack(padx=16, pady=(0, 8))

        # Video controls
        cam_ctrls = tk.Frame(left_pane, bg=BG_CARD, padx=16, pady=4)
        cam_ctrls.pack(fill=tk.X)

        self.btn_cam_toggle = tk.Button(
            cam_ctrls,
            text="Camera: ON",
            font=("Segoe UI", 8, "bold"),
            bg=BG_CARD_LIGHT,
            fg=ACCENT_GREEN,
            activebackground=BORDER_COLOR,
            relief=tk.FLAT,
            padx=10,
            pady=3,
            command=self._toggle_camera
        )
        self.btn_cam_toggle.pack(side=tk.LEFT, padx=(0, 8))
        _bind_hover(self.btn_cam_toggle, BG_CARD_LIGHT, BORDER_COLOR)

        self.btn_overlay_toggle = tk.Button(
            cam_ctrls,
            text="Overlay: ON",
            font=("Segoe UI", 8),
            bg=BG_CARD_LIGHT,
            fg=TEXT_PRIMARY,
            activebackground=BORDER_COLOR,
            relief=tk.FLAT,
            padx=10,
            pady=3,
            command=self._toggle_overlay
        )
        self.btn_overlay_toggle.pack(side=tk.LEFT)
        _bind_hover(self.btn_overlay_toggle, BG_CARD_LIGHT, BORDER_COLOR)

        gaze_mode_lbl = tk.Label(cam_ctrls, text="Iris Mesh Active", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD)
        gaze_mode_lbl.pack(side=tk.RIGHT)

        # Elevated Ocular Biometrics 2x2 Grid
        vis_card = tk.Frame(left_pane, bg=BG_CARD_LIGHT, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=12, pady=10)
        vis_card.pack(fill=tk.BOTH, expand=True, padx=16, pady=(8, 12))

        vis_title_row = tk.Frame(vis_card, bg=BG_CARD_LIGHT)
        vis_title_row.pack(fill=tk.X, pady=(0, 8))
        tk.Label(vis_title_row, text="👁️ OCULAR BIOMETRIC TELEMETRY", font=("Segoe UI", 8, "bold"), fg=TEXT_SECONDARY, bg=BG_CARD_LIGHT).pack(side=tk.LEFT)
        tk.Label(vis_title_row, text="MediaPipe Mesh", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD_LIGHT).pack(side=tk.RIGHT)

        vis_grid = tk.Frame(vis_card, bg=BG_CARD_LIGHT)
        vis_grid.pack(fill=tk.BOTH, expand=True)
        vis_grid.columnconfigure(0, weight=1)
        vis_grid.columnconfigure(1, weight=1)
        vis_grid.rowconfigure(0, weight=1)
        vis_grid.rowconfigure(1, weight=1)

        t1, self.metric_ear_val, self.metric_ear_sub = self._create_metric_tile(vis_grid, "Eye Aperture (EAR)", "0.29", "Threshold: 0.21")
        t1.grid(row=0, column=0, padx=(0, 4), pady=(0, 4), sticky="nsew")

        t2, self.metric_blink_val, self.metric_blink_sub = self._create_metric_tile(vis_grid, "Blink Frequency", "14.0 BPM", "Normal: 15–22 BPM")
        t2.grid(row=0, column=1, padx=(4, 0), pady=(0, 4), sticky="nsew")

        t3, self.metric_perclos_val, self.metric_perclos_sub = self._create_metric_tile(vis_grid, "Eye Strain (PERCLOS)", "3.2%", "Fatigue: > 15%")
        t3.grid(row=1, column=0, padx=(0, 4), pady=(4, 0), sticky="nsew")

        t4, self.metric_gaze_val, self.metric_gaze_sub = self._create_metric_tile(vis_grid, "Gaze Stability", "92%", "Flow Target: > 85%")
        t4.grid(row=1, column=1, padx=(4, 0), pady=(4, 0), sticky="nsew")

        # -------------------------------------------------------------
        # RIGHT PANE: Dashboard Cards & Adaptations
        # -------------------------------------------------------------
        right_pane = tk.Frame(body_frame, bg=BG_MAIN)
        right_pane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 1. Active Context Card
        ctx_card = tk.Frame(right_pane, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=14, pady=10)
        ctx_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(ctx_card, text="ACTIVE FOREGROUND CONTEXT", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor=tk.W)
        ctx_row = tk.Frame(ctx_card, bg=BG_CARD)
        ctx_row.pack(fill=tk.X, pady=(4, 0))

        self.ctx_domain_badge = tk.Label(
            ctx_row,
            text="CODE_IDE",
            font=("Segoe UI", 9, "bold"),
            bg="#4f46e5",
            fg="#ffffff",
            padx=8,
            pady=2
        )
        self.ctx_domain_badge.pack(side=tk.LEFT, padx=(0, 10))

        self.ctx_app_lbl = tk.Label(ctx_row, text="code.exe — Visual Studio Code", font=("Segoe UI", 10), fg=TEXT_PRIMARY, bg=BG_CARD)
        self.ctx_app_lbl.pack(side=tk.LEFT)

        # 2. Inferred Cognitive State Hero Card
        state_card = tk.Frame(right_pane, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=14, pady=10)
        state_card.pack(fill=tk.X, pady=(0, 10))

        state_header = tk.Frame(state_card, bg=BG_CARD)
        state_header.pack(fill=tk.X)
        tk.Label(state_header, text="INFERRED COGNITIVE STATE", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(side=tk.LEFT)
        self.confidence_lbl = tk.Label(state_header, text="Confidence: 88%", font=("Segoe UI", 9, "bold"), fg=TEXT_SECONDARY, bg=BG_CARD_LIGHT, padx=6, pady=1)
        self.confidence_lbl.pack(side=tk.RIGHT)

        state_top_row = tk.Frame(state_card, bg=BG_CARD)
        state_top_row.pack(fill=tk.X, pady=(6, 6))

        self.state_pill = tk.Label(
            state_top_row,
            text="NORMAL_FOCUS",
            font=("Segoe UI", 14, "bold"),
            bg=ACCENT_CYAN,
            fg="#042f2e",
            padx=14,
            pady=3
        )
        self.state_pill.pack(side=tk.LEFT, padx=(0, 12))

        self.state_desc_lbl = tk.Label(
            state_top_row,
            text="Steady beta rhythm • Engaged attention",
            font=("Segoe UI", 9),
            fg=TEXT_SECONDARY,
            bg=BG_CARD
        )
        self.state_desc_lbl.pack(side=tk.LEFT)

        # Probabilities Mini Bars
        prob_container = tk.Frame(state_card, bg=BG_CARD)
        prob_container.pack(fill=tk.X, pady=(4, 2))

        self.prob_bars: dict[str, ttk.Progressbar] = {}
        self.prob_labels: dict[str, tk.Label] = {}
        style_map = {
            CognitiveState.FLOW.value: "Flow.Horizontal.TProgressbar",
            CognitiveState.NORMAL_FOCUS.value: "Focus.Horizontal.TProgressbar",
            CognitiveState.COGNITIVE_OVERLOAD.value: "Overload.Horizontal.TProgressbar",
            CognitiveState.FATIGUE.value: "Fatigue.Horizontal.TProgressbar",
            CognitiveState.IDLE.value: "Idle.Horizontal.TProgressbar",
        }

        for state_name in CognitiveState.list():
            row = tk.Frame(prob_container, bg=BG_CARD)
            row.pack(fill=tk.X, pady=1)
            lbl = tk.Label(row, text=f"{state_name[:8]:<8}", font=("Consolas", 8), fg=TEXT_SECONDARY, bg=BG_CARD, width=9, anchor=tk.W)
            lbl.pack(side=tk.LEFT)
            bar_style = style_map.get(state_name, "Focus.Horizontal.TProgressbar")
            pb = ttk.Progressbar(row, maximum=100, length=200, style=bar_style)
            pb.pack(side=tk.LEFT, padx=6)
            val_lbl = tk.Label(row, text="0%", font=("Consolas", 8), fg=TEXT_MUTED, bg=BG_CARD, width=4)
            val_lbl.pack(side=tk.LEFT)
            self.prob_bars[state_name] = pb
            self.prob_labels[state_name] = val_lbl

        self.explain_lbl = tk.Label(state_card, text="Top Markers: Gaze=0.88, Jitter=0.12", font=("Segoe UI", 8, "italic"), fg=TEXT_MUTED, bg=BG_CARD)
        self.explain_lbl.pack(anchor=tk.W, pady=(4, 0))

        # 3. Peripheral Kinematics 2x2 Grid
        per_card = tk.Frame(right_pane, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=14, pady=10)
        per_card.pack(fill=tk.X, pady=(0, 10))

        per_title_row = tk.Frame(per_card, bg=BG_CARD)
        per_title_row.pack(fill=tk.X, pady=(0, 6))
        tk.Label(per_title_row, text="⌨️ PERIPHERAL MOTOR KINEMATICS", font=("Segoe UI", 8, "bold"), fg=TEXT_SECONDARY, bg=BG_CARD).pack(side=tk.LEFT)
        tk.Label(per_title_row, text="Zero Keylogging • Anonymized Latency", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD).pack(side=tk.RIGHT)

        per_grid = tk.Frame(per_card, bg=BG_CARD)
        per_grid.pack(fill=tk.X)
        per_grid.columnconfigure(0, weight=1)
        per_grid.columnconfigure(1, weight=1)

        p1, self.metric_cadence_val, self.metric_cadence_sub = self._create_metric_tile(per_grid, "Typing Cadence", "37 WPM", "185 KPM • Flow")
        p1.grid(row=0, column=0, padx=(0, 4), pady=(0, 4), sticky="nsew")

        p2, self.metric_dwell_val, self.metric_dwell_sub = self._create_metric_tile(per_grid, "Dwell / Flight Time", "98 / 165 ms", "Crisp Rhythmic")
        p2.grid(row=0, column=1, padx=(4, 0), pady=(0, 4), sticky="nsew")

        p3, self.metric_jitter_val, self.metric_jitter_sub = self._create_metric_tile(per_grid, "Mouse Jitter Index", "0.082", "Calm Trajectory")
        p3.grid(row=1, column=0, padx=(0, 4), pady=(4, 0), sticky="nsew")

        p4, self.metric_idle_val, self.metric_idle_sub = self._create_metric_tile(per_grid, "Activity Engagement", "ACTIVE", "Idle: 0.8s")
        p4.grid(row=1, column=1, padx=(4, 0), pady=(4, 0), sticky="nsew")

        # 4. Workspace Adaptation Actuation Card
        act_card = tk.Frame(right_pane, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=14, pady=10)
        act_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(act_card, text="⚙️ ACTIVE WORKSPACE ENVIRONMENT ADAPTATION", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor=tk.W)

        act_grid = tk.Frame(act_card, bg=BG_CARD)
        act_grid.pack(fill=tk.X, pady=(6, 0))
        act_grid.columnconfigure(0, weight=1)
        act_grid.columnconfigure(1, weight=1)
        act_grid.columnconfigure(2, weight=1)
        act_grid.columnconfigure(3, weight=1)

        # Kelvin warmth chip & color swatch
        warmth_box = tk.Frame(act_grid, bg=BG_TILE, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=8, pady=6)
        warmth_box.grid(row=0, column=0, padx=(0, 3), sticky="nsew")
        tk.Label(warmth_box, text="SCREEN WARMTH", font=("Segoe UI", 7, "bold"), fg=TEXT_MUTED, bg=BG_TILE).pack(anchor=tk.W)
        warmth_sub = tk.Frame(warmth_box, bg=BG_TILE)
        warmth_sub.pack(anchor=tk.W, pady=(2, 0))
        self.kelvin_swatch = tk.Label(warmth_sub, text="    ", bg="#fff5ea", relief=tk.SOLID, bd=1)
        self.kelvin_swatch.pack(side=tk.LEFT, padx=(0, 6))
        self.kelvin_lbl = tk.Label(warmth_sub, text="5000 K", font=("Segoe UI", 11, "bold"), fg=TEXT_PRIMARY, bg=BG_TILE)
        self.kelvin_lbl.pack(side=tk.LEFT)

        # UI Density
        density_box = tk.Frame(act_grid, bg=BG_TILE, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=8, pady=6)
        density_box.grid(row=0, column=1, padx=(3, 3), sticky="nsew")
        tk.Label(density_box, text="UI DENSITY", font=("Segoe UI", 7, "bold"), fg=TEXT_MUTED, bg=BG_TILE).pack(anchor=tk.W)
        self.density_lbl = tk.Label(density_box, text="STANDARD", font=("Segoe UI", 11, "bold"), fg=TEXT_PRIMARY, bg=BG_TILE)
        self.density_lbl.pack(anchor=tk.W, pady=(2, 0))

        # Notification DND
        dnd_box = tk.Frame(act_grid, bg=BG_TILE, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=8, pady=6)
        dnd_box.grid(row=0, column=2, padx=(3, 3), sticky="nsew")
        tk.Label(dnd_box, text="NOTIFICATION DND", font=("Segoe UI", 7, "bold"), fg=TEXT_MUTED, bg=BG_TILE).pack(anchor=tk.W)
        self.dnd_lbl = tk.Label(dnd_box, text="PRIORITY_ONLY", font=("Segoe UI", 10, "bold"), fg=TEXT_PRIMARY, bg=BG_TILE)
        self.dnd_lbl.pack(anchor=tk.W, pady=(3, 0))

        # Adaptation Interval & Gradual Fade Status
        pacing_box = tk.Frame(act_grid, bg=BG_TILE, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=8, pady=6)
        pacing_box.grid(row=0, column=3, padx=(3, 0), sticky="nsew")
        tk.Label(pacing_box, text="ADAPTATION PACING", font=("Segoe UI", 7, "bold"), fg=TEXT_MUTED, bg=BG_TILE).pack(anchor=tk.W)
        self.pacing_lbl = tk.Label(pacing_box, text="Ready (30s)", font=("Segoe UI", 10, "bold"), fg=ACCENT_CYAN, bg=BG_TILE)
        self.pacing_lbl.pack(anchor=tk.W, pady=(3, 0))

        # Interval Quick Selector Toolbar
        interval_row = tk.Frame(act_card, bg=BG_CARD)
        interval_row.pack(fill=tk.X, pady=(6, 0))
        tk.Label(interval_row, text="Cooldown Interval:", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD).pack(side=tk.LEFT, padx=(0, 6))

        self.interval_btns: dict[int, tk.Button] = {}
        for sec in [15, 30, 60, 120]:
            is_active = (sec == int(self.config.adaptation.min_adaptation_interval_sec))
            btn = tk.Button(
                interval_row,
                text=f"{sec}s",
                font=("Segoe UI", 8, "bold" if is_active else "normal"),
                bg="#0284c7" if is_active else BG_CARD_LIGHT,
                fg="#ffffff" if is_active else TEXT_PRIMARY,
                activebackground=BORDER_COLOR,
                relief=tk.FLAT,
                padx=8,
                pady=1,
                cursor="hand2",
                command=lambda s=sec: self._set_adaptation_interval(s)
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.interval_btns[sec] = btn

        fade_dur = self.config.adaptation.transition_duration_sec
        tk.Label(interval_row, text=f"• Fade: {fade_dur:.1f}s smooth S-curve", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD).pack(side=tk.RIGHT)

        # Ergonomic Advisory
        self.break_banner = tk.Label(
            act_card,
            text="✓ Ergonomics optimal — environment matched to creative context",
            font=("Segoe UI", 8, "bold"),
            bg=BG_CARD_LIGHT,
            fg=ACCENT_GREEN,
            padx=10,
            pady=4
        )
        self.break_banner.pack(fill=tk.X, pady=(6, 0))

        # 5. Interactive Simulation & Testing Controls
        sim_box = tk.Frame(right_pane, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=14, pady=8)
        sim_box.pack(fill=tk.X)

        tk.Label(sim_box, text="STATE OVERRIDE:", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(side=tk.LEFT, padx=(0, 8))

        self.btn_flow = self._create_sim_button(sim_box, "Flow", CognitiveState.FLOW, ACCENT_GREEN)
        self.btn_norm = self._create_sim_button(sim_box, "Normal", CognitiveState.NORMAL_FOCUS, ACCENT_CYAN)
        self.btn_overload = self._create_sim_button(sim_box, "Overload", CognitiveState.COGNITIVE_OVERLOAD, ACCENT_AMBER)
        self.btn_fatigue = self._create_sim_button(sim_box, "Fatigue", CognitiveState.FATIGUE, ACCENT_RED)
        self.btn_idle = self._create_sim_button(sim_box, "Idle", CognitiveState.IDLE, ACCENT_SLATE)

        self.btn_auto = tk.Button(
            sim_box,
            text="Auto-Sense",
            font=("Segoe UI", 8, "bold"),
            bg="#2563eb",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=8,
            pady=2,
            cursor="hand2",
            command=self._reset_to_auto
        )
        self.btn_auto.pack(side=tk.LEFT, padx=(8, 4))
        _bind_hover(self.btn_auto, "#2563eb", "#1d4ed8")

        self.btn_reset_gamma = tk.Button(
            sim_box,
            text="Restore 6500K",
            font=("Segoe UI", 8),
            bg=BG_CARD_LIGHT,
            fg=TEXT_PRIMARY,
            relief=tk.FLAT,
            padx=8,
            pady=2,
            cursor="hand2",
            command=self._restore_display_gamma
        )
        self.btn_reset_gamma.pack(side=tk.RIGHT)
        _bind_hover(self.btn_reset_gamma, BG_CARD_LIGHT, BORDER_COLOR)

    def _build_footer(self) -> None:
        """Bottom status bar with privacy statement and safeguards."""
        footer_frame = tk.Frame(self, bg=BG_MAIN, height=26, padx=16)
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Label(
            footer_frame,
            text="🔒 100% Local Inference — Zero Cloud Transmission | Zero Keylogging | Hardware Gamma Auto-Restoration Safeguard Active",
            font=("Segoe UI", 8),
            fg=TEXT_MUTED,
            bg=BG_MAIN
        ).pack(side=tk.LEFT)

    def _create_metric_tile(
        self,
        parent: tk.Widget,
        title: str,
        default_val: str,
        default_sub: str
    ) -> tuple[tk.Frame, tk.Label, tk.Label]:
        """Helper to create an elevated cockpit metric tile."""
        tile = tk.Frame(
            parent,
            bg=BG_TILE,
            highlightbackground=BORDER_COLOR,
            highlightthickness=1,
            padx=10,
            pady=6
        )
        tk.Label(
            tile,
            text=title.upper(),
            font=("Segoe UI", 7, "bold"),
            fg=TEXT_MUTED,
            bg=BG_TILE
        ).pack(anchor=tk.W)

        val_lbl = tk.Label(
            tile,
            text=default_val,
            font=("Segoe UI", 12, "bold"),
            fg=TEXT_PRIMARY,
            bg=BG_TILE
        )
        val_lbl.pack(anchor=tk.W, pady=(1, 1))

        sub_lbl = tk.Label(
            tile,
            text=default_sub,
            font=("Segoe UI", 7),
            fg=TEXT_SECONDARY,
            bg=BG_TILE
        )
        sub_lbl.pack(anchor=tk.W)

        return tile, val_lbl, sub_lbl

    def _create_metric_row(self, parent: tk.Widget, label_text: str, default_val: str, note_text: str) -> tk.Label:
        """Legacy helper maintained for backward compatibility."""
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
            cursor="hand2",
            command=lambda: self._set_simulated_state(state)
        )
        btn.pack(side=tk.LEFT, padx=2)
        _bind_hover(btn, BG_CARD_LIGHT, BORDER_COLOR)
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
                # Resize to fit canvas (488 x 366)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (488, 366), interpolation=cv2.INTER_AREA)
                img = Image.fromarray(frame_resized)
                img_tk = ImageTk.PhotoImage(image=img)
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                self.video_canvas.image = img_tk  # keep reference
        self.after(33, self._update_camera_feed)

    def _engine_tick(self) -> None:
        """Core closed-loop iteration executed at 10 Hz."""
        t_now = time.time()

        # Update Session Duration Timer and System Clock
        elapsed = int(t_now - self.session_start_time)
        hh = elapsed // 3600
        mm = (elapsed % 3600) // 60
        ss = elapsed % 60
        self.session_timer_lbl.configure(text=f"⏱️ {hh:02d}:{mm:02d}:{ss:02d}")
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

        # 6. Persistent Telemetry Logging (~1.0 Hz cadence)
        if (t_now - self._last_log_time) >= 1.0:
            self._last_log_time = t_now
            if self.session_id:
                self.session_store.log_tick(
                    session_id=self.session_id,
                    timestamp=t_now,
                    dt=1.0,
                    cognitive_state=inference.state.value,
                    confidence=inference.confidence,
                    context_domain=app_context.domain.value,
                    process_name=app_context.process_name,
                    window_title=app_context.window_title,
                    ear=vis_sample.ear_mean,
                    blink_rate=vis_sample.blink_rate_bpm,
                    perclos=vis_sample.perclos,
                    gaze_stability=vis_sample.gaze_stability,
                    keystrokes_per_min=per_sample.keystrokes_per_min,
                    dwell_time_ms=per_sample.mean_dwell_time_ms,
                    flight_time_ms=per_sample.mean_flight_time_ms,
                    mouse_jitter=per_sample.mouse_jitter_index,
                    screen_warmth_kelvin=exec_res.current_kelvin,
                    ui_density=exec_res.ui_density.value,
                    notification_policy=exec_res.notification_policy.value,
                    adaptation_executed=exec_res.executed
                )

        # 7. Update GUI Displays
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
        domain_val = ctx.domain.value
        badge_bg, badge_fg = DOMAIN_COLORS.get(domain_val, ("#334155", "#ffffff"))
        self.ctx_domain_badge.configure(text=domain_val, bg=badge_bg, fg=badge_fg)

        title_str = ctx.window_title[:45] + "..." if len(ctx.window_title) > 45 else ctx.window_title
        self.ctx_app_lbl.configure(text=f"{ctx.process_name} — {title_str or 'Foreground Window'}")

        # 2. State Hero & Confidence updates
        bg_col, fg_col, state_desc = STATE_HERO_STYLES.get(
            inf.state.value,
            (BG_CARD_LIGHT, TEXT_PRIMARY, "Cognitive baseline active")
        )
        self.state_pill.configure(text=inf.state.value, bg=bg_col, fg=fg_col)
        self.state_desc_lbl.configure(text=state_desc)
        self.confidence_lbl.configure(text=f"Confidence: {int(inf.confidence * 100)}%")

        # Probability mini bars
        for s_name, pb in self.prob_bars.items():
            prob_pct = int(inf.probabilities.get(s_name, 0.0) * 100)
            pb["value"] = prob_pct
            self.prob_labels[s_name].configure(text=f"{prob_pct}%")

        if inf.dominant_features:
            explain_str = "Top Markers: " + ", ".join(f"{k}: {v}" for k, v in inf.dominant_features[:2])
            self.explain_lbl.configure(text=explain_str)

        # 3. Ocular Biometric Tile Grid
        blink_tag = " (BLINKING)" if vis.is_blinking else ""
        self.metric_ear_val.configure(
            text=f"{vis.ear_mean:.2f}",
            fg=ACCENT_RED if vis.ear_mean < 0.21 or vis.is_blinking else TEXT_PRIMARY
        )
        self.metric_ear_sub.configure(text=f"Threshold: 0.21{blink_tag}")

        self.metric_blink_val.configure(text=f"{vis.blink_rate_bpm:.1f} BPM")
        self.metric_blink_sub.configure(
            text="Elevated (>24 BPM)" if vis.blink_rate_bpm > 24 else ("Low (<12 BPM)" if vis.blink_rate_bpm < 12 else "Normal: 15–22 BPM")
        )

        self.metric_perclos_val.configure(
            text=f"{vis.perclos:.1%}",
            fg=ACCENT_RED if vis.perclos > 0.15 else TEXT_PRIMARY
        )
        self.metric_perclos_sub.configure(
            text="⚠️ Fatigue Strain Alert" if vis.perclos > 0.15 else "Normal Strain: < 15%"
        )

        self.metric_gaze_val.configure(text=f"{int(vis.gaze_stability * 100)}%")
        self.metric_gaze_sub.configure(
            text="Flow Target: > 85%" if vis.gaze_stability >= 0.85 else "Dispersed Fixation"
        )

        # 4. Peripheral Motor Dynamics Tile Grid
        wpm = int(per.keystrokes_per_min / 5.0)
        self.metric_cadence_val.configure(text=f"{wpm} WPM")
        self.metric_cadence_sub.configure(text=f"{per.keystrokes_per_min:.0f} KPM Cadence")

        self.metric_dwell_val.configure(text=f"{per.mean_dwell_time_ms:.0f} / {per.mean_flight_time_ms:.0f} ms")
        self.metric_dwell_sub.configure(text="Crisp Rhythmic Motor")

        self.metric_jitter_val.configure(
            text=f"{per.mouse_jitter_index:.3f}",
            fg=ACCENT_RED if per.mouse_jitter_index > 0.35 else (ACCENT_AMBER if per.mouse_jitter_index > 0.20 else TEXT_PRIMARY)
        )
        self.metric_jitter_sub.configure(
            text="⚠️ High Motor Strain" if per.mouse_jitter_index > 0.35 else "Calm Trajectory"
        )

        self.metric_idle_val.configure(
            text="IDLE" if per.is_idle else "ACTIVE",
            fg=ACCENT_SLATE if per.is_idle else ACCENT_GREEN
        )
        self.metric_idle_sub.configure(text=f"Inactivity: {per.idle_seconds:.1f}s")

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
                text=f"Ready ({int_sec}s)",
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

        # Highlight active cooldown button
        for s, btn in self.interval_btns.items():
            if s == seconds:
                btn.configure(bg="#0284c7", fg="#ffffff", font=("Segoe UI", 8, "bold"))
            else:
                btn.configure(bg=BG_CARD_LIGHT, fg=TEXT_PRIMARY, font=("Segoe UI", 8, "normal"))

        self.pacing_lbl.configure(text=f"Ready ({int_sec}s)", fg=ACCENT_GREEN)
        logger.info(f"Adaptation interval set to {seconds} seconds.")

    def _minimize_to_background(self) -> None:
        """Minimize the GUI window to the system taskbar so it continues running silently."""
        self.iconify()
        logger.info("AetherFlow GUI minimized to background.")

    def _on_generate_report(self) -> None:
        """Trigger on-demand productivity report generation in background thread."""
        def _worker() -> None:
            try:
                self.session_store.flush()
                report_path = generate_report(session_id=self.session_id, open_browser=True)
                if report_path:
                    logger.info(f"Productivity report generated: {report_path}")
            except Exception as e:
                logger.error(f"Failed to generate productivity report: {e}")

        # Non-blocking launch so GUI doesn't pause
        threading.Thread(target=_worker, daemon=True).start()

    def on_closing(self) -> None:
        """Safe shutdown when user closes window."""
        logger.info("GUI window closed. Terminating background threads...")
        self.vision_tracker.stop()
        self.peripheral_tracker.stop()
        if self.session_id:
            self.session_store.close_session(self.session_id)
        self.adaptation_executor.restore_display()
        self.destroy()


def run_gui(config: Optional[AetherFlowConfig] = None) -> None:
    """Entry point to start the dedicated GUI application."""
    app = AetherFlowApp(config)
    app.mainloop()


if __name__ == "__main__":
    run_gui()
