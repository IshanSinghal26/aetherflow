"""Dedicated Desktop GUI Frontend for AetherFlow.

Provides a clean, human-crafted 4-tab control room (Linear / Raycast / Apple macOS aesthetic):
1. [Overview]: Calm hero status card, 4 quick-glance metrics, and live HUD camera viewport.
2. [Biometrics]: Ocular health baselines, motor kinematics, and classifier probability bars.
3. [Environment]: Screen warmth Kelvin swatch, UI density, cooldown pacing, and state overrides.
4. [Session Report]: Executive KPI tiles, workday cognitive spectrum bar, and application table.
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
# Visual Palette (Linear / Raycast Clean Matte Dark Theme)
# =====================================================================
BG_MAIN = "#0f0f11"        # Matte Charcoal Void
BG_CARD = "#18181b"        # Zinc-900 Primary Panel
BG_CARD_LIGHT = "#222226"  # Zinc-850 Container
BG_TILE = "#18181b"        # Clean Matte Metric Tile
BORDER_COLOR = "#27272a"   # Clean 1px Zinc Border
BORDER_ACCENT = "#10b981"  # Emerald Accent

TEXT_PRIMARY = "#f4f4f5"   # Zinc-100 High Contrast
TEXT_SECONDARY = "#a1a1aa" # Zinc-400 Readable Secondary
TEXT_MUTED = "#71717a"     # Zinc-500 Muted Captions

# Accent States
ACCENT_CYAN = "#0284c7"    # Focus / Sky Accent
ACCENT_BLUE = "#0284c7"    # Primary Action
ACCENT_GREEN = "#10b981"   # Flow / Emerald
ACCENT_AMBER = "#f59e0b"   # Overload / Amber
ACCENT_RED = "#ef4444"     # Fatigue / Red
ACCENT_PURPLE = "#8b5cf6"  # Context Accent
ACCENT_SLATE = "#64748b"   # Idle / Standby

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
    CognitiveState.FLOW.value: (ACCENT_GREEN, "#f4f4f5", "Sustained attention, calm motor rhythm. Flow immersion active."),
    CognitiveState.NORMAL_FOCUS.value: (ACCENT_CYAN, "#f4f4f5", "Steady engagement with light context switching."),
    CognitiveState.COGNITIVE_OVERLOAD.value: (ACCENT_AMBER, "#f4f4f5", "Erratic input and rapid switching — consider slowing down."),
    CognitiveState.FATIGUE.value: (ACCENT_RED, "#f4f4f5", "Long blinks and drifting gaze detected. Rest recommended."),
    CognitiveState.IDLE.value: (ACCENT_SLATE, "#f4f4f5", "Sensor standby — no user input detected."),
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
    """Main desktop GUI window for AetherFlow featuring 4-tab Linear navigation."""

    def __init__(self, config: Optional[AetherFlowConfig] = None) -> None:
        super().__init__()
        self.config = config or AetherFlowConfig()

        self.title("AetherFlow — Adaptive Focus Engine")
        self.geometry("1180x760")
        self.minsize(1060, 700)
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
        self._build_tabs()
        self._build_footer()

        # Activate default tab: Overview
        self.switch_tab("overview")

        # Start background sensors
        self.vision_tracker.start()
        self.peripheral_tracker.start()

        # Start background Web Dashboard Server (localhost:8765)
        try:
            from aetherflow.web_server import start_web_server
            self._web_server = start_web_server()
        except Exception as e:
            logger.debug(f"Web dashboard server init: {e}")
            self._web_server = None

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
        """Top Linear navbar with brand, 4 tab buttons, live session timer, and action buttons."""
        header_frame = tk.Frame(self, bg=BG_CARD, height=56, padx=20, highlightbackground=BORDER_COLOR, highlightthickness=1)
        header_frame.pack(side=tk.TOP, fill=tk.X)
        header_frame.pack_propagate(False)

        # Brand Logo and Title
        brand_box = tk.Frame(header_frame, bg=BG_CARD)
        brand_box.pack(side=tk.LEFT)

        logo_lbl = tk.Label(brand_box, text="⚡", font=("Segoe UI", 14, "bold"), fg=ACCENT_GREEN, bg=BG_CARD)
        logo_lbl.pack(side=tk.LEFT, padx=(0, 6))

        title_lbl = tk.Label(brand_box, text="AetherFlow", font=("Segoe UI", 13, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD)
        title_lbl.pack(side=tk.LEFT, padx=(0, 20))

        # 4 Central Navigation Tab Buttons
        nav_box = tk.Frame(header_frame, bg=BG_CARD)
        nav_box.pack(side=tk.LEFT)

        self.tab_buttons: dict[str, tk.Button] = {}
        tabs = [
            ("overview", "Overview"),
            ("biometrics", "Biometrics"),
            ("environment", "Environment"),
            ("report", "Session Report"),
        ]

        for tab_id, tab_label in tabs:
            btn = tk.Button(
                nav_box,
                text=tab_label,
                font=("Segoe UI", 9, "bold"),
                bg=BG_CARD,
                fg=TEXT_MUTED,
                activebackground=BG_CARD_LIGHT,
                activeforeground=TEXT_PRIMARY,
                relief=tk.FLAT,
                padx=12,
                pady=4,
                cursor="hand2",
                command=lambda tid=tab_id: self.switch_tab(tid)
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.tab_buttons[tab_id] = btn

        # Right Header: Session Timer, Status Pill, Web Dashboard & Actions
        right_box = tk.Frame(header_frame, bg=BG_CARD)
        right_box.pack(side=tk.RIGHT)

        self.session_timer_lbl = tk.Label(
            right_box,
            text="⏱ 00:00:00",
            font=("Consolas", 10),
            fg=TEXT_SECONDARY,
            bg=BG_CARD
        )
        self.session_timer_lbl.pack(side=tk.LEFT, padx=(0, 12))

        # Status Pill
        pill_frame = tk.Frame(right_box, bg=BG_CARD_LIGHT, highlightbackground="#065f46", highlightthickness=1, padx=8, pady=2)
        pill_frame.pack(side=tk.LEFT, padx=(0, 12))
        self.status_dot = tk.Label(pill_frame, text="●", font=("Segoe UI", 9), fg=ACCENT_GREEN, bg=BG_CARD_LIGHT)
        self.status_dot.pack(side=tk.LEFT, padx=(0, 4))
        self.status_lbl = tk.Label(pill_frame, text="Active", font=("Segoe UI", 8, "bold"), fg=ACCENT_GREEN, bg=BG_CARD_LIGHT)
        self.status_lbl.pack(side=tk.LEFT)

        # Web Dashboard Button
        self.btn_web_dash = tk.Button(
            right_box,
            text="🌐 Web Dashboard",
            font=("Segoe UI", 8, "bold"),
            bg=BG_CARD_LIGHT,
            fg=TEXT_PRIMARY,
            activebackground=BORDER_COLOR,
            relief=tk.FLAT,
            padx=10,
            pady=3,
            cursor="hand2",
            command=self._open_web_dashboard
        )
        self.btn_web_dash.pack(side=tk.LEFT, padx=(0, 6))
        _bind_hover(self.btn_web_dash, BG_CARD_LIGHT, BORDER_COLOR)

        # Minimize to Tray Button
        self.btn_min_bg = tk.Button(
            right_box,
            text="🗕 Minimize",
            font=("Segoe UI", 8),
            bg=BG_CARD_LIGHT,
            fg=TEXT_PRIMARY,
            activebackground=BORDER_COLOR,
            relief=tk.FLAT,
            padx=8,
            pady=3,
            cursor="hand2",
            command=self._minimize_to_background
        )
        self.btn_min_bg.pack(side=tk.LEFT)
        _bind_hover(self.btn_min_bg, BG_CARD_LIGHT, BORDER_COLOR)

    def _build_tabs(self) -> None:
        """Construct the 4 tab page containers."""
        self.tab_container = tk.Frame(self, bg=BG_MAIN, padx=24, pady=20)
        self.tab_container.pack(fill=tk.BOTH, expand=True)

        self.tab_frames: dict[str, tk.Frame] = {}

        # 1. Overview Tab
        self.tab_frames["overview"] = self._build_overview_tab()

        # 2. Biometrics Tab
        self.tab_frames["biometrics"] = self._build_biometrics_tab()

        # 3. Environment Tab
        self.tab_frames["environment"] = self._build_environment_tab()

        # 4. Session Report Tab
        self.tab_frames["report"] = self._build_report_tab()

    def switch_tab(self, tab_id: str) -> None:
        """Switch active view cleanly with highlighted navbar tab."""
        for tid, frame in self.tab_frames.items():
            if tid == tab_id:
                frame.pack(fill=tk.BOTH, expand=True)
            else:
                frame.pack_forget()

        for tid, btn in self.tab_buttons.items():
            if tid == tab_id:
                btn.configure(bg=BG_CARD_LIGHT, fg=TEXT_PRIMARY)
            else:
                btn.configure(bg=BG_CARD, fg=TEXT_MUTED)

    # -------------------------------------------------------------
    # TAB 1: OVERVIEW (Calm, elegant first impression)
    # -------------------------------------------------------------
    def _build_overview_tab(self) -> tk.Frame:
        page = tk.Frame(self.tab_container, bg=BG_MAIN)

        # 5-Column Split (Left 3 cols: Hero + 4 stats; Right 2 cols: HUD camera)
        left_box = tk.Frame(page, bg=BG_MAIN)
        left_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 16))

        # Hero Card
        hero_panel = tk.Frame(left_box, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=24, pady=20)
        hero_panel.pack(fill=tk.X, pady=(0, 14))

        tk.Label(hero_panel, text="CURRENT COGNITIVE STATE", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor=tk.W)

        state_title_row = tk.Frame(hero_panel, bg=BG_CARD)
        state_title_row.pack(anchor=tk.W, pady=(8, 4))
        self.overview_state_dot = tk.Label(state_title_row, text="●", font=("Segoe UI", 14), fg=ACCENT_GREEN, bg=BG_CARD)
        self.overview_state_dot.pack(side=tk.LEFT, padx=(0, 8))
        self.overview_state_title = tk.Label(state_title_row, text="Deep Flow", font=("Segoe UI", 22, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD)
        self.overview_state_title.pack(side=tk.LEFT)

        self.overview_state_desc = tk.Label(
            hero_panel,
            text="Sustained attention, calm motor rhythm. Workspace adaptations active.",
            font=("Segoe UI", 10),
            fg=TEXT_SECONDARY,
            bg=BG_CARD
        )
        self.overview_state_desc.pack(anchor=tk.W)

        # Context Pill
        ctx_frame = tk.Frame(hero_panel, bg=BG_MAIN, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=10, pady=3)
        ctx_frame.pack(anchor=tk.W, pady=(16, 0))
        self.ctx_domain_badge = tk.Label(ctx_frame, text="CODE_IDE", font=("Segoe UI", 8, "bold"), fg=ACCENT_CYAN, bg=BG_MAIN)
        self.ctx_domain_badge.pack(side=tk.LEFT, padx=(0, 8))
        self.ctx_app_lbl = tk.Label(ctx_frame, text="code.exe — VS Code (Development)", font=("Consolas", 8), fg=TEXT_MUTED, bg=BG_MAIN)
        self.ctx_app_lbl.pack(side=tk.LEFT)

        # 4 Quick Glance KPI Cards (2x2)
        grid_2x2 = tk.Frame(left_box, bg=BG_MAIN)
        grid_2x2.pack(fill=tk.BOTH, expand=True)
        grid_2x2.columnconfigure(0, weight=1)
        grid_2x2.columnconfigure(1, weight=1)
        grid_2x2.rowconfigure(0, weight=1)
        grid_2x2.rowconfigure(1, weight=1)

        c1, self.overview_prod_val, self.overview_prod_sub = self._create_stat_card(grid_2x2, "Productivity", "88", "/ 100", "High cognitive immersion", ACCENT_GREEN)
        c1.grid(row=0, column=0, padx=(0, 6), pady=(0, 6), sticky="nsew")

        c2, self.overview_wpm_val, self.overview_wpm_sub = self._create_stat_card(grid_2x2, "Typing Cadence", "37", "WPM", "185 KPM steady cadence", ACCENT_GREEN)
        c2.grid(row=0, column=1, padx=(6, 0), pady=(0, 6), sticky="nsew")

        c3, self.overview_bpm_val, self.overview_bpm_sub = self._create_stat_card(grid_2x2, "Blink Frequency", "14.0", "BPM", "Normal ocular baseline", ACCENT_GREEN)
        c3.grid(row=1, column=0, padx=(0, 6), pady=(6, 0), sticky="nsew")

        c4, self.overview_kelvin_val, self.overview_kelvin_sub = self._create_stat_card(grid_2x2, "Screen Warmth", "5000", "K", "Hardware gamma active", ACCENT_GREEN)
        c4.grid(row=1, column=1, padx=(6, 0), pady=(6, 0), sticky="nsew")

        # Right Side: HUD Camera Viewport
        right_box = tk.Frame(page, bg=BG_CARD, width=420, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=16, pady=16)
        right_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        right_box.pack_propagate(False)

        cam_header = tk.Frame(right_box, bg=BG_CARD)
        cam_header.pack(fill=tk.X)
        tk.Label(cam_header, text="LIVE TELEMETRY HUD", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(side=tk.LEFT)

        cam_btn_box = tk.Frame(cam_header, bg=BG_CARD)
        cam_btn_box.pack(side=tk.RIGHT)

        self.btn_cam_toggle = tk.Button(
            cam_btn_box,
            text="Camera ON",
            font=("Segoe UI", 8, "bold"),
            bg=BG_CARD_LIGHT,
            fg=ACCENT_GREEN,
            activebackground=BORDER_COLOR,
            relief=tk.FLAT,
            padx=8,
            pady=2,
            cursor="hand2",
            command=self._toggle_camera
        )
        self.btn_cam_toggle.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_overlay_toggle = tk.Button(
            cam_btn_box,
            text="Mesh ON",
            font=("Segoe UI", 8),
            bg=BG_CARD_LIGHT,
            fg=TEXT_PRIMARY,
            activebackground=BORDER_COLOR,
            relief=tk.FLAT,
            padx=8,
            pady=2,
            cursor="hand2",
            command=self._toggle_overlay
        )
        self.btn_overlay_toggle.pack(side=tk.LEFT)

        # Video Canvas (388 x 291)
        self.video_canvas = tk.Canvas(right_box, bg="#000000", width=388, height=291, highlightthickness=1, highlightbackground=BORDER_COLOR)
        self.video_canvas.pack(fill=tk.BOTH, expand=True, pady=14)

        # Bottom HUD stats
        cam_footer = tk.Frame(right_box, bg=BG_CARD)
        cam_footer.pack(fill=tk.X)
        self.cam_ear_lbl = tk.Label(cam_footer, text="EAR 0.29", font=("Consolas", 8), fg=TEXT_SECONDARY, bg=BG_CARD)
        self.cam_ear_lbl.pack(side=tk.LEFT)
        self.cam_gaze_lbl = tk.Label(cam_footer, text="Gaze 92%", font=("Consolas", 8), fg=TEXT_SECONDARY, bg=BG_CARD)
        self.cam_gaze_lbl.pack(side=tk.LEFT, padx=(16, 0))
        self.cam_fps_lbl = tk.Label(cam_footer, text="30 fps", font=("Consolas", 8, "bold"), fg=ACCENT_GREEN, bg=BG_CARD)
        self.cam_fps_lbl.pack(side=tk.RIGHT)

        return page

    # -------------------------------------------------------------
    # TAB 2: BIOMETRICS (Deep-dive ocular & motor metrics)
    # -------------------------------------------------------------
    def _build_biometrics_tab(self) -> tk.Frame:
        page = tk.Frame(self.tab_container, bg=BG_MAIN)

        # Section 1: Ocular Health
        tk.Label(page, text="OCULAR BIOMETRICS", font=("Segoe UI", 9, "bold"), fg=TEXT_PRIMARY, bg=BG_MAIN).pack(anchor=tk.W, pady=(0, 2))
        tk.Label(page, text="Measured against clinical ocular and flow benchmarks.", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_MAIN).pack(anchor=tk.W, pady=(0, 8))

        grid_ocular = tk.Frame(page, bg=BG_MAIN)
        grid_ocular.pack(fill=tk.X, pady=(0, 16))
        for i in range(4):
            grid_ocular.columnconfigure(i, weight=1)

        o1, self.metric_ear_val, self.metric_ear_sub = self._create_stat_card(grid_ocular, "Eye Aperture (EAR)", "0.29", "", "Threshold: 0.21", ACCENT_GREEN)
        o1.grid(row=0, column=0, padx=(0, 4), sticky="nsew")

        o2, self.metric_blink_val, self.metric_blink_sub = self._create_stat_card(grid_ocular, "Blink Frequency", "14.0", "BPM", "Normal: 15–22 BPM", ACCENT_GREEN)
        o2.grid(row=0, column=1, padx=(4, 4), sticky="nsew")

        o3, self.metric_perclos_val, self.metric_perclos_sub = self._create_stat_card(grid_ocular, "Eye Strain (PERCLOS)", "3.2", "%", "Fatigue Alert: >15%", ACCENT_GREEN)
        o3.grid(row=0, column=2, padx=(4, 4), sticky="nsew")

        o4, self.metric_gaze_val, self.metric_gaze_sub = self._create_stat_card(grid_ocular, "Gaze Fixation", "92", "%", "Flow benchmark: >85%", ACCENT_GREEN)
        o4.grid(row=0, column=3, padx=(4, 0), sticky="nsew")

        # Section 2: Motor Kinematics
        tk.Label(page, text="PERIPHERAL MOTOR KINEMATICS", font=("Segoe UI", 9, "bold"), fg=TEXT_PRIMARY, bg=BG_MAIN).pack(anchor=tk.W, pady=(0, 2))
        tk.Label(page, text="Zero keylogging — anonymized keystroke timing and pointer dynamics.", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_MAIN).pack(anchor=tk.W, pady=(0, 8))

        grid_motor = tk.Frame(page, bg=BG_MAIN)
        grid_motor.pack(fill=tk.X, pady=(0, 16))
        for i in range(4):
            grid_motor.columnconfigure(i, weight=1)

        m1, self.metric_cadence_val, self.metric_cadence_sub = self._create_stat_card(grid_motor, "Typing Cadence", "37", "WPM", "185 KPM Cadence", ACCENT_GREEN)
        m1.grid(row=0, column=0, padx=(0, 4), sticky="nsew")

        m2, self.metric_dwell_val, self.metric_dwell_sub = self._create_stat_card(grid_motor, "Key Latency", "98 / 165", "ms", "Dwell / Flight time", ACCENT_GREEN)
        m2.grid(row=0, column=1, padx=(4, 4), sticky="nsew")

        m3, self.metric_jitter_val, self.metric_jitter_sub = self._create_stat_card(grid_motor, "Mouse Jitter", "0.082", "", "Calm trajectory", ACCENT_GREEN)
        m3.grid(row=0, column=2, padx=(4, 4), sticky="nsew")

        m4, self.metric_idle_val, self.metric_idle_sub = self._create_stat_card(grid_motor, "Activity Engagement", "ACTIVE", "", "Inactivity: 0.8s", ACCENT_GREEN)
        m4.grid(row=0, column=3, padx=(4, 0), sticky="nsew")

        # Section 3: State Probability Bars
        prob_panel = tk.Frame(page, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=20, pady=16)
        prob_panel.pack(fill=tk.X)

        tk.Label(prob_panel, text="COGNITIVE STATE PROBABILITY DISTRIBUTION", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor=tk.W, pady=(0, 10))

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
            row = tk.Frame(prob_panel, bg=BG_CARD)
            row.pack(fill=tk.X, pady=2)
            lbl = tk.Label(row, text=f"{state_name.title():<12}", font=("Segoe UI", 8), fg=TEXT_SECONDARY, bg=BG_CARD, width=12, anchor=tk.W)
            lbl.pack(side=tk.LEFT)
            bar_style = style_map.get(state_name, "Focus.Horizontal.TProgressbar")
            pb = ttk.Progressbar(row, maximum=100, length=300, style=bar_style)
            pb.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
            val_lbl = tk.Label(row, text="0%", font=("Consolas", 8), fg=TEXT_MUTED, bg=BG_CARD, width=6)
            val_lbl.pack(side=tk.RIGHT)
            self.prob_bars[state_name] = pb
            self.prob_labels[state_name] = val_lbl

        return page

    # -------------------------------------------------------------
    # TAB 3: ENVIRONMENT (Screen warmth, adaptations, overrides)
    # -------------------------------------------------------------
    def _build_environment_tab(self) -> tk.Frame:
        page = tk.Frame(self.tab_container, bg=BG_MAIN)
        page.columnconfigure(0, weight=1)
        page.columnconfigure(1, weight=1)
        page.rowconfigure(0, weight=1)
        page.rowconfigure(1, weight=1)

        # 1. Screen Warmth Actuation
        p1 = tk.Frame(page, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=20, pady=16)
        p1.grid(row=0, column=0, padx=(0, 8), pady=(0, 8), sticky="nsew")

        tk.Label(p1, text="SCREEN WARMTH ACTUATION", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor=tk.W)
        w_row = tk.Frame(p1, bg=BG_CARD)
        w_row.pack(anchor=tk.W, pady=(12, 10))
        self.kelvin_swatch = tk.Label(w_row, text="      ", bg="#fff4e8", relief=tk.SOLID, bd=1)
        self.kelvin_swatch.pack(side=tk.LEFT, padx=(0, 12))
        self.kelvin_lbl = tk.Label(w_row, text="5000 K", font=("Segoe UI", 20, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD)
        self.kelvin_lbl.pack(side=tk.LEFT)

        # Quick Kelvin Selector Buttons
        k_btn_row = tk.Frame(p1, bg=BG_CARD)
        k_btn_row.pack(anchor=tk.W, pady=(6, 0))
        for k_val in [3000, 4200, 5000, 6500]:
            btn = tk.Button(
                k_btn_row,
                text=f"{k_val}K",
                font=("Segoe UI", 8),
                bg=BG_CARD_LIGHT,
                fg=TEXT_PRIMARY,
                activebackground=BORDER_COLOR,
                relief=tk.FLAT,
                padx=8,
                pady=2,
                cursor="hand2",
                command=lambda k=k_val: self._apply_kelvin_direct(k)
            )
            btn.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_reset_gamma = tk.Button(
            p1,
            text="Restore 6500K Pure Daylight",
            font=("Segoe UI", 8),
            bg=BG_CARD_LIGHT,
            fg=TEXT_PRIMARY,
            activebackground=BORDER_COLOR,
            relief=tk.FLAT,
            padx=10,
            pady=3,
            cursor="hand2",
            command=self._restore_display_gamma
        )
        self.btn_reset_gamma.pack(anchor=tk.W, pady=(12, 0))

        # 2. UI Density & Notifications
        p2 = tk.Frame(page, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=20, pady=16)
        p2.grid(row=0, column=1, padx=(8, 0), pady=(0, 8), sticky="nsew")

        tk.Label(p2, text="UI DENSITY & NOTIFICATIONS", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor=tk.W)
        d_row = tk.Frame(p2, bg=BG_CARD)
        d_row.pack(fill=tk.X, pady=(10, 8))
        tk.Label(d_row, text="Active UI Scaling:", font=("Segoe UI", 8), fg=TEXT_SECONDARY, bg=BG_CARD).pack(side=tk.LEFT)
        self.density_lbl = tk.Label(d_row, text="STANDARD", font=("Segoe UI", 10, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD)
        self.density_lbl.pack(side=tk.RIGHT)

        dnd_row = tk.Frame(p2, bg=BG_CARD)
        dnd_row.pack(fill=tk.X, pady=(0, 8))
        tk.Label(dnd_row, text="Notification Policy:", font=("Segoe UI", 8), fg=TEXT_SECONDARY, bg=BG_CARD).pack(side=tk.LEFT)
        self.dnd_lbl = tk.Label(dnd_row, text="PRIORITY_ONLY", font=("Segoe UI", 9, "bold"), fg=ACCENT_CYAN, bg=BG_CARD)
        self.dnd_lbl.pack(side=tk.RIGHT)

        self.break_banner = tk.Label(
            p2,
            text="✓ Ergonomics optimal — environment matched",
            font=("Segoe UI", 8, "bold"),
            bg=BG_CARD_LIGHT,
            fg=ACCENT_GREEN,
            padx=10,
            pady=4
        )
        self.break_banner.pack(fill=tk.X, pady=(6, 0))

        # 3. Adaptation Cooldown Pacing
        p3 = tk.Frame(page, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=20, pady=16)
        p3.grid(row=1, column=0, padx=(0, 8), pady=(8, 0), sticky="nsew")

        tk.Label(p3, text="COOLDOWN PACING INTERVAL", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor=tk.W)
        self.pacing_lbl = tk.Label(p3, text="Ready (30s interval)", font=("Segoe UI", 10, "bold"), fg=ACCENT_GREEN, bg=BG_CARD)
        self.pacing_lbl.pack(anchor=tk.W, pady=(8, 8))

        interval_row = tk.Frame(p3, bg=BG_CARD)
        interval_row.pack(anchor=tk.W)
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
                pady=2,
                cursor="hand2",
                command=lambda s=sec: self._set_adaptation_interval(s)
            )
            btn.pack(side=tk.LEFT, padx=(0, 6))
            self.interval_btns[sec] = btn

        # 4. Cognitive Simulation Override
        p4 = tk.Frame(page, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=20, pady=16)
        p4.grid(row=1, column=1, padx=(8, 0), pady=(8, 0), sticky="nsew")

        tk.Label(p4, text="COGNITIVE SIMULATION OVERRIDE", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor=tk.W)
        tk.Label(p4, text="Manually test workplace adaptations on demand.", font=("Segoe UI", 8), fg=TEXT_SECONDARY, bg=BG_CARD).pack(anchor=tk.W, pady=(2, 8))

        sim_row = tk.Frame(p4, bg=BG_CARD)
        sim_row.pack(anchor=tk.W)

        self.btn_auto = tk.Button(
            sim_row,
            text="Auto-Sense",
            font=("Segoe UI", 8, "bold"),
            bg="#2563eb",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=8,
            pady=3,
            cursor="hand2",
            command=self._reset_to_auto
        )
        self.btn_auto.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_flow = self._create_sim_button(sim_row, "Flow", CognitiveState.FLOW, ACCENT_GREEN)
        self.btn_norm = self._create_sim_button(sim_row, "Normal", CognitiveState.NORMAL_FOCUS, ACCENT_CYAN)
        self.btn_overload = self._create_sim_button(sim_row, "Overload", CognitiveState.COGNITIVE_OVERLOAD, ACCENT_AMBER)
        self.btn_fatigue = self._create_sim_button(sim_row, "Fatigue", CognitiveState.FATIGUE, ACCENT_RED)
        self.btn_idle = self._create_sim_button(sim_row, "Idle", CognitiveState.IDLE, ACCENT_SLATE)

        return page

    # -------------------------------------------------------------
    # TAB 4: SESSION REPORT (Executive analytics & PDF export)
    # -------------------------------------------------------------
    def _build_report_tab(self) -> tk.Frame:
        page = tk.Frame(self.tab_container, bg=BG_MAIN)

        rep_header = tk.Frame(page, bg=BG_MAIN)
        rep_header.pack(fill=tk.X, pady=(0, 14))

        title_box = tk.Frame(rep_header, bg=BG_MAIN)
        title_box.pack(side=tk.LEFT)
        tk.Label(title_box, text="EXECUTIVE SESSION SUMMARY", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=BG_MAIN).pack(anchor=tk.W)
        tk.Label(title_box, text="Cognitive Productivity & Health Report", font=("Segoe UI", 16, "bold"), fg=TEXT_PRIMARY, bg=BG_MAIN).pack(anchor=tk.W)

        self.btn_export_full = tk.Button(
            rep_header,
            text="🖨️ Export PDF / Open HTML Report",
            font=("Segoe UI", 9, "bold"),
            bg="#0284c7",
            fg="#ffffff",
            activebackground="#0369a1",
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
            command=self._on_generate_report
        )
        self.btn_export_full.pack(side=tk.RIGHT)

        # 6 KPI Tiles
        kpi_grid = tk.Frame(page, bg=BG_MAIN)
        kpi_grid.pack(fill=tk.X, pady=(0, 16))
        for i in range(3):
            kpi_grid.columnconfigure(i, weight=1)

        c1, self.report_prod_val, self.report_prod_sub = self._create_stat_card(kpi_grid, "Productivity Score", "88.5", "/ 100", "Solid Productivity", ACCENT_GREEN)
        c1.grid(row=0, column=0, padx=(0, 4), pady=(0, 6), sticky="nsew")

        c2, self.report_flow_val, self.report_flow_sub = self._create_stat_card(kpi_grid, "Flow Immersion Rate", "74.0", "%", "Deep flow engagement", ACCENT_GREEN)
        c2.grid(row=0, column=1, padx=(4, 4), pady=(0, 6), sticky="nsew")

        c3, self.report_wpm_val, self.report_wpm_sub = self._create_stat_card(kpi_grid, "Typing Speed", "37", "WPM", "185 KPM Cadence", ACCENT_GREEN)
        c3.grid(row=0, column=2, padx=(4, 0), pady=(0, 6), sticky="nsew")

        c4, self.report_bpm_val, self.report_bpm_sub = self._create_stat_card(kpi_grid, "Blink Frequency", "15.8", "BPM", "Normal: 15–22 BPM", ACCENT_GREEN)
        c4.grid(row=1, column=0, padx=(0, 4), pady=(6, 0), sticky="nsew")

        c5, self.report_perclos_val, self.report_perclos_sub = self._create_stat_card(kpi_grid, "Eye Strain (PERCLOS)", "3.2", "%", "Fatigue Risk <15%", ACCENT_GREEN)
        c5.grid(row=1, column=1, padx=(4, 4), pady=(6, 0), sticky="nsew")

        c6, self.report_jitter_val, self.report_jitter_sub = self._create_stat_card(kpi_grid, "Motor Smoothness", "91.8", "/ 100", "Calm Trajectory", ACCENT_GREEN)
        c6.grid(row=1, column=2, padx=(4, 0), pady=(6, 0), sticky="nsew")

        # Cognitive Timeline Spectrum Bar
        timeline_panel = tk.Frame(page, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=20, pady=16)
        timeline_panel.pack(fill=tk.X)

        tk.Label(timeline_panel, text="WORKDAY COGNITIVE SPECTRUM BAR", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor=tk.W)

        bar_frame = tk.Frame(timeline_panel, bg=BG_MAIN, height=22, highlightbackground=BORDER_COLOR, highlightthickness=1)
        bar_frame.pack(fill=tk.X, pady=(10, 8))
        bar_frame.pack_propagate(False)

        tk.Label(bar_frame, bg=ACCENT_SLATE, width=4).pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(bar_frame, bg=ACCENT_CYAN, width=12).pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(bar_frame, bg=ACCENT_GREEN, width=28).pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(bar_frame, bg=ACCENT_AMBER, width=5).pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(bar_frame, bg=ACCENT_RED, width=3).pack(side=tk.LEFT, fill=tk.Y)

        legend_row = tk.Frame(timeline_panel, bg=BG_CARD)
        legend_row.pack(anchor=tk.W)
        for name, col in [("Flow", ACCENT_GREEN), ("Focus", ACCENT_CYAN), ("Overload", ACCENT_AMBER), ("Fatigue", ACCENT_RED), ("Idle", ACCENT_SLATE)]:
            item = tk.Frame(legend_row, bg=BG_CARD)
            item.pack(side=tk.LEFT, padx=(0, 14))
            tk.Label(item, text="●", font=("Segoe UI", 9), fg=col, bg=BG_CARD).pack(side=tk.LEFT, padx=(0, 4))
            tk.Label(item, text=name, font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD).pack(side=tk.LEFT)

        return page

    def _build_footer(self) -> None:
        """Bottom status bar with privacy statement and safeguards."""
        footer_frame = tk.Frame(self, bg=BG_MAIN, height=24, padx=20)
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Label(
            footer_frame,
            text="🔒 100% Local Inference — Zero Cloud Transmission | Zero Keylogging | Hardware Gamma Auto-Restoration Safeguard Active",
            font=("Segoe UI", 8),
            fg=TEXT_MUTED,
            bg=BG_MAIN
        ).pack(side=tk.LEFT)

    def _create_stat_card(
        self,
        parent: tk.Widget,
        label: str,
        val: str,
        unit: str,
        note: str,
        dot_col: str
    ) -> tuple[tk.Frame, tk.Label, tk.Label]:
        """Helper to create standardized Linear-style stat card."""
        card = tk.Frame(parent, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=16, pady=12)

        tk.Label(card, text=label.upper(), font=("Segoe UI", 7, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor=tk.W)

        val_row = tk.Frame(card, bg=BG_CARD)
        val_row.pack(anchor=tk.W, pady=(4, 2))

        val_lbl = tk.Label(val_row, text=val, font=("Consolas", 15, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD)
        val_lbl.pack(side=tk.LEFT)

        if unit:
            tk.Label(val_row, text=f" {unit}", font=("Segoe UI", 8), fg=TEXT_SECONDARY, bg=BG_CARD).pack(side=tk.LEFT)

        note_row = tk.Frame(card, bg=BG_CARD)
        note_row.pack(anchor=tk.W)
        dot_lbl = tk.Label(note_row, text="●", font=("Segoe UI", 7), fg=dot_col, bg=BG_CARD)
        dot_lbl.pack(side=tk.LEFT, padx=(0, 4))
        sub_lbl = tk.Label(note_row, text=note, font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD)
        sub_lbl.pack(side=tk.LEFT)

        return card, val_lbl, sub_lbl

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

    def _apply_kelvin_direct(self, kelvin: int) -> None:
        """Directly adjust screen warmth from GUI."""
        self.adaptation_executor._apply_display_warmth(kelvin)
        self.adaptation_executor.current_kelvin = kelvin
        self.kelvin_lbl.configure(text=f"{kelvin} K")
        self.kelvin_swatch.configure(bg=kelvin_to_hex(kelvin))
        self.overview_kelvin_val.configure(text=str(kelvin))

    def _set_simulated_state(self, state: CognitiveState) -> None:
        """Manually force a cognitive state to test workplace adaptations immediately."""
        self.simulated_state = state
        self.status_lbl.configure(text=f"Sim: {state.value[:8]}", fg=ACCENT_AMBER)
        self.status_dot.configure(fg=ACCENT_AMBER)

    def _reset_to_auto(self) -> None:
        """Resume real-time biometric and peripheral sensor fusion."""
        self.simulated_state = None
        self.status_lbl.configure(text="Active", fg=ACCENT_GREEN)
        self.status_dot.configure(fg=ACCENT_GREEN)

    def _restore_display_gamma(self) -> None:
        """Restore screen display to default pristine gamma ramp."""
        self.adaptation_executor.restore_display()
        self.kelvin_lbl.configure(text="6500 K")
        self.kelvin_swatch.configure(bg="#ffffff")
        self.overview_kelvin_val.configure(text="6500")

    def _toggle_camera(self) -> None:
        """Toggle camera stream on/off."""
        self.camera_enabled = not self.camera_enabled
        self.btn_cam_toggle.configure(
            text="Camera ON" if self.camera_enabled else "Camera OFF",
            fg=ACCENT_GREEN if self.camera_enabled else ACCENT_RED
        )

    def _toggle_overlay(self) -> None:
        """Toggle landmark overlay on video."""
        self.show_overlay = not self.show_overlay
        self.btn_overlay_toggle.configure(
            text="Mesh ON" if self.show_overlay else "Mesh OFF"
        )

    # -------------------------------------------------------------
    # Main Engine Tick and Camera Update
    # -------------------------------------------------------------
    def _update_camera_feed(self) -> None:
        """Fetch frame from VisionTracker and blit to Tkinter canvas at ~30 FPS."""
        if self.camera_enabled:
            frame, _ = self.vision_tracker.get_latest_frame()
            if frame is not None:
                # Resize to fit canvas (388 x 291)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (388, 291), interpolation=cv2.INTER_AREA)
                img = Image.fromarray(frame_resized)
                img_tk = ImageTk.PhotoImage(image=img)
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                self.video_canvas.image = img_tk
        self.after(33, self._update_camera_feed)

    def _engine_tick(self) -> None:
        """Core closed-loop iteration executed at 10 Hz."""
        t_now = time.time()

        # Update Session Duration Timer
        elapsed = int(t_now - self.session_start_time)
        hh = elapsed // 3600
        mm = (elapsed % 3600) // 60
        ss = elapsed % 60
        self.session_timer_lbl.configure(text=f"⏱ {hh:02d}:{mm:02d}:{ss:02d}")

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

        # 7. Sync with Web Dashboard if active
        if hasattr(self, "_web_server") and self._web_server is not None:
            from aetherflow.web_server import dashboard_state
            if dashboard_state.override_state is not None:
                for cs in CognitiveState:
                    if cs.value == dashboard_state.override_state:
                        self.simulated_state = cs
                        break
                if dashboard_state.override_state == "null":
                    self.simulated_state = None
            if dashboard_state.minimize_requested:
                dashboard_state.minimize_requested = False
                self._minimize_to_background()

            dashboard_state.update_telemetry({
                "session_time": f"{hh:02d}:{mm:02d}:{ss:02d}",
                "state": inference.state.value,
                "state_title": inference.state.value.replace("_", " ").title(),
                "productivity_score": 92.0 if inference.state == CognitiveState.FLOW else (78.0 if inference.state == CognitiveState.NORMAL_FOCUS else 45.0),
                "wpm": per_sample.keystrokes_per_min / 5.0,
                "kpm": per_sample.keystrokes_per_min,
                "bpm": vis_sample.blink_rate_bpm,
                "ear": vis_sample.ear_mean,
                "perclos": vis_sample.perclos,
                "gaze": vis_sample.gaze_stability,
                "kelvin": exec_res.current_kelvin,
                "density": exec_res.ui_density.value,
                "dnd": exec_res.notification_policy.value,
                "context": f"{app_context.process_name} — {app_context.window_title}",
                "probabilities": {k.lower(): v for k, v in inference.probabilities.items()}
            })

        # 8. Update GUI Displays
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
        # Context & App
        domain_val = ctx.domain.value
        self.ctx_domain_badge.configure(text=domain_val)
        title_str = ctx.window_title[:38] + "..." if len(ctx.window_title) > 38 else ctx.window_title
        self.ctx_app_lbl.configure(text=f"{ctx.process_name} — {title_str or 'Foreground Window'}")

        # State Hero
        dot_col, text_col, state_desc = STATE_HERO_STYLES.get(
            inf.state.value,
            (ACCENT_GREEN, TEXT_PRIMARY, "Cognitive baseline active.")
        )
        self.overview_state_dot.configure(fg=dot_col)
        self.overview_state_title.configure(text=inf.state.value.replace("_", " ").title())
        self.overview_state_desc.configure(text=state_desc)

        # Overview Quick Numbers
        prod_val = 92 if inf.state == CognitiveState.FLOW else (78 if inf.state == CognitiveState.NORMAL_FOCUS else 45)
        self.overview_prod_val.configure(text=str(prod_val))
        wpm = int(per.keystrokes_per_min / 5.0)
        self.overview_wpm_val.configure(text=str(wpm))
        self.overview_wpm_sub.configure(text=f"{per.keystrokes_per_min:.0f} KPM steady")
        self.overview_bpm_val.configure(text=f"{vis.blink_rate_bpm:.1f}")
        self.overview_kelvin_val.configure(text=str(res.current_kelvin))

        # Camera HUD badges
        self.cam_ear_lbl.configure(text=f"EAR {vis.ear_mean:.2f}")
        self.cam_gaze_lbl.configure(text=f"Gaze {int(vis.gaze_stability * 100)}%")

        # Biometrics Tab
        blink_tag = " [BLINK]" if vis.is_blinking else ""
        self.metric_ear_val.configure(text=f"{vis.ear_mean:.2f}{blink_tag}", fg=ACCENT_RED if vis.ear_mean < 0.21 else TEXT_PRIMARY)
        self.metric_blink_val.configure(text=f"{vis.blink_rate_bpm:.1f}")
        self.metric_perclos_val.configure(text=f"{vis.perclos * 100:.1f}")
        self.metric_gaze_val.configure(text=f"{int(vis.gaze_stability * 100)}")

        self.metric_cadence_val.configure(text=str(wpm))
        self.metric_cadence_sub.configure(text=f"{per.keystrokes_per_min:.0f} KPM Cadence")
        self.metric_dwell_val.configure(text=f"{per.mean_dwell_time_ms:.0f} / {per.mean_flight_time_ms:.0f}")
        self.metric_jitter_val.configure(text=f"{per.mouse_jitter_index:.3f}")
        self.metric_idle_val.configure(text="ACTIVE" if not per.is_idle else "IDLE")

        # Probability Bars
        for s_name, pb in self.prob_bars.items():
            prob_pct = int(inf.probabilities.get(s_name, 0.0) * 100)
            pb["value"] = prob_pct
            self.prob_labels[s_name].configure(text=f"{prob_pct}%")

        # Environment Tab
        self.kelvin_lbl.configure(text=f"{res.current_kelvin} K")
        self.kelvin_swatch.configure(bg=kelvin_to_hex(res.current_kelvin))
        self.density_lbl.configure(text=res.ui_density.value)
        self.dnd_lbl.configure(text=res.notification_policy.value)

        if res.cooldown_remaining_sec > 0:
            self.pacing_lbl.configure(text=f"Locked: {res.cooldown_remaining_sec:.0f}s cooldown", fg=ACCENT_AMBER)
        else:
            int_sec = int(self.config.adaptation.min_adaptation_interval_sec)
            self.pacing_lbl.configure(text=f"Ready ({int_sec}s interval)", fg=ACCENT_GREEN)

        if res.break_recommendation == BreakRecommendation.MANDATORY_BREAK:
            self.break_banner.configure(text="⚠️ FATIGUE DETECTED — Take an ergonomic 5-minute break now!", bg=ACCENT_RED, fg="#ffffff")
        elif res.break_recommendation == BreakRecommendation.GENTLE_PROMPT:
            self.break_banner.configure(text="⏳ Cognitive strain elevated — gently pace yourself", bg=ACCENT_AMBER, fg="#000000")
        else:
            self.break_banner.configure(text="✓ Ergonomics optimal — environment matched", bg=BG_CARD_LIGHT, fg=ACCENT_GREEN)

        # Report Tab
        self.report_prod_val.configure(text=str(prod_val))
        self.report_wpm_val.configure(text=str(wpm))
        self.report_bpm_val.configure(text=f"{vis.blink_rate_bpm:.1f}")
        self.report_perclos_val.configure(text=f"{vis.perclos * 100:.1f}")
        motor_smooth = max(0.0, min(100.0, 100.0 * (1.0 - per.mouse_jitter_index * 1.6)))
        self.report_jitter_val.configure(text=f"{motor_smooth:.1f}")

    def _set_adaptation_interval(self, seconds: int) -> None:
        """Dynamically update the adaptation interval / cooldown period."""
        self.config.adaptation.min_adaptation_interval_sec = float(seconds)
        self.adaptation_executor.config.min_adaptation_interval_sec = float(seconds)
        int_sec = int(seconds)

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

        threading.Thread(target=_worker, daemon=True).start()

    def _open_web_dashboard(self) -> None:
        """Launch the modern 4-tab Linear/Raycast web dashboard in browser."""
        import webbrowser
        from aetherflow.web_server import start_web_server
        if not hasattr(self, "_web_server") or self._web_server is None:
            self._web_server = start_web_server()
        try:
            webbrowser.open("http://127.0.0.1:8765")
        except Exception as e:
            logger.debug(f"Could not open browser: {e}")

    def on_closing(self) -> None:
        """Safe shutdown when user closes window."""
        logger.info("GUI window closed. Terminating background threads...")
        if hasattr(self, "_web_server") and self._web_server is not None:
            self._web_server.stop()
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
