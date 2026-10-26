"""Persistent SQLite session storage and analytics database for AetherFlow.

Efficiently logs real-time cognitive states, active application contexts,
biometric telemetry, and workspace adaptation events for on-demand
productivity reporting and longitudinal cognitive health analysis.
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aetherflow.config import DATA_DIR, CognitiveState, ContextDomain

logger = logging.getLogger(__name__)

DB_PATH = DATA_DIR / "aetherflow_sessions.db"


@dataclass
class ProductivityMetrics:
    """Consolidated productivity and cognitive health metrics for a session."""
    session_id: str
    start_time: float
    end_time: float
    duration_sec: float
    active_duration_sec: float
    idle_duration_sec: float
    
    # State durations (in seconds)
    state_durations: Dict[str, float]
    state_percentages: Dict[str, float]
    
    # Context domain durations (in seconds)
    domain_durations: Dict[str, float]
    top_applications: List[Tuple[str, float]] # (app_name, duration_sec)
    
    # Cognitive & Productivity Scores (0 - 100)
    flow_score: float             # % of active time in flow
    productivity_score: float     # Weighted productivity index
    fatigue_index: float          # Average PERCLOS / eye strain ratio
    stability_score: float        # Gaze and motor stability composite
    
    # Biometric averages
    avg_ear: float
    avg_blink_rate: float
    avg_perclos: float
    avg_gaze_stability: float
    
    # Peripheral averages
    avg_cadence_kpm: float
    avg_dwell_ms: float
    avg_flight_ms: float
    avg_mouse_jitter: float
    
    # Adaptations
    total_adaptations: int
    blue_light_reduced_sec: float # Time display warmth was < 6500K

    # Typing speed & blink frequency metrics
    avg_wpm: float = 0.0          # Typing speed in Words Per Minute (KPM / 5.0)
    avg_bpm: float = 0.0          # Ocular blink frequency in Blinks Per Minute (alias of avg_blink_rate)


class SessionStore:
    """Manages SQLite persistence for telemetry logs and session analytics."""

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self._batch_buffer: List[tuple] = []
        self._last_flush_time = time.time()

    def _get_connection(self) -> sqlite3.Connection:
        """Create sqlite3 connection with WAL mode for high concurrency."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_database(self) -> None:
        """Initialize database schema with indexes for fast analytics queries."""
        with self._get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time REAL NOT NULL,
                end_time REAL,
                total_duration_sec REAL DEFAULT 0.0,
                status TEXT DEFAULT 'ACTIVE'
            );
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                dt REAL DEFAULT 1.0,
                cognitive_state TEXT NOT NULL,
                confidence REAL NOT NULL,
                context_domain TEXT NOT NULL,
                process_name TEXT NOT NULL,
                window_title TEXT,
                ear REAL,
                blink_rate REAL,
                perclos REAL,
                gaze_stability REAL,
                keystrokes_per_min REAL,
                dwell_time_ms REAL,
                flight_time_ms REAL,
                mouse_jitter REAL,
                screen_warmth_kelvin INTEGER,
                ui_density TEXT,
                notification_policy TEXT,
                adaptation_executed INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_session ON telemetry_ticks(session_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_time ON telemetry_ticks(timestamp);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_state ON telemetry_ticks(cognitive_state);")
            conn.commit()

    def start_session(self, session_id: Optional[str] = None) -> str:
        """Record the start of a new monitoring session."""
        sid = session_id or f"sess_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        t_now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, start_time, status) VALUES (?, ?, 'ACTIVE')",
                (sid, t_now)
            )
            conn.commit()
        logger.info(f"Started persistent session {sid}")
        return sid

    def log_tick(
        self,
        session_id: str,
        timestamp: float,
        dt: float,
        cognitive_state: str,
        confidence: float,
        context_domain: str,
        process_name: str,
        window_title: str,
        ear: float,
        blink_rate: float,
        perclos: float,
        gaze_stability: float,
        keystrokes_per_min: float,
        dwell_time_ms: float,
        flight_time_ms: float,
        mouse_jitter: float,
        screen_warmth_kelvin: int,
        ui_density: str,
        notification_policy: str,
        adaptation_executed: bool = False
    ) -> None:
        """Buffer and persist a single event loop telemetry tick."""
        record = (
            session_id,
            timestamp,
            dt,
            cognitive_state,
            confidence,
            context_domain,
            process_name,
            window_title[:120] if window_title else "",
            round(ear, 3),
            round(blink_rate, 1),
            round(perclos, 3),
            round(gaze_stability, 3),
            round(keystrokes_per_min, 1),
            round(dwell_time_ms, 1),
            round(flight_time_ms, 1),
            round(mouse_jitter, 3),
            screen_warmth_kelvin,
            ui_density,
            notification_policy,
            1 if adaptation_executed else 0
        )
        self._batch_buffer.append(record)

        # Flush batch every 5 records or 5 seconds to minimize disk write cycles
        if len(self._batch_buffer) >= 5 or (time.time() - self._last_flush_time) > 5.0:
            self.flush()

    def flush(self) -> None:
        """Commit buffered telemetry ticks to disk."""
        if not self._batch_buffer:
            return
        records_to_insert = list(self._batch_buffer)
        self._batch_buffer.clear()
        self._last_flush_time = time.time()

        try:
            with self._get_connection() as conn:
                conn.executemany("""
                INSERT INTO telemetry_ticks (
                    session_id, timestamp, dt, cognitive_state, confidence,
                    context_domain, process_name, window_title, ear, blink_rate,
                    perclos, gaze_stability, keystrokes_per_min, dwell_time_ms,
                    flight_time_ms, mouse_jitter, screen_warmth_kelvin, ui_density,
                    notification_policy, adaptation_executed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records_to_insert)
                conn.commit()
        except Exception as e:
            logger.error(f"Error flushing telemetry ticks to SQLite: {e}")

    def close_session(self, session_id: str) -> None:
        """Mark session completed and update total duration."""
        self.flush()
        t_now = time.time()
        with self._get_connection() as conn:
            row = conn.execute("SELECT start_time FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if row:
                duration = t_now - row["start_time"]
                conn.execute(
                    "UPDATE sessions SET end_time = ?, total_duration_sec = ?, status = 'COMPLETED' WHERE session_id = ?",
                    (t_now, duration, session_id)
                )
                conn.commit()
        logger.info(f"Closed session {session_id}")

    def get_latest_session_id(self) -> Optional[str]:
        """Query the most recent session ID that contains telemetry ticks (or latest registered)."""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT s.session_id 
                FROM sessions s 
                JOIN telemetry_ticks t ON s.session_id = t.session_id 
                ORDER BY t.timestamp DESC LIMIT 1
            """).fetchone()
            if row:
                return row["session_id"]
            row = conn.execute("SELECT session_id FROM sessions ORDER BY start_time DESC LIMIT 1").fetchone()
            return row["session_id"] if row else None

    def compute_session_metrics(self, session_id: Optional[str] = None) -> Optional[ProductivityMetrics]:
        """Aggregate statistical metrics for a specific or latest session."""
        self.flush()
        sid = session_id or self.get_latest_session_id()
        if not sid:
            return None

        with self._get_connection() as conn:
            sess = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (sid,)).fetchone()
            if not sess:
                return None

            ticks = conn.execute(
                "SELECT * FROM telemetry_ticks WHERE session_id = ? ORDER BY timestamp ASC",
                (sid,)
            ).fetchall()

            if not ticks:
                return None

            start_t = ticks[0]["timestamp"]
            end_t = ticks[-1]["timestamp"]
            total_duration = max(1.0, max(end_t - start_t, sum((t["dt"] or 1.0) for t in ticks)))

            # State breakdown
            state_durations = {s: 0.0 for s in CognitiveState.list()}
            domain_durations = {d: 0.0 for d in ContextDomain.list()}
            app_durations: Dict[str, float] = {}
            
            ears, blinks, perclos_list, gazes = [], [], [], []
            kpms, dwells, flights, jitters = [], [], [], []
            adaptations_count = 0
            warmth_time_sec = 0.0

            for t in ticks:
                dt = t["dt"] or 1.0
                state = t["cognitive_state"]
                domain = t["context_domain"]
                app = t["process_name"]

                state_durations[state] = state_durations.get(state, 0.0) + dt
                domain_durations[domain] = domain_durations.get(domain, 0.0) + dt
                app_durations[app] = app_durations.get(app, 0.0) + dt

                if t["ear"] is not None and t["ear"] > 0:
                    ears.append(t["ear"])
                if t["blink_rate"] is not None:
                    blinks.append(t["blink_rate"])
                if t["perclos"] is not None:
                    perclos_list.append(t["perclos"])
                if t["gaze_stability"] is not None:
                    gazes.append(t["gaze_stability"])

                if t["keystrokes_per_min"] is not None:
                    kpms.append(t["keystrokes_per_min"])
                if t["dwell_time_ms"] is not None and t["dwell_time_ms"] > 0:
                    dwells.append(t["dwell_time_ms"])
                if t["flight_time_ms"] is not None and t["flight_time_ms"] > 0:
                    flights.append(t["flight_time_ms"])
                if t["mouse_jitter"] is not None:
                    jitters.append(t["mouse_jitter"])

                if t["adaptation_executed"]:
                    adaptations_count += 1
                if t["screen_warmth_kelvin"] and t["screen_warmth_kelvin"] < 6500:
                    warmth_time_sec += dt

            # Compute percentages
            state_percentages = {
                s: (dur / total_duration) * 100.0 for s, dur in state_durations.items()
            }

            idle_dur = state_durations.get(CognitiveState.IDLE.value, 0.0)
            active_dur = max(1.0, total_duration - idle_dur)

            # Top apps
            sorted_apps = sorted(app_durations.items(), key=lambda x: x[1], reverse=True)[:5]

            # Flow Score (% of active time spent in FLOW)
            flow_dur = state_durations.get(CognitiveState.FLOW.value, 0.0)
            flow_score = min(100.0, (flow_dur / active_dur) * 100.0)

            # Productivity Score: Weighted combination of focused vs distracted vs sluggish time
            norm_dur = state_durations.get(CognitiveState.NORMAL_FOCUS.value, 0.0)
            overload_dur = state_durations.get(CognitiveState.COGNITIVE_OVERLOAD.value, 0.0)
            fatigue_dur = state_durations.get(CognitiveState.FATIGUE.value, 0.0)

            raw_prod = (flow_dur * 1.0 + norm_dur * 0.80 - overload_dur * 0.35 - fatigue_dur * 0.25) / active_dur
            productivity_score = round(max(0.0, min(100.0, raw_prod * 100.0)), 1)

            # Health & stability index
            avg_perclos = float(sum(perclos_list) / len(perclos_list)) if perclos_list else 0.03
            fatigue_index = round(min(100.0, (avg_perclos / 0.20) * 100.0), 1)

            avg_gaze = float(sum(gazes) / len(gazes)) if gazes else 0.80
            avg_jitter = float(sum(jitters) / len(jitters)) if jitters else 0.10
            stability_score = round(max(0.0, min(100.0, (avg_gaze * 60.0 + (1.0 - min(1.0, avg_jitter * 2.5)) * 40.0))), 1)

            return ProductivityMetrics(
                session_id=sid,
                start_time=start_t,
                end_time=end_t,
                duration_sec=round(total_duration, 1),
                active_duration_sec=round(active_dur, 1),
                idle_duration_sec=round(idle_dur, 1),
                state_durations=state_durations,
                state_percentages=state_percentages,
                domain_durations=domain_durations,
                top_applications=sorted_apps,
                flow_score=round(flow_score, 1),
                productivity_score=productivity_score,
                fatigue_index=fatigue_index,
                stability_score=stability_score,
                avg_ear=round(float(sum(ears)/len(ears)) if ears else 0.28, 3),
                avg_blink_rate=round(float(sum(blinks)/len(blinks)) if blinks else 16.0, 1),
                avg_perclos=round(avg_perclos, 3),
                avg_gaze_stability=round(avg_gaze, 3),
                avg_cadence_kpm=round(float(sum(kpms)/len(kpms)) if kpms else 120.0, 1),
                avg_dwell_ms=round(float(sum(dwells)/len(dwells)) if dwells else 105.0, 1),
                avg_flight_ms=round(float(sum(flights)/len(flights)) if flights else 220.0, 1),
                avg_mouse_jitter=round(avg_jitter, 3),
                total_adaptations=adaptations_count,
                blue_light_reduced_sec=round(warmth_time_sec, 1),
                avg_wpm=round((float(sum(kpms)/len(kpms)) if kpms else 120.0) / 5.0, 1),
                avg_bpm=round(float(sum(blinks)/len(blinks)) if blinks else 16.0, 1)
            )
