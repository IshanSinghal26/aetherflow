"""Peripheral telemetry tracking for keyboard cadence and mouse dynamics.

Captures flight time, dwell time, typing cadence, mouse jitter, velocity,
and idle intervals using non-invasive timing hooks via pynput.
STRICT PRIVACY POLICY: Never records key symbols, characters, or text.
Only timestamps and intervals are stored.
"""

from __future__ import annotations

import collections
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from aetherflow.config import PeripheralConfig

logger = logging.getLogger(__name__)


@dataclass
class PeripheralSample:
    """Snapshot of interaction dynamics derived from peripheral events."""
    timestamp: float
    
    # Keyboard cadence metrics
    mean_dwell_time_ms: float = 0.0          # Key-down to key-up duration
    std_dwell_time_ms: float = 0.0           # Dwell variance
    mean_flight_time_ms: float = 0.0         # Key-up(n-1) to key-down(n) interval
    std_flight_time_ms: float = 0.0          # Flight variance
    keystrokes_per_min: float = 0.0          # Current typing rate
    typing_pause_ratio: float = 1.0          # Ratio of non-typing time in window
    typing_burstiness: float = 0.0           # Coefficient of variation of inter-keystroke intervals
    
    # Mouse dynamics metrics
    mean_mouse_velocity_px_s: float = 0.0    # Mean cursor velocity
    max_mouse_velocity_px_s: float = 0.0     # Peak cursor velocity
    mouse_jitter_index: float = 0.0          # Directional entropy and path tortuosity (0.0 to 1.0)
    mouse_click_rate_hz: float = 0.0         # Clicks per second
    mouse_scroll_rate_hz: float = 0.0        # Scrolls per second
    
    # Overall activity status
    idle_seconds: float = 0.0                # Seconds since last user action
    is_idle: bool = True


class PeripheralTracker:
    """Passive, non-invasive monitor for keyboard and mouse kinematics."""

    def __init__(self, config: Optional[PeripheralConfig] = None) -> None:
        self.config = config or PeripheralConfig()
        self._running = False
        self._lock = threading.Lock()
        
        # Privacy-preserving keyboard state (only hashable token -> press timestamp)
        self._key_press_times: Dict[int, float] = {}
        self._last_key_release_time: Optional[float] = None
        
        # Event history deques (timestamps and metrics)
        self._dwell_times: collections.deque[Tuple[float, float]] = collections.deque(maxlen=self.config.max_history_events) # (timestamp, dwell_ms)
        self._flight_times: collections.deque[Tuple[float, float]] = collections.deque(maxlen=self.config.max_history_events) # (timestamp, flight_ms)
        self._keystroke_timestamps: collections.deque[float] = collections.deque(maxlen=self.config.max_history_events)
        
        # Mouse event history
        self._mouse_positions: collections.deque[Tuple[float, float, float]] = collections.deque(maxlen=1000) # (timestamp, x, y)
        self._mouse_clicks: collections.deque[float] = collections.deque(maxlen=300)
        self._mouse_scrolls: collections.deque[float] = collections.deque(maxlen=300)
        self._last_move_heading: Optional[float] = None
        self._angular_changes: collections.deque[Tuple[float, float]] = collections.deque(maxlen=500) # (timestamp, delta_theta)
        
        # Activity timestamps
        self._last_activity_time = time.time()
        self._latest_sample = PeripheralSample(timestamp=self._last_activity_time)
        
        # Pynput listener instances
        self._kb_listener = None
        self._mouse_listener = None

    def start(self) -> None:
        """Initialize and start background pynput hooks."""
        if self._running or self.config.synthetic_mode:
            return
        
        self._running = True
        try:
            from pynput import keyboard, mouse

            self._kb_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release
            )
            self._kb_listener.daemon = True
            self._kb_listener.start()

            self._mouse_listener = mouse.Listener(
                on_move=self._on_mouse_move,
                on_click=self._on_mouse_click,
                on_scroll=self._on_mouse_scroll
            )
            self._mouse_listener.daemon = True
            self._mouse_listener.start()

            logger.info("PeripheralTracker listeners started.")
        except Exception as e:
            logger.warning(f"Could not hook pynput listeners ({e}). Falling back to synthetic peripheral mode.")
            self.config.synthetic_mode = True

    def stop(self) -> None:
        """Stop background listeners safely."""
        self._running = False
        if self._kb_listener:
            try:
                self._kb_listener.stop()
            except Exception:
                pass
            self._kb_listener = None

        if self._mouse_listener:
            try:
                self._mouse_listener.stop()
            except Exception:
                pass
            self._mouse_listener = None

        logger.info("PeripheralTracker listeners stopped.")

    # ---------------------------------------------------------
    # Keyboard Event Handlers (Zero string/character logging)
    # ---------------------------------------------------------
    def _on_key_press(self, key) -> None:
        """Record key-down timestamp anonymously."""
        t_now = time.time()
        with self._lock:
            self._last_activity_time = t_now
            key_id = id(key)
            
            # Flight time: interval since last key release
            if self._last_key_release_time is not None:
                flight = (t_now - self._last_key_release_time) * 1000.0
                # Filter out long breaks (> 3.0s is pause, not typing cadence)
                if 5.0 <= flight <= 3000.0:
                    self._flight_times.append((t_now, flight))
            
            if key_id not in self._key_press_times:
                self._key_press_times[key_id] = t_now
                self._keystroke_timestamps.append(t_now)

    def _on_key_release(self, key) -> None:
        """Record key-up timestamp and compute dwell duration."""
        t_now = time.time()
        with self._lock:
            self._last_activity_time = t_now
            self._last_key_release_time = t_now
            key_id = id(key)
            
            if key_id in self._key_press_times:
                t_down = self._key_press_times.pop(key_id)
                dwell = (t_now - t_down) * 1000.0
                # Reasonable physical dwell filter (10ms to 2000ms)
                if 10.0 <= dwell <= 2000.0:
                    self._dwell_times.append((t_now, dwell))

    # ---------------------------------------------------------
    # Mouse Event Handlers
    # ---------------------------------------------------------
    def _on_mouse_move(self, x: float, y: float) -> None:
        """Compute cursor displacement, instantaneous speed, and trajectory jitter."""
        t_now = time.time()
        with self._lock:
            self._last_activity_time = t_now
            
            if self._mouse_positions:
                t_prev, x_prev, y_prev = self._mouse_positions[-1]
                dt = t_now - t_prev
                if dt > 1e-4:
                    dx = x - x_prev
                    dy = y - y_prev
                    dist = math.hypot(dx, dy)
                    if dist > 1.0: # Filter stationary jitter noise
                        current_heading = math.atan2(dy, dx)
                        if self._last_move_heading is not None:
                            d_theta = abs(current_heading - self._last_move_heading)
                            # Wrap angle diff to [-pi, pi]
                            d_theta = abs((d_theta + math.pi) % (2.0 * math.pi) - math.pi)
                            self._angular_changes.append((t_now, d_theta))
                        self._last_move_heading = current_heading

            self._mouse_positions.append((t_now, float(x), float(y)))

    def _on_mouse_click(self, x: float, y: float, button, pressed: bool) -> None:
        """Record click event timing."""
        if pressed:
            t_now = time.time()
            with self._lock:
                self._last_activity_time = t_now
                self._mouse_clicks.append(t_now)

    def _on_mouse_scroll(self, x: float, y: float, dx: int, dy: int) -> None:
        """Record scroll event timing."""
        t_now = time.time()
        with self._lock:
            self._last_activity_time = t_now
            self._mouse_scrolls.append(t_now)

    # ---------------------------------------------------------
    # Telemetry Sampling & Aggregation
    # ---------------------------------------------------------
    def sample(self, window_sec: float = 60.0) -> PeripheralSample:
        """Aggregate recent peripheral kinematics over a sliding window."""
        t_now = time.time()
        
        if self.config.synthetic_mode:
            return self._generate_synthetic_sample(t_now)

        with self._lock:
            cutoff = t_now - window_sec
            idle_sec = max(0.0, t_now - self._last_activity_time)
            is_idle = idle_sec >= self.config.idle_timeout_sec

            # 1. Dwell times
            recent_dwells = [d for t, d in self._dwell_times if t >= cutoff]
            if recent_dwells:
                mean_dwell = float(np_mean(recent_dwells))
                std_dwell = float(np_std(recent_dwells))
            else:
                mean_dwell = 100.0
                std_dwell = 0.0

            # 2. Flight times
            recent_flights = [f for t, f in self._flight_times if t >= cutoff]
            if recent_flights:
                mean_flight = float(np_mean(recent_flights))
                std_flight = float(np_std(recent_flights))
            else:
                mean_flight = 250.0
                std_flight = 0.0

            # 3. Keystrokes per minute & burstiness
            recent_keys = [t for t in self._keystroke_timestamps if t >= cutoff]
            effective_window_min = min(window_sec, max(5.0, t_now - (recent_keys[0] if recent_keys else t_now))) / 60.0
            kpm = (len(recent_keys) / effective_window_min) if effective_window_min > 0 else 0.0

            # Burstiness: variance of inter-keystroke intervals
            if len(recent_keys) > 2:
                ikis = [t2 - t1 for t1, t2 in zip(recent_keys[:-1], recent_keys[1:])]
                m_iki = np_mean(ikis)
                s_iki = np_std(ikis)
                burstiness = float(s_iki / m_iki) if m_iki > 1e-4 else 0.0
            else:
                burstiness = 0.0

            # Typing pause ratio (proportion of inactive intervals > 2s)
            pause_ratio = min(1.0, max(0.0, idle_sec / window_sec))

            # 4. Mouse velocity & peak speed
            recent_mouse = [p for p in self._mouse_positions if p[0] >= cutoff]
            velocities: List[float] = []
            for i in range(1, len(recent_mouse)):
                dt = recent_mouse[i][0] - recent_mouse[i-1][0]
                if dt > 1e-4:
                    dist = math.hypot(
                        recent_mouse[i][1] - recent_mouse[i-1][1],
                        recent_mouse[i][2] - recent_mouse[i-1][2]
                    )
                    velocities.append(dist / dt)

            if velocities:
                mean_vel = float(np_mean(velocities))
                max_vel = float(max(velocities))
            else:
                mean_vel = 0.0
                max_vel = 0.0

            # 5. Mouse Jitter Metric (combination of angular variance and path tortuosity)
            recent_angles = [a for t, a in self._angular_changes if t >= cutoff]
            if recent_angles and len(recent_mouse) > 5:
                # Mean angular change in direction (erratic zigzag = higher)
                mean_angular_turn = float(np_mean(recent_angles))
                
                # Path tortuosity: actual path length / Euclidean chord
                actual_path = sum(
                    math.hypot(
                        recent_mouse[i][1] - recent_mouse[i-1][1],
                        recent_mouse[i][2] - recent_mouse[i-1][2]
                    ) for i in range(1, len(recent_mouse))
                )
                chord = math.hypot(
                    recent_mouse[-1][1] - recent_mouse[0][1],
                    recent_mouse[-1][2] - recent_mouse[0][2]
                ) + 10.0
                tortuosity = min(5.0, actual_path / chord)
                
                # Composite normalized jitter index [0.0, 1.0]
                raw_jitter = (mean_angular_turn / math.pi) * 0.6 + (min(3.0, tortuosity - 1.0) / 3.0) * 0.4
                mouse_jitter = float(min(1.0, max(0.0, raw_jitter)))
            else:
                mouse_jitter = 0.05

            # 6. Click & scroll rates
            recent_clicks = [t for t in self._mouse_clicks if t >= cutoff]
            recent_scrolls = [t for t in self._mouse_scrolls if t >= cutoff]
            click_rate = float(len(recent_clicks) / window_sec)
            scroll_rate = float(len(recent_scrolls) / window_sec)

            sample = PeripheralSample(
                timestamp=t_now,
                mean_dwell_time_ms=round(mean_dwell, 2),
                std_dwell_time_ms=round(std_dwell, 2),
                mean_flight_time_ms=round(mean_flight, 2),
                std_flight_time_ms=round(std_flight, 2),
                keystrokes_per_min=round(kpm, 2),
                typing_pause_ratio=round(pause_ratio, 4),
                typing_burstiness=round(burstiness, 4),
                mean_mouse_velocity_px_s=round(mean_vel, 2),
                max_mouse_velocity_px_s=round(max_vel, 2),
                mouse_jitter_index=round(mouse_jitter, 4),
                mouse_click_rate_hz=round(click_rate, 4),
                mouse_scroll_rate_hz=round(scroll_rate, 4),
                idle_seconds=round(idle_sec, 2),
                is_idle=is_idle
            )
            self._latest_sample = sample
            return sample

    def _generate_synthetic_sample(self, timestamp: float) -> PeripheralSample:
        """Simulate realistic peripheral behavior for automated test runs or headless instances."""
        t = timestamp
        # Rhythmic active state simulation
        is_active = (int(t / 20) % 2) == 0
        dwell = 95.0 + 10.0 * math.sin(t * 0.3)
        flight = 180.0 + 30.0 * math.cos(t * 0.2)
        kpm = 220.0 if is_active else 0.0
        jitter = 0.12 if is_active else 0.01

        sample = PeripheralSample(
            timestamp=timestamp,
            mean_dwell_time_ms=round(dwell, 2),
            std_dwell_time_ms=15.0,
            mean_flight_time_ms=round(flight, 2),
            std_flight_time_ms=35.0,
            keystrokes_per_min=round(kpm, 2),
            typing_pause_ratio=0.15 if is_active else 1.0,
            typing_burstiness=0.6,
            mean_mouse_velocity_px_s=140.0 if is_active else 0.0,
            max_mouse_velocity_px_s=450.0 if is_active else 0.0,
            mouse_jitter_index=round(jitter, 4),
            mouse_click_rate_hz=0.4 if is_active else 0.0,
            mouse_scroll_rate_hz=0.2 if is_active else 0.0,
            idle_seconds=0.5 if is_active else 25.0,
            is_idle=not is_active
        )
        self._latest_sample = sample
        return sample

    def inject_synthetic_keystroke(self, dwell_ms: float = 110.0, flight_ms: float = 180.0) -> None:
        """Helper to programmatically feed keystrokes for testing."""
        t_now = time.time()
        with self._lock:
            self._last_activity_time = t_now
            self._keystroke_timestamps.append(t_now)
            self._dwell_times.append((t_now, dwell_ms))
            self._flight_times.append((t_now, flight_ms))

    def inject_synthetic_mouse(self, x: float, y: float) -> None:
        """Helper to programmatically feed mouse movements for testing."""
        t_now = time.time()
        with self._lock:
            self._last_activity_time = t_now
            self._mouse_positions.append((t_now, x, y))


def np_mean(arr: List[float]) -> float:
    """Fast float average without requiring heavyweight allocations."""
    return sum(arr) / len(arr) if arr else 0.0


def np_std(arr: List[float]) -> float:
    """Sample standard deviation calculation."""
    if len(arr) < 2:
        return 0.0
    m = np_mean(arr)
    variance = sum((x - m) ** 2 for x in arr) / len(arr)
    return math.sqrt(variance)
