"""Unit tests for AetherFlow session persistence and productivity reporting."""

import tempfile
import time
from pathlib import Path
import pytest

from aetherflow.config import CognitiveState, ContextDomain
from aetherflow.storage import SessionStore, ProductivityMetrics
from aetherflow.report import generate_html_report, print_cli_report, generate_report, format_duration


@pytest.fixture
def temp_db(tmp_path):
    """Provide a temporary SQLite database path."""
    return tmp_path / "test_aetherflow.db"


def test_session_lifecycle_and_tick_logging(temp_db):
    """Verify session start, batch tick logging, flush, and session closing."""
    store = SessionStore(db_path=temp_db)
    
    session_id = store.start_session()
    assert session_id is not None
    assert session_id.startswith("sess_")

    t_now = time.time()
    # Log 6 ticks across various cognitive states and applications
    store.log_tick(
        session_id=session_id,
        timestamp=t_now,
        dt=1.0,
        cognitive_state=CognitiveState.FLOW.value,
        confidence=0.92,
        context_domain=ContextDomain.CODE_IDE.value,
        process_name="Code.exe",
        window_title="main.py - Visual Studio Code",
        ear=0.29,
        blink_rate=16.0,
        perclos=0.03,
        gaze_stability=0.91,
        keystrokes_per_min=180.0,
        dwell_time_ms=95.0,
        flight_time_ms=160.0,
        mouse_jitter=0.08,
        screen_warmth_kelvin=4200,
        ui_density="COMPACT",
        notification_policy="SUPPRESS_ALL",
        adaptation_executed=True
    )

    store.log_tick(
        session_id=session_id,
        timestamp=t_now + 1.0,
        dt=1.0,
        cognitive_state=CognitiveState.FLOW.value,
        confidence=0.88,
        context_domain=ContextDomain.CODE_IDE.value,
        process_name="Code.exe",
        window_title="main.py - Visual Studio Code",
        ear=0.28,
        blink_rate=15.0,
        perclos=0.03,
        gaze_stability=0.89,
        keystrokes_per_min=175.0,
        dwell_time_ms=98.0,
        flight_time_ms=165.0,
        mouse_jitter=0.09,
        screen_warmth_kelvin=4200,
        ui_density="COMPACT",
        notification_policy="SUPPRESS_ALL",
        adaptation_executed=False
    )

    store.log_tick(
        session_id=session_id,
        timestamp=t_now + 2.0,
        dt=1.0,
        cognitive_state=CognitiveState.NORMAL_FOCUS.value,
        confidence=0.85,
        context_domain=ContextDomain.RESEARCH_BROWSING.value,
        process_name="chrome.exe",
        window_title="Documentation - Google Chrome",
        ear=0.27,
        blink_rate=18.0,
        perclos=0.04,
        gaze_stability=0.84,
        keystrokes_per_min=80.0,
        dwell_time_ms=110.0,
        flight_time_ms=230.0,
        mouse_jitter=0.12,
        screen_warmth_kelvin=5500,
        ui_density="COMFORTABLE",
        notification_policy="PRIORITY_ONLY",
        adaptation_executed=True
    )

    store.close_session(session_id)

    metrics = store.compute_session_metrics(session_id)
    assert metrics is not None
    assert metrics.session_id == session_id
    assert metrics.duration_sec >= 2.0
    assert metrics.active_duration_sec >= 2.0
    assert metrics.total_adaptations == 2
    assert metrics.flow_score > 0.0
    assert metrics.productivity_score > 50.0
    assert metrics.avg_ear > 0.2
    assert metrics.avg_wpm > 0.0
    assert metrics.avg_bpm > 0.0
    assert len(metrics.top_applications) >= 1


def test_format_duration():
    """Verify human-readable duration formatting."""
    assert format_duration(45) == "45s"
    assert format_duration(125) == "2m 5s"
    assert format_duration(3665) == "1h 1m 5s"


def test_html_and_cli_report_generation(temp_db, tmp_path):
    """Verify HTML executive report file generation and CLI printing."""
    store = SessionStore(db_path=temp_db)
    sid = store.start_session()
    
    t0 = time.time()
    store.log_tick(
        session_id=sid,
        timestamp=t0,
        dt=1.0,
        cognitive_state=CognitiveState.FLOW.value,
        confidence=0.95,
        context_domain=ContextDomain.CODE_IDE.value,
        process_name="Code.exe",
        window_title="Engine.py",
        ear=0.30,
        blink_rate=16.0,
        perclos=0.02,
        gaze_stability=0.94,
        keystrokes_per_min=160.0,
        dwell_time_ms=90.0,
        flight_time_ms=170.0,
        mouse_jitter=0.06,
        screen_warmth_kelvin=4200,
        ui_density="COMPACT",
        notification_policy="SUPPRESS_ALL",
        adaptation_executed=True
    )
    store.flush()

    metrics = store.compute_session_metrics(sid)
    assert metrics is not None

    # Test CLI printing (should not raise exceptions)
    print_cli_report(metrics)

    # Test HTML report export
    report_file = tmp_path / "test_report.html"
    out_path = generate_html_report(metrics, report_file)
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "AETHERFLOW PRODUCTIVITY & HEALTH REPORT" in content
    assert sid in content
    assert "Code.exe" in content
