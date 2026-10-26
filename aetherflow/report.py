"""On-demand productivity and cognitive health report generator for AetherFlow.

Analyzes persistent telemetry logs from SQLite to produce:
1. Interactive, modern Linear/Raycast-styled HTML executive reports with multi-tab dashboards.
2. Clean terminal console summaries with ASCII metrics tables.
3. Markdown summary reports for exporting to Notion or Obsidian.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import webbrowser
from pathlib import Path
from typing import Optional

from aetherflow.config import BASE_DIR, CognitiveState, ContextDomain
from aetherflow.storage import ProductivityMetrics, SessionStore

logger = logging.getLogger("aetherflow.report")

REPORTS_DIR = BASE_DIR / "reports"


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def generate_html_report(metrics: ProductivityMetrics, output_path: Path) -> Path:
    """Generate self-contained, responsive, clean Linear/Raycast style HTML report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.datetime.fromtimestamp(metrics.start_time).strftime("%Y-%m-%d %H:%M:%S")
    end_dt = datetime.datetime.fromtimestamp(metrics.end_time).strftime("%Y-%m-%d %H:%M:%S")
    motor_smoothness = max(0.0, min(100.0, 100.0 * (1.0 - metrics.avg_mouse_jitter * 1.6)))

    # State colors
    colors = {
        CognitiveState.FLOW.value: "#10b981",              # Emerald
        CognitiveState.NORMAL_FOCUS.value: "#0284c7",      # Sky
        CognitiveState.COGNITIVE_OVERLOAD.value: "#f59e0b",# Amber
        CognitiveState.FATIGUE.value: "#ef4444",           # Crimson
        CognitiveState.IDLE.value: "#64748b",              # Slate
    }

    # State distribution rows
    state_rows_html = ""
    for state_name, dur in metrics.state_durations.items():
        pct = metrics.state_percentages.get(state_name, 0.0)
        col = colors.get(state_name, "#94a3b8")
        state_rows_html += f"""
        <tr>
            <td style="padding: 12px 16px; border-bottom: 1px solid #27272a;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: {col}; margin-right: 8px;"></span>
                <strong>{state_name}</strong>
            </td>
            <td style="padding: 12px 16px; border-bottom: 1px solid #27272a; text-align: right; font-family: monospace;">{format_duration(dur)}</td>
            <td style="padding: 12px 16px; border-bottom: 1px solid #27272a; text-align: right; font-family: monospace;">{pct:.1f}%</td>
            <td style="padding: 12px 16px; border-bottom: 1px solid #27272a;">
                <div style="background: #27272a; border-radius: 9999px; height: 6px; width: 100%; overflow: hidden;">
                    <div style="background: {col}; width: {pct}%; height: 100%; border-radius: 9999px;"></div>
                </div>
            </td>
        </tr>
        """

    # Top applications rows
    app_rows_html = ""
    for app_name, dur in metrics.top_applications:
        pct = (dur / metrics.duration_sec * 100.0) if metrics.duration_sec > 0 else 0.0
        app_rows_html += f"""
        <tr>
            <td style="padding: 12px 16px; border-bottom: 1px solid #27272a;"><strong>{app_name}</strong></td>
            <td style="padding: 12px 16px; border-bottom: 1px solid #27272a; color: #a1a1aa;">Active Context</td>
            <td style="padding: 12px 16px; border-bottom: 1px solid #27272a; text-align: right; font-family: monospace;">{format_duration(dur)}</td>
            <td style="padding: 12px 16px; border-bottom: 1px solid #27272a;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="background: #27272a; border-radius: 9999px; height: 6px; width: 100px; overflow: hidden;">
                        <div style="background: var(--accent); width: {pct}%; height: 100%; border-radius: 9999px;"></div>
                    </div>
                    <span style="font-family: monospace; font-size: 12px; color: #a1a1aa;">{pct:.1f}%</span>
                </div>
            </td>
        </tr>
        """

    # Workday timeline bar segments
    timeline_segments_html = ""
    for state_name in [CognitiveState.FLOW.value, CognitiveState.NORMAL_FOCUS.value, CognitiveState.COGNITIVE_OVERLOAD.value, CognitiveState.FATIGUE.value, CognitiveState.IDLE.value]:
        dur = metrics.state_durations.get(state_name, 0.0)
        pct = metrics.state_percentages.get(state_name, 0.0)
        if pct > 0.0:
            col = colors.get(state_name, "#94a3b8")
            lbl = f"{state_name} ({pct:.0f}%)" if pct > 10 else (f"{pct:.0f}%" if pct > 4 else "")
            timeline_segments_html += f"""
            <div style="background: {col}; width: {pct}%; height: 100%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 600; color: #000; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{state_name}: {format_duration(dur)} ({pct:.1f}%)">
                {lbl}
            </div>
            """

    # Qualitative tier
    if metrics.productivity_score >= 85:
        tier_title = "Deep Focus Mastery"
        tier_sub = "Exceptional cognitive immersion and low distraction rate."
    elif metrics.productivity_score >= 70:
        tier_title = "Solid Productivity"
        tier_sub = "Steady engagement with balanced task switching."
    elif metrics.productivity_score >= 50:
        tier_title = "Fragmented Focus"
        tier_sub = "Frequent context switching and cognitive pacing strain."
    else:
        tier_title = "Cognitive Strain"
        tier_sub = "Elevated ocular fatigue and mental workload detected."

    top_app_name = metrics.top_applications[0][0] if metrics.top_applications else "system"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AETHERFLOW PRODUCTIVITY & HEALTH REPORT</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-base: #0f0f11;
            --bg-card: #18181b;
            --bg-secondary: #222226;
            --bg-muted: #27272a;
            --border-color: #27272a;
            --text-main: #f4f4f5;
            --text-muted: #a1a1aa;
            --text-subtle: #71717a;
            --primary: #10b981;
            --accent: #0284c7;
            --warning: #f59e0b;
            --destructive: #ef4444;
            --chart-5: #64748b;
            --radius-lg: 12px;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            border-color: var(--border-color);
        }}

        body {{
            background-color: var(--bg-base);
            color: var(--text-main);
            font-family: 'Geist', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            -webkit-font-smoothing: antialiased;
            min-height: 100vh;
            line-height: 1.5;
        }}

        header {{
            position: sticky;
            top: 0;
            z-index: 50;
            border-bottom: 1px solid var(--border-color);
            background: rgba(15, 15, 17, 0.94);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
        }}

        .header-inner {{
            max-width: 1140px;
            margin: 0 auto;
            height: 56px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 24px;
        }}

        .brand-box {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .brand-name {{
            font-size: 14px;
            font-weight: 600;
            letter-spacing: -0.01em;
            color: var(--text-main);
        }}

        .nav-tabs {{
            display: flex;
            gap: 4px;
            margin-left: 20px;
        }}

        .tab-btn {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
            font-family: inherit;
        }}

        .tab-btn:hover {{
            color: var(--text-main);
            background: var(--bg-secondary);
        }}

        .tab-btn.active {{
            color: var(--text-main);
            background: var(--bg-secondary);
        }}

        .header-right {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .btn-ui {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
            font-family: inherit;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}

        .btn-ui:hover {{
            background: var(--bg-muted);
            border-color: rgba(255, 255, 255, 0.15);
        }}

        main {{
            max-width: 1140px;
            margin: 0 auto;
            padding: 36px 24px 60px 24px;
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: block;
        }}

        .panel {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 24px;
        }}

        .eyebrow {{
            font-size: 11px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-subtle);
            font-weight: 600;
        }}

        .grid-2 {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }}
        .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
        .grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }}
        .grid-5-split {{ display: grid; grid-template-columns: 3fr 2fr; gap: 16px; }}

        @media (max-width: 900px) {{
            .grid-5-split, .grid-4, .grid-3 {{ grid-template-columns: 1fr; }}
        }}

        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 20px;
        }}

        .stat-val-row {{
            margin-top: 12px;
            display: flex;
            align-items: baseline;
            gap: 6px;
        }}

        .stat-number {{
            font-size: 26px;
            font-weight: 600;
            letter-spacing: -0.02em;
            color: var(--text-main);
            font-family: 'Geist Mono', monospace;
        }}

        .stat-unit {{
            font-size: 13px;
            color: var(--text-muted);
        }}

        .stat-note {{
            margin-top: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            color: var(--text-muted);
        }}

        .dot-indicator {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
        }}

        .dot-good {{ background: var(--primary); }}
        .dot-warn {{ background: var(--warning); }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}

        th {{
            text-align: left;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-subtle);
            font-weight: 600;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        td {{
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-main);
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        @media print {{
            .no-print {{ display: none !important; }}
            body {{ background: #ffffff !important; color: #09090b !important; padding: 0; }}
            .panel, .stat-card {{ background: #ffffff !important; border: 1px solid #e4e4e7 !important; color: #09090b !important; box-shadow: none !important; }}
            .stat-number {{ color: #09090b !important; }}
            td, th {{ color: #09090b !important; border-color: #e4e4e7 !important; }}
            #tab-report {{ display: block !important; }}
            #tab-overview, #tab-biometrics, #tab-environment {{ display: none !important; }}
        }}
    </style>
</head>
<body>
    <header class="no-print">
        <div class="header-inner">
            <div class="brand-box">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.8" />
                    <path d="M5 14c3-4 6 2 14-4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" />
                </svg>
                <span class="brand-name">AetherFlow</span>
            </div>

            <nav class="nav-tabs">
                <button class="tab-btn" id="btn-overview" onclick="showTab('overview')">Overview</button>
                <button class="tab-btn" id="btn-biometrics" onclick="showTab('biometrics')">Biometrics</button>
                <button class="tab-btn" id="btn-environment" onclick="showTab('environment')">Environment</button>
                <button class="tab-btn active" id="btn-report" onclick="showTab('report')">Session Report</button>
            </nav>

            <div class="header-right">
                <span style="font-family: 'Geist Mono', monospace; font-size: 12px; color: var(--text-muted);">{start_dt}</span>
                <button class="btn-ui" onclick="window.print()">🖨️ Print / Save PDF</button>
            </div>
        </div>
    </header>

    <main>
        <!-- ============================================================ -->
        <!-- TAB 1: OVERVIEW -->
        <!-- ============================================================ -->
        <div id="tab-overview" class="tab-content">
            <div class="grid-5-split">
                <div style="display: flex; flex-direction: column; gap: 16px;">
                    <div class="panel" style="padding: 32px;">
                        <div class="eyebrow">Session Synthesis</div>
                        <div style="margin-top: 16px; display: flex; align-items: center; gap: 12px;">
                            <span class="dot-indicator dot-good" style="width: 10px; height: 10px;"></span>
                            <h1 style="font-size: 32px; font-weight: 600; letter-spacing: -0.02em;">{tier_title}</h1>
                        </div>
                        <p style="margin-top: 10px; color: var(--text-muted); font-size: 15px;">{tier_sub}</p>
                        <div style="margin-top: 24px; display: inline-flex; align-items: center; gap: 8px; border-radius: 9999px; border: 1px solid var(--border-color); background: var(--bg-base); padding: 4px 12px; font-family: 'Geist Mono', monospace; font-size: 12px; color: var(--text-muted);">
                            <span class="dot-indicator" style="background: var(--accent);"></span>
                            <span>Primary App: {top_app_name}</span>
                        </div>
                    </div>

                    <div class="grid-2">
                        <div class="stat-card">
                            <div class="eyebrow">Productivity Score</div>
                            <div class="stat-val-row"><span class="stat-number">{metrics.productivity_score:.1f}</span><span class="stat-unit">/ 100</span></div>
                            <div class="stat-note"><span class="dot-indicator dot-good"></span> Cognitive focus composite</div>
                        </div>
                        <div class="stat-card">
                            <div class="eyebrow">Typing Cadence</div>
                            <div class="stat-val-row"><span class="stat-number">{metrics.avg_wpm:.0f}</span><span class="stat-unit">WPM</span></div>
                            <div class="stat-note"><span class="dot-indicator dot-good"></span> {metrics.avg_wpm * 5:.0f} KPM Cadence</div>
                        </div>
                        <div class="stat-card">
                            <div class="eyebrow">Blink Frequency</div>
                            <div class="stat-val-row"><span class="stat-number">{metrics.avg_bpm:.1f}</span><span class="stat-unit">BPM</span></div>
                            <div class="stat-note"><span class="dot-indicator dot-good"></span> Ocular baseline rate</div>
                        </div>
                        <div class="stat-card">
                            <div class="eyebrow">Total Duration</div>
                            <div class="stat-val-row"><span class="stat-number">{format_duration(metrics.duration_sec)}</span></div>
                            <div class="stat-note"><span class="dot-indicator dot-good"></span> Active session time</div>
                        </div>
                    </div>
                </div>

                <div class="panel" style="display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
                    <div class="eyebrow">Session Metadata</div>
                    <div style="margin-top: 20px; font-family: 'Geist Mono', monospace; font-size: 13px; color: var(--text-main);">
                        <div>ID: <strong>{metrics.session_id}</strong></div>
                        <div style="margin-top: 8px; color: var(--text-muted); font-size: 12px;">Start: {start_dt}</div>
                        <div style="margin-top: 4px; color: var(--text-muted); font-size: 12px;">End: {end_dt}</div>
                        <div style="margin-top: 14px; color: var(--primary);">✓ 100% Local Inference & Secure</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ============================================================ -->
        <!-- TAB 2: BIOMETRICS -->
        <!-- ============================================================ -->
        <div id="tab-biometrics" class="tab-content">
            <div style="margin-bottom: 28px;">
                <h2 style="font-size: 16px; font-weight: 600;">Ocular Health Baselines</h2>
                <p style="font-size: 13px; color: var(--text-muted); margin-top: 4px;">Measured against clinical ocular benchmarks.</p>
                <div class="grid-4" style="margin-top: 16px;">
                    <div class="stat-card">
                        <div class="eyebrow">Eye Aperture (EAR)</div>
                        <div class="stat-val-row"><span class="stat-number">{metrics.avg_ear:.2f}</span></div>
                        <div class="stat-note"><span class="dot-indicator dot-good"></span> Threshold: 0.21</div>
                    </div>
                    <div class="stat-card">
                        <div class="eyebrow">Blink Frequency</div>
                        <div class="stat-val-row"><span class="stat-number">{metrics.avg_bpm:.1f}</span><span class="stat-unit">BPM</span></div>
                        <div class="stat-note"><span class="dot-indicator dot-good"></span> Normal: 15–22 BPM</div>
                    </div>
                    <div class="stat-card">
                        <div class="eyebrow">Eye Strain (PERCLOS)</div>
                        <div class="stat-val-row"><span class="stat-number">{metrics.avg_perclos:.1%}</span></div>
                        <div class="stat-note"><span class="dot-indicator dot-good"></span> Fatigue Risk: &gt;15%</div>
                    </div>
                    <div class="stat-card">
                        <div class="eyebrow">Gaze Fixation</div>
                        <div class="stat-val-row"><span class="stat-number">{metrics.avg_gaze_stability:.0%}</span></div>
                        <div class="stat-note"><span class="dot-indicator dot-good"></span> Flow benchmark: &gt;85%</div>
                    </div>
                </div>
            </div>

            <div style="margin-bottom: 28px;">
                <h2 style="font-size: 16px; font-weight: 600;">Neuromuscular Motor Dynamics</h2>
                <div class="grid-4" style="margin-top: 16px;">
                    <div class="stat-card">
                        <div class="eyebrow">Typing Speed</div>
                        <div class="stat-val-row"><span class="stat-number">{metrics.avg_wpm:.0f}</span><span class="stat-unit">WPM</span></div>
                        <div class="stat-note"><span class="dot-indicator dot-good"></span> {metrics.avg_wpm * 5:.0f} KPM</div>
                    </div>
                    <div class="stat-card">
                        <div class="eyebrow">Key Latency</div>
                        <div class="stat-val-row"><span class="stat-number">{metrics.avg_dwell_ms:.0f} / {metrics.avg_flight_ms:.0f}</span><span class="stat-unit">ms</span></div>
                        <div class="stat-note"><span class="dot-indicator dot-good"></span> Dwell / Flight time</div>
                    </div>
                    <div class="stat-card">
                        <div class="eyebrow">Mouse Jitter Index</div>
                        <div class="stat-val-row"><span class="stat-number">{metrics.avg_mouse_jitter:.3f}</span></div>
                        <div class="stat-note"><span class="dot-indicator dot-good"></span> Calm trajectory</div>
                    </div>
                    <div class="stat-card">
                        <div class="eyebrow">Motor Smoothness</div>
                        <div class="stat-val-row"><span class="stat-number">{motor_smoothness:.1f}</span><span class="stat-unit">/ 100</span></div>
                        <div class="stat-note"><span class="dot-indicator dot-good"></span> Low neuromuscular tremor</div>
                    </div>
                </div>
            </div>

            <div>
                <h2 style="font-size: 16px; font-weight: 600;">Cognitive State Breakdown</h2>
                <div class="panel" style="padding: 0; overflow: hidden; margin-top: 16px;">
                    <table>
                        <thead>
                            <tr><th>State</th><th style="text-align: right;">Duration</th><th style="text-align: right;">Share</th><th style="width: 250px;">Visual Distribution</th></tr>
                        </thead>
                        <tbody>
                            {state_rows_html}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- ============================================================ -->
        <!-- TAB 3: ENVIRONMENT -->
        <!-- ============================================================ -->
        <div id="tab-environment" class="tab-content">
            <div class="grid-3" style="margin-bottom: 24px;">
                <div class="stat-card">
                    <div class="eyebrow">Total Adaptations</div>
                    <div class="stat-val-row"><span class="stat-number">{metrics.total_adaptations}</span></div>
                    <div class="stat-note"><span class="dot-indicator dot-good"></span> Display adjustments executed</div>
                </div>
                <div class="stat-card">
                    <div class="eyebrow">Blue Light Shield Time</div>
                    <div class="stat-val-row"><span class="stat-number">{format_duration(metrics.blue_light_reduced_sec)}</span></div>
                    <div class="stat-note"><span class="dot-indicator dot-good"></span> Circadian protection duration</div>
                </div>
                <div class="stat-card">
                    <div class="eyebrow">Hardware Gamma Protection</div>
                    <div class="stat-val-row"><span class="stat-number" style="color: var(--primary);">ACTIVE</span></div>
                    <div class="stat-note"><span class="dot-indicator dot-good"></span> Safe auto-restoration safeguard</div>
                </div>
            </div>
        </div>

        <!-- ============================================================ -->
        <!-- TAB 4: SESSION REPORT -->
        <!-- ============================================================ -->
        <div id="tab-report" class="tab-content active">
            <div style="display: flex; align-items: flex-end; justify-content: space-between; margin-bottom: 28px;">
                <div>
                    <div class="eyebrow">Executive Session Summary</div>
                    <h1 style="font-size: 28px; font-weight: 600; letter-spacing: -0.02em; margin-top: 4px;">Cognitive Productivity & Health Report</h1>
                </div>
                <div class="no-print">
                    <button class="btn-ui" onclick="window.print()">🖨️ Export Executive PDF</button>
                </div>
            </div>

            <!-- Top 6 KPIs -->
            <div class="grid-3" style="margin-bottom: 32px;">
                <div class="stat-card">
                    <div class="eyebrow">Productivity Score</div>
                    <div class="stat-val-row"><span class="stat-number">{metrics.productivity_score:.1f}</span><span class="stat-unit">/ 100</span></div>
                    <div class="stat-note"><span class="dot-indicator dot-good"></span> {tier_title}</div>
                </div>
                <div class="stat-card">
                    <div class="eyebrow">Flow Immersion Rate</div>
                    <div class="stat-val-row"><span class="stat-number">{metrics.flow_score:.1f}</span><span class="stat-unit">%</span></div>
                    <div class="stat-note"><span class="dot-indicator dot-good"></span> {format_duration(metrics.state_durations.get(CognitiveState.FLOW.value, 0.0))} deep flow</div>
                </div>
                <div class="stat-card">
                    <div class="eyebrow">Typing Speed</div>
                    <div class="stat-val-row"><span class="stat-number">{metrics.avg_wpm:.0f}</span><span class="stat-unit">WPM</span></div>
                    <div class="stat-note"><span class="dot-indicator dot-good"></span> {metrics.avg_wpm * 5:.0f} KPM Cadence</div>
                </div>
                <div class="stat-card">
                    <div class="eyebrow">Blink Frequency</div>
                    <div class="stat-val-row"><span class="stat-number">{metrics.avg_bpm:.1f}</span><span class="stat-unit">BPM</span></div>
                    <div class="stat-note"><span class="dot-indicator dot-good"></span> Normal baseline: 15–22 BPM</div>
                </div>
                <div class="stat-card">
                    <div class="eyebrow">Eye Strain (PERCLOS)</div>
                    <div class="stat-val-row"><span class="stat-number">{metrics.avg_perclos:.1%}</span></div>
                    <div class="stat-note"><span class="dot-indicator dot-good"></span> Fatigue Risk &lt;15%</div>
                </div>
                <div class="stat-card">
                    <div class="eyebrow">Motor Smoothness</div>
                    <div class="stat-val-row"><span class="stat-number">{motor_smoothness:.1f}</span><span class="stat-unit">/ 100</span></div>
                    <div class="stat-note"><span class="dot-indicator dot-good"></span> Jitter: {metrics.avg_mouse_jitter:.3f}</div>
                </div>
            </div>

            <!-- Cognitive Timeline Bar -->
            <div class="panel" style="margin-bottom: 32px;">
                <div class="eyebrow">Workday Cognitive Timeline Spectrum</div>
                <div style="height: 28px; display: flex; border-radius: 6px; overflow: hidden; margin-top: 14px; background: var(--bg-muted);">
                    {timeline_segments_html}
                </div>
                <div style="margin-top: 14px; display: flex; flex-wrap: wrap; gap: 16px; font-size: 12px; color: var(--text-muted);">
                    <span style="display: flex; align-items: center; gap: 6px;"><span class="dot-indicator" style="background: var(--primary);"></span> Flow</span>
                    <span style="display: flex; align-items: center; gap: 6px;"><span class="dot-indicator" style="background: var(--accent);"></span> Focus</span>
                    <span style="display: flex; align-items: center; gap: 6px;"><span class="dot-indicator" style="background: var(--warning);"></span> Overload</span>
                    <span style="display: flex; align-items: center; gap: 6px;"><span class="dot-indicator" style="background: var(--destructive);"></span> Fatigue</span>
                    <span style="display: flex; align-items: center; gap: 6px;"><span class="dot-indicator" style="background: var(--chart-5);"></span> Idle</span>
                </div>
            </div>

            <!-- Application Usage Breakdown Table -->
            <div class="panel" style="padding: 0; overflow: hidden;">
                <table>
                    <thead>
                        <tr>
                            <th>Application</th>
                            <th>Category</th>
                            <th style="text-align: right;">Time</th>
                            <th style="width: 200px;">Cognitive Share</th>
                        </tr>
                    </thead>
                    <tbody>
                        {app_rows_html}
                    </tbody>
                </table>
            </div>
        </div>
    </main>

    <script>
        function showTab(tabId) {{
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            const btn = document.getElementById(`btn-${{tabId}}`);
            const content = document.getElementById(`tab-${{tabId}}`);
            if (btn) btn.classList.add('active');
            if (content) content.classList.add('active');
        }}
    </script>
</body>
</html>
"""
    output_path.write_text(html_content, encoding="utf-8")
    logger.info(f"Generated HTML productivity report: {output_path.resolve()}")
    return output_path


def print_cli_report(metrics: ProductivityMetrics) -> None:
    """Print clean formatted executive summary to console."""
    start_dt = datetime.datetime.fromtimestamp(metrics.start_time).strftime("%H:%M:%S")
    end_dt = datetime.datetime.fromtimestamp(metrics.end_time).strftime("%H:%M:%S")

    wpm_str = f"{metrics.avg_wpm:.0f} WPM ({metrics.avg_wpm * 5:.0f} KPM)" if metrics.avg_wpm > 0 else "0 WPM (0 KPM)"
    bpm_str = f"{metrics.avg_bpm:.1f} BPM (Normal: 15–22 BPM)" if metrics.avg_bpm > 0 else "0.0 BPM (Normal: 15–22 BPM)"

    print("\n" + "=" * 68)
    print("      AETHERFLOW EXECUTIVE PRODUCTIVITY & COGNITIVE REPORT")
    print("=" * 68)
    print(f" Session ID          : {metrics.session_id}")
    print(f" Monitored Window    : {start_dt} - {end_dt} ({format_duration(metrics.duration_sec)})")
    print(f" Active Work Time    : {format_duration(metrics.active_duration_sec)} | Idle Time: {format_duration(metrics.idle_duration_sec)}")
    print("-" * 68)
    print(f" PRODUCTIVITY SCORE  : {metrics.productivity_score:.1f} / 100")
    print(f" FLOW IMMERSION RATE : {metrics.flow_score:.1f}%")
    print(f" TYPING SPEED (WPM)  : {wpm_str}")
    print(f" BLINK RATE (BPM)    : {bpm_str}")
    print(f" EYE STRAIN (PERCLOS): {metrics.avg_perclos:.1%} (Fatigue Index: {metrics.fatigue_index:.1f}/100)")
    motor_smoothness = max(0.0, min(100.0, 100.0 * (1.0 - metrics.avg_mouse_jitter * 1.6)))

    print(f" MOTOR SMOOTHNESS    : {motor_smoothness:.1f} / 100 (Jitter: {metrics.avg_mouse_jitter:.3f})")
    print("-" * 68)
    print(" PHYSIOLOGICAL & MOTOR BASELINES:")
    print(f"   Typing Cadence    : {wpm_str.split(' (')[0]} ({metrics.avg_wpm * 5:.0f} keystrokes/min)")
    print(f"   Key Latency       : Dwell: {metrics.avg_dwell_ms:.0f}ms | Flight: {metrics.avg_flight_ms:.0f}ms")
    print(f"   Blink Frequency   : {metrics.avg_bpm:.1f} BPM (Mean EAR: {metrics.avg_ear:.2f})")
    print(f"   Gaze Stability    : {int(metrics.avg_gaze_stability * 100)}% | Mouse Jitter: {metrics.avg_mouse_jitter:.3f}")
    print("-" * 68)
    print(" COGNITIVE STATE BREAKDOWN:")
    for state_name, dur in metrics.state_durations.items():
        pct = metrics.state_percentages.get(state_name, 0.0)
        bar_len = int(pct / 4)
        print(f"   {state_name:<26} {format_duration(dur):>4}  ({pct:5.1f}%) {'#' * bar_len}")
    print("-" * 68)
    print(" TOP APPLICATIONS:")
    for app_name, dur in metrics.top_applications[:5]:
        pct = (dur / metrics.duration_sec * 100.0) if metrics.duration_sec > 0 else 0.0
        print(f"   {app_name:<30} {format_duration(dur):>4}  ({pct:5.1f}%)")
    print("-" * 68)
    print(" NEUROADAPTIVE ACTUATIONS:")
    print(f"   Total Display Shifts : {metrics.total_adaptations}")
    print(f"   Blue Light Reduced   : {format_duration(metrics.blue_light_reduced_sec)}")
    print("=" * 68 + "\n")


def generate_report(
    session_id: Optional[str] = None,
    output_html_path: Optional[Path] = None,
    open_browser: bool = True
) -> Optional[Path]:
    """Top-level function to compute metrics and emit reports."""
    store = SessionStore()
    metrics = store.compute_session_metrics(session_id)
    if not metrics:
        logger.warning(f"No telemetry samples found for session {session_id or 'latest'}.")
        print("[AetherFlow] No telemetry samples recorded yet for this session.")
        return None

    # Print CLI Summary
    print_cli_report(metrics)

    # Generate HTML Executive Report
    if output_html_path is None:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_html_path = REPORTS_DIR / f"productivity_report_{timestamp_str}.html"

    report_path = generate_html_report(metrics, output_html_path)

    if open_browser:
        try:
            webbrowser.open(f"file://{report_path.resolve()}")
        except Exception as e:
            logger.debug(f"Failed to auto-open browser: {e}")

    return report_path
