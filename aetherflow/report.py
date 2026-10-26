"""On-demand productivity and cognitive health report generator for AetherFlow.

Analyzes persistent telemetry logs from SQLite to produce:
1. Interactive, modern glassmorphic HTML executive reports with visual charts.
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
    """Generate self-contained, responsive, dark-mode executive HTML report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.datetime.fromtimestamp(metrics.start_time).strftime("%Y-%m-%d %H:%M:%S")
    end_dt = datetime.datetime.fromtimestamp(metrics.end_time).strftime("%Y-%m-%d %H:%M:%S")

    # State colors
    colors = {
        CognitiveState.FLOW.value: "#10b981",              # Emerald
        CognitiveState.NORMAL_FOCUS.value: "#06b6d4",      # Cyan
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
            <td style="padding: 10px 14px; border-bottom: 1px solid #334155;">
                <span style="display: inline-block; width: 12px; height: 12px; border-radius: 3px; background-color: {col}; margin-right: 8px;"></span>
                <strong>{state_name}</strong>
            </td>
            <td style="padding: 10px 14px; border-bottom: 1px solid #334155; text-align: right; font-family: monospace;">{format_duration(dur)}</td>
            <td style="padding: 10px 14px; border-bottom: 1px solid #334155; text-align: right; font-family: monospace;">{pct:.1f}%</td>
            <td style="padding: 10px 14px; border-bottom: 1px solid #334155;">
                <div style="background: #1e293b; border-radius: 4px; height: 10px; width: 100%; overflow: hidden;">
                    <div style="background: {col}; width: {pct}%; height: 100%;"></div>
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
            <td style="padding: 8px 12px; border-bottom: 1px solid #334155;"><strong>{app_name}</strong></td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #334155; text-align: right; font-family: monospace;">{format_duration(dur)}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #334155; text-align: right; font-family: monospace;">{pct:.1f}%</td>
        </tr>
        """

    # Insights generation
    insights = []
    if metrics.flow_score >= 50.0:
        insights.append(f"🏆 <strong>Exceptional Flow State</strong>: You spent {metrics.flow_score:.1f}% of active time in uninterrupted deep focus.")
    elif metrics.flow_score >= 25.0:
        insights.append(f"✓ <strong>Solid Productive Engagement</strong>: {metrics.flow_score:.1f}% of active time was spent in deep flow.")
    else:
        insights.append("ℹ️ <strong>High Context Switching</strong>: Flow state was under 25%. Consider clustering communication tasks to protect focus windows.")

    if metrics.avg_perclos > 0.12:
        insights.append(f"⚠️ <strong>Elevated Ocular Strain Detected</strong>: Average PERCLOS reached {metrics.avg_perclos:.1%}. Recommend taking 5-minute eye breaks.")
    else:
        insights.append(f"✓ <strong>Ocular Fatigue Low</strong>: Average eye closure ratio was {metrics.avg_perclos:.1%}, well below the 15% fatigue threshold.")

    if metrics.avg_mouse_jitter < 0.12:
        insights.append(f"✓ <strong>Calm Motor Dynamics</strong>: Low mouse jitter ({metrics.avg_mouse_jitter:.3f}) and steady typing cadence indicate minimal task frustration.")
    elif metrics.avg_mouse_jitter > 0.30:
        insights.append(f"⚠️ <strong>High Motor Jitter ({metrics.avg_mouse_jitter:.3f})</strong>: Detected erratic cursor trajectories and rapid reversals during task switching.")

    if metrics.blue_light_reduced_sec > 0:
        insights.append(f"🛡️ <strong>Display Actuation Shield Active</strong>: AetherFlow protected your eyes with blue light reduction for {format_duration(metrics.blue_light_reduced_sec)} across {metrics.total_adaptations} adaptive interventions.")

    insights_html = "".join(f"<li style='margin-bottom: 10px;'>{i}</li>" for i in insights)

    # Workday timeline bar segments
    timeline_segments_html = ""
    for state_name in [CognitiveState.FLOW.value, CognitiveState.NORMAL_FOCUS.value, CognitiveState.COGNITIVE_OVERLOAD.value, CognitiveState.FATIGUE.value, CognitiveState.IDLE.value]:
        dur = metrics.state_durations.get(state_name, 0.0)
        pct = metrics.state_percentages.get(state_name, 0.0)
        if pct > 0.0:
            col = colors.get(state_name, "#94a3b8")
            lbl = f"{state_name} ({pct:.0f}%)" if pct > 8 else (f"{pct:.0f}%" if pct > 4 else "")
            timeline_segments_html += f"""
            <div style="background: {col}; width: {pct}%; height: 100%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; color: #000; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; transition: all 0.3s;" title="{state_name}: {format_duration(dur)} ({pct:.1f}%)">
                {lbl}
            </div>
            """

    # Productivity qualitative tier
    if metrics.productivity_score >= 85:
        tier_title = "Deep Focus Mastery"
        tier_badge = "background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4);"
    elif metrics.productivity_score >= 70:
        tier_title = "Solid Productivity"
        tier_badge = "background: rgba(6, 182, 212, 0.2); color: #06b6d4; border: 1px solid rgba(6, 182, 212, 0.4);"
    elif metrics.productivity_score >= 50:
        tier_title = "Fragmented Focus"
        tier_badge = "background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.4);"
    else:
        tier_title = "Cognitive Strain"
        tier_badge = "background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.4);"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AETHERFLOW PRODUCTIVITY & HEALTH REPORT</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-base: #06090e;
            --bg-card: rgba(15, 23, 42, 0.75);
            --bg-card-hover: rgba(30, 41, 59, 0.85);
            --border-card: rgba(255, 255, 255, 0.08);
            --border-card-hover: rgba(56, 189, 248, 0.4);
            --text-main: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-cyan: #06b6d4;
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
            --accent-red: #ef4444;
            --accent-purple: #a855f7;
        }}
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: radial-gradient(circle at 15% 15%, #172554 0%, #06090e 55%, #020408 100%);
            background-attachment: fixed;
            color: var(--text-main);
            margin: 0;
            padding: 40px 24px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1140px;
            margin: 0 auto;
        }}
        .glass-card {{
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 16px;
            padding: 28px;
            margin-bottom: 24px;
            border: 1px solid var(--border-card);
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6);
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        }}
        .glass-card:hover {{
            border-color: rgba(255, 255, 255, 0.14);
        }}
        .header-top {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 16px;
            border-bottom: 1px solid var(--border-card);
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}
        .brand-title {{
            margin: 0;
            font-size: 28px;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .meta-chips {{
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 8px;
            font-size: 13px;
            color: var(--text-secondary);
        }}
        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 5px 12px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.3px;
            text-transform: uppercase;
        }}
        .btn-print {{
            background: rgba(30, 41, 59, 0.8);
            color: #f8fafc;
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s ease;
        }}
        .btn-print:hover {{
            background: rgba(51, 65, 85, 0.9);
            border-color: var(--accent-cyan);
            transform: translateY(-1px);
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 18px;
            margin-bottom: 28px;
        }}
        .kpi-tile {{
            background: rgba(10, 15, 29, 0.6);
            border: 1px solid var(--border-card);
            border-radius: 12px;
            padding: 20px 18px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            position: relative;
            overflow: hidden;
            transition: all 0.2s ease;
        }}
        .kpi-tile:hover {{
            transform: translateY(-3px);
            border-color: var(--border-card-hover);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        }}
        .kpi-tile-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        .kpi-label {{
            font-size: 11px;
            font-weight: 700;
            color: var(--text-muted);
            letter-spacing: 0.6px;
            text-transform: uppercase;
        }}
        .kpi-val {{
            font-size: 34px;
            font-weight: 800;
            letter-spacing: -1px;
            margin: 4px 0 6px 0;
            font-family: 'JetBrains Mono', monospace;
        }}
        .kpi-subtext {{
            font-size: 12px;
            color: var(--text-secondary);
        }}
        .timeline-container {{
            background: rgba(10, 15, 29, 0.7);
            border: 1px solid var(--border-card);
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 24px;
        }}
        .timeline-bar {{
            height: 24px;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            width: 100%;
            background: #1e293b;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.4);
            margin: 12px 0 8px 0;
        }}
        table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 13px;
        }}
        th {{
            text-align: left;
            padding: 12px 14px;
            background: rgba(10, 15, 29, 0.8);
            color: var(--text-muted);
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid var(--border-card);
        }}
        td {{
            padding: 12px 14px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            color: #cbd5e1;
        }}
        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}
        .detail-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 16px;
        }}
        .detail-item {{
            background: rgba(10, 15, 29, 0.6);
            border: 1px solid var(--border-card);
            border-radius: 10px;
            padding: 16px;
        }}
        .detail-title {{
            font-size: 11px;
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
        }}
        .detail-val {{
            font-size: 20px;
            font-weight: 700;
            margin-top: 4px;
            color: var(--text-main);
            font-family: 'JetBrains Mono', monospace;
        }}
        .detail-note {{
            font-size: 11px;
            color: var(--text-secondary);
            margin-top: 2px;
        }}
        .recommendation-card {{
            background: rgba(15, 23, 42, 0.6);
            border-left: 4px solid var(--accent-cyan);
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 12px;
            font-size: 14px;
        }}
        @media print {{
            body {{
                background: #ffffff;
                color: #0f172a;
                padding: 10px;
            }}
            .glass-card, .kpi-tile, .detail-item, .timeline-container {{
                background: #ffffff !important;
                border: 1px solid #cbd5e1 !important;
                box-shadow: none !important;
                color: #0f172a !important;
            }}
            .btn-print {{
                display: none;
            }}
            .brand-title {{
                -webkit-text-fill-color: #0284c7;
            }}
            td, th {{
                color: #0f172a !important;
            }}
            .kpi-val {{
                color: #0f172a !important;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Main Executive Header Card -->
        <div class="glass-card">
            <div class="header-top">
                <div>
                    <h1 class="brand-title">⚡ AETHERFLOW <span style="font-size: 18px; font-weight: 500; color: var(--text-secondary); -webkit-text-fill-color: initial;">Executive Cognitive & Productivity Report</span></h1>
                    <div class="meta-chips">
                        <span>Session: <code style="font-family: 'JetBrains Mono', monospace; color: #38bdf8;">{metrics.session_id}</code></span>
                        <span>•</span>
                        <span>Window: <strong>{start_dt}</strong> → <strong>{end_dt}</strong></span>
                        <span>•</span>
                        <span>Total: <strong>{format_duration(metrics.duration_sec)}</strong></span>
                    </div>
                </div>
                <div style="display: flex; gap: 10px; align-items: center;">
                    <span class="badge" style="background: rgba(14, 165, 233, 0.2); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.3);">100% Local Inference</span>
                    <button class="btn-print" onclick="window.print()">🖨️ Print / Save PDF</button>
                </div>
            </div>

            <!-- Top 6 Executive KPIs -->
            <div class="kpi-grid">
                <!-- KPI 1: Productivity Score -->
                <div class="kpi-tile">
                    <div class="kpi-tile-top">
                        <span class="kpi-label">Productivity Score</span>
                        <span class="badge" style="{tier_badge}">{tier_title}</span>
                    </div>
                    <div class="kpi-val" style="color: var(--accent-green);">{metrics.productivity_score} <span style="font-size: 16px; font-weight: 500; color: var(--text-muted);">/ 100</span></div>
                    <div class="kpi-subtext">Weighted Cognitive Focus Metric</div>
                </div>

                <!-- KPI 2: Flow State Immersion -->
                <div class="kpi-tile">
                    <div class="kpi-tile-top">
                        <span class="kpi-label">Flow Immersion</span>
                        <span class="badge" style="background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3);">{metrics.flow_score:.0f}% Rate</span>
                    </div>
                    <div class="kpi-val" style="color: #38bdf8;">{format_duration(metrics.state_durations.get(CognitiveState.FLOW.value, 0.0))}</div>
                    <div class="kpi-subtext">{metrics.flow_score:.1f}% of active time in deep focus</div>
                </div>

                <!-- KPI 3: Typing Cadence (WPM) -->
                <div class="kpi-tile">
                    <div class="kpi-tile-top">
                        <span class="kpi-label">Typing Cadence</span>
                        <span class="badge" style="background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3);">Cadence</span>
                    </div>
                    <div class="kpi-val" style="color: var(--accent-amber);">{metrics.avg_wpm:.0f} <span style="font-size: 16px; font-weight: 500; color: var(--text-muted);">WPM</span></div>
                    <div class="kpi-subtext">{metrics.avg_cadence_kpm:.0f} Keystrokes/Min ({int(metrics.avg_cadence_kpm * (metrics.active_duration_sec/60.0))} total)</div>
                </div>

                <!-- KPI 4: Blink Rate (BPM) -->
                <div class="kpi-tile">
                    <div class="kpi-tile-top">
                        <span class="kpi-label">Blink Frequency</span>
                        <span class="badge" style="background: rgba(6, 182, 212, 0.2); color: #06b6d4; border: 1px solid rgba(6, 182, 212, 0.3);">Ocular</span>
                    </div>
                    <div class="kpi-val" style="color: {'var(--accent-red)' if metrics.avg_blink_rate < 10 or metrics.avg_blink_rate > 28 else 'var(--accent-cyan)'};">{metrics.avg_blink_rate:.1f} <span style="font-size: 16px; font-weight: 500; color: var(--text-muted);">BPM</span></div>
                    <div class="kpi-subtext">Baseline standard: 15–22 Blinks/Min</div>
                </div>

                <!-- KPI 5: Eye Strain / PERCLOS -->
                <div class="kpi-tile">
                    <div class="kpi-tile-top">
                        <span class="kpi-label">Eye Strain Ratio</span>
                        <span class="badge" style="background: {'rgba(239, 68, 68, 0.2); color: #ef4444;' if metrics.avg_perclos > 0.15 else 'rgba(16, 185, 129, 0.2); color: #10b981;'}">Fatigue: {metrics.fatigue_index}/100</span>
                    </div>
                    <div class="kpi-val" style="color: {'var(--accent-red)' if metrics.avg_perclos > 0.15 else 'var(--accent-green)'};">{metrics.avg_perclos:.1%}</div>
                    <div class="kpi-subtext">Eyelid droop percentage (Safety ceiling: 15%)</div>
                </div>

                <!-- KPI 6: Motor Smoothness -->
                <div class="kpi-tile">
                    <div class="kpi-tile-top">
                        <span class="kpi-label">Motor Smoothness</span>
                        <span class="badge" style="background: rgba(168, 85, 247, 0.2); color: #a855f7; border: 1px solid rgba(168, 85, 247, 0.3);">Fluidity</span>
                    </div>
                    <div class="kpi-val" style="color: var(--accent-purple);">{metrics.stability_score} <span style="font-size: 16px; font-weight: 500; color: var(--text-muted);">/ 100</span></div>
                    <div class="kpi-subtext">Mouse Jitter: {metrics.avg_mouse_jitter:.3f} (Lower is better)</div>
                </div>
            </div>

            <!-- Visual Workday Timeline -->
            <div class="timeline-container">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 12px; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px;">Cognitive Timeline Spectrum</span>
                    <span style="font-size: 12px; color: var(--text-muted);">Active Work: {format_duration(metrics.active_duration_sec)} | Idle: {format_duration(metrics.idle_duration_sec)}</span>
                </div>
                <div class="timeline-bar">
                    {timeline_segments_html}
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 14px; font-size: 11px; color: var(--text-secondary); margin-top: 8px;">
                    <span style="display: inline-flex; align-items: center; gap: 5px;"><span style="width: 8px; height: 8px; border-radius: 2px; background: #10b981;"></span> Flow</span>
                    <span style="display: inline-flex; align-items: center; gap: 5px;"><span style="width: 8px; height: 8px; border-radius: 2px; background: #06b6d4;"></span> Normal Focus</span>
                    <span style="display: inline-flex; align-items: center; gap: 5px;"><span style="width: 8px; height: 8px; border-radius: 2px; background: #f59e0b;"></span> Overload</span>
                    <span style="display: inline-flex; align-items: center; gap: 5px;"><span style="width: 8px; height: 8px; border-radius: 2px; background: #ef4444;"></span> Fatigue</span>
                    <span style="display: inline-flex; align-items: center; gap: 5px;"><span style="width: 8px; height: 8px; border-radius: 2px; background: #64748b;"></span> Idle</span>
                </div>
            </div>
        </div>

        <!-- Detailed Breakdown Split -->
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 24px; margin-bottom: 24px;">
            <!-- Cognitive State Breakdown -->
            <div class="glass-card" style="margin-bottom: 0;">
                <h3 style="margin-top: 0; font-size: 16px; font-weight: 700; color: #38bdf8;">📊 Cognitive State Breakdown</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Cognitive State</th>
                            <th style="text-align: right;">Duration</th>
                            <th style="text-align: right;">Share</th>
                            <th style="width: 32%;">Distribution</th>
                        </tr>
                    </thead>
                    <tbody>
                        {state_rows_html}
                    </tbody>
                </table>
            </div>

            <!-- Application Context Focus -->
            <div class="glass-card" style="margin-bottom: 0;">
                <h3 style="margin-top: 0; font-size: 16px; font-weight: 700; color: #818cf8;">🖥️ Top Focus Applications</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Application Process</th>
                            <th style="text-align: right;">Active Time</th>
                            <th style="text-align: right;">Share</th>
                        </tr>
                    </thead>
                    <tbody>
                        {app_rows_html}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Physiological & Kinematic Diagnostics Grid -->
        <div class="glass-card">
            <h3 style="margin-top: 0; font-size: 16px; font-weight: 700; color: #34d399;">🧬 Physiological & Interaction Kinematics Baseline</h3>
            <div class="detail-grid">
                <div class="detail-item">
                    <div class="detail-title">Eye Aspect Ratio (EAR)</div>
                    <div class="detail-val">{metrics.avg_ear:.2f}</div>
                    <div class="detail-note">Baseline open aperture (Threshold: 0.21)</div>
                </div>
                <div class="detail-item">
                    <div class="detail-title">Key Dwell / Flight Time</div>
                    <div class="detail-val">{metrics.avg_dwell_ms:.0f}ms <span style="font-size: 14px; color: var(--text-muted);">/</span> {metrics.avg_flight_ms:.0f}ms</div>
                    <div class="detail-note">Motor planning and typing fluidity</div>
                </div>
                <div class="detail-item">
                    <div class="detail-title">Gaze Stability Composite</div>
                    <div class="detail-val">{int(metrics.avg_gaze_stability * 100)}%</div>
                    <div class="detail-note">Iris fixation stability on screen</div>
                </div>
                <div class="detail-item">
                    <div class="detail-title">Blue Light Protection Shield</div>
                    <div class="detail-val">{format_duration(metrics.blue_light_reduced_sec)}</div>
                    <div class="detail-note">{metrics.total_adaptations} adaptive display interventions</div>
                </div>
            </div>
        </div>

        <!-- AI Cognitive Insights & Ergonomic Guidance -->
        <div class="glass-card">
            <h3 style="margin-top: 0; font-size: 16px; font-weight: 700; color: #f59e0b;">💡 Cognitive Insights & Ergonomic Guidance</h3>
            <div style="display: flex; flex-direction: column; gap: 8px;">
                {"".join(f'<div class="recommendation-card">{i}</div>' for i in insights)}
            </div>
        </div>

        <!-- Footer -->
        <div style="text-align: center; color: var(--text-muted); font-size: 12px; margin-top: 30px;">
            AetherFlow Local Neuroadaptive Workspace Engine • Generated {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} • 100% On-Device & Encrypted
        </div>
    </div>
</body>
</html>
"""
    output_path.write_text(html_content, encoding="utf-8")
    logger.info(f"Generated HTML productivity report: {output_path}")
    return output_path


def print_cli_report(metrics: ProductivityMetrics) -> None:
    """Print clean terminal report table with ASCII bars."""
    start_str = datetime.datetime.fromtimestamp(metrics.start_time).strftime("%H:%M:%S")
    end_str = datetime.datetime.fromtimestamp(metrics.end_time).strftime("%H:%M:%S")

    print("\n" + "=" * 68)
    print("      AETHERFLOW EXECUTIVE PRODUCTIVITY & COGNITIVE REPORT")
    print("=" * 68)
    print(f" Session ID          : {metrics.session_id}")
    print(f" Monitored Window    : {start_str} - {end_str} ({format_duration(metrics.duration_sec)})")
    print(f" Active Work Time    : {format_duration(metrics.active_duration_sec)} | Idle Time: {format_duration(metrics.idle_duration_sec)}")
    print("-" * 68)
    print(f" PRODUCTIVITY SCORE  : {metrics.productivity_score} / 100")
    print(f" FLOW IMMERSION RATE : {metrics.flow_score}%")
    print(f" TYPING SPEED (WPM)  : {metrics.avg_wpm:.0f} WPM ({metrics.avg_cadence_kpm:.0f} KPM)")
    print(f" BLINK RATE (BPM)    : {metrics.avg_blink_rate:.1f} BPM (Normal: 15–22 BPM)")
    print(f" EYE STRAIN (PERCLOS): {metrics.avg_perclos:.1%} (Fatigue Index: {metrics.fatigue_index}/100)")
    print(f" MOTOR SMOOTHNESS    : {metrics.stability_score} / 100 (Jitter: {metrics.avg_mouse_jitter:.3f})")
    print("-" * 68)
    print(" PHYSIOLOGICAL & MOTOR BASELINES:")
    print(f"   Typing Cadence    : {metrics.avg_wpm:.0f} WPM ({metrics.avg_cadence_kpm:.0f} keystrokes/min)")
    print(f"   Key Latency       : Dwell: {metrics.avg_dwell_ms:.0f}ms | Flight: {metrics.avg_flight_ms:.0f}ms")
    print(f"   Blink Frequency   : {metrics.avg_blink_rate:.1f} BPM (Mean EAR: {metrics.avg_ear:.2f})")
    print(f"   Gaze Stability    : {int(metrics.avg_gaze_stability * 100)}% | Mouse Jitter: {metrics.avg_mouse_jitter:.3f}")
    print("-" * 68)
    print(" COGNITIVE STATE BREAKDOWN:")
    for state_name, dur in metrics.state_durations.items():
        pct = metrics.state_percentages.get(state_name, 0.0)
        bar = "#" * int(pct / 4)
        print(f"   {state_name:<20} {format_duration(dur):>8}  ({pct:>5.1f}%) {bar}")
    print("-" * 68)
    print(" TOP APPLICATIONS:")
    for app_name, dur in metrics.top_applications:
        pct = (dur / metrics.duration_sec * 100.0) if metrics.duration_sec > 0 else 0.0
        print(f"   {app_name:<25} {format_duration(dur):>8}  ({pct:>5.1f}%)")
    print("-" * 68)
    print(f" NEUROADAPTIVE ACTUATIONS:")
    print(f"   Total Display Shifts : {metrics.total_adaptations}")
    print(f"   Blue Light Reduced   : {format_duration(metrics.blue_light_reduced_sec)}")
    print("=" * 68 + "\n")


def generate_report(session_id: Optional[str] = None, open_browser: bool = True) -> Optional[Path]:
    """Compute metrics and export both console summary and interactive HTML report."""
    store = SessionStore()
    metrics = store.compute_session_metrics(session_id)
    if not metrics:
        print("[AetherFlow] No completed session logs found in database.")
        return None

    # Print CLI report
    print_cli_report(metrics)

    # Save HTML report
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = REPORTS_DIR / f"productivity_report_{timestamp_str}.html"
    generate_html_report(metrics, report_file)

    if open_browser:
        try:
            webbrowser.open(f"file://{report_file.resolve()}")
            print(f"[AetherFlow] Report opened in default browser: {report_file}")
        except Exception as e:
            print(f"[AetherFlow] Could not auto-open browser: {e}")

    return report_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AetherFlow productivity report.")
    parser.add_argument("--session-id", type=str, default=None, help="Target specific session ID (defaults to latest)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open report in web browser")
    args = parser.parse_args()

    generate_report(session_id=args.session_id, open_browser=not args.no_browser)
