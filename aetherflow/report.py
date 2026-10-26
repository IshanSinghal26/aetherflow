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

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AetherFlow Productivity & Cognitive Health Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 30px 20px;
        }}
        .container {{
            max-width: 1080px;
            margin: 0 auto;
        }}
        .card {{
            background-color: #1e293b;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid #334155;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        }}
        .header-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #334155;
            padding-bottom: 18px;
            margin-bottom: 24px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background-color: #0f172a;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 18px;
            text-align: center;
        }}
        .kpi-val {{
            font-size: 32px;
            font-weight: bold;
            margin: 8px 0;
        }}
        .kpi-sub {{
            font-size: 12px;
            color: #94a3b8;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th {{
            text-align: left;
            padding: 10px 14px;
            background-color: #0f172a;
            color: #94a3b8;
            font-weight: 600;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="card">
            <div class="header-bar">
                <div>
                    <h1 style="margin: 0; font-size: 26px; color: #38bdf8;">⚡ AETHERFLOW PRODUCTIVITY & HEALTH REPORT</h1>
                    <div style="color: #94a3b8; font-size: 13px; margin-top: 6px;">
                        Session ID: <code style="color: #cbd5e1;">{metrics.session_id}</code> | Monitored Period: <strong>{start_dt}</strong> to <strong>{end_dt}</strong>
                    </div>
                </div>
                <div>
                    <span class="badge" style="background-color: #0369a1; color: #e0f2fe; font-size: 13px;">100% Local Inference</span>
                </div>
            </div>

            <!-- 6 Core Executive KPIs including WPM and BPM -->
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-sub">PRODUCTIVITY SCORE</div>
                    <div class="kpi-val" style="color: #10b981;">{metrics.productivity_score} / 100</div>
                    <div class="kpi-sub">Weighted Focus Efficiency</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-sub">FLOW IMMERSION RATIO</div>
                    <div class="kpi-val" style="color: #38bdf8;">{metrics.flow_score}%</div>
                    <div class="kpi-sub">{format_duration(metrics.state_durations.get(CognitiveState.FLOW.value, 0.0))} in Deep Flow</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-sub">TYPING SPEED (WPM)</div>
                    <div class="kpi-val" style="color: #f59e0b;">{metrics.avg_wpm:.0f} WPM</div>
                    <div class="kpi-sub">{metrics.avg_cadence_kpm:.0f} Keystrokes/Min (KPM)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-sub">BLINK RATE (BPM)</div>
                    <div class="kpi-val" style="color: {'#ef4444' if metrics.avg_blink_rate < 10 or metrics.avg_blink_rate > 28 else '#06b6d4'};">{metrics.avg_blink_rate:.1f} BPM</div>
                    <div class="kpi-sub">Normal: 15–22 Blinks/Min</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-sub">EYE STRAIN / PERCLOS</div>
                    <div class="kpi-val" style="color: {'#ef4444' if metrics.avg_perclos > 0.15 else '#10b981'};">{metrics.avg_perclos:.1%}</div>
                    <div class="kpi-sub">Fatigue Index: {metrics.fatigue_index}/100</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-sub">MOTOR SMOOTHNESS</div>
                    <div class="kpi-val" style="color: #a855f7;">{metrics.stability_score} / 100</div>
                    <div class="kpi-sub">Jitter: {metrics.avg_mouse_jitter:.3f} (Lower is better)</div>
                </div>
            </div>

            <div style="display: flex; gap: 20px; font-size: 13px; color: #94a3b8; border-top: 1px solid #334155; padding-top: 14px;">
                <div>Total Duration: <strong style="color: #f8fafc;">{format_duration(metrics.duration_sec)}</strong></div>
                <div>Active Work Time: <strong style="color: #f8fafc;">{format_duration(metrics.active_duration_sec)}</strong></div>
                <div>Idle / Away Time: <strong style="color: #f8fafc;">{format_duration(metrics.idle_duration_sec)}</strong></div>
                <div>Typing Output: <strong style="color: #f59e0b;">{metrics.avg_wpm:.0f} WPM</strong></div>
                <div>Ocular Rate: <strong style="color: #06b6d4;">{metrics.avg_blink_rate:.1f} BPM</strong></div>
                <div>Adaptations: <strong style="color: #38bdf8;">{metrics.total_adaptations}</strong></div>
            </div>
        </div>

        <!-- Cognitive State Distribution & Context Split -->
        <div style="display: grid; grid-template-columns: 3fr 2fr; gap: 24px; margin-bottom: 24px;">
            <div class="card" style="margin-bottom: 0;">
                <h3 style="margin-top: 0; color: #f8fafc; font-size: 16px;">Cognitive State Distribution</h3>
                <table>
                    <thead>
                        <tr>
                            <th>State</th>
                            <th style="text-align: right;">Time</th>
                            <th style="text-align: right;">Share</th>
                            <th style="width: 35%;">Timeline Proportion</th>
                        </tr>
                    </thead>
                    <tbody>
                        {state_rows_html}
                    </tbody>
                </table>
            </div>

            <div class="card" style="margin-bottom: 0;">
                <h3 style="margin-top: 0; color: #f8fafc; font-size: 16px;">Top Focus Applications</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Application</th>
                            <th style="text-align: right;">Duration</th>
                            <th style="text-align: right;">Share</th>
                        </tr>
                    </thead>
                    <tbody>
                        {app_rows_html}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Biometric & Peripheral Health Summary -->
        <div class="card">
            <h3 style="margin-top: 0; color: #f8fafc; font-size: 16px;">Physiological & Kinematic Baseline Summary</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;">
                <div style="background: #0f172a; padding: 14px; border-radius: 8px; border: 1px solid #334155;">
                    <div style="font-size: 12px; color: #94a3b8;">Average Blink Rate</div>
                    <div style="font-size: 20px; font-weight: bold; margin-top: 4px;">{metrics.avg_blink_rate} BPM</div>
                    <div style="font-size: 11px; color: #64748b; margin-top: 2px;">Normal: 15–22 BPM</div>
                </div>
                <div style="background: #0f172a; padding: 14px; border-radius: 8px; border: 1px solid #334155;">
                    <div style="font-size: 12px; color: #94a3b8;">Eye Aspect Ratio (EAR)</div>
                    <div style="font-size: 20px; font-weight: bold; margin-top: 4px;">{metrics.avg_ear:.2f}</div>
                    <div style="font-size: 11px; color: #64748b; margin-top: 2px;">Blink Threshold: 0.21</div>
                </div>
                <div style="background: #0f172a; padding: 14px; border-radius: 8px; border: 1px solid #334155;">
                    <div style="font-size: 12px; color: #94a3b8;">Typing Speed (KPM / WPM)</div>
                    <div style="font-size: 20px; font-weight: bold; margin-top: 4px;">{metrics.avg_cadence_kpm:.0f} KPM ({int(metrics.avg_cadence_kpm/5.0)} WPM)</div>
                    <div style="font-size: 11px; color: #64748b; margin-top: 2px;">Active generation throughput</div>
                </div>
                <div style="background: #0f172a; padding: 14px; border-radius: 8px; border: 1px solid #334155;">
                    <div style="font-size: 12px; color: #94a3b8;">Key Dwell / Flight Latency</div>
                    <div style="font-size: 20px; font-weight: bold; margin-top: 4px;">{metrics.avg_dwell_ms:.0f}ms / {metrics.avg_flight_ms:.0f}ms</div>
                    <div style="font-size: 11px; color: #64748b; margin-top: 2px;">Motor planning fluidity</div>
                </div>
            </div>
        </div>

        <!-- AI Cognitive Insights & Ergonomic Guidance -->
        <div class="card">
            <h3 style="margin-top: 0; color: #38bdf8; font-size: 16px;">💡 Cognitive Insights & Ergonomic Recommendations</h3>
            <ul style="font-size: 14px; line-height: 1.6; color: #cbd5e1; padding-left: 20px;">
                {insights_html}
            </ul>
        </div>

        <!-- Footer -->
        <div style="text-align: center; color: #64748b; font-size: 12px; margin-top: 24px;">
            AetherFlow Local Neuroadaptive Workspace Engine • Generated {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} • 100% Private & On-Device
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
