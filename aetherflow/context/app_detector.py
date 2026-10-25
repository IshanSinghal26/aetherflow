"""Foreground application detection and creative context domain classification.

Passively identifies the currently active window and categorizes it into
creative domains (CODE_IDE, CREATIVE_DESIGN, MEDIA_VIDEO, WRITING_TEXT, etc.)
using native OS APIs with cross-platform defensive fallbacks.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import psutil

from aetherflow.config import ContextDomain

logger = logging.getLogger(__name__)


@dataclass
class ActiveAppContext:
    """Snapshot of active user focus context."""
    process_name: str
    window_title: str
    pid: int
    domain: ContextDomain
    timestamp: float


class AppDetector:
    """Detects active foreground application and classifies creative context domain."""

    # Process names -> ContextDomain mapping
    PROCESS_DOMAIN_MAP = {
        # CODE_IDE
        "code.exe": ContextDomain.CODE_IDE,
        "cursor.exe": ContextDomain.CODE_IDE,
        "windsurf.exe": ContextDomain.CODE_IDE,
        "pycharm64.exe": ContextDomain.CODE_IDE,
        "pycharm.exe": ContextDomain.CODE_IDE,
        "idea64.exe": ContextDomain.CODE_IDE,
        "idea.exe": ContextDomain.CODE_IDE,
        "sublime_text.exe": ContextDomain.CODE_IDE,
        "devenv.exe": ContextDomain.CODE_IDE,
        "windowsterminal.exe": ContextDomain.CODE_IDE,
        "powershell.exe": ContextDomain.CODE_IDE,
        "cmd.exe": ContextDomain.CODE_IDE,
        "alacritty.exe": ContextDomain.CODE_IDE,
        "wezterm-gui.exe": ContextDomain.CODE_IDE,
        "nvim-qt.exe": ContextDomain.CODE_IDE,
        "eclipse.exe": ContextDomain.CODE_IDE,
        "code": ContextDomain.CODE_IDE,
        "sublime_text": ContextDomain.CODE_IDE,

        # CREATIVE_DESIGN
        "photoshop.exe": ContextDomain.CREATIVE_DESIGN,
        "illustrator.exe": ContextDomain.CREATIVE_DESIGN,
        "figma.exe": ContextDomain.CREATIVE_DESIGN,
        "blender.exe": ContextDomain.CREATIVE_DESIGN,
        "canva.exe": ContextDomain.CREATIVE_DESIGN,
        "premiere.exe": ContextDomain.CREATIVE_DESIGN,
        "afterfx.exe": ContextDomain.CREATIVE_DESIGN,
        "indesign.exe": ContextDomain.CREATIVE_DESIGN,
        "gimp.exe": ContextDomain.CREATIVE_DESIGN,
        "inkscape.exe": ContextDomain.CREATIVE_DESIGN,
        "resolve.exe": ContextDomain.CREATIVE_DESIGN,

        # MEDIA_VIDEO
        "vlc.exe": ContextDomain.MEDIA_VIDEO,
        "mpv.exe": ContextDomain.MEDIA_VIDEO,
        "wmplayer.exe": ContextDomain.MEDIA_VIDEO,
        "spotify.exe": ContextDomain.MEDIA_VIDEO,
        "netflix.exe": ContextDomain.MEDIA_VIDEO,

        # WRITING_TEXT
        "winword.exe": ContextDomain.WRITING_TEXT,
        "wordpad.exe": ContextDomain.WRITING_TEXT,
        "obsidian.exe": ContextDomain.WRITING_TEXT,
        "notion.exe": ContextDomain.WRITING_TEXT,
        "slack.exe": ContextDomain.WRITING_TEXT,
        "teams.exe": ContextDomain.WRITING_TEXT,
        "discord.exe": ContextDomain.WRITING_TEXT,
        "onenote.exe": ContextDomain.WRITING_TEXT,
        "typora.exe": ContextDomain.WRITING_TEXT,

        # SYSTEM_UTILITY
        "explorer.exe": ContextDomain.SYSTEM_UTILITY,
        "taskmgr.exe": ContextDomain.SYSTEM_UTILITY,
        "systemsettings.exe": ContextDomain.SYSTEM_UTILITY,
    }

    BROWSER_PROCESSES = {
        "chrome.exe", "firefox.exe", "msedge.exe", "brave.exe", "opera.exe", "vivaldi.exe",
        "chrome", "firefox", "msedge", "brave", "chromium"
    }

    # Keyword rules for browser tab titles
    MEDIA_TITLE_KEYWORDS = [
        "youtube", "netflix", "twitch", "vimeo", "disney+", "prime video", "coursera", "udemy"
    ]
    WRITING_TITLE_KEYWORDS = [
        "google docs", "google keep", "notion", "obsidian", "overleaf", "medium", "substack"
    ]
    DEV_TITLE_KEYWORDS = [
        "github", "gitlab", "stack overflow", "chatgpt", "claude", "docs.python.org", "w3schools"
    ]

    def __init__(self, mock_domain: Optional[ContextDomain] = None) -> None:
        self.mock_domain = mock_domain
        self._last_context = ActiveAppContext(
            process_name="unknown",
            window_title="unknown",
            pid=0,
            domain=ContextDomain.GENERAL,
            timestamp=time.time()
        )

    def get_active_app(self) -> ActiveAppContext:
        """Query native OS for foreground window and classify context domain."""
        t_now = time.time()
        
        if self.mock_domain is not None:
            return ActiveAppContext(
                process_name="mock_process",
                window_title="Mock Window",
                pid=1234,
                domain=self.mock_domain,
                timestamp=t_now
            )

        if sys.platform == "win32":
            proc_name, title, pid = self._get_win32_foreground()
        elif sys.platform.startswith("linux"):
            proc_name, title, pid = self._get_linux_foreground()
        elif sys.platform == "darwin":
            proc_name, title, pid = self._get_macos_foreground()
        else:
            proc_name, title, pid = "unknown", "unknown", 0

        # Classify domain
        domain = self._classify_domain(proc_name, title)

        context = ActiveAppContext(
            process_name=proc_name,
            window_title=title,
            pid=pid,
            domain=domain,
            timestamp=t_now
        )
        self._last_context = context
        return context

    def _get_win32_foreground(self) -> tuple[str, str, int]:
        """Query foreground window details via Windows Win32 / ctypes."""
        try:
            from ctypes import wintypes
            user32 = ctypes.windll.user32

            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return "desktop", "Desktop", 0

            # Window title
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buffer = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buffer, length + 1)
                title = buffer.value
            else:
                title = ""

            # Process PID
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            process_pid = pid.value

            if process_pid <= 4:
                return "system", title or "System", process_pid

            # Process name
            try:
                proc = psutil.Process(process_pid)
                proc_name = proc.name().lower()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                proc_name = "unknown"

            return proc_name, title, process_pid
        except Exception as e:
            logger.debug(f"Win32 foreground detection failed: {e}")
            return "unknown", "unknown", 0

    def _get_linux_foreground(self) -> tuple[str, str, int]:
        """Fallback detection on Linux using xdotool or procfs."""
        try:
            import subprocess
            cmd_window = ["xdotool", "getwindowfocus", "getwindowname"]
            title = subprocess.check_output(cmd_window, timeout=0.3).decode("utf-8", errors="ignore").strip()
            
            cmd_pid = ["xdotool", "getwindowfocus", "getwindowpid"]
            pid_str = subprocess.check_output(cmd_pid, timeout=0.3).decode("utf-8", errors="ignore").strip()
            pid = int(pid_str) if pid_str.isdigit() else 0
            
            proc_name = psutil.Process(pid).name().lower() if pid > 0 else "unknown"
            return proc_name, title, pid
        except Exception:
            return "unknown", "unknown", 0

    def _get_macos_foreground(self) -> tuple[str, str, int]:
        """Fallback detection on macOS using osascript."""
        try:
            import subprocess
            ascript = 'tell application "System Events" to get name of first process whose frontmost is true'
            proc_name = subprocess.check_output(["osascript", "-e", ascript], timeout=0.5).decode("utf-8").strip().lower()
            return proc_name, proc_name, 0
        except Exception:
            return "unknown", "unknown", 0

    def _classify_domain(self, proc_name: str, title: str) -> ContextDomain:
        """Classify (process_name, window_title) into a ContextDomain."""
        proc = proc_name.lower()
        title_lower = title.lower()

        # 1. Exact executable match
        if proc in self.PROCESS_DOMAIN_MAP:
            return self.PROCESS_DOMAIN_MAP[proc]

        # 2. Browser tab heuristic classification
        if proc in self.BROWSER_PROCESSES:
            for kw in self.MEDIA_TITLE_KEYWORDS:
                if kw in title_lower:
                    return ContextDomain.MEDIA_VIDEO
            for kw in self.WRITING_TITLE_KEYWORDS:
                if kw in title_lower:
                    return ContextDomain.WRITING_TEXT
            for kw in self.DEV_TITLE_KEYWORDS:
                if kw in title_lower:
                    return ContextDomain.CODE_IDE
            return ContextDomain.RESEARCH_BROWSING

        # 3. Window title substring heuristics
        if any(ext in title_lower for ext in [".py", ".ts", ".js", ".rs", ".go", ".cpp", "visual studio code", "intellij"]):
            return ContextDomain.CODE_IDE
        if any(term in title_lower for term in ["figma", "photoshop", "blender", "illustrator", "canvas"]):
            return ContextDomain.CREATIVE_DESIGN
        if any(term in title_lower for term in ["word", "notion", "obsidian", "document"]):
            return ContextDomain.WRITING_TEXT

        return ContextDomain.GENERAL
