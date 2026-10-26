"""Unit tests for Layer 4: Active Application and Creative Context Detection."""

import pytest

from aetherflow.config import ContextDomain
from aetherflow.context.app_detector import ActiveAppContext, AppDetector


class TestAppDetector:
    """Tests for process classification and window title heuristic parsing."""

    def test_mock_domain_override(self):
        detector = AppDetector(mock_domain=ContextDomain.CREATIVE_DESIGN)
        ctx = detector.get_active_app()
        assert isinstance(ctx, ActiveAppContext)
        assert ctx.domain == ContextDomain.CREATIVE_DESIGN
        assert ctx.process_name == "mock_process"

    def test_process_domain_mappings(self):
        detector = AppDetector()
        assert detector._classify_domain("code.exe", "") == ContextDomain.CODE_IDE
        assert detector._classify_domain("pycharm64.exe", "") == ContextDomain.CODE_IDE
        assert detector._classify_domain("photoshop.exe", "") == ContextDomain.CREATIVE_DESIGN
        assert detector._classify_domain("figma.exe", "") == ContextDomain.CREATIVE_DESIGN
        assert detector._classify_domain("vlc.exe", "") == ContextDomain.MEDIA_VIDEO
        assert detector._classify_domain("winword.exe", "") == ContextDomain.WRITING_TEXT
        assert detector._classify_domain("obsidian.exe", "") == ContextDomain.WRITING_TEXT
        assert detector._classify_domain("explorer.exe", "") == ContextDomain.SYSTEM_UTILITY

    def test_browser_title_heuristics(self):
        detector = AppDetector()
        
        # YouTube in Chrome
        assert detector._classify_domain("chrome.exe", "Watching Machine Learning - YouTube") == ContextDomain.MEDIA_VIDEO
        # Google Docs in Edge
        assert detector._classify_domain("msedge.exe", "Research Paper - Google Docs") == ContextDomain.WRITING_TEXT
        # GitHub in Firefox
        assert detector._classify_domain("firefox.exe", "GitHub - deepmind/aetherflow") == ContextDomain.CODE_IDE
        # Figma in Chrome/Edge/Brave
        assert detector._classify_domain("chrome.exe", "Mobile App UI – Figma - Google Chrome") == ContextDomain.CREATIVE_DESIGN
        assert detector._classify_domain("msedge.exe", "Design System – Figma - Microsoft Edge") == ContextDomain.CREATIVE_DESIGN
        # Canva in Firefox
        assert detector._classify_domain("firefox.exe", "Canva - Presentation Banner") == ContextDomain.CREATIVE_DESIGN
        # Generic browsing
        assert detector._classify_domain("chrome.exe", "Wikipedia, the free encyclopedia") == ContextDomain.RESEARCH_BROWSING

    def test_title_fallback_heuristics(self):
        detector = AppDetector()
        assert detector._classify_domain("unknown.exe", "main.py - Visual Studio Code") == ContextDomain.CODE_IDE
        assert detector._classify_domain("unknown.exe", "Figma - Workspace Design System") == ContextDomain.CREATIVE_DESIGN
        assert detector._classify_domain("unknown.exe", "Random Desktop Window") == ContextDomain.GENERAL

    def test_win32_defensive_query(self):
        # Query active app; must never throw an unhandled exception
        detector = AppDetector()
        ctx = detector.get_active_app()
        assert isinstance(ctx, ActiveAppContext)
        assert ctx.domain in ContextDomain
