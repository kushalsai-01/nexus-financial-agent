from __future__ import annotations

import pytest

from nexus.ui.themes import (
    NEXUS_THEME,
    STATUS_COLORS,
    AGENT_COLORS,
    TEAM_LABELS,
    NEXUS_BANNER,
    get_status_style,
    get_agent_style,
    format_currency,
    format_pct,
    format_duration,
)


class TestThemes:
    def test_nexus_theme_keys(self) -> None:
        assert "primary" in NEXUS_THEME
        assert "background" in NEXUS_THEME
        assert "success" in NEXUS_THEME
        assert "error" in NEXUS_THEME

    def test_status_colors(self) -> None:
        assert "pending" in STATUS_COLORS
        assert "running" in STATUS_COLORS
        assert "completed" in STATUS_COLORS
        assert "failed" in STATUS_COLORS

    def test_agent_colors_all_agents(self) -> None:
        all_agents = set()
        for agents in TEAM_LABELS.values():
            all_agents.update(agents)
        for agent in all_agents:
            assert agent in AGENT_COLORS, f"Missing color for agent: {agent}"

    def test_team_labels(self) -> None:
        assert len(TEAM_LABELS) == 5

    def test_banner_not_empty(self) -> None:
        assert len(NEXUS_BANNER) > 0

    def test_get_status_style(self) -> None:
        style = get_status_style("running")
        assert "cyan" in style.lower() or len(style) > 0

    def test_get_agent_style(self) -> None:
        style = get_agent_style("technical")
        assert len(style) > 0

    def test_format_currency(self) -> None:
        assert format_currency(1_500_000) == "$1.50M"
        assert format_currency(50_000) == "$50.00K"
        assert format_currency(500) == "$500.00"

    def test_format_pct(self) -> None:
        result = format_pct(0.1234)
        assert result == "+0.12%"

    def test_format_duration(self) -> None:
        assert format_duration(90) == "00:01:30"
        assert format_duration(5) == "00:00:05"
