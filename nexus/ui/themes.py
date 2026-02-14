from __future__ import annotations

from typing import Any

NEXUS_THEME = {
    "primary": "#00d4ff",
    "secondary": "#7c3aed",
    "accent": "#10b981",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "success": "#22c55e",
    "muted": "#6b7280",
    "background": "#0f172a",
    "surface": "#1e293b",
    "text": "#f8fafc",
    "border": "#334155",
}

STATUS_COLORS: dict[str, str] = {
    "pending": "yellow",
    "running": "cyan",
    "completed": "green",
    "failed": "red",
    "idle": "dim white",
    "active": "bright_cyan",
    "error": "bright_red",
    "warning": "bright_yellow",
}

AGENT_COLORS: dict[str, str] = {
    "market_data": "bright_blue",
    "technical": "cyan",
    "fundamental": "bright_green",
    "quantitative": "bright_magenta",
    "sentiment": "bright_yellow",
    "macro": "blue",
    "event": "yellow",
    "rl_agent": "magenta",
    "bull": "bright_green",
    "bear": "bright_red",
    "risk": "bright_yellow",
    "portfolio": "bright_cyan",
    "execution": "bright_white",
    "coordinator": "bright_magenta",
}

TEAM_LABELS: dict[str, list[str]] = {
    "Analyst": ["market_data", "sentiment", "fundamental"],
    "Research": ["bull", "bear", "coordinator"],
    "Quant": ["technical", "quantitative", "macro"],
    "Strategy": ["event", "rl_agent", "risk"],
    "Execution": ["portfolio", "execution"],
}

SIGNAL_COLORS: dict[str, str] = {
    "buy": "bright_green",
    "strong_buy": "green",
    "sell": "bright_red",
    "strong_sell": "red",
    "hold": "bright_yellow",
}

MESSAGE_TYPE_STYLES: dict[str, tuple[str, str]] = {
    "tool": ("bright_blue", "Tool"),
    "reason": ("bright_cyan", "Reason"),
    "signal": ("bright_green", "Signal"),
    "error": ("bright_red", "Error"),
    "info": ("white", "Info"),
    "cost": ("bright_yellow", "Cost"),
    "risk": ("bright_magenta", "Risk"),
}


NEXUS_BANNER = """
███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗
████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝
██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗
██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║
██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║
╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
""".strip()


def get_status_style(status: str) -> str:
    return STATUS_COLORS.get(status.lower(), "white")


def get_agent_style(agent_name: str) -> str:
    return AGENT_COLORS.get(agent_name, "white")


def get_signal_style(signal: str) -> str:
    return SIGNAL_COLORS.get(signal.lower(), "white")


def format_currency(value: float, prefix: str = "$") -> str:
    if abs(value) >= 1_000_000:
        return f"{prefix}{value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"{prefix}{value / 1_000:,.2f}K"
    return f"{prefix}{value:,.2f}"


def format_pct(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def format_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
