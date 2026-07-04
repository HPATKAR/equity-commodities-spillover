"""
Timezone helpers. The terminal is anchored to America/Chicago (the app's
market/reporting timezone per project convention); server-local time is wrong
the moment this is deployed anywhere but a CT machine (e.g. Render runs UTC).

Use these instead of naive datetime.now()/date.today() anywhere a user-facing
clock, a "today" boundary, or a market-date rollover is involved.
"""

from __future__ import annotations

from datetime import date, datetime

try:
    from zoneinfo import ZoneInfo          # stdlib ≥ 3.9
    _CT = ZoneInfo("America/Chicago")
except Exception:                          # pragma: no cover - zoneinfo always present on 3.10+
    _CT = None

# Label shown next to a CT clock in the UI.
CT_LABEL = "CT"


def now_ct() -> datetime:
    """Timezone-aware current time in America/Chicago (naive local as fallback)."""
    return datetime.now(_CT) if _CT is not None else datetime.now()


def today_ct() -> date:
    """Current calendar date in America/Chicago — the correct 'today' for
    day-over-day rollovers and market-date logic."""
    return now_ct().date()
