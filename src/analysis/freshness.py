"""
Data Freshness Registry.

Tracks when each data source was last successfully fetched and whether it
is live, recent, or stale. Every chart calls add_freshness_label() to stamp
its bottom-right corner with a source / latency badge.

Usage:
    from src.analysis.freshness import record_fetch, add_freshness_label, get_status

    # After a successful data fetch:
    record_fetch("yfinance_prices")

    # On a Plotly figure:
    fig = add_freshness_label(fig, "yfinance_prices")
"""

from __future__ import annotations

import datetime
from typing import Optional

# ── Thresholds ────────────────────────────────────────────────────────────────

_THRESHOLDS: dict[str, dict] = {
    "yfinance_prices":    {"warn_h": 4,    "stale_h": 26},
    "yfinance_vix":       {"warn_h": 4,    "stale_h": 26},
    "fred_macro":         {"warn_h": 48,   "stale_h": 168},   # 2d warn, 7d stale
    "rss_headlines":      {"warn_h": 1,    "stale_h": 4},
    "conflict_manual":    {"warn_h": 168,  "stale_h": 720},   # 7d warn, 30d stale
    "cot_positioning":    {"warn_h": 72,   "stale_h": 240},
    "fred_spreads":       {"warn_h": 4,    "stale_h": 26},   # yfinance proxy — same cadence as prices
    "risk_score":         {"warn_h": 4,    "stale_h": 26},
    "conflict_model":     {"warn_h": 24,   "stale_h": 72},
    # Live API sources added 2026-04-19
    "portwatch":          {"warn_h": 6,    "stale_h": 30},    # IMF ArcGIS — daily cadence
    "gdelt":              {"warn_h": 4,    "stale_h": 12},    # 3h cache, media volume
    "acled":              {"warn_h": 8,    "stale_h": 48},    # 6h cache, conflict events
    "eia_inventory":      {"warn_h": 24,   "stale_h": 168},   # weekly Wednesday update
}

_SOURCE_LABELS: dict[str, str] = {
    "yfinance_prices":   "YF",
    "yfinance_vix":      "YF/VIX",
    "fred_macro":        "FRED",
    "rss_headlines":     "RSS",
    "conflict_manual":   "Manual",
    "cot_positioning":   "CFTC/COT",
    "fred_spreads":      "YF/HYG·LQD",
    "risk_score":        "Computed",
    "conflict_model":    "Model",
    "portwatch":         "PortWatch",
    "gdelt":             "GDELT",
    "acled":             "ACLED",
    "eia_inventory":     "EIA",
}

# ── Internal store (module-level dict, survives within a Streamlit process) ──
_FETCH_TIMES:   dict[str, datetime.datetime] = {}
_FAIL_REGISTRY: dict[str, dict] = {}   # source → {ts, message, count}

# Critical sources — if any fail, show a visible warning banner
_CRITICAL_SOURCES = {"yfinance_prices", "yfinance_vix"}


def record_fetch(source: str, ts: Optional[datetime.datetime] = None) -> None:
    """Record a successful fetch for a named source. Also clears any failure record."""
    _FETCH_TIMES[source] = ts or datetime.datetime.now()
    _FAIL_REGISTRY.pop(source, None)  # clear failure on successful fetch


def record_failure(source: str, message: str = "") -> None:
    """
    Record a data fetch failure for a named source.

    Call this in except blocks where data is absent or fetch failed.
    Replaces the silent `except: pass` pattern — failures become visible
    on the dashboard via get_failures() / data_health_html().
    """
    existing = _FAIL_REGISTRY.get(source, {})
    _FAIL_REGISTRY[source] = {
        "ts":      datetime.datetime.now(),
        "message": message or f"{source} unavailable",
        "count":   existing.get("count", 0) + 1,
    }


def get_failures() -> dict[str, dict]:
    """Return all currently tracked data failures."""
    return dict(_FAIL_REGISTRY)


def clear_failure(source: str) -> None:
    """Manually clear a failure (e.g., after manual retry)."""
    _FAIL_REGISTRY.pop(source, None)


def data_health_html() -> str:
    """
    Return an HTML warning block if any critical data sources are failing.
    Returns empty string if all critical sources are healthy.
    """
    critical_failures = {s: v for s, v in _FAIL_REGISTRY.items() if s in _CRITICAL_SOURCES}
    all_failures      = _FAIL_REGISTRY

    if not all_failures:
        return ""

    lines = []
    for src, info in all_failures.items():
        severity = "critical" if src in _CRITICAL_SOURCES else "warn"
        color    = "#e74c3c" if severity == "critical" else "#e67e22"
        label    = _SOURCE_LABELS.get(src, src)
        msg      = info["message"][:80]
        cnt      = info["count"]
        lines.append(
            f'<span style="color:{color};font-size:10px;font-family:\'JetBrains Mono\',monospace;">'
            f'✗ {label}: {msg} (×{cnt})</span>'
        )

    severity_color = "#e74c3c" if critical_failures else "#e67e22"
    severity_label = "DATA UNAVAILABLE" if critical_failures else "DATA DEGRADED"
    items_html = "<br>".join(lines)
    return (
        f'<div style="background:#0d0505;border-left:3px solid {severity_color};'
        f'border-radius:4px;padding:10px 14px;margin:8px 0;">'
        f'<span style="color:{severity_color};font-family:\'JetBrains Mono\',monospace;'
        f'font-size:11px;font-weight:700;letter-spacing:.08em;">⚠ {severity_label}</span><br>'
        f'{items_html}</div>'
    )


def get_status(source: str) -> dict:
    """
    Returns dict with:
      status: "live" | "recent" | "stale" | "unknown"
      label:  human-readable stamp
      color:  hex color
      hours_ago: float or None
    """
    ts = _FETCH_TIMES.get(source)
    if ts is None:
        return {
            "status":    "unknown",
            "label":     f"No data · {_SOURCE_LABELS.get(source, source)}",
            "color":     "#555960",
            "hours_ago": None,
        }

    thresholds = _THRESHOLDS.get(source, {"warn_h": 4, "stale_h": 24})
    delta_h    = (datetime.datetime.now() - ts).total_seconds() / 3600

    src_label = _SOURCE_LABELS.get(source, source)

    if delta_h < 0.5:
        return {"status": "live",   "label": f"Live · {src_label}",
                "color": "#27ae60", "hours_ago": delta_h}
    elif delta_h < thresholds["warn_h"]:
        mins = int(delta_h * 60) if delta_h < 1 else None
        ago  = f"{mins}m" if mins else f"{int(delta_h)}h"
        return {"status": "recent", "label": f"{src_label} · {ago} ago",
                "color": "#CFB991", "hours_ago": delta_h}
    elif delta_h < thresholds["stale_h"]:
        return {"status": "warn",   "label": f"{src_label} · {int(delta_h)}h ago",
                "color": "#e67e22", "hours_ago": delta_h}
    else:
        return {"status": "stale",  "label": f"STALE · {src_label} · {int(delta_h)}h",
                "color": "#c0392b", "hours_ago": delta_h}


def freshness_badge_html(source: str, extra_style: str = "") -> str:
    """Return an inline HTML span with the freshness badge for st.markdown()."""
    info = get_status(source)
    return (
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        f'color:{info["color"]};letter-spacing:0.06em;{extra_style}">'
        f'{info["label"]}</span>'
    )


def add_freshness_label(fig, source: str, y_offset: float = -0.07):
    """
    Stamp a Plotly figure's bottom-right corner with a freshness badge annotation.
    Returns the figure (in-place mutation, but also returned for chaining).
    """
    info = get_status(source)
    fig.add_annotation(
        xref="paper", yref="paper",
        x=1.0, y=y_offset,
        text=info["label"],
        showarrow=False,
        font=dict(size=7.5, family="JetBrains Mono, monospace", color=info["color"]),
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(0,0,0,0)",
    )
    return fig


def all_statuses() -> dict[str, dict]:
    """Return freshness status for all registered sources."""
    return {src: get_status(src) for src in _THRESHOLDS}


def any_stale(sources: Optional[list[str]] = None) -> bool:
    """Return True if any of the given sources (or all) are stale."""
    check = sources or list(_THRESHOLDS.keys())
    return any(get_status(s)["status"] in ("stale", "unknown") for s in check)
