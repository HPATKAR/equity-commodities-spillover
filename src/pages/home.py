"""
Home — Geopolitical & Cross-Asset Intelligence Terminal
Command Center for the Equity-Commodities Spillover Dashboard.

Page hierarchy:
  1.  Masthead         — terminal identity, date, situation state
  2.  Geo Risk Score   — dominant block: score, decomposition, drivers, freshness
  3.  Context Narrative — data-driven plain-language interpretation
  4.  Intel Panel      — conflict table (left) + transmission channels (right)
  5.  Scenario Switch  — compact, integrated lens selector
  6.  Where To Go Now  — live-data-driven recommendations
  7.  Navigate Terminal — grouped quick-jump shortcuts
  8.  Live Signals     — strait snapshot + what-changed delta strip
"""

from __future__ import annotations

import datetime
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis.agent_state import init_agents, AGENTS, pending_count, log_activity
from src.analysis.scenario_state import (
    SCENARIOS, SCENARIO_ORDER, init_scenario,
    get_scenario, get_scenario_id, set_scenario,
)
from src.analysis.conflict_model import score_all_conflicts, aggregate_portfolio_scores
from src.analysis.freshness import freshness_badge_html, record_fetch
from src.data.loader import load_returns
from src.analysis.correlations import average_cross_corr_series
from src.analysis.risk_score import risk_score_history, plot_risk_history
from src.ui.shared import _page_header, _page_footer

_F    = "font-family:'DM Sans',sans-serif;"
_M    = "font-family:'JetBrains Mono',monospace;"
_GOLD = "#CFB991"


# ─────────────────────────────────────────────────────────────────────────────
# CSS — one block, loaded once
# ─────────────────────────────────────────────────────────────────────────────

_STYLE = """<style>
/*
  Typography system — Command Center
  T1  Section header  Mono 10px 700 uppercase .18em  #DCE4F0
  T2  Panel label     Mono 10px 600 uppercase .12em  #C8D4E0
  T3  Body text       Sans 12px 400            —     #C8D4E0
  T4  Data value      Mono 12px 700            —     (state-colored)
  T5  Caption / meta  Mono 10px 400            —     #A8B8C8
*/
/* T1 — section header */
.hm-label{font-family:'JetBrains Mono',monospace!important;font-size:10px!important;
  font-weight:700!important;text-transform:uppercase;letter-spacing:.18em;
  color:#DCE4F0!important;display:block}
/* T3 — body text */
.hm-sub{font-family:'DM Sans',sans-serif!important;font-size:12px!important;
  color:#C8D4E0!important;display:block;line-height:1.6}
/* T4 — delta values */
.hm-up{font-family:'JetBrains Mono',monospace!important;font-size:12px!important;
  font-weight:700!important;color:#c0392b!important}
.hm-dn{font-family:'JetBrains Mono',monospace!important;font-size:12px!important;
  font-weight:700!important;color:#27ae60!important}
.hm-fl{font-family:'JetBrains Mono',monospace!important;font-size:12px!important;
  font-weight:700!important;color:#DCE4F0!important}
/* ── Section rule ── */
.hm-rule{border:none;border-top:1px solid #1e1e1e;margin:.3rem 0 .25rem}
/* ── Conflict table rows ── */
.hm-crow{display:flex;align-items:center;gap:8px;padding:5px 0;
  border-bottom:1px solid #111}
/* ── Nav card ── */
.hm-nav{background:#0f0f0f;border:1px solid #1e1e1e;
  padding:6px 8px;margin-bottom:3px;
  transition:box-shadow .15s ease,border-color .15s ease}
.hm-nav:hover{box-shadow:0 0 0 1px #CFB991,inset 0 0 8px rgba(207,185,145,.07);
  border-color:#CFB991!important}
.hm-nav:hover .hm-sc{border-color:#CFB991!important;color:#CFB991!important}
/* ── Use-case tags ── */
.hm-tag{display:inline-block;font-family:'JetBrains Mono',monospace!important;
  font-size:10px!important;font-weight:700!important;letter-spacing:.12em;
  text-transform:uppercase;padding:1px 4px;border-radius:1px;margin-left:3px;
  vertical-align:middle}
.hm-tag-daily{background:#0a1a2e;color:#2980b9!important}
.hm-tag-alert{background:#1a0a00;color:#e67e22!important}
.hm-tag-deep {background:#0a1a0a;color:#27ae60!important}
/* ── Shortcut hint (static badge) ── */
.hm-sc{display:inline-block;font-family:'JetBrains Mono',monospace!important;
  font-size:10px!important;font-weight:700;color:#C8D4E0!important;
  border:1px solid #2a2a2a;border-radius:2px;padding:0 3px;
  margin-left:3px;vertical-align:middle;line-height:1.6}
/* ── Nav arrow buttons inside .hm-nav cards ── */
.hm-nav+div [data-testid="stButton"]>button{
  width:100%!important;text-align:center!important}
/* ── Recommendation row ── */
.hm-rec{border-left:3px solid;padding:.25rem .6rem;margin-bottom:3px;
  background:#0a0a0a;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
/* ── Pulse dot for high-risk ── */
@keyframes hm-pulse{0%,100%{opacity:1}50%{opacity:.3}}
.hm-dot{display:inline-block;width:7px;height:7px;border-radius:50%;
  animation:hm-pulse 1.8s ease-in-out infinite;vertical-align:middle;margin-right:4px}
/* ── Terminal-style buttons (scenario pills + nav arrows) ── */
[data-testid="stButton"]>button{
  font-family:'JetBrains Mono',monospace!important;
  font-size:11px!important;font-weight:700!important;
  letter-spacing:.06em!important;text-transform:uppercase!important;
  border-radius:1px!important;padding:3px 6px!important;
  height:auto!important;min-height:0!important;line-height:1.6!important}
[data-testid="stButton"]>button[kind="secondary"]{
  background:transparent!important;border:1px solid #2a2a2a!important;color:#C8D4E0!important}
[data-testid="stButton"]>button[kind="secondary"]:hover{
  border-color:#CFB991!important;color:#CFB991!important;background:rgba(207,185,145,.04)!important}
[data-testid="stButton"]>button[kind="primary"]{
  background:#0f0f0f!important;border:1px solid #CFB991!important;color:#CFB991!important}
[data-testid="stButton"]>button[kind="primary"]:hover{
  background:rgba(207,185,145,.08)!important}
</style>"""


# ─────────────────────────────────────────────────────────────────────────────
# Score history loader (cached — market data only fetched once per TTL)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def _load_market_risk(start: str, end: str, scenario_id: str = "base") -> tuple[dict, pd.Series]:
    """
    Load market returns, compute avg_corr, then run the full 3-layer risk score
    (identical to what overview.py uses: compute_risk_score(avg_corr, cmd_r, eq_r)).
    Also computes the historical series for the chart.
    Returns (risk_result, score_history).
    """
    from src.analysis.risk_score import compute_risk_score
    try:
        eq_r, cmd_r = load_returns(start, end)
        if eq_r.empty or cmd_r.empty:
            return {}, pd.Series(dtype=float)
        avg_corr  = average_cross_corr_series(eq_r, cmd_r, window=60)
        risk      = compute_risk_score(avg_corr, cmd_r, eq_r=eq_r)
        hist      = risk_score_history(avg_corr, cmd_r, eq_r=eq_r, window=252)
        return risk, hist
    except Exception:
        return {}, pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# § 1  MASTHEAD
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def _home_logo_b64() -> str:
    import base64
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent.parent / "assets" / "logo.png"
    if p.exists():
        return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
    return ""


def _render_masthead(conflict_agg: dict) -> None:
    now      = datetime.datetime.now()
    cis      = conflict_agg.get("portfolio_cis", conflict_agg.get("cis",  50.0))
    n_act    = conflict_agg.get("n_active", sum(
        1 for v in st.session_state.get("_conflict_results_cache", {}).values()
        if v.get("state") == "active"
    ))

    if cis >= 70:
        sit_color, sit_label = "#c0392b", "CRITICAL"
    elif cis >= 50:
        sit_color, sit_label = "#e67e22", "ELEVATED"
    else:
        sit_color, sit_label = "#27ae60", "MODERATE"

    scenario    = get_scenario()
    scenario_id = get_scenario_id()
    sc_note     = (
        f'SCENARIO: <b style="color:{scenario["color"]}">{scenario["label"].upper()}</b>'
        if scenario_id != "base"
        else f'SCENARIO: <span style="color:#C8D4E0">BASE</span>'
    )

    _logo = _home_logo_b64()
    _logo_img = (
        f'<img src="{_logo}" alt="" '
        f'style="height:14px;width:auto;object-fit:contain;opacity:0.55;'
        f'flex-shrink:0;display:block;margin-right:8px;" />'
        if _logo else ""
    )

    # ── Hero identity block ────────────────────────────────────────────────
    st.markdown(
        f'<div style="border-left:2px solid {_GOLD};padding-left:12px;margin-bottom:0.75rem">'
        # Eyebrow: logo mark + programme label
        f'<div style="display:flex;align-items:center;margin-bottom:5px">'
        f'{_logo_img}'
        f'<span style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#DCE4F0">'
        f'Cross-Asset Spillover Monitor &nbsp;·&nbsp; Purdue Daniels &nbsp;·&nbsp; MGMT 69000-120'
        f'</span>'
        f'</div>'
        # Page title
        f'<h1 style="font-family:\'DM Sans\',sans-serif;font-size:1.25rem;font-weight:700;'
        f'color:#e8e8e8;margin:0 0 3px">Command Center</h1>'
        # Subtitle
        f'<p style="{_M}font-size:10px;color:#C8D4E0;margin:0;letter-spacing:.06em">'
        f'Geopolitical &amp; Cross-Asset Intelligence Terminal &nbsp;·&nbsp; '
        f'Equity · Commodity · FX · Fixed Income</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Status bar ────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:#080808;border:1px solid #1e1e1e;'
        f'padding:.3rem 1rem;display:flex;align-items:center;gap:16px;'
        f'flex-wrap:wrap;margin-bottom:.4rem">'
        f'<span style="{_M}font-size:11px;color:#C8D4E0">'
        f'{now.strftime("%a %d %b %Y · %H:%M")}'
        f'</span>'
        f'<span style="{_M}font-size:10px;color:#2a2a2a">│</span>'
        f'<span style="{_M}font-size:11px;color:#C8D4E0">'
        f'{n_act} active conflict{"s" if n_act != 1 else ""}&nbsp;·&nbsp;'
        f'{sc_note}&nbsp;·&nbsp;'
        f'CIS <b style="color:{sit_color}">{cis:.0f}</b>'
        f'</span>'
        f'<span style="background:{sit_color};color:#fff;margin-left:auto;'
        f'{_M}font-size:10px;font-weight:700;padding:2px 7px;letter-spacing:.12em">'
        f'{sit_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § 2  GEOPOLITICAL RISK SCORE — dominant block
# ─────────────────────────────────────────────────────────────────────────────

def _sparkline_svg(
    values: list,
    width: int = 90,
    height: int = 28,
    line_color: str = "#CFB991",
) -> str:
    if len(values) < 2:
        return ""
    vmin, vmax = min(values), max(values)
    span = (vmax - vmin) or 1.0
    pad  = 3
    n    = len(values)
    def _x(i): return i / (n - 1) * width
    def _y(v):  return height - pad - (v - vmin) / span * (height - 2 * pad)
    pts  = [f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(values)]
    path = "M " + " L ".join(pts)
    lx, ly = pts[-1].split(",")
    velocity = values[-1] - values[0]
    if velocity > 1.5:   v_col, v_lbl = "#c0392b", "▲ RISING"
    elif velocity < -1.5: v_col, v_lbl = "#27ae60", "▼ FALLING"
    else:                v_col, v_lbl = "#8E9AAA", "— STABLE"
    return (
        f'<svg width="{width}" height="{height}" style="overflow:visible;vertical-align:middle;margin-left:8px">'
        f'<path d="{path} L{_x(n-1):.1f},{height} L{_x(0):.1f},{height} Z" '
        f'fill="{line_color}" fill-opacity="0.08"/>'
        f'<path d="{path}" fill="none" stroke="{line_color}" stroke-width="1.5" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{lx}" cy="{ly}" r="2.5" fill="{line_color}"/>'
        f'</svg>'
        f'<span style="{_M}font-size:11px;color:{v_col};margin-left:5px;'
        f'vertical-align:middle;font-weight:700">{v_lbl}</span>'
    )


def _bar_row(label: str, value: float, weight: float, color: str, note: str = "") -> str:
    """One layer row: label · weighted bar · value · note"""
    pct = min(value, 100)
    weighted_contribution = value * weight
    return (
        f'<div style="display:flex;align-items:center;gap:7px;padding:4px 0;'
        f'border-bottom:1px solid #0d0d0d">'
        f'<span style="{_M}font-size:10px;color:#DCE4F0;min-width:95px;white-space:nowrap">'
        f'{label}</span>'
        f'<div style="flex:1;background:#111;height:5px;border-radius:1px">'
        f'<div style="width:{pct:.0f}%;height:5px;background:{color};border-radius:1px"></div></div>'
        f'<span style="{_M}font-size:12px;font-weight:700;color:{color};'
        f'min-width:26px;text-align:right">{value:.0f}</span>'
        f'<span style="{_M}font-size:10px;color:#A8B8C8;min-width:52px">'
        f'×{weight:.0%}={weighted_contribution:.0f}</span>'
        f'<span style="{_F}font-size:12px;color:#C8D4E0">{note}</span>'
        f'</div>'
    )


def _build_speedometer_svg(
    score: float,
    color: str,
    label: str,
    delta: float,
) -> str:
    """
    Complete recomposition. Hub lives in the upper third (cy=148).
    Score readout is a fully separate zone starting 60px below hub bottom —
    zero overlap is geometrically impossible given the layout.

    Geometry:
      hub      → cy=148,  r_hub=16  → bottom edge at y=164
      readout  → starts at y=232    → 68px clear gap (≈ 4× hub diameter)
      viewBox  → 400 × 315
    """
    # ── Layout constants ──────────────────────────────────────────────────
    cx, cy   = 200, 148   # pivot — upper portion of canvas
    R        = 118        # arc centerline radius
    SW       = 32         # arc band stroke-width
    R_IN     = R - SW // 2   # inner edge  = 102
    R_OUT    = R + SW // 2   # outer edge  = 134
    # outer decorative rim sits 6px beyond the arc band
    R_RIM    = R_OUT + 6

    # Readout panel — anchored to absolute y, NOT relative to cy
    PANEL_Y  = 218        # top of readout rectangle
    PANEL_H  = 88         # height of readout rectangle
    TY_SCORE = 268        # score number baseline  (cy=148 + hub_r=16 + 104 gap → no collision)
    TY_LABEL = TY_SCORE + 26
    TY_DELTA = TY_LABEL + 15

    # ── Helpers ───────────────────────────────────────────────────────────
    def pt(radius: float, deg: float):
        a = math.radians(deg)
        return cx + radius * math.cos(a), cy - radius * math.sin(a)

    def arc(d1: float, d2: float, radius: float = R, sw: int = 1) -> str:
        """sweep=1 → clockwise in SVG → arc travels UPWARD (top of canvas)."""
        x1, y1 = pt(radius, d1)
        x2, y2 = pt(radius, d2)
        lg = 1 if abs(d1 - d2) > 180 else 0
        return f"M {x1:.2f},{y1:.2f} A {radius},{radius} 0 {lg},{sw} {x2:.2f},{y2:.2f}"

    # ── Zone table ────────────────────────────────────────────────────────
    zones = [
        (180, 135, "#091f10", "#27ae60", 157.5, "LOW"),
        (135,  90, "#201a07", "#c9a800", 112.5, "MOD"),
        ( 90,  45, "#281200", "#e67e22",  67.5, "ELEV"),
        ( 45,   0, "#280505", "#e74c3c",  22.5, "HIGH"),
    ]

    # ── Delta ─────────────────────────────────────────────────────────────
    if abs(delta) < 0.3:
        delta_str, dcol = "STABLE", "#3a3f48"
    elif delta > 0:
        delta_str, dcol = f"▲ +{delta:.1f}", "#e74c3c"
    else:
        delta_str, dcol = f"▼ {delta:.1f}", "#27ae60"

    # ── Needle ───────────────────────────────────────────────────────────
    sc       = max(0.5, min(score, 99.5))
    sdeg     = 180 - sc * 1.8
    tip_r    = R - 5
    nx, ny   = pt(tip_r, sdeg)
    perp     = math.radians(sdeg + 90)
    bw       = 4.5
    b1x = cx + bw * math.cos(perp);  b1y = cy - bw * math.sin(perp)
    b2x = cx - bw * math.cos(perp);  b2y = cy + bw * math.sin(perp)
    npts     = f"{b1x:.2f},{b1y:.2f} {b2x:.2f},{b2y:.2f} {nx:.2f},{ny:.2f}"

    S: list[str] = []
    S.append(
        '<svg viewBox="0 0 400 315" xmlns="http://www.w3.org/2000/svg" '
        'style="width:100%;max-height:315px;display:block">'
    )

    # ── Defs ─────────────────────────────────────────────────────────────
    S.append(
        '<defs>'
        # Broad glow — for needle + score
        '<filter id="gB" x="-80%" y="-80%" width="260%" height="260%">'
        '<feGaussianBlur stdDeviation="7" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
        '</filter>'
        # Tight glow — for inner detail
        '<filter id="gT" x="-50%" y="-50%" width="200%" height="200%">'
        '<feGaussianBlur stdDeviation="3" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
        '</filter>'
        # Score number glow (bloom only)
        '<filter id="gS" x="-30%" y="-30%" width="160%" height="160%">'
        '<feGaussianBlur stdDeviation="4" result="b"/>'
        '<feComposite in="b" in2="SourceGraphic" operator="over"/>'
        '</filter>'
        # Hub radial gradient
        f'<radialGradient id="hg" cx="50%" cy="50%" r="50%">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity="0.7"/>'
        f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/>'
        f'</radialGradient>'
        '</defs>'
    )

    # ════════════════════════════════════════════════════════════════════
    # DIAL MECHANICS  (cy=148 zone)
    # ════════════════════════════════════════════════════════════════════

    # Outer decorative rim (thin ring around the entire band)
    S.append(
        f'<path d="{arc(181,-1, R_RIM)}" fill="none" stroke="#252525" '
        f'stroke-width="1.5" stroke-linecap="butt"/>'
    )

    # Background track (dark substrate for the band)
    S.append(
        f'<path d="{arc(181,-1)}" fill="none" stroke="#0e0e0e" '
        f'stroke-width="{SW + 8}" stroke-linecap="butt"/>'
    )
    S.append(
        f'<path d="{arc(181,-1)}" fill="none" stroke="#161616" '
        f'stroke-width="{SW}" stroke-linecap="butt"/>'
    )

    # Zone arcs — rich dark fill body + bright inner accent line
    for d1, d2, fill, bright, mid, lbl in zones:
        S.append(
            f'<path d="{arc(d1,d2)}" fill="none" stroke="{fill}" '
            f'stroke-width="{SW}" stroke-linecap="butt"/>'
        )
        # Bright inner edge (makes the zone readable without being gaudy)
        S.append(
            f'<path d="{arc(d1,d2, R_IN+3)}" fill="none" stroke="{bright}" '
            f'stroke-width="2.5" stroke-linecap="butt" opacity="0.6"/>'
        )

    # Zone boundary separators (clean radial cuts)
    for sep in [135, 90, 45]:
        x1, y1 = pt(R_IN - 2, sep)
        x2, y2 = pt(R_OUT + 2, sep)
        S.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#0d0d0d" stroke-width="4.5"/>'
        )

    # Zone labels — inside band at arc centerline
    for d1, d2, fill, bright, mid, lbl in zones:
        tx, ty = pt(R, mid)
        S.append(
            f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" '
            f'dominant-baseline="middle" font-family="JetBrains Mono,monospace" '
            f'font-size="9" font-weight="700" fill="{bright}" opacity="0.88">{lbl}</text>'
        )

    # Score progress trail — thin glowing arc from 0 → current
    if score > 1:
        S.append(
            f'<path d="{arc(180, sdeg, R_IN+4)}" fill="none" stroke="{color}" '
            f'stroke-width="{SW//2-2}" stroke-linecap="round" opacity="0.09" filter="url(#gT)"/>'
        )
        S.append(
            f'<path d="{arc(180, sdeg, R_IN+4)}" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linecap="round" opacity="0.78"/>'
        )

    # Needle — glow bloom + solid triangle + bright spine
    S.append(
        f'<polygon points="{npts}" fill="{color}" opacity="0.18" filter="url(#gB)"/>'
    )
    S.append(f'<polygon points="{npts}" fill="{color}" opacity="0.97"/>')
    S.append(
        f'<line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}" '
        f'stroke="rgba(255,255,255,0.28)" stroke-width="1" stroke-linecap="round"/>'
    )
    # Needle tip glow
    S.append(f'<circle cx="{nx:.2f}" cy="{ny:.2f}" r="6" fill="{color}" opacity="0.18"/>')

    # Hub — three-ring instrument pivot
    S.append(f'<circle cx="{cx}" cy="{cy}" r="26" fill="url(#hg)" opacity="0.35"/>')
    S.append(f'<circle cx="{cx}" cy="{cy}" r="16" fill="#0a0a0a" stroke="{color}" stroke-width="2"/>')
    S.append(f'<circle cx="{cx}" cy="{cy}" r="9"  fill="{color}" opacity="0.9"/>')
    S.append(f'<circle cx="{cx}" cy="{cy}" r="3.5" fill="#d8d8d8" opacity="0.55"/>')

    # ════════════════════════════════════════════════════════════════════
    # SCORE READOUT PANEL  (completely separate zone, y ≥ 218)
    # Hub bottom edge: cy + 16 = 164.  Readout top: 218.  Gap = 54px.
    # ════════════════════════════════════════════════════════════════════

    # Subtle panel background
    S.append(
        f'<rect x="{cx-80}" y="{PANEL_Y}" width="160" height="{PANEL_H}" '
        f'rx="4" ry="4" fill="{color}" fill-opacity="0.04" '
        f'stroke="{color}" stroke-opacity="0.08" stroke-width="1"/>'
    )
    # Top accent line on panel
    S.append(
        f'<line x1="{cx-40}" y1="{PANEL_Y}" x2="{cx+40}" y2="{PANEL_Y}" '
        f'stroke="{color}" stroke-opacity="0.3" stroke-width="1"/>'
    )

    # SCORE — hero number
    S.append(
        f'<text x="{cx}" y="{TY_SCORE}" text-anchor="middle" '
        f'font-family="JetBrains Mono,monospace" font-size="66" font-weight="700" '
        f'letter-spacing="-3" fill="{color}" filter="url(#gT)">{score:.0f}</text>'
    )
    # /100 suffix
    S.append(
        f'<text x="{cx+58}" y="{TY_SCORE-20}" text-anchor="start" '
        f'font-family="JetBrains Mono,monospace" font-size="13" '
        f'fill="{color}" opacity="0.35">/100</text>'
    )

    # Zone label (HIGH / ELEVATED / etc.)
    S.append(
        f'<text x="{cx}" y="{TY_LABEL}" text-anchor="middle" '
        f'font-family="JetBrains Mono,monospace" font-size="11" font-weight="700" '
        f'letter-spacing="6" fill="{color}" opacity="0.75">{label.upper()}</text>'
    )

    # Delta
    S.append(
        f'<text x="{cx}" y="{TY_DELTA}" text-anchor="middle" '
        f'font-family="JetBrains Mono,monospace" font-size="9" '
        f'fill="{dcol}" opacity="0.88">{delta_str}</text>'
    )

    S.append('</svg>')
    return "\n".join(S)


def _render_geo_risk_block(
    risk: dict,
    conflict_agg: dict,
    conflict_results: dict,
    score_history: pd.Series | None = None,
) -> None:
    """
    Full-width geopolitical risk score block.
    Left panel  : Plotly speedometer (gauge) + sparkline + confidence.
    Right panel : 3-layer decomposition (CIS/TPS/MCS) + zone reference.
    Bottom      : Historical risk score chart with event markers + regime bands.
    Footer      : Lead conflict · top channel · news GPR · scenario · freshness.
    """
    score     = risk["score"]
    label     = risk["label"]
    color     = risk["color"]
    cis       = risk["cis"]
    tps       = risk["tps"]
    mcs       = risk.get("mcs", 50.0)
    conf      = risk.get("confidence", 0.5)
    top_c     = risk.get("top_conflict") or conflict_agg.get("top_conflict") or "—"
    scenario  = get_scenario()
    geo_mult  = scenario.get("geo_mult", 1.0)
    sc_label  = scenario.get("label", "Base")
    sc_color  = scenario.get("color", _GOLD)
    news_gpr  = risk.get("news_gpr")
    n_threat  = risk.get("n_threat", 0)
    n_act_hl  = risk.get("n_act", 0)
    freshness = freshness_badge_html("conflict_model")

    # Score history → sparkline
    history: list = list(st.session_state.get("_score_history", []))
    history.append(float(score))
    history = history[-10:]
    st.session_state["_score_history"] = history
    spark_html = _sparkline_svg(history, line_color=color)

    # Delta vs previous run
    prev_score = st.session_state.get("_prev_geo_score_v2", score)
    delta      = score - prev_score
    st.session_state["_prev_geo_score_v2"] = score

    # Top transmission channel
    ch_scores: dict[str, float] = {}
    for r in conflict_results.values():
        if r.get("state") != "active":
            continue
        w = r["cis"] / 100
        for ch, v in r.get("transmission", {}).items():
            ch_scores[ch] = ch_scores.get(ch, 0.0) + v * w
    top_ch     = max(ch_scores, key=ch_scores.get) if ch_scores else "—"
    top_ch_val = ch_scores.get(top_ch, 0.0)

    # MCS dominant signal note
    mcs_comp = risk.get("mcs_components", {})
    mcs_note = ""
    if mcs_comp:
        top_mcs_sig = max(mcs_comp, key=lambda k: abs(mcs_comp.get(k, 0)))
        mcs_note = top_mcs_sig.replace("_", " ")

    # Color per layer
    cis_color = "#c0392b" if cis >= 65 else "#e67e22" if cis >= 45 else "#8E9AAA"
    tps_color = "#c0392b" if tps >= 65 else "#e67e22" if tps >= 45 else "#CFB991"
    mcs_color = "#c0392b" if mcs >= 65 else "#e67e22" if mcs >= 45 else "#8E9AAA"
    conf_color = "#27ae60" if conf >= 0.7 else "#e67e22" if conf >= 0.5 else "#c0392b"

    top_c_disp  = top_c.replace("_", " ").title() if top_c and top_c != "—" else "—"
    top_ch_disp = top_ch.replace("_", " ").upper() if top_ch != "—" else "—"
    news_gpr_val = f'{news_gpr:.0f}' if news_gpr is not None else '—'
    news_gpr_sub = f'{n_threat}T / {n_act_hl}A' if news_gpr is not None else 'awaiting'

    # ── Panel header — full-width, sits above the two columns ─────────────
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;'
        f'border-top:3px solid {color};border-bottom:1px solid #1e1e1e;'
        f'background:#0a0a0a;padding:.3rem .8rem;margin-bottom:.1rem">'
        f'<span style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#DCE4F0">Geopolitical Risk Score</span>'
        f'<span style="background:{color};color:#000;{_M}font-size:11px;font-weight:700;'
        f'padding:1px 7px;letter-spacing:.10em">{label.upper()}&nbsp;{score:.0f}</span>'
        f'<span style="{_M}font-size:10px;color:#C8D4E0;margin-left:4px">'
        f'Confidence&nbsp;<b style="color:{conf_color}">{conf:.0%}</b>'
        f'&nbsp;·&nbsp;{spark_html}</span>'
        f'<span style="margin-left:auto">{freshness}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Two-column hero: gauge | history chart ─────────────────────────────
    col_gauge, col_hist = st.columns([1, 1.65], gap="medium")

    with col_gauge:
        st.markdown(
            f'<p style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
            f'text-transform:uppercase;color:#DCE4F0;margin:0 0 4px">Risk Gauge</p>',
            unsafe_allow_html=True,
        )
        svg_gauge = _build_speedometer_svg(score, color, label, delta)
        st.markdown(svg_gauge, unsafe_allow_html=True)
        st.markdown(
            f'<p style="{_M}font-size:10px;color:#C8D4E0;margin:.3rem 0 0;'
            f'padding:0 2px">40% CIS&nbsp;·&nbsp;35% TPS&nbsp;·&nbsp;25% MCS</p>',
            unsafe_allow_html=True,
        )

    with col_hist:
        st.markdown(
            f'<p style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
            f'text-transform:uppercase;color:#DCE4F0;margin:0 0 0">'
            f'Historical Risk Score'
            f'<span style="font-weight:400;letter-spacing:.06em;color:#A8B8C8;'
            f'margin-left:8px;text-transform:none">market-observable proxy · corr · oil-gold · vol</span>'
            f'</p>',
            unsafe_allow_html=True,
        )
        if score_history is not None and not score_history.empty:
            fig_hist = plot_risk_history(score_history, height=310)
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#DCE4F0", "family": "JetBrains Mono, monospace", "size": 12},
                title_text="",
                margin=dict(l=44, r=16, t=6, b=28),
                xaxis=dict(
                    showgrid=False,
                    tickfont={"size": 11, "color": "#C8D4E0", "family": "JetBrains Mono"},
                    linecolor="#1e1e1e",
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="#1a1a1a",
                    tickfont={"size": 11, "color": "#C8D4E0", "family": "JetBrains Mono"},
                    linecolor="#1e1e1e",
                    title_text="",
                    tickvals=[0, 25, 50, 75, 100],
                    ticktext=["0", "25 LOW", "50 MOD", "75 ELEV", "100"],
                ),
                legend=dict(
                    font={"size": 11, "color": "#DCE4F0"},
                    bgcolor="rgba(0,0,0,0)",
                    borderwidth=0,
                ),
            )
            fig_hist.add_hline(
                y=float(score),
                line=dict(color=color, width=1.2, dash="dot"),
                annotation_text=f'NOW {score:.0f}',
                annotation_font={"size": 8, "color": color, "family": "JetBrains Mono"},
                annotation_position="right",
            )
            st.plotly_chart(fig_hist, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.markdown(
                f'<div style="{_F}font-size:12px;color:#A8B8C8;padding:4rem 0;'
                f'text-align:center;border:1px solid #1a1a1a;margin-top:4px">'
                f'Historical series computing — loads after market data.</div>',
                unsafe_allow_html=True,
            )

    # ── Score decomposition — pure CSS grid (no st.columns) ───────────────
    comp = risk.get("components", {})
    dw   = risk.get("weights", {})

    if comp:
        bars_html = ""
        for name, val in comp.items():
            c_col  = "#c0392b" if val > 70 else "#e67e22" if val > 45 else "#2e7d32"
            pct    = min(val, 100)
            indent = name.startswith("  ")
            bars_html += (
                f'<div style="{"padding-left:10px;border-left:2px solid #1e1e1e;" if indent else ""}">'
                f'<div style="display:flex;justify-content:space-between;'
                f'{_F}font-size:{"11" if indent else "12"}px;margin-bottom:2px">'
                f'<span style="color:{"#A8B8C8" if indent else "#C8D4E0"}">{name.strip()}</span>'
                f'<span style="{_M}font-size:12px;font-weight:700;color:{c_col}">{val:.0f}</span>'
                f'</div>'
                f'<div style="height:{"3" if indent else "4"}px;background:#111;border-radius:1px;margin-bottom:5px">'
                f'<div style="width:{pct:.0f}%;height:{"3" if indent else "4"}px;'
                f'background:{c_col};border-radius:1px"></div></div></div>'
            )
        decomp_inner = (
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0 24px">'
            f'{bars_html}</div>'
        )
    else:
        bars_html = ""
        for lbl, val, col in [
            ("Conflict Intensity (CIS · 40%)",    cis, cis_color),
            ("Transmission Pressure (TPS · 35%)", tps, tps_color),
            ("Market Confirmation (MCS · 25%)",   mcs, mcs_color),
        ]:
            pct = min(val, 100)
            bars_html += (
                f'<div>'
                f'<div style="display:flex;justify-content:space-between;'
                f'{_F}font-size:12px;margin-bottom:2px">'
                f'<span style="color:#D8E0EC">{lbl}</span>'
                f'<span style="{_M}font-weight:700;color:{col}">{val:.0f}</span></div>'
                f'<div style="height:4px;background:#111;border-radius:1px;margin-bottom:5px">'
                f'<div style="width:{pct:.0f}%;height:4px;background:{col};'
                f'border-radius:1px"></div></div></div>'
            )
        decomp_inner = (
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0 24px">'
            f'{bars_html}</div>'
        )

    wt_html = ""
    if dw:
        wt_items = "".join(
            f'<span style="{_M}font-size:10px;color:#C8D4E0">'
            f'{k.replace("_", " ")}&nbsp;<b style="color:{_GOLD}">{v*100:.0f}%</b></span>'
            for k, v in dw.items()
        )
        wt_html = (
            f'<div style="display:flex;gap:10px;flex-wrap:wrap;'
            f'padding-top:5px;border-top:1px solid #111;margin-top:2px">'
            f'<span style="{_M}font-size:11px;letter-spacing:.16em;text-transform:uppercase;'
            f'color:#A8B8C8;align-self:center">Dynamic&nbsp;Weights</span>'
            f'{wt_items}</div>'
        )

    st.markdown(
        f'<div style="border-top:1px solid #1e1e1e;margin-top:.2rem;'
        f'padding:.3rem .4rem .2rem">'
        f'<p style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#DCE4F0;margin:0 0 .45rem">Score Decomposition</p>'
        f'{decomp_inner}'
        f'{wt_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── KPI strip — CSS grid guarantees equal-width tiles ─────────────────
    def _kt(lbl: str, val: str, vc: str, sub: str = "") -> str:
        return (
            f'<div style="padding:.4rem .65rem;background:#080808;'
            f'border:1px solid #1a1a1a;border-top:2px solid {vc}">'
            f'<div style="{_M}font-size:10px;font-weight:700;letter-spacing:.16em;'
            f'text-transform:uppercase;color:#A8B8C8;margin-bottom:3px">{lbl}</div>'
            f'<div style="{_M}font-size:1.0rem;font-weight:700;color:{vc};line-height:1.1">{val}</div>'
            + (f'<div style="{_M}font-size:11px;color:#C8D4E0;margin-top:2px">{sub}</div>' if sub else "")
            + f'</div>'
        )

    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:5px;'
        f'margin-top:.3rem;padding-top:.3rem;border-top:1px solid #1e1e1e">'
        + _kt("Conflict Intensity",    f'{cis:.0f}',       cis_color, "CIS · 40% weight")
        + _kt("Transmission Pressure", f'{tps:.0f}',       tps_color, "TPS · 35% weight")
        + _kt("Market Confirmation",   f'{mcs:.0f}',       mcs_color, mcs_note or "MCS · 25% weight")
        + _kt("Lead Conflict",         top_c_disp,         color,     "highest CIS actor")
        + _kt("News GPR",              news_gpr_val,       _GOLD,     news_gpr_sub)
        + _kt("Scenario",              sc_label.upper(),   sc_color,  f'geo ×{geo_mult:.2f}')
        + f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § 3  CONTEXT NARRATIVE
# ─────────────────────────────────────────────────────────────────────────────

def _render_context_narrative(risk: dict, conflict_results: dict) -> None:
    """
    2-sentence data-driven plain-language interpretation of the current score.
    Generated from live CIS/TPS/MCS values — not templated filler.
    """
    score    = risk["score"]
    cis      = risk["cis"]
    tps      = risk["tps"]
    mcs      = risk.get("mcs", 50.0)
    top_c    = (risk.get("top_conflict") or "an unidentified conflict")
    top_c    = top_c.replace("_", " ").title()
    scenario = get_scenario()
    sc_id    = get_scenario_id()

    # Top transmission channel
    ch_scores: dict[str, float] = {}
    for r in conflict_results.values():
        if r.get("state") != "active":
            continue
        w = r["cis"] / 100
        for ch, v in r.get("transmission", {}).items():
            ch_scores[ch] = ch_scores.get(ch, 0.0) + v * w
    top_ch = max(ch_scores, key=ch_scores.get) if ch_scores else None
    top_ch_disp = top_ch.replace("_", " ") if top_ch else "commodity markets"

    # Sentence 1: What is driving the score
    if cis >= 70 and tps >= 60:
        s1 = (
            f"<b style='color:{risk['color']}'>{top_c}</b> is the dominant driver "
            f"with CIS {cis:.0f}/100 — critical conflict intensity — and an open "
            f"<b>{top_ch_disp}</b> transmission channel carrying that risk into asset prices at TPS {tps:.0f}."
        )
    elif cis >= 50 and tps >= 50:
        s1 = (
            f"<b>{top_c}</b> is sustaining elevated conflict intensity "
            f"(CIS {cis:.0f}) with measurable transmission through the "
            f"<b>{top_ch_disp}</b> channel (TPS {tps:.0f}). "
            f"Asset price impact is active but not at critical levels."
        )
    elif cis >= 50 and tps < 40:
        s1 = (
            f"Conflict intensity is elevated (CIS {cis:.0f} — <b>{top_c}</b> is the "
            f"primary driver), but the transmission channel to asset markets "
            f"is currently suppressed (TPS {tps:.0f}). "
            f"Risk exists but is not yet flowing into prices."
        )
    elif cis < 40 and mcs >= 60:
        s1 = (
            f"Fundamental conflict intensity is moderate (CIS {cis:.0f}), but the "
            f"market confirmation layer (MCS {mcs:.0f}) is registering elevated signals — "
            f"markets are pricing more risk than the conflict scorecard reflects."
        )
    else:
        s1 = (
            f"Conflict intensity is contained (CIS {cis:.0f}, TPS {tps:.0f}). "
            f"<b>{top_c}</b> remains the watch list leader "
            f"but transmission pressure into asset markets is subdued."
        )

    # Sentence 2: Scenario + market implication
    if sc_id != "base":
        s2 = (
            f"The <b style='color:{scenario['color']}'>{scenario['label']}</b> scenario is active "
            f"(geo multiplier ×{scenario['geo_mult']:.2f}), amplifying all scores — "
            f"refer to Trade Ideas and Scenario Engine for forward risk under this lens."
        )
    elif score >= 65:
        s2 = (
            f"At this score level, historical spillover patterns show elevated correlation "
            f"between oil-linked equities and commodities. The Stress Test and Correlation "
            f"pages will quantify portfolio-level impact."
        )
    elif score >= 45:
        s2 = (
            f"Score sits in the Elevated band — consistent with past pre-escalation windows. "
            f"Monitor Threat vs Act and Conflict Intel for early signals before scores move further."
        )
    else:
        s2 = (
            f"Risk environment is contained. Standard monitoring posture: "
            f"daily Overview check, weekly Spillover review, "
            f"and Trade Ideas for any regime-matched positioning opportunities."
        )

    st.markdown(
        f'<div style="background:#080808;border:1px solid #1e1e1e;'
        f'border-left:3px solid #555960;padding:.35rem .9rem;margin-bottom:.35rem">'
        f'<span style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#DCE4F0;display:block;margin-bottom:6px">'
        f'Current Situation</span>'
        f'<p style="{_F}font-size:13px;color:#D8E0EC;line-height:1.65;margin:0 0 4px">'
        f'{s1}</p>'
        f'<p style="{_F}font-size:13px;color:#C8D4E0;line-height:1.65;margin:0">'
        f'{s2}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § 4  INTEL PANEL — conflict table (left) + channels (right)
# ─────────────────────────────────────────────────────────────────────────────

def _render_intel_panel(conflict_results: dict) -> None:
    # Shared section header above both columns
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;'
        f'border-bottom:1px solid #1e1e1e;padding-bottom:.2rem;margin-bottom:.3rem">'
        f'<span style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#DCE4F0">Intelligence Panel</span>'
        f'<span style="{_F}font-size:12px;color:#C8D4E0">'
        f'Active conflicts · CIS/TPS scores · transmission channel pressure</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1.6, 1], gap="medium")

    # ── Left: Conflict table ───────────────────────────────────────────────
    with col_left:
        _TREND_I = {"rising": "▲", "stable": "—", "falling": "▼"}
        _TREND_C = {"rising": "#c0392b", "stable": "#8E9AAA", "falling": "#27ae60"}

        _TH = f'{_M}font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:#A8B8C8'
        col_header = (
            f'<div style="display:flex;gap:8px;padding:0 0 4px;'
            f'border-bottom:1px solid #1e1e1e;margin-bottom:3px">'
            f'<span style="{_TH};min-width:70px">Conflict</span>'
            f'<span style="{_TH};flex:1">CIS Intensity</span>'
            f'<span style="{_TH};min-width:26px;text-align:right">Val</span>'
            f'<span style="{_TH};min-width:16px"> </span>'
            f'<span style="{_TH};min-width:34px">TPS</span>'
            f'<span style="{_TH}">Top Channel</span>'
            f'</div>'
        )
        rows = ""
        for cid, r in sorted(conflict_results.items(), key=lambda x: x[1]["cis"], reverse=True):
            tx      = r.get("transmission", {})
            top_ch  = max(tx, key=tx.get) if tx else "—"
            t_icon  = _TREND_I.get(r.get("trend", "stable"), "—")
            t_color = _TREND_C.get(r.get("trend", "stable"), "#8E9AAA")
            cis_col = "#c0392b" if r["cis"] >= 65 else "#e67e22" if r["cis"] >= 45 else "#8E9AAA"
            tps_val = r.get("tps", r.get("transmission_pressure", 50))
            tps_col = "#c0392b" if tps_val >= 65 else "#e67e22" if tps_val >= 45 else "#8E9AAA"
            rows += (
                f'<div class="hm-crow">'
                f'<span style="{_M}font-size:11px;font-weight:700;color:{r["color"]};'
                f'min-width:70px">{r["label"]}</span>'
                f'<div style="flex:1;background:#111;height:4px;border-radius:1px">'
                f'<div style="width:{r["cis"]:.0f}%;height:4px;background:{cis_col};'
                f'border-radius:1px"></div></div>'
                f'<span style="{_M}font-size:12px;color:{cis_col};font-weight:700;'
                f'min-width:26px;text-align:right">{r["cis"]:.0f}</span>'
                f'<span style="{_M}font-size:12px;color:{t_color};min-width:16px">{t_icon}</span>'
                f'<span style="{_M}font-size:11px;color:{tps_col};min-width:34px">'
                f'TPS {tps_val:.0f}</span>'
                f'<span style="{_F}font-size:12px;color:#C8D4E0">'
                f'{top_ch.replace("_", " ")}</span>'
                f'</div>'
            )
        st.markdown(
            f'<div style="background:#0a0a0a;border:1px solid #1e1e1e;padding:.45rem .7rem">'
            f'<p style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
            f'text-transform:uppercase;color:#DCE4F0;margin:0 0 .35rem">'
            f'Active Conflict Monitor &nbsp;·&nbsp; CIS / TPS / Top Channel</p>'
            + col_header + rows
            + f'</div>',
            unsafe_allow_html=True,
        )

    # ── Right: Top transmission channels ──────────────────────────────────
    with col_right:
        ch_scores: dict[str, float] = {}
        for r in conflict_results.values():
            if r.get("state") != "active":
                continue
            w = r["cis"] / 100
            for ch, v in r.get("transmission", {}).items():
                ch_scores[ch] = ch_scores.get(ch, 0.0) + v * w
        if ch_scores:
            ranked  = sorted(ch_scores.items(), key=lambda x: x[1], reverse=True)[:7]
            max_val = ranked[0][1] if ranked else 1.0
            rows_ch = ""
            for ch, val in ranked:
                pct    = min(val / max_val * 100, 100)
                ch_col = "#c0392b" if pct >= 80 else "#e67e22" if pct >= 55 else _GOLD
                rows_ch += (
                    f'<div style="display:flex;align-items:center;gap:7px;padding:3px 0;'
                    f'border-bottom:1px solid #111">'
                    f'<span style="{_M}font-size:10px;color:#C8D4E0;min-width:90px;'
                    f'white-space:nowrap">{ch.replace("_", " ").upper()}</span>'
                    f'<div style="flex:1;background:#111;height:4px;border-radius:1px">'
                    f'<div style="width:{pct:.0f}%;height:4px;background:{ch_col};'
                    f'border-radius:1px"></div></div>'
                    f'<span style="{_M}font-size:11px;color:{ch_col};min-width:28px;'
                    f'text-align:right">{val:.2f}</span>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="background:#0a0a0a;border:1px solid #1e1e1e;padding:.45rem .7rem">'
                f'<p style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
                f'text-transform:uppercase;color:#DCE4F0;margin:0 0 .35rem">'
                f'Top Transmission Channels</p>'
                f'{rows_ch}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="background:#0a0a0a;border:1px solid #1e1e1e;padding:.45rem .7rem">'
                f'<p style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
                f'text-transform:uppercase;color:#DCE4F0;margin:0 0 .35rem">'
                f'Top Transmission Channels</p>'
                f'<span style="{_F}font-size:12px;color:#A8B8C8">'
                f'No active transmission data.</span></div>',
                unsafe_allow_html=True,
            )



# ─────────────────────────────────────────────────────────────────────────────
# § 5  SCENARIO SWITCH
# ─────────────────────────────────────────────────────────────────────────────

_IMPACT = {
    "base":            "All scores at face value. No multipliers applied.",
    "escalation":      "Geo ×1.45 · Vol ×1.30 · Safe-haven assets re-weighted up.",
    "de_escalation":   "Geo ×0.60 · Short-bias active · Geo premiums compress.",
    "supply_shock":    "TPS +50% · Commodity supply chains amplified.",
    "sanctions_shock": "FX and sanctions channels elevated · Safe-haven bid active.",
    "shipping_shock":  "Chokepoint TPS ×1.60 · All strait risk scores amplified.",
    "risk_off":        "Broad vol ×1.50 · Equity shorts favoured · Gold / TLT bid.",
    "recovery":        "Geo ×0.70 · Cyclical longs favoured · Risk premiums deflate.",
}


def _render_scenario_switch() -> None:
    current_sid = get_scenario_id()
    current_def = SCENARIOS[current_sid]
    impact      = _IMPACT.get(current_sid, "")

    st.markdown(
        f'<div style="background:#0d0d0d;border:1px solid #1e1e1e;'
        f'padding:.35rem .9rem;margin:.35rem 0">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:5px;flex-wrap:wrap">'
        f'<span style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#DCE4F0">Scenario Lens</span>'
        f'<span style="background:{current_def["color"]};color:#000;'
        f'{_M}font-size:10px;font-weight:700;padding:1px 6px;letter-spacing:.10em">'
        f'{current_def["label"].upper()}</span>'
        f'<span style="{_F}font-size:12px;color:#D8E0EC">{current_def["desc"]}</span>'
        f'<span style="{_M}font-size:11px;color:#C8D4E0;margin-left:auto">{impact}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(len(SCENARIO_ORDER))
    for i, sid in enumerate(SCENARIO_ORDER):
        sdef   = SCENARIOS[sid]
        active = (sid == current_sid)
        if cols[i].button(
            sdef["label"],
            key=f"scen_{sid}",
            type="primary" if active else "secondary",
            use_container_width=True,
        ):
            set_scenario(sid)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# § 6  WHERE TO GO NOW — live-data-driven recommendations
# ─────────────────────────────────────────────────────────────────────────────

def _render_next_action(conflict_agg: dict, conflict_results: dict) -> None:
    cis         = conflict_agg.get("portfolio_cis", conflict_agg.get("cis", 50.0))
    tps         = conflict_agg.get("portfolio_tps", conflict_agg.get("tps", 50.0))
    scenario_id = get_scenario_id()
    scenario    = get_scenario()
    recs: list[dict] = []

    if cis >= 55:
        top_c = (conflict_agg.get("top_conflict") or "active conflict").replace("_"," ").title()
        recs.append({
            "color": "#c0392b", "tag": "⚡ ALERT",
            "label": "Conflict Intel", "page_id": "conflict_intelligence",
            "text": (
                f"Portfolio CIS is <b style='color:#c0392b'>{cis:.0f}/100</b>. "
                f"<b>{top_c}</b> is the lead driver. "
                f"Full CIS/TPS scorecard and per-channel breakdown."
            ),
        })
    if tps >= 55:
        recs.append({
            "color": "#e67e22", "tag": "⚡ ALERT",
            "label": "Transmission Matrix", "page_id": "transmission_matrix",
            "text": (
                f"Transmission Pressure at <b style='color:#e67e22'>{tps:.0f}/100</b>. "
                f"Risk is actively flowing through commodity and equity channels. "
                f"Matrix heatmap shows exactly which channels are open and at what velocity."
            ),
        })
    if scenario_id != "base":
        recs.append({
            "color": scenario["color"], "tag": "▶ SCENARIO",
            "label": "Scenario Engine", "page_id": "scenario_engine",
            "text": (
                f"<b>{scenario['label']}</b> scenario active "
                f"(geo ×{scenario['geo_mult']:.2f}, vol ×{scenario['vol_mult']:.2f}). "
                f"Full stressed P&amp;L and regime simulation under this lens."
            ),
        })
    n_pending = st.session_state.get("_stored_pending", 0) or 0
    if n_pending > 0:
        recs.append({
            "color": "#27ae60", "tag": "📋 ACTION",
            "label": "Trade Ideas", "page_id": "trade_ideas",
            "text": (
                f"<b style='color:#27ae60'>{n_pending} trade idea"
                f"{'s' if n_pending != 1 else ''}</b> awaiting review. "
                f"Conflict-driven candidates filtered to current regime with payoff tables."
            ),
        })
    if not recs:
        recs.append({
            "color": "#2980b9", "tag": "▶ START HERE",
            "label": "Overview", "page_id": "overview",
            "text": (
                "Risk environment is contained. Start with <b>Overview</b> — "
                "regime classification, correlation heatmap, and the AI analyst morning briefing."
            ),
        })

    recs = recs[:3]

    st.markdown(
        f'<div style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#DCE4F0;margin:.35rem 0 .2rem">'
        f'Where To Go Now</div>'
        f'<div style="{_F}font-size:12px;color:#C8D4E0;margin-bottom:.3rem">'
        f'Live recommendations based on current scores — updates every refresh</div>',
        unsafe_allow_html=True,
    )

    for r in recs:
        st.markdown(
            f'<div class="hm-rec" style="border-color:{r["color"]}">'
            f'<span style="{_M}font-size:11px;font-weight:700;color:{r["color"]};'
            f'white-space:nowrap">{r["tag"]}</span>'
            f'<span style="{_M}font-size:11px;font-weight:700;color:#e8e9ed;'
            f'white-space:nowrap;min-width:130px">{r["label"]}</span>'
            f'<span style="{_F}font-size:12px;color:#C8D4E0;flex:1">{r["text"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# § 7  NAVIGATE TERMINAL — grouped quick-jump
# ─────────────────────────────────────────────────────────────────────────────

_JUMP_GROUPS = [
    {
        "group":   "Market Analysis",
        "color":   "#2980b9",
        "caption": "Daily: regime · correlations · spillover direction · map exposure.",
        "items": [
            ("Overview",       "overview",       "Regime · risk score · AI briefing",      "daily", "1"),
            ("Correlation",    "correlation",    "Rolling cross-asset correlation",         "daily", "2"),
            ("Spillover",      "spillover",      "Diebold-Yilmaz directional flows",       "deep",  "3"),
            ("War Impact Map", "war_impact_map", "Conflict → asset exposure overlay",      "alert", "4"),
        ],
    },
    {
        "group":   "Geopolitical Intelligence",
        "color":   "#CFB991",
        "caption": "Open when a conflict score moves or a headline requires deeper context.",
        "items": [
            ("Conflict Intel",   "conflict_intelligence", "CIS/TPS breakdown per conflict",     "alert", "5"),
            ("Threat vs Act",    "threat_act_monitor",    "Live news GPR · threat/act signal",  "alert", "6"),
            ("Transmission",     "transmission_matrix",   "Active transmission channel heatmap","alert", "7"),
            ("Exposure Scoring", "exposure_scoring",      "Per-asset scenario-adjusted SAS",    "deep",  "8"),
            ("Geopolitical",     "geopolitical",          "Event history · RSS · GPR timeline", "deep",  "9"),
            ("Strait Watch",     "strait_watch",          "Chokepoint disruption monitor",      "alert", "10"),
        ],
    },
    {
        "group":   "Strategy Tools",
        "color":   "#27ae60",
        "caption": "Use when you have a view and need to size, stress-test, or structure a position.",
        "items": [
            ("Trade Ideas",     "trade_ideas",     "Conflict-driven regime-filtered trades", "daily", "11"),
            ("Scenario Engine", "scenario_engine", "Scenario P&L · what-if simulation",     "deep",  "12"),
            ("Stress Test",     "stress_test",     "Portfolio drawdown under shock events",  "deep",  "13"),
            ("Macro Dashboard", "macro_dashboard", "Rates · inflation · yield curve",        "daily", "14"),
        ],
    },
]

_TAG_META = {
    "daily": ("DAILY",     "hm-tag-daily"),
    "alert": ("ON ALERT",  "hm-tag-alert"),
    "deep":  ("DEEP-DIVE", "hm-tag-deep"),
}


def _render_quickjump() -> None:
    # Header + legend
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin:.25rem 0 .2rem;flex-wrap:wrap">'
        f'<span style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#DCE4F0">Navigate Terminal</span>'
        f'<span style="{_F}font-size:12px;color:#C8D4E0">14 modules</span>'
        f'<span style="margin-left:auto;display:flex;gap:8px;align-items:center">'
        f'<span class="hm-tag hm-tag-daily">DAILY</span>'
        f'<span style="{_M}font-size:10px;color:#C8D4E0">open every session</span>'
        f'<span class="hm-tag hm-tag-alert" style="margin-left:4px">ON ALERT</span>'
        f'<span style="{_M}font-size:10px;color:#C8D4E0">when a score moves</span>'
        f'<span class="hm-tag hm-tag-deep" style="margin-left:4px">DEEP-DIVE</span>'
        f'<span style="{_M}font-size:10px;color:#C8D4E0">research &amp; sizing</span>'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    for group in _JUMP_GROUPS:
        g_color = group["color"]
        st.markdown(
            f'<div style="display:flex;align-items:baseline;gap:8px;'
            f'border-left:2px solid {g_color};padding-left:7px;margin:5px 0 3px">'
            f'<span style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
            f'text-transform:uppercase;color:{g_color}">{group["group"]}</span>'
            f'<span style="{_F}font-size:12px;color:#C8D4E0">{group["caption"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        items  = group["items"]
        n_cols = min(len(items), 4)
        cols   = st.columns(n_cols, gap="small")

        for i, (label, page_id, desc, tag, _sc) in enumerate(items):
            tag_label, tag_cls = _TAG_META.get(tag, ("", ""))
            with cols[i % n_cols]:
                st.markdown(
                    f'<div class="hm-nav" style="border-top:2px solid {g_color};min-height:52px">'
                    f'<div style="display:flex;align-items:center;margin-bottom:3px">'
                    f'<span style="{_M}font-size:11px;font-weight:700;color:{g_color}">{label}</span>'
                    f'<span class="hm-tag {tag_cls}">{tag_label}</span>'
                    f'</div>'
                    f'<div style="{_F}font-size:12px;color:#C8D4E0;line-height:1.4">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button("→", key=f"qj_{page_id}", use_container_width=True):
                    st.query_params["page"] = page_id
                    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# § 8  LIVE SIGNALS — strait snapshot + what-changed delta
# ─────────────────────────────────────────────────────────────────────────────

_STRAITS = [
    {"name": "Strait of Hormuz",  "base_risk": 82, "conflict": "Iran/Hormuz",      "flow": "~20% global oil",    "color": "#c0392b"},
    {"name": "Bab-el-Mandeb",     "base_risk": 68, "conflict": "Red Sea / Houthi", "flow": "~10% global trade",  "color": "#e67e22"},
    {"name": "Taiwan Strait",     "base_risk": 44, "conflict": "Taiwan",            "flow": "~50% container",     "color": "#2980b9"},
    {"name": "Suez Canal",        "base_risk": 55, "conflict": "Red Sea",           "flow": "~12% global trade",  "color": "#e67e22"},
    {"name": "Strait of Malacca", "base_risk": 28, "conflict": "Taiwan / India",    "flow": "~25% global oil",    "color": "#8E9AAA"},
]


def _delta_html(delta, unit: str = "") -> str:
    if delta is None:
        return f'<span style="{_M}font-size:11px;color:#A8B8C8">— first run</span>'
    if abs(delta) < 0.3:
        return f'<span class="hm-fl">— flat</span>'
    if delta > 0:
        return f'<span class="hm-up">▲ +{delta:.1f}{unit}</span>'
    return f'<span class="hm-dn">▼ {delta:.1f}{unit}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# § 1.5  MARKET PULSE — macro KPI strip
# ─────────────────────────────────────────────────────────────────────────────

_PULSE_TICKERS = [
    ("^VIX",      "VIX",     "",   False),   # label, suffix, invert_color
    ("DX-Y.NYB",  "DXY",     "",   False),
    ("^GSPC",     "S&P 500", "",   False),
    ("CL=F",      "WTI",     "/b", False),
    ("GC=F",      "Gold",    "/oz",False),
    ("^TNX",      "10Y Yld", "%",  False),
]


@st.cache_data(ttl=300, show_spinner=False)
def _load_market_pulse() -> list[dict]:
    """Fetch last 2 closes for macro pulse tickers. Returns list of dicts."""
    try:
        import yfinance as yf
        syms = [t[0] for t in _PULSE_TICKERS]
        raw  = yf.download(syms, period="5d", auto_adjust=True, progress=False, threads=True)
        if raw.empty:
            return []
        close = raw["Close"] if "Close" in raw.columns else raw
        results = []
        for sym, label, suffix, _ in _PULSE_TICKERS:
            if sym not in close.columns:
                continue
            s = close[sym].dropna()
            if len(s) < 2:
                continue
            val  = float(s.iloc[-1])
            prev = float(s.iloc[-2])
            chg  = val - prev
            pct  = chg / prev * 100 if prev else 0.0
            results.append({
                "sym": sym, "label": label, "suffix": suffix,
                "val": val, "chg": chg, "pct": pct,
            })
        return results
    except Exception:
        return []


def _render_market_pulse() -> None:
    data = _load_market_pulse()
    if not data:
        return

    def _tile(d: dict) -> str:
        pct   = d["pct"]
        chg   = d["chg"]
        # VIX: rising = bad (red); everything else: up = green for equities/commodities
        # We keep standard: red = negative, green = positive for price
        # For VIX specifically, invert: rising VIX = red
        is_vix = d["sym"] == "^VIX"
        if abs(pct) < 0.05:
            c, arrow = "#555960", "—"
        elif pct > 0:
            c, arrow = ("#c0392b" if is_vix else "#27ae60"), "▲"
        else:
            c, arrow = ("#27ae60" if is_vix else "#c0392b"), "▼"

        val_fmt = f'{d["val"]:.2f}' if d["val"] < 10000 else f'{d["val"]:,.0f}'
        chg_fmt = f'{arrow} {abs(pct):.2f}%'
        return (
            f'<div style="flex:1;min-width:80px;padding:.35rem .6rem;background:#080808;'
            f'border:1px solid #1a1a1a;border-top:2px solid {c}">'
            f'<div style="{_M}font-size:10px;font-weight:700;letter-spacing:.16em;'
            f'text-transform:uppercase;color:#A8B8C8;margin-bottom:2px">{d["label"]}</div>'
            f'<div style="{_M}font-size:11px;font-weight:700;color:#e8e9ed;line-height:1.1">'
            f'{val_fmt}<span style="font-size:10px;color:#C8D4E0">{d["suffix"]}</span></div>'
            f'<div style="{_M}font-size:10px;color:{c};margin-top:1px">{chg_fmt}</div>'
            f'</div>'
        )

    tiles_html = "".join(_tile(d) for d in data)
    st.markdown(
        f'<div style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#A8B8C8;margin-bottom:2px">Market Pulse</div>'
        f'<div style="display:flex;gap:4px;flex-wrap:nowrap;margin-bottom:.3rem">'
        f'{tiles_html}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § 1.6  PORTFOLIO PULSE — NAV + 1-day P&L + top movers (conditional)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _load_portfolio_returns(tickers_key: str, tickers: tuple) -> dict[str, float]:
    """
    Fetch 2-day closes for portfolio tickers, return {ticker: pct_change_1d}.
    tickers_key is a hashable cache key (joined string).
    """
    if not tickers:
        return {}
    try:
        import yfinance as yf
        raw   = yf.download(list(tickers), period="5d", auto_adjust=True, progress=False, threads=True)
        if raw.empty:
            return {}
        close = raw["Close"] if "Close" in raw.columns else raw
        result: dict[str, float] = {}
        if isinstance(close, pd.Series):
            s = close.dropna()
            if len(s) >= 2:
                result[tickers[0]] = (s.iloc[-1] - s.iloc[-2]) / s.iloc[-2] * 100
            return result
        for tk in tickers:
            if tk in close.columns:
                s = close[tk].dropna()
                if len(s) >= 2:
                    result[tk] = (float(s.iloc[-1]) - float(s.iloc[-2])) / float(s.iloc[-2]) * 100
        return result
    except Exception:
        return {}


def _render_portfolio_pulse() -> None:
    from src.data.portfolio_loader import get_portfolio
    port = get_portfolio()
    if not port:
        return

    positions  = port.get("positions", [])
    total_usd  = port.get("total_usd", 0.0)
    n          = port.get("n", 0)
    loaded_at  = port.get("loaded_at", "—")[:10]

    if not positions or total_usd <= 0:
        return

    # Fetch 1-day returns for all tickers
    tickers   = tuple(p["ticker"] for p in positions)
    tk_key    = "|".join(tickers)
    day_ret   = _load_portfolio_returns(tk_key, tickers)

    # Portfolio-level estimated 1d P&L
    port_ret  = sum(p["weight"] * day_ret.get(p["ticker"], 0.0) for p in positions)
    dollar_pl = port_ret / 100 * total_usd

    pl_color  = "#27ae60" if dollar_pl >= 0 else "#c0392b"
    pl_arrow  = "▲" if dollar_pl >= 0 else "▼"
    pl_sign   = "+" if dollar_pl >= 0 else ""

    # Top 3 movers
    movers = sorted(
        [(p, day_ret.get(p["ticker"])) for p in positions if p["ticker"] in day_ret],
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:3]

    def _mover_html(p: dict, ret: float) -> str:
        col   = "#27ae60" if ret >= 0 else "#c0392b"
        arrow = "▲" if ret >= 0 else "▼"
        w_pct = p["weight"] * 100
        return (
            f'<div style="display:flex;align-items:center;gap:6px;'
            f'padding:3px 0;border-bottom:1px solid #0f0f0f">'
            f'<span style="{_M}font-size:11px;font-weight:700;color:{col};min-width:52px">'
            f'{p["ticker"]}</span>'
            f'<span style="{_M}font-size:10px;color:#C8D4E0;flex:1">'
            f'wt {w_pct:.1f}%</span>'
            f'<span style="{_M}font-size:11px;color:{col};font-weight:700">'
            f'{arrow} {abs(ret):.2f}%</span>'
            f'</div>'
        )

    movers_html = "".join(_mover_html(p, r) for p, r in movers) if movers else (
        f'<span style="{_M}font-size:11px;color:#A8B8C8">Returns pending…</span>'
    )

    col_nav, col_movers = st.columns([1, 1], gap="small")

    with col_nav:
        st.markdown(
            f'<div style="background:#080808;border:1px solid #1a1a1a;'
            f'border-top:2px solid {_GOLD};padding:.4rem .65rem">'
            f'<div style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
            f'text-transform:uppercase;color:#A8B8C8;margin-bottom:4px">Portfolio NAV</div>'
            f'<div style="{_M}font-size:1.1rem;font-weight:700;color:{_GOLD};line-height:1.1">'
            f'${total_usd:,.0f}</div>'
            f'<div style="{_M}font-size:10px;color:#C8D4E0;margin-top:2px">'
            f'{n} positions &nbsp;·&nbsp; as of {loaded_at}</div>'
            f'<div style="margin-top:.4rem;padding-top:.35rem;border-top:1px solid #1a1a1a">'
            f'<div style="{_M}font-size:10px;font-weight:700;letter-spacing:.16em;'
            f'text-transform:uppercase;color:#A8B8C8;margin-bottom:2px">Est. 1-Day P&amp;L</div>'
            f'<div style="{_M}font-size:0.95rem;font-weight:700;color:{pl_color}">'
            f'{pl_arrow} {pl_sign}${dollar_pl:,.0f}'
            f'<span style="font-size:11px;margin-left:5px">{pl_sign}{port_ret:.2f}%</span>'
            f'</div></div></div>',
            unsafe_allow_html=True,
        )

    with col_movers:
        st.markdown(
            f'<div style="background:#080808;border:1px solid #1a1a1a;'
            f'border-top:2px solid #2a2a2a;padding:.4rem .65rem">'
            f'<div style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
            f'text-transform:uppercase;color:#A8B8C8;margin-bottom:5px">Top Movers</div>'
            f'{movers_html}</div>',
            unsafe_allow_html=True,
        )


def _render_live_signals() -> None:
    col_straits, col_delta = st.columns([1.3, 1], gap="medium")

    # ── Strait snapshot ────────────────────────────────────────────────────
    with col_straits:
        st.markdown(
            f'<div style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
            f'text-transform:uppercase;color:#DCE4F0;margin-bottom:2px">'
            f'Chokepoint Disruption Snapshot</div>'
            f'<div style="{_F}font-size:12px;color:#C8D4E0;margin-bottom:3px">'
            f'Strait disruption raises energy and food costs within 2–6 weeks. '
            f'Score: conflict intensity × routing dependence.</div>',
            unsafe_allow_html=True,
        )
        scenario = get_scenario()
        tps_mult = scenario.get("tps_mult", 1.0) if scenario.get("label") == "Shipping Shock" else 1.0

        rows_s = ""
        for s in _STRAITS:
            risk = min(int(s["base_risk"] * tps_mult), 100)
            if risk >= 70:   tier, tc = "HIGH",     s["color"]
            elif risk >= 45: tier, tc = "ELEVATED", "#e67e22"
            else:            tier, tc = "MODERATE", "#27ae60"
            pulse = (
                f'<span class="hm-dot" style="background:{tc}"></span>'
                if risk >= 70 else
                f'<span style="display:inline-block;width:7px;height:7px;'
                f'border-radius:50%;background:{tc};vertical-align:middle;'
                f'margin-right:4px;opacity:.7"></span>'
            )
            rows_s += (
                f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;'
                f'border-bottom:1px solid #0d0d0d">'
                + pulse
                + f'<span style="{_M}font-size:11px;font-weight:700;color:{tc};'
                f'min-width:75px;white-space:nowrap">{s["name"][:18]}</span>'
                f'<div style="flex:1;background:#111;height:4px;border-radius:1px">'
                f'<div style="width:{risk}%;height:4px;background:{tc};border-radius:1px"></div></div>'
                f'<span style="{_M}font-size:12px;color:{tc};font-weight:700;min-width:22px;'
                f'text-align:right">{risk}</span>'
                f'<span style="background:{tc};color:#000;{_M}font-size:11px;'
                f'font-weight:700;padding:1px 4px;letter-spacing:.08em;min-width:46px;'
                f'text-align:center">{tier}</span>'
                f'<span style="{_F}font-size:10px;color:#C8D4E0">{s["flow"]}</span>'
                f'</div>'
            )

        st.markdown(
            f'<div style="background:#0d0d0d;border:1px solid #1e1e1e;padding:.4rem .7rem">'
            f'{rows_s}</div>',
            unsafe_allow_html=True,
        )

    # ── What changed ───────────────────────────────────────────────────────
    with col_delta:
        st.markdown(
            f'<div style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
            f'text-transform:uppercase;color:#DCE4F0;margin-bottom:2px">'
            f'What Changed Since Last Run</div>'
            f'<div style="{_F}font-size:12px;color:#C8D4E0;margin-bottom:3px">'
            f'Deltas vs your previous page load. '
            f'First load shows baseline.</div>',
            unsafe_allow_html=True,
        )

        first_run = all(
            st.session_state.get(k) is None
            for k in ("_delta_geo_score", "_delta_cis", "_delta_tps")
        )

        if first_run:
            st.markdown(
                f'<div style="background:#0d0d0d;border:1px solid #1e1e1e;'
                f'padding:.5rem .7rem;{_F}font-size:12px;color:#C8D4E0">'
                f'Establishing baseline — deltas appear from the second visit onward.</div>',
                unsafe_allow_html=True,
            )
        else:
            portfolio_items = [
                ("Geo Risk Score",   st.session_state.get("_delta_geo_score"), "/100"),
                ("Portfolio CIS",    st.session_state.get("_delta_cis"),       "/100"),
                ("Portfolio TPS",    st.session_state.get("_delta_tps"),       "/100"),
            ]
            _WATCH = [
                ("iran_conflict",    "Iran / Hormuz"),
                ("ukraine_russia",   "UA / RU"),
                ("red_sea_houthi",   "Red Sea"),
                ("taiwan_strait",    "Taiwan"),
                ("india_pakistan",   "India / Pak"),
            ]

            port_rows = ""
            for lbl, delta, unit in portfolio_items:
                port_rows += (
                    f'<div style="display:flex;align-items:baseline;gap:6px;'
                    f'padding:3px 0;border-bottom:1px solid #111">'
                    f'<span style="{_F}font-size:12px;color:#D8E0EC;min-width:120px">'
                    f'{lbl}</span>'
                    f'{_delta_html(delta, unit)}</div>'
                )

            conf_rows = ""
            any_c = False
            for cid, display in _WATCH:
                d    = st.session_state.get(f"_delta_cis_{cid}")
                prev = st.session_state.get(f"_stored_cis_{cid}")
                if prev is None:
                    continue
                any_c = True
                conf_rows += (
                    f'<div style="display:flex;align-items:baseline;gap:6px;'
                    f'padding:2px 0 2px 8px;border-left:2px solid #1e1e1e;margin-bottom:2px">'
                    f'<span style="{_M}font-size:10px;color:{_GOLD};min-width:75px">'
                    f'{display}</span>'
                    f'<span style="{_M}font-size:10px;color:#C8D4E0">'
                    f'CIS {prev:.0f}</span>'
                    f'{_delta_html(d, "")}</div>'
                )

            conf_section = (
                f'<div style="{_M}font-size:11px;text-transform:uppercase;letter-spacing:.14em;'
                f'color:#A8B8C8;margin:5px 0 3px">Per-Conflict CIS</div>'
                + conf_rows
                if any_c else ""
            )

            st.markdown(
                f'<div style="background:#0d0d0d;border:1px solid #1e1e1e;'
                f'padding:.4rem .7rem">'
                + port_rows
                + conf_section
                + f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# § 9  AI AGENT ACTIVITY — compact strip (shown only when active)
# ─────────────────────────────────────────────────────────────────────────────

def _render_agent_strip() -> None:
    feed = st.session_state.get("agent_activity", [])[:4]
    if not feed:
        return

    st.markdown(
        f'<div style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#DCE4F0;margin:.6rem 0 .3rem">'
        f'AI Analyst Activity</div>',
        unsafe_allow_html=True,
    )

    rows = ""
    for entry in feed:
        ag     = AGENTS.get(entry["agent_id"], {})
        color  = ag.get("color", "#8E9AAA")
        ts_str = entry["ts"].strftime("%H:%M") if isinstance(entry["ts"], datetime.datetime) else "—"
        rows += (
            f'<div style="display:flex;gap:8px;align-items:baseline;padding:3px 0;'
            f'border-bottom:1px solid #111">'
            f'<span style="{_M}font-size:10px;color:{color};min-width:26px;font-weight:700">'
            f'{ag.get("short","?")}</span>'
            f'<span style="{_F}font-size:12px;color:#D8E0EC;flex:1">'
            f'{entry["action"]}: {entry["detail"][:65]}</span>'
            f'<span style="{_M}font-size:10px;color:#C8D4E0">{ts_str}</span>'
            f'</div>'
        )

    n_pending = pending_count()
    pending_note = (
        f'<div style="{_M}font-size:11px;color:#e67e22;margin-top:4px">'
        f'{n_pending} item{"s" if n_pending != 1 else ""} awaiting human review</div>'
        if n_pending else ""
    )

    st.markdown(
        f'<div style="background:#0d0d0d;border:1px solid #1e1e1e;padding:.4rem .7rem">'
        f'{rows}{pending_note}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────

def page_home(start: str, end: str, fred_key: str = "") -> None:
    st.markdown(_STYLE, unsafe_allow_html=True)
    init_agents()
    init_scenario()

    # ── Load conflict model ────────────────────────────────────────────────
    with st.spinner("Loading conflict model…"):
        try:
            conflict_results = score_all_conflicts()
            conflict_agg     = aggregate_portfolio_scores(conflict_results)
            record_fetch("conflict_model")
        except Exception:
            conflict_results = {}
            conflict_agg = {
                "cis": 50.0, "tps": 50.0,
                "portfolio_cis": 50.0, "portfolio_tps": 50.0,
                "confidence": 0.5, "top_conflict": None,
            }

    # Cache conflict results for n_active count in masthead
    st.session_state["_conflict_results_cache"] = conflict_results

    # Inject n_active into agg for masthead
    conflict_agg["n_active"] = sum(
        1 for r in conflict_results.values() if r.get("state") == "active"
    )

    # ── Full 3-layer risk score + history (market-data-fed, cached) ───────
    with st.spinner("Computing risk score…"):
        risk, _score_hist = _load_market_risk(start, end, get_scenario_id())
        if not risk:
            # Fallback: conflict-model-only estimate when market data unavailable
            cis_f = conflict_agg.get("portfolio_cis", conflict_agg.get("cis", 50.0))
            tps_f = conflict_agg.get("portfolio_tps", conflict_agg.get("tps", 50.0))
            raw   = round(0.40 * cis_f + 0.35 * tps_f + 0.25 * 50, 1)
            if raw < 25:   lbl, col = "Low",      "#2e7d32"
            elif raw < 50: lbl, col = "Moderate", "#8E9AAA"
            elif raw < 75: lbl, col = "Elevated", "#e67e22"
            else:          lbl, col = "High",     "#c0392b"
            risk = {
                "score": raw, "label": lbl, "color": col,
                "cis": cis_f, "tps": tps_f, "mcs": 50.0,
                "confidence": conflict_agg.get("confidence", 0.5),
                "top_conflict": conflict_agg.get("top_conflict"),
                "news_gpr": None, "n_threat": 0, "n_act": 0,
                "mcs_components": {}, "components": {}, "weights": {},
                "conflict_detail": {},
            }

    # ── Populate session-state deltas ──────────────────────────────────────
    _new_cis  = conflict_agg.get("portfolio_cis", conflict_agg.get("cis", 50.0))
    _new_tps  = conflict_agg.get("portfolio_tps", conflict_agg.get("tps", 50.0))
    _new_geo  = float(risk["score"])
    _prev_cis = st.session_state.get("_stored_cis", None)
    _prev_tps = st.session_state.get("_stored_tps", None)
    _prev_geo = st.session_state.get("_stored_geo_score", None)
    st.session_state["_delta_cis"]       = round(_new_cis - _prev_cis, 1) if _prev_cis is not None else None
    st.session_state["_delta_tps"]       = round(_new_tps - _prev_tps, 1) if _prev_tps is not None else None
    st.session_state["_delta_geo_score"] = round(_new_geo - _prev_geo, 1) if _prev_geo is not None else None
    st.session_state["_stored_cis"]      = _new_cis
    st.session_state["_stored_tps"]      = _new_tps
    st.session_state["_stored_geo_score"] = _new_geo

    for _cid, _cr in conflict_results.items():
        _cis_now  = float(_cr.get("cis", 0))
        _cis_prev = st.session_state.get(f"_stored_cis_{_cid}", None)
        st.session_state[f"_delta_cis_{_cid}"]  = (
            round(_cis_now - _cis_prev, 1) if _cis_prev is not None else None
        )
        st.session_state[f"_stored_cis_{_cid}"] = _cis_now

    try:
        from src.analysis.agent_state import pending_count as _pc
        _new_pend = _pc()
        _prev_pend = st.session_state.get("_stored_pending", None)
        st.session_state["_delta_pending"]  = (_new_pend - _prev_pend) if _prev_pend is not None else None
        st.session_state["_stored_pending"] = _new_pend
    except Exception:
        pass

    # ══════════════════════════════════════════════════════════════════════
    # RENDER SECTIONS
    # ══════════════════════════════════════════════════════════════════════

    # § 1  Masthead
    _render_masthead(conflict_agg)

    # § 1.5  Market Pulse — macro KPI strip
    _render_market_pulse()

    # § 1.6  Portfolio Pulse — NAV + 1-day P&L (conditional on upload)
    _render_portfolio_pulse()

    # § 2  Geopolitical Risk Score — DOMINANT ELEMENT
    _render_geo_risk_block(risk, conflict_agg, conflict_results, _score_hist)

    # § 3  Context Narrative
    _render_context_narrative(risk, conflict_results)

    # § 4  Intel Panel
    _render_intel_panel(conflict_results)

    # § 5  Scenario Switch
    _render_scenario_switch()

    # § 6  Where To Go Now
    _render_next_action(conflict_agg, conflict_results)

    # § 7  Navigate Terminal
    st.markdown('<hr class="hm-rule">', unsafe_allow_html=True)
    _render_quickjump()

    # § 8  Live Signals
    st.markdown('<hr class="hm-rule">', unsafe_allow_html=True)
    _render_live_signals()

    # § 9  AI Agent Activity (optional strip)
    _render_agent_strip()

    _page_footer()
