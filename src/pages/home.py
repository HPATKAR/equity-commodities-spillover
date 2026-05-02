"""
Home - Geopolitical & Cross-Asset Intelligence Terminal
Command Center for the Equity-Commodities Spillover Dashboard.

Page hierarchy:
  1.  Masthead         - terminal identity, date, situation state
  2.  Geo Risk Score   - dominant block: score, decomposition, drivers, freshness
  3.  Context Narrative - data-driven plain-language interpretation
  4.  Intel Panel      - conflict table (left) + transmission channels (right)
  5.  Scenario Switch  - compact, integrated lens selector
  6.  Where To Go Now  - live-data-driven recommendations
  7.  Navigate Terminal - grouped quick-jump shortcuts
  8.  Live Signals     - strait snapshot + what-changed delta strip
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
    SCENARIOS, SCENARIO_ORDER, SCENARIO_COMPOUNDS,
    init_scenario, get_scenario, get_scenario_id,
    set_scenario, set_compound_scenario,
)
from src.analysis.conflict_model import score_all_conflicts, aggregate_portfolio_scores, transmission_lag_signal
from src.analysis.freshness import freshness_badge_html, record_fetch
from src.data.loader import load_returns
from src.analysis.correlations import average_cross_corr_series
from src.analysis.risk_score import risk_score_history, plot_risk_history
from src.ui.shared import _page_header, _page_footer

_F    = "font-family:'DM Sans',sans-serif;"
_M    = "font-family:'JetBrains Mono',monospace;"
_GOLD = "#CFB991"


# ─────────────────────────────────────────────────────────────────────────────
# CSS - one block, loaded once
# ─────────────────────────────────────────────────────────────────────────────

_STYLE = """<style>
/*
  Typography system - Command Center
  T1  Section header  Mono 10px 700 uppercase .18em  #DCE4F0
  T2  Panel label     Mono 10px 600 uppercase .12em  #C8D4E0
  T3  Body text       Sans 12px 400            -     #C8D4E0
  T4  Data value      Mono 12px 700            -     (state-colored)
  T5  Caption / meta  Mono 10px 400            -     #A8B8C8
*/
/* T1 - section header */
.hm-label{font-family:'JetBrains Mono',monospace!important;font-size:10px!important;
  font-weight:700!important;text-transform:uppercase;letter-spacing:.18em;
  color:#DCE4F0!important;display:block}
/* T3 - body text */
.hm-sub{font-family:'DM Sans',sans-serif!important;font-size:12px!important;
  color:#C8D4E0!important;display:block;line-height:1.6}
/* T4 - delta values */
.hm-up{font-family:'JetBrains Mono',monospace!important;font-size:12px!important;
  font-weight:700!important;color:#c0392b!important}
.hm-dn{font-family:'JetBrains Mono',monospace!important;font-size:12px!important;
  font-weight:700!important;color:#27ae60!important}
.hm-fl{font-family:'JetBrains Mono',monospace!important;font-size:12px!important;
  font-weight:700!important;color:#DCE4F0!important}
/* ── Section rule ── */
.hm-rule{border:none;border-top:1px solid #1e1e1e;margin:.3rem 0 .25rem}
/* ── Conflict table rows ── */
.hm-crow{display:flex;align-items:center;gap:10px;padding:8px 10px;
  border-bottom:1px solid #111;border-left:3px solid transparent;
  background:#0a0a0a;transition:background .12s ease,border-left-color .12s ease}
.hm-crow:hover{background:#0f0f0f;border-left-color:#CFB991}
/* ── Nav card ── */
.hm-nav{background:#0f0f0f;border:1px solid #1e1e1e;
  padding:12px 14px;margin-bottom:5px;min-height:72px;
  display:flex;flex-direction:column;justify-content:space-between;
  transition:box-shadow .15s ease,border-color .15s ease,background .15s ease}
.hm-nav:hover{box-shadow:0 0 0 1px #CFB991,inset 0 0 14px rgba(207,185,145,.09);
  border-color:#CFB991!important;background:rgba(20,20,20,0.9)!important}
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
.hm-rec{border-left:3px solid;padding:.35rem .8rem;margin-bottom:4px;
  background:#0f0f0f;border-top:1px solid #1e1e1e;border-bottom:1px solid #1e1e1e;
  display:flex;align-items:center;gap:10px;flex-wrap:wrap}
/* ── Pulse dot for high-risk ── */
@keyframes hm-pulse{0%,100%{opacity:1}50%{opacity:.3}}
.hm-dot{display:inline-block;width:7px;height:7px;border-radius:50%;
  animation:hm-pulse 1.8s ease-in-out infinite;vertical-align:middle;margin-right:4px}
/* ── Terminal-style buttons (scenario pills + nav arrows) ── */
[data-testid="stButton"]>button{
  font-family:'JetBrains Mono',monospace!important;
  font-size:9px!important;font-weight:700!important;
  letter-spacing:.05em!important;text-transform:uppercase!important;
  border-radius:1px!important;padding:3px 5px!important;
  height:auto!important;min-height:0!important;line-height:1.5!important;
  white-space:normal!important;overflow-wrap:break-word!important}
[data-testid="stButton"]>button[kind="secondary"]{
  background:transparent!important;border:1px solid #2a2a2a!important;color:#C8D4E0!important}
[data-testid="stButton"]>button[kind="secondary"]:hover{
  border-color:#CFB991!important;color:#CFB991!important;background:rgba(207,185,145,.04)!important}
[data-testid="stButton"]>button[kind="primary"]{
  background:#0f0f0f!important;border:1px solid #CFB991!important;color:#CFB991!important}
[data-testid="stButton"]>button[kind="primary"]:hover{
  background:rgba(207,185,145,.08)!important}
/* ── Reduce top page padding only ── */
.block-container{padding-top:1rem!important}
</style>"""


# ─────────────────────────────────────────────────────────────────────────────
# Score history loader (cached - market data only fetched once per TTL)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def _load_market_risk(start: str, end: str, scenario_id: str = "base") -> tuple[dict, pd.Series]:
    """
    Load market returns, compute avg_corr, then run the full 3-layer risk score.
    Returns (risk_result, score_history).

    risk_result always contains:
      _computed_at  : ISO timestamp string of when this computation ran
      _market_fallback : True when market data was unavailable
      _is_eod       : True when latest data is from a prior close (not intraday)
      _data_date    : str date of the latest available market close
    """
    from src.analysis.risk_score import compute_risk_score
    computed_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        eq_r, cmd_r = load_returns(start, end)
        if eq_r.empty or cmd_r.empty:
            return {
                "_computed_at": computed_at,
                "_market_fallback": True,
                "_is_eod": None,
                "_data_date": None,
            }, pd.Series(dtype=float)

        # Detect whether latest data is a prior close (EOD) or same-day
        today = datetime.date.today()
        last_date = cmd_r.index[-1].date() if hasattr(cmd_r.index[-1], "date") else None
        is_eod = last_date is not None and last_date < today

        avg_corr = average_cross_corr_series(eq_r, cmd_r, window=60)
        risk     = compute_risk_score(avg_corr, cmd_r, eq_r=eq_r)
        hist     = risk_score_history(avg_corr, cmd_r, eq_r=eq_r, window=252)

        risk["_computed_at"]      = computed_at
        risk["_market_fallback"]  = False
        risk["_is_eod"]           = is_eod
        risk["_data_date"]        = str(last_date) if last_date else None

        record_fetch("risk_score")
        return risk, hist

    except Exception:
        return {
            "_computed_at": computed_at,
            "_market_fallback": True,
            "_is_eod": None,
            "_data_date": None,
        }, pd.Series(dtype=float)


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

    # ── Header ────────────────────────────────────────────────────────────
    _page_header(
        "Command Center",
        "Geopolitical & Cross-Asset Intelligence · Equity · Commodity · FX · Fixed Income",
    )

    # ── Status bar ────────────────────────────────────────────────────────
    _sit_rgb = {"#c0392b": "192,57,43", "#e67e22": "230,126,34", "#27ae60": "39,174,96"}.get(sit_color, "207,185,145")
    st.markdown(
        f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;'
        f'border-left:3px solid {sit_color};'
        f'padding:.4rem 1rem;display:flex;align-items:center;gap:16px;'
        f'flex-wrap:wrap;margin-bottom:.5rem">'
        f'<span class="nx-live-dot"></span>'
        f'<span style="{_M}font-size:11px;color:#C8D4E0;letter-spacing:.02em">'
        f'{now.strftime("%a %d %b %Y")}&nbsp;'
        f'<span style="color:#2a2a2a">│</span>&nbsp;'
        f'{now.strftime("%H:%M")} LOCAL'
        f'</span>'
        f'<span style="background:rgba({_sit_rgb},0.15);color:{sit_color};'
        f'border:1px solid rgba({_sit_rgb},0.35);'
        f'{_M}font-size:9px;font-weight:700;padding:2px 9px;letter-spacing:.14em;border-radius:1px">'
        f'■ {n_act} CONFLICT{"S" if n_act != 1 else ""} ACTIVE'
        f'</span>'
        f'<span style="{_M}font-size:10px;color:#A8B8C8">{sc_note}</span>'
        f'<span style="{_M}font-size:10px;color:#A8B8C8">'
        f'CIS&nbsp;<b style="color:{sit_color}">{cis:.0f}</b>'
        f'</span>'
        f'<span style="margin-left:auto;background:{sit_color};color:#000;'
        f'{_M}font-size:9px;font-weight:700;padding:3px 11px;letter-spacing:.16em">'
        f'{sit_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § 2  GEOPOLITICAL RISK SCORE - dominant block
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
    else:                v_col, v_lbl = "#8E9AAA", "- STABLE"
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
        f'<div style="display:flex;align-items:center;gap:7px;padding:5px 0;'
        f'border-bottom:1px solid #111">'
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
    Score readout is a fully separate zone starting 60px below hub bottom -
    zero overlap is geometrically impossible given the layout.

    Geometry:
      hub      → cy=148,  r_hub=16  → bottom edge at y=164
      readout  → starts at y=232    → 68px clear gap (≈ 4× hub diameter)
      viewBox  → 400 × 315
    """
    # ── Layout constants ──────────────────────────────────────────────────
    cx, cy   = 200, 148   # pivot - upper portion of canvas
    R        = 118        # arc centerline radius
    SW       = 32         # arc band stroke-width
    R_IN     = R - SW // 2   # inner edge  = 102
    R_OUT    = R + SW // 2   # outer edge  = 134
    # outer decorative rim sits 6px beyond the arc band
    R_RIM    = R_OUT + 6

    # Readout panel - anchored to absolute y, NOT relative to cy
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
        # Broad glow - for needle + score
        '<filter id="gB" x="-80%" y="-80%" width="260%" height="260%">'
        '<feGaussianBlur stdDeviation="7" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
        '</filter>'
        # Tight glow - for inner detail
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

    # Zone arcs - rich dark fill body + bright inner accent line
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

    # Zone labels - inside band at arc centerline
    for d1, d2, fill, bright, mid, lbl in zones:
        tx, ty = pt(R, mid)
        S.append(
            f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" '
            f'dominant-baseline="middle" font-family="JetBrains Mono,monospace" '
            f'font-size="9" font-weight="700" fill="{bright}" opacity="0.88">{lbl}</text>'
        )

    # Score progress trail - thin glowing arc from 0 → current
    if score > 1:
        S.append(
            f'<path d="{arc(180, sdeg, R_IN+4)}" fill="none" stroke="{color}" '
            f'stroke-width="{SW//2-2}" stroke-linecap="round" opacity="0.09" filter="url(#gT)"/>'
        )
        S.append(
            f'<path d="{arc(180, sdeg, R_IN+4)}" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linecap="round" opacity="0.78"/>'
        )

    # Needle - glow bloom + solid triangle + bright spine
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

    # Hub - three-ring instrument pivot
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

    # SCORE - hero number
    S.append(
        f'<text x="{cx}" y="{TY_SCORE}" text-anchor="middle" '
        f'font-family="JetBrains Mono,monospace" font-size="56" font-weight="700" '
        f'letter-spacing="-2" fill="{color}" filter="url(#gT)">{score:.1f}</text>'
    )
    # /100 suffix
    S.append(
        f'<text x="{cx+68}" y="{TY_SCORE-20}" text-anchor="start" '
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
    top_c     = risk.get("top_conflict") or conflict_agg.get("top_conflict") or "-"
    scenario  = get_scenario()
    geo_mult  = scenario.get("geo_mult", 1.0)
    sc_label  = scenario.get("label", "Base")
    sc_color  = scenario.get("color", _GOLD)
    news_gpr  = risk.get("news_gpr")
    n_threat  = risk.get("n_threat", 0)
    n_act_hl  = risk.get("n_act", 0)

    # ── Freshness: track risk_score, not conflict_model ────────────────────
    is_fallback  = risk.get("_market_fallback", False)
    computed_at  = risk.get("_computed_at", "")
    is_eod       = risk.get("_is_eod", None)
    data_date    = risk.get("_data_date", "")

    from src.analysis.freshness import get_status
    rs_status = get_status("risk_score")
    if is_fallback:
        freshness_color = "#e67e22"
        freshness_text  = f"CONFLICT MODEL ONLY · No market data · {computed_at[11:16] if computed_at else '-'}"
    elif is_eod:
        freshness_color = "#CFB991"
        freshness_text  = f"EOD Close · {data_date} · computed {computed_at[11:16] if computed_at else '-'}"
    else:
        freshness_color = rs_status["color"]
        freshness_text  = f"{'LIVE' if rs_status['status']=='live' else rs_status['label']} · {computed_at[11:16] if computed_at else '-'}"

    freshness = (
        f'<span style="{_M}font-size:9px;color:{freshness_color};letter-spacing:.06em">'
        f'{freshness_text}</span>'
    )

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
    top_ch     = max(ch_scores, key=ch_scores.get) if ch_scores else "-"
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

    top_c_disp  = top_c.replace("_", " ").title() if top_c and top_c != "-" else "-"
    top_ch_disp = top_ch.replace("_", " ").upper() if top_ch != "-" else "-"
    news_gpr_val = f'{news_gpr:.0f}' if news_gpr is not None else '-'
    news_gpr_sub = f'{n_threat}T / {n_act_hl}A' if news_gpr is not None else 'awaiting'

    # ── Panel header - full-width, sits above the two columns ─────────────
    _hdr_col, _btn_col = st.columns([10, 1], gap="small")
    with _hdr_col:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;'
            f'border-top:3px solid {color};border-bottom:1px solid #1e1e1e;'
            f'background:#0f0f0f;padding:.4rem .9rem;margin-bottom:.1rem">'
            f'<span style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
            f'text-transform:uppercase;color:#DCE4F0">Geopolitical Risk Score</span>'
            f'<span style="background:{color};color:#000;{_M}font-size:11px;font-weight:700;'
            f'padding:1px 7px;letter-spacing:.10em">{label.upper()}&nbsp;{score:.1f}</span>'
            f'<span style="{_M}font-size:10px;color:#C8D4E0;margin-left:4px">'
            f'Confidence&nbsp;<b style="color:{conf_color}">{conf:.0%}</b>'
            f'&nbsp;·&nbsp;{spark_html}</span>'
            f'<span style="margin-left:auto">{freshness}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with _btn_col:
        if st.button("↻", key="geo_refresh_btn", help="Force-refresh market data and recompute score"):
            from src.data.loader import load_returns as _lr, load_all_prices as _lap
            _load_market_risk.clear()
            _lr.clear()
            _lap.clear()
            st.rerun()

    # ── Fallback warning - shown only when market data was unavailable ──────
    if is_fallback:
        st.markdown(
            f'<div style="background:#1a0a00;border:1px solid #e67e22;border-left:3px solid #e67e22;'
            f'padding:.3rem .8rem;margin-bottom:.4rem;{_M}font-size:10px;color:#e67e22">'
            f'⚠ MARKET DATA UNAVAILABLE - score is Conflict Model estimate only '
            f'(40% CIS + 35% TPS). MCS layer set to neutral 50. '
            f'Hit ↻ to retry.</div>',
            unsafe_allow_html=True,
        )
    elif is_eod:
        st.markdown(
            f'<div style="background:#0d0d08;border:1px solid #CFB991;border-left:3px solid #CFB991;'
            f'padding:.2rem .8rem;margin-bottom:.4rem;{_M}font-size:10px;color:#CFB991">'
            f'Latest market close: {data_date} · Market Confirmation layer uses prior-day prices.</div>',
            unsafe_allow_html=True,
        )

    # ── Gauge: full-width dominant centerpiece ────────────────────────────
    # Render full-width so it takes the entire center column — prominent by design.
    # History chart stacks below as a compact 220px strip.
    _gauge_col, _gauge_meta = st.columns([1.6, 1.0], gap="small")
    with _gauge_col:
        svg_gauge = _build_speedometer_svg(score, color, label, delta)
        # Override max-height to let the SVG breathe at full column width
        svg_gauge = svg_gauge.replace(
            'style="width:100%;max-height:315px;display:block"',
            'style="width:100%;max-height:400px;display:block"',
        )
        st.markdown(svg_gauge, unsafe_allow_html=True)
        st.markdown(
            f'<p style="{_M}font-size:9px;color:#A8B8C8;margin:.2rem 0 0;'
            f'padding:0 2px">40% CIS&nbsp;·&nbsp;35% TPS&nbsp;·&nbsp;25% MCS</p>',
            unsafe_allow_html=True,
        )
    with _gauge_meta:
        # Score decomposition inline next to gauge
        st.markdown(
            f'<div style="padding:.5rem 0">'
            f'<div style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
            f'text-transform:uppercase;color:#8890a1;margin-bottom:10px">Score Layers</div>',
            unsafe_allow_html=True,
        )
        for _lbl, _val, _col in [
            ("Conflict Intensity", cis, cis_color),
            ("Transmission Press.", tps, tps_color),
            ("Market Confirm.", mcs, mcs_color),
        ]:
            _pct = min(_val, 100)
            st.markdown(
                f'<div style="margin-bottom:10px">'
                f'<div style="display:flex;justify-content:space-between;'
                f'{_F}font-size:11px;margin-bottom:3px">'
                f'<span style="color:#C8D4E0">{_lbl}</span>'
                f'<span style="{_M}font-weight:700;color:{_col}">{_val:.0f}</span>'
                f'</div>'
                f'<div style="height:4px;background:#1a1a1a;border-radius:1px">'
                f'<div style="width:{_pct:.0f}%;height:4px;background:{_col};'
                f'border-radius:1px"></div></div></div>',
                unsafe_allow_html=True,
            )
        # News GPR if available
        if news_gpr is not None:
            st.markdown(
                f'<div style="margin-top:8px;padding-top:8px;border-top:1px solid #1e1e1e">'
                f'<div style="{_M}font-size:8px;color:#8890a1;letter-spacing:.12em;margin-bottom:3px">NEWS GPR</div>'
                f'<div style="{_M}font-size:1.1rem;font-weight:700;color:{_GOLD}">'
                f'{news_gpr:.0f}</div>'
                f'<div style="{_M}font-size:9px;color:#555960">'
                f'{"▲" if n_threat else ""}{n_threat} threat &nbsp;·&nbsp; '
                f'{"▲" if n_act_hl else ""}{n_act_hl} act</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── History chart: compact strip below the gauge ───────────────────────
    st.markdown(
        f'<p style="{_M}font-size:9px;font-weight:700;letter-spacing:.16em;'
        f'text-transform:uppercase;color:#8890a1;margin:.4rem 0 0">'
        f'Historical Risk Score'
        f'<span style="font-weight:400;color:#555960;margin-left:8px">'
        f'corr · oil-gold · vol proxy</span></p>',
        unsafe_allow_html=True,
    )
    if score_history is not None and not score_history.empty:
        fig_hist = plot_risk_history(score_history, height=220)
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#DCE4F0", "family": "JetBrains Mono, monospace", "size": 11},
            title_text="",
            margin=dict(l=44, r=16, t=4, b=24),
            xaxis=dict(
                showgrid=False,
                tickfont={"size": 10, "color": "#C8D4E0", "family": "JetBrains Mono"},
                linecolor="#1e1e1e",
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#1a1a1a",
                tickfont={"size": 10, "color": "#C8D4E0", "family": "JetBrains Mono"},
                linecolor="#1e1e1e",
                title_text="",
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["0", "25", "50", "75", "100"],
            ),
            legend=dict(
                font={"size": 10, "color": "#DCE4F0"},
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
        st.plotly_chart(fig_hist, width="stretch", config={"displayModeBar": False})
    else:
        st.markdown(
            f'<div style="{_F}font-size:12px;color:#A8B8C8;padding:2rem 0;'
            f'text-align:center;border:1px solid #1e1e1e;margin-top:4px">'
            f'No market data available - check connection or date range.</div>',
            unsafe_allow_html=True,
        )

    # ── Sub-component breakdown (when model provides detailed weights) ────────
    comp = risk.get("components", {})
    dw   = risk.get("weights", {})

    if comp:
        # Rich sub-component grid — only when the model produces them
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
                f'<div style="height:{"3" if indent else "4"}px;background:#1a1a1a;border-radius:1px;margin-bottom:5px">'
                f'<div style="width:{pct:.0f}%;height:{"3" if indent else "4"}px;'
                f'background:{c_col};border-radius:1px"></div></div></div>'
            )
        st.markdown(
            f'<div style="border-top:1px solid #1e1e1e;margin-top:.2rem;padding:.25rem .4rem .1rem">'
            f'<p style="{_M}font-size:9px;font-weight:700;letter-spacing:.16em;'
            f'text-transform:uppercase;color:#8890a1;margin:0 0 .35rem">Sub-Components</p>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0 24px">{bars_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if dw:
        wt_items = "".join(
            f'<span style="{_M}font-size:10px;color:#C8D4E0">'
            f'{k.replace("_", " ")}&nbsp;<b style="color:{_GOLD}">{v*100:.0f}%</b></span>'
            for k, v in dw.items()
        )
        st.markdown(
            f'<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;'
            f'padding:.2rem .4rem;border-top:1px solid #1e1e1e;margin-top:.1rem">'
            f'<span style="{_M}font-size:9px;letter-spacing:.14em;text-transform:uppercase;'
            f'color:#555960">Dynamic Weights</span>'
            f'{wt_items}</div>',
            unsafe_allow_html=True,
        )
    # ── Score interpretation note ─────────────────────────────────────────
    # Shown when GRS is materially below the lead conflict's CIS - i.e., when
    # a user might reasonably ask "why is the score low with active wars?"
    # Explains the market-corroboration design without cluttering the main view.
    _max_conflict_cis = max(
        (r["cis"] for r in conflict_results.values()), default=cis
    )
    _n_active = sum(1 for r in conflict_results.values() if r.get("state") == "active")
    if score < _max_conflict_cis - 6 and cis >= 45:
        # Build dynamic explanation based on current values
        _mcs_phrase = (
            "markets have partially priced in these conflicts - equity vol and commodity "
            "moves are elevated vs 2019 but are no longer at acute-crisis z-scores"
            if mcs < 55 else
            "market stress signals are rising - equity vol and safe-haven demand "
            "are beginning to confirm the geopolitical signal"
        )
        _top_cis_conflict = max(conflict_results.items(), key=lambda x: x[1]["cis"])
        _lead_name  = _top_cis_conflict[1]["name"]
        _lead_cis   = _top_cis_conflict[1]["cis"]
        _gap        = _lead_cis - score
        st.markdown(
            f'<div style="background:#0a0a0a;border:1px solid #1e1e1e;border-left:3px solid rgba(207,185,145,.35);'
            f'padding:.35rem .8rem .35rem;margin:.35rem 0 0">'
            f'<div style="display:flex;align-items:baseline;gap:9px;flex-wrap:wrap">'
            f'<span style="{_M}font-size:7px;font-weight:700;letter-spacing:.18em;'
            f'text-transform:uppercase;color:rgba(207,185,145,.45);flex-shrink:0">'
            f'SCORING NOTE</span>'
            f'<span style="{_F}font-size:11px;color:#6E7A8A;line-height:1.58">'
            f'GRS is <b style="color:#8E9AAA">{score:.0f}</b> vs. '
            f'{_lead_name}\'s CIS of <b style="color:#8E9AAA">{_lead_cis:.0f}</b> '
            f'({_gap:.0f}-pt gap) because the score is market-corroborated: '
            f'the MCS layer ({mcs:.0f}/100, 25% weight) anchors to <em>relative</em> '
            f'volatility via EWM z-scores - {_mcs_phrase}. '
            f'Portfolio CIS is also a weighted mean across all {_n_active} active conflicts, '
            f'not the worst-case actor alone. '
            f'The live signal that will push this score higher is PortWatch tanker disruption '
            f'and commodity vol z-scores breaking above 1σ - those feed MCS directly.'
            f'</span></div></div>',
            unsafe_allow_html=True,
        )

    # ── Market-freshness of lead conflict (affects ranking, not raw CIS) ─────
    _lead_mf = 1.0
    for _r in conflict_results.values():
        if _r.get("name") == top_c:
            _lead_mf = float(_r.get("market_freshness", 1.0))
            break
    # Only annotate when the multiplier is meaningfully different from neutral (1.0)
    _lead_mf_sub = (
        f'mkt x{_lead_mf:.2f} - {"boosted" if _lead_mf > 1.05 else "discounted"} by live signals'
        if abs(_lead_mf - 1.0) > 0.05
        else "highest CIS actor"
    )

    # ── KPI strip - CSS grid guarantees equal-width tiles ─────────────────
    def _kt(lbl: str, val: str, vc: str, sub: str = "") -> str:
        return (
            f'<div style="padding:.5rem .75rem;background:#0f0f0f;'
            f'border:1px solid #1e1e1e;border-top:3px solid {vc}">'
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
        + _kt("Lead Conflict",         top_c_disp,         color,     _lead_mf_sub)
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
    Generated from live CIS/TPS/MCS values - not templated filler.
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
            f"with CIS {cis:.0f}/100 - critical conflict intensity - and an open "
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
            f"Conflict intensity is elevated (CIS {cis:.0f} - <b>{top_c}</b> is the "
            f"primary driver), but the transmission channel to asset markets "
            f"is currently suppressed (TPS {tps:.0f}). "
            f"Risk exists but is not yet flowing into prices."
        )
    elif cis < 40 and mcs >= 60:
        s1 = (
            f"Fundamental conflict intensity is moderate (CIS {cis:.0f}), but the "
            f"market confirmation layer (MCS {mcs:.0f}) is registering elevated signals - "
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
            f"(geo multiplier ×{scenario['geo_mult']:.2f}), amplifying all scores - "
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
            f"Score sits in the Elevated band - consistent with past pre-escalation windows. "
            f"Monitor Threat vs Act and Conflict Intel for early signals before scores move further."
        )
    else:
        s2 = (
            f"Risk environment is contained. Standard monitoring posture: "
            f"daily Overview check, weekly Spillover review, "
            f"and Trade Ideas for any regime-matched positioning opportunities."
        )

    st.markdown(
        f'<div style="background:#0a0a0a;border:1px solid #1e1e1e;'
        f'border-left:3px solid #555960;padding:.4rem .9rem;margin-bottom:.4rem">'
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
# § 3.5  RISK BRIEFING PANEL  (right column — Nexus "Risk Officer Briefing")
# ─────────────────────────────────────────────────────────────────────────────

def _render_risk_briefing_panel(
    risk: dict,
    conflict_results: dict,
    alerts: list | None = None,
) -> None:
    """
    Compact right-column panel matching Nexus 'Risk Officer Briefing'.
    Shows: status + top narrative sentence + top 3 critical alerts as feed items
    + live conflict scores + agent activity.
    """
    score    = risk["score"]
    color    = risk["color"]
    label    = risk["label"]
    cis      = risk["cis"]
    tps      = risk["tps"]
    top_c    = (risk.get("top_conflict") or "-").replace("_", " ").title()

    # Severity level for badge
    if score >= 65:
        badge_level, badge_text = "critical", "CRITICAL"
    elif score >= 45:
        badge_level, badge_text = "warning", "ELEVATED"
    else:
        badge_level, badge_text = "nominal", "NOMINAL"

    # Top transmission channel
    ch_scores: dict[str, float] = {}
    for r in conflict_results.values():
        if r.get("state") != "active":
            continue
        w = r["cis"] / 100
        for ch, v in r.get("transmission", {}).items():
            ch_scores[ch] = ch_scores.get(ch, 0.0) + v * w
    top_ch = max(ch_scores, key=ch_scores.get).replace("_", " ") if ch_scores else "—"

    # Header
    st.markdown(
        f'<div class="nx-panel-header">'
        f'<span class="nx-panel-title">&#x26A0; Risk Officer Briefing</span>'
        f'<span class="nx-badge nx-badge-{badge_level}">{badge_text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Situation summary
    st.markdown(
        f'<div style="padding:0.55rem 0;border-bottom:1px solid #111">'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.52rem;'
        f'font-weight:700;letter-spacing:0.14em;text-transform:uppercase;'
        f'color:{color};margin-bottom:4px">GEO RISK SCORE</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:1.60rem;'
        f'font-weight:700;color:{color};line-height:1">{score:.0f}'
        f'<span style="font-size:0.62rem;color:#8890a1">/100 · {label}</span></div>'
        f'<div style="margin-top:6px;display:flex;gap:12px">'
        f'<span style="font-size:0.58rem;font-family:\'JetBrains Mono\',monospace;color:#8890a1">'
        f'CIS <b style="color:#e8e9ed">{cis:.0f}</b></span>'
        f'<span style="font-size:0.58rem;font-family:\'JetBrains Mono\',monospace;color:#8890a1">'
        f'TPS <b style="color:#e8e9ed">{tps:.0f}</b></span>'
        f'<span style="font-size:0.58rem;font-family:\'JetBrains Mono\',monospace;color:#8890a1">'
        f'CH <b style="color:#CFB991">{top_ch[:16]}</b></span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Critical alerts as feed items
    critical_alerts = [a for a in (alerts or []) if getattr(a, "severity", "") == "critical"][:3]
    warning_alerts  = [a for a in (alerts or []) if getattr(a, "severity", "") == "warning"][:2]

    if critical_alerts or warning_alerts:
        st.markdown(
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
            'font-weight:700;letter-spacing:0.14em;text-transform:uppercase;'
            'color:#CFB991;padding:0.45rem 0 0.3rem">Active Alerts</div>',
            unsafe_allow_html=True,
        )
        for a in critical_alerts:
            st.markdown(
                f'<div class="nx-feed-item critical" style="padding:0.4rem 0">'
                f'<div class="nx-feed-item-header">'
                f'<span class="nx-feed-item-type">ALARM</span>'
                f'</div>'
                f'<div class="nx-feed-item-body" style="line-height:1.5">'
                f'{getattr(a, "title", str(a))[:110]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        for a in warning_alerts:
            st.markdown(
                f'<div class="nx-feed-item warning" style="padding:0.4rem 0">'
                f'<div class="nx-feed-item-header">'
                f'<span class="nx-feed-item-type" style="color:#e67e22">WARNING</span>'
                f'</div>'
                f'<div class="nx-feed-item-body" style="line-height:1.5">'
                f'{getattr(a, "title", str(a))[:110]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Top 4 active conflicts as compact intel rows
    active = sorted(
        [(cid, r) for cid, r in conflict_results.items() if r.get("state") == "active"],
        key=lambda x: x[1]["cis"], reverse=True,
    )[:4]
    if active:
        st.markdown(
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
            'font-weight:700;letter-spacing:0.14em;text-transform:uppercase;'
            'color:#CFB991;padding:0.45rem 0 0.3rem">Active Conflicts</div>',
            unsafe_allow_html=True,
        )
        rows_html = ""
        for cid, r in active:
            cis_v = r["cis"]
            bar_c = "#c0392b" if cis_v >= 65 else "#e67e22" if cis_v >= 45 else "#8890a1"
            rows_html += (
                f'<div style="display:flex;align-items:center;gap:6px;'
                f'padding:4px 0;border-bottom:1px solid #111">'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.58rem;'
                f'font-weight:700;color:{r["color"]};min-width:52px;white-space:nowrap">'
                f'{r["label"][:10]}</span>'
                f'<div style="flex:1;background:#111;height:3px;border-radius:1px">'
                f'<div style="width:{cis_v:.0f}%;height:3px;background:{bar_c};border-radius:1px"></div>'
                f'</div>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.58rem;'
                f'font-weight:700;color:{bar_c};min-width:24px;text-align:right">{cis_v:.0f}</span>'
                f'</div>'
            )
        st.markdown(rows_html, unsafe_allow_html=True)

    # Agent activity (last 3 events)
    feed = st.session_state.get("agent_activity", [])[:3]
    if feed:
        st.markdown(
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
            'font-weight:700;letter-spacing:0.14em;text-transform:uppercase;'
            'color:#CFB991;padding:0.45rem 0 0.3rem">AI Analyst Activity</div>',
            unsafe_allow_html=True,
        )
        for entry in feed:
            ag     = AGENTS.get(entry["agent_id"], {})
            ag_col = ag.get("color", "#8890a1")
            ts_str = entry["ts"].strftime("%H:%M") if isinstance(entry["ts"], datetime.datetime) else "-"
            st.markdown(
                f'<div style="display:flex;gap:6px;align-items:flex-start;'
                f'padding:3px 0;border-bottom:1px solid #111">'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.52rem;'
                f'color:{ag_col};font-weight:700;min-width:28px">{ag.get("short","?")}</span>'
                f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.62rem;'
                f'color:#b8b8b8;flex:1;line-height:1.45">'
                f'{entry.get("action","")}: {entry.get("detail","")[:52]}</span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
                f'color:#555960;white-space:nowrap">{ts_str}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Updated timestamp
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
        f'color:#555960;padding-top:0.5rem;border-top:1px solid #111;margin-top:0.4rem">'
        f'Updated: {datetime.datetime.now().strftime("%H:%M:%S")} UTC</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § L  INTELLIGENCE FEED  (left column — Stitch-style severity card feed)
# ─────────────────────────────────────────────────────────────────────────────

def _render_intelligence_feed(
    risk: dict,
    conflict_results: dict,
    alerts: list | None = None,
) -> None:
    """Left column: alert severity cards + active conflicts + agent activity."""
    score = risk["score"]
    color = risk["color"]
    label = risk["label"]

    # Header
    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'padding-bottom:.4rem;border-bottom:1px solid #1e1e1e;margin-bottom:.55rem">'
        f'<span style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#DCE4F0">Intelligence Feed</span>'
        f'<span class="nx-badge nx-badge-live">LIVE</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # GRS score tile
    if score >= 65:   sc_badge, sc_bg = "CRITICAL",  "rgba(192,57,43,0.12)"
    elif score >= 45: sc_badge, sc_bg = "ELEVATED",  "rgba(230,126,34,0.10)"
    else:             sc_badge, sc_bg = "NOMINAL",   "rgba(39,174,96,0.08)"
    st.markdown(
        f'<div style="background:{sc_bg};border:1px solid #1e1e1e;'
        f'border-left:3px solid {color};padding:.45rem .6rem;margin-bottom:.45rem">'
        f'<div style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:{color};margin-bottom:3px">SYSTEMIC MONITOR</div>'
        f'<div style="{_M}font-size:1.5rem;font-weight:700;color:{color};line-height:1">'
        f'{score:.0f}'
        f'<span style="font-size:.55rem;color:#8890a1;margin-left:4px">/100 · {label.upper()}</span>'
        f'</div>'
        f'<div style="{_M}font-size:8px;font-weight:700;background:{color};color:#000;'
        f'display:inline-block;padding:1px 6px;margin-top:4px;letter-spacing:.12em">{sc_badge}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Alert cards (CRITICAL → ELEVATED → NORMAL)
    all_alerts = sorted(
        (alerts or []),
        key=lambda a: 0 if getattr(a, "severity", "") == "critical"
                   else 1 if getattr(a, "severity", "") == "warning" else 2,
    )[:6]

    for a in all_alerts:
        sev = getattr(a, "severity", "warning")
        if sev == "critical":
            border_c, badge_bg, badge_c, badge_lbl = "#c0392b", "rgba(192,57,43,0.12)", "#e05241", "CRITICAL"
        elif sev == "warning":
            border_c, badge_bg, badge_c, badge_lbl = "#e67e22", "rgba(230,126,34,0.10)", "#e8902a", "ELEVATED"
        else:
            border_c, badge_bg, badge_c, badge_lbl = "#8890a1", "rgba(136,144,161,0.08)", "#8890a1", "NORMAL"

        title  = getattr(a, "title",  str(a))[:68]
        detail = (getattr(a, "detail", None) or getattr(a, "message", ""))[:115]
        ts_str = datetime.datetime.now().strftime("%H:%M")
        st.markdown(
            f'<div style="border-left:2px solid {border_c};background:#0f0f0f;'
            f'border-top:1px solid #1e1e1e;border-right:1px solid #1e1e1e;'
            f'border-bottom:1px solid #1e1e1e;padding:.4rem .55rem;margin-bottom:.3rem">'
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;margin-bottom:4px">'
            f'<span style="{_M}font-size:8px;font-weight:700;letter-spacing:.14em;'
            f'background:{badge_bg};color:{badge_c};padding:1px 5px">{badge_lbl}</span>'
            f'<span style="{_M}font-size:9px;color:#555960">{ts_str}</span>'
            f'</div>'
            f'<div style="{_M}font-size:11px;font-weight:700;color:#e8e9ed;'
            f'line-height:1.3;margin-bottom:3px">{title}</div>'
            + (f'<div style="{_F}font-size:11px;color:#8890a1;line-height:1.45">{detail}</div>' if detail else "")
            + f'</div>',
            unsafe_allow_html=True,
        )

    if not all_alerts:
        st.markdown(
            f'<div style="border-left:2px solid #8890a1;background:#0f0f0f;'
            f'border:1px solid #1e1e1e;padding:.4rem .55rem;margin-bottom:.3rem">'
            f'<span style="{_M}font-size:8px;font-weight:700;color:#8890a1;letter-spacing:.14em">NOMINAL</span>'
            f'<div style="{_F}font-size:11px;color:#8890a1;margin-top:3px">No active alerts</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Active conflicts
    active = sorted(
        [(cid, r) for cid, r in conflict_results.items() if r.get("state") == "active"],
        key=lambda x: x[1]["cis"], reverse=True,
    )[:4]
    if active:
        st.markdown(
            f'<div style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
            f'text-transform:uppercase;color:#CFB991;padding:.4rem 0 .3rem;'
            f'margin-top:.3rem;border-top:1px solid #1e1e1e">Active Conflicts</div>',
            unsafe_allow_html=True,
        )
        for cid, r in active:
            cv    = r["cis"]
            bar_c = "#c0392b" if cv >= 65 else "#e67e22" if cv >= 45 else "#8890a1"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:6px;padding:5px .55rem;'
                f'border-left:2px solid {bar_c};background:#0f0f0f;'
                f'border-bottom:1px solid #1a1a1a;margin-bottom:2px">'
                f'<span style="{_M}font-size:10px;font-weight:700;color:{r["color"]};'
                f'min-width:52px;white-space:nowrap">{r["label"][:10]}</span>'
                f'<div style="flex:1;background:#1a1a1a;height:3px;border-radius:1px">'
                f'<div style="width:{min(cv,100):.0f}%;height:3px;background:{bar_c}"></div></div>'
                f'<span style="{_M}font-size:10px;font-weight:700;color:{bar_c};'
                f'min-width:22px;text-align:right">{cv:.0f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Agent activity
    feed = st.session_state.get("agent_activity", [])[:3]
    if feed:
        st.markdown(
            f'<div style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
            f'text-transform:uppercase;color:#CFB991;padding:.4rem 0 .3rem;'
            f'margin-top:.3rem;border-top:1px solid #1e1e1e">AI Analyst</div>',
            unsafe_allow_html=True,
        )
        for entry in feed:
            ag     = AGENTS.get(entry["agent_id"], {})
            ag_col = ag.get("color", "#8890a1")
            ts_str = entry["ts"].strftime("%H:%M") if isinstance(entry["ts"], datetime.datetime) else "-"
            st.markdown(
                f'<div style="display:flex;gap:6px;align-items:flex-start;'
                f'padding:4px 0;border-bottom:1px solid #111">'
                f'<span style="{_M}font-size:.52rem;color:{ag_col};font-weight:700;min-width:28px">'
                f'{ag.get("short","?")}</span>'
                f'<span style="{_F}font-size:.62rem;color:#b8b8b8;flex:1;line-height:1.45">'
                f'{entry.get("action","")}: {entry.get("detail","")[:52]}</span>'
                f'<span style="{_M}font-size:.50rem;color:#555960;white-space:nowrap">{ts_str}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Chokepoint Strait Watch (fills remaining left-col space) ─────────────
    scenario = get_scenario()
    tps_mult = scenario.get("tps_mult", 1.0) if "shipping" in scenario.get("id", "") else 1.0
    st.markdown(
        f'<div style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#CFB991;padding:.4rem 0 .3rem;'
        f'margin-top:.35rem;border-top:1px solid #1e1e1e">Chokepoint Watch</div>',
        unsafe_allow_html=True,
    )
    strait_rows = ""
    for s in _STRAITS:
        s_risk = min(int(s["base_risk"] * tps_mult), 100)
        tc     = "#c0392b" if s_risk >= 70 else "#e67e22" if s_risk >= 45 else "#27ae60"
        tier   = "HIGH" if s_risk >= 70 else "ELEV" if s_risk >= 45 else "MOD"
        pulse_dot = (
            f'<span class="hm-dot" style="background:{tc};flex-shrink:0"></span>'
            if s_risk >= 70 else
            f'<span style="display:inline-block;width:6px;height:6px;border-radius:50%;'
            f'background:{tc};opacity:.7;flex-shrink:0"></span>'
        )
        strait_rows += (
            f'<div style="display:flex;align-items:center;gap:6px;padding:4px 0;'
            f'border-bottom:1px solid #111">'
            f'{pulse_dot}'
            f'<span style="{_M}font-size:10px;font-weight:700;color:{tc};min-width:66px;'
            f'white-space:nowrap">{s["name"][:16]}</span>'
            f'<div style="flex:1;background:#1a1a1a;height:3px;border-radius:1px">'
            f'<div style="width:{s_risk}%;height:3px;background:{tc};border-radius:1px"></div></div>'
            f'<span style="{_M}font-size:10px;color:{tc};font-weight:700;min-width:20px;'
            f'text-align:right">{s_risk}</span>'
            f'<span style="background:{tc};color:#000;{_M}font-size:8px;font-weight:700;'
            f'padding:1px 4px;letter-spacing:.06em;min-width:32px;text-align:center">{tier}</span>'
            f'</div>'
        )
    st.markdown(
        f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;'
        f'padding:.35rem .55rem">{strait_rows}</div>',
        unsafe_allow_html=True,
    )

    # ── What-changed delta strip ──────────────────────────────────────────────
    _dg  = st.session_state.get("_delta_geo_score")
    _dc  = st.session_state.get("_delta_cis")
    _dt  = st.session_state.get("_delta_tps")
    if any(v is not None for v in [_dg, _dc, _dt]):
        def _dchip(label, delta):
            if delta is None:
                return f'<span style="{_M}font-size:9px;color:#555960">{label}&nbsp;—</span>'
            col  = "#c0392b" if delta > 0.3 else "#27ae60" if delta < -0.3 else "#555960"
            sign = f"▲ +{delta:.1f}" if delta > 0.3 else f"▼ {delta:.1f}" if delta < -0.3 else "— flat"
            return (
                f'<span style="{_M}font-size:9px;color:#555960">{label}&nbsp;</span>'
                f'<span style="{_M}font-size:9px;font-weight:700;color:{col}">{sign}</span>'
            )
        st.markdown(
            f'<div style="margin-top:.35rem;padding:.3rem .55rem;background:#0f0f0f;'
            f'border:1px solid #1e1e1e;display:flex;gap:12px;flex-wrap:wrap;align-items:center">'
            f'<span style="{_M}font-size:8px;letter-spacing:.14em;text-transform:uppercase;'
            f'color:#555960">Δ vs last load</span>'
            f'{_dchip("GRS", _dg)}'
            f'{_dchip("CIS", _dc)}'
            f'{_dchip("TPS", _dt)}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="{_M}font-size:.50rem;color:#555960;padding-top:.45rem;'
        f'border-top:1px solid #111;margin-top:.4rem">'
        f'Updated {datetime.datetime.now().strftime("%H:%M:%S")} UTC</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § R  MARKET PULSE CARDS  (right column — Stitch vertical card layout)
# ─────────────────────────────────────────────────────────────────────────────

def _render_market_pulse_cards() -> None:
    """Right column: market instruments as vertical stacked cards."""
    data = _load_market_pulse()
    if not data:
        return

    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'padding-bottom:.4rem;border-bottom:1px solid #1e1e1e;margin-bottom:.55rem">'
        f'<span style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#DCE4F0">Market Pulse</span>'
        f'<span style="{_M}font-size:9px;color:#8890a1;letter-spacing:.10em">REAL-TIME</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    for d in data:
        pct    = d["pct"]
        is_vix = d["sym"] == "^VIX"
        if abs(pct) < 0.05:
            c, arrow = "#555960", "—"
        elif pct > 0:
            c, arrow = ("#c0392b" if is_vix else "#27ae60"), "▲"
        else:
            c, arrow = ("#27ae60" if is_vix else "#c0392b"), "▼"

        val_fmt = f'{d["val"]:.2f}' if d["val"] < 10000 else f'{d["val"]:,.0f}'
        chg_fmt = f'{arrow} {abs(pct):.2f}%'

        # 5-day sparkline SVG
        series = d.get("series", [])
        spark_svg = ""
        if len(series) >= 3:
            mn, mx = min(series), max(series)
            span   = (mx - mn) or 1.0
            W, H   = 80, 26
            pts    = " ".join(
                f'{i/(len(series)-1)*W:.1f},{H - (v-mn)/span*H:.1f}'
                for i, v in enumerate(series)
            )
            spark_svg = (
                f'<svg width="{W}" height="{H}" style="display:block;margin-top:5px">'
                f'<polyline points="{pts}" fill="none" stroke="{c}" '
                f'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.8"/>'
                f'</svg>'
            )

        # 5d high/low range
        rng_html = ""
        if len(series) >= 2:
            lo, hi = min(series), max(series)
            lo_fmt = f'{lo:.2f}' if lo < 10000 else f'{lo:,.0f}'
            hi_fmt = f'{hi:.2f}' if hi < 10000 else f'{hi:,.0f}'
            rng_html = (
                f'<div style="{_M}font-size:9px;color:#555960;margin-top:3px">'
                f'5d&nbsp;&nbsp;L&nbsp;<b style="color:#8890a1">{lo_fmt}</b>'
                f'&nbsp;·&nbsp;H&nbsp;<b style="color:#8890a1">{hi_fmt}</b></div>'
            )

        st.markdown(
            f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;'
            f'border-left:3px solid {c};padding:.5rem .7rem;margin-bottom:.3rem">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<span style="{_M}font-size:9px;font-weight:700;letter-spacing:.14em;'
            f'text-transform:uppercase;color:#A8B8C8">{d["label"]}</span>'
            f'<span style="{_M}font-size:10px;font-weight:700;color:{c}">{chg_fmt}</span>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-end">'
            f'<div>'
            f'<div style="{_M}font-size:1.2rem;font-weight:700;color:#e8e9ed;line-height:1.1;margin-top:4px">'
            f'{val_fmt}<span style="font-size:.6rem;color:#555960;margin-left:3px">{d["suffix"]}</span></div>'
            f'{rng_html}'
            f'</div>'
            f'{spark_svg}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# § 4  INTEL PANEL - conflict table (left) + channels (right)
# ─────────────────────────────────────────────────────────────────────────────

def _render_intel_panel(conflict_results: dict) -> None:
    n_active = sum(1 for r in conflict_results.values() if r.get("state") == "active")
    st.markdown(
        f'<div class="nx-panel-header" style="margin-bottom:0.5rem">'
        f'<span class="nx-panel-title">&#x2318; Intelligence Panel</span>'
        f'<span style="display:flex;align-items:center;gap:8px">'
        f'<span class="nx-badge nx-badge-critical">{n_active} ACTIVE CONFLICTS</span>'
        f'<span class="nx-panel-meta">CIS · TPS · Transmission Channels</span>'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1.6, 1], gap="medium")

    # ── Left: Conflict table ───────────────────────────────────────────────
    with col_left:
        _TREND_I = {"rising": "▲", "stable": "-", "falling": "▼"}
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
            top_ch  = max(tx, key=tx.get) if tx else "-"
            t_icon  = _TREND_I.get(r.get("trend", "stable"), "-")
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
            f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;padding:.5rem .8rem">'
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
                f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;padding:.5rem .8rem">'
                f'<p style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
                f'text-transform:uppercase;color:#DCE4F0;margin:0 0 .35rem">'
                f'Top Transmission Channels</p>'
                f'{rows_ch}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;padding:.5rem .8rem">'
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


def _render_scenario_switch(narrow: bool = False) -> None:
    current_sid = get_scenario_id()
    current_def = get_scenario()  # handles both single and compound
    is_compound = "+" in current_sid
    impact      = (_IMPACT.get(current_sid, "") if not is_compound
                   else f"geo ×{current_def.get('geo_mult', 1):.2f} · vol ×{current_def.get('vol_mult', 1):.2f}")
    _compound_badge = (
        f'<span style="{_M}font-size:8px;color:#c0392b;letter-spacing:.1em">COMPOUND</span>'
        if is_compound else ""
    )

    st.markdown(
        f'<div style="background:#0a0a0a;border:1px solid #1e1e1e;'
        f'padding:.25rem .7rem;margin:.2rem 0">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:5px;flex-wrap:wrap">'
        f'<span style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#DCE4F0">Scenario Lens</span>'
        f'<span style="background:{current_def["color"]};color:#000;'
        f'{_M}font-size:10px;font-weight:700;padding:1px 6px;letter-spacing:.10em">'
        f'{current_def["label"].upper()}</span>'
        f'{_compound_badge}'
        f'<span style="{_F}font-size:12px;color:#D8E0EC">{current_def["desc"]}</span>'
        f'<span style="{_M}font-size:11px;color:#C8D4E0;margin-left:auto">{impact}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # In narrow mode (left column) render as 4-wide grid (2 rows of 4)
    _ncols = 4 if narrow else len(SCENARIO_ORDER)
    cols = st.columns(_ncols)
    for i, sid in enumerate(SCENARIO_ORDER):
        sdef   = SCENARIOS[sid]
        active = (not is_compound and sid == current_sid)
        if cols[i % _ncols].button(
            sdef["label"],
            key=f"scen_{sid}",
            type="primary" if active else "secondary",
            width="stretch",
        ):
            set_scenario(sid)
            st.rerun()

    # Compound scenario presets (GAP 14 fix)
    with st.expander("Compound Scenarios - stack multiple shocks simultaneously", expanded=False):
        st.markdown(
            f'<p style="{_F}font-size:0.60rem;color:#8E9AAA;margin:0 0 6px">'
            f'Compound scenarios multiply their multipliers (geo, vol, CIS, TPS) - '
            f'enabling realistic combined shocks like Iran military escalation AND Hormuz blockade.</p>',
            unsafe_allow_html=True,
        )
        _cmp_cols = st.columns(min(len(SCENARIO_COMPOUNDS), 2))
        for _ci, _cmp in enumerate(SCENARIO_COMPOUNDS):
            _col = _cmp_cols[_ci % 2]
            _active_cmp = is_compound and set(current_sid.split("+")) == set(_cmp["scenarios"])
            if _col.button(
                _cmp["label"],
                key=f"cmp_{'_'.join(_cmp['scenarios'])}",
                type="primary" if _active_cmp else "secondary",
                width="stretch",
                help=_cmp["desc"],
            ):
                set_compound_scenario(_cmp["scenarios"])
                st.rerun()
        if is_compound:
            if st.button("Clear compound → back to Base", key="cmp_clear", width="content"):
                set_scenario("base")
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# § 6  WHERE TO GO NOW - live-data-driven recommendations
# ─────────────────────────────────────────────────────────────────────────────

def _render_next_action(conflict_agg: dict, conflict_results: dict, compact: bool = False) -> None:
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
                "Risk environment is contained. Start with <b>Overview</b> - "
                "regime classification, correlation heatmap, and the AI analyst morning briefing."
            ),
        })

    recs = recs[:3]

    st.markdown(
        f'<div style="{_M}font-size:10px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#DCE4F0;margin:.2rem 0 .3rem">'
        f'Where To Go Now</div>',
        unsafe_allow_html=True,
    )

    for r in recs:
        if compact:
            # Stacked card format for narrow columns
            st.markdown(
                f'<div style="border-left:3px solid {r["color"]};background:#0f0f0f;'
                f'border-top:1px solid #1e1e1e;border-right:1px solid #1e1e1e;'
                f'border-bottom:1px solid #1e1e1e;padding:.4rem .55rem;margin-bottom:.3rem">'
                f'<div style="display:flex;align-items:center;gap:5px;margin-bottom:3px">'
                f'<span style="{_M}font-size:8px;font-weight:700;letter-spacing:.12em;'
                f'color:{r["color"]}">{r["tag"]}</span>'
                f'</div>'
                f'<div style="{_M}font-size:11px;font-weight:700;color:#e8e9ed;margin-bottom:3px">'
                f'{r["label"]}</div>'
                f'<div style="{_F}font-size:11px;color:#8890a1;line-height:1.4">{r["text"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
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
# § 7  NAVIGATE TERMINAL - grouped quick-jump
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
    # Flatten all groups into a single 4-col grid — no separate per-group st.columns calls
    # Group identity lives in the card's top-border color; rendered as one pass.
    all_items: list[tuple[str, str, str, str, str, str]] = []  # label, page_id, desc, tag, _sc, g_color
    for group in _JUMP_GROUPS:
        for item in group["items"]:
            all_items.append((*item, group["color"]))

    # Legend strip (single markdown)
    st.markdown(
        f'<div style="display:flex;gap:10px;align-items:center;'
        f'padding:.2rem 0 .35rem;flex-wrap:wrap;border-bottom:1px solid #1e1e1e;margin-bottom:.3rem">'
        + "".join(
            f'<span style="display:flex;align-items:center;gap:5px">'
            f'<span style="width:8px;height:8px;border-radius:50%;background:{g["color"]};'
            f'flex-shrink:0;display:inline-block"></span>'
            f'<span style="{_M}font-size:8px;font-weight:700;letter-spacing:.12em;'
            f'text-transform:uppercase;color:{g["color"]}">{g["group"]}</span>'
            f'</span>'
            for g in _JUMP_GROUPS
        )
        + f'<span style="margin-left:auto;display:flex;gap:6px;align-items:center">'
        f'<span class="hm-tag hm-tag-daily">DAILY</span>'
        f'<span class="hm-tag hm-tag-alert">ON ALERT</span>'
        f'<span class="hm-tag hm-tag-deep">DEEP-DIVE</span>'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    # One flat 4-col grid for all 14 modules
    cols = st.columns(4, gap="small")
    for i, (label, page_id, desc, tag, _sc, g_color) in enumerate(all_items):
        tag_label, tag_cls = _TAG_META.get(tag, ("", ""))
        with cols[i % 4]:
            st.markdown(
                f'<div class="hm-nav" style="border-top:2px solid {g_color};min-height:58px">'
                f'<div style="display:flex;align-items:center;gap:4px;margin-bottom:3px">'
                f'<span style="{_M}font-size:11px;font-weight:700;color:{g_color}">{label}</span>'
                f'<span class="hm-tag {tag_cls}" style="font-size:7px!important">{tag_label}</span>'
                f'</div>'
                f'<div style="{_F}font-size:11px;color:#8890a1;line-height:1.35">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("→", key=f"qj_{page_id}", width="stretch"):
                st.query_params["page"] = page_id
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# § 8  LIVE SIGNALS - strait snapshot + what-changed delta
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
        return f'<span style="{_M}font-size:11px;color:#A8B8C8">- first run</span>'
    if abs(delta) < 0.3:
        return f'<span class="hm-fl">- flat</span>'
    if delta > 0:
        return f'<span class="hm-up">▲ +{delta:.1f}{unit}</span>'
    return f'<span class="hm-dn">▼ {delta:.1f}{unit}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# § 1.5  MARKET PULSE - macro KPI strip
# ─────────────────────────────────────────────────────────────────────────────

_PULSE_TICKERS = [
    ("^VIX",      "VIX",     "",   False),   # label, suffix, invert_color
    ("DX-Y.NYB",  "DXY",     "",   False),
    ("^GSPC",     "S&P 500", "",   False),
    ("CL=F",      "WTI",     "/b", False),
    ("GC=F",      "Gold",    "/oz",False),
    ("^TNX",      "10Y Yld", "%",  False),
]


@st.cache_data(ttl=900, show_spinner=False)
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
                "series": [float(v) for v in s.tolist()],  # up to 5 days for sparkline
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
            c, arrow = "#555960", "-"
        elif pct > 0:
            c, arrow = ("#c0392b" if is_vix else "#27ae60"), "▲"
        else:
            c, arrow = ("#27ae60" if is_vix else "#c0392b"), "▼"

        val_fmt = f'{d["val"]:.2f}' if d["val"] < 10000 else f'{d["val"]:,.0f}'
        chg_fmt = f'{arrow} {abs(pct):.2f}%'
        return (
            f'<div style="flex:1;min-width:90px;padding:.65rem .85rem;background:#0f0f0f;'
            f'border:1px solid #1e1e1e;border-top:3px solid {c};'
            f'transition:background .12s ease">'
            f'<div style="{_M}font-size:9px;font-weight:700;letter-spacing:.18em;'
            f'text-transform:uppercase;color:#8890a1;margin-bottom:5px">{d["label"]}</div>'
            f'<div style="{_M}font-size:16px;font-weight:700;color:#e8e9ed;line-height:1.1">'
            f'{val_fmt}<span style="font-size:10px;color:#8890a1">{d["suffix"]}</span></div>'
            f'<div style="{_M}font-size:10px;color:{c};font-weight:700;margin-top:4px">{chg_fmt}</div>'
            f'</div>'
        )

    tiles_html = "".join(_tile(d) for d in data)
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">'
        f'<span style="{_M}font-size:9px;font-weight:700;letter-spacing:.20em;'
        f'text-transform:uppercase;color:#8890a1">Market Pulse</span>'
        f'<span class="nx-badge nx-badge-live">LIVE</span>'
        f'</div>'
        f'<div style="display:flex;gap:5px;flex-wrap:nowrap;margin-bottom:.5rem">'
        f'{tiles_html}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § 1.6  PORTFOLIO PULSE - NAV + 1-day P&L + top movers (conditional)
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
    loaded_at  = port.get("loaded_at", "-")[:10]

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
            f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;'
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
            f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;'
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
                f'<div style="background:#0a0a0a;border:1px solid #1e1e1e;'
                f'padding:.5rem .7rem;{_F}font-size:12px;color:#C8D4E0">'
                f'Establishing baseline - deltas appear from the second visit onward.</div>',
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
                f'<div style="background:#0a0a0a;border:1px solid #1e1e1e;'
                f'padding:.4rem .7rem">'
                + port_rows
                + conf_section
                + f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# § 9  AI AGENT ACTIVITY - compact strip (shown only when active)
# ─────────────────────────────────────────────────────────────────────────────

def _render_agent_strip() -> None:
    feed = st.session_state.get("agent_activity", [])[:5]
    if not feed:
        return

    n_pending = pending_count()
    pending_badge = (
        f'<span class="nx-badge nx-badge-warning">{n_pending} PENDING</span>'
        if n_pending else
        '<span class="nx-live-dot"></span><span style="font-family:\'JetBrains Mono\',monospace;'
        'font-size:0.50rem;color:#27ae60;letter-spacing:0.10em">ALL PROCESSED</span>'
    )

    rows_html = ""
    for entry in feed:
        ag     = AGENTS.get(entry["agent_id"], {})
        color  = ag.get("color", "#8E9AAA")
        ts_str = entry["ts"].strftime("%H:%M:%S") if isinstance(entry["ts"], datetime.datetime) else "-"
        action = entry.get("action", "")
        detail = entry.get("detail", "")[:72]
        rows_html += (
            f'<div class="nx-intel-row">'
            f'<span class="nx-intel-ts">{ts_str}</span>'
            f'<span class="nx-intel-entity" style="color:{color}">{ag.get("short","?")}</span>'
            f'<span class="nx-intel-condition">{action}: {detail}</span>'
            f'</div>'
        )

    st.markdown(
        f'<div class="nx-feed-panel" style="margin-top:0.6rem">'
        f'<div class="nx-feed-panel-header">'
        f'<span class="nx-panel-title">AI Analyst Activity</span>'
        f'<span style="display:flex;align-items:center;gap:6px">{pending_badge}</span>'
        f'</div>'
        f'<div style="padding:0 0.85rem 0.4rem">{rows_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § L2  CORRELATION PULSE  (left column — equity-commodity sparkline)
# ─────────────────────────────────────────────────────────────────────────────

def _render_correlation_pulse(
    corr_series: "pd.Series | None",
    regimes: "pd.Series | None",
) -> None:
    """Left column filler: 60-day equity-commodity correlation sparkline + regime."""
    st.markdown(
        f'<div style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#CFB991;padding:.4rem 0 .3rem;'
        f'margin-top:.4rem;border-top:1px solid #1e1e1e">Correlation Pulse</div>',
        unsafe_allow_html=True,
    )

    if corr_series is None or len(corr_series.dropna()) < 5:
        st.markdown(
            f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;'
            f'padding:.35rem .55rem;{_M}font-size:9px;color:#555960">'
            f'Correlation data unavailable</div>',
            unsafe_allow_html=True,
        )
        return

    series = corr_series.dropna().iloc[-60:].tolist()
    cur    = series[-1]

    if regimes is not None and not regimes.empty:
        r_val = int(regimes.dropna().iloc[-1])
        if r_val == 3:   regime_lbl, regime_c = "HIGH COUPLING", "#c0392b"
        elif r_val == 1: regime_lbl, regime_c = "DECOUPLED",     "#27ae60"
        else:            regime_lbl, regime_c = "TRANSITIONING", "#e67e22"
    else:
        regime_lbl, regime_c = "COMPUTING", "#8890a1"

    cur_c    = "#c0392b" if cur > 0.35 else "#27ae60" if cur < 0.0 else "#e67e22"
    cur_sign = "+" if cur >= 0 else ""

    W, H  = 200, 52
    mn, mx = -1.0, 1.0
    span   = 2.0
    n      = len(series)

    def _fy(v: float) -> float:
        return H - max(0.0, min(float(H), (v - mn) / span * H))

    zero_y = _fy(0.0)
    thr_y  = _fy(0.35)
    pts    = " ".join(
        f'{i / max(n - 1, 1) * W:.1f},{_fy(v):.1f}'
        for i, v in enumerate(series)
    )
    dot_y = _fy(cur)

    svg = (
        f'<svg width="{W}" height="{H + 12}" viewBox="0 0 {W} {H + 12}" '
        f'style="display:block;overflow:visible">'
        # shaded coupling zone (above 0.35 threshold)
        f'<rect x="0" y="0" width="{W}" height="{thr_y:.1f}" '
        f'fill="rgba(192,57,43,0.07)"/>'
        # shaded decoupled zone (below 0)
        f'<rect x="0" y="{zero_y:.1f}" width="{W}" height="{H - zero_y:.1f}" '
        f'fill="rgba(39,174,96,0.05)"/>'
        # zero reference line
        f'<line x1="0" y1="{zero_y:.1f}" x2="{W}" y2="{zero_y:.1f}" '
        f'stroke="#2e2e2e" stroke-width="1"/>'
        # 0.35 high-coupling threshold
        f'<line x1="0" y1="{thr_y:.1f}" x2="{W}" y2="{thr_y:.1f}" '
        f'stroke="#c0392b" stroke-width="0.8" stroke-dasharray="3,3" opacity="0.45"/>'
        # correlation sparkline
        f'<polyline points="{pts}" fill="none" stroke="{cur_c}" '
        f'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" opacity="0.95"/>'
        # terminal dot
        f'<circle cx="{W:.1f}" cy="{dot_y:.1f}" r="3.5" fill="{cur_c}"/>'
        # axis labels
        f'<text x="0" y="{H + 11}" font-size="8" fill="#333333" '
        f'font-family="JetBrains Mono,monospace">60d ago</text>'
        f'<text x="{W}" y="{H + 11}" font-size="8" fill="#333333" '
        f'font-family="JetBrains Mono,monospace" text-anchor="end">now</text>'
        # zero label
        f'<text x="{W + 3}" y="{zero_y + 3:.1f}" font-size="7" fill="#2e2e2e" '
        f'font-family="JetBrains Mono,monospace">0</text>'
        f'</svg>'
    )

    st.markdown(
        f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;'
        f'border-left:3px solid {cur_c};padding:.5rem .65rem">'
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:center;margin-bottom:6px">'
        f'<div>'
        f'<span style="{_M}font-size:1.05rem;font-weight:700;color:{cur_c};line-height:1">'
        f'{cur_sign}{cur:.3f}</span>'
        f'<span style="{_M}font-size:8px;color:#555960;margin-left:5px">eq↔cmd 60d</span>'
        f'</div>'
        f'<span style="background:{regime_c}1a;border:1px solid {regime_c}44;'
        f'{_M}font-size:7px;font-weight:700;letter-spacing:.10em;color:{regime_c};'
        f'padding:1px 5px;text-transform:uppercase">{regime_lbl}</span>'
        f'</div>'
        f'<div style="padding-bottom:2px">{svg}</div>'
        f'<div style="{_M}font-size:8px;color:#333;margin-top:4px">'
        f'<span style="color:#c0392b22;border-bottom:1px dashed #c0392b55">·</span>'
        f'&nbsp;<span style="color:#555960">0.35 coupling threshold</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § R2  RISK DECOMPOSITION  (right column — component meter bars)
# ─────────────────────────────────────────────────────────────────────────────

def _render_risk_arc(risk: dict) -> None:
    """Right column filler: GRS score breakdown by component (CIS / TPS / MCS)."""
    cis   = float(risk.get("cis",   50.0))
    tps   = float(risk.get("tps",   50.0))
    mcs   = float(risk.get("mcs",   50.0))
    score = float(risk.get("score", 50.0))

    weights = risk.get("weights") or {}
    w_cis = float(weights.get("cis", 0.40))
    w_tps = float(weights.get("tps", 0.35))
    w_mcs = float(weights.get("mcs", 0.25))

    def _cc(v: float) -> str:
        return "#c0392b" if v >= 65 else "#e67e22" if v >= 45 else "#27ae60"

    def _cl(v: float) -> str:
        return "HIGH" if v >= 65 else "ELEV" if v >= 45 else "NOM"

    def _meter(label: str, wpct: int, val: float, contrib: float) -> str:
        c   = _cc(val)
        lbl = _cl(val)
        bw  = min(int(val), 100)
        # SVG sparkline-style bar with gradient fill
        bar_svg = (
            f'<svg width="100%" height="5" style="display:block;border-radius:2px;overflow:hidden">'
            f'<defs><linearGradient id="bg_{label}" x1="0" y1="0" x2="1" y2="0">'
            f'<stop offset="0%" stop-color="{c}" stop-opacity="0.85"/>'
            f'<stop offset="100%" stop-color="{c}" stop-opacity="0.45"/>'
            f'</linearGradient></defs>'
            f'<rect width="100%" height="5" fill="#1a1a1a"/>'
            f'<rect width="{bw}%" height="5" fill="url(#bg_{label})"/>'
            f'</svg>'
        )
        return (
            f'<div style="padding:.32rem 0;border-bottom:1px solid #111">'
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;margin-bottom:5px">'
            f'<div style="display:flex;align-items:center;gap:5px">'
            f'<span style="{_M}font-size:9px;font-weight:700;letter-spacing:.12em;'
            f'text-transform:uppercase;color:#8890a1">{label}</span>'
            f'<span style="background:#1a1a1a;border:1px solid #222;'
            f'{_M}font-size:7px;color:#555960;padding:0 4px;letter-spacing:.04em">'
            f'wt {wpct}%</span>'
            f'</div>'
            f'<div style="display:flex;align-items:center;gap:5px">'
            f'<span style="{_M}font-size:1.0rem;font-weight:700;color:{c};line-height:1">'
            f'{val:.0f}</span>'
            f'<span style="background:{c}22;border:1px solid {c}44;'
            f'{_M}font-size:7px;font-weight:700;color:{c};padding:1px 4px">{lbl}</span>'
            f'</div>'
            f'</div>'
            f'{bar_svg}'
            f'<div style="{_M}font-size:8px;color:#444;margin-top:4px">'
            f'adds <b style="color:{c}">{contrib:.1f} pts</b> to composite</div>'
            f'</div>'
        )

    score_c = _cc(score)
    html = (
        f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;padding:.5rem .65rem">'
        # header row — composite score
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding-bottom:.3rem;border-bottom:1px solid #1e1e1e;margin-bottom:.1rem">'
        f'<span style="{_M}font-size:8px;font-weight:700;letter-spacing:.14em;'
        f'text-transform:uppercase;color:#555960">GRS · Composite</span>'
        f'<div style="display:flex;align-items:baseline;gap:3px">'
        f'<span style="{_M}font-size:1.05rem;font-weight:700;color:{score_c}">'
        f'{score:.0f}</span>'
        f'<span style="{_M}font-size:8px;color:#555960">/100</span>'
        f'</div>'
        f'</div>'
        + _meter("CIS", int(w_cis * 100), cis, cis * w_cis)
        + _meter("TPS", int(w_tps * 100), tps, tps * w_tps)
        + _meter("MCS", int(w_mcs * 100), mcs, mcs * w_mcs)
        + f'<div style="{_M}font-size:8px;color:#333;margin-top:.35rem">'
        f'CIS=Conflict Intensity · TPS=Transmission · MCS=Market</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# § L4  ESCALATION TRACKER  (left column — trend / CIS velocity per conflict)
# ─────────────────────────────────────────────────────────────────────────────

def _render_escalation_tracker(conflict_results: dict) -> None:
    """Left column: per-conflict trend direction and escalation status."""
    active = [
        (cid, r) for cid, r in conflict_results.items()
        if r.get("state") == "active"
    ]
    if not active:
        return

    st.markdown(
        f'<div style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#CFB991;padding:.4rem 0 .3rem;'
        f'margin-top:.4rem;border-top:1px solid #1e1e1e">Escalation Tracker</div>',
        unsafe_allow_html=True,
    )

    rows = ""
    for cid, r in sorted(active, key=lambda x: x[1].get("cis", 0), reverse=True):
        trend     = r.get("trend", "stable")
        esc       = r.get("escalation", "stable")
        cis_v     = float(r.get("cis", 0))
        col       = r.get("color", "#8890a1")
        lbl       = r.get("label", cid)

        if trend == "rising":
            t_icon, t_col = "▲", "#c0392b"
        elif trend == "falling":
            t_icon, t_col = "▼", "#27ae60"
        else:
            t_icon, t_col = "→", "#e67e22"

        if esc == "escalating":
            esc_bg, esc_c, esc_txt = "rgba(192,57,43,0.12)", "#e05241", "ESC ↑"
        elif esc == "de-escalating":
            esc_bg, esc_c, esc_txt = "rgba(39,174,96,0.10)", "#27ae60", "DE-ESC"
        else:
            esc_bg, esc_c, esc_txt = "rgba(136,144,161,0.08)", "#8890a1", "STABLE"

        bar_w = min(int(cis_v), 100)
        rows += (
            f'<div style="display:flex;align-items:center;gap:6px;padding:5px .55rem;'
            f'border-bottom:1px solid #111;background:#0f0f0f">'
            # trend arrow
            f'<span style="{_M}font-size:11px;font-weight:700;color:{t_col};'
            f'min-width:12px;flex-shrink:0">{t_icon}</span>'
            # name
            f'<span style="{_M}font-size:9px;font-weight:700;color:{col};'
            f'min-width:72px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
            f'{lbl[:14]}</span>'
            # bar
            f'<div style="flex:1;background:#1a1a1a;height:3px;border-radius:1px;min-width:20px">'
            f'<div style="width:{bar_w}%;height:3px;background:{col};border-radius:1px;opacity:.8"></div></div>'
            # CIS value
            f'<span style="{_M}font-size:9px;font-weight:700;color:{col};min-width:22px;text-align:right">'
            f'{cis_v:.0f}</span>'
            # escalation badge
            f'<span style="background:{esc_bg};border:1px solid {esc_c}44;'
            f'{_M}font-size:7px;font-weight:700;color:{esc_c};padding:1px 4px;'
            f'letter-spacing:.06em;flex-shrink:0">{esc_txt}</span>'
            f'</div>'
        )

    st.markdown(
        f'<div style="border:1px solid #1e1e1e;background:#0a0a0a;overflow:hidden">{rows}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § L5  TOP AFFECTED COMMODITIES  (left column — commodity exposure hit-list)
# ─────────────────────────────────────────────────────────────────────────────

def _render_top_commodities(conflict_results: dict) -> None:
    """Left column: commodities most exposed across active conflicts."""
    from collections import Counter

    # Collect commodities from active conflicts, weighted by CIS
    ctr: Counter = Counter()
    conflict_count: dict[str, int] = {}
    for cid, r in conflict_results.items():
        if r.get("state") != "active":
            continue
        cis_w = float(r.get("cis", 50)) / 100.0
        for com in r.get("affected_commodities", []):
            key = com.strip()
            ctr[key] += cis_w
            conflict_count[key] = conflict_count.get(key, 0) + 1

    if not ctr:
        return

    top = ctr.most_common(6)
    max_score = top[0][1] if top else 1.0

    st.markdown(
        f'<div style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#CFB991;padding:.4rem 0 .3rem;'
        f'margin-top:.4rem;border-top:1px solid #1e1e1e">Commodity Exposure</div>',
        unsafe_allow_html=True,
    )

    rows = ""
    for com, score in top:
        n_c   = conflict_count.get(com, 1)
        bar_w = int(score / max_score * 100)
        # Color by exposure level
        if score / max_score > 0.66:
            bar_c = "#c0392b"
        elif score / max_score > 0.33:
            bar_c = "#e67e22"
        else:
            bar_c = "#8890a1"

        rows += (
            f'<div style="padding:.3rem .55rem;border-bottom:1px solid #111">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px">'
            f'<span style="{_M}font-size:9px;font-weight:700;color:#C8D4E0;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:120px">'
            f'{com}</span>'
            f'<span style="{_M}font-size:8px;color:{bar_c}">'
            f'{n_c} conflict{"s" if n_c > 1 else ""}</span>'
            f'</div>'
            f'<div style="background:#1a1a1a;height:3px;border-radius:1px">'
            f'<div style="width:{bar_w}%;height:3px;background:{bar_c};border-radius:1px;opacity:.85"></div>'
            f'</div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;overflow:hidden">{rows}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § L3  CONFLICT LANDSCAPE  (left column — CIS × TPS 2-D scatter)
# ─────────────────────────────────────────────────────────────────────────────

def _render_conflict_landscape(conflict_results: dict) -> None:
    """Left column: scatter of all tracked conflicts on a CIS × TPS plane."""
    st.markdown(
        f'<div style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#CFB991;padding:.4rem 0 .3rem;'
        f'margin-top:.4rem;border-top:1px solid #1e1e1e">Conflict Landscape</div>',
        unsafe_allow_html=True,
    )
    if not conflict_results:
        st.markdown(
            f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;'
            f'padding:.35rem .55rem;{_M}font-size:9px;color:#555960">No conflicts tracked</div>',
            unsafe_allow_html=True,
        )
        return

    # Canvas
    W, H   = 216, 162
    ML, MB = 22, 18   # left/bottom margin for axis labels
    PW     = W - ML   # usable plot width
    PH     = H - MB   # usable plot height

    def _fx(v: float) -> float:   # CIS → SVG x
        return ML + v / 100.0 * PW

    def _fy(v: float) -> float:   # TPS → SVG y  (inverted: 0 at bottom)
        return H - MB - v / 100.0 * PH

    mx, my = _fx(50), _fy(50)

    # Quadrant fills
    fills = (
        f'<rect x="{mx:.1f}" y="0" width="{W - mx:.1f}" height="{my:.1f}" fill="rgba(192,57,43,0.07)"/>'
        f'<rect x="{ML}" y="{my:.1f}" width="{mx - ML:.1f}" height="{PH - my:.1f}" fill="rgba(230,126,34,0.04)"/>'
    )

    # Grid lines at 25 / 50 / 75
    grid = ""
    for v in (25, 50, 75):
        gx, gy = _fx(v), _fy(v)
        c = "#2a2a2a" if v == 50 else "#191919"
        sw = "0.8" if v == 50 else "0.4"
        grid += (
            f'<line x1="{gx:.1f}" y1="0" x2="{gx:.1f}" y2="{PH:.1f}" stroke="{c}" stroke-width="{sw}"/>'
            f'<line x1="{ML}" y1="{gy:.1f}" x2="{W}" y2="{gy:.1f}" stroke="{c}" stroke-width="{sw}"/>'
        )

    # Quadrant text labels
    qlbls = (
        f'<text x="{_fx(75):.1f}" y="9" font-size="6.5" fill="rgba(192,57,43,0.55)" '
        f'font-family="JetBrains Mono,monospace" text-anchor="middle" font-weight="700">CRITICAL</text>'
        f'<text x="{_fx(25):.1f}" y="9" font-size="6.5" fill="#222" '
        f'font-family="JetBrains Mono,monospace" text-anchor="middle">VOLATILE</text>'
        f'<text x="{_fx(75):.1f}" y="{PH - 3:.1f}" font-size="6.5" fill="#222" '
        f'font-family="JetBrains Mono,monospace" text-anchor="middle">ISOLATED</text>'
        f'<text x="{_fx(25):.1f}" y="{PH - 3:.1f}" font-size="6.5" fill="#222" '
        f'font-family="JetBrains Mono,monospace" text-anchor="middle">LATENT</text>'
    )

    # Axis tick labels
    ticks = ""
    for v in (0, 50, 100):
        ticks += (
            f'<text x="{_fx(v):.1f}" y="{H - 2}" font-size="6" fill="#2a2a2a" '
            f'font-family="JetBrains Mono,monospace" text-anchor="middle">{v}</text>'
            f'<text x="{ML - 2}" y="{_fy(v) + 3:.1f}" font-size="6" fill="#2a2a2a" '
            f'font-family="JetBrains Mono,monospace" text-anchor="end">{v}</text>'
        )
    axis_lbls = (
        f'<text x="{W}" y="{H}" font-size="7" fill="#444" '
        f'font-family="JetBrains Mono,monospace" text-anchor="end">CIS →</text>'
        f'<text x="2" y="8" font-size="7" fill="#444" '
        f'font-family="JetBrains Mono,monospace">↑ TPS</text>'
    )

    # Conflict dots — monitoring first (behind), active on top
    active_items, mon_items = [], []
    for cid, r in conflict_results.items():
        entry = (float(r.get("cis", 0)), float(r.get("tps", 0)),
                 r.get("color", "#8890a1"), r.get("label", cid)[:3].upper())
        if r.get("state") == "active":
            active_items.append(entry)
        else:
            mon_items.append(entry)

    dots = ""
    for cis_v, tps_v, col, lbl in mon_items:
        px, py = _fx(cis_v), _fy(tps_v)
        dots += (
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" '
            f'fill="{col}" fill-opacity="0.18" stroke="{col}" stroke-width="0.6" stroke-opacity="0.4"/>'
        )
    for cis_v, tps_v, col, lbl in active_items:
        px, py = _fx(cis_v), _fy(tps_v)
        dots += (
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="9" '
            f'fill="{col}" fill-opacity="0.18" stroke="{col}" stroke-width="1.5"/>'
            f'<text x="{px:.1f}" y="{py + 2.5:.1f}" font-size="6" fill="{col}" '
            f'font-family="JetBrains Mono,monospace" text-anchor="middle" font-weight="700">{lbl}</text>'
        )

    svg = (
        f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="display:block">'
        + fills + grid + qlbls + ticks + axis_lbls + dots
        + f'</svg>'
    )

    n_act = len(active_items)
    n_mon = len(mon_items)
    st.markdown(
        f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;padding:.5rem .65rem">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">'
        f'<span style="{_M}font-size:8px;color:#555960">CIS × TPS · all tracked</span>'
        f'<div style="display:flex;gap:4px">'
        f'<span style="background:rgba(192,57,43,0.12);border:1px solid rgba(192,57,43,0.28);'
        f'{_M}font-size:7px;color:#e05241;padding:1px 5px">{n_act} ACTIVE</span>'
        f'<span style="background:#111;border:1px solid #1e1e1e;'
        f'{_M}font-size:7px;color:#555960;padding:1px 5px">{n_mon} MON</span>'
        f'</div>'
        f'</div>'
        f'{svg}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § R3  RISK COMPASS  (right column — 5-axis radar)
# ─────────────────────────────────────────────────────────────────────────────

def _render_risk_compass(risk: dict, corr_val: float | None = None) -> None:
    """Right column: pentagon radar across CIS, TPS, MCS, Volatility, Coupling."""
    cis   = float(risk.get("cis",   50.0))
    tps   = float(risk.get("tps",   50.0))
    mcs   = float(risk.get("mcs",   50.0))
    score = float(risk.get("score", 50.0))

    # Volatility score: VIX 40 = 100%
    vix_val = 20.0
    for d in _load_market_pulse():
        if d["sym"] == "^VIX":
            vix_val = float(d["val"])
            break
    vol_score = min(vix_val / 40.0 * 100.0, 100.0)

    # Coupling score: correlation [-1, 1] → [5, 95]
    if corr_val is not None:
        coup_score = 5.0 + (corr_val + 1.0) / 2.0 * 90.0
    else:
        coup_score = 50.0

    # 5 axes — ordered to make the pentagon visually balanced
    axes = [
        ("CIS",  cis,        "#c0392b"),   # top
        ("MCS",  mcs,        "#2980b9"),   # upper-right
        ("VOL",  vol_score,  "#8E6F3E"),   # lower-right
        ("COUP", coup_score, "#27ae60"),   # lower-left
        ("TPS",  tps,        "#e67e22"),   # upper-left
    ]
    n = len(axes)

    cx, cy, R = 105, 100, 64
    LOFF = 16   # label offset beyond ring

    def _ang(i: int) -> float:
        return math.radians(-90.0 + i * 360.0 / n)

    def _pt(i: int, val: float):
        a = _ang(i)
        r = val / 100.0 * R
        return cx + r * math.cos(a), cy + r * math.sin(a)

    # Concentric pentagon grid rings
    grid_html = ""
    for pct in (25, 50, 75, 100):
        pts = " ".join(
            f'{cx + pct/100*R*math.cos(_ang(i)):.1f},{cy + pct/100*R*math.sin(_ang(i)):.1f}'
            for i in range(n)
        )
        c  = "#2a2a2a" if pct == 50 else "#191919"
        sw = "0.9"    if pct == 50 else "0.5"
        grid_html += f'<polygon points="{pts}" fill="none" stroke="{c}" stroke-width="{sw}"/>'

    # Axis spokes
    spoke_html = "".join(
        f'<line x1="{cx}" y1="{cy}" '
        f'x2="{cx + R * math.cos(_ang(i)):.1f}" y2="{cy + R * math.sin(_ang(i)):.1f}" '
        f'stroke="#1e1e1e" stroke-width="1"/>'
        for i in range(n)
    )

    # Data polygon
    data_pts = " ".join(f'{_pt(i, axes[i][1])[0]:.1f},{_pt(i, axes[i][1])[1]:.1f}' for i in range(n))
    fill_c = "#c0392b" if score >= 65 else "#e67e22" if score >= 45 else "#27ae60"

    # Labels + endpoint dots per axis
    lbl_html = ""
    for i, (name, val, col) in enumerate(axes):
        a   = _ang(i)
        cos_a, sin_a = math.cos(a), math.sin(a)
        lx  = cx + (R + LOFF) * cos_a
        ly  = cy + (R + LOFF) * sin_a
        anc = "middle" if abs(cos_a) < 0.3 else ("start" if cos_a > 0 else "end")
        # stagger name / value vertically so they don't overlap
        n_dy = -5 if sin_a < -0.25 else (3 if sin_a > 0.25 else -4)
        v_dy = n_dy + 10
        lbl_html += (
            f'<text x="{lx:.1f}" y="{ly + n_dy:.1f}" font-size="8" fill="#8890a1" '
            f'font-family="JetBrains Mono,monospace" text-anchor="{anc}">{name}</text>'
            f'<text x="{lx:.1f}" y="{ly + v_dy:.1f}" font-size="8.5" fill="{col}" '
            f'font-family="JetBrains Mono,monospace" text-anchor="{anc}" font-weight="700">{val:.0f}</text>'
        )
        px, py = _pt(i, val)
        lbl_html += f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{col}" opacity="0.9"/>'

    # Score in center
    ctr_html = (
        f'<text x="{cx}" y="{cy - 5}" font-size="15" fill="{fill_c}" '
        f'font-family="JetBrains Mono,monospace" text-anchor="middle" font-weight="700">{score:.0f}</text>'
        f'<text x="{cx}" y="{cy + 9}" font-size="7" fill="#555960" '
        f'font-family="JetBrains Mono,monospace" text-anchor="middle">GRS</text>'
    )

    SVG_W, SVG_H = 210, 195
    svg = (
        f'<svg width="{SVG_W}" height="{SVG_H}" viewBox="0 0 {SVG_W} {SVG_H}" '
        f'style="display:block;margin:0 auto">'
        + grid_html + spoke_html
        + f'<polygon points="{data_pts}" fill="{fill_c}" fill-opacity="0.13" '
        f'stroke="{fill_c}" stroke-width="1.5" stroke-linejoin="round"/>'
        + lbl_html + ctr_html
        + f'</svg>'
    )

    severity_lbl = "HIGH" if score >= 65 else "ELEVATED" if score >= 45 else "NOMINAL"
    st.markdown(
        f'<div style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#CFB991;padding:.4rem 0 .3rem;'
        f'margin-top:.35rem;border-top:1px solid #1e1e1e">Risk Compass</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;padding:.5rem .65rem">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'
        f'<span style="{_M}font-size:8px;color:#555960">5-dimension risk profile</span>'
        f'<span style="background:{fill_c}1a;border:1px solid {fill_c}44;'
        f'{_M}font-size:7px;font-weight:700;letter-spacing:.10em;color:{fill_c};'
        f'padding:1px 5px">{severity_lbl}</span>'
        f'</div>'
        f'{svg}'
        f'<div style="{_M}font-size:7.5px;color:#2e2e2e;margin-top:3px">'
        f'VOL=VIX stress · COUP=eq↔cmd coupling · MCS=market conditions</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § R4  5-DAY RETURNS HEATMAP  (right column — colored day-by-day asset grid)
# ─────────────────────────────────────────────────────────────────────────────

def _render_returns_heatmap() -> None:
    """Right column: 5-day daily-return heat grid for each pulse instrument."""
    data = _load_market_pulse()
    if not data:
        return

    # Need at least 2 price points to compute any return
    rows_data = []
    n_days = 0
    for d in data:
        series = d.get("series", [])
        if len(series) < 2:
            continue
        returns = [
            (series[i + 1] - series[i]) / series[i] * 100.0
            for i in range(len(series) - 1)
        ]
        rows_data.append((d["label"], d["sym"], returns))
        n_days = max(n_days, len(returns))

    if not rows_data or n_days == 0:
        return

    # Day column headers — count back from today
    day_labels = [f"D-{n_days - i}" for i in range(n_days - 1)] + ["TODAY"]

    # Color helpers
    def _cell_bg(pct: float, is_vix: bool) -> str:
        pos = pct > 0
        if is_vix:
            pos = not pos   # rising VIX = bad
        intensity = min(abs(pct) / 2.0, 1.0)   # cap at ±2%
        if pos:
            r = int(20 + (1 - intensity) * 15)
            g = int(60 + intensity * 50)
            b = int(20 + (1 - intensity) * 15)
        else:
            r = int(60 + intensity * 70)
            g = int(20 + (1 - intensity) * 15)
            b = int(20 + (1 - intensity) * 15)
        return f"rgb({r},{g},{b})"

    def _cell_txt(pct: float) -> str:
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.2f}"

    # Header row
    hdr_cells = "".join(
        f'<th style="{_M}font-size:7px;font-weight:700;letter-spacing:.08em;'
        f'color:#555960;text-align:center;padding:3px 4px;border-right:1px solid #111">'
        f'{lbl}</th>'
        for lbl in day_labels
    )
    header = (
        f'<tr><th style="{_M}font-size:7px;color:#555960;padding:3px 6px;'
        f'text-align:left;border-right:1px solid #1e1e1e;min-width:52px">ASSET</th>'
        + hdr_cells + f'</tr>'
    )

    # Data rows
    tbody = ""
    for label, sym, returns in rows_data:
        is_vix = sym == "^VIX"
        # Pad with blanks if fewer days than max
        padded = [None] * (n_days - len(returns)) + returns
        cells  = ""
        for pct in padded:
            if pct is None:
                cells += f'<td style="background:#111;padding:4px 3px"></td>'
            else:
                bg  = _cell_bg(pct, is_vix)
                txt = _cell_txt(pct)
                cells += (
                    f'<td style="background:{bg};{_M}font-size:7.5px;font-weight:700;'
                    f'color:#e8e9ed;text-align:center;padding:4px 3px;'
                    f'border-right:1px solid #111;white-space:nowrap">{txt}</td>'
                )
        tbody += (
            f'<tr style="border-bottom:1px solid #111">'
            f'<td style="{_M}font-size:8px;font-weight:700;color:#A8B8C8;'
            f'padding:4px 6px;border-right:1px solid #1e1e1e;white-space:nowrap">'
            f'{label}</td>'
            + cells + f'</tr>'
        )

    st.markdown(
        f'<div style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#CFB991;padding:.4rem 0 .3rem;'
        f'margin-top:.35rem;border-top:1px solid #1e1e1e">Daily Returns</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;overflow:hidden">'
        f'<table style="width:100%;border-collapse:collapse">'
        f'<thead style="background:#0a0a0a;border-bottom:1px solid #1e1e1e">{header}</thead>'
        f'<tbody>{tbody}</tbody>'
        f'</table>'
        f'<div style="{_M}font-size:7px;color:#2a2a2a;padding:3px 6px">'
        f'day-over-day % · VIX inverted (↑ VIX = red)</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § R5  TRANSMISSION CHANNELS  (right column — channel-level TPS breakdown)
# ─────────────────────────────────────────────────────────────────────────────

_CHANNEL_GROUPS = {
    "Energy":        ("oil_gas", "energy_infra"),
    "Shipping":      ("shipping", "chokepoint", "supply_chain"),
    "Metals":        ("metals",),
    "FX / Sanctions":("fx", "sanctions"),
    "Equity / Infl": ("equity_sector", "credit", "inflation"),
}

def _render_transmission_channels(conflict_results: dict, risk: dict) -> None:
    """Right column: aggregate transmission pressure by channel group."""
    active = [(cid, r) for cid, r in conflict_results.items() if r.get("state") == "active"]
    if not active:
        return

    # Aggregate channel scores across active conflicts (weighted by CIS)
    totals: dict[str, float] = {g: 0.0 for g in _CHANNEL_GROUPS}
    weights_sum: dict[str, float] = {g: 0.0 for g in _CHANNEL_GROUPS}

    for _, r in active:
        tx  = r.get("transmission", {}) or {}
        cis_w = float(r.get("cis", 50)) / 100.0
        for group, keys in _CHANNEL_GROUPS.items():
            vals = [float(tx.get(k, 0.0)) for k in keys if k in tx]
            if vals:
                totals[group]      += (sum(vals) / len(vals)) * cis_w
                weights_sum[group] += cis_w

    # Normalize to 0-100
    scores: dict[str, float] = {}
    for g in _CHANNEL_GROUPS:
        if weights_sum[g] > 0:
            scores[g] = min(totals[g] / weights_sum[g] * 100.0, 100.0)
        else:
            scores[g] = 0.0

    # Fall back to TPS-scaled estimate if all zeroes (sparse transmission data)
    if all(v < 1.0 for v in scores.values()):
        tps = float(risk.get("tps", 50.0))
        default = {"Energy": tps * 0.90, "Shipping": tps * 0.75,
                   "Metals": tps * 0.55, "FX / Sanctions": tps * 0.45, "Equity / Infl": tps * 0.40}
        scores = default

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    st.markdown(
        f'<div style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#CFB991;padding:.4rem 0 .3rem;'
        f'margin-top:.35rem;border-top:1px solid #1e1e1e">Transmission Channels</div>',
        unsafe_allow_html=True,
    )

    rows = ""
    for group, score in sorted_scores:
        bw = int(score)
        if score >= 65:   bc = "#c0392b"
        elif score >= 40: bc = "#e67e22"
        else:             bc = "#2980b9"
        rows += (
            f'<div style="padding:.28rem .55rem;border-bottom:1px solid #111">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px">'
            f'<span style="{_M}font-size:8.5px;font-weight:700;color:#8890a1">{group}</span>'
            f'<span style="{_M}font-size:9px;font-weight:700;color:{bc}">{score:.0f}</span>'
            f'</div>'
            f'<div style="background:#1a1a1a;height:4px;border-radius:2px">'
            f'<div style="width:{bw}%;height:4px;background:{bc};border-radius:2px;opacity:.85"></div>'
            f'</div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;overflow:hidden">{rows}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § EXTRA DATA FETCHERS  (cached; warm after first load)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def _load_yield_curve() -> dict[str, float]:
    """Fetch 4 yield-curve points: 3M / 5Y / 10Y / 30Y via yfinance."""
    try:
        import yfinance as yf
        syms = {"3M": "^IRX", "5Y": "^FVX", "10Y": "^TNX", "30Y": "^TYX"}
        raw  = yf.download(list(syms.values()), period="5d",
                           auto_adjust=True, progress=False, threads=True)
        close = raw["Close"] if "Close" in raw.columns else raw
        result = {}
        for label, sym in syms.items():
            if sym in close.columns:
                s = close[sym].dropna()
                if len(s):
                    result[label] = float(s.iloc[-1])
        return result
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def _load_vol_trio() -> dict[str, dict]:
    """Fetch VIX / OVX / GVZ with 1-year high/low for gauge normalisation."""
    try:
        import yfinance as yf
        syms = {"VIX": "^VIX", "OVX": "^OVX", "GVZ": "^GVZ"}
        raw  = yf.download(list(syms.values()), period="1y",
                           auto_adjust=True, progress=False, threads=True)
        close = raw["Close"] if "Close" in raw.columns else raw
        result = {}
        for label, sym in syms.items():
            if sym in close.columns:
                s = close[sym].dropna()
                if len(s) >= 2:
                    result[label] = {
                        "cur": float(s.iloc[-1]),
                        "lo":  float(s.min()),
                        "hi":  float(s.max()),
                        "pct_rank": float((s.iloc[-1] - s.min()) / max(s.max() - s.min(), 0.01)),
                    }
        return result
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# § L6  REGIME HISTORY — 60-day colour-coded regime strip
# ─────────────────────────────────────────────────────────────────────────────

def _render_regime_history(regimes: "pd.Series | None") -> None:
    """Left column: 60-day day-by-day correlation-regime strip."""
    st.markdown(
        f'<div style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#CFB991;padding:.4rem 0 .3rem;'
        f'margin-top:.35rem;border-top:1px solid #1e1e1e">Regime History · 60d</div>',
        unsafe_allow_html=True,
    )

    _REG_COL = {1: "#27ae60", 2: "#e8a838", 3: "#c0392b"}
    _REG_LAB = {1: "DECOUPLED", 2: "TRANSITIONING", 3: "HIGH COUPLING"}

    if regimes is None or len(regimes.dropna()) < 5:
        st.markdown(
            f'<div style="{_M}font-size:9px;color:#555960;padding:.3rem 0">Regime data unavailable</div>',
            unsafe_allow_html=True,
        )
        return

    vals = list(regimes.dropna().astype(int).iloc[-60:])
    N    = len(vals)
    W, H, gap = 210, 18, 1
    cell_w = max(1, (W - gap * (N - 1)) / N)

    cells = ""
    for i, v in enumerate(vals):
        c = _REG_COL.get(v, "#333")
        x = i * (cell_w + gap)
        # fade older cells slightly
        op = 0.45 + 0.55 * (i / max(N - 1, 1))
        cells += f'<rect x="{x:.1f}" y="0" width="{cell_w:.1f}" height="{H}" fill="{c}" opacity="{op:.2f}" rx="1"/>'

    # current regime badge
    cur_v   = int(vals[-1]) if vals else 1
    cur_col = _REG_COL.get(cur_v, "#555")
    cur_lbl = _REG_LAB.get(cur_v, "UNKNOWN")

    # count of days in each regime
    from collections import Counter
    cnt = Counter(vals)

    legend = ""
    for rv, rc in _REG_COL.items():
        n = cnt.get(rv, 0)
        lx = (rv - 1) * 72
        legend += (
            f'<rect x="{lx}" y="0" width="8" height="8" fill="{rc}" rx="1" opacity=".8"/>'
            f'<text x="{lx + 11}" y="8" font-size="7.5" fill="#8890a1" font-family="JetBrains Mono,monospace">'
            f'{_REG_LAB[rv][:4]} {n}d</text>'
        )

    st.markdown(
        f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;padding:.45rem .55rem;border-radius:2px">'
        f'<svg width="{W}" height="{H}" style="display:block">{cells}</svg>'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:.35rem">'
        f'<svg width="216" height="12">{legend}</svg>'
        f'</div>'
        f'<div style="margin-top:.3rem;{_M}font-size:8.5px;font-weight:700;color:{cur_col}">'
        f'NOW: {cur_lbl}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § L7  ALERT SUMMARY — severity breakdown of proactive alerts
# ─────────────────────────────────────────────────────────────────────────────

def _render_alert_summary(alerts: list) -> None:
    """Left column: compact count of alerts by severity + most recent message."""
    if not alerts:
        return

    from collections import Counter, OrderedDict
    sev_order = ["critical", "high", "medium", "low"]
    sev_col   = {"critical": "#c0392b", "high": "#e67e22", "medium": "#e8a838", "low": "#2980b9"}
    cnt = Counter(getattr(a, "severity", "low") for a in alerts)

    st.markdown(
        f'<div style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#CFB991;padding:.4rem 0 .3rem;'
        f'margin-top:.35rem;border-top:1px solid #1e1e1e">Active Alerts · {len(alerts)}</div>',
        unsafe_allow_html=True,
    )

    badges = ""
    for sev in sev_order:
        n = cnt.get(sev, 0)
        if n == 0:
            continue
        c = sev_col[sev]
        badges += (
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:.28rem">'
            f'<span style="background:{c}22;border:1px solid {c}66;'
            f'{_M}font-size:8px;font-weight:700;color:{c};padding:1px 6px;'
            f'border-radius:2px;letter-spacing:.08em;min-width:20px;text-align:center">{n}</span>'
            f'<span style="{_M}font-size:9px;color:#8890a1;text-transform:uppercase;'
            f'letter-spacing:.08em">{sev}</span>'
            f'</div>'
        )

    # Most recent alert message
    recent = alerts[0]
    msg    = (getattr(recent, "message", "") or "")[:80]
    rc     = sev_col.get(getattr(recent, "severity", "low"), "#555")

    st.markdown(
        f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;padding:.45rem .55rem;border-radius:2px">'
        f'{badges}'
        f'<div style="border-top:1px solid #1a1a1a;margin-top:.3rem;padding-top:.3rem">'
        f'<span style="{_M}font-size:8.5px;color:{rc};font-weight:600">LATEST:</span>'
        f'<span style="{_M}font-size:8.5px;color:#8890a1;margin-left:4px">{msg}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § R6  GRS TREND — 60-day composite risk score sparkline
# ─────────────────────────────────────────────────────────────────────────────

def _render_grs_trend(score_hist: "pd.Series | None") -> None:
    """Right column: 60-day GRS sparkline with risk zone bands and delta."""
    st.markdown(
        f'<div style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#CFB991;padding:.4rem 0 .3rem;'
        f'margin-top:.35rem;border-top:1px solid #1e1e1e">Risk Score Trend · 60d</div>',
        unsafe_allow_html=True,
    )

    import pandas as _pd
    if score_hist is None or (isinstance(score_hist, _pd.Series) and len(score_hist.dropna()) < 5):
        st.markdown(
            f'<div style="{_M}font-size:9px;color:#555960;padding:.3rem 0">Score history unavailable</div>',
            unsafe_allow_html=True,
        )
        return

    vals = list(score_hist.dropna().iloc[-60:])
    N    = len(vals)
    W, H = 210, 56
    ML, MR, MT, MB = 4, 4, 4, 4
    PW = W - ML - MR
    PH = H - MT - MB

    def _fx(i):   return ML + i / max(N - 1, 1) * PW
    def _fy(v):   return MT + (1.0 - max(0.0, min(1.0, v / 100.0))) * PH

    cur   = vals[-1]
    delta = cur - vals[max(0, N - 30)] if N >= 5 else 0.0
    d_col = "#c0392b" if delta > 0 else "#27ae60"
    d_sym = "▲" if delta > 0 else "▼"
    s_col = "#c0392b" if cur >= 70 else "#e8a838" if cur >= 45 else "#27ae60"

    # zone bands
    y_high = _fy(70)
    y_mid  = _fy(45)
    bands  = (
        f'<rect x="{ML}" y="{MT}" width="{PW}" height="{y_high - MT}" '
        f'fill="#c0392b" opacity=".06" rx="0"/>'
        f'<rect x="{ML}" y="{y_high}" width="{PW}" height="{y_mid - y_high}" '
        f'fill="#e8a838" opacity=".06" rx="0"/>'
        f'<rect x="{ML}" y="{y_mid}" width="{PW}" height="{MT + PH - y_mid}" '
        f'fill="#27ae60" opacity=".05" rx="0"/>'
    )

    # threshold lines
    lines = (
        f'<line x1="{ML}" y1="{y_high:.1f}" x2="{ML + PW}" y2="{y_high:.1f}" '
        f'stroke="#c0392b" stroke-width=".6" stroke-dasharray="3,3" opacity=".5"/>'
        f'<line x1="{ML}" y1="{y_mid:.1f}" x2="{ML + PW}" y2="{y_mid:.1f}" '
        f'stroke="#e8a838" stroke-width=".6" stroke-dasharray="3,3" opacity=".4"/>'
    )

    pts = " ".join(f'{_fx(i):.1f},{_fy(v):.1f}' for i, v in enumerate(vals))

    # fill polygon under line
    fill_pts = (
        f'{_fx(0):.1f},{MT + PH} ' + pts + f' {_fx(N-1):.1f},{MT + PH}'
    )

    # current dot
    dot_x, dot_y = _fx(N - 1), _fy(cur)

    svg = (
        f'<svg width="{W}" height="{H}" style="display:block;overflow:visible">'
        f'{bands}{lines}'
        f'<polygon points="{fill_pts}" fill="{s_col}" opacity=".08"/>'
        f'<polyline points="{pts}" fill="none" stroke="{s_col}" '
        f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" opacity=".9"/>'
        f'<circle cx="{dot_x:.1f}" cy="{dot_y:.1f}" r="3" fill="{s_col}" opacity="1"/>'
        f'</svg>'
    )

    st.markdown(
        f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;padding:.45rem .55rem;border-radius:2px">'
        f'{svg}'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:.3rem">'
        f'<span style="{_M}font-size:8.5px;color:#555960">GRS NOW</span>'
        f'<span style="{_M}font-size:11px;font-weight:700;color:{s_col}">{cur:.1f}</span>'
        f'<span style="{_M}font-size:8.5px;color:{d_col}">{d_sym} {abs(delta):.1f} vs 30d</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § R7  MACRO SNAPSHOT — 4-cell grid: VIX / 10Y / DXY / WTI
# ─────────────────────────────────────────────────────────────────────────────

def _render_macro_snapshot() -> None:
    """Right column: 2×2 grid of macro regime indicators from pulse tickers."""
    data = _load_market_pulse()
    if not data:
        return

    by_sym = {d["sym"]: d for d in data}

    def _cell(sym, label, fmt_fn, regime_fn):
        d = by_sym.get(sym)
        if not d:
            return None
        val   = d["val"]
        pct   = d["pct"]
        vstr  = fmt_fn(val)
        reg, rc = regime_fn(val)
        arr   = "▲" if pct > 0.05 else "▼" if pct < -0.05 else "—"
        ac    = "#c0392b" if arr == "▲" else "#27ae60" if arr == "▼" else "#555960"
        # flip colour for VIX (up = bad)
        if sym == "^VIX":
            ac = "#c0392b" if arr == "▲" else "#27ae60" if arr == "▼" else "#555960"
        return (label, vstr, reg, rc, arr, ac)

    vix_cell = _cell("^VIX",   "VIX",
        fmt_fn=lambda v: f"{v:.1f}",
        regime_fn=lambda v: ("CALM", "#27ae60") if v < 15 else
                            ("NORMAL", "#8890a1") if v < 25 else
                            ("ELEVATED", "#e8a838") if v < 35 else
                            ("STRESS", "#c0392b"))

    tny_cell = _cell("^TNX",   "10Y YLD",
        fmt_fn=lambda v: f"{v:.2f}%",
        regime_fn=lambda v: ("LOW", "#2980b9") if v < 3.0 else
                            ("NORMAL", "#8890a1") if v < 4.5 else
                            ("HIGH", "#e8a838") if v < 5.5 else
                            ("EXTREME", "#c0392b"))

    dxy_cell = _cell("DX-Y.NYB", "DXY",
        fmt_fn=lambda v: f"{v:.1f}",
        regime_fn=lambda v: ("WEAK", "#27ae60") if v < 98 else
                            ("NEUTRAL", "#8890a1") if v < 104 else
                            ("STRONG", "#e8a838") if v < 110 else
                            ("EXTREME", "#c0392b"))

    wti_cell = _cell("CL=F",   "WTI",
        fmt_fn=lambda v: f"${v:.1f}",
        regime_fn=lambda v: ("LOW", "#27ae60") if v < 60 else
                            ("MID", "#8890a1") if v < 80 else
                            ("HIGH", "#e8a838") if v < 100 else
                            ("SPIKE", "#c0392b"))

    cells = [vix_cell, tny_cell, dxy_cell, wti_cell]
    if not any(cells):
        return

    st.markdown(
        f'<div style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#CFB991;padding:.4rem 0 .3rem;'
        f'margin-top:.35rem;border-top:1px solid #1e1e1e">Macro Snapshot</div>',
        unsafe_allow_html=True,
    )

    # Render 2×2 grid
    grid_rows = [cells[:2], cells[2:]]
    for row in grid_rows:
        cols_html = ""
        for cell in row:
            if cell is None:
                cols_html += '<div style="flex:1"></div>'
                continue
            label, vstr, reg, rc, arr, ac = cell
            cols_html += (
                f'<div style="flex:1;background:#0d0d0d;border:1px solid #1a1a1a;'
                f'padding:.4rem .5rem;border-radius:2px;min-width:0">'
                f'<div style="{_M}font-size:7.5px;font-weight:700;letter-spacing:.12em;'
                f'color:#555960;text-transform:uppercase;margin-bottom:2px">{label}</div>'
                f'<div style="{_M}font-size:1.0rem;font-weight:700;color:#e8e9ed;line-height:1">{vstr}</div>'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:3px">'
                f'<span style="{_M}font-size:7.5px;font-weight:700;color:{rc};'
                f'letter-spacing:.08em">{reg}</span>'
                f'<span style="{_M}font-size:9px;font-weight:700;color:{ac}">{arr}</span>'
                f'</div>'
                f'</div>'
            )
        st.markdown(
            f'<div style="display:flex;gap:4px;margin-bottom:4px">{cols_html}</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# § L8  COMMODITY SECTOR RETURNS — Energy / Metals / Agri 5-day performance
# ─────────────────────────────────────────────────────────────────────────────

def _render_commodity_sector_returns(cmd_r: "pd.DataFrame | None") -> None:
    """Left column: 5-day average return by commodity sector group."""
    import pandas as _pd
    if cmd_r is None or (isinstance(cmd_r, _pd.DataFrame) and cmd_r.empty):
        return

    from src.data.config import COMMODITY_GROUPS, COMMODITY_TICKERS
    rev_map = {v: k for k, v in COMMODITY_TICKERS.items()}

    # map column names (could be ticker or asset name) to group
    group_rets: dict[str, list[float]] = {g: [] for g in COMMODITY_GROUPS}
    for col in cmd_r.columns:
        asset = col if col in COMMODITY_TICKERS else rev_map.get(col, col)
        for g, members in COMMODITY_GROUPS.items():
            if asset in members:
                s = cmd_r[col].dropna()
                if len(s) >= 5:
                    ret5 = float((s.iloc[-1] / s.iloc[-5] - 1) * 100) if s.iloc[-5] != 0 else 0.0
                    group_rets[g].append(ret5)
                break

    rows_html = ""
    _G_COL = {
        "Energy":           "#e67e22",
        "Precious Metals":  "#f1c40f",
        "Industrial Metals":"#95a5a6",
        "Agriculture":      "#27ae60",
    }
    for g, rets in group_rets.items():
        if not rets:
            continue
        avg = sum(rets) / len(rets)
        col = "#27ae60" if avg >= 0 else "#c0392b"
        bar_w  = min(abs(avg) / 5.0 * 100, 100)   # cap at ±5% full width
        bar_dir = "right" if avg >= 0 else "left"
        sign   = "+" if avg >= 0 else ""
        gc     = _G_COL.get(g, "#8890a1")
        abbr   = {"Energy": "ENGY", "Precious Metals": "PREC",
                  "Industrial Metals": "INDU", "Agriculture": "AGRI"}.get(g, g[:4].upper())
        rows_html += (
            f'<div style="padding:.28rem .55rem;border-bottom:1px solid #111">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px">'
            f'<span style="{_M}font-size:8px;font-weight:700;color:{gc}">{abbr}</span>'
            f'<span style="{_M}font-size:9px;font-weight:700;color:{col}">{sign}{avg:.2f}%</span>'
            f'</div>'
            f'<div style="background:#1a1a1a;height:4px;border-radius:2px;overflow:hidden">'
            f'<div style="float:{bar_dir};width:{bar_w:.0f}%;height:4px;'
            f'background:{col};border-radius:2px;opacity:.85"></div>'
            f'</div>'
            f'</div>'
        )

    if not rows_html:
        return

    st.markdown(
        f'<div style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#CFB991;padding:.4rem 0 .3rem;'
        f'margin-top:.35rem;border-top:1px solid #1e1e1e">Commodity Sectors · 5d Rtn</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;overflow:hidden">{rows_html}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § L9  CROSS-CORRELATION LAG — does commodity lead or lag equity?
# ─────────────────────────────────────────────────────────────────────────────

def _render_cross_corr_lag(eq_r: "pd.DataFrame | None", cmd_r: "pd.DataFrame | None") -> None:
    """Left column: cross-correlation of avg equity vs avg commodity at lags 0–5."""
    import pandas as _pd, numpy as _np
    if eq_r is None or cmd_r is None:
        return
    if isinstance(eq_r, _pd.DataFrame) and eq_r.empty:
        return
    if isinstance(cmd_r, _pd.DataFrame) and cmd_r.empty:
        return

    try:
        avg_eq  = eq_r.mean(axis=1).dropna()
        avg_cmd = cmd_r.mean(axis=1).dropna()
        idx     = avg_eq.index.intersection(avg_cmd.index)
        if len(idx) < 30:
            return
        eq_s  = avg_eq.loc[idx]
        cmd_s = avg_cmd.loc[idx]

        lags = list(range(0, 6))
        corrs = []
        for lag in lags:
            shifted = cmd_s.shift(lag)
            aligned = _pd.concat([eq_s, shifted], axis=1).dropna()
            if len(aligned) >= 20:
                corrs.append(float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])))
            else:
                corrs.append(0.0)
    except Exception:
        return

    peak_lag = int(lags[corrs.index(max(corrs, key=abs))])
    peak_cor = corrs[peak_lag]

    W, H  = 210, 52
    ML, MR, MT, MB = 22, 8, 6, 16
    PW    = W - ML - MR
    PH    = H - MT - MB
    n     = len(lags)
    bw    = PW / n * 0.55
    gap   = PW / n

    bars_svg = ""
    for i, (lag, cor) in enumerate(zip(lags, corrs)):
        bh   = max(abs(cor) * PH, 1.5)
        col  = "#27ae60" if cor >= 0 else "#c0392b"
        op   = 1.0 if lag == peak_lag else 0.5
        x    = ML + i * gap + gap / 2 - bw / 2
        y    = MT + PH / 2 - bh / 2 if cor >= 0 else MT + PH / 2
        bars_svg += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{col}" opacity="{op}" rx="1"/>'

    # midline
    mid_y = MT + PH / 2
    bars_svg += f'<line x1="{ML}" y1="{mid_y:.1f}" x2="{ML+PW}" y2="{mid_y:.1f}" stroke="#333" stroke-width=".6"/>'

    # x-axis lag labels
    for i, lag in enumerate(lags):
        lx = ML + i * gap + gap / 2
        bars_svg += f'<text x="{lx:.1f}" y="{H-2}" font-size="7" fill="#555960" text-anchor="middle" font-family="JetBrains Mono,monospace">+{lag}d</text>'

    # y-axis ticks (+0.5 / 0 / -0.5)
    for yv, yt in [(0.5, MT + PH * 0.0), (0.0, MT + PH * 0.5), (-0.5, MT + PH * 1.0)]:
        bars_svg += f'<text x="{ML-2}" y="{yt:.1f}" font-size="6.5" fill="#444" text-anchor="end" font-family="JetBrains Mono,monospace" dominant-baseline="middle">{yv:+.1f}</text>'

    lead_msg = (
        f"CMD leads EQ by {peak_lag}d" if peak_lag > 0
        else "Contemporaneous (lag 0 peak)"
    )

    st.markdown(
        f'<div style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#CFB991;padding:.4rem 0 .3rem;'
        f'margin-top:.35rem;border-top:1px solid #1e1e1e">CMD→EQ Lead-Lag · 60d</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;padding:.45rem .55rem;border-radius:2px">'
        f'<svg width="{W}" height="{H}" style="display:block;overflow:visible">{bars_svg}</svg>'
        f'<div style="display:flex;justify-content:space-between;margin-top:.25rem">'
        f'<span style="{_M}font-size:8px;color:#8890a1">{lead_msg}</span>'
        f'<span style="{_M}font-size:8px;font-weight:700;color:{"#27ae60" if peak_cor>=0 else "#c0392b"}">'
        f'ρ={peak_cor:.2f}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § R8  YIELD CURVE SNAPSHOT — 3M / 5Y / 10Y / 30Y shape
# ─────────────────────────────────────────────────────────────────────────────

def _render_yield_curve_snap() -> None:
    """Right column: live yield curve shape across 4 tenors."""
    yc = _load_yield_curve()
    tenors  = ["3M", "5Y", "10Y", "30Y"]
    present = [(t, yc[t]) for t in tenors if t in yc]
    if len(present) < 2:
        return

    W, H  = 210, 60
    ML, MR, MT, MB = 28, 8, 8, 18
    PW    = W - ML - MR
    PH    = H - MT - MB

    vals  = [v for _, v in present]
    mn    = min(vals) - 0.1
    mx    = max(vals) + 0.1
    span  = mx - mn or 0.5

    def _fx(i):  return ML + i / (len(present) - 1) * PW
    def _fy(v):  return MT + (1.0 - (v - mn) / span) * PH

    pts = " ".join(f"{_fx(i):.1f},{_fy(v):.1f}" for i, (_, v) in enumerate(present))

    # colour by slope: short-end vs long-end
    slope = present[-1][1] - present[0][1]
    line_col = "#27ae60" if slope > 0.1 else "#c0392b" if slope < -0.1 else "#e8a838"
    shape_lbl = "NORMAL" if slope > 0.1 else "INVERTED" if slope < -0.1 else "FLAT"

    # fill under curve
    fill_pts = f"{_fx(0):.1f},{MT+PH} " + pts + f" {_fx(len(present)-1):.1f},{MT+PH}"

    # dots + labels
    dots = ""
    for i, (t, v) in enumerate(present):
        dx, dy = _fx(i), _fy(v)
        la = "start" if i == 0 else "end" if i == len(present)-1 else "middle"
        dots += (
            f'<circle cx="{dx:.1f}" cy="{dy:.1f}" r="2.5" fill="{line_col}" opacity=".9"/>'
            f'<text x="{dx:.1f}" y="{H-2}" font-size="7.5" fill="#555960" text-anchor="middle" '
            f'font-family="JetBrains Mono,monospace">{t}</text>'
            f'<text x="{dx:.1f}" y="{dy-6:.1f}" font-size="7" fill="#8890a1" text-anchor="{la}" '
            f'font-family="JetBrains Mono,monospace">{v:.2f}</text>'
        )

    # y-axis tick lines
    grid = ""
    for yv in [mn + span * 0.25, mn + span * 0.5, mn + span * 0.75]:
        gy = _fy(yv)
        grid += f'<line x1="{ML}" y1="{gy:.1f}" x2="{ML+PW}" y2="{gy:.1f}" stroke="#1a1a1a" stroke-width=".6"/>'
        grid += f'<text x="{ML-3}" y="{gy:.1f}" font-size="6.5" fill="#444" text-anchor="end" dominant-baseline="middle" font-family="JetBrains Mono,monospace">{yv:.1f}</text>'

    svg = (
        f'<svg width="{W}" height="{H}" style="display:block;overflow:visible">'
        f'{grid}'
        f'<polygon points="{fill_pts}" fill="{line_col}" opacity=".07"/>'
        f'<polyline points="{pts}" fill="none" stroke="{line_col}" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'{dots}'
        f'</svg>'
    )

    st.markdown(
        f'<div style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#CFB991;padding:.4rem 0 .3rem;'
        f'margin-top:.35rem;border-top:1px solid #1e1e1e">Yield Curve</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;padding:.45rem .55rem;border-radius:2px">'
        f'{svg}'
        f'<div style="display:flex;justify-content:space-between;margin-top:.2rem">'
        f'<span style="{_M}font-size:8.5px;font-weight:700;color:{line_col}">{shape_lbl}</span>'
        f'<span style="{_M}font-size:8px;color:#555960">'
        f'spread {slope:+.2f}% (3M→30Y)</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § R9  VOL REGIME TRIO — VIX / OVX / GVZ gauge bars
# ─────────────────────────────────────────────────────────────────────────────

def _render_vol_trio() -> None:
    """Right column: VIX / OVX / GVZ as gauge bars normalised to 1-year range."""
    vols = _load_vol_trio()
    if not vols:
        return

    _VOL_DESC = {
        "VIX": ("Equity Vol", lambda v: ("CALM",     "#27ae60") if v < 15 else
                                         ("NORMAL",   "#8890a1") if v < 25 else
                                         ("ELEVATED", "#e8a838") if v < 35 else
                                         ("STRESS",   "#c0392b")),
        "OVX": ("Oil Vol",    lambda v: ("LOW",      "#27ae60") if v < 25 else
                                         ("MID",      "#8890a1") if v < 45 else
                                         ("HIGH",     "#e8a838") if v < 65 else
                                         ("EXTREME",  "#c0392b")),
        "GVZ": ("Gold Vol",   lambda v: ("LOW",      "#27ae60") if v < 12 else
                                         ("MID",      "#8890a1") if v < 20 else
                                         ("HIGH",     "#e8a838") if v < 30 else
                                         ("EXTREME",  "#c0392b")),
    }

    st.markdown(
        f'<div style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#CFB991;padding:.4rem 0 .3rem;'
        f'margin-top:.35rem;border-top:1px solid #1e1e1e">Volatility Regime · VIX / OVX / GVZ</div>',
        unsafe_allow_html=True,
    )

    rows_html = ""
    for key in ["VIX", "OVX", "GVZ"]:
        d = vols.get(key)
        if not d:
            continue
        sub_lbl, reg_fn = _VOL_DESC[key]
        reg, rc = reg_fn(d["cur"])
        pct_r   = d["pct_rank"]          # 0..1 position in 1-year range
        bar_w   = int(pct_r * 100)
        lo_str  = f'{d["lo"]:.1f}'
        hi_str  = f'{d["hi"]:.1f}'
        rows_html += (
            f'<div style="padding:.32rem .55rem;border-bottom:1px solid #111">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'
            f'<div>'
            f'<span style="{_M}font-size:9px;font-weight:700;color:#DCE4F0">{key}</span>'
            f'<span style="{_M}font-size:7.5px;color:#555960;margin-left:5px">{sub_lbl}</span>'
            f'</div>'
            f'<div style="text-align:right">'
            f'<span style="{_M}font-size:10px;font-weight:700;color:{rc}">{d["cur"]:.1f}</span>'
            f'<span style="{_M}font-size:7.5px;font-weight:700;color:{rc};'
            f'margin-left:5px;letter-spacing:.06em">{reg}</span>'
            f'</div>'
            f'</div>'
            # gauge track
            f'<div style="position:relative;background:#1a1a1a;height:5px;border-radius:3px">'
            f'<div style="position:absolute;left:0;top:0;width:{bar_w}%;height:5px;'
            f'background:{rc};border-radius:3px;opacity:.8"></div>'
            f'</div>'
            # range labels
            f'<div style="display:flex;justify-content:space-between;margin-top:2px">'
            f'<span style="{_M}font-size:7px;color:#333">{lo_str}</span>'
            f'<span style="{_M}font-size:7px;color:#333">1y range</span>'
            f'<span style="{_M}font-size:7px;color:#333">{hi_str}</span>'
            f'</div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;overflow:hidden">{rows_html}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § L10  THREAT RADAR — polar scatter: conflicts plotted by CIS (radius) × TPS (angle)
# ─────────────────────────────────────────────────────────────────────────────

def _render_threat_radar(conflict_results: dict, risk: dict) -> None:
    """Left column: radar-style polar scatter of active conflicts (CIS=radius, TPS=angle)."""
    import math
    active = [(cid, r) for cid, r in conflict_results.items() if r.get("state") == "active"]
    if not active:
        return

    CX, CY, R = 105, 98, 80
    W, H       = 210, 210

    grs   = float(risk.get("score", 50))
    g_col = "#c0392b" if grs >= 70 else "#e8a838" if grs >= 45 else "#27ae60"

    # ── background ────────────────────────────────────────────────────────────
    # risk zone rings
    bg = (
        f'<circle cx="{CX}" cy="{CY}" r="{R}" fill="#c0392b" opacity=".055"/>'
        f'<circle cx="{CX}" cy="{CY}" r="{int(R*0.65)}" fill="#e8a838" opacity=".07"/>'
        f'<circle cx="{CX}" cy="{CY}" r="{int(R*0.32)}" fill="#27ae60" opacity=".10"/>'
    )
    # ring outlines
    for ri in [R, int(R*0.65), int(R*0.32)]:
        bg += f'<circle cx="{CX}" cy="{CY}" r="{ri}" fill="none" stroke="#222" stroke-width=".7"/>'

    # radial spokes (every 45°)
    spokes = ""
    for deg in range(0, 360, 45):
        rad = math.radians(deg)
        spokes += (
            f'<line x1="{CX}" y1="{CY}" '
            f'x2="{CX + R*math.cos(rad):.1f}" y2="{CY + R*math.sin(rad):.1f}" '
            f'stroke="#1e1e1e" stroke-width=".8"/>'
        )

    # ring labels (CIS values)
    ring_labels = ""
    for pct, label in [(0.32, "25"), (0.65, "50"), (1.0, "75+")]:
        lx = CX + R * pct + 2
        ring_labels += (
            f'<text x="{lx:.1f}" y="{CY-2}" font-size="6.5" fill="#333" '
            f'font-family="JetBrains Mono,monospace">{label}</text>'
        )

    # ── conflict blips ────────────────────────────────────────────────────────
    blips = ""
    pulse_css = ""
    top_conflict = max(active, key=lambda x: x[1].get("cis", 0))

    for cid, r in active:
        cis   = float(r.get("cis", 50))
        tps   = float(r.get("tps", 50))
        color = r.get("color", "#e8a838")
        label = (r.get("label") or cid).replace("_", " ")[:12]

        # angle: TPS maps 0→360, starting from 12 o'clock (−90°)
        theta = math.radians(tps / 100.0 * 360.0 - 90.0)
        rad_r = cis / 100.0 * R
        bx    = CX + rad_r * math.cos(theta)
        by    = CY + rad_r * math.sin(theta)
        dot_r = 4.5 + cis / 100.0 * 5.5

        is_top = (cid == top_conflict[0])

        # glow ring for top conflict
        if is_top:
            blips += (
                f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="{dot_r+5:.1f}" '
                f'fill="none" stroke="{color}" stroke-width="1" opacity=".3" '
                f'class="radar-pulse"/>'
                f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="{dot_r+10:.1f}" '
                f'fill="none" stroke="{color}" stroke-width=".5" opacity=".15" '
                f'class="radar-pulse2"/>'
            )

        blips += (
            f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="{dot_r:.1f}" '
            f'fill="{color}" opacity=".85"/>'
            f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="{dot_r:.1f}" '
            f'fill="none" stroke="{color}" stroke-width="1.2" opacity=".5"/>'
        )

        # label placement: push outward from center
        lrad = rad_r + dot_r + 5
        lx   = CX + lrad * math.cos(theta)
        ly   = CY + lrad * math.sin(theta)
        anchor = "start" if math.cos(theta) >= 0 else "end"
        blips += (
            f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="7" fill="{color}" '
            f'text-anchor="{anchor}" dominant-baseline="middle" '
            f'font-family="DM Sans,sans-serif" font-weight="700">{label[:10]}</text>'
        )

    # ── centre GRS ────────────────────────────────────────────────────────────
    centre = (
        f'<circle cx="{CX}" cy="{CY}" r="18" fill="#0a0a0a" stroke="{g_col}" stroke-width="1.5"/>'
        f'<text x="{CX}" y="{CY-3}" font-size="13" font-weight="700" fill="{g_col}" '
        f'text-anchor="middle" font-family="JetBrains Mono,monospace">{grs:.0f}</text>'
        f'<text x="{CX}" y="{CY+9}" font-size="6" fill="#555960" '
        f'text-anchor="middle" font-family="JetBrains Mono,monospace">GRS</text>'
    )

    # compass labels
    compass = (
        f'<text x="{CX}" y="{CY-R-5}" font-size="7" fill="#333" text-anchor="middle" '
        f'font-family="JetBrains Mono,monospace">TPS 0</text>'
        f'<text x="{CX+R+4}" y="{CY+3}" font-size="7" fill="#333" text-anchor="start" '
        f'font-family="JetBrains Mono,monospace">90</text>'
        f'<text x="{CX}" y="{CY+R+12}" font-size="7" fill="#333" text-anchor="middle" '
        f'font-family="JetBrains Mono,monospace">180</text>'
        f'<text x="{CX-R-4}" y="{CY+3}" font-size="7" fill="#333" text-anchor="end" '
        f'font-family="JetBrains Mono,monospace">270</text>'
    )

    pulse_style = (
        "<style>"
        "@keyframes radar-pulse{0%{opacity:.3;r:10}50%{opacity:.05;r:18}100%{opacity:.3;r:10}}"
        "@keyframes radar-pulse2{0%{opacity:.15;r:18}50%{opacity:.02;r:26}100%{opacity:.15;r:18}}"
        ".radar-pulse{animation:radar-pulse 2s ease-in-out infinite}"
        ".radar-pulse2{animation:radar-pulse2 2s ease-in-out infinite .4s}"
        "</style>"
    )

    svg = (
        f'<svg width="{W}" height="{H}" style="display:block;overflow:visible">'
        f'{pulse_style}'
        f'{bg}{spokes}{ring_labels}'
        f'{blips}{centre}{compass}'
        f'</svg>'
    )

    st.markdown(
        f'<div style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#CFB991;padding:.4rem 0 .3rem;'
        f'margin-top:.35rem;border-top:1px solid #1e1e1e">Threat Radar  ·  CIS × TPS</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="background:#080808;border:1px solid #1e1e1e;'
        f'padding:.4rem .3rem .2rem;border-radius:2px;text-align:center">'
        f'{svg}'
        f'<div style="display:flex;justify-content:center;gap:14px;margin-top:.15rem">'
        f'<span style="{_M}font-size:7px;color:#333">● radius = CIS</span>'
        f'<span style="{_M}font-size:7px;color:#333">● angle = TPS</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# § R10  RISK CONVERGENCE — GRS / Coupling / Vol stress area overlay (60d)
# ─────────────────────────────────────────────────────────────────────────────

def _render_risk_convergence(
    score_hist: "pd.Series | None",
    corr_series: "pd.Series | None",
) -> None:
    """Right column: 3 risk signals as overlapping filled area charts (60 days)."""
    import pandas as _pd, math as _math

    st.markdown(
        f'<div style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
        f'text-transform:uppercase;color:#CFB991;padding:.4rem 0 .3rem;'
        f'margin-top:.35rem;border-top:1px solid #1e1e1e">Risk Signal Convergence · 60d</div>',
        unsafe_allow_html=True,
    )

    W, H  = 210, 90
    ML, MR, MT, MB = 20, 8, 6, 16
    PW    = W - ML - MR
    PH    = H - MT - MB
    N     = 60

    def _norm(s: "_pd.Series", lo: float = 0, hi: float = 100) -> list[float]:
        s = s.dropna().iloc[-N:]
        mn, mx = float(s.min()), float(s.max())
        span   = mx - mn or 1.0
        return [lo + (float(v) - mn) / span * (hi - lo) for v in s]

    def _pts(vals: list[float]) -> str:
        n = len(vals)
        return " ".join(
            f'{ML + i/(n-1)*PW:.1f},{MT + (1 - v/100)*PH:.1f}'
            for i, v in enumerate(vals)
        )

    def _fill(vals: list[float]) -> str:
        n = len(vals)
        base_y = MT + PH
        return (
            f'{ML:.1f},{base_y} '
            + " ".join(f'{ML + i/(n-1)*PW:.1f},{MT + (1 - v/100)*PH:.1f}' for i, v in enumerate(vals))
            + f' {ML + PW:.1f},{base_y}'
        )

    series_data = []

    # Series 1: GRS (0–100 native)
    if isinstance(score_hist, _pd.Series) and len(score_hist.dropna()) >= 10:
        v1 = _norm(score_hist, 0, 100)
        if len(v1) >= 5:
            series_data.append(("GRS", v1, "#c0392b", float(v1[-1])))

    # Series 2: Coupling (avg_corr → 0–100 via percentile rank)
    if isinstance(corr_series, _pd.Series) and len(corr_series.dropna()) >= 10:
        v2 = _norm(corr_series, 0, 100)
        if len(v2) >= 5:
            series_data.append(("COUP", v2, "#2980b9", float(v2[-1])))

    # Series 3: vol stress proxy from pulse (VIX current → flat line as baseline context)
    try:
        pulse = _load_market_pulse()
        vix_d = next((d for d in pulse if d["sym"] == "^VIX"), None)
        if vix_d and len(vix_d.get("series", [])) >= 3:
            vix_s = _pd.Series(vix_d["series"])
            # normalise using 5-day series and scale to 0-100 (VIX 40 = 100)
            vix_norm = [min(v / 40.0 * 100.0, 100.0) for v in vix_d["series"]]
            # pad to N by repeating first value
            while len(vix_norm) < 5:
                vix_norm.insert(0, vix_norm[0])
            series_data.append(("VIX", vix_norm, "#27ae60", float(vix_norm[-1])))
    except Exception:
        pass

    if not series_data:
        st.markdown(
            f'<div style="{_M}font-size:9px;color:#555960;padding:.3rem 0">Signal data unavailable</div>',
            unsafe_allow_html=True,
        )
        return

    # grid lines
    grid = ""
    for pct in [25, 50, 75]:
        gy = MT + (1 - pct/100) * PH
        grid += (
            f'<line x1="{ML}" y1="{gy:.1f}" x2="{ML+PW}" y2="{gy:.1f}" '
            f'stroke="#181818" stroke-width=".7" stroke-dasharray="3,3"/>'
            f'<text x="{ML-3}" y="{gy:.1f}" font-size="6.5" fill="#2e2e2e" '
            f'text-anchor="end" dominant-baseline="middle" '
            f'font-family="JetBrains Mono,monospace">{pct}</text>'
        )

    areas = ""
    lines = ""
    end_dots = ""
    legend_items = ""

    for name, vals, col, cur in series_data:
        n = len(vals)
        pts  = _pts(vals)
        fill = _fill(vals)
        areas += f'<polygon points="{fill}" fill="{col}" opacity=".12"/>'
        lines += (
            f'<polyline points="{pts}" fill="none" stroke="{col}" '
            f'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity=".85"/>'
        )
        ex = ML + PW
        ey = MT + (1 - cur/100) * PH
        end_dots += (
            f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="2.5" fill="{col}"/>'
            f'<text x="{ex+4}" y="{ey:.1f}" font-size="7" fill="{col}" '
            f'dominant-baseline="middle" font-family="JetBrains Mono,monospace">'
            f'{cur:.0f}</text>'
        )
        legend_items += (
            f'<span style="display:inline-flex;align-items:center;gap:3px;margin-right:10px">'
            f'<svg width="12" height="4"><line x1="0" y1="2" x2="12" y2="2" '
            f'stroke="{col}" stroke-width="2"/></svg>'
            f'<span style="{_M}font-size:7.5px;color:{col}">{name}</span>'
            f'</span>'
        )

    # x-axis: just label start and end
    x_labels = (
        f'<text x="{ML}" y="{H-2}" font-size="7" fill="#333" font-family="JetBrains Mono,monospace">−60d</text>'
        f'<text x="{ML+PW}" y="{H-2}" font-size="7" fill="#333" text-anchor="end" font-family="JetBrains Mono,monospace">NOW</text>'
    )

    svg = (
        f'<svg width="{W}" height="{H}" style="display:block;overflow:visible">'
        f'{grid}{areas}{lines}{end_dots}{x_labels}'
        f'</svg>'
    )

    st.markdown(
        f'<div style="background:#080808;border:1px solid #1e1e1e;'
        f'padding:.5rem .55rem .35rem;border-radius:2px">'
        f'{svg}'
        f'<div style="display:flex;flex-wrap:wrap;margin-top:.3rem">{legend_items}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §C1  Asset Correlation Heatmap — center-column mini heatmap
# ─────────────────────────────────────────────────────────────────────────────

def _render_corr_heatmap(eq_r: "pd.DataFrame | None", cmd_r: "pd.DataFrame | None") -> None:
    """60-day rolling correlation heatmap for top equity + commodity pairs."""
    _HDR = (
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
        'font-weight:700;letter-spacing:.1em;color:#8a8a9a;margin-bottom:4px">'
        'ASSET CORRELATION MATRIX · 60D</div>'
    )
    try:
        import numpy as _np
        import pandas as _pd

        _EQ_LABELS  = {"^GSPC": "SPX", "^IXIC": "NDX", "^DJI": "DOW",
                       "^RUT": "RUT", "EEM": "EEM", "GLD": "GLD"}
        _CMD_LABELS = {"CL=F": "WTI", "GC=F": "Gold", "SI=F": "Silver",
                       "NG=F": "NatGas", "ZW=F": "Wheat", "ZC=F": "Corn"}

        frames: list["pd.DataFrame"] = []
        labels: list[str] = []

        if eq_r is not None and not eq_r.empty:
            for col in eq_r.columns[:4]:
                lbl = _EQ_LABELS.get(col, col[:5])
                frames.append(eq_r[col].rename(lbl))
                labels.append(lbl)

        if cmd_r is not None and not cmd_r.empty:
            for col in cmd_r.columns[:4]:
                lbl = _CMD_LABELS.get(col, col[:5])
                frames.append(cmd_r[col].rename(lbl))
                labels.append(lbl)

        if len(frames) < 2:
            raise ValueError("insufficient data")

        combined = _pd.concat(frames, axis=1).dropna()
        if len(combined) > 60:
            combined = combined.iloc[-60:]
        corr = combined.corr()
        n    = len(corr)

        CELL = 28
        PAD  = 52
        W    = PAD + n * CELL
        H    = PAD + n * CELL

        def _corr_color(v: float) -> str:
            if v >= 0.7:  return "#c0392b"
            if v >= 0.4:  return "#e67e22"
            if v >= 0.1:  return "#f1c40f"
            if v >= -0.1: return "#7f8c8d"
            if v >= -0.4: return "#2980b9"
            return "#1a5276"

        cells = ""
        for i, ri in enumerate(corr.index):
            for j, ci in enumerate(corr.columns):
                v   = corr.loc[ri, ci]
                col = _corr_color(v)
                x   = PAD + j * CELL
                y   = PAD + i * CELL
                op  = 0.25 if i == j else max(0.3, abs(v) * 0.9)
                txt = "1.0" if i == j else f"{v:+.2f}"[1:] if abs(v) >= 0.1 else f"{v:+.2f}"
                cells += (
                    f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                    f'fill="{col}" opacity="{op:.2f}" rx="2"/>'
                    f'<text x="{x + CELL//2}" y="{y + CELL//2 + 4}" '
                    f'font-family="JetBrains Mono,monospace" font-size="7.5" '
                    f'fill="white" text-anchor="middle">{txt}</text>'
                )

        row_labels = ""
        col_labels = ""
        for i, lbl in enumerate(corr.index):
            row_labels += (
                f'<text x="{PAD - 4}" y="{PAD + i * CELL + CELL//2 + 4}" '
                f'font-family="JetBrains Mono,monospace" font-size="8" '
                f'fill="#a0a0b0" text-anchor="end">{lbl}</text>'
            )
            col_labels += (
                f'<text x="{PAD + i * CELL + CELL//2}" y="{PAD - 6}" '
                f'font-family="JetBrains Mono,monospace" font-size="8" '
                f'fill="#a0a0b0" text-anchor="middle">{lbl}</text>'
            )

        legend = (
            '<div style="display:flex;gap:8px;margin-top:4px;flex-wrap:wrap">'
            + "".join(
                f'<span style="font-family:JetBrains Mono,monospace;font-size:8px;'
                f'color:#a0a0b0"><span style="display:inline-block;width:8px;height:8px;'
                f'background:{c};border-radius:1px;margin-right:2px;vertical-align:middle">'
                f'</span>{lbl}</span>'
                for c, lbl in [
                    ("#c0392b","≥0.7"), ("#e67e22","0.4–0.7"),
                    ("#f1c40f","0.1–0.4"), ("#7f8c8d","-0.1–0.1"),
                    ("#2980b9","-0.4–-0.1"), ("#1a5276","≤-0.4"),
                ]
            )
            + "</div>"
        )

        svg = (
            f'<svg width="100%" viewBox="0 0 {W} {H}" '
            f'xmlns="http://www.w3.org/2000/svg">'
            f'{cells}{row_labels}{col_labels}'
            f'</svg>'
        )
        st.markdown(
            _HDR +
            '<div style="background:#0a0a14;border:1px solid #1e1e2e;'
            'padding:.5rem .55rem .35rem;border-radius:2px">'
            + svg + legend + "</div>",
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            _HDR +
            '<div style="background:#0a0a14;border:1px solid #1e1e2e;'
            'padding:.5rem;border-radius:2px;color:#555;font-family:JetBrains Mono,monospace;'
            'font-size:10px">Insufficient data for correlation matrix</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# §C2  Risk Signal Waterfall — multi-model signal stack
# ─────────────────────────────────────────────────────────────────────────────

def _render_risk_signal_waterfall(
    risk: dict,
    conflict_results,
    regimes: "pd.Series | None",
    alerts: list,
) -> None:
    """Stacked signal bars from all analytical layers — geo / regime / vol / FI."""
    _HDR = (
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
        'font-weight:700;letter-spacing:.1em;color:#8a8a9a;margin-bottom:4px">'
        'CROSS-MODEL SIGNAL STACK</div>'
    )
    try:
        import numpy as _np
        if isinstance(conflict_results, dict):
            conflict_results = list(conflict_results.values())

        # Gather signals from available data
        signals: list[tuple[str, float, str, str]] = []  # (label, 0-100, color, note)

        # 1. Geo Risk Score
        grs = float(risk.get("score", 0))
        grs_col = "#c0392b" if grs >= 70 else "#e67e22" if grs >= 45 else "#27ae60"
        signals.append(("GEO RISK", grs, grs_col, f"{grs:.0f}/100"))

        # 2. Conflict Intensity Score
        cis = float(risk.get("cis", 0))
        cis_col = "#c0392b" if cis >= 70 else "#e67e22" if cis >= 45 else "#27ae60"
        signals.append(("CONFLICT INTENSITY", cis, cis_col, f"{cis:.0f}/100"))

        # 3. Transmission Pressure
        tps = float(risk.get("tps", 0))
        tps_col = "#c0392b" if tps >= 70 else "#e67e22" if tps >= 45 else "#2980b9"
        signals.append(("TRANSMISSION PRESSURE", tps, tps_col, f"{tps:.0f}/100"))

        # 4. Market Coupling Score
        mcs = float(risk.get("mcs", 0))
        mcs_col = "#c0392b" if mcs >= 70 else "#e67e22" if mcs >= 45 else "#27ae60"
        signals.append(("MARKET COUPLING", mcs, mcs_col, f"{mcs:.0f}/100"))

        # 5. Correlation regime (1=Decoupled, 2=Transitioning, 3=High Coupling)
        if regimes is not None and not regimes.empty:
            reg = int(regimes.dropna().iloc[-1]) if len(regimes.dropna()) > 0 else 1
            reg_pct = (reg - 1) / 2 * 100
            reg_col = "#27ae60" if reg == 1 else "#e8a838" if reg == 2 else "#c0392b"
            reg_lbl = ["DECOUPLED", "TRANSITIONING", "HIGH COUPLING"][reg - 1]
            signals.append(("REGIME", reg_pct, reg_col, reg_lbl))

        # 6. Active critical alerts pressure
        crit = len([a for a in alerts if getattr(a, "severity", "") == "critical"])
        high = len([a for a in alerts if getattr(a, "severity", "") == "high"])
        alert_pct = min((crit * 25 + high * 10), 100)
        alert_col = "#c0392b" if crit >= 3 else "#e67e22" if crit >= 1 else "#27ae60"
        signals.append(("ALERT PRESSURE", alert_pct, alert_col, f"{crit}c {high}h alerts"))

        # 7. Active conflicts pressure
        n_conflicts = len([c for c in conflict_results if c.get("active", True)])
        conf_pct = min(n_conflicts / 10 * 100, 100)
        conf_col = "#c0392b" if n_conflicts >= 6 else "#e67e22" if n_conflicts >= 3 else "#27ae60"
        signals.append(("ACTIVE CONFLICTS", conf_pct, conf_col, f"{n_conflicts} tracked"))

        # Render SVG waterfall
        BAR_H  = 12
        GAP    = 6
        LABEL_W = 120
        BAR_W   = 120
        PAD_Y   = 6
        H = PAD_Y * 2 + len(signals) * (BAR_H + GAP) - GAP
        W = LABEL_W + BAR_W + 45

        bars = ""
        for i, (lbl, pct, col, note) in enumerate(signals):
            y    = PAD_Y + i * (BAR_H + GAP)
            bw   = max(pct / 100 * BAR_W, 2)
            bars += (
                # label
                f'<text x="{LABEL_W - 6}" y="{y + BAR_H//2 + 4}" '
                f'font-family="JetBrains Mono,monospace" font-size="8" '
                f'fill="#8a8a9a" text-anchor="end">{lbl}</text>'
                # track
                f'<rect x="{LABEL_W}" y="{y}" width="{BAR_W}" height="{BAR_H}" '
                f'fill="#1a1a2a" rx="2"/>'
                # bar
                f'<rect x="{LABEL_W}" y="{y}" width="{bw:.1f}" height="{BAR_H}" '
                f'fill="{col}" opacity="0.85" rx="2"/>'
                # note
                f'<text x="{LABEL_W + BAR_W + 5}" y="{y + BAR_H//2 + 4}" '
                f'font-family="JetBrains Mono,monospace" font-size="8" '
                f'fill="{col}">{note}</text>'
            )

        svg = (
            f'<svg width="100%" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">'
            f'{bars}</svg>'
        )
        st.markdown(
            _HDR +
            '<div style="background:#0a0a14;border:1px solid #1e1e2e;'
            'padding:.5rem .55rem .35rem;border-radius:2px">'
            + svg + "</div>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# §C3  Conflict × Commodity Impact Matrix
# ─────────────────────────────────────────────────────────────────────────────

def _render_conflict_commodity_matrix(conflict_results) -> None:
    """Grid of active conflicts × commodity groups with impact intensity cells."""
    _HDR = (
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
        'font-weight:700;letter-spacing:.1em;color:#8a8a9a;margin-bottom:4px">'
        'CONFLICT × COMMODITY IMPACT MATRIX</div>'
    )
    try:
        if isinstance(conflict_results, dict):
            conflict_results = [{"name": k, **v} for k, v in conflict_results.items()]
        GROUPS = ["ENERGY", "METALS", "AGRI", "FI/FX"]
        # Map commodity names to groups
        _G = {
            "WTI Crude Oil": "ENERGY", "Brent Crude": "ENERGY",
            "Natural Gas": "ENERGY", "Heating Oil": "ENERGY",
            "Gold": "METALS", "Silver": "METALS", "Copper": "METALS",
            "Platinum": "METALS", "Palladium": "METALS",
            "Wheat": "AGRI", "Corn": "AGRI", "Soybeans": "AGRI",
            "Sugar": "AGRI", "Cotton": "AGRI",
        }

        active = [c for c in conflict_results if c.get("cis", 0) > 10][:8]
        if not active:
            raise ValueError("no conflicts")

        # Build matrix: conflict × group → max exposure score
        import numpy as _np
        matrix = _np.zeros((len(active), len(GROUPS)))
        for i, c in enumerate(active):
            for exp in c.get("commodity_exposures", []):
                cname = exp.get("commodity", "")
                score = float(exp.get("exposure_score", 0))
                g = _G.get(cname)
                if g and g in GROUPS:
                    j = GROUPS.index(g)
                    matrix[i, j] = max(matrix[i, j], score * float(c.get("cis", 50)) / 100)

        CELL_W = 48
        CELL_H = 20
        LABEL_W = 100
        HDR_H   = 22
        W = LABEL_W + len(GROUPS) * CELL_W + 4
        H = HDR_H + len(active) * CELL_H + 4

        def _cell_col(v: float) -> tuple[str, float]:
            if v >= 60: return "#c0392b", 0.9
            if v >= 40: return "#e67e22", 0.8
            if v >= 20: return "#e8a838", 0.7
            if v >  5:  return "#2980b9", 0.5
            return "#1a1a2a", 1.0

        cells = ""
        # Column headers
        for j, g in enumerate(GROUPS):
            x = LABEL_W + j * CELL_W
            cells += (
                f'<rect x="{x}" y="0" width="{CELL_W}" height="{HDR_H}" '
                f'fill="#1e2a3a" rx="0"/>'
                f'<text x="{x + CELL_W//2}" y="{HDR_H//2 + 4}" '
                f'font-family="JetBrains Mono,monospace" font-size="8" '
                f'fill="#CFB991" text-anchor="middle" font-weight="700">{g}</text>'
            )
        # Rows
        for i, c in enumerate(active):
            y = HDR_H + i * CELL_H
            name = c.get("name", f"Conflict {i+1}").replace("_", " ")
            short = name[:12] + "…" if len(name) > 12 else name
            cis_v = float(c.get("cis", 0))
            cis_col = "#c0392b" if cis_v >= 70 else "#e67e22" if cis_v >= 45 else "#8a8a9a"
            bg = "#0e1420" if i % 2 == 0 else "#0a1018"
            cells += (
                f'<rect x="0" y="{y}" width="{W}" height="{CELL_H}" fill="{bg}"/>'
                f'<text x="4" y="{y + CELL_H//2 + 4}" '
                f'font-family="JetBrains Mono,monospace" font-size="8" '
                f'fill="#c0c0d0">{short}</text>'
                f'<text x="{LABEL_W - 5}" y="{y + CELL_H//2 + 4}" '
                f'font-family="JetBrains Mono,monospace" font-size="8" '
                f'fill="{cis_col}" text-anchor="end">{cis_v:.0f}</text>'
            )
            for j in range(len(GROUPS)):
                v = matrix[i, j]
                col, op = _cell_col(v)
                x = LABEL_W + j * CELL_W
                txt = f"{v:.0f}" if v > 5 else "·"
                cells += (
                    f'<rect x="{x+1}" y="{y+1}" width="{CELL_W-2}" height="{CELL_H-2}" '
                    f'fill="{col}" opacity="{op:.2f}" rx="2"/>'
                    f'<text x="{x + CELL_W//2}" y="{y + CELL_H//2 + 4}" '
                    f'font-family="JetBrains Mono,monospace" font-size="8.5" '
                    f'fill="white" text-anchor="middle" font-weight="700">{txt}</text>'
                )

        svg = (
            f'<svg width="100%" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">'
            f'{cells}</svg>'
        )
        note = (
            '<div style="font-family:JetBrains Mono,monospace;font-size:8px;'
            'color:#555;margin-top:3px">cell value = exposure × CIS/100 · '
            '<span style="color:#c0392b">■</span> ≥60 '
            '<span style="color:#e67e22">■</span> 40-60 '
            '<span style="color:#e8a838">■</span> 20-40 '
            '<span style="color:#2980b9">■</span> 5-20</div>'
        )
        st.markdown(
            _HDR +
            '<div style="background:#0a0a14;border:1px solid #1e1e2e;'
            'padding:.5rem .55rem .35rem;border-radius:2px">'
            + svg + note + "</div>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# §C4  Regime × Signal Heatmap Ticker — scrolling ticker of regime-signal pairs
# ─────────────────────────────────────────────────────────────────────────────

def _render_geo_event_timeline(conflict_results) -> None:
    """Horizontal timeline of conflict severity with intensity gradient bars."""
    _HDR = (
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
        'font-weight:700;letter-spacing:.1em;color:#8a8a9a;margin-bottom:4px">'
        'CONFLICT SEVERITY TIMELINE</div>'
    )
    try:
        if isinstance(conflict_results, dict):
            conflict_results = [{"name": k, **v} for k, v in conflict_results.items()]
        active = sorted(
            [c for c in conflict_results if c.get("cis", 0) > 5],
            key=lambda x: -x.get("cis", 0),
        )[:10]
        if not active:
            raise ValueError("no data")

        W, H = 310, len(active) * 22 + 20
        BAR_MAX = 170
        LABEL_W = 90
        SCORE_W = 28

        bars = (
            f'<text x="{LABEL_W + BAR_MAX//2}" y="12" font-family="JetBrains Mono,monospace" '
            f'font-size="7" fill="#555" text-anchor="middle">CIS ← → TPS</text>'
        )
        ROW_H = 22
        for i, c in enumerate(active):
            y    = 16 + i * ROW_H
            cis  = float(c.get("cis", 0))
            tps  = float(c.get("tps", 0))
            name = c.get("name", "").replace("_", " ")
            short = name[:11] + "…" if len(name) > 11 else name
            bw   = cis / 100 * BAR_MAX

            col1 = "#c0392b" if cis >= 70 else "#e67e22" if cis >= 45 else "#e8a838"
            tps_x = LABEL_W + tps / 100 * BAR_MAX
            bg = "#0e1420" if i % 2 == 0 else "#0a1018"
            bars += (
                f'<rect x="0" y="{y}" width="{W}" height="{ROW_H}" fill="{bg}"/>'
                f'<text x="{LABEL_W - 4}" y="{y + 13}" '
                f'font-family="JetBrains Mono,monospace" font-size="7.5" '
                f'fill="#c0c0d0" text-anchor="end">{short}</text>'
                f'<rect x="{LABEL_W}" y="{y + 5}" width="{BAR_MAX}" height="10" '
                f'fill="#1a1a2a" rx="2"/>'
                f'<rect x="{LABEL_W}" y="{y + 5}" width="{bw:.1f}" height="10" '
                f'fill="{col1}" opacity="0.8" rx="2"/>'
                f'<line x1="{tps_x:.1f}" y1="{y + 3}" x2="{tps_x:.1f}" y2="{y + 17}" '
                f'stroke="#CFB991" stroke-width="1.5" opacity="0.9"/>'
                f'<text x="{LABEL_W + BAR_MAX + 4}" y="{y + 13}" '
                f'font-family="JetBrains Mono,monospace" font-size="7.5" '
                f'fill="{col1}">{cis:.0f}</text>'
            )

        legend = (
            '<div style="font-family:JetBrains Mono,monospace;font-size:8px;'
            'color:#555;margin-top:3px">'
            '■ bar = CIS (conflict intensity) · '
            '<span style="color:#CFB991">│</span> = TPS (transmission pressure)'
            '</div>'
        )
        svg = (
            f'<svg width="100%" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">'
            f'{bars}</svg>'
        )
        st.markdown(
            _HDR +
            '<div style="background:#0a0a14;border:1px solid #1e1e2e;'
            'padding:.5rem .55rem .35rem;border-radius:2px">'
            + svg + legend + "</div>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────

def page_home(start: str, end: str, fred_key: str = "") -> None:
    st.markdown(_STYLE, unsafe_allow_html=True)
    init_agents()
    init_scenario()

    # ── Parallel data load ────────────────────────────────────────────────
    # score_all_conflicts (GDELT HTTP) and _load_market_pulse (yfinance 6-ticker)
    # run in background threads while _load_market_risk (yfinance 30+ tickers,
    # has st.session_state write) runs on the main thread. This overlaps the two
    # largest I/O operations and cuts cold-start time by ~50%.
    from concurrent.futures import ThreadPoolExecutor

    with st.spinner("Loading market data & intelligence…"):
        _scenario_id = get_scenario_id()
        with ThreadPoolExecutor(max_workers=2) as _pool:
            _f_conflict = _pool.submit(score_all_conflicts)
            _f_pulse    = _pool.submit(_load_market_pulse)
            # Main thread: market risk (contains st.session_state write — must not thread)
            risk, _score_hist = _load_market_risk(start, end, _scenario_id)

        # Collect thread results after main-thread work completes
        try:
            conflict_results = _f_conflict.result()
            record_fetch("conflict_model")
        except Exception:
            conflict_results = {}
        try:
            _f_pulse.result()   # cache is now warm; result consumed by _render_market_pulse_cards
        except Exception:
            pass

    conflict_agg = aggregate_portfolio_scores(conflict_results)
    if not conflict_agg:
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

    # Apply conflict-model fallback if market data was unavailable.
    if risk.get("_market_fallback") or not risk.get("score"):
        cis_f = conflict_agg.get("portfolio_cis", conflict_agg.get("cis", 50.0))
        tps_f = conflict_agg.get("portfolio_tps", conflict_agg.get("tps", 50.0))
        raw   = round(0.40 * cis_f + 0.35 * tps_f + 0.25 * 50, 1)
        if raw < 25:   lbl, col = "Low",      "#2e7d32"
        elif raw < 50: lbl, col = "Moderate", "#8E9AAA"
        elif raw < 75: lbl, col = "Elevated", "#e67e22"
        else:          lbl, col = "High",     "#c0392b"
        _fallback_at = risk.get("_computed_at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        risk = {
            "score": raw, "label": lbl, "color": col,
            "cis": cis_f, "tps": tps_f, "mcs": 50.0,
            "confidence": conflict_agg.get("confidence", 0.5),
            "top_conflict": conflict_agg.get("top_conflict"),
            "news_gpr": None, "n_threat": 0, "n_act": 0,
            "mcs_components": {}, "components": {}, "weights": {},
            "conflict_detail": {},
            "_market_fallback": True,
            "_computed_at": _fallback_at,
            "_is_eod": None,
            "_data_date": None,
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
    # PRE-RENDER: compute alerts before column split so all columns can use them
    # ══════════════════════════════════════════════════════════════════════
    _cached_alerts: list = st.session_state.get("_cached_alerts", [])
    _al_regimes    = None
    _al_regime_insuf = False
    _al_corr       = None
    try:
        from src.analysis.proactive_alerts import compute_alerts
        from src.analysis.correlations import detect_correlation_regime
        _al_eq_r, _al_cmd_r = load_returns(start, end)
        if not _al_eq_r.empty and not _al_cmd_r.empty:
            _al_corr         = average_cross_corr_series(_al_eq_r, _al_cmd_r, window=60)
            _al_regimes      = detect_correlation_regime(_al_corr)
            _al_regime_insuf = bool(_al_regimes.attrs.get("insufficient_data", False))
            _cached_alerts   = compute_alerts(
                eq_r=_al_eq_r, cmd_r=_al_cmd_r,
                avg_corr=_al_corr, regimes=_al_regimes,
                risk_score=float(risk["score"]),
                risk_history=_score_hist if isinstance(_score_hist, pd.Series)
                             else pd.Series(dtype=float),
            )
            st.session_state["_cached_alerts"] = _cached_alerts
    except Exception:
        pass

    # ══════════════════════════════════════════════════════════════════════
    # RENDER SECTIONS
    # ══════════════════════════════════════════════════════════════════════

    # § 0.5  Data Health Banner
    try:
        from src.analysis.freshness import data_health_html
        _dh_html = data_health_html()
        if _dh_html:
            st.markdown(_dh_html, unsafe_allow_html=True)
    except Exception:
        pass

    # § 1  Masthead — full-width above 3-col
    _render_masthead(conflict_agg)

    # ── 3-col layout: intel feed | dominant gauge | market pulse cards ────────
    # Column heights are balanced by design:
    #   Left  = intelligence feed + scenario switch
    #   Center = market pulse strip + geo risk block (gauge + history + decomp)
    #   Right  = market pulse cards + where-to-go-now recommendations
    # Context narrative + intel panel + morning briefing go BELOW as full-width.
    # ─────────────────────────────────────────────────────────────────────────
    _col_left, _col_ctr, _col_right = st.columns([1.0, 2.2, 1.0], gap="medium")

    with _col_left:
        # § L1  Intelligence feed — live alerts + morning briefing + chokepoint watch
        _render_intelligence_feed(risk, conflict_results, alerts=_cached_alerts)
        # § L2  Threat radar — visually striking showpiece directly under intel feed
        #        polar scatter: CIS radius × TPS angle, animated pulse on top conflict
        try:
            _render_threat_radar(conflict_results, risk)
        except Exception:
            pass
        # § L3  Correlation pulse — 60-day equity↔commodity sparkline + regime badge
        _render_correlation_pulse(_al_corr, _al_regimes)
        # § L4  Conflict landscape — CIS×TPS 2-D scatter of all tracked conflicts
        _render_conflict_landscape(conflict_results)
        # § L5  Escalation tracker — trend/escalation velocity per active conflict
        _render_escalation_tracker(conflict_results)
        # § L6  Top commodities — exposure-weighted commodity risk ranking
        _render_top_commodities(conflict_results)
        # § L7  Regime history — 60-day day-by-day correlation regime colour strip
        _render_regime_history(_al_regimes)
        # § L8  Alert summary — severity breakdown of active proactive alerts
        _render_alert_summary(_cached_alerts)
        # § L9  Commodity sector returns — Energy / Metals / Agri 5-day performance
        try:
            _render_commodity_sector_returns(_al_cmd_r)
        except Exception:
            pass
        # § L10 Lead-lag bars — CMD→EQ cross-correlation at lags 0–5d
        try:
            _render_cross_corr_lag(_al_eq_r, _al_cmd_r)
        except Exception:
            pass

    with _col_ctr:
        # Market pulse horizontal strip
        _render_market_pulse()
        # Portfolio pulse (conditional — hidden unless CSV uploaded)
        _render_portfolio_pulse()
        # Geo risk block: gauge + history chart + decomposition — the CENTERPIECE
        _render_geo_risk_block(risk, conflict_agg, conflict_results, _score_hist)
        # Critical alert banner (inline, no empty space when no alerts)
        try:
            from src.ui.alert_banner import render_alert_banner
            _critical = [a for a in _cached_alerts if a.severity == "critical"]
            if _critical and _al_regimes is not None:
                render_alert_banner(_critical, market_context=(
                    f"Geo risk {risk['score']:.0f}/100 ({risk['label']}). "
                    f"CIS {risk['cis']:.0f} · TPS {risk['tps']:.0f}. "
                    f"Regime: {_al_regimes.iloc[-1] if not _al_regimes.empty else 1}/3"
                    f"{' [INSUF DATA]' if _al_regime_insuf else ''}. "
                    f"Lead conflict: {(conflict_agg.get('top_conflict') or 'none').replace('_',' ')}."
                ))
        except Exception:
            pass
        # Transmission lag signal (only shown when active)
        try:
            _lag = transmission_lag_signal(_al_cmd_r, _al_eq_r)
            if _lag["active"]:
                _lc = "#e8a838" if _lag["lag_signal"] == "In progress" else "#e74c3c"
                _li = "⏳" if _lag["lag_signal"] == "In progress" else "⚡"
                st.markdown(
                    f'<div style="background:#0d0c04;border-left:3px solid {_lc};'
                    f'border:1px solid {_lc}33;padding:8px 14px;margin:6px 0">'
                    f'<span style="color:{_lc};{_M}font-size:11px;font-weight:700;'
                    f'letter-spacing:.08em">{_li} TRANSMISSION LAG · {_lag["lag_signal"].upper()}</span><br>'
                    f'<span style="color:#b0b0b0;{_M}font-size:11px">{_lag["detail"]}</span></div>',
                    unsafe_allow_html=True,
                )
        except Exception:
            pass
        # § C1–C4  2-column sub-grid: each panel ~half the center width (~340px)
        st.markdown('<hr class="hm-rule" style="margin:.5rem 0">', unsafe_allow_html=True)
        _cc_l, _cc_r = st.columns(2, gap="small")
        with _cc_l:
            # § C1  Asset Correlation Heatmap
            try:
                _render_corr_heatmap(_al_eq_r, _al_cmd_r)
            except Exception:
                pass
            st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
            # § C3  Conflict × Commodity Impact Matrix
            try:
                _render_conflict_commodity_matrix(conflict_results)
            except Exception:
                pass
        with _cc_r:
            # § C2  Cross-Model Signal Waterfall
            try:
                _render_risk_signal_waterfall(risk, conflict_results, _al_regimes, _cached_alerts)
            except Exception:
                pass
            st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
            # § C4  Conflict Severity Timeline — CIS bar + TPS overlay per conflict
            try:
                _render_geo_event_timeline(conflict_results)
            except Exception:
                pass

    with _col_right:
        # § R1  Market pulse cards — 6 live instrument cards with sparklines
        _render_market_pulse_cards()
        # § R2  Risk arc — GRS component decomposition bars (CIS/TPS/MCS)
        st.markdown('<hr class="hm-rule" style="margin:.5rem 0">', unsafe_allow_html=True)
        _render_risk_arc(risk)
        # § R3  Risk convergence — showpiece directly under risk arc:
        #        60d overlapping area chart of GRS / coupling / VIX stress signals
        st.markdown('<hr class="hm-rule" style="margin:.5rem 0">', unsafe_allow_html=True)
        try:
            _render_risk_convergence(_score_hist, _al_corr)
        except Exception:
            pass
        # § R4  Next action — routing recommendation based on dominant risk driver
        st.markdown('<hr class="hm-rule" style="margin:.5rem 0">', unsafe_allow_html=True)
        _render_next_action(conflict_agg, conflict_results, compact=True)
        # § R5  Risk compass — 5-axis radar (CIS, TPS, MCS, Volatility, Coupling)
        st.markdown('<hr class="hm-rule" style="margin:.5rem 0">', unsafe_allow_html=True)
        _corr_cur = (
            float(_al_corr.dropna().iloc[-1])
            if _al_corr is not None and len(_al_corr.dropna()) >= 1
            else None
        )
        _render_risk_compass(risk, corr_val=_corr_cur)
        # § R6  Returns heatmap — 5-day day-over-day asset return grid
        st.markdown('<hr class="hm-rule" style="margin:.5rem 0">', unsafe_allow_html=True)
        _render_returns_heatmap()
        # § R7  Transmission channels — CIS-weighted channel pressure breakdown
        st.markdown('<hr class="hm-rule" style="margin:.5rem 0">', unsafe_allow_html=True)
        _render_transmission_channels(conflict_results, risk)
        # § R8  GRS trend — 60-day composite risk score sparkline with zone bands
        st.markdown('<hr class="hm-rule" style="margin:.5rem 0">', unsafe_allow_html=True)
        _render_grs_trend(_score_hist if isinstance(_score_hist, __import__("pandas").Series) else None)
        # § R9  Macro snapshot — VIX / 10Y / DXY / WTI 2×2 regime grid
        st.markdown('<hr class="hm-rule" style="margin:.5rem 0">', unsafe_allow_html=True)
        _render_macro_snapshot()
        # § R10 Yield curve — 3M / 5Y / 10Y / 30Y shape (normal / inverted / flat)
        st.markdown('<hr class="hm-rule" style="margin:.5rem 0">', unsafe_allow_html=True)
        _render_yield_curve_snap()
        # § R11 Vol regime trio — VIX / OVX / GVZ gauge bars vs 1-year range
        st.markdown('<hr class="hm-rule" style="margin:.5rem 0">', unsafe_allow_html=True)
        _render_vol_trio()

    # ── Full-width below 3-col ────────────────────────────────────────────────
    st.markdown('<hr class="hm-rule" style="margin:.4rem 0 .2rem">', unsafe_allow_html=True)

    # Scenario switch — full-width, 8 buttons in one row
    _render_scenario_switch()

    # Context narrative + Intel panel — collapsed by default to save space
    with st.expander("Context & Intelligence  ·  Narrative · Conflict Monitor · Channels", expanded=False):
        _col_ctx, _col_intel = st.columns([1.2, 1.0], gap="medium")
        with _col_ctx:
            _render_context_narrative(risk, conflict_results)
        with _col_intel:
            _render_intel_panel(conflict_results)

    # Morning Briefing Chain — full-width expander
    try:
        from src.ui.agent_panel import render_morning_briefing_panel
        _top_texts    = [getattr(a, "title", "") for a in _cached_alerts[:3] if getattr(a, "title", "")]
        _top_conflict = conflict_agg.get("top_conflict")
        _risk_val     = float(risk["score"])
        _expanded     = _risk_val >= 50
        _panel_label  = (
            f"⚡ AI Analyst Team · Morning Briefing (Risk {_risk_val:.0f}/100)"
            if _expanded else
            f"AI Analyst Team · Morning Briefing (Risk {_risk_val:.0f}/100)"
        )
        with st.expander(_panel_label, expanded=_expanded):
            render_morning_briefing_panel(
                risk_score=_risk_val,
                top_alerts=_top_texts,
                top_conflict=_top_conflict,
                auto_run=True,
                start=start,
                end=end,
            )
    except Exception:
        pass

    # Navigate Terminal — collapsed by default; open when switching pages
    with st.expander("Navigate Terminal  ·  14 modules", expanded=False):
        _render_quickjump()

    # AI Agent Activity strip (optional — only shown when agents are active)
    _render_agent_strip()

    _page_footer()
