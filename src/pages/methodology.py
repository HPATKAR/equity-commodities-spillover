"""
Model Methodology & Documentation Page.

Comprehensive technical documentation for all analytical modules in the
Cross-Asset Spillover Monitor. Covers data sources, formulae, assumptions,
implementation choices, and limitations - following industry model-documentation
standards (SR 11-7 / OCC 2011-12 model risk management guidance).
"""

from __future__ import annotations

import streamlit as st
from src.ui.shared import (
    _page_header, _page_footer, _page_intro,
    _definition_block, _takeaway_block, _section_note,
    _h2, _h3,
)

_M  = "font-family:'JetBrains Mono',monospace;"
_S  = "font-family:'DM Sans',sans-serif;"
_G  = "#CFB991"
_DIM = "#8E9AAA"
_MUT = "#555960"

def _prose(text: str) -> None:
    st.markdown(
        f'<p style="{_S}font-size:0.75rem;color:#b8b8b8;line-height:1.75;'
        f'margin:.2rem 0 .5rem">{text}</p>',
        unsafe_allow_html=True,
    )

def _formula(latex_text: str, caption: str = "") -> None:
    """Render a formula block in monospace with optional caption."""
    st.markdown(
        f'<div style="background:#080808;border:1px solid #1e1e1e;'
        f'border-left:3px solid {_G};padding:.6rem 1rem;margin:.4rem 0 .3rem">'
        f'<code style="{_M}font-size:0.69rem;color:{_G};white-space:pre-wrap">'
        f'{latex_text}</code>'
        + (f'<br><span style="{_S}font-size:0.62rem;color:{_MUT}">{caption}</span>' if caption else "")
        + f'</div>',
        unsafe_allow_html=True,
    )

def _weight_table(rows: list[tuple[str, str, float, str]]) -> None:
    """
    Render a compact weight table.
    rows: [(dimension_key, display_name, weight, note)]
    """
    header = (
        f'<table style="width:100%;border-collapse:collapse;{_S}font-size:0.70rem">'
        f'<thead><tr>'
        f'<th style="{_M}font-size:0.50rem;color:{_MUT};text-align:left;border-bottom:1px solid #2a2a2a;padding:3px 8px">DIMENSION / CHANNEL</th>'
        f'<th style="{_M}font-size:0.50rem;color:{_MUT};text-align:right;border-bottom:1px solid #2a2a2a;padding:3px 8px">WEIGHT</th>'
        f'<th style="{_M}font-size:0.50rem;color:{_MUT};text-align:left;border-bottom:1px solid #2a2a2a;padding:3px 8px">RATIONALE</th>'
        f'</tr></thead><tbody>'
    )
    rows_html = ""
    for _, name, weight, note in rows:
        bar = int(weight / max(r[2] for r in rows) * 60)
        rows_html += (
            f'<tr style="border-bottom:1px solid #111">'
            f'<td style="color:#e8e9ed;padding:4px 8px">{name}</td>'
            f'<td style="text-align:right;padding:4px 8px">'
            f'<span style="{_M}color:{_G};font-weight:700">{weight:.0%}</span>'
            f'<div style="background:#1a1a1a;height:2px;margin-top:2px;width:80px;margin-left:auto">'
            f'<div style="background:{_G};width:{bar}px;height:2px"></div></div>'
            f'</td>'
            f'<td style="color:{_DIM};font-size:0.65rem;padding:4px 8px">{note}</td>'
            f'</tr>'
        )
    st.markdown(header + rows_html + '</tbody></table>', unsafe_allow_html=True)

def _signal_card(
    key: str, name: str, weight: float, color: str,
    tagline: str, formula_lines: str, note: str = "",
) -> None:
    """Compact signal card: weight badge + name + tagline + inline formula block."""
    pct = f"{weight:.0%}"
    st.markdown(
        f'<div style="background:#080808;border:1px solid #1e1e1e;'
        f'border-top:2px solid {color};padding:.55rem .75rem;height:100%">'
        # header row
        f'<div style="display:flex;align-items:center;gap:7px;margin-bottom:.35rem">'
        f'<span style="{_M}font-size:0.56rem;font-weight:700;color:#0a0a0a;'
        f'background:{color};padding:1px 5px;flex-shrink:0">{pct}</span>'
        f'<span style="{_M}font-size:0.50rem;font-weight:700;color:#e8e9ed;'
        f'letter-spacing:.08em;text-transform:uppercase">{name}</span>'
        f'</div>'
        # tagline
        f'<p style="{_S}font-size:0.67rem;color:{_DIM};margin:0 0 .35rem;line-height:1.5">'
        f'{tagline}</p>'
        # formula block
        f'<div style="background:#040404;border-left:2px solid {color};'
        f'padding:.3rem .5rem;margin:.2rem 0">'
        f'<code style="{_M}font-size:0.53rem;color:{color};white-space:pre-wrap;line-height:1.6">'
        f'{formula_lines}</code></div>'
        + (f'<p style="{_S}font-size:0.60rem;color:{_MUT};margin:.3rem 0 0;'
           f'line-height:1.4;font-style:italic">{note}</p>' if note else "")
        + '</div>',
        unsafe_allow_html=True,
    )


def _arch_flow() -> None:
    """Visual 4-layer architecture flow for the composite risk score (C&I + ACLED-style)."""
    _BG  = "#080808"
    layers = [
        ("NEWS GPR", "40%", "#c0392b",  "News GPR",           "RSS threat+act classification · Caldara-Iacoviello style · live feed · fallback from escalation signals"),
        ("CIS",      "30%", "#e67e22",  "Conflict Events",    "ACLED-calibrated event intensity · additive breadth bonus · staleness cap · static registry fallback"),
        ("CP",       "20%", "#2980b9",  "Chokepoint Stress",  "PortWatch throughput disruption · Hormuz live tanker count · active transmission pressure"),
        ("MCS",      "10%", "#CFB991",  "Market Confirmation","Oil-Gold joint signal + commodity vol ONLY · equity vol removed (fires on non-geo stress)"),
    ]
    boxes_html = ""
    for abbr, pct, col, full, detail in layers:
        boxes_html += (
            f'<div style="flex:1;background:{_BG};border:1px solid #2a2a2a;'
            f'border-top:3px solid {col};padding:.5rem .7rem;min-width:0">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'
            f'<span style="{_M}font-size:0.69rem;font-weight:700;color:{col}">{abbr}</span>'
            f'<span style="{_M}font-size:0.56rem;font-weight:700;color:{col};'
            f'background:rgba(255,255,255,0.04);padding:1px 5px">{pct}</span></div>'
            f'<div style="{_M}font-size:0.50rem;font-weight:700;color:#e8e9ed;'
            f'letter-spacing:.08em;text-transform:uppercase;margin-bottom:3px">{full}</div>'
            f'<div style="{_S}font-size:0.62rem;color:{_MUT};line-height:1.4">{detail}</div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="margin:.6rem 0">'
        f'<div style="display:flex;gap:6px;margin-bottom:6px">{boxes_html}</div>'
        f'<div style="display:flex;align-items:center;gap:0;margin-bottom:6px">'
        f'<div style="flex:1;height:1px;background:#2a2a2a"></div>'
        f'<div style="padding:0 8px;{_M}font-size:0.50rem;color:{_MUT}">▼ CIS-weighted conflict ranking · PortWatch disruption · RSS feed</div>'
        f'<div style="flex:1;height:1px;background:#2a2a2a"></div>'
        f'</div>'
        f'<div style="display:flex;gap:6px;margin-bottom:6px">'
        f'<div style="flex:2;background:{_BG};border:1px solid #2a2a2a;padding:.4rem .7rem">'
        f'<span style="{_M}font-size:0.50rem;font-weight:700;color:#e8e9ed;letter-spacing:.1em">ASSEMBLY</span><br>'
        f'<code style="{_M}font-size:0.56rem;color:{_G}">GRS_raw = 0.40·GPR + 0.30·CIS + 0.20·CP + 0.10·MCS</code>'
        f'</div>'
        f'<div style="flex:1;background:{_BG};border:1px solid #2a2a2a;padding:.4rem .7rem">'
        f'<span style="{_M}font-size:0.50rem;font-weight:700;color:#e8e9ed;letter-spacing:.1em">MARKET FRESHNESS</span><br>'
        f'<code style="{_M}font-size:0.56rem;color:#8E9AAA">rank by CIS × freshness ∈ [0.7, 1.5]</code>'
        f'</div>'
        f'<div style="flex:1;background:{_BG};border:1px solid #2a2a2a;padding:.4rem .7rem">'
        f'<span style="{_M}font-size:0.50rem;font-weight:700;color:#e8e9ed;letter-spacing:.1em">SCENARIO MULT</span><br>'
        f'<code style="{_M}font-size:0.56rem;color:#8E9AAA">× geo_mult  (default 1.0)</code>'
        f'</div>'
        f'</div>'
        f'<div style="display:flex;align-items:center;gap:0;margin-bottom:6px">'
        f'<div style="flex:1;height:1px;background:#2a2a2a"></div>'
        f'<div style="padding:0 8px;{_M}font-size:0.50rem;color:{_MUT}">▼ clip(·, 0, 100)</div>'
        f'<div style="flex:1;height:1px;background:#2a2a2a"></div>'
        f'</div>'
        f'<div style="background:#0a0a0a;border:1px solid {_G};padding:.45rem 1rem;text-align:center">'
        f'<span style="{_M}font-size:0.56rem;font-weight:700;color:{_G};letter-spacing:.18em">GRS</span>'
        f'<span style="{_M}font-size:0.50rem;color:{_MUT};margin-left:10px">0 – 100 · Low / Moderate / Elevated / High</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _freshness_range() -> None:
    """Visual [0.7 … 1.0 … 1.5] range bar for the market freshness multiplier."""
    st.markdown(
        f'<div style="margin:.5rem 0 .8rem;padding:.5rem .8rem;background:#080808;border:1px solid #1e1e1e">'
        f'<div style="{_M}font-size:0.50rem;color:{_MUT};letter-spacing:.12em;margin-bottom:.4rem">MARKET FRESHNESS RANGE  ∈  [0.7, 1.5]</div>'
        f'<div style="position:relative;height:6px;background:#1a1a1a;margin:.3rem 0">'
        # colored gradient bar
        f'<div style="position:absolute;left:0;top:0;width:100%;height:6px;'
        f'background:linear-gradient(to right,#2e7d32,#8E9AAA 37.5%,#CFB991 62.5%,#c0392b)"></div>'
        # tick marks + labels
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:.2rem">'
        f'<span style="{_M}font-size:0.50rem;color:#2e7d32">0.7× quiet market</span>'
        f'<span style="{_M}font-size:0.50rem;color:#8E9AAA">1.0× neutral</span>'
        f'<span style="{_M}font-size:0.50rem;color:#c0392b">1.5× crisis signal</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _confidence_component(label: str, weight: str, formula: str, note: str, color: str = _G) -> None:
    st.markdown(
        f'<div style="display:flex;gap:10px;align-items:flex-start;'
        f'padding:.4rem .6rem;border-bottom:1px solid #111;background:#080808">'
        f'<div style="flex-shrink:0;text-align:right;width:36px">'
        f'<span style="{_M}font-size:0.63rem;font-weight:700;color:{color}">{weight}</span></div>'
        f'<div style="flex:1">'
        f'<div style="{_M}font-size:0.50rem;font-weight:700;color:#e8e9ed;letter-spacing:.07em;'
        f'text-transform:uppercase;margin-bottom:2px">{label}</div>'
        f'<code style="{_M}font-size:0.53rem;color:{_DIM}">{formula}</code><br>'
        f'<span style="{_S}font-size:0.62rem;color:{_MUT}">{note}</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def _assumption(text: str) -> None:
    st.markdown(
        f'<div style="display:flex;gap:8px;margin:.15rem 0">'
        f'<span style="{_M}font-size:0.50rem;color:#e67e22;margin-top:1px">▲</span>'
        f'<span style="{_S}font-size:0.70rem;color:#b8b8b8">{text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

def _limitation(text: str) -> None:
    st.markdown(
        f'<div style="display:flex;gap:8px;margin:.15rem 0">'
        f'<span style="{_M}font-size:0.50rem;color:#c0392b;margin-top:1px">■</span>'
        f'<span style="{_S}font-size:0.70rem;color:#b8b8b8">{text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

def _data_source(name: str, what: str, freq: str, lag: str) -> None:
    st.markdown(
        f'<div style="background:#080808;border:1px solid #1e1e1e;padding:.45rem .8rem;margin:.3rem 0">'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<span style="{_M}font-size:0.56rem;font-weight:700;color:{_G}">{name}</span>'
        f'<span style="{_M}font-size:0.50rem;color:{_MUT}">Freq: {freq} · Lag: {lag}</span>'
        f'</div>'
        f'<span style="{_S}font-size:0.70rem;color:{_DIM}">{what}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Page ───────────────────────────────────────────────────────────────────────

def page_methodology(start: str = "", end: str = "", fred_key: str = "") -> None:
    _page_header(
        "Model Methodology",
        "Formulae · Weights · Assumptions · Data sources · Limitations",
        "DOCUMENTATION / MODEL RISK",
    )
    _page_intro(
        "This document provides complete technical documentation for all analytical models "
        "in the Cross-Asset Spillover Monitor. It is structured following the documentation "
        "conventions recommended by <strong>SR 11-7 / OCC 2011-12</strong> model risk management guidance: "
        "purpose, design logic, implementation, assumptions, limitations, and ongoing monitoring. "
        "Each section covers one analytical module - from raw data ingestion through to "
        "the composite risk score, illustrative trade structure generation, and portfolio stress testing."
    )

    # ── Navigation index ───────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:#080808;border:1px solid #1e1e1e;padding:.6rem 1rem;margin-bottom:1rem">'
        f'<p style="{_M}font-size:0.50rem;color:{_MUT};letter-spacing:.15em;margin-bottom:5px">CONTENTS</p>'
        + "".join(
            f'<span style="{_M}font-size:0.50rem;color:{_DIM};margin-right:16px">'
            f'<span style="color:{_G}">{i+1}.</span> {s}</span>'
            for i, s in enumerate([
                "Data Architecture", "Live Intelligence Layer",
                "Conflict Intensity Score (CIS)",
                "Transmission Pressure Score (TPS)", "Market Confirmation Score (MCS)",
                "Composite Risk Score", "News GPR Layer",
                "Correlation Regime Engine", "Spillover Network",
                "LP-IRF · Conflict Shock",
                "Markov Regime Chain", "Portfolio Exposure Engine",
                "Scenario Engine", "Assumptions & Limitations",
            ])
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 1. DATA ARCHITECTURE
    # ══════════════════════════════════════════════════════════════════════════
    _h2("1 · Data Architecture")
    _prose(
        "All market data is sourced from publicly available APIs - no proprietary feeds. "
        "Every layer caches aggressively to minimise latency and API rate-limit exposure. "
        "Stale-data badges appear on charts when the last fetch exceeds the TTL."
    )

    # Source grid - 3 columns
    _h3("Data Sources")
    sources = [
        ("#c0392b", "MARKET DATA",   "Yahoo Finance",
         "15 equity indices · 17 commodity futures · FX spot rates · implied vol proxies (VIX, OVX, GVZ)",
         "Daily close", "1 day", "1 h"),
        ("#e67e22", "MACRO DATA",    "FRED · St. Louis Fed",
         "10Y/2Y Treasury yields · yield spread · Fed Funds Rate · CPI · PCE · payrolls · GDP",
         "Monthly / Daily", "1–30 days", "24 h"),
        ("#8E9AAA", "POSITIONING",   "CFTC COT Reports",
         "Large speculator net longs for commodity futures - contrarian sentiment signal",
         "Weekly", "3 days", "24 h"),
        ("#CFB991", "NEWS & INTEL",  "RSS Feeds",
         "Reuters · AP · BBC · NYT · WSJ · FT - keyword-tagged, severity-scored, conflict-routed",
         "Near-real-time", "~15 min", "15 min"),
        ("#27ae60", "REGISTRY",      "Manual Conflict Registry",
         "CONFLICTS config: intensity dimensions · transmission weights · state · last_updated",
         "Manual", "Varies", "on change"),
        ("#8E9AAA", "FX CONVERSION", "Yahoo Finance (FX Spot)",
         "Live spot rates for portfolio conversion - GBPUSD=X, EURUSD=X. 5-day lookback",
         "Daily", "1 day", "1 h"),
    ]
    row1, row2 = sources[:3], sources[3:]
    for row in (row1, row2):
        cols = st.columns(3)
        for (col, cat, name, desc, freq, lag, ttl), c in zip(row, cols):
            with c:
                st.markdown(
                    f'<div style="background:#080808;border:1px solid #1e1e1e;'
                    f'border-top:2px solid {col};padding:.5rem .65rem;margin-bottom:6px">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'
                    f'<span style="{_M}font-size:0.50rem;font-weight:700;color:{col};letter-spacing:.12em">{cat}</span>'
                    f'<span style="{_M}font-size:0.50rem;color:{_MUT};background:#111;padding:1px 4px">'
                    f'TTL {ttl}</span></div>'
                    f'<div style="{_M}font-size:0.50rem;font-weight:700;color:#e8e9ed;margin-bottom:3px">{name}</div>'
                    f'<div style="{_S}font-size:0.64rem;color:{_DIM};line-height:1.45;margin-bottom:4px">{desc}</div>'
                    f'<div style="display:flex;gap:8px">'
                    f'<span style="{_M}font-size:0.50rem;color:{_MUT}">freq: {freq}</span>'
                    f'<span style="{_M}font-size:0.50rem;color:{_MUT}">lag: {lag}</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

    # Data flow strip
    _h3("Ingestion Pipeline")
    st.markdown(
        f'<div style="display:flex;align-items:stretch;gap:0;margin:.4rem 0;border:1px solid #1e1e1e">'
        + "".join(
            f'<div style="flex:1;padding:.4rem .6rem;background:#080808;'
            f'border-right:1px solid #1e1e1e;position:relative">'
            f'<div style="{_M}font-size:0.50rem;font-weight:700;color:{c};letter-spacing:.1em;margin-bottom:2px">{step}</div>'
            f'<div style="{_S}font-size:0.62rem;color:{_MUT}">{desc}</div>'
            f'</div>'
            for step, desc, c in [
                ("FETCH",    "yfinance · FRED · GDELT · EIA · PortWatch · ACLED · RSS",  _G),
                ("CACHE",    "TTL-keyed @st.cache_data per layer",                     _G),
                ("VALIDATE", "Empty-frame guards · fallback to 50 (neutral)",          _G),
                ("TRANSFORM","Log returns · EWM z-scores · rolling windows",           _G),
                ("SCORE",    "CIS / TPS / MCS / GRS · per conflict & portfolio",       _G),
                ("DISPLAY",  "Plotly charts · Streamlit UI · session_state publish",   _G),
            ]
        )
        + '</div>',
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 2. LIVE INTELLIGENCE LAYER
    # ══════════════════════════════════════════════════════════════════════════
    _h2("2 · Live Intelligence Layer")
    _prose(
        "Four real-time intelligence feeds extend the static conflict registry with event-driven, "
        "media-corroborated, and physical-market signals. All feeds are free-tier or open APIs, "
        "update on independent TTL schedules, and feed directly into CIS dimension replacement, "
        "proactive alerts, and the AI analyst context."
    )
    _h3("Live Sources")

    live_sources = [
        ("#27ae60", "CONFLICT MEDIA",  "GDELT 2.0 DOC API",
         "7-day article volume time-series per conflict · average tone · WoW volume trend · "
         "escalation signal (escalating / stable / de-escalating). No API key required.",
         "Hourly", "~15 min", "4 h"),
        ("#27ae60", "PHYSICAL STOCKS", "EIA Open Data",
         "U.S. crude oil · gasoline · distillate stocks · crude imports · natural gas storage. "
         "WoW, YoY, and vs-5-year seasonal average. Signal: draw (bullish) / build (bearish). "
         "DEMO_KEY accepted - production key unlocks higher rate limits.",
         "Weekly", "3 days", "24 h"),
        ("#27ae60", "CHOKEPOINT TRAFFIC", "IMF PortWatch",
         "Daily vessel transit counts for Strait of Hormuz · Bab el-Mandeb · Malacca · "
         "Suez · Taiwan Strait · Bosphorus · Panama · Cape of Good Hope. "
         "24h delta and 7-day moving average per strait. No API key required.",
         "Daily", "~1 day", "6 h"),
        ("#CFB991", "CONFLICT EVENTS", "ACLED",
         "Armed Conflict Location & Event Database - fatality counts · event type distribution · "
         "subnational geographic spread · escalation trend. Replaces static CIS dimensions "
         "(deadliness · geographic diffusion · escalation trend) when configured. "
         "Free academic API key required.",
         "Weekly", "~1 week", "8 h"),
    ]
    _live_cols = st.columns(4)
    for (col, cat, name, desc, freq, lag, ttl), c in zip(live_sources, _live_cols):
        with c:
            st.markdown(
                f'<div style="background:#080808;border:1px solid #1e1e1e;'
                f'border-top:2px solid {col};padding:.5rem .65rem;margin-bottom:6px">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'
                f'<span style="{_M}font-size:0.50rem;font-weight:700;color:{col};letter-spacing:.12em">{cat}</span>'
                f'<span style="{_M}font-size:0.50rem;color:{_MUT};background:#111;padding:1px 4px">'
                f'TTL {ttl}</span></div>'
                f'<div style="{_M}font-size:0.50rem;font-weight:700;color:#e8e9ed;margin-bottom:3px">{name}</div>'
                f'<div style="{_S}font-size:0.64rem;color:{_DIM};line-height:1.45;margin-bottom:4px">{desc}</div>'
                f'<div style="display:flex;gap:8px">'
                f'<span style="{_M}font-size:0.50rem;color:{_MUT}">freq: {freq}</span>'
                f'<span style="{_M}font-size:0.50rem;color:{_MUT}">lag: {lag}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    _h3("Signal Integration")
    _prose(
        "GDELT and ACLED signals are fused via a corroboration function: when both sources "
        "agree on direction, confidence is <em>high</em>; when one is neutral and one signals "
        "escalation, confidence is <em>medium</em>; contradictory signals default to "
        "<em>stable</em>. The final consensus signal replaces the static "
        "<code>escalation_trend</code> CIS dimension. EIA inventory draws/builds feed the "
        "Proactive Alerts engine as supply-side commodity pressure signals. PortWatch transit "
        "counts feed the Strait Watch page and the AI analyst context window."
    )
    st.markdown(
        f'<div style="background:#080808;border:1px solid #1e1e1e;'
        f'border-left:3px solid #27ae60;padding:.5rem .9rem;margin:.3rem 0">'
        f'<code style="{_M}font-size:0.56rem;color:#27ae60;white-space:pre-wrap">'
        f'GDELT escalation + ACLED escalation  → corroboration()\n'
        f'  both escalating     → final = "escalating"  · confidence = high\n'
        f'  one escalating      → final = "escalating"  · confidence = medium\n'
        f'  contradictory       → final = "stable"       · confidence = low\n'
        f'  neither configured  → static registry value used as fallback'
        f'</code></div>',
        unsafe_allow_html=True,
    )

    _h3("Freshness Registry")
    _prose(
        "All six live sources are tracked by a central freshness registry "
        "(<code>src/analysis/freshness.py</code>). Each source has two thresholds: "
        "<em>warn</em> (amber badge) and <em>stale</em> (red badge). "
        "Badges appear on any page that uses that data. The data health banner on the "
        "home page aggregates freshness across all sources."
    )
    freshness_rows = [
        ("YFinance",      "4 h",  "26 h"),   # src/analysis/freshness.py yfinance_prices
        ("FRED",          "48 h", "168 h"),  # fred_macro: 2d warn, 7d stale
        ("GDELT",         "4 h",  "12 h"),   # 3h cache cadence
        ("EIA Inventory", "24 h", "168 h"),  # weekly Wednesday update
        ("PortWatch",     "6 h",  "30 h"),   # IMF ArcGIS daily cadence
        ("ACLED",         "8 h",  "48 h"),   # 6h cache, conflict events
    ]
    header_html = (
        f'<table style="width:100%;border-collapse:collapse;{_S}font-size:0.70rem;margin:.3rem 0">'
        f'<thead><tr>'
        f'<th style="{_M}font-size:0.50rem;color:{_MUT};text-align:left;border-bottom:1px solid #2a2a2a;padding:3px 8px">SOURCE</th>'
        f'<th style="{_M}font-size:0.50rem;color:{_MUT};text-align:center;border-bottom:1px solid #2a2a2a;padding:3px 8px">WARN AFTER</th>'
        f'<th style="{_M}font-size:0.50rem;color:{_MUT};text-align:center;border-bottom:1px solid #2a2a2a;padding:3px 8px">STALE AFTER</th>'
        f'</tr></thead><tbody>'
    )
    rows_html = "".join(
        f'<tr style="border-bottom:1px solid #111">'
        f'<td style="{_M}font-size:0.50rem;color:#e8e9ed;padding:4px 8px">{src}</td>'
        f'<td style="{_M}font-size:0.50rem;color:#e67e22;text-align:center;padding:4px 8px">{warn}</td>'
        f'<td style="{_M}font-size:0.50rem;color:#c0392b;text-align:center;padding:4px 8px">{stale}</td>'
        f'</tr>'
        for src, warn, stale in freshness_rows
    )
    st.markdown(header_html + rows_html + '</tbody></table>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. CIS
    # ══════════════════════════════════════════════════════════════════════════
    _h2("3 · Conflict Intensity Score (CIS)")
    _prose(
        "The CIS is a per-conflict composite score (0–100) quantifying how intense and "
        "market-relevant a conflict is <em>right now</em>. Seven orthogonalized dimensions "
        "are blended by empirical association with commodity price disruption, then "
        "multiplied by a state factor reflecting whether hostilities are active, latent, or frozen."
    )

    _formula(
        "CIS(conflict) = [ Σᵢ wᵢ × dᵢ ] × state_mult × 100\n\n"
        "dᵢ ∈ [0,1]   - normalized dimension value\n"
        "wᵢ           - dimension weight (Σwᵢ = 1.0)\n"
        "state_mult   - active=1.00 · latent=0.35 · frozen=0.15",
        "Staleness cap: CIS hard-capped at 65.0 when last_updated > 180 days."
    )

    _h3("Seven Dimensions")
    dims = [
        ("Deadliness",           0.22, "#c0392b",
         "Direct casualty count, normalised 0–1. Highest weight - most reliably observable, most directly priced by markets."),
        ("Escalation Trend",     0.20, "#e67e22",
         "escalating = 1.0 · stable = 0.50 · de-escalating = 0.0. Direction drives forward-looking risk, not just current level."),
        ("Civilian Danger",      0.15, "#e67e22",
         "Targeting of civilian infrastructure, population displacement. Proxy for conflict indiscriminacy and duration risk."),
        ("Recency",              0.13, _G,
         "Exponential decay: 1.0 at day 1 → 0.30 at 365d (half-life ≈ 2 years). Latent: flat 0.35. Frozen: flat 0.15."),
        ("Geographic Diffusion", 0.12, _G,
         "Spatial spread of hostilities across borders. Higher diffusion → broader supply-chain disruption and FX stress."),
        ("Source Coverage",      0.10, _DIM,
         "Breadth of credible reporting across Reuters, AP, BBC, NYT, WSJ, FT. Low coverage penalises the score."),
        ("Fragmentation",        0.08, _DIM,
         "Number of distinct armed factions. More factions → harder to resolve → longer tail risk horizon."),
    ]
    # 2-column card grid (4 left, 3 right)
    cis_l, cis_r = st.columns(2)
    for i, (name, wt, col, desc) in enumerate(dims):
        bar_w = int(wt / 0.22 * 100)
        card_html = (
            f'<div style="background:#080808;border:1px solid #1e1e1e;'
            f'border-left:3px solid {col};padding:.4rem .65rem;margin-bottom:5px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px">'
            f'<span style="{_M}font-size:0.50rem;font-weight:700;color:{col}">{name}</span>'
            f'<span style="{_M}font-size:0.50rem;font-weight:700;color:{col}">{wt:.0%}</span>'
            f'</div>'
            f'<div style="background:#1a1a1a;height:2px;margin-bottom:4px">'
            f'<div style="background:{col};width:{bar_w}%;height:2px"></div></div>'
            f'<div style="{_S}font-size:0.63rem;color:{_DIM};line-height:1.45">{desc}</div>'
            f'</div>'
        )
        if i < 4:
            with cis_l:
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            with cis_r:
                st.markdown(card_html, unsafe_allow_html=True)

    _h3("State Multipliers")
    st.markdown(
        f'<div style="display:flex;gap:6px;margin:.4rem 0">'
        + "".join(
            f'<div style="flex:1;background:#080808;border:1px solid #1e1e1e;'
            f'border-top:3px solid {c};padding:.5rem .7rem">'
            f'<div style="{_M}font-size:0.50rem;font-weight:700;color:{c};letter-spacing:.1em;margin-bottom:4px">{s}</div>'
            f'<div style="{_M}font-size:22px;font-weight:700;color:{c};margin-bottom:2px">{m}×</div>'
            f'<div style="background:#1a1a1a;height:3px;margin-bottom:5px">'
            f'<div style="background:{c};width:{int(float(m)*100)}%;height:3px"></div></div>'
            f'<div style="{_S}font-size:0.64rem;color:{_DIM}">{d}</div>'
            f'</div>'
            for s, m, c, d in [
                ("ACTIVE",  "1.00", "#c0392b", "Full score - hostilities actively ongoing. No discount."),
                ("LATENT",  "0.35", "#e67e22", "No active fighting but structural risk persists - sanctions, troops massed, unresolved territorial disputes."),
                ("FROZEN",  "0.15", _DIM,      "Formal ceasefire or prolonged stalemate. Low near-term risk; tail event potential retained."),
            ]
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 3. TPS
    # ══════════════════════════════════════════════════════════════════════════
    _h2("4 · Transmission Pressure Score (TPS)")
    _prose(
        "The TPS measures how strongly a conflict's geography and nature transmit into "
        "commodity and financial markets - independent of CIS intensity. A conflict can be "
        "intense (high CIS) but geographically isolated (low TPS). Both layers are required. "
        "<code style='font-family:monospace;font-size:0.75em'>tx_ch ∈ [0,1]</code> is the "
        "analyst-assigned per-channel transmission strength; <code style='font-family:monospace;"
        "font-size:0.75em'>w_ch</code> is the channel's structural market importance weight."
    )

    _formula(
        "TPS(conflict) = [ Σ_ch w_ch × tx_ch ] × state_mult × 100\n\n"
        "tx_ch ∈ [0, 1]  - analyst-coded transmission strength per channel\n"
        "w_ch            - structural market importance weight (Σw = 1.0)\n"
        "state_mult      - same multiplier as CIS (active/latent/frozen)",
        "12 transmission channels. Higher tx_ch = conflict directly disrupts that channel."
    )

    _h3("12 Channels - Grouped by Category")
    tps_groups = [
        ("#c0392b", "ENERGY", [
            ("Oil / Gas",     0.18, "Primary geopolitical shock mechanism - energy supply disruption raises input costs globally."),
            ("Energy Infra",  0.02, "Pipeline and grid attacks. Narrow but catastrophic when triggered."),
        ]),
        ("#e67e22", "PHYSICAL FLOWS", [
            ("Shipping",      0.12, "Rerouting costs, war-risk insurance premia. Critical when straits threatened."),
            ("Chokepoint",    0.10, "Hormuz · Suez · Bosphorus - ~51% of seaborne oil transit. Binary risk."),
            ("Agriculture",   0.08, "Black Sea corridor, wheat/corn supply. Russia/Ukraine primary driver."),
            ("Supply Chain",  0.05, "Input shortages, factory closures, logistics re-routing. Slow-moving."),
            ("Metals",        0.10, "Nickel, aluminium, copper - defence, construction, EV inputs."),
        ]),
        ("#CFB991", "FINANCIAL", [
            ("Sanctions",     0.12, "Financial sanctions → market dislocation. Commodity sanctions amplify."),
            ("FX",            0.06, "Safe-haven USD bid, EM currency stress, petrocurrency correlations."),
            ("Credit",        0.02, "Sovereign CDS spreads, EM bond stress. Lagging indicator."),
        ]),
        (_DIM, "MACRO / EQUITY", [
            ("Equity Sector", 0.08, "Direct equity: defence, energy, shipping. First-order market impact."),
            ("Inflation",     0.07, "Second-order: commodity shock → CPI → central bank response → re-rating."),
        ]),
    ]
    for grp_col, grp_name, channels in tps_groups:
        st.markdown(
            f'<div style="margin:.35rem 0 0;padding:.2rem .6rem;background:#0a0a0a;'
            f'border-left:3px solid {grp_col};border-bottom:1px solid #1e1e1e">'
            f'<span style="{_M}font-size:0.50rem;font-weight:700;color:{grp_col};letter-spacing:.15em">'
            f'{grp_name}</span></div>',
            unsafe_allow_html=True,
        )
        for ch_name, ch_wt, ch_desc in channels:
            bar_w = int(ch_wt / 0.18 * 100)
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;padding:.3rem .6rem;'
                f'background:#080808;border-bottom:1px solid #111">'
                f'<div style="width:130px;flex-shrink:0">'
                f'<span style="{_M}font-size:0.50rem;font-weight:700;color:#e8e9ed">{ch_name}</span>'
                f'</div>'
                f'<div style="width:60px;flex-shrink:0;text-align:right">'
                f'<span style="{_M}font-size:0.50rem;font-weight:700;color:{grp_col}">{ch_wt:.0%}</span>'
                f'</div>'
                f'<div style="width:80px;flex-shrink:0">'
                f'<div style="background:#1a1a1a;height:3px">'
                f'<div style="background:{grp_col};width:{bar_w}%;height:3px"></div></div>'
                f'</div>'
                f'<div style="{_S}font-size:0.63rem;color:{_DIM}">{ch_desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # 4. MCS
    # ══════════════════════════════════════════════════════════════════════════
    _h2("5 · Market Confirmation Score (MCS)")
    _prose(
        "The MCS is a <em>confirmatory</em> layer only (10% of GRS) — it uses two geo-specific "
        "market signals to verify what news flow and conflict events already indicate. "
        "Equity vol, rates vol, safe-haven bid, and correlation acceleration were removed: "
        "all four fire on rate cycles, earnings shocks, and COVID-style demand crashes that are "
        "not geopolitical in origin, producing false positives and diluting the score."
    )

    _formula(
        "MCS = 0.70 · OilGold + 0.30 · CmdVol",
        "Two geo-specific signals only. Both EWM z-scored (span=252) before weighting."
    )

    _h3("Signal Cards")
    mc1, mc2 = st.columns(2)
    with mc1:
        _signal_card(
            "oil_gold", "Oil-Gold Joint Signal", 0.70, "#e67e22",
            "EWM z-scores of 20d cumulative log-returns for Gold and front Oil contract (span=252). "
            "Gold↑ + Oil↑ = war/supply-shock. Gold↑ + Oil↓ = safe-haven flight only. "
            "Gold↓ + Oil↑ = pure supply shock. Smooth conflict bonus ramps as both exceed neutral.",
            "score = 50 + g_z×17 + o_z×11 + conflict_bonus\n"
            "conflict_bonus = clip((g_excess + o_excess)×7.5, 0, 22)\n"
            "  g_excess = max(0, g_z − 0.7)\n"
            "  o_excess = max(0, o_z − 0.7)",
            "span=252 prevents 'war normal' adaptation (span=60 normalises after 2+ months). "
            "WTI preferred; falls back to Brent."
        )
    with mc2:
        _signal_card(
            "cmd_vol", "Commodity Vol", 0.30, "#8E9AAA",
            "20d annualized realized vol across energy/metals basket "
            "(WTI, Brent, NatGas, Gold, Silver, Copper). EWM z-scored (span=252). "
            "No OLS residualization — removing equity beta eliminates the legitimate co-movement "
            "during war/blockade, when oil vol and equity fear rise together.",
            "rv_cmd = mean annualized 20d vol  [energy/metals basket]\n"
            "z      = EWM_zscore(rv_cmd, span=252)\n"
            "CmdVol = (50 + z × 14).clip(0, 100)",
            "Symmetric: fires on price crash as well as spike — both indicate supply disruption."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 5. COMPOSITE RISK SCORE
    # ══════════════════════════════════════════════════════════════════════════
    _h2("6 · Composite Geopolitical Risk Score")
    _prose(
        "The GRS is a 4-layer index inspired by Caldara &amp; Iacoviello (2022) GPR methodology "
        "and ACLED-style event-driven conflict measurement. "
        "News flow leads as the primary layer &mdash; it is the most responsive and least "
        "market-contaminated signal available in near-real-time. "
        "Equity vol has been removed from the scoring pipeline entirely: it fires on rate cycles, "
        "earnings, and growth scares that are not geopolitical in origin, producing false "
        "positives during peacetime and undercounting genuine conflict periods once markets stabilise."
    )

    _h3("Architecture")
    _arch_flow()

    _h3("Layer Weights &amp; Score Bands")
    col_lw, col_sb = st.columns([1, 1])
    with col_lw:
        for abbr, full, pct, col, val, rationale in [
            ("NEWS GPR", "News GPR",          "40%", "#c0392b", 40,
             "Caldara-Iacoviello style: threat + act classification of RSS headlines. "
             "Falls back to conflict escalation proxy when feed unavailable."),
            ("CIS",      "Conflict Events",   "30%", "#e67e22", 30,
             "ACLED-calibrated event intensity. Additive breadth bonus (log scale) "
             "prevents inflation when multiple low-intensity conflicts coexist."),
            ("CP",       "Chokepoint Stress", "20%", "#2980b9", 20,
             "PortWatch live Hormuz disruption + active conflict transmission pressure. "
             "Explicit physical channel — not a judgment about future impact."),
            ("MCS",      "Market Conf.",      "10%", _G,        10,
             "Oil-gold joint signal + commodity vol. Confirmatory only — "
             "not primary. Geo supply-shock signature, not generic market stress."),
        ]:
            st.markdown(
                f'<div style="margin:.35rem 0">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px">'
                f'<span style="{_M}font-size:0.50rem;font-weight:700;color:{col}">{abbr} · {full}</span>'
                f'<span style="{_M}font-size:0.56rem;font-weight:700;color:{col}">{pct}</span></div>'
                f'<div style="background:#1a1a1a;height:4px;position:relative">'
                f'<div style="background:{col};width:{val*2}px;height:4px;max-width:100%"></div></div>'
                f'<div style="{_S}font-size:0.62rem;color:{_MUT};margin-top:2px">{rationale}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    with col_sb:
        st.markdown(
            f'<div style="background:#080808;border:1px solid #1e1e1e;padding:.5rem .7rem">'
            f'<p style="{_M}font-size:0.50rem;color:{_MUT};letter-spacing:.12em;margin-bottom:.4rem">SCORE BANDS</p>'
            + "".join(
                f'<div style="display:flex;align-items:center;gap:8px;padding:.25rem 0;'
                f'border-bottom:1px solid #111">'
                f'<div style="width:4px;height:24px;background:{c};flex-shrink:0"></div>'
                f'<div>'
                f'<span style="{_M}font-size:0.50rem;font-weight:700;color:{c}">{band} · {lo}–{hi}</span><br>'
                f'<span style="{_S}font-size:0.62rem;color:{_MUT}">{interp}</span>'
                f'</div></div>'
                for band, lo, hi, c, interp in [
                    ("HIGH/CRISIS", 75, 100, "#c0392b", "Active conflict confirmed by news flow + chokepoint disruption"),
                    ("ELEVATED",    50,  75, "#e67e22", "Significant news activity or chokepoint stress building"),
                    ("MODERATE",    25,  50, _G,        "Latent risk — news background noise or isolated events"),
                    ("LOW",          0,  25, "#27ae60", "Quiet — minimal news, no chokepoint disruption"),
                ]
            )
            + f'</div>',
            unsafe_allow_html=True,
        )

    _h3("Portfolio Aggregation")
    _formula(
        "CIS_portfolio = weighted_mean(CIS_i) + breadth_bonus\n"
        "breadth_bonus = min(8, 2.5 × ln(n_eff + 1))   ← n_eff = 1/HHI (conflict concentration)\n\n"
        "GRS_raw = 0.40 × NewsGPR + 0.30 × CIS_portfolio + 0.20 × Chokepoint + 0.10 × MCS\n"
        "GRS     = clip( GRS_raw × geo_mult , 0, 100 )",
        "Additive breadth bonus (log scale) replaces the prior n_eff^0.25 multiplicative factor, "
        "which inflated a 64-point weighted mean to 97 when 6 conflicts coexisted. "
        "Log bonus adds ≤8 pts regardless of portfolio size."
    )

    _h3("Market Freshness Multiplier")
    _prose(
        "Which conflict is <em>currently moving markets</em> differs session-to-session. "
        "The freshness layer maps each conflict's live transmission channel activity to a multiplier "
        "∈ [0.7, 1.5]. The command center's Lead Conflict is ranked by "
        "<code style='font-family:monospace;font-size:0.75em'>CIS × market_freshness</code>, "
        "not raw CIS - so a quiet Russia day won't outrank an active Hormuz blockade."
    )
    _freshness_range()
    _formula(
        "market_freshness(c) = clip( 0.7 + mean(signals) × 0.8 ,  0.7, 1.5 )\n\n"
        "signals (fired only if conflict transmission weight > 0.3):\n"
        "  oil_sig  = max(|brent_1d|/3, |wti_1d|/3, tanker_disruption) × oil_weight\n"
        "  ng_sig   = (|natgas_1d| / 5)  × energy_infra_weight\n"
        "  agri_sig = (|wheat_1d|  / 4)  × agriculture_weight\n"
        "  eq_sig   = (|vix_1d|    / 5)  × equity_sector_weight",
        "Live inputs: Brent/WTI/NatGas/Wheat 1d % changes + PortWatch tanker disruption index. TTL=300s."
    )
    # Example boxes
    st.markdown(
        f'<div style="display:flex;gap:8px;margin:.4rem 0">'
        f'<div style="flex:1;background:#080808;border:1px solid #1e1e1e;'
        f'border-left:3px solid #c0392b;padding:.45rem .7rem">'
        f'<p style="{_M}font-size:0.50rem;font-weight:700;color:#c0392b;margin:0 0 4px;letter-spacing:.08em">'
        f'EXAMPLE  ·  HORMUZ BLOCKADE  (high freshness)</p>'
        f'<p style="{_S}font-size:0.67rem;color:{_DIM};margin:0;line-height:1.5">'
        f'Iran/Hormuz: oil_gas=0.97, chokepoint=0.97. Brent +4% intraday, PortWatch disruption=0.40.<br>'
        f'oil_sig ≈ 0.97 × max(1.33, 1.33, 0.40) = 1.29 → mean([1.0]) = 1.0 →<br>'
        f'<b style="color:#c0392b">market_freshness = 1.5 (capped) → Hormuz ranks #1</b></p>'
        f'</div>'
        f'<div style="flex:1;background:#080808;border:1px solid #1e1e1e;'
        f'border-left:3px solid #27ae60;padding:.45rem .7rem">'
        f'<p style="{_M}font-size:0.50rem;font-weight:700;color:#27ae60;margin:0 0 4px;letter-spacing:.08em">'
        f'EXAMPLE  ·  RUSSIA-UKRAINE  (quiescent day)</p>'
        f'<p style="{_S}font-size:0.67rem;color:{_DIM};margin:0;line-height:1.5">'
        f'NatGas flat (+0.3%), Wheat +1%, no oil move. Both signals below 0.3 threshold → not fired.<br>'
        f'No active signals → mean([]) = 0 →<br>'
        f'<b style="color:#27ae60">market_freshness = 0.7 → Russia drops in ranking</b></p>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    _h3("Staleness Enforcement")
    _prose("Conflicts not updated within threshold windows receive automatic penalties - "
           "preventing stale manual data from outranking freshly-confirmed crises.")
    st.markdown(
        f'<div style="border:1px solid #1e1e1e;margin:.4rem 0">'
        # timeline bar
        f'<div style="display:flex;height:5px">'
        f'<div style="flex:1;background:#27ae60"></div>'
        f'<div style="flex:1;background:#e67e22"></div>'
        f'<div style="flex:1;background:#c0392b"></div>'
        f'</div>'
        f'<div style="display:flex">'
        + "".join(
            f'<div style="flex:1;padding:.45rem .7rem;background:#080808;border-right:1px solid #111">'
            f'<div style="{_M}font-size:0.50rem;font-weight:700;color:{c};letter-spacing:.08em;margin-bottom:3px">{s}</div>'
            f'<div style="{_M}font-size:0.50rem;color:#e8e9ed;margin-bottom:2px">{threshold}</div>'
            f'<div style="{_S}font-size:0.65rem;color:{_MUT}">{effect}</div>'
            f'</div>'
            for s, c, threshold, effect in [
                ("FRESH",  "#27ae60", "≤ 90 days", "Full confidence. No penalty applied."),
                ("WARN",   "#e67e22", "91 – 180 days", "Confidence score × 0.90. CIS unaffected; data is ageing."),
                ("STALE",  "#c0392b", "> 180 days", "CIS hard-capped at 65.0 · confidence × 0.70. Cannot rank as lead conflict."),
            ]
        )
        + f'</div></div>',
        unsafe_allow_html=True,
    )

    _h3("Confidence Score & Uncertainty Intervals")
    _prose(
        "Every GRS output includes a confidence value [0,1] and symmetric ±uncertainty band. "
        "Confidence weights three independent quality signals. The uncertainty band uses "
        "quadrature (RSS) combination of per-layer errors - wider when data is stale or market signals disagree."
    )
    # Confidence components
    st.markdown(
        f'<div style="border:1px solid #1e1e1e;margin:.4rem 0 .2rem">'
        f'<div style="background:#0a0a0a;padding:.35rem .7rem;border-bottom:1px solid #1e1e1e">'
        f'<span style="{_M}font-size:0.50rem;font-weight:700;color:{_G};letter-spacing:.12em">'
        f'CONFIDENCE  =  0.35 × conflict_conf  +  0.35 × news_conf  +  0.20 × mcs_agreement  +  0.10 × data_availability</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    _confidence_component(
        "Conflict Confidence", "35%",
        "0.30×source_cov + 0.25×data_conf + 0.25×freshness + 0.20×completeness",
        "CIS-intensity-weighted avg across active conflicts. Staleness multipliers applied (×0.90 at 90d, ×0.70 at 180d).",
        "#c0392b",
    )
    _confidence_component(
        "News Confidence", "35%",
        "0.90 if live RSS feed active · 0.45 if conflict-escalation fallback used",
        "Live RSS feed → high confidence. Fallback (feed unavailable or no classified headlines) → lower weight.",
        "#e67e22",
    )
    _confidence_component(
        "MCS Signal Agreement", "20%",
        "1 − std(MCS sub-signals) / 50",
        "High agreement (both oil-gold and commodity vol pointing same direction) → score near 1.0.",
        _G,
    )
    _confidence_component(
        "Data Availability", "10%",
        "1.0 if avg_corr non-empty  ·  0.5 otherwise",
        "Penalizes sessions where cross-asset correlation series is unavailable (data load failure).",
        _DIM,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    _formula(
        "σ_CIS  = (1 − conflict_conf)     × CIS_portfolio × 0.30\n"
        "σ_news = (1 − news_conf)          × news_layer   × 0.40\n"
        "σ_MCS  = (1 − mcs_agreement)      × MCS          × 0.10\n"
        "σ_CP   = 0.0  (PortWatch present or static fallback; no partial uncertainty)\n\n"
        "Uncertainty = √( σ_CIS² + σ_news² + σ_MCS² )\n"
        "score_low   = max(0,   GRS − Uncertainty)\n"
        "score_high  = min(100, GRS + Uncertainty)",
        "Quadrature (RSS) - assumes layer errors are independent. "
        "At 100% confidence → Uncertainty = 0. At 0% → max layer-weighted error."
    )

    _h3("Historical Score  (risk_score_history)")
    _prose(
        "The historical chart plots a daily VIX-mimic index — cross-asset realized volatility "
        "z-scored against a 3-year trailing baseline. Structural CIS/News GPR/Chokepoint layers "
        "are not available at daily historical frequency, so the series uses realized fear directly: "
        "when markets are calm, the score is low; when they are in distress, it spikes."
    )
    _weight_table([
        ("eq_vol",  "Equity Realized Vol",     0.55, "20d annualized realized vol across all equity indices. Core VIX analogue — spikes during GFC, COVID, Ukraine selloffs."),
        ("cmd_vol", "Commodity Realized Vol",  0.45, "20d annualized realized vol across WTI, Brent, NatGas, Gold, Silver, Copper. Captures geo supply shocks and energy crises."),
    ])
    _formula(
        "composite_vol(t) = 0.55 · eq_rv(t) + 0.45 · cmd_rv(t)\n\n"
        "rv(t)      = rolling(20).std() × √252 × 100   [annualized %]\n"
        "z(t)       = EWM_zscore(composite_vol, span=756)   [3-year baseline]\n"
        "HistScore  = 100 / (1 + exp(−0.56 · z))   [sigmoid → 0–100]\n\n"
        "z=0 → 50  ·  z=+2 → 77  ·  z=+3 → 88  ·  z=−2 → 23",
        "span=756 (3-year EWM): baseline adapts slowly so multi-year crises stay elevated. "
        "span=252 adapted in ~12 months, normalising Ukraine 2023 to z≈0."
    )
    _section_note(
        "HistScore ≠ live GRS on any given day — the live model uses news flow, conflict events, "
        "and chokepoint data unavailable historically. Treat HistScore as a market fear proxy: "
        "2008 GFC and 2020 COVID score highest; 2012-13 QE era (VIX ~15) scores low; "
        "2022 Ukraine invasion spikes; recent active war environment reads Elevated."
    )

    _h3("GRS vs Global Conflict Risk Map - Separate Pipelines")
    _prose(
        "The top-level <b>Geopolitical Risk Score (GRS)</b> and the <b>Global Conflict Risk Map</b> "
        "are <em>independent pipelines</em> tracking the same underlying conflicts through different models."
    )
    st.markdown(
        f'<div style="display:flex;gap:8px;margin:.4rem 0">'
        f'<div style="flex:1;background:#080808;border:1px solid #1e1e1e;'
        f'border-left:3px solid {_G};padding:.5rem .8rem">'
        f'<p style="{_M}font-size:0.50rem;font-weight:700;color:{_G};letter-spacing:.12em;margin:0 0 5px">GRS PIPELINE</p>'
        f'<p style="{_S}font-size:0.65rem;color:{_DIM};margin:0;line-height:1.6">'
        f'<b style="color:#e8e9ed">Source:</b> <code>conflict_model.py</code> · CONFLICTS registry (6 parametric conflicts)<br>'
        f'<b style="color:#e8e9ed">Inputs:</b> News GPR (RSS classification) + CIS (conflict events) + Chokepoint (PortWatch) + MCS (oil-gold)<br>'
        f'<b style="color:#e8e9ed">Output:</b> Single composite GRS 0–100 · used on Overview, Geopolitical, Watchlist, and 12 other pages</p>'
        f'</div>'
        f'<div style="flex:1;background:#080808;border:1px solid #1e1e1e;'
        f'border-left:3px solid #8E9AAA;padding:.5rem .8rem">'
        f'<p style="{_M}font-size:0.50rem;font-weight:700;color:#8E9AAA;letter-spacing:.12em;margin:0 0 5px">MAP PIPELINE</p>'
        f'<p style="{_S}font-size:0.65rem;color:{_DIM};margin:0;line-height:1.6">'
        f'<b style="color:#e8e9ed">Source:</b> <code>war_country_scores.py</code> · WAR_DATA (3 war base scores)<br>'
        f'<b style="color:#e8e9ed">Inputs:</b> Static base scores × live commodity z-score multipliers (gas/oil/gold) × per-country structural weights<br>'
        f'<b style="color:#e8e9ed">Output:</b> Per-country composite 0–100 · used for choropleth and country exposure alerts</p>'
        f'</div>'
        f'</div>'
        f'<div style="background:#0a0a0a;border:1px solid #1e1e1e;padding:.4rem .8rem;margin-top:4px">'
        f'<p style="{_M}font-size:0.50rem;font-weight:700;color:#e67e22;letter-spacing:.1em;margin:0 0 3px">LINKAGE</p>'
        f'<p style="{_S}font-size:0.65rem;color:{_DIM};margin:0;line-height:1.5">'
        f'Map scores feed <code>proactive_alerts._check_country_exposure()</code> via <code>war_country_scores.compute_country_scores()</code>. '
        f'When any equity-indexed country\'s composite score ≥ 65, a country exposure alert fires. '
        f'This is the only downstream consumer of map scores outside the visualization layer. '
        f'GRS and map scores do not share a computation path.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 6. NEWS GPR LAYER
    # ══════════════════════════════════════════════════════════════════════════
    _h2("7 · News GPR Layer - Threat / Act Classification")
    _prose(
        "The News GPR layer is the <strong>primary driver</strong> of the composite GRS (40% weight), "
        "following the Caldara &amp; Iacoviello (2022) intuition that news flow is the most responsive "
        "and least market-contaminated geopolitical risk signal. "
        "Headlines are classified as <b>Threat</b> (leading: rhetoric, military signalling, diplomatic breakdown) "
        "or <b>Act</b> (contemporaneous: strikes, sanctions imposed, vessels seized). "
        "When the live feed returns a weak signal, a conflict-state floor prevents it from "
        "overriding the structural reality of active wars: "
        "<code>news_layer = max(live_gpr, fallback × 0.60)</code>."
    )

    _h3("Ingestion Pipeline")
    # Horizontal flow
    pipeline_steps = [
        (_G,       "01 INGEST",    "RSS polled every 15 min",          "Reuters · AP · BBC · NYT · WSJ · FT"),
        (_G,       "02 TAG",       "Keyword taxonomy match",           "8 regions · 6 commodity categories"),
        (_G,       "03 SCORE",     "Relevance scoring",                "(regions×10 + cmd×12 + severity) × src_wt"),
        (_G,       "04 CLASSIFY",  "Threat vs Act split",              "keyword lists → binary label"),
        ("#e67e22","05 ROUTE",     "High-confidence dispatch",         "relevance ≥ 65 → Geopolitical Analyst queue"),
    ]
    st.markdown(
        f'<div style="display:flex;gap:0;margin:.5rem 0;border:1px solid #1e1e1e">',
        unsafe_allow_html=True,
    )
    for col, step, title, detail in pipeline_steps:
        st.markdown(
            f'<div style="flex:1;padding:.4rem .6rem;background:#080808;border-right:1px solid #1e1e1e">'
            f'<div style="{_M}font-size:0.50rem;font-weight:700;color:{col};letter-spacing:.12em;margin-bottom:3px">{step}</div>'
            f'<div style="{_M}font-size:0.50rem;font-weight:700;color:#e8e9ed;margin-bottom:2px">{title}</div>'
            f'<div style="{_S}font-size:0.62rem;color:{_MUT}">{detail}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Relevance formula + Threat vs Act side by side
    _h3("Scoring & Classification")
    col_sc, col_cls = st.columns([1, 1])
    with col_sc:
        _formula(
            "Relevance = (regions×10 + commodities×12\n"
            "           + severity_score) × source_weight\n\n"
            "source_weight: Reuters/AP=1.0 · BBC=0.9 · NYT/WSJ/FT=0.85",
            "Relevance ≥ 65 → routed for agent review. Below threshold → archived only."
        )
    with col_cls:
        st.markdown(
            f'<div style="display:flex;gap:6px;height:100%">'
            + "".join(
                f'<div style="flex:1;background:#080808;border:1px solid #1e1e1e;'
                f'border-top:2px solid {c};padding:.45rem .6rem">'
                f'<div style="{_M}font-size:0.50rem;font-weight:700;color:{c};letter-spacing:.1em;margin-bottom:5px">{label}</div>'
                f'<div style="{_S}font-size:0.63rem;color:{_DIM};margin-bottom:4px">{timing}</div>'
                f'<div style="{_S}font-size:0.61rem;color:{_MUT};font-style:italic">{kws}</div>'
                f'</div>'
                for label, c, timing, kws in [
                    ("THREAT",  "#e67e22",
                     "Leading indicator - precedes market reaction",
                     "warned · mobilize · ultimatum · sanctions threatened · diplomatic breakdown"),
                    ("ACT",     "#c0392b",
                     "Contemporaneous - market already reacting",
                     "attack · seized · explosion · sanctions imposed · vessel boarded · airstrike"),
                ]
            )
            + '</div>',
            unsafe_allow_html=True,
        )

    _h3("Dynamic Alpha Weighting")
    _prose(
        "The alpha parameter dynamically shifts weight toward Act headlines when realized events dominate, "
        "and toward Threat headlines during rhetoric-heavy periods. This prevents a news cycle of "
        "pure rhetoric from scoring as high as a day of actual strikes."
    )
    _formula(
        "News GPR = α · Act_score + (1 − α) · Threat_score\n\n"
        "α = clip( [n_act / (n_act + n_threat + ε)] × 2,  0.2, 0.8 )\n\n"
        "α → 0.8  when acts dominate  (realized-event-heavy cycle)\n"
        "α → 0.2  when threats dominate (rhetoric-heavy cycle)",
        "ε = 1e-6 to avoid division by zero. α is bounded to [0.2, 0.8] - "
        "Threat signal always retained even in a pure-act cycle."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 7. CORRELATION REGIME ENGINE
    # ══════════════════════════════════════════════════════════════════════════
    _h2("8 · Correlation Regime Engine")
    _prose(
        "The regime engine classifies the current equity-commodity relationship into one of four states "
        "using rolling Pearson correlation. DCC-GARCH is computed separately as a supplementary "
        "visualization — it does not feed into the regime state variable. Rolling correlation provides "
        "a fast, interpretable regime signal that maps directly to the four-state ladder below."
    )

    # Regime ladder - full width
    _h3("Four-State Regime Ladder")
    regimes = [
        ("0 · DECORRELATED", "#2e7d32", "avg_corr < P₂₅",       "Diversification intact. Equity and commodity shocks are not co-moving - safe to hold cross-asset portfolios."),
        ("1 · NORMAL",       _DIM,      "P₂₅ ≤ avg_corr < P₅₀", "Baseline co-movement consistent with macro linkages. No action signal."),
        ("2 · ELEVATED",     "#e67e22", "P₅₀ ≤ avg_corr < P₇₅", "Stress building. Correlation rising - monitor for regime shift. Consider partial hedges."),
        ("3 · CRISIS",       "#c0392b", "avg_corr ≥ P₇₅",        "Full spillover regime. Diversification fails. Geopolitical shock likely active - activate hedges."),
    ]
    # gradient bar
    st.markdown(
        f'<div style="display:flex;height:4px;margin:.4rem 0 0">'
        f'<div style="flex:1;background:#2e7d32"></div>'
        f'<div style="flex:1;background:{_DIM}"></div>'
        f'<div style="flex:1;background:#e67e22"></div>'
        f'<div style="flex:1;background:#c0392b"></div>'
        f'</div>'
        f'<div style="display:flex;margin-bottom:.5rem">',
        unsafe_allow_html=True,
    )
    for name, col, threshold, interp in regimes:
        st.markdown(
            f'<div style="flex:1;background:#080808;border:1px solid #1e1e1e;'
            f'border-top:none;padding:.45rem .6rem">'
            f'<div style="{_M}font-size:0.50rem;font-weight:700;color:{col};margin-bottom:2px">{name}</div>'
            f'<div style="{_M}font-size:0.50rem;color:{_MUT};margin-bottom:4px">{threshold}</div>'
            f'<div style="{_S}font-size:0.63rem;color:{_DIM};line-height:1.4">{interp}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Methods - 2 columns
    col_a, col_b = st.columns(2)
    with col_a:
        _h3("Rolling Pearson Correlation")
        _prose(
            "Pairwise rolling Pearson between log-return series of equity indices and commodity futures. "
            "Multiple windows expose different timescales of co-movement. "
            "A 300-day burn-in period is loaded beyond the selected range so all windows have "
            "sufficient history at the period boundary. <code style='font-family:monospace;"
            "font-size:0.75em'>avg_corr</code> = mean of all pairwise ρ for regime detection."
        )
        _formula(
            "ρ(A,B,t,w) = Cov(r_A[t-w:t], r_B[t-w:t])\n"
            "             / [ σ(r_A[t-w:t]) · σ(r_B[t-w:t]) ]\n\n"
            "r_t = log(P_t / P_{t-1})\n"
            "w ∈ { 21d, 42d, 63d, 126d, 252d }",
            "Thresholds P₂₅/P₅₀/P₇₅ set from full historical avg_corr distribution."
        )
        _h3("DCC  (Engle, 2002)  ·  EWMA Pre-whitening")
        _prose(
            "Two-step procedure: (1) EWMA volatility pre-whitening (λ=0.94, RiskMetrics daily standard) "
            "standardises returns before the DCC recursion — equivalent to GARCH(1,1) with ω=0, α=0.06, β=0.94 "
            "but with fixed rather than MLE-estimated parameters; "
            "(2) DCC(1,1) recursion on the standardised residuals to obtain time-varying conditional correlation R_t."
        )
        _formula(
            "Step 1  EWMA pre-whitening (fixed params, not MLE-fit):\n"
            "  h_t = λ·h_{t-1} + (1-λ)·x_{t-1}²,   λ=0.94\n"
            "  ε̃_t = x_t / √h_t   (standardised residuals)\n\n"
            "Step 2  DCC(1,1):\n"
            "  Q_t = (1-a-b)·Q̄ + a·(ε̃_{t-1}·ε̃_{t-1}ᵀ) + b·Q_{t-1}\n"
            "  R_t = diag(Q_t)^{-½} · Q_t · diag(Q_t)^{-½}",
            "Q̄ = unconditional cov of ε̃. Parameters: a=0.05, b=0.90. "
            "Full GARCH(1,1) MLE (ω, α, β data-estimated) would require the arch package."
        )
    with col_b:
        _h3("Regime Detection Detail")
        _prose(
            "Regime is assigned at each date by comparing the 60-day rolling average cross-asset "
            "correlation against its empirical percentile distribution estimated over the full "
            "available history. Thresholds are re-calibrated as data accumulates."
        )
        _formula(
            "avg_corr(t) = mean_over_all_pairs [ ρ(equity_i, commodity_j, t, 60) ]\n\n"
            "regime(t) = 0  if avg_corr(t) < P₂₅\n"
            "          = 1  if P₂₅ ≤ avg_corr(t) < P₅₀\n"
            "          = 2  if P₅₀ ≤ avg_corr(t) < P₇₅\n"
            "          = 3  if avg_corr(t) ≥ P₇₅",
            "Percentiles P₂₅/P₅₀/P₇₅ from full historical avg_corr distribution."
        )
        _h3("Early Warning Signals")
        ew_signals = [
            ("Regime Acceleration",  "Rate of change of avg_corr > threshold → imminent shift"),
            ("Correlation Spike",    "Single-day correlation jump > 2σ above 60d mean"),
            ("Cross-Vol Divergence", "Commodity vol rising while equity vol falls → idiosyncratic stress"),
            ("Safe-Haven Bid",       "Gold z-score > 1 and TLT positive simultaneously"),
        ]
        for sig, desc in ew_signals:
            st.markdown(
                f'<div style="display:flex;gap:8px;padding:.3rem .5rem;'
                f'background:#080808;border-bottom:1px solid #111;align-items:flex-start">'
                f'<span style="{_M}font-size:0.50rem;color:{_G};flex-shrink:0;margin-top:1px">▶</span>'
                f'<div>'
                f'<span style="{_M}font-size:0.50rem;font-weight:700;color:#e8e9ed">{sig}</span>'
                f'<span style="{_S}font-size:0.63rem;color:{_DIM};margin-left:6px">{desc}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # 8. SPILLOVER NETWORK
    # ══════════════════════════════════════════════════════════════════════════
    _h2("9 · Spillover Network")
    _prose(
        "Three complementary measures quantify cross-asset shock transmission. "
        "Each captures a distinct aspect: Granger tests linear predictability, "
        "Transfer Entropy captures nonlinear information flow, and Diebold-Yilmaz "
        "decomposes portfolio-level forecast error variance. Together they triangulate "
        "the direction, magnitude, and statistical significance of spillovers."
    )

    # Method type legend
    st.markdown(
        f'<div style="display:flex;gap:8px;margin:.4rem 0 .6rem;align-items:center">'
        f'<span style="{_M}font-size:0.50rem;color:{_MUT};letter-spacing:.12em">METHOD TYPE ▸</span>'
        f'<span style="{_M}font-size:0.50rem;background:#1a1a1a;border:1px solid #c0392b;'
        f'color:#c0392b;padding:1px 7px">DIRECTIONAL</span>'
        f'<span style="{_M}font-size:0.50rem;color:{_MUT};margin:0 4px">-</span>'
        f'<span style="{_S}font-size:0.67rem;color:{_DIM}">identifies which asset leads / lags</span>'
        f'<span style="{_M}font-size:0.50rem;background:#1a1a1a;border:1px solid {_G};'
        f'color:{_G};padding:1px 7px;margin-left:12px">AGGREGATE</span>'
        f'<span style="{_M}font-size:0.50rem;color:{_MUT};margin:0 4px">-</span>'
        f'<span style="{_S}font-size:0.67rem;color:{_DIM}">portfolio-level total spillover index</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown(
            f'<div style="background:#080808;border:1px solid #1e1e1e;'
            f'border-top:2px solid #c0392b;padding:.55rem .75rem;height:100%">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.35rem">'
            f'<span style="{_M}font-size:0.56rem;font-weight:700;color:#c0392b">GRANGER CAUSALITY</span>'
            f'<span style="{_M}font-size:0.50rem;border:1px solid #c0392b;color:#c0392b;padding:1px 5px">'
            f'DIRECTIONAL · LINEAR</span>'
            f'</div>'
            f'<p style="{_S}font-size:0.67rem;color:{_DIM};margin:0 0 .4rem;line-height:1.5">'
            f'Tests whether lagged X improves the forecast of Y beyond Y\'s own lags. '
            f'Rejects H₀ if X contains predictive information about Y.</p>'
            f'<div style="background:#040404;border-left:2px solid #c0392b;padding:.35rem .5rem;margin:.3rem 0">'
            f'<code style="{_M}font-size:0.50rem;color:#c0392b;white-space:pre-wrap;line-height:1.6">'
            f'H₀: X does not Granger-cause Y\n'
            f'Lag p* selected by BIC over p ∈ {{1, …, 5}}\n'
            f'F-test run at BIC-optimal lag only\n'
            f'Reported: min p-value · significance · lag p*</code></div>'
            f'<p style="{_S}font-size:0.62rem;color:{_MUT};margin:.35rem 0 0;line-height:1.4;font-style:italic">'
            f'▲ Assumes linear dependence. Misses threshold or regime-switching effects.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_s2:
        st.markdown(
            f'<div style="background:#080808;border:1px solid #1e1e1e;'
            f'border-top:2px solid #e67e22;padding:.55rem .75rem;height:100%">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.35rem">'
            f'<span style="{_M}font-size:0.56rem;font-weight:700;color:#e67e22">TRANSFER ENTROPY</span>'
            f'<span style="{_M}font-size:0.50rem;border:1px solid #e67e22;color:#e67e22;padding:1px 5px">'
            f'DIRECTIONAL · NONLINEAR</span>'
            f'</div>'
            f'<p style="{_S}font-size:0.67rem;color:{_DIM};margin:0 0 .4rem;line-height:1.5">'
            f'Model-free directed information flow from X to Y. Captures nonlinear dependencies '
            f'Granger cannot detect. Net TE isolates the dominant transmission direction.</p>'
            f'<div style="background:#040404;border-left:2px solid #e67e22;padding:.35rem .5rem;margin:.3rem 0">'
            f'<code style="{_M}font-size:0.50rem;color:#e67e22;white-space:pre-wrap;line-height:1.6">'
            f'TE(X→Y) = Σ p(y_t+1, y_t, x_t)\n'
            f'          · log[ p(y_t+1 | y_t, x_t)\n'
            f'               / p(y_t+1 | y_t) ]\n'
            f'Net TE  = TE(X→Y) − TE(Y→X)</code></div>'
            f'<p style="{_S}font-size:0.62rem;color:{_MUT};margin:.35rem 0 0;line-height:1.4;font-style:italic">'
            f'▲ Histogram-based entropy estimation using equal-probability binning. Sensitive to bin-count choice.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_s3:
        st.markdown(
            f'<div style="background:#080808;border:1px solid #1e1e1e;'
            f'border-top:2px solid {_G};padding:.55rem .75rem;height:100%">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.35rem">'
            f'<span style="{_M}font-size:0.56rem;font-weight:700;color:{_G}">DIEBOLD-YILMAZ (2012)</span>'
            f'<span style="{_M}font-size:0.50rem;border:1px solid {_G};color:{_G};padding:1px 5px">'
            f'AGGREGATE · VARIANCE DECOMP</span>'
            f'</div>'
            f'<p style="{_S}font-size:0.67rem;color:{_DIM};margin:0 0 .4rem;line-height:1.5">'
            f'Cholesky-orthogonalized forecast error variance decomposition of a VAR. '
            f'θᵢⱼ(H) = fraction of asset i\'s H-step forecast variance explained by shocks to j.</p>'
            f'<div style="background:#040404;border-left:2px solid {_G};padding:.35rem .5rem;margin:.3rem 0">'
            f'<code style="{_M}font-size:0.50rem;color:{_G};white-space:pre-wrap;line-height:1.6">'
            f'θᵢⱼ(H) = Cholesky-orthogonalized FEVD\n'
            f'  (statsmodels VAR.fevd(), Cholesky P)\n'
            f'Total SI = (Σᵢ≠ⱼ θᵢⱼ) / N × 100</code></div>'
            f'<p style="{_S}font-size:0.62rem;color:{_MUT};margin:.35rem 0 0;line-height:1.4;font-style:italic">'
            f'H = 10d horizon. VAR lag by BIC. Cholesky is order-dependent; '
            f'D-Y (2012) uses Pesaran-Shin generalized FEVD (order-invariant) — '
            f'that extension requires custom implementation beyond statsmodels.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Bottom interpretation strip
    st.markdown(
        f'<div style="display:flex;gap:0;margin:.6rem 0 0;border:1px solid #1e1e1e">'
        f'<div style="flex:1;padding:.4rem .7rem;border-right:1px solid #1e1e1e;background:#080808">'
        f'<div style="{_M}font-size:0.50rem;color:{_MUT};letter-spacing:.1em;margin-bottom:3px">'
        f'INTERPRETATION HIERARCHY</div>'
        f'<span style="{_S}font-size:0.67rem;color:{_DIM}">'
        f'Granger → establishes statistical lead/lag.  '
        f'TE → confirms nonlinear direction.  '
        f'DY → sizes total spillover exposure.</span>'
        f'</div>'
        f'<div style="flex:0 0 auto;padding:.4rem .7rem;background:#080808">'
        f'<div style="{_M}font-size:0.50rem;color:{_MUT};letter-spacing:.1em;margin-bottom:3px">'
        f'CONSENSUS RULE</div>'
        f'<span style="{_S}font-size:0.67rem;color:{_DIM}">'
        f'Flag spillover only when ≥ 2 of 3 methods agree on direction and significance.</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 10. LP-IRF · CONFLICT SHOCK TRANSMISSION
    # ══════════════════════════════════════════════════════════════════════════
    _h2("10 · LP-IRF · Conflict Shock Transmission")
    _prose(
        "The Local-Projection IRF (Jordà 2005) measures how asset returns respond to an unexpected "
        "conflict escalation shock over horizons h = 0..20 trading days. Unlike VAR-based IRFs, "
        "local projections are robust to dynamic misspecification — each horizon is a separate "
        "OLS regression, so mis-specifying the lag structure at one horizon does not contaminate others. "
        "The shock is the standardised AR residual of daily GDELT news-volume, capturing <em>unexpected</em> "
        "escalation rather than anticipated conflict intensity level."
    )

    col_lp1, col_lp2 = st.columns(2)
    with col_lp1:
        _h3("Step 1 · Shock Identification")
        _prose(
            "For each conflict, the daily GDELT article-volume series is filtered through an "
            "AR(<em>p</em>) model with BIC lag selection (p ∈ {1..5}). "
            "The standardised residual is the conflict-intensity shock: it is orthogonal to "
            "the conflict's own recent history, isolating the <em>unexpected</em> escalation component."
        )
        _formula(
            "GDELT volume series v_t  (daily article count)\n"
            "s_t = log(1 + v_t)               (variance-stabilise)\n\n"
            "AR(p*) fit, BIC lag selection p* ∈ {1..5}:\n"
            "  s_t = μ + φ₁·s_{t-1} + … + φ_{p*}·s_{t-p*} + ε_t\n\n"
            "Shock: shock_t = ε_t / σ̂_ε       (standardised to σ=1)",
            "BIC search over p=1..5; min 40 obs per lag step. "
            "shock_t ≡ 1-σ unexpected escalation event. Source: GDELT TimelineVol API, 1y timespan."
        )
    with col_lp2:
        _h3("Step 2 · Local-Projection Regression")
        _prose(
            "At each horizon <em>h</em>, the h-day cumulative log-return is regressed on the "
            "contemporaneous shock plus lag controls. The IRF coefficient β<sub>h</sub> gives the "
            "expected cumulative return response to a 1-σ shock at <em>h</em> trading days ahead."
        )
        _formula(
            "y_{a,t→t+h} = Σ_{k=0}^{h} r_{a,t+k} × 100  (cum. log-return, %)\n\n"
            "y_{a,t→t+h} = α_h + β_h · shock_t\n"
            "             + Σ_{j=1}^{4} γ_{h,j} · shock_{t-j}\n"
            "             + Σ_{j=1}^{2} δ_{h,j} · r_{a,t-j} × 100  +  ε_{h,t}\n\n"
            "β_h  =  IRF at horizon h  (%, per 1-σ shock)\n"
            "Displayed: 90% confidence band",
            "Estimated separately for each asset a and horizon h = 0..20. "
            "Lag controls: 4 shock lags + 2 return lags. Min 80 obs required per horizon."
        )

    _h3("Step 3 · Standard Error Specification")
    _prose(
        "Overlapping cumulative returns introduce MA(<em>h</em>−1) serial correlation in LP residuals at horizon h. "
        "Newey-West HAC with bandwidth = h is the canonical correction "
        "(Jordà 2005, fn. 4; Montiel Olea &amp; Plagborg-Møller 2021, Theorem 2). "
        "At h = 0 there is no overlap, so HC3 (heteroscedasticity-robust only) is used instead."
    )

    _se_spec_rows = [
        ("h = 0",  "HC3",           "No overlap — cumulative return = single-period return. HC3 corrects heteroscedasticity; no autocorrelation correction needed.", "#2e7d32"),
        ("h ≥ 1",  "NW-HAC  bw=h",  "Overlapping h+1-day sums → MA(h-1) residual structure. NW with maxlags=h is uniformly valid (MOP 2021). Bands widen monotonically.", _G),
    ]
    _se_html = '<div style="border:1px solid #1e1e1e;overflow:hidden;margin:.3rem 0 .6rem">'
    for _sh, _sn, _sd, _sc in _se_spec_rows:
        _se_html += (
            f'<div style="display:flex;gap:0;align-items:stretch;border-bottom:1px solid #111">'
            f'<div style="width:3px;background:{_sc};flex-shrink:0"></div>'
            f'<div style="padding:.3rem .6rem;width:70px;flex-shrink:0;display:flex;align-items:center">'
            f'<code style="{_M}font-size:0.50rem;color:{_sc}">{_sh}</code></div>'
            f'<div style="padding:.3rem .6rem;width:140px;flex-shrink:0;display:flex;align-items:center">'
            f'<span style="{_M}font-size:0.50rem;color:#e8e9ed">{_sn}</span></div>'
            f'<div style="padding:.3rem .6rem;flex:1;display:flex;align-items:center">'
            f'<span style="{_S}font-size:0.65rem;color:{_DIM}">{_sd}</span></div>'
            f'</div>'
        )
    _se_html += '</div>'
    st.markdown(_se_html, unsafe_allow_html=True)

    col_lp3, col_lp4 = st.columns(2)
    with col_lp3:
        _h3("Data Source · GDELT TimelineVol")
        _data_source(
            "GDELT Project  (gdeltproject.org)",
            "Daily article-volume series via GDELT Doc 2.0 API (TimelineVol mode). "
            "Each conflict uses a keyword query + CAMEO theme filter. 1-year timespan (~250 trading days). "
            "Rate-limited at 1 request per 5 seconds; 6-second inter-conflict sleep enforced. "
            "Cached 6 hours to match GDELT update cadence.",
            "Daily",
            "~0 (live API)",
        )
        _prose(
            "GDELT date strings include a trailing <code>Z</code> "
            "(<code>20250702T000000Z</code>) stripped before parsing. "
            "Two CONFLICTS registry IDs are remapped to GDELT query keys: "
            "<code>israel_gaza → israel_hamas</code>, <code>iran_conflict → iran_regional</code>."
        )
    with col_lp4:
        _h3("Conflict-to-Asset Mapping")
        _prose(
            "For each conflict, the <code>affected_equities</code> and "
            "<code>affected_commodities</code> fields in the CONFLICTS registry "
            "define the response-asset universe. Only assets present in the loaded "
            "return data are included. The mapping is static — update the registry to "
            "add or remove assets from a conflict's IRF panel."
        )
        st.markdown(
            f'<div style="background:#040404;border:1px solid #1e1e1e;padding:.4rem .6rem;margin:.2rem 0">'
            f'<div style="{_M}font-size:0.50rem;color:{_MUT};margin-bottom:5px;letter-spacing:.1em">REGISTRY FIELDS USED</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:4px">'
            f'<code style="{_M}font-size:0.50rem;color:{_G};background:#111;padding:2px 7px">affected_equities[ ]</code>'
            f'<code style="{_M}font-size:0.50rem;color:{_G};background:#111;padding:2px 7px">affected_commodities[ ]</code>'
            f'<code style="{_M}font-size:0.50rem;color:{_DIM};background:#111;padding:2px 7px">id</code>'
            f'</div>'
            f'<p style="{_S}font-size:0.63rem;color:{_MUT};margin:.4rem 0 0;line-height:1.4">'
            f'6 conflicts × affected assets → separate IRF panel per conflict. '
            f'Asset columns filtered to those available in yfinance return data.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    col_lpa, col_lpb = st.columns(2)
    with col_lpa:
        _h3("Assumptions")
        _assumption("GDELT article volume is a noisy but unbiased proxy for conflict news intensity at daily frequency.")
        _assumption("AR(p) BIC-selected residuals are approximately white noise — conflict media dynamics are linear and stationary within the 1-year window.")
        _assumption("Lag controls (4 shock lags + 2 return lags) are sufficient to condition out confounders at each horizon independently.")
        _assumption("1-year GDELT window (~250 trading days) provides adequate statistical power for the maximum horizon h=20.")
    with col_lpb:
        _h3("Limitations")
        _limitation("GDELT coverage skews toward English-language sources; regional conflicts with limited Western press are systematically under-covered.")
        _limitation("AR residuals may retain serial structure if conflict media dynamics are non-linear or subject to abrupt regime changes in coverage.")
        _limitation("Conflict-to-asset mapping is manually configured; assets not in the registry are excluded even if materially exposed to that conflict.")
        _limitation("NW-HAC finite-sample performance deteriorates at long horizons (h approaching T/4); interpret h ≥ 15 confidence bands with caution.")

    st.markdown(
        f'<div style="display:flex;gap:0;margin:.5rem 0 0;border:1px solid #1e1e1e">'
        f'<div style="flex:1;padding:.35rem .7rem;border-right:1px solid #1e1e1e;background:#040404">'
        f'<div style="{_M}font-size:0.50rem;color:{_MUT};letter-spacing:.1em;margin-bottom:2px">PRIMARY REFERENCE</div>'
        f'<span style="{_S}font-size:0.65rem;color:{_DIM}">'
        f'Jordà, Ò. (2005). Estimation and Inference of Impulse Responses by Local Projections. '
        f'<em>American Economic Review</em>, 95(1), 161–182.</span>'
        f'</div>'
        f'<div style="flex:1;padding:.35rem .7rem;background:#040404">'
        f'<div style="{_M}font-size:0.50rem;color:{_MUT};letter-spacing:.1em;margin-bottom:2px">SE SPECIFICATION</div>'
        f'<span style="{_S}font-size:0.65rem;color:{_DIM}">'
        f'Montiel Olea, J.L. &amp; Plagborg-Møller, M. (2021). Local Projection Inference is Simpler and '
        f'More Robust Than You Think. <em>Econometrica</em>, 89(4), 1789–1823.</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 11. MARKOV REGIME CHAIN
    # ══════════════════════════════════════════════════════════════════════════
    _h2("11 · Markov Regime Chain")
    _prose(
        "The four-state regime series (Decorrelated / Normal / Elevated / Crisis) is modelled "
        "as a first-order Markov chain. The empirical transition matrix P is estimated from "
        "the full historical regime sequence, enabling steady-state probability estimation, "
        "mean first passage time computation, and probabilistic forward regime forecasts."
    )

    # 4-state visual bar
    _mc_states = [
        ("0", "DECORRELATED", "#2e7d32"),
        ("1", "NORMAL",       _DIM),
        ("2", "ELEVATED",     "#e67e22"),
        ("3", "CRISIS",       "#c0392b"),
    ]
    _mc_boxes = ""
    for _idx, (_num, _lbl, _col) in enumerate(_mc_states):
        _sep = "border-right:1px solid #111;" if _idx < 3 else ""
        _mc_boxes += (
            f'<div style="flex:1;background:#080808;{_sep}'
            f'border-top:3px solid {_col};padding:.4rem .5rem;text-align:center;min-width:0">'
            f'<div style="{_M}font-size:20px;font-weight:700;color:{_col};line-height:1">{_num}</div>'
            f'<div style="{_M}font-size:0.50rem;font-weight:700;letter-spacing:.1em;color:#e8e9ed;'
            f'text-transform:uppercase;margin-top:3px">{_lbl}</div>'
            f'</div>'
        )
    st.markdown(
        f'<div style="display:flex;gap:0;margin:.4rem 0 .6rem;border:1px solid #1e1e1e;overflow:hidden">'
        + _mc_boxes +
        f'</div>',
        unsafe_allow_html=True,
    )

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(
            f'<div style="background:#080808;border:1px solid #1e1e1e;'
            f'border-top:2px solid {_G};padding:.55rem .75rem;margin-bottom:.5rem">'
            f'<div style="{_M}font-size:0.50rem;font-weight:700;color:{_G};letter-spacing:.1em;'
            f'text-transform:uppercase;margin-bottom:.4rem">TRANSITION MATRIX</div>'
            f'<div style="background:#040404;border-left:2px solid {_G};padding:.35rem .5rem;margin:.3rem 0">'
            f'<code style="{_M}font-size:0.53rem;color:{_G};white-space:pre-wrap;line-height:1.6">'
            f'P[i,j] = count(r_t=i, r_t+1=j)\n'
            f'          / count(r_t=i)\n'
            f'Shape: 4×4  ·  row-stochastic</code></div>'
            f'<p style="{_S}font-size:0.65rem;color:{_DIM};margin:.3rem 0 0;line-height:1.5">'
            f'Estimated from full historical regime sequence. Each row sums to 1. '
            f'High diagonal values indicate strong regime persistence.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="background:#080808;border:1px solid #1e1e1e;'
            f'border-top:2px solid {_DIM};padding:.55rem .75rem">'
            f'<div style="{_M}font-size:0.50rem;font-weight:700;color:{_DIM};letter-spacing:.1em;'
            f'text-transform:uppercase;margin-bottom:.4rem">STEADY-STATE DISTRIBUTION</div>'
            f'<div style="background:#040404;border-left:2px solid {_DIM};padding:.35rem .5rem;margin:.3rem 0">'
            f'<code style="{_M}font-size:0.53rem;color:{_DIM};white-space:pre-wrap;line-height:1.6">'
            f'π = π · P   s.t.  Σπᵢ = 1\n'
            f'Left eigenvector of P for λ = 1</code></div>'
            f'<p style="{_S}font-size:0.65rem;color:{_DIM};margin:.3rem 0 0;line-height:1.5">'
            f'Long-run probability of being in each regime. '
            f'If π[CRISIS] > 0.30, system is structurally stressed.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            f'<div style="background:#080808;border:1px solid #1e1e1e;'
            f'border-top:2px solid #e67e22;padding:.55rem .75rem;margin-bottom:.5rem">'
            f'<div style="{_M}font-size:0.50rem;font-weight:700;color:#e67e22;letter-spacing:.1em;'
            f'text-transform:uppercase;margin-bottom:.4rem">MEAN FIRST PASSAGE TIME</div>'
            f'<div style="background:#040404;border-left:2px solid #e67e22;padding:.35rem .5rem;margin:.3rem 0">'
            f'<code style="{_M}font-size:0.53rem;color:#e67e22;white-space:pre-wrap;line-height:1.6">'
            f'm[i,j] = 1 + Σ_{{k≠j}}  P[i,k] · m[k,j]\n'
            f'Solved as linear system\n'
            f'(fundamental matrix method)</code></div>'
            f'<p style="{_S}font-size:0.65rem;color:{_DIM};margin:.3rem 0 0;line-height:1.5">'
            f'm[CRISIS, NORMAL] = expected trading days to recover. '
            f'Primary input to drawdown duration estimates.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        _run_rows = "".join([
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:.25rem .4rem;background:#040404;border-left:2px solid {_rc};'
            f'margin-bottom:2px">'
            f'<span style="{_M}font-size:0.50rem;color:{_rc};text-transform:uppercase">{_rl}</span>'
            f'<span style="{_S}font-size:0.62rem;color:{_DIM}">mean run · P75 · longest</span>'
            f'</div>'
            for _rl, _rc in [
                ("Decorrelated", "#2e7d32"), ("Normal", _DIM),
                ("Elevated", "#e67e22"), ("Crisis", "#c0392b"),
            ]
        ])
        st.markdown(
            f'<div style="background:#080808;border:1px solid #1e1e1e;'
            f'border-top:2px solid #c0392b;padding:.55rem .75rem">'
            f'<div style="{_M}font-size:0.50rem;font-weight:700;color:#c0392b;letter-spacing:.1em;'
            f'text-transform:uppercase;margin-bottom:.4rem">RUN STATISTICS</div>'
            + _run_rows +
            f'<p style="{_S}font-size:0.62rem;color:{_MUT};margin:.35rem 0 0;line-height:1.4;font-style:italic">'
            f'Consecutive-day distributions per regime. Calibrates expected Crisis duration.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 12. PORTFOLIO EXPOSURE ENGINE
    # ══════════════════════════════════════════════════════════════════════════
    _h2("12 · Portfolio Exposure Engine")
    _prose(
        "The exposure engine maps each asset against every active conflict across four "
        "compounding dimensions: structural linkage, CIS-weighted total exposure, "
        "scenario-scaled score, and conflict beta. Assets are ranked by total exposure "
        "and conflict beta to surface hedge candidates and concentration risk."
    )

    # Flow diagram: SES → TAE → SAS → β
    _pe_steps = [
        ("SES", "Structural Exposure",  "#c0392b", "analyst config\nper-asset × conflict"),
        ("TAE", "Total Asset Exposure", "#e67e22", "CIS-weighted sum\nacross all conflicts"),
        ("SAS", "Scenario-Adjusted",    _G,        "× geo_mult\nclipped [0, 100]"),
        ("β",   "Conflict Beta",        _DIM,      "SES × TPS / 100\nper-conflict ranking"),
    ]
    _pe_flow = '<div style="display:flex;align-items:stretch;gap:0;margin:.4rem 0 .7rem;border:1px solid #1e1e1e;overflow:hidden">'
    for _pi, (_pa, _pb, _pc, _pd) in enumerate(_pe_steps):
        _pe_flow += (
            f'<div style="flex:1;background:#080808;border-top:3px solid {_pc};'
            f'{"border-right:1px solid #1e1e1e;" if _pi < 3 else ""}'
            f'padding:.5rem .6rem;min-width:0;text-align:center">'
            f'<div style="{_M}font-size:16px;font-weight:700;color:{_pc};line-height:1">{_pa}</div>'
            f'<div style="{_M}font-size:0.50rem;font-weight:700;letter-spacing:.09em;color:#e8e9ed;'
            f'text-transform:uppercase;margin:.3rem 0 .25rem">{_pb}</div>'
            f'<code style="{_M}font-size:0.50rem;color:{_MUT};white-space:pre-wrap;line-height:1.4">{_pd}</code>'
            f'</div>'
        )
        if _pi < 3:
            _pe_flow += (
                f'<div style="display:flex;align-items:center;padding:0 4px;background:#050505">'
                f'<span style="{_M}font-size:0.75rem;color:{_MUT}">→</span>'
                f'</div>'
            )
    _pe_flow += '</div>'
    st.markdown(_pe_flow, unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        _formula(
            "SES(asset, conflict)\n"
            "  = structural[asset][conflict]  ∈ [0, 1]\n\n"
            "TAE(asset)\n"
            "  = Σ_c  SES(asset, c) × CIS(c) / 100",
            "SES coded in config.SECURITY_EXPOSURE - reflects supply-chain, revenue, or "
            "regulatory linkage. TAE: higher CIS conflicts contribute proportionally more."
        )
    with col_p2:
        _formula(
            "SAS(asset)\n"
            "  = clip( TAE(asset) × geo_mult × 100,  0, 100 )\n\n"
            "β(asset, conflict)\n"
            "  = SES(asset, conflict) × TPS(conflict) / 100",
            "SAS scales up in stressed scenarios via geo_mult > 1.0. "
            "β measures asset sensitivity per unit of TPS - primary hedge ranking input."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 13. SCENARIO ENGINE
    # ══════════════════════════════════════════════════════════════════════════
    _h2("13 · Scenario Engine")
    _prose(
        "The Scenario Engine applies a named stress lens across the entire dashboard - "
        "modifying composite risk scores, asset exposure, and trade signals without "
        "altering underlying data. Enables side-by-side comparison of scenario assumptions."
    )

    _h3("Scenario Parameter Registry")
    _se_params = [
        ("geo_mult",  "Geopolitical Multiplier", _G,        "Scales GRS. >1 amplifies, <1 dampens.",           "float · default 1.0"),
        ("vol_mult",  "Volatility Multiplier",   "#e67e22", "Applied to modelled drawdown. >1 = fat-tail.",     "float · default 1.0"),
        ("safe_haven","Safe-Haven Overlay",       _DIM,     "Adds positive bias to Gold, UST, CHF, JPY.",       "bool"),
        ("short_bias","Short-Bias Overlay",       "#c0392b","Penalises long equity, surfaces hedge signals.",    "bool"),
        ("desc",      "Scenario Narrative",       _MUT,     "Human-readable label for stress event windows.",   "str"),
    ]
    _se_html = '<div style="border:1px solid #1e1e1e;overflow:hidden;margin:.3rem 0 .6rem">'
    for _si, (_sk, _sn, _sc, _sd, _sv) in enumerate(_se_params):
        _sbg = "#040404" if _si % 2 == 0 else "#080808"
        _se_html += (
            f'<div style="display:flex;gap:0;align-items:stretch;background:{_sbg};border-bottom:1px solid #111">'
            f'<div style="width:3px;background:{_sc};flex-shrink:0"></div>'
            f'<div style="padding:.35rem .6rem;width:105px;flex-shrink:0;display:flex;align-items:center">'
            f'<code style="{_M}font-size:0.50rem;color:{_sc}">{_sk}</code></div>'
            f'<div style="padding:.35rem .6rem;width:160px;flex-shrink:0;display:flex;align-items:center">'
            f'<span style="{_S}font-size:0.69rem;color:#e8e9ed">{_sn}</span></div>'
            f'<div style="padding:.35rem .6rem;flex:1;display:flex;align-items:center">'
            f'<span style="{_S}font-size:0.67rem;color:{_DIM}">{_sd}</span></div>'
            f'<div style="padding:.35rem .6rem;width:110px;flex-shrink:0;display:flex;align-items:center;justify-content:flex-end">'
            f'<span style="{_M}font-size:0.50rem;color:{_MUT}">{_sv}</span></div>'
            f'</div>'
        )
    _se_html += '</div>'
    st.markdown(_se_html, unsafe_allow_html=True)

    _h3("Stress Test Event Methodology")
    _prose(
        "For each catalogued geopolitical event, the stress tester extracts a 30-day pre-event "
        "window, the event duration, and a 60-day post-event window. Portfolio returns are "
        "computed as dollar-weighted log-returns. Max drawdown and Sharpe are reported "
        "for the event duration window."
    )

    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        _formula(
            "Portfolio_t = Σᵢ  wᵢ · (Pᵢ_t / Pᵢ_t₀)\n"
            "              (normalised to base 100)\n\n"
            "Max Drawdown =\n"
            "  min_t [ (Port_t − max_{{s≤t}} Port_s)\n"
            "          / max_{{s≤t}} Port_s ]",
            "Weights wᵢ sum to 1. Single-period log-returns (no compounding correction)."
        )
    with col_sc2:
        _formula(
            "Event Sharpe = μ_daily / σ_daily × √252\n\n"
            "Window layout:\n"
            "  [−30d PRE]  [← EVENT →]  [+60d POST]",
            "Sharpe computed over event duration only. Pre/post windows provide regime context."
        )
        st.markdown(
            f'<div style="display:flex;gap:0;margin:.4rem 0;height:28px;'
            f'border:1px solid #1e1e1e;overflow:hidden">'
            f'<div style="flex:2;background:#1a2a1a;display:flex;align-items:center;'
            f'justify-content:center;border-right:1px solid #2a2a2a">'
            f'<span style="{_M}font-size:0.50rem;color:#2e7d32">−30d PRE</span></div>'
            f'<div style="flex:3;background:#2a1a0a;display:flex;align-items:center;'
            f'justify-content:center;border-right:1px solid #2a2a2a">'
            f'<span style="{_M}font-size:0.50rem;color:#e67e22">← EVENT →</span></div>'
            f'<div style="flex:4;background:#0a0a1a;display:flex;align-items:center;'
            f'justify-content:center">'
            f'<span style="{_M}font-size:0.50rem;color:{_DIM}">+60d POST</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 14. ASSUMPTIONS & LIMITATIONS
    # ══════════════════════════════════════════════════════════════════════════
    _h2("14 · Assumptions, Limitations & Ongoing Maintenance")

    def _badge_html(level: str) -> str:
        _bc = {"HIGH": "#c0392b", "MED": "#e67e22", "LOW": _DIM}.get(level, _DIM)
        return (
            f'<span style="{_M}font-size:0.50rem;border:1px solid {_bc};color:{_bc};'
            f'padding:0 4px;flex-shrink:0;align-self:flex-start;margin-top:2px">{level}</span>'
        )

    def _assumption_b(text: str, level: str = "MED") -> None:
        st.markdown(
            f'<div style="display:flex;gap:7px;margin:.2rem 0;align-items:flex-start">'
            f'<span style="{_M}font-size:0.50rem;color:#e67e22;flex-shrink:0;margin-top:2px">▲</span>'
            f'<span style="{_S}font-size:0.69rem;color:#b8b8b8;flex:1;line-height:1.5">{text}</span>'
            + _badge_html(level) +
            f'</div>',
            unsafe_allow_html=True,
        )

    def _limitation_b(text: str, level: str = "MED") -> None:
        st.markdown(
            f'<div style="display:flex;gap:7px;margin:.2rem 0;align-items:flex-start">'
            f'<span style="{_M}font-size:0.50rem;color:#c0392b;flex-shrink:0;margin-top:2px">■</span>'
            f'<span style="{_S}font-size:0.69rem;color:#b8b8b8;flex:1;line-height:1.5">{text}</span>'
            + _badge_html(level) +
            f'</div>',
            unsafe_allow_html=True,
        )

    col_a2, col_l2 = st.columns(2)
    with col_a2:
        _h3("Key Assumptions")
        _assumption_b("Conflict intensity dimensions are observable and monotonically related to market risk.", "MED")
        _assumption_b("Transmission channel weights are stable across conflict types - not conflict-specific.", "HIGH")
        _assumption_b("Four-state Markov chain adequately captures the regime space - higher granularity not supported by sample size.", "MED")
        _assumption_b("Rolling Pearson correlation is a sufficient proxy for time-varying dependence at daily frequency.", "MED")
        _assumption_b("Yahoo Finance daily close prices are unbiased proxies for commodity spot prices (front-month futures).", "LOW")
        _assumption_b("DCC(1,1) parameters (a=0.05, b=0.90) and EWMA λ=0.94 are fixed globally - not MLE-estimated or conflict-specific.", "HIGH")
        _assumption_b("SECURITY_EXPOSURE structural weights are point-in-time estimates and subject to drift.", "HIGH")
        _assumption_b("RSS feed headlines are a representative sample of geopolitical events - coverage gaps exist.", "MED")
    with col_l2:
        _h3("Known Limitations")
        _limitation_b("CIS and TPS dimensions are manually scored - subject to analyst judgment and update lag.", "HIGH")
        _limitation_b("Model uses public equity indices and commodity futures; direct exposure data (loan books, supply contracts) unavailable.", "HIGH")
        _limitation_b("Granger and DY tests assume linear VAR dynamics - nonlinear regime-switching relationships not fully captured.", "MED")
        _limitation_b("News GPR uses keyword matching, not semantic NLP - 'no attack' style negations may be misclassified.", "HIGH")
        _limitation_b("Conflict registry requires manual maintenance; stale data degrades CIS/TPS accuracy.", "HIGH")
        _limitation_b("Scenario multipliers applied uniformly - no differentiation by asset class or geography.", "MED")
        _limitation_b("COT positioning data has a 3-day publication lag - contrarian signal reflects week-old positioning.", "LOW")
        _limitation_b("Transfer entropy estimation is sensitive to bin-width and sample size - small samples produce noise.", "MED")

    _h3("Model Maintenance Protocol  ·  SR 11-7")
    _maint_rows = [
        ("WEEKLY",    _G,        "Review conflict registry for state changes (active → latent). Update last_updated fields. Check freshness cap violations."),
        ("MONTHLY",   "#e67e22", "Recalibrate regime thresholds as new data accumulates. Validate DCC-style correlation stationarity. Review MCS z-score distributions."),
        ("QUARTERLY", "#c0392b", "Re-examine CIS dimension weights against realized commodity price responses to recent conflicts. Backtest risk score vs. VIX spikes."),
        ("ANNUALLY",  _DIM,     "Full model review - reassess 4-regime Markov structure; re-estimate TPS channel weights; refresh SECURITY_EXPOSURE structural weights."),
    ]
    _maint_html = '<div style="border:1px solid #1e1e1e;overflow:hidden;margin:.3rem 0">'
    for _ml, _mc, _md in _maint_rows:
        _maint_html += (
            f'<div style="display:flex;gap:0;align-items:stretch;border-bottom:1px solid #111">'
            f'<div style="width:3px;background:{_mc};flex-shrink:0"></div>'
            f'<div style="padding:.35rem .7rem;width:90px;flex-shrink:0;display:flex;'
            f'align-items:center;background:#040404">'
            f'<span style="{_M}font-size:0.50rem;font-weight:700;color:{_mc}">{_ml}</span></div>'
            f'<div style="padding:.35rem .7rem;flex:1;background:#080808">'
            f'<span style="{_S}font-size:0.68rem;color:{_DIM}">{_md}</span></div>'
            f'</div>'
        )
    _maint_html += '</div>'
    st.markdown(_maint_html, unsafe_allow_html=True)

    _takeaway_block(
        "This dashboard implements a four-layer geopolitical risk framework grounded in "
        "academic spillover literature (Caldara-Iacoviello 2022 GPR, Diebold-Yilmaz 2009/2012, Engle DCC 2002) and "
        "practitioner scenario design conventions (SR 11-7, CCAR). "
        "News flow leads as the primary signal; conflict event intensity, chokepoint disruption, "
        "and oil-gold market confirmation follow. All formulae, weights, "
        "and assumptions are fully transparent and documented above. The model is designed "
        "to be challenged, stress-tested, and improved — contributions to the conflict registry "
        "and news GPR keyword taxonomy are the highest-leverage points of intervention."
    )

    _page_footer()
