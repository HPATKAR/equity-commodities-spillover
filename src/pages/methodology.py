"""
Model Methodology & Documentation Page.

Comprehensive technical documentation for all analytical modules in the
Cross-Asset Spillover Monitor. Covers data sources, formulae, assumptions,
implementation choices, and limitations — following industry model-documentation
standards (SR 11-7 / OCC 2011-12 model risk management guidance).
"""

from __future__ import annotations

import streamlit as st
from src.ui.shared import (
    _page_header, _page_footer, _page_intro,
    _definition_block, _takeaway_block, _section_note,
)

_M  = "font-family:'JetBrains Mono',monospace;"
_S  = "font-family:'DM Sans',sans-serif;"
_G  = "#CFB991"
_DIM = "#8E9AAA"
_MUT = "#555960"

# ── Sub-components ─────────────────────────────────────────────────────────────

def _h2(text: str) -> None:
    st.markdown(
        f'<p style="{_M}font-size:8px;font-weight:700;letter-spacing:.20em;'
        f'color:{_G};text-transform:uppercase;border-bottom:1px solid #1e1e1e;'
        f'padding-bottom:5px;margin:1.4rem 0 .6rem">{text}</p>',
        unsafe_allow_html=True,
    )

def _h3(text: str) -> None:
    st.markdown(
        f'<p style="{_M}font-size:8px;font-weight:700;letter-spacing:.12em;'
        f'color:{_DIM};text-transform:uppercase;margin:.8rem 0 .3rem">{text}</p>',
        unsafe_allow_html=True,
    )

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
        f'<code style="{_M}font-size:11px;color:{_G};white-space:pre-wrap">'
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
        f'<th style="{_M}font-size:7px;color:{_MUT};text-align:left;border-bottom:1px solid #2a2a2a;padding:3px 8px">DIMENSION / CHANNEL</th>'
        f'<th style="{_M}font-size:7px;color:{_MUT};text-align:right;border-bottom:1px solid #2a2a2a;padding:3px 8px">WEIGHT</th>'
        f'<th style="{_M}font-size:7px;color:{_MUT};text-align:left;border-bottom:1px solid #2a2a2a;padding:3px 8px">RATIONALE</th>'
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
        f'<span style="{_M}font-size:9px;font-weight:700;color:#0a0a0a;'
        f'background:{color};padding:1px 5px;flex-shrink:0">{pct}</span>'
        f'<span style="{_M}font-size:7.5px;font-weight:700;color:#e8e9ed;'
        f'letter-spacing:.08em;text-transform:uppercase">{name}</span>'
        f'</div>'
        # tagline
        f'<p style="{_S}font-size:0.67rem;color:{_DIM};margin:0 0 .35rem;line-height:1.5">'
        f'{tagline}</p>'
        # formula block
        f'<div style="background:#040404;border-left:2px solid {color};'
        f'padding:.3rem .5rem;margin:.2rem 0">'
        f'<code style="{_M}font-size:8.5px;color:{color};white-space:pre-wrap;line-height:1.6">'
        f'{formula_lines}</code></div>'
        + (f'<p style="{_S}font-size:0.60rem;color:{_MUT};margin:.3rem 0 0;'
           f'line-height:1.4;font-style:italic">{note}</p>' if note else "")
        + '</div>',
        unsafe_allow_html=True,
    )


def _arch_flow() -> None:
    """Visual 3-layer architecture flow for the composite risk score."""
    _BG  = "#080808"
    _BRD = "#1e1e1e"
    layers = [
        ("CIS", "40%", "#c0392b",  "Conflict Intensity",   "7 weighted dimensions · per-conflict · state multiplier · staleness cap"),
        ("TPS", "35%", "#e67e22",  "Transmission Pressure","12 channel weights · supply chain · sanctions · chokepoint · FX"),
        ("MCS", "25%", "#CFB991",  "Market Confirmation",  "6 orthogonalized price signals · EWM z-scores · live feed"),
    ]
    # build layer boxes
    boxes_html = ""
    for abbr, pct, col, full, detail in layers:
        boxes_html += (
            f'<div style="flex:1;background:{_BG};border:1px solid #2a2a2a;'
            f'border-top:3px solid {col};padding:.5rem .7rem;min-width:0">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'
            f'<span style="{_M}font-size:13px;font-weight:700;color:{col}">{abbr}</span>'
            f'<span style="{_M}font-size:9px;font-weight:700;color:{col};'
            f'background:rgba(255,255,255,0.04);padding:1px 5px">{pct}</span></div>'
            f'<div style="{_M}font-size:7px;font-weight:700;color:#e8e9ed;'
            f'letter-spacing:.08em;text-transform:uppercase;margin-bottom:3px">{full}</div>'
            f'<div style="{_S}font-size:0.62rem;color:{_MUT};line-height:1.4">{detail}</div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="margin:.6rem 0">'
        # top row — 3 layer boxes
        f'<div style="display:flex;gap:6px;margin-bottom:6px">{boxes_html}</div>'
        # arrow row
        f'<div style="display:flex;align-items:center;gap:0;margin-bottom:6px">'
        f'<div style="flex:1;height:1px;background:#2a2a2a"></div>'
        f'<div style="padding:0 8px;{_M}font-size:7px;color:{_MUT}">▼ CIS-weighted avg → portfolio score</div>'
        f'<div style="flex:1;height:1px;background:#2a2a2a"></div>'
        f'</div>'
        # middle row — assembly + freshness
        f'<div style="display:flex;gap:6px;margin-bottom:6px">'
        f'<div style="flex:2;background:{_BG};border:1px solid #2a2a2a;padding:.4rem .7rem">'
        f'<span style="{_M}font-size:7px;font-weight:700;color:#e8e9ed;letter-spacing:.1em">ASSEMBLY</span><br>'
        f'<code style="{_M}font-size:9px;color:{_G}">GRS_raw = 0.40·CIS + 0.35·TPS + 0.25·MCS</code>'
        f'</div>'
        f'<div style="flex:1;background:{_BG};border:1px solid #2a2a2a;padding:.4rem .7rem">'
        f'<span style="{_M}font-size:7px;font-weight:700;color:#e8e9ed;letter-spacing:.1em">MARKET FRESHNESS</span><br>'
        f'<code style="{_M}font-size:9px;color:#8E9AAA">rank by CIS × freshness ∈ [0.7, 1.5]</code>'
        f'</div>'
        f'<div style="flex:1;background:{_BG};border:1px solid #2a2a2a;padding:.4rem .7rem">'
        f'<span style="{_M}font-size:7px;font-weight:700;color:#e8e9ed;letter-spacing:.1em">SCENARIO MULT</span><br>'
        f'<code style="{_M}font-size:9px;color:#8E9AAA">× geo_mult  (default 1.0)</code>'
        f'</div>'
        f'</div>'
        # arrow row
        f'<div style="display:flex;align-items:center;gap:0;margin-bottom:6px">'
        f'<div style="flex:1;height:1px;background:#2a2a2a"></div>'
        f'<div style="padding:0 8px;{_M}font-size:7px;color:{_MUT}">▼ clip(·, 0, 100)</div>'
        f'<div style="flex:1;height:1px;background:#2a2a2a"></div>'
        f'</div>'
        # output box
        f'<div style="background:#0a0a0a;border:1px solid {_G};padding:.45rem 1rem;text-align:center">'
        f'<span style="{_M}font-size:9px;font-weight:700;color:{_G};letter-spacing:.18em">GRS</span>'
        f'<span style="{_M}font-size:8px;color:{_MUT};margin-left:10px">0 – 100 · Low / Moderate / Elevated / High</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _freshness_range() -> None:
    """Visual [0.7 … 1.0 … 1.5] range bar for the market freshness multiplier."""
    st.markdown(
        f'<div style="margin:.5rem 0 .8rem;padding:.5rem .8rem;background:#080808;border:1px solid #1e1e1e">'
        f'<div style="{_M}font-size:7px;color:{_MUT};letter-spacing:.12em;margin-bottom:.4rem">MARKET FRESHNESS RANGE  ∈  [0.7, 1.5]</div>'
        f'<div style="position:relative;height:6px;background:#1a1a1a;margin:.3rem 0">'
        # colored gradient bar
        f'<div style="position:absolute;left:0;top:0;width:100%;height:6px;'
        f'background:linear-gradient(to right,#2e7d32,#8E9AAA 37.5%,#CFB991 62.5%,#c0392b)"></div>'
        # tick marks + labels
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:.2rem">'
        f'<span style="{_M}font-size:7.5px;color:#2e7d32">0.7× quiet market</span>'
        f'<span style="{_M}font-size:7.5px;color:#8E9AAA">1.0× neutral</span>'
        f'<span style="{_M}font-size:7.5px;color:#c0392b">1.5× crisis signal</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _confidence_component(label: str, weight: str, formula: str, note: str, color: str = _G) -> None:
    st.markdown(
        f'<div style="display:flex;gap:10px;align-items:flex-start;'
        f'padding:.4rem .6rem;border-bottom:1px solid #111;background:#080808">'
        f'<div style="flex-shrink:0;text-align:right;width:36px">'
        f'<span style="{_M}font-size:10px;font-weight:700;color:{color}">{weight}</span></div>'
        f'<div style="flex:1">'
        f'<div style="{_M}font-size:7.5px;font-weight:700;color:#e8e9ed;letter-spacing:.07em;'
        f'text-transform:uppercase;margin-bottom:2px">{label}</div>'
        f'<code style="{_M}font-size:8.5px;color:{_DIM}">{formula}</code><br>'
        f'<span style="{_S}font-size:0.62rem;color:{_MUT}">{note}</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def _assumption(text: str) -> None:
    st.markdown(
        f'<div style="display:flex;gap:8px;margin:.15rem 0">'
        f'<span style="{_M}font-size:8px;color:#e67e22;margin-top:1px">▲</span>'
        f'<span style="{_S}font-size:0.70rem;color:#b8b8b8">{text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

def _limitation(text: str) -> None:
    st.markdown(
        f'<div style="display:flex;gap:8px;margin:.15rem 0">'
        f'<span style="{_M}font-size:8px;color:#c0392b;margin-top:1px">■</span>'
        f'<span style="{_S}font-size:0.70rem;color:#b8b8b8">{text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

def _data_source(name: str, what: str, freq: str, lag: str) -> None:
    st.markdown(
        f'<div style="background:#080808;border:1px solid #1e1e1e;padding:.45rem .8rem;margin:.3rem 0">'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<span style="{_M}font-size:9px;font-weight:700;color:{_G}">{name}</span>'
        f'<span style="{_M}font-size:7px;color:{_MUT}">Freq: {freq} · Lag: {lag}</span>'
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
        "in the Cross-Asset Spillover Monitor. It follows the structure recommended by "
        "<strong>SR 11-7 / OCC 2011-12</strong> model risk management guidance: "
        "purpose, design logic, implementation, assumptions, limitations, and ongoing monitoring. "
        "Each section covers one analytical module — from raw data ingestion through to "
        "the composite risk score, trade signal generation, and portfolio stress testing."
    )

    # ── Navigation index ───────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:#080808;border:1px solid #1e1e1e;padding:.6rem 1rem;margin-bottom:1rem">'
        f'<p style="{_M}font-size:7px;color:{_MUT};letter-spacing:.15em;margin-bottom:5px">CONTENTS</p>'
        + "".join(
            f'<span style="{_M}font-size:8px;color:{_DIM};margin-right:16px">'
            f'<span style="color:{_G}">{i+1}.</span> {s}</span>'
            for i, s in enumerate([
                "Data Architecture", "Conflict Intensity Score (CIS)",
                "Transmission Pressure Score (TPS)", "Market Confirmation Score (MCS)",
                "Composite Risk Score", "News GPR Layer",
                "Correlation Regime Engine", "Spillover Network",
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
        "All market data is sourced from publicly available APIs. No proprietary or paid data "
        "feeds are used. Data is cached to reduce latency (yfinance: 1h TTL; FRED: 24h TTL; "
        "RSS: 15 min TTL). Stale-data indicators are shown on each chart."
    )

    col1, col2 = st.columns(2)
    with col1:
        _data_source("Yahoo Finance", "15 equity indices, 17 commodity futures, FX spot rates, implied vol proxies (VIX, OVX, GVZ)", "Daily close", "1 day")
        _data_source("FRED (St. Louis Fed)", "10Y / 2Y Treasury yields, yield spread, Fed Funds Rate, CPI, PCE, industrial production, payrolls, GDP", "Monthly / Daily", "1–30 days")
        _data_source("CFTC COT Reports", "Commitment of Traders positioning data for commodity futures — large speculator net longs as contrarian signal", "Weekly", "3 days")
    with col2:
        _data_source("RSS News Feeds", "Reuters, AP, BBC, NYT, WSJ, FT geopolitical headlines — keyword-tagged, severity-scored, conflict-routed", "Near-real-time", "~15 min")
        _data_source("Manual Conflict Registry", "CONFLICTS config: per-conflict intensity dimensions, transmission channel weights, state, last_updated", "Manually maintained", "Varies")
        _data_source("Yahoo Finance (FX Spot)", "Live spot rates for portfolio FX conversion — e.g. GBPUSD=X, EURUSD=X. 5-day lookback for latest close", "Daily", "1 day")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. CIS
    # ══════════════════════════════════════════════════════════════════════════
    _h2("2 · Conflict Intensity Score (CIS)")
    _prose(
        "The CIS is a per-conflict composite score (0–100) that quantifies how intense and "
        "market-relevant a given armed conflict is <em>right now</em>. It is built from seven "
        "orthogonalized intensity dimensions, weighted by their historical association with "
        "commodity price disruption, and adjusted by a state multiplier that reflects whether "
        "the conflict is actively evolving or has stabilised."
    )

    _h3("Formula")
    _formula(
        "CIS(conflict) = [ Σᵢ wᵢ × dᵢ ] × state_mult × 100",
        "where dᵢ ∈ [0,1] is the normalized value of dimension i, wᵢ is its weight, "
        "and state_mult reflects active / latent / frozen classification."
    )

    _h3("Dimension Weights")
    _weight_table([
        ("deadliness",           "Deadliness",               0.22, "Direct casualty count normalised to 0–1 scale. Highest weight — most reliably observable dimension."),
        ("escalation_trend",     "Escalation Trend",          0.20, "escalating=1.0, stable=0.5, de-escalating=0.0. Second-highest: direction drives forward-looking risk."),
        ("civilian_danger",      "Civilian Danger",           0.15, "Targeting of civilian infrastructure. Proxy for conflict indiscriminacy and displacement risk."),
        ("recency",              "Recency",                   0.13, "Exponential decay: 1.0 at day 1 → 0.3 at 365 days (half-life ~2 years). Latent/frozen: flat 0.35/0.15."),
        ("source_coverage",      "Source Coverage",           0.10, "Breadth of credible reporting. Proxy for data confidence; low coverage penalises the score."),
        ("geographic_diffusion", "Geographic Diffusion",      0.12, "Spatial spread of hostilities. Higher diffusion → broader supply-chain and FX disruption."),
        ("fragmentation",        "Fragmentation",             0.08, "Number of armed factions. Higher fragmentation → harder to resolve → longer tail risk."),
    ])

    _h3("State Multipliers")
    st.markdown(
        f'<div style="display:flex;gap:10px;margin:.4rem 0">'
        + "".join(
            f'<div style="background:#080808;border:1px solid #1e1e1e;padding:.4rem .7rem;flex:1">'
            f'<span style="{_M}font-size:8px;font-weight:700;color:{c}">{s}</span><br>'
            f'<span style="{_M}font-size:16px;font-weight:700;color:{c}">{m}×</span><br>'
            f'<span style="{_S}font-size:0.65rem;color:{_DIM}">{d}</span></div>'
            for s, m, c, d in [
                ("ACTIVE",  "1.00", "#c0392b", "Full score — hostilities ongoing"),
                ("LATENT",  "0.35", "#e67e22", "No active fighting; structural risk persists"),
                ("FROZEN",  "0.15", _DIM,      "Ceasefire or prolonged stalemate"),
            ]
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 3. TPS
    # ══════════════════════════════════════════════════════════════════════════
    _h2("3 · Transmission Pressure Score (TPS)")
    _prose(
        "The TPS measures how strongly a conflict's geography and nature are expected to "
        "transmit into specific commodity and financial markets — independent of CIS intensity. "
        "A conflict can be intense (high CIS) but geographically isolated from key supply chains "
        "(low TPS). Both dimensions are needed."
    )

    _h3("Formula")
    _formula(
        "TPS(conflict) = [ Σ_ch w_ch × tx_ch ] × state_mult × 100",
        "where tx_ch ∈ [0,1] is the analyst-assigned transmission weight for channel ch, "
        "and w_ch is the channel's structural market importance weight."
    )

    _h3("Channel Weights")
    _weight_table([
        ("oil_gas",       "Oil / Gas",         0.18, "Highest weight — energy supply disruption is the primary geopolitical commodity shock mechanism."),
        ("sanctions",     "Sanctions",          0.12, "Financial sanctions create direct market dislocations; commodity sanctions amplify oil/metals moves."),
        ("shipping",      "Shipping / Freight", 0.12, "Rerouting costs, insurance premia, freight indices. Critical when straits are threatened."),
        ("chokepoint",    "Chokepoint",         0.10, "Hormuz, Suez, Bosphorus — ~51% of global seaborne oil transit. Binary risk event."),
        ("metals",        "Metals",             0.10, "Nickel, aluminium, copper — key inputs for defence, construction, EVs."),
        ("inflation",     "Inflation",          0.07, "Second-order: commodity shock → CPI → central bank response → equity re-rating."),
        ("equity_sector", "Equity Sector",      0.08, "Direct equity exposure: defence, energy, shipping sectors."),
        ("agriculture",   "Agriculture",        0.08, "Black Sea corridor, wheat/corn supply routes. Highly relevant for Russia/Ukraine."),
        ("fx",            "FX",                 0.06, "Safe-haven USD bid, emerging-market currency stress, petrocurrency correlations."),
        ("supply_chain",  "Supply Chain",       0.05, "Input shortages, factory closures, logistics re-routing. Slower-moving signal."),
        ("credit",        "Credit",             0.02, "Sovereign CDS spreads, EM bond stress. Lagging indicator."),
        ("energy_infra",  "Energy Infra",       0.02, "Pipeline and grid attacks. Narrow but high-impact when triggered."),
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # 4. MCS
    # ══════════════════════════════════════════════════════════════════════════
    _h2("4 · Market Confirmation Score (MCS)")
    _prose(
        "The MCS is the market-observable layer of the risk score — it uses six live price signals to "
        "<em>confirm or contradict</em> the structural CIS/TPS verdict. High CIS with calm markets → MCS "
        "dampens the score (markets have priced it in). Low CIS with spiking volatility → MCS amplifies. "
        "All signals are orthogonalized to remove multicollinearity before weighting."
    )

    _formula(
        "MCS = 0.22·SafeHaven + 0.18·OilGold + 0.15·EqVol\n"
        "    + 0.15·RatesVol  + 0.15·CmdVol  + 0.15·CorrAccel",
        "All six sub-signals normalised to [0, 100] via EWM z-scoring before weighting. "
        "CorrAccel replaces correlation-level percentile to avoid double-counting CIS geographic_diffusion."
    )

    _h3("Signal Cards")
    # Row 1
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        _signal_card(
            "safe_haven", "Safe-Haven Bid", 0.22, "#CFB991",
            "Gold 20d cumulative return EWM z-scored (span=60). "
            "TLT bond ETF adds corroboration: if Gold z > 0.5 and TLT return z > 0.3, "
            "+min(tlt_z×4, 10) pts. Filters USD-specific gold rallies from genuine flight-to-safety.",
            "SafeHaven = clip(50 + g_z×14 + tlt_boost, 0, 100)\n"
            "tlt_boost = min(tlt_z×4, 10) if g_z>0.5 and tlt_z>0.3\n"
            "          = 0  otherwise",
            "tlt_z = 20d TLT return / (std × √20). Cached fetch, TTL=300."
        )
    with r1c2:
        _signal_card(
            "oil_gold", "Oil-Gold Signal", 0.18, "#e67e22",
            "Simultaneous EWM z-scores of 20d cumulative log-returns for Gold and front Oil contract. "
            "Flat +5 bonus when both z > 1 — regime-invariant joint geopolitical premium. "
            "Avoids multiplicative zero-collapse when one signal is near-neutral.",
            "OilGold = clip(50 + g_z×14 + o_z×8 + bonus, 0, 100)\n"
            "bonus   = 5.0  if g_z > 1.0 and o_z > 1.0\n"
            "        = 0.0  otherwise",
            "WTI preferred; falls back to Brent. EWM span=60, clip g_z/o_z to [−4, +4]."
        )
    with r1c3:
        _signal_card(
            "eq_vol", "Equity Vol", 0.15, "#8E9AAA",
            "20d realized annualized volatility across S&P 500, Eurostoxx 50, and Nikkei 225 — "
            "averaged, then EWM z-scored (span=60). Measures broad equity fear. "
            "z = 0 → score 50; z = +2 → score ≈ 74.",
            "rv     = eq_r.rolling(20).std() × √252 × 100\n"
            "z      = EWM_zscore(rv.mean(axis=1), span=60)\n"
            "EqVol  = clip(50 + z×12, 0, 100)",
            "Falls back to first 3 columns if named indices unavailable."
        )
    # Row 2
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        _signal_card(
            "rates_vol", "Rates Vol", 0.15, "#8E9AAA",
            "MOVE-proxy: 20d realized vol of TLT (iShares 20Y+ Treasury ETF) price returns. "
            "More orthogonal to equity vol than VIX. Captures flight-to-safety and rates-channel "
            "transmission of geopolitical shocks (yield compression = risk-off bid).",
            "tlt_r     = TLT.pct_change()\n"
            "rv_tlt    = tlt_r.rolling(20).std() × √252 × 100\n"
            "RatesVol  = clip(50 + z×10, 0, 100)",
            "Shared cached 200d TLT fetch (TTL=300). Scale=10 (less sensitive than equity)."
        )
    with r2c2:
        _signal_card(
            "cmd_vol", "Commodity Vol (residual)", 0.15, "#8E9AAA",
            "Energy/metals 20d annualized vol (WTI, Brent, NatGas, Gold, Silver, Copper) "
            "OLS-residualized on equity vol over trailing 252 days. "
            "Strips equity-fear component — isolates commodity-specific supply-disruption stress.",
            "resid  = rv_cmd − β_OLS × rv_eq\n"
            "β_OLS  = polyfit(rv_eq[-252:], rv_cmd[-252:], 1)[0]\n"
            "CmdVol = clip(50 + EWM_zscore(resid)×11, 0, 100)",
            "Falls back to raw rv_cmd if equity data unavailable or < 30 overlapping obs."
        )
    with r2c3:
        _signal_card(
            "corr_accel", "Correlation Accel", 0.15, "#27ae60",
            "Second derivative of the EWM-smoothed (span=20) average equity-commodity correlation. "
            "Captures whether cross-asset coupling is accelerating, not just its level. "
            "Orthogonal to CIS geographic_diffusion (level-based) — eliminates double-counting.",
            "smooth    = avg_corr.ewm(span=20).mean()\n"
            "velocity  = smooth.diff()\n"
            "accel     = velocity.ewm(span=20).mean()\n"
            "CorrAccel = pct_rank(accel) × 100  ∈ [0, 100]",
            "Fallback: 50 if < 90 observations. Replaced old correlation-percentile signal."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 5. COMPOSITE RISK SCORE
    # ══════════════════════════════════════════════════════════════════════════
    _h2("5 · Composite Geopolitical Risk Score")
    _prose(
        "The composite score assembles CIS, TPS, and MCS into a single 0–100 index. "
        "A session-dynamic market freshness multiplier re-ranks conflicts by <em>today's</em> market moves "
        "before aggregation — ensuring the command center reflects live crises, not just historical intensity. "
        "A scenario geo-multiplier is applied after assembly for stress testing."
    )

    _h3("Architecture")
    _arch_flow()

    _h3("Layer Weights & Score Bands")
    col_lw, col_sb = st.columns([1, 1])
    with col_lw:
        # Proportional weight bars
        for abbr, full, pct, col, val, rationale in [
            ("CIS", "Conflict Intensity",    "40%", "#c0392b", 40,
             "Structural severity — direction and magnitude of active hostilities."),
            ("TPS", "Transmission Pressure", "35%", "#e67e22", 35,
             "Channel specificity — which commodities and markets are exposed."),
            ("MCS", "Market Confirmation",   "25%", _G,        25,
             "Live signal — confirms or dampens the structural signal."),
        ]:
            st.markdown(
                f'<div style="margin:.35rem 0">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px">'
                f'<span style="{_M}font-size:7px;font-weight:700;color:{col}">{abbr} · {full}</span>'
                f'<span style="{_M}font-size:9px;font-weight:700;color:{col}">{pct}</span></div>'
                f'<div style="background:#1a1a1a;height:4px;position:relative">'
                f'<div style="background:{col};width:{val*2}px;height:4px;max-width:100%"></div></div>'
                f'<div style="{_S}font-size:0.62rem;color:{_MUT};margin-top:2px">{rationale}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    with col_sb:
        st.markdown(
            f'<div style="background:#080808;border:1px solid #1e1e1e;padding:.5rem .7rem">'
            f'<p style="{_M}font-size:7px;color:{_MUT};letter-spacing:.12em;margin-bottom:.4rem">SCORE BANDS</p>'
            + "".join(
                f'<div style="display:flex;align-items:center;gap:8px;padding:.25rem 0;'
                f'border-bottom:1px solid #111">'
                f'<div style="width:4px;height:24px;background:{c};flex-shrink:0"></div>'
                f'<div>'
                f'<span style="{_M}font-size:7px;font-weight:700;color:{c}">{band} · {lo}–{hi}</span><br>'
                f'<span style="{_S}font-size:0.62rem;color:{_MUT}">{interp}</span>'
                f'</div></div>'
                for band, lo, hi, c, interp in [
                    ("HIGH/CRISIS", 75, 100, "#c0392b", "Active conflict + market stress confirmed"),
                    ("ELEVATED",    50,  75, "#e67e22", "Significant transmission pressure building"),
                    ("MODERATE",    25,  50, _G,        "Latent risk or mild market dislocations"),
                    ("LOW",          0,  25, "#27ae60", "Calm — no active confirmed spillover"),
                ]
            )
            + f'</div>',
            unsafe_allow_html=True,
        )

    _h3("Portfolio Aggregation")
    _formula(
        "CIS_portfolio = Σᵢ [ CIS(i) × CIS(i) ] / Σᵢ CIS(i)   ← intensity-weighted avg\n"
        "TPS_portfolio = Σᵢ [ TPS(i) × CIS(i) ] / Σᵢ CIS(i)   ← same weights\n\n"
        "GRS_raw = 0.40 × CIS_portfolio + 0.35 × TPS_portfolio + 0.25 × MCS\n"
        "GRS     = clip( GRS_raw × geo_mult , 0, 100 )",
        "High-CIS conflicts dominate. Latent/frozen conflicts already discounted via state_mult in CIS computation."
    )

    _h3("Market Freshness Multiplier")
    _prose(
        "Which conflict is <em>currently moving markets</em> differs session-to-session. "
        "The freshness layer maps each conflict's live transmission channel activity to a multiplier "
        "∈ [0.7, 1.5]. The command center's Lead Conflict is ranked by "
        "<code style='font-family:monospace;font-size:0.75em'>CIS × market_freshness</code>, "
        "not raw CIS — so a quiet Russia day won't outrank an active Hormuz blockade."
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
        f'<p style="{_M}font-size:7px;font-weight:700;color:#c0392b;margin:0 0 4px;letter-spacing:.08em">'
        f'EXAMPLE  ·  HORMUZ BLOCKADE  (high freshness)</p>'
        f'<p style="{_S}font-size:0.67rem;color:{_DIM};margin:0;line-height:1.5">'
        f'Iran/Hormuz: oil_gas=0.97, chokepoint=0.97. Brent +4% intraday, PortWatch disruption=0.40.<br>'
        f'oil_sig ≈ 0.97 × max(1.33, 1.33, 0.40) = 1.29 → mean([1.0]) = 1.0 →<br>'
        f'<b style="color:#c0392b">market_freshness = 1.5 (capped) → Hormuz ranks #1</b></p>'
        f'</div>'
        f'<div style="flex:1;background:#080808;border:1px solid #1e1e1e;'
        f'border-left:3px solid #27ae60;padding:.45rem .7rem">'
        f'<p style="{_M}font-size:7px;font-weight:700;color:#27ae60;margin:0 0 4px;letter-spacing:.08em">'
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
    _prose("Conflicts not updated within threshold windows receive automatic penalties — "
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
            f'<div style="{_M}font-size:7px;font-weight:700;color:{c};letter-spacing:.08em;margin-bottom:3px">{s}</div>'
            f'<div style="{_M}font-size:8px;color:#e8e9ed;margin-bottom:2px">{threshold}</div>'
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
        "quadrature (RSS) combination of per-layer errors — wider when data is stale or market signals disagree."
    )
    # Confidence components
    st.markdown(
        f'<div style="border:1px solid #1e1e1e;margin:.4rem 0 .2rem">'
        f'<div style="background:#0a0a0a;padding:.35rem .7rem;border-bottom:1px solid #1e1e1e">'
        f'<span style="{_M}font-size:7px;font-weight:700;color:{_G};letter-spacing:.12em">'
        f'CONFIDENCE  =  0.50 × conflict_conf  +  0.30 × mcs_agreement  +  0.20 × data_availability</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    _confidence_component(
        "Conflict Confidence", "50%",
        "0.30×source_cov + 0.25×data_conf + 0.25×freshness + 0.20×completeness",
        "CIS-intensity-weighted avg across active conflicts. Staleness multipliers applied (×0.90 at 90d, ×0.70 at 180d).",
        "#c0392b",
    )
    _confidence_component(
        "MCS Signal Agreement", "30%",
        "1 − std(MCS sub-signals) / 50",
        "High agreement (all 6 signals point same direction) → score near 1.0. Divergence → lower weight on MCS.",
        "#e67e22",
    )
    _confidence_component(
        "Data Availability", "20%",
        "1.0 if avg_corr non-empty  ·  0.5 otherwise",
        "Penalizes sessions where cross-asset correlation series is unavailable (data load failure).",
        _G,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    _formula(
        "σ_CIS  = (1 − conflict_conf) × CIS_portfolio × 0.40\n"
        "σ_TPS  = (1 − conflict_conf) × TPS_portfolio × 0.35\n"
        "σ_MCS  = (1 − mcs_agreement) × MCS           × 0.25\n\n"
        "Uncertainty = √( σ_CIS² + σ_TPS² + σ_MCS² )\n"
        "score_low   = max(0,   GRS − Uncertainty)\n"
        "score_high  = min(100, GRS + Uncertainty)",
        "Quadrature (RSS) — assumes layer errors are independent. "
        "At 100% confidence → Uncertainty = 0. At 0% → uncertainty = max layer-weighted error."
    )

    _h3("Historical Score  (risk_score_history)")
    _prose(
        "The historical chart plots a daily rolling index over the selected date range. "
        "Structural CIS/TPS are not available at daily frequency (point-in-time analyst snapshots), "
        "so the historical series uses market-observable signals only — weighted to approximate "
        "the live model's dominant drivers."
    )
    _weight_table([
        ("eq_vol",     "Equity Vol (hist.)",    0.40, "Dominant live driver — 20d realized vol EWM z-score. Matches live construction exactly."),
        ("oil_gold",   "Oil-Gold (hist.)",       0.35, "Same construction as live: g_z×14 + o_z×8 + flat +5 bonus. Rolling EWM z-scores."),
        ("cmd_vol",    "Commodity Vol (hist.)",  0.13, "Residualized commodity vol — same construction as live MCS sub-signal."),
        ("corr_accel", "Corr. Accel (hist.)",   0.12, "2nd derivative of corr — same construction as live. Replaced old corr-pct (was 0.30 weight)."),
    ])
    _formula(
        "HistScore(t) = 0.40·EqVol(t) + 0.35·OilGold(t)\n"
        "             + 0.13·CmdVol(t) + 0.12·CorrAccel(t)",
        "Rolling daily. All signals EWM z-scored (span=60), clipped to [0, 100] before weighting."
    )
    _section_note(
        "HistScore ≠ GRS on any given day. The live score includes 40%+35% structural layers (CIS/TPS) "
        "which are not backcasted. Treat HistScore as a market-stress proxy — directionally aligned "
        "with GRS but not numerically equivalent."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 6. NEWS GPR LAYER
    # ══════════════════════════════════════════════════════════════════════════
    _h2("6 · News GPR Layer — Threat / Act Classification")
    _prose(
        "The News GPR layer extends the structural CIS/TPS signal with a real-time news feed "
        "that classifies headlines into <b>Threat</b> (rhetoric, military signalling, diplomatic breakdown) "
        "and <b>Act</b> (realized events: strikes, sanctions imposed, vessels seized). "
        "This distinction matters: Threat scores are leading indicators; Act scores are contemporaneous."
    )

    _h3("Pipeline")
    steps = [
        ("Ingest", "RSS feeds polled every 15 min from Reuters, AP, BBC, NYT, WSJ, FT"),
        ("Tag", "Keyword matching against regional taxonomy (8 regions) and commodity taxonomy (6 categories)"),
        ("Score", "Relevance score = (regions×10 + commodities×12 + severity_score) × source_weight"),
        ("Classify", "Threat keywords (warned, mobilize, ultimatum…) vs Act keywords (attack, seized, explosion…)"),
        ("Route", "High-confidence headlines (relevance ≥ 65) auto-routed to Geopolitical Analyst agent review queue"),
    ]
    for i, (step, desc) in enumerate(steps):
        st.markdown(
            f'<div style="display:flex;gap:10px;align-items:flex-start;margin:.2rem 0">'
            f'<span style="{_M}font-size:7px;font-weight:700;color:{_G};'
            f'background:#0a0a0a;border:1px solid #2a2a2a;padding:2px 6px;flex-shrink:0">'
            f'0{i+1}</span>'
            f'<div><span style="{_M}font-size:8px;font-weight:700;color:#e8e9ed">{step}</span>'
            f'<span style="{_S}font-size:0.70rem;color:{_DIM};margin-left:8px">{desc}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    _h3("Dynamic Alpha Weighting")
    _formula(
        "News GPR = α · Act_score + (1 − α) · Threat_score\n"
        "α = clip( n_act / (n_act + n_threat + ε) × 2, 0.2, 0.8 )",
        "α rises when act headlines dominate; falls when only rhetoric is present. "
        "The News GPR layer is diagnostic — it does not feed directly into the composite GRS."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 7. CORRELATION REGIME ENGINE
    # ══════════════════════════════════════════════════════════════════════════
    _h2("7 · Correlation Regime Engine")

    col_a, col_b = st.columns(2)
    with col_a:
        _h3("Rolling Pearson Correlation")
        _prose(
            "Pairwise rolling Pearson coefficient between log-return series of equity "
            "indices and commodity futures. Windows: 21d (1M), 63d (3M), 126d (6M), 252d (1Y). "
            "A 300-day burn-in period is loaded beyond the selected date range to ensure all "
            "windows have sufficient history at the period boundary."
        )
        _formula(
            "ρ(A,B,t,w) = Cov(r_A[t-w:t], r_B[t-w:t])\n"
            "             / [σ(r_A) · σ(r_B)]",
            "r = log(P_t / P_{t-1}). Window w ∈ {21, 42, 63, 126, 252}."
        )

        _h3("DCC-GARCH")
        _prose(
            "Dynamic Conditional Correlation (Engle, 2002) captures time-varying correlations "
            "that spike during stress events — a signature of genuine spillover rather than "
            "stable co-movement. The implementation uses a simplified two-step DCC: "
            "(1) fit univariate GARCH(1,1) to each series; (2) apply DCC to the standardised residuals."
        )
        _formula(
            "Q_t = (1-a-b)·Q̄ + a·(ε_{t-1}·ε_{t-1}ᵀ) + b·Q_{t-1}\n"
            "R_t = diag(Q_t)^{-1/2} · Q_t · diag(Q_t)^{-1/2}",
            "Q̄ = unconditional covariance of standardised residuals. "
            "Typical parameters: a=0.05, b=0.92."
        )

    with col_b:
        _h3("Regime Detection")
        _prose(
            "Four-state correlation regime assigned by thresholding the 60-day average "
            "cross-asset correlation, calibrated to its historical distribution. "
            "Thresholds are set at the 25th, 50th, and 75th percentiles of the full history."
        )
        for name, col, threshold, interp in [
            ("0 · Decorrelated", "#2e7d32", "avg_corr < P25",  "Diversification intact; equity and commodity shocks not co-moving"),
            ("1 · Normal",       _DIM,      "P25 ≤ avg_corr < P50", "Baseline; mild co-movement consistent with macro linkages"),
            ("2 · Elevated",     "#e67e22", "P50 ≤ avg_corr < P75", "Stress building; correlation rising — monitor for regime shift"),
            ("3 · Crisis",       "#c0392b", "avg_corr ≥ P75",  "Full spillover regime; diversification fails; geopolitical shock active"),
        ]:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin:.2rem 0">'
                f'<span style="{_M}font-size:7px;font-weight:700;color:{col};width:90px">{name}</span>'
                f'<span style="{_M}font-size:8px;color:{_MUT}">{threshold}</span>'
                f'</div>'
                f'<div style="margin-left:98px;margin-top:-2px;margin-bottom:4px">'
                f'<span style="{_S}font-size:0.65rem;color:{_DIM}">{interp}</span></div>',
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # 8. SPILLOVER NETWORK
    # ══════════════════════════════════════════════════════════════════════════
    _h2("8 · Spillover Network")
    _prose(
        "Three complementary spillover measures quantify the direction and magnitude of "
        "cross-asset shock transmission. Each captures a different aspect: Granger tests "
        "linear predictability, Transfer Entropy captures nonlinear information flow, and "
        "Diebold-Yilmaz provides a portfolio-level variance decomposition."
    )

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        _h3("Granger Causality")
        _prose("Tests whether lagged values of series X improve the forecast of Y beyond Y's own lags.")
        _formula(
            "H₀: X does not Granger-cause Y\n"
            "F-test on VAR(p), p ∈ {1, …, 5}\n"
            "Reported: F-stat, p-value, optimal lag",
            "Reject H₀ at p < 0.05 → X leads Y."
        )
        _assumption("Assumes linear dependence; misses threshold or regime-switching effects.")
    with col_s2:
        _h3("Transfer Entropy")
        _prose("Nonlinear, model-free measure of directed information flow from X to Y.")
        _formula(
            "TE(X→Y) = Σ p(y_{t+1}, y_t, x_t)\n"
            "          · log[ p(y_{t+1}|y_t, x_t)\n"
            "               / p(y_{t+1}|y_t) ]",
            "Estimated with kernel density on binned log-returns. "
            "Net TE = TE(X→Y) − TE(Y→X)."
        )
        _assumption("Sensitive to bin-width choice; computationally intensive for many pairs.")
    with col_s3:
        _h3("Diebold-Yilmaz (2012)")
        _prose("Forecast error variance decomposition of a VAR — what fraction of asset i's forecast variance is explained by shocks to asset j?")
        _formula(
            "θᵢⱼ(H) = [Σ_{h=0}^{H-1} (eᵢᵀAₕΣeⱼ)²]\n"
            "          / [Σ_{h=0}^{H-1} (eᵢᵀAₕΣAₕᵀeᵢ)]",
            "H = forecast horizon (default 10d). "
            "Total spillover index = (Σᵢ≠ⱼ θᵢⱼ) / N × 100."
        )
        _assumption("VAR lag length selected by BIC; assumes covariance stationarity.")

    # ══════════════════════════════════════════════════════════════════════════
    # 9. MARKOV REGIME CHAIN
    # ══════════════════════════════════════════════════════════════════════════
    _h2("9 · Markov Regime Chain")
    _prose(
        "The regime series (Decorrelated / Normal / Elevated / Crisis) is modelled as a "
        "first-order Markov chain. The empirical transition matrix is estimated from the "
        "full historical regime sequence. This allows computation of steady-state probabilities, "
        "mean first passage times, and forward regime forecasts."
    )

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        _h3("Transition Matrix")
        _formula(
            "P[i,j] = count(regime_t=i, regime_{t+1}=j)\n"
            "          / count(regime_t=i)",
            "Estimated from full historical regime sequence. "
            "Row-stochastic: each row sums to 1."
        )
        _h3("Steady-State Distribution")
        _formula(
            "π = π · P   subject to Σπᵢ = 1\n"
            "Solved as left eigenvector of P for eigenvalue 1.",
            "π gives the long-run fraction of time in each regime."
        )
    with col_m2:
        _h3("Mean First Passage Time")
        _formula(
            "m[i,j] = 1 + Σ_{k≠j} P[i,k] · m[k,j]\n"
            "Solved as linear system (fundamental matrix method).",
            "m[i,j] = expected trading days to first reach regime j from regime i."
        )
        _h3("Run Statistics")
        _prose(
            "Empirical distribution of consecutive days spent in each regime. "
            "Reported: mean run length, 75th percentile, longest observed run. "
            "Used to calibrate 'how long does a Crisis regime typically last?'"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 10. PORTFOLIO EXPOSURE ENGINE
    # ══════════════════════════════════════════════════════════════════════════
    _h2("10 · Portfolio Exposure Engine")
    _prose(
        "The exposure engine scores each asset against every active conflict across three "
        "dimensions: structural exposure (hard-coded analyst assessment), transmission-adjusted "
        "exposure (TPS-weighted), and a scenario-adjusted score. Assets are also ranked by "
        "conflict beta and hedge potential."
    )

    _h3("Structural Exposure Score (SES)")
    _formula(
        "SES(asset, conflict) = structural[asset][conflict]  ∈ [0, 1]",
        "Manually coded per-asset, per-conflict exposure weight in config.SECURITY_EXPOSURE. "
        "Reflects supply-chain, revenue, or regulatory linkage."
    )
    _h3("Total Asset Exposure (TAE)")
    _formula(
        "TAE(asset) = Σ_conflicts  SES(asset, c) × CIS(c) / 100",
        "CIS-weighted sum of structural exposures. Conflicts with higher CIS contribute more."
    )
    _h3("Scenario-Adjusted Score (SAS)")
    _formula(
        "SAS(asset) = clip( TAE(asset) × geo_mult × 100, 0, 100 )",
        "geo_mult from active scenario (default 1.0). Scales up in stressed scenarios."
    )
    _h3("Conflict Beta")
    _formula(
        "β(asset, conflict) = structural[asset][conflict] × TPS(conflict) / 100",
        "Measures how much the asset moves per unit of TPS from a given conflict."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 11. SCENARIO ENGINE
    # ══════════════════════════════════════════════════════════════════════════
    _h2("11 · Scenario Engine")
    _prose(
        "The Scenario Engine applies a named stress lens to the entire dashboard. "
        "When a scenario is active, it modifies the composite risk score, asset exposure "
        "scores, and trade signal generation without altering the underlying data. "
        "This enables simultaneous comparison of multiple scenario assumptions."
    )

    _h3("Scenario Parameters")
    param_rows = [
        ("geo_mult",    "Geopolitical Multiplier", "Scales the composite GRS. geo_mult > 1 amplifies; < 1 dampens. Default: 1.0."),
        ("vol_mult",    "Volatility Multiplier",   "Applied to modelled drawdown in stress simulation. > 1 = fat-tailed stress."),
        ("safe_haven",  "Safe-Haven Overlay",      "Boolean. When True, adds positive bias to Gold, Treasuries, CHF, JPY in trade signals."),
        ("short_bias",  "Short-Bias Overlay",      "Boolean. When True, penalises long equity signals and surfaces short/hedge ideas."),
        ("desc",        "Narrative Description",   "Human-readable scenario summary — used in stress test event windows."),
    ]
    for key, name, note in param_rows:
        st.markdown(
            f'<div style="display:flex;gap:10px;margin:.15rem 0;align-items:baseline">'
            f'<code style="{_M}font-size:8px;color:{_G};background:#0a0a0a;'
            f'padding:1px 6px;border:1px solid #2a2a2a;flex-shrink:0">{key}</code>'
            f'<span style="{_S}font-size:0.70rem;color:#e8e9ed;width:150px;flex-shrink:0">{name}</span>'
            f'<span style="{_S}font-size:0.68rem;color:{_DIM}">{note}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    _h3("Stress Test Event Methodology")
    _prose(
        "For each catalogued geopolitical event, the stress tester extracts a 30-day pre-event "
        "window, the event duration, and a 60-day post-event window. Portfolio returns are "
        "computed as dollar-weighted log-returns of the selected assets. Max drawdown and "
        "Sharpe ratio are reported for the event duration window."
    )
    _formula(
        "Portfolio_t = Σᵢ  wᵢ · (Pᵢ_t / Pᵢ_t₀)   (normalised to base 100)\n"
        "Max Drawdown = min_t [ (Portfolio_t − max_{s≤t} Portfolio_s) / max_{s≤t} Portfolio_s ]\n"
        "Event Sharpe = μ_daily / σ_daily × √252",
        "Weights wᵢ normalised to sum to 1. Single-period returns used (no compounding correction)."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 12. ASSUMPTIONS & LIMITATIONS
    # ══════════════════════════════════════════════════════════════════════════
    _h2("12 · Assumptions, Limitations & Ongoing Maintenance")

    col_a2, col_l2 = st.columns(2)
    with col_a2:
        _h3("Key Assumptions")
        for a in [
            "Conflict intensity dimensions are observable and monotonically related to market risk.",
            "Transmission channel weights are stable across conflict types (not conflict-specific).",
            "Four-state Markov chain adequately captures the regime space — higher granularity not supported by sample size.",
            "Rolling Pearson correlation is a sufficient proxy for time-varying dependence at daily frequency.",
            "Yahoo Finance daily close prices are unbiased proxies for commodity spot prices (uses futures front-month).",
            "DCC-GARCH parameters (a=0.05, b=0.92) are calibrated globally — not conflict-specific.",
            "SECURITY_EXPOSURE structural weights are point-in-time estimates and subject to drift.",
            "RSS feed headlines are a representative sample of geopolitical events — coverage gaps exist.",
        ]:
            _assumption(a)
    with col_l2:
        _h3("Known Limitations")
        for l in [
            "CIS and TPS dimensions are manually scored — subject to analyst judgment and update lag.",
            "The model uses public equity indices and commodity futures as proxies; direct exposure data (bank loan books, supply contracts) is not available.",
            "Granger and DY tests assume linear VAR dynamics — nonlinear regime-dependent relationships are not fully captured.",
            "News GPR layer uses keyword matching, not semantic NLP — headline nuance (e.g. 'no attack') may be misclassified.",
            "Conflict registry requires manual maintenance; stale data degrades CIS/TPS accuracy.",
            "Scenario Engine multipliers are applied uniformly — no differentiation by asset class or geography.",
            "COT positioning data has a 3-day publication lag — contrarian signal reflects week-old positioning.",
            "Transfer entropy estimation is sensitive to bin-width and sample size — small samples produce noisy estimates.",
        ]:
            _limitation(l)

    _h3("Model Maintenance Protocol")
    _prose(
        "Per SR 11-7 guidance, the following ongoing monitoring actions are recommended: "
        "<br>• <b>Weekly:</b> Review conflict registry for state changes (active → latent) and update last_updated fields. "
        "<br>• <b>Monthly:</b> Recalibrate regime thresholds as new data accumulates. Validate DCC-GARCH stationarity. "
        "<br>• <b>Quarterly:</b> Re-examine CIS dimension weights against realized commodity price responses to recent conflicts. "
        "<br>• <b>Annually:</b> Full model review — assess whether the 4-regime Markov structure still holds; consider re-estimation of TPS channel weights against updated empirical evidence."
    )

    _takeaway_block(
        "This dashboard implements a three-layer geopolitical risk framework grounded in "
        "academic spillover literature (Diebold-Yilmaz 2009/2012, Engle DCC 2002) and "
        "practitioner scenario design conventions (SR 11-7, CCAR). All formulae, weights, "
        "and assumptions are fully transparent and documented above. The model is designed "
        "to be challenged, stress-tested, and improved — contributions to the conflict registry "
        "and channel weight calibration are the highest-leverage points of intervention."
    )

    _page_footer()
