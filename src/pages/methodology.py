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
        "The MCS is the market-observable layer of the risk score. It uses six signals extracted "
        "from live price data to confirm or contradict the structural CIS/TPS signal. If CIS and TPS "
        "are high but markets are calm, the MCS dampens the overall score. If markets are already "
        "showing stress, MCS amplifies it. All six sub-signals are orthogonalized to reduce "
        "multicollinearity."
    )

    _h3("Sub-signal Weights")
    _weight_table([
        ("safe_haven",  "Safe-Haven Bid",             0.22, "Gold/USD upside relative to equity vol — clearest geopolitical signal in prices."),
        ("oil_gold",    "Oil-Gold Signal",             0.18, "Simultaneous oil and gold spikes: joint move = geopolitical, not purely cyclical."),
        ("eq_vol",      "Equity Vol (orthog.)",        0.15, "VIX proxy — residualized to remove cyclical component. Stress indicator."),
        ("rates_vol",   "Rates Vol (TLT proxy)",       0.15, "Treasury price volatility. Flight-to-safety during geopolitical shocks raises bond prices."),
        ("cmd_vol",     "Commodity Vol (residual)",    0.15, "Commodity return volatility after removing equity component. Pure commodity stress."),
        ("spillover",   "Cross-Asset Spillover",       0.15, "Percentile of current 60d avg equity-commodity correlation in its historical distribution."),
    ])

    _formula(
        "MCS = 0.22·SafeHaven + 0.18·OilGold + 0.15·EqVol\n"
        "    + 0.15·RatesVol + 0.15·CmdVol + 0.15·Spillover",
        "All sub-signals normalised to [0, 100] before weighting."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 5. COMPOSITE RISK SCORE
    # ══════════════════════════════════════════════════════════════════════════
    _h2("5 · Composite Geopolitical Risk Score")
    _prose(
        "The composite score assembles the three layers into a single 0–100 index. "
        "CIS and TPS carry the largest weights because they are structurally grounded in "
        "analyst assessment of active conflicts. MCS serves as a real-time market confirmation "
        "signal. A scenario geo-multiplier is applied last, allowing stress scenario analysis "
        "to scale the score without altering the underlying layers."
    )

    _formula(
        "GRS = clip( [0.40 × CIS_portfolio + 0.35 × TPS_portfolio + 0.25 × MCS] × geo_mult , 0, 100)",
        "CIS_portfolio and TPS_portfolio are CIS-intensity-weighted averages across all active conflicts."
    )

    col_l, col_r = st.columns([1, 1])
    with col_l:
        _h3("Score Bands")
        for band, lo, hi, col, interp in [
            ("CRITICAL", 75, 100, "#c0392b", "Multiple active conflicts + market stress confirmed"),
            ("ELEVATED", 50,  75, "#e67e22", "Active conflict with significant transmission pressure"),
            ("MODERATE", 25,  50, _G,        "Latent conflict risk or mild market dislocations"),
            ("LOW",       0,  25, "#27ae60", "No active conflicts; market signals calm"),
        ]:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin:.2rem 0">'
                f'<span style="{_M}font-size:7px;font-weight:700;color:{col};width:65px">{band}</span>'
                f'<span style="{_M}font-size:8px;color:{_DIM}">{lo}–{hi}</span>'
                f'<span style="{_S}font-size:0.65rem;color:{_MUT};margin-left:4px">{interp}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    with col_r:
        _h3("Layer Rationale")
        _prose(
            "<b style='color:#CFB991'>40% CIS</b> — Structural conflict intensity is the primary "
            "driver. Markets respond to actual conflict severity first.<br><br>"
            "<b style='color:#CFB991'>35% TPS</b> — Transmission channel specificity ensures "
            "commodity markets are connected to the right conflicts.<br><br>"
            "<b style='color:#CFB991'>25% MCS</b> — Market confirmation prevents false signals: "
            "a high CIS with calm markets suggests pricing-in has already occurred."
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
