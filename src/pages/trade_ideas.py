"""
Page 6 - Trade Ideas
Regime-triggered cross-asset trade cards, rationale, entry/exit levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date

from src.data.loader import load_returns, load_commodity_prices
from src.data.config import GEOPOLITICAL_EVENTS, PALETTE
from src.analysis.correlations import (
    average_cross_corr_series, detect_correlation_regime, rolling_correlation,
    composite_stress_index,
)
from src.ui.shared import (
    _style_fig, _chart, _page_intro, _thread, _section_note,
    _definition_block, _takeaway_block, _page_conclusion, _page_footer,
    _insight_note,
)

_REGIME_NAMES  = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis"}
_REGIME_COLORS = {0: "#2e7d32",      1: "#555960", 2: "#e67e22",  3: "#c0392b"}


# ── Trade idea library ─────────────────────────────────────────────────────
_TRADE_LIBRARY = [
    {
        "regime":    [2, 3],
        "trigger":   "Elevated/Crisis correlation",
        "name":      "Long Gold / Short Eurostoxx 50",
        "rationale": "When cross-asset correlation spikes, equities and commodities sell off together. "
                     "Gold decouples as safe-haven demand absorbs panic flows. "
                     "Short European equities amplifies energy-cost transmission risk.",
        "entry":     "Enter when 60d avg cross-asset corr > 0.45 and DCC(Gold/SPX) < −0.1",
        "exit":      "Close when correlation regime drops back to Normal for 5+ days",
        "risk":      "Correlation snap-back; Central bank intervention can flip gold",
        "assets":    ["Gold", "Eurostoxx 50"],
        "direction": ["Long", "Short"],
        "category":  "Crisis Hedge",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Energy supply shock",
        "name":      "Long Natural Gas / Short Nikkei 225",
        "rationale": "Japan is the world's largest LNG importer. "
                     "Natural gas supply shocks directly impair Japanese manufacturing margins "
                     "and current account. Energy spikes translate to yen weakness and equity underperformance.",
        "entry":     "Enter on Ukraine escalation OR Strait of Hormuz closure signal + NG vol spike",
        "exit":      "Peace signal, supply restoration, or NG price mean-reversion (−20% from entry)",
        "risk":      "BOJ FX intervention; domestic LNG stockpile release",
        "assets":    ["Natural Gas", "Nikkei 225"],
        "direction": ["Long", "Short"],
        "category":  "Geopolitical",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Wheat/food supply disruption (Ukraine War)",
        "name":      "Long Wheat / Long Gold / Short Emerging Markets",
        "rationale": "Food price spikes trigger inflation in EM countries with high cereal import ratios. "
                     "Combined with USD strength (gold hedge), EM equities face dual pressure from "
                     "import inflation and capital outflows.",
        "entry":     "Wheat 30d return > +15% + political instability signals in MENA/SSA",
        "exit":      "Wheat normalises to 12M average; EM carry recovery",
        "risk":      "IMF/World Bank intervention; US export restrictions",
        "assets":    ["Wheat", "Gold", "Sensex"],
        "direction": ["Long", "Long", "Short"],
        "category":  "Macro",
    },
    {
        "regime":    [0, 1],
        "trigger":   "Global growth recovery (low correlation regime)",
        "name":      "Long Copper / Long S&P 500",
        "rationale": "Copper is the premier global growth bellwether. "
                     "When correlation is low, commodities and equities price "
                     "independent fundamentals - copper rising with equities signals "
                     "genuine demand expansion, not just liquidity.",
        "entry":     "Copper 60d momentum > 0, ISM Manufacturing > 50, cross-asset corr < 0.20",
        "exit":      "Copper momentum reversal; ISM contraction; Fed pause signals",
        "risk":      "China property market collapse; USD spike from geopolitical safe-haven",
        "assets":    ["Copper", "S&P 500"],
        "direction": ["Long", "Long"],
        "category":  "Growth",
    },
    {
        "regime":    [1, 2],
        "trigger":   "Oil-equity divergence (supply shock premium)",
        "name":      "Long WTI Crude / Short S&P 500 Energy-Heavy Sectors",
        "rationale": "When oil spikes from a supply shock (not demand), "
                     "energy futures gain while broader equities face margin compression. "
                     "The long/short captures the spread between commodity producer "
                     "and equity consumer dynamics.",
        "entry":     "Brent-WTI spread widens + OPEC+ surprise cut + SPX P/E compression",
        "exit":      "Supply restoration event; recession pricing dominates oil",
        "risk":      "Demand destruction flips correlation; tech-led equity rally decouples",
        "assets":    ["WTI Crude Oil", "S&P 500"],
        "direction": ["Long", "Short"],
        "category":  "Macro",
    },
    {
        "regime":    [3],
        "trigger":   "Full crisis - all correlations elevated",
        "name":      "Long Gold, Long Silver / Short Copper, Short Shanghai Comp",
        "rationale": "Full crisis regime: precious metals outperform as industrial metals "
                     "and EM equities (particularly China) collapse under dollar strength "
                     "and risk-off flows. Gold/Silver spread also captures "
                     "the industrial-vs-monetary metals divergence.",
        "entry":     "Crisis regime active > 3 days; VIX > 35; DXY trending up",
        "exit":      "Regime drops below Elevated; Fed emergency action; VIX < 25",
        "risk":      "Chinese stimulus; commodity demand front-running recovery",
        "assets":    ["Gold", "Silver", "Copper", "Shanghai Comp"],
        "direction": ["Long", "Long", "Short", "Short"],
        "category":  "Crisis Hedge",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Private credit stress / HY spread widening + BDC underperformance",
        "name":      "Short BDC Basket / Long HY Credit Protection",
        "rationale": "Private credit ($2T+ AUM) is illiquid and marked-to-model quarterly. "
                     "When HY OAS widens >35bps in 90 days and BKLN underperforms SPY, "
                     "BDC equity (ARCC, OBDC, FSK) reprices before private marks surface - "
                     "a lagged NAV writedown is typically confirmed 1–2 quarters later. "
                     "The CDX HY 5Y index provides liquid short exposure to the same credit universe. "
                     "Gold long captures contagion into safe-haven flows as financial sector sells off.",
        "entry":     "HY OAS >350bps and rising >30bps/month; BKLN below 200d MA; "
                     "BDC basket -5% vs SPY on 60d basis; elevated/crisis equity-commodity regime",
        "exit":      "HY OAS stabilises <300bps; Fed emergency rate cut signal; "
                     "BDC premiums re-compress; credit facilities extended at par",
        "risk":      "Fed emergency cuts collapse floating-rate costs rapidly; "
                     "sponsor rescue financing delays visible defaults; "
                     "interval fund gates mask redemption pressure for 12–18 months",
        "assets":    ["Ares Capital (ARCC)", "Blue Owl (OBDC)", "Gold"],
        "direction": ["Short", "Short", "Long"],
        "category":  "Private Credit",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Flight to quality - elevated/crisis correlation regime",
        "name":      "Long TLT / Short HYG (Flight to Quality)",
        "rationale": "In elevated and crisis correlation regimes, investors rotate from credit risk to duration safety. "
                     "TLT captures the safe-haven Treasury bid while HYG shorts the credit spread widening. "
                     "The trade isolates the quality spread compression that accompanies every risk-off episode.",
        "entry":     "Crisis/Elevated regime active; HY OAS rising >30bps in 30 days; VIX >25; TLT above 200d MA",
        "exit":      "Regime drops to Normal; HY OAS stabilises; Fed pivot signal",
        "risk":      "Bear steepening (long rates rise with HY spreads simultaneously); fiscal dominance narrative",
        "assets":    ["US 20Y+ Treasury (TLT)", "HY Corporate (HYG)"],
        "direction": ["Long", "Short"],
        "category":  "Fixed Income",
    },
    {
        "regime":    [1, 2],
        "trigger":   "Inflation breakeven expansion / stagflation risk",
        "name":      "Long TIP / Short TLT (Inflation Breakeven Trade)",
        "rationale": "When commodity prices spike (energy, food) and the Fed is behind the curve, "
                     "real yields compress while nominal yields stay elevated. "
                     "TIPS outperform nominal Treasuries as breakeven inflation widens. "
                     "This trade directly monetises the commodity-to-bond inflation transmission channel.",
        "entry":     "5Y breakeven inflation rising >20bps in 60 days; WTI or wheat up >15% in 30 days; CPI surprise positive",
        "exit":      "Breakeven inflation peaks; commodity prices mean-revert; Fed delivers credible inflation response",
        "risk":      "Demand destruction flips inflation to deflation; recession pricing overrides inflation premium",
        "assets":    ["TIPS / Inflation (TIP)", "US 20Y+ Treasury (TLT)"],
        "direction": ["Long", "Short"],
        "category":  "Fixed Income",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Oil spike + INR depreciation (India crude import stress)",
        "name":      "Long Brent Crude / Short Nifty 50 (India Import Shock)",
        "rationale": "India imports ~85% of its crude oil needs (~5 mb/d). "
                     "When Brent spikes >15% in 60 days and USD/INR is depreciating simultaneously, "
                     "India's current account deficit widens sharply - historically correlating with "
                     "Nifty 50 underperformance of -6% to -12% vs global peers. "
                     "The long Brent / short Nifty trade monetises the commodity-to-EM-equity transmission channel specific to India.",
        "entry":     "Brent up >15% in 60 days AND USD/INR rising >3% in 30 days; elevated/crisis regime active",
        "exit":      "Oil supply restored; INR stabilises / RBI intervention; India CAD narrows",
        "risk":      "RBI forex reserve intervention caps INR weakness; India domestic demand surprises to upside; OPEC+ production cut reversal",
        "assets":    ["Brent Crude", "Nifty 50"],
        "direction": ["Long", "Short"],
        "category":  "India/EM",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Geopolitical stress + dollar strength - India gold import sensitivity",
        "name":      "Long Gold / Short INR (India Geopolitical Hedge)",
        "rationale": "India is the world's #2 gold consumer (~800-900 tonnes/year). "
                     "During geopolitical stress (Middle East escalation, South Asia tensions), "
                     "gold demand surges from Indian households as a safe-haven AND currency hedge. "
                     "Simultaneously, USD/INR typically rises (INR weakens) under global risk-off conditions. "
                     "The long gold / short INR (via USD/INR long) trade captures both legs of this transmission.",
        "entry":     "Crisis/Elevated regime; VIX >25; USD/INR rising >2% in 20 days; Gold 20d momentum positive",
        "exit":      "Geopolitical de-escalation; gold mean-reverts -8% from peak; INR stabilises on RBI action",
        "risk":      "RBI aggressively defends INR using $620B forex reserves; Gold sell-off on Fed hawkish surprise",
        "assets":    ["Gold", "USD/INR"],
        "direction": ["Long", "Long"],
        "category":  "India/EM",
    },
    {
        "regime":    [0, 1],
        "trigger":   "Dollar weakness cycle / EM credit rally",
        "name":      "Long EMB / Short DXY (Dollar Debasement - EM Relief)",
        "rationale": "A weakening dollar reduces dollar-denominated debt service costs for EM sovereigns, "
                     "compresses EM credit spreads, and attracts capital inflows into EM assets. "
                     "EMB captures the bond price appreciation; short DXY amplifies the currency leg. "
                     "Gold is the commodity expression of the same dollar debasement theme.",
        "entry":     "DXY down >3% in 60 days; Fed on hold or cutting; EM current accounts improving; commodity prices rising",
        "exit":      "Dollar reversal; Fed hawkish pivot; EM-specific credit event; commodity demand collapse",
        "risk":      "EM-specific credit events (sovereign default, political crisis); commodity demand collapse flips EM outlook",
        "assets":    ["EM USD Bonds (EMB)", "DXY (Dollar Index)", "Gold"],
        "direction": ["Long", "Short", "Long"],
        "category":  "Fixed Income",
    },
]

_CATEGORY_COLORS = {
    "Crisis Hedge":   "#c0392b",
    "Geopolitical":   "#e67e22",
    "Macro":          "#2980b9",
    "Growth":         "#2e7d32",
    "Private Credit": "#8e44ad",
    "Fixed Income":   "#2980b9",
    "India/EM":       "#16a085",
}


def page_trade_ideas(start: str, end: str, fred_key: str = "") -> None:
    st.markdown(
        '<h1 style="font-family:\'DM Sans\',sans-serif;font-size:1.25rem;'
        'font-weight:700;margin-bottom:0.3rem">Trade Ideas</h1>',
        unsafe_allow_html=True,
    )
    _page_intro(
        "Spillover analysis is only useful if it generates actionable positioning. "
        "<strong>Each idea here is a direct translation of a spillover or correlation regime signal into a trade.</strong> "
        "When Granger tests show oil leading equity returns, there is a pairs trade. When the correlation "
        "regime flips from Elevated to Decorrelated, there is a diversification opportunity. "
        "When the macro cycle enters late-stage and commodity-equity co-movement breaks down, "
        "there are defensive rotations. Ideas only show as active when the current regime matches "
        "their structural trigger - dead cards mean the regime has not yet validated that setup."
    )

    with st.spinner("Loading data…"):
        eq_r, cmd_r = load_returns(start, end)

    if eq_r.empty or cmd_r.empty:
        st.error("Market data unavailable.")
        return

    # ── Current regime ─────────────────────────────────────────────────────
    avg_corr = average_cross_corr_series(eq_r, cmd_r, window=60)
    regimes  = detect_correlation_regime(avg_corr)
    current  = int(regimes.iloc[-1]) if not regimes.empty else 1
    r_name   = _REGIME_NAMES[current]
    r_color  = _REGIME_COLORS[current]

    # ── Signal summary KPI strip ────────────────────────────────────────────
    _all_triggered = [t for t in _TRADE_LIBRARY if current in t["regime"]]
    _n_bull = sum(1 for t in _all_triggered if "Long" in t["direction"] and t["direction"][0] == "Long")
    _n_bear = sum(1 for t in _all_triggered if "Short" in t["direction"] and t["direction"][0] == "Short")
    _cats   = list(dict.fromkeys(t["category"] for t in _all_triggered))

    _F_ti = "font-family:'DM Sans',sans-serif;"
    def _ti_kpi(col, label, value, value_color="#000"):
        col.markdown(
            f'<div style="border:1px solid #E8E5E0;border-radius:4px;padding:0.6rem 0.85rem;background:#fafaf8">'
            f'<div style="{_F_ti}font-size:0.55rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.14em;color:#8E6F3E;margin-bottom:3px">{label}</div>'
            f'<div style="{_F_ti}font-size:1.0rem;font-weight:700;color:{value_color}">{value}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    _sk1, _sk2, _sk3, _sk4, _sk5 = st.columns(5)
    _sk1.markdown(
        f'<div style="border:1px solid {r_color};border-left:4px solid {r_color};border-radius:4px;'
        f'padding:0.6rem 0.85rem;background:#fafaf8">'
        f'<div style="{_F_ti}font-size:0.55rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.14em;color:#8E6F3E;margin-bottom:3px">Regime</div>'
        f'<div style="{_F_ti}font-size:1.0rem;font-weight:700;color:{r_color}">{r_name}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    _ti_kpi(_sk2, "Avg Corr (60d)", f"{avg_corr.iloc[-1]:.3f}")
    _ti_kpi(_sk3, "Ideas Triggered", str(len(_all_triggered)), "#8E6F3E")
    _ti_kpi(_sk4, "Long-First Ideas", str(_n_bull), "#2e7d32")
    _ti_kpi(_sk5, "Short-First Ideas", str(_n_bear), "#c0392b")

    # ── Filter by regime / category ────────────────────────────────────────
    c1, c2 = st.columns(2)
    show_all = c1.checkbox("Show all regimes", value=False)
    cat_filter = c2.selectbox(
        "Filter by category",
        ["All", "Crisis Hedge", "Geopolitical", "Macro", "Growth", "India/EM", "Private Credit", "Fixed Income"],
    )

    active_trades = [
        t for t in _TRADE_LIBRARY
        if (show_all or current in t["regime"])
        and (cat_filter == "All" or t["category"] == cat_filter)
    ]

    if not active_trades:
        st.info(f"No trade ideas for {r_name} regime. Enable 'Show all regimes' to see all.")
    else:
        st.markdown(
            f'<p style="font-family:\'DM Sans\',sans-serif;font-size:0.70rem;color:#8890a1;'
            f'margin-bottom:0.6rem">'
            f'<b>{len(active_trades)}</b> idea{"s" if len(active_trades)>1 else ""} '
            f'{"shown" if show_all else "triggered"} for <b style=\'color:{r_color}\'>{r_name}</b></p>',
            unsafe_allow_html=True,
        )

    all_r_concat = pd.concat([eq_r, cmd_r], axis=1)

    _thread(
        "With the current regime confirmed above, the ideas below have been filtered to match the "
        "prevailing market structure. Each card shows the directional thesis, entry trigger, target, "
        "stop, and the historical correlation data that supports it."
    )

    # ── 2-column card grid ──────────────────────────────────────────────────
    for row_start in range(0, len(active_trades), 2):
        pair = active_trades[row_start:row_start + 2]
        card_cols = st.columns(len(pair), gap="medium")

        for col, trade in zip(card_cols, pair):
            with col:
                cat_col = _CATEGORY_COLORS.get(trade["category"], "#CFB991")
                dir_html = " &nbsp;|&nbsp; ".join(
                    f'<span style="color:{"#2e7d32" if d=="Long" else "#c0392b"};font-weight:700">{d}</span> {a}'
                    for a, d in zip(trade["assets"], trade["direction"])
                )
                regime_pills = " ".join(
                    f'<span style="background:{_REGIME_COLORS[r]};color:#fff;padding:1px 5px;'
                    f'border-radius:2px;font-size:0.56rem">{_REGIME_NAMES[r]}</span>'
                    for r in trade["regime"]
                )
                st.markdown(
                    f'<div style="border:1px solid #E8E5E0;border-radius:5px;overflow:hidden;margin-bottom:0.5rem">'
                    f'<div style="background:{cat_col};padding:0.4rem 0.85rem;'
                    f'display:flex;justify-content:space-between;align-items:center">'
                    f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.57rem;font-weight:700;'
                    f'letter-spacing:0.10em;text-transform:uppercase;color:#fff">'
                    f'{trade["category"]} · {trade["trigger"]}</div>'
                    f'<div>{regime_pills}</div></div>'
                    f'<div style="padding:0.75rem 0.85rem;background:#fff">'
                    f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.84rem;font-weight:700;'
                    f'color:#000;margin-bottom:3px">{trade["name"]}</div>'
                    f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.70rem;'
                    f'color:#333;margin-bottom:6px">{dir_html}</div>'
                    f'<p style="font-family:\'DM Sans\',sans-serif;font-size:0.70rem;color:#2A2A2A;'
                    f'line-height:1.65;margin-bottom:8px">{trade["rationale"]}</p>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;font-size:0.70rem">'
                    f'<div style="background:#f9f8f6;padding:0.35rem 0.5rem;border-radius:3px">'
                    f'<div style="font-weight:700;color:#8E6F3E;font-size:0.56rem;text-transform:uppercase;'
                    f'letter-spacing:0.10em;margin-bottom:2px">Entry</div>{trade["entry"]}</div>'
                    f'<div style="background:#f9f8f6;padding:0.35rem 0.5rem;border-radius:3px">'
                    f'<div style="font-weight:700;color:#333;font-size:0.56rem;text-transform:uppercase;'
                    f'letter-spacing:0.10em;margin-bottom:2px">Exit</div>{trade["exit"]}</div>'
                    f'<div style="background:#fff0f0;padding:0.35rem 0.5rem;border-radius:3px">'
                    f'<div style="font-weight:700;color:#c0392b;font-size:0.56rem;text-transform:uppercase;'
                    f'letter-spacing:0.10em;margin-bottom:2px">Risks</div>{trade["risk"]}</div>'
                    f'</div></div></div>',
                    unsafe_allow_html=True,
                )

                # Mini correlation chart inside the card column
                if len(trade["assets"]) >= 2:
                    a1, a2 = trade["assets"][0], trade["assets"][1]
                    if a1 in all_r_concat.columns and a2 in all_r_concat.columns:
                        rc = rolling_correlation(all_r_concat[a1], all_r_concat[a2], 60)
                        fig_mini = go.Figure()
                        r, g, b = int(cat_col[1:3],16), int(cat_col[3:5],16), int(cat_col[5:7],16)
                        fig_mini.add_trace(go.Scatter(
                            x=rc.index, y=rc.values,
                            name=f"{a1}/{a2}",
                            line=dict(color=cat_col, width=1.4),
                            fill="tozeroy",
                            fillcolor=f"rgba({r},{g},{b},0.07)",
                        ))
                        fig_mini.add_hline(y=0, line=dict(color="#ABABAB", width=1, dash="dot"))
                        for ev in GEOPOLITICAL_EVENTS:
                            fig_mini.add_vrect(x0=str(ev["start"]), x1=str(ev["end"]),
                                fillcolor=ev["color"], opacity=0.04, layer="below", line_width=0)
                        fig_mini.update_layout(
                            template="purdue", height=180,
                            title=dict(text=f"60d Corr: {a1} / {a2}", font=dict(size=10)),
                            showlegend=False,
                            margin=dict(l=36, r=12, t=28, b=24),
                            xaxis=dict(rangeslider=dict(visible=False)),
                        )
                        _chart(fig_mini)
                        _insight_note(
                            "Shows the rolling 60-day correlation between the commodity driver and the equity target for this trade. "
                            "A rising positive correlation means the relationship underpinning the trade thesis is strengthening. "
                            "A declining or negative reading is a warning that the causal link may be breaking down."
                        )

    # ── Download report ─────────────────────────────────────────────────────
    st.markdown(
        '<p style="font-size:0.74rem;font-weight:700;letter-spacing:0.06em;'
        'text-transform:uppercase;color:#CFB991;margin-bottom:0.5rem">'
        'Institution Report Download</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:0.72rem;color:#8890a1;line-height:1.7;margin-bottom:0.8rem">'
        'Generate a professionally formatted A4 research report covering the current regime, '
        'all active trade ideas, geopolitical context, and methodology - suitable for '
        'academic submission or institutional review.</p>',
        unsafe_allow_html=True,
    )

    if st.button("Generate PDF Report", key="gen_report", type="primary"):
        try:
            from src.reports.report_generator import generate_report
            with st.spinner("Building report - generating charts…"):
                stress = composite_stress_index(eq_r, cmd_r, avg_corr=avg_corr)
                pdf_bytes = generate_report(
                    start=start,
                    end=end,
                    avg_corr_series=avg_corr,
                    current_regime=current,
                    regimes=regimes,
                    active_trades=active_trades,
                    all_trades=_TRADE_LIBRARY,
                    eq_r=eq_r,
                    cmd_r=cmd_r,
                    stress_series=stress,
                    geopolitical_events=GEOPOLITICAL_EVENTS,
                )
            filename = (
                f"purdue_spillover_report_"
                f"{start.replace('-','')}_to_{end.replace('-','')}.pdf"
            )
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                key="download_report",
            )
        except ImportError:
            st.error(
                "reportlab is required for PDF generation. "
                "Run: `pip install reportlab>=4.2.0`"
            )
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")

    _section_note(
        "Trade ideas are generated from historical cross-asset patterns and regime signals. "
        "All ideas are illustrative and must be validated against current market structure, "
        "liquidity, and position sizing constraints before implementation."
    )

    _page_conclusion(
        "Framework",
        "The regime-based trade generation framework matches historical spillover patterns "
        "to current correlation regimes. Crisis regimes activate hedging and divergence plays; "
        "normal regimes favour growth-correlated long positioning. "
        "Use Granger and transfer entropy results from the Spillover page to validate lead-lag direction."
    )

    # ── AI Trade Structurer + Pending Review Panel ─────────────────────────
    try:
        from src.agents.trade_structurer import run as _ts_run
        from src.ui.agent_panel import render_agent_output_block, render_pending_review
        from src.analysis.agent_state import is_enabled, get_agent

        if is_enabled("trade_structurer"):
            _anthropic_key = _openai_key = ""
            try:
                _keys = st.secrets.get("keys", {})
                _anthropic_key = _keys.get("anthropic_api_key", "") or ""
                _openai_key    = _keys.get("openai_api_key",    "") or ""
            except Exception:
                pass
            _provider = "anthropic" if _anthropic_key else ("openai" if _openai_key else None)
            _api_key  = _anthropic_key or _openai_key

            # Build context from what's been computed on this page
            _ts_ctx: dict = {
                "regime_name":   r_name,
                "regime_level":  current,
                "avg_corr":      float(avg_corr.iloc[-1]) if not avg_corr.empty else 0.0,
            }
            try:
                from src.analysis.risk_score import compute_risk_score
                from src.data.loader import load_returns as _lr3
                _eq_r3, _cmd_r3 = _lr3(start, end)
                from src.analysis.correlations import average_cross_corr_series as _acs
                _avg_c3 = _acs(_eq_r3, _cmd_r3, window=60)
                _rs = compute_risk_score(_avg_c3, _cmd_r3, _eq_r3)
                _ts_ctx["risk_score"] = float(_rs.get("score", 0))
                if len(_cmd_r3) >= 5:
                    _w5c = _cmd_r3.iloc[-5:].sum()
                    _ts_ctx["top_commodity"]     = str(_w5c.idxmax())
                    _ts_ctx["top_commodity_ret"] = float(_w5c.max()) * 100
                if len(_eq_r3) >= 5:
                    _w5e = _eq_r3.iloc[-5:].sum()
                    _ts_ctx["worst_equity"]      = str(_w5e.idxmin())
                    _ts_ctx["worst_equity_ret"]  = float(_w5e.min()) * 100
            except Exception:
                pass

            # Include peer agent outputs
            _peer_signals = {}
            for _pid in ("macro_strategist", "geopolitical_analyst"):
                _pa = get_agent(_pid)
                if _pa.get("last_output"):
                    _peer_signals[_pid] = _pa["last_output"][:120]
            if _peer_signals:
                _ts_ctx["peer_signals"] = _peer_signals

            with st.spinner("AI Trade Structurer generating idea…"):
                _ts_result = _ts_run(_ts_ctx, _provider, _api_key)

            if _ts_result.get("narrative"):
                st.markdown("---")
                render_agent_output_block("trade_structurer", _ts_result)

        # Always show the review queue on this page
        st.markdown("---")
        render_pending_review()

    except Exception:
        pass

    _page_footer()
