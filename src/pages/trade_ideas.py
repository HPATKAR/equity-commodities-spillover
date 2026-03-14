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
    _style_fig, _chart, _page_intro, _section_note,
    _definition_block, _takeaway_block, _page_conclusion, _page_footer,
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
]

_CATEGORY_COLORS = {
    "Crisis Hedge": "#c0392b",
    "Geopolitical": "#e67e22",
    "Macro":        "#2980b9",
    "Growth":       "#2e7d32",
}


def page_trade_ideas(start: str, end: str, fred_key: str = "") -> None:
    st.markdown(
        '<h1 style="font-family:\'DM Sans\',sans-serif;font-size:1.25rem;'
        'font-weight:700;margin-bottom:0.3rem">Trade Ideas</h1>',
        unsafe_allow_html=True,
    )
    _page_intro(
        "Regime-triggered cross-asset trade ideas based on the current correlation regime, "
        "geopolitical context, and historical spillover patterns. Each card includes "
        "entry trigger, exit signal, rationale, and key risks. "
        "Not investment advice - for research and education only."
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

    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:0.8rem;
        padding:0.7rem 1rem;border:1px solid {r_color};border-radius:4px;
        background:#fafaf8;margin-bottom:1rem">
        <div style="width:10px;height:10px;border-radius:50%;
        background:{r_color};flex-shrink:0"></div>
        <div>
        <span style="font-size:0.60rem;text-transform:uppercase;letter-spacing:0.12em;
        color:#666666;font-weight:600">Current Regime · </span>
        <span style="font-size:0.88rem;font-weight:700;color:{r_color}">{r_name}</span>
        <span style="font-size:0.74rem;color:#333333;margin-left:0.5rem">
        (avg cross-asset corr: {avg_corr.iloc[-1]:.3f})</span>
        </div></div>""",
        unsafe_allow_html=True,
    )

    # ── Filter by regime / category ────────────────────────────────────────
    st.markdown("---")
    c1, c2 = st.columns(2)
    show_all = c1.checkbox("Show all regimes", value=False)
    cat_filter = c2.selectbox(
        "Filter by category",
        ["All", "Crisis Hedge", "Geopolitical", "Macro", "Growth"],
    )

    active_trades = [
        t for t in _TRADE_LIBRARY
        if (show_all or current in t["regime"])
        and (cat_filter == "All" or t["category"] == cat_filter)
    ]

    if not active_trades:
        st.info(f"No trade ideas active for current regime ({r_name}). Enable 'Show all regimes' to see all ideas.")
    else:
        st.markdown(
            f'<p style="font-size:0.74rem;color:#333333;margin-bottom:0.8rem">'
            f'{len(active_trades)} idea{"s" if len(active_trades) > 1 else ""} '
            f'{"shown" if show_all else "triggered"} for <b>{r_name}</b> regime</p>',
            unsafe_allow_html=True,
        )

    for trade in active_trades:
        cat_col = _CATEGORY_COLORS.get(trade["category"], "#CFB991")
        dir_html = " | ".join(
            f'<span style="color:{"#2e7d32" if d=="Long" else "#c0392b"};'
            f'font-weight:700">{d}</span> {a}'
            for a, d in zip(trade["assets"], trade["direction"])
        )
        regime_pills = " ".join(
            f'<span style="background:{_REGIME_COLORS[r]};color:#fff;padding:1px 6px;'
            f'border-radius:2px;font-size:0.58rem">{_REGIME_NAMES[r]}</span>'
            for r in trade["regime"]
        )

        st.markdown(
            f"""<div style="border:1px solid #E8E5E0;border-radius:5px;
            overflow:hidden;margin-bottom:1rem">
            <div style="background:{cat_col};padding:0.5rem 1rem;
            display:flex;justify-content:space-between;align-items:center">
              <div style="font-size:0.60rem;font-weight:700;letter-spacing:0.10em;
              text-transform:uppercase;color:#fff">{trade['category']} · {trade['trigger']}</div>
              <div>{regime_pills}</div>
            </div>
            <div style="padding:0.9rem 1rem;background:#fff">
              <div style="font-size:0.88rem;font-weight:700;color:#000;
              margin-bottom:0.4rem">{trade['name']}</div>
              <div style="font-size:0.74rem;color:#333333;margin-bottom:0.5rem">{dir_html}</div>
              <p style="font-size:0.78rem;color:#2A2A2A;line-height:1.7;margin-bottom:0.5rem">
              {trade['rationale']}</p>
              <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.5rem;
              font-size:0.74rem">
                <div style="background:#f9f8f6;padding:0.4rem 0.6rem;border-radius:3px">
                  <div style="font-weight:600;color:#8E6F3E;font-size:0.60rem;
                  text-transform:uppercase;letter-spacing:0.10em;margin-bottom:2px">Entry</div>
                  {trade['entry']}
                </div>
                <div style="background:#f9f8f6;padding:0.4rem 0.6rem;border-radius:3px">
                  <div style="font-weight:600;color:#333333;font-size:0.60rem;
                  text-transform:uppercase;letter-spacing:0.10em;margin-bottom:2px">Exit</div>
                  {trade['exit']}
                </div>
                <div style="background:#fff0f0;padding:0.4rem 0.6rem;border-radius:3px">
                  <div style="font-weight:600;color:#c0392b;font-size:0.60rem;
                  text-transform:uppercase;letter-spacing:0.10em;margin-bottom:2px">Risks</div>
                  {trade['risk']}
                </div>
              </div>
            </div></div>""",
            unsafe_allow_html=True,
        )

        # Mini chart: rolling correlation of the first two assets
        if len(trade["assets"]) >= 2:
            all_r = pd.concat([eq_r, cmd_r], axis=1)
            a1, a2 = trade["assets"][0], trade["assets"][1]
            if a1 in all_r.columns and a2 in all_r.columns:
                rc = rolling_correlation(all_r[a1], all_r[a2], 60)
                fig_mini = go.Figure()
                fig_mini.add_trace(go.Scatter(
                    x=rc.index, y=rc.values,
                    name=f"{a1}/{a2} 60d corr",
                    line=dict(color=cat_col, width=1.5),
                    fill="tozeroy",
                    fillcolor=f"rgba({int(cat_col[1:3],16)},{int(cat_col[3:5],16)},{int(cat_col[5:7],16)},0.08)",
                ))
                fig_mini.add_hline(y=0, line=dict(color="#ABABAB", width=1, dash="dot"))
                for ev in GEOPOLITICAL_EVENTS:
                    fig_mini.add_vrect(
                        x0=str(ev["start"]), x1=str(ev["end"]),
                        fillcolor=ev["color"], opacity=0.05, layer="below", line_width=0,
                    )
                fig_mini.update_layout(
                    template="purdue", height=200,
                    title=dict(text=f"Rolling 60d Correlation: {a1} / {a2}", font=dict(size=11)),
                    showlegend=False, margin=dict(l=40, r=20, t=35, b=30),
                    xaxis=dict(rangeslider=dict(visible=False)),
                )
                _chart(fig_mini)

    # ── Download report ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.74rem;font-weight:700;letter-spacing:0.06em;'
        'text-transform:uppercase;color:#333;margin-bottom:0.5rem">'
        'Institution Report Download</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:0.78rem;color:#555960;line-height:1.7;margin-bottom:0.8rem">'
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
    _page_footer()
