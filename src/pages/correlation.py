"""
Page 3 — Correlation Analysis
Rolling correlations, DCC-GARCH dynamic pairs, regime detection, pair explorer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_returns
from src.data.config import (
    GEOPOLITICAL_EVENTS, EQUITY_REGIONS, COMMODITY_GROUPS, PALETTE,
)
from src.analysis.correlations import (
    rolling_correlation, cross_asset_corr, dcc_correlation,
    average_cross_corr_series, detect_correlation_regime,
)
from src.ui.shared import (
    _style_fig, _chart, _page_intro, _section_note,
    _definition_block, _takeaway_block, _page_conclusion, _page_footer,
    _add_event_bands,
)


_REGIME_NAMES  = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis"}
_REGIME_COLORS = {0: "#2e7d32",      1: "#555960", 2: "#e67e22",  3: "#c0392b"}


def page_correlation(start: str, end: str, fred_key: str = "") -> None:
    st.markdown(
        '<h1 style="font-family:\'DM Sans\',sans-serif;font-size:1.25rem;'
        'font-weight:700;margin-bottom:0.3rem">Correlation Analysis</h1>',
        unsafe_allow_html=True,
    )
    _page_intro(
        "Rolling Pearson correlations and DCC-GARCH dynamic conditional correlations "
        "between global equity indices and commodity futures. Regime detection flags "
        "when correlations enter historically elevated or crisis territory."
    )

    with st.spinner("Loading returns…"):
        eq_r, cmd_r = load_returns(start, end)

    if eq_r.empty or cmd_r.empty:
        st.error("Market data unavailable.")
        return

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Rolling Correlation", "DCC-GARCH", "Regime Detection"])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — Rolling Correlation
    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Rolling Pairwise Correlation Explorer")
        _definition_block(
            "Rolling Pearson Correlation",
            "Measures how closely two assets move together over a trailing window. "
            "+1 = perfect co-movement, −1 = perfect divergence. "
            "Crisis periods typically push correlations toward +1 as markets sell off uniformly.",
        )

        c1, c2, c3 = st.columns(3)
        eq_options  = [c for c in eq_r.columns  if c in eq_r.columns]
        cmd_options = [c for c in cmd_r.columns if c in cmd_r.columns]
        all_opts    = eq_options + cmd_options

        asset_a  = c1.selectbox("Asset A", all_opts,  index=0)
        asset_b  = c2.selectbox("Asset B", all_opts,  index=len(eq_options))
        window   = c3.select_slider("Window (days)", [21, 42, 63, 126, 252], value=63)

        all_r = pd.concat([eq_r, cmd_r], axis=1)
        if asset_a in all_r.columns and asset_b in all_r.columns:
            rc = rolling_correlation(all_r[asset_a], all_r[asset_b], window)

            fig_rc = go.Figure()
            fig_rc.add_trace(go.Scatter(
                x=rc.index, y=rc.values,
                name=f"{asset_a} / {asset_b}",
                line=dict(color=PALETTE[1], width=1.8),
                fill="tozeroy", fillcolor="rgba(207,185,145,0.1)",
            ))
            fig_rc.add_hline(y=0,    line=dict(color="#ABABAB", width=1, dash="dot"))
            fig_rc.add_hline(y=0.5,  line=dict(color="#e67e22", width=0.8, dash="dot"),
                             annotation_text="Elevated", annotation_font_size=8)
            fig_rc.add_hline(y=-0.5, line=dict(color="#2980b9", width=0.8, dash="dot"),
                             annotation_text="Diverge", annotation_font_size=8)
            _add_event_bands(fig_rc)
            _chart(_style_fig(fig_rc, height=400))

            latest_corr = rc.dropna().iloc[-1] if not rc.dropna().empty else 0
            pctile = (rc.dropna() <= latest_corr).mean() * 100
            _takeaway_block(
                f"<b>{asset_a} / {asset_b}</b>: current {window}d correlation = "
                f"<b>{latest_corr:.3f}</b> — "
                f"{pctile:.0f}th historical percentile."
            )

        # Multi-pair overlay
        st.subheader("Multi-Pair Rolling Correlation Overlay")
        default_pairs = [
            ("S&P 500", "WTI Crude Oil"),
            ("S&P 500", "Gold"),
            ("Nikkei 225", "WTI Crude Oil"),
            ("Eurostoxx 50", "Natural Gas"),
        ]
        default_pairs = [(a, b) for a, b in default_pairs
                         if a in all_r.columns and b in all_r.columns]

        fig_multi = go.Figure()
        for i, (a, b) in enumerate(default_pairs):
            rc2 = rolling_correlation(all_r[a], all_r[b], 63)
            fig_multi.add_trace(go.Scatter(
                x=rc2.index, y=rc2.values,
                name=f"{a} / {b}",
                line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
            ))
        fig_multi.add_hline(y=0, line=dict(color="#ABABAB", width=1, dash="dot"))
        _add_event_bands(fig_multi)
        _chart(_style_fig(fig_multi, height=400))

        _section_note(
            "Note how S&P 500 / Gold correlation turns sharply negative during market stress "
            "(flight-to-safety), while S&P 500 / WTI correlation spikes positive during "
            "demand-driven risk-off episodes and negative during supply shocks."
        )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — DCC-GARCH
    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("DCC-GARCH Dynamic Conditional Correlation")
        _definition_block(
            "DCC-GARCH (Engle, 2002)",
            "Unlike rolling Pearson, DCC-GARCH accounts for heteroskedasticity — "
            "it explicitly models the time-varying variance structure of each asset. "
            "DCC captures correlation dynamics more accurately during volatile periods "
            "and gives smoother, more statistically robust estimates.",
        )

        c1, c2 = st.columns(2)
        dcc_eq  = c1.selectbox("Equity", eq_options,  index=0, key="dcc_eq")
        dcc_cmd = c2.selectbox("Commodity", cmd_options, index=0, key="dcc_cmd")

        if dcc_eq in eq_r.columns and dcc_cmd in cmd_r.columns:
            with st.spinner("Computing DCC-GARCH…"):
                dcc_series = dcc_correlation(eq_r[dcc_eq], cmd_r[dcc_cmd])
                roll_series = rolling_correlation(eq_r[dcc_eq], cmd_r[dcc_cmd], 63)

            fig_dcc = go.Figure()
            fig_dcc.add_trace(go.Scatter(
                x=dcc_series.index, y=dcc_series.values,
                name="DCC-GARCH",
                line=dict(color=PALETTE[0], width=2),
            ))
            fig_dcc.add_trace(go.Scatter(
                x=roll_series.index, y=roll_series.values,
                name="Rolling 63d Pearson",
                line=dict(color=PALETTE[1], width=1.5, dash="dot"),
            ))
            fig_dcc.add_hline(y=0, line=dict(color="#ABABAB", width=1, dash="dot"))
            _add_event_bands(fig_dcc)
            _chart(_style_fig(fig_dcc, height=420))

            latest_dcc = dcc_series.dropna().iloc[-1] if not dcc_series.dropna().empty else 0
            _takeaway_block(
                f"<b>{dcc_eq} / {dcc_cmd}</b>: DCC correlation = <b>{latest_dcc:.3f}</b>. "
                "DCC typically diverges from rolling Pearson during regime transitions — "
                "the GARCH component captures volatility clustering that biases static estimates."
            )

        # DCC comparison: multiple commodity pairs for one equity
        st.subheader("DCC Comparison: One Equity vs Multiple Commodities")
        ref_eq = st.selectbox("Reference equity", eq_options, index=0, key="dcc_ref")
        cmd_sel_multi = st.multiselect(
            "Commodities",
            cmd_options,
            default=cmd_options[:5],
            key="dcc_cmds",
        )

        if ref_eq in eq_r.columns and cmd_sel_multi:
            fig_dcc2 = go.Figure()
            for i, cmd in enumerate(cmd_sel_multi):
                if cmd in cmd_r.columns:
                    dcc_s = dcc_correlation(eq_r[ref_eq], cmd_r[cmd])
                    fig_dcc2.add_trace(go.Scatter(
                        x=dcc_s.index, y=dcc_s.values,
                        name=cmd,
                        line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
                    ))
            fig_dcc2.add_hline(y=0, line=dict(color="#ABABAB", width=1, dash="dot"))
            _add_event_bands(fig_dcc2)
            _chart(_style_fig(fig_dcc2, height=400))

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — Regime Detection
    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Cross-Asset Correlation Regime Detection")
        _definition_block(
            "Regime Classification",
            "The daily average absolute cross-asset correlation is classified into four regimes: "
            "Decorrelated (< 0.15), Normal (0.15–0.45), Elevated (0.45–0.60), "
            "Crisis (≥ 0.60 for 3+ consecutive days). "
            "Regime detection helps identify when cross-asset hedges break down.",
        )

        avg_corr = average_cross_corr_series(eq_r, cmd_r, window=60)
        regimes  = detect_correlation_regime(avg_corr)

        if not avg_corr.empty:
            # Colour the series by regime
            fig_reg = go.Figure()

            # Background regime bands
            for r_val in [0, 1, 2, 3]:
                mask = (regimes == r_val)
                # Add coloured scatter as region markers
                fig_reg.add_trace(go.Scatter(
                    x=avg_corr.index[mask],
                    y=avg_corr.values[mask],
                    mode="markers",
                    marker=dict(color=_REGIME_COLORS[r_val], size=2, opacity=0.5),
                    name=_REGIME_NAMES[r_val],
                    showlegend=True,
                ))

            fig_reg.add_trace(go.Scatter(
                x=avg_corr.index, y=avg_corr.values,
                name="Avg |Corr|",
                line=dict(color="#000000", width=1.2),
                showlegend=False,
            ))
            for thresh, label, color in [
                (0.15, "Decorrelated", "#2e7d32"),
                (0.45, "Elevated",     "#e67e22"),
                (0.60, "Crisis",       "#c0392b"),
            ]:
                fig_reg.add_hline(
                    y=thresh,
                    line=dict(color=color, width=1, dash="dash"),
                    annotation_text=label,
                    annotation_font=dict(size=9, color=color),
                )
            _add_event_bands(fig_reg)
            _chart(_style_fig(fig_reg, height=420))

            # Regime summary table
            regime_counts = regimes.value_counts().sort_index()
            total = len(regimes)
            regime_df = pd.DataFrame({
                "Regime":     [_REGIME_NAMES[i] for i in regime_counts.index],
                "Days":       regime_counts.values,
                "% of Time":  (regime_counts.values / total * 100).round(1),
            })
            st.dataframe(regime_df, use_container_width=True, hide_index=True)

            current = regimes.iloc[-1]
            _takeaway_block(
                f"Current regime: <b style='color:{_REGIME_COLORS[current]}'>"
                f"{_REGIME_NAMES[current]}</b>. "
                "Crisis regimes (historically ~8–12% of trading days) coincide with "
                "the most profitable cross-asset hedging windows — but also the highest "
                "basis risk as correlations can snap back abruptly."
            )

    _page_footer()
