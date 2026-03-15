"""
Page 4 - Spillover Analytics
Granger causality grid, transfer entropy flows, Diebold-Yilmaz spillover index.
All four analyses pre-run and displayed in a grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_returns
from src.data.config import PALETTE, EQUITY_REGIONS, COMMODITY_GROUPS
from src.analysis.spillover import (
    granger_grid, transfer_entropy_matrix, net_flow_matrix, diebold_yilmaz,
)
from src.analysis.network import (
    build_dy_graph, build_granger_graph,
    plot_dy_network, plot_granger_network, plot_net_transmitter_bar,
)
from src.ui.shared import (
    _style_fig, _chart, _page_intro, _section_note,
    _definition_block, _takeaway_block, _page_conclusion, _page_footer,
    _insight_note,
)

_F = "font-family:'DM Sans',sans-serif;"

_DEFAULT_EQ  = ["S&P 500", "DAX", "Nikkei 225", "Shanghai Comp", "Sensex"]
_DEFAULT_CMD = ["WTI Crude Oil", "Gold", "Wheat", "Copper", "Natural Gas"]
_DEFAULT_ALL = _DEFAULT_EQ + _DEFAULT_CMD


def _label(txt: str) -> None:
    st.markdown(
        f'<p style="{_F}font-size:0.58rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.14em;color:#8E6F3E;margin:0 0 5px 0">{txt}</p>',
        unsafe_allow_html=True,
    )


def _panel_note(txt: str) -> None:
    st.markdown(
        f'<p style="{_F}font-size:0.64rem;color:#666;line-height:1.5;margin:4px 0 0 0">{txt}</p>',
        unsafe_allow_html=True,
    )


def page_spillover(start: str, end: str, fred_key: str = "") -> None:
    st.markdown(
        '<h1 style="font-family:\'DM Sans\',sans-serif;font-size:1.25rem;'
        'font-weight:700;margin-bottom:0.1rem">Spillover Analytics</h1>'
        '<p style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;color:#555;'
        'margin:0 0 0.7rem">Granger Causality · Transfer Entropy · Diebold-Yilmaz · Network Graph</p>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading returns…"):
        eq_r, cmd_r = load_returns(start, end)

    if eq_r.empty or cmd_r.empty:
        st.error("Market data unavailable.")
        return

    all_r   = pd.concat([eq_r, cmd_r], axis=1)
    all_col = list(all_r.columns)

    # Validate defaults
    def_eq  = [c for c in _DEFAULT_EQ  if c in eq_r.columns]
    def_cmd = [c for c in _DEFAULT_CMD if c in cmd_r.columns]
    def_all = [c for c in _DEFAULT_ALL if c in all_r.columns]

    # ── Asset selection strip ───────────────────────────────────────────────
    with st.expander("Asset selection (applies to all panels)", expanded=False):
        sel_l, sel_r = st.columns(2)
        sel_eq  = sel_l.multiselect("Equities",    list(eq_r.columns),  default=def_eq,  key="sp_eq")
        sel_cmd = sel_r.multiselect("Commodities", list(cmd_r.columns), default=def_cmd, key="sp_cmd")
        sel_all = st.multiselect("VAR assets (DY + Network)", all_col, default=def_all, key="sp_all")
        max_lag   = st.slider("Granger max lag (days)", 1, 10, 5, key="sp_lag")
        dy_thresh = st.slider("DY edge threshold (%)",  1, 15,  4, key="sp_thresh")
        layout    = st.selectbox("Network layout", ["bipartite","spring","circular"], key="sp_layout")

    sel_eq  = [c for c in sel_eq  if c in eq_r.columns]  or def_eq
    sel_cmd = [c for c in sel_cmd if c in cmd_r.columns] or def_cmd
    sel_all = [c for c in sel_all if c in all_r.columns]  or def_all

    st.markdown('<div style="margin:0.4rem 0 0.5rem;border-top:1px solid #E8E5E0"></div>',
                unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # ROW 1: Granger Causality (wider) | Transfer Entropy (narrower)
    # ══════════════════════════════════════════════════════════════════════
    col_gc, col_te = st.columns([1.2, 1], gap="medium")

    # ── Panel 1: Granger Causality ─────────────────────────────────────────
    with col_gc:
        _label("Granger Causality: Commodity → Equity p-values")
        with st.spinner("Running Granger causality tests…"):
            granger_df = granger_grid(eq_r[sel_eq], cmd_r[sel_cmd], max_lag=max_lag)

        if granger_df.empty:
            st.warning("Not enough data for Granger tests.")
        else:
            sig_df = granger_df[granger_df["significant"]].copy()
            c2e    = sig_df[sig_df["direction"] == "Commodity → Equity"]

            # Metrics row
            m1, m2, m3 = st.columns(3)
            m1.metric("Tested pairs",      len(granger_df))
            m2.metric("Significant (p<.05)", len(sig_df))
            m3.metric("Cmd → Equity links", len(c2e))

            if not c2e.empty:
                pivot = c2e.pivot(index="effect", columns="cause", values="min_p")
                fig_gc = go.Figure(go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns.tolist(),
                    y=pivot.index.tolist(),
                    colorscale=[[0,"#c0392b"],[0.05,"#e74c3c"],[0.1,"#f9f3ea"],[1,"#ffffff"]],
                    zmin=0, zmax=0.1,
                    text=pivot.round(3).values,
                    texttemplate="%{text}",
                    textfont=dict(size=8, family="JetBrains Mono, monospace"),
                    colorbar=dict(title="p-val", thickness=10),
                ))
                fig_gc.update_layout(
                    template="purdue", height=320,
                    xaxis=dict(tickangle=-35, tickfont=dict(size=8)),
                    yaxis=dict(tickfont=dict(size=8)),
                    margin=dict(l=100, r=40, t=20, b=90),
                )
                _chart(fig_gc)
                _panel_note("Red = strong lead (low p-value). Energy commodities typically lead equities by 1–3 days.")
                _insight_note(
                    "Red cells indicate the column commodity statistically 'Granger-causes' the row "
                    "equity - meaning its past price moves help predict where that equity is heading. "
                    "This is not coincidence: it means commodity price changes consistently arrive "
                    "before equity price changes, giving a 1–3 day early warning window."
                )
            else:
                _panel_note("No significant Commodity → Equity Granger links with current selection.")

            # Top pairs table in expander
            if not sig_df.empty:
                with st.expander(f"Top significant pairs ({len(sig_df)})"):
                    st.dataframe(
                        sig_df[["cause","effect","direction","min_p","best_lag"]]
                        .sort_values("min_p").head(20),
                        use_container_width=True, hide_index=True,
                    )

    # ── Panel 2: Transfer Entropy ──────────────────────────────────────────
    with col_te:
        _label("Transfer Entropy: Net Information Flow")
        with st.spinner("Computing transfer entropy…"):
            te_c2e, te_e2c = transfer_entropy_matrix(eq_r[sel_eq], cmd_r[sel_cmd])
            net_te = net_flow_matrix(te_c2e, te_e2c)

        if net_te.empty:
            st.warning("Transfer entropy computation failed.")
        else:
            fig_te = go.Figure(go.Heatmap(
                z=net_te.values.astype(float),
                x=net_te.columns.tolist(),
                y=net_te.index.tolist(),
                colorscale=[[0,"#c0392b"],[0.5,"#ffffff"],[1,"#2e7d32"]],
                zmid=0,
                text=net_te.round(4).values.astype(float),
                texttemplate="%{text:.3f}",
                textfont=dict(size=8, family="JetBrains Mono, monospace"),
                colorbar=dict(title="Net TE", thickness=10),
            ))
            fig_te.update_layout(
                template="purdue", height=320,
                xaxis=dict(tickangle=-35, tickfont=dict(size=8)),
                yaxis=dict(tickfont=dict(size=8)),
                margin=dict(l=100, r=40, t=20, b=90),
            )
            _chart(fig_te)
            _panel_note("Green = commodity leads equity. Red = equity leads commodity.")
            _insight_note(
                "Transfer entropy measures the net direction of information flow between two assets - "
                "which one is telling the story, and which one is listening. Green cells mean the "
                "commodity is driving the equity. Red cells mean the equity is driving the commodity. "
                "This goes beyond correlation by capturing the directionality of influence."
            )

    st.markdown('<div style="margin:0.5rem 0;border-top:1px solid #E8E5E0"></div>',
                unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # ROW 2: Diebold-Yilmaz FEVD (wider) | Network graph (narrower)
    # ══════════════════════════════════════════════════════════════════════
    col_dy, col_net = st.columns([1.1, 1], gap="medium")

    dy_valid = [c for c in sel_all if c in all_r.columns]
    dy_result, dy_table, total_sp, G_dy = None, pd.DataFrame(), 0.0, None

    if len(dy_valid) >= 3:
        combined = all_r[dy_valid].dropna()
        with st.spinner("Fitting VAR (Diebold-Yilmaz)…"):
            dy_result = diebold_yilmaz(combined, top_n=len(dy_valid))
        dy_table = dy_result["spillover_table"]
        total_sp = dy_result["total_spillover"]
        if not dy_table.empty:
            G_dy = build_dy_graph(dy_table, threshold=dy_thresh)

    # ── Panel 3: Diebold-Yilmaz FEVD ──────────────────────────────────────
    with col_dy:
        _label("Diebold-Yilmaz: Forecast Error Variance Decomposition")
        if dy_table.empty:
            st.warning("Select ≥ 3 VAR assets.")
        else:
            st.metric("Total Spillover Index", f"{total_sp:.1f}%",
                      help="% of FEVD from cross-asset shocks. >50% = high interconnectedness.")
            fig_dy = go.Figure(go.Heatmap(
                z=dy_table.values,
                x=dy_table.columns.tolist(),
                y=dy_table.index.tolist(),
                colorscale="YlOrRd", zmin=0,
                text=dy_table.values,
                texttemplate="%{text:.1f}%",
                textfont=dict(size=8, family="JetBrains Mono, monospace"),
                colorbar=dict(title="% FEVD", thickness=10),
            ))
            fig_dy.update_layout(
                template="purdue", height=340,
                xaxis=dict(tickangle=-35, tickfont=dict(size=8)),
                yaxis=dict(tickfont=dict(size=8)),
                margin=dict(l=100, r=40, t=20, b=90),
            )
            _chart(fig_dy)
            _insight_note(
                "Each cell shows what percentage of one asset's price uncertainty (forecast error) "
                "is caused by shocks originating in another asset. The diagonal is self-driven. "
                "A high total spillover index (above 50%) means markets are deeply interconnected - "
                "a shock anywhere in the system propagates everywhere quickly."
            )
            _panel_note(
                f"Total spillover: <b>{total_sp:.1f}%</b> · "
                "Off-diagonal = cross-asset variance explained. "
                "High values = systemic transmission risk."
            )

    # ── Panel 4: Spillover Network ─────────────────────────────────────────
    with col_net:
        _label("Spillover Network Graph")
        if G_dy is None or dy_table.empty:
            st.warning("Insufficient data for network graph.")
        else:
            # Net transmitter bar (compact)
            _chart(plot_net_transmitter_bar(G_dy, height=max(200, len(dy_valid) * 22)))
            _panel_note("Green = net transmitter (spills shocks outward). Red = net receiver.")
            _insight_note(
                "Green bars are 'shock exporters' - when they move, other assets tend to follow. "
                "Red bars are 'shock absorbers' - they react to moves elsewhere. Energy commodities "
                "like WTI tend to be strong transmitters; equity markets tend to be large receivers "
                "of commodity-originated shocks during supply disruption events."
            )

            # Full network in expander
            with st.expander("View full network graph"):
                _chart(plot_dy_network(
                    G_dy,
                    title=f"DY Network · Total {total_sp:.1f}%",
                    layout=layout,
                    top_edges=min(len(dy_valid) * 4, 32),
                ))
                # Granger network
                eq_net  = [c for c in dy_valid if c in eq_r.columns]
                cmd_net = [c for c in dy_valid if c in cmd_r.columns]
                if eq_net and cmd_net and not granger_df.empty:
                    G_gr = build_granger_graph(granger_df)
                    _chart(plot_granger_network(G_gr,
                        title="Granger Causality Network (p < 0.05)", layout=layout))

    _page_footer()
