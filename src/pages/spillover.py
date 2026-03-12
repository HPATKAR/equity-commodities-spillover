"""
Page 4 - Spillover Analytics
Granger causality grid, transfer entropy flows, Diebold-Yilmaz spillover index.
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
)


def page_spillover(start: str, end: str, fred_key: str = "") -> None:
    st.markdown(
        '<h1 style="font-family:\'DM Sans\',sans-serif;font-size:1.25rem;'
        'font-weight:700;margin-bottom:0.3rem">Spillover Analytics</h1>',
        unsafe_allow_html=True,
    )
    _page_intro(
        "Directional spillover analysis quantifies <em>which</em> market leads and "
        "<em>which</em> follows during stress. Granger causality tests whether past commodity "
        "returns improve the forecast of equity returns (and vice versa). "
        "Transfer entropy captures nonlinear information flow. "
        "Diebold-Yilmaz provides a network-level spillover index."
    )

    with st.spinner("Loading returns…"):
        eq_r, cmd_r = load_returns(start, end)

    if eq_r.empty or cmd_r.empty:
        st.error("Market data unavailable.")
        return

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Granger Causality", "Transfer Entropy", "Diebold-Yilmaz Index", "Spillover Network"
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 - Granger Causality
    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Granger Causality: Commodity → Equity & Equity → Commodity")
        _definition_block(
            "Granger Causality",
            "Series X Granger-causes series Y if past values of X significantly "
            "improve forecasts of Y beyond Y's own history. "
            "This is a predictive (not structural) test - "
            "it identifies statistical lead-lag relationships, not economic causation. "
            "p < 0.05 → significant at 5% level.",
        )

        # Subset selector
        c1, c2 = st.columns(2)
        eq_region   = c1.selectbox("Equity region", ["All"] + list(EQUITY_REGIONS.keys()))
        cmd_group   = c2.selectbox("Commodity group", ["All"] + list(COMMODITY_GROUPS.keys()))
        max_lag     = st.slider("Max lag (days)", 1, 10, 5)

        eq_subset = (
            [c for c in EQUITY_REGIONS[eq_region] if c in eq_r.columns]
            if eq_region != "All" else list(eq_r.columns)
        )
        cmd_subset = (
            [c for c in COMMODITY_GROUPS[cmd_group] if c in cmd_r.columns]
            if cmd_group != "All" else list(cmd_r.columns)
        )

        if st.button("Run Granger Tests", type="primary"):
            with st.spinner("Running Granger causality tests… (this may take ~30s)"):
                granger_df = granger_grid(
                    eq_r[eq_subset], cmd_r[cmd_subset], max_lag=max_lag
                )

            if granger_df.empty:
                st.warning("Not enough overlapping data.")
            else:
                st.success(f"Tested {len(granger_df)} pairs.")

                # Significant only
                sig_df = granger_df[granger_df["significant"]].copy()
                all_df = granger_df.copy()

                col_a, col_b = st.columns(2)
                col_a.metric("Significant pairs (p<0.05)", len(sig_df))
                col_b.metric(
                    "Commodity → Equity",
                    len(sig_df[sig_df["direction"] == "Commodity → Equity"]),
                )

                if not sig_df.empty:
                    # Heatmap: p-values for commodity → equity
                    c2e = sig_df[sig_df["direction"] == "Commodity → Equity"].copy()
                    if not c2e.empty:
                        pivot = c2e.pivot(index="effect", columns="cause", values="min_p")
                        fig_gc = go.Figure(go.Heatmap(
                            z=pivot.values,
                            x=pivot.columns.tolist(),
                            y=pivot.index.tolist(),
                            colorscale=[
                                [0.0, "#c0392b"],
                                [0.05, "#e74c3c"],
                                [0.1, "#f9f3ea"],
                                [1.0, "#ffffff"],
                            ],
                            zmin=0, zmax=0.1,
                            text=pivot.round(3).values,
                            texttemplate="%{text}",
                            textfont=dict(size=8, family="JetBrains Mono, monospace"),
                            colorbar=dict(title="p-value", thickness=12),
                        ))
                        fig_gc.update_layout(
                            template="purdue", height=400,
                            title="Commodity → Equity: Granger p-values (red = significant)",
                            xaxis=dict(tickangle=-35, tickfont=dict(size=8)),
                            yaxis=dict(tickfont=dict(size=8)),
                            margin=dict(l=120, r=40, t=50, b=120),
                        )
                        _chart(fig_gc)

                    st.dataframe(
                        sig_df[["cause", "effect", "direction", "min_p", "best_lag"]]
                        .sort_values("min_p")
                        .head(30),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.info("No significant Granger relationships found with current settings.")

        _section_note(
            "Energy commodities (WTI, Natural Gas) typically Granger-cause equity returns "
            "with 1–3 day lags during supply shocks. Gold tends to Granger-cause equity "
            "returns negatively during flight-to-safety episodes."
        )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 - Transfer Entropy
    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Transfer Entropy: Information Flow Direction")
        _definition_block(
            "Transfer Entropy (Schreiber, 2000)",
            "A model-free, nonlinear measure of directional information flow. "
            "TE(X→Y) quantifies how much knowing X's past reduces uncertainty about Y's future, "
            "beyond Y's own past. Unlike Granger, TE detects nonlinear dependencies. "
            "Net flow = TE(commodity→equity) − TE(equity→commodity): "
            "positive = commodity leads equity.",
        )

        c1, c2 = st.columns(2)
        te_eq_sel  = c1.multiselect(
            "Equities",  list(eq_r.columns),
            default=list(eq_r.columns)[:5], key="te_eq",
        )
        te_cmd_sel = c2.multiselect(
            "Commodities", list(cmd_r.columns),
            default=["WTI Crude Oil", "Gold", "Wheat", "Copper", "Natural Gas"],
            key="te_cmd",
        )

        if st.button("Compute Transfer Entropy", type="primary"):
            te_cmd_sel_valid = [c for c in te_cmd_sel if c in cmd_r.columns]
            te_eq_sel_valid  = [c for c in te_eq_sel  if c in eq_r.columns]

            if not te_cmd_sel_valid or not te_eq_sel_valid:
                st.warning("Select valid assets.")
            else:
                with st.spinner("Computing transfer entropy… (may take ~20s)"):
                    te_c2e, te_e2c = transfer_entropy_matrix(
                        eq_r[te_eq_sel_valid],
                        cmd_r[te_cmd_sel_valid],
                    )
                    net_te = net_flow_matrix(te_c2e, te_e2c)

                st.subheader("Net Transfer Entropy (Commodity → Equity direction)")
                _section_note(
                    "Green = commodity information flows TO equity (commodity leads). "
                    "Red = equity information flows TO commodity (equity leads). "
                    "Magnitude indicates strength of the directional link."
                )
                fig_te = go.Figure(go.Heatmap(
                    z=net_te.values.astype(float),
                    x=net_te.columns.tolist(),
                    y=net_te.index.tolist(),
                    colorscale=[
                        [0.0,  "#c0392b"],
                        [0.5,  "#ffffff"],
                        [1.0,  "#2e7d32"],
                    ],
                    zmid=0,
                    text=net_te.round(4).values.astype(float),
                    texttemplate="%{text:.4f}",
                    textfont=dict(size=8, family="JetBrains Mono, monospace"),
                    colorbar=dict(title="Net TE", thickness=12),
                ))
                fig_te.update_layout(
                    template="purdue", height=420,
                    xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
                    yaxis=dict(tickfont=dict(size=9)),
                    margin=dict(l=120, r=40, t=30, b=100),
                )
                _chart(fig_te)

        _takeaway_block(
            "Energy commodities (WTI, Brent, Natural Gas) typically show the strongest "
            "net transfer entropy INTO equities - particularly for Europe (DAX, Eurostoxx) "
            "which has high energy import dependency. Gold's TE often flows FROM equities "
            "during risk-off, reflecting reactive safe-haven demand rather than a leading signal."
        )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 - Diebold-Yilmaz
    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Diebold-Yilmaz Spillover Index")
        _definition_block(
            "Diebold-Yilmaz (2012) Spillover Index",
            "Uses a VAR-based forecast error variance decomposition (FEVD) to measure "
            "how much of asset i's forecast error variance is explained by shocks to asset j. "
            "The total spillover index aggregates all cross-asset contributions into a single "
            "network-level measure of interconnectedness.",
        )

        # Select assets for VAR
        all_cols = list(eq_r.columns) + list(cmd_r.columns)
        dy_assets = st.multiselect(
            "Assets for VAR (max 8 recommended)",
            all_cols,
            default=["S&P 500", "Eurostoxx 50", "Nikkei 225",
                     "WTI Crude Oil", "Gold", "Wheat"],
        )

        if st.button("Run Diebold-Yilmaz", type="primary"):
            dy_valid = [c for c in dy_assets if c in eq_r.columns or c in cmd_r.columns]
            if len(dy_valid) < 3:
                st.warning("Select at least 3 assets.")
            else:
                combined = pd.concat([eq_r, cmd_r], axis=1)[dy_valid].dropna()
                with st.spinner("Fitting VAR model…"):
                    result = diebold_yilmaz(combined, top_n=len(dy_valid))

                total = result["total_spillover"]
                table = result["spillover_table"]

                st.metric("Total Spillover Index", f"{total:.1f}%",
                          help="Share of FEVD explained by cross-asset shocks. Higher = more interconnected.")

                if not table.empty:
                    fig_dy = go.Figure(go.Heatmap(
                        z=table.values,
                        x=table.columns.tolist(),
                        y=table.index.tolist(),
                        colorscale="YlOrRd",
                        zmin=0,
                        text=table.values,
                        texttemplate="%{text:.1f}%",
                        textfont=dict(size=9, family="JetBrains Mono, monospace"),
                        colorbar=dict(title="% FEVD", thickness=12),
                    ))
                    fig_dy.update_layout(
                        template="purdue", height=420,
                        title="Forecast Error Variance Decomposition (%)",
                        xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
                        yaxis=dict(tickfont=dict(size=9)),
                        margin=dict(l=100, r=40, t=50, b=100),
                    )
                    _chart(fig_dy)

                    _takeaway_block(
                        f"Total spillover index: <b>{total:.1f}%</b>. "
                        f"Values above 50% indicate high systemic interconnectedness - "
                        "typical of crisis periods. "
                        "Off-diagonal entries show the largest net transmitters and receivers."
                    )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 - Spillover Network
    # ══════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("Spillover Network Graph")
        _definition_block(
            "Network Interpretation",
            "<b>DY Network:</b> Nodes = assets. Directed edge j → i = j's shocks explain "
            "a significant share of i's forecast error variance. "
            "Node size ∝ |net spillover score| (Transmit − Receive). "
            "Large nodes = strong net transmitters (positive) or receivers (negative).<br><br>"
            "<b>Granger Network:</b> Edge cause → effect = cause significantly Granger-causes effect. "
            "Red = commodity → equity flow. Blue = equity → commodity flow.",
        )

        # ── Controls ──
        all_cols = list(eq_r.columns) + list(cmd_r.columns)
        c1, c2, c3 = st.columns([2, 1, 1])

        net_assets = c1.multiselect(
            "Assets (max 10 recommended)",
            all_cols,
            default=["S&P 500", "DAX", "Nikkei 225", "Shanghai Comp", "Sensex",
                     "WTI Crude Oil", "Gold", "Wheat", "Copper", "Natural Gas"],
            key="net_assets",
        )
        layout_opt = c2.selectbox(
            "Layout",
            ["bipartite", "spring", "circular"],
            key="net_layout",
        )
        dy_thresh = c3.slider(
            "DY edge threshold (%)",
            min_value=1, max_value=15, value=4,
            help="Minimum % FEVD for an edge to appear",
            key="net_thresh",
        )

        if st.button("Build Network Graphs", type="primary", key="net_build"):
            net_valid = [c for c in net_assets if c in eq_r.columns or c in cmd_r.columns]
            if len(net_valid) < 3:
                st.warning("Select at least 3 assets.")
            else:
                combined = pd.concat([eq_r, cmd_r], axis=1)[net_valid].dropna()

                # ── DY Network ──────────────────────────────────────────
                with st.spinner("Fitting VAR for Diebold-Yilmaz…"):
                    dy_result = diebold_yilmaz(combined, top_n=len(net_valid))

                dy_table = dy_result["spillover_table"]
                total_sp = dy_result["total_spillover"]

                if not dy_table.empty:
                    st.markdown("---")
                    st.markdown("#### Diebold-Yilmaz Spillover Network")

                    m1, m2 = st.columns(2)
                    m1.metric("Total Spillover Index", f"{total_sp:.1f}%")
                    m2.metric("Assets", len(net_valid))

                    G_dy = build_dy_graph(dy_table, threshold=dy_thresh)
                    st.session_state["_dy_graph"] = G_dy

                    # Net transmitter bar
                    _chart(plot_net_transmitter_bar(G_dy, height=max(280, len(net_valid) * 24)))

                    _section_note(
                        "Green bars = net transmitters (their shocks spill INTO other assets). "
                        "Red bars = net receivers (absorb shocks from others). "
                        "Energy commodities and the S&P 500 are typically the largest net transmitters."
                    )

                    # Network graph
                    _chart(plot_dy_network(
                        G_dy,
                        title=f"DY Spillover Network · Total Index {total_sp:.1f}%",
                        layout=layout_opt,
                        top_edges=min(len(net_valid) * 4, 32),
                    ))

                # ── Granger Network ─────────────────────────────────────
                st.markdown("---")
                st.markdown("#### Granger Causality Network")

                eq_net  = [c for c in net_valid if c in eq_r.columns]
                cmd_net = [c for c in net_valid if c in cmd_r.columns]

                if eq_net and cmd_net:
                    with st.spinner("Running Granger tests for network…"):
                        g_df = granger_grid(eq_r[eq_net], cmd_r[cmd_net], max_lag=5)

                    sig_count = g_df["significant"].sum() if not g_df.empty else 0
                    st.metric("Significant Granger links", int(sig_count))

                    if sig_count > 0:
                        G_gr = build_granger_graph(g_df)
                        st.session_state["_granger_graph"] = G_gr
                        _chart(plot_granger_network(
                            G_gr,
                            title="Granger Causality Network (p < 0.05)",
                            layout=layout_opt,
                        ))
                        _takeaway_block(
                            "Node size = number of assets it Granger-causes. "
                            "Red arrows = commodity information flowing into equity markets. "
                            "Blue arrows = equity-driven flows back into commodities. "
                            "Energy commodities typically dominate outbound red arrows."
                        )
                    else:
                        st.info("No significant Granger links found. Try a longer date range or looser lag settings.")

    _page_footer()
