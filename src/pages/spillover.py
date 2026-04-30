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

from src.data.loader import load_returns, load_fixed_income_returns, load_fx_returns
from src.data.config import PALETTE, EQUITY_REGIONS, COMMODITY_GROUPS
from src.analysis.spillover import (
    granger_grid, transfer_entropy_matrix, net_flow_matrix, diebold_yilmaz,
    rolling_diebold_yilmaz, regime_conditional_spillover,
)
from src.analysis.network import (
    build_dy_graph, build_granger_graph,
    plot_dy_network, plot_granger_network, plot_net_transmitter_bar,
)
from src.ui.shared import (
    _style_fig, _chart, _page_intro, _thread, _section_note,
    _definition_block, _takeaway_block, _page_conclusion, _page_header, _page_footer,
    _insight_note, _no_api_key_banner,
)

_F = "font-family:'DM Sans',sans-serif;"

_DEFAULT_EQ  = ["S&P 500", "DAX", "Nikkei 225", "Shanghai Comp", "Sensex"]
_DEFAULT_CMD = ["WTI Crude Oil", "Gold", "Wheat", "Copper", "Natural Gas"]
_DEFAULT_ALL = _DEFAULT_EQ + _DEFAULT_CMD


def _label(txt: str) -> None:
    st.markdown(
        f'<p style="{_F}font-size:0.58rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.14em;color:#8890a1;margin:0.8rem 0 0.4rem 0">{txt}</p>',
        unsafe_allow_html=True,
    )


def _panel_note(txt: str) -> None:
    st.markdown(
        f'<p style="{_F}font-size:0.64rem;color:#8890a1;line-height:1.5;margin:4px 0 0 0">{txt}</p>',
        unsafe_allow_html=True,
    )


def page_spillover(start: str, end: str, fred_key: str = "") -> None:
    _page_header("Spillover Network",
                 "Granger Causality · Transfer Entropy · Diebold-Yilmaz · Network Graph")
    _no_api_key_banner("AI spillover interpretation")
    _page_intro(
        "Correlation tells you <em>that</em> two markets move together. Spillover tells you <em>why</em> - "
        "and more importantly, <strong>which market is driving which.</strong> "
        "This is the analytical core of the dashboard. Granger causality tests whether past equity returns "
        "statistically predict future commodity returns (or the reverse). Transfer entropy measures "
        "directional information flow without assuming linearity. Diebold-Yilmaz decomposes forecast error "
        "variance to assign a transmitter/receiver score to every asset. When equities rank as net "
        "transmitters to commodities, the spillover is equity-led - a risk-off equity selloff is leaking "
        "into commodity markets before prices reflect it."
    )

    # ── Conflict transmission context banner ──────────────────────────────────
    try:
        from src.analysis.conflict_model import score_all_conflicts, aggregate_portfolio_scores
        _sp_cr  = score_all_conflicts()
        _sp_agg = aggregate_portfolio_scores(_sp_cr)
        _sp_tps = _sp_agg.get("portfolio_tps", 50.0)
        _sp_cis = _sp_agg.get("portfolio_cis", 50.0)

        # Top transmission channels across active conflicts
        _channel_scores: dict = {}
        for _cr_r in _sp_cr.values():
            if _cr_r.get("state") != "active":
                continue
            _w = _cr_r["cis"] / 100
            for _ch, _v in _cr_r.get("transmission", {}).items():
                _channel_scores[_ch] = _channel_scores.get(_ch, 0.0) + _v * _w
        _top_channels = sorted(_channel_scores.items(), key=lambda x: x[1], reverse=True)[:4]

        if _top_channels:
            _sp_color = "#c0392b" if _sp_tps >= 65 else "#e67e22" if _sp_tps >= 45 else "#CFB991"
            _ch_tags  = "".join(
                f'<span style="background:#0a1a0a;color:#27ae60;'
                f'font-family:\'JetBrains Mono\',monospace;font-size:7px;'
                f'padding:2px 6px;margin-right:5px;border:1px solid #27ae60;opacity:.8">'
                f'{ch.replace("_"," ").upper()}</span>'
                for ch, _ in _top_channels
            )
            st.markdown(
                f'<div style="background:#080808;border:1px solid #1e1e1e;'
                f'border-left:3px solid {_sp_color};padding:.4rem .9rem;'
                f'margin-bottom:.6rem;display:flex;align-items:center;gap:10px;flex-wrap:wrap">'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
                f'font-weight:700;color:{_sp_color};white-space:nowrap">ACTIVE TRANSMISSION CHANNELS</span>'
                f'{_ch_tags}'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
                f'color:#8E9AAA;margin-left:auto">'
                f'TPS&nbsp;<b style="color:{_sp_color}">{_sp_tps:.0f}</b>&nbsp;·&nbsp;'
                f'CIS&nbsp;<b style="color:#e67e22">{_sp_cis:.0f}</b>&nbsp;·&nbsp;'
                f'High TPS = geopolitical risk is actively flowing into asset prices</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        pass

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

    st.markdown('<div style="margin:0.4rem 0 0.5rem;border-top:1px solid #2a2a2a"></div>',
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
            sig_df  = granger_df[granger_df["significant"]].copy()
            holm_df = granger_df[granger_df.get("holm_significant", granger_df["significant"])].copy()
            c2e     = sig_df[sig_df["direction"] == "Commodity → Equity"]
            c2e_holm = holm_df[holm_df["direction"] == "Commodity → Equity"]

            # Metrics row — show both raw p<.05 count and Holm-corrected count
            m1, m2, m3 = st.columns(3)
            m1.metric("Tested pairs",          len(granger_df))
            m2.metric("Sig (p<.05 unadj.)",    len(sig_df),
                      delta=f"{len(holm_df)} Holm-corrected",
                      delta_color="off")
            m3.metric("Cmd → Equity (Holm)",   len(c2e_holm))

            if not c2e.empty and c2e["cause"].nunique() > 0 and c2e["effect"].nunique() > 0:
                pivot = c2e.pivot(index="effect", columns="cause", values="min_p")
                fig_gc = go.Figure(go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns.tolist(),
                    y=pivot.index.tolist(),
                    colorscale=[[0,"#c0392b"],[0.05,"#e74c3c"],[0.1,"#1e1e1e"],[1,"#111111"]],
                    zmin=0, zmax=0.1,
                    text=pivot.round(3).values,
                    texttemplate="%{text}",
                    textfont=dict(size=8, family="JetBrains Mono, monospace", color="#e8e9ed"),
                    colorbar=dict(title="p-val", thickness=10),
                ))
                fig_gc.update_layout(
                    template="purdue", height=320,
                    paper_bgcolor="#000", plot_bgcolor="#080808",
                    font=dict(color="#e8e9ed"),
                    xaxis=dict(tickangle=-35, tickfont=dict(size=8, color="#8890a1"), rangeslider=dict(visible=False)),
                    yaxis=dict(tickfont=dict(size=8, color="#8890a1")),
                    margin=dict(l=100, r=40, t=20, b=90),
                )
                try:
                    from src.analysis.freshness import add_freshness_label
                    fig_gc = add_freshness_label(fig_gc, "yfinance_prices")
                except Exception:
                    pass
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
                    _TBL_CSS = """
<style>
.ec-table{width:100%;border-collapse:collapse;font-family:'DM Sans',sans-serif;font-size:0.78rem}
.ec-table th{background:#1c1c1c;color:#CFB991;padding:7px 10px;text-align:left;
    border-bottom:1px solid rgba(207,185,145,0.3);font-weight:600;
    letter-spacing:0.06em;text-transform:uppercase;font-size:0.68rem}
.ec-table td{padding:5px 10px;border-bottom:1px solid #1e1e1e;color:#e8e9ed}
.ec-table tr:nth-child(even) td{background:#171717}
.ec-table tr:nth-child(odd) td{background:#111111}
.ec-table tr:hover td{background:#202020}
</style>"""
                    tbl_df = (
                        sig_df[["cause","effect","direction","min_p","best_lag"]]
                        .sort_values("min_p").head(20)
                    )
                    rows_html = ""
                    for _, row in tbl_df.iterrows():
                        p_val = row["min_p"]
                        p_color = "#4ade80" if p_val < 0.01 else "#e8e9ed"
                        rows_html += (
                            f"<tr>"
                            f"<td style='color:#b8b8b8'>{row['cause']}</td>"
                            f"<td style='color:#b8b8b8'>{row['effect']}</td>"
                            f"<td style='color:#8890a1'>{row['direction']}</td>"
                            f"<td style='color:{p_color}'>{p_val:.4f}</td>"
                            f"<td style='color:#8890a1'>{row['best_lag']}</td>"
                            f"</tr>"
                        )
                    html_tbl = (
                        _TBL_CSS
                        + "<table class='ec-table'>"
                        + "<thead><tr>"
                        + "<th>Cause</th><th>Effect</th><th>Direction</th><th>Min P</th><th>Best Lag</th>"
                        + "</tr></thead><tbody>"
                        + rows_html
                        + "</tbody></table>"
                    )
                    st.markdown(html_tbl, unsafe_allow_html=True)

    # ── Panel 2: Transfer Entropy ──────────────────────────────────────────
    _thread(
        "Granger causality tells you whether a predictive relationship exists. Transfer entropy "
        "below goes further - it measures how much information is actually flowing in each "
        "direction, filtering out spurious correlations and isolating the true signal channel."
    )
    with col_te:
        _label("Transfer Entropy: Net Information Flow")
        with st.spinner("Computing transfer entropy + shuffle significance…"):
            te_c2e, te_e2c, pval_c2e, pval_e2c = transfer_entropy_matrix(
                eq_r[sel_eq], cmd_r[sel_cmd]
            )
            net_te = net_flow_matrix(te_c2e, te_e2c)

        if net_te.empty:
            st.warning("Transfer entropy computation failed.")
        else:
            # Overlay significance: hatched or annotated for p < 0.05
            # Use net-direction p-value: c2e where net > 0, e2c where net < 0
            sig_mask = (net_te > 0) & (pval_c2e < 0.05) | (net_te <= 0) & (pval_e2c < 0.05)
            text_ann = net_te.round(3).astype(str)
            for r in net_te.index:
                for c in net_te.columns:
                    if sig_mask.loc[r, c]:
                        text_ann.loc[r, c] = text_ann.loc[r, c] + "*"
            fig_te = go.Figure(go.Heatmap(
                z=net_te.values.astype(float),
                x=net_te.columns.tolist(),
                y=net_te.index.tolist(),
                colorscale=[[0,"#c0392b"],[0.5,"#1e1e1e"],[1,"#2e7d32"]],
                zmid=0,
                text=text_ann.values,
                texttemplate="%{text}",
                textfont=dict(size=8, family="JetBrains Mono, monospace", color="#e8e9ed"),
                colorbar=dict(title="Net TE", thickness=10),
            ))
            fig_te.update_layout(
                template="purdue", height=320,
                paper_bgcolor="#000", plot_bgcolor="#080808",
                font=dict(color="#e8e9ed"),
                xaxis=dict(tickangle=-35, tickfont=dict(size=8, color="#8890a1"), rangeslider=dict(visible=False)),
                yaxis=dict(tickfont=dict(size=8, color="#8890a1")),
                margin=dict(l=100, r=40, t=20, b=90),
            )
            _chart(fig_te)
            _panel_note("Green = commodity leads equity. Red = equity leads commodity. * = p < 0.05 (shuffle test, Schreiber 2000).")
            _insight_note(
                "Transfer entropy measures the net direction of information flow between two assets - "
                "which one is telling the story, and which one is listening. Green cells mean the "
                "commodity is driving the equity. Red cells mean the equity is driving the commodity. "
                "Asterisked cells pass the shuffle significance test (p < 0.05): the observed TE "
                "exceeds 95% of null-distribution TEs from 200 random source permutations."
            )

    st.markdown('<div style="margin:0.5rem 0;border-top:1px solid #2a2a2a"></div>',
                unsafe_allow_html=True)
    _thread(
        "Transfer entropy identifies direction; Diebold-Yilmaz measures magnitude. The FEVD "
        "table below shows what fraction of each asset's price variance is explained by shocks "
        "originating elsewhere - the higher the number, the more that asset is a price-taker "
        "rather than a price-setter."
    )

    # ══════════════════════════════════════════════════════════════════════
    # ROW 2: Diebold-Yilmaz FEVD (wider) | Network graph (narrower)
    # ══════════════════════════════════════════════════════════════════════
    col_dy, col_net = st.columns([1.1, 1], gap="medium")

    dy_valid = [c for c in sel_all if c in all_r.columns]
    dy_result, dy_table, total_sp, G_dy = None, pd.DataFrame(), 0.0, None
    dy_from = dy_to = dy_net = pd.Series(dtype=float)
    dy_top_tx = dy_top_rx = dy_dir = ""

    if len(dy_valid) >= 3:
        combined = all_r[dy_valid].dropna()
        with st.spinner("Fitting VAR (Diebold-Yilmaz)…"):
            dy_result = diebold_yilmaz(combined, top_n=len(dy_valid))
        dy_table  = dy_result["spillover_table"]
        total_sp  = dy_result["total_spillover"]
        dy_from   = dy_result["from_spillover"]
        dy_to     = dy_result["to_spillover"]
        dy_net    = dy_result["net_spillover"]
        dy_top_tx = dy_result["top_transmitter"]
        dy_top_rx = dy_result["top_receiver"]
        dy_dir    = dy_result["direction_label"]
        if not dy_table.empty:
            G_dy = build_dy_graph(dy_table, threshold=dy_thresh)

    # ── DY Headline banner (full-width, above panels) ─────────────────────
    if not dy_table.empty and np.isfinite(total_sp):
        _sp_level = "HIGH" if total_sp >= 55 else ("MODERATE" if total_sp >= 35 else "LOW")
        _sp_color = "#c0392b" if total_sp >= 55 else ("#e67e22" if total_sp >= 35 else "#27ae60")
        _dir_icon = "→" if "Commodity" in dy_dir else ("←" if "Equity" in dy_dir else "↔")
        st.markdown(
            f'<div style="background:#080808;border:1px solid #1e1e1e;'
            f'border-left:4px solid {_sp_color};padding:.55rem 1rem;margin-bottom:.6rem;'
            f'display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap">'
            f'<div>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
            f'font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:#8E9AAA">'
            f'TOTAL SPILLOVER INDEX</span><br>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:22px;'
            f'font-weight:700;color:{_sp_color};line-height:1.1">{total_sp:.1f}%</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
            f'font-weight:700;color:{_sp_color};margin-left:6px">{_sp_level}</span>'
            f'</div>'
            f'<div style="border-left:1px solid #2a2a2a;padding-left:1.2rem">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
            f'color:#8E9AAA;text-transform:uppercase;letter-spacing:.14em">DIRECTION</span><br>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
            f'font-weight:700;color:#CFB991">{_dir_icon} {dy_dir}</span>'
            f'</div>'
            f'<div style="border-left:1px solid #2a2a2a;padding-left:1.2rem">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
            f'color:#8E9AAA;text-transform:uppercase;letter-spacing:.14em">TOP TRANSMITTER</span><br>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
            f'font-weight:700;color:#27ae60">{dy_top_tx} '
            f'<span style="font-size:9px;color:#8E9AAA">'
            f'(+{dy_net.get(dy_top_tx, 0):.1f}%)</span></span>'
            f'</div>'
            f'<div style="border-left:1px solid #2a2a2a;padding-left:1.2rem">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
            f'color:#8E9AAA;text-transform:uppercase;letter-spacing:.14em">TOP RECEIVER</span><br>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
            f'font-weight:700;color:#c0392b">{dy_top_rx} '
            f'<span style="font-size:9px;color:#8E9AAA">'
            f'({dy_net.get(dy_top_rx, 0):.1f}%)</span></span>'
            f'</div>'
            f'<div style="margin-left:auto;font-family:\'DM Sans\',sans-serif;'
            f'font-size:9px;color:#555960;text-align:right">'
            f'Diebold-Yilmaz (2012) FEVD<br>VAR({4}) · H={10} · {len(dy_valid)} assets</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Panel 3: Diebold-Yilmaz FEVD + FROM/TO/NET ────────────────────────
    with col_dy:
        _label("Diebold-Yilmaz: Forecast Error Variance Decomposition")
        if dy_table.empty:
            st.warning("Select ≥ 3 VAR assets.")
        else:
            # FROM / TO / NET compact bar chart
            if not dy_net.empty:
                _colors = ["#27ae60" if v >= 0 else "#c0392b" for v in dy_net.values]
                fig_net_bar = go.Figure(go.Bar(
                    x=dy_net.index.tolist(),
                    y=dy_net.values.tolist(),
                    marker_color=_colors,
                    text=[f"{v:+.1f}%" for v in dy_net.values],
                    textposition="outside",
                    textfont=dict(size=8, color="#e8e9ed", family="JetBrains Mono, monospace"),
                ))
                fig_net_bar.update_layout(
                    template="purdue", height=180,
                    paper_bgcolor="#080808", plot_bgcolor="#080808",
                    font=dict(color="#e8e9ed"),
                    xaxis=dict(tickfont=dict(size=8, color="#8890a1"), tickangle=-25),
                    yaxis=dict(tickfont=dict(size=8, color="#8890a1"),
                               title=dict(text="NET (TO−FROM %)", font=dict(size=8))),
                    margin=dict(l=55, r=20, t=10, b=60),
                    showlegend=False,
                )
                _chart(fig_net_bar)
                _panel_note(
                    "Green = net transmitter (exports shocks). Red = net receiver (absorbs shocks). "
                    f"Current leader: <b>{dy_top_tx}</b>."
                )

            # Raw FEVD heatmap in expander to keep page clean
            with st.expander("View raw FEVD table"):
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
                    paper_bgcolor="#000", plot_bgcolor="#080808",
                    font=dict(color="#e8e9ed"),
                    xaxis=dict(tickangle=-35, tickfont=dict(size=8, color="#8890a1"),
                               rangeslider=dict(visible=False)),
                    yaxis=dict(tickfont=dict(size=8, color="#8890a1")),
                    margin=dict(l=100, r=40, t=20, b=90),
                )
                _chart(fig_dy)
            _insight_note(
                "Each cell shows what percentage of one asset's price uncertainty (forecast error) "
                "is caused by shocks originating in another asset. The diagonal is self-driven. "
                "A high total spillover index (above 50%) means markets are deeply interconnected — "
                "a shock anywhere in the system propagates everywhere quickly."
            )

    # ── Panel 4: Spillover Network ─────────────────────────────────────────
    _thread(
        "The three methods above each illuminate a different facet. The network graph synthesises "
        "them: transmitters (thick outgoing edges, large nodes) are the price-setters you need "
        "to monitor; receivers are the assets most vulnerable when those price-setters move."
    )
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

    # ══════════════════════════════════════════════════════════════════════
    # SECTION: Rolling Spillover + Regime-Conditional
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div style="margin:0.8rem 0;border-top:1px solid #2a2a2a"></div>',
                unsafe_allow_html=True)
    st.markdown(
        '<h2 style="font-family:\'DM Sans\',sans-serif;font-size:1.0rem;font-weight:700;'
        'color:#CFB991;margin-bottom:0.1rem">Spillover Over Time &amp; Regime Context</h2>'
        '<p style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;color:#8890a1;margin:0 0 0.5rem">'
        'Rolling 200-day DY spillover index shows how interconnectedness evolves. '
        'Regime-conditional table answers: <em>in the current regime, what is the expected spillover?</em></p>',
        unsafe_allow_html=True,
    )

    col_roll, col_regime = st.columns([1.4, 1], gap="medium")

    with col_roll:
        _label("Rolling Diebold-Yilmaz Spillover Index (200-day window)")
        if len(dy_valid) >= 3:
            with st.spinner("Computing rolling DY spillover…"):
                roll_dy = rolling_diebold_yilmaz(
                    all_r[dy_valid].dropna(), window=200, step=5,
                    lag_order=2, horizon=10,
                )
            if not roll_dy.empty and "total_spillover" in roll_dy.columns:
                fig_roll = go.Figure()
                fig_roll.add_trace(go.Scatter(
                    x=roll_dy.index, y=roll_dy["total_spillover"],
                    mode="lines", line=dict(color="#CFB991", width=1.5),
                    fill="tozeroy", fillcolor="rgba(207,185,145,0.08)",
                    name="Total Spillover %",
                ))
                # Threshold lines
                fig_roll.add_hline(y=55, line_dash="dot", line_color="#c0392b",
                                   annotation_text="High (55%)", annotation_font_size=8)
                fig_roll.add_hline(y=35, line_dash="dot", line_color="#e67e22",
                                   annotation_text="Moderate (35%)", annotation_font_size=8)
                fig_roll.update_layout(
                    template="purdue", height=280,
                    paper_bgcolor="#080808", plot_bgcolor="#080808",
                    font=dict(color="#e8e9ed"),
                    xaxis=dict(tickfont=dict(size=8, color="#8890a1"),
                               rangeslider=dict(visible=False)),
                    yaxis=dict(tickfont=dict(size=8, color="#8890a1"),
                               title=dict(text="Spillover %", font=dict(size=8))),
                    margin=dict(l=55, r=20, t=15, b=40),
                    showlegend=False,
                )
                _chart(fig_roll)
                _panel_note(
                    "Peaks correspond to known systemic events (GFC 2008, COVID 2020, Ukraine 2022). "
                    "High spillover = shock propagates everywhere; low = markets are segmented."
                )
            else:
                st.info("Insufficient data for rolling DY computation.")
        else:
            st.info("Select ≥ 3 VAR assets to compute rolling spillover.")

    with col_regime:
        _label("Regime-Conditional Spillover")
        if len(dy_valid) >= 3:
            try:
                from src.analysis.correlations import (
                    average_cross_corr_series, detect_correlation_regime,
                )
                _rc_eq  = [c for c in dy_valid if c in eq_r.columns]
                _rc_cmd = [c for c in dy_valid if c in cmd_r.columns]
                if _rc_eq and _rc_cmd:
                    with st.spinner("Computing regime-conditional spillover…"):
                        _avg_c = average_cross_corr_series(
                            eq_r[_rc_eq], cmd_r[_rc_cmd], window=60,
                        )
                        _regimes = detect_correlation_regime(_avg_c)
                        _rc_result = regime_conditional_spillover(
                            all_r[dy_valid].dropna(), _regimes,
                            lag_order=3, horizon=10,
                        )
                    _cur_regime = int(_regimes.dropna().iloc[-1]) if not _regimes.dropna().empty else 1
                    _RNAMES = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis"}
                    _RCOLORS = {0: "#27ae60", 1: "#CFB991", 2: "#e67e22", 3: "#c0392b"}

                    for _rid in range(4):
                        _rdata  = _rc_result.get(_rid, {})
                        _rsp    = _rdata.get("total_spillover", np.nan)
                        _rn     = _rdata.get("n_obs", 0)
                        _rtx    = _rdata.get("top_transmitter", "—")
                        _rcolor = _RCOLORS[_rid]
                        _is_cur = _rid == _cur_regime
                        _border = f"border-left:3px solid {_rcolor}" if _is_cur else f"border-left:1px solid #2a2a2a"
                        _bg     = "#0d0d0d" if _is_cur else "#080808"
                        _label_sfx = " ◀ CURRENT" if _is_cur else ""
                        st.markdown(
                            f'<div style="background:{_bg};{_border};'
                            f'padding:.4rem .7rem;margin-bottom:.3rem">'
                            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
                            f'font-weight:700;letter-spacing:.14em;text-transform:uppercase;'
                            f'color:{_rcolor}">{_RNAMES[_rid]}{_label_sfx}</span>'
                            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:18px;'
                            f'font-weight:700;color:{_rcolor if _is_cur else "#c8c8c8"};'
                            f'display:block;line-height:1.2">'
                            f'{"—" if not np.isfinite(_rsp) else f"{_rsp:.1f}%"}</span>'
                            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:9px;'
                            f'color:#8E9AAA">top tx: {_rtx} · {_rn} obs</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    _panel_note(
                        "Expected spillover in each historical regime. Current regime highlighted. "
                        "Crisis regime spillover substantially exceeds normal — use to set hedge sizing."
                    )
                else:
                    st.info("Need equity + commodity assets selected for regime detection.")
            except Exception as _rc_e:
                st.caption(f"Regime-conditional computation unavailable: {type(_rc_e).__name__}")
        else:
            st.info("Select ≥ 3 VAR assets.")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION: Granger Causality Network (GAP 24)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div style="margin:0.8rem 0;border-top:1px solid #2a2a2a"></div>',
                unsafe_allow_html=True)
    st.markdown(
        '<h2 style="font-family:\'DM Sans\',sans-serif;font-size:1.0rem;font-weight:700;'
        'color:#CFB991;margin-bottom:0.1rem">Granger Causality Network — Full Pair Analysis</h2>'
        '<p style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;color:#8890a1;margin:0 0 0.5rem">'
        'Directed network across all equity-commodity pairs (p &lt; 0.05). '
        'Node size = number of significant outgoing Granger links. '
        'Hub commodities (many significant outlinks into equities) are the dominant price-setters.</p>',
        unsafe_allow_html=True,
    )

    if granger_df.empty:
        st.info("Run the Granger analysis (select assets above) to see the causality network.")
    else:
        _sig = granger_df[granger_df["significant"] == True]
        _n_sig = len(_sig)
        _n_cmd_to_eq = len(_sig[_sig["direction"] == "Commodity → Equity"])
        _n_eq_to_cmd = len(_sig[_sig["direction"] == "Equity → Commodity"])

        # Hub commodity: most significant outgoing Granger links toward equities
        if _n_cmd_to_eq > 0:
            _hub_counts = _sig[_sig["direction"] == "Commodity → Equity"]["cause"].value_counts()
            _hub_cmd = _hub_counts.index[0] if len(_hub_counts) > 0 else "—"
            _hub_n   = int(_hub_counts.iloc[0]) if len(_hub_counts) > 0 else 0
            _hub_eq_counts = _sig[_sig["direction"] == "Equity → Commodity"]["cause"].value_counts()
            _hub_eq  = _hub_eq_counts.index[0] if len(_hub_eq_counts) > 0 else "—"
            _hub_eq_n = int(_hub_eq_counts.iloc[0]) if len(_hub_eq_counts) > 0 else 0

            # Hub summary banner
            st.markdown(
                f'<div style="background:#080808;border-left:3px solid #CFB991;'
                f'border-radius:4px;padding:8px 14px;margin:6px 0;display:flex;gap:40px;">'
                f'<div><span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
                f'font-weight:700;letter-spacing:.1em;color:#8E9AAA;text-transform:uppercase">'
                f'Hub Commodity</span><br>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:16px;'
                f'font-weight:700;color:#CFB991">{_hub_cmd}</span>'
                f'<span style="font-family:\'DM Sans\',sans-serif;font-size:10px;color:#8E9AAA">'
                f' → {_hub_n} equity markets</span></div>'
                f'<div><span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
                f'font-weight:700;letter-spacing:.1em;color:#8E9AAA;text-transform:uppercase">'
                f'Hub Equity</span><br>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:16px;'
                f'font-weight:700;color:#e67e22">{_hub_eq}</span>'
                f'<span style="font-family:\'DM Sans\',sans-serif;font-size:10px;color:#8E9AAA">'
                f' → {_hub_eq_n} commodities</span></div>'
                f'<div><span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
                f'font-weight:700;letter-spacing:.1em;color:#8E9AAA;text-transform:uppercase">'
                f'Significant Links</span><br>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:16px;'
                f'font-weight:700;color:#c8c8c8">{_n_sig}</span>'
                f'<span style="font-family:\'DM Sans\',sans-serif;font-size:10px;color:#8E9AAA">'
                f' · cmd→eq {_n_cmd_to_eq} · eq→cmd {_n_eq_to_cmd}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Full Granger network graph (promoted from expander)
        _col_gr_net, _col_gr_hub = st.columns([1.8, 1], gap="medium")
        with _col_gr_net:
            try:
                G_gr_full = build_granger_graph(granger_df)
                if G_gr_full.number_of_nodes() > 0:
                    _chart(plot_granger_network(
                        G_gr_full,
                        title=f"Granger Causality Network · {_n_sig} significant pairs (p < 0.05)",
                        layout=layout,
                    ))
                    _panel_note(
                        "Red arrows = commodity Granger-causes equity. "
                        "Blue arrows = equity Granger-causes commodity. "
                        "Node size ∝ number of significant outgoing links (hub = large node)."
                    )
            except Exception as _gr_err:
                st.caption(f"Network render unavailable: {type(_gr_err).__name__}")

        with _col_gr_hub:
            _label("Top Granger Transmitters")
            if not granger_df.empty:
                _cause_counts = (
                    granger_df[granger_df["significant"] == True]
                    .groupby("cause").size()
                    .sort_values(ascending=False)
                    .head(10)
                )
                if not _cause_counts.empty:
                    _hub_fig = go.Figure(go.Bar(
                        x=_cause_counts.values,
                        y=_cause_counts.index.tolist(),
                        orientation="h",
                        marker=dict(
                            color=["#d35400" if c in cmd_r.columns else "#2980b9"
                                   for c in _cause_counts.index],
                        ),
                        text=_cause_counts.values,
                        textfont=dict(size=8, color="#e8e9ed",
                                      family="JetBrains Mono, monospace"),
                    ))
                    _hub_fig.update_layout(
                        template="purdue", height=280,
                        paper_bgcolor="#000", plot_bgcolor="#080808",
                        font=dict(color="#e8e9ed"),
                        xaxis=dict(title="# significant Granger links",
                                   tickfont=dict(size=8, color="#8890a1")),
                        yaxis=dict(tickfont=dict(size=8, color="#c8c8c8")),
                        margin=dict(l=110, r=20, t=10, b=30),
                    )
                    _chart(_hub_fig)
                    _panel_note("Orange = commodity. Blue = equity. Bar length = number of Granger-causal pairs.")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION: Cross-Asset Spillover: Rates & FX
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div style="margin:0.8rem 0;border-top:1px solid #2a2a2a"></div>',
                unsafe_allow_html=True)
    st.markdown(
        '<h2 style="font-family:\'DM Sans\',sans-serif;font-size:1.0rem;font-weight:700;'
        'color:#CFB991;margin-bottom:0.2rem">Cross-Asset Spillover: Rates &amp; FX</h2>'
        '<p style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;color:#8890a1;margin:0 0 0.5rem">'
        'Targeted Granger causality across the 6 most representative cross-asset signals</p>',
        unsafe_allow_html=True,
    )

    try:
        fi_r = load_fixed_income_returns(start, end)
        fx_r = load_fx_returns(start, end)

        # Build focused universe
        _CA_ASSETS_SOURCES = {
            "S&P 500":                 eq_r,
            "Gold":                    cmd_r,
            "WTI Crude Oil":           cmd_r,
            "US 20Y+ Treasury (TLT)":  fi_r,
            "HY Corporate (HYG)":      fi_r,
            "DXY (Dollar Index)":      fx_r,
        }

        _ca_series = {}
        for _name, _src in _CA_ASSETS_SOURCES.items():
            if not _src.empty and _name in _src.columns:
                _ca_series[_name] = _src[_name]

        if len(_ca_series) >= 4:
            _ca_df = pd.DataFrame(_ca_series).dropna(how="all")

            if len(_ca_df) >= 60:
                with st.spinner("Running cross-asset Granger causality (6-asset universe)…"):
                    _ca_assets = list(_ca_df.columns)
                    _granger_results = {}
                    for _cause in _ca_assets:
                        for _effect in _ca_assets:
                            if _cause == _effect:
                                continue
                            try:
                                from statsmodels.tsa.stattools import grangercausalitytests
                                _data = _ca_df[[_effect, _cause]].dropna()
                                if len(_data) < 30:
                                    continue
                                _gc = grangercausalitytests(_data, maxlag=5, verbose=False)
                                _min_p = min(
                                    _gc[lag][0]["ssr_ftest"][1]
                                    for lag in range(1, 6)
                                    if lag in _gc
                                )
                                _granger_results[(_cause, _effect)] = _min_p
                            except Exception:
                                pass

                if _granger_results:
                    _pivot_data = {}
                    for _effect in _ca_assets:
                        _pivot_data[_effect] = {}
                        for _cause in _ca_assets:
                            if _cause == _effect:
                                _pivot_data[_effect][_cause] = None
                            else:
                                _pivot_data[_effect][_cause] = _granger_results.get((_cause, _effect))

                    _pivot_df = pd.DataFrame(_pivot_data).T

                    # Build HTML table with color-coded p-values
                    _TBL_CSS_CA = """
<style>
.ec-table{width:100%;border-collapse:collapse;font-family:'DM Sans',sans-serif;font-size:0.78rem}
.ec-table th{background:#1c1c1c;color:#CFB991;padding:7px 10px;text-align:left;
    border-bottom:1px solid rgba(207,185,145,0.3);font-weight:600;
    letter-spacing:0.06em;text-transform:uppercase;font-size:0.68rem}
.ec-table td{padding:5px 10px;border-bottom:1px solid #1e1e1e;color:#e8e9ed}
.ec-table tr:nth-child(even) td{background:#171717}
.ec-table tr:nth-child(odd) td{background:#111111}
.ec-table tr:hover td{background:#202020}
</style>"""
                    _header_cells = "<th>Effect \\ Cause</th>" + "".join(
                        f"<th>{c}</th>" for c in _pivot_df.columns
                    )
                    _body_rows = ""
                    for _eff, _row in _pivot_df.iterrows():
                        _cells = f"<td style='color:#b8b8b8;font-weight:600'>{_eff}</td>"
                        for _caus in _pivot_df.columns:
                            _p = _row.get(_caus)
                            if _p is None:
                                _cells += "<td style='color:#8890a1'>-</td>"
                            elif _p < 0.05:
                                _cells += f"<td style='color:#4ade80;font-weight:700'>{_p:.3f}</td>"
                            elif _p < 0.10:
                                _cells += f"<td style='color:#e67e22;font-weight:600'>{_p:.3f}</td>"
                            else:
                                _cells += f"<td style='color:#8890a1'>{_p:.3f}</td>"
                        _body_rows += f"<tr>{_cells}</tr>"

                    _html_ca = (
                        _TBL_CSS_CA
                        + "<table class='ec-table'>"
                        + f"<thead><tr>{_header_cells}</tr></thead>"
                        + f"<tbody>{_body_rows}</tbody>"
                        + "</table>"
                    )
                    st.markdown(_html_ca, unsafe_allow_html=True)
                    st.markdown(
                        f'<p style="font-family:\'DM Sans\',sans-serif;font-size:0.64rem;color:#8890a1;'
                        f'margin-top:6px;line-height:1.6">'
                        f'<span style="color:#4ade80;font-weight:700">Green p&lt;0.05</span> = significant Granger causality at 5%. '
                        f'<span style="color:#e67e22;font-weight:700">Amber p&lt;0.10</span> = marginal signal. '
                        f'<span style="color:#8890a1">Grey</span> = not significant. '
                        f'Row = effect; Column = cause. E.g. row TLT, col S&P 500 = "does S&P 500 Granger-cause TLT?"</p>',
                        unsafe_allow_html=True,
                    )

                    # Net transmitter score
                    _transmitter_scores = {}
                    for _asset in _ca_assets:
                        _sent = sum(
                            1 for (_c, _e), _p in _granger_results.items()
                            if _c == _asset and _p < 0.05
                        )
                        _recv = sum(
                            1 for (_c, _e), _p in _granger_results.items()
                            if _e == _asset and _p < 0.05
                        )
                        _transmitter_scores[_asset] = _sent - _recv

                    _primary_tx = max(_transmitter_scores, key=_transmitter_scores.get)
                    _primary_score = _transmitter_scores[_primary_tx]
                    _tx_color = "#4ade80" if _primary_score > 0 else "#f87171" if _primary_score < 0 else "#8890a1"
                    st.markdown(
                        f'<div style="border:1px solid #2a2a2a;'
                        f'border-radius:0;padding:0.65rem 1rem;background:#1c1c1c;margin-top:0.6rem">'
                        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.56rem;font-weight:700;'
                        f'text-transform:uppercase;letter-spacing:0.12em;color:{_tx_color};margin-bottom:3px">'
                        f'Primary Cross-Asset Transmitter</div>'
                        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.84rem;font-weight:700;'
                        f'color:#e8e9ed">{_primary_tx} '
                        f'<span style="font-size:0.70rem;color:#8890a1">'
                        f'(net score: {_primary_score:+d} significant Granger links)</span></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Insufficient data for cross-asset Granger analysis. Extend the date range.")
        else:
            st.info("Fixed income or FX data unavailable. Cross-asset spillover requires internet connectivity.")
    except Exception as _e:
        pass  # Cross-asset spillover section unavailable — skip silently

    # ══════════════════════════════════════════════════════════════════════
    # SECTION: Commodity Futures Curve — Backwardation / Contango (GAP 3/25)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div style="margin:0.8rem 0;border-top:1px solid #2a2a2a"></div>',
                unsafe_allow_html=True)
    st.markdown(
        '<h2 style="font-family:\'DM Sans\',sans-serif;font-size:1.0rem;font-weight:700;'
        'color:#CFB991;margin-bottom:0.1rem">Commodity Futures Curve — Backwardation / Contango</h2>'
        '<p style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;color:#8890a1;margin:0 0 0.5rem">'
        'Front-month vs 6-month deferred contracts. '
        'Backwardation (spot &gt; futures) signals near-term supply tightness — corroborates geopolitical risk. '
        'Contango (spot &lt; futures) signals oversupply or weak demand — contradicts a high GRS.</p>',
        unsafe_allow_html=True,
    )

    try:
        from src.analysis.futures_curve import (
            fetch_curve_snapshot, geopolitical_corroboration, fetch_rolling_basis,
        )

        with st.spinner("Fetching futures curve data…"):
            _curve_df = fetch_curve_snapshot()

        if _curve_df.empty:
            st.info(
                "Futures curve data unavailable. Deferred contract tickers (e.g., CLQ25.NYM) "
                "may not be available on yfinance for the current date. "
                "Spot prices are used for all other analyses."
            )
        else:
            # GRS corroboration check
            try:
                _grs_val = float(st.session_state.get("_stored_geo_score", 50.0))
            except Exception:
                _grs_val = 50.0

            _corr = geopolitical_corroboration(_curve_df, _grs_val)
            _corr_colors = {
                "Corroborated": "#27ae60",
                "Contradicted": "#e74c3c",
                "Inconclusive": "#CFB991",
            }
            _corr_color = _corr_colors.get(_corr["overall_signal"], "#8890a1")
            st.markdown(
                f'<div style="background:#080808;border-left:3px solid {_corr_color};'
                f'border-radius:4px;padding:10px 14px;margin:6px 0;">'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
                f'font-weight:700;color:{_corr_color};letter-spacing:.08em;">'
                f'MARKET STRUCTURE: {_corr["overall_signal"].upper()} · GRS={_grs_val:.0f}</span><br>'
                f'<span style="font-family:\'DM Sans\',sans-serif;font-size:11px;color:#b0b0b0;">'
                f'{_corr["detail"]}</span></div>',
                unsafe_allow_html=True,
            )

            # Curve snapshot cards
            _n_cols = min(len(_curve_df), 3)
            _curve_cols = st.columns(_n_cols)
            for _ci, (_crow_idx, _crow) in enumerate(_curve_df.iterrows()):
                if _ci >= _n_cols:
                    break
                with _curve_cols[_ci]:
                    _bp = float(_crow["basis_pct"])
                    _sc = _crow["structure_color"]
                    _struct = _crow["structure"]
                    _arrow = "▼" if _bp < 0 else "▲"
                    st.markdown(
                        f'<div style="background:#080808;border:1px solid #1e1e1e;'
                        f'border-top:2px solid {_sc};border-radius:4px;padding:10px 12px;">'
                        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
                        f'font-weight:700;letter-spacing:.1em;color:#8E9AAA;'
                        f'text-transform:uppercase">{_crow["name"]}</span><br>'
                        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:18px;'
                        f'font-weight:700;color:{_sc}">{_struct}</span><br>'
                        f'<span style="font-family:\'DM Sans\',sans-serif;font-size:11px;'
                        f'color:#8E9AAA">Basis {_arrow} {abs(_bp):.2f}%'
                        f'<br>Front: {_crow["front_price"]} {_crow["unit"]}'
                        f'<br>6M: {_crow["deferred_price"]} {_crow["unit"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Second row of cards
            if len(_curve_df) > 3:
                _curve_cols2 = st.columns(min(len(_curve_df) - 3, 3))
                for _ci2, (_crow_idx, _crow) in enumerate(_curve_df.iloc[3:].iterrows()):
                    if _ci2 >= 3:
                        break
                    with _curve_cols2[_ci2]:
                        _bp = float(_crow["basis_pct"])
                        _sc = _crow["structure_color"]
                        _struct = _crow["structure"]
                        _arrow = "▼" if _bp < 0 else "▲"
                        st.markdown(
                            f'<div style="background:#080808;border:1px solid #1e1e1e;'
                            f'border-top:2px solid {_sc};border-radius:4px;padding:10px 12px;">'
                            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
                            f'font-weight:700;letter-spacing:.1em;color:#8E9AAA;'
                            f'text-transform:uppercase">{_crow["name"]}</span><br>'
                            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:18px;'
                            f'font-weight:700;color:{_sc}">{_struct}</span><br>'
                            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:11px;'
                            f'color:#8E9AAA">Basis {_arrow} {abs(_bp):.2f}%'
                            f'<br>Front: {_crow["front_price"]} {_crow["unit"]}'
                            f'<br>6M: {_crow["deferred_price"]} {_crow["unit"]}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            _panel_note(
                "Basis = (6M deferred − front-month) / front-month × 100%. "
                "Backwardation means market pays a premium for near-term delivery — "
                "supply tightness or demand spike. Contango means storage cost is priced in — "
                "supply is adequate. Source: yfinance (CME/ICE futures)."
            )

            # Optional: rolling basis chart for selected commodity
            _fc_cmd = st.selectbox(
                "View rolling basis for",
                options=_curve_df["name"].tolist(),
                index=0,
                key="sp_futures_curve_sel",
            )
            if _fc_cmd:
                with st.spinner(f"Fetching 6-month rolling basis for {_fc_cmd}…"):
                    _rb = fetch_rolling_basis(_fc_cmd, period="6mo")
                if not _rb.empty and "basis_pct" in _rb.columns:
                    _fc_color = _curve_df[_curve_df["name"] == _fc_cmd]["color"].values[0] if len(
                        _curve_df[_curve_df["name"] == _fc_cmd]) > 0 else "#CFB991"
                    _fc_fig = go.Figure()
                    _fc_fig.add_trace(go.Scatter(
                        x=_rb.index, y=_rb["basis_pct"],
                        mode="lines", line=dict(color=_fc_color, width=1.5),
                        fill="tozeroy",
                        fillcolor=f"rgba(207,185,145,0.08)",
                        name="Basis %",
                    ))
                    _fc_fig.add_hline(y=0, line_dash="solid", line_color="#555",
                                      annotation_text="Flat", annotation_font_size=8)
                    _fc_fig.update_layout(
                        template="purdue", height=200,
                        paper_bgcolor="#000", plot_bgcolor="#080808",
                        font=dict(color="#e8e9ed"),
                        xaxis=dict(tickfont=dict(size=8, color="#8890a1")),
                        yaxis=dict(title="Basis %", tickfont=dict(size=8, color="#8890a1"),
                                   zeroline=True, zerolinecolor="#555"),
                        margin=dict(l=50, r=20, t=10, b=30),
                    )
                    _chart(_fc_fig)
                    _panel_note(
                        "Above zero = contango (futures > spot). Below zero = backwardation. "
                        "Geopolitical supply shock events typically push energy curves into backwardation."
                    )
                else:
                    st.caption(
                        f"Rolling basis data for {_fc_cmd} unavailable "
                        f"(deferred contract ticker may not be liquid on this date)."
                    )
    except Exception as _fc_err:
        st.caption(f"Futures curve section unavailable: {type(_fc_err).__name__}: {_fc_err}")

    _page_conclusion(
        "Transmission Map",
        "Assets identified as strong transmitters across all three methods - Granger, Transfer "
        "Entropy, and Diebold-Yilmaz - are your first-order risk factors. A price move in a "
        "high-transmitter commodity is not isolated; it will propagate. Use this map to identify "
        "which equity markets to hedge when a key commodity breaks out."
    )

    # ── Conflict Layer: which active conflicts are driving the current spillover ──
    st.markdown(
        f'<div style="margin:1rem 0 0.4rem;border-top:1px solid #1e1e1e;padding-top:0.8rem">'
        f'<p style="{_F}font-size:0.58rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.14em;color:#8E9AAA;margin:0 0 4px">Conflict Transmission Layer</p>'
        f'<p style="{_F}font-size:0.68rem;color:#8890a1;margin:0 0 8px;line-height:1.5">'
        f'Active conflicts driving current spillover pressure. '
        f'CIS × TPS per-conflict contribution to cross-asset transmission.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    try:
        from src.analysis.conflict_model import aggregate_portfolio_scores, score_all_conflicts
        _sc = score_all_conflicts()
        _sa = aggregate_portfolio_scores(_sc)
        _conf_detail = _sa.get("conflict_detail", {})
        if _conf_detail:
            _cl_sorted = sorted(_conf_detail.items(), key=lambda x: x[1]["tps"], reverse=True)
            _cl_cols = st.columns(min(len(_cl_sorted), 6))
            for _ci2, (_cid2, _cr2) in enumerate(_cl_sorted[:6]):
                with _cl_cols[_ci2]:
                    _cc2 = _cr2.get("color", "#8E9AAA")
                    _cis2 = _cr2["cis"]; _tps2 = _cr2["tps"]
                    _trans_score = round(_cis2 * _tps2 / 100, 1)
                    st.markdown(
                        f'<div style="background:#0d0d0d;border:1px solid #1e1e1e;'
                        f'border-left:2px solid {_cc2};padding:6px 8px;margin-bottom:4px">'
                        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
                        f'font-weight:700;color:{_cc2}">{_cr2.get("label","?")}</span>'
                        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
                        f'font-weight:700;color:#CFB991;margin-top:2px">'
                        f'CIS×TPS {_trans_score:.0f}</div>'
                        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
                        f'color:#555960">CIS {_cis2:.0f} · TPS {_tps2:.0f}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
    except Exception:
        pass

    # CQO runs silently - output visible in About > AI Workforce
    try:
        from src.agents.quality_officer import run as _cqo_run
        from src.analysis.agent_state import is_enabled
        if is_enabled("quality_officer"):
            _anthropic_key = _openai_key = ""
            try:
                _keys = st.secrets.get("keys", {})
                _anthropic_key = _keys.get("anthropic_api_key", "") or ""
                _openai_key    = _keys.get("openai_api_key",    "") or ""
            except Exception:
                pass
            _provider = "anthropic" if _anthropic_key else ("openai" if _openai_key else None)
            _api_key  = _anthropic_key or _openai_key
            _n_obs = len(all_r.dropna(how="all"))
            _cqo_ctx = {
                "n_obs": _n_obs, "date_range": f"{start} to {end}",
                "n_assets": len(sel_all), "sample_warning": _n_obs < 252,
                "model": "VAR + Granger causality + Diebold-Yilmaz FEVD",
                "assumption_count": 4, "lookahead_risk": False,
                "notes": [
                    f"Granger max lag: {max_lag} days (user-selected, not AIC-optimised)",
                    f"DY edge threshold: {dy_thresh}% (arbitrary cutoff affects network topology)",
                    f"VAR assets selected: {len(sel_all)} - degrees of freedom constraint at N<{len(sel_all)*5}",
                    "Transfer entropy estimated non-parametrically - sensitive to bin width choice",
                    "DY FEVD uses 10-step-ahead horizon - results vary materially at 4 vs 10 steps",
                ],
            }
            _cqo_run(_cqo_ctx, _provider, _api_key, page="Spillover Analysis")
    except Exception:
        pass

    _page_footer()
