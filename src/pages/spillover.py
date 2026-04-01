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
)
from src.analysis.network import (
    build_dy_graph, build_granger_graph,
    plot_dy_network, plot_granger_network, plot_net_transmitter_bar,
)
from src.ui.shared import (
    _style_fig, _chart, _page_intro, _thread, _section_note,
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
        f'<p style="{_F}font-size:0.64rem;color:#8890a1;line-height:1.5;margin:4px 0 0 0">{txt}</p>',
        unsafe_allow_html=True,
    )


def page_spillover(start: str, end: str, fred_key: str = "") -> None:
    st.markdown(
        '<h1 style="font-family:\'DM Sans\',sans-serif;font-size:1.25rem;'
        'font-weight:700;margin-bottom:0.1rem">Spillover Analytics</h1>'
        '<p style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;color:#8890a1;'
        'margin:0 0 0.7rem">Granger Causality · Transfer Entropy · Diebold-Yilmaz · Network Graph</p>',
        unsafe_allow_html=True,
    )
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
                    colorscale=[[0,"#c0392b"],[0.05,"#e74c3c"],[0.1,"#1e1e1e"],[1,"#111111"]],
                    zmin=0, zmax=0.1,
                    text=pivot.round(3).values,
                    texttemplate="%{text}",
                    textfont=dict(size=8, family="JetBrains Mono, monospace", color="#e8e9ed"),
                    colorbar=dict(title="p-val", thickness=10),
                ))
                fig_gc.update_layout(
                    template="purdue", height=320,
                    paper_bgcolor="#111111", plot_bgcolor="#111111",
                    font=dict(color="#e8e9ed"),
                    xaxis=dict(tickangle=-35, tickfont=dict(size=8, color="#8890a1"), rangeslider=dict(visible=False)),
                    yaxis=dict(tickfont=dict(size=8, color="#8890a1")),
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
                colorscale=[[0,"#c0392b"],[0.5,"#1e1e1e"],[1,"#2e7d32"]],
                zmid=0,
                text=net_te.round(4).values.astype(float),
                texttemplate="%{text:.3f}",
                textfont=dict(size=8, family="JetBrains Mono, monospace", color="#e8e9ed"),
                colorbar=dict(title="Net TE", thickness=10),
            ))
            fig_te.update_layout(
                template="purdue", height=320,
                paper_bgcolor="#111111", plot_bgcolor="#111111",
                font=dict(color="#e8e9ed"),
                xaxis=dict(tickangle=-35, tickfont=dict(size=8, color="#8890a1"), rangeslider=dict(visible=False)),
                yaxis=dict(tickfont=dict(size=8, color="#8890a1")),
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
                paper_bgcolor="#111111", plot_bgcolor="#111111",
                font=dict(color="#e8e9ed"),
                xaxis=dict(tickangle=-35, tickfont=dict(size=8, color="#8890a1"), rangeslider=dict(visible=False)),
                yaxis=dict(tickfont=dict(size=8, color="#8890a1")),
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
                        f'<div style="border:1px solid #2a2a2a;border-left:4px solid {_tx_color};'
                        f'border-radius:0 6px 6px 0;padding:0.65rem 1rem;background:#1c1c1c;margin-top:0.6rem">'
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
        st.warning(f"Cross-asset spillover section unavailable: {_e}")

    _page_conclusion(
        "Transmission Map",
        "Assets identified as strong transmitters across all three methods - Granger, Transfer "
        "Entropy, and Diebold-Yilmaz - are your first-order risk factors. A price move in a "
        "high-transmitter commodity is not isolated; it will propagate. Use this map to identify "
        "which equity markets to hedge when a key commodity breaks out."
    )

    # ── Chief Quality Officer ─────────────────────────────────────────────────
    try:
        from src.agents.quality_officer import run as _cqo_run
        from src.ui.agent_panel import render_agent_output_block
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
                "n_obs":           _n_obs,
                "date_range":      f"{start} to {end}",
                "n_assets":        len(sel_all),
                "sample_warning":  _n_obs < 252,
                "model":           "VAR + Granger causality + Diebold-Yilmaz FEVD",
                "assumption_count": 4,  # stationarity, linearity, lag selection, normality
                "lookahead_risk":  False,
                "notes": [
                    f"Granger max lag: {max_lag} days (user-selected, not AIC-optimised)",
                    f"DY edge threshold: {dy_thresh}% (arbitrary cutoff affects network topology)",
                    f"VAR assets selected: {len(sel_all)} — degrees of freedom constraint at N<{len(sel_all)*5}",
                    "Transfer entropy estimated non-parametrically — sensitive to bin width choice",
                    "DY FEVD uses 10-step-ahead horizon — results vary materially at 4 vs 10 steps",
                ],
            }
            with st.spinner("CQO auditing spillover methodology…"):
                _cqo_result = _cqo_run(_cqo_ctx, _provider, _api_key, page="Spillover Analysis")
            if _cqo_result.get("narrative"):
                st.markdown("---")
                render_agent_output_block("quality_officer", _cqo_result)
    except Exception:
        pass

    _page_footer()
