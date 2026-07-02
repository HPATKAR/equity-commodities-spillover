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
    rolling_diebold_yilmaz, rolling_frequency_connectedness,
    bootstrap_dy_ci, regime_conditional_spillover,
)
from src.analysis.network import (
    build_dy_graph, build_granger_graph,
    plot_dy_network, plot_granger_network, plot_net_transmitter_bar,
)
from src.ui.shared import (
    _style_fig, _chart, _page_intro, _thread, _section_note,
    _definition_block, _takeaway_block, _page_conclusion, _page_header, _page_footer,
    _insight_note, _no_api_key_banner,
    _label, _panel_note, _F, EC_TABLE_CSS,
)

_DEFAULT_EQ  = ["S&P 500", "DAX", "Nikkei 225", "Shanghai Comp", "Sensex"]
_DEFAULT_CMD = ["WTI Crude Oil", "Gold", "Wheat", "Copper", "Natural Gas"]
_DEFAULT_ALL = _DEFAULT_EQ + _DEFAULT_CMD


def page_spillover(start: str, end: str, fred_key: str = "") -> None:
    _page_header("Spillover Network",
                 "Step 4a of 7 · Direction Test · Granger Causality · Transfer Entropy · Diebold-Yilmaz · Network Graph")
    _no_api_key_banner("AI spillover interpretation")
    _page_intro(
        "<strong>Research question for this page: which market is statistically leading the other — "
        "are equity returns Granger-preceding commodity returns, or the reverse?</strong> "
        "Correlation (on the previous page) tells you <em>that</em> two markets move together. "
        "Spillover analysis tests <em>which direction the statistical lead runs</em>. "
        "Granger causality tests whether past equity returns statistically precede future commodity returns. "
        "Transfer entropy measures directional information flow without assuming linearity. "
        "Diebold-Yilmaz decomposes forecast error variance to assign a net transmitter or receiver "
        "score to every asset. The dominant direction here — equity-led or commodity-led — is the "
        "key input to the regime classification on the Overview page."
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
                f'font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
                f'padding:2px 6px;margin-right:5px;border:1px solid #27ae60;opacity:.8">'
                f'{ch.replace("_"," ").upper()}</span>'
                for ch, _ in _top_channels
            )
            st.markdown(
                f'<div style="background:#080808;border:1px solid #1e1e1e;'
                f'border-left:3px solid {_sp_color};padding:.4rem .9rem;'
                f'margin-bottom:.6rem;display:flex;align-items:center;gap:10px;flex-wrap:wrap">'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
                f'font-weight:700;color:{_sp_color};white-space:nowrap">ACTIVE TRANSMISSION CHANNELS</span>'
                f'{_ch_tags}'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
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
    # Pre-compute all three analyses in parallel (each is @st.cache_data,
    # pure-numpy/statsmodels — no st.* calls inside, safe to thread).
    # Cold first-visit goes from ~17 s sequential → ~7 s parallel.
    # ══════════════════════════════════════════════════════════════════════
    dy_valid = [c for c in sel_all if c in all_r.columns]
    _dy_input = all_r[dy_valid].dropna() if len(dy_valid) >= 3 else None

    from concurrent.futures import ThreadPoolExecutor
    with st.spinner("Running Granger causality, transfer entropy, and Diebold-Yilmaz in parallel…"):
        with ThreadPoolExecutor(max_workers=3) as _sp_pool:
            _f_gc = _sp_pool.submit(granger_grid, eq_r[sel_eq], cmd_r[sel_cmd], max_lag=max_lag)
            _f_te = _sp_pool.submit(transfer_entropy_matrix, eq_r[sel_eq], cmd_r[sel_cmd])
            _f_dy = (_sp_pool.submit(diebold_yilmaz, _dy_input, top_n=len(dy_valid))
                     if _dy_input is not None else None)
            granger_df     = _f_gc.result()
            _te_raw        = _f_te.result()
            _dy_result_raw = _f_dy.result() if _f_dy is not None else None

    te_c2e, te_e2c, pval_c2e, pval_e2c = _te_raw
    net_te = net_flow_matrix(te_c2e, te_e2c)

    # ══════════════════════════════════════════════════════════════════════
    # ROW 1: Granger Causality (wider) | Transfer Entropy (narrower)
    # ══════════════════════════════════════════════════════════════════════
    col_gc, col_te = st.columns([1.2, 1], gap="medium")

    # ── Panel 1: Granger Causality ─────────────────────────────────────────
    with col_gc:
        _label("Granger Causality: Commodity → Equity p-values")

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
                    "Red cells indicate the column commodity statistically Granger-precedes the row "
                    "equity — meaning its past price moves are associated with subsequent equity moves "
                    "in this sample. Commodity price changes have historically preceded equity price "
                    "changes in these pairs, suggesting a 1–3 day predictive association (not established causation)."
                )
            else:
                _panel_note("No significant Commodity → Equity Granger links with current selection.")

            # Top pairs table in expander
            if not sig_df.empty:
                with st.expander(f"Top significant pairs ({len(sig_df)})"):
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
                        EC_TABLE_CSS
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
        "Granger causality tests whether a linear predictive relationship exists. Transfer entropy "
        "below complements it — it measures directional information flow without assuming linearity, "
        "capturing asymmetric associations that linear tests may miss."
    )
    with col_te:
        _label("Transfer Entropy: Net Information Flow")

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

    dy_result, dy_table, total_sp, G_dy = None, pd.DataFrame(), 0.0, None
    dy_from = dy_to = dy_net = pd.Series(dtype=float)
    dy_top_tx = dy_top_rx = dy_dir = ""

    if _dy_result_raw is not None:
        dy_result = _dy_result_raw
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

    # ── Stationarity warning badge ────────────────────────────────────────
    _ns_assets = dy_result.get("non_stationary_assets", []) if dy_result else []
    if _ns_assets:
        _ns_list = ", ".join(_ns_assets)
        st.markdown(
            f'<div style="background:#1a1000;border:1px solid #5a3e00;'
            f'border-left:3px solid #e67e22;padding:.45rem .85rem;'
            f'margin-bottom:.5rem;font-family:\'JetBrains Mono\',monospace;font-size:0.63rem">'
            f'<span style="color:#e67e22;font-weight:700">⚠ STATIONARITY WARNING</span>'
            f'<span style="color:#8E9AAA;margin-left:.6rem">'
            f'ADF test did not reject unit root for: '
            f'<span style="color:#CFB991">{_ns_list}</span> — '
            f'VAR inference on I(1) series without differencing may be unreliable.'
            f'</span></div>',
            unsafe_allow_html=True,
        )

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
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
            f'font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:#8E9AAA">'
            f'TOTAL SPILLOVER INDEX</span><br>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:22px;'
            f'font-weight:700;color:{_sp_color};line-height:1.1">{total_sp:.1f}%</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.56rem;'
            f'font-weight:700;color:{_sp_color};margin-left:6px">{_sp_level}</span>'
            f'</div>'
            f'<div style="border-left:1px solid #2a2a2a;padding-left:1.2rem">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
            f'color:#8E9AAA;text-transform:uppercase;letter-spacing:.14em">DIRECTION</span><br>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.69rem;'
            f'font-weight:700;color:#CFB991">{_dir_icon} {dy_dir}</span>'
            f'</div>'
            f'<div style="border-left:1px solid #2a2a2a;padding-left:1.2rem">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
            f'color:#8E9AAA;text-transform:uppercase;letter-spacing:.14em">TOP TRANSMITTER</span><br>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.69rem;'
            f'font-weight:700;color:#27ae60">{dy_top_tx} '
            f'<span style="font-size:0.56rem;color:#8E9AAA">'
            f'(+{dy_net.get(dy_top_tx, 0):.1f}%)</span></span>'
            f'</div>'
            f'<div style="border-left:1px solid #2a2a2a;padding-left:1.2rem">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
            f'color:#8E9AAA;text-transform:uppercase;letter-spacing:.14em">TOP RECEIVER</span><br>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.69rem;'
            f'font-weight:700;color:#c0392b">{dy_top_rx} '
            f'<span style="font-size:0.56rem;color:#8E9AAA">'
            f'({dy_net.get(dy_top_rx, 0):.1f}%)</span></span>'
            f'</div>'
            f'<div style="margin-left:auto;font-family:\'DM Sans\',sans-serif;'
            f'font-size:0.56rem;color:#555960;text-align:right">'
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

            # ── Baruník-Křehlík frequency breakdown ─────────────────────
            _bk_bands = (dy_result.get("bk_bands", {}) if dy_result else {})
            if _bk_bands:
                with st.expander("Baruník-Křehlík Frequency Connectedness (2018)"):
                    b_short  = _bk_bands.get("short",  {})
                    b_medium = _bk_bands.get("medium", {})
                    b_long   = _bk_bands.get("long",   {})
                    tc_s = b_short.get("total_connectedness",  float("nan"))
                    tc_m = b_medium.get("total_connectedness", float("nan"))
                    tc_l = b_long.get("total_connectedness",   float("nan"))

                    # KPI row: total connectedness per band
                    _kc1, _kc2, _kc3 = st.columns(3)
                    for _col, _label_txt, _tc, _period, _color in [
                        (_kc1, "SHORT  (1–5d)",  tc_s, "ω ∈ [2π/5, π]",    "#c0392b"),
                        (_kc2, "MEDIUM (5–22d)", tc_m, "ω ∈ [2π/22, 2π/5]", "#e67e22"),
                        (_kc3, "LONG   (22d+)",  tc_l, "ω ∈ [0, 2π/22]",   "#CFB991"),
                    ]:
                        _col.markdown(
                            f'<div style="background:#0d0d0d;border:1px solid #2a2a2a;'
                            f'border-top:2px solid {_color};padding:.5rem .7rem;text-align:center">'
                            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
                            f'color:#8E9AAA;text-transform:uppercase;letter-spacing:.14em">{_label_txt}</div>'
                            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:20px;'
                            f'font-weight:700;color:{_color}">'
                            f'{"—" if not isinstance(_tc, float) or np.isnan(_tc) else f"{_tc:.1f}%"}</div>'
                            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
                            f'color:#555960">{_period}</div></div>',
                            unsafe_allow_html=True,
                        )

                    # Per-asset NET bar chart for each band
                    _net_rows = []
                    for _band_key, _band_data in [
                        ("Short",  b_short),
                        ("Medium", b_medium),
                        ("Long",   b_long),
                    ]:
                        _net_s: pd.Series = _band_data.get("net_connectedness", pd.Series(dtype=float))
                        if not _net_s.empty:
                            _net_rows.append((_band_key, _net_s))

                    if _net_rows:
                        st.markdown(
                            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
                            'color:#8E9AAA;margin:.6rem 0 .2rem">NET within-band connectedness '
                            '(TO − FROM). Green = transmitter at that horizon.</p>',
                            unsafe_allow_html=True,
                        )
                        _nb_cols = st.columns(len(_net_rows))
                        _band_colors = {"Short": "#c0392b", "Medium": "#e67e22", "Long": "#CFB991"}
                        for _ci, (_bname, _net_s) in enumerate(_net_rows):
                            _bc = _band_colors.get(_bname, "#CFB991")
                            _bar_colors = ["#27ae60" if v >= 0 else "#c0392b"
                                           for v in _net_s.values]
                            _fig_bk_bar = go.Figure(go.Bar(
                                y=_net_s.index.tolist(),
                                x=_net_s.values.tolist(),
                                orientation="h",
                                marker_color=_bar_colors,
                                text=[f"{v:+.1f}" for v in _net_s.values],
                                textposition="outside",
                                textfont=dict(size=7, color="#e8e9ed",
                                              family="JetBrains Mono, monospace"),
                            ))
                            _fig_bk_bar.update_layout(
                                title=dict(text=f"{_bname}-term", font=dict(size=9, color=_bc),
                                           x=0.5, xanchor="center"),
                                template="purdue", height=max(160, len(_net_s) * 22),
                                paper_bgcolor="#080808", plot_bgcolor="#080808",
                                font=dict(color="#e8e9ed"),
                                xaxis=dict(tickfont=dict(size=7, color="#8890a1"),
                                           title=dict(text="NET %", font=dict(size=7))),
                                yaxis=dict(tickfont=dict(size=7, color="#8890a1")),
                                margin=dict(l=90, r=40, t=28, b=30),
                                showlegend=False,
                            )
                            _nb_cols[_ci].plotly_chart(_fig_bk_bar, use_container_width=True)

                    # Diagnostic: band-sum invariant and D-Y comparison
                    _diag = _bk_bands.get("_diagnostic", {})
                    if _diag:
                        _bk_sum = _diag.get("band_sum",      float("nan"))
                        _bk_ful = _diag.get("full_gfevd_tc", float("nan"))
                        _bk_gap = _diag.get("gap",           float("nan"))
                        _dy_tc  = _diag.get("dy_cholesky_tc", float("nan"))
                        _gap_color = "#c0392b" if _bk_gap > 0.5 else "#27ae60"
                        st.markdown(
                            f'<div style="font-family:\'JetBrains Mono\',monospace;'
                            f'font-size:0.50rem;color:#8E9AAA;margin:.5rem 0 0">'
                            f'Invariant check — '
                            f'Σ bands: <b style="color:#CFB991">{_bk_sum:.2f}%</b> '
                            f'| Full-spectrum GFEVD: <b style="color:#CFB991">{_bk_ful:.2f}%</b> '
                            f'| gap: <b style="color:{_gap_color}">{_bk_gap:.4f}%</b>'
                            f'{"  ⚠ normalisation error" if _bk_gap > 0.5 else "  ✓"}'
                            f'<br>D-Y Cholesky TC: <b style="color:#8890a1">{_dy_tc:.2f}%</b> '
                            f'<span style="color:#555960">(different FEVD method — gap is expected)</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    _panel_note(
                        "Baruník-Křehlík (2018) spectral GFEVD. Short = high-frequency noise "
                        "& HFT. Medium = earnings & macro cycles. Long = structural regime shifts."
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
                # ── Bootstrap CI (cached 24h) ─────────────────────────────
                with st.spinner("Computing bootstrap CI…"):
                    _bci = bootstrap_dy_ci(
                        all_r[dy_valid].dropna(),
                        n_boot=300, lag_order=2, horizon=10,
                    )

                _ts   = roll_dy["total_spillover"]
                _dates = list(roll_dy.index)

                fig_roll = go.Figure()

                # CI ribbon — rendered first so it sits behind the center line
                if _bci and "total_p50" in _bci:
                    _hw_lo = max(0.0, _bci["total_p50"] - _bci["total_p05"])
                    _hw_hi = max(0.0, _bci["total_p95"] - _bci["total_p50"])
                    _ci_lo = (_ts - _hw_lo).clip(lower=0.0)
                    _ci_hi = _ts + _hw_hi
                    # Closed polygon: upper bound → lower bound reversed
                    fig_roll.add_trace(go.Scatter(
                        x=_dates + _dates[::-1],
                        y=list(_ci_hi) + list(_ci_lo)[::-1],
                        fill="toself",
                        fillcolor="rgba(207,185,145,0.20)",
                        line=dict(color="rgba(207,185,145,0.35)", width=0.5),
                        hoverinfo="skip",
                        showlegend=True,
                        name=f"90% CI (n={_bci.get('n_success', 0)}, block={_bci.get('block_len', '?')}d)",
                        legendgroup="ci",
                    ))

                # Center line
                fig_roll.add_trace(go.Scatter(
                    x=_dates, y=list(_ts),
                    mode="lines",
                    line=dict(color="#CFB991", width=1.5),
                    name="Total Spillover %",
                ))

                # Threshold lines
                fig_roll.add_hline(y=55, line_dash="dot", line_color="#c0392b",
                                   annotation_text="High (55%)", annotation_font_size=8)
                fig_roll.add_hline(y=35, line_dash="dot", line_color="#e67e22",
                                   annotation_text="Moderate (35%)", annotation_font_size=8)
                fig_roll.update_layout(
                    template="purdue", height=295,
                    paper_bgcolor="#080808", plot_bgcolor="#080808",
                    font=dict(color="#e8e9ed"),
                    xaxis=dict(tickfont=dict(size=8, color="#8890a1"),
                               rangeslider=dict(visible=False)),
                    yaxis=dict(tickfont=dict(size=8, color="#8890a1"),
                               title=dict(text="Spillover %", font=dict(size=8))),
                    margin=dict(l=55, r=20, t=15, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                font=dict(size=8), bgcolor="rgba(0,0,0,0)"),
                    hovermode="x unified",
                )
                _chart(fig_roll)

                _ci_note = ""
                if _bci and "total_p05" in _bci:
                    _ci_note = (
                        f" · Bootstrap CI: [{_bci['total_p05']:.1f}%, "
                        f"{_bci['total_p95']:.1f}%] "
                        f"(n={_bci['n_success']}, block={_bci['block_len']}d)"
                    )
                _panel_note(
                    "Peaks correspond to known systemic events (GFC 2008, COVID 2020, Ukraine 2022). "
                    "High spillover = shock propagates everywhere; low = markets are segmented."
                    + _ci_note
                )

                # Per-asset NET bootstrap CI (expander)
                if _bci and _bci.get("net_bands"):
                    with st.expander("Per-asset NET spillover bootstrap CI"):
                        _nb  = _bci["net_bands"]
                        _fig_nb = go.Figure()
                        _sorted_assets = sorted(
                            _nb.keys(),
                            key=lambda c: (_nb[c]["p05"] + _nb[c]["p95"]) / 2,
                        )
                        for _ac in _sorted_assets:
                            _p05v = _nb[_ac]["p05"]
                            _p95v = _nb[_ac]["p95"]
                            _mid  = (_p05v + _p95v) / 2
                            _col  = "#27ae60" if _mid >= 0 else "#c0392b"
                            # Error bar trace
                            _fig_nb.add_trace(go.Scatter(
                                x=[_mid], y=[_ac],
                                mode="markers",
                                marker=dict(color=_col, size=7, symbol="circle"),
                                error_x=dict(
                                    type="data", symmetric=False,
                                    array=[_p95v - _mid],
                                    arrayminus=[_mid - _p05v],
                                    color=_col, thickness=1.5, width=5,
                                ),
                                showlegend=False,
                            ))
                        _fig_nb.add_vline(x=0, line=dict(color="#555960", width=1, dash="dot"))
                        _fig_nb.update_layout(
                            template="purdue", height=max(200, len(_nb) * 26),
                            paper_bgcolor="#080808", plot_bgcolor="#080808",
                            font=dict(color="#e8e9ed"),
                            xaxis=dict(tickfont=dict(size=8, color="#8890a1"),
                                       title=dict(text="NET spillover % (TO − FROM)", font=dict(size=8))),
                            yaxis=dict(tickfont=dict(size=8, color="#8890a1")),
                            margin=dict(l=100, r=30, t=15, b=35),
                        )
                        _chart(_fig_nb)
                        _panel_note(
                            "Dot = median NET. Whiskers = 5th/95th bootstrap percentiles. "
                            "Green = net transmitter across all resamples; "
                            "red = net receiver. Wide whiskers = unstable direction."
                        )

                # Rolling BK frequency stacked area chart
                _label("Baruník-Křehlík Frequency Decomposition (Rolling)")
                with st.spinner("Computing rolling frequency connectedness…"):
                    roll_bk = rolling_frequency_connectedness(
                        all_r[dy_valid].dropna(),
                        window=200, step=5, lag_order=2, n_freqs=60,
                    )
                if not roll_bk.empty and {"tc_short", "tc_medium", "tc_long"}.issubset(roll_bk.columns):
                    _fig_bk = go.Figure()
                    # Stack order: long at bottom, short on top (so short = most visible)
                    for _bname, _col, _line_color, _fill_color in [
                        ("Long (22d+)",    "tc_long",   "#CFB991", "rgba(207,185,145,0.55)"),
                        ("Medium (5–22d)", "tc_medium", "#e67e22", "rgba(230,126,34,0.55)"),
                        ("Short (1–5d)",   "tc_short",  "#c0392b", "rgba(192,57,43,0.65)"),
                    ]:
                        _fig_bk.add_trace(go.Scatter(
                            x=roll_bk.index,
                            y=roll_bk[_col],
                            name=_bname,
                            mode="lines",
                            line=dict(color=_line_color, width=0.6),
                            stackgroup="bk",
                            fillcolor=_fill_color,
                        ))
                    _fig_bk.update_layout(
                        template="purdue", height=250,
                        paper_bgcolor="#080808", plot_bgcolor="#080808",
                        font=dict(color="#e8e9ed"),
                        xaxis=dict(tickfont=dict(size=8, color="#8890a1"),
                                   rangeslider=dict(visible=False)),
                        yaxis=dict(tickfont=dict(size=8, color="#8890a1"),
                                   title=dict(text="TC %", font=dict(size=8))),
                        margin=dict(l=55, r=20, t=15, b=40),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    font=dict(size=8), bgcolor="rgba(0,0,0,0)"),
                        hovermode="x unified",
                    )
                    _chart(_fig_bk)
                    _panel_note(
                        "Stacked area = frequency decomposition of total BK connectedness. "
                        "Rising red = short-run contagion (panic). Rising gold = structural integration."
                    )
                else:
                    st.info("Insufficient data for rolling frequency decomposition.")
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
                            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
                            f'font-weight:700;letter-spacing:.14em;text-transform:uppercase;'
                            f'color:{_rcolor}">{_RNAMES[_rid]}{_label_sfx}</span>'
                            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:18px;'
                            f'font-weight:700;color:{_rcolor if _is_cur else "#c8c8c8"};'
                            f'display:block;line-height:1.2">'
                            f'{"—" if not np.isfinite(_rsp) else f"{_rsp:.1f}%"}</span>'
                            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.56rem;'
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
                st.caption("Regime-conditional computation unavailable — see logs.")
        else:
            st.info("Select ≥ 3 VAR assets.")

    # ── Rolling Risk Index Fan Chart (market-data uncertainty bootstrap) ─────
    st.markdown('<div style="margin:0.8rem 0;border-top:1px solid #2a2a2a"></div>',
                unsafe_allow_html=True)
    st.markdown(
        '<h2 style="font-family:\'DM Sans\',sans-serif;font-size:1.0rem;font-weight:700;'
        'color:#CFB991;margin-bottom:0.1rem">Rolling Risk Index — Bootstrap Scenario Fan</h2>'
        '<p style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;color:#8890a1;margin:0 0 0.5rem">'
        'Moving-block bootstrap (n=300, block≈√T) resamples equity + commodity returns jointly. '
        'Band = 5th/95th percentile of scores under randomly re-ordered return histories '
        '(market-data inputs only; analyst layers — news GPR, CIS, chokepoint — fixed at current values).</p>',
        unsafe_allow_html=True,
    )
    try:
        from src.analysis.risk_score import risk_score_history, bootstrap_risk_history_ci
        with st.spinner("Computing risk index history and bootstrap CI (n=150)…"):
            _sp_score_hist = risk_score_history(pd.Series(dtype=float), cmd_r, eq_r)
            _sp_bci_df     = bootstrap_risk_history_ci(cmd_r, eq_r, n_boot=150)

        if isinstance(_sp_score_hist, pd.Series) and len(_sp_score_hist.dropna()) >= 20:
            _fig_fan = go.Figure()

            # Risk zone backgrounds
            for _y0, _y1, _fc in [
                (0,  25,  "rgba(46,125,50,0.07)"),
                (25, 50,  "rgba(100,100,100,0.04)"),
                (50, 75,  "rgba(230,126,34,0.07)"),
                (75, 100, "rgba(192,57,43,0.10)"),
            ]:
                _fig_fan.add_hrect(y0=_y0, y1=_y1, fillcolor=_fc,
                                   opacity=1.0, layer="below", line_width=0)

            # Bootstrap CI ribbon
            _sp_ci_label = ""
            if not _sp_bci_df.empty and {"p05", "p95"}.issubset(_sp_bci_df.columns):
                _sp_shared = _sp_score_hist.index.intersection(_sp_bci_df.index)
                if len(_sp_shared) > 5:
                    _sp_p05      = _sp_bci_df.loc[_sp_shared, "p05"]
                    _sp_p95      = _sp_bci_df.loc[_sp_shared, "p95"]
                    _sp_dates_ci = list(_sp_shared)
                    _fig_fan.add_trace(go.Scatter(
                        x=_sp_dates_ci + _sp_dates_ci[::-1],
                        y=list(_sp_p95) + list(_sp_p05)[::-1],
                        fill="toself",
                        fillcolor="rgba(207,185,145,0.18)",
                        line=dict(color="rgba(207,185,145,0.30)", width=0.5),
                        hoverinfo="skip",
                        showlegend=True,
                        name="90% CI — market-data only",
                    ))
                    if "p50" in _sp_bci_df.columns:
                        _fig_fan.add_trace(go.Scatter(
                            x=_sp_dates_ci,
                            y=list(_sp_bci_df.loc[_sp_shared, "p50"]),
                            mode="lines",
                            line=dict(color="rgba(207,185,145,0.40)", width=0.8, dash="dot"),
                            name="Bootstrap median",
                            showlegend=True,
                        ))
                    _sp_n_boot = 300
                    _sp_ci_label = (
                        f" Bootstrap n={_sp_n_boot}, block≈{max(5, int(len(_sp_score_hist)**0.5))}d."
                    )

            # Point estimate — realized market risk index
            _fig_fan.add_trace(go.Scatter(
                x=list(_sp_score_hist.index),
                y=list(_sp_score_hist.values),
                mode="lines",
                line=dict(color="#CFB991", width=1.8),
                name="Risk Index (realized)",
            ))

            _fig_fan.update_layout(
                height=300,
                paper_bgcolor="#080808", plot_bgcolor="#080808",
                font=dict(color="#e8e9ed"),
                xaxis=dict(tickfont=dict(size=8, color="#8890a1"),
                           rangeslider=dict(visible=False)),
                yaxis=dict(tickfont=dict(size=8, color="#8890a1"),
                           title=dict(text="Risk Index (0–100)", font=dict(size=8)),
                           range=[0, 100]),
                margin=dict(l=55, r=20, t=15, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            font=dict(size=8), bgcolor="rgba(0,0,0,0)"),
                hovermode="x unified",
            )
            _chart(_fig_fan)
            _panel_note(
                "Gold line = realized market risk index (equity + commodity realized vol, EWM z-scored). "
                "Band = 5th/95th percentile across 300 block-resampled return histories (block≈√T). "
                "Interpretation: the band shows the range of scores that are compatible with randomly "
                "re-ordered blocks of actual returns — not a confidence interval around the gold line. "
                "Periods where the gold line sits above p95 indicate concentrated volatility episodes "
                "that random block shuffling cannot replicate (expected; not a bias). "
                "Analyst layers — news GPR (40%), CIS (30%), chokepoint (20%) — fixed at current values."
                + _sp_ci_label
            )
        else:
            st.caption("Insufficient data for risk index history (need ≥ 120 days of returns).")
    except Exception as _sp_fan_err:
        st.caption("Risk index fan chart unavailable — see logs.")

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
                f'<div><span style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
                f'font-weight:700;letter-spacing:.1em;color:#8E9AAA;text-transform:uppercase">'
                f'Hub Commodity</span><br>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:16px;'
                f'font-weight:700;color:#CFB991">{_hub_cmd}</span>'
                f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.63rem;color:#8E9AAA">'
                f' → {_hub_n} equity markets</span></div>'
                f'<div><span style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
                f'font-weight:700;letter-spacing:.1em;color:#8E9AAA;text-transform:uppercase">'
                f'Hub Equity</span><br>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:16px;'
                f'font-weight:700;color:#e67e22">{_hub_eq}</span>'
                f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.63rem;color:#8E9AAA">'
                f' → {_hub_eq_n} commodities</span></div>'
                f'<div><span style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
                f'font-weight:700;letter-spacing:.1em;color:#8E9AAA;text-transform:uppercase">'
                f'Significant Links</span><br>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:16px;'
                f'font-weight:700;color:#c8c8c8">{_n_sig}</span>'
                f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.63rem;color:#8E9AAA">'
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
                st.caption("Network render unavailable — see logs.")

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
                        EC_TABLE_CSS
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
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.69rem;'
                f'font-weight:700;color:{_corr_color};letter-spacing:.08em;">'
                f'MARKET STRUCTURE: {_corr["overall_signal"].upper()} · GRS={_grs_val:.0f}</span><br>'
                f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.69rem;color:#b0b0b0;">'
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
                        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
                        f'font-weight:700;letter-spacing:.1em;color:#8E9AAA;'
                        f'text-transform:uppercase">{_crow["name"]}</span><br>'
                        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:18px;'
                        f'font-weight:700;color:{_sc}">{_struct}</span><br>'
                        f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.69rem;'
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
                            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
                            f'font-weight:700;letter-spacing:.1em;color:#8E9AAA;'
                            f'text-transform:uppercase">{_crow["name"]}</span><br>'
                            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:18px;'
                            f'font-weight:700;color:{_sc}">{_struct}</span><br>'
                            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.69rem;'
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
        st.caption("Futures curve section unavailable — see logs.")

    _page_conclusion(
        "Transmission Map",
        "Assets identified as strong transmitters across all three methods - Granger, Transfer "
        "Entropy, and Diebold-Yilmaz - are your first-order risk factors. A price move in a "
        "high-transmitter commodity is not isolated; it will propagate. Use this map to identify "
        "which equity markets to hedge when a key commodity breaks out."
    )

    # ── ΔCoVaR — Adrian & Brunnermeier Systemic Risk ─────────────────────────
    st.markdown(
        f'<div style="margin:1.2rem 0 0.5rem;border-top:1px solid #1e1e1e;padding-top:0.9rem">'
        f'<p style="{_F}font-size:0.58rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.14em;color:#8E9AAA;margin:0 0 3px">ΔCoVaR — Systemic Risk Amplification</p>'
        f'<p style="{_F}font-size:0.68rem;color:#8890a1;margin:0;line-height:1.55">'
        f'Adrian &amp; Brunnermeier (2016). For each asset, QuantReg of the equal-weighted system return '
        f'on the asset at τ=5% and τ=50% gives CoVaR at distress and median states. '
        f'<b style="color:#c8c8c8">ΔCoVaR = CoVaR(5%) − CoVaR(50%)</b> — '
        f'more negative means the asset amplifies system losses when it is itself in distress. '
        f'Estimation window: full sample. System = equal-weighted cross-asset index.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    try:
        from src.analysis.covar import compute_covar, rolling_covar, _MAX_ITER

        _covar_input = all_r[sel_all].dropna(how="all") if sel_all else all_r.dropna(how="all")

        with st.spinner("Computing ΔCoVaR (quantile regressions)…"):
            _covar_df = compute_covar(_covar_input, min_obs=126)

        if _covar_df.empty:
            st.caption("Insufficient data for ΔCoVaR — need ≥ 126 overlapping observations per asset.")
        else:
            # ── Convergence audit ──────────────────────────────────────────
            _no_conv = (
                _covar_df.index[~_covar_df["converged"]].tolist()
                if "converged" in _covar_df.columns
                else []
            )
            if _no_conv:
                st.warning(
                    f"QuantReg hit max iterations ({_MAX_ITER}) for "
                    f"{len(_no_conv)} asset(s): **{', '.join(_no_conv)}** — "
                    "estimates retained but marked ⚠ (ranking may be unreliable for flagged assets).",
                    icon="⚠️",
                )

            _cv_col_bar, _cv_col_roll = st.columns([1, 1.4], gap="medium")

            # ── Left: ranked horizontal bar ────────────────────────────────
            with _cv_col_bar:
                _dc = _covar_df["delta_covar"]
                # Non-converged bars rendered at reduced opacity
                _converged_mask = (
                    _covar_df["converged"].values
                    if "converged" in _covar_df.columns
                    else [True] * len(_covar_df)
                )
                # Color: most-negative = red (#c0392b) → least-negative = gold (#CFB991)
                _norm = (_dc - _dc.min()) / (_dc.max() - _dc.min() + 1e-9)
                _bar_colors = [
                    f"rgba({int(192 + (207-192)*v)},{int(57 + (185-57)*v)},{int(43 + (145-43)*v)},{0.85 if ok else 0.35})"
                    for v, ok in zip(_norm.values, _converged_mask)
                ]
                # Append ⚠ to y-axis labels for non-converged assets
                _y_labels = [
                    f"{a} ⚠" if a in _no_conv else a
                    for a in _dc.index.tolist()
                ]
                _fig_bar = go.Figure(go.Bar(
                    x=_dc.values,
                    y=_y_labels,
                    orientation="h",
                    marker=dict(color=_bar_colors, line=dict(width=0)),
                    text=[f"{v:+.2f}%" for v in _dc.values],
                    textposition="outside",
                    textfont=dict(family="JetBrains Mono, monospace", size=9, color="#8890a1"),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "ΔCoVaR: %{x:.3f}%<br>"
                        "<extra></extra>"
                    ),
                ))
                _fig_bar.update_layout(
                    height=max(240, len(_dc) * 28 + 40),
                    margin=dict(l=0, r=60, t=28, b=28),
                    paper_bgcolor="#000000",
                    plot_bgcolor="#000000",
                    font=dict(family="JetBrains Mono, monospace", color="#8890a1", size=10),
                    title=dict(
                        text="ΔCoVaR by Asset  (most systemic → top)",
                        font=dict(size=11, color="#CFB991", family="JetBrains Mono, monospace"),
                        x=0, xanchor="left", pad=dict(l=4, b=6),
                    ),
                    xaxis=dict(
                        title=dict(text="ΔCoVaR (%)", font=dict(size=9, color="#555960")),
                        tickfont=dict(size=9, color="#555960"),
                        gridcolor="#111111", zerolinecolor="#2a2a2a",
                        showline=False,
                    ),
                    yaxis=dict(
                        tickfont=dict(size=9, color="#c8c8c8"),
                        gridcolor="#111111",
                        showline=False, autorange="reversed",
                    ),
                    showlegend=False,
                )
                _chart(_fig_bar)

                # Compact stats table under the bar
                _top3   = _covar_df.head(3)
                _tbl_md = (
                    f'<div style="margin-top:8px">'
                    f'<table style="width:100%;border-collapse:collapse;'
                    f'font-family:\'JetBrains Mono\',monospace;font-size:0.56rem">'
                    f'<thead><tr>'
                    f'<th style="color:#555960;text-align:left;padding:3px 6px;'
                    f'border-bottom:1px solid #1e1e1e;letter-spacing:.08em">ASSET</th>'
                    f'<th style="color:#555960;text-align:right;padding:3px 6px;'
                    f'border-bottom:1px solid #1e1e1e;letter-spacing:.08em">VaR 5%</th>'
                    f'<th style="color:#555960;text-align:right;padding:3px 6px;'
                    f'border-bottom:1px solid #1e1e1e;letter-spacing:.08em">CoVaR 5%</th>'
                    f'<th style="color:#555960;text-align:right;padding:3px 6px;'
                    f'border-bottom:1px solid #1e1e1e;letter-spacing:.08em">ΔCoVaR</th>'
                    f'<th style="color:#555960;text-align:right;padding:3px 6px;'
                    f'border-bottom:1px solid #1e1e1e;letter-spacing:.08em">β(5%)</th>'
                    f'</tr></thead><tbody>'
                )
                for _an, _row in _top3.iterrows():
                    _dc_v    = _row["delta_covar"]
                    _dc_col  = "#c0392b" if _dc_v < -0.3 else "#e67e22" if _dc_v < -0.1 else "#8890a1"
                    _nc_flag = not _row.get("converged", True)
                    _name_td = (
                        f'{_an} <span title="QuantReg hit max_iter={_MAX_ITER}; estimate unreliable" '
                        f'style="color:#e67e22">⚠</span>'
                        if _nc_flag else _an
                    )
                    _tbl_md += (
                        f'<tr style="border-bottom:1px solid #0d0d0d;'
                        f'{"opacity:0.55;" if _nc_flag else ""}">'
                        f'<td style="color:#e8e9ed;padding:3px 6px">{_name_td}</td>'
                        f'<td style="color:#8890a1;text-align:right;padding:3px 6px">'
                        f'{_row["var5"]:+.2f}%</td>'
                        f'<td style="color:#8890a1;text-align:right;padding:3px 6px">'
                        f'{_row["covar5"]:+.2f}%</td>'
                        f'<td style="color:{_dc_col};font-weight:700;text-align:right;padding:3px 6px">'
                        f'{_dc_v:+.2f}%</td>'
                        f'<td style="color:#8890a1;text-align:right;padding:3px 6px">'
                        f'{_row["beta"]:.3f}</td>'
                        f'</tr>'
                    )
                _tbl_md += "</tbody></table></div>"
                st.markdown(_tbl_md, unsafe_allow_html=True)

            # ── Right: rolling ΔCoVaR for top contributors ─────────────────
            with _cv_col_roll:
                _roll_window = st.select_slider(
                    "Rolling window",
                    options=[63, 126, 252],
                    value=126,
                    format_func=lambda v: f"{v}d",
                    key="sp_covar_window",
                )
                _top_n_roll = min(5, len(_covar_df))

                with st.spinner("Rolling ΔCoVaR…"):
                    _roll_df = rolling_covar(
                        _covar_input,
                        window=_roll_window,
                        step=5,
                        top_n=_top_n_roll,
                        _n_rows=len(_covar_input),
                    )

                if _roll_df.empty:
                    st.caption("Not enough history for rolling ΔCoVaR.")
                else:
                    _ROLL_PALETTE = [
                        "#c0392b", "#e67e22", "#CFB991", "#2980b9", "#27ae60",
                    ]
                    _fig_roll = go.Figure()
                    for _i, _col in enumerate(_roll_df.columns):
                        _clr = _ROLL_PALETTE[_i % len(_ROLL_PALETTE)]
                        _fig_roll.add_trace(go.Scatter(
                            x=_roll_df.index,
                            y=_roll_df[_col],
                            mode="lines",
                            name=_col,
                            line=dict(color=_clr, width=1.5),
                            hovertemplate=f"<b>{_col}</b><br>%{{x|%Y-%m-%d}}<br>ΔCoVaR: %{{y:.2f}}%<extra></extra>",
                        ))
                    # Zero reference line
                    _fig_roll.add_hline(
                        y=0, line=dict(color="#2a2a2a", width=1, dash="dot"),
                    )
                    _fig_roll.update_layout(
                        height=max(240, len(_dc) * 28 + 40),
                        margin=dict(l=0, r=10, t=28, b=28),
                        paper_bgcolor="#000000",
                        plot_bgcolor="#000000",
                        font=dict(family="JetBrains Mono, monospace", color="#8890a1", size=10),
                        title=dict(
                            text=f"Rolling {_roll_window}d ΔCoVaR — Top {_top_n_roll} Systemic",
                            font=dict(size=11, color="#CFB991", family="JetBrains Mono, monospace"),
                            x=0, xanchor="left", pad=dict(l=4, b=6),
                        ),
                        xaxis=dict(
                            tickfont=dict(size=9, color="#555960"),
                            gridcolor="#0d0d0d", zerolinecolor="#1a1a1a",
                            showline=False,
                        ),
                        yaxis=dict(
                            title=dict(text="ΔCoVaR (%)", font=dict(size=9, color="#555960")),
                            tickfont=dict(size=9, color="#555960"),
                            gridcolor="#111111", zerolinecolor="#2a2a2a",
                            showline=False,
                        ),
                        legend=dict(
                            font=dict(size=9, color="#8890a1"),
                            bgcolor="rgba(0,0,0,0)",
                            bordercolor="#1e1e1e",
                            borderwidth=1,
                            orientation="h",
                            x=0, y=-0.12, xanchor="left",
                        ),
                        hovermode="x unified",
                    )
                    _chart(_fig_roll)

                    st.markdown(
                        f'<p style="{_F}font-size:0.60rem;color:#555960;margin-top:4px;line-height:1.45">'
                        f'QuantReg at τ=5% and τ=50% re-estimated on each {_roll_window}-day window, '
                        f'stepped every 5 days. A deepening negative trend signals growing systemic '
                        f'tail-risk contribution from that asset in the current regime.</p>',
                        unsafe_allow_html=True,
                    )

    except Exception as _cov_err:
        st.caption("ΔCoVaR unavailable — see logs.")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION: Jordà Local-Projection IRF
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div style="margin:0.8rem 0;border-top:1px solid #2a2a2a"></div>',
                unsafe_allow_html=True)
    st.markdown(
        '<h2 style="font-family:\'DM Sans\',sans-serif;font-size:1.0rem;font-weight:700;'
        'color:#CFB991;margin-bottom:0.1rem">Conflict Shock Impulse Response (Jordà LP)</h2>'
        '<p style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;color:#8890a1;'
        'margin:0 0 0.5rem;line-height:1.55">'
        'How do asset returns respond to an unexpected escalation in conflict intensity? '
        'Shock = AR-residual of GDELT daily news-volume for each conflict '
        '(log-transformed, BIC lag selection, standardised to 1-σ). '
        'At each horizon h = 0–20 trading days, '
        'h-day cumulative return is regressed on the shock plus lag controls '
        '(Jordà 2005). SEs: Newey-West HAC, bandwidth = max(h, 2). '
        '90% confidence bands. IRF in % per 1-σ shock.</p>',
        unsafe_allow_html=True,
    )

    try:
        from src.analysis.local_projection import lp_irf_all_conflicts
        from src.data.config import CONFLICTS as _LP_CONFLICTS

        # Conflict selector ────────────────────────────────────────────────────
        _lp_names  = {c["id"]: c["name"] for c in _LP_CONFLICTS}
        _lp_labels = [f'{c["name"]}' for c in _LP_CONFLICTS]
        _lp_ids    = [c["id"]   for c in _LP_CONFLICTS]
        _lp_sel_label = st.selectbox(
            "Conflict",
            options=_lp_labels,
            index=0,
            key="lp_conflict_sel",
            label_visibility="collapsed",
        )
        _lp_sel_id = _lp_ids[_lp_labels.index(_lp_sel_label)]

        with st.spinner("Fetching GDELT conflict-news series and computing LP-IRFs…"):
            _lp_results = lp_irf_all_conflicts(
                eq_r, cmd_r,
                _n_rows=len(eq_r),
                max_h=20,
            )

        _lp_entry = _lp_results.get(_lp_sel_id, {})
        _lp_irf   = _lp_entry.get("irf", pd.DataFrame())
        _lp_err   = _lp_entry.get("error")
        _lp_nobs  = _lp_entry.get("n_obs_shock", 0)
        _lp_ok    = _lp_entry.get("ar_ok", False)

        # Status chip ──────────────────────────────────────────────────────────
        _F2 = "font-family:'JetBrains Mono',monospace;"
        if _lp_ok and not _lp_irf.empty:
            _lp_n_assets = _lp_irf["asset"].nunique()
            st.markdown(
                f'<span style="{_F2}font-size:0.50rem;color:#27ae60;'
                f'background:#0a1a0a;border:1px solid #27ae60;padding:2px 8px">'
                f'SHOCK T={_lp_nobs}d · {_lp_n_assets} ASSETS · NW-HAC 90% CI</span>',
                unsafe_allow_html=True,
            )
        elif _lp_err:
            st.markdown(
                f'<span style="{_F2}font-size:0.50rem;color:#c0392b;'
                f'background:#1a0a0a;border:1px solid #c0392b;padding:2px 8px">'
                f'NO DATA — {_lp_err}</span>',
                unsafe_allow_html=True,
            )

        # IRF charts ───────────────────────────────────────────────────────────
        if not _lp_irf.empty:
            _lp_assets = sorted(_lp_irf["asset"].unique())
            _lp_n      = len(_lp_assets)
            _lp_ncols  = min(3, _lp_n)
            _lp_cols   = st.columns(_lp_ncols, gap="small")

            for _ai, _asset in enumerate(_lp_assets):
                _adf = _lp_irf[_lp_irf["asset"] == _asset].sort_values("horizon")
                if _adf.empty:
                    continue

                _horizons = _adf["horizon"].tolist()
                _coef     = _adf["coef"].tolist()
                _ci_lo    = _adf["ci_lo"].tolist()
                _ci_hi    = _adf["ci_hi"].tolist()
                _pvals    = _adf["pval"].tolist()

                # Colour: gold if max |IRF| is significant at 10%, else muted
                _sig_any  = any(p < 0.10 for p in _pvals)
                _line_col = "#CFB991" if _sig_any else "#5a5a6a"
                _fill_col = "rgba(207,185,145,0.14)" if _sig_any else "rgba(90,90,106,0.12)"
                _fill_bdr = "rgba(207,185,145,0.30)" if _sig_any else "rgba(90,90,106,0.25)"

                _fig_irf = go.Figure()

                # CI ribbon
                _fig_irf.add_trace(go.Scatter(
                    x=_horizons + _horizons[::-1],
                    y=_ci_hi + _ci_lo[::-1],
                    fill="toself",
                    fillcolor=_fill_col,
                    line=dict(color=_fill_bdr, width=0.5),
                    hoverinfo="skip",
                    showlegend=False,
                ))

                # Zero reference
                _fig_irf.add_hline(y=0, line_color="#c0392b",
                                   line_dash="dot", line_width=0.8)

                # IRF line
                _nobs_label = int(_adf["nobs"].iloc[-1]) if len(_adf) > 0 else 0
                _fig_irf.add_trace(go.Scatter(
                    x=_horizons, y=_coef,
                    mode="lines+markers",
                    line=dict(color=_line_col, width=1.6),
                    marker=dict(
                        size=[5 if p < 0.10 else 3 for p in _pvals],
                        color=[_line_col if p < 0.10 else "#555" for p in _pvals],
                        symbol="circle",
                    ),
                    name=_asset,
                    hovertemplate=(
                        "h=%{x}d<br>IRF=%{y:.3f}%<extra></extra>"
                    ),
                    showlegend=False,
                ))

                _fig_irf.update_layout(
                    template="purdue",
                    height=220,
                    paper_bgcolor="#000000",
                    plot_bgcolor="#080808",
                    margin=dict(l=42, r=12, t=32, b=36),
                    title=dict(
                        text=(
                            f'<span style="font-size:0.56rem;color:#CFB991;'
                            f'font-family:JetBrains Mono,monospace">{_asset}</span>'
                            + (
                                f'<span style="font-size:0.50rem;color:#27ae60"> ●</span>'
                                if _sig_any else ""
                            )
                        ),
                        x=0.0, xanchor="left",
                        font=dict(size=9),
                        y=0.97,
                    ),
                    xaxis=dict(
                        title=dict(text="Horizon (trading days)",
                                   font=dict(size=7, color="#8890a1")),
                        tickfont=dict(size=7, color="#8890a1"),
                        gridcolor="#111",
                        dtick=5,
                        zeroline=False,
                    ),
                    yaxis=dict(
                        title=dict(text="Cum. return (%)",
                                   font=dict(size=7, color="#8890a1")),
                        tickfont=dict(size=7, color="#8890a1"),
                        gridcolor="#111",
                        zeroline=False,
                    ),
                    hovermode="x unified",
                )

                with _lp_cols[_ai % _lp_ncols]:
                    st.plotly_chart(_fig_irf, use_container_width=True,
                                    config={"displayModeBar": False})

            _panel_note(
                "Gold dot = significant at 10% (NW-HAC). Muted = not significant. "
                "IRF = % cumulative log-return per 1-σ unexpected escalation shock. "
                f"Shock series: {_lp_nobs} trading-day obs, AR BIC-selected, standardised. "
                "Positive IRF at h=0 = same-day price increase after news escalation. "
                "CI width grows with h due to compounding uncertainty and shrinking sample."
            )

        elif _lp_ok:
            st.caption("LP regression returned no results — insufficient overlapping history.")

    except Exception as _lp_top_err:
        st.caption("LP-IRF unavailable — see logs.")

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
                        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
                        f'font-weight:700;color:{_cc2}">{_cr2.get("label","?")}</span>'
                        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.69rem;'
                        f'font-weight:700;color:#CFB991;margin-top:2px">'
                        f'CIS×TPS {_trans_score:.0f}</div>'
                        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
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
