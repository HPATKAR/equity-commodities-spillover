"""
Page 3 - Correlation Analysis
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
    regime_transition_matrix, regime_steady_state,
    regime_mean_first_passage, regime_run_statistics,
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

    # ── Hoist regime computation (shared across tabs) ──────────────────────
    avg_corr = average_cross_corr_series(eq_r, cmd_r, window=60)
    regimes  = detect_correlation_regime(avg_corr)

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Rolling Correlation", "DCC-GARCH", "Regime Detection", "Regime Forecast",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 - Rolling Correlation
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
                f"<b>{latest_corr:.3f}</b> - "
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
    # TAB 2 - DCC-GARCH
    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("DCC-GARCH Dynamic Conditional Correlation")
        _definition_block(
            "DCC-GARCH (Engle, 2002)",
            "Unlike rolling Pearson, DCC-GARCH accounts for heteroskedasticity - "
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
                "DCC typically diverges from rolling Pearson during regime transitions - "
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
    # TAB 3 - Regime Detection
    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Cross-Asset Correlation Regime Detection")
        _definition_block(
            "Regime Classification",
            "The daily average absolute cross-asset correlation is classified into four regimes: "
            "Decorrelated (< 0.15), Normal (0.15-0.45), Elevated (0.45-0.60), "
            "Crisis (>= 0.60 for 3+ consecutive days). "
            "Regime detection helps identify when cross-asset hedges break down.",
        )

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
                "the most profitable cross-asset hedging windows - but also the highest "
                "basis risk as correlations can snap back abruptly."
            )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 - Regime Forecast (Markov Chain)
    # ══════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("Regime Transition Forecast - Markov Chain Analysis")
        _definition_block(
            "Markov Regime Transition Model",
            "A first-order Markov chain models the correlation regime as a stochastic process "
            "where tomorrow's regime depends only on today's. Transition probabilities are "
            "estimated from the full historical regime sequence. "
            "Key outputs: transition matrix (P(j|i) = probability of moving from regime i to j), "
            "steady-state distribution (long-run time in each regime), and "
            "Mean First Passage Time - expected trading days until Crisis from any starting regime.",
        )

        if regimes.empty or len(regimes) < 60:
            st.warning("Insufficient regime history for Markov analysis. Extend the date range.")
        else:
            trans = regime_transition_matrix(regimes)
            steady = regime_steady_state(trans)
            mfpt   = regime_mean_first_passage(trans, target=3)
            run_stats = regime_run_statistics(regimes)

            current_r  = int(regimes.dropna().iloc[-1])
            r_names    = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis"}
            r_colors   = {0: "#2e7d32",      1: "#555960", 2: "#e67e22", 3: "#c0392b"}
            r_labels   = [r_names[i] for i in range(4)]

            # ── Section A: Transition Probability Heatmap ────────────────────
            st.subheader("Transition Probability Matrix")
            z_vals = trans.values * 100   # convert to percentages

            fig_tm = go.Figure(go.Heatmap(
                z=z_vals,
                x=r_labels,
                y=r_labels,
                colorscale=[
                    [0.00, "#f5f2ee"],
                    [0.20, "#fde0c8"],
                    [0.50, "#e05c3a"],
                    [1.00, "#7a0e0e"],
                ],
                zmin=0, zmax=100,
                text=[[f"{v:.1f}%" for v in row] for row in z_vals],
                texttemplate="%{text}",
                textfont=dict(size=13, family="JetBrains Mono, monospace"),
                colorbar=dict(
                    title=dict(text="P(%)", font=dict(size=9)),
                    thickness=12, len=0.7,
                    tickvals=[0, 25, 50, 75, 100],
                    ticktext=["0%", "25%", "50%", "75%", "100%"],
                    tickfont=dict(size=8, family="JetBrains Mono, monospace"),
                ),
                hoverongaps=False,
                hovertemplate=(
                    "<b>From: %{y}</b><br>"
                    "<b>To: %{x}</b><br>"
                    "Probability: %{z:.1f}%<extra></extra>"
                ),
            ))
            fig_tm.update_layout(
                template="purdue",
                height=340,
                margin=dict(l=100, r=80, t=50, b=80),
                xaxis=dict(title="To Regime", tickfont=dict(size=11)),
                yaxis=dict(title="From Regime", tickfont=dict(size=11)),
                title=dict(
                    text="Regime Transition Probability Matrix (From Row -> To Column)",
                    font=dict(size=11, family="DM Sans, sans-serif"),
                ),
            )
            # Highlight current regime row with a rectangle
            fig_tm.add_shape(
                type="rect",
                x0=-0.5, x1=3.5,
                y0=current_r - 0.5, y1=current_r + 0.5,
                line=dict(color=r_colors[current_r], width=2.5),
                fillcolor="rgba(0,0,0,0)",
            )
            _chart(fig_tm)

            st.markdown(
                f'<p style="font-size:0.68rem;color:#555;margin-top:4px">'
                f'Highlighted row = current regime (<b style="color:{r_colors[current_r]}">'
                f'{r_names[current_r]}</b>). Read across to see probability of each next-day outcome.</p>',
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Section B: 3-column forward metrics ─────────────────────────
            col_ss, col_mfpt, col_runs = st.columns(3)

            with col_ss:
                st.subheader("Steady-State Distribution")
                _section_note(
                    "Long-run proportion of time market spends in each regime, "
                    "implied by the transition matrix."
                )
                fig_ss = go.Figure(go.Bar(
                    x=[f"<b>{r_labels[i]}</b>" for i in range(4)],
                    y=[steady[i] * 100 for i in range(4)],
                    marker_color=[r_colors[i] for i in range(4)],
                    text=[f"{steady[i]*100:.1f}%" for i in range(4)],
                    textposition="outside",
                    textfont=dict(size=10, family="JetBrains Mono, monospace"),
                ))
                fig_ss.update_layout(
                    template="purdue",
                    height=260,
                    margin=dict(l=20, r=20, t=20, b=60),
                    yaxis=dict(range=[0, max(steady)*130, ], title="%"),
                    showlegend=False,
                )
                _chart(fig_ss)

            with col_mfpt:
                st.subheader("Expected Days to Crisis")
                _section_note(
                    "Mean First Passage Time: expected trading days "
                    "until Crisis regime, starting from each regime."
                )
                mfpt_data = {
                    "Regime":         [r_names[i] for i in range(4)],
                    "Expected Days":  [
                        "0 (already in Crisis)" if i == 3
                        else (f"{mfpt.get(i, float('nan')):.0f} days"
                              if mfpt.get(i, float('nan')) < 5000
                              else ">5,000 (rare)")
                        for i in range(4)
                    ],
                }
                mfpt_df = pd.DataFrame(mfpt_data)
                # Highlight current regime
                def _mfpt_style(row):
                    regime_idx = list(r_names.values()).index(row["Regime"])
                    if regime_idx == current_r:
                        return [f"background:{r_colors[current_r]}22;font-weight:700"] * 2
                    return [""] * 2
                st.dataframe(
                    mfpt_df.style.apply(_mfpt_style, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )

                # Big metric for current regime
                cur_mfpt = mfpt.get(current_r, float("nan"))
                cur_mfpt_str = (
                    "0 (in Crisis)" if current_r == 3
                    else (f"{cur_mfpt:.0f} days" if cur_mfpt < 5000 else ">5,000")
                )
                st.markdown(
                    f'<div style="margin-top:12px;padding:10px;background:#fafaf8;'
                    f'border-left:4px solid {r_colors[current_r]};border-radius:0 4px 4px 0">'
                    f'<div style="font-size:0.55rem;font-weight:700;letter-spacing:0.12em;'
                    f'text-transform:uppercase;color:#888">From {r_names[current_r]}</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:1.30rem;'
                    f'font-weight:700;color:{r_colors[current_r]}">{cur_mfpt_str}</div>'
                    f'<div style="font-size:0.62rem;color:#555">to next Crisis</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with col_runs:
                st.subheader("Regime Run Statistics")
                _section_note(
                    "Historical consecutive-day run lengths per regime "
                    "(mean / median / max days)."
                )
                run_display = pd.DataFrame({
                    "Regime": [r_names[i] for i in run_stats.index if i in r_names],
                    "Avg Days":    run_stats["mean"].values,
                    "Median":      run_stats["median"].values,
                    "Max Days":    run_stats["max"].values,
                    "Episodes":    run_stats["count"].values.astype(int),
                })
                st.dataframe(
                    run_display.style
                    .format({"Avg Days": "{:.1f}", "Median": "{:.1f}",
                             "Max Days": "{:.0f}", "Episodes": "{:.0f}"}),
                    use_container_width=True,
                    hide_index=True,
                )

            # ── Section C: Takeaway ──────────────────────────────────────────
            p_stay    = float(trans.loc[current_r, current_r]) * 100
            p_crisis  = float(trans.loc[current_r, 3]) * 100
            avg_dur   = float(run_stats.loc[current_r, "mean"]) if current_r in run_stats.index else 0
            ss_crisis = steady[3] * 100

            _takeaway_block(
                f"<b>Current regime: <span style='color:{r_colors[current_r]}'>"
                f"{r_names[current_r]}</span></b>. "
                f"Probability of staying in {r_names[current_r]} tomorrow: "
                f"<b>{p_stay:.1f}%</b>. "
                f"Probability of transitioning to Crisis: <b>{p_crisis:.1f}%</b>. "
                f"Expected {cur_mfpt_str} until next Crisis episode. "
                f"Historical average {r_names[current_r]} run: <b>{avg_dur:.0f} days</b>. "
                f"Long-run, markets spend <b>{ss_crisis:.1f}%</b> of time in Crisis regime."
            )

    _page_footer()
