"""
Page 3 - Correlation Analysis
Rolling correlations, DCC-GARCH dynamic pairs, regime detection, pair explorer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_returns, load_fixed_income_returns, load_fx_returns
from src.data.config import (
    GEOPOLITICAL_EVENTS, EQUITY_REGIONS, COMMODITY_GROUPS, PALETTE,
    FIXED_INCOME_GROUPS, FX_GROUPS,
)
from src.analysis.correlations import (
    rolling_correlation, cross_asset_corr, dcc_correlation,
    average_cross_corr_series, detect_correlation_regime,
    regime_transition_matrix, regime_steady_state,
    regime_mean_first_passage, regime_run_statistics,
)
from src.ui.shared import (
    _style_fig, _chart, _page_intro, _thread, _section_note,
    _definition_block, _takeaway_block, _page_conclusion, _page_footer,
    _add_event_bands, _insight_note, _line_style, _EQUITY_REGIONS,
)

_REGIME_NAMES  = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis"}
_REGIME_COLORS = {0: "#2e7d32",      1: "#555960", 2: "#e67e22",  3: "#c0392b"}
_F = "font-family:'DM Sans',sans-serif;"


def _label(txt: str) -> None:
    st.markdown(
        f'<p style="{_F}font-size:0.58rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.14em;color:#8E6F3E;margin:0 0 5px 0">{txt}</p>',
        unsafe_allow_html=True,
    )


def _panel_note(txt: str) -> None:
    st.markdown(
        f'<p style="{_F}font-size:0.64rem;color:#8890a1;line-height:1.55;margin:4px 0 0 0">{txt}</p>',
        unsafe_allow_html=True,
    )


def page_correlation(start: str, end: str, fred_key: str = "") -> None:
    st.markdown(
        '<h1 style="font-family:\'DM Sans\',sans-serif;font-size:1.25rem;'
        'font-weight:700;margin-bottom:0.1rem">Correlation Analysis</h1>'
        '<p style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;color:#8890a1;'
        'margin:0 0 0.7rem">Rolling Pearson · DCC-GARCH · Regime Detection · Markov Forecast</p>',
        unsafe_allow_html=True,
    )
    _page_intro(
        "Correlation is the <em>first observable signal</em> of spillover: when equity returns and commodity "
        "returns start moving together, the channel between the two markets is open. "
        "<strong>This page measures how open that channel is right now.</strong> "
        "Rolling Pearson gives the direction and magnitude. DCC-GARCH shows whether the relationship "
        "is structural or noise - non-linear dependence that spikes during stress is the hallmark of "
        "a genuine spillover regime. The Markov forecast tells you where the regime is headed. "
        "A Crisis regime here means equity shocks <em>are</em> transmitting into commodities in real time."
    )

    with st.spinner("Loading returns…"):
        eq_r, cmd_r = load_returns(start, end)

    if eq_r.empty or cmd_r.empty:
        st.error("Market data unavailable.")
        return

    try:
        fi_r = load_fixed_income_returns(start, end)
    except Exception:
        fi_r = pd.DataFrame()
    try:
        fx_r = load_fx_returns(start, end)
    except Exception:
        fx_r = pd.DataFrame()

    avg_corr = average_cross_corr_series(eq_r, cmd_r, window=60)
    regimes  = detect_correlation_regime(avg_corr)

    # Build combined returns including FI and FX
    _r_frames = [eq_r, cmd_r]
    if not fi_r.empty:
        _r_frames.append(fi_r)
    if not fx_r.empty:
        _r_frames.append(fx_r)
    all_r = pd.concat(_r_frames, axis=1)

    eq_options  = list(eq_r.columns)
    cmd_options = list(cmd_r.columns)
    fi_options  = list(fi_r.columns) if not fi_r.empty else []
    fx_options  = list(fx_r.columns) if not fx_r.empty else []

    current_r = int(regimes.dropna().iloc[-1]) if not regimes.empty else 1

    # ══════════════════════════════════════════════════════════════════════
    # ROW 1: Rolling Correlation | DCC-GARCH
    # ══════════════════════════════════════════════════════════════════════
    col_rc, col_dcc = st.columns(2, gap="medium")

    # ── Panel 1: Rolling Correlation ──────────────────────────────────────
    with col_rc:
        _label("Rolling Pairwise Correlation")
        ca1, ca2, ca3 = st.columns([1, 1, 0.8])
        all_opts = eq_options + cmd_options + fi_options + fx_options
        asset_a = ca1.selectbox("Asset A", all_opts, index=0, key="rc_a", label_visibility="collapsed")
        asset_b = ca2.selectbox("Asset B", all_opts, index=len(eq_options), key="rc_b", label_visibility="collapsed")
        window  = ca3.select_slider("Win", [21, 42, 63, 126, 252], value=63, label_visibility="collapsed")

        if asset_a in all_r.columns and asset_b in all_r.columns:
            rc = rolling_correlation(all_r[asset_a], all_r[asset_b], window)
            latest_corr = rc.dropna().iloc[-1] if not rc.dropna().empty else 0
            pctile = float((rc.dropna() <= latest_corr).mean() * 100)

            fig_rc = go.Figure()
            fig_rc.add_trace(go.Scatter(
                x=rc.index, y=rc.values,
                name=f"{asset_a} / {asset_b}",
                line=dict(color=PALETTE[1], width=1.6),
                fill="tozeroy", fillcolor="rgba(207,185,145,0.10)",
            ))
            fig_rc.add_hline(y=0,    line=dict(color="#ABABAB", width=1, dash="dot"))
            fig_rc.add_hline(y=0.5,  line=dict(color="#e67e22", width=0.7, dash="dot"),
                             annotation_text="Elevated", annotation_font_size=8)
            fig_rc.add_hline(y=-0.5, line=dict(color="#2980b9", width=0.7, dash="dot"),
                             annotation_text="Diverge", annotation_font_size=8)
            _add_event_bands(fig_rc)
            _chart(_style_fig(fig_rc, height=300))
            _insight_note(
                "Shows whether commodities and equities are moving together or apart over time. "
                "When the line rises above 0.4, stress is spreading across asset classes. "
                "When it drops below 0, commodity and equity prices are moving in opposite directions - the classic diversification condition."
            )
            _panel_note(
                f"<b>{asset_a} / {asset_b}</b> · {window}d corr = <b>{latest_corr:.3f}</b> · "
                f"{pctile:.0f}th historical percentile"
            )

        # Multi-pair overlay in expander
        with st.expander("Multi-pair overlay (4 key pairs · 63d)"):
            default_pairs = [
                ("S&P 500", "WTI Crude Oil"), ("S&P 500", "Gold"),
                ("Nikkei 225", "WTI Crude Oil"), ("Eurostoxx 50", "Natural Gas"),
            ]
            fig_multi = go.Figure()
            eq_i = cmd_i = 0
            for (a, b) in default_pairs:
                if a in all_r.columns and b in all_r.columns:
                    pair_name = f"{a} / {b}"
                    ls = _line_style(pair_name, eq_i, cmd_i)
                    if pair_name in _EQUITY_REGIONS: eq_i += 1
                    else: cmd_i += 1
                    fig_multi.add_trace(go.Scatter(
                        x=rolling_correlation(all_r[a], all_r[b], 63).index,
                        y=rolling_correlation(all_r[a], all_r[b], 63).values,
                        name=pair_name,
                        line=ls,
                    ))
            fig_multi.add_hline(y=0, line=dict(color="#ABABAB", width=1, dash="dot"))
            _add_event_bands(fig_multi)
            _chart(_style_fig(fig_multi, height=280))
            _insight_note(
                "Shows four key equity-commodity correlations on a single chart for quick comparison. "
                "Lines rising together suggest a broad market stress event is pushing all relationships in the same direction. "
                "Diverging lines mean each pair is responding to its own specific fundamental driver."
            )

    # ── Panel 2: DCC-GARCH ────────────────────────────────────────────────
    _thread(
        "Simple rolling correlation smooths over the past equally. DCC-GARCH below does better: "
        "it lets correlation update dynamically, giving more weight to recent observations and "
        "producing a cleaner estimate of the current coupling strength."
    )
    with col_dcc:
        _label("DCC-GARCH Dynamic Conditional Correlation")
        cd1, cd2 = st.columns(2)
        dcc_eq  = cd1.selectbox("Asset A (Equity / FI / FX)", eq_options + fi_options + fx_options, index=0, key="dcc_eq", label_visibility="collapsed")
        dcc_cmd = cd2.selectbox("Asset B (Commodity / FI / FX)", cmd_options + fi_options + fx_options, index=0, key="dcc_cmd", label_visibility="collapsed")

        if dcc_eq in eq_r.columns and dcc_cmd in cmd_r.columns:
            with st.spinner("Computing DCC-GARCH…"):
                dcc_series  = dcc_correlation(eq_r[dcc_eq], cmd_r[dcc_cmd])
                roll_series = rolling_correlation(eq_r[dcc_eq], cmd_r[dcc_cmd], 63)

            latest_dcc = dcc_series.dropna().iloc[-1] if not dcc_series.dropna().empty else 0

            fig_dcc = go.Figure()
            fig_dcc.add_trace(go.Scatter(
                x=dcc_series.index, y=dcc_series.values,
                name="DCC-GARCH",
                line=dict(color=PALETTE[0], width=2),
            ))
            fig_dcc.add_trace(go.Scatter(
                x=roll_series.index, y=roll_series.values,
                name="Rolling 63d Pearson",
                line=dict(color=PALETTE[1], width=1.4, dash="dot"),
            ))
            fig_dcc.add_hline(y=0, line=dict(color="#ABABAB", width=1, dash="dot"))
            _add_event_bands(fig_dcc)
            _chart(_style_fig(fig_dcc, height=300))
            _insight_note(
                "A more sophisticated statistical model than simple rolling correlation. "
                "It accounts for the fact that market co-movement spikes during crises and relaxes during calm periods. "
                "The adaptive nature of DCC means it reacts faster to sudden regime shifts than a fixed rolling window."
            )
            _panel_note(
                f"<b>{dcc_eq} / {dcc_cmd}</b> · DCC = <b>{latest_dcc:.3f}</b> · "
                "DCC captures volatility clustering better than static Pearson"
            )

        with st.expander("DCC vs multiple commodities"):
            ref_eq = st.selectbox("Reference equity", eq_options, index=0, key="dcc_ref")
            cmd_sel = st.multiselect("Commodities", cmd_options, default=cmd_options[:5], key="dcc_cmds")
            if ref_eq in eq_r.columns and cmd_sel:
                fig_dcc2 = go.Figure()
                eq_i = cmd_i = 0
                for cmd in cmd_sel:
                    if cmd in cmd_r.columns:
                        s = dcc_correlation(eq_r[ref_eq], cmd_r[cmd])
                        ls = _line_style(cmd, eq_i, cmd_i)
                        if cmd in _EQUITY_REGIONS: eq_i += 1
                        else: cmd_i += 1
                        fig_dcc2.add_trace(go.Scatter(
                            x=s.index, y=s.values, name=cmd,
                            line=ls,
                        ))
                fig_dcc2.add_hline(y=0, line=dict(color="#ABABAB", width=1, dash="dot"))
                _add_event_bands(fig_dcc2)
                _chart(_style_fig(fig_dcc2, height=260))
                _insight_note(
                    "Compares the DCC-GARCH dynamic correlation between a single equity index and multiple commodities simultaneously. "
                    "Lines that spike upward at the same time indicate a broad risk-off event rather than a commodity-specific move. "
                    "Use this view to identify which commodity pairs are most tightly linked to the selected equity market."
                )

    st.markdown('<div style="margin:0.5rem 0;border-top:1px solid #2a2a2a"></div>',
                unsafe_allow_html=True)
    _thread(
        "Knowing the exact correlation number is useful. Knowing which regime you are in - and "
        "how stable that regime has been - is more actionable. Below, we classify the market "
        "into three states: low coupling (diversification working), medium (partial), and high "
        "(breakdown)."
    )

    # ══════════════════════════════════════════════════════════════════════
    # ROW 2: Regime Detection | Regime Forecast (Markov)
    # ══════════════════════════════════════════════════════════════════════
    col_reg, col_mkov = st.columns(2, gap="medium")

    # ── Panel 3: Regime Detection ─────────────────────────────────────────
    with col_reg:
        _label("Correlation Regime Detection")

        if not avg_corr.empty:
            fig_reg = go.Figure()
            for r_val in [0, 1, 2, 3]:
                mask = (regimes == r_val)
                fig_reg.add_trace(go.Scatter(
                    x=avg_corr.index[mask],
                    y=avg_corr.values[mask],
                    mode="markers",
                    marker=dict(color=_REGIME_COLORS[r_val], size=2, opacity=0.5),
                    name=_REGIME_NAMES[r_val],
                ))
            fig_reg.add_trace(go.Scatter(
                x=avg_corr.index, y=avg_corr.values,
                name="Avg |Corr|",
                line=dict(color="#e8e9ed", width=1.2),
                showlegend=False,
            ))
            for thresh, label, color in [
                (0.15, "Decorrelated", "#2e7d32"),
                (0.45, "Elevated",     "#e67e22"),
                (0.60, "Crisis",       "#c0392b"),
            ]:
                fig_reg.add_hline(y=thresh,
                    line=dict(color=color, width=1, dash="dash"),
                    annotation_text=label,
                    annotation_font=dict(size=8, color=color),
                )
            _add_event_bands(fig_reg)
            _chart(_style_fig(fig_reg, height=300))
            _insight_note(
                "Classifies each period into one of three regimes - Normal, Elevated, or Crisis - based on how correlated and volatile markets are. "
                "Colour bands mark when the model identified each regime. "
                "Look for transitions into Crisis regime as early warning signals."
            )

            # Compact regime summary table inline
            regime_counts = regimes.value_counts().sort_index()
            total = len(regimes)
            rows = ""
            for i in sorted(regime_counts.index):
                pct = regime_counts[i] / total * 100
                c = _REGIME_COLORS[i]
                is_cur = "font-weight:700;" if i == current_r else ""
                rows += (
                    f'<tr><td style="padding:3px 8px;{is_cur}font-size:0.66rem;'
                    f'color:{c}">{_REGIME_NAMES[i]}</td>'
                    f'<td style="padding:3px 8px;font-family:JetBrains Mono,monospace;'
                    f'font-size:0.66rem;{is_cur}color:#e8e9ed">{regime_counts[i]}</td>'
                    f'<td style="padding:3px 8px;font-size:0.66rem;{is_cur}color:#e8e9ed">{pct:.1f}%</td>'
                    f'<td style="padding:3px 8px;width:80px">'
                    f'<div style="background:#2a2a2a;height:4px;border-radius:2px">'
                    f'<div style="width:{pct:.0f}%;height:4px;background:{c};border-radius:2px"></div>'
                    f'</div></td></tr>'
                )
            st.markdown(
                f'<table style="width:100%;border-collapse:collapse;{_F};margin-top:4px">'
                f'<thead><tr style="background:#1c1c1c">'
                f'<th style="padding:3px 8px;font-size:0.56rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#CFB991;text-align:left">Regime</th>'
                f'<th style="padding:3px 8px;font-size:0.56rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#CFB991;text-align:left">Days</th>'
                f'<th style="padding:3px 8px;font-size:0.56rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#CFB991;text-align:left">% Time</th>'
                f'<th style="padding:3px 8px;font-size:0.56rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#CFB991;text-align:left">Bar</th>'
                f'</tr></thead><tbody>{rows}</tbody></table>',
                unsafe_allow_html=True,
            )
            _panel_note(
                f"Current: <b style='color:{_REGIME_COLORS[current_r]}'>{_REGIME_NAMES[current_r]}</b> · "
                "Crisis = correlations break down hedges; most profitable window for cross-asset strategies"
            )

    # ── Panel 4: Markov Regime Forecast ───────────────────────────────────
    _thread(
        "Regime detection tells you where you are today. The Markov chain below tells you where "
        "you are statistically likely to go next, based on how often each regime has historically "
        "transitioned into another."
    )
    with col_mkov:
        _label("Regime Forecast: Markov Chain")

        if regimes.empty or len(regimes) < 60:
            st.warning("Insufficient regime history. Extend date range.")
        else:
            trans    = regime_transition_matrix(regimes)
            steady   = regime_steady_state(trans)
            mfpt     = regime_mean_first_passage(trans, target=3)
            run_stats = regime_run_statistics(regimes)

            r_labels = [_REGIME_NAMES[i] for i in range(4)]
            z_vals   = trans.values * 100

            fig_tm = go.Figure(go.Heatmap(
                z=z_vals, x=r_labels, y=r_labels,
                colorscale=[[0,"#1c1c1c"],[0.2,"#fde0c8"],[0.5,"#e05c3a"],[1,"#7a0e0e"]],
                zmin=0, zmax=100,
                text=[[f"{v:.0f}%" for v in row] for row in z_vals],
                texttemplate="%{text}",
                textfont=dict(size=11, family="JetBrains Mono, monospace", color="#e8e9ed"),
                colorbar=dict(title="P(%)", thickness=10, len=0.7,
                              tickfont=dict(size=8, family="JetBrains Mono, monospace")),
            ))
            fig_tm.add_shape(type="rect",
                x0=-0.5, x1=3.5, y0=current_r-0.5, y1=current_r+0.5,
                line=dict(color=_REGIME_COLORS[current_r], width=2),
                fillcolor="rgba(0,0,0,0)",
            )
            fig_tm.update_layout(
                template="purdue", height=260,
                paper_bgcolor="#111111", plot_bgcolor="#111111",
                font=dict(color="#e8e9ed"),
                margin=dict(l=90, r=60, t=20, b=80),
                xaxis=dict(title="To Regime", tickfont=dict(size=10, color="#8890a1"), rangeslider=dict(visible=False)),
                yaxis=dict(title="From Regime", tickfont=dict(size=10, color="#8890a1")),
            )
            _chart(fig_tm)
            _insight_note(
                "A probability matrix showing the likelihood of moving from one regime to another. "
                "A high probability of staying in Crisis means the model sees no immediate relief. "
                "A high probability of transitioning from Elevated to Normal is a positive signal."
            )

            # 3-col stats row below the heatmap
            ms1, ms2, ms3 = st.columns(3)
            with ms1:
                _label("Steady State")
                for i in range(4):
                    c = _REGIME_COLORS[i]
                    pct = steady[i] * 100
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;align-items:center;'
                        f'margin-bottom:3px">'
                        f'<span style="{_F}font-size:0.64rem;color:{c};font-weight:600">{_REGIME_NAMES[i]}</span>'
                        f'<span style="font-family:JetBrains Mono,monospace;font-size:0.64rem;font-weight:700">{pct:.1f}%</span>'
                        f'</div>'
                        f'<div style="background:#2a2a2a;height:3px;border-radius:2px;margin-bottom:5px">'
                        f'<div style="width:{pct:.0f}%;height:3px;background:{c};border-radius:2px"></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            with ms2:
                _label("Days to Crisis")
                for i in range(4):
                    v = mfpt.get(i, float("nan"))
                    disp = "0 (in Crisis)" if i == 3 else (f"{v:.0f}d" if v < 5000 else ">5,000d")
                    is_cur = f"border-left:2px solid {_REGIME_COLORS[i]};padding-left:4px;" if i == current_r else ""
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;{is_cur}">'
                        f'<span style="{_F}font-size:0.64rem;color:{_REGIME_COLORS[i]}">{_REGIME_NAMES[i]}</span>'
                        f'<span style="font-family:JetBrains Mono,monospace;font-size:0.64rem;font-weight:700">{disp}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            with ms3:
                _label("Avg Run Length")
                for i in run_stats.index:
                    if i not in _REGIME_NAMES:
                        continue
                    avg_d = float(run_stats.loc[i, "mean"])
                    max_d = int(run_stats.loc[i, "max"])
                    st.markdown(
                        f'<div style="margin-bottom:4px">'
                        f'<div style="display:flex;justify-content:space-between">'
                        f'<span style="{_F}font-size:0.64rem;color:{_REGIME_COLORS[i]}">{_REGIME_NAMES[i]}</span>'
                        f'<span style="font-family:JetBrains Mono,monospace;font-size:0.64rem;font-weight:700">{avg_d:.1f}d avg</span>'
                        f'</div>'
                        f'<div style="{_F}font-size:0.58rem;color:#6b7280">max {max_d}d</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            p_stay   = float(trans.loc[current_r, current_r]) * 100
            p_crisis = float(trans.loc[current_r, 3]) * 100
            _panel_note(
                f"Current: <b style='color:{_REGIME_COLORS[current_r]}'>{_REGIME_NAMES[current_r]}</b> · "
                f"Stay probability: <b>{p_stay:.1f}%</b> · "
                f"Transition to Crisis: <b>{p_crisis:.1f}%</b>"
            )

    # ── Equity-Bond Correlation Regime Note ───────────────────────────────────
    if not fi_r.empty and "US 20Y+ Treasury (TLT)" in fi_r.columns and "S&P 500" in eq_r.columns:
        try:
            _tlt_spx_rc = rolling_correlation(all_r["S&P 500"], all_r["US 20Y+ Treasury (TLT)"], 63)
            _latest_tlt_spx = float(_tlt_spx_rc.dropna().iloc[-1]) if not _tlt_spx_rc.dropna().empty else None
            if _latest_tlt_spx is not None:
                _corr_signal = "POSITIVE" if _latest_tlt_spx > 0 else "NEGATIVE"
                _corr_color = "#c0392b" if _latest_tlt_spx > 0.1 else "#2e7d32" if _latest_tlt_spx < -0.1 else "#e67e22"
                _corr_interp = (
                    "60/40 HEDGE BREAKDOWN: equities and bonds are both falling together. "
                    "This is a classic inflationary or fiscal dominance regime where traditional diversification fails."
                    if _latest_tlt_spx > 0.1 else
                    "60/40 HEDGE WORKING: bonds rising when equities fall (negative correlation). "
                    "Traditional portfolio diversification is functioning normally."
                    if _latest_tlt_spx < -0.1 else
                    "NEUTRAL: equity-bond correlation near zero. No clear flight-to-quality signal."
                )
                st.markdown(
                    f'<div style="border:1px solid #2a2a2a;border-left:4px solid {_corr_color};'
                    f'border-radius:0 6px 6px 0;padding:0.75rem 1rem;background:#1c1c1c;margin:0.8rem 0">'
                    f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.56rem;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:0.12em;color:{_corr_color};margin-bottom:4px">'
                    f'Equity-Bond Correlation Regime</div>'
                    f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.80rem;font-weight:700;'
                    f'color:#e8e9ed;margin-bottom:4px">S&P 500 / TLT 63d Correlation: '
                    f'<span style="color:{_corr_color}">{_latest_tlt_spx:+.3f}</span> ({_corr_signal})</div>'
                    f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;color:#b8b8b8;line-height:1.55">'
                    f'{_corr_interp}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        except Exception:
            pass

    _page_conclusion(
        "Regime & Diversification Signal",
        "If correlation is elevated and the Markov model assigns high probability of remaining "
        "in the high-coupling regime, treat your equity and commodity positions as one risk - "
        "not two. Hedge accordingly or reduce gross exposure. Low correlation with a high "
        "probability of staying low is the green light for diversified positioning. "
        "Monitor the equity-bond correlation regime above: when TLT and S&P 500 become positively "
        "correlated, the traditional 60/40 hedge breaks down and requires alternative protection."
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

            _n_obs = len(eq_r.dropna(how="all"))
            _current_regime = regimes.iloc[-1] if not regimes.empty else "Unknown"
            _regime_changed = (
                len(regimes) > 5 and regimes.iloc[-1] != regimes.iloc[-6]
            ) if not regimes.empty else False

            # Top correlated pairs from cross-asset matrix
            try:
                import numpy as np
                _cm = eq_r.join(cmd_r).corr().abs()
                _max_corr = float(_cm.where(
                    ~(_cm == 1.0)
                ).stack().max())
            except Exception:
                _max_corr = None

            _cqo_ctx = {
                "n_obs":           _n_obs,
                "date_range":      f"{start} to {end}",
                "model":           "Rolling Pearson correlation + DCC-GARCH + Markov regime",
                "regime":          str(_current_regime),
                "regime_change":   _regime_changed,
                "max_correlation": _max_corr,
                "assumption_count": 3,
                "notes": [
                    "Rolling window correlation is non-stationary — window length (60d default) is arbitrary",
                    "Pearson correlation assumes linearity; equity-commodity relationships are often non-linear during crises",
                    "DCC-GARCH assumes elliptical distributions — tail dependence is underestimated",
                    "Markov regime model uses 2 states only — real market has continuous regime transitions",
                    "Correlation ≠ causation — no causal inference drawn from this page",
                ],
            }
            with st.spinner("CQO auditing correlation methodology…"):
                _cqo_result = _cqo_run(_cqo_ctx, _provider, _api_key, page="Correlation Analysis")
            if _cqo_result.get("narrative"):
                st.markdown("---")
                render_agent_output_block("quality_officer", _cqo_result)
    except Exception:
        pass

    _page_footer()
