"""
Early-Warning Radar - critical slowing down before a regime flips.

Imports tipping-point theory from ecology/climate (Scheffer 2009; Dakos 2012):
as a system nears a critical transition it recovers ever more slowly from
shocks, leaving two statistical fingerprints - rising lag-1 autocorrelation and
rising variance - that appear BEFORE it flips. This page runs those indicators on
a market driver (average cross-asset correlation or D-Y connectedness) to flag an
impending correlation-regime transition ahead of the terminal's Markov classifier,
and validates the lead time honestly against realized flips (false alarms shown).
Engine: src/analysis/critical_slowing.py.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_returns
from src.analysis.correlations import (
    average_cross_corr_series, detect_correlation_regime,
)
from src.analysis import critical_slowing as cs
from src.ui.shared import (
    _chart, _page_intro, _definition_block, _page_header, _page_footer,
)

_GOLD  = "#CFB991"
_RED   = "#c0392b"
_GREEN = "#27ae60"
_AMBER = "#e67e22"
_MUTED = "#8890a1"
_BLUE  = "#4a90d9"

_HISTORY_START = "2008-01-01"
_REGIME_NAMES = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis"}
_REGIME_COL   = {0: "#3d566e", 1: _MUTED, 2: _AMBER, 3: _RED}
_STATUS_COL   = {
    "TRANSITION RISK BUILDING": _RED,
    "WATCH - MIXED SIGNAL":     _AMBER,
    "STABLE":                   _GREEN,
    "INSUFFICIENT DATA":        _MUTED,
}


def _section_label(text: str) -> None:
    st.markdown(
        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;font-weight:700;'
        f'letter-spacing:0.12em;text-transform:uppercase;color:#e8e9ed;'
        f'border-bottom:1px solid #1e1e1e;padding-bottom:0.35rem;margin:1.4rem 0 0.8rem">'
        f'{text}</div>',
        unsafe_allow_html=True,
    )


# ── Cached data builders ─────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False, max_entries=3)
def _avg_corr_driver(start: str, end: str, window: int) -> pd.Series:
    eq_r, cmd_r = load_returns(start, end)
    return average_cross_corr_series(eq_r, cmd_r, window=window)


@st.cache_data(ttl=86400, show_spinner=False, max_entries=2)
def _dy_driver(start: str, end: str) -> pd.Series:
    """Rolling D-Y total spillover connectedness - slow first run, cached daily."""
    from src.analysis.spillover import rolling_diebold_yilmaz
    eq_r, cmd_r = load_returns(start, end)
    cols = [c for c in ("S&P 500", "DAX", "Nikkei 225",
                        "WTI Crude Oil", "Gold", "Copper")
            if c in pd.concat([eq_r, cmd_r], axis=1).columns]
    combined = pd.concat([eq_r, cmd_r], axis=1)[cols]
    df = rolling_diebold_yilmaz(combined, window=200, step=5, lag_order=2)
    s = df["total_spillover"] if "total_spillover" in df.columns else pd.Series(dtype=float)
    # Reindex to daily and forward-fill so it aligns with the daily regime series.
    if s.empty:
        return s
    daily = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="B")).ffill()
    daily.name = "connectedness"
    return daily


@st.cache_data(ttl=3600, show_spinner=False, max_entries=3)
def _regime_series(start: str, end: str, corr_window: int) -> pd.Series:
    eq_r, cmd_r = load_returns(start, end)
    acc = average_cross_corr_series(eq_r, cmd_r, window=corr_window)
    return detect_correlation_regime(acc)


_AI_SYSTEM = (
    "You are the AI Transition Analyst embedded in the Cross-Asset Spillover Monitor "
    "at Purdue University Daniels School of Business. You read a critical-slowing-down "
    "early-warning signal (rising lag-1 autocorrelation + rising variance ahead of a "
    "correlation-regime flip) and explain, in plain language, what it currently implies. "
    "You produce research analysis for an academic dashboard - not investment advice. "
    "Critical slowing down is a LEADING but NOISY precursor: never state a flip is coming, "
    "only that transition risk is elevated or not, and always respect the historical "
    "false-alarm rate you are given. Distinguish evidence from inference."
)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=8)
def _ai_transition_read(context_str: str, provider: str, api_key: str) -> str:
    """Plain-language read of the current early-warning signal. Cached 1 hour,
    keyed by the context string so it only regenerates when the signal changes."""
    prompt = (
        f"CURRENT EARLY-WARNING STATE (live):\n{context_str}\n\n"
        "Write a 3–4 sentence read of this signal covering: "
        "1) whether critical slowing down is genuinely present right now "
        "(both AR(1) autocorrelation AND variance rising) or only partial, "
        "2) how much weight it deserves given the historical hit rate, median lead "
        "time, and false-alarm rate shown above - be explicit that this is a noisy "
        "precursor, not a forecast, "
        "3) one specific thing to watch that would confirm or kill the signal. "
        "Be quantitative and terse. Do not give trade advice.\n\n"
        "End with these labeled lines:\n"
        "VERDICT: [one line - elevated transition risk / mixed / stable, and why]\n"
        "INVALIDATED IF: [what reading would contradict this]"
    )
    try:
        if provider == "anthropic":
            import anthropic as _ant
            client = _ant.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=380,
                system=_AI_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        else:
            from openai import OpenAI as _OAI
            client = _OAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o", max_tokens=380, temperature=0.2,
                messages=[{"role": "system", "content": _AI_SYSTEM},
                          {"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
    except Exception:
        return ""   # silent - section is skipped rather than surfacing API errors


# ── Charts ───────────────────────────────────────────────────────────────────

def _radar_gauge(composite: float, threshold: float) -> go.Figure:
    col = _RED if composite >= threshold else (_AMBER if composite >= threshold - 15 else _GREEN)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=composite,
        number=dict(font=dict(family="JetBrains Mono, monospace", size=34, color=col),
                    suffix=""),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=_MUTED,
                      tickfont=dict(size=8, family="JetBrains Mono, monospace")),
            bar=dict(color=col, thickness=0.28),
            bgcolor="#080808", borderwidth=0,
            steps=[
                dict(range=[0, threshold - 15], color="rgba(39,174,96,0.15)"),
                dict(range=[threshold - 15, threshold], color="rgba(230,126,34,0.18)"),
                dict(range=[threshold, 100], color="rgba(192,57,43,0.20)"),
            ],
            threshold=dict(line=dict(color=_GOLD, width=2.5), thickness=0.85, value=threshold),
        ),
    ))
    fig.update_layout(
        height=210, paper_bgcolor="#000",
        font=dict(family="DM Sans, sans-serif", color="#c8c8c8"),
        margin=dict(l=24, r=24, t=18, b=6),
        title=dict(text="Composite early-warning level", x=0.5, xanchor="center",
                   font=dict(family="JetBrains Mono, monospace", size=10, color=_MUTED)),
    )
    return fig


def _driver_chart(ews: pd.DataFrame, regime: pd.Series, flips: pd.Series,
                  threshold: float, driver_name: str) -> go.Figure:
    fig = go.Figure()
    # Composite warning level (right axis), shaded above threshold
    fig.add_trace(go.Scatter(
        x=ews.index, y=ews["composite"], name="Warning level",
        line=dict(color=_GOLD, width=1.3), yaxis="y2",
        hovertemplate="%{x|%Y-%m-%d}: warning %{y:.0f}<extra></extra>",
    ))
    fig.add_hline(y=threshold, line=dict(color=_RED, width=1, dash="dot"),
                  yref="y2", opacity=0.5)
    # Driver level (left axis)
    fig.add_trace(go.Scatter(
        x=ews.index, y=ews["driver"], name=driver_name,
        line=dict(color=_BLUE, width=1.0), opacity=0.75,
        hovertemplate="%{x|%Y-%m-%d}: " + driver_name + " %{y:.3f}<extra></extra>",
    ))
    # Realized regime flips into stress - vertical markers
    for fdate, target in flips.items():
        fig.add_vline(x=fdate, line=dict(color=_REGIME_COL.get(int(target), _MUTED),
                                         width=0.8, dash="dot"), opacity=0.5)
    fig.update_layout(
        template="plotly_dark", height=340, paper_bgcolor="#000", plot_bgcolor="#080808",
        font=dict(family="DM Sans, sans-serif", color="#c8c8c8", size=10),
        title=dict(text=f"Warning level vs {driver_name} - dotted lines = realized flips into Elevated/Crisis",
                   x=0, xanchor="left",
                   font=dict(family="JetBrains Mono, monospace", size=11, color=_MUTED)),
        xaxis=dict(showgrid=False),
        yaxis=dict(title=driver_name, showgrid=True, gridcolor="#1a1a1a"),
        yaxis2=dict(title="Warning level", overlaying="y", side="right",
                    range=[0, 105], showgrid=False),
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=9),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=52, r=52, t=34, b=52),
    )
    return fig


def _indicator_chart(ews: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ews.index, y=ews["ar1_z"], name="AR(1) autocorrelation (z)",
        line=dict(color=_AMBER, width=1.2),
        hovertemplate="%{x|%Y-%m-%d}: AR(1) %{y:+.2f}σ<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ews.index, y=ews["var_z"], name="Variance (z)",
        line=dict(color=_RED, width=1.2),
        hovertemplate="%{x|%Y-%m-%d}: variance %{y:+.2f}σ<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="#2a2a2a", width=1))
    fig.update_layout(
        template="plotly_dark", height=300, paper_bgcolor="#000", plot_bgcolor="#080808",
        font=dict(family="DM Sans, sans-serif", color="#c8c8c8", size=10),
        title=dict(text="Critical-slowing-down indicators - both rising together is the signal",
                   x=0, xanchor="left",
                   font=dict(family="JetBrains Mono, monospace", size=11, color=_MUTED)),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Std. deviations vs own history", showgrid=True, gridcolor="#1a1a1a"),
        legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=9),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=52, r=16, t=34, b=52),
    )
    return fig


def _stat_card(label: str, value: str, col: str, sub: str = "") -> str:
    sub_html = (f'<div style="font-size:0.52rem;color:{_MUTED};margin-top:.15rem">{sub}</div>'
                if sub else "")
    return (
        f'<div style="flex:1;background:#080808;border:1px solid #1e1e1e;'
        f'border-top:2px solid {col};padding:.6rem .8rem">'
        f'<div style="font-size:0.53rem;letter-spacing:.1em;text-transform:uppercase;'
        f'color:{_MUTED}">{label}</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:1.15rem;'
        f'font-weight:700;color:{col};margin-top:.2rem">{value}</div>{sub_html}</div>'
    )


# ── Page ─────────────────────────────────────────────────────────────────────

def page_early_warning(start: str, end: str, fred_key: str | None = None) -> None:
    _page_header(
        "Early-Warning Radar - Critical Slowing Down",
        "Tipping-point theory · rising autocorrelation + variance · lead-time validated · false alarms shown",
    )
    _page_intro(
        "Borrowed from ecology and climate science (Scheffer et al., <em>Nature</em> 2009; "
        "Dakos et al., <em>PLoS ONE</em> 2012): as a system approaches a critical transition it "
        "recovers more and more slowly from small shocks - <strong>critical slowing down</strong>. "
        "Two statistical fingerprints of that appear <em>before</em> the system actually flips: "
        "lag-1 autocorrelation climbs toward 1 (memory lengthens) and variance rises (the basin "
        "of attraction flattens). This page runs those indicators on a market driver and asks "
        "whether they build ahead of a correlation-regime flip that the Markov classifier "
        "(Correlation page) only confirms after the fact. The honest catch - CSD is a noisy "
        "leading signal, so the validation panel below reports false alarms, not just hits."
    )

    # ── Controls ────────────────────────────────────────────────────────────
    _section_label("Signal configuration")
    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
    driver_choice = c1.selectbox(
        "Driver series", ["Average cross-correlation", "D-Y connectedness index"],
        key="ew_driver",
        help="What the CSD indicators run on. Cross-correlation is daily and fast. "
             "Connectedness (rolling Diebold-Yilmaz) is structural and slower to compute "
             "(~30–60s first run, then cached for the day).",
    )
    window = c2.slider("Indicator window (days)", 40, 120, 60, 10, key="ew_win",
                       help="Rolling window for AR(1) and variance.")
    detrend_bw = c3.slider("Detrend bandwidth", 15, 60, 30, 5, key="ew_bw",
                           help="Rolling-mean width removed before measuring fluctuations.")
    threshold = c4.slider("Alert threshold", 50, 85, 62, 1, key="ew_thr",
                          help="Warning level above which an alert fires. Higher = fewer "
                               "false alarms but less lead time.")

    today = datetime.date.today()
    corr_window = 60

    # ── Build driver + EWS ──────────────────────────────────────────────────
    with st.spinner("Computing critical-slowing-down indicators…"):
        if driver_choice.startswith("Average"):
            driver = _avg_corr_driver(_HISTORY_START, str(today), corr_window)
            driver_name = "Avg cross-correlation"
        else:
            driver = _dy_driver(_HISTORY_START, str(today))
            driver_name = "D-Y connectedness"
        regime = _regime_series(_HISTORY_START, str(today), corr_window)
        ews = cs.compute_ews(driver, detrend_bw=detrend_bw, window=window)

    if ews.empty:
        st.warning("Insufficient history to compute early-warning indicators for this driver.")
        _page_footer()
        return

    taus = cs.trend_tau(ews, window=max(30, window // 2))
    reading = cs.latest_reading(ews, taus, alert_threshold=threshold)
    flips = cs.detect_regime_flips(regime)
    ev = cs.evaluate_lead_time(ews, regime, alert_threshold=threshold)

    # ── Headline radar ──────────────────────────────────────────────────────
    status = reading["status"]
    s_col = _STATUS_COL.get(status, _MUTED)
    st.markdown(
        f'<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-left:4px solid {s_col};'
        f'padding:.7rem 1rem;margin:.4rem 0 .9rem;display:flex;align-items:center;gap:1rem">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-weight:700;font-size:1rem;'
        f'color:{s_col}">{status}</span>'
        f'<span style="font-size:0.62rem;color:{_MUTED}">as of '
        f'<b style="color:{_GOLD}">{reading["date"].date()}</b> · driver: {driver_name} · '
        f'Markov regime currently: <b style="color:{_REGIME_COL.get(int(regime.iloc[-1]),_MUTED)}">'
        f'{_REGIME_NAMES.get(int(regime.iloc[-1])," - ")}</b></span></div>',
        unsafe_allow_html=True,
    )

    lc, rc = st.columns([1, 1.9])
    with lc:
        _chart(_radar_gauge(reading["composite"], threshold))
    with rc:
        def _arrow(t):
            if t is None or np.isnan(t):
                return f'<span style="color:{_MUTED}">–</span>'
            if t > 0.1:
                return f'<span style="color:{_RED}">▲ rising ({t:+.2f})</span>'
            if t < -0.1:
                return f'<span style="color:{_GREEN}">▼ falling ({t:+.2f})</span>'
            return f'<span style="color:{_MUTED}">→ flat ({t:+.2f})</span>'
        cards = "".join([
            _stat_card("AR(1) autocorr", f'{reading["ar1"]:.2f}', _AMBER,
                       f'{reading["ar1_z"]:+.2f}σ vs history'),
            _stat_card("Variance (z)", f'{reading["var_z"]:+.2f}σ', _RED,
                       "vs own history"),
        ])
        st.markdown(
            f'<div style="display:flex;gap:.6rem;margin-bottom:.6rem">{cards}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="background:#080808;border:1px solid #1e1e1e;padding:.7rem .9rem;'
            f'font-size:0.72rem;color:#c8c8c8;line-height:1.9">'
            f'AR(1) autocorrelation trend: {_arrow(reading["ar1_tau"])}<br>'
            f'Variance trend: {_arrow(reading["var_tau"])}<br>'
            f'<span style="color:{_MUTED};font-size:0.62rem">Critical slowing down requires '
            f'<b>both</b> rising together. One alone is not a transition signal.</span></div>',
            unsafe_allow_html=True,
        )

    # ── Time series ─────────────────────────────────────────────────────────
    _section_label("Warning level over history")
    _chart(_driver_chart(ews, regime, flips, threshold, driver_name))
    _chart(_indicator_chart(ews))

    # ── Validation scorecard ────────────────────────────────────────────────
    _section_label("Did it actually lead the flips? - honest scorecard")
    hr = ev["hit_rate"]
    far = ev["false_alarm_rate"]
    ml = ev["median_lead"]
    hr_col = _GREEN if (hr == hr and hr >= 0.5) else (_AMBER if (hr == hr and hr >= 0.3) else _RED)
    far_col = _GREEN if (far == far and far <= 0.4) else (_AMBER if (far == far and far <= 0.6) else _RED)
    cards = "".join([
        _stat_card("Flips caught", f'{ev["n_caught"]}/{ev["n_flips"]}', hr_col,
                   f'{hr*100:.0f}% hit rate' if hr == hr else " - "),
        _stat_card("Median lead", f'{ml:.0f}d' if ml == ml else " - ", _GOLD,
                   "calendar days before flip"),
        _stat_card("False-alarm rate", f'{far*100:.0f}%' if far == far else " - ", far_col,
                   f'{ev["n_false_alarms"]}/{ev["n_alert_episodes"]} alert episodes'),
        _stat_card("Realized flips", f'{ev["n_flips"]}', _MUTED,
                   "into Elevated/Crisis since 2008"),
    ])
    st.markdown(
        f'<div style="display:flex;gap:.6rem;margin-bottom:.8rem">{cards}</div>',
        unsafe_allow_html=True,
    )

    # Recent flips table
    recent = [f for f in ev["flips"] if f.get("date")][-8:]
    if recent:
        rows = ""
        for f in reversed(recent):
            caught = f["caught"]
            v_col = _GREEN if caught else _RED
            lead = f'{f["lead_days"]}d' if f["lead_days"] is not None else " - "
            rows += (
                f'<tr>'
                f'<td style="padding:.3rem .6rem;font-family:\'JetBrains Mono\',monospace;'
                f'font-size:.7rem;color:{_GOLD}">{f["date"].date()}</td>'
                f'<td style="padding:.3rem .6rem;font-size:.66rem;text-align:center;'
                f'color:{_REGIME_COL.get(f["regime"],_MUTED)}">{_REGIME_NAMES.get(f["regime"]," - ")}</td>'
                f'<td style="padding:.3rem .6rem;font-family:\'JetBrains Mono\',monospace;'
                f'font-size:.7rem;text-align:right;color:#c8c8c8">{lead}</td>'
                f'<td style="padding:.3rem .6rem;font-size:.66rem;font-weight:700;'
                f'text-align:center;color:{v_col}">{"WARNED" if caught else "MISSED"}</td>'
                f'</tr>'
            )
        hdr = "".join(
            f'<th style="padding:.35rem .6rem;font-size:.53rem;letter-spacing:.1em;'
            f'text-transform:uppercase;color:{_MUTED};text-align:{a}">{h}</th>'
            for h, a in [("Flip date", "left"), ("Into regime", "center"),
                         ("Lead time", "right"), ("Signal", "center")]
        )
        st.markdown(
            f'<div style="overflow:auto;border:1px solid #1e1e1e;margin-bottom:.8rem">'
            f'<table style="width:100%;border-collapse:collapse;background:#080808">'
            f'<thead><tr style="border-bottom:1px solid #1e1e1e">{hdr}</tr></thead>'
            f'<tbody>{rows}</tbody></table></div>',
            unsafe_allow_html=True,
        )

    # ── AI read on the signal ───────────────────────────────────────────────
    _ai_key, _ai_provider = "", ""
    try:
        _sec = st.secrets.get("keys", {})
        if _sec.get("anthropic_api_key", ""):
            _ai_key, _ai_provider = _sec["anthropic_api_key"], "anthropic"
        elif _sec.get("openai_api_key", ""):
            _ai_key, _ai_provider = _sec["openai_api_key"], "openai"
    except Exception:
        pass

    if _ai_key:
        def _tw(t):
            return "n/a" if t is None else ("rising" if t > 0.1 else "falling" if t < -0.1 else "flat")
        _ctx = (
            f"Driver: {driver_name}\n"
            f"Engine status: {reading['status']}\n"
            f"Composite warning level: {reading['composite']:.0f}/100 (alert bar {threshold:.0f})\n"
            f"Current Markov correlation regime: {_REGIME_NAMES.get(int(regime.iloc[-1]),'?')}\n"
            f"AR(1) autocorrelation: {reading['ar1']:.2f} ({reading['ar1_z']:+.2f} sd vs history), "
            f"trend {_tw(reading.get('ar1_tau'))}\n"
            f"Variance: {reading['var_z']:+.2f} sd vs history, trend {_tw(reading.get('var_tau'))}\n"
            f"Both CSD indicators rising together: {reading.get('both_rising')}\n"
            f"--- Historical validation of this driver+threshold ---\n"
            f"Flips caught: {ev['n_caught']}/{ev['n_flips']} "
            f"(hit rate {ev['hit_rate']*100:.0f}%)\n"
            f"Median lead time: {ev['median_lead']:.0f} calendar days\n"
            f"False-alarm rate: {ev['false_alarm_rate']*100:.0f}% "
            f"({ev['n_false_alarms']}/{ev['n_alert_episodes']} alert episodes did NOT precede a flip)"
        ) if ev["hit_rate"] == ev["hit_rate"] else ""

        _read = _ai_transition_read(_ctx, _ai_provider, _ai_key) if _ctx else ""
        if _read:
            _section_label("AI read on the signal")
            st.markdown(
                f'<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-left:3px solid {_GOLD};'
                f'padding:.85rem 1.05rem;margin-bottom:.9rem">'
                f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.78rem;color:#e8e9ed;'
                f'line-height:1.75">{_read.replace(chr(10), "<br>")}</div>'
                f'<div style="font-size:0.52rem;color:{_MUTED};margin-top:.5rem;'
                f'border-top:1px solid #1e1e1e;padding-top:.4rem">AI Transition Analyst · reads the '
                f'engine output and its track record · research analysis, not investment advice</div></div>',
                unsafe_allow_html=True,
            )

    _definition_block(
        "What this engine is - and is not",
        "The indicators are computed exactly as in the tipping-point literature: detrend the "
        "driver with a centred rolling mean, then measure lag-1 autocorrelation and variance in a "
        "rolling window; the warning is their upward trend (Kendall's tau), squashed into a 0–100 "
        "composite. Every value at day t uses data ≤ t except the expanding-window standardization, "
        "which is the same in-sample normalization the terminal's other analog engines use. This is "
        "a genuinely LEADING but NOISY signal: the false-alarm rate above is deliberately shown "
        "because critical slowing down produces precursors that sometimes fizzle without a flip. "
        "Raising the alert threshold cuts false alarms at the cost of lead time - the slider lets you "
        "see that trade-off. Treat it as a heightened-vigilance flag that stacks with the Correlation "
        "regime model and Pattern Memory, not a standalone forecast."
    )

    _page_footer()
