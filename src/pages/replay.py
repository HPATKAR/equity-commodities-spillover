"""
Replay Mode — point-in-time case studies.

Points the whole terminal at a past date (preset: the tracked conflict
breakouts) and shows what it WOULD have said then — GRS, regime, transmission
channels, trade theses — computed strictly from data available up to that
timestamp, next to what actually happened.

Cutoff enforcement lives in src/analysis/replay.py (pit_slice / pit_assert /
LookaheadError). This page only renders the frozen call and, separately, the
post-cutoff outcome.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_returns
from src.ui.shared import (
    _chart, _page_intro, _definition_block, _page_header, _page_footer,
)
from src.analysis.replay import (
    LookaheadError, replay_presets, terminal_call,
    actual_outcome, trade_leg_outcomes, OUTCOME_HORIZONS,
)

_GOLD = "#CFB991"
_RED = "#c0392b"
_GREEN = "#27ae60"
_MUTED = "#8890a1"

_LOOKBACK_YEARS = 4      # history window feeding the PIT computation


def _section_label(text: str) -> None:
    st.markdown(
        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;font-weight:700;'
        f'letter-spacing:0.12em;text-transform:uppercase;color:#e8e9ed;'
        f'border-bottom:1px solid #1e1e1e;padding-bottom:0.35rem;margin:1.4rem 0 0.8rem">'
        f'{text}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=3600, show_spinner=False, max_entries=8)
def _replay_call_cached(cutoff_iso: str, focus: str) -> dict:
    """Load history ending at cutoff and compute the frozen terminal call."""
    cutoff = datetime.date.fromisoformat(cutoff_iso)
    start = str(cutoff - datetime.timedelta(days=int(365.25 * _LOOKBACK_YEARS)))
    # Loader is asked for end=cutoff but terminal_call re-slices + asserts —
    # the fetch layer is never trusted (see replay.py, THE ONE RULE).
    eq_r, cmd_r = load_returns(start, str(cutoff))
    return terminal_call(eq_r, cmd_r, cutoff, focus_conflict=focus or None)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=8)
def _outcome_cached(cutoff_iso: str, conflict_id: str) -> dict:
    return actual_outcome(datetime.date.fromisoformat(cutoff_iso), conflict_id or None)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=8)
def _trade_outcomes_cached(_trades: list, cutoff_iso: str, cache_key: str) -> list:
    # _trades is unhashable (leading underscore skips hashing); cache_key carries
    # the focus-conflict identity so different replays at the same date don't collide.
    return trade_leg_outcomes(_trades, datetime.date.fromisoformat(cutoff_iso), horizon=30)


def _grs_chart(series: pd.Series, cutoff: datetime.date, grs: float) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=series.index, y=series.values, mode="lines",
        line=dict(color=_GOLD, width=1.6),
        hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}<extra></extra>",
    ))
    fig.add_hline(y=60, line_color=_RED, line_width=1, line_dash="dot",
                  annotation_text="stress ≥ 60",
                  annotation_position="top left",
                  annotation_font=dict(size=8, color=_RED))
    fig.add_trace(go.Scatter(
        x=[series.index[-1]], y=[grs], mode="markers+text",
        marker=dict(size=10, color=_RED, symbol="diamond"),
        text=[f"  {grs:.0f} @ cutoff"], textposition="middle right",
        textfont=dict(family="JetBrains Mono, monospace", size=10, color=_RED),
        showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        template="plotly_dark", height=240,
        paper_bgcolor="#000", plot_bgcolor="#080808",
        font=dict(family="DM Sans, sans-serif", color="#c8c8c8", size=10),
        title=dict(text=f"GRS (market-confirmation proxy) — trailing year to {cutoff}",
                   x=0, xanchor="left",
                   font=dict(family="JetBrains Mono, monospace", size=11, color=_MUTED)),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Score (0–100)", showgrid=True, gridcolor="#1a1a1a", range=[0, 100]),
        margin=dict(l=48, r=90, t=34, b=28),
        showlegend=False,
    )
    return fig


def _forward_path_chart(paths: dict[str, pd.Series], cutoff: datetime.date) -> go.Figure:
    palette = {"WTI Crude Oil": "#e67e22", "Gold": _GOLD, "S&P 500": "#4a90d9",
               "Freight (BDRY)": "#8e44ad"}
    fig = go.Figure()
    for name, s in paths.items():
        if s is None or len(s) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=list(range(1, len(s) + 1)), y=s.values, mode="lines",
            name=name, line=dict(color=palette.get(name, _GREEN), width=1.8),
            hovertemplate=f"{name} · day %{{x}}: %{{y:+.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        template="plotly_dark", height=300,
        paper_bgcolor="#000", plot_bgcolor="#080808",
        font=dict(family="DM Sans, sans-serif", color="#c8c8c8", size=10),
        title=dict(text=f"What actually happened — trading days after {cutoff}",
                   x=0, xanchor="left",
                   font=dict(family="JetBrains Mono, monospace", size=11, color=_MUTED)),
        xaxis=dict(title="Trading days after cutoff", showgrid=False),
        yaxis=dict(title="Cumulative return (%)", showgrid=True, gridcolor="#1a1a1a",
                   zeroline=True, zerolinecolor="#2a2a2a"),
        legend=dict(orientation="h", y=-0.28, x=0, font=dict(size=9),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=48, r=16, t=34, b=64),
    )
    return fig


def page_replay(start: str, end: str, fred_key: str | None = None) -> None:
    _page_header("Replay Mode — Point-in-Time Case Studies",
                 "Strict PIT reconstruction · No lookahead · Frozen call vs. actual outcome")
    _page_intro(
        "Point the terminal at a past date and see what it WOULD have said — GRS, "
        "correlation regime, transmission channels, and trade theses — computed strictly "
        "from data available up to that timestamp, then fast-forward to what actually "
        "happened. Every input passes a hard cutoff assertion: a single post-date "
        "observation raises LookaheadError and the replay refuses to render. "
        "One leaked future point would make the terminal look brilliant, and it would be a lie."
    )

    # ── Date selection ─────────────────────────────────────────────────────
    presets = replay_presets()
    today = datetime.date.today()
    usable = [p for p in presets if p["date"] < today - datetime.timedelta(days=7)]
    options = [f"{p['label']} breakout — {p['date']}" for p in usable] + ["Custom date"]
    pick = st.selectbox("Replay date", options, index=0, key="replay_pick")

    if pick == "Custom date":
        cutoff = st.date_input(
            "Custom replay date", value=datetime.date(2022, 2, 24),
            min_value=datetime.date(2010, 1, 1),
            max_value=today - datetime.timedelta(days=7),
            key="replay_custom",
        )
        focus_id = ""
        focus_label = "custom date"
    else:
        p = usable[options.index(pick)]
        cutoff, focus_id, focus_label = p["date"], p["id"], p["label"]

    # ── Frozen terminal call (PIT side) ────────────────────────────────────
    try:
        with st.spinner(f"Reconstructing the terminal as of {cutoff} — strict point-in-time…"):
            call = _replay_call_cached(str(cutoff), focus_id)
    except LookaheadError as e:
        st.error(f"REPLAY ABORTED — {e}")
        st.markdown(
            f'<p style="font-size:0.7rem;color:{_MUTED}">This is the enforcement working as '
            f'designed: post-cutoff data reached a PIT computation and the replay refused to '
            f'render rather than show a contaminated result. See src/analysis/replay.py.</p>',
            unsafe_allow_html=True,
        )
        _page_footer()
        return

    # PIT proof banner — newest observation per source, all ≤ cutoff by assertion
    pit_bits = " · ".join(f"{k}: {v}" for k, v in call["pit"].items())
    st.markdown(
        f'<div style="background:#080808;border:1px solid #1e1e1e;border-left:3px solid {_GREEN};'
        f'padding:.45rem .9rem;margin:.4rem 0 .8rem;font-family:\'JetBrains Mono\',monospace;'
        f'font-size:0.58rem;color:{_MUTED}">'
        f'<b style="color:{_GREEN}">CUTOFF ENFORCED — {cutoff}</b>&nbsp;·&nbsp;'
        f'newest observation per source: {pit_bits}&nbsp;·&nbsp;'
        f'enforced in <b style="color:{_GOLD}">src/analysis/replay.py :: pit_slice / pit_assert</b> '
        f'(LookaheadError on violation — never silent)</div>',
        unsafe_allow_html=True,
    )

    # ── The call ───────────────────────────────────────────────────────────
    _section_label(f"What the terminal said — {focus_label} · {cutoff}")

    m1, m2, m3, m4, m5 = st.columns(5)
    grs = call["grs"]
    grs_col = _RED if grs >= 60 else "#e67e22" if grs >= 45 else _GOLD
    m1.metric("GRS (proxy layer)", f"{grs:.0f} / 100")
    m2.metric("Correlation regime", call["regime_name"])
    m3.metric("Avg eq-cmd corr (60d)", f"{call['avg_corr']:.3f}")
    m4.metric("D-Y spillover index", "n/a" if np.isnan(call.get("dy_total", np.nan))
              else f"{call['dy_total']:.0f}%")
    m5.metric("Top transmitter", call.get("dy_top_transmitter") or "n/a")

    if len(call.get("grs_series", [])) > 20:
        _chart(_grs_chart(call["grs_series"], cutoff, grs))

    # Transmission channels
    ch_rows = "".join(
        f'<tr><td style="padding:.28rem .7rem;font-size:.72rem;color:#c8c8c8">'
        f'{c["commodity"]} → {c["equity"]}</td>'
        f'<td style="padding:.28rem .7rem;font-family:\'JetBrains Mono\',monospace;font-size:.72rem;'
        f'color:{_GOLD};text-align:right">{c["corr_60d"]:+.3f}</td>'
        f'<td style="padding:.28rem .7rem;font-family:\'JetBrains Mono\',monospace;font-size:.72rem;'
        f'color:#c8c8c8;text-align:right">{c["beta_252d"]:+.3f}</td></tr>'
        for c in call["channels"]
    )
    cf_rows = "".join(
        f'<tr><td style="padding:.28rem .7rem;font-size:.72rem;color:#c8c8c8">{r["label"]}'
        + (' <b style="color:#e74c3c">◀ THIS BREAKOUT</b>' if r["id"] == focus_id else "")
        + f'</td>'
        f'<td style="padding:.28rem .7rem;font-family:\'JetBrains Mono\',monospace;font-size:.72rem;'
        f'color:{_GOLD};text-align:right">{r["cis"]:.0f}</td>'
        f'<td style="padding:.28rem .7rem;font-family:\'JetBrains Mono\',monospace;font-size:.72rem;'
        f'color:#c8c8c8;text-align:right">{r["tps"]:.0f}</td>'
        f'<td style="padding:.28rem .7rem;font-size:.68rem;color:{_MUTED};text-align:right">'
        f'{r["escalation"]}</td></tr>'
        for r in sorted(call["conflicts"].values(), key=lambda x: x["cis"], reverse=True)
    )
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(
            f'<div style="overflow:auto;border:1px solid #1e1e1e"><table style="width:100%;'
            f'border-collapse:collapse;background:#080808">'
            f'<thead><tr style="border-bottom:1px solid #1e1e1e">'
            f'<th style="padding:.35rem .7rem;font-size:.55rem;letter-spacing:.12em;text-transform:uppercase;'
            f'color:{_MUTED};text-align:left">Transmission channel</th>'
            f'<th style="padding:.35rem .7rem;font-size:.55rem;letter-spacing:.12em;text-transform:uppercase;'
            f'color:{_MUTED};text-align:right">Corr 60d</th>'
            f'<th style="padding:.35rem .7rem;font-size:.55rem;letter-spacing:.12em;text-transform:uppercase;'
            f'color:{_MUTED};text-align:right">Beta 252d</th></tr></thead>'
            f'<tbody>{ch_rows}</tbody></table></div>',
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            f'<div style="overflow:auto;border:1px solid #1e1e1e"><table style="width:100%;'
            f'border-collapse:collapse;background:#080808">'
            f'<thead><tr style="border-bottom:1px solid #1e1e1e">'
            f'<th style="padding:.35rem .7rem;font-size:.55rem;letter-spacing:.12em;text-transform:uppercase;'
            f'color:{_MUTED};text-align:left">Conflicts known at cutoff</th>'
            f'<th style="padding:.35rem .7rem;font-size:.55rem;letter-spacing:.12em;text-transform:uppercase;'
            f'color:{_MUTED};text-align:right">CIS</th>'
            f'<th style="padding:.35rem .7rem;font-size:.55rem;letter-spacing:.12em;text-transform:uppercase;'
            f'color:{_MUTED};text-align:right">TPS</th>'
            f'<th style="padding:.35rem .7rem;font-size:.55rem;letter-spacing:.12em;text-transform:uppercase;'
            f'color:{_MUTED};text-align:right">Trend</th></tr></thead>'
            f'<tbody>{cf_rows}</tbody></table></div>',
            unsafe_allow_html=True,
        )

    # Trade theses
    trades = call.get("trades", [])
    if trades:
        _section_label("Trade theses the terminal would have generated")
        for t in trades:
            st.markdown(
                f'<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-left:3px solid {_GOLD};'
                f'padding:.6rem .9rem;margin-bottom:.5rem">'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.74rem;font-weight:700;'
                f'color:#e8e9ed">{t.get("name", "")}'
                f'<span style="float:right;font-size:.6rem;color:{_MUTED}">confidence '
                f'{t.get("confidence", 0) * 100:.0f}%</span></div>'
                f'<div style="font-size:.64rem;color:{_MUTED};margin-top:2px">{t.get("trigger", "")}</div>'
                f'<div style="font-size:.62rem;color:#8890a1;margin-top:4px">'
                f'Entry: {t.get("entry", "")}</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No trade theses cleared the CIS ≥ 45 gate at this date.")

    # ══════════════════════════════════════════════════════════════════════
    # Fast-forward — post-cutoff data, structurally separate from the call
    # ══════════════════════════════════════════════════════════════════════
    _section_label("Fast-forward — what actually happened")
    st.markdown(
        f'<p style="font-size:0.62rem;color:{_MUTED};margin:0 0 .6rem">Everything below is '
        f'post-cutoff data, loaded by a separate function (actual_outcome) whose output never '
        f'feeds the computation above. It is displayed for comparison only.</p>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading post-cutoff outcomes…"):
        outcome = _outcome_cached(str(cutoff), focus_id)

    if outcome["assets"]:
        hz = outcome["horizons"]
        head = "".join(
            f'<th style="padding:.35rem .7rem;font-size:.55rem;letter-spacing:.12em;'
            f'text-transform:uppercase;color:{_MUTED};text-align:right">+{h}d</th>'
            for h in hz
        )
        rows = ""
        for a, vals in outcome["assets"].items():
            cells = ""
            for h in hz:
                v = vals.get(h, float("nan"))
                if np.isnan(v):
                    cells += (f'<td style="padding:.28rem .7rem;font-size:.72rem;'
                              f'color:{_MUTED};text-align:right">—</td>')
                else:
                    col = _GREEN if v > 0 else _RED
                    cells += (f'<td style="padding:.28rem .7rem;font-family:\'JetBrains Mono\','
                              f'monospace;font-size:.72rem;color:{col};text-align:right">'
                              f'{v * 100:+.1f}%</td>')
            rows += (f'<tr><td style="padding:.28rem .7rem;font-size:.72rem;color:#c8c8c8">{a}</td>'
                     f'{cells}</tr>')
        st.markdown(
            f'<div style="overflow:auto;border:1px solid #1e1e1e;margin-bottom:.8rem">'
            f'<table style="width:100%;border-collapse:collapse;background:#080808">'
            f'<thead><tr style="border-bottom:1px solid #1e1e1e">'
            f'<th style="padding:.35rem .7rem;font-size:.55rem;letter-spacing:.12em;'
            f'text-transform:uppercase;color:{_MUTED};text-align:left">Actual outcome</th>{head}'
            f'</tr></thead><tbody>{rows}</tbody></table></div>',
            unsafe_allow_html=True,
        )
        _chart(_forward_path_chart(outcome["paths"], cutoff))

    # Trade P&L
    if trades:
        with st.spinner("Pricing trade legs forward…"):
            tr_out = _trade_outcomes_cached(trades, str(cutoff), focus_id)
        if tr_out:
            _section_label("Did the theses pay? (+30 trading days, equal-weight legs)")
            for r in tr_out:
                v_col = {"PAID": _GREEN, "LOST": _RED, "FLAT": "#e67e22", "—": _MUTED}[r["verdict"]]
                legs_txt = " · ".join(
                    f'{l["direction"]} {l["asset"]} '
                    + ("—" if np.isnan(l["ret"]) else f'{l["ret"] * 100:+.1f}%')
                    for l in r["legs"]
                )
                pnl_txt = "—" if np.isnan(r["pnl"]) else f'{r["pnl"] * 100:+.1f}%'
                st.markdown(
                    f'<div style="background:#0d0d0d;border:1px solid #1e1e1e;'
                    f'border-left:3px solid {v_col};padding:.5rem .9rem;margin-bottom:.4rem">'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:.62rem;'
                    f'font-weight:700;color:{v_col}">{r["verdict"]} {pnl_txt}</span>'
                    f'<span style="font-size:.68rem;color:#c8c8c8;margin-left:10px">{r["name"]}</span>'
                    f'<div style="font-size:.6rem;color:{_MUTED};margin-top:2px">{legs_txt}</div></div>',
                    unsafe_allow_html=True,
                )

    # ── Case-study verdict ─────────────────────────────────────────────────
    _section_label("Case-study verdict")
    wti30 = outcome["assets"].get("WTI Crude Oil", {}).get(30, float("nan"))
    gold30 = outcome["assets"].get("Gold", {}).get(30, float("nan"))
    spx30 = outcome["assets"].get("S&P 500", {}).get(30, float("nan"))
    flag_txt = ("flagged stress" if grs >= 60 else
                "was elevated" if grs >= 45 else "did not flag stress")
    tr_paid = sum(1 for r in (tr_out if trades else []) if r["verdict"] == "PAID")
    tr_n = len(tr_out) if trades else 0

    def _fmt(v):
        return "n/a" if np.isnan(v) else f"{v * 100:+.1f}%"

    st.markdown(
        f'<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-top:2px solid {_GOLD};'
        f'padding:.8rem 1rem;font-size:.72rem;color:#c8c8c8;line-height:1.7">'
        f'On <b style="color:{_GOLD}">{cutoff}</b>, the terminal\'s market-confirmation layer '
        f'<b style="color:{grs_col}">{flag_txt}</b> (GRS {grs:.0f}, regime {call["regime_name"]}), '
        f'with <b>{call.get("dy_top_transmitter") or "n/a"}</b> as the top spillover transmitter. '
        f'Over the next 30 trading days: WTI <b>{_fmt(wti30)}</b>, Gold <b>{_fmt(gold30)}</b>, '
        f'S&amp;P 500 <b>{_fmt(spx30)}</b>.'
        + (f' Trade theses: <b>{tr_paid}/{tr_n} paid</b>.' if tr_n else "")
        + f'</div>',
        unsafe_allow_html=True,
    )

    # ── Honesty block ──────────────────────────────────────────────────────
    _definition_block(
        "What is and is not point-in-time here",
        "STRICTLY PIT: all price/return series — sliced to ≤ cutoff and asserted via "
        "pit_slice()/pit_assert() in src/analysis/replay.py; any post-date observation raises "
        "LookaheadError and the replay refuses to render. The GRS shown is the market-confirmation "
        "proxy layer (risk_score_history), the only GRS layer computable from historical data. "
        "NOT REPLAYABLE (excluded, never substituted with today's values): GDELT, ACLED, "
        "PortWatch, EIA, COT, RSS — no historical API snapshots exist. "
        "DISCLOSED HINDSIGHT: the CONFLICTS registry's structural transmission weights were "
        "calibrated recently, with knowledge of how these conflicts played out; conflict CIS/TPS "
        "at the cutoff are therefore structural reconstructions (recency and escalation are "
        "cutoff-aware by stated rule), not live-signal replays. Conflicts that had not started "
        "by the cutoff are excluded entirely. Freight outcome uses BDRY (dry-bulk ETF, 2018+) "
        "as proxy; unavailable before its listing."
    )

    _page_footer()
