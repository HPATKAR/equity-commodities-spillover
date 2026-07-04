"""
Pattern Memory — "the market has seen this before."

Whole-state analog recall: each historical day is compressed into a
fingerprint of the terminal's full configuration (geo events + chokepoint
pressure + spillover structure + correlation regime + market state) and
today's fingerprint is matched against the entire state space.
Engine: src/analysis/fingerprint.py.
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
from src.analysis.fingerprint import (
    BLOCKS, BLOCK_FEATURES, FWD_HORIZONS,
    build_fingerprints, match_fingerprint, match_outcomes,
    base_rates, aggregate_verdicts, active_event_labels,
)

_GOLD = "#CFB991"
_RED = "#c0392b"
_GREEN = "#27ae60"
_AMBER = "#e67e22"
_MUTED = "#8890a1"

_HISTORY_START = "2008-01-01"
_REGIME_NAMES = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis", None: "—"}
_VERDICT_COL = {"GOOD": _GREEN, "MIXED": _AMBER, "BAD": _RED, "—": _MUTED}

_BLOCK_LABELS = {
    "geo": "Geo events", "chokepoint": "Chokepoints",
    "spillover": "Spillover", "regime": "Regime", "market": "Market",
}


def _section_label(text: str) -> None:
    st.markdown(
        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;font-weight:700;'
        f'letter-spacing:0.12em;text-transform:uppercase;color:#e8e9ed;'
        f'border-bottom:1px solid #1e1e1e;padding-bottom:0.35rem;margin:1.4rem 0 0.8rem">'
        f'{text}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=86400, show_spinner=False, max_entries=2)
def _dy_series_cached(start: str, end: str) -> pd.Series:
    """Rolling D-Y total spillover — slow first run, cached for the day."""
    from src.analysis.spillover import rolling_diebold_yilmaz
    eq_r, cmd_r = load_returns(start, end)
    cols = [c for c in ("S&P 500", "DAX", "Nikkei 225",
                        "WTI Crude Oil", "Gold", "Copper")
            if c in pd.concat([eq_r, cmd_r], axis=1).columns]
    combined = pd.concat([eq_r, cmd_r], axis=1)[cols]
    df = rolling_diebold_yilmaz(combined, window=200, step=10, lag_order=2)
    return df["total_spillover"] if "total_spillover" in df.columns else pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=2)
def _fingerprints_cached(start: str, end: str, use_dy: bool):
    eq_r, cmd_r = load_returns(start, end)
    dy = None
    if use_dy:
        try:
            dy = _dy_series_cached(start, end)
        except Exception:
            dy = None
    fp, regimes = build_fingerprints(eq_r, cmd_r, dy_series=dy)
    combined = pd.concat([eq_r, cmd_r], axis=1)
    return fp, regimes, combined


def _today_z_chart(today: dict, features: list[str]) -> go.Figure:
    """Today's fingerprint as z-scores, grouped and colored by block."""
    block_of = {f: b for b, fs in BLOCK_FEATURES.items() for f in fs}
    palette = {"geo": "#e74c3c", "chokepoint": "#8e44ad", "spillover": "#4a90d9",
               "regime": _AMBER, "market": _GOLD}
    feats = [f for f in features if f in today["z"]]
    zs = [today["z"][f] for f in feats]
    cols = [palette.get(block_of.get(f, ""), _MUTED) for f in feats]
    fig = go.Figure(go.Bar(
        x=zs, y=feats, orientation="h", marker_color=cols,
        text=[f"{z:+.1f}σ" for z in zs], textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=9),
        hovertemplate="%{y}: %{x:+.2f}σ vs full history<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark", height=max(320, 22 * len(feats)),
        paper_bgcolor="#000", plot_bgcolor="#080808",
        font=dict(family="DM Sans, sans-serif", color="#c8c8c8", size=10),
        title=dict(text="Today's fingerprint — z-scores vs full history",
                   x=0, xanchor="left",
                   font=dict(family="JetBrains Mono, monospace", size=11, color=_MUTED)),
        xaxis=dict(title="Standard deviations from historical mean",
                   showgrid=True, gridcolor="#1a1a1a",
                   zeroline=True, zerolinecolor="#2a2a2a", zerolinewidth=1.5),
        yaxis=dict(showgrid=False, tickfont=dict(size=9, family="JetBrains Mono, monospace")),
        margin=dict(l=140, r=60, t=32, b=36),
    )
    return fig


def _spaghetti_chart(outcomes: list[dict], combined: pd.DataFrame, asset: str) -> go.Figure:
    fig = go.Figure()
    for o in outcomes:
        pos_arr = np.where(combined.index == o["date"])[0]
        if not len(pos_arr) or asset not in combined.columns:
            continue
        pos = int(pos_arr[0])
        fwd = combined[asset].iloc[pos + 1: pos + 61].fillna(0)
        cum = (np.exp(np.log1p(fwd).cumsum()) - 1.0) * 100
        fig.add_trace(go.Scatter(
            x=list(range(1, len(cum) + 1)), y=cum.values, mode="lines",
            name=f"{o['date'].date()} ({o['verdict']})",
            line=dict(color=_VERDICT_COL[o["verdict"]], width=1.4),
            opacity=0.85,
            hovertemplate=f"{o['date'].date()} · day %{{x}}: %{{y:+.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        template="plotly_dark", height=320,
        paper_bgcolor="#000", plot_bgcolor="#080808",
        font=dict(family="DM Sans, sans-serif", color="#c8c8c8", size=10),
        title=dict(text=f"{asset} — 60 trading days after each analog "
                        f"(colored by verdict)",
                   x=0, xanchor="left",
                   font=dict(family="JetBrains Mono, monospace", size=11, color=_MUTED)),
        xaxis=dict(title="Trading days after analog date", showgrid=False),
        yaxis=dict(title="Cumulative return (%)", showgrid=True, gridcolor="#1a1a1a",
                   zeroline=True, zerolinecolor="#2a2a2a"),
        legend=dict(orientation="h", y=-0.3, x=0, font=dict(size=8),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=48, r=16, t=32, b=70),
    )
    return fig


def _verdict_vs_base_chart(agg: dict, br: dict) -> go.Figure:
    cats = ["GOOD", "MIXED", "BAD"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=f"Today's {agg.get('n', 0)} analogs (similarity-weighted)",
        x=cats, y=[agg.get(c, 0) * 100 for c in cats],
        marker_color=[_GREEN, _AMBER, _RED],
        text=[f"{agg.get(c, 0) * 100:.0f}%" for c in cats], textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=11),
    ))
    fig.add_trace(go.Bar(
        name=f"Base rate — all {br.get('n', 0):,} historical days",
        x=cats, y=[br.get(c, 0) * 100 for c in cats],
        marker_color="rgba(136,144,161,0.45)",
        text=[f"{br.get(c, 0) * 100:.0f}%" for c in cats], textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=11, color=_MUTED),
    ))
    fig.update_layout(
        template="plotly_dark", height=300, barmode="group",
        paper_bgcolor="#000", plot_bgcolor="#080808",
        font=dict(family="DM Sans, sans-serif", color="#c8c8c8", size=10),
        title=dict(text="Where did this fingerprint lead? — analogs vs unconditional base rate",
                   x=0, xanchor="left",
                   font=dict(family="JetBrains Mono, monospace", size=11, color=_MUTED)),
        yaxis=dict(title="Share of outcomes (%)", showgrid=True, gridcolor="#1a1a1a",
                   range=[0, 105]),
        legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=9),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=48, r=16, t=32, b=56),
    )
    return fig


def page_pattern_memory(start: str, end: str, fred_key: str | None = None) -> None:
    _page_header("Pattern Memory — The Market Has Seen This Before",
                 "Whole-state analog recall · Geo + chokepoint + spillover + regime + market · Base-rate honest")
    _page_intro(
        "Every historical day since 2008 is compressed into a fingerprint of the terminal's "
        "full state — active geopolitical events, chokepoint pressure, spillover structure, "
        "correlation regime, and market conditions. Today's fingerprint is matched against the "
        "entire state space at once: a past day only ranks if the geopolitical shape AND the "
        "market shape both rhyme. Not a price-similarity screen (that exists on the Scenario "
        "Engine); this is pattern-memory over the whole configuration, with every verdict shown "
        "against the unconditional base rate."
    )

    # ── Block weights ──────────────────────────────────────────────────────
    _section_label("Fingerprint block weights")
    wcols = st.columns(len(BLOCKS) + 1)
    block_weights = {}
    for col, b in zip(wcols, BLOCKS):
        block_weights[b] = col.slider(_BLOCK_LABELS[b], 0.0, 2.0, 1.0, 0.25,
                                      key=f"pm_w_{b}")
    use_dy = wcols[-1].checkbox("D-Y feature", value=True, key="pm_dy",
                                help="Rolling Diebold-Yilmaz total spillover as a feature. "
                                     "Slow on first run (~30–60s), then cached for the day.")
    if all(v == 0 for v in block_weights.values()):
        st.warning("All block weights are zero — set at least one above zero.")
        _page_footer()
        return

    # ── Build + match ──────────────────────────────────────────────────────
    today = datetime.date.today()
    with st.spinner("Building 18 years of state fingerprints…"):
        fp, regimes, combined = _fingerprints_cached(_HISTORY_START, str(today), use_dy)
    if fp.empty:
        st.warning("Insufficient history to build fingerprints.")
        _page_footer()
        return

    res = match_fingerprint(fp, block_weights=block_weights)
    if not res.get("matches"):
        st.warning("Not enough candidate history for matching.")
        _page_footer()
        return
    outcomes = match_outcomes(res, combined, regimes)
    br = base_rates(combined, regimes)
    agg = aggregate_verdicts(outcomes)

    st.markdown(
        f'<div style="background:#080808;border:1px solid #1e1e1e;border-left:3px solid {_GOLD};'
        f'padding:.4rem .9rem;margin:.4rem 0 .8rem;font-family:\'JetBrains Mono\',monospace;'
        f'font-size:0.58rem;color:{_MUTED}">'
        f'FINGERPRINT AS OF <b style="color:{_GOLD}">{res["today"]["date"].date()}</b>'
        f'&nbsp;·&nbsp;{len(res["features"])} features · 5 blocks · '
        f'{res["n_candidates"]:,} candidate days searched · matches non-overlapping (±40d), '
        f'trailing 63d excluded</div>',
        unsafe_allow_html=True,
    )

    # ── Today's fingerprint ────────────────────────────────────────────────
    _chart(_today_z_chart(res["today"], res["features"]))

    # ── Match table ────────────────────────────────────────────────────────
    _section_label("Closest historical configurations")
    rows = ""
    for o in outcomes:
        events = ", ".join(active_event_labels(o["date"].date())) or "quiet"
        r = o["returns"]
        v_col = _VERDICT_COL[o["verdict"]]

        def _pct(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return f'<span style="color:{_MUTED}">—</span>'
            c = _GREEN if v > 0 else _RED
            return f'<span style="color:{c}">{v * 100:+.1f}%</span>'

        rows += (
            f'<tr>'
            f'<td style="padding:.3rem .6rem;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:.7rem;color:{_GOLD}">{o["date"].date()}</td>'
            f'<td style="padding:.3rem .6rem;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:.68rem;color:#c8c8c8;text-align:right">{o["similarity"]:.1f}%</td>'
            f'<td style="padding:.3rem .6rem;font-size:.64rem;color:#c8c8c8">{events}</td>'
            f'<td style="padding:.3rem .6rem;font-size:.66rem;color:{_MUTED};text-align:center">'
            f'{_REGIME_NAMES[o["regime_then"]]} → {_REGIME_NAMES[o["regime_60"]]}</td>'
            f'<td style="padding:.3rem .6rem;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:.7rem;text-align:right">{_pct(r["S&P 500"][60])}</td>'
            f'<td style="padding:.3rem .6rem;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:.7rem;text-align:right">{_pct(r["WTI Crude Oil"][60])}</td>'
            f'<td style="padding:.3rem .6rem;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:.7rem;text-align:right">{_pct(r["Gold"][60])}</td>'
            f'<td style="padding:.3rem .6rem;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:.7rem;color:{_RED};text-align:right">{o["spx_max_dd_60"] * 100:+.1f}%</td>'
            f'<td style="padding:.3rem .6rem;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:.66rem;font-weight:700;color:{v_col};text-align:center">{o["verdict"]}</td>'
            f'</tr>'
        )
    hdr = "".join(
        f'<th style="padding:.35rem .6rem;font-size:.53rem;letter-spacing:.1em;'
        f'text-transform:uppercase;color:{_MUTED};text-align:{a}">{h}</th>'
        for h, a in [("Date", "left"), ("Similarity", "right"),
                     ("What was happening", "left"), ("Regime → +60d", "center"),
                     ("S&P +60d", "right"), ("WTI +60d", "right"), ("Gold +60d", "right"),
                     ("S&P max DD", "right"), ("Verdict", "center")]
    )
    st.markdown(
        f'<div style="overflow:auto;border:1px solid #1e1e1e;margin-bottom:.8rem">'
        f'<table style="width:100%;border-collapse:collapse;background:#080808">'
        f'<thead><tr style="border-bottom:1px solid #1e1e1e">{hdr}</tr></thead>'
        f'<tbody>{rows}</tbody></table></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="font-size:0.58rem;color:{_MUTED};margin:-0.3rem 0 0.8rem">Similarity = share '
        f'of all candidate days farther from today than this match ("closer than X% of history"). '
        f'Verdict: BAD = S&amp;P ≤ −5% in 60d or regime hit Crisis; GOOD = S&amp;P ≥ +2% and '
        f'no Crisis; else MIXED.</p>',
        unsafe_allow_html=True,
    )

    # ── Where did it lead ──────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        _chart(_verdict_vs_base_chart(agg, br))
    with c2:
        _chart(_spaghetti_chart(outcomes, combined, "S&P 500"))

    # Punchline
    bad_lift = (agg.get("BAD", 0) - br.get("BAD", 0)) * 100
    good_lift = (agg.get("GOOD", 0) - br.get("GOOD", 0)) * 100
    lift_txt = (
        f'Analogs of today\'s configuration went BAD '
        f'<b style="color:{_RED if bad_lift > 0 else _GREEN}">{bad_lift:+.0f}pp</b> vs base rate '
        f'and GOOD <b style="color:{_GREEN if good_lift > 0 else _RED}">{good_lift:+.0f}pp</b> '
        f'vs base rate.'
    )
    st.markdown(
        f'<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-top:2px solid {_GOLD};'
        f'padding:.8rem 1rem;font-size:.72rem;color:#c8c8c8;line-height:1.7">'
        f'{lift_txt} With {agg.get("n", 0)} analogs this is pattern recall, not statistical '
        f'proof — treat lifts under ~15pp as noise.</div>',
        unsafe_allow_html=True,
    )

    _definition_block(
        "What this engine is — and is not",
        "Fingerprint blocks: GEO and CHOKEPOINT are structural reconstructions from the event/"
        "conflict registry (weights calibrated recently — same disclosed hindsight as Replay "
        "Mode; recency and activity are computed day-by-day, and no event exists before its "
        "onset). SPILLOVER, REGIME, and MARKET are genuinely historical (rolling D-Y, rolling "
        "betas, correlation regime, realized vol, momentum). Every raw feature at day t uses "
        "data ≤ t; feature standardization and regime percentile thresholds are full-sample "
        "(in-sample normalization, standard for analog recall) — verified in "
        "tests/test_fingerprint.py. Matching is weighted Euclidean distance in z-space across "
        "the whole configuration; block weights are the sliders above. This is pattern-memory, "
        "NOT a forecast: n=7 analogs cannot establish significance, which is why the base-rate "
        "comparison is always displayed."
    )

    _page_footer()
