"""
Threat vs Act Monitor Page.

Real-time split view of:
  Threat Score  — rhetoric, force build-up, military signaling
  Act Score     — realized adverse events (strikes, seizures, sanctions)

Layout:
  Row 1: dual KPI gauges (Threat / Act) + News GPR composite
  Row 2: live headline feed (Threat tab | Act tab | All tab)
  Row 3: per-conflict news GPR bar chart
  Row 4: alpha dial + score methodology footnote
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from src.ui.shared import _page_header, _page_footer


# ── Gauge helper ──────────────────────────────────────────────────────────────

def _mini_gauge(value: float, label: str, color: str, height: int = 160) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(
            font=dict(size=28, family="JetBrains Mono, monospace", color=color),
        ),
        title=dict(
            text=label,
            font=dict(size=9, family="JetBrains Mono, monospace", color="#8E9AAA"),
        ),
        gauge=dict(
            axis=dict(range=[0, 100], visible=False),
            bar=dict(color=color, thickness=0.22),
            bgcolor="rgba(20,20,20,0.8)",
            borderwidth=0,
            steps=[
                dict(range=[0,  33], color="rgba(39,174,96,0.12)"),
                dict(range=[33, 66], color="rgba(230,126,34,0.12)"),
                dict(range=[66, 100], color="rgba(192,57,43,0.15)"),
            ],
        ),
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Per-conflict news GPR bar chart ──────────────────────────────────────────

def _render_conflict_news_gpr(per_conflict: dict) -> None:
    if not per_conflict:
        return

    # Sort by news_gpr desc
    items = sorted(per_conflict.items(), key=lambda x: x[1].get("news_gpr", 0), reverse=True)
    labels  = [cid.replace("_", " ").title() for cid, _ in items]
    threat  = [v.get("threat", 0) for _, v in items]
    act     = [v.get("act", 0)    for _, v in items]
    gpr     = [v.get("news_gpr", 0) for _, v in items]

    fig = go.Figure()
    fig.add_bar(
        name="Threat",
        x=labels, y=threat,
        marker_color="#e67e22",
        opacity=0.75,
    )
    fig.add_bar(
        name="Act",
        x=labels, y=act,
        marker_color="#c0392b",
        opacity=0.85,
    )
    fig.add_scatter(
        name="News GPR",
        x=labels, y=gpr,
        mode="lines+markers",
        line=dict(color="#CFB991", width=1.5),
        marker=dict(size=5, symbol="diamond"),
    )

    fig.update_layout(
        barmode="group",
        height=210,
        margin=dict(l=10, r=10, t=10, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h", x=0, y=1.08,
            font=dict(size=8, family="JetBrains Mono, monospace"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            tickfont=dict(size=8, family="JetBrains Mono, monospace"),
            tickangle=-25,
        ),
        yaxis=dict(
            range=[0, 100],
            tickfont=dict(size=7, family="JetBrains Mono, monospace"),
            gridcolor="#1e1e1e",
        ),
        bargap=0.18,
        bargroupgap=0.06,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Alpha gauge (donut-style) ─────────────────────────────────────────────────

def _render_alpha_display(alpha: float) -> None:
    alpha_pct = alpha * 100
    label = "ACT-DOMINATED" if alpha >= 0.65 else "BALANCED" if alpha >= 0.50 else "THREAT-DOMINATED"
    color = "#c0392b" if alpha >= 0.65 else "#e67e22" if alpha >= 0.50 else "#CFB991"
    st.markdown(
        f'<div style="text-align:center;padding:12px 8px;'
        f'background:#0d0d0d;border:1px solid #2a2a2a">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
        f'color:#555960;letter-spacing:.16em;text-transform:uppercase">α (Act Weight)</span>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:28px;'
        f'font-weight:700;color:{color};margin:4px 0">{alpha_pct:.0f}%</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
        f'color:{color}">{label}</div>'
        f'<div style="margin-top:8px;height:4px;background:#1a1a1a">'
        f'<div style="width:{alpha_pct:.0f}%;height:100%;background:{color}"></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Methodology note ──────────────────────────────────────────────────────────

def _render_methodology() -> None:
    st.markdown(
        '<div style="background:#0a0a0a;border:1px solid #1e1e1e;'
        'padding:10px 14px;margin-top:8px">'
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
        'color:#555960;letter-spacing:.16em;margin:0 0 4px">METHODOLOGY</p>'
        '<p style="font-family:\'DM Sans\',sans-serif;font-size:9px;'
        'color:#8E9AAA;line-height:1.6;margin:0">'
        'News GPR = α · Act Score + (1−α) · Threat Score. '
        'α rises dynamically when realized act headlines dominate '
        '(α_max 0.80); defaults to 0.55 in balanced environments. '
        'Threat Score: rhetoric, warnings, force build-up, military signaling, '
        'sanctions threats. '
        'Act Score: realized events — strikes, seizures, sanctions imposed, '
        'shipping blockages. '
        'Scores are EWM-dampened to avoid spike-then-reversal overcounting. '
        'Source: Reuters, FT, WSJ, BBC, AP (15-min cache).'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Main page ─────────────────────────────────────────────────────────────────

def page_threat_act_monitor(start=None, end=None, fred_key="") -> None:
    _page_header("News GPR · Threat/Act Monitor",
                 "RSS geo-risk scoring · Threat/act classification · Conflict routing · Review queue",
                 "INTELLIGENCE / NEWS GPR")

    # ── Fetch data ─────────────────────────────────────────────────────────
    try:
        from src.analysis.gpr_news import get_news_gpr_layer, render_threat_act_feed
        result = get_news_gpr_layer()
    except Exception as e:
        st.error(f"News GPR layer unavailable: {e}")
        _render_methodology()
        return

    # Surface data availability — never show silent zeros
    data_status = result.get("data_status", "live")
    feed_error  = result.get("feed_error", "")

    if data_status == "no_feed":
        st.markdown(
            f'<div style="background:#130808;border:1px solid #c0392b;'
            f'padding:12px 16px;margin-bottom:12px">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            f'font-weight:700;color:#c0392b">■ RSS FEED UNAVAILABLE</span><br>'
            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:10px;'
            f'color:#8E9AAA;margin-top:4px;display:block">{feed_error or "No headlines returned from feeds."}</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            f'color:#555960">Scores show 0 — not a low-risk signal. Fix feed to restore.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        _render_methodology()
        return

    if data_status == "no_classified":
        n_raw = result.get("n_raw", 0)
        st.markdown(
            f'<div style="background:#111;border:1px solid #e67e22;'
            f'padding:10px 16px;margin-bottom:12px">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            f'font-weight:700;color:#e67e22">■ NO CLASSIFIED HEADLINES</span><br>'
            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:10px;'
            f'color:#8E9AAA">'
            f'{n_raw} headlines fetched · 0 matched Threat/Act taxonomy. '
            f'This is a <b style="color:#e67e22">no-signal</b> reading — '
            f'not a low-risk reading. Zero does not mean calm; it means the news '
            f'cycle currently lacks geopolitical language matching the classification taxonomy.'
            f'</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Render NO-SIGNAL tiles instead of zero gauges to avoid false calm signal
        _ns_cols = st.columns([1, 1, 1, 0.7])
        for _nc, (_lbl, _col) in zip(_ns_cols[:3], [
            ("THREAT SCORE", "#e67e22"),
            ("ACT SCORE",    "#c0392b"),
            ("NEWS GPR",     "#CFB991"),
        ]):
            _nc.markdown(
                f'<div style="background:#0d0d0d;border:1px solid #2a2a2a;'
                f'height:160px;display:flex;flex-direction:column;'
                f'align-items:center;justify-content:center;gap:4px">'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
                f'color:#555960;letter-spacing:.16em;text-transform:uppercase">{_lbl}</span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:22px;'
                f'font-weight:700;color:{_col};opacity:.4">— —</span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
                f'color:#555960">NO SIGNAL · {n_raw} raw headlines</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        _ns_cols[3].markdown(
            f'<div style="background:#0d0d0d;border:1px solid #2a2a2a;'
            f'height:160px;display:flex;flex-direction:column;'
            f'align-items:center;justify-content:center;gap:4px;margin-top:10px">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
            f'color:#555960;letter-spacing:.16em">α (ACT WEIGHT)</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:18px;'
            f'font-weight:700;color:#555960;opacity:.4">—</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
            f'color:#555960">AWAITING SIGNAL</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        _render_methodology()
        return

    threat_score = result["threat_score"]
    act_score    = result["act_score"]
    news_gpr     = result["news_gpr"]
    alpha        = result["alpha"]
    n_threat     = result["n_threat"]
    n_act        = result["n_act"]
    per_conflict = result["per_conflict"]

    # ── Row 1: KPI gauges ──────────────────────────────────────────────────
    g1, g2, g3, g4 = st.columns([1, 1, 1, 0.7])

    with g1:
        st.plotly_chart(
            _mini_gauge(threat_score, "THREAT SCORE", "#e67e22"),
            use_container_width=True, config={"displayModeBar": False},
        )
    with g2:
        st.plotly_chart(
            _mini_gauge(act_score, "ACT SCORE", "#c0392b"),
            use_container_width=True, config={"displayModeBar": False},
        )
    with g3:
        st.plotly_chart(
            _mini_gauge(news_gpr, "NEWS GPR", "#CFB991"),
            use_container_width=True, config={"displayModeBar": False},
        )
    with g4:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        _render_alpha_display(alpha)

    # ── Row 2: headline counts ─────────────────────────────────────────────
    st.markdown(
        f'<div style="display:flex;gap:20px;margin:4px 0 12px;padding:6px 12px;'
        f'background:#0d0d0d;border:1px solid #1e1e1e">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        f'color:#e67e22">THREAT {n_threat} headlines</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        f'color:#c0392b">ACT {n_act} headlines</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        f'color:#8E9AAA">{result.get("n_raw", "?")} raw · fetched {result["fetched_at"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Row 3: headline tabs ───────────────────────────────────────────────
    tab_all, tab_act, tab_threat = st.tabs(["All", "Act", "Threat"])

    with tab_all:
        render_threat_act_feed(news_type=None, max_items=10)
    with tab_act:
        render_threat_act_feed(news_type="act", max_items=10)
    with tab_threat:
        render_threat_act_feed(news_type="threat", max_items=10)

    # ── Row 4: per-conflict breakdown ──────────────────────────────────────
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        'color:#CFB991;letter-spacing:.16em;border-bottom:1px solid #1e1e1e;padding-bottom:4px;margin:1.4rem 0 .6rem">'
        'PER-CONFLICT NEWS GPR</p>',
        unsafe_allow_html=True,
    )
    _render_conflict_news_gpr(per_conflict)

    # ── Methodology ────────────────────────────────────────────────────────
    _render_methodology()
    _page_footer()
