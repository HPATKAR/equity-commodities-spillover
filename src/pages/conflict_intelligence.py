"""
Conflict Intelligence Page.

Per-conflict scorecard grid — one card per conflict showing:
  CIS, TPS, confidence, trend, state, freshness

Detailed drill-down for selected conflict:
  - Intensity dimension breakdown (radar/bar)
  - Transmission pressure heatmap (conflicts × channels)
  - Top affected assets table
  - Live news headlines (Threat/Act classified)

Uses: conflict_model.py, gpr_news.py, config.CONFLICTS, config.SECURITY_EXPOSURE
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis.conflict_model import (
    score_all_conflicts,
    aggregate_portfolio_scores,
    top_affected_assets,
)
from src.data.config import CONFLICTS, PALETTE


# ── Colour helpers ─────────────────────────────────────────────────────────────

def _cis_color(cis: float) -> str:
    if cis >= 70:  return "#c0392b"
    if cis >= 45:  return "#e67e22"
    if cis >= 25:  return "#CFB991"
    return "#27ae60"

def _tps_color(tps: float) -> str:
    if tps >= 60:  return "#c0392b"
    if tps >= 35:  return "#e67e22"
    return "#8E9AAA"

def _trend_marker(trend: str) -> str:
    return {"rising": "▲", "stable": "■", "falling": "▼"}.get(trend, "■")

def _state_badge(state: str) -> tuple[str, str]:
    return {
        "active":  ("ACTIVE",  "#c0392b"),
        "latent":  ("LATENT",  "#e67e22"),
        "frozen":  ("FROZEN",  "#8E9AAA"),
    }.get(state, ("UNKNOWN", "#555960"))

def _freshness_color(label: str) -> str:
    return {
        "live":   "#27ae60",
        "recent": "#CFB991",
        "aging":  "#e67e22",
        "stale":  "#c0392b",
    }.get(label, "#555960")


# ── Scorecard grid ────────────────────────────────────────────────────────────

def _render_scorecard_grid(results: dict) -> str | None:
    """
    Render conflict scorecards in a 3-column grid.
    Returns the selected conflict_id from radio, or None.
    """
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        'color:#8E9AAA;letter-spacing:2px;text-transform:uppercase;'
        'margin-bottom:0.4rem">CONFLICT SCORECARD GRID</p>',
        unsafe_allow_html=True,
    )

    # Sort by CIS descending
    ranked = sorted(results.values(), key=lambda r: r["cis"], reverse=True)
    n = len(ranked)
    cols_per_row = 3

    selected_id = st.session_state.get("ci_selected_conflict",
                                        ranked[0]["id"] if ranked else None)

    for row_start in range(0, n, cols_per_row):
        batch = ranked[row_start: row_start + cols_per_row]
        cols  = st.columns(len(batch))
        for col, r in zip(cols, batch):
            with col:
                state_text, state_col = _state_badge(r["state"])
                cis_col  = _cis_color(r["cis"])
                tps_col  = _tps_color(r["tps"])
                trend    = _trend_marker(r["trend"])
                fresh_col = _freshness_color(r.get("freshness", "aging"))
                is_sel   = (r["id"] == selected_id)
                border   = f"1px solid {r['color']}" if is_sel else "1px solid #2a2a2a"

                card_html = (
                    f'<div style="background:#0d0d0d;border:{border};'
                    f'padding:10px 12px;cursor:pointer">'
                    # Header row
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;margin-bottom:6px">'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:9px;font-weight:700;color:{r["color"]}">'
                    f'{r["label"]}</span>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:7px;color:{state_col};border:1px solid {state_col};'
                    f'padding:1px 4px">{state_text}</span>'
                    f'</div>'
                    # Name
                    f'<div style="font-family:\'DM Sans\',sans-serif;font-size:10px;'
                    f'color:#c8cdd8;margin-bottom:8px;line-height:1.3">'
                    f'{r["name"]}</div>'
                    # CIS / TPS
                    f'<div style="display:flex;gap:16px;margin-bottom:4px">'
                    f'<div>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:7px;color:#555960">CIS</span><br>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:16px;font-weight:700;color:{cis_col}">'
                    f'{r["cis"]:.0f}</span>'
                    f'</div>'
                    f'<div>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:7px;color:#555960">TPS</span><br>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:16px;font-weight:700;color:{tps_col}">'
                    f'{r["tps"]:.0f}</span>'
                    f'</div>'
                    f'<div>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:7px;color:#555960">CONF</span><br>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:16px;font-weight:700;color:#8E9AAA">'
                    f'{r["confidence"]:.0%}</span>'
                    f'</div>'
                    f'</div>'
                    # Trend + freshness
                    f'<div style="display:flex;justify-content:space-between;'
                    f'margin-top:4px">'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:8px;color:{cis_col}">{trend} {r["trend"].upper()}</span>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:7px;color:{fresh_col}">'
                    f'{r.get("freshness","aging").upper()}</span>'
                    f'</div>'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

                if st.button(
                    f"{'▶ ' if is_sel else ''}Select",
                    key=f"ci_sel_{r['id']}",
                    use_container_width=True,
                ):
                    st.session_state["ci_selected_conflict"] = r["id"]
                    selected_id = r["id"]
                    st.rerun()

    return selected_id


# ── Transmission heatmap ──────────────────────────────────────────────────────

def _render_transmission_heatmap(results: dict) -> None:
    """Conflicts × channels transmission pressure heatmap."""
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        'color:#8E9AAA;letter-spacing:2px;margin:1.2rem 0 0.4rem">'
        'TRANSMISSION PRESSURE HEATMAP</p>',
        unsafe_allow_html=True,
    )

    _CHANNELS = [
        "oil_gas", "metals", "agriculture", "shipping", "chokepoint",
        "sanctions", "equity_sector", "fx", "inflation",
        "supply_chain", "credit", "energy_infra",
    ]
    _CH_LABELS = [
        "Oil/Gas", "Metals", "Agri", "Shipping", "Chokepoint",
        "Sanctions", "Equity", "FX", "Inflation",
        "Supply Chain", "Credit", "Infra",
    ]

    ranked = sorted(results.values(), key=lambda r: r["cis"], reverse=True)
    conflict_labels = [r["label"] for r in ranked]

    from src.data.config import CONFLICTS
    conf_map = {c["id"]: c for c in CONFLICTS}

    z_data = []
    for r in ranked:
        conf = conf_map.get(r["id"], {})
        tx   = conf.get("transmission", {})
        row  = [float(tx.get(ch, 0.0)) for ch in _CHANNELS]
        z_data.append(row)

    z  = np.array(z_data)
    fig = go.Figure(go.Heatmap(
        z=z,
        x=_CH_LABELS,
        y=conflict_labels,
        colorscale=[
            [0.00, "#0d0d0d"],
            [0.15, "#1a1f2e"],
            [0.35, "#2c3a5a"],
            [0.55, "#8E6F3E"],
            [0.75, "#e67e22"],
            [1.00, "#c0392b"],
        ],
        zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont=dict(size=7, family="JetBrains Mono, monospace"),
        hoverongaps=False,
        showscale=True,
        colorbar=dict(
            thickness=8, len=0.7,
            tickfont=dict(size=7, family="JetBrains Mono, monospace"),
            title=dict(text="", side="right"),
        ),
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=40, t=10, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            tickfont=dict(size=7.5, family="JetBrains Mono, monospace"),
            tickangle=-35,
        ),
        yaxis=dict(
            tickfont=dict(size=8, family="JetBrains Mono, monospace"),
            autorange="reversed",
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Intensity dimension breakdown ─────────────────────────────────────────────

def _render_intensity_breakdown(conflict_id: str, result: dict) -> None:
    """Horizontal bar chart of CIS dimension contributions for selected conflict."""
    from src.data.config import CONFLICTS
    from src.analysis.conflict_model import _CIS_WEIGHTS, _ESCALATION_MAP, _recency_score

    conf = next((c for c in CONFLICTS if c["id"] == conflict_id), None)
    if conf is None:
        return

    dims = {
        "Deadliness":          float(conf.get("deadliness",           0.5)),
        "Civilian Danger":     float(conf.get("civilian_danger",      0.5)),
        "Geo Diffusion":       float(conf.get("geographic_diffusion", 0.3)),
        "Fragmentation":       float(conf.get("fragmentation",        0.2)),
        "Escalation Trend":    _ESCALATION_MAP.get(
                                    conf.get("escalation_trend", "stable"), 0.5),
        "Recency":             _recency_score(conf),
        "Source Coverage":     float(conf.get("source_coverage",      0.7)),
    }
    wt_keys = {
        "Deadliness":          "deadliness",
        "Civilian Danger":     "civilian_danger",
        "Geo Diffusion":       "geographic_diffusion",
        "Fragmentation":       "fragmentation",
        "Escalation Trend":    "escalation_trend",
        "Recency":             "recency",
        "Source Coverage":     "source_coverage",
    }

    labels   = list(dims.keys())
    values   = [dims[k] for k in labels]
    weights  = [_CIS_WEIGHTS.get(wt_keys[k], 0) for k in labels]
    weighted = [v * w * 100 for v, w in zip(values, weights)]

    bar_colors = [
        "#c0392b" if v >= 0.75 else "#e67e22" if v >= 0.45 else "#8E9AAA"
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=weighted,
        y=labels,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v:.1f}  (raw {r:.2f})" for v, r in zip(weighted, values)],
        textposition="outside",
        textfont=dict(size=8, family="JetBrains Mono, monospace", color="#8E9AAA"),
        hovertemplate="%{y}: weighted=%{x:.1f}<extra></extra>",
    ))
    fig.update_layout(
        height=230,
        margin=dict(l=10, r=80, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, max(weighted) * 1.35 if weighted else 30],
            tickfont=dict(size=7, family="JetBrains Mono, monospace"),
            showgrid=True, gridcolor="#1e1e1e",
        ),
        yaxis=dict(
            tickfont=dict(size=8, family="JetBrains Mono, monospace"),
            autorange="reversed",
        ),
        bargap=0.28,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── TPS channel breakdown for selected conflict ───────────────────────────────

def _render_tps_channels(conflict_id: str) -> None:
    from src.data.config import CONFLICTS
    from src.analysis.conflict_model import _TPS_WEIGHTS

    conf = next((c for c in CONFLICTS if c["id"] == conflict_id), None)
    if conf is None:
        return

    tx = conf.get("transmission", {})
    _CH_LABELS = {
        "oil_gas": "Oil/Gas", "metals": "Metals", "agriculture": "Agriculture",
        "shipping": "Shipping", "chokepoint": "Chokepoint", "sanctions": "Sanctions",
        "equity_sector": "Equity Sector", "fx": "FX", "inflation": "Inflation",
        "supply_chain": "Supply Chain", "credit": "Credit", "energy_infra": "Energy Infra",
    }

    channels = list(_TPS_WEIGHTS.keys())
    vals     = [float(tx.get(ch, 0.0)) for ch in channels]
    weighted = [v * _TPS_WEIGHTS[ch] * 100 for v, ch in zip(vals, channels)]
    labels   = [_CH_LABELS.get(ch, ch) for ch in channels]

    # Sort by weighted contribution desc
    order = sorted(range(len(weighted)), key=lambda i: weighted[i], reverse=True)
    labels_s   = [labels[i]   for i in order]
    vals_s     = [vals[i]     for i in order]
    weighted_s = [weighted[i] for i in order]

    bar_colors = [
        "#c0392b" if v >= 0.75 else "#e67e22" if v >= 0.45 else "#8E9AAA"
        for v in vals_s
    ]

    fig = go.Figure(go.Bar(
        x=weighted_s,
        y=labels_s,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v:.1f}" for v in weighted_s],
        textposition="outside",
        textfont=dict(size=8, family="JetBrains Mono, monospace", color="#8E9AAA"),
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=60, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            tickfont=dict(size=7, family="JetBrains Mono, monospace"),
            showgrid=True, gridcolor="#1e1e1e",
        ),
        yaxis=dict(
            tickfont=dict(size=8, family="JetBrains Mono, monospace"),
            autorange="reversed",
        ),
        bargap=0.25,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Top affected assets ───────────────────────────────────────────────────────

def _render_affected_assets(conflict_id: str, conflict_result: dict) -> None:
    assets = top_affected_assets(conflict_id, n=8)
    if not assets:
        # Fall back to config affected_commodities / affected_equities
        from src.data.config import CONFLICTS
        conf = next((c for c in CONFLICTS if c["id"] == conflict_id), None)
        if conf:
            all_assets = (
                [(a, "commodity") for a in conf.get("affected_commodities", [])]
                + [(a, "equity")   for a in conf.get("affected_equities",    [])]
                + [(a, "hedge")    for a in conf.get("hedge_assets",         [])]
            )
            assets = [{"asset": a, "exposure": None, "type": t} for a, t in all_assets[:8]]

    if not assets:
        st.caption("No affected assets configured.")
        return

    rows_html = ""
    for item in assets:
        asset  = item["asset"]
        exp    = item.get("exposure")
        exp_str = f"{exp:.2f}" if exp is not None else "—"
        bar_w  = int((exp or 0.5) * 80)
        rows_html += (
            f'<div style="display:flex;align-items:center;gap:8px;'
            f'padding:3px 0;border-bottom:1px solid #1a1a1a">'
            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:10px;'
            f'color:#c8cdd8;width:140px;flex-shrink:0">{asset}</span>'
            f'<div style="flex:1;height:6px;background:#1a1a1a;border-radius:0">'
            f'<div style="width:{bar_w}%;height:100%;'
            f'background:#CFB991;border-radius:0"></div>'
            f'</div>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            f'color:#CFB991;width:30px;text-align:right">{exp_str}</span>'
            f'</div>'
        )

    st.markdown(
        f'<div style="background:#0a0a0a;border:1px solid #1e1e1e;'
        f'padding:8px 12px">{rows_html}</div>',
        unsafe_allow_html=True,
    )


# ── News panel for selected conflict ─────────────────────────────────────────

def _render_conflict_news(conflict_id: str) -> None:
    try:
        from src.analysis.gpr_news import render_threat_act_feed, get_news_gpr_layer
        result = get_news_gpr_layer()
        pc = result["per_conflict"].get(conflict_id, {})
        if pc:
            t_score = pc.get("threat", 0)
            a_score = pc.get("act", 0)
            gpr     = pc.get("news_gpr", 0)
            n_hl    = pc.get("n_headlines", 0)
            st.markdown(
                f'<div style="display:flex;gap:16px;margin-bottom:8px">'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
                f'color:#e67e22">THREAT {t_score:.0f}</span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
                f'color:#c0392b">ACT {a_score:.0f}</span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
                f'color:#CFB991">NEWS GPR {gpr:.0f}</span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
                f'color:#8E9AAA">{n_hl} headlines</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        render_threat_act_feed(conflict_id=conflict_id, max_items=6)
    except Exception:
        st.caption("News feed unavailable.")


# ── Main page ─────────────────────────────────────────────────────────────────

def page_conflict_intelligence(start=None, end=None, fred_key="") -> None:
    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
        'color:#8E9AAA;letter-spacing:3px;text-transform:uppercase;margin:0">'
        'INTELLIGENCE / CONFLICT SCORECARD</p>'
        '<h2 style="font-family:\'DM Sans\',sans-serif;font-size:1.35rem;'
        'font-weight:700;color:#e8e9ed;margin:4px 0 16px">Conflict Intelligence</h2>',
        unsafe_allow_html=True,
    )

    # ── Load scores ────────────────────────────────────────────────────────
    try:
        results = score_all_conflicts()
    except Exception as e:
        st.error(f"Error loading conflict scores: {e}")
        return

    if not results:
        st.warning("No conflict data available.")
        return

    # ── Scorecard grid ─────────────────────────────────────────────────────
    selected_id = _render_scorecard_grid(results)
    if not selected_id or selected_id not in results:
        selected_id = next(iter(results))

    selected = results[selected_id]

    # ── Heatmap ────────────────────────────────────────────────────────────
    _render_transmission_heatmap(results)

    # ── Conflict detail section ────────────────────────────────────────────
    st.markdown(
        f'<div style="border-left:3px solid {selected["color"]};'
        f'padding:4px 12px;margin:1.2rem 0 0.6rem">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
        f'font-weight:700;color:{selected["color"]}">{selected["label"]}</span>'
        f'<span style="font-family:\'DM Sans\',sans-serif;font-size:13px;'
        f'color:#e8e9ed;margin-left:10px">{selected["name"]}</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        f'color:#8E9AAA;margin-left:12px">'
        f'CIS {selected["cis"]:.0f} · TPS {selected["tps"]:.0f} · '
        f'{selected["trend"].upper()} · {selected["state"].upper()}'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Two-column detail layout ───────────────────────────────────────────
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown(
            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            'color:#8E9AAA;letter-spacing:2px">INTENSITY DIMENSIONS</p>',
            unsafe_allow_html=True,
        )
        _render_intensity_breakdown(selected_id, selected)

    with col_r:
        st.markdown(
            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            'color:#8E9AAA;letter-spacing:2px">TRANSMISSION CHANNELS</p>',
            unsafe_allow_html=True,
        )
        _render_tps_channels(selected_id)

    # ── Bottom row: affected assets + news ─────────────────────────────────
    col_a, col_n = st.columns([1, 1])

    with col_a:
        st.markdown(
            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            'color:#8E9AAA;letter-spacing:2px;margin-bottom:6px">'
            'TOP AFFECTED ASSETS</p>',
            unsafe_allow_html=True,
        )
        _render_affected_assets(selected_id, selected)

    with col_n:
        st.markdown(
            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            'color:#8E9AAA;letter-spacing:2px;margin-bottom:6px">'
            'LIVE INTELLIGENCE FEED</p>',
            unsafe_allow_html=True,
        )
        _render_conflict_news(selected_id)

    # ── AI Analyst Deliberation ────────────────────────────────────────────────
    # Trigger and display the analyst team's assessment of the selected conflict.
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        'color:#8E9AAA;letter-spacing:2px;margin:1.2rem 0 0.4rem">'
        'AI ANALYST TEAM — CONFLICT ASSESSMENT</p>',
        unsafe_allow_html=True,
    )
    try:
        from src.analysis.agent_dialogue import (
            get_subject_threads, send_message, compute_consensus,
        )
        from src.ui.agent_panel import render_deliberation_panel

        _conf_subject = f"conflict_{selected_id}"
        _existing_msgs = get_subject_threads(_conf_subject)

        _col_debate, _col_trigger = st.columns([4, 1])
        with _col_trigger:
            if st.button("Run Assessment", key=f"ci_debate_{selected_id}",
                         use_container_width=True):
                _cis = selected["cis"]
                _tps = selected["tps"]
                _trend = selected["trend"]
                _state = selected["state"]

                # Risk Officer opens the thread
                _tid = send_message(
                    sender="risk_officer",
                    recipient="geopolitical_analyst",
                    msg_type="query",
                    content=(
                        f"{selected['label']} — CIS {_cis:.0f}/100, TPS {_tps:.0f}/100, "
                        f"state {_state.upper()}, trend {_trend.upper()}. "
                        f"What is driving the current intensity and which transmission "
                        f"channels carry the most near-term market risk?"
                    ),
                    subject_id=_conf_subject,
                )
                # Geopolitical Analyst responds
                _trend_context = {
                    "rising":  "Escalation risk is real. Recommend elevated positioning.",
                    "stable":  "Holding steady — monitor for shift triggers.",
                    "falling": "De-escalation signal. Watch for false dawns.",
                }.get(_trend, "Trend ambiguous — insufficient data to project.")
                send_message(
                    sender="geopolitical_analyst",
                    recipient="commodities_specialist",
                    msg_type="handoff",
                    content=(
                        f"Intensity analysis complete. "
                        f"Conflict is {_trend.upper()} — {_trend_context} "
                        f"Handing off to commodity specialist for transmission impact."
                    ),
                    subject_id=_conf_subject,
                    thread_id=_tid,
                )
                # Commodities Specialist assesses transmission
                _top_ch = max(
                    selected.get("transmission", {}).items(),
                    key=lambda x: x[1], default=("unknown", 0)
                )
                send_message(
                    sender="commodities_specialist",
                    recipient="macro_strategist",
                    msg_type="handoff",
                    content=(
                        f"Top transmission channel: {_top_ch[0].replace('_',' ').title()} "
                        f"({_top_ch[1]:.0%} weight). "
                        f"TPS {_tps:.0f} indicates "
                        + ("active market transmission — commodity repricing underway."
                           if _tps >= 50 else
                           "pressure building but not yet reflected in prices.")
                    ),
                    subject_id=_conf_subject,
                    thread_id=_tid,
                )
                # Macro Strategist concludes
                _conf_label = (
                    "HIGH CONFIDENCE — act on signal" if selected.get("confidence", 0.5) >= 0.7
                    else "MODERATE CONFIDENCE — size conservatively"
                    if selected.get("confidence", 0.5) >= 0.5
                    else "LOW CONFIDENCE — watch but do not trade yet"
                )
                send_message(
                    sender="macro_strategist",
                    recipient="risk_officer",
                    msg_type="resolve",
                    content=(
                        f"Assessment: {_conf_label}. "
                        f"Regime alignment: {selected['state'].upper()} conflict with "
                        f"{_trend} trajectory. Recommend "
                        + ("hedging commodity exposure and reviewing safe-haven allocations."
                           if _cis >= 60 else
                           "monitoring. Reopen assessment if CIS exceeds 60.")
                    ),
                    subject_id=_conf_subject,
                    thread_id=_tid,
                )
                st.rerun()

        with _col_debate:
            if _existing_msgs:
                render_deliberation_panel(
                    subject_id=_conf_subject,
                    title=f"Analyst Assessment — {selected['label']}",
                    max_msgs=8,
                    show_consensus=True,
                )
            else:
                st.markdown(
                    '<p style="font-family:\'DM Sans\',sans-serif;font-size:9px;'
                    'color:#555960;font-style:italic">'
                    'Click Run Assessment to initiate analyst team deliberation '
                    'for this conflict.</p>',
                    unsafe_allow_html=True,
                )
    except Exception as exc:
        st.caption(f"Agent deliberation unavailable: {exc}")
