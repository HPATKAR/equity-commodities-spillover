"""
Transmission Matrix Page.

Conflict × channel transmission pressure analysis:
  - Full heatmap: conflicts × 12 channels (weighted by CIS)
  - Sankey flow: conflict → channel → asset class
  - Most-affected assets ranked table (by weighted exposure)
  - Channel dominance: which channels are most stressed portfolio-wide
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.ui.shared import _page_header, _page_footer

from src.analysis.conflict_model import (
    score_all_conflicts,
    aggregate_portfolio_scores,
    conflict_commodity_matrix,
    top_affected_assets,
)
from src.data.config import CONFLICTS, PALETTE


# ── Channel metadata ──────────────────────────────────────────────────────────

_CHANNELS = [
    "oil_gas", "metals", "agriculture", "shipping", "chokepoint",
    "sanctions", "equity_sector", "fx", "inflation",
    "supply_chain", "credit", "energy_infra",
]
_CH_LABELS = {
    "oil_gas":       "Oil/Gas",
    "metals":        "Metals",
    "agriculture":   "Agriculture",
    "shipping":      "Shipping",
    "chokepoint":    "Chokepoint",
    "sanctions":     "Sanctions",
    "equity_sector": "Equity Sector",
    "fx":            "FX",
    "inflation":     "Inflation",
    "supply_chain":  "Supply Chain",
    "credit":        "Credit",
    "energy_infra":  "Energy Infra",
}
_CH_ASSET_MAP = {
    "oil_gas":       "Commodities",
    "metals":        "Commodities",
    "agriculture":   "Commodities",
    "shipping":      "Logistics",
    "chokepoint":    "Logistics",
    "sanctions":     "Policy",
    "equity_sector": "Equities",
    "fx":            "FX",
    "inflation":     "Macro",
    "supply_chain":  "Logistics",
    "credit":        "Fixed Income",
    "energy_infra":  "Commodities",
}


# ── CIS-weighted transmission heatmap ─────────────────────────────────────────

def _render_weighted_heatmap(results: dict) -> None:
    """
    Full heatmap with rows = conflicts, cols = channels.
    Cell values are channel transmission × (conflict CIS / max CIS).
    """
    if not results:
        st.caption("No conflict data.")
        return

    max_cis = max((r["cis"] for r in results.values()), default=100) + 1e-9

    conflict_ids   = list(results.keys())
    conflict_names = [results[cid]["label"] for cid in conflict_ids]

    z = []
    for cid in conflict_ids:
        r  = results[cid]
        tx = r.get("transmission", {})
        weight = r["cis"] / max_cis
        row = [float(tx.get(ch, 0.0)) * weight for ch in _CHANNELS]
        z.append(row)

    ch_labels = [_CH_LABELS.get(ch, ch) for ch in _CHANNELS]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=ch_labels,
        y=conflict_names,
        colorscale=[
            [0.0,  "#0a0a0a"],
            [0.15, "#1a1210"],
            [0.40, "#5c2a0e"],
            [0.70, "#c0392b"],
            [1.0,  "#ff6b6b"],
        ],
        zmin=0, zmax=1,
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>",
        showscale=True,
        colorbar=dict(
            thickness=12, len=0.85,
            tickfont=dict(family="JetBrains Mono", size=8, color="#8E9AAA"),
            title=dict(text="CIS-Weighted", font=dict(family="JetBrains Mono", size=8, color="#555960")),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
        margin=dict(l=10, r=10, t=10, b=60),
        height=max(200, len(conflict_ids) * 42),
        xaxis=dict(
            tickfont=dict(family="JetBrains Mono", size=8, color="#8E9AAA"),
            tickangle=-40, showgrid=False,
        ),
        yaxis=dict(
            tickfont=dict(family="JetBrains Mono", size=8, color="#8E9AAA"),
            showgrid=False,
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Channel dominance bar chart ───────────────────────────────────────────────

def _render_channel_dominance(results: dict) -> None:
    """
    Portfolio-wide channel stress: sum of CIS-weighted transmission per channel.
    """
    if not results:
        st.caption("No data.")
        return

    max_cis = max((r["cis"] for r in results.values()), default=100) + 1e-9
    channel_stress: dict[str, float] = {ch: 0.0 for ch in _CHANNELS}

    for r in results.values():
        tx = r.get("transmission", {})
        w  = r["cis"] / max_cis
        for ch in _CHANNELS:
            channel_stress[ch] += float(tx.get(ch, 0.0)) * w

    # Normalize to 0-1
    max_stress = max(channel_stress.values()) + 1e-9
    sorted_ch  = sorted(channel_stress.items(), key=lambda x: x[1])
    labels = [_CH_LABELS.get(k, k) for k, _ in sorted_ch]
    values = [v / max_stress for _, v in sorted_ch]

    bar_colors = [
        "#c0392b" if v >= 0.70 else "#e67e22" if v >= 0.45 else "#CFB991" if v >= 0.25 else "#555960"
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.0%}" for v in values],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=9, color="#8E9AAA"),
        hovertemplate="%{y}: %{x:.0%}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
        margin=dict(l=10, r=50, t=10, b=10),
        height=320,
        xaxis=dict(range=[0, 1.25], tickformat=".0%",
                   tickfont=dict(family="JetBrains Mono", size=8, color="#555960"),
                   gridcolor="#1e1e1e", showgrid=True),
        yaxis=dict(tickfont=dict(family="JetBrains Mono", size=8, color="#8E9AAA"),
                   showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Sankey: conflict → channel → asset class ──────────────────────────────────

def _render_sankey(results: dict) -> None:
    """
    Sankey diagram: conflict nodes → channel nodes → asset class nodes.
    Only includes links above a minimum threshold.
    """
    if not results:
        st.caption("No data.")
        return

    MIN_WEIGHT = 0.10
    max_cis = max((r["cis"] for r in results.values()), default=100) + 1e-9

    # Asset class display names — must NOT collide with any channel label.
    # "FX" channel and "FX" asset class share the same string, causing a self-loop.
    _ASSET_DISPLAY = {
        "Commodities":  "Commodities",
        "Equities":     "Equities",
        "Fixed Income": "Fixed Income",
        "FX":           "FX Markets",   # renamed to avoid collision with FX channel
        "Logistics":    "Logistics",
        "Macro":        "Macro",
        "Policy":       "Policy",
    }

    # Build node lists
    conflict_labels = [results[cid]["label"] for cid in results]
    channel_labels  = [_CH_LABELS[ch] for ch in _CHANNELS]
    asset_labels    = sorted(set(
        _ASSET_DISPLAY.get(v, v) for v in _CH_ASSET_MAP.values()
    ))

    all_labels = conflict_labels + channel_labels + asset_labels
    label_idx  = {lbl: i for i, lbl in enumerate(all_labels)}

    sources, targets, values, link_colors = [], [], [], []

    # Conflict → Channel
    for cid, r in results.items():
        tx  = r.get("transmission", {})
        w   = r["cis"] / max_cis
        for ch in _CHANNELS:
            ch_val = float(tx.get(ch, 0.0)) * w
            if ch_val >= MIN_WEIGHT:
                sources.append(label_idx[r["label"]])
                targets.append(label_idx[_CH_LABELS[ch]])
                values.append(ch_val)
                link_colors.append(f"rgba({int(r['color'][1:3],16)},{int(r['color'][3:5],16)},{int(r['color'][5:7],16)},0.25)")

    # Channel → Asset class
    ch_asset_flow: dict[tuple, float] = {}
    for cid, r in results.items():
        tx = r.get("transmission", {})
        w  = r["cis"] / max_cis
        for ch in _CHANNELS:
            ch_val = float(tx.get(ch, 0.0)) * w
            if ch_val >= MIN_WEIGHT:
                raw_asset = _CH_ASSET_MAP[ch]
                key = (_CH_LABELS[ch], _ASSET_DISPLAY.get(raw_asset, raw_asset))
                ch_asset_flow[key] = ch_asset_flow.get(key, 0.0) + ch_val

    for (ch_lbl, asset_lbl), flow in ch_asset_flow.items():
        sources.append(label_idx[ch_lbl])
        targets.append(label_idx[asset_lbl])
        values.append(flow)
        link_colors.append("rgba(207,185,145,0.15)")

    if not sources:
        st.caption("Insufficient transmission data for Sankey.")
        return

    # Node colors
    n_conflicts = len(conflict_labels)
    n_channels  = len(channel_labels)
    conflict_colors = [results[cid]["color"] for cid in results]
    channel_colors  = ["#555960"] * n_channels
    asset_colors    = ["#CFB991"] * len(asset_labels)
    node_colors = conflict_colors + channel_colors + asset_colors

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=12, thickness=14,
            label=all_labels,
            color=node_colors,
            line=dict(color="#0a0a0a", width=0.5),
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color=link_colors,
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#0a0a0a",
        font=dict(family="JetBrains Mono, monospace", size=8, color="#8E9AAA"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=340,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Commodity exposure table ──────────────────────────────────────────────────

def _render_commodity_exposure(results: dict) -> None:
    """
    Table of commodities ranked by CIS-weighted exposure across all conflicts.
    """
    try:
        matrix = conflict_commodity_matrix()
    except Exception:
        st.caption("Commodity matrix unavailable.")
        return

    max_cis = max((r["cis"] for r in results.values()), default=100) + 1e-9
    commodity_scores: dict[str, float] = {}

    for cid, r in results.items():
        w = r["cis"] / max_cis
        for commodity, rel in matrix.get(cid, {}).items():
            commodity_scores[commodity] = commodity_scores.get(commodity, 0.0) + rel * w

    if not commodity_scores:
        st.caption("No commodity exposure data.")
        return

    sorted_commodities = sorted(commodity_scores.items(), key=lambda x: x[1], reverse=True)
    max_score = max(v for _, v in sorted_commodities) + 1e-9

    cols = st.columns(min(len(sorted_commodities), 6))
    for i, (commodity, score) in enumerate(sorted_commodities[:6]):
        norm  = score / max_score
        color = "#c0392b" if norm >= 0.7 else "#e67e22" if norm >= 0.45 else "#CFB991" if norm >= 0.25 else "#555960"
        bar_w = int(norm * 100)
        with cols[i]:
            st.markdown(
                f'<div style="background:#0a0a0a;border:1px solid #2a2a2a;padding:.5rem .7rem">'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;color:#555960;margin-bottom:3px">'
                f'{commodity.upper()}</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:16px;font-weight:700;'
                f'color:{color}">{norm:.0%}</div>'
                f'<div style="background:#1a1a1a;height:3px;margin-top:5px;border-radius:1px">'
                f'<div style="background:{color};width:{bar_w}%;height:3px;border-radius:1px"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )


# ── Page entry point ──────────────────────────────────────────────────────────

def page_transmission_matrix(start=None, end=None, fred_key: str = "") -> None:
    _page_header(
        "Market Transmission Matrix",
        "Channel-level TPS · Conflict-to-market routing · Active pathway breakdown",
        "INTELLIGENCE / TRANSMISSION",
    )

    # ── Load scores ────────────────────────────────────────────────────────
    try:
        results = score_all_conflicts()
    except Exception as e:
        st.error(f"Error loading conflict scores: {e}")
        _page_footer()
        return

    if not results:
        st.warning("No conflict data available.")
        _page_footer()
        return

    # ── Section 1: Heatmap ─────────────────────────────────────────────────
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        'color:#CFB991;letter-spacing:.16em;border-bottom:1px solid #1e1e1e;padding-bottom:4px;margin-bottom:8px">'
        'CIS-WEIGHTED CONFLICT × CHANNEL HEATMAP</p>',
        unsafe_allow_html=True,
    )
    _render_weighted_heatmap(results)

    # ── Section 2: Channel dominance + Sankey ──────────────────────────────
    col_l, col_r = st.columns([1, 1.4])

    with col_l:
        st.markdown(
            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            'color:#CFB991;letter-spacing:.16em;border-bottom:1px solid #1e1e1e;padding-bottom:4px;margin-bottom:8px">'
            'PORTFOLIO CHANNEL STRESS</p>',
            unsafe_allow_html=True,
        )
        _render_channel_dominance(results)

    with col_r:
        st.markdown(
            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            'color:#CFB991;letter-spacing:.16em;border-bottom:1px solid #1e1e1e;padding-bottom:4px;margin-bottom:8px">'
            'CONFLICT → CHANNEL → ASSET FLOW</p>',
            unsafe_allow_html=True,
        )
        _render_sankey(results)

    # ── Section 3: Commodity exposure ──────────────────────────────────────
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        'color:#CFB991;letter-spacing:.16em;border-bottom:1px solid #1e1e1e;padding-bottom:4px;margin:1.4rem 0 .6rem">'
        'PORTFOLIO COMMODITY EXPOSURE  '
        '<span style="color:#555960">(CIS-weighted relevance)</span></p>',
        unsafe_allow_html=True,
    )
    _render_commodity_exposure(results)

    _page_footer()
