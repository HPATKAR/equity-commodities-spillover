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
    Cell values are channel transmission × (conflict CIS / max CIS) — so dominant
    conflicts show proportionally stronger colour.
    """
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        'color:#8E9AAA;letter-spacing:2px;margin-bottom:4px">'
        'CONFLICT × CHANNEL TRANSMISSION MATRIX  '
        '<span style="color:#555960">(cell = raw channel intensity)</span></p>',
        unsafe_allow_html=True,
    )

    ranked = sorted(results.values(), key=lambda r: r["cis"], reverse=True)
    max_cis = max(r["cis"] for r in ranked) + 1e-9

    conf_map = {c["id"]: c for c in CONFLICTS}
    z_raw, z_weighted, labels_y = [], [], []

    for r in ranked:
        conf = conf_map.get(r["id"], {})
        tx   = conf.get("transmission", {})
        raw_row      = [float(tx.get(ch, 0.0)) for ch in _CHANNELS]
        weighted_row = [v * (r["cis"] / max_cis) for v in raw_row]
        z_raw.append(raw_row)
        z_weighted.append(weighted_row)
        labels_y.append(f'{r["label"]}  [{r["cis"]:.0f}]')

    ch_labels_x = [_CH_LABELS[ch] for ch in _CHANNELS]

    tab_raw, tab_wt = st.tabs(["Raw Intensity", "CIS-Weighted"])

    for tab, z_data, title in [
        (tab_raw, z_raw,      "Raw"),
        (tab_wt,  z_weighted, "CIS-Weighted"),
    ]:
        with tab:
            fig = go.Figure(go.Heatmap(
                z=z_data,
                x=ch_labels_x,
                y=labels_y,
                colorscale=[
                    [0.00, "#090909"],
                    [0.15, "#151c2b"],
                    [0.30, "#1e3050"],
                    [0.50, "#8E6F3E"],
                    [0.70, "#e67e22"],
                    [1.00, "#c0392b"],
                ],
                zmin=0,
                zmax=1 if title == "Raw" else (max_cis / 100),
                text=[[f"{v:.2f}" for v in row] for row in z_data],
                texttemplate="%{text}",
                textfont=dict(size=7, family="JetBrains Mono, monospace"),
                hoverongaps=False,
                showscale=True,
                colorbar=dict(
                    thickness=8, len=0.65,
                    tickfont=dict(size=7, family="JetBrains Mono, monospace"),
                ),
            ))
            fig.update_layout(
                height=240,
                margin=dict(l=10, r=40, t=10, b=70),
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
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})


# ── Channel dominance bar chart ───────────────────────────────────────────────

def _render_channel_dominance(results: dict) -> None:
    """
    Portfolio-level channel stress: intensity-weighted average per channel.
    """
    conf_map = {c["id"]: c for c in CONFLICTS}
    cis_vals = {r["id"]: r["cis"] for r in results.values()}
    total_cis = sum(cis_vals.values()) + 1e-9

    channel_scores = {ch: 0.0 for ch in _CHANNELS}
    for cid, cis in cis_vals.items():
        conf = conf_map.get(cid, {})
        tx   = conf.get("transmission", {})
        w    = cis / total_cis
        for ch in _CHANNELS:
            channel_scores[ch] += float(tx.get(ch, 0.0)) * w

    # Normalize to 0–100
    max_score = max(channel_scores.values()) + 1e-9
    sorted_ch = sorted(channel_scores.items(), key=lambda x: x[1], reverse=True)

    labels = [_CH_LABELS[ch] for ch, _ in sorted_ch]
    values = [v * 100 / max_score for _, v in sorted_ch]
    raw_v  = [v for _, v in sorted_ch]

    bar_colors = [
        "#c0392b" if v >= 0.75 else "#e67e22" if v >= 0.45 else "#8E9AAA"
        for v in raw_v
    ]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v:.0f}" for v in values],
        textposition="outside",
        textfont=dict(size=8, family="JetBrains Mono, monospace", color="#8E9AAA"),
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=60, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 120],
            tickfont=dict(size=7, family="JetBrains Mono, monospace"),
            showgrid=True, gridcolor="#1a1a1a",
        ),
        yaxis=dict(
            tickfont=dict(size=8.5, family="JetBrains Mono, monospace"),
            autorange="reversed",
        ),
        bargap=0.22,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Sankey: conflict → channel → asset class ─────────────────────────────────

def _render_sankey(results: dict) -> None:
    conf_map = {c["id"]: c for c in CONFLICTS}
    cis_vals = {r["id"]: r["cis"] for r in results.values()}

    # Nodes: conflicts + channels + asset classes
    conflict_ids  = list(results.keys())
    channel_ids   = _CHANNELS
    asset_classes = list(dict.fromkeys(_CH_ASSET_MAP.values()))  # ordered unique

    all_nodes = conflict_ids + channel_ids + asset_classes
    node_idx  = {n: i for i, n in enumerate(all_nodes)}

    conflict_colors = {r["id"]: r["color"] for r in results.values()}
    ch_color = "#8E6F3E"
    ac_colors = {
        "Commodities":   "#e67e22",
        "Logistics":     "#2980b9",
        "Policy":        "#c0392b",
        "Equities":      "#27ae60",
        "FX":            "#8E9AAA",
        "Macro":         "#CFB991",
        "Fixed Income":  "#6c3483",
    }

    node_colors = (
        [conflict_colors.get(cid, "#8E9AAA") for cid in conflict_ids]
        + [ch_color] * len(channel_ids)
        + [ac_colors.get(ac, "#8E9AAA") for ac in asset_classes]
    )

    sources, targets, values, link_colors = [], [], [], []

    min_cis = 5.0  # minimum flow threshold
    for cid in conflict_ids:
        conf = conf_map.get(cid, {})
        tx   = conf.get("transmission", {})
        cis  = cis_vals.get(cid, 30.0)
        for ch in channel_ids:
            raw = float(tx.get(ch, 0.0))
            flow = raw * cis
            if flow < min_cis:
                continue
            sources.append(node_idx[cid])
            targets.append(node_idx[ch])
            values.append(round(flow, 1))
            base_c = conflict_colors.get(cid, "#8E9AAA")
            link_colors.append(base_c.replace("#", "rgba(") + ",0.35)")

    # Fall back to proper rgba
    link_colors_clean = []
    for lc, base in zip(link_colors, [conflict_colors.get(conflict_ids[s - 0], "#8E9AAA")
                                       for s in sources]):
        try:
            r = int(base[1:3], 16)
            g = int(base[3:5], 16)
            b = int(base[5:7], 16)
            link_colors_clean.append(f"rgba({r},{g},{b},0.35)")
        except Exception:
            link_colors_clean.append("rgba(142,154,170,0.30)")

    # Channel → asset class flows
    for ch in channel_ids:
        ac = _CH_ASSET_MAP.get(ch, "Commodities")
        # Sum flow into this channel
        ch_total = sum(
            float(conf_map.get(cid, {}).get("transmission", {}).get(ch, 0.0)) * cis_vals.get(cid, 0)
            for cid in conflict_ids
        )
        if ch_total < min_cis:
            continue
        sources.append(node_idx[ch])
        targets.append(node_idx[ac])
        values.append(round(ch_total, 1))
        ac_color = ac_colors.get(ac, "#8E9AAA")
        try:
            r = int(ac_color[1:3], 16)
            g = int(ac_color[3:5], 16)
            b = int(ac_color[5:7], 16)
            link_colors_clean.append(f"rgba({r},{g},{b},0.25)")
        except Exception:
            link_colors_clean.append("rgba(142,154,170,0.20)")

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=12, thickness=14,
            line=dict(color="#0a0a0a", width=0.3),
            label=all_nodes,
            color=node_colors,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors_clean,
            hovertemplate="Flow: %{value:.0f}<extra></extra>",
        ),
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=8, family="JetBrains Mono, monospace", color="#8E9AAA"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Most-affected commodities table ──────────────────────────────────────────

def _render_commodity_exposure(results: dict) -> None:
    """
    Conflict-commodity relevance matrix aggregated to portfolio level.
    Shows top 12 most-affected commodities.
    """
    matrix = conflict_commodity_matrix()
    cis_vals = {r["id"]: r["cis"] for r in results.values()}
    total_cis = sum(cis_vals.values()) + 1e-9

    # Portfolio-level commodity exposure = CIS-weighted average
    commodity_scores: dict[str, float] = {}
    for cid, cm in matrix.items():
        w = cis_vals.get(cid, 0) / total_cis
        for commodity, score in cm.items():
            commodity_scores[commodity] = commodity_scores.get(commodity, 0) + score * w

    ranked = sorted(commodity_scores.items(), key=lambda x: x[1], reverse=True)[:12]
    if not ranked:
        return

    rows_html = ""
    for commodity, score in ranked:
        bar_w = int(score * 100)
        color = "#c0392b" if score >= 0.75 else "#e67e22" if score >= 0.45 else "#8E9AAA"
        rows_html += (
            f'<div style="display:flex;align-items:center;gap:8px;'
            f'padding:3px 0;border-bottom:1px solid #1a1a1a">'
            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:10px;'
            f'color:#c8cdd8;width:160px;flex-shrink:0">{commodity}</span>'
            f'<div style="flex:1;height:5px;background:#1a1a1a">'
            f'<div style="width:{bar_w}%;height:100%;background:{color}"></div>'
            f'</div>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            f'color:{color};width:35px;text-align:right">{score:.2f}</span>'
            f'</div>'
        )

    st.markdown(
        f'<div style="background:#0a0a0a;border:1px solid #1e1e1e;'
        f'padding:8px 12px">{rows_html}</div>',
        unsafe_allow_html=True,
    )


# ── Main page ─────────────────────────────────────────────────────────────────

def page_transmission_matrix(start=None, end=None, fred_key="") -> None:
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
        'color:#8E9AAA;letter-spacing:3px;text-transform:uppercase;margin:0">'
        'INTELLIGENCE / TRANSMISSION</p>'
        '<h2 style="font-family:\'DM Sans\',sans-serif;font-size:1.35rem;'
        'font-weight:700;color:#e8e9ed;margin:4px 0 16px">Transmission Matrix</h2>',
        unsafe_allow_html=True,
    )

    try:
        results = score_all_conflicts()
    except Exception as e:
        st.error(f"Error loading conflict scores: {e}")
        return

    if not results:
        st.warning("No conflict data available.")
        return

    # ── Section 1: Heatmap ─────────────────────────────────────────────────
    _render_weighted_heatmap(results)

    # ── Section 2: Channel dominance + Sankey ──────────────────────────────
    col_l, col_r = st.columns([1, 1.4])

    with col_l:
        st.markdown(
            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            'color:#8E9AAA;letter-spacing:2px;margin-bottom:4px">'
            'PORTFOLIO CHANNEL STRESS</p>',
            unsafe_allow_html=True,
        )
        _render_channel_dominance(results)

    with col_r:
        st.markdown(
            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            'color:#8E9AAA;letter-spacing:2px;margin-bottom:4px">'
            'CONFLICT → CHANNEL → ASSET FLOW</p>',
            unsafe_allow_html=True,
        )
        _render_sankey(results)

    # ── Section 3: Commodity exposure ──────────────────────────────────────
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        'color:#8E9AAA;letter-spacing:2px;margin:1.2rem 0 0.4rem">'
        'PORTFOLIO COMMODITY EXPOSURE  '
        '<span style="color:#555960">(CIS-weighted relevance)</span></p>',
        unsafe_allow_html=True,
    )
    _render_commodity_exposure(results)
