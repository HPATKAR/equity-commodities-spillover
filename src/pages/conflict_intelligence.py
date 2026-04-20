"""
Conflict Intelligence Page.

Per-conflict scorecard grid — one card per conflict showing:
  CIS, TPS, confidence, trend, state, freshness

Detailed drill-down for selected conflict:
  - Intensity dimension breakdown (bar)
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
from src.ui.shared import _page_header, _page_footer

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
    Returns the selected conflict_id.
    """
    ids = list(results.keys())
    if not ids:
        return None

    # ── Selection state ────────────────────────────────────────────────────
    if "ci_selected" not in st.session_state or st.session_state.ci_selected not in ids:
        st.session_state.ci_selected = ids[0]

    # ── Render cards in groups of 3 ────────────────────────────────────────
    for row_start in range(0, len(ids), 3):
        row_ids = ids[row_start:row_start + 3]
        cols = st.columns(len(row_ids))

        for col, cid in zip(cols, row_ids):
            r = results[cid]
            is_selected = (cid == st.session_state.ci_selected)
            border_color = r["color"] if is_selected else "#2a2a2a"
            bg_color     = "#0f0f0f" if is_selected else "#0a0a0a"
            state_lbl, state_col = _state_badge(r["state"])
            trend_sym = _trend_marker(r["trend"])
            trend_col = {"rising": "#c0392b", "stable": "#8E9AAA", "falling": "#27ae60"}.get(r["trend"], "#8E9AAA")

            with col:
                st.markdown(
                    f'<div style="border:1px solid {border_color};border-top:3px solid {r["color"]};'
                    f'background:{bg_color};padding:.7rem .9rem;border-radius:0;">'
                    # Label + state badge
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;font-weight:700;'
                    f'color:{r["color"]}">{r["label"]}</span>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;font-weight:700;'
                    f'color:{state_col};border:1px solid {state_col};padding:1px 5px">{state_lbl}</span>'
                    f'</div>'
                    # Name
                    f'<div style="font-family:\'DM Sans\',sans-serif;font-size:11px;font-weight:600;'
                    f'color:#e8e9ed;margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
                    f'{r["name"]}</div>'
                    # CIS / TPS scores
                    f'<div style="display:flex;gap:10px;margin-bottom:5px">'
                    f'<div><span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;color:#555960">CIS</span><br>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:18px;font-weight:700;'
                    f'color:{_cis_color(r["cis"])}">{r["cis"]:.0f}</span></div>'
                    f'<div><span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;color:#555960">TPS</span><br>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:18px;font-weight:700;'
                    f'color:{_tps_color(r["tps"])}">{r["tps"]:.0f}</span></div>'
                    f'<div style="margin-left:auto;text-align:right">'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;color:#555960">CONF</span><br>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#CFB991">'
                    f'{r["confidence"]:.0%}</span></div>'
                    f'</div>'
                    # Trend + freshness
                    f'<div style="display:flex;justify-content:space-between;margin-top:3px">'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;font-weight:700;'
                    f'color:{trend_col}">{trend_sym} {r["trend"].upper()}</span>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
                    f'color:{_freshness_color(r["freshness"])}">{r["freshness"].upper()}</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button("Select", key=f"ci_sel_{cid}", width="stretch"):
                    st.session_state.ci_selected = cid
                    st.rerun()

    return st.session_state.ci_selected


# ── Intensity dimension breakdown ─────────────────────────────────────────────

def _render_intensity_breakdown(selected_id: str, selected: dict) -> None:
    """
    Horizontal bar chart of CIS intensity dimensions.
    Uses the 7 standard CIS dimensions, pulling values from the CONFLICTS registry.
    """
    from src.data.config import CONFLICTS

    conflict_raw = next((c for c in CONFLICTS if c["id"] == selected_id), None)
    if conflict_raw is None:
        st.caption("No dimension data.")
        return

    from src.analysis.conflict_model import _ESCALATION_MAP, _recency_score

    dims = {
        "Deadliness":           float(conflict_raw.get("deadliness",           0.5)),
        "Civilian Danger":      float(conflict_raw.get("civilian_danger",      0.5)),
        "Geo Diffusion":        float(conflict_raw.get("geographic_diffusion", 0.3)),
        "Fragmentation":        float(conflict_raw.get("fragmentation",        0.2)),
        "Escalation Trend":     _ESCALATION_MAP.get(
                                    conflict_raw.get("escalation_trend", "stable"), 0.5),
        "Recency":              _recency_score(conflict_raw),
        "Source Coverage":      float(conflict_raw.get("source_coverage",      0.7)),
    }

    labels = list(dims.keys())[::-1]
    values = [dims[k] for k in labels]
    bar_colors = [
        "#c0392b" if v >= 0.7 else "#e67e22" if v >= 0.45 else "#CFB991" if v >= 0.25 else "#8E9AAA"
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=9, color="#8E9AAA"),
        hovertemplate="%{y}: %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
        margin=dict(l=10, r=40, t=10, b=10),
        height=220,
        xaxis=dict(range=[0, 1.15], tickfont=dict(family="JetBrains Mono", size=8, color="#555960"),
                   gridcolor="#1e1e1e", showgrid=True),
        yaxis=dict(tickfont=dict(family="JetBrains Mono", size=8, color="#8E9AAA"),
                   showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ── TPS channel breakdown ─────────────────────────────────────────────────────

def _render_tps_channels(selected_id: str) -> None:
    """
    Horizontal bar chart of TPS transmission channel weights for the selected conflict.
    """
    from src.data.config import CONFLICTS

    conflict_raw = next((c for c in CONFLICTS if c["id"] == selected_id), None)
    if conflict_raw is None:
        st.caption("No transmission data.")
        return

    tx = conflict_raw.get("transmission", {})
    if not tx:
        st.caption("No transmission channels defined.")
        return

    _CH_LABELS = {
        "oil_gas": "Oil/Gas", "metals": "Metals", "agriculture": "Agriculture",
        "shipping": "Shipping", "chokepoint": "Chokepoint", "sanctions": "Sanctions",
        "equity_sector": "Equity Sector", "fx": "FX", "inflation": "Inflation",
        "supply_chain": "Supply Chain", "credit": "Credit", "energy_infra": "Energy Infra",
    }

    sorted_tx = sorted(tx.items(), key=lambda x: x[1])
    channels = [_CH_LABELS.get(k, k) for k, _ in sorted_tx]
    weights  = [v for _, v in sorted_tx]

    bar_colors = [
        "#c0392b" if v >= 0.6 else "#e67e22" if v >= 0.35 else "#CFB991" if v >= 0.15 else "#555960"
        for v in weights
    ]

    fig = go.Figure(go.Bar(
        x=weights, y=channels, orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.0%}" for v in weights],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=9, color="#8E9AAA"),
        hovertemplate="%{y}: %{x:.0%}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
        margin=dict(l=10, r=50, t=10, b=10),
        height=220,
        xaxis=dict(range=[0, 1.20], tickformat=".0%",
                   tickfont=dict(family="JetBrains Mono", size=8, color="#555960"),
                   gridcolor="#1e1e1e", showgrid=True),
        yaxis=dict(tickfont=dict(family="JetBrains Mono", size=8, color="#8E9AAA"),
                   showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ── Top affected assets ───────────────────────────────────────────────────────

def _render_affected_assets(selected_id: str, selected: dict) -> None:
    """
    Compact table of top affected commodities, equities, and hedge assets.
    """
    rows = []

    # Structured exposure scores
    struct_assets = top_affected_assets(selected_id, n=5)
    for item in struct_assets:
        rows.append({"Asset": item["asset"], "Exposure": f'{item["exposure"]:.2f}', "Type": "Structured"})

    # Fallback to config lists if no structured exposure
    if not rows:
        for a in selected.get("affected_commodities", [])[:3]:
            rows.append({"Asset": a, "Exposure": "—", "Type": "Commodity"})
        for a in selected.get("affected_equities", [])[:3]:
            rows.append({"Asset": a, "Exposure": "—", "Type": "Equity"})

    if not rows:
        st.caption("No asset exposure data.")
        return

    # Hedge assets strip
    hedge = selected.get("hedge_assets", [])

    # Render as HTML table
    header = (
        '<table style="width:100%;border-collapse:collapse;font-family:\'JetBrains Mono\',monospace;font-size:9px">'
        '<thead><tr>'
        '<th style="color:#555960;text-align:left;border-bottom:1px solid #2a2a2a;padding:3px 6px">ASSET</th>'
        '<th style="color:#555960;text-align:right;border-bottom:1px solid #2a2a2a;padding:3px 6px">EXPOSURE</th>'
        '<th style="color:#555960;text-align:right;border-bottom:1px solid #2a2a2a;padding:3px 6px">TYPE</th>'
        '</tr></thead><tbody>'
    )
    body = ""
    for row in rows[:6]:
        body += (
            f'<tr style="border-bottom:1px solid #1a1a1a">'
            f'<td style="color:#e8e9ed;padding:4px 6px">{row["Asset"]}</td>'
            f'<td style="color:#CFB991;text-align:right;padding:4px 6px">{row["Exposure"]}</td>'
            f'<td style="color:#555960;text-align:right;padding:4px 6px">{row["Type"]}</td>'
            f'</tr>'
        )
    footer = '</tbody></table>'

    st.markdown(header + body + footer, unsafe_allow_html=True)

    if hedge:
        hedge_str = " · ".join(hedge[:4])
        st.markdown(
            f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            f'color:#27ae60;margin-top:6px">HEDGE: {hedge_str}</p>',
            unsafe_allow_html=True,
        )


# ── Live conflict news ────────────────────────────────────────────────────────

def _render_conflict_news(selected_id: str) -> None:
    """
    Render GPR-classified news headlines for the selected conflict.
    Falls back to generic RSS panel if GPR layer unavailable.
    """
    try:
        from src.analysis.gpr_news import get_news_gpr_layer
        result = get_news_gpr_layer()
        per_conflict = result.get("per_conflict", {})
        headlines = result.get("headlines", [])

        # Filter to headlines mentioning this conflict
        conflict_headlines = [
            h for h in headlines
            if selected_id in getattr(h, "conflicts", [])
        ]

        if not conflict_headlines:
            # Fall back to first 4 headlines sorted by act_score + threat_score
            conflict_headlines = sorted(
                headlines,
                key=lambda h: getattr(h, "act_score", 0) + getattr(h, "threat_score", 0),
                reverse=True,
            )[:4]

        if not conflict_headlines:
            st.caption("No intelligence feed available.")
            return

        _TYPE_COLOR = {"threat": "#e67e22", "act": "#c0392b", "neutral": "#555960"}
        _TYPE_LABEL = {"threat": "THR", "act": "ACT", "neutral": "NEU"}

        rows_html = ""
        for h in conflict_headlines[:5]:
            news_type = getattr(h, "news_type", "neutral")
            tc  = _TYPE_COLOR.get(news_type, "#555960")
            tl  = _TYPE_LABEL.get(news_type, "NEU")
            url_attr = f'href="{h.url}" target="_blank"' if getattr(h, "url", "") else ""
            src = getattr(h, "source", "")
            title = getattr(h, "title", "")
            rows_html += (
                f'<div style="border-bottom:1px solid #1a1a1a;padding:5px 0">'
                f'<div style="display:flex;gap:6px;align-items:flex-start">'
                f'<span style="font-size:7px;font-weight:700;color:{tc};border:1px solid {tc};'
                f'padding:1px 4px;flex-shrink:0;margin-top:1px">{tl}</span>'
                f'<div>'
                f'<a {url_attr} style="font-family:\'DM Sans\',sans-serif;font-size:10px;'
                f'font-weight:500;color:#e8e9ed;text-decoration:none;line-height:1.4;display:block">'
                f'{title}</a>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;color:#555960">'
                f'{src}</span>'
                f'</div></div></div>'
            )

        st.markdown(
            f'<div style="background:#0a0a0a;border:1px solid #2a2a2a;padding:.4rem .8rem">'
            f'{rows_html}</div>',
            unsafe_allow_html=True,
        )

    except Exception:
        # Fallback to generic RSS panel
        try:
            from src.ingestion.geo_rss import render_rss_panel
            render_rss_panel(max_items=4)
        except Exception:
            st.caption("Intelligence feed unavailable.")


# ── Transmission heatmap ──────────────────────────────────────────────────────

def _render_transmission_heatmap(results: dict) -> None:
    """
    Heatmap of conflicts × transmission channels, cells weighted by CIS.
    """
    _CHANNELS = [
        "oil_gas", "metals", "agriculture", "shipping", "chokepoint",
        "sanctions", "equity_sector", "fx", "inflation",
        "supply_chain", "credit", "energy_infra",
    ]
    _CH_LABELS = {
        "oil_gas": "Oil/Gas", "metals": "Metals", "agriculture": "Agriculture",
        "shipping": "Shipping", "chokepoint": "Chokepoint", "sanctions": "Sanctions",
        "equity_sector": "Eq. Sector", "fx": "FX", "inflation": "Inflation",
        "supply_chain": "Supply Chain", "credit": "Credit", "energy_infra": "Energy Infra",
    }

    from src.data.config import CONFLICTS
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
            thickness=10, len=0.8,
            tickfont=dict(family="JetBrains Mono", size=8, color="#8E9AAA"),
            title=dict(text="Weight", font=dict(family="JetBrains Mono", size=8, color="#555960")),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
        margin=dict(l=10, r=10, t=10, b=50),
        height=max(180, len(conflict_ids) * 38),
        xaxis=dict(
            tickfont=dict(family="JetBrains Mono", size=8, color="#8E9AAA"),
            tickangle=-35, showgrid=False,
        ),
        yaxis=dict(
            tickfont=dict(family="JetBrains Mono", size=8, color="#8E9AAA"),
            showgrid=False,
        ),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ── Page entry point ──────────────────────────────────────────────────────────

def page_conflict_intelligence(start=None, end=None, fred_key: str = "") -> None:
    _page_header(
        "Conflict Intelligence Center",
        "CIS scoring · 7-dimension intensity · State multiplier · Portfolio weighting",
        "INTELLIGENCE / CONFLICT SCORECARD",
    )

    # ── Live Data Status Banner (ACLED + GDELT) ────────────────────────────
    try:
        from src.data.acled import acled_configured, acled_setup_instructions
        from src.data.gdelt import fetch_all_gdelt_signals

        acled_live = acled_configured()

        # GDELT runs regardless — no key needed; just check it responded
        gdelt_signals = {}
        try:
            gdelt_signals = fetch_all_gdelt_signals(timespan="7d")
        except Exception:
            pass
        gdelt_live = any(v.get("data_available") for v in gdelt_signals.values())

        if acled_live and gdelt_live:
            st.markdown(
                '<div style="background:#050a05;border-left:3px solid #27ae60;'
                'border-radius:4px;padding:8px 14px;margin:4px 0;">'
                '<span style="color:#27ae60;font-family:\'JetBrains Mono\',monospace;'
                'font-size:10px;font-weight:700">● ACLED + GDELT LIVE</span>'
                '<span style="color:#8E9AAA;font-family:\'JetBrains Mono\',monospace;'
                'font-size:10px"> · CIS escalation_trend cross-validated by two independent '
                'live sources. ACLED: event/fatality counts (6h cache). '
                'GDELT: media volume signals (3h cache).</span></div>',
                unsafe_allow_html=True,
            )
        elif gdelt_live:
            st.markdown(
                '<div style="background:#050a0f;border-left:3px solid #2980b9;'
                'border-radius:4px;padding:8px 14px;margin:4px 0;">'
                '<span style="color:#2980b9;font-family:\'JetBrains Mono\',monospace;'
                'font-size:10px;font-weight:700">● GDELT LIVE</span>'
                '<span style="color:#8E9AAA;font-family:\'JetBrains Mono\',monospace;'
                'font-size:10px"> · CIS escalation_trend driven by GDELT media-volume signals '
                '(no API key). Add ACLED API key for event-count corroboration.</span></div>',
                unsafe_allow_html=True,
            )
            with st.expander("Enable ACLED for full corroboration (free, academic)", expanded=False):
                st.markdown(acled_setup_instructions())
        elif acled_live:
            st.markdown(
                '<div style="background:#050a05;border-left:3px solid #27ae60;'
                'border-radius:4px;padding:8px 14px;margin:4px 0;">'
                '<span style="color:#27ae60;font-family:\'JetBrains Mono\',monospace;'
                'font-size:10px;font-weight:700">● ACLED LIVE</span>'
                '<span style="color:#8E9AAA;font-family:\'JetBrains Mono\',monospace;'
                'font-size:10px"> · CIS scores augmented with ACLED event counts '
                '(30-day window). GDELT media signals unavailable.</span></div>',
                unsafe_allow_html=True,
            )
        else:
            with st.expander("Enable live conflict data (ACLED API — free for academic use)", expanded=False):
                st.markdown(acled_setup_instructions())

        # ── GDELT signal summary table (when available) ────────────────────
        if gdelt_live and gdelt_signals:
            with st.expander("GDELT live conflict signals (media volume escalation)", expanded=False):
                rows = []
                for cid, gd in gdelt_signals.items():
                    if not gd.get("data_available"):
                        continue
                    rows.append({
                        "Conflict":     cid.replace("_", " ").title(),
                        "Signal":       gd["escalation_signal"].capitalize(),
                        "Vol Trend":    f"{gd['volume_trend']:+.0%}",
                        "Recent Vol":   gd["volume_recent"],
                        "Tone":         f"{gd['tone_recent']:+.1f}",
                        "Source":       gd["source"],
                        "As Of":        gd["as_of"],
                    })
                if rows:
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                else:
                    st.caption("No GDELT data rows to display.")

    except Exception:
        pass

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

    # ── Scorecard grid ─────────────────────────────────────────────────────
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        'color:#CFB991;letter-spacing:.16em;border-bottom:1px solid #1e1e1e;padding-bottom:4px;margin-bottom:8px">CONFLICT SCORECARDS</p>',
        unsafe_allow_html=True,
    )
    selected_id = _render_scorecard_grid(results)
    if not selected_id or selected_id not in results:
        selected_id = next(iter(results))

    selected = results[selected_id]

    # ── Heatmap ────────────────────────────────────────────────────────────
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        'color:#CFB991;letter-spacing:.16em;border-bottom:1px solid #1e1e1e;padding-bottom:4px;margin:1.4rem 0 .6rem">'
        'TRANSMISSION PRESSURE — CONFLICTS × CHANNELS</p>',
        unsafe_allow_html=True,
    )
    _render_transmission_heatmap(results)

    # ── Conflict detail header ─────────────────────────────────────────────
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
            'color:#8E9AAA;letter-spacing:.16em">INTENSITY DIMENSIONS</p>',
            unsafe_allow_html=True,
        )
        _render_intensity_breakdown(selected_id, selected)

    with col_r:
        st.markdown(
            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            'color:#8E9AAA;letter-spacing:.16em">TRANSMISSION CHANNELS</p>',
            unsafe_allow_html=True,
        )
        _render_tps_channels(selected_id)

    # ── Bottom row: affected assets + news ─────────────────────────────────
    col_a, col_n = st.columns([1, 1])

    with col_a:
        st.markdown(
            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            'color:#CFB991;letter-spacing:.16em;border-bottom:1px solid #1e1e1e;padding-bottom:4px;margin-bottom:8px">'
            'TOP AFFECTED ASSETS</p>',
            unsafe_allow_html=True,
        )
        _render_affected_assets(selected_id, selected)

    with col_n:
        st.markdown(
            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            'color:#CFB991;letter-spacing:.16em;border-bottom:1px solid #1e1e1e;padding-bottom:4px;margin-bottom:8px">'
            'LIVE INTELLIGENCE FEED</p>',
            unsafe_allow_html=True,
        )
        _render_conflict_news(selected_id)

    # ── AI Analyst Deliberation ────────────────────────────────────────────
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        'color:#CFB991;letter-spacing:.16em;border-bottom:1px solid #1e1e1e;padding-bottom:4px;margin:1.4rem 0 .6rem">'
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
                         width="stretch"):
                _cis   = selected["cis"]
                _tps   = selected["tps"]
                _trend = selected["trend"]
                _state = selected["state"]

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

    _page_footer()
