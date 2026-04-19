"""
Exposure Scoring Page.

Per-security geopolitical exposure framework:
  - Scenario-adjusted exposure score (SAS 0–100) for every tracked asset
  - Conflict beta decomposition: which conflict drives each asset's exposure
  - Ranked exposure table (filterable by direction, sector, conflict)
  - Top hedge assets
  - Conflict selector: full affected universe for a given conflict

Uses: exposure.py, conflict_model.py, scenario_state.py
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from src.ui.shared import _page_header, _page_footer

from src.analysis.exposure import (
    score_all_assets,
    ranked_by_exposure,
    ranked_hedges,
    conflict_affected_universe,
    exposure_summary_stats,
)
from src.analysis.conflict_model import score_all_conflicts
from src.analysis.scenario_state import get_scenario, get_scenario_id
from src.data.config import CONFLICTS


# ── Color helpers ─────────────────────────────────────────────────────────────

def _sas_color(sas: float) -> str:
    if sas >= 60: return "#c0392b"
    if sas >= 35: return "#e67e22"
    if sas >= 15: return "#CFB991"
    return "#8E9AAA"

def _dir_badge(direction: str) -> tuple[str, str]:
    return {
        "long_geo_risk": ("GEO-LONG",   "#e67e22"),
        "safe_haven":    ("SAFE HAVEN", "#27ae60"),
        "neutral":       ("NEUTRAL",    "#8E9AAA"),
    }.get(direction, ("-", "#555960"))


# ── KPI header row ────────────────────────────────────────────────────────────

def _render_kpi_row(stats: dict, scenario: dict) -> None:
    geo_mult = scenario.get("geo_mult", 1.0)
    mult_color = "#c0392b" if geo_mult > 1.1 else "#27ae60" if geo_mult < 0.9 else "#CFB991"

    items = [
        ("Assets Tracked",    str(stats.get("n_assets", 0)),          "#CFB991"),
        ("Mean Exposure",     f'{stats.get("mean_sas", 0):.1f}',      _sas_color(stats.get("mean_sas", 0))),
        ("Peak Exposure",     f'{stats.get("max_sas", 0):.1f}',       _sas_color(stats.get("max_sas", 0))),
        ("High Exp Assets",   str(stats.get("high_exp", 0)),          "#c0392b"),
        ("Scenario Mult",     f'×{geo_mult:.2f}',                     mult_color),
        ("Top Hedge",         stats.get("top_hedge", "-") or "-",    "#27ae60"),
    ]

    cols = st.columns(len(items))
    for col, (label, value, color) in zip(cols, items):
        with col:
            st.markdown(
                f'<div style="background:#0d0d0d;border:1px solid #1e1e1e;'
                f'padding:8px 10px">'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
                f'color:#555960;letter-spacing:.16em;text-transform:uppercase;'
                f'display:block">{label}</span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:18px;'
                f'font-weight:700;color:{color};line-height:1.2;display:block">{value}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Exposure bar table ────────────────────────────────────────────────────────

def _render_exposure_table(items: list[dict], title: str = "RANKED EXPOSURE") -> None:
    st.markdown(
        f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        f'color:#CFB991;letter-spacing:.16em;border-bottom:1px solid #1e1e1e;padding-bottom:4px;margin-bottom:8px">{title}</p>',
        unsafe_allow_html=True,
    )
    if not items:
        st.caption("No assets match current filters.")
        return

    rows_html = ""
    for a in items:
        sas      = a["sas"]
        sas_col  = _sas_color(sas)
        dir_text, dir_col = _dir_badge(a["direction"])
        top_c    = a.get("top_conflict", "") or ""
        top_c_label = top_c.replace("_", " ").upper()
        # sector tags - first two
        sectors  = " · ".join(
            s.replace("_", " ").title() for s in a.get("sector_tags", [])[:2]
        )
        bar_w    = int(sas)

        rows_html += (
            f'<div style="display:flex;align-items:center;gap:8px;'
            f'padding:4px 0;border-bottom:1px solid #141414">'
            # Asset name
            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:10px;'
            f'color:#c8cdd8;width:150px;flex-shrink:0;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
            f'{a["asset"]}</span>'
            # Direction badge
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
            f'color:{dir_col};border:1px solid {dir_col};padding:1px 4px;'
            f'flex-shrink:0;width:72px;text-align:center">{dir_text}</span>'
            # SAS bar
            f'<div style="flex:1;height:5px;background:#1a1a1a;min-width:80px">'
            f'<div style="width:{bar_w}%;height:100%;background:{sas_col}"></div>'
            f'</div>'
            # SAS value
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
            f'color:{sas_col};width:30px;text-align:right;flex-shrink:0">'
            f'{sas:.0f}</span>'
            # Top conflict
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
            f'color:#555960;width:80px;flex-shrink:0;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
            f'{top_c_label}</span>'
            f'</div>'
        )

    st.markdown(
        f'<div style="background:#0a0a0a;border:1px solid #1e1e1e;'
        f'padding:6px 12px">{rows_html}</div>',
        unsafe_allow_html=True,
    )


# ── Conflict beta heatmap for selected asset ──────────────────────────────────

def _render_beta_chart(asset_data: dict) -> None:
    beta = asset_data.get("beta", {})
    if not beta:
        return

    # Sort conflicts by beta descending
    items = sorted(beta.items(), key=lambda x: x[1], reverse=True)
    labels = [cid.replace("_", " ").title() for cid, _ in items]
    vals   = [v * 100 for _, v in items]   # scale to 0–100 for display

    bar_colors = [
        "#c0392b" if v >= 60 else "#e67e22" if v >= 30 else "#8E9AAA"
        for v in vals
    ]

    fig = go.Figure(go.Bar(
        x=labels, y=vals,
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v:.0f}" for v in vals],
        textposition="outside",
        textfont=dict(size=8, family="JetBrains Mono, monospace", color="#8E9AAA"),
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            tickfont=dict(size=8, family="JetBrains Mono, monospace"),
            tickangle=-20,
        ),
        yaxis=dict(
            range=[0, max(vals) * 1.35 if vals else 30],
            tickfont=dict(size=7, family="JetBrains Mono, monospace"),
            showgrid=True, gridcolor="#1a1a1a",
            title=dict(text="Beta ×100", font=dict(size=8)),
        ),
        bargap=0.25,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Conflict radar: all assets beta to one conflict ───────────────────────────

def _render_conflict_affected_chart(conflict_id: str, conflict_color: str) -> None:
    items = conflict_affected_universe(conflict_id, min_beta=0.08)
    if not items:
        st.caption("No assets with meaningful beta to this conflict.")
        return

    items = items[:15]
    labels = [a["asset"] for a in items]
    betas  = [a["beta"].get(conflict_id, 0) * 100 for a in items]
    dirs   = [a["direction"] for a in items]

    bar_colors = [
        "#27ae60" if d == "safe_haven" else "#e67e22" if d == "long_geo_risk" else "#8E9AAA"
        for d in dirs
    ]

    fig = go.Figure(go.Bar(
        x=labels, y=betas,
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v:.0f}" for v in betas],
        textposition="outside",
        textfont=dict(size=8, family="JetBrains Mono, monospace", color="#8E9AAA"),
        hovertemplate="%{x}: beta=%{y:.1f}<extra></extra>",
    ))
    fig.add_hline(
        y=float(np.mean(betas)),
        line=dict(color=conflict_color, width=1, dash="dot"),
        annotation_text="avg",
        annotation_font=dict(size=7, color=conflict_color),
    )
    fig.update_layout(
        height=230,
        margin=dict(l=10, r=10, t=10, b=70),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            tickfont=dict(size=7.5, family="JetBrains Mono, monospace"),
            tickangle=-35,
        ),
        yaxis=dict(
            tickfont=dict(size=7, family="JetBrains Mono, monospace"),
            showgrid=True, gridcolor="#1a1a1a",
        ),
        bargap=0.22,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Hedge asset panel ─────────────────────────────────────────────────────────

def _render_hedge_panel() -> None:
    hedges = ranked_hedges(n=8)
    if not hedges:
        return

    rows_html = ""
    for a in hedges:
        hs  = a["hedge_score"]
        bar = int(hs)
        rows_html += (
            f'<div style="display:flex;align-items:center;gap:8px;'
            f'padding:4px 0;border-bottom:1px solid #141414">'
            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:10px;'
            f'color:#c8cdd8;width:170px;flex-shrink:0">{a["asset"]}</span>'
            f'<div style="flex:1;height:5px;background:#1a1a1a">'
            f'<div style="width:{bar}%;height:100%;background:#27ae60"></div>'
            f'</div>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
            f'color:#27ae60;width:30px;text-align:right">{hs:.0f}</span>'
            f'</div>'
        )

    st.markdown(
        f'<div style="background:#0a0a0a;border:1px solid #1e1e1e;'
        f'padding:6px 12px">{rows_html}</div>',
        unsafe_allow_html=True,
    )


# ── Filters sidebar ────────────────────────────────────────────────────────────

def _exposure_filters() -> dict:
    with st.expander("Filters", expanded=False):
        direction = st.selectbox(
            "Direction",
            ["All", "long_geo_risk", "safe_haven", "neutral"],
            key="exp_dir_filter",
        )
        sector = st.selectbox(
            "Sector",
            ["All", "energy", "agriculture", "precious_metals", "industrial",
             "tech", "broad_equity", "safe_haven", "defense"],
            key="exp_sector_filter",
        )
        conflict_filter = st.selectbox(
            "Filter by conflict",
            ["All"] + [c["id"] for c in CONFLICTS],
            format_func=lambda x: x if x == "All" else next(
                (c["name"] for c in CONFLICTS if c["id"] == x), x
            ),
            key="exp_conf_filter",
        )
        n_show = st.slider("Max assets", 5, 32, 20, key="exp_n_show")

    return {
        "direction": None if direction == "All" else direction,
        "sector":    None if sector    == "All" else sector,
        "conflict":  None if conflict_filter == "All" else conflict_filter,
        "n":         n_show,
    }


# ── Main page ─────────────────────────────────────────────────────────────────

def page_exposure_scoring(start=None, end=None, fred_key="") -> None:
    _page_header("Exposure Scoring",
                 "Per-asset SES · TAE · Scenario-adjusted score · Conflict beta · Hedge ranking",
                 "INTELLIGENCE / EXPOSURE")

    # ── Scenario context ───────────────────────────────────────────────────
    scenario    = get_scenario()
    scenario_id = get_scenario_id()
    geo_mult    = scenario.get("geo_mult", 1.0)
    mult_color  = "#c0392b" if geo_mult > 1.1 else "#27ae60" if geo_mult < 0.9 else "#CFB991"
    st.markdown(
        f'<div style="background:#0d0d0d;border-left:3px solid {mult_color};'
        f'padding:6px 12px;margin-bottom:12px">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        f'color:{mult_color};font-weight:700">{scenario.get("label","Base").upper()} SCENARIO</span>'
        f'<span style="font-family:\'DM Sans\',sans-serif;font-size:10px;color:#8E9AAA;'
        f'margin-left:10px">{scenario.get("desc","")}</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        f'color:{mult_color};margin-left:12px">geo_mult ×{geo_mult:.2f}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Load data ──────────────────────────────────────────────────────────
    try:
        conflict_results = score_all_conflicts()
        all_assets       = score_all_assets(conflict_results)
    except Exception as e:
        st.error(f"Exposure model error: {e}")
        return

    stats = exposure_summary_stats(all_assets)

    # ── KPI row ────────────────────────────────────────────────────────────
    _render_kpi_row(stats, scenario)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Filters + main table ───────────────────────────────────────────────
    col_main, col_side = st.columns([2.2, 1], gap="large")

    with col_main:
        filters = _exposure_filters()
        ranked  = ranked_by_exposure(
            n=filters["n"],
            direction=filters["direction"],
            sector=filters["sector"],
            conflict_id=filters["conflict"],
        )
        _render_exposure_table(ranked, title="RANKED EXPOSURE - SCENARIO ADJUSTED SCORE")

    with col_side:
        st.markdown(
            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            'color:#CFB991;letter-spacing:.16em;border-bottom:1px solid #1e1e1e;padding-bottom:4px;margin-bottom:8px">TOP HEDGES</p>',
            unsafe_allow_html=True,
        )
        _render_hedge_panel()

    # ── Asset drill-down ───────────────────────────────────────────────────
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        'color:#CFB991;letter-spacing:.16em;border-bottom:1px solid #1e1e1e;padding-bottom:4px;margin:1.4rem 0 .6rem">'
        'ASSET CONFLICT BETA DECOMPOSITION</p>',
        unsafe_allow_html=True,
    )

    asset_names = sorted(all_assets.keys())
    selected_asset = st.selectbox(
        "Select asset",
        asset_names,
        index=asset_names.index("WTI Crude Oil") if "WTI Crude Oil" in asset_names else 0,
        key="exp_asset_select",
        label_visibility="collapsed",
    )

    if selected_asset and selected_asset in all_assets:
        a = all_assets[selected_asset]
        dir_text, dir_col = _dir_badge(a["direction"])
        st.markdown(
            f'<div style="border-left:3px solid {_sas_color(a["sas"])};'
            f'padding:4px 12px;margin-bottom:8px">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
            f'font-weight:700;color:{_sas_color(a["sas"])}">{selected_asset}</span>'
            f'<span style="margin-left:10px;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:7px;color:{dir_col};border:1px solid {dir_col};'
            f'padding:1px 4px">{dir_text}</span>'
            f'<span style="margin-left:10px;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:8px;color:#8E9AAA">'
            f'SES {a["ses"]:.2f} · TAE {a["tae"]:.2f} · SAS {a["sas"]:.0f}'
            + (f' (raw {a["sas_raw"]:.0f} - capped at 100 by scenario mult)' if a.get("sas_capped") else "")
            + f'</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        _render_beta_chart(a)

    # ── Conflict-specific affected universe ────────────────────────────────
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        'color:#CFB991;letter-spacing:.16em;border-bottom:1px solid #1e1e1e;padding-bottom:4px;margin:1.4rem 0 .6rem">'
        'CONFLICT AFFECTED UNIVERSE</p>',
        unsafe_allow_html=True,
    )

    conflict_options = {c["id"]: c for c in CONFLICTS}
    conflict_id_sel  = st.selectbox(
        "Select conflict",
        list(conflict_options.keys()),
        format_func=lambda x: conflict_options[x]["name"],
        key="exp_conflict_select",
        label_visibility="collapsed",
    )

    if conflict_id_sel:
        conf = conflict_options[conflict_id_sel]
        st.markdown(
            f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            f'color:#CFB991;letter-spacing:.16em;border-bottom:1px solid #1e1e1e;padding-bottom:4px;margin-bottom:8px">'
            f'ASSETS EXPOSED TO {conf["name"].upper()}</p>',
            unsafe_allow_html=True,
        )
        _render_conflict_affected_chart(conflict_id_sel, conf.get("color", "#CFB991"))

        # Table view of full affected universe
        affected = conflict_affected_universe(conflict_id_sel, min_beta=0.05)
        if affected:
            # convert to same format as exposure table uses
            _render_exposure_table(
                affected[:20],
                title=f"FULL AFFECTED UNIVERSE - {conf['label'].upper()}",
            )

    # ── Methodology footnote ───────────────────────────────────────────────
    st.markdown(
        '<div style="background:#0a0a0a;border:1px solid #1e1e1e;'
        'padding:10px 14px;margin-top:16px">'
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
        'color:#555960;letter-spacing:.16em;margin:0 0 4px">METHODOLOGY</p>'
        '<p style="font-family:\'DM Sans\',sans-serif;font-size:9px;'
        'color:#8E9AAA;line-height:1.6;margin:0">'
        'SES (Structural Exposure Score): CIS-weighted average of per-conflict structural '
        'exposure values from SECURITY_EXPOSURE registry. '
        'TAE (Transmission-Adjusted Exposure): SES scaled by each conflict\'s TPS - '
        'reflects how much structural exposure is being transmitted through markets. '
        'SAS (Scenario-Adjusted Score): TAE × scenario geo_mult × 100. '
        'Conflict Beta: structural[asset][conflict] × TPS[conflict] / 100. '
        'Hedge Score: CIS-weighted presence in conflict hedge_assets lists.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    _page_footer()
