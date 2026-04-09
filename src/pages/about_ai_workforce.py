"""
AI Workforce - dedicated page showing the full AI agent pipeline.
Pipeline block diagram, all agent outputs, CQO flags, remediation results,
and the activity feed. Keeps analysis pages clean.
"""

from __future__ import annotations

import datetime
import streamlit as st

from src.analysis.agent_state import AGENTS, STATUSES, init_agents, get_agent, is_enabled
from src.ui.shared import _page_header, _about_page_styles, _page_footer

_GOLD = "#CFB991"
_F    = "font-family:'DM Sans',sans-serif;"
_M    = "font-family:'JetBrains Mono',monospace;"

# Pipeline round metadata
_ROUNDS = [
    {
        "n": 1, "label": "Round 1 - Data Gatherers",
        "desc": "Independent - no peer dependencies. Each pulls live market data.",
        "agents": ["signal_auditor", "macro_strategist", "geopolitical_analyst"],
        "color": "#2980b9",
    },
    {
        "n": 2, "label": "Round 2 - Synthesisers",
        "desc": "Consume Round 1 outputs. Cross-signal synthesis.",
        "agents": ["risk_officer", "commodities_specialist"],
        "color": "#8E6F3E",
    },
    {
        "n": 3, "label": "Round 3 - Action Layer",
        "desc": "Consume Rounds 1-2. Generate trade ideas and stress scenarios.",
        "agents": ["stress_engineer", "trade_structurer"],
        "color": "#e67e22",
    },
    {
        "n": 4, "label": "Round 4 - Quality Audit",
        "desc": "Per-page. Flags methodological issues and triggers remediation.",
        "agents": ["quality_officer"],
        "color": "#c0392b",
    },
]

# Data inputs per agent (for the block diagram)
_INPUTS = {
    "signal_auditor":       ["Granger hit rates", "Model accuracy", "Confidence history"],
    "macro_strategist":     ["FRED yields", "CPI / inflation", "GDP", "Fed rate"],
    "geopolitical_analyst": ["Active conflict events", "Strait Watch", "Commodity exposure"],
    "risk_officer":         ["Round 1 peer outputs", "Alert feed", "Regime + risk score"],
    "commodities_specialist":["Geo analyst context", "COT positioning", "Price action"],
    "stress_engineer":      ["Risk Officer context", "Stress scenarios", "Tail risk metrics"],
    "trade_structurer":     ["All Round 1-2 outputs", "Regime", "Corr delta", "Risk score"],
    "quality_officer":      ["Page-specific context", "Sample sizes", "p-values", "Assumptions"],
}


def _pipeline_diagram() -> None:
    """Render the agent pipeline as an HTML block diagram."""

    # Build round boxes
    def _agent_node(aid: str, color: str) -> str:
        meta = AGENTS.get(aid, {})
        a    = get_agent(aid)
        conf = a.get("confidence")
        conf_str = f"{int(conf*100)}%" if conf is not None else "-"
        status = a.get("status", "idle")
        status_meta = STATUSES.get(status, STATUSES.get("idle", {}))
        s_color = status_meta.get("color", "#555960") if status_meta else "#555960"

        return (
            f'<div style="background:#131313;border:1px solid #2a2a2a;'
            f'border-top:2px solid {color};padding:0.55rem 0.7rem;'
            f'min-width:140px;flex:1">'
            f'<div style="{_M}font-size:0.80rem;margin-bottom:3px">{meta.get("icon","")}</div>'
            f'<div style="{_F}font-size:0.58rem;font-weight:700;color:{color};'
            f'letter-spacing:0.04em;margin-bottom:2px">{meta.get("short","")}</div>'
            f'<div style="{_F}font-size:0.54rem;color:#555960;line-height:1.4">'
            f'{meta.get("desc","")}</div>'
            f'<div style="margin-top:6px;display:flex;gap:6px;align-items:center">'
            f'<span style="{_M}font-size:0.46rem;font-weight:700;letter-spacing:0.10em;'
            f'text-transform:uppercase;color:{s_color}">{status.replace("_"," ")}</span>'
            f'<span style="{_M}font-size:0.46rem;color:#8890a1">{conf_str}</span>'
            f'</div></div>'
        )

    def _input_node(items: list[str]) -> str:
        bullets = "".join(
            f'<div style="margin:1px 0">{i}</div>' for i in items
        )
        return (
            f'<div style="background:#0d0d0d;border:1px solid #1e1e1e;'
            f'padding:0.4rem 0.6rem;min-width:130px">'
            f'<div style="{_M}font-size:0.44rem;font-weight:700;letter-spacing:0.14em;'
            f'text-transform:uppercase;color:#555960;margin-bottom:4px">Inputs</div>'
            f'<div style="{_F}font-size:0.53rem;color:#8890a1;line-height:1.5">{bullets}</div>'
            f'</div>'
        )

    arrow = (
        '<div style="display:flex;align-items:center;padding:0 4px;color:#2a2a2a;'
        'font-size:0.9rem;align-self:center">&#x25B6;</div>'
    )
    down_arrow = (
        '<div style="text-align:center;font-size:0.7rem;color:#2a2a2a;'
        'margin:4px 0">&#x25BC;</div>'
    )

    html_parts = [
        '<div style="overflow-x:auto;padding-bottom:0.5rem">',
        f'<div style="{_F}font-size:0.50rem;font-weight:700;letter-spacing:0.16em;'
        f'text-transform:uppercase;color:#555960;margin-bottom:0.9rem">'
        f'Agent Pipeline Architecture</div>',
    ]

    for i, rnd in enumerate(_ROUNDS):
        color = rnd["color"]
        agents = rnd["agents"]

        # Round header
        html_parts.append(
            f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.4rem">'
            f'<div style="width:18px;height:18px;background:{color};display:flex;'
            f'align-items:center;justify-content:center;'
            f'{_M}font-size:0.52rem;font-weight:700;color:#fff">{rnd["n"]}</div>'
            f'<div style="{_F}font-size:0.60rem;font-weight:700;color:{color}">{rnd["label"]}</div>'
            f'<div style="{_F}font-size:0.54rem;color:#555960">&nbsp;&middot;&nbsp;{rnd["desc"]}</div>'
            f'</div>'
        )

        # Agent row: [inputs → agent] per agent
        row_parts = ['<div style="display:flex;gap:0.8rem;flex-wrap:wrap;margin-bottom:0.5rem">']
        for aid in agents:
            inputs = _INPUTS.get(aid, [])
            row_parts.append('<div style="display:flex;align-items:stretch;gap:0">')
            row_parts.append(_input_node(inputs))
            row_parts.append(arrow)
            row_parts.append(_agent_node(aid, color))
            row_parts.append('</div>')
        row_parts.append('</div>')
        html_parts.append("".join(row_parts))

        # Arrow between rounds (not after last)
        if i < len(_ROUNDS) - 1:
            html_parts.append(
                f'<div style="{_F}font-size:0.48rem;color:#555960;'
                f'margin:0.2rem 0 0.5rem;padding-left:0.2rem">'
                f'&#x25BC; outputs pass to Round {rnd["n"]+1}</div>'
            )

    html_parts.append(
        # Remediation loop note
        f'<div style="margin-top:0.7rem;border-top:1px solid #1e1e1e;padding-top:0.5rem;'
        f'{_F}font-size:0.54rem;color:#555960">'
        f'<span style="color:#c0392b;font-weight:700">CQO flags</span>'
        f' are parsed and routed to specialist agents for active correction '
        f'(Signal Auditor, Macro Strategist, Stress Engineer, Trade Structurer, '
        f'Geo Analyst, Commodities Specialist). Remediation runs automatically for '
        f'Medium / High / Critical severity. Cached 30 min per page.'
        f'</div>'
    )
    html_parts.append('</div>')

    st.markdown("".join(html_parts), unsafe_allow_html=True)


def _conf_bar_html(conf: float | None) -> str:
    if conf is None:
        return ""
    pct = int(conf * 100)
    color = "#27ae60" if pct >= 70 else ("#e67e22" if pct >= 45 else "#c0392b")
    return (
        f'<div style="margin-top:5px">'
        f'<div style="{_F}font-size:0.46rem;color:#555960;margin-bottom:2px">'
        f'Confidence - {pct}%</div>'
        f'<div style="background:#1e1e1e;height:2px;width:100%">'
        f'<div style="background:{color};width:{pct}%;height:2px"></div>'
        f'</div></div>'
    )


def _render_agent_full(aid: str) -> None:
    """Render a single agent's full output block."""
    init_agents()
    meta   = AGENTS.get(aid, {})
    state  = get_agent(aid)
    status = state.get("status", "idle")
    s_meta = STATUSES.get(status, {})
    s_color = s_meta.get("color", "#555960") if s_meta else "#555960"
    conf   = state.get("confidence")
    output = state.get("last_output") or ""
    last_run = state.get("last_run")
    ts_str = ""
    if last_run:
        diff = (datetime.datetime.now() - last_run).total_seconds()
        if diff < 60:   ts_str = "just now"
        elif diff < 3600: ts_str = f"{int(diff/60)}m ago"
        else:             ts_str = f"{int(diff/3600)}h ago"

    if not output:
        st.markdown(
            f'<div style="border:1px solid #1e1e1e;padding:0.5rem 0.8rem;'
            f'display:flex;align-items:center;gap:0.5rem">'
            f'<span style="font-size:0.75rem">{meta.get("icon","")}</span>'
            f'<span style="{_F}font-size:0.58rem;font-weight:700;color:{meta.get("color","#888")}">'
            f'{meta.get("short","")}</span>'
            f'<span style="{_M}font-size:0.50rem;color:#555960">No output yet</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    # Format narrative: bold "Agent:" prefix lines
    lines = output.split("\n")
    formatted = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("⚠", "✓", "FLAG", "PASS")):
            formatted.append(f'<span style="color:#e8e9ed;font-weight:600">{line}</span>')
        elif "SEVERITY:" in stripped:
            sev = stripped.replace("SEVERITY:", "").strip()
            sev_color = (
                "#c0392b" if "Critical" in sev else
                "#e67e22" if "High" in sev else
                "#e67e22" if "Medium" in sev else "#27ae60"
            )
            formatted.append(
                f'<span style="{_M}font-size:0.50rem;font-weight:700;letter-spacing:0.10em;'
                f'text-transform:uppercase;color:{sev_color}">SEVERITY: {sev}</span>'
            )
        elif "CONFIDENCE:" in stripped:
            formatted.append(
                f'<span style="{_M}font-size:0.46rem;color:#555960">{line}</span>'
            )
        elif ":" in line[:50] and not line.startswith(" "):
            head, _, rest = line.partition(":")
            formatted.append(f'<b style="color:#e8e9ed">{head}:</b>{rest}')
        else:
            formatted.append(line)
    formatted_html = "<br>".join(formatted)

    conf_html = _conf_bar_html(conf)

    st.markdown(
        f'<div style="{_F}background:#0d0d0d;border:1px solid #1e1e1e;'
        f'border-top:2px solid {meta.get("color","#CFB991")};'
        f'padding:0.8rem 1rem;margin-bottom:0.5rem">'

        # Header row
        f'<div style="display:flex;align-items:center;gap:0.55rem;margin-bottom:0.6rem;'
        f'border-bottom:1px solid #1e1e1e;padding-bottom:0.5rem">'
        f'<span style="font-size:0.85rem">{meta.get("icon","")}</span>'
        f'<div style="flex:1">'
        f'<div style="font-size:0.60rem;font-weight:700;color:{meta.get("color","#CFB991")};'
        f'letter-spacing:0.04em">{meta.get("name","")}</div>'
        f'<div style="font-size:0.52rem;color:#555960;margin-top:1px">{meta.get("desc","")}</div>'
        f'</div>'
        f'<div style="text-align:right">'
        f'<div style="{_M}font-size:0.48rem;font-weight:700;letter-spacing:0.10em;'
        f'text-transform:uppercase;color:{s_color}">{status.replace("_"," ")}</div>'
        f'<div style="{_M}font-size:0.46rem;color:#555960;margin-top:1px">{ts_str}</div>'
        f'</div>'
        f'</div>'

        # Narrative
        f'<div style="font-size:0.72rem;color:#c8c8c8;line-height:1.78;'
        f'white-space:pre-wrap">{formatted_html}</div>'

        + conf_html +
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_cqo_remediations() -> None:
    """Render all stored CQO remediation results across all pages."""
    remediations: dict = st.session_state.get("cqo_remediations", {})
    if not remediations:
        st.markdown(
            f'<div style="{_F}font-size:0.62rem;color:#555960;'
            f'padding:0.5rem 0;font-style:italic">'
            f'No remediation runs yet. CQO flags trigger active corrections automatically '
            f'when severity is Medium or higher.</div>',
            unsafe_allow_html=True,
        )
        return

    for page_name, page_rems in remediations.items():
        st.markdown(
            f'<div style="{_M}font-size:0.50rem;font-weight:700;letter-spacing:0.14em;'
            f'text-transform:uppercase;color:#c0392b;margin:0.8rem 0 0.4rem">'
            f'{page_name}</div>',
            unsafe_allow_html=True,
        )
        for aid, rem in page_rems.items():
            meta       = AGENTS.get(aid, {})
            color      = meta.get("color", "#CFB991")
            icon       = rem.get("agent_icon") or meta.get("icon", "")
            name       = rem.get("agent_name") or meta.get("short", aid)
            flags      = rem.get("flags", [])
            response   = rem.get("response", "")
            ts         = rem.get("ts")
            ts_str     = ts.strftime("%H:%M") if ts else ""
            flag_titles = " / ".join(f["title"] for f in flags[:3])

            st.markdown(
                f'<div style="{_F}background:#0f0f0f;border:1px solid #1e1e1e;'
                f'border-top:2px solid {color};padding:0.65rem 0.9rem;margin:0.3rem 0">'

                f'<div style="display:flex;align-items:center;gap:0.45rem;margin-bottom:0.4rem">'
                f'<span style="font-size:0.78rem">{icon}</span>'
                f'<span style="font-size:0.58rem;font-weight:700;color:{color}">{name}</span>'
                f'<span style="{_M}font-size:0.46rem;font-weight:700;letter-spacing:0.10em;'
                f'text-transform:uppercase;color:#c0392b;background:rgba(192,57,43,0.10);'
                f'padding:1px 4px">CORRECTING {len(flags)} FLAG{"S" if len(flags)!=1 else ""}</span>'
                f'<span style="{_M}font-size:0.46rem;color:#555960;margin-left:auto">{ts_str}</span>'
                f'</div>'

                f'<div style="{_F}font-size:0.56rem;color:#555960;'
                f'font-style:italic;margin-bottom:0.35rem">{flag_titles}</div>'

                f'<div style="font-size:0.70rem;color:#c8c8c8;line-height:1.72;'
                f'white-space:pre-wrap">{response}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_activity_feed() -> None:
    """Render the global agent activity feed."""
    feed = st.session_state.get("agent_activity", [])
    if not feed:
        st.markdown(
            f'<div style="{_F}font-size:0.62rem;color:#555960;padding:0.5rem 0;'
            f'font-style:italic">No activity yet.</div>',
            unsafe_allow_html=True,
        )
        return

    for entry in feed[:60]:
        aid      = entry.get("agent", "")
        meta     = AGENTS.get(aid, {})
        icon     = meta.get("icon", "")
        color    = meta.get("color", "#8890a1")
        action   = entry.get("action", "")
        detail   = entry.get("detail", "")
        sev      = entry.get("severity", "info")
        sev_color = "#c0392b" if sev == "warning" else ("#e67e22" if sev == "escalated" else "#555960")
        ts       = entry.get("ts")
        ts_str   = ts.strftime("%H:%M:%S") if ts else ""

        st.markdown(
            f'<div style="display:flex;align-items:baseline;gap:0.5rem;'
            f'border-bottom:1px solid #0f0f0f;padding:0.25rem 0">'
            f'<span style="{_M}font-size:0.46rem;color:#333">{ts_str}</span>'
            f'<span style="font-size:0.65rem">{icon}</span>'
            f'<span style="{_F}font-size:0.54rem;font-weight:600;color:{color}">{action}</span>'
            f'<span style="{_F}font-size:0.52rem;color:#555960;flex:1">{detail}</span>'
            f'<span style="{_M}font-size:0.42rem;font-weight:700;letter-spacing:0.10em;'
            f'text-transform:uppercase;color:{sev_color}">{sev}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _get_api_creds() -> tuple[str | None, str]:
    """Read API credentials from st.secrets. Returns (provider, api_key)."""
    try:
        sec = st.secrets.get("keys", st.secrets)
        ant = sec.get("anthropic_api_key", "")
        oai = sec.get("openai_api_key", "")
        if ant:
            return "anthropic", ant
        if oai:
            return "openai", oai
    except Exception:
        pass
    return None, ""


def _build_minimal_context() -> dict:
    """
    Build the best market context we can from session state.
    Falls back to empty dict - all agents degrade gracefully.
    """
    ctx: dict = {}

    # Pull cached overview context if the overview page has already run
    ov = st.session_state.get("overview_market_context")
    if ov:
        ctx.update(ov)
        return ctx

    # Lightweight fallback: pull regime + risk score from session state if saved
    for key in ("regime_name", "regime_level", "risk_score", "avg_corr",
                "corr_delta", "n_alerts"):
        if key in st.session_state:
            ctx[key] = st.session_state[key]

    # Pull FRED data if cached
    fred = st.session_state.get("fred_data")
    if fred:
        ctx["fed_rate"]          = fred.get("FEDFUNDS")
        ctx["cpi_yoy"]           = fred.get("CPIAUCSL")
        ctx["gdp_growth"]        = fred.get("GDP")
        ctx["yield_curve_spread"]= fred.get("T10Y2Y")

    return ctx


def _run_pipeline(force: bool = False) -> None:
    """Run the full 3-round agent pipeline and rerun the page on completion."""
    from src.analysis.agent_orchestrator import get_orchestrator

    provider, api_key = _get_api_creds()
    if not provider:
        st.error("No API key found. Add `anthropic_api_key` or `openai_api_key` to `.streamlit/secrets.toml`.")
        return

    orch = get_orchestrator(provider, api_key)
    if force:
        orch.invalidate()

    mkt_ctx = _build_minimal_context()

    rounds_done = 0
    progress = st.progress(0, text="Starting pipeline...")
    try:
        # Round 1
        progress.progress(10, text="Round 1 - Signal Auditor, Macro Strategist, Geo Intel...")
        from src.analysis.agent_orchestrator import PIPELINE
        r1_agents = [p for p in PIPELINE if p["round"] == 1]
        for spec in r1_agents:
            aid = spec["id"]
            if not is_enabled(aid):
                continue
            ctx = orch._build_context(aid, mkt_ctx)
            orch._run_agent(aid, ctx)
        rounds_done = 1
        progress.progress(38, text="Round 1 complete. Starting Round 2...")

        # Round 2
        progress.progress(42, text="Round 2 - Risk Officer, Commodities Specialist...")
        r2_agents = [p for p in PIPELINE if p["round"] == 2]
        for spec in r2_agents:
            aid = spec["id"]
            if not is_enabled(aid):
                continue
            ctx = orch._build_context(aid, mkt_ctx)
            orch._run_agent(aid, ctx)
        rounds_done = 2
        progress.progress(68, text="Round 2 complete. Starting Round 3...")

        # Round 3
        progress.progress(72, text="Round 3 - Stress Engineer, Trade Structurer...")
        r3_agents = [p for p in PIPELINE if p["round"] == 3]
        for spec in r3_agents:
            aid = spec["id"]
            if not is_enabled(aid):
                continue
            ctx = orch._build_context(aid, mkt_ctx)
            orch._run_agent(aid, ctx)
        rounds_done = 3
        progress.progress(100, text="Pipeline complete.")

        st.success(f"All {rounds_done} rounds complete - 7 agents updated. Switch to the Agent Outputs tab to review.")
    except Exception as e:
        progress.empty()
        st.error(f"Pipeline failed after Round {rounds_done}: {e}")


def page_about_ai_workforce() -> None:
    _about_page_styles()
    init_agents()

    # Page header
    _page_header("AI Agent Team",
                 "8 specialist agents · Monitoring · Analysis · Trade structuring · Quality assurance")

    # ── Run controls ──────────────────────────────────────────────────────────
    from src.analysis.agent_orchestrator import _is_fresh, PIPELINE as _PIPELINE

    n_fresh  = sum(1 for p in _PIPELINE if _is_fresh(p["id"]))
    n_total  = len(_PIPELINE)  # 7 pipeline agents (not CQO)
    n_stale  = n_total - n_fresh
    last_run = st.session_state.get("orchestrator", {}).get("pipeline_run")
    last_run_str = (
        last_run.strftime("%H:%M") if last_run else "never"
    )

    # Status bar
    status_color = "#27ae60" if n_stale == 0 else ("#e67e22" if n_stale <= 3 else "#c0392b")
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:1rem;'
        f'background:#0d0d0d;border:1px solid #1e1e1e;'
        f'padding:0.55rem 0.9rem;margin-bottom:0.8rem">'
        f'<div style="{_M}font-size:0.50rem;font-weight:700;letter-spacing:0.12em;'
        f'text-transform:uppercase;color:{status_color}">'
        f'{"ALL AGENTS FRESH" if n_stale == 0 else f"{n_stale} AGENT{"S" if n_stale != 1 else ""} NEED REFRESH"}'
        f'</div>'
        f'<div style="{_M}font-size:0.48rem;color:#555960">'
        f'{n_fresh}/{n_total} fresh &nbsp;·&nbsp; last run {last_run_str}</div>'
        f'<div style="{_F}font-size:0.52rem;color:#555960;flex:1;text-align:right">'
        f'Round 4 (CQO) runs per analysis page visit.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_run, col_force, col_spacer = st.columns([1.3, 1.5, 5])
    with col_run:
        run_clicked = st.button(
            "Run All Agents Now",
            key="workforce_run_btn",
            use_container_width=True,
            type="primary",
            disabled=(n_stale == 0),
            help="Run the full 3-round pipeline (Rounds 1-3). CQO runs per page.",
        )
    with col_force:
        force_clicked = st.button(
            "Force Refresh All",
            key="workforce_force_btn",
            use_container_width=True,
            type="secondary",
            help="Invalidate all cached outputs and re-run the full pipeline.",
        )

    if run_clicked:
        _run_pipeline(force=False)
    elif force_clicked:
        _run_pipeline(force=True)

    st.markdown('<div style="margin-bottom:0.5rem"></div>', unsafe_allow_html=True)

    tab_diagram, tab_outputs, tab_cqo, tab_feed = st.tabs([
        "Pipeline Diagram", "Agent Outputs", "CQO Flags & Corrections", "Activity Feed"
    ])

    with tab_diagram:
        st.markdown(
            f'<div style="{_F}font-size:0.62rem;color:#8890a1;margin-bottom:1rem">'
            f'Each agent receives structured inputs from upstream peers - not just a summary '
            f'string but the full narratives. The Risk Officer synthesises all Round 1 outputs. '
            f'Round 4 (CQO) runs separately per page and feeds back into specialist agents.</div>',
            unsafe_allow_html=True,
        )
        _pipeline_diagram()

        # Legend
        st.markdown(
            f'<div style="display:flex;gap:1.2rem;flex-wrap:wrap;margin-top:1rem;'
            f'border-top:1px solid #1e1e1e;padding-top:0.6rem">'
            + "".join(
                f'<div style="display:flex;align-items:center;gap:5px">'
                f'<div style="width:10px;height:10px;background:{r["color"]}"></div>'
                f'<span style="{_F}font-size:0.52rem;color:#8890a1">Round {r["n"]} - {r["label"].split(" - ")[1]}</span>'
                f'</div>'
                for r in _ROUNDS
            )
            + '</div>',
            unsafe_allow_html=True,
        )

    with tab_outputs:
        for rnd in _ROUNDS:
            color = rnd["color"]
            st.markdown(
                f'<div style="border-top:2px solid {color};margin:1.2rem 0 0.6rem;'
                f'padding-top:0.4rem;display:flex;align-items:baseline;gap:0.6rem">'
                f'<span style="{_M}font-size:0.50rem;font-weight:700;letter-spacing:0.14em;'
                f'text-transform:uppercase;color:{color}">Round {rnd["n"]}</span>'
                f'<span style="{_F}font-size:0.60rem;font-weight:700;color:#c8c8c8">'
                f'{rnd["label"].split(" - ")[1]}</span>'
                f'<span style="{_F}font-size:0.54rem;color:#555960">&middot; {rnd["desc"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            for aid in rnd["agents"]:
                _render_agent_full(aid)

    with tab_cqo:
        st.markdown(
            f'<div style="{_F}font-size:0.62rem;color:#8890a1;margin-bottom:0.8rem">'
            f'The CQO raises numbered flags on each analysis page. '
            f'Flags at Medium severity or above are automatically routed to specialist agents '
            f'for active correction. Results below are cached 30 min per page.</div>',
            unsafe_allow_html=True,
        )

        # CQO narrative itself
        cqo_state = get_agent("quality_officer")
        cqo_output = cqo_state.get("last_output", "")
        if cqo_output:
            st.markdown(
                f'<div style="{_M}font-size:0.50rem;font-weight:700;letter-spacing:0.14em;'
                f'text-transform:uppercase;color:#c0392b;margin-bottom:0.4rem">'
                f'Latest CQO Audit</div>',
                unsafe_allow_html=True,
            )
            _render_agent_full("quality_officer")

        st.markdown(
            f'<div style="{_M}font-size:0.50rem;font-weight:700;letter-spacing:0.14em;'
            f'text-transform:uppercase;color:#c0392b;margin:1rem 0 0.4rem">'
            f'Active Corrections by Page</div>',
            unsafe_allow_html=True,
        )
        _render_cqo_remediations()

    with tab_feed:
        st.markdown(
            f'<div style="{_F}font-size:0.62rem;color:#8890a1;margin-bottom:0.6rem">'
            f'Timestamped log of all agent activity this session. '
            f'Last 60 entries shown.</div>',
            unsafe_allow_html=True,
        )
        _render_activity_feed()

    _page_footer()
