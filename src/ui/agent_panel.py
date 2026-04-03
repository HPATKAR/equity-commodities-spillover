"""
Agent Control Panel
Renders a compact persistent strip at the top of every page showing all 7 AI
agents with status, toggle, and confidence. Also renders:
  - Activity Feed  (full on Overview, collapsible elsewhere)
  - Pending Review Panel (Approve / Reject / Escalate queue)
"""

from __future__ import annotations

import datetime
import streamlit as st

from src.analysis.agent_state import (
    AGENTS, STATUSES, init_agents,
    toggle_agent, review_item, get_agent,
    pending_count, is_enabled,
)

_F = "font-family:'DM Sans',sans-serif;"
_M = "font-family:'JetBrains Mono',monospace;"


# ── Shared agent output block ─────────────────────────────────────────────────

def render_agent_output_block(agent_id: str, result: dict) -> None:
    """
    Render the standard AI agent output block on a page.
    Shows narrative, confidence, status, and ownership tag.
    Call this after the agent's run() completes.
    """
    if not result or result.get("status") in ("monitoring", None) and not result.get("narrative"):
        return

    init_agents()
    ag_meta   = AGENTS.get(agent_id, {})
    a_state   = st.session_state["agents"].get(agent_id, {})
    status    = a_state.get("status", "monitoring")
    conf      = a_state.get("confidence")
    narrative = result.get("narrative") or a_state.get("last_output") or ""

    if not narrative:
        return

    status_meta = STATUSES.get(status, STATUSES["monitoring"])
    s_color     = status_meta["color"]
    ag_color    = ag_meta.get("color", "#CFB991")

    routed_html = ""
    if result.get("routed_to"):
        rt = AGENTS.get(result["routed_to"], {})
        routed_html = (
            f'<span style="{_F}font-size:0.50rem;color:#555960;margin-left:0.5rem">'
            f'&#x2192; {rt.get("short","")}</span>'
        )

    # Format narrative: highlight flag lines, severity, confidence
    lines = narrative.split("\n")
    fmt_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("⚠", "\u26a0")):
            fmt_lines.append(f'<span style="color:#e67e22;font-weight:600">{line}</span>')
        elif stripped.startswith(("✓", "\u2713")):
            fmt_lines.append(f'<span style="color:#27ae60;font-weight:600">{line}</span>')
        elif "SEVERITY:" in stripped.upper():
            sev_text = stripped.split(":", 1)[-1].strip()
            sev_color = (
                "#c0392b" if "critical" in sev_text.lower() else
                "#e67e22" if "high" in sev_text.lower() or "medium" in sev_text.lower()
                else "#27ae60"
            )
            fmt_lines.append(
                f'<span style="{_M}font-size:0.48rem;font-weight:700;letter-spacing:0.10em;'
                f'color:{sev_color}">SEVERITY: {sev_text}</span>'
            )
        elif "CONFIDENCE:" in stripped.upper():
            fmt_lines.append(
                f'<span style="{_M}font-size:0.46rem;color:#555960">{line}</span>'
            )
        else:
            fmt_lines.append(line)
    formatted = "<br>".join(fmt_lines)

    conf_html = _conf_bar(conf) if conf is not None else ""

    st.markdown(
        f'<div style="background:#0d0d0d;border:1px solid #1e1e1e;'
        f'border-top:2px solid {ag_color};padding:0.75rem 1rem;margin:0.6rem 0">'

        # Header
        f'<div style="display:flex;align-items:center;gap:0.5rem;'
        f'border-bottom:1px solid #1a1a1a;padding-bottom:0.45rem;margin-bottom:0.5rem">'
        f'<span style="font-size:0.82rem">{ag_meta.get("icon","")}</span>'
        f'<div style="flex:1">'
        f'<span style="{_F}font-size:0.58rem;font-weight:700;color:{ag_color};'
        f'letter-spacing:0.04em">{ag_meta.get("name","Agent")}</span>'
        f'<span style="{_F}font-size:0.52rem;color:#555960;margin-left:0.5rem">'
        f'{ag_meta.get("desc","")}</span>'
        f'</div>'
        f'<span style="{_M}font-size:0.48rem;font-weight:700;letter-spacing:0.10em;'
        f'text-transform:uppercase;color:{s_color}">{status_meta["label"]}</span>'
        f'{routed_html}'
        f'</div>'

        # Narrative
        f'<div style="{_F}font-size:0.72rem;color:#c8c8c8;line-height:1.78">'
        f'{formatted}</div>'

        + conf_html +
        f'</div>',
        unsafe_allow_html=True,
    )


def render_agent_inline_signal(agent_id: str, result: dict, page_key: str = "") -> None:
    """
    Compact 2-line signal shown on analysis pages instead of the full block.
    Links user to the AI Workforce page for the full output.
    """
    if not result or not result.get("narrative"):
        return

    init_agents()
    ag_meta   = AGENTS.get(agent_id, {})
    a_state   = st.session_state["agents"].get(agent_id, {})
    conf      = a_state.get("confidence")
    narrative = result.get("narrative") or ""
    ag_color  = ag_meta.get("color", "#CFB991")

    # Extract first meaningful line as the teaser
    first_line = next(
        (l.strip() for l in narrative.split("\n") if l.strip() and "CONFIDENCE" not in l.upper()),
        narrative[:120],
    )
    if len(first_line) > 140:
        first_line = first_line[:137] + "..."

    conf_str = f"{int(conf*100)}%" if conf is not None else ""

    # CQO-specific: show severity + flag count
    extra_html = ""
    if agent_id == "quality_officer":
        import re
        sev_m = re.search(r"SEVERITY:\s*(Critical|High|Medium|Low)", narrative, re.IGNORECASE)
        n_flags = len(re.findall(r"FLAG\s*\d+:", narrative, re.IGNORECASE))
        if sev_m:
            sev = sev_m.group(1)
            sev_color = (
                "#c0392b" if sev.lower() == "critical" else
                "#e67e22" if sev.lower() in ("high", "medium") else "#27ae60"
            )
            extra_html = (
                f'<span style="{_M}font-size:0.46rem;font-weight:700;letter-spacing:0.10em;'
                f'text-transform:uppercase;color:{sev_color};background:rgba(0,0,0,0.4);'
                f'padding:1px 5px;margin-left:0.4rem">'
                f'{sev.upper()} &middot; {n_flags} flag{"s" if n_flags != 1 else ""}</span>'
            )

    st.markdown(
        f'<div style="background:#0a0a0a;border:1px solid #181818;'
        f'padding:0.4rem 0.75rem;margin:0.4rem 0;'
        f'display:flex;align-items:baseline;gap:0.5rem;flex-wrap:wrap">'
        f'<span style="font-size:0.70rem">{ag_meta.get("icon","")}</span>'
        f'<span style="{_F}font-size:0.54rem;font-weight:700;color:{ag_color}">'
        f'{ag_meta.get("short","")}</span>'
        f'{extra_html}'
        f'<span style="{_F}font-size:0.58rem;color:#8890a1;flex:1;'
        f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
        f'{first_line}</span>'
        f'<span style="{_M}font-size:0.46rem;color:#555960">{conf_str}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts(dt: datetime.datetime | None) -> str:
    if dt is None:
        return "-"
    now = datetime.datetime.now()
    diff = (now - dt).total_seconds()
    if diff < 60:
        return "just now"
    if diff < 3600:
        return f"{int(diff/60)}m ago"
    if diff < 86400:
        return f"{int(diff/3600)}h ago"
    return dt.strftime("%b %d")


def _conf_bar(conf: float | None) -> str:
    """Return an HTML mini confidence bar."""
    if conf is None:
        return ""
    pct = int(conf * 100)
    bar_color = "#27ae60" if pct >= 70 else ("#e67e22" if pct >= 45 else "#c0392b")
    return (
        f'<div style="margin-top:4px">'
        f'<div style="{_F}font-size:0.48rem;color:#8890a1;margin-bottom:2px">'
        f'Confidence {pct}%</div>'
        f'<div style="background:#2a2a2a;border-radius:2px;height:3px;width:100%">'
        f'<div style="background:{bar_color};width:{pct}%;height:3px;border-radius:2px"></div>'
        f'</div></div>'
    )


# ── Agent Control Panel strip ─────────────────────────────────────────────────

def _render_workforce_content(agents_state: dict, agent_list: list,
                               n_pending: int, key_prefix: str = "") -> None:
    """Render agent tiles + toggles inline (no expander wrapper)."""
    tiles_html = (
        '<div style="display:flex;gap:5px;margin-bottom:6px;'
        'flex-wrap:wrap">'
    )
    for aid, meta in agent_list:
        a           = agents_state.get(aid, {})
        status_key  = a.get("status", "idle")
        status_meta = STATUSES.get(status_key, STATUSES["idle"])
        enabled     = a.get("enabled", True)
        conf        = a.get("confidence")
        last_run    = a.get("last_run")

        opacity     = "1" if enabled else "0.38"
        left_border = f"border-left:2px solid {meta['color']}" if enabled \
                      else "border-left:2px solid #2a2a2a"
        conf_html   = ""
        if conf is not None:
            pct     = int(conf * 100)
            bar_col = "#27ae60" if pct >= 70 else ("#e67e22" if pct >= 45 else "#c0392b")
            conf_html = (
                f'<div style="margin-top:3px">'
                f'<div style="background:#222;border-radius:1px;height:2px">'
                f'<div style="background:{bar_col};width:{pct}%;height:2px;'
                f'border-radius:1px"></div></div></div>'
            )

        tiles_html += (
            f'<div style="{_F}width:calc(25% - 4px);min-width:80px;background:#161616;'
            f'border:1px solid #222;{left_border};'
            f'border-radius:2px;padding:0.35rem 0.5rem;opacity:{opacity}">'
            f'<div style="display:flex;align-items:center;gap:4px;margin-bottom:2px">'
            f'<span style="font-size:0.62rem;flex-shrink:0">{meta["icon"]}</span>'
            f'<span style="font-size:0.52rem;font-weight:700;color:#c8c8c8;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0">'
            f'{meta["short"]}</span>'
            f'</div>'
            f'<span style="font-size:0.40rem;font-weight:700;letter-spacing:0.06em;'
            f'text-transform:uppercase;color:{status_meta["color"]}">'
            f'{status_meta["label"]}</span>'
            f'<div style="font-size:0.40rem;color:#444;margin-top:2px">'
            f'{_ts(last_run) if last_run else "-"}</div>'
            + conf_html +
            f'</div>'
        )
    tiles_html += '</div>'
    st.markdown(tiles_html, unsafe_allow_html=True)

    # Toggle buttons - 4 per row to fit popover width
    cols_a = st.columns(4, gap="small")
    cols_b = st.columns(4, gap="small")
    for i, (aid, meta) in enumerate(agent_list):
        enabled = agents_state.get(aid, {}).get("enabled", True)
        label   = meta["short"]
        col = (cols_a if i < 4 else cols_b)[i % 4]
        with col:
            btn_type = "secondary" if enabled else "primary"
            if st.button(
                ("● " if enabled else "○ ") + label,
                key=f"{key_prefix}agent_toggle_{aid}",
                use_container_width=True,
                help=f"{'Disable' if enabled else 'Enable'} {meta['name']}",
            ):
                toggle_agent(aid)
                st.rerun()

    if n_pending:
        st.markdown(
            f'<p style="{_F}font-size:0.52rem;color:#e67e22;margin:6px 0 0">'
            f'⚠ {n_pending} pending review - Trade Ideas page</p>',
            unsafe_allow_html=True,
        )


def render_agent_panel(show_panel: bool = True) -> None:
    """Kept for backward compatibility - no longer used in main layout."""
    pass


# ── Activity Feed ─────────────────────────────────────────────────────────────

def render_activity_feed(max_entries: int = 20, collapsible: bool = False) -> None:
    """
    Render the timestamped agent activity feed.
    On Overview: collapsible=False (shown inline).
    On other pages: collapsible=True (inside expander).
    """
    init_agents()
    feed = st.session_state.get("agent_activity", [])

    _sev_color = {"critical": "#c0392b", "warning": "#e67e22", "info": "#2980b9"}

    def _render_feed_inner():
        if not feed:
            st.markdown(
                f'<p style="{_F}font-size:0.65rem;color:#555960;'
                f'font-style:italic;padding:0.3rem 0">No agent activity yet.</p>',
                unsafe_allow_html=True,
            )
            return

        shown = feed[:max_entries]
        rows_html = ""
        for entry in shown:
            ts_str   = entry["ts"].strftime("%H:%M:%S") if isinstance(entry["ts"], datetime.datetime) else str(entry["ts"])
            sev_col  = _sev_color.get(entry.get("severity", "info"), "#2980b9")
            ag_meta  = AGENTS.get(entry["agent_id"], {})
            ag_icon  = ag_meta.get("icon", "●")
            ag_color = ag_meta.get("color", "#8890a1")
            routed   = entry.get("routed_to")
            routed_html = ""
            if routed:
                rt_meta = AGENTS.get(routed, {})
                routed_html = (
                    f'<span style="color:#555960"> → {rt_meta.get("short", routed)}</span>'
                )

            rows_html += (
                f'<div style="display:flex;gap:0.6rem;align-items:flex-start;'
                f'padding:0.35rem 0;border-bottom:1px solid #1e1e1e">'

                # Timestamp
                f'<span style="{_M}font-size:0.52rem;color:#555960;flex-shrink:0;'
                f'padding-top:1px">{ts_str}</span>'

                # Agent icon + name
                f'<span style="font-size:0.60rem;flex-shrink:0;color:{ag_color}">'
                f'{ag_icon}</span>'
                f'<span style="{_F}font-size:0.58rem;font-weight:600;'
                f'color:{ag_color};flex-shrink:0">'
                f'{entry["agent_name"]}</span>'

                # Severity dot
                f'<span style="width:5px;height:5px;border-radius:50%;'
                f'background:{sev_col};flex-shrink:0;margin-top:5px"></span>'

                # Action + detail + routing
                f'<span style="{_F}font-size:0.62rem;color:#c8c8c8;line-height:1.5">'
                f'<strong style="color:#e8e9ed">{entry["action"]}</strong> '
                f'- {entry["detail"]}'
                f'{routed_html}</span>'

                f'</div>'
            )

        st.markdown(
            f'<div style="background:#131313;border:1px solid #1e1e1e;'
            f'border-radius:0;padding:0.4rem 0.7rem;'
            f'max-height:280px;overflow-y:auto">'
            f'{rows_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

        if len(feed) > max_entries:
            st.markdown(
                f'<p style="{_F}font-size:0.52rem;color:#555960;margin-top:0.3rem">'
                f'+ {len(feed) - max_entries} earlier entries</p>',
                unsafe_allow_html=True,
            )

    if collapsible:
        with st.expander("Agent Activity Feed", expanded=False):
            _render_feed_inner()
    else:
        st.markdown(
            f'<div style="{_F}font-size:0.52rem;font-weight:700;letter-spacing:0.14em;'
            f'text-transform:uppercase;color:#8890a1;margin:0.8rem 0 0.4rem">Activity Feed</div>',
            unsafe_allow_html=True,
        )
        _render_feed_inner()


# ── Pending Review Panel ──────────────────────────────────────────────────────

def render_pending_review() -> None:
    """
    Render the Pending Review Panel with Approve / Reject / Escalate actions.
    Shows all pending items from all agents.
    """
    init_agents()
    queue = st.session_state.get("pending_review", [])
    pending = [i for i in queue if i["status"] == "pending"]

    _sev_border = {"critical": "#c0392b", "warning": "#e67e22", "info": "#2980b9"}

    with st.expander(
        f"Pending Review Queue ({len(pending)} item{'s' if len(pending) != 1 else ''})",
        expanded=(len(pending) > 0),
    ):
        if not pending:
            st.markdown(
                f'<p style="{_F}font-size:0.68rem;color:#555960;'
                f'font-style:italic">Queue is clear.</p>',
                unsafe_allow_html=True,
            )
            return

        for item in pending:
            ag_meta = AGENTS.get(item["agent_id"], {})
            border_col = _sev_border.get(item["severity"], "#2980b9")
            conf_pct = int(item["confidence"] * 100)

            st.markdown(
                f'<div style="{_F}background:#1c1c1c;border:1px solid #2a2a2a;'
                f'border-radius:0;'
                f'padding:0.7rem 1rem;margin-bottom:0.6rem">'

                # Header row: agent + title
                f'<div style="display:flex;align-items:center;gap:0.5rem;'
                f'margin-bottom:0.4rem">'
                f'<span style="font-size:0.78rem">{ag_meta.get("icon","●")}</span>'
                f'<span style="font-size:0.60rem;font-weight:700;'
                f'color:{ag_meta.get("color","#CFB991")}">'
                f'{item["agent_name"]}</span>'
                f'<span style="font-size:0.78rem;font-weight:700;color:#e8e9ed;'
                f'margin-left:0.4rem">{item["title"]}</span>'
                f'</div>'

                # Summary
                f'<div style="font-size:0.70rem;color:#c8c8c8;margin-bottom:0.35rem">'
                f'{item["summary"]}</div>'

                # Confidence + timestamp
                f'<div style="display:flex;gap:1.2rem;font-size:0.54rem;color:#555960">'
                f'<span>Confidence: <strong style="color:#CFB991">{conf_pct}%</strong></span>'
                f'<span>Submitted: {_ts(item["created_at"])}</span>'
                f'</div>'

                f'</div>',
                unsafe_allow_html=True,
            )

            # Rationale preview
            with st.expander("Rationale", expanded=False):
                st.markdown(
                    f'<div style="{_F}font-size:0.70rem;color:#c8c8c8;'
                    f'line-height:1.7;padding:0.4rem">{item["rationale"]}</div>',
                    unsafe_allow_html=True,
                )

            # Action buttons
            a1, a2, a3, _ = st.columns([1, 1, 1, 4])
            if a1.button("Approve", key=f"apr_{item['id']}", type="primary"):
                review_item(item["id"], "approved")
                st.rerun()
            if a2.button("Reject", key=f"rej_{item['id']}"):
                review_item(item["id"], "rejected")
                st.rerun()
            if a3.button("Escalate", key=f"esc_{item['id']}"):
                review_item(item["id"], "escalated",
                            escalation_peer="signal_auditor")
                st.rerun()

            st.markdown("---")

        # Recently resolved
        resolved = [i for i in queue if i["status"] != "pending"][-5:]
        if resolved:
            with st.expander("Recent Decisions", expanded=False):
                for item in reversed(resolved):
                    status_color = (
                        "#27ae60" if item["status"] == "approved"
                        else "#c0392b" if item["status"] == "rejected"
                        else "#e67e22"
                    )
                    st.markdown(
                        f'<div style="{_F}display:flex;align-items:center;'
                        f'gap:0.6rem;padding:0.3rem 0;'
                        f'border-bottom:1px solid #1e1e1e">'
                        f'<span style="font-size:0.52rem;font-weight:700;'
                        f'text-transform:uppercase;color:{status_color};'
                        f'min-width:60px">{item["status"].upper()}</span>'
                        f'<span style="font-size:0.62rem;color:#8890a1">'
                        f'{item["agent_name"]}</span>'
                        f'<span style="font-size:0.65rem;color:#c8c8c8">'
                        f'{item["title"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ── CQO Remediation Panel ─────────────────────────────────────────────────────

def render_remediation_panel(page: str) -> None:
    """
    Render the active CQO remediation panel for the current page.
    Shows which agents are addressing which flags and what they concluded.
    Call this directly after render_agent_output_block for the quality_officer.
    """
    remediations: dict = st.session_state.get("cqo_remediations", {}).get(page, {})
    if not remediations:
        return

    st.markdown(
        f'<div style="border-top:2px solid #c0392b;margin:0.5rem 0 0.2rem">'
        f'<span style="{_M}font-size:0.50rem;font-weight:700;letter-spacing:0.14em;'
        f'text-transform:uppercase;color:#c0392b">CQO Remediation - Active Corrections</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    for agent_id, rem in remediations.items():
        ag_meta    = AGENTS.get(agent_id, {})
        icon       = rem.get("agent_icon") or ag_meta.get("icon", "")
        name       = rem.get("agent_name") or ag_meta.get("name", agent_id)
        color      = ag_meta.get("color", "#CFB991")
        flags      = rem.get("flags", [])
        response   = rem.get("response", "")
        ts         = rem.get("ts")
        ts_str     = ts.strftime("%H:%M") if ts else ""
        n_flags    = len(flags)
        flag_titles = " / ".join(f["title"] for f in flags[:3])

        st.markdown(
            f'<div style="{_F}background:#0f0f0f;border:1px solid #1e1e1e;'
            f'border-radius:0;padding:0.65rem 0.9rem;margin:0.4rem 0">'

            # Header
            f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.45rem">'
            f'<span style="font-size:0.80rem">{icon}</span>'
            f'<span style="font-size:0.58rem;font-weight:700;color:{color};letter-spacing:0.04em">'
            f'{name}</span>'
            f'<span style="{_M}font-size:0.48rem;font-weight:700;letter-spacing:0.10em;'
            f'text-transform:uppercase;color:#c0392b;background:rgba(192,57,43,0.12);'
            f'padding:1px 5px">'
            f'CORRECTING {n_flags} FLAG{"S" if n_flags != 1 else ""}</span>'
            f'<span style="{_M}font-size:0.48rem;color:#555960;margin-left:auto">{ts_str}</span>'
            f'</div>'

            # Flags being addressed
            f'<div style="{_F}font-size:0.58rem;color:#555960;margin-bottom:0.4rem;'
            f'font-style:italic">{flag_titles}</div>'

            # Response
            f'<div style="font-size:0.72rem;color:#c8c8c8;line-height:1.70;'
            f'white-space:pre-wrap">{response}</div>'

            f'</div>',
            unsafe_allow_html=True,
        )
