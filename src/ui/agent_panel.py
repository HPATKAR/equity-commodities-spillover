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
    ag_meta  = AGENTS.get(agent_id, {})
    a_state  = st.session_state["agents"].get(agent_id, {})
    status   = a_state.get("status", "monitoring")
    conf     = a_state.get("confidence")
    narrative = result.get("narrative") or a_state.get("last_output") or ""

    if not narrative:
        return

    status_meta  = STATUSES.get(status, STATUSES["monitoring"])
    routed_html  = ""
    if result.get("routed_to"):
        rt = AGENTS.get(result["routed_to"], {})
        routed_html = (
            f'<span style="{_F}font-size:0.52rem;color:#555960;margin-left:0.6rem">'
            f'→ routed to {rt.get("short","")}</span>'
        )

    conf_html = _conf_bar(conf) if conf is not None else ""

    st.markdown(
        f'<div style="{_F}background:#111318;border:1px solid #1e2130;'
        f'border-left:3px solid {ag_meta.get("color","#CFB991")};'
        f'border-radius:0 4px 4px 0;padding:0.75rem 1rem;margin:0.8rem 0">'

        # Header row
        f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem">'
        f'<span style="font-size:0.82rem">{ag_meta.get("icon","●")}</span>'
        f'<span style="font-size:0.58rem;font-weight:700;color:{ag_meta.get("color","#CFB991")};'
        f'letter-spacing:0.04em">{ag_meta.get("name","Agent")}</span>'
        f'<span style="font-size:0.50rem;font-weight:700;letter-spacing:0.10em;'
        f'text-transform:uppercase;color:{status_meta["color"]};'
        f'background:rgba(0,0,0,0.4);padding:1px 5px;border-radius:2px">'
        f'{status_meta["label"]}</span>'
        f'{routed_html}'
        f'</div>'

        # Narrative
        f'<div style="font-size:0.74rem;color:#c8cdd8;line-height:1.7;margin-bottom:0.4rem">'
        f'{narrative}</div>'

        # Confidence
        + conf_html +

        f'</div>',
        unsafe_allow_html=True,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts(dt: datetime.datetime | None) -> str:
    if dt is None:
        return "—"
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
        f'<div style="background:#2a2d3a;border-radius:2px;height:3px;width:100%">'
        f'<div style="background:{bar_color};width:{pct}%;height:3px;border-radius:2px"></div>'
        f'</div></div>'
    )


# ── Agent Control Panel strip ─────────────────────────────────────────────────

def render_agent_panel(show_panel: bool = True) -> None:
    """
    Render the agent control panel inside a collapsible expander.
    All 7 tiles live in one pure-HTML flex row so they never wrap.
    Toggle buttons sit in a single st.columns(7) row below the tiles.
    """
    init_agents()

    n_pending    = pending_count()
    agents_state = st.session_state["agents"]
    agent_list   = list(AGENTS.items())   # ordered, length 7

    # ── Build expander label with live status summary ──────────────────────
    n_active = sum(1 for a in agents_state.values() if a.get("enabled", True))
    n_inv    = sum(1 for a in agents_state.values() if a.get("status") == "investigating")
    pend_tag = f"  ·  ⚠ {n_pending} pending" if n_pending else ""
    inv_tag  = f"  ·  {n_inv} investigating" if n_inv else ""
    exp_label = f"AI Workforce  ·  {n_active}/7 active{inv_tag}{pend_tag}"

    with st.expander(exp_label, expanded=False):

        # ── Pure-HTML flex row — all 7 tiles always on one line ───────────
        tiles_html = (
            '<div style="display:flex;gap:8px;margin-bottom:8px;'
            'overflow-x:auto;padding-bottom:2px">'
        )
        for aid, meta in agent_list:
            a           = agents_state.get(aid, {})
            status_key  = a.get("status", "idle")
            status_meta = STATUSES.get(status_key, STATUSES["idle"])
            enabled     = a.get("enabled", True)
            conf        = a.get("confidence")
            last_run    = a.get("last_run")

            opacity     = "1" if enabled else "0.40"
            left_border = f"border-left:2px solid {meta['color']}" if enabled \
                          else "border-left:2px solid #3a3d4a"
            conf_html   = ""
            if conf is not None:
                pct       = int(conf * 100)
                bar_col   = "#27ae60" if pct >= 70 else ("#e67e22" if pct >= 45 else "#c0392b")
                conf_html = (
                    f'<div style="margin-top:5px">'
                    f'<div style="{_F}font-size:0.43rem;color:#555960;margin-bottom:2px">'
                    f'Conf {pct}%</div>'
                    f'<div style="background:#2a2d3a;border-radius:2px;height:2px">'
                    f'<div style="background:{bar_col};width:{pct}%;height:2px;'
                    f'border-radius:2px"></div></div></div>'
                )

            tiles_html += (
                f'<div style="{_F}flex:1;min-width:0;background:#1a1d27;'
                f'border:1px solid #2a2d3a;{left_border};'
                f'border-radius:4px;padding:0.5rem 0.65rem;opacity:{opacity}">'

                # Icon + short name
                f'<div style="display:flex;align-items:center;gap:5px;margin-bottom:3px">'
                f'<span style="font-size:0.70rem;flex-shrink:0">{meta["icon"]}</span>'
                f'<span style="font-size:0.58rem;font-weight:700;color:#e8e9ed;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0">'
                f'{meta["short"]}</span>'
                f'</div>'

                # Status badge
                f'<span style="font-size:0.43rem;font-weight:700;letter-spacing:0.08em;'
                f'text-transform:uppercase;color:{status_meta["color"]};'
                f'background:rgba(0,0,0,0.35);padding:1px 4px;border-radius:2px">'
                f'{status_meta["label"]}</span>'

                # Last run
                f'<div style="font-size:0.43rem;color:#555960;margin-top:3px">'
                f'{_ts(last_run) if last_run else "not run"}</div>'

                + conf_html +

                f'</div>'
            )
        tiles_html += '</div>'
        st.markdown(tiles_html, unsafe_allow_html=True)

        # ── Toggle buttons — one st.columns(7) row ────────────────────────
        btn_cols = st.columns(7, gap="small")
        for i, (aid, meta) in enumerate(agent_list):
            enabled = agents_state.get(aid, {}).get("enabled", True)
            label   = "Disable" if enabled else "Enable"
            with btn_cols[i]:
                if st.button(label, key=f"agent_toggle_{aid}",
                             use_container_width=True):
                    toggle_agent(aid)
                    st.rerun()

        # ── Pending notice (inside expander) ─────────────────────────────
        if n_pending:
            st.markdown(
                f'<p style="{_F}font-size:0.55rem;color:#e67e22;margin:6px 0 0">'
                f'⚠ {n_pending} trade idea(s) awaiting your review — '
                f'see the Trade Ideas page.</p>',
                unsafe_allow_html=True,
            )


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
                f'padding:0.35rem 0;border-bottom:1px solid #1e2130">'

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
                f'<span style="{_F}font-size:0.62rem;color:#c8cdd8;line-height:1.5">'
                f'<strong style="color:#e8e9ed">{entry["action"]}</strong> '
                f'— {entry["detail"]}'
                f'{routed_html}</span>'

                f'</div>'
            )

        st.markdown(
            f'<div style="background:#111318;border:1px solid #1e2130;'
            f'border-radius:4px;padding:0.4rem 0.7rem;'
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
                f'<div style="{_F}background:#1a1d27;border:1px solid #2a2d3a;'
                f'border-left:3px solid {border_col};border-radius:0 4px 4px 0;'
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
                f'<div style="font-size:0.70rem;color:#c8cdd8;margin-bottom:0.35rem">'
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
                    f'<div style="{_F}font-size:0.70rem;color:#c8cdd8;'
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
                        f'border-bottom:1px solid #1e2130">'
                        f'<span style="font-size:0.52rem;font-weight:700;'
                        f'text-transform:uppercase;color:{status_color};'
                        f'min-width:60px">{item["status"].upper()}</span>'
                        f'<span style="font-size:0.62rem;color:#8890a1">'
                        f'{item["agent_name"]}</span>'
                        f'<span style="font-size:0.65rem;color:#c8cdd8">'
                        f'{item["title"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
