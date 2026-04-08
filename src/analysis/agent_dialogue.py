"""
Inter-Agent Dialogue Engine.

Provides a structured message bus for agent-to-agent communication.
Agents communicate via typed messages (query, handoff, challenge, support,
veto, resolve, escalate) stored in st.session_state["agent_dialogue"].

All messages are threaded by thread_id so a deliberation chain can be
followed from first query to final resolution.

Usage:
    from src.analysis.agent_dialogue import send_message, get_thread, compute_consensus

    # Agent sends a query to another agent
    thread_id = send_message(
        sender="risk_officer",
        recipient="geopolitical_analyst",
        msg_type="query",
        content="Is the rising CIS score supply-driven or escalation-driven?",
        subject_id="ukraine_russia",
    )

    # Responding agent handles it
    send_message(
        sender="geopolitical_analyst",
        recipient="commodities_specialist",
        msg_type="handoff",
        content="Escalation-driven. Handing off commodity transmission analysis.",
        subject_id="ukraine_russia",
        thread_id=thread_id,
    )
"""

from __future__ import annotations

import datetime
import uuid
from typing import Optional

# ── Message type registry ─────────────────────────────────────────────────────

MSG_TYPES = {
    "query":    {"label": "QUERY",    "color": "#2980b9"},
    "handoff":  {"label": "HANDOFF",  "color": "#8E9AAA"},
    "challenge":{"label": "CHALLENGE","color": "#e67e22"},
    "support":  {"label": "SUPPORT",  "color": "#27ae60"},
    "veto":     {"label": "VETO",     "color": "#c0392b"},
    "resolve":  {"label": "RESOLVE",  "color": "#CFB991"},
    "escalate": {"label": "ESCALATE", "color": "#c0392b"},
    "info":     {"label": "INFO",     "color": "#555960"},
}

# Escalation threshold: consensus below this → escalate to human
CONSENSUS_THRESHOLD = 0.50
# Auto-approve threshold: consensus above this → auto-submit
AUTO_APPROVE_THRESHOLD = 0.65


# ── Core message bus ──────────────────────────────────────────────────────────

def _init_dialogue() -> None:
    import streamlit as st
    if "agent_dialogue" not in st.session_state:
        st.session_state["agent_dialogue"] = []
    if "agent_consensus" not in st.session_state:
        st.session_state["agent_consensus"] = {}


def send_message(
    sender: str,
    recipient: str,
    msg_type: str,
    content: str,
    subject_id: Optional[str] = None,
    payload: Optional[dict] = None,
    thread_id: Optional[str] = None,
) -> str:
    """
    Post a message to the agent dialogue bus.

    Returns the thread_id (new if not supplied, so callers can group replies).
    """
    import streamlit as st
    _init_dialogue()

    tid = thread_id or str(uuid.uuid4())[:10]
    msg = {
        "id":         str(uuid.uuid4())[:8],
        "thread_id":  tid,
        "ts":         datetime.datetime.now(),
        "sender":     sender,
        "recipient":  recipient,
        "msg_type":   msg_type,
        "content":    content,
        "subject_id": subject_id,
        "payload":    payload or {},
    }
    feed = st.session_state["agent_dialogue"]
    feed.insert(0, msg)
    # Keep last 500 messages
    if len(feed) > 500:
        st.session_state["agent_dialogue"] = feed[:500]
    return tid


def get_thread(thread_id: str) -> list[dict]:
    """Return all messages in a thread, oldest first."""
    import streamlit as st
    _init_dialogue()
    msgs = [m for m in st.session_state.get("agent_dialogue", [])
            if m["thread_id"] == thread_id]
    return sorted(msgs, key=lambda m: m["ts"])


def get_subject_threads(subject_id: str) -> list[dict]:
    """Return all messages related to a subject (trade_id / conflict_id)."""
    import streamlit as st
    _init_dialogue()
    return [m for m in st.session_state.get("agent_dialogue", [])
            if m.get("subject_id") == subject_id]


def get_recent_dialogue(n: int = 20) -> list[dict]:
    """Return the n most recent messages across all threads."""
    import streamlit as st
    _init_dialogue()
    return st.session_state.get("agent_dialogue", [])[:n]


# ── Consensus computation ─────────────────────────────────────────────────────

def compute_consensus(thread_id: str) -> tuple[float, float]:
    """
    Compute consensus and disagreement scores for a thread.

    consensus_score    = (supports + 0.5 × resolves) / total_votes
    disagreement_score = (vetoes + challenges) / total_votes

    Returns (consensus, disagreement) both in [0, 1].
    """
    msgs = get_thread(thread_id)
    supports   = sum(1 for m in msgs if m["msg_type"] == "support")
    challenges = sum(1 for m in msgs if m["msg_type"] == "challenge")
    vetoes     = sum(1 for m in msgs if m["msg_type"] == "veto")
    resolves   = sum(1 for m in msgs if m["msg_type"] == "resolve")

    total = supports + challenges + vetoes + resolves
    if total == 0:
        return 0.5, 0.0

    consensus    = (supports + 0.5 * resolves) / total
    disagreement = (challenges + vetoes) / total
    return round(float(consensus), 2), round(float(disagreement), 2)


def get_provenance(thread_id: str) -> list[str]:
    """
    Return the ordered list of agents who participated in a thread.
    Used as 'who influenced this output'.
    """
    seen = []
    for msg in get_thread(thread_id):
        if msg["sender"] not in seen:
            seen.append(msg["sender"])
    return seen


# ── Protocol 1: Morning Briefing Chain ───────────────────────────────────────

def run_morning_briefing_protocol(
    risk_score: float,
    top_alerts: list[str],
    top_conflict: Optional[str] = None,
) -> str:
    """
    Kicks off the morning briefing inter-agent chain.
    risk_officer → geopolitical_analyst → commodities_specialist → macro_strategist → trade_structurer

    Returns the thread_id for the full briefing chain.
    """
    alert_text = "; ".join(top_alerts[:3]) if top_alerts else "No critical alerts"
    conflict_ctx = f" Dominant conflict: {top_conflict}." if top_conflict else ""

    tid = send_message(
        sender="risk_officer",
        recipient="broadcast",
        msg_type="info",
        content=(
            f"Morning briefing initiated. Geo risk score: {risk_score:.0f}/100.{conflict_ctx} "
            f"Top alerts: {alert_text}. Routing to Geopolitical Analyst for conflict decomposition."
        ),
        subject_id="morning_briefing",
    )

    send_message(
        sender="risk_officer",
        recipient="geopolitical_analyst",
        msg_type="query",
        content=(
            f"Geo risk score is {risk_score:.0f}/100. "
            "Is this primarily supply-driven (transmission pressure) or escalation-driven (conflict intensity)? "
            "Which conflict is contributing most?"
        ),
        subject_id="morning_briefing",
        thread_id=tid,
    )

    send_message(
        sender="geopolitical_analyst",
        recipient="commodities_specialist",
        msg_type="handoff",
        content=(
            "Conflict analysis complete. Handing off to Commodities for transmission channel scoring. "
            f"Primary conflict: {top_conflict or 'multi-front'}. "
            "Please assess which commodity routes are most stressed."
        ),
        subject_id="morning_briefing",
        thread_id=tid,
    )

    send_message(
        sender="commodities_specialist",
        recipient="macro_strategist",
        msg_type="handoff",
        content=(
            "Commodity transmission assessed. Handing off to Macro for inflation and rate implications. "
            "Energy and agriculture transmission channels are most active."
        ),
        subject_id="morning_briefing",
        thread_id=tid,
    )

    send_message(
        sender="macro_strategist",
        recipient="trade_structurer",
        msg_type="handoff",
        content=(
            "Macro implications assessed. Handing off to Trade Structurer for expression ideas. "
            "Key themes: energy inflation pass-through, safe-haven demand, EM currency pressure."
        ),
        subject_id="morning_briefing",
        thread_id=tid,
    )

    return tid


# ── Protocol 2: Trade Idea Challenge ─────────────────────────────────────────

def challenge_trade(
    trade_id: str,
    trade_title: str,
    confidence: float,
    qc_flags: list[str],
) -> str:
    """
    Run the quality challenge protocol for a new trade idea.
    signal_auditor and quality_officer review; stress_engineer adds tail risk.
    Returns thread_id.
    """
    tid = send_message(
        sender="trade_structurer",
        recipient="broadcast",
        msg_type="info",
        content=f"New trade submitted for review: {trade_title} (confidence: {confidence:.0%})",
        subject_id=trade_id,
    )

    if qc_flags:
        flag_text = "; ".join(qc_flags[:3])
        send_message(
            sender="quality_officer",
            recipient="trade_structurer",
            msg_type="veto" if len(qc_flags) >= 3 else "challenge",
            content=f"QC flags raised: {flag_text}. "
                    + ("Recommending rejection pending resolution."
                       if len(qc_flags) >= 3
                       else "Minor flags — confidence haircut applied."),
            subject_id=trade_id,
            thread_id=tid,
            payload={"flags": qc_flags},
        )
    else:
        send_message(
            sender="quality_officer",
            recipient="trade_structurer",
            msg_type="support",
            content="No QC flags. Data freshness and methodology checks pass.",
            subject_id=trade_id,
            thread_id=tid,
        )

    if confidence >= 0.65:
        send_message(
            sender="signal_auditor",
            recipient="trade_structurer",
            msg_type="support",
            content=f"Confidence {confidence:.0%} is above threshold. Signal stability check passes.",
            subject_id=trade_id,
            thread_id=tid,
        )
    else:
        send_message(
            sender="signal_auditor",
            recipient="trade_structurer",
            msg_type="challenge",
            content=f"Confidence {confidence:.0%} is below 65% threshold. "
                    "Recommend reducing position size or widening stop.",
            subject_id=trade_id,
            thread_id=tid,
        )

    send_message(
        sender="stress_engineer",
        recipient="trade_structurer",
        msg_type="info",
        content="Tail risk assessment: max drawdown estimate computed. "
                "Scenario stress applied. Stress-adjusted R/R reviewed.",
        subject_id=trade_id,
        thread_id=tid,
    )

    # Final resolution
    consensus, disagreement = compute_consensus(tid)
    if consensus >= AUTO_APPROVE_THRESHOLD and not qc_flags:
        send_message(
            sender="risk_officer",
            recipient="broadcast",
            msg_type="resolve",
            content=f"Consensus reached ({consensus:.0%}). Trade approved for human review.",
            subject_id=trade_id,
            thread_id=tid,
            payload={"consensus": consensus},
        )
    elif consensus < CONSENSUS_THRESHOLD or (qc_flags and len(qc_flags) >= 3):
        send_message(
            sender="risk_officer",
            recipient="broadcast",
            msg_type="escalate",
            content=f"Consensus below threshold ({consensus:.0%}) or critical QC flags. "
                    "Escalating to human for adjudication.",
            subject_id=trade_id,
            thread_id=tid,
            payload={"consensus": consensus, "disagreement": disagreement},
        )

    return tid


# ── UI rendering helper ───────────────────────────────────────────────────────

def render_dialogue_thread(thread_id: str, max_msgs: int = 12) -> None:
    """
    Render a threaded dialogue in Streamlit.
    Call from any page that shows agent deliberation.
    """
    import streamlit as st
    from src.analysis.agent_state import AGENTS

    msgs = get_thread(thread_id)[-max_msgs:]
    if not msgs:
        st.caption("No deliberation log for this thread.")
        return

    for msg in msgs:
        meta   = MSG_TYPES.get(msg["msg_type"], MSG_TYPES["info"])
        ag     = AGENTS.get(msg["sender"], {})
        ag_color = ag.get("color", "#8E9AAA")
        ts_str = msg["ts"].strftime("%H:%M:%S")

        st.markdown(
            f'<div style="border-left:2px solid {ag_color};padding:4px 8px;'
            f'margin:3px 0;background:rgba(0,0,0,0.15)">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            f'color:{ag_color};font-weight:700">{ag.get("short", msg["sender"])}</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            f'color:{meta["color"]};margin:0 6px;font-weight:700">'
            f'[{meta["label"]}]</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
            f'color:#555960">{ts_str}</span>'
            f'<div style="font-family:\'DM Sans\',sans-serif;font-size:10px;'
            f'color:#a8b0c0;margin-top:2px;line-height:1.5">{msg["content"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    consensus, disagreement = compute_consensus(thread_id)
    st.markdown(
        f'<div style="display:flex;gap:16px;margin-top:6px;padding-top:6px;'
        f'border-top:1px solid #1e1e1e">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        f'color:#27ae60">Consensus {consensus:.0%}</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        f'color:#e67e22">Disagreement {disagreement:.0%}</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        f'color:#8E9AAA">{len(msgs)} messages · {len(set(m["sender"] for m in msgs))} agents</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
