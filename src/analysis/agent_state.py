"""
Shared agent state manager.
Manages the session_state schema for all 8 AI agents:
  risk_officer, macro_strategist, commodities_specialist,
  geopolitical_analyst, stress_engineer, signal_auditor, trade_structurer,
  quality_officer

All functions operate on st.session_state directly. Call init_agents() once
per app load (idempotent - skips if already initialised).
"""

from __future__ import annotations

import datetime
import uuid
from typing import Optional

# ── Agent registry ────────────────────────────────────────────────────────────

AGENTS = {
    "risk_officer": {
        "name":  "AI Risk Officer",
        "short": "Risk Officer",
        "icon":  "RO",
        "pages": ["overview"],
        "color": "#c0392b",
        "desc":  "Morning briefing · Cross-asset stress · Routing hub",
    },
    "macro_strategist": {
        "name":  "AI Macro Strategist",
        "short": "Macro Strat",
        "icon":  "MS",
        "pages": ["macro_dashboard"],
        "color": "#2980b9",
        "desc":  "Yield curve · Inflation · Fed policy · GDP",
    },
    "commodities_specialist": {
        "name":  "AI Commodities Specialist",
        "short": "Cmdties",
        "icon":  "CS",
        "pages": ["watchlist"],
        "color": "#8E6F3E",
        "desc":  "COT positioning · Supply shocks · Sector rotation",
    },
    "geopolitical_analyst": {
        "name":  "AI Geopolitical Analyst",
        "short": "Geo Intel",
        "icon":  "GA",
        "pages": ["geopolitical", "war_impact_map"],
        "color": "#6c3483",
        "desc":  "Conflict risk · Sanctions · Supply disruption",
    },
    "stress_engineer": {
        "name":  "AI Stress Engineer",
        "short": "Stress Eng",
        "icon":  "SE",
        "pages": ["stress_test"],
        "color": "#e67e22",
        "desc":  "Scenario stress · Tail risk · Portfolio shock",
    },
    "signal_auditor": {
        "name":  "AI Signal Auditor",
        "short": "Auditor",
        "icon":  "SA",
        "pages": ["model_accuracy"],
        "color": "#27ae60",
        "desc":  "Calibration · Confidence scoring · Model validation",
    },
    "trade_structurer": {
        "name":  "AI Trade Structurer",
        "short": "Trade Str",
        "icon":  "TS",
        "pages": ["trade_ideas"],
        "color": "#CFB991",
        "desc":  "Trade ideas · Entry/exit · Awaiting human approval",
    },
    "quality_officer": {
        "name":  "AI Chief Quality Officer",
        "short": "CQO",
        "icon":  "CQO",
        "pages": ["spillover", "correlation", "geopolitical", "war_impact_map",
                  "trade_ideas", "stress_test", "insights", "overview", "strait_watch"],
        "color": "#e74c3c",
        "desc":  "Data integrity · Assumption auditing · Bullshit detection",
    },
}

# Valid status values
STATUSES = {
    "monitoring":        {"label": "Monitoring",         "color": "#27ae60"},
    "investigating":     {"label": "Investigating",      "color": "#e67e22"},
    "awaiting_input":    {"label": "Awaiting Input",     "color": "#2980b9"},
    "idle":              {"label": "Idle",                "color": "#555960"},
    "paused":            {"label": "Paused",             "color": "#555960"},
    "awaiting_approval": {"label": "Awaiting Approval",  "color": "#e67e22"},
    "approved":          {"label": "Approved",           "color": "#27ae60"},
    "rejected":          {"label": "Rejected",           "color": "#c0392b"},
    "escalated":         {"label": "Escalated",          "color": "#c0392b"},
    "overridden":        {"label": "Overridden",         "color": "#8890a1"},
}


def _default_agent(agent_id: str) -> dict:
    return {
        "status":      "idle",
        "enabled":     True,
        "last_run":    None,       # datetime or None
        "last_output": None,       # str narrative
        "confidence":  None,       # float 0-1 or None
        "findings":    [],         # list of finding dicts
        "peers_seen":  [],         # which peer outputs this agent has digested
    }


def init_agents() -> None:
    """Idempotently initialise all agent state in st.session_state."""
    import streamlit as st

    if "agents" not in st.session_state:
        st.session_state["agents"] = {
            aid: _default_agent(aid) for aid in AGENTS
        }
    else:
        # Ensure any new agents added to registry are present
        for aid in AGENTS:
            if aid not in st.session_state["agents"]:
                st.session_state["agents"][aid] = _default_agent(aid)

    if "agent_activity" not in st.session_state:
        st.session_state["agent_activity"] = []   # newest first

    if "pending_review" not in st.session_state:
        st.session_state["pending_review"] = []


# ── State mutators ────────────────────────────────────────────────────────────

def set_status(agent_id: str, status: str) -> None:
    import streamlit as st
    init_agents()
    if agent_id in st.session_state["agents"]:
        st.session_state["agents"][agent_id]["status"] = status


def set_output(agent_id: str, output: str,
               confidence: Optional[float] = None,
               findings: Optional[list] = None) -> None:
    import streamlit as st
    init_agents()
    a = st.session_state["agents"].get(agent_id)
    if a is None:
        return
    a["last_output"]  = output
    a["last_run"]     = datetime.datetime.now()
    if confidence is not None:
        a["confidence"] = confidence
    if findings is not None:
        a["findings"] = findings
    a["status"] = "monitoring"


def log_activity(agent_id: str, action: str, detail: str,
                 severity: str = "info",
                 routed_to: Optional[str] = None) -> None:
    """Prepend a timestamped entry to the shared activity feed (newest first)."""
    import streamlit as st
    init_agents()
    entry = {
        "id":         str(uuid.uuid4())[:8],
        "ts":         datetime.datetime.now(),
        "agent_id":   agent_id,
        "agent_name": AGENTS.get(agent_id, {}).get("short", agent_id),
        "action":     action,
        "detail":     detail,
        "severity":   severity,
        "routed_to":  routed_to,
    }
    feed = st.session_state["agent_activity"]
    feed.insert(0, entry)
    # Keep last 200 entries
    if len(feed) > 200:
        st.session_state["agent_activity"] = feed[:200]


def add_pending(
    agent_id: str,
    title: str,
    summary: str,
    rationale: str,
    confidence: float,
    severity: str = "info",
    extra: Optional[dict] = None,
) -> str:
    """Add a trade idea / finding to the Pending Review queue. Returns item ID.

    Deduplicates: if a non-rejected item with the same agent_id + title already
    exists in the queue (any status), skip insertion and return its existing ID.
    This prevents re-runs of an agent from re-queuing already-reviewed items.
    """
    import streamlit as st
    init_agents()

    # Deduplicate: skip if same agent+title already exists in any non-rejected state
    existing = st.session_state.get("pending_review", [])
    for _existing in existing:
        if (
            _existing["agent_id"] == agent_id
            and _existing["title"] == title
            and _existing["status"] != "rejected"
        ):
            return _existing["id"]

    item_id = str(uuid.uuid4())[:8]
    item = {
        "id":             item_id,
        "agent_id":       agent_id,
        "agent_name":     AGENTS.get(agent_id, {}).get("name", agent_id),
        "title":          title,
        "summary":        summary,
        "rationale":      rationale,
        "confidence":     confidence,
        "severity":       severity,
        "created_at":     datetime.datetime.now(),
        "status":         "pending",   # pending|approved|rejected|escalated
        "escalation_peer": None,
        "escalation_note": None,
        "extra":          extra or {},
    }
    st.session_state["pending_review"].insert(0, item)
    set_status(agent_id, "awaiting_approval")
    log_activity(agent_id, "submitted for review", title, severity)
    return item_id


def review_item(item_id: str, decision: str, note: str = "",
                escalation_peer: Optional[str] = None) -> None:
    """Apply approve / reject / escalate to a pending review item."""
    import streamlit as st
    init_agents()
    for item in st.session_state["pending_review"]:
        if item["id"] == item_id:
            item["status"] = decision
            item["escalation_note"] = note
            if escalation_peer:
                item["escalation_peer"] = escalation_peer
            # Log the decision
            log_activity(
                item["agent_id"],
                f"review: {decision}",
                item["title"],
                "info",
            )
            # Update agent status
            set_status(item["agent_id"], decision)
            break


def is_enabled(agent_id: str) -> bool:
    import streamlit as st
    init_agents()
    return st.session_state["agents"].get(agent_id, {}).get("enabled", True)


def toggle_agent(agent_id: str) -> None:
    import streamlit as st
    init_agents()
    a = st.session_state["agents"].get(agent_id)
    if a:
        a["enabled"] = not a["enabled"]
        if not a["enabled"]:
            a["status"] = "paused"
        else:
            a["status"] = "monitoring"


def get_agent(agent_id: str) -> dict:
    import streamlit as st
    init_agents()
    return st.session_state["agents"].get(agent_id, _default_agent(agent_id))


def pending_count() -> int:
    import streamlit as st
    init_agents()
    return sum(1 for i in st.session_state.get("pending_review", [])
               if i["status"] == "pending")


def calibrate_confidence(raw: float, agent_id: str) -> float:
    """
    Signal Auditor calibration layer.
    Applies a per-agent historical shrinkage factor if the auditor has run.
    Falls back to raw confidence if auditor hasn't calibrated yet.
    """
    import streamlit as st
    init_agents()
    auditor = st.session_state["agents"].get("signal_auditor", {})
    calibration = auditor.get("extra", {}).get("calibration", {})
    factor = calibration.get(agent_id, 1.0)
    return min(max(raw * factor, 0.0), 1.0)
