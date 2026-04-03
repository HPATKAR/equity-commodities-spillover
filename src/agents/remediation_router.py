"""
CQO Remediation Router - active correction layer for quality flags.

When the Chief Quality Officer raises a flag, this module:
  1. Parses structured flags out of CQO narrative text
  2. Routes each flag to the specialist agent best positioned to address it
  3. Runs a targeted corrective LLM call per agent (not the full agent run)
  4. Stores results so the UI can display agents actively working on corrections

This is Round 5 of the pipeline - always triggered by CQO output.
Cached 30 minutes (same as CQO TTL).
"""

from __future__ import annotations

import re
import datetime
import streamlit as st

from src.analysis.agent_state import log_activity, AGENTS

_AGENT = "quality_officer"

# ── Flag-to-agent routing table ──────────────────────────────────────────────
# Keywords in flag text → responsible specialist agent
_ROUTING: list[tuple[list[str], str]] = [
    (["look-ahead", "lookahead", "future", "forward-looking"], "signal_auditor"),
    (["spurious", "multicollinearity", "correlation", "r-squared", "overfitted"], "signal_auditor"),
    (["hardcoded", "static", "not live", "divorced from", "hit rate", "calibrat"], "signal_auditor"),
    (["sample", "observation", "n <", "insufficient data", "too few", "short window"], "macro_strategist"),
    (["regime", "mismatch", "mis-calibrated", "structural break", "non-stationary"], "macro_strategist"),
    (["assumption", "stacking", "stacked", "parametric", "distributional"], "stress_engineer"),
    (["stress", "tail", "scenario", "drawdown", "var", "cvar", "fat tail"], "stress_engineer"),
    (["trade", "stop loss", "overconfident", "entry", "exit", "conviction"], "trade_structurer"),
    (["geopolitical", "event window", "event study", "conflict", "n events"], "geopolitical_analyst"),
    (["commodity", "supply", "positioning", "cot", "physical"], "commodities_specialist"),
]

# Per-agent remediation system prompts (corrective posture, not full analysis)
_REMEDIATION_SYSTEMS: dict[str, str] = {
    "signal_auditor": (
        "You are the AI Signal Auditor in the Cross-Asset Spillover Monitor. "
        "The Chief Quality Officer has flagged methodological or statistical issues in the current analysis. "
        "Your job is to directly address each assigned flag with specific corrective actions: "
        "propose alternative test approaches, state the minimum sample N required, "
        "name which signal pairs are suspect and should be de-weighted, "
        "and specify any recalibration steps. Be concrete and actionable. "
        "Do not restate the flag - go straight to the fix."
    ),
    "macro_strategist": (
        "You are the AI Macro Strategist in the Cross-Asset Spillover Monitor. "
        "The Chief Quality Officer has flagged data or regime issues in the current analysis. "
        "Address each flag directly: if sample size is too short, propose the appropriate window "
        "and explain what regimes are excluded by the short window. "
        "If regime mismatch is flagged, specify which sub-regime the model should be conditioned on "
        "and what FRED/macro data would improve signal quality. Be specific."
    ),
    "stress_engineer": (
        "You are the AI Stress Engineer in the Cross-Asset Spillover Monitor. "
        "The Chief Quality Officer has flagged assumption-stacking or scenario design issues. "
        "Address each flag: identify which assumptions can be relaxed or replaced with empirical distributions, "
        "suggest fat-tail adjustments (e.g. EVT, historical simulation), "
        "and flag which stress scenarios are under-specified. Be concrete."
    ),
    "trade_structurer": (
        "You are the AI Trade Structurer in the Cross-Asset Spillover Monitor. "
        "The Chief Quality Officer has flagged issues with trade confidence or risk management. "
        "Address each flag: if confidence is overconfident, revise it downward with rationale, "
        "add a specific stop-loss level if missing, "
        "and state which conditions would invalidate the trade thesis. Be direct."
    ),
    "geopolitical_analyst": (
        "You are the AI Geopolitical Analyst in the Cross-Asset Spillover Monitor. "
        "The Chief Quality Officer has flagged event-study design or geopolitical data issues. "
        "Address each flag: if the event N is too low, name comparable historical precedents to add, "
        "propose a tighter or wider event window with justification, "
        "and specify which commodity channels are most exposed but underweighted. Be specific."
    ),
    "commodities_specialist": (
        "You are the AI Commodities Specialist in the Cross-Asset Spillover Monitor. "
        "The Chief Quality Officer has flagged commodity data or positioning issues. "
        "Address each flag: identify which supply/demand signals are missing, "
        "flag stale COT data if relevant, and propose alternative price benchmarks "
        "or positioning proxies where the flagged data is weak. Be actionable."
    ),
}


def parse_flags(cqo_narrative: str) -> list[dict]:
    """
    Extract structured flags from CQO output text.
    Returns list of {number, title, explanation, severity}.
    """
    flags = []
    severity = "Unknown"

    # Extract severity
    sev_match = re.search(r"SEVERITY:\s*(Critical|High|Medium|Low)", cqo_narrative, re.IGNORECASE)
    if sev_match:
        severity = sev_match.group(1)

    # Extract individual flags: "⚠ FLAG N: title - explanation"
    flag_pattern = re.compile(
        r"[⚠\u26a0]\s*FLAG\s*(\d+):\s*([^-\n]+?)\s*[-]\s*(.+?)(?=\n[⚠\u26a0\u2713]|\nSEVERITY|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    for m in flag_pattern.finditer(cqo_narrative):
        flags.append({
            "number":      int(m.group(1)),
            "title":       m.group(2).strip(),
            "explanation": m.group(3).strip().replace("\n", " "),
            "severity":    severity,
        })

    return flags


def route_flags(flags: list[dict]) -> dict[str, list[dict]]:
    """
    Map each flag to the agent best positioned to address it.
    Returns {agent_id: [flag, ...]}. One flag can go to only one agent.
    """
    routing: dict[str, list[dict]] = {}
    for flag in flags:
        text = (flag["title"] + " " + flag["explanation"]).lower()
        assigned = None
        for keywords, agent_id in _ROUTING:
            if any(kw in text for kw in keywords):
                assigned = agent_id
                break
        if assigned is None:
            assigned = "signal_auditor"  # default fallback
        routing.setdefault(assigned, []).append(flag)
    return routing


@st.cache_data(ttl=1800, show_spinner=False)
def _call_remediation_ai(
    agent_id: str,
    flags_text: str,
    page: str,
    provider: str,
    api_key: str,
) -> str:
    """LLM call for one agent's corrective response. Cached 30 min."""
    system = _REMEDIATION_SYSTEMS.get(agent_id, _REMEDIATION_SYSTEMS["signal_auditor"])
    prompt = (
        f"PAGE: {page}\n\n"
        f"CQO FLAGS ASSIGNED TO YOU:\n{flags_text}\n\n"
        "Address each flag above with specific corrective actions or workarounds. "
        "Be direct. No preamble. Start with the flag number."
    )
    try:
        if provider == "anthropic":
            import anthropic as _ant
            client = _ant.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
                system=system,
            )
            return resp.content[0].text.strip()
        else:
            from openai import OpenAI as _OAI
            client = _OAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=400, temperature=0.15,
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Remediation unavailable: {e}"


def run_remediation(
    cqo_output: dict,
    page: str,
    provider: str | None,
    api_key: str,
) -> dict[str, dict]:
    """
    Parse CQO flags, route to agents, run corrective calls.

    Returns:
        {agent_id: {"agent_name": str, "flags": list[dict], "response": str, "ts": datetime}}

    Result is also stored in st.session_state["cqo_remediations"][page].
    """
    narrative = cqo_output.get("narrative", "")
    if not narrative or not provider:
        return {}

    flags = parse_flags(narrative)
    if not flags:
        return {}

    routing = route_flags(flags)
    results: dict[str, dict] = {}

    for agent_id, assigned_flags in routing.items():
        agent_meta = AGENTS.get(agent_id, {})
        flags_text = "\n".join(
            f"FLAG {f['number']}: {f['title']} - {f['explanation']}"
            for f in assigned_flags
        )
        log_activity(
            _AGENT,
            f"routing {len(assigned_flags)} flag(s) to {agent_meta.get('short', agent_id)}",
            f"page: {page}",
            "info",
        )

        response = _call_remediation_ai(agent_id, flags_text, page, provider, api_key)

        log_activity(
            agent_id,
            f"CQO remediation: {page}",
            f"addressed {len(assigned_flags)} flag(s)",
            "info",
        )

        results[agent_id] = {
            "agent_name": agent_meta.get("name", agent_id),
            "agent_icon": agent_meta.get("icon", ""),
            "flags":      assigned_flags,
            "response":   response,
            "ts":         datetime.datetime.now(),
        }

    # Cache in session state for UI rendering
    if "cqo_remediations" not in st.session_state:
        st.session_state["cqo_remediations"] = {}
    st.session_state["cqo_remediations"][page] = results

    return results
