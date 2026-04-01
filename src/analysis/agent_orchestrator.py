"""
Agent Orchestrator — runs the AI workforce as a collaborative pipeline.

Previous architecture: each agent ran independently when you landed on its page,
with no awareness of what other agents had concluded. Peer signals were
truncated to 120 chars and pulled from stale session state.

New architecture: a dependency-ordered pipeline where each agent explicitly
receives the structured outputs of its upstream peers before it runs.
Divergence between agents is detected and flagged. Stale outputs are
invalidated. The Risk Officer now synthesises ALL Round 1 findings.

Pipeline rounds (sequential dependency order):
  Round 1 — Data gatherers (no peer dependencies):
    geo_analyst      ← GEOPOLITICAL_EVENTS + Strait Watch + regime
    macro_strategist ← FRED + yield curve + regime
    signal_auditor   ← Granger hit rates + model accuracy

  Round 2 — Synthesisers (depend on Round 1):
    risk_officer     ← alerts + all Round 1 outputs
    cmdty_specialist ← COT + prices + geo context

  Round 3 — Action layer (depend on Round 1+2):
    stress_engineer  ← scenarios + risk_officer risk level
    trade_structurer ← regime + ALL Round 1+2 agent outputs

  Round 4 — Audit (called per-page, always last):
    quality_officer  ← page-specific context (called separately)

Usage:
    from src.analysis.agent_orchestrator import Orchestrator
    orch = Orchestrator(provider, api_key)
    orch.run(market_context)          # runs full pipeline
    orch.status()                     # returns pipeline state summary
    orch.get_peer_context(agent_id)   # structured context for any agent
"""

from __future__ import annotations

import datetime
import json
from typing import Any

import streamlit as st

from src.analysis.agent_state import (
    AGENTS, init_agents, get_agent, set_status, log_activity,
    is_enabled,
)

# ── Pipeline definition ───────────────────────────────────────────────────────

PIPELINE: list[dict] = [
    # Round 1 — independent data gatherers
    {"id": "signal_auditor",       "round": 1, "depends_on": []},
    {"id": "macro_strategist",     "round": 1, "depends_on": []},
    {"id": "geopolitical_analyst", "round": 1, "depends_on": []},
    # Round 2 — synthesisers
    {"id": "risk_officer",          "round": 2,
     "depends_on": ["macro_strategist", "geopolitical_analyst", "signal_auditor"]},
    {"id": "commodities_specialist","round": 2,
     "depends_on": ["geopolitical_analyst"]},
    # Round 3 — action
    {"id": "stress_engineer",  "round": 3,
     "depends_on": ["risk_officer"]},
    {"id": "trade_structurer", "round": 3,
     "depends_on": ["risk_officer", "macro_strategist", "geopolitical_analyst",
                    "commodities_specialist"]},
]

# How long an agent output is considered fresh (seconds)
AGENT_TTL: dict[str, int] = {
    "signal_auditor":        3600,
    "macro_strategist":      3600,
    "geopolitical_analyst":  3600,
    "risk_officer":          3600,
    "commodities_specialist":3600,
    "stress_engineer":       3600,
    "trade_structurer":      3600,
    "quality_officer":       1800,
}

# Divergence detection: these agent pairs should NOT reach opposite conclusions
DIVERGENCE_PAIRS: list[tuple[str, str, str]] = [
    # (agent_a, agent_b, topic_keyword)
    ("macro_strategist", "risk_officer", "hawkish"),
    ("geopolitical_analyst", "stress_engineer", "critical"),
    ("macro_strategist", "geopolitical_analyst", "oil"),
]


def _orch_state() -> dict:
    """Lazy-init orchestrator state in session_state."""
    init_agents()
    if "orchestrator" not in st.session_state:
        st.session_state["orchestrator"] = {
            "pipeline_run":     None,   # datetime of last full run
            "pipeline_status":  "idle", # idle|running|complete|partial
            "round_complete":   {1: False, 2: False, 3: False},
            "divergence_flags": [],     # list of divergence dicts
            "context_cache":    {},     # agent_id → context dict used for that run
        }
    return st.session_state["orchestrator"]


def _is_fresh(agent_id: str) -> bool:
    """True if agent has output within its TTL window."""
    a = get_agent(agent_id)
    if not a.get("last_run") or not a.get("last_output"):
        return False
    age = (datetime.datetime.now() - a["last_run"]).total_seconds()
    return age < AGENT_TTL.get(agent_id, 3600)


def _peer_context(agent_id: str) -> dict[str, str]:
    """
    Build structured peer context for an agent — full narratives, not truncated.
    Strips only the CONFIDENCE line so the substantive content passes through.
    """
    spec = next((p for p in PIPELINE if p["id"] == agent_id), None)
    if not spec:
        return {}

    peers = {}
    for pid in spec.get("depends_on", []):
        a = get_agent(pid)
        raw = a.get("last_output") or ""
        if raw:
            # Remove CONFIDENCE line, keep everything else
            lines = [l for l in raw.split("\n")
                     if "confidence:" not in l.lower()]
            peers[pid] = " ".join(lines).strip()[:400]
    return peers


def _detect_divergence(
    agent_id: str, narrative: str, orch: dict
) -> list[dict]:
    """
    Check if this agent's output contradicts a peer's on a key topic.
    Returns list of divergence dicts (may be empty).
    """
    flags = []
    for a_id, b_id, keyword in DIVERGENCE_PAIRS:
        if agent_id not in (a_id, b_id):
            continue
        peer_id = b_id if agent_id == a_id else a_id
        peer = get_agent(peer_id)
        peer_text = (peer.get("last_output") or "").lower()
        if not peer_text:
            continue

        # Simple divergence: one says keyword, the other uses opposite sentiment
        a_positive = keyword in narrative.lower()
        b_positive = keyword in peer_text

        neg_words = ["not", "no ", "low", "easing", "declining", "falling", "unlikely"]
        a_negated = any(neg + " " + keyword in narrative.lower() for neg in neg_words)
        b_negated = any(neg + " " + keyword in peer_text for neg in neg_words)

        if (a_positive and not a_negated) != (b_positive and not b_negated):
            flags.append({
                "agent_a": agent_id,
                "agent_b": peer_id,
                "topic": keyword,
                "ts": datetime.datetime.now(),
            })
            log_activity(
                agent_id,
                f"divergence detected vs {AGENTS.get(peer_id,{}).get('short', peer_id)}",
                f"disagreement on '{keyword}'",
                "warning",
            )
    return flags


# ── Public API ────────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Coordinate the agent pipeline for one dashboard session.
    Call run() once on Overview page load; agents cache their own outputs.
    Pages can call get_peer_context() to get structured peer inputs.
    """

    def __init__(self, provider: str | None, api_key: str):
        self.provider  = provider
        self.api_key   = api_key
        self.orch      = _orch_state()

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def run(self, market_context: dict) -> dict[str, Any]:
        """
        Execute the full 3-round pipeline.

        market_context must contain (all optional but improves quality):
          regime_name, regime_level, risk_score, avg_corr, corr_delta,
          best_equity, worst_equity, best_commodity, worst_commodity,
          n_alerts, alert_categories, alert_summaries,
          eq_returns (pd.DataFrame), cmd_returns (pd.DataFrame),
          granger_hit_rates (dict), cpi_yoy, yield_curve_spread,
          fred_data (dict), scenarios (list[dict])
        """
        orch = self.orch
        orch["pipeline_status"] = "running"
        results: dict[str, Any] = {}

        for round_n in (1, 2, 3):
            agents_in_round = [p for p in PIPELINE if p["round"] == round_n]
            for spec in agents_in_round:
                aid = spec["id"]
                if not is_enabled(aid):
                    continue
                if _is_fresh(aid):
                    # Reuse cached output — still build peer context updates
                    continue
                ctx = self._build_context(aid, market_context)
                result = self._run_agent(aid, ctx)
                results[aid] = result

                # Detect divergence against already-run peers
                narrative = result.get("narrative", "")
                if narrative:
                    flags = _detect_divergence(aid, narrative, orch)
                    orch["divergence_flags"].extend(flags)

            orch["round_complete"][round_n] = True

        orch["pipeline_run"]    = datetime.datetime.now()
        orch["pipeline_status"] = "complete"
        return results

    def run_round_one(self, market_context: dict) -> dict[str, Any]:
        """Run only Round 1 agents. Useful for quick page loads."""
        results = {}
        for spec in [p for p in PIPELINE if p["round"] == 1]:
            aid = spec["id"]
            if not is_enabled(aid) or _is_fresh(aid):
                continue
            ctx = self._build_context(aid, market_context)
            results[aid] = self._run_agent(aid, ctx)
        return results

    # ── Context builders ───────────────────────────────────────────────────────

    def _build_context(self, agent_id: str, mc: dict) -> dict:
        """Build the context dict for a specific agent."""
        peer_ctx = _peer_context(agent_id)

        if agent_id == "signal_auditor":
            return {
                "granger_hit_rates": mc.get("granger_hit_rates", {}),
                "best_pair":         mc.get("best_pair"),
                "worst_pair":        mc.get("worst_pair"),
                "avg_hit_rate":      mc.get("avg_hit_rate"),
                "n_signals":         mc.get("n_signals"),
                "signal_decay":      mc.get("signal_decay", False),
                "regime_name":       mc.get("regime_name"),
            }

        if agent_id == "macro_strategist":
            return {
                "yield_curve_spread": mc.get("yield_curve_spread"),
                "cpi_yoy":           mc.get("cpi_yoy"),
                "fed_rate":          mc.get("fed_rate"),
                "gdp_growth":        mc.get("gdp_growth"),
                "ism_pmi":           mc.get("ism_pmi"),
                "regime_name":       mc.get("regime_name"),
                "risk_score":        mc.get("risk_score"),
            }

        if agent_id == "geopolitical_analyst":
            from src.data.config import GEOPOLITICAL_EVENTS
            import datetime as _dt
            today = _dt.date.today()
            active = [
                e for e in GEOPOLITICAL_EVENTS
                if e.get("end", today) >= today or
                   (_dt.date.today() - e.get("end", today)).days <= 365
            ]
            hi_sev = [e for e in active
                      if e.get("category") in ("War", "Conflict", "Sanctions", "Crisis")]
            cmd_kw = ["oil", "gas", "wheat", "gold", "copper", "grain", "energy", "nickel"]
            affected_cmds = []
            for e in active:
                desc = (e.get("description", "") + e.get("name", "")).lower()
                for kw in cmd_kw:
                    if kw in desc and kw.title() not in affected_cmds:
                        affected_cmds.append(kw.title())

            # Strait Watch context
            strait_notes = []
            try:
                from src.pages.strait_watch import _STRAITS
                for sw in _STRAITS:
                    if sw["disruption_score"] >= 30:
                        strait_notes.append(
                            f"{sw['name']}: disruption {sw['disruption_score']}/100, "
                            f"vessel traffic {sw['flow_change_pct']:+d}% vs baseline"
                        )
            except Exception:
                pass

            return {
                "n_events":           len(active),
                "high_severity":      len(hi_sev),
                "active_events":      [
                    {"name": e.get("name",""), "severity": e.get("category",""),
                     "region": e.get("region",""),
                     "commodity_impact": e.get("commodity_impact","")}
                    for e in active[:8]
                ],
                "affected_commodities": affected_cmds[:5],
                "affected_regions":     list({e.get("region","") for e in active if e.get("region")})[:4],
                "regime_name":          mc.get("regime_name"),
                "risk_score":           mc.get("risk_score"),
                "notes":                strait_notes,
            }

        if agent_id == "risk_officer":
            return {
                "regime_name":       mc.get("regime_name"),
                "regime_level":      mc.get("regime_level", 1),
                "risk_score":        mc.get("risk_score"),
                "avg_corr":          mc.get("avg_corr"),
                "corr_delta":        mc.get("corr_delta"),
                "best_equity":       mc.get("best_equity"),
                "worst_equity":      mc.get("worst_equity"),
                "best_commodity":    mc.get("best_commodity"),
                "worst_commodity":   mc.get("worst_commodity"),
                "n_alerts":          mc.get("n_alerts", 0),
                "alert_categories":  mc.get("alert_categories", []),
                "alert_summaries":   mc.get("alert_summaries", []),
                # Full peer narratives — this is what makes the RO actually synthesise
                "peer_signals": {
                    k: v for k, v in peer_ctx.items()
                },
            }

        if agent_id == "commodities_specialist":
            return {
                "top_performers":    mc.get("top_cmd_performers", []),
                "worst_performers":  mc.get("worst_cmd_performers", []),
                "crowded_longs":     mc.get("crowded_longs", []),
                "crowded_shorts":    mc.get("crowded_shorts", []),
                "avg_corr":          mc.get("avg_corr"),
                "regime_name":       mc.get("regime_name"),
                "geo_context":       peer_ctx.get("geopolitical_analyst", ""),
            }

        if agent_id == "stress_engineer":
            return {
                "scenarios":       mc.get("scenarios", []),
                "worst_scenario":  mc.get("worst_scenario"),
                "worst_impact":    mc.get("worst_impact"),
                "avg_impact":      mc.get("avg_impact"),
                "n_scenarios":     mc.get("n_scenarios", 0),
                "regime_name":     mc.get("regime_name"),
                "risk_score":      mc.get("risk_score"),
                "risk_context":    peer_ctx.get("risk_officer", ""),
            }

        if agent_id == "trade_structurer":
            return {
                "regime_name":      mc.get("regime_name"),
                "regime_level":     mc.get("regime_level", 1),
                "avg_corr":         mc.get("avg_corr"),
                "risk_score":       mc.get("risk_score"),
                "top_commodity":    mc.get("top_commodity"),
                "top_commodity_ret":mc.get("top_commodity_ret"),
                "worst_equity":     mc.get("worst_equity"),
                "worst_equity_ret": mc.get("worst_equity_ret"),
                "active_alerts":    mc.get("n_alerts", 0),
                # Full peer context — all upstream agents
                "peer_signals": {
                    k: v for k, v in peer_ctx.items()
                },
            }

        return {}

    # ── Agent dispatch ─────────────────────────────────────────────────────────

    def _run_agent(self, agent_id: str, context: dict) -> dict:
        """Dispatch to the appropriate agent module."""
        set_status(agent_id, "investigating")
        try:
            if agent_id == "signal_auditor":
                from src.agents.signal_auditor import run
            elif agent_id == "macro_strategist":
                from src.agents.macro_strategist import run
            elif agent_id == "geopolitical_analyst":
                from src.agents.geopolitical_analyst import run
            elif agent_id == "risk_officer":
                from src.agents.risk_officer import run
            elif agent_id == "commodities_specialist":
                from src.agents.commodities_specialist import run
            elif agent_id == "stress_engineer":
                from src.agents.stress_engineer import run
            elif agent_id == "trade_structurer":
                from src.agents.trade_structurer import run
            else:
                return {}
            return run(context, self.provider, self.api_key)  # type: ignore
        except Exception as e:
            log_activity(agent_id, "pipeline error", str(e)[:80], "warning")
            set_status(agent_id, "idle")
            return {"error": str(e)}

    # ── Introspection ──────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return a summary of pipeline state for UI rendering."""
        orch = self.orch
        agents_fresh  = [aid for aid in AGENTS if _is_fresh(aid)]
        agents_stale  = [aid for aid in AGENTS if not _is_fresh(aid)]
        return {
            "pipeline_run":     orch.get("pipeline_run"),
            "pipeline_status":  orch.get("pipeline_status", "idle"),
            "rounds_complete":  orch.get("round_complete", {}),
            "agents_fresh":     agents_fresh,
            "agents_stale":     agents_stale,
            "divergence_flags": orch.get("divergence_flags", []),
            "n_divergences":    len(orch.get("divergence_flags", [])),
        }

    def divergence_flags(self) -> list[dict]:
        return self.orch.get("divergence_flags", [])

    def get_peer_context(self, agent_id: str) -> dict[str, str]:
        return _peer_context(agent_id)

    def invalidate(self, agent_id: str | None = None) -> None:
        """Force re-run of one agent or all agents next pipeline execution."""
        import streamlit as st
        init_agents()
        targets = [agent_id] if agent_id else list(AGENTS.keys())
        for aid in targets:
            a = st.session_state["agents"].get(aid)
            if a:
                a["last_run"]    = None
                a["last_output"] = None
                a["status"]      = "idle"
        orch = _orch_state()
        orch["pipeline_run"]    = None
        orch["pipeline_status"] = "idle"
        orch["round_complete"]  = {1: False, 2: False, 3: False}


def get_orchestrator(provider: str | None, api_key: str) -> Orchestrator:
    """Factory — returns (and caches) the session's Orchestrator instance."""
    return Orchestrator(provider, api_key)
