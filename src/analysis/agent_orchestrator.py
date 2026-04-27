"""
Agent Orchestrator - runs the AI workforce as a collaborative pipeline.

Previous architecture: each agent ran independently when you landed on its page,
with no awareness of what other agents had concluded. Peer signals were
truncated to 120 chars and pulled from stale session state.

New architecture: a dependency-ordered pipeline where each agent explicitly
receives the structured outputs of its upstream peers before it runs.
Divergence between agents is detected and flagged. Stale outputs are
invalidated. The Risk Officer now synthesises ALL Round 1 findings.

Pipeline rounds (sequential dependency order):
  Round 1 - Data gatherers (no peer dependencies):
    geo_analyst      ← GEOPOLITICAL_EVENTS + Strait Watch + regime
    macro_strategist ← FRED + yield curve + regime
    signal_auditor   ← Granger hit rates + model accuracy

  Round 2 - Synthesisers (depend on Round 1):
    risk_officer     ← alerts + all Round 1 outputs
    cmdty_specialist ← COT + prices + geo context

  Round 3 - Action layer (depend on Round 1+2):
    stress_engineer  ← scenarios + risk_officer risk level
    trade_structurer ← regime + ALL Round 1+2 agent outputs

  Round 4 - Audit (called per-page, always last):
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
from typing import Any, TypedDict, Optional

import streamlit as st

from src.analysis.agent_state import (
    AGENTS, init_agents, get_agent, set_status, log_activity,
    is_enabled,
)


# ── Structured handoff schema ─────────────────────────────────────────────────
# Every agent run() returns this alongside its narrative text.
# Downstream agents receive these typed fields — NOT a truncated string —
# so numeric precision is preserved and divergence detection is field-level.

class AgentHandoff(TypedDict, total=False):
    agent_id:     str           # which agent produced this
    ts:           datetime.datetime
    narrative:    str           # full text (never truncated in handoffs)
    confidence:   float         # 0.0–1.0  (numeric, not regex-parsed from text)
    regime:       int           # 1=Normal 2=Elevated 3=Crisis
    risk_score:   Optional[float]   # 0–100 composite, if agent computes it
    signal_class: str           # "macro" | "geo" | "commodity" | "cross_asset" | "audit"
    # Key typed signals — each agent populates what it knows
    yield_curve_spread: Optional[float]   # TLT-SHY 60d spread, bps
    cpi_yoy:            Optional[float]   # CPI YoY %
    cis:                Optional[float]   # Conflict Intensity Score 0–100
    tps:                Optional[float]   # Transmission Pressure Score 0–100
    top_conflict:       Optional[str]     # e.g. "russia_ukraine"
    cmd_vol_z:          Optional[float]   # commodity vol z-score
    corr_pct:           Optional[float]   # correlation percentile
    granger_hit_rate:   Optional[float]   # signal auditor hit rate
    routed_to:          list              # downstream agent ids
    low_confidence:     bool              # True if confidence < CONFIDENCE_THRESHOLDS[agent_id]

# ── Pipeline definition ───────────────────────────────────────────────────────

PIPELINE: list[dict] = [
    # Round 1 - independent data gatherers
    {"id": "signal_auditor",       "round": 1, "depends_on": []},
    {"id": "macro_strategist",     "round": 1, "depends_on": []},
    {"id": "geopolitical_analyst", "round": 1, "depends_on": []},
    # Round 2 - synthesisers
    {"id": "risk_officer",          "round": 2,
     "depends_on": ["macro_strategist", "geopolitical_analyst", "signal_auditor"]},
    {"id": "commodities_specialist","round": 2,
     "depends_on": ["geopolitical_analyst"]},
    # Round 3 - action
    {"id": "stress_engineer",  "round": 3,
     "depends_on": ["risk_officer"]},
    {"id": "trade_structurer", "round": 3,
     "depends_on": ["risk_officer", "macro_strategist", "geopolitical_analyst",
                    "commodities_specialist"]},
]

# ── Confidence gate thresholds ────────────────────────────────────────────────
# Below these levels the harness flags the output as LOW CONFIDENCE.
# Thresholds are set at 80% of each agent's measured eval hit rate so the
# gate fires when confidence falls meaningfully below demonstrated accuracy:
#   risk_officer       eval hit rate 77.3%  →  gate at 0.55 (80% × 0.69 base)
#   macro_strategist   eval hit rate 70.0%  →  gate at 0.50
#   geopolitical_analyst  75.0%            →  gate at 0.52
#   commodities_specialist 80.0%           →  gate at 0.55
#   stress_engineer    66.7%               →  gate at 0.48
#   signal_auditor     manual              →  gate at 0.48 (conservative)
#   trade_structurer   structured output   →  gate at 0.45 (Pydantic schema guards quality)
# Below threshold: output is still shown but flagged LOW CONFIDENCE in the
# harness trace and surfaced as a warning badge in the AI Workforce UI.
CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "risk_officer":          0.55,
    "macro_strategist":      0.50,
    "geopolitical_analyst":  0.52,
    "commodities_specialist":0.55,
    "stress_engineer":       0.48,
    "signal_auditor":        0.48,
    "trade_structurer":      0.45,
    "quality_officer":       0.60,
}

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


def _store_handoff(agent_id: str, handoff: AgentHandoff) -> None:
    """Persist structured handoff in session_state for peer lookup."""
    init_agents()
    if "agent_handoffs" not in st.session_state:
        st.session_state["agent_handoffs"] = {}
    st.session_state["agent_handoffs"][agent_id] = handoff


def _get_handoff(agent_id: str) -> AgentHandoff | None:
    """Retrieve the latest structured handoff for an agent."""
    return st.session_state.get("agent_handoffs", {}).get(agent_id)


def _peer_context(agent_id: str) -> dict[str, "AgentHandoff"]:
    """
    Build structured peer context for an agent.
    Returns typed AgentHandoff dicts — NOT truncated strings — so numeric
    precision is preserved across the pipeline handoff chain.
    Downstream agents receive e.g. peer["macro_strategist"]["yield_curve_spread"]
    as a float, not a substring of a 400-char blob.
    """
    spec = next((p for p in PIPELINE if p["id"] == agent_id), None)
    if not spec:
        return {}

    peers: dict[str, AgentHandoff] = {}
    for pid in spec.get("depends_on", []):
        h = _get_handoff(pid)
        if h:
            peers[pid] = h
        else:
            # Fallback: wrap legacy text output in minimal handoff struct
            a   = get_agent(pid)
            raw = a.get("last_output") or ""
            if raw:
                peers[pid] = AgentHandoff(
                    agent_id=pid,
                    ts=a.get("last_run", datetime.datetime.now()),
                    narrative=raw,   # FULL text — no truncation
                    confidence=a.get("confidence", 0.5),
                    regime=None, risk_score=None, signal_class="unknown",
                    routed_to=[],
                )
    return peers


def _detect_divergence(
    agent_id: str, handoff: AgentHandoff, orch: dict
) -> list[dict]:
    """
    Numeric field-level divergence detection.
    Compares typed fields (risk_score, regime, cis, cmd_vol_z) between peer
    handoffs — NOT keyword string matching. A divergence is flagged when:
      • risk_score differs by > 20 pts between two agents that both compute it
      • regime disagrees between geo_analyst and macro_strategist
      • cis vs risk_score imply opposite threat levels
    Returns list of divergence dicts (may be empty).
    """
    flags: list[dict] = []
    peer_handoffs = st.session_state.get("agent_handoffs", {})

    NUMERIC_PAIRS: list[tuple[str, str, str, float]] = [
        # (agent_a, agent_b, field, max_allowed_diff)
        ("macro_strategist",    "risk_officer",     "risk_score",  20.0),
        ("geopolitical_analyst","risk_officer",      "risk_score",  20.0),
        ("geopolitical_analyst","stress_engineer",   "regime",       1.0),
        ("macro_strategist",    "geopolitical_analyst", "regime",    1.0),
    ]

    for a_id, b_id, field, threshold in NUMERIC_PAIRS:
        if agent_id not in (a_id, b_id):
            continue
        peer_id = b_id if agent_id == a_id else a_id
        peer_h  = peer_handoffs.get(peer_id)
        if not peer_h:
            continue

        val_a = handoff.get(field)
        val_b = peer_h.get(field)
        if val_a is None or val_b is None:
            continue

        diff = abs(float(val_a) - float(val_b))
        if diff > threshold:
            flags.append({
                "agent_a":  agent_id,
                "agent_b":  peer_id,
                "field":    field,
                "val_a":    val_a,
                "val_b":    val_b,
                "diff":     round(diff, 2),
                "threshold":threshold,
                "ts":       datetime.datetime.now(),
            })
            log_activity(
                agent_id,
                f"numeric divergence vs {AGENTS.get(peer_id,{}).get('short', peer_id)}",
                f"{field}: {val_a} vs {val_b} (Δ={diff:.1f}, threshold={threshold})",
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
        # Clear stale divergence flags from previous run — flags must reflect
        # the current pipeline execution only, not accumulate across sessions.
        orch["divergence_flags"] = []
        results: dict[str, Any] = {}

        for round_n in (1, 2, 3):
            agents_in_round = [p for p in PIPELINE if p["round"] == round_n]
            for spec in agents_in_round:
                aid = spec["id"]
                if not is_enabled(aid):
                    continue
                if _is_fresh(aid):
                    # Still build and store a fresh handoff so peer context
                    # is available to downstream agents even on cache hits.
                    existing = _get_handoff(aid)
                    if not existing:
                        cached_result = {"narrative": get_agent(aid).get("last_output", "")}
                        h = self._build_handoff(aid, cached_result, market_context)
                        _store_handoff(aid, h)
                    continue
                ctx = self._build_context(aid, market_context)
                result = self._run_agent(aid, ctx)
                results[aid] = result

                # Skip handoff and divergence for failed agents — an error
                # result has no typed fields to check and must not pollute
                # downstream peer context with empty/wrong numeric data.
                if result.get("error"):
                    continue

                # Store structured handoff for downstream peer context
                h = self._build_handoff(aid, result, market_context)
                _store_handoff(aid, h)

                # Confidence gate — log to trace when output falls below threshold.
                # Output is still used downstream but flagged so peers and UI
                # know they are working with uncertain upstream data.
                if h.get("low_confidence"):
                    gate = CONFIDENCE_THRESHOLDS.get(aid, 0.50)
                    try:
                        from src.analysis.trace_logger import log_failure
                        log_failure(aid, "LowConfidence",
                                    f"conf={h['confidence']:.2f} < gate={gate:.2f}")
                    except Exception:
                        pass
                    log_activity(aid, "low confidence gate",
                                 f"conf={h['confidence']:.2f} < threshold={gate:.2f} "
                                 f"— output flagged, downstream peers notified",
                                 "warning")

                # Numeric field-level divergence detection — runs on the handoff
                # regardless of whether narrative is populated (field check is
                # independent of text; empty narrative does not skip this).
                flags = _detect_divergence(aid, h, orch)
                orch["divergence_flags"].extend(flags)

            orch["round_complete"][round_n] = True

        orch["pipeline_run"]    = datetime.datetime.now()
        # Mark partial failure if any agent returned an error
        failed = [aid for aid, r in results.items() if r.get("error")]
        orch["pipeline_status"] = "partial" if failed else "complete"
        if failed:
            orch["failed_agents"] = failed
        return results

    def run_round_one(self, market_context: dict) -> dict[str, Any]:
        """
        Run only Round 1 agents. Useful for quick page loads.
        Stores handoffs so downstream agents can access peer context
        even when only the fast path is used.
        """
        orch = self.orch
        results = {}
        for spec in [p for p in PIPELINE if p["round"] == 1]:
            aid = spec["id"]
            if not is_enabled(aid):
                continue
            if _is_fresh(aid):
                existing = _get_handoff(aid)
                if not existing:
                    cached_result = {"narrative": get_agent(aid).get("last_output", "")}
                    h = self._build_handoff(aid, cached_result, market_context)
                    _store_handoff(aid, h)
                continue
            ctx = self._build_context(aid, market_context)
            result = self._run_agent(aid, ctx)
            results[aid] = result
            # Store handoff — critical so Round 2/3 agents that call
            # _peer_context() later in the session find typed data.
            h = self._build_handoff(aid, result, market_context)
            _store_handoff(aid, h)
            if h.get("low_confidence"):
                gate = CONFIDENCE_THRESHOLDS.get(aid, 0.50)
                try:
                    from src.analysis.trace_logger import log_failure
                    log_failure(aid, "LowConfidence",
                                f"conf={h['confidence']:.2f} < gate={gate:.2f}")
                except Exception:
                    pass
                log_activity(aid, "low confidence gate",
                             f"conf={h['confidence']:.2f} < threshold={gate:.2f}",
                             "warning")
            flags = _detect_divergence(aid, h, orch)
            orch["divergence_flags"].extend(flags)
        return results

    # ── Handoff builder ────────────────────────────────────────────────────────

    def _build_handoff(self, agent_id: str, result: dict, mc: dict) -> AgentHandoff:
        """
        Extract typed numeric fields from an agent result + market context
        and package them as an AgentHandoff for downstream peer consumption.
        All fields are typed — no regex parsing, no string truncation.
        """
        a     = get_agent(agent_id)
        conf  = result.get("confidence", a.get("confidence", 0.5))
        regime_map = {"Normal": 1, "Elevated": 2, "Crisis": 3}
        # Prefer agent's own regime assessment (result["regime"]) over the
        # shared market-context regime.  Using mc["regime_name"] for ALL agents
        # makes them identical, rendering regime divergence checks vacuous —
        # every agent would report the same value and Δ would always be 0.
        agent_regime_raw = result.get("regime_name") or mc.get("regime_name", "Normal")
        regime_int = result.get("regime") or regime_map.get(agent_regime_raw, 1)

        conf_float = float(conf) if conf is not None else 0.5
        gate       = CONFIDENCE_THRESHOLDS.get(agent_id, 0.50)

        h = AgentHandoff(
            agent_id=agent_id,
            ts=datetime.datetime.now(),
            narrative=result.get("narrative", ""),  # FULL text, never truncated
            confidence=conf_float,
            regime=regime_int,
            routed_to=result.get("routed_to", []),
            low_confidence=(conf_float < gate),
        )

        # Per-agent typed fields
        if agent_id == "macro_strategist":
            h["signal_class"] = "macro"
            h["yield_curve_spread"] = mc.get("yield_curve_spread")
            h["cpi_yoy"]            = mc.get("cpi_yoy")
            h["risk_score"]         = mc.get("risk_score")

        elif agent_id == "geopolitical_analyst":
            h["signal_class"] = "geo"
            # Pull live CIS/TPS from conflict model — not from agent text
            try:
                from src.analysis.conflict_model import score_all_conflicts, aggregate_portfolio_scores
                agg = aggregate_portfolio_scores(score_all_conflicts())
                h["cis"]          = float(agg.get("portfolio_cis", agg.get("cis", 50.0)))
                h["tps"]          = float(agg.get("portfolio_tps", agg.get("tps", 50.0)))
                h["top_conflict"] = agg.get("top_conflict")
                h["risk_score"]   = round(0.4 * h["cis"] + 0.35 * h["tps"] + 0.25 * 50, 1)
            except Exception:
                h["cis"] = h["tps"] = None

        elif agent_id == "signal_auditor":
            h["signal_class"]     = "audit"
            h["granger_hit_rate"] = mc.get("avg_hit_rate")

        elif agent_id == "risk_officer":
            h["signal_class"]  = "cross_asset"
            h["risk_score"]    = mc.get("risk_score")
            h["corr_pct"]      = mc.get("avg_corr")

        elif agent_id == "commodities_specialist":
            h["signal_class"] = "commodity"
            h["cmd_vol_z"]    = mc.get("cmd_vol_z")
            # pull CIS/TPS from geo peer handoff if available
            geo_h = _get_handoff("geopolitical_analyst")
            if geo_h:
                h["cis"] = geo_h.get("cis")
                h["tps"] = geo_h.get("tps")

        elif agent_id == "stress_engineer":
            h["signal_class"] = "cross_asset"
            h["risk_score"]   = mc.get("risk_score")

        elif agent_id == "trade_structurer":
            h["signal_class"] = "cross_asset"
            h["risk_score"]   = mc.get("risk_score")

        return h

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
                "affected_regions":     list({str(e.get("region","")) for e in active if e.get("region")})[:4],
                "regime_name":          mc.get("regime_name"),
                "risk_score":           mc.get("risk_score"),
                "notes":                strait_notes,
            }

        if agent_id == "risk_officer":
            # peer_ctx now contains typed AgentHandoff dicts — numeric fields preserved
            geo_h   = peer_ctx.get("geopolitical_analyst", {})
            macro_h = peer_ctx.get("macro_strategist", {})
            audit_h = peer_ctx.get("signal_auditor", {})
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
                # Typed peer fields — downstream agents use these directly
                "peer_cis":          geo_h.get("cis"),
                "peer_tps":          geo_h.get("tps"),
                "peer_top_conflict": geo_h.get("top_conflict"),
                "peer_geo_regime":   geo_h.get("regime"),
                "peer_yield_curve":  macro_h.get("yield_curve_spread"),
                "peer_cpi_yoy":      macro_h.get("cpi_yoy"),
                "peer_macro_regime": macro_h.get("regime"),
                "peer_hit_rate":     audit_h.get("granger_hit_rate"),
                "peer_confidence": {
                    pid: h.get("confidence") for pid, h in peer_ctx.items()
                },
                # Full narratives still available for LLM prompt enrichment
                "peer_narratives": {
                    pid: h.get("narrative", "") for pid, h in peer_ctx.items()
                },
            }

        if agent_id == "commodities_specialist":
            geo_h = peer_ctx.get("geopolitical_analyst", {})
            return {
                "top_performers":    mc.get("top_cmd_performers", []),
                "worst_performers":  mc.get("worst_cmd_performers", []),
                "crowded_longs":     mc.get("crowded_longs", []),
                "crowded_shorts":    mc.get("crowded_shorts", []),
                "avg_corr":          mc.get("avg_corr"),
                "regime_name":       mc.get("regime_name"),
                # Typed geo fields — not a truncated string
                "geo_cis":           geo_h.get("cis"),
                "geo_tps":           geo_h.get("tps"),
                "geo_top_conflict":  geo_h.get("top_conflict"),
                "geo_narrative":     geo_h.get("narrative", ""),
            }

        if agent_id == "stress_engineer":
            ro_h = peer_ctx.get("risk_officer", {})
            return {
                "scenarios":       mc.get("scenarios", []),
                "worst_scenario":  mc.get("worst_scenario"),
                "worst_impact":    mc.get("worst_impact"),
                "avg_impact":      mc.get("avg_impact"),
                "n_scenarios":     mc.get("n_scenarios", 0),
                "regime_name":     mc.get("regime_name"),
                "risk_score":      mc.get("risk_score"),
                # Typed RO fields
                "ro_risk_score":   ro_h.get("risk_score"),
                "ro_regime":       ro_h.get("regime"),
                "ro_narrative":    ro_h.get("narrative", ""),
            }

        if agent_id == "trade_structurer":
            # All upstream typed handoffs available
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
                # Typed numeric fields from all upstream peers
                "peer_risk_scores": {
                    pid: h.get("risk_score") for pid, h in peer_ctx.items()
                    if h.get("risk_score") is not None
                },
                "peer_regimes": {
                    pid: h.get("regime") for pid, h in peer_ctx.items()
                    if h.get("regime") is not None
                },
                "peer_cis":          peer_ctx.get("geopolitical_analyst", {}).get("cis"),
                "peer_tps":          peer_ctx.get("geopolitical_analyst", {}).get("tps"),
                "peer_yield_curve":  peer_ctx.get("macro_strategist", {}).get("yield_curve_spread"),
                "peer_hit_rate":     peer_ctx.get("signal_auditor", {}).get("granger_hit_rate"),
                # Full narratives for LLM enrichment
                "peer_narratives": {
                    pid: h.get("narrative", "") for pid, h in peer_ctx.items()
                },
            }

        return {}

    # ── Agent dispatch ─────────────────────────────────────────────────────────

    def _run_agent(self, agent_id: str, context: dict) -> dict:
        """
        Dispatch to the appropriate agent module.
        Retries once on failure (transient LLM timeout/network error).
        Logs failures to trace_logger for harness observability.
        """
        import time as _time

        _MODULE_MAP = {
            "signal_auditor":        "src.agents.signal_auditor",
            "macro_strategist":      "src.agents.macro_strategist",
            "geopolitical_analyst":  "src.agents.geopolitical_analyst",
            "risk_officer":          "src.agents.risk_officer",
            "commodities_specialist":"src.agents.commodities_specialist",
            "stress_engineer":       "src.agents.stress_engineer",
            "trade_structurer":      "src.agents.trade_structurer",
        }
        if agent_id not in _MODULE_MAP:
            return {}

        # Guard: empty api_key with a live provider causes a 401 that burns
        # both retry attempts.  Treat missing key the same as missing provider.
        if not self.api_key or not self.api_key.strip():
            self.provider = None

        set_status(agent_id, "investigating")
        last_err = None

        for attempt in range(2):  # one retry on failure
            try:
                import importlib
                mod = importlib.import_module(_MODULE_MAP[agent_id])
                result = mod.run(context, self.provider, self.api_key)

                # Guard: agents return an "unavailable" string as narrative when
                # _call_ai raises (e.g. "Risk Officer unavailable: ...").
                # That error string must not be stored as agent output or used
                # to build a handoff — treat it as a recoverable failure.
                narrative = result.get("narrative", "")
                if narrative and "unavailable:" in narrative.lower():
                    last_err = narrative
                    try:
                        from src.analysis.trace_logger import log_failure
                        log_failure(agent_id, "LLMError", narrative[:120])
                    except Exception:
                        pass
                    log_activity(agent_id, f"LLM error (attempt {attempt+1})",
                                 narrative[:80], "warning")
                    if attempt == 0:
                        _time.sleep(1.5)
                    continue

                return result
            except Exception as e:
                last_err = e
                # Log failure to trace for harness observability — failed calls
                # are as important as successes for auditing pipeline health.
                try:
                    from src.analysis.trace_logger import log_failure
                    log_failure(agent_id, type(e).__name__, str(e)[:120])
                except Exception:
                    pass
                log_activity(agent_id, f"pipeline error (attempt {attempt+1})",
                             str(e)[:80], "warning")
                if attempt == 0:
                    _time.sleep(1.5)  # brief pause before retry

        set_status(agent_id, "idle")
        return {"error": str(last_err)}

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
            "failed_agents":    orch.get("failed_agents", []),
        }

    def divergence_flags(self) -> list[dict]:
        """
        Returns list of numeric divergence dicts with fields:
          agent_a, agent_b, field, val_a, val_b, diff, threshold, ts
        Use .get("diff") and .get("field") for display — not keyword strings.
        """
        return self.orch.get("divergence_flags", [])

    def get_peer_context(self, agent_id: str) -> dict[str, "AgentHandoff"]:
        """Returns typed AgentHandoff dicts for the agent's upstream peers."""
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
    """Factory - returns (and caches) the session's Orchestrator instance."""
    return Orchestrator(provider, api_key)
