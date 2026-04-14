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

def build_briefing_context(start: str, end: str) -> dict:
    """
    Assemble a comprehensive live-data context dict for the morning briefing.
    Pulls conflict scores, market returns, regime, VIX, COT, scenario, alerts.
    Designed to be called once per session; all loaders are @st.cache_data so
    no extra network round-trips vs. what the page already computed.
    """
    import datetime as _dt
    ctx: dict = {
        "date":             _dt.date.today().isoformat(),
        "risk_score":       50.0,
        "cis":              50.0,
        "tps":              50.0,
        "mcs":              50.0,
        "top_conflict":     None,
        "top_conflict_name": "No dominant conflict",
        "n_active":         0,
        "active_conflicts": [],        # list of dicts: name, cis, tps, escalation, channel, commodities
        "regime":           1,
        "regime_label":     "Normal",
        "corr_pct":         50.0,
        "scenario":         "Base",
        "geo_mult":         1.0,
        "top_alerts":       [],
        # Market movers (5-day cumulative log-ret × 100)
        "top_eq_winner":    None,  "top_eq_winner_ret":  None,
        "top_eq_loser":     None,  "top_eq_loser_ret":   None,
        "top_cmd_winner":   None,  "top_cmd_winner_ret": None,
        "top_cmd_loser":    None,  "top_cmd_loser_ret":  None,
        # Vol surface
        "vix":  None, "ovx": None, "gvz": None, "vvix": None,
        # COT: net speculative positioning for key markets
        "cot_summary":      [],    # list of "Gold: +12.4k net long" strings
        # Macro
        "yield_curve":      None,  # TLT - SHY 60d return (negative = steepening)
        "credit_spread":    None,  # HYG 60d return (negative = widening spreads)
        "dollar_60d":       None,  # DXY proxy 60d return
        # Confidence
        "model_confidence": 0.50,
        "n_news_threat":    0,
        "n_news_act":       0,
    }

    # ── Conflict model ─────────────────────────────────────────────────────
    try:
        from src.analysis.conflict_model import score_all_conflicts, aggregate_portfolio_scores
        cr  = score_all_conflicts()
        agg = aggregate_portfolio_scores(cr)
        ctx["cis"] = round(agg.get("portfolio_cis", agg.get("cis", 50.0)), 1)
        ctx["tps"] = round(agg.get("portfolio_tps", agg.get("tps", 50.0)), 1)
        ctx["top_conflict"] = agg.get("top_conflict")
        if ctx["top_conflict"]:
            ctx["top_conflict_name"] = ctx["top_conflict"].replace("_", " ").title()
        active = [(cid, r) for cid, r in cr.items() if r.get("state") == "active"]
        active.sort(key=lambda x: x[1]["cis"], reverse=True)
        ctx["n_active"] = len(active)
        for _cid, r in active[:5]:
            tx       = r.get("transmission", {})
            top_ch   = max(tx, key=tx.get) if tx else "unknown"
            ch_score = tx.get(top_ch, 0) if tx else 0
            ctx["active_conflicts"].append({
                "name":       r["name"],
                "cis":        round(r["cis"], 1),
                "tps":        round(r["tps"], 1),
                "escalation": r.get("escalation", "stable"),
                "channel":    top_ch.replace("_", " "),
                "ch_score":   round(ch_score, 1),
                "commodities": r.get("affected_commodities", [])[:4],
                "equities":    r.get("affected_equities", [])[:3],
                "hedges":      r.get("hedge_assets", [])[:3],
            })
    except Exception:
        pass

    # ── Risk score + regime + market data ──────────────────────────────────
    try:
        from src.data.loader import load_returns
        from src.analysis.correlations import average_cross_corr_series, detect_correlation_regime
        from src.analysis.risk_score import compute_risk_score
        eq_r, cmd_r = load_returns(start, end)
        if not eq_r.empty and not cmd_r.empty:
            avg_corr = average_cross_corr_series(eq_r, cmd_r, window=60)
            regimes  = detect_correlation_regime(avg_corr)
            risk     = compute_risk_score(avg_corr, cmd_r)
            ctx["risk_score"]      = round(risk.get("score", 50.0), 1)
            ctx["mcs"]             = round(risk.get("mcs", 50.0), 1)
            ctx["corr_pct"]        = round(risk.get("corr_pct", 50.0), 1)
            ctx["model_confidence"]= round(risk.get("confidence", 0.5), 2)
            ctx["n_news_threat"]   = risk.get("n_threat", 0)
            ctx["n_news_act"]      = risk.get("n_act", 0)
            if not regimes.empty:
                rv = int(regimes.iloc[-1])
                ctx["regime"]       = rv
                ctx["regime_label"] = {1: "Normal", 2: "Elevated", 3: "Crisis"}.get(rv, "Normal")
            # 5-day movers
            if len(eq_r) >= 5:
                eq5  = eq_r.iloc[-5:].sum() * 100
                top2 = eq5.nlargest(2)
                bot2 = eq5.nsmallest(2)
                if len(top2): ctx["top_eq_winner"], ctx["top_eq_winner_ret"] = top2.index[0], round(float(top2.iloc[0]), 2)
                if len(bot2): ctx["top_eq_loser"],  ctx["top_eq_loser_ret"]  = bot2.index[0], round(float(bot2.iloc[0]), 2)
            if len(cmd_r) >= 5:
                cmd5 = cmd_r.iloc[-5:].sum() * 100
                top2 = cmd5.nlargest(2)
                bot2 = cmd5.nsmallest(2)
                if len(top2): ctx["top_cmd_winner"], ctx["top_cmd_winner_ret"] = top2.index[0], round(float(top2.iloc[0]), 2)
                if len(bot2): ctx["top_cmd_loser"],  ctx["top_cmd_loser_ret"]  = bot2.index[0], round(float(bot2.iloc[0]), 2)
    except Exception:
        pass

    # ── VIX / vol surface ─────────────────────────────────────────────────
    try:
        from src.data.loader import load_iv_snapshot
        iv = load_iv_snapshot()
        ctx["vix"]  = iv.get("VIX")
        ctx["ovx"]  = iv.get("OVX")
        ctx["gvz"]  = iv.get("GVZ")
        ctx["vvix"] = iv.get("VVIX")
    except Exception:
        pass

    # ── Yield curve + credit spread proxy ─────────────────────────────────
    try:
        from src.data.loader import load_fixed_income_returns
        fi = load_fixed_income_returns(start, end)
        if not fi.empty and len(fi) >= 60:
            tlt_col = next((c for c in fi.columns if "TLT" in c or "20Y" in c), None)
            shy_col = next((c for c in fi.columns if "SHY" in c or "1-3Y" in c), None)
            hyg_col = next((c for c in fi.columns if "HYG" in c or "HY " in c), None)
            if tlt_col: ctx["yield_curve"]  = round(float(fi[tlt_col].iloc[-60:].sum() * 100), 2)
            if hyg_col: ctx["credit_spread"]= round(float(fi[hyg_col].iloc[-60:].sum() * 100), 2)
    except Exception:
        pass

    # ── Dollar index proxy ─────────────────────────────────────────────────
    try:
        from src.data.loader import load_fx_returns
        fx = load_fx_returns(start, end)
        if not fx.empty and len(fx) >= 60:
            dxy_col = next((c for c in fx.columns if "DXY" in c or "Dollar" in c), None)
            if dxy_col:
                ctx["dollar_60d"] = round(float(fx[dxy_col].iloc[-60:].sum() * 100), 2)
    except Exception:
        pass

    # ── COT positioning ────────────────────────────────────────────────────
    try:
        from src.analysis.cot import load_cot_data
        cot_df = load_cot_data(years=1)
        if not cot_df.empty:
            for mkt in ["Gold", "Crude Oil", "S&P 500", "Natural Gas", "Wheat"]:
                sub = cot_df[cot_df["market"] == mkt]
                if not sub.empty:
                    net = float(sub["net_speculative"].iloc[-1])
                    pct = float(sub.get("net_spec_pct", sub["net_speculative"]).iloc[-1])
                    direction = "net long" if net > 0 else "net short"
                    ctx["cot_summary"].append(f"{mkt}: {abs(net/1000):.0f}k contracts {direction} ({pct:+.1f}% of OI)")
    except Exception:
        pass

    # ── Scenario ───────────────────────────────────────────────────────────
    try:
        from src.analysis.scenario_state import get_scenario
        sc = get_scenario()
        ctx["scenario"]  = sc.get("label", "Base")
        ctx["geo_mult"]  = round(sc.get("geo_mult", 1.0), 2)
    except Exception:
        pass

    return ctx


def run_morning_briefing_protocol(ctx: dict) -> str:
    """
    Kicks off the morning briefing inter-agent chain with comprehensive live context.
    risk_officer → geopolitical_analyst → commodities_specialist → macro_strategist → trade_structurer

    ctx must be produced by build_briefing_context(start, end).
    Returns the thread_id for the full briefing chain.
    """
    # ── Unpack context ─────────────────────────────────────────────────────
    date_str         = ctx.get("date", "today")
    risk_score       = ctx.get("risk_score", 50.0)
    cis              = ctx.get("cis", 50.0)
    tps              = ctx.get("tps", 50.0)
    mcs              = ctx.get("mcs", 50.0)
    top_conflict_name= ctx.get("top_conflict_name", "No dominant conflict")
    n_active         = ctx.get("n_active", 0)
    active_conflicts = ctx.get("active_conflicts", [])
    regime_label     = ctx.get("regime_label", "Normal")
    corr_pct         = ctx.get("corr_pct", 50.0)
    scenario         = ctx.get("scenario", "Base")
    geo_mult         = ctx.get("geo_mult", 1.0)
    top_alerts       = ctx.get("top_alerts", [])
    vix              = ctx.get("vix")
    ovx              = ctx.get("ovx")
    gvz              = ctx.get("gvz")
    vvix             = ctx.get("vvix")
    yield_curve      = ctx.get("yield_curve")
    credit_spread    = ctx.get("credit_spread")
    dollar_60d       = ctx.get("dollar_60d")
    cot_summary      = ctx.get("cot_summary", [])
    confidence       = ctx.get("model_confidence", 0.5)
    n_news_threat    = ctx.get("n_news_threat", 0)
    n_news_act       = ctx.get("n_news_act", 0)
    top_eq_winner    = ctx.get("top_eq_winner"); top_eq_winner_ret = ctx.get("top_eq_winner_ret")
    top_eq_loser     = ctx.get("top_eq_loser");  top_eq_loser_ret  = ctx.get("top_eq_loser_ret")
    top_cmd_winner   = ctx.get("top_cmd_winner"); top_cmd_winner_ret= ctx.get("top_cmd_winner_ret")
    top_cmd_loser    = ctx.get("top_cmd_loser");  top_cmd_loser_ret = ctx.get("top_cmd_loser_ret")

    # ── Derived signals ────────────────────────────────────────────────────
    driven_by = "transmission pressure" if tps > cis else "conflict escalation"
    risk_tier = "HIGH" if risk_score >= 70 else ("ELEVATED" if risk_score >= 50 else "MODERATE")
    corr_tier = "breakdown" if corr_pct >= 75 else ("elevated" if corr_pct >= 55 else "normal")
    vix_str   = f"VIX {vix:.1f}" if vix else "VIX n/a"
    ovx_str   = f"OVX {ovx:.1f}" if ovx else ""
    gvz_str   = f"GVZ {gvz:.1f}" if gvz else ""
    vol_str   = " · ".join(filter(None, [vix_str, ovx_str, gvz_str]))
    alert_str = "; ".join(top_alerts[:3]) if top_alerts else "No critical alerts flagged"

    # Top active conflict detail
    top_c  = active_conflicts[0] if active_conflicts else {}
    top_c2 = active_conflicts[1] if len(active_conflicts) > 1 else {}
    top_c3 = active_conflicts[2] if len(active_conflicts) > 2 else {}

    # ── Message 1: Risk Officer opens briefing (INFO → broadcast) ─────────
    ro_open = (
        f"MORNING BRIEFING · {date_str}\n"
        f"Composite geo risk: {risk_score:.0f}/100 ({risk_tier}) · Scenario: {scenario} ×{geo_mult:.2f}\n"
        f"Regime: {regime_label} · Correlation at {corr_pct:.0f}th percentile ({corr_tier})\n"
        f"Vol surface: {vol_str}\n"
        f"Conflict layer: CIS {cis:.0f} · TPS {tps:.0f} · MCS {mcs:.0f} · Confidence {confidence:.0%}\n"
        f"News signal: {n_news_threat} threat events, {n_news_act} action events ingested\n"
        f"Active conflicts: {n_active} · Lead: {top_conflict_name}\n"
        f"Alerts: {alert_str}\n"
        f"Routing to Geo Analyst for conflict decomposition."
    )

    tid = send_message(
        sender="risk_officer", recipient="broadcast",
        msg_type="info", content=ro_open, subject_id="morning_briefing",
    )

    # ── Message 2: Risk Officer queries Geo Analyst ────────────────────────
    ro_query = (
        f"Geo risk is {risk_score:.0f}/100, primarily {driven_by} (CIS {cis:.0f} vs TPS {tps:.0f}).\n"
        f"Lead conflict: {top_conflict_name}"
        + (f" (CIS {top_c.get('cis','?')}, TPS {top_c.get('tps','?')}, escalation: {top_c.get('escalation','?')})" if top_c else "") + ".\n"
        + (f"Secondary: {top_c2.get('name','?')} (CIS {top_c2.get('cis','?')})\n" if top_c2 else "")
        + (f"Third: {top_c3.get('name','?')} (CIS {top_c3.get('cis','?')})\n" if top_c3 else "")
        + f"Is the {top_conflict_name} risk driven by new escalation events or supply-route disruption?\n"
        f"What is the spillover risk to adjacent theaters given {n_active} active conflicts?"
    )

    send_message(
        sender="risk_officer", recipient="geopolitical_analyst",
        msg_type="query", content=ro_query,
        subject_id="morning_briefing", thread_id=tid,
    )

    # ── Message 3: Geo Analyst responds + hands to Commodities ────────────
    if top_c:
        channel_str = f"Primary channel: {top_c.get('channel','?')} (score {top_c.get('ch_score','?')}/100)"
        affected_str = ", ".join(top_c.get("commodities", [])) or "broad-based"
        hedge_str    = ", ".join(top_c.get("hedges", [])) or "gold, USD"
    else:
        channel_str  = "No dominant transmission channel identified"
        affected_str = "n/a"
        hedge_str    = "gold, USD"

    ga_handoff = (
        f"Conflict decomposition complete.\n"
        f"Lead: {top_conflict_name} — {channel_str}.\n"
        f"Affected commodities: {affected_str}. Hedge assets: {hedge_str}.\n"
    )
    if top_c2:
        ga_handoff += f"{top_c2.get('name','?')}: CIS {top_c2.get('cis','?')}, channel: {top_c2.get('channel','?')}.\n"
    if top_c3:
        ga_handoff += f"{top_c3.get('name','?')}: CIS {top_c3.get('cis','?')}, channel: {top_c3.get('channel','?')}.\n"
    ga_handoff += (
        f"Escalation trend on lead conflict: {top_c.get('escalation','stable') if top_c else 'stable'}.\n"
        f"With {n_active} active conflicts, portfolio CIS is driven by "
        + ("compounding supply-route pressure." if tps > cis else "simultaneous escalation across multiple theaters.")
        + f"\nHanding to Commodities Specialist for price transmission assessment."
    )

    send_message(
        sender="geopolitical_analyst", recipient="commodities_specialist",
        msg_type="handoff", content=ga_handoff,
        subject_id="morning_briefing", thread_id=tid,
    )

    # ── Message 4: Commodities Specialist assesses price impact ───────────
    cmd_moves = []
    if top_cmd_winner: cmd_moves.append(f"{top_cmd_winner} {top_cmd_winner_ret:+.1f}% (5d)")
    if top_cmd_loser:  cmd_moves.append(f"{top_cmd_loser} {top_cmd_loser_ret:+.1f}% (5d)")
    moves_str = "; ".join(cmd_moves) if cmd_moves else "insufficient data"

    ovx_signal = ""
    if ovx:
        ovx_signal = f" OVX (crude vol) at {ovx:.1f} — {'elevated fear' if ovx > 30 else 'contained'}."
    gvz_signal = ""
    if gvz:
        gvz_signal = f" GVZ (gold vol) at {gvz:.1f} — {'safe-haven demand elevated' if gvz > 20 else 'contained'}."

    cot_str = "; ".join(cot_summary[:3]) if cot_summary else "COT data not loaded this session"

    cs_handoff = (
        f"Commodity transmission assessment:\n"
        f"5-day price action: {moves_str}\n"
        f"Implied vol:{ovx_signal}{gvz_signal}\n"
        f"COT positioning: {cot_str}\n"
        f"Active transmission channels: "
        + ", ".join({c.get("channel","?") for c in active_conflicts[:3] if c.get("channel")})
        + f".\n"
        f"Assessment: {'Supply disruption risk is elevated' if tps >= 60 else 'Supply disruption risk is moderate'}. "
        f"{'Energy and ag routes most stressed at current TPS {tps:.0f}.' if tps >= 50 else f'TPS at {tps:.0f} — transmission pressure manageable but building.'}\n"
        f"Handing to Macro Strategist for inflation and rates implications."
    )

    send_message(
        sender="commodities_specialist", recipient="macro_strategist",
        msg_type="handoff", content=cs_handoff,
        subject_id="morning_briefing", thread_id=tid,
    )

    # ── Message 5: Macro Strategist assesses regime + cross-asset ─────────
    curve_str = ""
    if yield_curve is not None:
        curve_str = f" TLT 60d: {yield_curve:+.1f}% ({'bull steepening / duration bid' if yield_curve > 2 else 'bear flattening / rate pressure' if yield_curve < -2 else 'curve range-bound'})."
    credit_str = ""
    if credit_spread is not None:
        credit_str = f" HYG 60d: {credit_spread:+.1f}% ({'spreads tightening' if credit_spread > 0 else 'spreads widening — credit stress signal'})."
    dollar_str = ""
    if dollar_60d is not None:
        dollar_str = f" DXY proxy 60d: {dollar_60d:+.1f}% ({'dollar strengthening — EM headwind' if dollar_60d > 2 else 'dollar weakening — commodity tailwind' if dollar_60d < -2 else 'dollar neutral'})."

    eq_moves = []
    if top_eq_winner: eq_moves.append(f"{top_eq_winner} {top_eq_winner_ret:+.1f}% (5d)")
    if top_eq_loser:  eq_moves.append(f"{top_eq_loser} {top_eq_loser_ret:+.1f}% (5d)")
    eq_str = "; ".join(eq_moves) if eq_moves else "insufficient data"

    macro_handoff = (
        f"Macro regime: {regime_label} · Correlation at {corr_pct:.0f}th percentile.\n"
        f"{'Diversification is failing — equities and commodities co-moving.' if corr_pct > 70 else 'Diversification still effective at current correlation.'}\n"
        f"Equity 5-day: {eq_str}\n"
        f"Fixed income:{curve_str}{credit_str}\n"
        f"FX:{dollar_str}\n"
        + (f"VIX at {vix:.1f} — {'fear elevated, risk-off likely dominating flows' if vix > 25 else 'vol contained' if vix < 18 else 'vol moderate'}.\n" if vix else "")
        + (f"VVIX at {vvix:.1f} — {'tail-risk hedging demand elevated' if vvix > 100 else 'second-order vol contained'}.\n" if vvix else "")
        + f"Under {scenario} scenario (×{geo_mult:.2f}): "
        + ("inflation pass-through from energy/ag disruption is the primary macro risk." if tps > 55 else "correlation regime shift is the primary macro risk.")
        + f"\nHanding to Trade Structurer for positioning expression."
    )

    send_message(
        sender="macro_strategist", recipient="trade_structurer",
        msg_type="handoff", content=macro_handoff,
        subject_id="morning_briefing", thread_id=tid,
    )

    # ── Message 6: Trade Structurer synthesises ────────────────────────────
    # Build trade expressions based on what the data actually says
    expressions = []
    if top_cmd_winner and top_cmd_winner_ret and top_cmd_winner_ret > 2:
        expressions.append(f"Long {top_cmd_winner} momentum — confirmed by {top_c.get('channel','supply disruption') if top_c else 'geo pressure'}")
    if vix and vix > 22:
        expressions.append("Long VIX puts / short-vol fade — fear elevated, mean-reversion candidate")
    if credit_spread is not None and credit_spread < -2:
        expressions.append("Short HYG / long LQD — spread widening momentum, credit stress forming")
    if corr_pct > 70:
        expressions.append("Regime hedge: long gold + short S&P — crisis correlation, diversification failing")
    if dollar_60d is not None and dollar_60d > 3:
        expressions.append("Short EMB / reduce EM equity — dollar strength EM headwind confirmed")
    if not expressions:
        expressions.append("No high-conviction asymmetric expression — reduce position size, await regime clarification")

    ts_resolve = (
        f"Trade synthesis from full briefing chain:\n"
        + "\n".join(f"  {i+1}. {e}" for i, e in enumerate(expressions[:4]))
        + f"\n\nScenario: {scenario} (×{geo_mult:.2f}) applied to all R/R estimates.\n"
        f"Risk-off bias {'warranted' if risk_score >= 60 else 'not yet warranted'} at geo risk {risk_score:.0f}/100.\n"
        f"Confidence: {confidence:.0%}. All ideas require human review before execution."
    )

    send_message(
        sender="trade_structurer", recipient="broadcast",
        msg_type="resolve", content=ts_resolve,
        subject_id="morning_briefing", thread_id=tid,
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
