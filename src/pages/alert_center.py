"""
Alert Center - the terminal watches your exposure and tells you when it matters.

The terminal already computes the signals (market-stress score, correlation regime,
commodity volatility, COT positioning, country exposure, GDELT conflict media,
EIA inventory) via src/analysis/proactive_alerts.compute_alerts. That engine
returns structured Alert objects but has no notion of the user's money. This page
adds the missing piece: the user supplies an exposure profile (a notional and a
coarse allocation across Equities / Energy / Metals / Agriculture / Rates-FX), and
every fired alert is translated into an estimated dollar impact on the relevant
sleeve, ranked, and composed into a briefing. Delivery is in-terminal for now;
email / Slack / SMS push is on the roadmap (it needs accounts + a channel).
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd
import streamlit as st

from src.ui.shared import (_page_header, _page_footer, _page_intro,
                           _definition_block)

_M = "font-family:'JetBrains Mono',monospace;"
_GOLD, _GRN, _ORG, _RED = "#CFB991", "#27ae60", "#e67e22", "#c0392b"
_TXT, _MUT, _FNT, _BRD = "#e8e9ed", "#8890a1", "#555960", "#1e1e1e"

_BUCKETS = ["Equities", "Energy", "Metals", "Agriculture", "Rates/FX"]
_ENERGY = {"WTI Crude Oil", "Brent Crude", "Natural Gas", "Gasoline (RBOB)",
           "Heating Oil", "Crude Oil", "Gasoline", "Brent", "WTI"}
_METALS = {"Gold", "Silver", "Platinum", "Palladium", "Copper", "Aluminum", "Nickel"}
_AG = {"Wheat", "Corn", "Soybeans", "Soybean Oil", "Sugar #11", "Sugar",
       "Coffee", "Cotton"}

_SEV = {"critical": (_RED, "CRITICAL"), "warning": (_ORG, "WARNING"),
        "info": (_MUT, "INFO")}
_CAT = {"stress": "Market Stress", "regime": "Regime", "correlation": "Correlation",
        "volatility": "Volatility", "cot": "Positioning (COT)",
        "country_exposure": "Country Exposure", "conflict": "Conflict (GDELT)",
        "supply": "Supply (EIA)"}


def _money(x: float) -> str:
    a = abs(x)
    if a >= 1e9:  return f'${x/1e9:.2f}B'
    if a >= 1e6:  return f'${x/1e6:.2f}M'
    if a >= 1e3:  return f'${x/1e3:.0f}K'
    return f'${x:,.0f}'


def _commodity_bucket(name: str) -> str:
    if name in _ENERGY:  return "Energy"
    if name in _METALS:  return "Metals"
    if name in _AG:      return "Agriculture"
    return "Energy"


@st.cache_data(show_spinner=False, ttl=900, max_entries=4)
def _alert_inputs(start: str, end: str) -> dict | None:
    """Load returns, compute the alert-engine inputs, run compute_alerts, and
    return plain dicts (cache-friendly). None if market data is unavailable."""
    try:
        from src.data.loader import load_returns
        from src.analysis.correlations import (average_cross_corr_series,
                                               detect_correlation_regime)
        from src.analysis.risk_score import compute_risk_score, risk_score_history
        from src.analysis.proactive_alerts import compute_alerts
    except Exception:
        return None

    eq_r, cmd_r = load_returns(start, end)
    if eq_r is None or cmd_r is None or eq_r.empty or cmd_r.empty:
        return None

    avg_corr = average_cross_corr_series(eq_r, cmd_r, window=60)
    regimes = detect_correlation_regime(avg_corr)
    try:
        risk = compute_risk_score(avg_corr, cmd_r, eq_r=eq_r)
        rscore = float(risk.get("score", 0.0)) if isinstance(risk, dict) else float(risk)
    except Exception:
        rscore = 0.0
    try:
        rhist = risk_score_history(avg_corr, cmd_r, eq_r)
        if not isinstance(rhist, pd.Series):
            rhist = pd.Series(dtype=float)
    except Exception:
        rhist = pd.Series(dtype=float)
    try:
        from src.analysis.cot import load_cot_data
        cot_df = load_cot_data(years=2)
    except Exception:
        cot_df = pd.DataFrame()

    try:
        alerts = compute_alerts(eq_r=eq_r, cmd_r=cmd_r, avg_corr=avg_corr,
                                regimes=regimes, risk_score=rscore,
                                risk_history=rhist, cot_df=cot_df)
    except Exception:
        alerts = []

    ctx = {
        "risk_score": rscore,
        "regime": int(regimes.iloc[-1]) if not regimes.empty else 1,
        "avg_corr": float(avg_corr.iloc[-1]) if not avg_corr.empty else 0.0,
        "asof": str(cmd_r.index[-1].date()) if len(cmd_r) else "",
    }
    out = [{"severity": a.severity, "category": a.category, "title": a.title,
            "body": a.body, "page_hint": a.page_hint, "data": dict(a.data or {})}
           for a in alerts]
    return {"alerts": out, "ctx": ctx}


def _dollarize(alert: dict, notional: float, w: dict) -> tuple[str, float, float, str]:
    """Map an alert to (bucket, adverse_move_fraction, dollar_impact, basis_text).
    Every figure is an explicit estimate; the basis text states the assumption."""
    cat, d, sev = alert["category"], alert.get("data", {}), alert["severity"]
    _rn = ["Decorrelated", "Normal", "Elevated", "Crisis"]

    if cat == "stress":
        rs = float(d.get("risk_score", 55.0))
        bucket = "Equities"
        move = 0.12 if rs >= 65 else 0.06 if rs >= 50 else 0.03
        basis = f"stress {rs:.0f}/100, est. {move:.0%} 30d equity drawdown"
    elif cat == "regime":
        cr = int(d.get("current_regime", 2))
        bucket = "Equities"
        move = {3: 0.15, 2: 0.08, 1: 0.03, 0: 0.02}.get(cr, 0.05)
        basis = f"{_rn[cr]} regime, est. {move:.0%} equity move"
    elif cat == "correlation":
        bucket = "Equities"
        move = 0.03
        basis = "diversification loss, nominal 3% equity proxy (not directional)"
    elif cat == "volatility":
        nm = str(d.get("commodity", ""))
        bucket = _commodity_bucket(nm)
        vol = float(d.get("vol_60d", 30.0))
        move = float(min(vol / 100.0 / np.sqrt(12.0), 0.25))
        basis = f"{nm} 60d vol {vol:.0f}%, est. 1-month 1σ {move:.0%}"
    elif cat == "cot":
        nm = str(d.get("market", ""))
        bucket = _commodity_bucket(nm)
        move = 0.12
        basis = f"{nm} crowded positioning, est. {move:.0%} reversal"
    elif cat == "country_exposure":
        bucket = "Equities"
        move = 0.06
        basis = "elevated country equity exposure, est. 6% index move"
    elif cat == "conflict":
        tr = float(d.get("volume_trend", 0.30))
        bucket = "Energy"
        move = float(min(0.04 + tr * 0.10, 0.15))
        basis = f"media surge {tr:+.0%}, est. {move:.0%} energy repricing"
    elif cat == "supply":
        bucket = "Energy"
        move = 0.10 if sev == "critical" else 0.06
        basis = f"inventory shortfall, est. {move:.0%} energy move"
    else:
        bucket = "Equities"
        move = 0.03
        basis = "general market signal, nominal 3% proxy"

    dollar = float(notional) * float(w.get(bucket, 0.0)) * move
    return bucket, move, dollar, basis


def page_alert_center(start: str, end: str, fred_key: str = "") -> None:
    _page_header("Alert Center",
                 "Watched signals, ranked and priced against your exposure")
    _page_intro(
        "The terminal already scores market stress, correlation regime, commodity "
        "volatility, COT positioning, country exposure, conflict media and energy "
        "inventory. This page runs that whole battery against live data and does the "
        "one thing a signal feed usually will not: it translates each fired alert "
        "into an <strong>estimated dollar impact on your book</strong>. Give it an "
        "exposure profile, and it ranks what is firing by how much it can cost you, "
        "then composes a briefing you can take out of the terminal."
    )
    _definition_block(
        "How the dollar figures work",
        "You supply a notional and a coarse allocation across five sleeves. Each alert "
        "maps to the sleeve it hits (a natural-gas vol spike hits Energy, a regime flip "
        "hits Equities) and an estimated adverse move drawn from the signal itself "
        "(the stress level, the regime, the realised vol, the positioning extreme). "
        "Dollar impact is notional &times; sleeve weight &times; that move. These are "
        "sizing estimates, deliberately simple and clearly labelled, not a joint VaR. "
        "Delivery is in-terminal; push to email / Slack is on the roadmap."
    )

    # ── Exposure profile ─────────────────────────────────────────────────────
    st.markdown(
        f'<div style="{_M}font-size:.62rem;font-weight:700;letter-spacing:.12em;'
        f'color:#e8e9ed;margin:.3rem 0 .1rem">YOUR EXPOSURE PROFILE</div>',
        unsafe_allow_html=True)
    _cN, _c1, _c2, _c3, _c4, _c5 = st.columns([1.3, 1, 1, 1, 1, 1], gap="small")
    with _cN:
        notl_m = st.number_input("Notional ($M)", min_value=1.0, value=100.0,
                                 step=10.0, key="_alc_notl")
    with _c1:
        w_eq = st.number_input("Equities %", 0.0, 100.0, 50.0, 5.0, key="_alc_eq")
    with _c2:
        w_en = st.number_input("Energy %", 0.0, 100.0, 15.0, 5.0, key="_alc_en")
    with _c3:
        w_me = st.number_input("Metals %", 0.0, 100.0, 10.0, 5.0, key="_alc_me")
    with _c4:
        w_ag = st.number_input("Agriculture %", 0.0, 100.0, 10.0, 5.0, key="_alc_ag")
    with _c5:
        w_rf = st.number_input("Rates/FX %", 0.0, 100.0, 15.0, 5.0, key="_alc_rf")

    notional = float(notl_m) * 1e6
    raw = {"Equities": w_eq, "Energy": w_en, "Metals": w_me,
           "Agriculture": w_ag, "Rates/FX": w_rf}
    tot = sum(raw.values()) or 1.0
    w = {k: v / tot for k, v in raw.items()}

    _cf1, _cf2 = st.columns([1.4, 1], gap="medium")
    with _cf1:
        sev_filter = st.radio("Show", ["All", "Warning and above", "Critical only"],
                             horizontal=True, key="_alc_sev")
    with _cf2:
        min_k = st.number_input("Hide impact below ($K)", 0.0, value=0.0, step=50.0,
                               key="_alc_mink")

    with st.spinner("Scanning live signals..."):
        res = _alert_inputs(start, end)

    if res is None:
        st.error("Could not load market data to evaluate alerts. Check the connection "
                 "and try again.")
        _page_footer()
        return

    alerts, ctx = res["alerts"], res["ctx"]

    # Dollarize + filter
    _sev_min = {"All": 0, "Warning and above": 1, "Critical only": 2}[sev_filter]
    _sev_rank = {"critical": 2, "warning": 1, "info": 0}
    priced = []
    for a in alerts:
        bucket, move, dollar, basis = _dollarize(a, notional, w)
        priced.append({**a, "bucket": bucket, "move": move, "dollar": dollar,
                       "basis": basis})
    shown = [p for p in priced
             if _sev_rank.get(p["severity"], 0) >= _sev_min
             and p["dollar"] >= float(min_k) * 1e3]
    # rank: severity first, then dollar impact
    shown.sort(key=lambda p: (_sev_rank.get(p["severity"], 0), p["dollar"]),
               reverse=True)

    n_crit = sum(1 for p in priced if p["severity"] == "critical")
    n_warn = sum(1 for p in priced if p["severity"] == "warning")
    n_info = sum(1 for p in priced if p["severity"] == "info")
    gross = sum(p["dollar"] for p in shown)
    top = shown[0] if shown else None

    # ── Summary strip ────────────────────────────────────────────────────────
    def _stat(lbl, val, sub, col=_TXT, last=False):
        br = "" if last else f"border-right:1px solid {_BRD}"
        return (f'<div style="flex:1;padding:.45rem .7rem;{br}">'
                f'<div style="{_M}font-size:.5rem;letter-spacing:.1em;color:{_MUT}">{lbl}</div>'
                f'<div style="{_M}font-size:1.05rem;font-weight:700;color:{col};margin:1px 0">{val}</div>'
                f'<div style="{_M}font-size:.48rem;color:{_FNT}">{sub}</div></div>')

    _asof = ctx.get("asof", "")
    _hdr_col = _RED if n_crit else _ORG if n_warn else _GRN
    st.markdown(
        f'<div style="border:1px solid {_BRD};background:#0a0a0a;margin:.3rem 0 .2rem">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
        f'padding:.4rem .8rem;border-bottom:1px solid {_BRD}">'
        f'<span style="{_M}font-size:.6rem;font-weight:700;letter-spacing:.14em;color:{_TXT}">'
        f'ALERT SUMMARY</span><span style="{_M}font-size:.5rem;color:{_MUT}">'
        f'live signals &middot; data to {_asof} &middot; {_money(notional)} exposure'
        f'<span style="background:{_hdr_col};color:#000;font-weight:700;padding:1px 7px;'
        f'margin-left:6px;letter-spacing:.08em">{"ACTION" if n_crit else "MONITOR" if n_warn else "CLEAR"}</span>'
        f'</span></div>'
        f'<div style="display:flex;border-bottom:1px solid {_BRD}">'
        + _stat("ALERTS FIRING", f'{len(priced)}',
                f'{n_crit} critical &middot; {n_warn} warning &middot; {n_info} info', _hdr_col)
        + _stat("SHOWN", f'{len(shown)}', f'after your filters')
        + _stat("GROSS $-AT-RISK", _money(gross),
                'sum of shown, not a joint VaR', _RED if gross else _TXT)
        + _stat("LARGEST SINGLE", _money(top["dollar"]) if top else "$0",
                (top["title"][:34] if top else "nothing firing"),
                _RED if top else _GRN, last=True)
        + '</div></div>',
        unsafe_allow_html=True)

    _rmap = ["Decorrelated", "Normal", "Elevated", "Crisis"]
    st.caption(f"Context: market stress {ctx.get('risk_score',0):.0f}/100, "
               f"{_rmap[int(ctx.get('regime',1))]} regime, 60d avg |corr| "
               f"{ctx.get('avg_corr',0):.3f}.")

    # ── Alert feed ───────────────────────────────────────────────────────────
    if not shown:
        st.success("No alerts above your filters. The watched signals are within "
                   "normal ranges for your exposure.")
    for p in shown:
        col, lbl = _SEV.get(p["severity"], (_MUT, p["severity"].upper()))
        catlbl = _CAT.get(p["category"], p["category"].title())
        st.markdown(
            f'<div style="border:1px solid {_BRD};border-left:3px solid {col};'
            f'background:#0a0a0a;margin:.25rem 0;padding:.5rem .8rem">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:10px">'
            f'<span><span style="background:{col};color:#000;font-weight:700;padding:1px 7px;'
            f'{_M}font-size:.5rem;letter-spacing:.08em">{lbl}</span>'
            f'<span style="{_M}font-size:.5rem;color:{_MUT};margin-left:7px;'
            f'letter-spacing:.08em">{catlbl.upper()}</span>'
            f'<span style="{_M}font-size:.66rem;font-weight:700;color:{_TXT};'
            f'margin-left:9px">{p["title"]}</span></span>'
            f'<span style="text-align:right;white-space:nowrap">'
            f'<span style="{_M}font-size:.92rem;font-weight:700;color:{col}">{_money(p["dollar"])}</span>'
            f'<span style="{_M}font-size:.46rem;color:{_MUT};display:block">'
            f'est. on {p["bucket"]}</span></span></div>'
            f'<div style="{_M}font-size:.56rem;color:#c9ccd4;line-height:1.5;margin-top:5px">{p["body"]}</div>'
            f'<div style="{_M}font-size:.48rem;color:{_FNT};margin-top:4px">'
            f'BASIS: {p["basis"]}  &middot;  drill down: {p["page_hint"].replace("_"," ")}</div>'
            f'</div>', unsafe_allow_html=True)

    # ── Briefing (takeaway artifact) ─────────────────────────────────────────
    _today = _dt.date.today().strftime("%d %b %Y")
    lines = [f"ALERT BRIEFING - {_today}",
             f"Exposure: {_money(notional)} "
             + ", ".join(f"{k} {w[k]*100:.0f}%" for k in _BUCKETS if w[k] > 0) + ".",
             f"Context: stress {ctx.get('risk_score',0):.0f}/100, "
             f"{_rmap[int(ctx.get('regime',1))]} regime, avg |corr| {ctx.get('avg_corr',0):.3f}.",
             f"{len(priced)} alerts firing ({n_crit} critical, {n_warn} warning). "
             f"Gross estimated $-at-risk (shown): {_money(gross)}.",
             ""]
    if shown:
        for i, p in enumerate(shown, 1):
            lines.append(f"{i}. [{p['severity'].upper()}] {p['title']} "
                         f"- est. {_money(p['dollar'])} on {p['bucket']}.")
            lines.append(f"   {p['basis']}.")
    else:
        lines.append("No alerts above the current filters.")
    lines += ["",
              "Dollar figures are simple sizing estimates (notional x sleeve weight x "
              "an estimated adverse move), not a joint VaR. For research and risk "
              "monitoring only; not investment advice."]
    briefing = "\n".join(lines)

    st.markdown(
        f'<div style="border-top:1px solid {_BRD};margin:1rem 0 .4rem;padding-top:.6rem">'
        f'<span style="{_M}font-size:.62rem;font-weight:700;letter-spacing:.1em;'
        f'color:{_TXT}">TAKE THE BRIEFING</span>'
        f'<div style="{_M}font-size:.54rem;color:{_MUT};margin-top:2px">A plain-text '
        f'digest of what is firing and its priced impact. Push to email / Slack / SMS '
        f'is on the roadmap (needs accounts and a delivery channel); for now, take it '
        f'with you.</div></div>', unsafe_allow_html=True)
    _b1, _b2 = st.columns([1, 3], gap="medium")
    with _b1:
        st.download_button("Download Briefing", data=briefing,
                          file_name=f"alert_briefing_{_dt.date.today().isoformat()}.txt",
                          mime="text/plain", width="stretch")
    with st.expander("Preview briefing"):
        st.code(briefing, language="text")

    st.caption("Alert Center runs the terminal's proactive-alert engine and prices "
               "each hit against your exposure. Estimates only, not investment advice.")
    _page_footer()
