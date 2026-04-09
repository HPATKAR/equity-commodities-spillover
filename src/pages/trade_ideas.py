"""
Page 6 - Trade Ideas
Regime-triggered + conflict-driven cross-asset trade cards.
Integrates conflict exposure scoring, scenario-aware payoff tables, QC grading,
filter sidebar, and agent debate threads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date

from src.data.loader import load_returns, load_commodity_prices
from src.data.config import GEOPOLITICAL_EVENTS, PALETTE
from src.analysis.correlations import (
    average_cross_corr_series, detect_correlation_regime, rolling_correlation,
    composite_stress_index,
)
from src.ui.shared import (
    _style_fig, _chart, _page_intro, _thread, _section_note,
    _definition_block, _takeaway_block, _page_conclusion, _page_header, _page_footer,
    _insight_note,
)

_REGIME_NAMES  = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis"}
_REGIME_COLORS = {0: "#2e7d32",      1: "#555960", 2: "#e67e22",  3: "#c0392b"}


# ── Trade idea library ─────────────────────────────────────────────────────
_TRADE_LIBRARY = [
    {
        "regime":    [2, 3],
        "trigger":   "Elevated/Crisis correlation",
        "name":      "Long Gold / Short Eurostoxx 50",
        "rationale": "When cross-asset correlation spikes, equities and commodities sell off together. "
                     "Gold decouples as safe-haven demand absorbs panic flows. "
                     "Short European equities amplifies energy-cost transmission risk.",
        "entry":     "Enter when 60d avg cross-asset corr > 0.45 and DCC(Gold/SPX) < −0.1",
        "exit":      "Close when correlation regime drops back to Normal for 5+ days",
        "risk":      "Correlation snap-back; Central bank intervention can flip gold",
        "assets":    ["Gold", "Eurostoxx 50"],
        "direction": ["Long", "Short"],
        "category":  "Crisis Hedge",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Energy supply shock",
        "name":      "Long Natural Gas / Short Nikkei 225",
        "rationale": "Japan is the world's largest LNG importer. "
                     "Natural gas supply shocks directly impair Japanese manufacturing margins "
                     "and current account. Energy spikes translate to yen weakness and equity underperformance.",
        "entry":     "Enter on Ukraine escalation OR Strait of Hormuz closure signal + NG vol spike",
        "exit":      "Peace signal, supply restoration, or NG price mean-reversion (−20% from entry)",
        "risk":      "BOJ FX intervention; domestic LNG stockpile release",
        "assets":    ["Natural Gas", "Nikkei 225"],
        "direction": ["Long", "Short"],
        "category":  "Geopolitical",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Wheat/food supply disruption (Ukraine War)",
        "name":      "Long Wheat / Long Gold / Short Emerging Markets",
        "rationale": "Food price spikes trigger inflation in EM countries with high cereal import ratios. "
                     "Combined with USD strength (gold hedge), EM equities face dual pressure from "
                     "import inflation and capital outflows.",
        "entry":     "Wheat 30d return > +15% + political instability signals in MENA/SSA",
        "exit":      "Wheat normalises to 12M average; EM carry recovery",
        "risk":      "IMF/World Bank intervention; US export restrictions",
        "assets":    ["Wheat", "Gold", "Sensex"],
        "direction": ["Long", "Long", "Short"],
        "category":  "Macro",
    },
    {
        "regime":    [0, 1],
        "trigger":   "Global growth recovery (low correlation regime)",
        "name":      "Long Copper / Long S&P 500",
        "rationale": "Copper is the premier global growth bellwether. "
                     "When correlation is low, commodities and equities price "
                     "independent fundamentals - copper rising with equities signals "
                     "genuine demand expansion, not just liquidity.",
        "entry":     "Copper 60d momentum > 0, ISM Manufacturing > 50, cross-asset corr < 0.20",
        "exit":      "Copper momentum reversal; ISM contraction; Fed pause signals",
        "risk":      "China property market collapse; USD spike from geopolitical safe-haven",
        "assets":    ["Copper", "S&P 500"],
        "direction": ["Long", "Long"],
        "category":  "Growth",
    },
    {
        "regime":    [1, 2],
        "trigger":   "Oil-equity divergence (supply shock premium)",
        "name":      "Long WTI Crude / Short S&P 500 Energy-Heavy Sectors",
        "rationale": "When oil spikes from a supply shock (not demand), "
                     "energy futures gain while broader equities face margin compression. "
                     "The long/short captures the spread between commodity producer "
                     "and equity consumer dynamics.",
        "entry":     "Brent-WTI spread widens + OPEC+ surprise cut + SPX P/E compression",
        "exit":      "Supply restoration event; recession pricing dominates oil",
        "risk":      "Demand destruction flips correlation; tech-led equity rally decouples",
        "assets":    ["WTI Crude Oil", "S&P 500"],
        "direction": ["Long", "Short"],
        "category":  "Macro",
    },
    {
        "regime":    [3],
        "trigger":   "Full crisis - all correlations elevated",
        "name":      "Long Gold, Long Silver / Short Copper, Short Shanghai Comp",
        "rationale": "Full crisis regime: precious metals outperform as industrial metals "
                     "and EM equities (particularly China) collapse under dollar strength "
                     "and risk-off flows. Gold/Silver spread also captures "
                     "the industrial-vs-monetary metals divergence.",
        "entry":     "Crisis regime active > 3 days; VIX > 35; DXY trending up",
        "exit":      "Regime drops below Elevated; Fed emergency action; VIX < 25",
        "risk":      "Chinese stimulus; commodity demand front-running recovery",
        "assets":    ["Gold", "Silver", "Copper", "Shanghai Comp"],
        "direction": ["Long", "Long", "Short", "Short"],
        "category":  "Crisis Hedge",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Private credit stress / HY spread widening + BDC underperformance",
        "name":      "Short BDC Basket / Long HY Credit Protection",
        "rationale": "Private credit ($2T+ AUM) is illiquid and marked-to-model quarterly. "
                     "When HY OAS widens >35bps in 90 days and BKLN underperforms SPY, "
                     "BDC equity (ARCC, OBDC, FSK) reprices before private marks surface - "
                     "a lagged NAV writedown is typically confirmed 1–2 quarters later. "
                     "The CDX HY 5Y index provides liquid short exposure to the same credit universe. "
                     "Gold long captures contagion into safe-haven flows as financial sector sells off.",
        "entry":     "HY OAS >350bps and rising >30bps/month; BKLN below 200d MA; "
                     "BDC basket -5% vs SPY on 60d basis; elevated/crisis equity-commodity regime",
        "exit":      "HY OAS stabilises <300bps; Fed emergency rate cut signal; "
                     "BDC premiums re-compress; credit facilities extended at par",
        "risk":      "Fed emergency cuts collapse floating-rate costs rapidly; "
                     "sponsor rescue financing delays visible defaults; "
                     "interval fund gates mask redemption pressure for 12–18 months",
        "assets":    ["Ares Capital (ARCC)", "Blue Owl (OBDC)", "Gold"],
        "direction": ["Short", "Short", "Long"],
        "category":  "Private Credit",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Flight to quality - elevated/crisis correlation regime",
        "name":      "Long TLT / Short HYG (Flight to Quality)",
        "rationale": "In elevated and crisis correlation regimes, investors rotate from credit risk to duration safety. "
                     "TLT captures the safe-haven Treasury bid while HYG shorts the credit spread widening. "
                     "The trade isolates the quality spread compression that accompanies every risk-off episode.",
        "entry":     "Crisis/Elevated regime active; HY OAS rising >30bps in 30 days; VIX >25; TLT above 200d MA",
        "exit":      "Regime drops to Normal; HY OAS stabilises; Fed pivot signal",
        "risk":      "Bear steepening (long rates rise with HY spreads simultaneously); fiscal dominance narrative",
        "assets":    ["US 20Y+ Treasury (TLT)", "HY Corporate (HYG)"],
        "direction": ["Long", "Short"],
        "category":  "Fixed Income",
    },
    {
        "regime":    [1, 2],
        "trigger":   "Inflation breakeven expansion / stagflation risk",
        "name":      "Long TIP / Short TLT (Inflation Breakeven Trade)",
        "rationale": "When commodity prices spike (energy, food) and the Fed is behind the curve, "
                     "real yields compress while nominal yields stay elevated. "
                     "TIPS outperform nominal Treasuries as breakeven inflation widens. "
                     "This trade directly monetises the commodity-to-bond inflation transmission channel.",
        "entry":     "5Y breakeven inflation rising >20bps in 60 days; WTI or wheat up >15% in 30 days; CPI surprise positive",
        "exit":      "Breakeven inflation peaks; commodity prices mean-revert; Fed delivers credible inflation response",
        "risk":      "Demand destruction flips inflation to deflation; recession pricing overrides inflation premium",
        "assets":    ["TIPS / Inflation (TIP)", "US 20Y+ Treasury (TLT)"],
        "direction": ["Long", "Short"],
        "category":  "Fixed Income",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Oil spike + INR depreciation (India crude import stress)",
        "name":      "Long Brent Crude / Short Nifty 50 (India Import Shock)",
        "rationale": "India imports ~85% of its crude oil needs (~5 mb/d). "
                     "When Brent spikes >15% in 60 days and USD/INR is depreciating simultaneously, "
                     "India's current account deficit widens sharply - historically correlating with "
                     "Nifty 50 underperformance of -6% to -12% vs global peers. "
                     "The long Brent / short Nifty trade monetises the commodity-to-EM-equity transmission channel specific to India.",
        "entry":     "Brent up >15% in 60 days AND USD/INR rising >3% in 30 days; elevated/crisis regime active",
        "exit":      "Oil supply restored; INR stabilises / RBI intervention; India CAD narrows",
        "risk":      "RBI forex reserve intervention caps INR weakness; India domestic demand surprises to upside; OPEC+ production cut reversal",
        "assets":    ["Brent Crude", "Nifty 50"],
        "direction": ["Long", "Short"],
        "category":  "India/EM",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Geopolitical stress + dollar strength - India gold import sensitivity",
        "name":      "Long Gold / Short INR (India Geopolitical Hedge)",
        "rationale": "India is the world's #2 gold consumer (~800-900 tonnes/year). "
                     "During geopolitical stress (Middle East escalation, South Asia tensions), "
                     "gold demand surges from Indian households as a safe-haven AND currency hedge. "
                     "Simultaneously, USD/INR typically rises (INR weakens) under global risk-off conditions. "
                     "The long gold / short INR (via USD/INR long) trade captures both legs of this transmission.",
        "entry":     "Crisis/Elevated regime; VIX >25; USD/INR rising >2% in 20 days; Gold 20d momentum positive",
        "exit":      "Geopolitical de-escalation; gold mean-reverts -8% from peak; INR stabilises on RBI action",
        "risk":      "RBI aggressively defends INR using $620B forex reserves; Gold sell-off on Fed hawkish surprise",
        "assets":    ["Gold", "USD/INR"],
        "direction": ["Long", "Long"],
        "category":  "India/EM",
    },
    {
        "regime":    [0, 1],
        "trigger":   "Dollar weakness cycle / EM credit rally",
        "name":      "Long EMB / Short DXY (Dollar Debasement - EM Relief)",
        "rationale": "A weakening dollar reduces dollar-denominated debt service costs for EM sovereigns, "
                     "compresses EM credit spreads, and attracts capital inflows into EM assets. "
                     "EMB captures the bond price appreciation; short DXY amplifies the currency leg. "
                     "Gold is the commodity expression of the same dollar debasement theme.",
        "entry":     "DXY down >3% in 60 days; Fed on hold or cutting; EM current accounts improving; commodity prices rising",
        "exit":      "Dollar reversal; Fed hawkish pivot; EM-specific credit event; commodity demand collapse",
        "risk":      "EM-specific credit events (sovereign default, political crisis); commodity demand collapse flips EM outlook",
        "assets":    ["EM USD Bonds (EMB)", "DXY (Dollar Index)", "Gold"],
        "direction": ["Long", "Short", "Long"],
        "category":  "Fixed Income",
    },
]

_CATEGORY_COLORS = {
    "Crisis Hedge":   "#c0392b",
    "Geopolitical":   "#e67e22",
    "Macro":          "#2980b9",
    "Growth":         "#2e7d32",
    "Private Credit": "#8e44ad",
    "Fixed Income":   "#2980b9",
    "India/EM":       "#16a085",
}


def _render_trade_card(
    col,
    trade: dict,
    all_r_concat: pd.DataFrame,
    current: int,
    trade_idx: int,
    asset_exposure: dict | None = None,
) -> None:
    """Render a single trade card with QC grade, confidence, payoff table, and debate thread."""
    cat_col = _CATEGORY_COLORS.get(trade["category"], "#CFB991")

    # ── QC scoring ─────────────────────────────────────────────────────────
    try:
        from src.analysis.trade_filter import score_trade_quality
        qc = score_trade_quality(trade)
    except Exception:
        qc = {"score": 60, "grade": "B", "flags": []}

    grade      = qc["grade"]
    qc_score   = qc["score"]
    grade_color = {"A": "#27ae60", "B": "#2980b9", "C": "#e67e22", "D": "#c0392b"}.get(grade, "#8890a1")
    confidence  = float(trade.get("confidence", 0.60))
    conf_pct    = f"{confidence*100:.0f}%"
    conf_color  = "#27ae60" if confidence >= 0.70 else "#e67e22" if confidence >= 0.55 else "#c0392b"

    # ── Badges ─────────────────────────────────────────────────────────────
    is_generated = trade.get("generated", False)
    conflict_id  = trade.get("conflict_id")
    source_badge = (
        '<span style="background:#1a1a2e;color:#CFB991;padding:1px 5px;'
        'font-size:0.52rem;font-weight:700;letter-spacing:0.10em;border-radius:2px">'
        'LIVE GEO</span>'
        if is_generated else
        '<span style="background:#2a2a2a;color:#8890a1;padding:1px 5px;'
        'font-size:0.52rem;letter-spacing:0.08em;border-radius:2px">STATIC</span>'
    )
    conflict_badge = (
        f'<span style="background:#3d1a00;color:#e67e22;padding:1px 5px;'
        f'font-size:0.52rem;font-weight:700;letter-spacing:0.08em;border-radius:2px">'
        f'{conflict_id.upper().replace("_"," ")}</span>'
        if conflict_id else ""
    )

    dir_html = " &nbsp;|&nbsp; ".join(
        f'<span style="color:{"#2e7d32" if d=="Long" else "#c0392b"};font-weight:700">{d}</span> {a}'
        for a, d in zip(trade["assets"], trade["direction"])
    )
    regime_pills = " ".join(
        f'<span style="background:{_REGIME_COLORS[r]};color:#fff;padding:1px 5px;'
        f'border-radius:2px;font-size:0.56rem">{_REGIME_NAMES[r]}</span>'
        for r in trade.get("regime", [])
    )

    with col:
        st.markdown(
            f'<div style="border:1px solid #1e1e1e;border-radius:0;overflow:hidden;'
            f'margin-bottom:0.5rem;background:#0d0d0d">'
            # Header bar
            f'<div style="background:{cat_col};padding:0.4rem 0.85rem;'
            f'display:flex;justify-content:space-between;align-items:center">'
            f'<div style="font-size:0.57rem;font-weight:700;letter-spacing:0.10em;'
            f'text-transform:uppercase;color:#fff">'
            f'{trade["category"]} · {trade["trigger"]}</div>'
            f'<div style="display:flex;gap:4px;align-items:center">'
            f'{regime_pills} {source_badge} {conflict_badge}</div></div>'
            # Body
            f'<div style="padding:0.75rem 0.85rem">'
            # Name + direction
            f'<div style="font-size:0.84rem;font-weight:700;color:#e8e8e8;margin-bottom:3px">'
            f'{trade["name"]}</div>'
            f'<div style="font-size:0.70rem;color:#aaa;margin-bottom:6px">{dir_html}</div>'
            # Confidence + QC row
            f'<div style="display:flex;gap:8px;align-items:center;margin-bottom:8px">'
            f'<span style="font-size:0.60rem;color:#8890a1;text-transform:uppercase;'
            f'letter-spacing:0.10em">Confidence</span>'
            f'<span style="font-weight:700;color:{conf_color};font-size:0.75rem">{conf_pct}</span>'
            f'<span style="font-size:0.60rem;color:#8890a1;text-transform:uppercase;'
            f'letter-spacing:0.10em;margin-left:8px">QC</span>'
            f'<span style="background:{grade_color};color:#fff;font-weight:700;'
            f'padding:1px 6px;border-radius:2px;font-size:0.65rem">{grade} ({qc_score})</span>'
            f'</div>'
            # Rationale
            f'<p style="font-size:0.70rem;color:#cccccc;line-height:1.65;margin-bottom:8px">'
            f'{trade["rationale"]}</p>'
            # Entry / Exit / Risk grid
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;font-size:0.68rem">'
            f'<div style="background:#111;padding:0.35rem 0.5rem">'
            f'<div style="font-weight:700;color:#CFB991;font-size:0.55rem;text-transform:uppercase;'
            f'letter-spacing:0.10em;margin-bottom:2px">Entry</div>'
            f'<span style="color:#ddd">{trade.get("entry","—")}</span></div>'
            f'<div style="background:#111;padding:0.35rem 0.5rem">'
            f'<div style="font-weight:700;color:#8890a1;font-size:0.55rem;text-transform:uppercase;'
            f'letter-spacing:0.10em;margin-bottom:2px">Exit</div>'
            f'<span style="color:#ddd">{trade.get("exit","—")}</span></div>'
            f'<div style="background:#1a0000;padding:0.35rem 0.5rem">'
            f'<div style="font-weight:700;color:#c0392b;font-size:0.55rem;text-transform:uppercase;'
            f'letter-spacing:0.10em;margin-bottom:2px">Risks</div>'
            f'<span style="color:#ddd">{trade.get("risk","—")}</span></div>'
            f'</div>'
            # ── Extended fields: stop / target / invalidation / holding ──
            + (
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;'
                f'gap:5px;font-size:0.62rem;margin-top:5px">'
                f'<div style="background:#0a100a;border:1px solid #1a2a1a;padding:0.3rem 0.5rem">'
                f'<div style="font-weight:700;color:#27ae60;font-size:0.50rem;text-transform:uppercase;'
                f'letter-spacing:0.10em;margin-bottom:2px">Stop</div>'
                f'<span style="color:#ddd">{trade.get("stop","—")}</span></div>'
                f'<div style="background:#0a100a;border:1px solid #1a2a1a;padding:0.3rem 0.5rem">'
                f'<div style="font-weight:700;color:#27ae60;font-size:0.50rem;text-transform:uppercase;'
                f'letter-spacing:0.10em;margin-bottom:2px">Target</div>'
                f'<span style="color:#ddd">{trade.get("target","—")}</span></div>'
                f'<div style="background:#0d0d14;border:1px solid #1a1a2a;padding:0.3rem 0.5rem">'
                f'<div style="font-weight:700;color:#2980b9;font-size:0.50rem;text-transform:uppercase;'
                f'letter-spacing:0.10em;margin-bottom:2px">Invalidation</div>'
                f'<span style="color:#ddd">{trade.get("invalidation","—")}</span></div>'
                f'<div style="background:#111;border:1px solid #1e1e1e;padding:0.3rem 0.5rem">'
                f'<div style="font-weight:700;color:#8890a1;font-size:0.50rem;text-transform:uppercase;'
                f'letter-spacing:0.10em;margin-bottom:2px">Holding Period</div>'
                f'<span style="color:#ddd">{trade.get("holding_period","—")}</span></div>'
                f'</div>'
                if any(trade.get(k) for k in ["stop", "target", "invalidation", "holding_period"])
                else ""
            )
            + f'</div></div>',
            unsafe_allow_html=True,
        )

        # ── Inline projected P&L row ───────────────────────────────────────
        try:
            from src.analysis.profit_projection import project_trade
            _proj = project_trade(trade)
            _epnl = _proj["expected_pnl"]
            _wpnl = _proj["worst_case_pnl"]
            _bprob = _proj["breakeven_prob"]
            _sharpe = _proj["sharpe_proxy"]
            _epnl_col = "#27ae60" if _epnl >= 0 else "#c0392b"
            _wpnl_col = "#c0392b" if _wpnl < -5 else "#e67e22"
            with col:
                st.markdown(
                    f'<div style="display:flex;gap:16px;background:#080808;'
                    f'border:1px solid #1e1e1e;padding:4px 10px;margin-top:-2px;'
                    f'margin-bottom:4px;align-items:center;flex-wrap:wrap">'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
                    f'color:#555960;letter-spacing:.12em">PROJECTED</span>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                    f'font-weight:700;color:{_epnl_col}">E[P&L]&nbsp;{_epnl:+.1f}%</span>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                    f'font-weight:700;color:{_wpnl_col}">Worst&nbsp;{_wpnl:+.1f}%</span>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                    f'color:#8E9AAA">Breakeven&nbsp;{_bprob*100:.0f}%</span>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                    f'color:#CFB991">Sharpe&nbsp;{_sharpe:.2f}</span>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
                    f'color:#555960;margin-left:auto">model estimate</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        except Exception:
            pass

        # QC flags
        if qc["flags"]:
            st.markdown(
                " ".join(
                    f'<span style="background:#1a0e00;color:#e67e22;padding:1px 6px;'
                    f'font-size:0.55rem;border-radius:2px;margin-right:3px">⚠ {f}</span>'
                    for f in qc["flags"]
                ),
                unsafe_allow_html=True,
            )

        # ── Why this trade passed filters ─────────────────────────────────
        try:
            _pass_reasons: list[str] = []
            if trade.get("generated"):
                _pass_reasons.append("Live GEO signal")
            if trade.get("conflict_id"):
                _cid = trade["conflict_id"].replace("_", " ").title()
                _pass_reasons.append(f"Conflict: {_cid}")
            _pass_reasons.append(f"Regime {_REGIME_NAMES.get(current, current)}")
            if confidence >= 0.70:
                _pass_reasons.append(f"Conf {confidence*100:.0f}%")
            _cat = trade.get("category", "")
            if _cat and _cat != "all":
                _pass_reasons.append(_cat)
            # Add SAS-based why-now signals from exposure data
            if asset_exposure:
                _trade_assets = trade.get("assets", [])
                _sas_vals = [asset_exposure[a]["sas"] for a in _trade_assets if a in asset_exposure]
                if _sas_vals:
                    _avg_sas = sum(_sas_vals) / len(_sas_vals)
                    if _avg_sas >= 60:
                        _pass_reasons.append(f"SAS {_avg_sas:.0f} — high exposure")
                    elif _avg_sas >= 35:
                        _pass_reasons.append(f"SAS {_avg_sas:.0f}")
                # Hedge signal
                _hedge_scores = [asset_exposure[a]["hedge_score"] for a in _trade_assets if a in asset_exposure]
                if any(h >= 40 for h in _hedge_scores):
                    _pass_reasons.append("Hedge signal active")
                # Scenario alignment
                _directions = [asset_exposure[a]["direction"] for a in _trade_assets if a in asset_exposure]
                if "safe_haven" in _directions and any(d.lower() == "long" for d in trade.get("direction", [])):
                    _pass_reasons.append("Safe-haven demand")
            if _pass_reasons:
                st.markdown(
                    '<div style="display:flex;gap:4px;flex-wrap:wrap;margin:4px 0 2px">'
                    + "".join(
                        f'<span style="background:#0a1a0a;color:#27ae60;'
                        f'font-family:\'JetBrains Mono\',monospace;font-size:6.5px;'
                        f'padding:2px 5px;border:1px solid #1a3a1a">✓ {r}</span>'
                        for r in _pass_reasons
                    )
                    + '</div>',
                    unsafe_allow_html=True,
                )
        except Exception:
            pass

        # ── Exposure strip: per-asset SAS + direction + top conflict beta ──
        if asset_exposure:
            try:
                _exp_items = []
                for _a, _d in zip(trade.get("assets", []), trade.get("direction", [])):
                    _ed = asset_exposure.get(_a)
                    if not _ed:
                        continue
                    _sas = _ed["sas"]
                    _dir = _ed["direction"]
                    _top_c = _ed.get("top_conflict") or ""
                    _top_beta = _ed["beta"].get(_top_c, 0.0) if _top_c else 0.0
                    _dir_icon = "↑" if _dir == "long_geo_risk" else "↓" if _dir == "safe_haven" else "→"
                    _sas_col = "#e67e22" if _sas >= 60 else "#CFB991" if _sas >= 30 else "#555960"
                    _top_c_label = _top_c.replace("_", " ").upper()[:12] if _top_c else ""
                    _beta_str = f"β={_top_beta:.2f}" if _top_c else ""
                    _item_html = (
                        f'<div style="background:#0a0a0a;border:1px solid #1a1a1a;'
                        f'padding:2px 6px;display:flex;flex-direction:column;gap:1px;min-width:90px">'
                        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:6px;'
                        f'color:#555960;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:110px">'
                        f'{_a[:18]}</div>'
                        f'<div style="display:flex;gap:5px;align-items:center">'
                        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
                        f'font-weight:700;color:{_sas_col}">SAS {_sas:.0f}</span>'
                        f'<span style="font-size:8px;color:#8890a1">{_dir_icon}</span>'
                        + (f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:6px;'
                           f'color:#8890a1">{_beta_str}</span>' if _beta_str else "")
                        + '</div>'
                        + (f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:5.5px;'
                           f'color:#444c5c">{_top_c_label}</div>' if _top_c_label else "")
                        + '</div>'
                    )
                    _exp_items.append(_item_html)
                if _exp_items:
                    st.markdown(
                        '<div style="display:flex;gap:4px;flex-wrap:wrap;margin:4px 0 2px">'
                        + "".join(_exp_items)
                        + '</div>',
                        unsafe_allow_html=True,
                    )
            except Exception:
                pass

        # ── Payoff table expander ──────────────────────────────────────────
        with st.expander(f"Scenario Payoff Table — {trade['name'][:40]}", expanded=False):
            try:
                from src.analysis.profit_projection import project_trade
                proj   = project_trade(trade)
                p_table = proj["payoff_table"]

                pm1, pm2, pm3, pm4 = st.columns(4)
                pm1.metric("Exp. P&L", f"{proj['expected_pnl']:+.1f}%")
                pm2.metric("Worst Case", f"{proj['worst_case_pnl']:+.1f}%")
                pm3.metric("Breakeven Prob", f"{proj['breakeven_prob']*100:.0f}%")
                pm4.metric("Sharpe Proxy", f"{proj['sharpe_proxy']:.2f}")

                # Payoff bar chart
                sc_labels = [r["label"] for r in p_table]
                sc_pnls   = [r["expected_pnl"] for r in p_table]
                sc_colors = [
                    "#27ae60" if pnl >= 0 else "#c0392b"
                    for pnl in sc_pnls
                ]
                fig_pf = go.Figure(go.Bar(
                    x=sc_labels, y=sc_pnls,
                    marker_color=sc_colors,
                    text=[f"{v:+.1f}%" for v in sc_pnls],
                    textposition="outside",
                ))
                fig_pf.update_layout(
                    template="plotly_dark",
                    height=220,
                    title=dict(text="Expected P&L by Scenario (%)", font=dict(size=11)),
                    margin=dict(l=40, r=20, t=36, b=40),
                    yaxis=dict(title="P&L %", zeroline=True,
                               zerolinecolor="#333", zerolinewidth=1),
                    showlegend=False,
                    plot_bgcolor="#0d0d0d",
                    paper_bgcolor="#0d0d0d",
                )
                _chart(fig_pf)

                # Scenario table
                pt_df = pd.DataFrame([{
                    "Scenario":   r["label"],
                    "Prob":       f"{r['prob']*100:.0f}%",
                    "Exp. P&L":   f"{r['expected_pnl']:+.1f}%",
                    "Vol":        f"{r['vol']:.1f}%",
                    "Wtd P&L":    f"{r['prob_weighted_pnl']:+.2f}%",
                    "Active":     "★" if r["is_current"] else "",
                } for r in p_table])
                st.dataframe(pt_df, use_container_width=True, hide_index=True)

            except Exception as exc:
                st.caption(f"Payoff projection unavailable: {exc}")

        # ── Agent debate thread ────────────────────────────────────────────
        # Auto-expand for conflict-driven ideas (higher urgency)
        _is_geo_trade  = trade.get("generated", False)
        _debate_open   = _is_geo_trade  # expand debate panel for geo trades by default
        _debate_label  = (
            f"⚡ Agent Debate — {trade['name'][:40]}"
            if _is_geo_trade
            else f"Agent Debate — {trade['name'][:40]}"
        )
        with st.expander(_debate_label, expanded=_debate_open):
            try:
                from src.ui.agent_panel import render_deliberation_panel
                from src.analysis.agent_dialogue import (
                    challenge_trade, get_subject_threads,
                )
                _trade_subject_id = trade.get("name", f"trade_{trade_idx}")
                # Look up by subject_id (trade name) — works across sessions
                _stored_key = f"_debate_tid_{trade_idx}"
                msgs = get_subject_threads(_trade_subject_id)

                if not msgs:
                    if _is_geo_trade:
                        # Auto-run debate for conflict-driven trades on first render
                        try:
                            _new_tid = challenge_trade(
                                trade_id=_trade_subject_id,
                                trade_title=trade.get("name", "Unknown Trade"),
                                confidence=float(trade.get("confidence", 0.60)),
                                qc_flags=list(trade.get("qc_flags", [])),
                            )
                            st.session_state[_stored_key] = _new_tid
                            msgs = get_subject_threads(_trade_subject_id)
                        except Exception:
                            pass
                    if not msgs:
                        if st.button(f"Run Agent Debate", key=f"debate_{trade_idx}"):
                            try:
                                _new_tid = challenge_trade(
                                    trade_id=_trade_subject_id,
                                    trade_title=trade.get("name", "Unknown Trade"),
                                    confidence=float(trade.get("confidence", 0.60)),
                                    qc_flags=list(trade.get("qc_flags", [])),
                                )
                                st.session_state[_stored_key] = _new_tid
                                st.rerun()
                            except Exception as exc:
                                st.caption(f"Debate unavailable: {exc}")

                if msgs:
                    # Use the stored thread_id if available, else use first message's thread
                    _render_tid = st.session_state.get(_stored_key) or msgs[0]["thread_id"]
                    render_deliberation_panel(
                        thread_id=_render_tid,
                        subject_id=_trade_subject_id,
                        title="Agent Deliberation",
                        max_msgs=8,
                        show_consensus=True,
                    )
            except Exception as exc:
                st.caption(f"Debate panel unavailable: {exc}")

        # Mini correlation chart
        if len(trade["assets"]) >= 2:
            a1, a2 = trade["assets"][0], trade["assets"][1]
            if a1 in all_r_concat.columns and a2 in all_r_concat.columns:
                rc = rolling_correlation(all_r_concat[a1], all_r_concat[a2], 60)
                fig_mini = go.Figure()
                r_hex, g_hex, b_hex = (
                    int(cat_col[1:3], 16),
                    int(cat_col[3:5], 16),
                    int(cat_col[5:7], 16),
                )
                fig_mini.add_trace(go.Scatter(
                    x=rc.index, y=rc.values,
                    name=f"{a1}/{a2}",
                    line=dict(color=cat_col, width=1.4),
                    fill="tozeroy",
                    fillcolor=f"rgba({r_hex},{g_hex},{b_hex},0.12)",
                ))
                fig_mini.add_hline(y=0, line=dict(color="#444", width=1, dash="dot"))
                for ev in GEOPOLITICAL_EVENTS:
                    fig_mini.add_vrect(
                        x0=str(ev["start"]), x1=str(ev["end"]),
                        fillcolor=ev["color"], opacity=0.04, layer="below", line_width=0,
                    )
                fig_mini.update_layout(
                    template="plotly_dark", height=180,
                    title=dict(text=f"60d Corr: {a1} / {a2}", font=dict(size=10)),
                    showlegend=False,
                    margin=dict(l=36, r=12, t=28, b=24),
                    xaxis=dict(rangeslider=dict(visible=False)),
                    plot_bgcolor="#0d0d0d",
                    paper_bgcolor="#0d0d0d",
                )
                _chart(fig_mini)
                _insight_note(
                    "Rolling 60-day correlation between the commodity driver and equity target. "
                    "Rising correlation = thesis strengthening. Declining = causal link breaking down."
                )


def page_trade_ideas(start: str, end: str, fred_key: str = "") -> None:
    _page_header("Structured Trade Ideas",
                 "Regime-driven · Conflict-linked · Exposure-ranked · QC-graded")
    _page_intro(
        "Spillover analysis is only useful if it generates actionable positioning. "
        "<strong>Each idea here is a direct translation of a spillover or correlation regime signal into a trade.</strong> "
        "Conflict-driven candidates are generated live from current CIS/TPS scores and exposure data. "
        "Static library ideas fire when the current regime matches their structural trigger. "
        "All ideas are QC-graded (A–D), scenario-payoff-projected, and debatable via agent threads."
    )

    # ── Geopolitical context & filter gate (shown BEFORE any trades) ──────────
    try:
        from src.analysis.conflict_model import score_all_conflicts, aggregate_portfolio_scores
        from src.analysis.scenario_state import get_scenario, get_scenario_id
        _ti_cr   = score_all_conflicts()
        _ti_agg  = aggregate_portfolio_scores(_ti_cr)
        _ti_sc   = get_scenario()
        _ti_sid  = get_scenario_id()
        _ti_cis  = _ti_agg.get("portfolio_cis", _ti_agg.get("cis", 50.0))
        _ti_tps  = _ti_agg.get("portfolio_tps", _ti_agg.get("tps", 50.0))
        _ti_top  = (_ti_agg.get("top_conflict", "—") or "—").replace("_", " ").title()
        _ti_mult = _ti_sc.get("geo_mult", 1.0)
        _ti_sc_color = _ti_sc.get("color", "#CFB991")

        if _ti_cis >= 70:    _ti_risk_color, _ti_risk_lbl = "#c0392b", "HIGH CONFLICT"
        elif _ti_cis >= 50:  _ti_risk_color, _ti_risk_lbl = "#e67e22", "ELEVATED"
        else:                _ti_risk_color, _ti_risk_lbl = "#CFB991", "MODERATE"

        st.markdown(
            f'<div style="background:#080808;border:1px solid #1e1e1e;'
            f'border-left:3px solid {_ti_risk_color};padding:.55rem 1rem;margin-bottom:.6rem">'
            f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
            f'font-weight:700;color:{_ti_risk_color};letter-spacing:.18em">'
            f'■ GEO CONTEXT · TRADE GENERATION INPUT</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            f'color:#e67e22">CIS&nbsp;<b>{_ti_cis:.0f}</b></span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            f'color:#CFB991">TPS&nbsp;<b>{_ti_tps:.0f}</b></span>'
            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:9px;'
            f'color:#8E9AAA">Lead:&nbsp;<b style="color:{_ti_risk_color}">{_ti_top}</b></span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            f'color:{_ti_sc_color}">{_ti_sc.get("label","Base").upper()}&nbsp;×{_ti_mult:.2f}</span>'
            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:8.5px;'
            f'color:#555960;margin-left:auto">'
            f'Conflict-driven ideas below reflect these live inputs. '
            f'Set filters before reviewing trades.</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    with st.spinner("Loading data…"):
        eq_r, cmd_r = load_returns(start, end)

    if eq_r.empty or cmd_r.empty:
        st.error("Market data unavailable.")
        return

    # ── Current regime ─────────────────────────────────────────────────────
    avg_corr = average_cross_corr_series(eq_r, cmd_r, window=60)
    regimes  = detect_correlation_regime(avg_corr)
    current  = int(regimes.iloc[-1]) if not regimes.empty else 1
    r_name   = _REGIME_NAMES[current]
    r_color  = _REGIME_COLORS[current]

    # ── Conflict-driven candidates ─────────────────────────────────────────
    generated: list[dict] = []
    conflict_betas: dict  = {}
    asset_exposure: dict  = {}
    try:
        from src.analysis.trade_generator import generate_conflict_trades, merge_with_library
        from src.analysis.exposure import score_all_assets
        from src.analysis.conflict_model import score_all_conflicts
        _cr  = score_all_conflicts()
        _aa  = score_all_assets(conflict_results=_cr)
        generated = generate_conflict_trades(regime=current, conflict_results=_cr, all_assets=_aa)
        # Retain full exposure data for ranking and card display
        asset_exposure = dict(_aa)
        conflict_betas = {
            name: {"beta": d["beta"]}
            for name, d in _aa.items()
        }
    except Exception:
        pass  # silently degrade if exposure/conflict data unavailable

    # ── Merge with static library ──────────────────────────────────────────
    try:
        from src.analysis.trade_generator import merge_with_library as _merge
        all_candidates = _merge(generated, _TRADE_LIBRARY, current)
    except Exception:
        all_candidates = list(_TRADE_LIBRARY)
        for t in all_candidates:
            t.setdefault("confidence", 0.60)
            t.setdefault("qc_flags",  [])

    # ── Apply filters ──────────────────────────────────────────────────────
    try:
        from src.analysis.trade_filter import apply_filters, build_filter_ui
        from src.analysis.scenario_state import get_scenario_id
        _filters      = build_filter_ui(key_prefix="ti")
        _scenario_id  = get_scenario_id()
        active_trades = apply_filters(
            all_candidates,
            filters=_filters,
            current_regime=current,
            current_scenario_id=_scenario_id,
            conflict_betas=conflict_betas,
            asset_exposure=asset_exposure,
        )
    except Exception:
        # Fallback: basic regime filter
        active_trades = [t for t in all_candidates if current in t.get("regime", [current])]

    # ── Re-rank by exposure × confidence composite ─────────────────────────
    if asset_exposure:
        def _exposure_rank_key(t: dict) -> float:
            conf = float(t.get("confidence", 0.5))
            sas_vals = [asset_exposure[a]["sas"] for a in t.get("assets", []) if a in asset_exposure]
            avg_sas = (sum(sas_vals) / len(sas_vals)) if sas_vals else 0.0
            return conf * (1 + avg_sas / 200)
        active_trades.sort(key=_exposure_rank_key, reverse=True)

    # ── KPI strip ──────────────────────────────────────────────────────────
    _n_geo  = sum(1 for t in active_trades if t.get("generated"))
    _n_stat = len(active_trades) - _n_geo
    _n_bull = sum(1 for t in active_trades if t.get("direction", [""])[0] == "Long")
    _n_bear = sum(1 for t in active_trades if t.get("direction", [""])[0] == "Short")

    _F_ti = "font-family:'DM Sans',sans-serif;"
    def _ti_kpi(col, label, value, value_color="#e8e8e8"):
        col.markdown(
            f'<div style="border:1px solid #1e1e1e;border-radius:0;padding:0.6rem 0.85rem;'
            f'background:#0d0d0d">'
            f'<div style="{_F_ti}font-size:0.55rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.14em;color:#CFB991;margin-bottom:3px">{label}</div>'
            f'<div style="{_F_ti}font-size:1.0rem;font-weight:700;color:{value_color}">{value}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.markdown(
        f'<div style="border:1px solid {r_color};border-radius:0;'
        f'padding:0.6rem 0.85rem;background:#0d0d0d">'
        f'<div style="{_F_ti}font-size:0.55rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.14em;color:#CFB991;margin-bottom:3px">Regime</div>'
        f'<div style="{_F_ti}font-size:1.0rem;font-weight:700;color:{r_color}">{r_name}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    _ti_kpi(k2, "Avg Corr (60d)",  f"{avg_corr.iloc[-1]:.3f}")
    _ti_kpi(k3, "Active Ideas",    str(len(active_trades)), "#CFB991")
    _ti_kpi(k4, "Conflict-Driven", str(_n_geo),  "#e67e22")
    _ti_kpi(k5, "Long-First",      str(_n_bull), "#27ae60")
    _ti_kpi(k6, "Short-First",     str(_n_bear), "#c0392b")

    if not active_trades:
        st.info(f"No trade ideas pass current filters for {r_name} regime.")
        st.stop()

    st.markdown(
        f'<p style="font-size:0.70rem;color:#8890a1;margin-bottom:0.6rem">'
        f'<b style="color:#e8e8e8">{len(active_trades)}</b> ideas — '
        f'<b style="color:#e67e22">{_n_geo}</b> conflict-driven · '
        f'<b style="color:#8890a1">{_n_stat}</b> static · '
        f'sorted by exposure × confidence · regime: '
        f'<b style="color:{r_color}">{r_name}</b></p>',
        unsafe_allow_html=True,
    )

    all_r_concat = pd.concat([eq_r, cmd_r], axis=1)

    _thread(
        "Conflict-driven ideas are generated live from current CIS/TPS/SAS scores. "
        "Static ideas fire when the regime matches their structural trigger. "
        "Both sets are QC-graded and filtered by the active filter settings above."
    )

    # ── 2-column card grid ──────────────────────────────────────────────────
    for row_start in range(0, len(active_trades), 2):
        pair = active_trades[row_start:row_start + 2]
        card_cols = st.columns(len(pair), gap="medium")
        for col, (trade, idx) in zip(card_cols, [(t, row_start + i) for i, t in enumerate(pair)]):
            _render_trade_card(col, trade, all_r_concat, current, idx, asset_exposure or None)

    # ── Download report ─────────────────────────────────────────────────────
    st.markdown(
        '<p style="font-size:0.74rem;font-weight:700;letter-spacing:0.06em;'
        'text-transform:uppercase;color:#CFB991;margin-bottom:0.5rem">'
        'Institution Report Download</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:0.72rem;color:#8890a1;line-height:1.7;margin-bottom:0.8rem">'
        'Generate a professionally formatted A4 research report covering the current regime, '
        'all active trade ideas, geopolitical context, and methodology - suitable for '
        'academic submission or institutional review.</p>',
        unsafe_allow_html=True,
    )

    if st.button("Generate PDF Report", key="gen_report", type="primary"):
        try:
            from src.reports.report_generator import generate_report
            with st.spinner("Building report - generating charts…"):
                stress = composite_stress_index(eq_r, cmd_r, avg_corr=avg_corr)
                pdf_bytes = generate_report(
                    start=start,
                    end=end,
                    avg_corr_series=avg_corr,
                    current_regime=current,
                    regimes=regimes,
                    active_trades=active_trades,
                    all_trades=_TRADE_LIBRARY,
                    eq_r=eq_r,
                    cmd_r=cmd_r,
                    stress_series=stress,
                    geopolitical_events=GEOPOLITICAL_EVENTS,
                )
            filename = (
                f"purdue_spillover_report_"
                f"{start.replace('-','')}_to_{end.replace('-','')}.pdf"
            )
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                key="download_report",
            )
        except ImportError:
            st.error(
                "reportlab is required for PDF generation. "
                "Run: `pip install reportlab>=4.2.0`"
            )
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")

    _section_note(
        "Trade ideas are generated from historical cross-asset patterns and regime signals. "
        "All ideas are illustrative and must be validated against current market structure, "
        "liquidity, and position sizing constraints before implementation."
    )

    _page_conclusion(
        "Framework",
        "The regime-based trade generation framework matches historical spillover patterns "
        "to current correlation regimes. Crisis regimes activate hedging and divergence plays; "
        "normal regimes favour growth-correlated long positioning. "
        "Use Granger and transfer entropy results from the Spillover page to validate lead-lag direction."
    )

    # ── AI Trade Structurer + Pending Review Panel ─────────────────────────
    try:
        from src.agents.trade_structurer import run as _ts_run
        from src.ui.agent_panel import render_agent_output_block, render_pending_review
        from src.analysis.agent_state import is_enabled, get_agent

        if is_enabled("trade_structurer"):
            _anthropic_key = _openai_key = ""
            try:
                _keys = st.secrets.get("keys", {})
                _anthropic_key = _keys.get("anthropic_api_key", "") or ""
                _openai_key    = _keys.get("openai_api_key",    "") or ""
            except Exception:
                pass
            _provider = "anthropic" if _anthropic_key else ("openai" if _openai_key else None)
            _api_key  = _anthropic_key or _openai_key

            # Build context from what's been computed on this page
            _ts_ctx: dict = {
                "regime_name":   r_name,
                "regime_level":  current,
                "avg_corr":      float(avg_corr.iloc[-1]) if not avg_corr.empty else 0.0,
            }
            try:
                from src.analysis.risk_score import compute_risk_score
                from src.data.loader import load_returns as _lr3
                _eq_r3, _cmd_r3 = _lr3(start, end)
                from src.analysis.correlations import average_cross_corr_series as _acs
                _avg_c3 = _acs(_eq_r3, _cmd_r3, window=60)
                _rs = compute_risk_score(_avg_c3, _cmd_r3, _eq_r3)
                _ts_ctx["risk_score"] = float(_rs.get("score", 0))
                if len(_cmd_r3) >= 5:
                    _w5c = _cmd_r3.iloc[-5:].sum()
                    _ts_ctx["top_commodity"]     = str(_w5c.idxmax())
                    _ts_ctx["top_commodity_ret"] = float(_w5c.max()) * 100
                if len(_eq_r3) >= 5:
                    _w5e = _eq_r3.iloc[-5:].sum()
                    _ts_ctx["worst_equity"]      = str(_w5e.idxmin())
                    _ts_ctx["worst_equity_ret"]  = float(_w5e.min()) * 100
            except Exception:
                pass

            # Peer context from orchestrator (structured, freshness-aware)
            _peer_signals = {}
            try:
                from src.analysis.agent_orchestrator import get_orchestrator as _get_orch_ts
                _peer_signals = _get_orch_ts(_provider, _api_key).get_peer_context("trade_structurer")
            except Exception:
                # Fallback: read directly from agent state
                for _pid in ("macro_strategist", "geopolitical_analyst",
                             "stress_engineer", "commodities_specialist"):
                    _pa = get_agent(_pid)
                    if _pa.get("last_output"):
                        _raw = _pa["last_output"]
                        _lines = [l for l in _raw.split("\n") if "confidence:" not in l.lower()]
                        _peer_signals[_pid] = " ".join(_lines).strip()[:250]
            if _peer_signals:
                _ts_ctx["peer_signals"] = _peer_signals

            with st.spinner("AI Trade Structurer generating idea…"):
                _ts_result = _ts_run(_ts_ctx, _provider, _api_key)

            if _ts_result.get("narrative"):
                st.markdown("---")
                render_agent_output_block("trade_structurer", _ts_result)

        # Always show the review queue on this page
        st.markdown("---")
        render_pending_review()

    except Exception:
        pass

    # CQO runs silently - output visible in About > AI Workforce
    try:
        from src.agents.quality_officer import run as _cqo_run
        from src.analysis.agent_state import is_enabled
        if is_enabled("quality_officer"):
            _anthropic_key = _openai_key = ""
            try:
                _keys = st.secrets.get("keys", {})
                _anthropic_key = _keys.get("anthropic_api_key", "") or ""
                _openai_key    = _keys.get("openai_api_key",    "") or ""
            except Exception:
                pass
            _provider = "anthropic" if _anthropic_key else ("openai" if _openai_key else None)
            _api_key  = _anthropic_key or _openai_key
            _cqo_ctx = {
                "n_obs": len(eq_r.dropna(how="all")), "date_range": f"{start} to {end}",
                "model": "Conflict-driven + regime-filtered trade ideas", "regime": r_name,
                "assumption_count": 5, "trade_has_stop": True,
                "notes": [
                    f"Current regime index: {current}/3 — {len(active_trades)} ideas active after filters",
                    f"{_n_geo} conflict-generated candidates, {_n_stat} static library ideas",
                    "Trade entry/exit levels are illustrative ranges, not live-calibrated prices",
                    "Correlation-based regime uses 60d rolling window - whipsaws in trending vol regimes",
                    "No walk-forward backtest - ideas are not validated against historical win rates",
                    "All ideas assume liquid markets - slippage and execution risk not modelled",
                ],
            }
            _cqo_run(_cqo_ctx, _provider, _api_key, page="Structured Trade Ideas")
    except Exception:
        pass

    _page_footer()
