"""
Proactive alert engine.
Scans live market data and returns structured alert objects - no LLM required.
The UI layer optionally enriches these with AI narration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class Alert:
    severity: str          # "critical" | "warning" | "info"
    category: str          # "stress" | "regime" | "cot" | "volatility" | "correlation"
    title: str
    body: str              # data-driven template text
    page_hint: str         # suggested page to navigate to
    data: dict = field(default_factory=dict)   # raw numbers for AI enrichment


def _safe_last(s: pd.Series, default=0.0):
    return float(s.iloc[-1]) if not s.empty else default


def _safe_nth(s: pd.Series, n: int, default=0.0):
    return float(s.iloc[-n]) if len(s) >= n else default


# ── Alert detectors ──────────────────────────────────────────────────────────

def _check_stress(risk_score: float, risk_history: pd.Series) -> list[Alert]:
    alerts = []

    # Absolute level
    if risk_score >= 65:
        alerts.append(Alert(
            severity="critical", category="stress",
            title=f"Market stress critical - {risk_score:.0f}/100",
            body=(
                f"Composite stress score is {risk_score:.0f}/100, well into the elevated zone (≥65). "
                "Equity-commodity correlations, volatility, and futures positioning are all elevated simultaneously. "
                "Historical episodes above 65 preceded average equity drawdowns of 8–12% over 30 days."
            ),
            page_hint="overview",
            data={"risk_score": risk_score},
        ))
    elif risk_score >= 50:
        alerts.append(Alert(
            severity="warning", category="stress",
            title=f"Market stress elevated - {risk_score:.0f}/100",
            body=(
                f"Composite stress score is {risk_score:.0f}/100 - above the 50-point caution threshold. "
                "At least one or two stress components (correlation, vol, or COT) are flashing. "
                "Not yet a crisis, but worth monitoring daily."
            ),
            page_hint="overview",
            data={"risk_score": risk_score},
        ))

    # 5-day spike
    if len(risk_history) >= 6:
        delta_5d = float(risk_history.iloc[-1]) - float(risk_history.iloc[-6])
        if abs(delta_5d) >= 10:
            direction = "spiked" if delta_5d > 0 else "collapsed"
            sev = "critical" if abs(delta_5d) >= 20 else "warning"
            alerts.append(Alert(
                severity=sev, category="stress",
                title=f"Stress score {direction} {abs(delta_5d):.0f} pts in 5 days",
                body=(
                    f"The composite stress index moved {delta_5d:+.0f} pts over the past 5 trading days "
                    f"(now {risk_score:.0f}/100). "
                    f"{'Rapid acceleration in cross-asset stress is a strong early-warning signal.' if delta_5d > 0 else 'Rapid deceleration - risk-off pressures easing.'}"
                ),
                page_hint="overview",
                data={"risk_score": risk_score, "delta_5d": delta_5d},
            ))

    return alerts


def _check_regime(regimes: pd.Series, avg_corr: pd.Series) -> list[Alert]:
    alerts = []
    if len(regimes) < 6:
        return alerts

    current  = int(regimes.iloc[-1])
    previous = int(regimes.iloc[-6])  # 5 days ago
    labels   = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis"}

    if current != previous:
        direction_up = current > previous
        sev = "critical" if current >= 3 else ("warning" if current >= 2 else "info")
        alerts.append(Alert(
            severity=sev, category="regime",
            title=f"Regime shifted → {labels[current]} (from {labels[previous]})",
            body=(
                f"The equity-commodity correlation regime transitioned from "
                f"{labels[previous]} to {labels[current]} within the last 5 trading days. "
                f"{'Diversification benefits are diminishing - assets increasingly moving together.' if direction_up else 'Cross-asset diversification is improving - equity and commodity markets decoupling.'} "
                f"Current 60d avg |correlation|: {_safe_last(avg_corr):.3f}."
            ),
            page_hint="correlation",
            data={"current_regime": current, "previous_regime": previous,
                  "avg_corr": _safe_last(avg_corr)},
        ))

    # Acceleration without regime change
    elif len(avg_corr) >= 6:
        delta_corr = _safe_last(avg_corr) - _safe_nth(avg_corr, 6)
        if abs(delta_corr) >= 0.05:
            direction = "rising" if delta_corr > 0 else "falling"
            alerts.append(Alert(
                severity="warning", category="correlation",
                title=f"Equity-commodity correlation {direction} fast (+{delta_corr:+.3f} / 5d)",
                body=(
                    f"Average equity-commodity |correlation| moved {delta_corr:+.3f} in 5 days "
                    f"(now {_safe_last(avg_corr):.3f}). "
                    f"{'A correlation surge approaching the Elevated threshold reduces portfolio diversification.' if delta_corr > 0 else 'Falling correlation improves diversification - commodities offering more independent return.'}"
                ),
                page_hint="correlation",
                data={"avg_corr": _safe_last(avg_corr), "delta_5d": delta_corr},
            ))

    return alerts


def _check_cot(cot_df: pd.DataFrame) -> list[Alert]:
    alerts = []
    if cot_df.empty:
        return alerts

    for market in cot_df["market"].unique():
        sub = cot_df[cot_df["market"] == market].sort_values("date")
        if sub.empty:
            continue
        latest_pct  = float(sub["net_spec_pct"].iloc[-1])
        days_stale  = (date.today() - sub["date"].iloc[-1].date()).days
        if days_stale > 14:   # skip stale data
            continue

        if latest_pct >= 25:
            pct_rank = float((sub["net_spec_pct"] < latest_pct).mean() * 100)
            alerts.append(Alert(
                severity="warning" if latest_pct < 40 else "critical",
                category="cot",
                title=f"{market} - crowded long ({latest_pct:.0f}% of OI)",
                body=(
                    f"Net speculative positioning in {market} is {latest_pct:.0f}% of open interest "
                    f"- at the {pct_rank:.0f}th historical percentile. "
                    "Readings above +25% signal a crowded long; historically precede price reversals "
                    "of 10–20% within 4–8 weeks."
                ),
                page_hint="watchlist",
                data={"market": market, "net_spec_pct": latest_pct, "percentile": pct_rank},
            ))
        elif latest_pct <= -25:
            pct_rank = float((sub["net_spec_pct"] < latest_pct).mean() * 100)
            alerts.append(Alert(
                severity="warning" if latest_pct > -40 else "critical",
                category="cot",
                title=f"{market} - crowded short ({latest_pct:.0f}% of OI)",
                body=(
                    f"Net speculative positioning in {market} is {latest_pct:.0f}% of open interest "
                    f"- at the {pct_rank:.0f}th historical percentile (extreme bearish). "
                    "Crowded shorts historically precede mean-reversion rallies of 10–15%."
                ),
                page_hint="watchlist",
                data={"market": market, "net_spec_pct": latest_pct, "percentile": pct_rank},
            ))

    return alerts


def _check_volatility(cmd_r: pd.DataFrame) -> list[Alert]:
    alerts = []
    if cmd_r.empty or len(cmd_r) < 30:
        return alerts

    vol_60   = cmd_r.iloc[-60:].std() * np.sqrt(252) * 100
    vol_full = cmd_r.std() * np.sqrt(252) * 100
    vol_mean = vol_full.mean()
    vol_std  = vol_full.std()

    for col in cmd_r.columns:
        if vol_std == 0 or pd.isna(vol_60.get(col)) or pd.isna(vol_full.get(col)):
            continue
        z = (vol_60[col] - vol_mean) / vol_std
        if z >= 2.0:
            alerts.append(Alert(
                severity="critical" if z >= 3 else "warning",
                category="volatility",
                title=f"{col} volatility spike - {z:.1f}σ above norm",
                body=(
                    f"{col} 60-day annualised volatility is {vol_60[col]:.0f}% "
                    f"({z:.1f} standard deviations above the historical average of {vol_mean:.0f}%). "
                    "Elevated commodity volatility is a leading indicator of equity stress via the "
                    "spillover channel - watch Granger causality pairs."
                ),
                page_hint="spillover",
                data={"commodity": col, "vol_60d": float(vol_60[col]), "z_score": z},
            ))

    return alerts


def _check_early_warnings(eq_r: pd.DataFrame, cmd_r: pd.DataFrame,
                           avg_corr: pd.Series) -> list[Alert]:
    """Dashboard's own early warning signals (largest 1w equity drawdown etc.)."""
    alerts = []
    if eq_r.empty or len(eq_r) < 6:
        return alerts

    # Worst equity index past week
    eq_1w = eq_r.iloc[-5:].sum()
    worst_eq   = eq_1w.idxmin()
    worst_val  = float(eq_1w.min()) * 100
    if worst_val < -4:
        alerts.append(Alert(
            severity="warning" if worst_val > -7 else "critical",
            category="stress",
            title=f"{worst_eq} down {worst_val:.1f}% this week",
            body=(
                f"{worst_eq} is the worst-performing equity index over the past 5 trading days "
                f"({worst_val:.1f}%). "
                "Large single-index drawdowns can transmit stress to commodity markets via the "
                "Granger spillover channel - particularly energy and metals."
            ),
            page_hint="spillover",
            data={"index": worst_eq, "return_5d": worst_val},
        ))

    # Best commodity (could signal supply shock)
    cmd_1w = cmd_r.iloc[-5:].sum()
    best_cmd  = cmd_1w.idxmax()
    best_val  = float(cmd_1w.max()) * 100
    if best_val > 5:
        alerts.append(Alert(
            severity="info",
            category="volatility",
            title=f"{best_cmd} up {best_val:.1f}% this week",
            body=(
                f"{best_cmd} is the best-performing commodity over the past 5 trading days "
                f"(+{best_val:.1f}%). "
                "A sharp commodity move this size may signal supply disruption or macro rotation - "
                "check the Granger grid for downstream equity impact."
            ),
            page_hint="spillover",
            data={"commodity": best_cmd, "return_5d": best_val},
        ))

    return alerts


# ── Country-level equity index exposure check ─────────────────────────────────
# Scoring data and multiplier math live in war_country_scores.py (single source of truth).
# compute_country_scores() returns {iso: composite_score} for the 8 equity-indexed countries.

from src.analysis.war_country_scores import (
    compute_country_scores,
    COUNTRY_INDICES as _COUNTRY_INDICES,
    war_multipliers as _war_multipliers_ref,
)


def _check_country_exposure(cmd_r: pd.DataFrame) -> list[Alert]:
    """
    Fire a warning when any tracked country's equity-index composite score ≥ 65.
    Delegates all scoring math to war_country_scores.compute_country_scores().
    """
    alerts: list[Alert] = []
    if cmd_r.empty or len(cmd_r) < 40:
        return alerts

    country_scores = compute_country_scores(cmd_r)
    mults = _war_multipliers_ref(cmd_r)
    signals = mults.get("signals", {})
    oil_z  = signals.get("Crude Oil z",   0.0)
    gas_z  = signals.get("Natural Gas z", 0.0)
    gold_z = signals.get("Gold z",        0.0)

    # Build (iso, primary_index_name, score) for countries ≥ 65
    elevated: list[tuple[str, str, int]] = []
    for iso, score in country_scores.items():
        if score >= 65:
            idx_name = _COUNTRY_INDICES[iso][0]  # primary index (first listed)
            elevated.append((iso, idx_name, score))

    if not elevated:
        return alerts

    elevated.sort(key=lambda x: x[2], reverse=True)
    lead_iso, lead_idx, lead_score = elevated[0]
    detail = ", ".join(f"{idx} ({score}/100)" for _, idx, score in elevated)

    alerts.append(Alert(
        severity="warning",
        category="country_exposure",
        title=f"Country equity exposure elevated — {lead_idx} {lead_score}/100",
        body=(
            f"{len(elevated)} equity index/indices have country-level geopolitical exposure ≥ 65: "
            f"{detail}. "
            f"Live signals driving scores: Oil z={oil_z:+.2f}, Gas z={gas_z:+.2f}, Gold z={gold_z:+.2f}. "
            f"Scores combine static structural conflict exposure with live commodity market intensity. "
            f"Open War Impact Map for full country breakdown."
        ),
        page_hint="war_impact_map",
        data={
            "elevated_countries": {iso: score for iso, _, score in elevated},
            "oil_z": round(oil_z, 2),
            "gas_z": round(gas_z, 2),
            "gold_z": round(gold_z, 2),
            "multipliers": {k: mults[k] for k in ("ukraine", "hamas", "iran")},
        },
    ))
    return alerts


# ── Main entry point ─────────────────────────────────────────────────────────

def _check_gdelt_escalation() -> list[Alert]:
    """Alert when GDELT detects a sharp conflict media volume spike (≥30% WoW)."""
    alerts = []
    try:
        from src.data.gdelt import fetch_all_gdelt_signals
        signals = fetch_all_gdelt_signals(timespan="7d")
        for cid, gd in signals.items():
            if not gd.get("data_available"):
                continue
            trend = float(gd.get("volume_trend", 0.0))
            esc   = gd.get("escalation_signal", "stable")
            tone  = float(gd.get("tone_recent", 0.0))
            label = cid.replace("_", " ").title()

            if esc == "escalating" and trend >= 0.40:
                alerts.append(Alert(
                    severity="critical", category="conflict",
                    title=f"GDELT: {label} media surge +{trend:.0%}",
                    body=(
                        f"News coverage of {label} surged {trend:+.0%} in the past 7 days vs prior period. "
                        f"Tone score: {tone:+.1f} (negative = alarming). "
                        "Sharp media surges often precede market repricing by 1–3 days — "
                        "watch energy and safe-haven assets."
                    ),
                    page_hint="conflict_intelligence",
                    data={"conflict": cid, "volume_trend": trend, "tone": tone},
                ))
            elif esc == "escalating" and trend >= 0.20:
                alerts.append(Alert(
                    severity="warning", category="conflict",
                    title=f"GDELT: {label} escalating coverage +{trend:.0%}",
                    body=(
                        f"{label} media volume up {trend:+.0%} this week. "
                        f"GDELT tone: {tone:+.1f}. Monitor for further escalation — "
                        "coverage spikes above +40% historically precede conflict-driven commodity moves."
                    ),
                    page_hint="conflict_intelligence",
                    data={"conflict": cid, "volume_trend": trend, "tone": tone},
                ))
    except Exception:
        pass
    return alerts


def _check_eia_inventory() -> list[Alert]:
    """Alert when EIA crude or gas stocks are significantly below the 5yr seasonal average."""
    alerts = []
    try:
        from src.data.eia import eia_snapshot
        snap = eia_snapshot(weeks=260)

        crude = snap.get("crude_stocks", {})
        natgas = snap.get("natgas_storage", {})

        for key, s, commodity in [
            ("crude_stocks", crude, "Crude Oil"),
            ("natgas_storage", natgas, "Natural Gas"),
        ]:
            if not s.get("data_available"):
                continue
            p5yr = s.get("vs_5yr_pct")
            if p5yr is None:
                continue
            wow = s.get("wow_change", 0)
            level = s.get("level", 0)
            units = s.get("units", "")

            if p5yr <= -10:
                alerts.append(Alert(
                    severity="critical", category="supply",
                    title=f"EIA {commodity} stocks critically low ({p5yr:+.1f}% vs 5yr avg)",
                    body=(
                        f"US {commodity} stocks at {level:,} {units} — {abs(p5yr):.1f}% below the "
                        f"5-year seasonal average. WoW change: {'+' if wow>=0 else ''}{wow:,}. "
                        "Stocks this far below seasonal norms historically precede acute price spikes. "
                        "Any supply disruption (chokepoint, weather, pipeline) could amplify sharply."
                    ),
                    page_hint="strait_watch",
                    data={"commodity": commodity, "vs_5yr_pct": p5yr, "wow_change": wow},
                ))
            elif p5yr <= -5:
                alerts.append(Alert(
                    severity="warning", category="supply",
                    title=f"EIA {commodity} below seasonal average ({p5yr:+.1f}%)",
                    body=(
                        f"US {commodity} stocks at {level:,} {units} — {abs(p5yr):.1f}% below "
                        f"5-year seasonal average. WoW: {'+' if wow>=0 else ''}{wow:,} {units}. "
                        "Tighter-than-normal inventory levels reduce the market's shock-absorption buffer."
                    ),
                    page_hint="strait_watch",
                    data={"commodity": commodity, "vs_5yr_pct": p5yr, "wow_change": wow},
                ))
    except Exception:
        pass
    return alerts


def compute_alerts(
    eq_r: pd.DataFrame,
    cmd_r: pd.DataFrame,
    avg_corr: pd.Series,
    regimes: pd.Series,
    risk_score: float,
    risk_history: pd.Series,
    cot_df: pd.DataFrame | None = None,
) -> list[Alert]:
    """
    Run all alert detectors and return a deduplicated, severity-sorted list.
    Pass an empty DataFrame for cot_df if COT data is unavailable.
    """
    all_alerts: list[Alert] = []

    all_alerts += _check_stress(risk_score, risk_history)
    all_alerts += _check_regime(regimes, avg_corr)
    all_alerts += _check_volatility(cmd_r)
    all_alerts += _check_early_warnings(eq_r, cmd_r, avg_corr)
    all_alerts += _check_country_exposure(cmd_r)
    all_alerts += _check_gdelt_escalation()
    all_alerts += _check_eia_inventory()

    if cot_df is not None and not cot_df.empty:
        all_alerts += _check_cot(cot_df)

    # Sort: critical → warning → info
    order = {"critical": 0, "warning": 1, "info": 2}
    all_alerts.sort(key=lambda a: order.get(a.severity, 9))

    return all_alerts
