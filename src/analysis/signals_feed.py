"""
Signals feed - assembles the terminal's headline signals into one versioned,
machine-readable payload.

This is the single source a data-feed product would serve. The Signals Export page
downloads exactly what build_signals_payload() returns, and a future HTTP endpoint
(GET /v1/signals) would return the identical dict. Keeping the assembly here, pure
and Streamlit-free, is what makes that future endpoint a thin wrapper rather than a
rewrite. Every section is guarded so one failing signal degrades to null rather
than taking the whole feed down.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

SCHEMA = "spillover-signals/v1"

_RNAME = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis"}


def _commodity_gsi_table(scores: dict, matrix: dict) -> list[dict]:
    """Per-commodity Geopolitical Stress Index: noisy-OR union of each active
    conflict's transmission relevance to the commodity with its live CIS."""
    out = []
    commodities = sorted({c for row in matrix.values() for c in row})
    for name in commodities:
        prod, best, drv = 1.0, 0.0, None
        for cid, row in matrix.items():
            s = scores.get(cid, {})
            if s.get("state", "active") != "active":
                continue
            c = float(row.get(name, 0.0)) * float(s.get("cis", 0.0)) / 100.0
            if c > 0:
                prod *= (1.0 - min(c, 0.999))
            if c > best:
                best, drv = c, s.get("name", cid)
        gsi = round((1.0 - prod) * 100.0, 1)
        if gsi > 0:
            out.append({"commodity": name, "gsi": gsi, "top_driver": drv})
    out.sort(key=lambda d: d["gsi"], reverse=True)
    return out


def build_signals_payload(start: str, end: str) -> dict:
    """Assemble the full signals payload. Pure function: no Streamlit, no I/O beyond
    the terminal's own cached data loaders. Safe to call from an HTTP handler."""
    payload: dict = {
        "schema": SCHEMA,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of_data": None,
        "source": "Cross-Asset Spillover Monitor - Purdue Daniels MSF Research Terminal",
        "provenance": ("Public data (Yahoo Finance, FRED, IMF PortWatch, GDELT, "
                       "CFTC). Research and educational use; NOT licensed for "
                       "institutional redistribution."),
        "disclaimer": ("Signals are model estimates for research and risk "
                       "monitoring. Not investment advice."),
        "market": None,
        "conflict": None,
        "commodity_gsi": None,
    }

    # ── Market: stress, regime, correlation, connectedness ───────────────────
    eq_r = cmd_r = None
    try:
        from src.data.loader import load_returns
        from src.analysis.correlations import (average_cross_corr_series,
                                               detect_correlation_regime)
        from src.analysis.risk_score import compute_risk_score
        eq_r, cmd_r = load_returns(start, end)
        if eq_r is not None and cmd_r is not None and not eq_r.empty and not cmd_r.empty:
            ac = average_cross_corr_series(eq_r, cmd_r, window=60)
            reg = detect_correlation_regime(ac)
            try:
                _rs = compute_risk_score(ac, cmd_r, eq_r=eq_r)
                stress = float(_rs.get("score", 0.0)) if isinstance(_rs, dict) else float(_rs)
            except Exception:
                stress = None
            _lvl = int(reg.iloc[-1]) if len(reg) else 1
            payload["as_of_data"] = str(cmd_r.index[-1].date()) if len(cmd_r) else None
            payload["market"] = {
                "stress_score": round(stress, 1) if stress is not None else None,
                "regime_level": _lvl,
                "regime_label": _RNAME.get(_lvl, "Normal"),
                "avg_cross_asset_corr_60d": round(float(ac.iloc[-1]), 4) if len(ac) else None,
                "dy_total_connectedness_pct": None,
            }
            # Diebold-Yilmaz total connectedness on the combined asset universe.
            try:
                from src.analysis.spillover import diebold_yilmaz
                combined = pd.concat([eq_r, cmd_r], axis=1).dropna()
                if combined.shape[1] >= 3 and combined.shape[0] > 200:
                    dy = diebold_yilmaz(combined, top_n=combined.shape[1])
                    payload["market"]["dy_total_connectedness_pct"] = round(
                        float(dy.get("total_spillover", 0.0)), 2)
                    payload["market"]["dy_universe_n"] = int(combined.shape[1])
            except Exception:
                pass
    except Exception:
        pass

    # ── Conflict intensity: portfolio + per-conflict CIS / TPS ───────────────
    try:
        from src.analysis.conflict_model import (score_all_conflicts,
                                                 aggregate_portfolio_scores,
                                                 conflict_commodity_matrix)
        scores = score_all_conflicts()
        agg = aggregate_portfolio_scores()
        conflicts = []
        for cid, s in scores.items():
            conflicts.append({
                "id": s.get("id", cid),
                "name": s.get("name"),
                "region": s.get("region"),
                "state": s.get("state", "active"),
                "cis": s.get("cis"),
                "tps": s.get("tps"),
                "confidence": s.get("confidence"),
                "trend": s.get("trend"),
                "escalation": s.get("escalation"),
            })
        conflicts.sort(key=lambda c: (c.get("cis") or 0), reverse=True)
        payload["conflict"] = {
            "portfolio_cis": agg.get("portfolio_cis"),
            "portfolio_tps": agg.get("portfolio_tps"),
            "confidence": agg.get("confidence"),
            "top_conflict": agg.get("top_conflict"),
            "n_active": sum(1 for c in conflicts if c["state"] == "active"),
            "conflicts": conflicts,
        }
        try:
            payload["commodity_gsi"] = _commodity_gsi_table(
                scores, conflict_commodity_matrix())
        except Exception:
            pass
    except Exception:
        pass

    return payload


def payload_to_long_rows(payload: dict) -> list[dict]:
    """Flatten the nested payload into tidy long rows for CSV:
    {category, entity, metric, value, as_of}. Universal, join-friendly shape."""
    asof = payload.get("as_of_data") or ""
    rows: list[dict] = []

    def _add(cat, entity, metric, value):
        if value is None:
            return
        rows.append({"category": cat, "entity": entity, "metric": metric,
                     "value": value, "as_of": asof})

    mk = payload.get("market") or {}
    for m in ("stress_score", "regime_level", "regime_label",
              "avg_cross_asset_corr_60d", "dy_total_connectedness_pct"):
        _add("market", "market", m, mk.get(m))

    cf = payload.get("conflict") or {}
    for m in ("portfolio_cis", "portfolio_tps", "confidence", "n_active"):
        _add("conflict", "portfolio", m, cf.get(m))
    for c in cf.get("conflicts", []) or []:
        ent = c.get("name") or c.get("id") or "conflict"
        for m in ("cis", "tps", "confidence", "state", "trend", "escalation"):
            _add("conflict", ent, m, c.get(m))

    for g in payload.get("commodity_gsi") or []:
        _add("commodity_gsi", g.get("commodity"), "gsi", g.get("gsi"))
        _add("commodity_gsi", g.get("commodity"), "top_driver", g.get("top_driver"))

    return rows


# ── Field documentation (drives the export page's schema table + future API docs)
FIELD_DOCS: list[tuple[str, str, str]] = [
    ("market.stress_score", "0-100", "Composite cross-asset stress index (correlation, volatility, positioning)."),
    ("market.regime_level", "0-3", "Correlation regime: 0 Decorrelated, 1 Normal, 2 Elevated, 3 Crisis."),
    ("market.avg_cross_asset_corr_60d", "0-1", "Mean absolute equity-commodity correlation, 60-day window."),
    ("market.dy_total_connectedness_pct", "0-100", "Diebold-Yilmaz (2012) total connectedness, generalized FEVD, order-invariant."),
    ("conflict.portfolio_cis", "0-100", "Breadth-weighted aggregate Conflict Intensity Score across active conflicts."),
    ("conflict.portfolio_tps", "0-100", "Aggregate Transmission Propensity Score (how strongly conflicts reach markets)."),
    ("conflict.conflicts[].cis", "0-100", "Per-conflict intensity (ACLED / GDELT where live, else scenario baseline)."),
    ("conflict.conflicts[].tps", "0-100", "Per-conflict market-transmission propensity."),
    ("commodity_gsi[].gsi", "0-100", "Per-commodity Geopolitical Stress Index: noisy-OR of transmission x CIS."),
]
