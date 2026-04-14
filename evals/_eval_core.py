"""
Core quantitative model runner for the standalone eval script.
Duplicates the essential computation from run_historical_snapshot() in
agent_benchmark.py — kept here to remain Streamlit-free and importable
without side effects.

Provides:
    run_snapshot(date_str) -> dict   # model output for a given date
"""
from __future__ import annotations

import datetime


def run_snapshot(date_str: str) -> dict:
    """
    Load 252 trading days of price data ending at date_str, run the
    quantitative model, and return typed output fields.

    Fields returned:
        risk_score, regime, corr_pct, cmd_vol_z,
        yield_curve_spread, granger_hit_rate (None — not back-testable),
        cis, tps (None — set by caller from VIX proxy),
        _computed (bool), _error (str | None)
    """
    import yfinance as yf
    import pandas as pd
    import numpy as np

    output: dict = {
        "risk_score":         None,
        "regime":             None,
        "corr_pct":           None,
        "cmd_vol_z":          None,
        "yield_curve_spread": None,
        "cpi_yoy":            None,
        "granger_hit_rate":   None,
        "cis":                None,
        "tps":                None,
        "_computed":          False,
        "_error":             None,
    }

    try:
        end_dt   = datetime.date.fromisoformat(date_str)
        start_dt = end_dt - datetime.timedelta(days=400)
        end_dl   = end_dt + datetime.timedelta(days=1)  # yfinance exclusive end

        # ── Download equity + commodity prices ────────────────────────────────
        equity_tickers    = ["SPY", "EFA", "EEM", "QQQ", "IWM"]
        commodity_tickers = ["GLD", "USO", "DBA", "PDBC", "CPER"]
        tlt_shy           = ["TLT", "SHY"]

        all_tickers = equity_tickers + commodity_tickers + tlt_shy
        raw = yf.download(
            all_tickers,
            start=str(start_dt), end=str(end_dl),
            auto_adjust=True, progress=False,
        )

        if raw.empty:
            output["_error"] = "No data returned from yfinance"
            return output

        prices = raw["Close"] if "Close" in raw.columns else raw
        prices = prices.ffill().dropna(how="all")

        if len(prices) < 60:
            output["_error"] = f"Insufficient data: {len(prices)} rows"
            return output

        eq_cols  = [c for c in equity_tickers    if c in prices.columns]
        cmd_cols = [c for c in commodity_tickers if c in prices.columns]

        if not eq_cols or not cmd_cols:
            output["_error"] = "Missing equity or commodity columns"
            return output

        eq_r  = prices[eq_cols].pct_change().dropna()
        cmd_r = prices[cmd_cols].pct_change().dropna()

        # ── 60-day rolling average cross-asset correlation ─────────────────
        window = min(60, len(eq_r) - 1)
        corr_vals = []
        for eq in eq_cols:
            for cmd in cmd_cols:
                combined = pd.concat([eq_r[eq], cmd_r[cmd]], axis=1).dropna()
                if len(combined) < window:
                    continue
                roll = combined.iloc[-window:].corr().iloc[0, 1]
                corr_vals.append(abs(roll))

        avg_corr = float(np.mean(corr_vals)) if corr_vals else 0.30

        # ── Regime classification ─────────────────────────────────────────
        # 60th / 80th percentile thresholds from the full 400-day window
        long_corr = []
        n = len(eq_r)
        step = max(window, 10)
        for i in range(window, n, step):
            vals = []
            sub_eq  = eq_r.iloc[max(0, i-window):i]
            sub_cmd = cmd_r.iloc[max(0, i-window):i]
            for eq in eq_cols:
                for cmd in cmd_cols:
                    c = pd.concat([sub_eq[eq], sub_cmd[cmd]], axis=1).dropna()
                    if len(c) >= 20:
                        vals.append(abs(c.corr().iloc[0, 1]))
            if vals:
                long_corr.append(float(np.mean(vals)))

        if long_corr and len(long_corr) >= 3:
            p60 = float(np.percentile(long_corr, 60))
            p80 = float(np.percentile(long_corr, 80))
        else:
            p60, p80 = 0.35, 0.50

        regime = 1
        if avg_corr >= p80:
            regime = 3
        elif avg_corr >= p60:
            regime = 2

        output["regime"]   = regime
        output["corr_pct"] = round(avg_corr * 100, 1)

        # ── Risk score (simplified MCS proxy) ────────────────────────────
        # Component 1: correlation level (0-40 pts)
        corr_score = min(avg_corr * 80, 40.0)

        # Component 2: recent equity volatility (0-35 pts)
        eq_vol = float(eq_r.iloc[-20:].std().mean()) * (252 ** 0.5)
        vol_score = min(eq_vol * 250, 35.0)

        # Component 3: commodity volatility relative to equity vol (0-25 pts)
        cmd_vol = float(cmd_r.iloc[-20:].std().mean()) * (252 ** 0.5)
        ratio   = cmd_vol / max(eq_vol, 0.001)
        ratio_score = min(ratio * 12, 25.0)

        risk_score = round(corr_score + vol_score + ratio_score, 1)
        output["risk_score"] = min(max(risk_score, 0.0), 100.0)

        # cmd_vol_z: standardise commodity vol vs. full-window mean
        cmd_vols_history = [
            float(cmd_r.iloc[max(0,i-20):i].std().mean()) * (252**0.5)
            for i in range(30, len(cmd_r), 10) if i <= len(cmd_r)
        ]
        if len(cmd_vols_history) >= 5:
            mu  = float(np.mean(cmd_vols_history))
            sig = float(np.std(cmd_vols_history)) or 0.01
            output["cmd_vol_z"] = round((cmd_vol - mu) / sig, 2)

        output["_computed"] = True

        # ── Yield curve proxy (TLT vs SHY 60d cumulative return spread) ──
        if "TLT" in prices.columns and "SHY" in prices.columns:
            tlt_r = prices["TLT"].pct_change().dropna()
            shy_r = prices["SHY"].pct_change().dropna()
            if len(tlt_r) >= 60:
                tlt_60 = float(tlt_r.iloc[-60:].sum() * 100)
                shy_60 = float(shy_r.iloc[-60:].sum() * 100)
                output["yield_curve_spread"] = round(tlt_60 - shy_60, 2)

    except Exception as e:
        output["_error"] = str(e)

    return output


def evaluate_assertions(output: dict, assertions: list) -> list[dict]:
    """Check a list of [field, comparator, expected] assertions against output."""
    _CMP = {
        "gt":      lambda a, b: a > b,
        "lt":      lambda a, b: a < b,
        "gte":     lambda a, b: a >= b,
        "lte":     lambda a, b: a <= b,
        "eq":      lambda a, b: a == b,
        "between": lambda a, b: b[0] <= a <= b[1],
    }
    results = []
    for field, cmp_op, expected in assertions:
        actual = output.get(field)
        try:
            fn = _CMP.get(cmp_op)
            passed = bool(fn(actual, expected)) if (fn and actual is not None) else False
        except Exception:
            passed = False
        results.append({"field": field, "comparator": cmp_op,
                        "expected": expected, "actual": actual, "passed": passed})
    return results
