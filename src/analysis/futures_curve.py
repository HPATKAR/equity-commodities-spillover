"""
Commodity Futures Curve Analysis — backwardation / contango detection.

Fetches front-month and deferred contract prices via yfinance and computes:
  - Curve slope (spot vs 6M contract): negative = backwardation, positive = contango
  - Structure label: "Backwardation" / "Contango" / "Flat"
  - Signal for geopolitical corroboration: if a high GRS score coincides with
    backwardation, the market structure corroborates the geopolitical signal.
  - Historical curve slope over time (rolling basis)

Ticker map uses liquid ETF proxies available on yfinance:
  WTI:        CL=F  (front-month)  vs  CLM25.NYM or USL (12M blend)
  Brent:      BZ=F  (front-month)  vs  deferred via ICE
  Nat Gas:    NG=F  (front-month)  vs  NGQ25.NYM
  Gold:       GC=F  (front-month)  vs  GCZ25.CMX
  Wheat:      ZW=F  (front-month)  vs  ZWZ25.CBT
  Copper:     HG=F  (front-month)  vs  HGZ25.CMX

Because deferred contract month codes rotate, we use the simplest approach:
fetch the two most liquid futures contracts and compute the calendar spread.
"""

from __future__ import annotations

import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


# ── Futures contract pairs ──────────────────────────────────────────────────

_CURVE_PAIRS: list[dict] = [
    {
        "name":       "WTI Crude Oil",
        "front":      "CL=F",          # front-month WTI
        "deferred":   "CLQ25.NYM",     # 6-month deferred (Aug 2025)
        "group":      "Energy",
        "unit":       "$/bbl",
        "color":      "#d35400",
    },
    {
        "name":       "Brent Crude",
        "front":      "BZ=F",          # front-month Brent
        "deferred":   "BRNQ25.NYM",    # 6-month deferred
        "group":      "Energy",
        "unit":       "$/bbl",
        "color":      "#e74c3c",
    },
    {
        "name":       "Natural Gas",
        "front":      "NG=F",          # front-month Henry Hub
        "deferred":   "NGQ25.NYM",     # ~6-month deferred
        "group":      "Energy",
        "unit":       "$/MMBtu",
        "color":      "#3498db",
    },
    {
        "name":       "Gold",
        "front":      "GC=F",          # front-month gold
        "deferred":   "GCZ25.CMX",     # Dec 2025 gold
        "group":      "Precious Metals",
        "unit":       "$/oz",
        "color":      "#CFB991",
    },
    {
        "name":       "Wheat",
        "front":      "ZW=F",          # front-month CBOT wheat
        "deferred":   "ZWZ25.CBT",     # Dec 2025 wheat
        "group":      "Agriculture",
        "unit":       "¢/bu",
        "color":      "#2e7d32",
    },
    {
        "name":       "Copper",
        "front":      "HG=F",          # front-month copper
        "deferred":   "HGZ25.CMX",     # Dec 2025 copper
        "group":      "Industrial Metals",
        "unit":       "$/lb",
        "color":      "#7f8c8d",
    },
]

# Annualised roll cost threshold for structure classification (as % of spot)
_BACKWARDATION_THRESHOLD = -0.5    # curve slope < -0.5% → backwardation
_CONTANGO_THRESHOLD      =  0.5    # curve slope >  0.5% → contango


# ── Data fetch ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_curve_snapshot() -> pd.DataFrame:
    """
    Fetch latest front-month and 6-month deferred prices for each commodity.

    Returns DataFrame with columns:
      name, front_price, deferred_price, basis_pct, structure,
      structure_color, corroboration_signal, group, unit, color

    basis_pct = (deferred − front) / front × 100
      Negative = backwardation (supply shock pricing)
      Positive = contango (oversupply / storage cost)
    """
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()

    rows = []
    for pair in _CURVE_PAIRS:
        try:
            tickers = [pair["front"], pair["deferred"]]
            data    = yf.download(tickers, period="5d", auto_adjust=True,
                                  progress=False, show_errors=False)["Close"]
            if data.empty:
                continue

            front_col    = pair["front"]
            deferred_col = pair["deferred"]

            # Some deferred contracts may not trade — fall back to NaN
            if front_col not in data.columns or deferred_col not in data.columns:
                # Try scalar download if only one ticker returned
                continue

            last_row = data.dropna(how="all").iloc[-1]
            front_px    = float(last_row.get(front_col,    np.nan))
            deferred_px = float(last_row.get(deferred_col, np.nan))

            if not np.isfinite(front_px) or front_px <= 0:
                continue
            if not np.isfinite(deferred_px) or deferred_px <= 0:
                # If deferred is unavailable, skip rather than show bad data
                continue

            basis_pct = (deferred_px - front_px) / front_px * 100.0

            if basis_pct < _BACKWARDATION_THRESHOLD:
                structure       = "Backwardation"
                structure_color = "#c0392b"
                # Backwardation + geopolitically stressed commodity = corroboration
                corroboration_signal = "Corroborates" if pair["group"] == "Energy" else "Neutral"
            elif basis_pct > _CONTANGO_THRESHOLD:
                structure       = "Contango"
                structure_color = "#27ae60"
                corroboration_signal = "Contradicts" if pair["group"] == "Energy" else "Neutral"
            else:
                structure       = "Flat"
                structure_color = "#CFB991"
                corroboration_signal = "Inconclusive"

            rows.append({
                "name":                 pair["name"],
                "front_price":          round(front_px, 2),
                "deferred_price":       round(deferred_px, 2),
                "basis_pct":            round(basis_pct, 3),
                "structure":            structure,
                "structure_color":      structure_color,
                "corroboration_signal": corroboration_signal,
                "group":                pair["group"],
                "unit":                 pair["unit"],
                "color":                pair["color"],
                "front_ticker":         front_col,
                "deferred_ticker":      deferred_col,
            })
        except Exception:
            continue

    return pd.DataFrame(rows) if rows else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_rolling_basis(
    commodity_name: str,
    period: str = "6mo",
) -> pd.DataFrame:
    """
    Fetch rolling daily basis (deferred − front) / front for one commodity.

    Returns DataFrame with DatetimeIndex and columns: front, deferred, basis_pct.
    basis_pct negative = backwardation.
    """
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()

    pair = next((p for p in _CURVE_PAIRS if p["name"] == commodity_name), None)
    if pair is None:
        return pd.DataFrame()

    try:
        tickers = [pair["front"], pair["deferred"]]
        data    = yf.download(tickers, period=period, auto_adjust=True,
                              progress=False, show_errors=False)["Close"]
        if data.empty:
            return pd.DataFrame()

        front_col    = pair["front"]
        deferred_col = pair["deferred"]

        if front_col not in data.columns or deferred_col not in data.columns:
            return pd.DataFrame()

        aligned = data[[front_col, deferred_col]].dropna()
        if aligned.empty:
            return pd.DataFrame()

        result = pd.DataFrame({
            "front":     aligned[front_col],
            "deferred":  aligned[deferred_col],
            "basis_pct": (aligned[deferred_col] - aligned[front_col]) / aligned[front_col] * 100.0,
        })
        return result

    except Exception:
        return pd.DataFrame()


def geopolitical_corroboration(
    curve_df: pd.DataFrame,
    grs_score: float,
    grs_threshold: float = 55.0,
) -> dict:
    """
    Check whether commodity market structure corroborates the current GRS signal.

    A high GRS score claiming supply disruption should coincide with energy
    commodities being in backwardation (near-term tightness priced in).
    If GRS is high but energy is in contango, the market structure contradicts
    the geopolitical narrative — worth flagging.

    Returns dict with:
      overall_signal   : "Corroborated" / "Contradicted" / "Inconclusive"
      energy_structure : list of (name, structure) for energy commodities
      detail           : str — human-readable corroboration summary
    """
    if curve_df.empty:
        return {
            "overall_signal": "Inconclusive",
            "energy_structure": [],
            "detail": "No futures curve data available for corroboration check.",
        }

    energy = curve_df[curve_df["group"] == "Energy"]
    if energy.empty:
        return {
            "overall_signal": "Inconclusive",
            "energy_structure": [],
            "detail": "No energy futures curve data available.",
        }

    n_backwardation = len(energy[energy["structure"] == "Backwardation"])
    n_contango      = len(energy[energy["structure"] == "Contango"])
    n_total         = len(energy)

    energy_structures = list(zip(energy["name"].tolist(), energy["structure"].tolist()))

    high_grs = grs_score >= grs_threshold

    if high_grs and n_backwardation >= n_total / 2:
        signal = "Corroborated"
        detail = (
            f"GRS={grs_score:.0f} (elevated). Energy futures in backwardation "
            f"({n_backwardation}/{n_total}): market is pricing near-term supply tightness. "
            f"Geopolitical risk signal and market structure are consistent."
        )
    elif high_grs and n_contango >= n_total / 2:
        signal = "Contradicted"
        detail = (
            f"GRS={grs_score:.0f} (elevated) BUT energy in contango "
            f"({n_contango}/{n_total}): market structure implies oversupply, "
            f"not the scarcity the GRS signal suggests. "
            f"Market may be pricing geopolitical risk in volatility, not spot. "
            f"Consider VIX and options skew rather than outrights."
        )
    elif not high_grs and n_backwardation >= n_total / 2:
        signal = "Inconclusive"
        detail = (
            f"GRS={grs_score:.0f} (low), but energy in backwardation "
            f"({n_backwardation}/{n_total}). Supply tightness may be demand-driven, "
            f"not geopolitical — or the GRS may be lagging market pricing."
        )
    else:
        signal = "Inconclusive"
        detail = (
            f"GRS={grs_score:.0f}. Mixed or flat energy curve "
            f"(backwardation: {n_backwardation}, contango: {n_contango}, "
            f"flat: {n_total - n_backwardation - n_contango}). "
            f"No clear corroboration or contradiction."
        )

    return {
        "overall_signal":   signal,
        "energy_structure": energy_structures,
        "detail":           detail,
    }
