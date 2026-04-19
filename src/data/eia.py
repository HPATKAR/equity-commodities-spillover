"""
EIA (Energy Information Administration) live weekly inventory data.

No API key required for the series used here.
API documentation: https://www.eia.gov/opendata/

Key series:
  PET.WCRSTUS1.W   — US crude oil stocks (weekly, thousand barrels)
  PET.WGTSTUS1.W   — US total gasoline stocks
  NG.NW2_EPG0_SWO_R48_BCF.W — US natural gas working storage (BCF)
  PET.WDIMUSTAL1.W — US crude oil imports (weekly, mb/d)

Endpoint: https://api.eia.gov/v2/seriesid/{series_id}?api_key=DEMO_KEY
  DEMO_KEY works for ~30 req/min from non-Streamlit Cloud environments.
  Add real key in .streamlit/secrets.toml:
    [keys]
    eia_api_key = "your_key"

Returned DataFrame columns: date, value, units, series_name
All series are weekly. We compute:
  - Level (current stock)
  - Week-on-week change
  - Year-on-year % change (same week prior year)
  - 5-year seasonal average (same week, 5yr average)
  - Surplus/deficit vs 5yr average
"""

from __future__ import annotations

import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

_EIA_BASE = "https://api.eia.gov/v2/seriesid"

_SERIES: dict[str, dict] = {
    "crude_stocks": {
        "id":    "PET.WCRSTUS1.W",
        "name":  "US Crude Oil Stocks",
        "units": "thousand barrels",
        "label": "Crude Stocks",
    },
    "gasoline_stocks": {
        "id":    "PET.WGTSTUS1.W",
        "name":  "US Total Gasoline Stocks",
        "units": "thousand barrels",
        "label": "Gasoline Stocks",
    },
    "natgas_storage": {
        "id":    "NG.NW2_EPG0_SWO_R48_BCF.W",
        "name":  "US Natural Gas Working Storage",
        "units": "BCF",
        "label": "NatGas Storage",
    },
    "crude_imports": {
        "id":    "PET.WCRIMUS2.W",
        "name":  "US Crude Oil Imports",
        "units": "thousand barrels per day",
        "label": "Crude Imports",
    },
    "distillate_stocks": {
        "id":    "PET.WDISTUS1.W",
        "name":  "US Distillate Fuel Stocks",
        "units": "thousand barrels",
        "label": "Distillate Stocks",
    },
}


def _get_eia_key() -> str:
    """Return EIA API key from secrets, or DEMO_KEY."""
    try:
        return st.secrets.get("keys", {}).get("eia_api_key", "") or "DEMO_KEY"
    except Exception:
        return "DEMO_KEY"


@st.cache_data(ttl=21600, show_spinner=False)  # 6-hour cache (EIA updates Wednesdays)
def fetch_eia_series(series_key: str, weeks: int = 260) -> pd.DataFrame:
    """
    Fetch a single EIA weekly series.

    Parameters
    ----------
    series_key : str — key from _SERIES dict (e.g., "crude_stocks")
    weeks      : int — number of weeks to fetch (default 260 = 5 years)

    Returns
    -------
    DataFrame with columns: date, value, units, series_name
    Returns empty DataFrame on failure.
    """
    if series_key not in _SERIES:
        return pd.DataFrame()

    cfg    = _SERIES[series_key]
    api_key = _get_eia_key()
    start   = (datetime.date.today() - datetime.timedelta(weeks=weeks)).isoformat()

    try:
        import requests
        url = f"{_EIA_BASE}/{cfg['id']}"
        params = {
            "api_key": api_key,
            "data[]":  "value",
            "start":   start,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length":  weeks + 5,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        rows = data.get("response", {}).get("data", [])
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["date"]  = pd.to_datetime(df["period"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
        df["units"]       = cfg["units"]
        df["series_name"] = cfg["name"]

        return df[["date", "value", "units", "series_name"]]

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_all_eia_inventory(weeks: int = 260) -> dict[str, pd.DataFrame]:
    """Fetch all tracked EIA series. Returns {series_key: DataFrame}."""
    return {k: fetch_eia_series(k, weeks=weeks) for k in _SERIES}


def eia_snapshot(weeks: int = 260) -> dict[str, dict]:
    """
    Compute a summary snapshot for each EIA series:
      level, wow_change, wow_pct, yoy_pct, vs_5yr_avg, vs_5yr_pct, signal

    signal: "draw" (level below 5yr avg = tight supply → bullish price)
            "build" (level above 5yr avg = excess supply → bearish price)
            "neutral"

    Returns {series_key: {label, level, units, wow_change, wow_pct,
                           yoy_pct, vs_5yr_avg, vs_5yr_pct, signal,
                           data_available, as_of}}
    """
    _empty = lambda label: {
        "label":         label,
        "level":         None,
        "units":         "",
        "wow_change":    None,
        "wow_pct":       None,
        "yoy_pct":       None,
        "vs_5yr_avg":    None,
        "vs_5yr_pct":    None,
        "signal":        "neutral",
        "data_available": False,
        "as_of":         None,
    }

    all_dfs   = fetch_all_eia_inventory(weeks=weeks)
    result    = {}

    for key, df in all_dfs.items():
        cfg   = _SERIES[key]
        label = cfg["label"]

        if df.empty or len(df) < 2:
            result[key] = _empty(label)
            continue

        # Week-on-week
        level      = float(df["value"].iloc[-1])
        prior_wk   = float(df["value"].iloc[-2])
        wow_change = level - prior_wk
        wow_pct    = (wow_change / prior_wk) * 100 if abs(prior_wk) > 1e-6 else 0.0

        # Year-on-year (same week, 52 rows back)
        yoy_pct = None
        if len(df) >= 53:
            prior_yr = float(df["value"].iloc[-53])
            yoy_pct  = ((level / prior_yr) - 1) * 100 if abs(prior_yr) > 1e-6 else None

        # 5-year seasonal average: same calendar week number, past 5 years
        # Only available if we have 260+ weeks of data
        vs_5yr_avg = vs_5yr_pct = None
        try:
            current_week = df["date"].iloc[-1].isocalendar()[1]
            df_week = df[df["date"].apply(lambda d: d.isocalendar()[1]) == current_week]
            # Exclude the current year
            current_yr = df["date"].iloc[-1].year
            df_hist = df_week[df_week["date"].dt.year < current_yr]
            if len(df_hist) >= 3:
                avg_5yr    = float(df_hist["value"].tail(5).mean())
                vs_5yr_avg = level - avg_5yr
                vs_5yr_pct = (vs_5yr_avg / avg_5yr) * 100 if abs(avg_5yr) > 1e-6 else None
        except Exception:
            pass

        # Price signal
        if vs_5yr_pct is not None:
            if vs_5yr_pct < -5:
                signal = "draw"    # below seasonal avg → tight supply
            elif vs_5yr_pct > 5:
                signal = "build"   # above seasonal avg → excess supply
            else:
                signal = "neutral"
        else:
            signal = "neutral"

        result[key] = {
            "label":          label,
            "level":          round(level),
            "units":          cfg["units"],
            "wow_change":     round(wow_change),
            "wow_pct":        round(wow_pct, 2),
            "yoy_pct":        round(yoy_pct, 1) if yoy_pct is not None else None,
            "vs_5yr_avg":     round(vs_5yr_avg) if vs_5yr_avg is not None else None,
            "vs_5yr_pct":     round(vs_5yr_pct, 1) if vs_5yr_pct is not None else None,
            "signal":         signal,
            "data_available": True,
            "as_of":          str(df["date"].iloc[-1].date()),
        }

    return result


def eia_price_signal(series_key: str = "crude_stocks") -> str:
    """
    Quick price-direction signal for a given inventory series.
    Returns "bullish" (draw), "bearish" (build), or "neutral".
    """
    snap = eia_snapshot()
    s    = snap.get(series_key, {})
    sig  = s.get("signal", "neutral")
    return {"draw": "bullish", "build": "bearish", "neutral": "neutral"}.get(sig, "neutral")
