"""
Data loader — yfinance + FRED with Streamlit caching.
Returns log-return DataFrames and raw price DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import date, timedelta
from typing import Optional

from src.data.config import (
    EQUITY_TICKERS, COMMODITY_TICKERS, FRED_SERIES,
    DEFAULT_START, DEFAULT_END,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _fetch_yf(tickers: dict[str, str], start: date, end: date) -> pd.DataFrame:
    """Download adjusted close prices for a dict of {name: ticker}."""
    reverse = {v: k for k, v in tickers.items()}
    raw = yf.download(
        list(tickers.values()),
        start=str(start),
        end=str(end),
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
    prices = prices.rename(columns=reverse)
    prices = prices[[c for c in tickers.keys() if c in prices.columns]]
    return prices.sort_index()


def _log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns, dropping all-NaN rows."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


def _fill_gaps(df: pd.DataFrame, method: str = "ffill", limit: int = 5) -> pd.DataFrame:
    return df.ffill(limit=limit).bfill(limit=2)


# ── Cached loaders ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_equity_prices(
    start: str = str(DEFAULT_START),
    end:   str = str(DEFAULT_END),
) -> pd.DataFrame:
    return _fill_gaps(_fetch_yf(EQUITY_TICKERS, start, end))


@st.cache_data(ttl=3600, show_spinner=False)
def load_commodity_prices(
    start: str = str(DEFAULT_START),
    end:   str = str(DEFAULT_END),
) -> pd.DataFrame:
    return _fill_gaps(_fetch_yf(COMMODITY_TICKERS, start, end))


@st.cache_data(ttl=3600, show_spinner=False)
def load_all_prices(
    start: str = str(DEFAULT_START),
    end:   str = str(DEFAULT_END),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (equity_prices, commodity_prices)."""
    eq  = load_equity_prices(start, end)
    cmd = load_commodity_prices(start, end)
    return eq, cmd


@st.cache_data(ttl=3600, show_spinner=False)
def load_returns(
    start: str = str(DEFAULT_START),
    end:   str = str(DEFAULT_END),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (equity_returns, commodity_returns) as log-return DataFrames."""
    eq, cmd = load_all_prices(start, end)
    return _log_returns(eq), _log_returns(cmd)


@st.cache_data(ttl=3600, show_spinner=False)
def load_combined_returns(
    start: str = str(DEFAULT_START),
    end:   str = str(DEFAULT_END),
) -> pd.DataFrame:
    """Single DataFrame: equity + commodity log returns, aligned on date index."""
    eq_r, cmd_r = load_returns(start, end)
    combined = pd.concat([eq_r, cmd_r], axis=1)
    return combined.dropna(how="all")


@st.cache_data(ttl=3600, show_spinner=False)
def load_fred_series(
    fred_api_key: Optional[str] = None,
    start: str = str(DEFAULT_START),
    end:   str = str(DEFAULT_END),
) -> pd.DataFrame:
    """Fetch FRED series if API key provided; else return empty DataFrame."""
    if not fred_api_key:
        return pd.DataFrame()
    try:
        from fredapi import Fred
        fred  = Fred(api_key=fred_api_key)
        dfs   = {}
        for name, series_id in FRED_SERIES.items():
            try:
                s = fred.get_series(series_id, observation_start=start, observation_end=end)
                dfs[name] = s
            except Exception:
                pass
        if dfs:
            return pd.DataFrame(dfs).sort_index()
    except Exception:
        pass
    return pd.DataFrame()


# ── Convenience: event-window slicer ──────────────────────────────────────

def slice_event(
    df: pd.DataFrame,
    event_start: date,
    event_end: date,
    pre_days:  int = 30,
    post_days: int = 60,
) -> pd.DataFrame:
    """Return df sliced to [event_start - pre_days, event_end + post_days]."""
    t0 = pd.Timestamp(event_start) - pd.Timedelta(days=pre_days)
    t1 = pd.Timestamp(event_end)   + pd.Timedelta(days=post_days)
    return df.loc[t0:t1]


# ── Hourly loaders (yfinance: max 730-day lookback) ───────────────────────

def _clamp_hourly_start(start: str) -> str:
    """yfinance hourly data is capped at 730 days. Clamp silently."""
    limit = date.today() - timedelta(days=729)
    requested = date.fromisoformat(start)
    return str(max(requested, limit))


def _fetch_yf_hourly(tickers: dict[str, str], start: str, end: str) -> pd.DataFrame:
    """Download hourly close prices. Drops timezone info, returns UTC-naive index."""
    reverse = {v: k for k, v in tickers.items()}
    raw = yf.download(
        list(tickers.values()),
        start=start,
        end=end,
        interval="1h",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
    prices = prices.rename(columns=reverse)
    prices = prices[[c for c in tickers.keys() if c in prices.columns]]
    prices.index = prices.index.tz_localize(None)   # drop tz → naive UTC
    return prices.sort_index()


@st.cache_data(ttl=1800, show_spinner=False)
def load_hourly_equity_prices(
    start: str = str(DEFAULT_START),
    end:   str = str(DEFAULT_END),
) -> pd.DataFrame:
    return _fill_gaps(_fetch_yf_hourly(EQUITY_TICKERS, _clamp_hourly_start(start), end))


@st.cache_data(ttl=1800, show_spinner=False)
def load_hourly_commodity_prices(
    start: str = str(DEFAULT_START),
    end:   str = str(DEFAULT_END),
) -> pd.DataFrame:
    return _fill_gaps(_fetch_yf_hourly(COMMODITY_TICKERS, _clamp_hourly_start(start), end))


@st.cache_data(ttl=1800, show_spinner=False)
def load_hourly_prices(
    start: str = str(DEFAULT_START),
    end:   str = str(DEFAULT_END),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (equity_hourly_prices, commodity_hourly_prices). Max ~730 days."""
    eq  = load_hourly_equity_prices(start, end)
    cmd = load_hourly_commodity_prices(start, end)
    return eq, cmd


@st.cache_data(ttl=1800, show_spinner=False)
def load_hourly_returns(
    start: str = str(DEFAULT_START),
    end:   str = str(DEFAULT_END),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (equity_hourly_returns, commodity_hourly_returns) as log-return DataFrames."""
    eq, cmd = load_hourly_prices(start, end)
    return _log_returns(eq), _log_returns(cmd)


# ── Live snapshot (no cache, called only in watchlist) ─────────────────────

def load_live_snapshot(tickers: dict[str, str]) -> pd.DataFrame:
    """
    Fetch 5 days of data for the given tickers and return a snapshot row:
    last price, 1d change %, 5d change %, YTD change %.
    """
    start_ytd  = date(date.today().year, 1, 1)
    start_5d   = date.today() - timedelta(days=7)
    rows = []
    for name, tk in tickers.items():
        try:
            hist = yf.Ticker(tk).history(start=str(start_ytd), auto_adjust=True)
            if hist.empty:
                continue
            last  = hist["Close"].iloc[-1]
            d1    = (hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1) * 100 if len(hist) >= 2 else 0
            d5    = (hist["Close"].iloc[-1] / hist["Close"].iloc[-6] - 1) * 100 if len(hist) >= 6 else 0
            ytd   = (hist["Close"].iloc[-1] / hist["Close"].iloc[0]  - 1) * 100 if len(hist) >= 2 else 0
            rows.append({
                "Name":    name,
                "Ticker":  tk,
                "Last":    round(last, 2),
                "1D %":    round(d1, 2),
                "5D %":    round(d5, 2),
                "YTD %":   round(ytd, 2),
            })
        except Exception:
            pass
    return pd.DataFrame(rows).set_index("Name") if rows else pd.DataFrame()
