"""
Data loader — LSEG (primary) + yfinance (fallback) + FRED.
Returns log-return DataFrames and raw price DataFrames.

LSEG setup (Purdue account):
  1. pip install lseg-data
  2. Add to .streamlit/secrets.toml:
       [lseg]
       app_key = "your_32_char_key_from_developers.lseg.com"
  3. Keep Eikon / LSEG Workspace open on the same machine.
  All loaders silently fall back to yfinance if the session fails.
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


# ── LSEG RIC map ───────────────────────────────────────────────────────────
# Maps the dashboard's asset names → LSEG RIC codes.
# Continuous futures use the "1!" convention (front-month roll).

_LSEG_RICS: dict[str, str] = {
    # ── Equity indices ──────────────────────────────────────────────────
    "S&P 500":        ".SPX",
    "Nasdaq 100":     ".NDX",
    "DJIA":           ".DJI",
    "Russell 2000":   ".RUT",
    "Eurostoxx 50":   ".STOXX50E",
    "DAX":            ".GDAXI",
    "CAC 40":         ".FCHI",
    "FTSE 100":       ".FTSE",
    "Nikkei 225":     ".N225",
    "TOPIX":          ".TOPX",
    "Hang Seng":      ".HSI",
    "Shanghai Comp":  ".SSEC",
    "CSI 300":        "CSI300.SS",
    "Sensex":         ".BSESN",
    "Nifty 50":       ".NSEI",
    # ── Commodities (continuous front-month futures) ─────────────────────
    "WTI Crude Oil":  "CL1!",
    "Brent Crude":    "LCO1!",
    "Natural Gas":    "NG1!",
    "Gasoline (RBOB)":"RB1!",
    "Heating Oil":    "HO1!",
    "Gold":           "GC1!",
    "Silver":         "SI1!",
    "Platinum":       "PL1!",
    "Copper":         "HG1!",
    "Aluminum":       "MAL1!",
    "Nickel":         "MNI1!",
    "Wheat":          "W1!",
    "Corn":           "C1!",
    "Soybeans":       "S1!",
    "Sugar #11":      "SB1!",
    "Coffee":         "KC1!",
    "Cotton":         "CT1!",
}

_RIC_TO_NAME = {v: k for k, v in _LSEG_RICS.items()}


# ── LSEG session management ────────────────────────────────────────────────

def _lseg_open() -> bool:
    """
    Open a Desktop Session using the app key in st.secrets["lseg"]["app_key"].
    Requires Eikon / LSEG Workspace to be running on the same machine.
    Returns True on success, False otherwise (fallback to yfinance).
    Stores result in st.session_state so we only attempt once per session.
    """
    if st.session_state.get("_lseg_ok") is not None:
        return bool(st.session_state["_lseg_ok"])
    try:
        import lseg.data as ld
        app_key = st.secrets.get("lseg", {}).get("app_key", "")
        if not app_key:
            st.session_state["_lseg_ok"] = False
            return False
        ld.open_session(
            name="desktop.workspace",
            app_key=app_key,
        )
        st.session_state["_lseg_ok"] = True
        st.session_state["_lseg_mod"] = ld
        return True
    except Exception:
        st.session_state["_lseg_ok"] = False
        return False


# ── LSEG fetch helpers ─────────────────────────────────────────────────────

def _fetch_lseg(names: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily Close prices from LSEG for a list of dashboard asset names.
    Returns DataFrame with asset names as columns; empty on any failure.
    """
    if not _lseg_open():
        return pd.DataFrame()
    ld = st.session_state.get("_lseg_mod")
    if ld is None:
        return pd.DataFrame()

    rics = [_LSEG_RICS[n] for n in names if n in _LSEG_RICS]
    if not rics:
        return pd.DataFrame()

    try:
        raw = ld.get_history(
            universe=rics,
            fields=["TRDPRC_1"],       # last trade price (exchange close)
            start=start,
            end=end,
            interval="daily",
        )
        if raw is None or raw.empty:
            return pd.DataFrame()

        # get_history may return MultiIndex columns — flatten to RICs
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.xs("TRDPRC_1", level=0, axis=1) if "TRDPRC_1" in raw.columns.get_level_values(0) \
                  else raw.droplevel(0, axis=1)

        raw = raw.rename(columns=_RIC_TO_NAME)
        raw.index = pd.to_datetime(raw.index).normalize()
        # Keep only requested names (in original order)
        keep = [n for n in names if n in raw.columns]
        return raw[keep].sort_index().astype(float)

    except Exception:
        return pd.DataFrame()


def _fetch_lseg_snapshot(names: list[str]) -> pd.DataFrame:
    """
    Real-time snapshot: last price, % change, volume for given asset names.
    Returns DataFrame indexed by asset name; empty on failure.
    """
    if not _lseg_open():
        return pd.DataFrame()
    ld = st.session_state.get("_lseg_mod")
    if ld is None:
        return pd.DataFrame()

    rics = [_LSEG_RICS[n] for n in names if n in _LSEG_RICS]
    if not rics:
        return pd.DataFrame()

    try:
        df, _ = ld.get_data(
            universe=rics,
            fields=["CF_LAST", "PCTCHNG", "CF_VOLUME", "CF_HIGH", "CF_LOW"],
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df["Asset"] = df["Instrument"].map(_RIC_TO_NAME)
        df = df.dropna(subset=["Asset"]).set_index("Asset").drop(columns=["Instrument"])
        df.columns = ["Last", "Change %", "Volume", "High", "Low"]
        return df.loc[[n for n in names if n in df.index]]
    except Exception:
        return pd.DataFrame()


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

@st.cache_data(ttl=300, show_spinner=False)   # 5 min when LSEG live; 1 hr otherwise
def load_equity_prices(
    start: str = str(DEFAULT_START),
    end:   str = str(DEFAULT_END),
) -> pd.DataFrame:
    names = list(EQUITY_TICKERS.keys())
    lseg  = _fetch_lseg(names, start, end)
    if not lseg.empty and len(lseg) > 10:
        return _fill_gaps(lseg)
    return _fill_gaps(_fetch_yf(EQUITY_TICKERS, start, end))


@st.cache_data(ttl=300, show_spinner=False)
def load_commodity_prices(
    start: str = str(DEFAULT_START),
    end:   str = str(DEFAULT_END),
) -> pd.DataFrame:
    names = list(COMMODITY_TICKERS.keys())
    lseg  = _fetch_lseg(names, start, end)
    if not lseg.empty and len(lseg) > 10:
        return _fill_gaps(lseg)
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


# ── S&P 500 individual stock loaders ──────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_constituents() -> pd.DataFrame:
    """
    Fetch S&P 500 constituent list from Wikipedia.
    Returns DataFrame with columns: ticker, name, sector.
    Tickers are already normalised for yfinance (dots → dashes).
    Falls back to a curated top-100 list if Wikipedia is unreachable.
    """
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            header=0,
        )
        df = tables[0][["Symbol", "Security", "GICS Sector"]].copy()
        df.columns = ["ticker", "name", "sector"]
        df["ticker"] = df["ticker"].str.strip().str.replace(".", "-", regex=False)
        return df.sort_values("ticker").reset_index(drop=True)
    except Exception:
        return _SP500_FALLBACK.copy()


# Top-100 S&P 500 fallback (market-cap ordered, approximate)
_SP500_FALLBACK = pd.DataFrame([
    ("AAPL", "Apple Inc.", "Information Technology"),
    ("MSFT", "Microsoft Corp.", "Information Technology"),
    ("NVDA", "NVIDIA Corp.", "Information Technology"),
    ("GOOGL", "Alphabet Inc. Class A", "Communication Services"),
    ("AMZN", "Amazon.com Inc.", "Consumer Discretionary"),
    ("META", "Meta Platforms Inc.", "Communication Services"),
    ("BRK-B", "Berkshire Hathaway Inc. Class B", "Financials"),
    ("LLY",  "Eli Lilly and Company", "Health Care"),
    ("TSLA", "Tesla Inc.", "Consumer Discretionary"),
    ("JPM",  "JPMorgan Chase & Co.", "Financials"),
    ("V",    "Visa Inc.", "Financials"),
    ("UNH",  "UnitedHealth Group Inc.", "Health Care"),
    ("XOM",  "Exxon Mobil Corp.", "Energy"),
    ("MA",   "Mastercard Inc.", "Financials"),
    ("AVGO", "Broadcom Inc.", "Information Technology"),
    ("HD",   "Home Depot Inc.", "Consumer Discretionary"),
    ("JNJ",  "Johnson & Johnson", "Health Care"),
    ("PG",   "Procter & Gamble Co.", "Consumer Staples"),
    ("COST", "Costco Wholesale Corp.", "Consumer Staples"),
    ("MRK",  "Merck & Co. Inc.", "Health Care"),
    ("ABBV", "AbbVie Inc.", "Health Care"),
    ("WMT",  "Walmart Inc.", "Consumer Staples"),
    ("CVX",  "Chevron Corp.", "Energy"),
    ("CRM",  "Salesforce Inc.", "Information Technology"),
    ("BAC",  "Bank of America Corp.", "Financials"),
    ("NFLX", "Netflix Inc.", "Communication Services"),
    ("AMD",  "Advanced Micro Devices Inc.", "Information Technology"),
    ("ACN",  "Accenture Plc", "Information Technology"),
    ("TMO",  "Thermo Fisher Scientific Inc.", "Health Care"),
    ("LIN",  "Linde Plc", "Materials"),
    ("ORCL", "Oracle Corp.", "Information Technology"),
    ("ADBE", "Adobe Inc.", "Information Technology"),
    ("CSCO", "Cisco Systems Inc.", "Information Technology"),
    ("MCD",  "McDonald's Corp.", "Consumer Discretionary"),
    ("DIS",  "Walt Disney Co.", "Communication Services"),
    ("GE",   "GE Aerospace", "Industrials"),
    ("TXN",  "Texas Instruments Inc.", "Information Technology"),
    ("INTU", "Intuit Inc.", "Information Technology"),
    ("WFC",  "Wells Fargo & Co.", "Financials"),
    ("CAT",  "Caterpillar Inc.", "Industrials"),
    ("IBM",  "IBM Corp.", "Information Technology"),
    ("NEE",  "NextEra Energy Inc.", "Utilities"),
    ("RTX",  "RTX Corp.", "Industrials"),
    ("SPGI", "S&P Global Inc.", "Financials"),
    ("AMGN", "Amgen Inc.", "Health Care"),
    ("LOW",  "Lowe's Companies Inc.", "Consumer Discretionary"),
    ("ISRG", "Intuitive Surgical Inc.", "Health Care"),
    ("BKNG", "Booking Holdings Inc.", "Consumer Discretionary"),
    ("GS",   "Goldman Sachs Group Inc.", "Financials"),
    ("HON",  "Honeywell International Inc.", "Industrials"),
    ("MS",   "Morgan Stanley", "Financials"),
    ("C",    "Citigroup Inc.", "Financials"),
    ("BSX",  "Boston Scientific Corp.", "Health Care"),
    ("UBER", "Uber Technologies Inc.", "Industrials"),
    ("DE",   "Deere & Co.", "Industrials"),
    ("MMC",  "Marsh & McLennan Companies Inc.", "Financials"),
    ("AXP",  "American Express Co.", "Financials"),
    ("SYK",  "Stryker Corp.", "Health Care"),
    ("REGN", "Regeneron Pharmaceuticals Inc.", "Health Care"),
    ("BLK",  "BlackRock Inc.", "Financials"),
    ("VRTX", "Vertex Pharmaceuticals Inc.", "Health Care"),
    ("PANW", "Palo Alto Networks Inc.", "Information Technology"),
    ("ADI",  "Analog Devices Inc.", "Information Technology"),
    ("BA",   "Boeing Co.", "Industrials"),
    ("SCHW", "Charles Schwab Corp.", "Financials"),
    ("MDT",  "Medtronic Plc", "Health Care"),
    ("CI",   "Cigna Group", "Health Care"),
    ("PLD",  "Prologis Inc.", "Real Estate"),
    ("MU",   "Micron Technology Inc.", "Information Technology"),
    ("SO",   "Southern Co.", "Utilities"),
    ("ELV",  "Elevance Health Inc.", "Health Care"),
    ("DUK",  "Duke Energy Corp.", "Utilities"),
    ("TJX",  "TJX Companies Inc.", "Consumer Discretionary"),
    ("ETN",  "Eaton Corp. Plc", "Industrials"),
    ("COF",  "Capital One Financial Corp.", "Financials"),
    ("APH",  "Amphenol Corp.", "Information Technology"),
    ("ICE",  "Intercontinental Exchange Inc.", "Financials"),
    ("COP",  "ConocoPhillips", "Energy"),
    ("ZTS",  "Zoetis Inc.", "Health Care"),
    ("AMAT", "Applied Materials Inc.", "Information Technology"),
    ("CME",  "CME Group Inc.", "Financials"),
    ("AON",  "Aon Plc", "Financials"),
    ("FI",   "Fiserv Inc.", "Financials"),
    ("NOC",  "Northrop Grumman Corp.", "Industrials"),
    ("LMT",  "Lockheed Martin Corp.", "Industrials"),
    ("PGR",  "Progressive Corp.", "Financials"),
    ("USB",  "U.S. Bancorp", "Financials"),
    ("AIG",  "American International Group Inc.", "Financials"),
    ("MO",   "Altria Group Inc.", "Consumer Staples"),
    ("PM",   "Philip Morris International Inc.", "Consumer Staples"),
    ("KO",   "Coca-Cola Co.", "Consumer Staples"),
    ("PEP",  "PepsiCo Inc.", "Consumer Staples"),
    ("GILD", "Gilead Sciences Inc.", "Health Care"),
    ("EW",   "Edwards Lifesciences Corp.", "Health Care"),
    ("KLAC", "KLA Corp.", "Information Technology"),
    ("LRCX", "Lam Research Corp.", "Information Technology"),
    ("ADM",  "Archer-Daniels-Midland Co.", "Consumer Staples"),
    ("NKE",  "Nike Inc.", "Consumer Discretionary"),
    ("T",    "AT&T Inc.", "Communication Services"),
    ("VZ",   "Verizon Communications Inc.", "Communication Services"),
], columns=["ticker", "name", "sector"])


@st.cache_data(ttl=3600, show_spinner=False)
def load_sp500_prices(
    tickers_tuple: tuple,          # tuple for hashability
    start: str = str(DEFAULT_START),
    end:   str = str(DEFAULT_END),
) -> pd.DataFrame:
    """
    Batch download S&P 500 individual stock close prices via yfinance.
    Accepts a tuple of ticker strings (already yfinance-normalised, dots→dashes).
    Returns a DataFrame with ticker symbols as column names.
    """
    if not tickers_tuple:
        return pd.DataFrame()
    raw = yf.download(
        list(tickers_tuple),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        if len(tickers_tuple) == 1:
            prices.columns = list(tickers_tuple)
    return _fill_gaps(prices)


# ── Live snapshot (no cache, called only in watchlist) ─────────────────────

def load_live_snapshot(tickers: dict[str, str]) -> pd.DataFrame:
    """
    Return last price, 1d/5d/YTD change % for each asset.
    Uses LSEG real-time snapshot when available; falls back to yfinance.
    """
    names = list(tickers.keys())

    # ── LSEG real-time path ───────────────────────────────────────────
    lseg_snap = _fetch_lseg_snapshot(names)
    if not lseg_snap.empty:
        # Supplement with YTD from a short historical pull
        ytd_start = str(date(date.today().year, 1, 1))
        ytd_hist  = _fetch_lseg(names, ytd_start, str(date.today()))
        rows = []
        for name in names:
            if name not in lseg_snap.index:
                continue
            row = lseg_snap.loc[name]
            ytd = np.nan
            if not ytd_hist.empty and name in ytd_hist.columns:
                s = ytd_hist[name].dropna()
                ytd = (s.iloc[-1] / s.iloc[0] - 1) * 100 if len(s) >= 2 else np.nan
            rows.append({
                "Name":    name,
                "Last":    round(float(row["Last"]),     2) if pd.notna(row["Last"])     else np.nan,
                "1D %":    round(float(row["Change %"]), 2) if pd.notna(row["Change %"]) else np.nan,
                "5D %":    np.nan,    # not in real-time snapshot; filled below if needed
                "YTD %":   round(float(ytd), 2) if pd.notna(ytd) else np.nan,
            })
        if rows:
            return pd.DataFrame(rows).set_index("Name")

    # ── yfinance fallback ─────────────────────────────────────────────
    start_ytd = date(date.today().year, 1, 1)
    rows = []
    for name, tk in tickers.items():
        try:
            hist = yf.Ticker(tk).history(start=str(start_ytd), auto_adjust=True)
            if hist.empty:
                continue
            last = hist["Close"].iloc[-1]
            d1   = (hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1) * 100 if len(hist) >= 2 else 0
            d5   = (hist["Close"].iloc[-1] / hist["Close"].iloc[-6] - 1) * 100 if len(hist) >= 6 else 0
            ytd  = (hist["Close"].iloc[-1] / hist["Close"].iloc[0]  - 1) * 100 if len(hist) >= 2 else 0
            rows.append({
                "Name":  name,
                "Last":  round(last, 2),
                "1D %":  round(d1,   2),
                "5D %":  round(d5,   2),
                "YTD %": round(ytd,  2),
            })
        except Exception:
            pass
    return pd.DataFrame(rows).set_index("Name") if rows else pd.DataFrame()
