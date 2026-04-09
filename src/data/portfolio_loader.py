"""
Portfolio Upload & Weighting Engine.

Accepts a CSV or Excel file with positions, fetches live market prices,
converts foreign-currency positions to USD at the prevailing spot rate,
and computes dollar-weighted portfolio allocations.

Expected upload columns (case-insensitive):
  Required: ticker, dollar_amount   (or market_value / position_value / amount)
  Optional: currency  (ISO 4217, e.g. EUR, GBP, JPY — defaults to USD)
            cusip, isin, name, sector, asset_class

Output stored in st.session_state["gbl_portfolio"] as:
  {
    "positions": list[dict]   — one row per holding with enriched fields
    "weights":   dict[str, float]  — {ticker: weight_0_to_1}
    "total_usd": float             — total portfolio value in USD
    "n":         int               — number of valid positions
    "loaded_at": str               — ISO timestamp
    "errors":    list[str]         — tickers that could not be priced
  }
"""

from __future__ import annotations

import io
import datetime
from typing import Optional

import numpy as np
import pandas as pd


# ── Template ───────────────────────────────────────────────────────────────────

TEMPLATE_COLUMNS = ["ticker", "dollar_amount", "currency", "cusip", "isin", "name", "sector"]

TEMPLATE_EXAMPLE = pd.DataFrame([
    {"ticker": "AAPL",  "dollar_amount": 50000, "currency": "USD", "cusip": "037833100", "isin": "US0378331005", "name": "Apple Inc.",         "sector": "Technology"},
    {"ticker": "XLE",   "dollar_amount": 30000, "currency": "USD", "cusip": "",           "isin": "",             "name": "Energy Select SPDR", "sector": "Energy"},
    {"ticker": "GLD",   "dollar_amount": 20000, "currency": "USD", "cusip": "",           "isin": "",             "name": "SPDR Gold Shares",   "sector": "Commodities"},
    {"ticker": "BP.L",  "dollar_amount": 15000, "currency": "GBP", "cusip": "",           "isin": "GB0007980591", "name": "BP plc",             "sector": "Energy"},
    {"ticker": "TLT",   "dollar_amount": 25000, "currency": "USD", "cusip": "",           "isin": "",             "name": "iShares 20Y+ Tsy",  "sector": "Fixed Income"},
])


def get_template_csv() -> bytes:
    """Return a CSV template as bytes for the download button."""
    return TEMPLATE_EXAMPLE.to_csv(index=False).encode()


# ── Column normaliser ──────────────────────────────────────────────────────────

_AMOUNT_ALIASES = {
    "dollar_amount", "market_value", "position_value", "amount",
    "mkt_value", "value", "notional", "market value", "position value",
}

def _find_column(cols: list[str], candidates: set[str]) -> Optional[str]:
    """Return the first column name (lowercased) matching any candidate."""
    col_map = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand in col_map:
            return col_map[cand]
    return None


def _normalise_df(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Standardise column names, infer missing optional columns.
    Returns (normalised_df, list_of_warnings).
    """
    warnings: list[str] = []
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]

    # Ticker
    if "ticker" not in df.columns:
        # Try symbol, security, asset
        for alt in ("symbol", "security", "asset", "stock", "code"):
            if alt in df.columns:
                df = df.rename(columns={alt: "ticker"})
                break
        else:
            raise ValueError("Upload must have a 'ticker' (or 'symbol' / 'security') column.")

    # Dollar amount
    amount_col = _find_column(list(df.columns), _AMOUNT_ALIASES)
    if amount_col is None:
        raise ValueError(
            "Upload must have a dollar amount column "
            "(e.g. 'dollar_amount', 'market_value', 'amount')."
        )
    df = df.rename(columns={amount_col.lower().strip(): "dollar_amount"})
    df["dollar_amount"] = pd.to_numeric(df["dollar_amount"], errors="coerce").fillna(0)

    # Currency — default USD
    if "currency" not in df.columns:
        df["currency"] = "USD"
        warnings.append("No 'currency' column found — assuming all positions are in USD.")
    else:
        df["currency"] = df["currency"].fillna("USD").str.upper().str.strip()

    # Optional columns
    for col in ("cusip", "isin", "name", "sector", "asset_class"):
        if col not in df.columns:
            df[col] = ""

    # Clean ticker
    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
    )

    # Drop rows with no ticker or zero amount
    n_before = len(df)
    df = df[df["ticker"].str.len() > 0]
    df = df[df["dollar_amount"] > 0]
    if len(df) < n_before:
        warnings.append(f"{n_before - len(df)} rows skipped (empty ticker or zero amount).")

    return df, warnings


# ── FX conversion ──────────────────────────────────────────────────────────────

def _fetch_spot_rate(ccy: str) -> float:
    """
    Fetch latest spot rate for {ccy}USD (i.e. 1 unit of ccy → USD).
    Returns 1.0 if ccy is USD or rate cannot be fetched.
    """
    if ccy.upper() in ("USD", ""):
        return 1.0
    try:
        import yfinance as yf
        pair = f"{ccy.upper()}USD=X"
        tk = yf.Ticker(pair)
        hist = tk.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return 1.0


def _convert_to_usd(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Add 'dollar_amount_usd' column. Fetches spot rates for each unique non-USD currency.
    Returns (df_with_usd_column, fx_rates_used).
    """
    fx_rates: dict[str, float] = {"USD": 1.0}
    unique_ccys = df["currency"].unique()
    for ccy in unique_ccys:
        if ccy not in fx_rates:
            fx_rates[ccy] = _fetch_spot_rate(ccy)

    df = df.copy()
    df["fx_rate"] = df["currency"].map(fx_rates).fillna(1.0)
    df["dollar_amount_usd"] = df["dollar_amount"] * df["fx_rate"]
    return df, fx_rates


# ── Price fetch ────────────────────────────────────────────────────────────────

def _fetch_latest_prices(tickers: list[str]) -> dict[str, float]:
    """
    Fetch the most recent closing price for each ticker via yfinance.
    Returns dict of {ticker: price}. Missing tickers are omitted.
    """
    if not tickers:
        return {}
    try:
        import yfinance as yf
        data = yf.download(
            tickers, period="5d", auto_adjust=True,
            progress=False, threads=True,
        )
        if data.empty:
            return {}

        # Multi-ticker: data["Close"] is a DataFrame; single ticker: data["Close"] is a Series
        close = data["Close"] if "Close" in data.columns else data
        if isinstance(close, pd.Series):
            # Single ticker
            price = float(close.dropna().iloc[-1]) if not close.dropna().empty else None
            return {tickers[0]: price} if price else {}

        prices: dict[str, float] = {}
        for tk in tickers:
            if tk in close.columns:
                s = close[tk].dropna()
                if not s.empty:
                    prices[tk] = float(s.iloc[-1])
        return prices
    except Exception:
        return {}


# ── Main builder ──────────────────────────────────────────────────────────────

def build_portfolio(file_obj) -> dict:
    """
    Parse an uploaded CSV or Excel file and return a portfolio dict.

    Args:
        file_obj: Streamlit UploadedFile (has .name and read() / seek()).

    Returns:
        {
            "positions": list[dict],
            "weights":   dict[str, float],
            "total_usd": float,
            "n":         int,
            "loaded_at": str,
            "errors":    list[str],
            "warnings":  list[str],
            "fx_rates":  dict[str, float],
        }
    Raises ValueError on unrecoverable parse errors.
    """
    errors:   list[str] = []
    warnings: list[str] = []

    # ── Parse file ─────────────────────────────────────────────────────────
    fname = getattr(file_obj, "name", "upload")
    if fname.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_obj)
    else:
        df = pd.read_csv(file_obj)

    if df.empty:
        raise ValueError("Uploaded file is empty.")

    df, parse_warnings = _normalise_df(df)
    warnings.extend(parse_warnings)

    if df.empty:
        raise ValueError("No valid positions found after parsing.")

    # ── FX conversion ──────────────────────────────────────────────────────
    df, fx_rates = _convert_to_usd(df)

    # ── Live price fetch (optional enrichment) ─────────────────────────────
    tickers = df["ticker"].tolist()
    live_prices = _fetch_latest_prices(tickers)

    missing_price = [t for t in tickers if t not in live_prices]
    if missing_price:
        errors.extend([f"Price not found for {t} — using uploaded dollar_amount as-is." for t in missing_price])

    # ── Weights ─────────────────────────────────────────────────────────────
    total_usd = df["dollar_amount_usd"].sum()
    if total_usd <= 0:
        raise ValueError("Total portfolio value is zero — check dollar_amount column.")

    df["weight"] = df["dollar_amount_usd"] / total_usd
    df["live_price"] = df["ticker"].map(live_prices)

    # ── Build position records ──────────────────────────────────────────────
    positions = []
    for _, row in df.iterrows():
        positions.append({
            "ticker":           row["ticker"],
            "name":             row.get("name", ""),
            "sector":           row.get("sector", ""),
            "asset_class":      row.get("asset_class", ""),
            "cusip":            row.get("cusip", ""),
            "isin":             row.get("isin", ""),
            "currency":         row["currency"],
            "dollar_amount":    float(row["dollar_amount"]),
            "fx_rate":          float(row["fx_rate"]),
            "dollar_amount_usd":float(row["dollar_amount_usd"]),
            "weight":           float(row["weight"]),
            "live_price":       float(row["live_price"]) if pd.notna(row.get("live_price")) else None,
        })

    weights = {p["ticker"]: p["weight"] for p in positions}

    return {
        "positions": positions,
        "weights":   weights,
        "total_usd": float(total_usd),
        "n":         len(positions),
        "loaded_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "errors":    errors,
        "warnings":  warnings,
        "fx_rates":  fx_rates,
    }


# ── Session helpers ────────────────────────────────────────────────────────────

def get_portfolio():
    """Return the current global portfolio from session_state, or None."""
    try:
        import streamlit as st
        return st.session_state.get("gbl_portfolio")
    except Exception:
        return None


def set_portfolio(portfolio: dict) -> None:
    """Store a processed portfolio in session_state."""
    try:
        import streamlit as st
        st.session_state["gbl_portfolio"] = portfolio
    except Exception:
        pass


def clear_portfolio() -> None:
    """Remove the current portfolio from session_state."""
    try:
        import streamlit as st
        st.session_state.pop("gbl_portfolio", None)
    except Exception:
        pass
