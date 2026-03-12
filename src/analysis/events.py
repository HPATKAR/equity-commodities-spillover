"""
Event-window analytics:
  - Cumulative return around a geopolitical trigger
  - Pre/post volatility comparison
  - Cross-asset performance attribution during an event
  - Correlation shift: pre vs during vs post event
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Optional


def event_window_returns(
    prices: pd.DataFrame,
    event_start: date,
    event_end: date,
    pre_days: int = 30,
    post_days: int = 60,
) -> dict:
    """
    Slice prices around an event and compute cumulative returns
    for pre-window, event, and post-window periods.

    Returns dict:
      pre:    DataFrame of cum returns in pre-window
      during: DataFrame of cum returns during event
      post:   DataFrame of cum returns in post-window
      full:   Full sliced prices DataFrame
    """
    t0_pre  = pd.Timestamp(event_start) - pd.Timedelta(days=pre_days)
    t0_evt  = pd.Timestamp(event_start)
    t1_evt  = pd.Timestamp(event_end)
    t1_post = pd.Timestamp(event_end) + pd.Timedelta(days=post_days)

    prices = prices.sort_index()
    full   = prices.loc[t0_pre:t1_post]

    def _cum_ret(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        return (df.iloc[-1] / df.iloc[0] - 1) * 100

    return {
        "pre":    _cum_ret(prices.loc[t0_pre:t0_evt]),
        "during": _cum_ret(prices.loc[t0_evt:t1_evt]),
        "post":   _cum_ret(prices.loc[t1_evt:t1_post]),
        "full":   full,
        "labels": {
            "pre":    f"Pre ({pre_days}d)",
            "during": "During Event",
            "post":   f"Post ({post_days}d)",
        },
    }


def event_normalised_prices(
    prices: pd.DataFrame,
    event_start: date,
    pre_days: int = 30,
    post_days: int = 90,
    assets: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Index prices to 100 at event_start for the given window.
    Useful for comparing cross-asset performance during an event.
    """
    t0 = pd.Timestamp(event_start) - pd.Timedelta(days=pre_days)
    t1 = pd.Timestamp(event_start) + pd.Timedelta(days=post_days)

    if assets:
        prices = prices[[c for c in assets if c in prices.columns]]

    sliced = prices.loc[t0:t1].copy()
    if sliced.empty:
        return sliced

    base  = sliced.loc[pd.Timestamp(event_start):].iloc[0]
    normed = (sliced / base) * 100
    return normed


def pre_post_volatility(
    returns: pd.DataFrame,
    event_start: date,
    event_end: date,
    window: int = 30,
) -> pd.DataFrame:
    """
    Annualised volatility (std * sqrt(252)) for:
      pre-event  : [event_start - window, event_start)
      post-event : (event_end, event_end + window]
    """
    t0 = pd.Timestamp(event_start)
    t1 = pd.Timestamp(event_end)

    pre_ret  = returns.loc[t0 - pd.Timedelta(days=window): t0]
    post_ret = returns.loc[t1: t1 + pd.Timedelta(days=window)]

    pre_vol  = pre_ret.std()  * np.sqrt(252) * 100
    post_vol = post_ret.std() * np.sqrt(252) * 100

    result = pd.DataFrame({
        "Pre-Event Vol %":  pre_vol,
        "Post-Event Vol %": post_vol,
    }).round(2)
    result["Vol Change pp"] = (result["Post-Event Vol %"] - result["Pre-Event Vol %"]).round(2)
    return result


def correlation_shift(
    equity_returns: pd.DataFrame,
    commodity_returns: pd.DataFrame,
    event_start: date,
    event_end: date,
    pre_days: int = 60,
    post_days: int = 60,
) -> pd.DataFrame:
    """
    Cross-asset correlation before vs during vs after an event.
    Returns DataFrame indexed by (equity, commodity) pair.
    """
    combined = pd.concat([equity_returns, commodity_returns], axis=1)
    eq_cols  = equity_returns.columns.tolist()
    cmd_cols = commodity_returns.columns.tolist()

    t0 = pd.Timestamp(event_start)
    t1 = pd.Timestamp(event_end)

    pre    = combined.loc[t0 - pd.Timedelta(days=pre_days): t0]
    during = combined.loc[t0: t1]
    post   = combined.loc[t1: t1 + pd.Timedelta(days=post_days)]

    rows = []
    for eq in eq_cols:
        for cmd in cmd_cols:
            def corr(df: pd.DataFrame) -> float:
                sub = df[[eq, cmd]].dropna()
                return sub.iloc[:, 0].corr(sub.iloc[:, 1]) if len(sub) >= 5 else np.nan

            rows.append({
                "Equity":    eq,
                "Commodity": cmd,
                "Pre":       round(corr(pre),    3),
                "During":    round(corr(during),  3),
                "Post":      round(corr(post),    3),
                "Shift":     round((corr(during) or 0) - (corr(pre) or 0), 3),
            })

    return pd.DataFrame(rows)
