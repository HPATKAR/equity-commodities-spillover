"""
Spillover analytics:
  - Granger causality (commodity → equity and vice versa)
  - Transfer entropy (directional information flow)
  - Diebold-Yilmaz spillover index (VAR-based)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from typing import Optional


# ── Granger causality ──────────────────────────────────────────────────────

def granger_test(
    cause: pd.Series,
    effect: pd.Series,
    max_lag: int = 5,
    significance: float = 0.05,
) -> dict:
    """
    Test whether `cause` Granger-causes `effect`.
    Returns dict: {lag: p_value, ..., 'min_p': ..., 'significant': bool}
    """
    combined = pd.concat([effect, cause], axis=1).dropna()
    if len(combined) < max_lag * 10:
        return {"min_p": np.nan, "significant": False, "results": {}}
    try:
        res = grangercausalitytests(combined.values, maxlag=max_lag, verbose=False)
        p_values = {
            lag: min(r[0][test][1] for test in ["ssr_ftest", "ssr_chi2test"])
            for lag, r in res.items()
        }
        min_p = min(p_values.values())
        return {
            "min_p": round(min_p, 4),
            "significant": min_p < significance,
            "results": p_values,
            "best_lag": min(p_values, key=p_values.get),
        }
    except Exception:
        return {"min_p": np.nan, "significant": False, "results": {}}


def granger_grid(
    equity_returns: pd.DataFrame,
    commodity_returns: pd.DataFrame,
    max_lag: int = 5,
    significance: float = 0.05,
) -> pd.DataFrame:
    """
    Full grid of Granger tests: commodity → equity AND equity → commodity.
    Returns tidy DataFrame: [cause, effect, direction, min_p, significant, best_lag].
    """
    rows = []
    eq_cols  = equity_returns.columns.tolist()
    cmd_cols = commodity_returns.columns.tolist()

    for eq in eq_cols:
        for cmd in cmd_cols:
            r_eq  = equity_returns[eq].dropna()
            r_cmd = commodity_returns[cmd].dropna()
            idx   = r_eq.index.intersection(r_cmd.index)
            if len(idx) < 60:
                continue
            r_eq  = r_eq.loc[idx]
            r_cmd = r_cmd.loc[idx]

            # commodity → equity
            res1 = granger_test(r_cmd, r_eq, max_lag, significance)
            rows.append({
                "cause": cmd, "effect": eq,
                "direction": "Commodity → Equity",
                **res1,
            })
            # equity → commodity
            res2 = granger_test(r_eq, r_cmd, max_lag, significance)
            rows.append({
                "cause": eq, "effect": cmd,
                "direction": "Equity → Commodity",
                **res2,
            })

    df = pd.DataFrame(rows)
    return df.drop(columns=["results"], errors="ignore")


# ── Transfer entropy ───────────────────────────────────────────────────────

def _discretize(x: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Bin continuous series into integer labels."""
    edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    return np.digitize(x, edges[:-1]) - 1


def transfer_entropy(
    source: pd.Series,
    target: pd.Series,
    lag: int = 1,
    n_bins: int = 5,
) -> float:
    """
    Transfer entropy: TE(source → target).
    Measures how much the past of `source` reduces uncertainty in `target`
    beyond `target`'s own past.
    """
    combined = pd.concat([source, target], axis=1).dropna()
    if len(combined) < lag + 20:
        return np.nan

    x = _discretize(combined.iloc[:, 0].values, n_bins)
    y = _discretize(combined.iloc[:, 1].values, n_bins)

    T = len(y) - lag
    y_fut  = y[lag:]
    y_past = y[:T]
    x_past = x[:T]

    def entropy(*arrays) -> float:
        combined_arr = np.column_stack(arrays)
        _, counts = np.unique(combined_arr, axis=0, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-12))

    h_yfut_given_ypast   = entropy(y_fut, y_past) - entropy(y_past)
    h_yfut_given_both    = entropy(y_fut, y_past, x_past) - entropy(y_past, x_past)
    te = h_yfut_given_ypast - h_yfut_given_both
    return float(max(te, 0.0))


def optimal_te_lag(
    source: pd.Series,
    target: pd.Series,
    max_lag: int = 5,
    n_bins: int = 5,
) -> int:
    """
    Find the lag (1..max_lag) that maximises TE(source → target).
    Commodity→equity transmission empirically peaks at 2–5 days.
    Returns the lag with the highest TE value (minimum 1).
    """
    best_lag, best_te = 1, -np.inf
    for lag in range(1, max_lag + 1):
        te = transfer_entropy(source, target, lag=lag, n_bins=n_bins)
        if np.isfinite(te) and te > best_te:
            best_te, best_lag = te, lag
    return best_lag


def transfer_entropy_matrix(
    equity_returns: pd.DataFrame,
    commodity_returns: pd.DataFrame,
    lag: int = 0,         # 0 = auto-select optimal lag per pair (recommended)
    n_bins: int = 5,
    max_lag: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns two DataFrames:
      te_cmd_to_eq[equity, commodity] = TE(commodity → equity) at optimal lag
      te_eq_to_cmd[equity, commodity] = TE(equity → commodity) at optimal lag

    When lag=0 (default), the optimal lag (1..max_lag) is selected per pair
    by maximising TE(commodity → equity). This captures the 2-5 day transmission
    delay that commodity shocks take to fully propagate into equity prices.
    """
    eq_cols  = equity_returns.columns.tolist()
    cmd_cols = commodity_returns.columns.tolist()

    te_c2e = pd.DataFrame(index=eq_cols, columns=cmd_cols, dtype=float)
    te_e2c = pd.DataFrame(index=eq_cols, columns=cmd_cols, dtype=float)

    for eq in eq_cols:
        for cmd in cmd_cols:
            r_eq  = equity_returns[eq]
            r_cmd = commodity_returns[cmd]

            # Auto-select the lag that maximises commodity→equity TE
            _lag = optimal_te_lag(r_cmd, r_eq, max_lag=max_lag, n_bins=n_bins) if lag == 0 else lag

            te_c2e.loc[eq, cmd] = transfer_entropy(r_cmd, r_eq,  _lag, n_bins)
            # Use same lag for reverse direction for comparability
            te_e2c.loc[eq, cmd] = transfer_entropy(r_eq,  r_cmd, _lag, n_bins)

    return te_c2e.astype(float), te_e2c.astype(float)


def net_flow_matrix(
    te_c2e: pd.DataFrame,
    te_e2c: pd.DataFrame,
) -> pd.DataFrame:
    """Net transfer entropy: TE(cmd→eq) − TE(eq→cmd). Positive = commodity leads."""
    return te_c2e - te_e2c


# ── Diebold-Yilmaz spillover index ────────────────────────────────────────

def diebold_yilmaz(
    returns: pd.DataFrame,
    lag_order: int = 4,
    horizon: int = 10,
    top_n: int = 6,
) -> dict:
    """
    Diebold-Yilmaz (2012) forecast error variance decomposition spillover index.

    Returns full FROM/TO/NET decomposition per asset plus total spillover index.

    FROM[i] = % of i's forecast variance explained by shocks from ALL OTHER assets
              → how much asset i is a net RECEIVER
    TO[j]   = % of ALL OTHER assets' variance explained by shocks from j
              → how much asset j is a net TRANSMITTER
    NET[i]  = TO[i] - FROM[i]  (positive = transmitter, negative = receiver)
    Total spillover index = mean(FROM) = (sum of all off-diagonal) / N

    Returns:
      spillover_table  : pd.DataFrame (n×n) — raw FEVD in %
      from_spillover   : pd.Series — received variance from others per asset (%)
      to_spillover     : pd.Series — variance sent to others per asset (%)
      net_spillover    : pd.Series — TO - FROM per asset (+ = transmitter)
      total_spillover  : float — overall interconnectedness (%)
      top_transmitter  : str — asset with highest net positive spillover
      top_receiver     : str — asset with most negative net spillover
      direction_label  : str — "Commodity → Equity dominant" or "Equity → Commodity dominant"
    """
    data = returns.dropna(how="all").iloc[:, :top_n]
    data = data.dropna()

    _empty = {
        "spillover_table": pd.DataFrame(),
        "from_spillover":  pd.Series(dtype=float),
        "to_spillover":    pd.Series(dtype=float),
        "net_spillover":   pd.Series(dtype=float),
        "total_spillover": np.nan,
        "top_transmitter": "",
        "top_receiver":    "",
        "direction_label": "",
    }

    if len(data) < lag_order * 10:
        return _empty

    try:
        model  = VAR(data)
        result = model.fit(lag_order)
        fevd   = result.fevd(horizon)

        n = len(data.columns)
        # fevd.decomp[-1] shape: (n_variables, n_variables)
        # table[i, j] = % of asset i's variance explained by asset j's shock
        table = pd.DataFrame(
            fevd.decomp[-1] * 100,
            index=data.columns,
            columns=data.columns,
        )

        # Zero the diagonal for off-diagonal sums
        tbl_offdiag = table.copy()
        np.fill_diagonal(tbl_offdiag.values, 0.0)

        # FROM[i]: sum of row i excluding diagonal — how much i receives
        from_sp = tbl_offdiag.sum(axis=1)

        # TO[j]: sum of column j excluding diagonal — how much j sends
        to_sp = tbl_offdiag.sum(axis=0)

        # NET[i] = TO[i] - FROM[i]
        net_sp = to_sp - from_sp

        # Total spillover = mean of FROM (= off-diagonal sum / N)
        total_sp = float(from_sp.mean())

        # Direction: are equity assets net transmitters or receivers on average?
        # Positive net_sp for equities → equity-led; positive for commodities → commodity-led
        # Identify equity vs commodity by column name heuristics
        _EQ_KEYWORDS = ["S&P", "DAX", "Nikkei", "FTSE", "Nasdaq", "Sensex",
                        "Hang Seng", "CAC", "Shanghai", "Nifty", "DJIA",
                        "Russell", "TOPIX", "CSI", "Eurostoxx"]
        eq_assets  = [c for c in data.columns if any(k in c for k in _EQ_KEYWORDS)]
        cmd_assets = [c for c in data.columns if c not in eq_assets]

        eq_net_avg  = float(net_sp[eq_assets].mean())  if eq_assets  else 0.0
        cmd_net_avg = float(net_sp[cmd_assets].mean()) if cmd_assets else 0.0

        if cmd_net_avg > eq_net_avg + 2.0:
            direction = "Commodity → Equity dominant"
        elif eq_net_avg > cmd_net_avg + 2.0:
            direction = "Equity → Commodity dominant"
        else:
            direction = "Bidirectional / no clear leader"

        top_tx  = str(net_sp.idxmax())
        top_rx  = str(net_sp.idxmin())

        return {
            "spillover_table": table.round(2),
            "from_spillover":  from_sp.round(2),
            "to_spillover":    to_sp.round(2),
            "net_spillover":   net_sp.round(2),
            "total_spillover": round(total_sp, 2),
            "top_transmitter": top_tx,
            "top_receiver":    top_rx,
            "direction_label": direction,
        }
    except Exception:
        return _empty


def rolling_diebold_yilmaz(
    returns: pd.DataFrame,
    window: int = 200,
    step: int = 5,
    lag_order: int = 2,
    horizon: int = 10,
) -> pd.DataFrame:
    """
    Rolling Diebold-Yilmaz total spillover index over time.

    Fits a VAR every `step` days on a rolling `window`-day window and records:
      - total_spillover (%)
      - top transmitter name
      - top receiver name

    Returns DataFrame with DatetimeIndex and columns
    [total_spillover, top_transmitter, top_receiver].

    Uses lag_order=2 and smaller window by default for speed; increase for precision.
    """
    data = returns.dropna(how="all").dropna()
    if len(data) < window:
        return pd.DataFrame(columns=["total_spillover", "top_transmitter", "top_receiver"])

    records = []
    indices = range(window, len(data) + 1, step)

    for end_i in indices:
        chunk = data.iloc[end_i - window: end_i]
        try:
            model  = VAR(chunk)
            result = model.fit(lag_order)
            fevd   = result.fevd(horizon)
            n      = len(chunk.columns)
            tbl    = pd.DataFrame(
                fevd.decomp[-1] * 100,
                index=chunk.columns, columns=chunk.columns,
            )
            np.fill_diagonal(tbl.values, 0.0)
            from_sp = tbl.sum(axis=1)
            to_sp   = tbl.sum(axis=0)
            net_sp  = to_sp - from_sp
            total   = float(from_sp.mean())
            records.append({
                "date":            data.index[end_i - 1],
                "total_spillover": round(total, 2),
                "top_transmitter": str(net_sp.idxmax()),
                "top_receiver":    str(net_sp.idxmin()),
            })
        except Exception:
            continue

    if not records:
        return pd.DataFrame(columns=["total_spillover", "top_transmitter", "top_receiver"])

    df = pd.DataFrame(records).set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    return df


def regime_conditional_spillover(
    returns: pd.DataFrame,
    regimes: pd.Series,
    lag_order: int = 3,
    horizon: int = 10,
) -> dict[int, dict]:
    """
    Compute DY total spillover stratified by correlation regime (0-3).

    For each regime, fits a VAR on the subset of dates in that regime and
    returns total spillover, average FROM, and top transmitter.

    Returns {regime_id: {total_spillover, from_mean, to_mean, top_tx, n_obs}}.
    """
    _REGIME_NAMES = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis"}
    results: dict[int, dict] = {}

    data = returns.dropna(how="all").dropna()

    for regime_id in range(4):
        reg_dates = regimes[regimes == regime_id].index
        subset = data.loc[data.index.intersection(reg_dates)]

        if len(subset) < lag_order * 15:
            results[regime_id] = {
                "regime_name":    _REGIME_NAMES[regime_id],
                "total_spillover": np.nan,
                "from_mean":      np.nan,
                "to_mean":        np.nan,
                "top_transmitter": "—",
                "n_obs":          len(subset),
            }
            continue

        try:
            model  = VAR(subset)
            result = model.fit(lag_order)
            fevd   = result.fevd(horizon)
            tbl    = pd.DataFrame(
                fevd.decomp[-1] * 100,
                index=subset.columns, columns=subset.columns,
            )
            np.fill_diagonal(tbl.values, 0.0)
            from_sp = tbl.sum(axis=1)
            to_sp   = tbl.sum(axis=0)
            net_sp  = to_sp - from_sp
            results[regime_id] = {
                "regime_name":     _REGIME_NAMES[regime_id],
                "total_spillover": round(float(from_sp.mean()), 2),
                "from_mean":       round(float(from_sp.mean()), 2),
                "to_mean":         round(float(to_sp.mean()), 2),
                "top_transmitter": str(net_sp.idxmax()),
                "n_obs":           len(subset),
            }
        except Exception:
            results[regime_id] = {
                "regime_name":     _REGIME_NAMES[regime_id],
                "total_spillover": np.nan,
                "from_mean":       np.nan,
                "to_mean":         np.nan,
                "top_transmitter": "—",
                "n_obs":           len(subset),
            }

    return results
