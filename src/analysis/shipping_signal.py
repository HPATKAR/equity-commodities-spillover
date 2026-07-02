"""
Lead-lag cross-correlation between chokepoint ship counts and crude oil returns.

Tests both directions:
  shipping → price  (lag > 0): shipping changes at t predict crude returns at t+lag
  price → shipping  (lag < 0): crude returns at t predict shipping changes at t-lag

Data constraints:
  - IMF PortWatch has ~7-day publication lag; lags 0–6 are unreliable for real-time use
  - Typical sample ~252–365 daily obs; 95% CI threshold is ~0.10–0.13
  - Day-over-day % changes used for stationarity; capacity_tanker used where available

No forward-filling or look-ahead: data aligned on shared observed dates only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_LAG    = 10
MIN_OBS    = 60   # below this return "insufficient data"
OIL_FRAC   = 0.60 # n_tanker → oil proxy (strips LNG, LPG)


# ── Core CCF computation ──────────────────────────────────────────────────────

def _ccf(x: np.ndarray, y: np.ndarray, lags: range) -> dict[int, float]:
    """
    Cross-correlation: corr(x[t], y[t+lag]) for each lag.
    Positive lag → x leads y. Negative lag → y leads x.
    Uses unbiased (overlapping) estimator.
    """
    mu_x, mu_y = x.mean(), y.mean()
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    if sx < 1e-10 or sy < 1e-10:
        return {lag: 0.0 for lag in lags}
    xz = (x - mu_x) / sx
    yz = (y - mu_y) / sy
    out: dict[int, float] = {}
    for lag in lags:
        if lag == 0:
            out[lag] = float(np.mean(xz * yz))
        elif lag > 0:
            # x leads y: align x[0..n-lag] with y[lag..n]
            out[lag] = float(np.mean(xz[:-lag] * yz[lag:]))
        else:
            # y leads x (lag < 0): align y[0..n+lag] with x[-lag..n]
            k = -lag
            out[lag] = float(np.mean(yz[:-k] * xz[k:]))
    return out


def compute_lead_lag(
    ship_series: pd.Series,
    price_series: pd.Series,
    max_lag: int = MAX_LAG,
    min_obs: int = MIN_OBS,
) -> dict:
    """
    Cross-correlation analysis between ship-count changes and crude returns.

    Parameters
    ----------
    ship_series  : pd.Series — daily ship counts (n_tanker or oil_tanker), date-indexed
    price_series : pd.Series — daily crude price levels (WTI or Brent), date-indexed

    Returns
    -------
    dict with:
      peak_lag   : int | None — positive = shipping leads price, negative = price leads shipping
      direction  : str — "shipping → price" | "price → shipping" | "contemporaneous" | "no signal" | "insufficient data"
      r          : float — correlation at peak lag
      significant: bool — |r| > 2/sqrt(n) (approx 95% CI under white-noise null)
      n_obs      : int — aligned observations used
      ci95       : float — significance threshold
      ccf        : dict[int, float] — full CCF at each lag
      sign_label : str — "positive" | "negative" (sign of r at peak)
      note       : str — honest data-quality note
    """
    ship_chg  = ship_series.pct_change().replace([np.inf, -np.inf], np.nan)
    price_ret = price_series.pct_change().replace([np.inf, -np.inf], np.nan)

    df = pd.DataFrame({"ship": ship_chg, "price": price_ret}).dropna()
    n  = len(df)

    _empty = {
        "peak_lag": None, "direction": "insufficient data",
        "r": 0.0, "significant": False, "n_obs": n,
        "ci95": 0.0, "ccf": {}, "sign_label": "—",
        "note": f"Only {n} aligned observations (minimum {min_obs} required).",
    }
    if n < min_obs:
        return _empty

    x = df["ship"].values
    y = df["price"].values

    lags = range(-max_lag, max_lag + 1)
    ccf  = _ccf(x, y, lags)

    ci95 = 2.0 / np.sqrt(n)

    # Peak by absolute magnitude
    peak_lag = max(ccf, key=lambda k: abs(ccf[k]))
    peak_r   = ccf[peak_lag]
    significant = abs(peak_r) > ci95

    if not significant:
        direction = "no signal"
    elif peak_lag > 0:
        direction = "shipping → price"
    elif peak_lag < 0:
        direction = "price → shipping"
    else:
        direction = "contemporaneous"

    sign_label = "positive" if peak_r > 0 else "negative"

    # Honest note about data quality
    note_parts = [f"n={n} daily obs."]
    if peak_lag is not None and abs(peak_lag) <= 6:
        note_parts.append("Peak lag ≤6 days — may overlap ~7d PortWatch publication lag.")
    if n < 120:
        note_parts.append("Short sample; treat as indicative.")

    return {
        "peak_lag":   peak_lag,
        "direction":  direction,
        "r":          round(peak_r, 3),
        "significant": significant,
        "n_obs":      n,
        "ci95":       round(ci95, 3),
        "ccf":        {k: round(v, 3) for k, v in ccf.items()},
        "sign_label": sign_label,
        "note":       " ".join(note_parts),
    }


# ── Multi-chokepoint runner ───────────────────────────────────────────────────

# portid → display label for the 5 tracked straits
_STRAIT_LABELS: dict[str, str] = {
    "hormuz":       "Hormuz",
    "malacca":      "Malacca",
    "suez":         "Suez",
    "bab_el_mandeb": "Bab-el-Mandeb",
    "turkish":      "Turkish Straits",
}

# turkish straits has no direct PortWatch portid; use Bosporus proxy if available
_STRAIT_PORTIDS: dict[str, str] = {
    "hormuz":        "chokepoint6",
    "malacca":       "chokepoint5",
    "suez":          "chokepoint1",
    "bab_el_mandeb": "chokepoint4",
    "turkish":       "chokepoint3",  # Bosporus
}


def run_all_strait_lead_lag(
    price_series: pd.Series,
    days: int = 400,
    max_lag: int = MAX_LAG,
) -> list[dict]:
    """
    Run lead-lag analysis for all 5 straits against `price_series` (WTI or Brent).

    Calls load_strait_tankers() for each portid — uses its own @st.cache_data TTL.
    Returns list of result dicts, each augmented with 'strait_id' and 'strait_label'.
    """
    from src.data.portwatch import load_strait_tankers

    results: list[dict] = []

    for strait_id, portid in _STRAIT_PORTIDS.items():
        label = _STRAIT_LABELS[strait_id]
        try:
            df = load_strait_tankers(portid, days=days)
            if df.empty or len(df) < MIN_OBS:
                result = {
                    "peak_lag": None, "direction": "no data",
                    "r": 0.0, "significant": False,
                    "n_obs": len(df), "ci95": 0.0, "ccf": {},
                    "sign_label": "—",
                    "note": "PortWatch returned insufficient data for this strait.",
                }
            else:
                ship_s = df.set_index("date")["n_tanker"]
                result = compute_lead_lag(ship_s, price_series, max_lag=max_lag)
        except Exception as exc:
            result = {
                "peak_lag": None, "direction": "error",
                "r": 0.0, "significant": False, "n_obs": 0, "ci95": 0.0,
                "ccf": {}, "sign_label": "—", "note": "Data fetch failed.",
            }

        result["strait_id"]    = strait_id
        result["strait_label"] = label
        results.append(result)

    return results
