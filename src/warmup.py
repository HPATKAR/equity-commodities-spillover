"""
Cache warm-up — runs once per process start in a background daemon thread.

Populates the in-process @st.cache_data store for Trade Ideas page
before any user visits, so the first real visitor always hits a warm cache.

Re-schedules itself every INTERVAL_H hours so the cache never expires on
idle servers (backtest TTL = 3600 s = 1 hr; data TTL = 1800 s).
"""
from __future__ import annotations

import logging
import threading
import time

_log = logging.getLogger("warmup")
INTERVAL_H = 3          # re-run 3 h after each completion (before TTL expires)

_lock    = threading.Lock()
_started = False         # module-level: True after the thread is first launched
_timer: threading.Timer | None = None


# ── Core warm-up logic ────────────────────────────────────────────────────────

def _run() -> None:
    """Execute all pre-warm calls. Never raises — errors are logged and swallowed."""
    t0 = time.monotonic()
    try:
        import pandas as pd
        from src.data.loader import load_equity_prices, load_commodity_prices, load_returns
        from src.analysis.correlations import average_cross_corr_series, detect_correlation_regime

        # 1. Data pipeline — network-heavy, cached 1800 s.
        # Warm both the default key (2005-01-01) and the app's typical key
        # (2010-01-01) so the Trade Ideas page primary @st.cache_data call hits
        # a warm cache rather than a cold yfinance download.
        from datetime import date as _date
        _today = str(_date.today())
        _app_start = "2010-01-01"
        load_equity_prices(_app_start, _today)
        load_commodity_prices(_app_start, _today)
        eq_r, cmd_r = load_returns(_app_start, _today)

        if eq_r.empty or cmd_r.empty:
            # Fall back to default date range as safety net
            load_equity_prices(); load_commodity_prices()
            eq_r, cmd_r = load_returns()

        if eq_r.empty or cmd_r.empty:
            _log.warning("warmup: return data empty — aborting")
            _reschedule()
            return

        # 2. Correlation pipeline
        all_r    = pd.concat([eq_r, cmd_r], axis=1).sort_index()
        avg_corr = average_cross_corr_series(eq_r, cmd_r)
        detect_correlation_regime(avg_corr)
        n_corr   = len(avg_corr)

        # 3. Walk-forward backtests for every static trade card, cached 3600 s
        from src.pages.trade_ideas import (
            _TRADE_LIBRARY, _wf_backtest_trade, _parse_holding_days,
        )
        for trade in _TRADE_LIBRARY:
            try:
                _wf_backtest_trade(
                    _all_r=all_r,
                    _avg_corr=avg_corr,
                    trade_name=trade["name"],
                    trigger_regimes=list(trade["regime"]),
                    assets=list(trade["assets"]),
                    directions=list(trade["direction"]),
                    holding_days=_parse_holding_days(trade),
                    leg_weights=None,
                    avg_corr_n=n_corr,
                )
            except Exception as exc:
                _log.debug("warmup: backtest skipped for '%s': %s", trade["name"], exc)

        # 4. Stock-price fetch — warms the yfinance connection for the full universe
        from src.pages.trade_ideas import _fetch_stock_prices
        _fetch_stock_prices(sectors=())

        _log.info("warmup: complete in %.1f s", time.monotonic() - t0)

    except Exception as exc:
        _log.warning("warmup: failed after %.1f s: %s", time.monotonic() - t0, exc)

    _reschedule()


def _reschedule() -> None:
    global _timer
    t = threading.Timer(INTERVAL_H * 3600, _run)
    t.daemon = True
    with _lock:
        _timer = t
    t.start()


# ── Public entry point ────────────────────────────────────────────────────────

def start() -> None:
    """
    Launch the warm-up daemon. Idempotent — safe to call on every Streamlit
    rerun because the module-level _started flag is set only once per process.
    """
    global _started
    if _started:
        return
    with _lock:
        if _started:
            return
        _started = True
    threading.Thread(target=_run, daemon=True, name="cache-warmup").start()
