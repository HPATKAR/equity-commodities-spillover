"""
Cache warm-up - runs once per process start in a background daemon thread.

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

def _run(reschedule: bool = True) -> None:
    """Execute all pre-warm calls. Never raises - errors are logged and swallowed.

    reschedule=True (default) re-arms the background timer for the in-process
    daemon. The standalone precompute entrypoint calls _run(reschedule=False):
    it warms + persists the artifact cache once and exits (no daemon timer)."""
    t0 = time.monotonic()
    try:
        import pandas as pd
        from src.data.loader import load_equity_prices, load_commodity_prices, load_returns
        from src.analysis.correlations import average_cross_corr_series, detect_correlation_regime

        # 1. Data pipeline - network-heavy, cached 1800 s.
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
            _log.warning("warmup: return data empty - aborting")
            _reschedule()
            return

        # 2. Correlation pipeline
        all_r    = pd.concat([eq_r, cmd_r], axis=1).sort_index()
        avg_corr = average_cross_corr_series(eq_r, cmd_r)
        detect_correlation_regime(avg_corr)
        n_corr   = len(avg_corr)

        # Guarded warmer - one slow source can't abort the rest. (Background
        # thread has no ScriptRunContext; @st.cache_data still populates the
        # shared process cache - the "missing ScriptRunContext" warning is benign.)
        def _warm(fn, *a, **k):
            try:
                fn(*a, **k)
            except Exception as exc:
                _log.debug("warmup: %s skipped: %s", getattr(fn, "__name__", fn), exc)

        # 3. LANDING-PAGE essentials FIRST - the deployed link opens on the
        # command center, so warm what it renders (conflict scores, portfolio
        # aggregate, the yfinance market tape, exposure) BEFORE the heavier
        # Trade-Ideas backtests, so the first visitor to the landing page hits
        # warm caches soonest.
        from src.analysis.conflict_model import (
            score_all_conflicts, aggregate_portfolio_scores)
        _cr = score_all_conflicts()
        _warm(aggregate_portfolio_scores, _cr)
        from src.pages.home import _load_market_pulse
        _warm(_load_market_pulse)
        try:
            from src.analysis.exposure import score_all_assets
            _warm(score_all_assets)
        except Exception:
            pass

        # Full Command Center render essentials - the 3-layer risk score and the
        # hot-stocks RSS feed, the two remaining home cold calls (both disk-backed
        # now, so this also persists their artifacts for a cold web process).
        try:
            from src.pages.home import _load_market_risk, _load_hot_stocks
            _warm(_load_market_risk, _app_start, _today, "base")
            _warm(_load_hot_stocks)
        except Exception:
            pass

        # 4. Walk-forward backtests for every static trade card, cached 3600 s
        from src.pages.trade_ideas import (
            _TRADE_LIBRARY_BASE as _TRADE_LIBRARY,   # read-only warmup pass
            _wf_backtest_trade, _parse_holding_days,
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

        # 5. Stock-price fetch - warms the yfinance connection for the full universe
        from src.pages.trade_ideas import _fetch_stock_prices
        _fetch_stock_prices(sectors=())

        # 5b. Single-stock log-returns (184-ticker fetch) - the heaviest Trade-
        # Ideas cold cost. _warm both hydrates the in-memory cache AND persists
        # the frame to the artifact cache, so a cold process reads it from disk.
        from src.pages.trade_ideas import _load_stock_returns
        _warm(_load_stock_returns, _app_start, _today)

        # 6. Trade Ideas extra legs (fixed income / FX / private credit) so the
        # ALERTS → Trade Ideas shortcut lands on warm caches too.
        from src.data.loader import (
            load_fixed_income_returns, load_fx_returns, load_private_credit_returns)
        _warm(load_fixed_income_returns, _app_start, _today)
        _warm(load_fx_returns, _app_start, _today)
        _warm(load_private_credit_returns, _app_start, _today)

        _log.info("warmup: complete in %.1f s", time.monotonic() - t0)

    except Exception as exc:
        _log.warning("warmup: failed after %.1f s: %s", time.monotonic() - t0, exc)

    if reschedule:
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
    Launch the warm-up daemon. Idempotent - safe to call on every Streamlit
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
