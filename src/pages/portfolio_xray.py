"""
Portfolio X-Ray - point the Book Risk Character audit at YOUR book.

The Trade Ideas page audits the terminal's own trade book (factor attribution,
factor-neutral skill test, rolling exposures, cost and capacity, hedge overlay,
out-of-sample validation). This page runs the identical suite on a portfolio the
user supplies as tickers and weights. The audit compute functions already accept
a generic `book`, so this is input plumbing over the existing engine, not new
analytics.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd
import streamlit as st

from src.ui.shared import (_page_header, _page_footer, _page_intro,
                           _definition_block)
# Reuse the audit engine + renderers from the Trade Ideas page.
from src.pages.trade_ideas import (
    _render_book_factor_decomp, _render_factor_neutral_skill,
    _render_rolling_exposures, _render_book_costs_capacity,
    _render_hedge_overlay, _render_hedge_oos,
    _compute_book_factor_decomp, _compute_factor_neutral_skill,
    _compute_rolling_exposures, _compute_book_costs_capacity,
    _compute_hedge_overlay, _compute_hedge_oos,
)

_M = "font-family:'JetBrains Mono',monospace;"


@st.cache_data(show_spinner=False, ttl=3600, max_entries=6)
def _load_portfolio_returns(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    """Daily log-returns for the user's tickers, keyed by ticker. Disk-cached.
    Empty frame on total failure; individual bad tickers are simply absent."""
    if not tickers:
        return pd.DataFrame()
    from src.utils.artifact_cache import read_artifact, write_artifact
    _key = f"pxr_{'_'.join(sorted(tickers))}_{end}"
    _hit = read_artifact(_key, max_age_s=3600)
    if _hit is not None:
        return _hit
    try:
        from src.data.loader import _yf_download
        _floor = str(_dt.date.today() - _dt.timedelta(days=6 * 365))
        _s = _floor if start < _floor else start
        raw = _yf_download(list(tickers), start=_s, end=end,
                           auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return pd.DataFrame()
        close = raw["Close"] if "Close" in raw.columns else raw
        if not hasattr(close, "columns"):          # single ticker -> Series
            close = close.to_frame(tickers[0])
        ret = np.log(close / close.shift(1)).dropna(how="all")
        ret = ret.loc[:, [c for c in tickers if c in ret.columns]]
        if not ret.empty:
            write_artifact(_key, ret)
        return ret
    except Exception:
        return pd.DataFrame()


def page_portfolio_xray(start: str, end: str, fred_key: str = "") -> None:
    _page_header(
        "Portfolio X-Ray",
        "Point the Book Risk Character audit at your own book")
    _page_intro(
        "This runs the exact suite the Trade Ideas page runs on the terminal's own "
        "book, on a portfolio <strong>you</strong> supply. Enter your holdings as "
        "tickers and weights, and the terminal prosecutes them the same way: factor "
        "attribution, a Fama-French factor-neutral skill test, rolling exposures, "
        "cost and capacity, a tradeable hedge overlay, and its out-of-sample "
        "validation. It tells you what your book actually is (factor and risk "
        "exposure), and what it is not (selection alpha)."
    )
    _definition_block(
        "What to enter",
        "Any liquid, yfinance-recognised tickers (US equities, ETFs, ADRs). Weights "
        "are normalised automatically, so they need not sum to 100. The book is "
        "treated as long-only for the attribution; short books are on the roadmap. "
        "Numbers are measured on real market returns, and the same honest caveats "
        "apply as on the terminal's own book (a modest risk signal, not proven "
        "alpha; the out-of-sample window is a single macro era)."
    )

    # ── Input: ticker + weight table ─────────────────────────────────────────
    if "_pxr_rows" not in st.session_state:
        st.session_state["_pxr_rows"] = pd.DataFrame({
            "Ticker": ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "GLD", "TLT"],
            "Weight %": [18.0, 16.0, 16.0, 14.0, 12.0, 14.0, 10.0],
        })
    _c1, _c2 = st.columns([3, 1.1], gap="medium")
    with _c1:
        edited = st.data_editor(
            st.session_state["_pxr_rows"], num_rows="dynamic", width="stretch",
            key="_pxr_editor",
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", width="medium",
                                                      help="e.g. AAPL, SPY, GLD"),
                "Weight %": st.column_config.NumberColumn("Weight %", min_value=0.0,
                                                          step=1.0, format="%.1f"),
            })
    with _c2:
        st.markdown('<div style="height:.2rem"></div>', unsafe_allow_html=True)
        _go = st.button("Run X-Ray", type="primary", width="stretch",
                        help="Fetches returns and runs the full audit (~10-20s first time)")
        if _go:
            st.session_state["_pxr_run"] = edited.copy()
            st.session_state.pop("_pxr_pdf", None)      # invalidate stale tearsheet

    _run = st.session_state.get("_pxr_run")
    if _run is None:
        st.info("Enter your holdings above and press **Run X-Ray**.")
        _page_footer()
        return

    # ── Parse + normalise ────────────────────────────────────────────────────
    df = _run.copy()
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df = df[df["Ticker"].str.len() > 0]
    df["Weight %"] = pd.to_numeric(df["Weight %"], errors="coerce").fillna(0.0)
    df = df[df["Weight %"] > 0]
    df = df.drop_duplicates(subset="Ticker", keep="first")
    if len(df) < 2:
        st.warning("Enter at least two positions with positive weights.")
        _page_footer()
        return
    tickers = tuple(df["Ticker"].tolist())
    w = df.set_index("Ticker")["Weight %"]

    with st.spinner("Fetching returns and running the audit..."):
        R = _load_portfolio_returns(tickers, start, end)
    if R.empty or R.shape[1] < 2:
        st.error("Could not load returns for those tickers. Check the symbols "
                 "(they must be yfinance-recognised) and try again.")
        _page_footer()
        return

    good = [t for t in tickers if t in R.columns]
    dropped = [t for t in tickers if t not in R.columns]
    if dropped:
        st.warning("No data for: **" + ", ".join(dropped) + "** (dropped from the book).")
    if len(good) < 2:
        st.error("Fewer than two positions have usable data. Check the symbols.")
        _page_footer()
        return
    w = w.loc[good]
    w = w / w.sum()
    all_r_gate = R[good].dropna(how="all")

    # Build the book in the audit engine's shape: each holding is a single-leg
    # long trade. `ticker` is set explicitly so cost/capacity resolves ADV.
    book = [{"name": t, "ticker": t, "assets": [t], "direction": ["Long"],
             "alloc_weight": float(w[t]), "holding_period": "8 weeks"}
            for t in good]

    # ── Portfolio summary strip ──────────────────────────────────────────────
    _obs = len(all_r_gate.dropna())
    _top = w.sort_values(ascending=False)
    _rows = " · ".join(f'{t} {w[t]*100:.0f}%' for t in _top.index[:8])
    if len(_top) > 8:
        _rows += f' · +{len(_top)-8} more'
    st.markdown(
        f'<div style="border:1px solid #1e1e1e;background:#0a0a0a;padding:.5rem .8rem;'
        f'margin:.2rem 0 .6rem"><span style="{_M}font-size:.6rem;font-weight:700;'
        f'letter-spacing:.14em;color:#e8e9ed">YOUR BOOK</span>'
        f'<span style="{_M}font-size:.5rem;color:#8890a1;margin-left:8px">'
        f'{len(good)} positions · {_obs} obs · long-only, weight-normalised</span>'
        f'<div style="{_M}font-size:.56rem;color:#c9ccd4;margin-top:4px">{_rows}</div></div>',
        unsafe_allow_html=True,
    )

    # ── Run the full Book Risk Character suite on the user's book ─────────────
    for _fn, _args in (
        (_render_book_factor_decomp,  (book, all_r_gate, start, end)),
        (_render_factor_neutral_skill, (book, all_r_gate, start, end, max(len(good), 1))),
        (_render_rolling_exposures,   (book, all_r_gate, start, end)),
        (_render_book_costs_capacity, (book, all_r_gate, end)),
        (_render_hedge_overlay,       (book, all_r_gate, start, end)),
        (_render_hedge_oos,           (book, all_r_gate, start, end)),
    ):
        try:
            _fn(*_args)
        except Exception:
            pass

    st.caption("Each panel is the same computation the Trade Ideas page runs on the "
               "terminal's book. Illustrative, not investment advice.")

    # ── White-label client / IC / LP tearsheet (PDF export) ──────────────────
    st.markdown(
        f'<div style="border-top:1px solid #1e1e1e;margin:1.1rem 0 .5rem;padding-top:.7rem">'
        f'<span style="{_M}font-size:.7rem;font-weight:700;letter-spacing:.1em;'
        f'color:#e8e9ed">CLIENT TEARSHEET</span>'
        f'<div style="{_M}font-size:.56rem;color:#8890a1;margin-top:2px">Export this '
        f'audit as a white-label, client- or IC-ready PDF risk tearsheet on your own '
        f'firm name. Same numbers, your letterhead.</div></div>',
        unsafe_allow_html=True)
    _t1, _t2, _t3, _t4 = st.columns([1.4, 1.3, 1.3, 1.0], gap="medium")
    with _t1:
        _firm = st.text_input("Firm name", value=st.session_state.get("_pxr_firm", ""),
                              placeholder="Your Firm LLP", key="_pxr_firm")
    with _t2:
        _prep = st.text_input("Prepared for", value="Investment Committee",
                             key="_pxr_prep")
    with _t3:
        _blab = st.text_input("Book label", value="Portfolio", key="_pxr_blab")
    with _t4:
        st.markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
        _mk = st.button("Generate Tearsheet (PDF)", width="stretch",
                        help="Builds a 6-page white-label risk tearsheet of this book")

    if _mk:
        try:
            with st.spinner("Building the tearsheet (~10s)..."):
                from src.reports.report_generator import generate_tearsheet
                _fd = _compute_book_factor_decomp(book, all_r_gate, start, end)
                _sk = _compute_factor_neutral_skill(book, all_r_gate, start, end,
                                                    max(len(good), 1))
                _ro = _compute_rolling_exposures(book, all_r_gate, start, end)
                _co = _compute_book_costs_capacity(book, all_r_gate, end)
                _he = _compute_hedge_overlay(book, all_r_gate, start, end)
                _oos = _compute_hedge_oos(book, all_r_gate, start, end)
                _pdf = generate_tearsheet(
                    book_rows=[(t, float(w[t] * 100)) for t in good],
                    firm=_firm or "Your Firm", prepared_for=_prep, book_label=_blab,
                    start=start, end=end,
                    factor_decomp=_fd, skill_decomp=_sk, rolling_decomp=_ro,
                    cost_decomp=_co, hedge_decomp=_he, hedge_oos_decomp=_oos)
            st.session_state["_pxr_pdf"] = _pdf
            st.session_state["_pxr_pdf_name"] = (
                (_blab or "portfolio").strip().replace(" ", "_").lower()
                + "_risk_tearsheet.pdf")
        except Exception as _e:
            st.error(f"Could not build the tearsheet: {_e}")

    if st.session_state.get("_pxr_pdf"):
        st.download_button(
            "Download Tearsheet (PDF)", data=st.session_state["_pxr_pdf"],
            file_name=st.session_state.get("_pxr_pdf_name", "risk_tearsheet.pdf"),
            mime="application/pdf", width="stretch")

    _page_footer()
