"""
Page 7 - Portfolio Stress Tester
Build a custom equity/commodity allocation (indices + S&P 500 individual stocks),
assign weights via an interactive table, and stress-test against every historical
geopolitical event in the dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date

from src.data.loader import (
    load_all_prices, load_sp500_prices, get_sp500_constituents,
    load_fixed_income_prices,
)
from src.data.portfolio_loader import (
    build_portfolio, get_portfolio, set_portfolio, clear_portfolio, get_template_csv,
)
from src.data.config import GEOPOLITICAL_EVENTS, PALETTE, EQUITY_REGIONS, COMMODITY_GROUPS
from src.ui.shared import (
    _style_fig, _chart, _page_intro, _thread, _section_note,
    _definition_block, _takeaway_block, _page_conclusion, _page_header, _page_footer,
    _no_api_key_banner,
)


_M = "font-family:'JetBrains Mono',monospace;"
_G = "#CFB991"
_DIM = "#8E9AAA"


def _sh(txt: str) -> None:
    """Styled terminal section header - replaces st.subheader()."""
    st.markdown(
        f'<p style="{_M}font-size:8px;font-weight:700;letter-spacing:.18em;'
        f'color:{_G};text-transform:uppercase;border-bottom:1px solid #1e1e1e;'
        f'padding-bottom:5px;margin:1.4rem 0 .6rem">{txt}</p>',
        unsafe_allow_html=True,
    )


# ── Portfolio maths ────────────────────────────────────────────────────────

def _portfolio_path(prices: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Daily portfolio value (base=100) from price DataFrame and weights dict."""
    assets = [a for a in weights if a in prices.columns]
    if not assets:
        return pd.Series(dtype=float)
    w = np.array([weights[a] for a in assets])
    w = w / w.sum()
    p = prices[assets].dropna(how="all").ffill()
    if p.empty:
        return pd.Series(dtype=float)
    p_norm = p / p.iloc[0]
    return (p_norm * w).sum(axis=1) * 100


def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    roll_max = series.cummax()
    dd = (series - roll_max) / roll_max * 100
    return float(dd.min())


def _sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    if returns.empty or returns.std() < 1e-9:
        return np.nan
    return float((returns.mean() - rf / 252) / returns.std() * np.sqrt(252))


def _event_stats(
    prices: pd.DataFrame,
    weights: dict[str, float],
    event: dict,
    pre_days: int = 30,
    post_days: int = 60,
) -> dict:
    today = date.today()
    t0 = pd.Timestamp(event["start"])
    t1 = pd.Timestamp(min(event["end"], today))

    full_slice = prices.loc[
        t0 - pd.Timedelta(days=pre_days + 5):
        t1 + pd.Timedelta(days=post_days + 5)
    ]
    if full_slice.empty:
        return {}
    port = _portfolio_path(full_slice, weights)
    if port.empty:
        return {}

    log_r = np.log(port / port.shift(1)).dropna()

    def _window_ret(a, b) -> float:
        sl = port.loc[a:b]
        return float((sl.iloc[-1] / sl.iloc[0] - 1) * 100) if len(sl) >= 2 else np.nan

    return {
        "event":      event["label"],
        "name":       event["name"],
        "color":      event["color"],
        "pre_ret":    _window_ret(t0 - pd.Timedelta(days=pre_days), t0),
        "during_ret": _window_ret(t0, t1),
        "post_ret":   _window_ret(t1, t1 + pd.Timedelta(days=post_days)),
        "max_dd":     _max_drawdown(port.loc[t0:t1]),
        "sharpe":     _sharpe(log_r.loc[t0:t1]),
        "port":       port,
        "t0":         t0,
        "t1":         t1,
    }


# ── Page ───────────────────────────────────────────────────────────────────

def page_stress_test(start: str, end: str, fred_key: str = "") -> None:
    _page_header("Stress Lab",
                 "Scenario shocks · Drawdown simulation · Tail risk · Cross-asset correlation stress")
    _no_api_key_banner("AI stress scenario commentary")
    _page_intro(
        "The core finding of spillover research is that equity-commodity correlation spikes during "
        "crises - precisely when diversification is most needed. A portfolio built assuming low "
        "cross-asset correlation will underperform exactly when it matters most. "
        "<strong>This page stress-tests that assumption directly.</strong> "
        "Build any equity-commodity mix and replay it through the historical geopolitical and macro "
        "events catalogued in this dashboard - the same events where spillover was highest. "
        "If your portfolio survives those windows with acceptable drawdown, the diversification "
        "assumption holds. If it does not, the spillover risk is not being priced correctly."
    )

    # ── Geo risk context banner ────────────────────────────────────────────
    try:
        from src.analysis.conflict_model import score_all_conflicts, aggregate_portfolio_scores
        _st_cr  = score_all_conflicts()
        _st_agg = aggregate_portfolio_scores(_st_cr)
        _st_cis = _st_agg.get("portfolio_cis", 50.0)
        _st_tps = _st_agg.get("portfolio_tps", 50.0)
        _st_active = [r for r in _st_cr.values() if r.get("state") == "active"]
        _st_col = "#c0392b" if _st_cis >= 65 else "#e67e22" if _st_cis >= 45 else "#CFB991"
        _st_note = (
            "Elevated active-war CIS - stress test against current conflicts is high-priority."
            if _st_cis >= 60 else
            "Moderate conflict intensity - watch oil/gold channels in event stress windows."
        )
        _st_tags = "".join(
            f'<span style="background:#0a0a1a;color:#8E9AAA;'
            f'font-family:\'JetBrains Mono\',monospace;font-size:7px;'
            f'padding:2px 6px;margin-right:4px;border:1px solid #2a2a2a">'
            f'{r["label"].upper()}&nbsp;CIS&nbsp;{r["cis"]:.0f}</span>'
            for r in sorted(_st_active, key=lambda x: x["cis"], reverse=True)[:3]
        )
        st.markdown(
            f'<div style="background:#080808;border:1px solid #1e1e1e;'
            f'border-left:3px solid {_st_col};padding:.4rem .9rem;'
            f'margin-bottom:.6rem;display:flex;align-items:center;gap:10px;flex-wrap:wrap">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
            f'font-weight:700;color:{_st_col};white-space:nowrap">GEO STRESS CONTEXT</span>'
            f'{_st_tags}'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            f'color:#8E9AAA;margin-left:auto">'
            f'CIS&nbsp;<b style="color:{_st_col}">{_st_cis:.0f}</b>&nbsp;·&nbsp;'
            f'TPS&nbsp;<b style="color:#CFB991">{_st_tps:.0f}</b>&nbsp;·&nbsp;'
            f'{_st_note}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    # ── Portfolio Import ───────────────────────────────────────────────────────
    _existing_portfolio = get_portfolio()
    with st.expander(
        "📂  Import Portfolio  -  CSV / Excel upload · dollar amounts · auto FX conversion · auto weights",
        expanded=(_existing_portfolio is None),
    ):
        _M = "font-family:'JetBrains Mono',monospace;"

        st.markdown(
            '<p style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;color:#8E9AAA;'
            'line-height:1.6;margin-bottom:6px">'
            'Upload your portfolio as a CSV or Excel file. Required columns: '
            '<b style="color:#CFB991">ticker</b>, '
            '<b style="color:#CFB991">dollar_amount</b>. '
            'Optional: <b>currency</b> (ISO 4217 - non-USD converted at live spot rate), '
            '<b>cusip</b>, <b>isin</b>, <b>name</b>, <b>sector</b>. '
            'Weights are computed automatically from dollar amounts - no manual percentage entry needed.'
            '</p>',
            unsafe_allow_html=True,
        )

        _ic1, _ic2, _ic3 = st.columns([2, 1, 1])

        with _ic1:
            _uploaded = st.file_uploader(
                "Portfolio file",
                type=["csv", "xlsx", "xls"],
                key="portfolio_upload",
                label_visibility="collapsed",
            )

        with _ic2:
            st.download_button(
                "Download Template CSV",
                data=get_template_csv(),
                file_name="portfolio_template.csv",
                mime="text/csv",
                width="stretch",
            )

        with _ic3:
            if _existing_portfolio and st.button(
                "Clear Portfolio", width="stretch", key="clear_pf_btn"
            ):
                clear_portfolio()
                st.rerun()

        if _uploaded is not None:
            with st.spinner("Fetching live prices and FX rates…"):
                try:
                    _pf = build_portfolio(_uploaded)
                    set_portfolio(_pf)
                    _existing_portfolio = _pf
                    st.success(
                        f"Portfolio loaded - {_pf['n']} positions · "
                        f"Total NAV: ${_pf['total_usd']:,.0f}"
                    )
                    for w in _pf.get("warnings", []):
                        st.warning(w)
                    for e in _pf.get("errors", [])[:5]:
                        st.caption(e)
                except ValueError as e:
                    st.error(f"Could not parse file: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

        # ── Show current portfolio summary ─────────────────────────────────
        if _existing_portfolio:
            _pf = _existing_portfolio
            positions = _pf["positions"]
            # Sort by weight desc
            top_pos = sorted(positions, key=lambda p: p["weight"], reverse=True)

            st.markdown(
                f'<div style="display:flex;gap:16px;flex-wrap:wrap;'
                f'margin:.6rem 0 .3rem;border-top:1px solid #1e1e1e;padding-top:.5rem">'
                f'<div><span style="{_M}font-size:7px;color:#555960">POSITIONS</span><br>'
                f'<span style="{_M}font-size:16px;font-weight:700;color:#e8e9ed">{_pf["n"]}</span></div>'
                f'<div><span style="{_M}font-size:7px;color:#555960">TOTAL NAV (USD)</span><br>'
                f'<span style="{_M}font-size:16px;font-weight:700;color:#CFB991">'
                f'${_pf["total_usd"]:,.0f}</span></div>'
                f'<div><span style="{_M}font-size:7px;color:#555960">LOADED</span><br>'
                f'<span style="{_M}font-size:11px;color:#8E9AAA">{_pf["loaded_at"][:16]}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Mini holdings table
            rows_html = "".join(
                f'<tr style="border-bottom:1px solid #1a1a1a">'
                f'<td style="{_M}font-size:9px;color:#CFB991;padding:3px 8px">{p["ticker"]}</td>'
                f'<td style="{_M}font-size:9px;color:#8E9AAA;padding:3px 8px">'
                f'{p["name"][:28] if p["name"] else "-"}</td>'
                f'<td style="{_M}font-size:9px;color:#e8e9ed;padding:3px 8px">'
                f'{p["sector"][:18] if p["sector"] else "-"}</td>'
                f'<td style="{_M}font-size:9px;color:#e8e9ed;text-align:right;padding:3px 8px">'
                f'${p["dollar_amount_usd"]:,.0f}</td>'
                f'<td style="{_M}font-size:9px;color:#e8e9ed;text-align:right;padding:3px 8px">'
                f'{p["weight"]:.1%}</td>'
                f'<td style="{_M}font-size:9px;color:#555960;text-align:right;padding:3px 8px">'
                f'{p["currency"]}</td>'
                f'</tr>'
                for p in top_pos[:10]
            )
            st.markdown(
                f'<table style="width:100%;border-collapse:collapse">'
                f'<thead><tr>'
                f'<th style="{_M}font-size:7px;color:#555960;text-align:left;'
                f'padding:3px 8px;border-bottom:1px solid #2a2a2a">TICKER</th>'
                f'<th style="{_M}font-size:7px;color:#555960;text-align:left;'
                f'padding:3px 8px;border-bottom:1px solid #2a2a2a">NAME</th>'
                f'<th style="{_M}font-size:7px;color:#555960;text-align:left;'
                f'padding:3px 8px;border-bottom:1px solid #2a2a2a">SECTOR</th>'
                f'<th style="{_M}font-size:7px;color:#555960;text-align:right;'
                f'padding:3px 8px;border-bottom:1px solid #2a2a2a">MKT VALUE (USD)</th>'
                f'<th style="{_M}font-size:7px;color:#555960;text-align:right;'
                f'padding:3px 8px;border-bottom:1px solid #2a2a2a">WEIGHT</th>'
                f'<th style="{_M}font-size:7px;color:#555960;text-align:right;'
                f'padding:3px 8px;border-bottom:1px solid #2a2a2a">CCY</th>'
                f'</tr></thead><tbody>{rows_html}</tbody></table>',
                unsafe_allow_html=True,
            )
            if len(positions) > 10:
                st.caption(f"+ {len(positions) - 10} more positions.")

            # ── Auto-populate weights into session state so Sections A/B pick them up ──
            _pf_weights = _pf["weights"]
            for tk, w in _pf_weights.items():
                st.session_state[f"w_{tk}"]  = round(w * 100, 2)
                st.session_state[f"sw_{tk}"] = round(w * 100, 2)

    with st.spinner("Loading index and commodity price data…"):
        eq_p, cmd_p = load_all_prices(start, end)

    if eq_p.empty or cmd_p.empty:
        st.error("Market data unavailable.")
        return

    try:
        fi_prices = load_fixed_income_prices(start, end)
    except Exception:
        fi_prices = pd.DataFrame()

    # ── Section A: Indices & Commodities ────────────────────────────────────
    _sh("A · Global Indices & Commodities")
    _definition_block(
        "Indices & Commodity Weights",
        "Select indices and commodity futures. Enter weights (%) for each. "
        "Weights across Section A and Section B are combined and renormalised to 100%.",
    )

    all_assets = list(eq_p.columns) + list(cmd_p.columns)
    presets = {
        "60/40 (S&P / Gold)":    {"S&P 500": 60, "Gold": 40},
        "Energy Macro":          {"S&P 500": 40, "WTI Crude Oil": 30, "Gold": 20, "Natural Gas": 10},
        "EM + Commodities":      {"Sensex": 25, "CSI 300": 25, "Copper": 25, "Wheat": 25},
        "Global Diversified":    {"S&P 500": 25, "Eurostoxx 50": 20, "Nikkei 225": 15,
                                  "Gold": 20, "WTI Crude Oil": 10, "Wheat": 10},
        "Custom (no preset)":    {},
    }

    c1, c2 = st.columns([1, 2])
    preset_name = c1.selectbox("Start from preset", list(presets.keys()), key="st_preset")
    preset_assets = list(presets[preset_name].keys()) if presets[preset_name] else []

    selected_assets = c2.multiselect(
        "Select indices / commodity futures",
        all_assets,
        default=[a for a in preset_assets if a in all_assets] or ["S&P 500", "Gold", "WTI Crude Oil"],
        key="st_assets",
    )

    # ── Free-form ticker entry ─────────────────────────────────────────────
    st.markdown(
        '<p style="font-size:0.60rem;font-weight:600;letter-spacing:0.12em;'
        'text-transform:uppercase;color:#8890a1;margin:0.6rem 0 0.2rem">'
        'Or enter stock tickers directly (comma-separated)</p>',
        unsafe_allow_html=True,
    )
    custom_ticker_input = st.text_input(
        "Custom tickers",
        value="",
        placeholder="e.g.  AAPL, MSFT, NVDA, JPM",
        key="custom_tickers",
        label_visibility="collapsed",
    )

    # Parse and fetch custom tickers
    custom_tickers_raw = [
        t.strip().upper().replace(".", "-")
        for t in custom_ticker_input.split(",")
        if t.strip()
    ]
    custom_prices: pd.DataFrame = pd.DataFrame()
    if custom_tickers_raw:
        with st.spinner(f"Fetching prices for {custom_tickers_raw}…"):
            custom_prices = load_sp500_prices(tuple(sorted(set(custom_tickers_raw))), start, end)
        if not custom_prices.empty:
            fetched = [t for t in custom_tickers_raw if t in custom_prices.columns]
            missing = [t for t in custom_tickers_raw if t not in custom_prices.columns]
            if fetched:
                st.success(f"Loaded: {', '.join(fetched)}")
            if missing:
                st.warning(f"Not found: {', '.join(missing)}")
            # Add to eq_p so they appear in all_prices later
            for tk in fetched:
                eq_p[tk] = custom_prices[tk]
            all_assets = list(eq_p.columns) + list(cmd_p.columns)

    weights_a: dict[str, float] = {}
    # Include any selected from the multiselect + fetched custom tickers
    all_selected_a = list(selected_assets) + [
        t for t in custom_tickers_raw if t in eq_p.columns and t not in selected_assets
    ]
    if all_selected_a:
        st.markdown(
            f'<p style="{_M}font-size:7.5px;font-weight:700;letter-spacing:.14em;'
            f'color:{_DIM};text-transform:uppercase;margin:.6rem 0 .3rem">Weights (%)</p>',
            unsafe_allow_html=True,
        )
        default_weights = presets.get(preset_name, {})
        weight_cols = st.columns(min(len(all_selected_a), 5))
        for i, asset in enumerate(all_selected_a):
            default_w = default_weights.get(asset, round(100 / len(all_selected_a), 1))
            w = weight_cols[i % 5].number_input(
                asset, min_value=0.0, max_value=100.0,
                value=float(default_w), step=1.0, key=f"w_{asset}",
            )
            weights_a[asset] = w

    # ── Section C: Fixed Income ──────────────────────────────────────────────
    _sh("C · Fixed Income")
    _definition_block(
        "Fixed Income Weights",
        "Treasuries, IG/HY Credit, TIPS, EM Bonds. "
        "Select instruments and enter weights (%). "
        "Weights are combined with Sections A and B and renormalised to 100%.",
    )

    _fi_asset_names = list(fi_prices.columns) if not fi_prices.empty else [
        "US 20Y+ Treasury (TLT)", "US 7-10Y Treasury (IEF)", "US 1-3Y Treasury (SHY)",
        "IG Corporate (LQD)", "HY Corporate (HYG)", "EM USD Bonds (EMB)", "TIPS / Inflation (TIP)",
    ]

    with st.expander("Fixed Income asset selector", expanded=False):
        _fi_selected = st.multiselect(
            "Select fixed income instruments",
            _fi_asset_names,
            default=[],
            key="st_fi_assets",
            placeholder="Add Treasuries, credit, or TIPS to portfolio…",
        )

    weights_c: dict[str, float] = {}
    if _fi_selected and not fi_prices.empty:
        st.markdown(
            f'<p style="{_M}font-size:7.5px;font-weight:700;letter-spacing:.14em;'
            f'color:{_DIM};text-transform:uppercase;margin:.6rem 0 .3rem">Fixed Income Weights (%)</p>',
            unsafe_allow_html=True,
        )
        _fi_weight_cols = st.columns(min(len(_fi_selected), 5))
        for i, asset in enumerate(_fi_selected):
            if asset in fi_prices.columns:
                _default_fi_w = round(100 / len(_fi_selected), 1)
                _w_fi = _fi_weight_cols[i % 5].number_input(
                    asset, min_value=0.0, max_value=100.0,
                    value=float(_default_fi_w), step=1.0, key=f"wfi_{asset}",
                )
                weights_c[asset] = _w_fi
    elif _fi_selected and fi_prices.empty:
        st.warning("Fixed income price data unavailable. Check connectivity.")

    # ── Section B: S&P 500 Individual Stocks ────────────────────────────────
    _sh("B · S&P 500 Individual Stocks")
    _definition_block(
        "Stock Selection & Weight Editor",
        "Search and select any S&P 500 constituent. "
        "Edit weights directly in the table below - or use the quick-fill buttons. "
        "Weights are combined with Section A and normalised to 100% at run time. "
        "Stock prices are fetched from Yahoo Finance on demand (may take a few seconds for large baskets).",
    )

    with st.spinner("Fetching S&P 500 constituents…"):
        sp500_df = get_sp500_constituents()

    sp500_dict     = dict(zip(sp500_df["ticker"], sp500_df["name"]))
    sp500_sector   = dict(zip(sp500_df["ticker"], sp500_df["sector"]))
    all_sectors    = sorted(sp500_df["sector"].unique().tolist())

    bs1, bs2, bs3 = st.columns([1, 1, 2])
    sector_filter = bs1.multiselect(
        "Filter by sector", all_sectors,
        default=[], key="sector_filter",
        placeholder="All sectors",
    )

    filtered_sp500 = sp500_df if not sector_filter else sp500_df[sp500_df["sector"].isin(sector_filter)]
    stock_options  = [
        f"{row['ticker']} - {row['name'][:40]}"
        for _, row in filtered_sp500.iterrows()
    ]
    option_to_ticker = {
        f"{row['ticker']} - {row['name'][:40]}": row['ticker']
        for _, row in filtered_sp500.iterrows()
    }

    selected_stock_opts = bs2.multiselect(
        "Select S&P 500 stocks (type to search)",
        stock_options,
        default=[],
        key="st_stocks",
        placeholder="Search ticker or company…",
    )
    selected_stocks = [option_to_ticker[o] for o in selected_stock_opts]

    # Weight editor
    weights_b: dict[str, float] = {}
    if selected_stocks:
        n_stocks = len(selected_stocks)
        eq_w = round(100.0 / n_stocks, 2)

        bcol1, bcol2, bcol3 = st.columns([1, 1, 3])
        if bcol1.button("Equal Weight", key="eq_weight_btn", width="stretch"):
            for tk in selected_stocks:
                st.session_state[f"sw_{tk}"] = eq_w

        # CSV upload
        uploaded_csv = bcol3.file_uploader(
            "Upload weights CSV  (columns: ticker, weight)",
            type="csv", key="weight_csv",
            label_visibility="collapsed",
        )
        csv_weights: dict[str, float] = {}
        if uploaded_csv:
            try:
                cdf = pd.read_csv(uploaded_csv)
                cdf.columns = [c.lower().strip() for c in cdf.columns]
                if "ticker" in cdf.columns and "weight" in cdf.columns:
                    cdf["ticker"] = cdf["ticker"].str.strip().str.upper().str.replace(".", "-", regex=False)
                    csv_weights = dict(zip(cdf["ticker"], cdf["weight"].astype(float)))
                    st.success(f"Loaded weights for {len(csv_weights)} tickers from CSV.")
                else:
                    st.warning("CSV must have 'ticker' and 'weight' columns.")
            except Exception as e:
                st.warning(f"Could not parse CSV: {e}")

        # Build editable weights DataFrame
        stock_rows = []
        for tk in selected_stocks:
            default_w = (
                csv_weights.get(tk)
                or st.session_state.get(f"sw_{tk}")
                or eq_w
            )
            stock_rows.append({
                "Ticker":     tk,
                "Company":    sp500_dict.get(tk, tk),
                "Sector":     sp500_sector.get(tk, ""),
                "Weight (%)": float(default_w),
            })
        stock_df = pd.DataFrame(stock_rows)

        edited_df = st.data_editor(
            stock_df,
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            key="stock_weight_editor",
            column_config={
                "Ticker":     st.column_config.TextColumn("Ticker", disabled=True, width="small"),
                "Company":    st.column_config.TextColumn("Company", disabled=True),
                "Sector":     st.column_config.TextColumn("Sector", disabled=True, width="medium"),
                "Weight (%)": st.column_config.NumberColumn(
                    "Weight (%)", min_value=0.0, max_value=100.0, step=0.5, format="%.1f",
                ),
            },
        )

        # Extract weights from edited table
        for _, row in edited_df.iterrows():
            weights_b[row["Ticker"]] = float(row["Weight (%)"])

        total_b = sum(weights_b.values())
        st.markdown(
            f'<p style="font-size:0.70rem;color:#d1d5db;margin:0.3rem 0">'
            f'Section B total: <b>{total_b:.1f}%</b> across {n_stocks} stocks</p>',
            unsafe_allow_html=True,
        )

    # ── Combined weight summary & validation ─────────────────────────────────
    combined_weights_raw = {**weights_a, **weights_c, **weights_b}
    total_w = sum(combined_weights_raw.values())

    if total_w > 0:
        # Color-coded total weight indicator
        if total_w > 100.5:
            tw_color, tw_bg = "#e67e22", "#1e1e1e"
            tw_status = "Exceeds 100% - will normalize on run"
        elif abs(total_w - 100.0) < 0.5:
            tw_color, tw_bg = "#2e7d32", "#1c1c1c"
            tw_status = "✓ Fully allocated"
        else:
            tw_color, tw_bg = "#555960", "#1c1c1c"
            tw_status = f"{100 - total_w:.1f}% unallocated - will normalize on run"

        tw_col, norm_col, _ = st.columns([3, 1, 2])
        tw_col.markdown(
            f'<div style="padding:0.45rem 0.8rem;'
            f'background:{tw_bg};margin:0.5rem 0;font-size:0.70rem;border-radius:0">'
            f'<b style="color:{tw_color};font-family:JetBrains Mono,monospace">'
            f'Total weight: {total_w:.1f}%</b>'
            f'<span style="color:#8890a1;margin-left:10px;font-size:0.64rem">'
            f'{tw_status}</span></div>',
            unsafe_allow_html=True,
        )
        if norm_col.button("Normalize to 100%", key="normalize_btn", width="stretch"):
            combined_weights_raw = {a: w / total_w * 100 for a, w in combined_weights_raw.items()}
            total_w = 100.0

        # Build normalized weights for portfolio computation
        norm_weights = {a: w / total_w for a, w in combined_weights_raw.items() if w > 0}

        # Show compact allocation bar
        if norm_weights:
            top5 = sorted(norm_weights.items(), key=lambda x: -x[1])[:5]
            summary_txt = " &nbsp;|&nbsp; ".join(
                f"<b>{a}</b>: {w*100:.1f}%" for a, w in top5
            )
            remaining = len(norm_weights) - min(5, len(norm_weights))
            if remaining > 0:
                summary_txt += f" &nbsp;+&nbsp; {remaining} more"
            st.markdown(
                f'<div style="padding:0.4rem 0.8rem;'
                f'background:#1c1c1c;margin:0.4rem 0 0.6rem;font-size:0.70rem;color:#d1d5db">'
                f'<b>Normalised portfolio</b> ({len(norm_weights)} assets): {summary_txt}</div>',
                unsafe_allow_html=True,
            )

            # Portfolio allocation bar chart
            alloc_sorted = sorted(norm_weights.items(), key=lambda x: -x[1])
            alloc_fig = go.Figure(go.Bar(
                y=[a for a, _ in alloc_sorted],
                x=[v * 100 for _, v in alloc_sorted],
                orientation="h",
                marker_color="#CFB991",
                marker_line_color="#B8A070",
                marker_line_width=0.5,
                text=[f"{v*100:.1f}%" for _, v in alloc_sorted],
                textposition="outside",
                textfont=dict(size=9, family="JetBrains Mono, monospace"),
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            ))
            alloc_fig.update_layout(
                template="purdue",
                height=max(120, len(alloc_sorted) * 26),
                title=dict(text="Portfolio Allocation", font=dict(size=11), x=0),
                xaxis=dict(
                    title="Weight (%)", ticksuffix="%",
                    range=[0, max(v * 100 for _, v in alloc_sorted) * 1.35],
                ),
                yaxis=dict(tickfont=dict(size=9, family="JetBrains Mono, monospace")),
                margin=dict(l=130, r=80, t=36, b=30),
            )
            _chart(alloc_fig)
    else:
        norm_weights = {}

    # ── Event selector & stress test controls ───────────────────────────────
    _thread(
        "Portfolio built. Now select which historical stress scenarios to apply. Each event is defined "
        "by a pre-shock window, an acute stress period, and a post-event recovery window - matching the "
        "same event definitions used in the Geopolitical Triggers page."
    )
    _sh("Event Selection & Run")

    c3, c4 = st.columns(2)
    pre_days  = c3.slider("Pre-event window (days)",  10, 60, 30, key="st_pre")
    post_days = c4.slider("Post-event window (days)", 10, 90, 45, key="st_post")

    event_labels = [f"{e['label']} - {e['name']}" for e in GEOPOLITICAL_EVENTS]
    sel_events = st.multiselect(
        "Events to include", event_labels, default=event_labels, key="st_events",
    )
    chosen_events = [GEOPOLITICAL_EVENTS[event_labels.index(s)] for s in sel_events]

    if not norm_weights:
        st.info("Configure a portfolio in Section A and/or Section B above.")
        _page_footer()
        return

    if total_w > 100.5:
        st.warning(
            f"Total weight is **{total_w:.1f}%** - portfolio will be normalized to 100% on run. "
            "Or click **Normalize to 100%** above to update the weights explicitly."
        )

    if not st.button("Run Stress Test", type="primary", key="st_run"):
        _page_footer()
        return

    # ── Load stock prices if needed ──────────────────────────────────────────
    # eq_p may already include custom tickers added above
    _price_frames = [eq_p, cmd_p]
    if not fi_prices.empty:
        _price_frames.append(fi_prices)
    all_prices = pd.concat(_price_frames, axis=1)

    if weights_b:
        with st.spinner(f"Fetching prices for {len(weights_b)} individual stocks…"):
            stock_tickers = tuple(sorted(weights_b.keys()))
            stock_prices  = load_sp500_prices(stock_tickers, start, end)

        if not stock_prices.empty:
            # Rename ticker columns to match weights_b keys
            available_stocks = {t: t for t in stock_tickers if t in stock_prices.columns}
            all_prices = pd.concat([all_prices, stock_prices[list(available_stocks.values())]], axis=1)
        else:
            st.warning("Could not load individual stock prices. Check ticker symbols and connectivity.")

    # ── Compute stress test results ──────────────────────────────────────────
    # Remap stock names: use ticker as both key and column name
    with st.spinner(f"Running stress tests across {len(chosen_events)} events…"):
        results = [
            _event_stats(all_prices, {k: v for k, v in norm_weights.items()}, ev, pre_days, post_days)
            for ev in chosen_events
        ]
        results = [r for r in results if r]

    if not results:
        st.warning("No data available for the selected events and date range.")
        _page_footer()
        return

    # ── Summary table ────────────────────────────────────────────────────────
    _thread(
        "Results below decompose your portfolio's behaviour across three phases. Focus on the 'During' "
        "column first - that is the peak stress you would have experienced. Then check the 'Post' column "
        "to understand how quickly (or slowly) your specific allocation would have recovered."
    )
    _sh("Stress Test Results")

    summary = pd.DataFrame([{
        "Event":      r["event"],
        "Name":       r["name"],
        "Pre (%)":    round(r["pre_ret"],   2) if not np.isnan(r["pre_ret"])   else None,
        "During (%)": round(r["during_ret"], 2) if not np.isnan(r["during_ret"]) else None,
        "Post (%)":   round(r["post_ret"],  2) if not np.isnan(r["post_ret"])  else None,
        "Max DD (%)": round(r["max_dd"],    2) if not np.isnan(r["max_dd"])    else None,
        "Sharpe":     round(r["sharpe"],    2) if r["sharpe"] and not np.isnan(r["sharpe"]) else None,
    } for r in results])

    _TBL_CSS_ST = """
<style>
.ec-table{width:100%;border-collapse:collapse;font-family:'DM Sans',sans-serif;font-size:0.78rem}
.ec-table th{background:#1c1c1c;color:#CFB991;padding:7px 10px;text-align:left;
    border-bottom:1px solid rgba(207,185,145,0.3);font-weight:600;
    letter-spacing:0.06em;text-transform:uppercase;font-size:0.68rem}
.ec-table td{padding:5px 10px;border-bottom:1px solid #1e1e1e;color:#e8e9ed}
.ec-table tr:nth-child(even) td{background:#171717}
.ec-table tr:nth-child(odd) td{background:#111111}
.ec-table tr:hover td{background:#202020}
</style>"""
    st_pct_cols = ["Pre (%)", "During (%)", "Post (%)"]
    st_rows_html = ""
    for _, row in summary.iterrows():
        cells = (
            f"<td style='color:#b8b8b8'>{row.get('Event','')}</td>"
            f"<td style='color:#8890a1'>{row.get('Name','')}</td>"
        )
        for col in st_pct_cols:
            v = row.get(col)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                cells += "<td style='color:#8890a1'>-</td>"
            elif v > 0:
                cells += f"<td style='color:#4ade80;font-weight:600'>{v:+.2f}%</td>"
            else:
                cells += f"<td style='color:#f87171;font-weight:600'>{v:+.2f}%</td>"
        dd_v = row.get("Max DD (%)")
        if dd_v is None or (isinstance(dd_v, float) and pd.isna(dd_v)):
            cells += "<td style='color:#8890a1'>-</td>"
        elif dd_v < -10:
            cells += f"<td style='color:#f87171;font-weight:700'>{dd_v:.2f}%</td>"
        else:
            cells += f"<td style='color:#e8e9ed'>{dd_v:.2f}%</td>"
        sharpe_v = row.get("Sharpe")
        if sharpe_v is None or (isinstance(sharpe_v, float) and pd.isna(sharpe_v)):
            cells += "<td style='color:#8890a1'>-</td>"
        else:
            cells += f"<td style='color:#e8e9ed'>{sharpe_v:.2f}</td>"
        st_rows_html += f"<tr>{cells}</tr>"
    html_st = (
        _TBL_CSS_ST
        + "<table class='ec-table'>"
        + "<thead><tr>"
        + "<th>Event</th><th>Name</th><th>Pre (%)</th><th>During (%)</th>"
        + "<th>Post (%)</th><th>Max DD (%)</th><th>Sharpe</th>"
        + "</tr></thead><tbody>"
        + st_rows_html
        + "</tbody></table>"
    )
    st.markdown(html_st, unsafe_allow_html=True)

    valid_during = [r["during_ret"] for r in results if not np.isnan(r["during_ret"])]
    valid_dd     = [r["max_dd"]     for r in results if not np.isnan(r["max_dd"])]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Worst Event Return", f"{min(valid_during):+.1f}%" if valid_during else "N/A")
    k2.metric("Best Event Return",  f"{max(valid_during):+.1f}%" if valid_during else "N/A")
    k3.metric("Worst Max Drawdown", f"{min(valid_dd):.1f}%"      if valid_dd else "N/A")
    k4.metric("Avg During Return",  f"{np.mean(valid_during):+.1f}%" if valid_during else "N/A")

    # ── Event returns heatmap (pre / during / post) ───────────────────────────
    _sh("Event Returns - Pre / During / Post")
    _section_note(
        f"Return (%) across three windows: {pre_days}d before, event period, "
        f"{post_days}d after. Green = gain, red = loss."
    )

    hm_metrics = ["Pre (%)", "During (%)", "Post (%)"]
    hm_z_raw   = [
        [r["pre_ret"], r["during_ret"], r["post_ret"]]
        for r in results
    ]
    hm_z    = [[v if (v is not None and not np.isnan(float(v))) else None for v in row]
               for row in hm_z_raw]
    hm_text = [[f"{v:+.1f}%" if v is not None else "–" for v in row] for row in hm_z]
    hm_y    = [r["event"] for r in results]
    abs_max = max(
        (abs(v) for row in hm_z for v in row if v is not None),
        default=10,
    )

    fig_hm = go.Figure(go.Heatmap(
        z=hm_z, x=hm_metrics, y=hm_y,
        colorscale=[[0, "#c0392b"], [0.5, "#1e1e1e"], [1, "#2e7d32"]],
        zmid=0, zmin=-abs_max, zmax=abs_max,
        text=hm_text,
        texttemplate="%{text}",
        textfont=dict(size=10, family="JetBrains Mono, monospace", color="#e8e9ed"),
        hovertemplate="%{y} | %{x}: %{text}<extra></extra>",
        colorbar=dict(title="Return (%)", thickness=12, len=0.8, ticksuffix="%"),
    ))
    fig_hm.update_layout(
        template="purdue",
        height=max(300, len(results) * 30 + 80),
        paper_bgcolor="#000", plot_bgcolor="#080808",
        font=dict(color="#e8e9ed"),
        xaxis=dict(side="top", tickfont=dict(size=10, color="#8890a1"), rangeslider=dict(visible=False)),
        yaxis=dict(
            tickfont=dict(size=9, family="JetBrains Mono, monospace", color="#8890a1"),
            autorange="reversed",
        ),
        margin=dict(l=130, r=40, t=60, b=20),
    )
    _chart(fig_hm)

    # ── Max drawdown comparison ───────────────────────────────────────────────
    _sh("Max Drawdown by Event")

    fig_dd = go.Figure(go.Bar(
        y=[r["event"] for r in results],
        x=[r["max_dd"] for r in results],
        orientation="h",
        marker_color="#c0392b",
        opacity=0.75,
        text=[f"{r['max_dd']:.1f}%" for r in results],
        textposition="outside",
        textfont=dict(size=9, family="JetBrains Mono, monospace"),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig_dd.add_vline(x=-10, line=dict(color="#e67e22", width=1, dash="dot"),
                     annotation_text="-10% threshold", annotation_font_size=8)
    fig_dd.update_layout(
        template="purdue",
        height=max(260, len(results) * 28),
        xaxis=dict(title="Max Drawdown (%)", ticksuffix="%"),
        yaxis=dict(tickfont=dict(size=9, family="JetBrains Mono, monospace")),
        margin=dict(l=130, r=80, t=30, b=30),
    )
    _chart(fig_dd)

    # ── Geo Risk Beta - CAPM-style geopolitical amplification (Benjamin) ──────
    try:
        from src.analysis.conflict_model import score_all_conflicts, aggregate_portfolio_scores

        _geo_cr  = score_all_conflicts()
        _geo_agg = aggregate_portfolio_scores(_geo_cr)
        _geo_cis = float(_geo_agg.get("portfolio_cis", 50.0))
        _geo_beta = _geo_cis / 100.0   # 0–1 scale; 1.0 = maximum geopolitical stress

        _geo_col = "#c0392b" if _geo_cis >= 65 else "#e67e22" if _geo_cis >= 45 else "#CFB991"
        _M_geo = "font-family:'JetBrains Mono',monospace;"
        _F_geo = "font-family:'DM Sans',sans-serif;"

        # Build comparison rows: base vs. geo-adjusted drawdown
        _geo_rows_html = ""
        for r in results:
            _base_dd  = r["max_dd"]     if not np.isnan(r["max_dd"])     else None
            _base_ret = r["during_ret"] if not np.isnan(r["during_ret"]) else None
            if _base_dd is None or _base_ret is None:
                continue
            # Adjusted drawdown: amplified by geo_beta (more negative)
            _adj_dd  = _base_dd  * (1 + _geo_beta)
            # Adjusted during return: losses amplified; gains muted (sign preserved)
            _sign = np.sign(_base_ret) if _base_ret != 0 else -1
            _adj_ret = _base_ret * (1 + _geo_beta * abs(_sign))

            _dd_worse = _adj_dd < _base_dd
            _ret_worse = (_adj_ret < _base_ret) if _base_ret < 0 else (_adj_ret > _base_ret and _base_ret >= 0)

            _geo_rows_html += (
                f'<tr style="border-bottom:1px solid #1a1a1a">'
                f'<td style="{_M_geo}font-size:0.68rem;color:#b8b8b8;padding:4px 8px">{r["event"]}</td>'
                f'<td style="{_F_geo}font-size:0.68rem;color:#8890a1;padding:4px 8px">{r["name"][:36]}</td>'
                f'<td style="{_M_geo}font-size:0.68rem;color:#f87171;padding:4px 8px;text-align:right">'
                f'{_base_dd:.2f}%</td>'
                f'<td style="{_M_geo}font-size:0.68rem;color:{"#f87171" if _dd_worse else "#4ade80"};'
                f'font-weight:700;padding:4px 8px;text-align:right">{_adj_dd:.2f}%</td>'
                f'<td style="{_M_geo}font-size:0.68rem;color:#e8e9ed;padding:4px 8px;text-align:right">'
                f'{_base_ret:+.2f}%</td>'
                f'<td style="{_M_geo}font-size:0.68rem;color:{"#f87171" if _base_ret < 0 else "#4ade80"};'
                f'font-weight:700;padding:4px 8px;text-align:right">{_adj_ret:+.2f}%</td>'
                f'</tr>'
            )

        if _geo_rows_html:
            st.markdown("---")
            _sh("Geo Risk Beta - Geopolitically Adjusted Drawdowns")
            _section_note(
                f"Current CIS of {_geo_cis:.0f}/100 implies a geo-risk beta of {_geo_beta:.2f}. "
                "Scenario drawdowns amplified accordingly - consistent with Benjamin's CAPM-style "
                "geo risk factor approach: countries/portfolios with higher observed geopolitical risk "
                "experience 100% of the stress × (Geopolitical Risk Factor)."
            )

            st.markdown(
                f'<div style="background:#080808;border:1px solid #1e1e1e;'
                f'border-left:3px solid {_geo_col};padding:.5rem .9rem;margin-bottom:.6rem">'
                f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:.3rem">'
                f'<span style="{_M_geo}font-size:0.52rem;font-weight:700;letter-spacing:.16em;'
                f'text-transform:uppercase;color:{_geo_col}">GEO RISK FACTOR</span>'
                f'<span style="{_M_geo}font-size:0.72rem;font-weight:700;color:{_geo_col}">'
                f'β = {_geo_beta:.2f}</span>'
                f'<span style="{_F_geo}font-size:0.62rem;color:#8890a1">'
                f'CIS {_geo_cis:.0f}/100 · Adjusted DD = Base DD × (1 + {_geo_beta:.2f})</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            _geo_tbl_css = """<style>
.geo-table{width:100%;border-collapse:collapse;font-family:'DM Sans',sans-serif;font-size:0.78rem}
.geo-table th{background:#1c1c1c;color:#CFB991;padding:6px 8px;text-align:left;
    border-bottom:1px solid rgba(207,185,145,0.3);font-weight:600;
    letter-spacing:0.06em;text-transform:uppercase;font-size:0.65rem}
.geo-table td{border-bottom:1px solid #1e1e1e}
.geo-table tr:nth-child(even) td{background:#171717}
.geo-table tr:nth-child(odd) td{background:#111111}
.geo-table tr:hover td{background:#202020}
</style>"""
            st.markdown(
                _geo_tbl_css
                + "<table class='geo-table'><thead><tr>"
                + "<th>Event</th><th>Name</th>"
                + "<th style='text-align:right'>Base DD</th>"
                + "<th style='text-align:right'>Geo-Adj DD</th>"
                + "<th style='text-align:right'>Base During</th>"
                + "<th style='text-align:right'>Geo-Adj During</th>"
                + "</tr></thead><tbody>"
                + _geo_rows_html
                + "</tbody></table>",
                unsafe_allow_html=True,
            )
    except Exception:
        pass

    # ── Portfolio path (relative days from event start) ───────────────────────
    _sh("Portfolio Value Path - Days from Event Start (Base = 100)")
    _section_note(
        "Each line is indexed to 100 at event start (day 0), making events "
        "directly comparable regardless of calendar date. "
        f"Negative days = {pre_days}d pre-event; positive = {post_days}d post-event."
    )

    fig_path = go.Figure()
    for i, r in enumerate(results):
        port = r["port"]
        t0, t1 = r["t0"], r["t1"]
        window_start = t0 - pd.Timedelta(days=pre_days)
        window_end   = t1 + pd.Timedelta(days=post_days)
        sl = port.loc[window_start:window_end]
        if sl.empty:
            continue
        ref = sl.loc[t0:]
        if ref.empty:
            continue
        base     = ref.iloc[0]
        sl_norm  = sl / base * 100
        rel_days = [(d - t0).days for d in sl_norm.index]
        fig_path.add_trace(go.Scatter(
            x=rel_days, y=sl_norm.values,
            name=r["event"],
            line=dict(color=r["color"], width=1.5),
            hovertemplate="Day %{x}: %{y:.1f}<extra>" + r["event"] + "</extra>",
        ))

    fig_path.add_vline(
        x=0,
        line=dict(color="#555960", width=1.5, dash="dash"),
        annotation_text="Event Start",
        annotation_font_size=8,
        annotation_position="top right",
    )
    fig_path.add_hline(y=100, line=dict(color="#ABABAB", width=1, dash="dot"))
    fig_path.update_layout(
        template="purdue", height=420,
        yaxis=dict(title="Portfolio Value (Base=100 at Event Start)"),
        xaxis=dict(title="Days from Event Start", zeroline=False),
        margin=dict(l=50, r=20, t=30, b=40),
    )
    _chart(fig_path)

    # ── Individual stock contribution (if stocks selected) ────────────────────
    if weights_b and not stock_prices.empty:
        st.markdown("---")
        _sh("Individual Stock Weight Summary")
        _section_note(
            "Stocks selected in Section B. Weights are normalised across the full portfolio."
        )
        stock_summary = []
        for tk in weights_b:
            norm_w = norm_weights.get(tk, 0) * 100
            stock_summary.append({
                "Ticker":          tk,
                "Company":         sp500_dict.get(tk, tk),
                "Sector":          sp500_sector.get(tk, ""),
                "Portfolio Wt (%)": round(norm_w, 2),
            })
        stock_sum_df = pd.DataFrame(stock_summary).sort_values("Portfolio Wt (%)", ascending=False)
        _TBL_CSS_STK = """
<style>
.ec-table{width:100%;border-collapse:collapse;font-family:'DM Sans',sans-serif;font-size:0.78rem}
.ec-table th{background:#1c1c1c;color:#CFB991;padding:7px 10px;text-align:left;
    border-bottom:1px solid rgba(207,185,145,0.3);font-weight:600;
    letter-spacing:0.06em;text-transform:uppercase;font-size:0.68rem}
.ec-table td{padding:5px 10px;border-bottom:1px solid #1e1e1e;color:#e8e9ed}
.ec-table tr:nth-child(even) td{background:#171717}
.ec-table tr:nth-child(odd) td{background:#111111}
.ec-table tr:hover td{background:#202020}
</style>"""
        stk_rows_html = ""
        for _, row in stock_sum_df.iterrows():
            wt = row.get("Portfolio Wt (%)", 0)
            stk_rows_html += (
                f"<tr>"
                f"<td style='color:#CFB991;font-weight:600'>{row.get('Ticker','')}</td>"
                f"<td style='color:#b8b8b8'>{row.get('Company','')}</td>"
                f"<td style='color:#8890a1'>{row.get('Sector','')}</td>"
                f"<td style='color:#e8e9ed'>{wt:.2f}%</td>"
                f"</tr>"
            )
        html_stk = (
            _TBL_CSS_STK
            + "<table class='ec-table'>"
            + "<thead><tr>"
            + "<th>Ticker</th><th>Company</th><th>Sector</th><th>Portfolio Wt (%)</th>"
            + "</tr></thead><tbody>"
            + stk_rows_html
            + "</tbody></table>"
        )
        st.markdown(html_stk, unsafe_allow_html=True)

    worst_ev    = min(results, key=lambda r: r["during_ret"] if not np.isnan(r["during_ret"]) else 0)
    best_ev     = max(results, key=lambda r: r["during_ret"] if not np.isnan(r["during_ret"]) else 0)
    worst_dd_ev = min(results, key=lambda r: r["max_dd"]     if not np.isnan(r["max_dd"])     else 0)
    avg_ret     = np.mean(valid_during) if valid_during else 0

    _takeaway_block(
        f"Across {len(results)} historical events, this {len(norm_weights)}-asset portfolio "
        f"averaged {avg_ret:+.1f}% during-event return. "
        f"Worst: {worst_ev['event']} ({worst_ev['during_ret']:+.1f}%). "
        f"Best: {best_ev['event']} ({best_ev['during_ret']:+.1f}%). "
        f"Steepest drawdown: {worst_dd_ev['max_dd']:.1f}% during {worst_dd_ev['event']}. "
        "High commodity weights amplify energy supply-shock events; "
        "high equity weights amplify correlation-spike drawdowns. "
        "Gold typically reduces max drawdown during crisis regimes."
    )

    _page_conclusion(
        "Stress Test Complete",
        f"Portfolio of {len(norm_weights)} assets tested across {len(results)} historical events. "
        "Combine with the Spillover Network to understand contagion channels "
        "and with Correlation Analysis to see how regime shifts affect your allocation.",
    )

    # ── AI Stress Engineer ────────────────────────────────────────────────
    try:
        from src.agents.stress_engineer import run as _se_run
        from src.ui.agent_panel import render_agent_output_block
        from src.analysis.agent_state import is_enabled

        if is_enabled("stress_engineer"):
            _anthropic_key = _openai_key = ""
            try:
                _keys = st.secrets.get("keys", {})
                _anthropic_key = _keys.get("anthropic_api_key", "") or ""
                _openai_key    = _keys.get("openai_api_key",    "") or ""
            except Exception:
                pass
            _provider = "anthropic" if _anthropic_key else ("openai" if _openai_key else None)
            _api_key  = _anthropic_key or _openai_key

            _se_ctx: dict = {
                "n_scenarios": len(results),
                "scenarios": [
                    {
                        "name": r["event"],
                        "impact_pct": r.get("during_ret", 0),
                        "transmission": "equity-commodity correlation spike",
                    }
                    for r in results[:5]
                ],
                "worst_scenario": worst_ev["event"],
                "worst_impact":   worst_ev.get("during_ret", 0),
                "avg_impact":     avg_ret,
            }
            # Enrich with regime + risk score
            try:
                from src.analysis.correlations import (
                    average_cross_corr_series as _acs_se,
                    detect_correlation_regime as _dcr_se,
                )
                from src.analysis.risk_score import compute_risk_score as _crs_se
                _avg_se = _acs_se(eq_r, cmd_r, window=60)
                _rlab_se = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis"}
                _dcr_out = _dcr_se(_avg_se)
                _se_ctx["regime_name"] = (
                    _rlab_se.get(int(_dcr_out.iloc[-1]), "Normal")
                    if not _dcr_out.empty else "Unknown"
                )
                _rs_se = _crs_se(_avg_se, cmd_r, eq_r)
                _se_ctx["risk_score"] = float(_rs_se.get("score", 0))
            except Exception:
                pass

            with st.spinner("AI Stress Engineer assessing…"):
                _se_result = _se_run(_se_ctx, _provider, _api_key)

            if _se_result.get("narrative"):
                st.markdown("---")
                render_agent_output_block("stress_engineer", _se_result)
    except Exception:
        pass

    # CQO runs silently - output visible in About > AI Workforce
    try:
        from src.agents.quality_officer import run as _cqo_run
        from src.analysis.agent_state import is_enabled
        if is_enabled("quality_officer"):
            _anthropic_key = _openai_key = ""
            try:
                _keys = st.secrets.get("keys", {})
                _anthropic_key = _keys.get("anthropic_api_key", "") or ""
                _openai_key    = _keys.get("openai_api_key",    "") or ""
            except Exception:
                pass
            _provider = "anthropic" if _anthropic_key else ("openai" if _openai_key else None)
            _api_key  = _anthropic_key or _openai_key
            _n_scenarios = len(results) if "results" in dir() else 0
            _scenario_names = [r["event"] for r in results] if "results" in dir() and results else []
            _cqo_ctx = {
                "n_obs": len(eq_r.dropna(how="all")), "date_range": f"{start} to {end}",
                "n_events": _n_scenarios, "model": "Historical scenario simulation using realised returns",
                "assumption_count": 5,
                "notes": [
                    "Stress test uses historical return realisations - assumes future crises resemble past ones",
                    "Portfolio weights are user-defined and not dynamically rebalanced during stress period",
                    "Correlation structure during stress events differs from pre-crisis: diversification overstated",
                    "Each event treated independently - no correlation between concurrent shocks modelled",
                    f"Events tested: {', '.join(_scenario_names[:4]) if _scenario_names else 'none'} - user-selected, not systematic",
                    "No Monte Carlo or parametric tail extension - fat tail losses underrepresented",
                    "Liquidity risk and bid-ask spread expansion during crises not captured",
                ],
            }
            _cqo_run(_cqo_ctx, _provider, _api_key, page="Stress Test")
    except Exception:
        pass

    _page_footer()
