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
)
from src.data.config import GEOPOLITICAL_EVENTS, PALETTE, EQUITY_REGIONS, COMMODITY_GROUPS
from src.ui.shared import (
    _style_fig, _chart, _page_intro, _section_note,
    _definition_block, _takeaway_block, _page_conclusion, _page_footer,
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
    st.markdown(
        '<h1 style="font-family:\'DM Sans\',sans-serif;font-size:1.25rem;'
        'font-weight:700;margin-bottom:0.3rem">Portfolio Stress Tester</h1>',
        unsafe_allow_html=True,
    )
    _page_intro(
        "Build a custom multi-asset portfolio - mix global equity indices, commodity futures, "
        "and any S&P 500 individual stock - assign weights via an interactive editor, "
        "and stress-test against every historical geopolitical and macro event. "
        "Metrics: cumulative return (pre/during/post), max drawdown, Sharpe ratio. "
        "Not investment advice - for educational purposes only."
    )

    with st.spinner("Loading index and commodity price data…"):
        eq_p, cmd_p = load_all_prices(start, end)

    if eq_p.empty or cmd_p.empty:
        st.error("Market data unavailable.")
        return

    # ── Section A: Indices & Commodities ────────────────────────────────────
    st.markdown("---")
    st.subheader("Section A - Global Indices & Commodities")
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
        'text-transform:uppercase;color:#555960;margin:0.6rem 0 0.2rem">'
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
        st.markdown("**Weights (%)**")
        default_weights = presets.get(preset_name, {})
        weight_cols = st.columns(min(len(all_selected_a), 5))
        for i, asset in enumerate(all_selected_a):
            default_w = default_weights.get(asset, round(100 / len(all_selected_a), 1))
            w = weight_cols[i % 5].number_input(
                asset, min_value=0.0, max_value=100.0,
                value=float(default_w), step=1.0, key=f"w_{asset}",
            )
            weights_a[asset] = w

    # ── Section B: S&P 500 Individual Stocks ────────────────────────────────
    st.markdown("---")
    st.subheader("Section B - S&P 500 Individual Stocks")
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
        if bcol1.button("Equal Weight", key="eq_weight_btn", use_container_width=True):
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
            use_container_width=True,
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
            f'<p style="font-size:0.70rem;color:#333333;margin:0.3rem 0">'
            f'Section B total: <b>{total_b:.1f}%</b> across {n_stocks} stocks</p>',
            unsafe_allow_html=True,
        )

    # ── Combined weight summary & validation ─────────────────────────────────
    combined_weights_raw = {**weights_a, **weights_b}
    total_w = sum(combined_weights_raw.values())

    if total_w > 0:
        # Color-coded total weight indicator
        if total_w > 100.5:
            tw_color, tw_bg = "#e67e22", "#fff8f0"
            tw_status = "Exceeds 100% - will normalize on run"
        elif abs(total_w - 100.0) < 0.5:
            tw_color, tw_bg = "#2e7d32", "#f0f7f0"
            tw_status = "✓ Fully allocated"
        else:
            tw_color, tw_bg = "#555960", "#f5f5f5"
            tw_status = f"{100 - total_w:.1f}% unallocated - will normalize on run"

        tw_col, norm_col, _ = st.columns([3, 1, 2])
        tw_col.markdown(
            f'<div style="border-left:4px solid {tw_color};padding:0.45rem 0.8rem;'
            f'background:{tw_bg};margin:0.5rem 0;font-size:0.70rem;border-radius:0 4px 4px 0">'
            f'<b style="color:{tw_color};font-family:JetBrains Mono,monospace">'
            f'Total weight: {total_w:.1f}%</b>'
            f'<span style="color:#555960;margin-left:10px;font-size:0.64rem">'
            f'{tw_status}</span></div>',
            unsafe_allow_html=True,
        )
        if norm_col.button("Normalize to 100%", key="normalize_btn", use_container_width=True):
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
                f'<div style="border-left:3px solid #CFB991;padding:0.4rem 0.8rem;'
                f'background:#fafaf8;margin:0.4rem 0 0.6rem;font-size:0.70rem;color:#333333">'
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
    st.markdown("---")
    st.subheader("Event Selection & Run")

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
    all_prices = pd.concat([eq_p, cmd_p], axis=1)

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
    st.markdown("---")
    st.subheader("Stress Test Results")

    summary = pd.DataFrame([{
        "Event":      r["event"],
        "Name":       r["name"],
        "Pre (%)":    round(r["pre_ret"],   2) if not np.isnan(r["pre_ret"])   else None,
        "During (%)": round(r["during_ret"], 2) if not np.isnan(r["during_ret"]) else None,
        "Post (%)":   round(r["post_ret"],  2) if not np.isnan(r["post_ret"])  else None,
        "Max DD (%)": round(r["max_dd"],    2) if not np.isnan(r["max_dd"])    else None,
        "Sharpe":     round(r["sharpe"],    2) if r["sharpe"] and not np.isnan(r["sharpe"]) else None,
    } for r in results])

    def _col_pct(val):
        if pd.isna(val): return ""
        if val > 0:  return "color:#2e7d32;font-weight:600"
        if val < 0:  return "color:#c0392b;font-weight:600"
        return ""

    styled = (
        summary.style
        .applymap(_col_pct, subset=["Pre (%)", "During (%)", "Post (%)"])
        .applymap(lambda v: "color:#c0392b;font-weight:600" if isinstance(v, float) and v < -10 else "",
                  subset=["Max DD (%)"])
        .format({
            "Pre (%)": "{:+.2f}%", "During (%)": "{:+.2f}%",
            "Post (%)": "{:+.2f}%", "Max DD (%)": "{:.2f}%", "Sharpe": "{:.2f}",
        }, na_rep="-")
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=340)

    valid_during = [r["during_ret"] for r in results if not np.isnan(r["during_ret"])]
    valid_dd     = [r["max_dd"]     for r in results if not np.isnan(r["max_dd"])]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Worst Event Return", f"{min(valid_during):+.1f}%" if valid_during else "N/A")
    k2.metric("Best Event Return",  f"{max(valid_during):+.1f}%" if valid_during else "N/A")
    k3.metric("Worst Max Drawdown", f"{min(valid_dd):.1f}%"      if valid_dd else "N/A")
    k4.metric("Avg During Return",  f"{np.mean(valid_during):+.1f}%" if valid_during else "N/A")

    # ── Event returns heatmap (pre / during / post) ───────────────────────────
    st.markdown("---")
    st.subheader("Event Returns: Pre / During / Post")
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
        colorscale=[[0, "#c0392b"], [0.5, "#f9f9f7"], [1, "#2e7d32"]],
        zmid=0, zmin=-abs_max, zmax=abs_max,
        text=hm_text,
        texttemplate="%{text}",
        textfont=dict(size=10, family="JetBrains Mono, monospace"),
        hovertemplate="%{y} | %{x}: %{text}<extra></extra>",
        colorbar=dict(title="Return (%)", thickness=12, len=0.8, ticksuffix="%"),
    ))
    fig_hm.update_layout(
        template="purdue",
        height=max(300, len(results) * 30 + 80),
        xaxis=dict(side="top", tickfont=dict(size=10)),
        yaxis=dict(
            tickfont=dict(size=9, family="JetBrains Mono, monospace"),
            autorange="reversed",
        ),
        margin=dict(l=130, r=40, t=60, b=20),
    )
    _chart(fig_hm)

    # ── Max drawdown comparison ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Max Drawdown by Event")

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

    # ── Portfolio path (relative days from event start) ───────────────────────
    st.markdown("---")
    st.subheader("Portfolio Value Path: Days from Event Start (Base = 100)")
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
        st.subheader("Individual Stock Weight Summary")
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
        st.dataframe(stock_sum_df, use_container_width=True, hide_index=True)

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
    _page_footer()
