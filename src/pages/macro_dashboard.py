"""
Macro Dashboard
Money flows · High-freq indicators · Retail participation · Valuations
Index performance · Bond yields · Yield spreads · Earnings growth · GDP
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from datetime import date, timedelta

from src.ui.shared import (
    _style_fig, _chart, _insight_note, _definition_block,
    _page_footer,
)

_F = "font-family:'DM Sans',sans-serif;"


def _label(txt: str) -> None:
    import streamlit as _st
    _st.markdown(
        f'<p style="{_F}font-size:0.58rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.14em;color:#8E6F3E;margin:0 0 5px 0">{txt}</p>',
        unsafe_allow_html=True,
    )
from src.data.config import PALETTE

# ── FRED series used on this page ──────────────────────────────────────────
_YIELD_SERIES = {
    "3M":  "DGS3MO",
    "2Y":  "DGS2",
    "5Y":  "DGS5",
    "10Y": "DGS10",
    "30Y": "DGS30",
}
_SPREAD_SERIES = {
    "10Y–2Y (Yield Curve)":  "T10Y2Y",
    "10Y–3M":                "T10Y3M",
    "IG Credit Spread":      "BAMLC0A0CM",
    "HY Credit Spread":      "BAMLH0A0HYM2",
}
_MACRO_SERIES = {
    "Real GDP Growth (QoQ %)":      "A191RL1Q225SBEA",
    "Industrial Production":        "INDPRO",
    "Retail Sales ex-Auto":         "RSXFS",
    "ISM Manufacturing PMI":        "NAPM",
    "Nonfarm Payrolls (MoM k)":     "PAYEMS",
    "Unemployment Rate (%)":        "UNRATE",
    "CPI YoY (%)":                  "CPIAUCSL",
    "5Y Breakeven Inflation (%)":   "T5YIE",
}
_MONEY_SERIES = {
    "M2 Money Supply ($B)":         "M2SL",
    "Fed Total Assets ($B)":        "WALCL",
}
_SENTIMENT_SERIES = {
    "Consumer Sentiment (Michigan)": "UMCSENT",
}

# Valuation ETF proxies (yfinance)
_VAL_ETFS = {
    "S&P 500":    "SPY",
    "Nasdaq 100": "QQQ",
    "Russell 2K": "IWM",
    "Europe":     "VGK",
    "Japan":      "EWJ",
    "EM":         "EEM",
}

# Index ETFs for performance
_INDEX_ETFS = {
    "S&P 500":    "^GSPC",
    "Nasdaq 100": "^NDX",
    "Russell 2K": "^RUT",
    "Eurostoxx":  "^STOXX50E",
    "Nikkei 225": "^N225",
    "Hang Seng":  "^HSI",
    "VIX":        "^VIX",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _fred(fred_key: str, series_id: str, start: str, end: str) -> pd.Series:
    """Fetch a single FRED series; return empty Series on failure."""
    try:
        from fredapi import Fred
        f = Fred(api_key=fred_key)
        s = f.get_series(series_id, observation_start=start, observation_end=end)
        s.name = series_id
        return s.dropna()
    except Exception:
        return pd.Series(dtype=float, name=series_id)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_yields(fred_key: str, start: str, end: str) -> pd.DataFrame:
    frames = {}
    for label, sid in _YIELD_SERIES.items():
        s = _fred(fred_key, sid, start, end)
        if not s.empty:
            frames[label] = s
    return pd.DataFrame(frames) if frames else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_spreads(fred_key: str, start: str, end: str) -> pd.DataFrame:
    frames = {}
    for label, sid in _SPREAD_SERIES.items():
        s = _fred(fred_key, sid, start, end)
        if not s.empty:
            frames[label] = s
    return pd.DataFrame(frames) if frames else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_macro(fred_key: str, start: str, end: str) -> dict[str, pd.Series]:
    out = {}
    for label, sid in _MACRO_SERIES.items():
        s = _fred(fred_key, sid, start, end)
        if not s.empty:
            # For CPI: compute YoY pct change
            if "CPI" in label:
                s = s.pct_change(12) * 100
            # For payrolls: take MoM difference
            if "Payroll" in label:
                s = s.diff()
            out[label] = s.dropna()
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _load_money(fred_key: str, start: str, end: str) -> dict[str, pd.Series]:
    out = {}
    for label, sid in _MONEY_SERIES.items():
        s = _fred(fred_key, sid, start, end)
        if not s.empty:
            out[label] = s
    # Sentiment
    s = _fred(fred_key, "UMCSENT", start, end)
    if not s.empty:
        out["Consumer Sentiment"] = s
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _load_valuations() -> pd.DataFrame:
    rows = []
    for name, ticker in _VAL_ETFS.items():
        try:
            info = yf.Ticker(ticker).fast_info
            # fast_info doesn't have PE; use .info
            full = yf.Ticker(ticker).info
            trail_pe  = full.get("trailingPE")
            fwd_pe    = full.get("forwardPE")
            pb        = full.get("priceToBook")
            eps_fwd   = full.get("forwardEps")
            eps_trail = full.get("trailingEps")
            if trail_pe and eps_fwd and eps_trail and eps_trail != 0:
                eps_growth = round((eps_fwd / eps_trail - 1) * 100, 1)
            else:
                eps_growth = None
            rows.append({
                "Market":          name,
                "Trailing P/E":    round(trail_pe, 1) if trail_pe else None,
                "Forward P/E":     round(fwd_pe, 1)   if fwd_pe  else None,
                "P/B":             round(pb, 2)        if pb      else None,
                "Earnings Yield %": round(100/fwd_pe, 2) if fwd_pe else None,
                "Fwd EPS Growth %": eps_growth,
            })
        except Exception:
            rows.append({"Market": name})
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800, show_spinner=False)
def _load_index_perf(start: str) -> pd.DataFrame:
    """Returns cumulative return from `start` for each index."""
    tickers = list(_INDEX_ETFS.values())
    raw = yf.download(tickers, start=start, progress=False, auto_adjust=True)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    rename = {v: k for k, v in _INDEX_ETFS.items()}
    raw.columns = [rename.get(c, c) for c in raw.columns]
    return raw


def _kpi(col, label: str, value: str, delta: str = "", delta_up: bool | None = None):
    arrow = ""
    if delta:
        if delta_up is True:
            arrow = f'<span style="color:#2e7d32">▲ {delta}</span>'
        elif delta_up is False:
            arrow = f'<span style="color:#c0392b">▼ {delta}</span>'
        else:
            arrow = f'<span style="color:#888">{delta}</span>'
    col.markdown(
        f'<div style="background:#fafaf8;border:1px solid #E8E5E0;border-radius:6px;'
        f'padding:10px 14px;margin-bottom:8px">'
        f'<div style="{_F}font-size:0.55rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.12em;color:#8E6F3E;margin-bottom:2px">{label}</div>'
        f'<div style="{_F}font-size:1.05rem;font-weight:700;color:#000;line-height:1.2">{value}</div>'
        f'<div style="{_F}font-size:0.65rem;margin-top:2px">{arrow}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Main page ───────────────────────────────────────────────────────────────

def page_macro_dashboard(start: str, end: str, fred_key: str = "") -> None:
    st.markdown(
        '<h1 style="font-family:\'DM Sans\',sans-serif;font-size:1.25rem;'
        'font-weight:700;margin-bottom:0.1rem">Macro Intelligence Dashboard</h1>'
        '<p style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;color:#555;'
        'margin:0 0 0.7rem">Money flows · Valuations · Bond yields · Yield spreads · '
        'Earnings growth · GDP · High-freq indicators · Retail participation</p>',
        unsafe_allow_html=True,
    )

    no_fred = not fred_key
    if no_fred:
        st.info("Add a FRED API key in Settings to unlock bond, macro, and money-flow data. "
                "Valuations and index performance load without it.")

    # ── SECTION 1: Index Performance ────────────────────────────────────────
    st.markdown('<div style="margin:0.4rem 0 0.5rem;border-top:1px solid #E8E5E0"></div>',
                unsafe_allow_html=True)
    _label("Index Performance")

    # Rolling windows for performance
    perf_windows = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "YTD": None}
    today = date.today()
    perf_start = (today - timedelta(days=400)).isoformat()

    with st.spinner("Loading index data…"):
        idx_prices = _load_index_perf(perf_start)

    if not idx_prices.empty:
        # KPI strip — latest returns for each window
        win_cols = st.columns(len(perf_windows))
        chosen_win = None
        for i, (wlabel, wdays) in enumerate(perf_windows.items()):
            win_cols[i].markdown(
                f'<div style="{_F}font-size:0.58rem;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:0.12em;color:#8E6F3E;text-align:center">{wlabel}</div>',
                unsafe_allow_html=True,
            )

        # Performance heatmap table
        perf_rows = []
        for name in idx_prices.columns:
            if name == "VIX":
                continue
            s = idx_prices[name].dropna()
            if s.empty:
                continue
            row = {"Index": name}
            for wlabel, wdays in perf_windows.items():
                if wdays is None:
                    ytd_start = pd.Timestamp(today.year, 1, 1)
                    s_ytd = s[s.index >= ytd_start]
                    if len(s_ytd) >= 2:
                        row[wlabel] = round((s_ytd.iloc[-1] / s_ytd.iloc[0] - 1) * 100, 2)
                    else:
                        row[wlabel] = None
                else:
                    if len(s) > wdays:
                        row[wlabel] = round((s.iloc[-1] / s.iloc[-wdays - 1] - 1) * 100, 2)
                    else:
                        row[wlabel] = None
            row["Latest"] = round(float(s.iloc[-1]), 2)
            perf_rows.append(row)

        perf_df = pd.DataFrame(perf_rows).set_index("Index")
        ret_cols = [c for c in perf_df.columns if c != "Latest"]

        def _color_ret(val):
            if pd.isna(val):
                return ""
            if val > 2:    return "background-color:#c8e6c9;color:#1b5e20"
            if val > 0:    return "background-color:#e8f5e9;color:#2e7d32"
            if val > -2:   return "background-color:#ffebee;color:#b71c1c"
            return              "background-color:#ffcdd2;color:#b71c1c"

        styled_perf = perf_df.style.applymap(_color_ret, subset=ret_cols).format(
            {c: "{:+.2f}%" for c in ret_cols},
            na_rep="—",
        )
        idx_l, idx_r = st.columns([1.6, 1])
        with idx_l:
            st.dataframe(styled_perf, use_container_width=True, height=280)

        with idx_r:
            # VIX trend
            if "VIX" in idx_prices.columns:
                vix = idx_prices["VIX"].dropna().last("90D")
                fig_vix = go.Figure()
                fig_vix.add_trace(go.Scatter(
                    x=vix.index, y=vix.values, name="VIX",
                    line=dict(color="#c0392b", width=1.8),
                    fill="tozeroy", fillcolor="rgba(192,57,43,0.08)",
                ))
                fig_vix.add_hline(y=20, line=dict(color="#CFB991", width=1, dash="dot"),
                                  annotation_text="20 (Caution)", annotation_font_size=8)
                fig_vix.add_hline(y=30, line=dict(color="#c0392b", width=1, dash="dot"),
                                  annotation_text="30 (Fear)", annotation_font_size=8)
                fig_vix.update_layout(
                    template="purdue", height=260, margin=dict(l=40, r=10, t=30, b=30),
                    title=dict(text="VIX Fear Index (90D)", font=dict(size=10)),
                    yaxis=dict(title="VIX"), showlegend=False,
                )
                _chart(fig_vix)

        _insight_note(
            "Performance heatmap shows returns across rolling windows. "
            "Green = positive return; red = negative. "
            "VIX above 20 signals elevated market stress; above 30 signals fear."
        )

    # ── SECTION 2: Bond Yields & Yield Curve ────────────────────────────────
    st.markdown('<div style="margin:0.5rem 0;border-top:1px solid #E8E5E0"></div>',
                unsafe_allow_html=True)
    _label("Bond Yields & Yield Curve")

    if no_fred:
        st.caption("Requires FRED API key.")
    else:
        with st.spinner("Loading yield data…"):
            yields_df = _load_yields(fred_key, start, end)

        if not yields_df.empty:
            y_l, y_r = st.columns([1.8, 1])

            with y_l:
                fig_yld = go.Figure()
                colors = ["#CFB991", "#8E6F3E", "#c0392b", "#2e7d32", "#3498db"]
                for i, col in enumerate(yields_df.columns):
                    fig_yld.add_trace(go.Scatter(
                        x=yields_df.index, y=yields_df[col],
                        name=col, line=dict(color=colors[i % len(colors)], width=1.5),
                    ))
                fig_yld.update_layout(
                    template="purdue", height=300,
                    title=dict(text="US Treasury Yields (%)", font=dict(size=10)),
                    yaxis=dict(title="Yield (%)", ticksuffix="%"),
                    margin=dict(l=40, r=10, t=35, b=30),
                    legend=dict(orientation="h", y=1.12, font=dict(size=9)),
                )
                _chart(fig_yld)

            with y_r:
                # Spot yield curve (latest snapshot)
                latest_yields = yields_df.iloc[-1].dropna()
                maturities = {"3M": 0.25, "2Y": 2, "5Y": 5, "10Y": 10, "30Y": 30}
                x_vals, y_vals = [], []
                for label, yrs in maturities.items():
                    if label in latest_yields.index:
                        x_vals.append(yrs)
                        y_vals.append(latest_yields[label])

                fig_curve = go.Figure()
                fig_curve.add_trace(go.Scatter(
                    x=x_vals, y=y_vals,
                    mode="lines+markers",
                    line=dict(color="#CFB991", width=2),
                    marker=dict(size=7, color="#8E6F3E"),
                    name="Yield Curve",
                ))
                fig_curve.update_layout(
                    template="purdue", height=300,
                    title=dict(text=f"Spot Yield Curve ({yields_df.index[-1].strftime('%d %b %Y')})",
                               font=dict(size=10)),
                    xaxis=dict(title="Maturity (Years)", tickvals=x_vals,
                               ticktext=[f"{v}Y" if v >= 1 else "3M" for v in x_vals]),
                    yaxis=dict(title="Yield (%)", ticksuffix="%"),
                    margin=dict(l=40, r=10, t=35, b=40),
                )
                _chart(fig_curve)

            _insight_note(
                "An inverted yield curve (short rates above long rates) has preceded every US recession "
                "since 1955. Watch the 10Y–2Y spread: negative = inversion = recession risk signal."
            )

    # ── SECTION 3: Yield Spreads ─────────────────────────────────────────────
    st.markdown('<div style="margin:0.5rem 0;border-top:1px solid #E8E5E0"></div>',
                unsafe_allow_html=True)
    _label("Yield Spreads & Credit Risk")

    if no_fred:
        st.caption("Requires FRED API key.")
    else:
        with st.spinner("Loading spread data…"):
            spreads_df = _load_spreads(fred_key, start, end)

        if not spreads_df.empty:
            sp_l, sp_r = st.columns(2)

            with sp_l:
                # Yield curve inversion
                curve_cols = [c for c in ["10Y–2Y (Yield Curve)", "10Y–3M"] if c in spreads_df]
                if curve_cols:
                    fig_inv = go.Figure()
                    for col in curve_cols:
                        s = spreads_df[col].dropna()
                        fig_inv.add_trace(go.Scatter(
                            x=s.index, y=s.values, name=col,
                            line=dict(width=1.5),
                        ))
                    fig_inv.add_hline(y=0, line=dict(color="#c0392b", width=1.2, dash="dash"),
                                      annotation_text="Inversion", annotation_font_size=8,
                                      annotation_font_color="#c0392b")
                    fig_inv.update_layout(
                        template="purdue", height=280,
                        title=dict(text="Yield Curve Spreads (%)", font=dict(size=10)),
                        yaxis=dict(title="Spread (%)", ticksuffix="%"),
                        margin=dict(l=40, r=10, t=35, b=30),
                        legend=dict(orientation="h", y=1.12, font=dict(size=9)),
                    )
                    _chart(fig_inv)

            with sp_r:
                # Credit spreads
                cred_cols = [c for c in ["IG Credit Spread", "HY Credit Spread"] if c in spreads_df]
                if cred_cols:
                    fig_cred = go.Figure()
                    cred_colors = {"IG Credit Spread": "#2e7d32", "HY Credit Spread": "#c0392b"}
                    for col in cred_cols:
                        s = spreads_df[col].dropna()
                        fig_cred.add_trace(go.Scatter(
                            x=s.index, y=s.values, name=col,
                            line=dict(color=cred_colors.get(col, "#CFB991"), width=1.5),
                        ))
                    fig_cred.update_layout(
                        template="purdue", height=280,
                        title=dict(text="Credit Spreads (OAS, %)", font=dict(size=10)),
                        yaxis=dict(title="Spread (%)", ticksuffix="%"),
                        margin=dict(l=40, r=10, t=35, b=30),
                        legend=dict(orientation="h", y=1.12, font=dict(size=9)),
                    )
                    _chart(fig_cred)

            _insight_note(
                "Credit spreads widen when investors demand more compensation for default risk. "
                "HY spreads above 600bps historically signal recession; IG spreads above 200bps "
                "indicate broad credit stress. Rising spreads = tightening financial conditions."
            )

    # ── SECTION 4: Valuations & Expected Earnings Growth ────────────────────
    st.markdown('<div style="margin:0.5rem 0;border-top:1px solid #E8E5E0"></div>',
                unsafe_allow_html=True)
    _label("Valuations & Expected Earnings Growth")

    with st.spinner("Loading valuation data…"):
        val_df = _load_valuations()

    if not val_df.empty and "Trailing P/E" in val_df.columns:
        v_l, v_r = st.columns([1.2, 1])

        with v_l:
            def _val_color(val):
                if pd.isna(val):
                    return ""
                return ""
            def _pe_color(val):
                if pd.isna(val): return ""
                if val > 30:   return "background-color:#ffcdd2;color:#b71c1c"
                if val > 22:   return "background-color:#fff9c4;color:#f57f17"
                if val > 15:   return "background-color:#e8f5e9;color:#2e7d32"
                return "background-color:#c8e6c9;color:#1b5e20"
            def _growth_color(val):
                if pd.isna(val): return ""
                if val > 15:   return "color:#2e7d32;font-weight:700"
                if val > 5:    return "color:#558b2f"
                if val > 0:    return "color:#888"
                return              "color:#c0392b;font-weight:700"

            display_df = val_df.set_index("Market") if "Market" in val_df.columns else val_df
            styled_val = (
                display_df.style
                .applymap(_pe_color, subset=[c for c in ["Trailing P/E", "Forward P/E"] if c in display_df.columns])
                .applymap(_growth_color, subset=[c for c in ["Fwd EPS Growth %"] if c in display_df.columns])
                .format("{:.1f}", subset=[c for c in ["Trailing P/E", "Forward P/E", "Earnings Yield %"] if c in display_df.columns], na_rep="—")
                .format("{:.2f}", subset=[c for c in ["P/B"] if c in display_df.columns], na_rep="—")
                .format("{:+.1f}%", subset=[c for c in ["Fwd EPS Growth %"] if c in display_df.columns], na_rep="—")
            )
            st.dataframe(styled_val, use_container_width=True, height=280)

        with v_r:
            # Forward P/E bar chart
            plot_df = val_df.dropna(subset=["Forward P/E"]) if "Forward P/E" in val_df.columns else pd.DataFrame()
            if not plot_df.empty:
                bar_colors = [
                    "#c0392b" if v > 25 else "#CFB991" if v > 18 else "#2e7d32"
                    for v in plot_df["Forward P/E"]
                ]
                fig_pe = go.Figure(go.Bar(
                    x=plot_df["Market"], y=plot_df["Forward P/E"],
                    marker_color=bar_colors,
                    text=[f"{v:.1f}x" for v in plot_df["Forward P/E"]],
                    textposition="outside",
                ))
                fig_pe.add_hline(y=20, line=dict(color="#CFB991", dash="dot", width=1),
                                 annotation_text="Expensive (20x)", annotation_font_size=8)
                fig_pe.add_hline(y=15, line=dict(color="#2e7d32", dash="dot", width=1),
                                 annotation_text="Fair Value (15x)", annotation_font_size=8)
                fig_pe.update_layout(
                    template="purdue", height=280,
                    title=dict(text="Forward P/E by Market", font=dict(size=10)),
                    yaxis=dict(title="Forward P/E (x)"),
                    margin=dict(l=40, r=10, t=35, b=40),
                )
                _chart(fig_pe)

        _insight_note(
            "Forward P/E above 20x signals expensive valuations; below 15x is historically cheap. "
            "Earnings yield (= 1 / Forward P/E) can be compared to the 10Y bond yield to assess the "
            "equity risk premium. Positive EPS growth with low P/E = best value setup."
        )

    # ── SECTION 5: GDP, Macro & High-Freq Indicators ─────────────────────────
    st.markdown('<div style="margin:0.5rem 0;border-top:1px solid #E8E5E0"></div>',
                unsafe_allow_html=True)
    _label("GDP, Growth & High-Frequency Indicators")

    if no_fred:
        st.caption("Requires FRED API key.")
    else:
        with st.spinner("Loading macro indicators…"):
            macro_data = _load_macro(fred_key, start, end)

        if macro_data:
            # Latest readings KPI strip
            kpi_keys = [
                "Real GDP Growth (QoQ %)",
                "ISM Manufacturing PMI",
                "Unemployment Rate (%)",
                "CPI YoY (%)",
            ]
            k_cols = st.columns(len(kpi_keys))
            for i, key in enumerate(kpi_keys):
                if key in macro_data and not macro_data[key].empty:
                    s = macro_data[key]
                    latest = s.iloc[-1]
                    prev   = s.iloc[-2] if len(s) > 1 else latest
                    delta  = latest - prev
                    is_up  = delta > 0
                    # For unemployment and CPI, up is bad
                    good_up = key not in ("Unemployment Rate (%)", "CPI YoY (%)")
                    _kpi(
                        k_cols[i], key,
                        f"{latest:.2f}{'%' if '%' in key else ''}",
                        f"{abs(delta):.2f}",
                        delta_up=is_up if good_up else not is_up,
                    )

            # Charts: GDP + Industrial Production | PMI + Retail Sales
            m_l, m_r = st.columns(2)

            with m_l:
                gdp_key = "Real GDP Growth (QoQ %)"
                ip_key  = "Industrial Production"
                fig_gdp = go.Figure()
                if gdp_key in macro_data:
                    s = macro_data[gdp_key].last("10Y")
                    fig_gdp.add_trace(go.Bar(
                        x=s.index, y=s.values, name="Real GDP Growth",
                        marker_color=["#2e7d32" if v >= 0 else "#c0392b" for v in s.values],
                    ))
                fig_gdp.add_hline(y=0, line=dict(color="#ABABAB", dash="dot", width=1))
                fig_gdp.update_layout(
                    template="purdue", height=280,
                    title=dict(text="Real GDP Growth (QoQ %)", font=dict(size=10)),
                    yaxis=dict(title="Growth (%)", ticksuffix="%"),
                    margin=dict(l=40, r=10, t=35, b=30),
                )
                _chart(fig_gdp)

            with m_r:
                pmi_key = "ISM Manufacturing PMI"
                fig_pmi = go.Figure()
                if pmi_key in macro_data:
                    s = macro_data[pmi_key].last("5Y")
                    fig_pmi.add_trace(go.Scatter(
                        x=s.index, y=s.values, name="ISM PMI",
                        line=dict(color="#CFB991", width=2),
                        fill="tozeroy", fillcolor="rgba(207,185,145,0.1)",
                    ))
                    fig_pmi.add_hline(y=50, line=dict(color="#c0392b", dash="dash", width=1.2),
                                      annotation_text="50 = Expansion/Contraction", annotation_font_size=8)
                fig_pmi.update_layout(
                    template="purdue", height=280,
                    title=dict(text="ISM Manufacturing PMI", font=dict(size=10)),
                    yaxis=dict(title="PMI"),
                    margin=dict(l=40, r=10, t=35, b=30),
                )
                _chart(fig_pmi)

            # Retail Sales + Payrolls
            m2_l, m2_r = st.columns(2)

            with m2_l:
                rs_key = "Retail Sales ex-Auto"
                if rs_key in macro_data:
                    s = macro_data[rs_key].last("5Y")
                    s_pct = s.pct_change(12) * 100
                    fig_rs = go.Figure()
                    fig_rs.add_trace(go.Scatter(
                        x=s_pct.index, y=s_pct.values, name="YoY %",
                        line=dict(color="#3498db", width=1.8),
                        fill="tozeroy",
                        fillcolor="rgba(52,152,219,0.08)",
                    ))
                    fig_rs.add_hline(y=0, line=dict(color="#ABABAB", dash="dot", width=1))
                    fig_rs.update_layout(
                        template="purdue", height=260,
                        title=dict(text="Retail Sales ex-Auto (YoY %)", font=dict(size=10)),
                        yaxis=dict(title="YoY (%)", ticksuffix="%"),
                        margin=dict(l=40, r=10, t=35, b=30),
                    )
                    _chart(fig_rs)

            with m2_r:
                pay_key = "Nonfarm Payrolls (MoM k)"
                if pay_key in macro_data:
                    s = macro_data[pay_key].last("5Y") / 1000  # convert to thousands
                    fig_pay = go.Figure(go.Bar(
                        x=s.index, y=s.values,
                        marker_color=["#2e7d32" if v >= 0 else "#c0392b" for v in s.values],
                        name="MoM Change",
                    ))
                    fig_pay.update_layout(
                        template="purdue", height=260,
                        title=dict(text="Nonfarm Payrolls (MoM, thousands)", font=dict(size=10)),
                        yaxis=dict(title="Change (k)"),
                        margin=dict(l=40, r=10, t=35, b=30),
                    )
                    _chart(fig_pay)

            _insight_note(
                "PMI above 50 = manufacturing expanding; below 50 = contraction. "
                "Retail sales and payrolls are key revenue indicators for consumer-facing sectors. "
                "Declining payrolls alongside falling retail sales historically precede equity drawdowns of 15-30%."
            )

    # ── SECTION 6: Money Flows & Liquidity ───────────────────────────────────
    st.markdown('<div style="margin:0.5rem 0;border-top:1px solid #E8E5E0"></div>',
                unsafe_allow_html=True)
    _label("Money Flows & Liquidity")

    if no_fred:
        st.caption("Requires FRED API key.")
    else:
        with st.spinner("Loading money flow data…"):
            money_data = _load_money(fred_key, start, end)

        if money_data:
            mf_l, mf_r, mf_s = st.columns(3)

            with mf_l:
                if "M2 Money Supply ($B)" in money_data:
                    s = money_data["M2 Money Supply ($B)"].last("10Y")
                    s_yoy = s.pct_change(12) * 100
                    fig_m2 = go.Figure()
                    fig_m2.add_trace(go.Scatter(
                        x=s_yoy.dropna().index, y=s_yoy.dropna().values,
                        name="M2 YoY %", line=dict(color="#CFB991", width=2),
                        fill="tozeroy", fillcolor="rgba(207,185,145,0.1)",
                    ))
                    fig_m2.add_hline(y=0, line=dict(color="#c0392b", dash="dash", width=1))
                    fig_m2.update_layout(
                        template="purdue", height=260,
                        title=dict(text="M2 Money Supply Growth (YoY %)", font=dict(size=10)),
                        yaxis=dict(title="YoY (%)", ticksuffix="%"),
                        margin=dict(l=40, r=10, t=35, b=30),
                    )
                    _chart(fig_m2)

            with mf_r:
                if "Fed Total Assets ($B)" in money_data:
                    s = money_data["Fed Total Assets ($B)"].last("10Y")
                    fig_fed = go.Figure()
                    fig_fed.add_trace(go.Scatter(
                        x=s.index, y=s.values / 1000,
                        name="Fed Assets ($T)", line=dict(color="#8E6F3E", width=2),
                        fill="tozeroy", fillcolor="rgba(142,111,62,0.1)",
                    ))
                    fig_fed.update_layout(
                        template="purdue", height=260,
                        title=dict(text="Fed Balance Sheet ($T)", font=dict(size=10)),
                        yaxis=dict(title="Assets ($T)", tickprefix="$"),
                        margin=dict(l=40, r=10, t=35, b=30),
                    )
                    _chart(fig_fed)

            with mf_s:
                if "Consumer Sentiment" in money_data:
                    s = money_data["Consumer Sentiment"].last("5Y")
                    fig_sent = go.Figure()
                    fig_sent.add_trace(go.Scatter(
                        x=s.index, y=s.values, name="Consumer Sentiment",
                        line=dict(color="#3498db", width=2),
                    ))
                    # Historical average
                    avg = float(s.mean())
                    fig_sent.add_hline(y=avg, line=dict(color="#ABABAB", dash="dot", width=1),
                                       annotation_text=f"Avg {avg:.0f}", annotation_font_size=8)
                    fig_sent.update_layout(
                        template="purdue", height=260,
                        title=dict(text="Consumer Sentiment (Michigan)", font=dict(size=10)),
                        yaxis=dict(title="Index"),
                        margin=dict(l=40, r=10, t=35, b=30),
                    )
                    _chart(fig_sent)

            _insight_note(
                "M2 contraction (negative YoY) is historically associated with asset price stress — "
                "the 2022 drawdown coincided with the first M2 decline since the 1930s. "
                "Fed balance sheet expansion (QE) injects liquidity and typically lifts risk assets; "
                "QT (reduction) tightens liquidity. Consumer sentiment below its long-run average "
                "signals reduced retail participation and lower consumer spending momentum."
            )

    _page_footer()
