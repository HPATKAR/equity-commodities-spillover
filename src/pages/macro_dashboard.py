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
    _page_footer, _section_header, _regime_banner, _narrative_box, _page_intro,
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


@st.cache_data(ttl=3600, show_spinner=False)
def _load_pc_proxies(start: str, end: str) -> pd.DataFrame:
    """BKLN, ARCC, OBDC, FSK, SPY daily close prices - private credit proxy basket."""
    raw = yf.download(
        ["BKLN", "ARCC", "OBDC", "FSK", "SPY"],
        start=start, end=end, auto_adjust=True, progress=False,
    )
    px = (raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw).ffill()
    return px


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


def _narrate_growth(macro_data: dict) -> str:
    parts = []
    gdp = macro_data.get("Real GDP Growth (QoQ %)")
    pmi = macro_data.get("ISM Manufacturing PMI")
    pay = macro_data.get("Nonfarm Payrolls (MoM k)")
    cpi = macro_data.get("CPI YoY (%)")
    un  = macro_data.get("Unemployment Rate (%)")

    if gdp is not None and not gdp.empty:
        v = gdp.iloc[-1]
        trend = "expanding" if v > 0 else "contracting"
        strength = "robustly" if v > 2.5 else ("modestly" if v > 0.5 else "weakly")
        parts.append(
            f"The US economy is {trend} {strength}, with real GDP growth at {v:.1f}% QoQ. "
            + ("Two consecutive quarters of contraction would confirm a technical recession." if v < 0 else
               "This pace is consistent with late-cycle deceleration." if v < 1.5 else
               "Expansion at this rate supports corporate earnings and commodity demand.")
        )
    if pmi is not None and not pmi.empty:
        v = pmi.iloc[-1]
        prev = pmi.iloc[-2] if len(pmi) > 1 else v
        direction = "improving" if v > prev else "deteriorating"
        state = "expansion" if v > 50 else "contraction"
        parts.append(
            f"ISM Manufacturing PMI stands at {v:.1f} - {state} territory, and {direction}. "
            + ("PMI above 50 is constructive for industrial metals (copper, aluminium) and energy demand." if v > 50 else
               "Sub-50 PMI historically leads to negative earnings revisions in industrials and weighs on base metals within 1–2 quarters.")
        )
    if pay is not None and not pay.empty:
        v = pay.iloc[-1] / 1000
        cpi_v = cpi.iloc[-1] if cpi is not None and not cpi.empty else None
        un_v  = un.iloc[-1]  if un  is not None and not un.empty  else None
        labour = "resilient" if v > 0.15 else ("mixed" if v > 0 else "weakening")
        parts.append(
            f"The labour market is {labour}: nonfarm payrolls added {v*1000:.0f}k jobs last month"
            + (f", against an unemployment rate of {un_v:.1f}%" if un_v else "")
            + (f" and CPI running at {cpi_v:.1f}% YoY" if cpi_v else "")
            + (". Strong employment underpins consumer spending and keeps inflationary pressure alive, "
               "limiting the scope for near-term rate cuts which would otherwise benefit high-multiple equities." if v > 0.15 else
               ". Softening employment reduces consumer demand - negative for retail and consumer discretionary equities, "
               "and moderately bearish for agricultural and energy commodities.")
        )
    return "\n\n".join(parts) if parts else "GDP and labour data unavailable."


def _narrate_liquidity(money_data: dict, yields_df: "pd.DataFrame") -> str:
    parts = []
    m2 = money_data.get("M2 Money Supply ($B)")
    fed = money_data.get("Fed Total Assets ($B)")
    sent = money_data.get("Consumer Sentiment")

    if m2 is not None and not m2.empty and len(m2) > 12:
        yoy = m2.pct_change(12).iloc[-1] * 100
        regime = "contracting" if yoy < 0 else ("growing slowly" if yoy < 4 else "expanding")
        parts.append(
            f"M2 money supply is {regime} at {yoy:.1f}% YoY. "
            + ("M2 contraction tightens financial conditions - historically this precedes "
               "equity drawdowns and softens commodity demand as dollar liquidity shrinks." if yoy < 0 else
               "Moderate M2 growth supports asset prices without stoking excess inflation." if yoy < 6 else
               "Rapid M2 expansion is historically reflationary - positive for hard commodities and "
               "real assets, though it risks keeping inflation elevated.")
        )
    if fed is not None and not fed.empty:
        v = fed.iloc[-1] / 1000
        prev12 = fed.iloc[-12] / 1000 if len(fed) > 12 else v
        direction = "expanding (QE)" if v > prev12 * 1.01 else ("shrinking (QT)" if v < prev12 * 0.99 else "broadly stable")
        parts.append(
            f"The Federal Reserve balance sheet stands at ${v:.2f}T and is {direction}. "
            + ("Balance sheet expansion injects reserves into the system, compressing risk premia and "
               "supporting equity multiples and commodity prices." if v > prev12 else
               "Quantitative tightening withdraws liquidity, raising the cost of capital and pressuring "
               "high-multiple equities and speculative commodity positions.")
        )
    if not yields_df.empty:
        y10 = yields_df["10Y"].iloc[-1] if "10Y" in yields_df.columns else None
        y2  = yields_df["2Y"].iloc[-1]  if "2Y"  in yields_df.columns else None
        if y10 and y2:
            parts.append(
                f"The 10-year Treasury yields {y10:.2f}% against a 2-year at {y2:.2f}%. "
                + ("An elevated 10Y raises the hurdle rate for equities - particularly growth stocks "
                   "whose valuations are most sensitive to discount rate changes. "
                   "For commodities, higher real rates strengthen the dollar and create a headwind for "
                   "gold and dollar-denominated raw materials." if y10 > 4.0 else
                   "Yields at this level are broadly neutral for equities but support gold as "
                   "the opportunity cost of holding bullion remains modest.")
            )
    return "\n\n".join(parts) if parts else "Liquidity data unavailable."


def _narrate_credit(spreads_df: "pd.DataFrame") -> str:
    if spreads_df.empty:
        return "Credit spread data unavailable."
    parts = []
    latest = spreads_df.iloc[-1].dropna()
    curve = latest.get("10Y–2Y (Yield Curve)")
    hy = latest.get("HY Credit Spread")
    ig = latest.get("IG Credit Spread")

    if curve is not None:
        inverted = curve < 0
        parts.append(
            f"The 10Y–2Y yield spread is {curve:.2f}%, indicating "
            + ("an inverted yield curve. Inversion has preceded every US recession since 1955. "
               "This is the single most powerful macro warning signal: equity markets have historically "
               "peaked 6–18 months after initial inversion, while commodity demand typically softens "
               "as growth expectations are repriced lower." if inverted else
               f"a positively sloped curve - consistent with a growth-positive regime that "
               f"supports cyclical equities and industrial commodities.")
        )
    if hy is not None:
        stress = "elevated (recession-level)" if hy > 6 else ("moderately wide" if hy > 4 else "contained")
        parts.append(
            f"High-yield credit spreads are {stress} at {hy:.2f}%. "
            + ("Spreads at these levels signal material default risk in the corporate sector. "
               "This is historically bearish for equities and deeply negative for energy and "
               "metals commodity demand as capital expenditure freezes." if hy > 6 else
               "Spread widening at this level signals caution in risk assets - watch for "
               "further deterioration as the leading indicator for equity drawdowns." if hy > 4 else
               "Contained spreads indicate healthy credit markets and support risk-on positioning "
               "across both equities and cyclical commodities.")
        )
    if ig is not None:
        parts.append(
            f"Investment-grade spreads at {ig:.2f}% suggest "
            + ("broad credit stress - even high-quality issuers face higher financing costs, "
               "which compresses margins and weighs on capital-intensive commodity producers." if ig > 2 else
               "benign credit conditions for corporate borrowers, supportive of capex and commodity demand.")
        )
    return "\n\n".join(parts)


def _narrate_valuations(val_df: "pd.DataFrame", yields_df: "pd.DataFrame") -> str:
    if val_df.empty or "Forward P/E" not in val_df.columns:
        return "Valuation data unavailable."
    parts = []
    y10 = yields_df["10Y"].iloc[-1] if (not yields_df.empty and "10Y" in yields_df.columns) else None
    for _, row in val_df.iterrows():
        mkt = row.get("Market", "")
        pe  = row.get("Forward P/E")
        ey  = row.get("Earnings Yield %")
        gr  = row.get("Fwd EPS Growth %")
        if not mkt or pd.isna(pe):
            continue
        verdict = "expensive" if pe > 25 else ("fair" if pe > 16 else "cheap")
        erp = (ey - y10) if (ey and y10) else None
        erp_text = (
            f" The equity risk premium vs the 10Y is {erp:.2f}pp - "
            + ("thin, meaning equities offer little compensation over risk-free bonds." if erp < 1.5 else
               "reasonable, supporting the case for equities over bonds." if erp > 3 else "at the margin.")
        ) if erp is not None else ""
        growth_text = (
            f" Forward EPS growth of {gr:+.1f}% "
            + ("must accelerate to justify this multiple." if pe > 22 and gr < 10 else
               "comfortably supports the current valuation." if gr > 15 else
               "provides modest support but leaves limited room for disappointment.")
        ) if gr and not pd.isna(gr) else ""
        parts.append(f"{mkt} trades at {pe:.1f}x forward earnings - {verdict}.{erp_text}{growth_text}")
    return "\n\n".join(parts) if parts else "Valuation data unavailable."


def _narrate_cross_asset(
    macro_data: dict, spreads_df: "pd.DataFrame",
    val_df: "pd.DataFrame", idx_prices: "pd.DataFrame",
) -> str:
    """Master narrative: how it all ties together for equities and commodities."""
    parts = []

    # Cycle assessment
    pmi_v   = macro_data.get("ISM Manufacturing PMI", pd.Series()).iloc[-1] if macro_data.get("ISM Manufacturing PMI") is not None and not macro_data.get("ISM Manufacturing PMI", pd.Series()).empty else None
    gdp_v   = macro_data.get("Real GDP Growth (QoQ %)", pd.Series()).iloc[-1] if macro_data.get("Real GDP Growth (QoQ %)") is not None and not macro_data.get("Real GDP Growth (QoQ %)", pd.Series()).empty else None
    curve_v = spreads_df["10Y–2Y (Yield Curve)"].iloc[-1] if (not spreads_df.empty and "10Y–2Y (Yield Curve)" in spreads_df.columns) else None
    hy_v    = spreads_df["HY Credit Spread"].iloc[-1]     if (not spreads_df.empty and "HY Credit Spread"     in spreads_df.columns) else None

    # Determine cycle phase
    late_cycle  = (pmi_v and pmi_v < 50) or (curve_v and curve_v < 0) or (hy_v and hy_v > 4.5)
    early_cycle = (pmi_v and pmi_v > 55) and (gdp_v and gdp_v > 2) and (curve_v is None or curve_v > 0.5)
    phase = "late-cycle" if late_cycle else ("early-cycle recovery" if early_cycle else "mid-cycle")

    parts.append(
        f"Cross-asset synthesis: The aggregate macro signal points to a {phase} environment. "
        + ("Late-cycle conditions historically favour: (1) defensive equities over cyclicals - "
           "healthcare, utilities, and consumer staples over technology and discretionary; "
           "(2) gold and agricultural commodities over base metals as growth slows; "
           "(3) shorter-duration fixed income over equities on a risk-adjusted basis." if late_cycle else
           "Early-cycle conditions historically favour: (1) small and mid-cap equities with high operating leverage; "
           "(2) industrial metals - copper and aluminium benefit most from re-accelerating manufacturing PMI; "
           "(3) energy commodities as demand recovers faster than supply responds." if early_cycle else
           "Mid-cycle conditions suggest: balanced equity exposure with a tilt toward quality growth; "
           "energy and industrial metals supported by stable demand; "
           "gold range-bound unless real rates shift materially.")
    )

    # Performance read
    if not idx_prices.empty:
        best, worst = None, None
        best_r, worst_r = -999, 999
        for name in idx_prices.columns:
            if name == "VIX": continue
            s = idx_prices[name].dropna()
            if len(s) > 21:
                r = (s.iloc[-1] / s.iloc[-22] - 1) * 100
                if r > best_r:  best_r  = r; best  = name
                if r < worst_r: worst_r = r; worst = name
        if best and worst:
            parts.append(
                f"Over the past month, {best} (+{best_r:.1f}%) led and {worst} ({worst_r:.1f}%) lagged. "
                + ("Divergence between regional indices is consistent with the macro regime: "
                   "markets most exposed to rate-sensitive sectors or commodity-importing economies "
                   "underperform when financial conditions tighten." if abs(best_r - worst_r) > 5 else
                   "Narrow dispersion across indices reflects a low-conviction macro environment "
                   "where neither growth optimism nor recession fears dominate positioning.")
            )

    return "\n\n".join(parts)


def page_macro_dashboard(start: str, end: str, fred_key: str = "") -> None:
    st.markdown(
        '<h1 style="font-family:\'DM Sans\',sans-serif;font-size:1.25rem;'
        'font-weight:700;margin-bottom:0.1rem">Macro Intelligence Dashboard</h1>'
        '<p style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;color:#8890a1;'
        'margin:0 0 0.7rem">Money flows · Valuations · Bond yields · Yield spreads · '
        'Earnings growth · GDP · High-freq indicators · Retail participation</p>',
        unsafe_allow_html=True,
    )
    _page_intro(
        "Macro conditions determine the <em>regime</em> under which equity-commodity spillover operates. "
        "In early-cycle recoveries, equities and commodities tend to rise together - spillover is "
        "bidirectional and reinforcing. In late-cycle contractions, the relationship inverts: commodity "
        "price shocks become a <em>drag</em> on equity returns rather than a signal of growth. "
        "Read the regime banner first. Then use the sections below to identify which macro forces "
        "are currently setting the structural backdrop for cross-asset behaviour."
    )

    no_fred = not fred_key
    if no_fred:
        st.info("Add a FRED API key in Settings to unlock bond, macro, and money-flow data. "
                "Valuations and index performance load without it.")

    today = date.today()

    # initialise all data stores so later sections and narratives can safely reference them
    macro_data: dict = {}
    money_data: dict = {}
    yields_df  = pd.DataFrame()
    spreads_df = pd.DataFrame()
    val_df     = pd.DataFrame()
    idx_prices = pd.DataFrame()

    # ── Regime banner - quick upfront read using spread data ─────────────────
    if fred_key:
        try:
            _sp = _load_spreads(fred_key, start, end)
            _curve = _sp["10Y–2Y (Yield Curve)"].iloc[-1] if (not _sp.empty and "10Y–2Y (Yield Curve)" in _sp.columns) else None
            _hy    = _sp["HY Credit Spread"].iloc[-1]     if (not _sp.empty and "HY Credit Spread" in _sp.columns) else None
            _m     = _load_macro(fred_key, start, end)
            _pmi   = _m.get("ISM Manufacturing PMI")
            _pmi_v = float(_pmi.iloc[-1]) if _pmi is not None and not _pmi.empty else None
            _late  = (_curve is not None and _curve < 0) or (_hy is not None and _hy > 4.5) or (_pmi_v is not None and _pmi_v < 50)
            _early = (_pmi_v is not None and _pmi_v > 55) and (_curve is None or _curve > 0.5)
            if _late:
                _phase, _pcolor = "LATE-CYCLE CAUTION", "#b03a2e"
                _psub = "Yield curve or credit spreads signal elevated recession risk - favour defensives and gold"
            elif _early:
                _phase, _pcolor = "EARLY-CYCLE RECOVERY", "#1e8449"
                _psub = "PMI expanding and curve positive - cyclicals, copper, and energy are the beneficiaries"
            else:
                _phase, _pcolor = "MID-CYCLE EXPANSION", "#8E6F3E"
                _psub = "Balanced conditions - quality growth equities and stable commodity demand"
            _regime_banner(_phase, _psub, _pcolor)
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════════
    # 1. GDP & GROWTH - Where is the economy?
    # ══════════════════════════════════════════════════════════════════════════
    _section_header("01", "Economic Growth", "GDP, industrial output - the broadest cycle context")

    if no_fred:
        st.caption("Requires FRED API key.")
    else:
        with st.spinner("Loading macro indicators…"):
            macro_data = _load_macro(fred_key, start, end)

        if macro_data:
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
                    good_up = key not in ("Unemployment Rate (%)", "CPI YoY (%)")
                    _kpi(
                        k_cols[i], key,
                        f"{latest:.2f}{'%' if '%' in key else ''}",
                        f"{abs(delta):.2f}",
                        delta_up=is_up if good_up else not is_up,
                    )

            g_l, g_r = st.columns(2)
            with g_l:
                if "Real GDP Growth (QoQ %)" in macro_data:
                    s = macro_data["Real GDP Growth (QoQ %)"].last("10Y")
                    fig_gdp = go.Figure(go.Bar(
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

            with g_r:
                if "Industrial Production" in macro_data:
                    s = macro_data["Industrial Production"].last("10Y")
                    s_yoy = s.pct_change(12) * 100
                    fig_ip = go.Figure(go.Scatter(
                        x=s_yoy.dropna().index, y=s_yoy.dropna().values,
                        line=dict(color="#8E6F3E", width=1.8),
                        fill="tozeroy", fillcolor="rgba(142,111,62,0.08)",
                        name="Ind. Production YoY %",
                    ))
                    fig_ip.add_hline(y=0, line=dict(color="#ABABAB", dash="dot", width=1))
                    fig_ip.update_layout(
                        template="purdue", height=280,
                        title=dict(text="Industrial Production (YoY %)", font=dict(size=10)),
                        yaxis=dict(title="YoY (%)", ticksuffix="%"),
                        margin=dict(l=40, r=10, t=35, b=30),
                    )
                    _chart(fig_ip)

            _insight_note(
                "GDP and industrial production set the broadest context: are we in expansion or contraction? "
                "Two consecutive negative GDP quarters = technical recession. Industrial production leads "
                "earnings revisions by roughly one quarter."
            )
            _narrative_box(_narrate_growth(macro_data))

    # ══════════════════════════════════════════════════════════════════════════
    # 2. HIGH-FREQ INDICATORS - Where is the economy heading?
    # ══════════════════════════════════════════════════════════════════════════
    _section_header("02", "High-Frequency Indicators", "PMI · retail sales · payrolls · unemployment")

    if no_fred:
        st.caption("Requires FRED API key.")
    elif macro_data:
        hf_l, hf_r = st.columns(2)
        with hf_l:
            if "ISM Manufacturing PMI" in macro_data:
                s = macro_data["ISM Manufacturing PMI"].last("5Y")
                fig_pmi = go.Figure(go.Scatter(
                    x=s.index, y=s.values, name="ISM PMI",
                    line=dict(color="#CFB991", width=2),
                    fill="tozeroy", fillcolor="rgba(207,185,145,0.1)",
                ))
                fig_pmi.add_hline(y=50, line=dict(color="#c0392b", dash="dash", width=1.2),
                                  annotation_text="50 = Expansion/Contraction", annotation_font_size=8)
                fig_pmi.update_layout(
                    template="purdue", height=260,
                    title=dict(text="ISM Manufacturing PMI", font=dict(size=10)),
                    yaxis=dict(title="PMI"),
                    margin=dict(l=40, r=10, t=35, b=30),
                )
                _chart(fig_pmi)

        with hf_r:
            if "Retail Sales ex-Auto" in macro_data:
                s = macro_data["Retail Sales ex-Auto"].last("5Y")
                s_pct = s.pct_change(12) * 100
                fig_rs = go.Figure(go.Scatter(
                    x=s_pct.dropna().index, y=s_pct.dropna().values,
                    line=dict(color="#3498db", width=1.8),
                    fill="tozeroy", fillcolor="rgba(52,152,219,0.08)",
                    name="YoY %",
                ))
                fig_rs.add_hline(y=0, line=dict(color="#ABABAB", dash="dot", width=1))
                fig_rs.update_layout(
                    template="purdue", height=260,
                    title=dict(text="Retail Sales ex-Auto (YoY %)", font=dict(size=10)),
                    yaxis=dict(title="YoY (%)", ticksuffix="%"),
                    margin=dict(l=40, r=10, t=35, b=30),
                )
                _chart(fig_rs)

        hf2_l, hf2_r = st.columns(2)
        with hf2_l:
            if "Nonfarm Payrolls (MoM k)" in macro_data:
                s = macro_data["Nonfarm Payrolls (MoM k)"].last("5Y") / 1000
                fig_pay = go.Figure(go.Bar(
                    x=s.index, y=s.values,
                    marker_color=["#2e7d32" if v >= 0 else "#c0392b" for v in s.values],
                    name="MoM Change",
                ))
                fig_pay.update_layout(
                    template="purdue", height=240,
                    title=dict(text="Nonfarm Payrolls (MoM, thousands)", font=dict(size=10)),
                    yaxis=dict(title="Change (k)"),
                    margin=dict(l=40, r=10, t=35, b=30),
                )
                _chart(fig_pay)

        with hf2_r:
            if "Unemployment Rate (%)" in macro_data:
                s = macro_data["Unemployment Rate (%)"].last("10Y")
                fig_un = go.Figure(go.Scatter(
                    x=s.index, y=s.values,
                    line=dict(color="#c0392b", width=1.8),
                    fill="tozeroy", fillcolor="rgba(192,57,43,0.06)",
                    name="Unemployment %",
                ))
                fig_un.update_layout(
                    template="purdue", height=240,
                    title=dict(text="Unemployment Rate (%)", font=dict(size=10)),
                    yaxis=dict(title="%", ticksuffix="%"),
                    margin=dict(l=40, r=10, t=35, b=30),
                )
                _chart(fig_un)

        _insight_note(
            "PMI is the earliest leading indicator - it turns before GDP by 1–2 quarters. "
            "Retail sales capture consumer revenue momentum; falling sales YoY typically precede "
            "earnings downgrades in consumer discretionary. Payrolls above +150k/month = healthy labour market."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 3. MONEY FLOWS & LIQUIDITY - What is driving the cycle?
    # ══════════════════════════════════════════════════════════════════════════
    _section_header("03", "Money Flows & Liquidity", "M2 · Fed balance sheet · consumer sentiment")

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
                    fig_m2 = go.Figure(go.Scatter(
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
                    fig_fed = go.Figure(go.Scatter(
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
                    avg = float(s.mean())
                    fig_sent = go.Figure(go.Scatter(
                        x=s.index, y=s.values, name="Consumer Sentiment",
                        line=dict(color="#3498db", width=2),
                    ))
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
                "M2 contraction (negative YoY) is historically associated with asset price stress - "
                "the 2022 drawdown coincided with the first M2 decline since the 1930s. "
                "Fed balance sheet expansion (QE) injects liquidity and typically lifts risk assets; "
                "QT (reduction) tightens it. Consumer sentiment below its long-run average signals "
                "reduced retail participation - a proxy for weakening demand-side flows."
            )

    # ══════════════════════════════════════════════════════════════════════════
    # 4. BOND YIELDS - What does the rates market price in?
    # ══════════════════════════════════════════════════════════════════════════
    _section_header("04", "Bond Yields", "Treasury yield curve - the risk-free rate backdrop")

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
                latest_yields = yields_df.iloc[-1].dropna()
                maturities = {"3M": 0.25, "2Y": 2, "5Y": 5, "10Y": 10, "30Y": 30}
                x_vals, y_vals = [], []
                for lbl, yrs in maturities.items():
                    if lbl in latest_yields.index:
                        x_vals.append(yrs)
                        y_vals.append(latest_yields[lbl])
                fig_curve = go.Figure(go.Scatter(
                    x=x_vals, y=y_vals, mode="lines+markers",
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
                "Higher long-end yields raise the discount rate on equities, compressing valuations. "
                "When the 10Y yield exceeds the S&P earnings yield, bonds become competitive with stocks - "
                "watch this spread as a regime signal. A steep upward-sloping curve signals growth optimism; "
                "a flat or inverted curve signals tightening or recession expectations."
            )
            _narrative_box(_narrate_liquidity(money_data, yields_df))

    # ══════════════════════════════════════════════════════════════════════════
    # 5. YIELD SPREADS & CREDIT - What does the credit market say?
    # ══════════════════════════════════════════════════════════════════════════
    _section_header("05", "Yield Spreads & Credit Risk", "Yield curve shape · IG and HY credit spreads")

    if no_fred:
        st.caption("Requires FRED API key.")
    else:
        with st.spinner("Loading spread data…"):
            spreads_df = _load_spreads(fred_key, start, end)

        if not spreads_df.empty:
            sp_l, sp_r = st.columns(2)

            with sp_l:
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
                "Credit spreads are the market's real-time verdict on default risk. "
                "They widen before equity markets sell off, making them a useful leading risk indicator. "
                "HY spreads above 600bps have historically coincided with recessions; IG above 200bps "
                "signals broad credit stress. An inverted yield curve alongside widening spreads = "
                "the strongest combined recession signal."
            )
            _narrative_box(_narrate_credit(spreads_df))

    # ══════════════════════════════════════════════════════════════════════════
    # 6. VALUATIONS & EARNINGS - Are equities priced for this environment?
    # ══════════════════════════════════════════════════════════════════════════
    _section_header("06", "Valuations & Expected Earnings Growth", "Forward P/E · earnings yield · equity risk premium")

    with st.spinner("Loading valuation data…"):
        val_df = _load_valuations()

    if not val_df.empty and "Trailing P/E" in val_df.columns:
        v_l, v_r = st.columns([1.2, 1])

        with v_l:
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
            _TBL_CSS_VAL = """
<style>
.ec-table{width:100%;border-collapse:collapse;font-family:'DM Sans',sans-serif;font-size:0.78rem}
.ec-table th{background:#1a1d27;color:#CFB991;padding:7px 10px;text-align:left;
    border-bottom:1px solid rgba(207,185,145,0.3);font-weight:600;
    letter-spacing:0.06em;text-transform:uppercase;font-size:0.68rem}
.ec-table td{padding:5px 10px;border-bottom:1px solid #1e2130;color:#e8e9ed}
.ec-table tr:nth-child(even) td{background:#141720}
.ec-table tr:nth-child(odd) td{background:#0f1117}
.ec-table tr:hover td{background:#1e2230}
</style>"""
            val_cols = list(display_df.columns)
            val_header_html = "<th>Market</th>" + "".join(f"<th>{c}</th>" for c in val_cols)
            val_rows_html = ""
            for mkt, row in display_df.iterrows():
                cells = f"<td style='color:#b8bec8'>{mkt}</td>"
                for col in val_cols:
                    v = row[col]
                    is_nan = pd.isna(v) if not isinstance(v, str) else False
                    if is_nan:
                        cells += "<td style='color:#8890a1'>-</td>"
                    elif col in ("Trailing P/E", "Forward P/E"):
                        if v > 30:
                            style = "color:#f87171;font-weight:600"
                        elif v > 22:
                            style = "color:#CFB991"
                        elif v > 15:
                            style = "color:#4ade80"
                        else:
                            style = "color:#4ade80;font-weight:600"
                        cells += f"<td style='{style}'>{v:.1f}</td>"
                    elif col == "Fwd EPS Growth %":
                        if v > 15:
                            style = "color:#4ade80;font-weight:700"
                        elif v > 5:
                            style = "color:#4ade80"
                        elif v > 0:
                            style = "color:#8890a1"
                        else:
                            style = "color:#f87171;font-weight:700"
                        cells += f"<td style='{style}'>{v:+.1f}%</td>"
                    elif col == "P/B":
                        cells += f"<td style='color:#e8e9ed'>{v:.2f}</td>"
                    elif col == "Earnings Yield %":
                        cells += f"<td style='color:#e8e9ed'>{v:.1f}</td>"
                    else:
                        cells += f"<td style='color:#e8e9ed'>{v}</td>"
                val_rows_html += f"<tr>{cells}</tr>"
            html_val = (
                _TBL_CSS_VAL
                + "<table class='ec-table'>"
                + f"<thead><tr>{val_header_html}</tr></thead>"
                + f"<tbody>{val_rows_html}</tbody>"
                + "</table>"
            )
            st.markdown(html_val, unsafe_allow_html=True)

        with v_r:
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
            "Forward P/E above 20x demands strong earnings growth to justify. "
            "With rates elevated (section 4), the equity risk premium compresses - "
            "making growth disappointments more painful. Markets priced at high multiples "
            "with slowing PMI and widening credit spreads (sections 2 & 5) = the classic "
            "late-cycle vulnerability setup."
        )
        _narrative_box(_narrate_valuations(val_df, yields_df))

    # ══════════════════════════════════════════════════════════════════════════
    # 7. INDEX PERFORMANCE - How have markets responded to all of the above?
    # ══════════════════════════════════════════════════════════════════════════
    _section_header("07", "Index Performance & Market Stress", "Rolling returns · VIX - the market's verdict on everything above")

    perf_windows = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "YTD": None}
    perf_start = (today - timedelta(days=400)).isoformat()

    with st.spinner("Loading index data…"):
        idx_prices = _load_index_perf(perf_start)

    if not idx_prices.empty:
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
                    row[wlabel] = round((s_ytd.iloc[-1] / s_ytd.iloc[0] - 1) * 100, 2) if len(s_ytd) >= 2 else None
                else:
                    row[wlabel] = round((s.iloc[-1] / s.iloc[-wdays - 1] - 1) * 100, 2) if len(s) > wdays else None
            row["Latest"] = round(float(s.iloc[-1]), 2)
            perf_rows.append(row)

        perf_df = pd.DataFrame(perf_rows).set_index("Index")
        ret_cols = [c for c in perf_df.columns if c != "Latest"]

        def _color_ret(val):
            if pd.isna(val): return ""
            if val > 2:  return "background-color:#c8e6c9;color:#1b5e20"
            if val > 0:  return "background-color:#e8f5e9;color:#2e7d32"
            if val > -2: return "background-color:#ffebee;color:#b71c1c"
            return            "background-color:#ffcdd2;color:#b71c1c"

        idx_l, idx_r = st.columns([1.6, 1])
        with idx_l:
            _TBL_CSS_PERF = """
<style>
.ec-table{width:100%;border-collapse:collapse;font-family:'DM Sans',sans-serif;font-size:0.78rem}
.ec-table th{background:#1a1d27;color:#CFB991;padding:7px 10px;text-align:left;
    border-bottom:1px solid rgba(207,185,145,0.3);font-weight:600;
    letter-spacing:0.06em;text-transform:uppercase;font-size:0.68rem}
.ec-table td{padding:5px 10px;border-bottom:1px solid #1e2130;color:#e8e9ed}
.ec-table tr:nth-child(even) td{background:#141720}
.ec-table tr:nth-child(odd) td{background:#0f1117}
.ec-table tr:hover td{background:#1e2230}
</style>"""
            perf_all_cols = ["Latest"] + ret_cols
            perf_header = "<th>Index</th>" + "".join(f"<th>{c}</th>" for c in perf_all_cols)
            perf_rows_html = ""
            for idx_name, row in perf_df.iterrows():
                cells = f"<td style='color:#b8bec8'>{idx_name}</td>"
                latest_v = row.get("Latest")
                cells += f"<td style='color:#8890a1'>{latest_v:.2f}</td>" if not pd.isna(latest_v) else "<td style='color:#8890a1'>-</td>"
                for rc in ret_cols:
                    v = row.get(rc)
                    if pd.isna(v):
                        cells += "<td style='color:#8890a1'>-</td>"
                    elif v > 2:
                        cells += f"<td style='color:#4ade80;font-weight:600'>{v:+.2f}%</td>"
                    elif v > 0:
                        cells += f"<td style='color:#4ade80'>{v:+.2f}%</td>"
                    elif v > -2:
                        cells += f"<td style='color:#f87171'>{v:+.2f}%</td>"
                    else:
                        cells += f"<td style='color:#f87171;font-weight:600'>{v:+.2f}%</td>"
                perf_rows_html += f"<tr>{cells}</tr>"
            html_perf = (
                _TBL_CSS_PERF
                + "<table class='ec-table'>"
                + f"<thead><tr>{perf_header}</tr></thead>"
                + f"<tbody>{perf_rows_html}</tbody>"
                + "</table>"
            )
            st.markdown(html_perf, unsafe_allow_html=True)

        with idx_r:
            if "VIX" in idx_prices.columns:
                vix = idx_prices["VIX"].dropna().last("90D")
                fig_vix = go.Figure(go.Scatter(
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
            "Index performance is the cumulative market verdict on everything above. "
            "Cross-reference with sections 1–6: strong GDP + low spreads + reasonable valuations "
            "should explain green returns. Drawdowns alongside widening spreads or inverted curves "
            "are the most dangerous combination - the macro backdrop confirms the price signal."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # CROSS-ASSET SYNTHESIS - The macro story in one place
    # ══════════════════════════════════════════════════════════════════════════
    _section_header("∑", "Macro Synthesis", "Cross-asset implications for equities and commodities")
    _narrative_box(_narrate_cross_asset(macro_data, spreads_df, val_df, idx_prices))

    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE CREDIT BUBBLE MONITOR
    # ══════════════════════════════════════════════════════════════════════════
    _section_header("⚠", "Private Credit Bubble Monitor",
                    "HY spreads · leveraged loans · BDC signals · anatomy of a bust")

    st.markdown(
        f'<p style="{_F}font-size:0.68rem;color:#8890a1;margin:0 0 1rem">'
        f'Private credit AUM has grown from ~$500B (2010) to $2T+ (2024). '
        f'Most loans are floating-rate SOFR-linked - at current rates, all-in borrower costs are 9–12% '
        f'on deals structured at 5–7× leverage. Quarterly mark-to-model valuation masks latent stress '
        f'until it cannot be hidden. This panel tracks the observable liquid proxies for that hidden risk.</p>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading private credit data…"):
        _pc_px = _load_pc_proxies(start, end)
        _pc_hy = _fred(fred_key, "BAMLH0A0HYM2", start, end) if not no_fred else pd.Series(dtype=float)
        _pc_ig = _fred(fred_key, "BAMLC0A0CM",   start, end) if not no_fred else pd.Series(dtype=float)

    # ── compute metrics ──────────────────────────────────────────────────────
    _hy_curr = _hy_pct = _hy_90d = _ig_curr = None
    _bkln_90d = _bdc_90d = _spx_90d = _bkln_alpha = _bdc_alpha = _pc_score = None

    if not _pc_hy.empty and len(_pc_hy) > 20:
        _hy_curr = float(_pc_hy.iloc[-1])
        _hy_pct  = float((_pc_hy < _hy_curr).mean() * 100)
        _hy_90d  = float(_pc_hy.iloc[-1] - _pc_hy.iloc[-91]) if len(_pc_hy) > 91 else 0.0
    if not _pc_ig.empty:
        _ig_curr = float(_pc_ig.iloc[-1])

    if not _pc_px.empty:
        _N = 91
        if "SPY" in _pc_px.columns:
            _spy_s = _pc_px["SPY"].dropna()
            if len(_spy_s) > _N:
                _spx_90d = float((_spy_s.iloc[-1] / _spy_s.iloc[-_N] - 1) * 100)
        if "BKLN" in _pc_px.columns:
            _bkln_s = _pc_px["BKLN"].dropna()
            if len(_bkln_s) > _N and _spx_90d is not None:
                _bkln_90d   = float((_bkln_s.iloc[-1] / _bkln_s.iloc[-_N] - 1) * 100)
                _bkln_alpha = _bkln_90d - _spx_90d
        _bdc_ts = [t for t in ["ARCC", "OBDC", "FSK"] if t in _pc_px.columns]
        if _bdc_ts and _spx_90d is not None:
            _bdc_vals = []
            for _t in _bdc_ts:
                _s = _pc_px[_t].dropna()
                if len(_s) > _N:
                    _bdc_vals.append(float((_s.iloc[-1] / _s.iloc[-_N] - 1) * 100))
            if _bdc_vals:
                _bdc_90d   = float(np.mean(_bdc_vals))
                _bdc_alpha = _bdc_90d - _spx_90d

    _sc = {}
    if _hy_pct is not None:  _sc["hy"]   = 100.0 - _hy_pct
    if _bkln_alpha is not None: _sc["bkln"] = min(100, max(0, -_bkln_alpha * 5))
    if _bdc_alpha  is not None: _sc["bdc"]  = min(100, max(0, -_bdc_alpha  * 4))
    if _sc:
        _w = {"hy": 0.5, "bkln": 0.3, "bdc": 0.2}
        _tw = sum(_w.get(k, 0) for k in _sc)
        _pc_score = sum(_sc[k] * _w.get(k, 0) for k in _sc) / max(_tw, 0.01)

    _pc_score_val = _pc_score if _pc_score is not None else 0.0
    _pc_color = ("#c0392b" if _pc_score_val >= 65 else
                 "#b7770d" if _pc_score_val >= 40 else "#1e8449")
    _pc_label = ("Elevated" if _pc_score_val >= 65 else
                 "Moderate" if _pc_score_val >= 40 else "Low")

    # ── KPI strip ────────────────────────────────────────────────────────────
    def _pc_kpi(col, label, value, sub="", bar_pct=None, bar_color="#CFB991"):
        _bar = ""
        if bar_pct is not None:
            _bar = (
                f'<div style="margin-top:6px;height:3px;background:#2a2d3a;border-radius:2px">'
                f'<div style="height:3px;width:{min(100,max(0,bar_pct))}%;'
                f'background:{bar_color};border-radius:2px"></div></div>'
            )
        col.markdown(
            f'<div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:6px;'
            f'padding:10px 14px;margin-bottom:8px">'
            f'<div style="{_F}font-size:0.54rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.12em;color:#8E6F3E;margin-bottom:2px">{label}</div>'
            f'<div style="{_F}font-size:1.0rem;font-weight:700;color:#e8e9ed;line-height:1.2">{value}</div>'
            f'<div style="{_F}font-size:0.62rem;color:#8890a1;margin-top:2px">{sub}</div>'
            f'{_bar}</div>',
            unsafe_allow_html=True,
        )

    _k1, _k2, _k3, _k4, _k5 = st.columns(5)
    _pc_kpi(_k1, "Private Credit AUM", "$2T+",
            sub="2024 estimate (+300% since 2015)",
            bar_pct=88, bar_color="#c0392b")
    _pc_kpi(_k2, "HY OAS (bps)",
            f"{_hy_curr:.0f}" if _hy_curr else "-",
            sub=f"{_hy_pct:.0f}th percentile since {start[:4]}" if _hy_pct is not None else "FRED key required",
            bar_pct=_hy_pct, bar_color=("#c0392b" if _hy_pct is not None and _hy_pct < 30 else
                                         "#b7770d" if _hy_pct is not None and _hy_pct < 50 else "#1e8449"))
    _pc_kpi(_k3, "HY OAS 90d Δ",
            f"{_hy_90d:+.0f} bps" if _hy_90d is not None else "-",
            sub="widening = stress onset" if _hy_90d and _hy_90d > 20 else "spread trend",
            bar_pct=min(100, abs(_hy_90d) * 1.5) if _hy_90d is not None else 0,
            bar_color="#c0392b" if _hy_90d and _hy_90d > 20 else "#1e8449")
    _pc_kpi(_k4, "BKLN / BDC Alpha",
            f"{_bkln_alpha:+.1f}% / {_bdc_alpha:+.1f}%" if _bkln_alpha is not None and _bdc_alpha is not None else "-",
            sub="vs S&P 500 (90d) - negative = stress",
            bar_pct=min(100, max(0, -(_bkln_alpha or 0) * 6)),
            bar_color="#c0392b" if _bkln_alpha and _bkln_alpha < -4 else "#b7770d")
    _pc_kpi(_k5, "PC Bubble Score",
            f"{_pc_score_val:.0f} / 100" if _pc_score is not None else "-",
            sub=_pc_label,
            bar_pct=_pc_score_val,
            bar_color=_pc_color)

    # ── Charts: HY OAS history + BKLN/BDC vs SPY ────────────────────────────
    _ch_left, _ch_right = st.columns([1.1, 1])

    with _ch_left:
        if not _pc_hy.empty and len(_pc_hy) > 50:
            _fig_hy = go.Figure()
            # Danger zone fill: <300 bps (bubble) and >700 bps (burst)
            _fig_hy.add_hrect(y0=0,   y1=300, fillcolor="rgba(183,119,13,0.08)",
                               line_width=0, annotation_text="Bubble zone (<300bps)",
                               annotation_font=dict(size=8, color="#b7770d"),
                               annotation_position="top left")
            _fig_hy.add_hrect(y0=700, y1=2200, fillcolor="rgba(192,57,43,0.08)",
                               line_width=0, annotation_text="Crisis zone (>700bps)",
                               annotation_font=dict(size=8, color="#c0392b"),
                               annotation_position="bottom left")
            # Historical mean
            _hy_mean = float(_pc_hy.mean())
            _fig_hy.add_hline(y=_hy_mean, line=dict(color="#6b7280", width=1, dash="dot"),
                               annotation_text=f"Hist. mean {_hy_mean:.0f}bps",
                               annotation_font=dict(size=8, color="#6b7280"))
            # Main OAS line
            _fig_hy.add_trace(go.Scatter(
                x=_pc_hy.index, y=_pc_hy.values,
                name="HY OAS", fill="tozeroy",
                fillcolor="rgba(192,57,43,0.12)",
                line=dict(color="#c0392b", width=1.5),
            ))
            # Current level marker
            if _hy_curr is not None:
                _fig_hy.add_hline(y=_hy_curr,
                                   line=dict(color="#CFB991", width=1.2, dash="dash"),
                                   annotation_text=f"Now {_hy_curr:.0f}bps ({_hy_pct:.0f}th pct)",
                                   annotation_font=dict(size=8, color="#CFB991"),
                                   annotation_position="top right")
            _fig_hy.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                template="purdue", height=300,
                title=dict(text="HY OAS - ICE BofA US High Yield (FRED)", font=dict(size=10, color="#e8e9ed")),
                yaxis=dict(title="Spread (bps)", tickfont=dict(color="#8890a1"), gridcolor="#1e2130"),
                xaxis=dict(tickfont=dict(color="#8890a1"), gridcolor="#1e2130",
                           rangeslider=dict(visible=False)),
                font=dict(color="#e8e9ed"),
                margin=dict(l=45, r=10, t=38, b=30),
                showlegend=False,
            )
            _chart(_fig_hy)
        else:
            st.markdown(
                f'<div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:6px;'
                f'padding:2rem;text-align:center;color:#8890a1;font-size:0.72rem">'
                f'HY OAS chart requires FRED API key (already configured in secrets.toml)</div>',
                unsafe_allow_html=True,
            )

    with _ch_right:
        if not _pc_px.empty and "SPY" in _pc_px.columns:
            _fig_pc = go.Figure()
            _cols_to_plot = {
                "SPY":  ("S&P 500",             "#6b7280", "dot",   1.0),
                "BKLN": ("BKLN (Lev. Loans)",   "#2980b9", "solid", 1.8),
                "ARCC": ("ARCC (BDC)",           "#e67e22", "solid", 1.2),
                "OBDC": ("OBDC (BDC)",           "#8e44ad", "solid", 1.2),
                "FSK":  ("FSK (BDC)",            "#16a085", "solid", 1.2),
            }
            _ref_n = min(252, len(_pc_px) - 1)  # normalise to 1 year ago
            for _tk, (_nm, _col, _dash, _wid) in _cols_to_plot.items():
                if _tk in _pc_px.columns:
                    _s = _pc_px[_tk].dropna()
                    if len(_s) > _ref_n:
                        _base = float(_s.iloc[-_ref_n])
                        _norm = _s.iloc[-_ref_n:] / _base * 100
                        _fig_pc.add_trace(go.Scatter(
                            x=_norm.index, y=_norm.values,
                            name=_nm,
                            line=dict(color=_col, width=_wid, dash=_dash),
                        ))
            _fig_pc.add_hline(y=100, line=dict(color="#2a2d3a", width=1))
            _fig_pc.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                template="purdue", height=300,
                title=dict(text="BKLN & BDC Basket vs S&P 500 (normalised, 1Y)", font=dict(size=10, color="#e8e9ed")),
                yaxis=dict(title="Indexed (base=100)", tickfont=dict(color="#8890a1"), gridcolor="#1e2130"),
                xaxis=dict(tickfont=dict(color="#8890a1"), gridcolor="#1e2130",
                           rangeslider=dict(visible=False)),
                font=dict(color="#e8e9ed"),
                margin=dict(l=45, r=10, t=38, b=30),
                legend=dict(orientation="h", y=-0.18, font=dict(size=8), bgcolor="rgba(0,0,0,0)"),
            )
            _chart(_fig_pc)
        else:
            st.markdown(
                f'<div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:6px;'
                f'padding:2rem;text-align:center;color:#8890a1;font-size:0.72rem">'
                f'Loading BKLN/BDC chart data…</div>',
                unsafe_allow_html=True,
            )

    # ── Trigger signal table ─────────────────────────────────────────────────
    def _sig_row(label, value_str, threshold_str, status_color, status_label):
        return (
            f'<tr>'
            f'<td style="font-weight:600;color:#c8cdd8">{label}</td>'
            f'<td style="font-family:\'JetBrains Mono\',monospace;color:#e8e9ed">{value_str}</td>'
            f'<td style="color:#8890a1;font-size:0.72rem">{threshold_str}</td>'
            f'<td><span style="background:{status_color}22;color:{status_color};'
            f'border:1px solid {status_color}44;border-radius:3px;padding:2px 8px;'
            f'font-size:0.65rem;font-weight:700;letter-spacing:0.08em">{status_label}</span></td>'
            f'</tr>'
        )

    _T_GREEN = "#1e8449"; _T_AMBER = "#b7770d"; _T_RED = "#c0392b"

    _sig_rows = []
    if _hy_pct is not None:
        _c = _T_RED if _hy_pct < 25 else (_T_AMBER if _hy_pct < 45 else _T_GREEN)
        _l = "BUBBLE" if _hy_pct < 25 else ("CAUTION" if _hy_pct < 45 else "NORMAL")
        _sig_rows.append(_sig_row("HY OAS Percentile", f"{_hy_pct:.0f}th pct",
                                   "Bubble: <30th · Stress: >70th", _c, _l))
    if _hy_90d is not None:
        _c = _T_RED if _hy_90d > 40 else (_T_AMBER if _hy_90d > 15 else _T_GREEN)
        _l = "WIDENING" if _hy_90d > 40 else ("RISING" if _hy_90d > 15 else "STABLE")
        _sig_rows.append(_sig_row("HY OAS 90d Trend", f"{_hy_90d:+.0f} bps",
                                   "Danger: >+40bps in 90d", _c, _l))
    if _bkln_alpha is not None:
        _c = _T_RED if _bkln_alpha < -8 else (_T_AMBER if _bkln_alpha < -3 else _T_GREEN)
        _l = "STRESS" if _bkln_alpha < -8 else ("CAUTION" if _bkln_alpha < -3 else "NORMAL")
        _sig_rows.append(_sig_row("BKLN vs SPY Alpha (90d)", f"{_bkln_alpha:+.1f}%",
                                   "Danger: <−5% underperformance", _c, _l))
    if _bdc_alpha is not None:
        _c = _T_RED if _bdc_alpha < -8 else (_T_AMBER if _bdc_alpha < -4 else _T_GREEN)
        _l = "STRESS" if _bdc_alpha < -8 else ("CAUTION" if _bdc_alpha < -4 else "NORMAL")
        _sig_rows.append(_sig_row("BDC Basket vs SPY (90d)", f"{_bdc_alpha:+.1f}%",
                                   "Discount-to-book widening: leads NAV cuts by 1–2Q", _c, _l))
    if _hy_curr is not None and _ig_curr is not None and _ig_curr > 0:
        _ratio = _hy_curr / _ig_curr
        _c = _T_RED if _ratio < 3.5 else (_T_AMBER if _ratio < 4.5 else _T_GREEN)
        _l = "COMPRESSED" if _ratio < 3.5 else ("TIGHT" if _ratio < 4.5 else "NORMAL")
        _sig_rows.append(_sig_row("HY / IG OAS Ratio", f"{_ratio:.2f}×",
                                   "Bubble: <3.5× (investors underpricing default risk)", _c, _l))
    # Coverage ratio proxy - qualitative based on known macro context
    _cov_c = _T_AMBER; _cov_l = "ELEVATED"
    _sig_rows.append(_sig_row("Coverage Ratio Risk",
                               "SOFR 4.3% + 500bps spread = ~9.5% all-in",
                               "Danger: all-in cost >9% on 5–7× levered borrowers",
                               _cov_c, _cov_l))

    if _sig_rows:
        st.markdown(
            f'<style>'
            f'.pc-tbl{{width:100%;border-collapse:collapse;font-family:\'DM Sans\',sans-serif;font-size:0.78rem}}'
            f'.pc-tbl th{{background:#1a1d27;color:#CFB991;padding:7px 10px;text-align:left;'
            f'    border-bottom:1px solid rgba(207,185,145,0.25);font-weight:600;'
            f'    letter-spacing:0.06em;text-transform:uppercase;font-size:0.65rem}}'
            f'.pc-tbl td{{padding:6px 10px;border-bottom:1px solid #1e2130}}'
            f'.pc-tbl tr:hover td{{background:#1e2230}}'
            f'</style>'
            f'<table class="pc-tbl">'
            f'<thead><tr>'
            f'<th>Signal</th><th>Current Level</th><th>Threshold / Context</th><th>Status</th>'
            f'</tr></thead><tbody>'
            + "".join(_sig_rows) +
            f'</tbody></table>',
            unsafe_allow_html=True,
        )

    # ── Anatomy of a bust - HTML flow infographic ────────────────────────────
    st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="{_F}font-size:0.58rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.14em;color:#8E6F3E;margin:0 0 0.6rem">Anatomy of a Private Credit Bust</p>',
        unsafe_allow_html=True,
    )

    _NODE_CSS = (
        "display:inline-flex;flex-direction:column;align-items:center;justify-content:center;"
        "background:#1a1d27;border:1px solid #2a2d3a;border-radius:6px;"
        "padding:0.55rem 0.7rem;min-width:95px;text-align:center;"
        "font-family:'DM Sans',sans-serif;"
    )
    _ARROW = '<span style="color:#CFB991;font-size:1.1rem;padding:0 4px;align-self:center">→</span>'

    def _node(icon, title, sub, border_color="#2a2d3a"):
        return (
            f'<div style="{_NODE_CSS}border-color:{border_color}">'
            f'<span style="font-size:1.0rem;margin-bottom:3px">{icon}</span>'
            f'<span style="font-size:0.68rem;font-weight:700;color:#e8e9ed;line-height:1.3">{title}</span>'
            f'<span style="font-size:0.58rem;color:#8890a1;margin-top:2px;line-height:1.3">{sub}</span>'
            f'</div>'
        )

    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px;'
        f'background:#0f1117;border:1px solid #1e2130;border-radius:8px;padding:1rem 1.1rem;">'
        + _node("📈", "Rates Stay High", "SOFR 4.3%+<br>all-in ~9–12%", "#b7770d")
        + _ARROW
        + _node("💸", "Coverage Squeeze", "EBITDA ÷ interest<br>&lt;1.5×", "#b7770d")
        + _ARROW
        + _node("📋", "PIK &amp; Amends", "Pay-in-kind<br>elections rise", "#b7770d")
        + _ARROW
        + _node("📉", "NAV Writedowns", "Quarterly marks<br>lag 1–2 quarters", "#c0392b")
        + _ARROW
        + _node("🚪", "Redemption Gates", "Semi-liquid fund<br>lockups trigger", "#c0392b")
        + _ARROW
        + _node("🔥", "HY OAS Widens", "Forced sales hit<br>public markets", "#c0392b")
        + _ARROW
        + _node("🏦", "Fin. Sector Hit", "PE sponsors &amp;<br>BDC equity falls", "#c0392b")
        + _ARROW
        + _node("🛢️", "Commodity Drop", "Copper/Oil demand<br>destruction", "#8e44ad")
        + f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
    _insight_note(
        "Private credit stress typically manifests 2–4 quarters AFTER the conditions that cause it - "
        "because quarterly marks can be managed, redemptions can be gated, and defaults can be "
        "extended via amendments. The observable signals (HY OAS, BKLN, BDC prices) lead the "
        "private marks by 1–2 quarters, making them the only real-time window into this $2T+ market."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PDF DOWNLOAD
    # ══════════════════════════════════════════════════════════════════════════
    _section_header("↓", "Download This Report", "Institutional-format PDF with all charts and narrative")
    st.markdown(
        f'<p style="{_F}font-size:0.68rem;color:#8890a1;margin:0 0 0.6rem">'
        f'Generates a formatted PDF brief of all seven sections above - '
        f'GDP, PMI, money flows, yields, credit spreads, valuations, and index performance.</p>',
        unsafe_allow_html=True,
    )

    if st.button("Generate PDF Report", type="primary", key="macro_pdf_btn"):
        with st.spinner("Building PDF…"):
            try:
                from src.reports.macro_pdf import build_macro_pdf

                # Reuse already-loaded data (cached); fall back to empty if FRED unavailable
                _yields  = _load_yields(fred_key, start, end)  if fred_key else pd.DataFrame()
                _spreads = _load_spreads(fred_key, start, end) if fred_key else pd.DataFrame()
                _macro   = _load_macro(fred_key, start, end)   if fred_key else {}
                _money   = _load_money(fred_key, start, end)   if fred_key else {}
                _val     = _load_valuations()
                _perf_start = (date.today() - timedelta(days=400)).isoformat()
                _idx    = _load_index_perf(_perf_start)

                # Build narratives for PDF executive summary
                _narrative = "\n\n".join(filter(None, [
                    _narrate_growth(_macro),
                    _narrate_liquidity(_money, _yields),
                    _narrate_credit(_spreads),
                    _narrate_valuations(_val, _yields),
                    _narrate_cross_asset(_macro, _spreads, _val, _idx),
                ]))

                pdf_bytes = build_macro_pdf(
                    yields_df=_yields,
                    spreads_df=_spreads,
                    macro_data=_macro,
                    money_data=_money,
                    val_df=_val,
                    idx_prices=_idx,
                    start=start,
                    end=end,
                    narrative=_narrative,
                )
                fname = f"macro_brief_{date.today().strftime('%Y%m%d')}.pdf"
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name=fname,
                    mime="application/pdf",
                    key="macro_pdf_dl",
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

    _page_footer()
