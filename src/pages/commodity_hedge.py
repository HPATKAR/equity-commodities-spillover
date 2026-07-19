"""
Commodity Hedge Desk - hedge a PHYSICAL commodity exposure in your own units.

The Book Risk Character hedge overlay (Trade Ideas / Portfolio X-Ray) neutralises
an equity book with an equity-ETF basket. That is useless to a fuel buyer, a grain
merchant or a metals treasury, who think in barrels / bushels / tonnes and hedge in
futures, not ETFs. This desk takes a physical exposure in native units and returns:

  1. A hedge ticket in the real listed future (contract, direction, contract count).
  2. The roll carry, read off the ACTUAL forward curve (contango vs backwardation).
  3. A geopolitical hedge-ratio overlay driven by the terminal's own CIS engine -
     the differentiator: hedge more of the exposure when the conflicts that transmit
     to this commodity are hot, less when they are quiet.

Everything is measured on real futures data. The honest caveats (basis, margin,
parametric VaR) are disclosed on the panel, not buried.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd
import streamlit as st

from src.ui.shared import (_page_header, _page_footer, _page_intro,
                           _definition_block)
from src.data.config import COMMODITY_TICKERS

_M = "font-family:'JetBrains Mono',monospace;"
_GOLD, _GRN, _ORG, _RED = "#CFB991", "#27ae60", "#e67e22", "#c0392b"
_TXT, _MUT, _FNT, _BRD = "#e8e9ed", "#8890a1", "#555960", "#1e1e1e"
_MC = "FGHJKMNQUVXZ"                       # CME month codes, Jan..Dec

# root, Yahoo exchange suffix, contract size, physical unit, price multiplier
# (0.01 = quoted in cents/unit), and the complex's valid delivery-month letters.
_SPECS: dict[str, dict] = {
    "WTI Crude Oil":   dict(root="CL",  suf="NYM", size=1000,   unit="barrels",     u1="barrel",   qmult=1.0,  months="FGHJKMNQUVXZ"),
    "Brent Crude":     dict(root="BZ",  suf="NYM", size=1000,   unit="barrels",     u1="barrel",   qmult=1.0,  months="FGHJKMNQUVXZ"),
    "Natural Gas":     dict(root="NG",  suf="NYM", size=10000,  unit="MMBtu",       u1="MMBtu",    qmult=1.0,  months="FGHJKMNQUVXZ"),
    "Gasoline (RBOB)": dict(root="RB",  suf="NYM", size=42000,  unit="gallons",     u1="gallon",   qmult=1.0,  months="FGHJKMNQUVXZ"),
    "Heating Oil":     dict(root="HO",  suf="NYM", size=42000,  unit="gallons",     u1="gallon",   qmult=1.0,  months="FGHJKMNQUVXZ"),
    "Gold":            dict(root="GC",  suf="CMX", size=100,    unit="troy ounces", u1="troy oz",  qmult=1.0,  months="GJMQVZ"),
    "Silver":          dict(root="SI",  suf="CMX", size=5000,   unit="troy ounces", u1="troy oz",  qmult=1.0,  months="FHKNUZ"),
    "Platinum":        dict(root="PL",  suf="NYM", size=50,     unit="troy ounces", u1="troy oz",  qmult=1.0,  months="FJNV"),
    "Copper":          dict(root="HG",  suf="CMX", size=25000,  unit="pounds",      u1="pound",    qmult=1.0,  months="HKNUZ"),
    "Aluminum":        dict(root="ALI", suf="CMX", size=25,     unit="metric tons", u1="tonne",    qmult=1.0,  months="FGHJKMNQUVXZ"),
    "Wheat":           dict(root="ZW",  suf="CBT", size=5000,   unit="bushels",     u1="bushel",   qmult=0.01, months="HKNUZ"),
    "Corn":            dict(root="ZC",  suf="CBT", size=5000,   unit="bushels",     u1="bushel",   qmult=0.01, months="HKNUZ"),
    "Soybeans":        dict(root="ZS",  suf="CBT", size=5000,   unit="bushels",     u1="bushel",   qmult=0.01, months="FHKNQUX"),
    "Sugar #11":       dict(root="SB",  suf="NYB", size=112000, unit="pounds",      u1="pound",    qmult=0.01, months="HKNV"),
    "Coffee":          dict(root="KC",  suf="NYB", size=37500,  unit="pounds",      u1="pound",    qmult=0.01, months="HKNUZ"),
    "Cotton":          dict(root="CT",  suf="NYB", size=50000,  unit="pounds",      u1="pound",    qmult=0.01, months="HKNVZ"),
}


# ── Forward curve (real dated contracts) ──────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def _curve(name: str, asof: str) -> dict:
    """Front (continuous) + up to 4 dated deferred prices, in USD per physical unit.
    Returns {front, points:[(months_ahead, price)], slope_ann, ok, tickers}.
    slope_ann > 0 is contango (deferred richer), < 0 backwardation."""
    import yfinance as yf
    sp = _SPECS[name]
    front_t = COMMODITY_TICKERS[name]
    today = _dt.date.today()
    base = today.year * 12 + (today.month - 1)
    cands: list[tuple[int, str]] = []
    for k in range(2, 12):                       # skip the ~expiring front month
        idx = base + k
        y, m = divmod(idx, 12)
        letter = _MC[m]
        if letter in sp["months"]:
            cands.append((k, f'{sp["root"]}{letter}{y % 100:02d}.{sp["suf"]}'))
        if len(cands) >= 4:
            break
    tickers = [front_t] + [c for _, c in cands]
    try:
        raw = yf.download(tickers, period="8d", progress=False,
                          auto_adjust=False)["Close"]
    except Exception:
        return {"ok": False, "front": None, "points": [], "slope_ann": 0.0,
                "tickers": tickers}
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])

    def _last(t):
        try:
            s = raw[t].dropna()
            return float(s.iloc[-1]) if len(s) else None
        except Exception:
            return None

    fr = _last(front_t)
    if fr is None:
        return {"ok": False, "front": None, "points": [], "slope_ann": 0.0,
                "tickers": tickers}
    fr *= sp["qmult"]
    pts: list[tuple[int, float]] = []
    for k, c in cands:
        p = _last(c)
        if p is not None and p > 0:
            pts.append((k, p * sp["qmult"]))
    if not pts:
        return {"ok": False, "front": fr, "points": [], "slope_ann": 0.0,
                "tickers": tickers}
    k_far, p_far = pts[-1]
    slope_ann = (p_far / fr) ** (12.0 / k_far) - 1.0
    return {"ok": True, "front": fr, "points": pts, "slope_ann": float(slope_ann),
            "tickers": tickers}


@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def _front_vol(name: str, end: str) -> float:
    """Annualised vol of the front continuous contract, ~3y of daily returns."""
    try:
        from src.data.loader import _yf_download
        t = COMMODITY_TICKERS[name]
        s = str(_dt.date.today() - _dt.timedelta(days=3 * 365))
        raw = _yf_download([t], start=s, end=end, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return 0.0
        close = raw["Close"] if "Close" in raw.columns else raw
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        r = np.log(close / close.shift(1)).dropna()
        return float(r.std() * np.sqrt(252)) if len(r) > 30 else 0.0
    except Exception:
        return 0.0


def _gsi(name: str):
    """Geopolitical Stress Index for this commodity in [0,1], its band, and the
    driving conflicts. Combines each active conflict's transmission relevance to
    the commodity with its live CIS via a noisy-OR union. Degrades to (None,...)
    if the conflict engine is unavailable."""
    try:
        from src.analysis.conflict_model import (score_all_conflicts,
                                                  conflict_commodity_matrix)
        scores = score_all_conflicts()
        mat = conflict_commodity_matrix()
    except Exception:
        return None, "N/A", []
    drivers = []
    prod = 1.0
    for cid, row in mat.items():
        sc = scores.get(cid, {})
        if sc.get("state", "active") != "active":
            continue
        rel = float(row.get(name, 0.0))
        cis = float(sc.get("cis", 0.0)) / 100.0
        c = rel * cis
        if c > 0:
            drivers.append((sc.get("name", cid), cis * 100.0, rel, c))
            prod *= (1.0 - min(c, 0.999))
    gsi = 1.0 - prod
    drivers.sort(key=lambda x: x[3], reverse=True)
    band = "ELEVATED" if gsi >= 0.60 else "MODERATE" if gsi >= 0.30 else "LOW"
    return gsi, band, drivers[:4]


# ── Small HTML helpers (match the Book Risk Character grammar) ─────────────────
def _stat(lbl, val, sub, col=_TXT, last=False):
    br = "" if last else f"border-right:1px solid {_BRD}"
    return (f'<div style="flex:1;padding:.45rem .7rem;{br}">'
            f'<div style="{_M}font-size:.5rem;letter-spacing:.1em;color:{_MUT}">{lbl}</div>'
            f'<div style="{_M}font-size:1.05rem;font-weight:700;color:{col};margin:1px 0">{val}</div>'
            f'<div style="{_M}font-size:.48rem;color:{_FNT}">{sub}</div></div>')


def _panel(title, meta_html, body_html, foot_html=""):
    foot = (f'<div style="padding:.3rem .8rem;border-top:1px solid {_BRD};{_M}'
            f'font-size:.48rem;color:{_FNT};line-height:1.5">{foot_html}</div>'
            if foot_html else "")
    return st.markdown(
        f'<div style="border:1px solid {_BRD};background:#0a0a0a;margin:.2rem 0 .8rem">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
        f'padding:.4rem .8rem;border-bottom:1px solid {_BRD}">'
        f'<span style="{_M}font-size:.6rem;font-weight:700;letter-spacing:.14em;'
        f'color:{_TXT}">{title}</span>'
        f'<span style="{_M}font-size:.5rem;color:{_MUT}">{meta_html}</span></div>'
        f'{body_html}{foot}</div>', unsafe_allow_html=True)


def _money(x: float) -> str:
    a = abs(x)
    if a >= 1e9:  return f'${x/1e9:.2f}B'
    if a >= 1e6:  return f'${x/1e6:.2f}M'
    if a >= 1e3:  return f'${x/1e3:.1f}K'
    return f'${x:,.0f}'


def page_commodity_hedge(start: str, end: str, fred_key: str = "") -> None:
    _page_header(
        "Commodity Hedge Desk",
        "Hedge a physical commodity exposure in your own units, geopolitics-scaled")
    _page_intro(
        "Enter a physical exposure the way a hedger actually carries it - barrels of "
        "jet fuel, bushels of wheat, tonnes of copper - and the terminal returns a "
        "hedge ticket in the real listed future: which contract, long or short, and "
        "how many. It reads the <strong>roll carry</strong> off the actual forward "
        "curve (contango costs a long, backwardation pays it), and it scales the "
        "recommended <strong>hedge ratio</strong> by the terminal's own geopolitical "
        "stress engine, so you hedge more when the conflicts that transmit to your "
        "commodity are hot. This is the equity hedge overlay's commodity-native "
        "sibling: futures and physical units, not an ETF basket."
    )
    _definition_block(
        "How to read it",
        "A <b>consumer</b> (you buy the commodity, an airline buying fuel, a mill "
        "buying wheat) is hurt by rising prices and hedges <b>long</b> futures. A "
        "<b>producer</b> (you sell it, a driller, a farmer) is hurt by falling prices "
        "and hedges <b>short</b>. VaR is parametric (normal, real vol). The hedge is "
        "the exchange future, so it neutralises price risk but not grade, location or "
        "timing basis, and it carries margin and roll. Illustrative, not advice."
    )

    # ── Inputs ───────────────────────────────────────────────────────────────
    names = list(_SPECS.keys())
    c1, c2, c3, c4 = st.columns([1.5, 1.3, 1.2, 1.0], gap="medium")
    with c1:
        name = st.selectbox("Commodity", names, index=0, key="_chd_name")
    sp = _SPECS[name]
    with c2:
        role = st.radio("Your exposure", ["Consumer (I buy it)",
                                          "Producer (I sell it)"],
                        key="_chd_role", horizontal=False)
        is_consumer = role.startswith("Consumer")
    with c3:
        qty = st.number_input(f"Quantity ({sp['unit']})", min_value=0.0,
                              value=float(500 * sp["size"]), step=float(sp["size"]),
                              key="_chd_qty",
                              help=f"Physical size you need to hedge, in {sp['unit']}.")
    with c4:
        horizon = st.slider("Horizon (months)", 1, 24, 6, key="_chd_h")

    if qty <= 0:
        st.info("Enter a positive quantity to size the hedge.")
        _page_footer()
        return

    with st.spinner("Pricing the curve and scoring geopolitical stress..."):
        cv = _curve(name, end)
        sig = _front_vol(name, end)
        gsi, band, drivers = _gsi(name)

    if not cv.get("front"):
        st.error(f"Could not price {name} right now (data feed). Try another "
                 "commodity or refresh in a moment.")
        _page_footer()
        return

    px = cv["front"]                                    # USD per physical unit
    notional = qty * px
    contract_notl = sp["size"] * px
    n_full = qty / sp["size"]

    # Horizon 95% parametric VaR on the unhedged physical.
    sig_h = sig * np.sqrt(horizon / 12.0)
    var_un = 1.645 * sig_h * notional

    # Geopolitical hedge ratio: 50% baseline + up to 50% geopolitical overlay.
    g = gsi if gsi is not None else 0.0
    h_ratio = float(np.clip(0.50 + 0.50 * g, 0.50, 1.0))
    n_hedge = round(h_ratio * qty / sp["size"])
    hedge_notl = n_hedge * contract_notl
    side = "LONG" if is_consumer else "SHORT"
    side_col = _GRN if is_consumer else _RED

    # Residual VaR after the hedge (8% residual basis on the hedged slice).
    _basis = 0.08
    hedged_frac = (1.0 - h_ratio) + h_ratio * _basis
    var_hed = var_un * hedged_frac
    var_cut = 1.0 - hedged_frac

    # Roll carry to the hedger: a long earns backwardation / pays contango; a
    # short is the mirror. slope_ann > 0 = contango.
    slope = cv.get("slope_ann", 0.0)
    carry_hedger = (-slope) if is_consumer else (+slope)
    carry_dollars = carry_hedger * hedge_notl
    if abs(slope) < 0.005:
        shape, shape_col = "FLAT", _MUT
    elif slope > 0:
        shape, shape_col = "CONTANGO", _ORG
    else:
        shape, shape_col = "BACKWARDATION", _GRN

    # ── EXPOSURE ─────────────────────────────────────────────────────────────
    exp_stats = (
        f'<div style="display:flex;border-bottom:1px solid {_BRD}">'
        + _stat("PHYSICAL EXPOSURE", f'{qty:,.0f}',
                f'{sp["unit"]} @ ${px:,.2f}/{sp["u1"]}')
        + _stat("NOTIONAL VALUE", _money(notional),
                f'{n_full:,.1f} contracts of {sp["size"]:,}')
        + _stat("ANNUAL VOL", f'{sig*100:.0f}%',
                'front contract, 3y realised')
        + _stat(f'{horizon}-MO 95% VaR', _money(var_un),
                f'{var_un/notional*100:.0f}% of notional, unhedged', _RED, last=True)
        + '</div>')
    _panel("EXPOSURE",
           f'{name} · {"consumer, price-rise risk" if is_consumer else "producer, price-fall risk"}',
           exp_stats,
           "Notional marks the physical at the front futures price. VaR is a "
           f'parametric 95% {horizon}-month move on real return vol, before any hedge.')

    # ── GEOPOLITICAL STRESS (the differentiator) ─────────────────────────────
    if gsi is not None:
        g_col = _RED if band == "ELEVATED" else _ORG if band == "MODERATE" else _GRN
        rows = ""
        for nm, cis, rel, _c in drivers:
            rows += (
                f'<div style="display:flex;align-items:center;gap:8px;padding:2px 0">'
                f'<span style="{_M}font-size:.56rem;color:{_TXT};min-width:150px;'
                f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{nm}</span>'
                f'<span style="{_M}font-size:.5rem;color:{_MUT};min-width:60px">CIS {cis:.0f}</span>'
                f'<div style="flex:1;height:7px;background:#141414"><div style="width:{rel*100:.0f}%;'
                f'height:7px;background:{_GOLD}"></div></div>'
                f'<span style="{_M}font-size:.52rem;color:{_TXT};min-width:74px;text-align:right">'
                f'{rel*100:.0f}% transmit</span></div>')
        gbar = (
            f'<div style="padding:.5rem .8rem;border-bottom:1px solid {_BRD}">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline">'
            f'<span style="{_M}font-size:.5rem;letter-spacing:.1em;color:{_MUT}">'
            f'GEOPOLITICAL STRESS INDEX (this commodity)</span>'
            f'<span style="{_M}font-size:.9rem;font-weight:700;color:{g_col}">{g*100:.0f}'
            f'<span style="font-size:.5rem;color:{_MUT}">/100</span>'
            f'<span style="background:{g_col};color:#000;font-weight:700;padding:1px 7px;'
            f'margin-left:8px;font-size:.5rem;letter-spacing:.08em">{band}</span></span></div>'
            f'<div style="height:9px;background:#141414;margin-top:5px"><div style="width:{g*100:.0f}%;'
            f'height:9px;background:{g_col}"></div></div></div>')
        if drivers:
            read = (
                f'The index unions each active conflict\'s transmission relevance to '
                f'<b>{name}</b> with its live CIS. It reads <b style="color:{g_col}">{band}</b> '
                f'at <b style="color:{g_col}">{g*100:.0f}</b>, which lifts the recommended hedge '
                f'ratio to <b style="color:{_TXT}">{h_ratio*100:.0f}%</b> of the exposure '
                f'(50% baseline + {(h_ratio-0.5)*100:.0f}% geopolitical overlay). Quiet the '
                f'conflicts and the ratio falls back toward the baseline.')
        else:
            read = (
                f'No active conflict transmits to <b>{name}</b> on the supply side, so the '
                f'index is <b style="color:{g_col}">{band}</b> and the hedge holds at the '
                f'<b style="color:{_TXT}">{h_ratio*100:.0f}%</b> baseline. Note this measures '
                f'supply-disruption stress; a safe-haven like gold is conflict-supported on '
                f'the <i>demand</i> side, so a buyer\'s downside risk, if anything, eases when '
                f'conflict runs hot. Size to your own view of that.')
        body = (gbar +
                f'<div style="display:flex;gap:16px;padding:.55rem .8rem;flex-wrap:wrap">'
                f'<div style="flex:1.2;min-width:320px">'
                f'<div style="{_M}font-size:.5rem;letter-spacing:.1em;color:{_MUT};'
                f'margin-bottom:3px">DRIVING CONFLICTS · relevance to {name}</div>{rows}</div>'
                f'<div style="flex:1;min-width:230px;{_M}font-size:.56rem;color:#c9ccd4;'
                f'line-height:1.55">{read}</div></div>')
        _panel("GEOPOLITICAL OVERLAY",
               'transmission &times; CIS &middot; noisy-OR'
               f'<span style="background:{_GOLD};color:#000;font-weight:700;padding:1px 7px;'
               f'margin-left:6px;letter-spacing:.08em">HEDGE-RATIO DRIVER</span>',
               body,
               "CIS is the terminal's Conflict Intensity Score (ACLED / GDELT where "
               "live, else scenario). Transmission relevance is the conflict's oil_gas "
               "/ metals / agriculture channel weight mapped to this commodity. The "
               "hedge-ratio rule (50% + 50%&times;GSI) is a house convention, not a "
               "market-implied optimum.")
    else:
        st.caption("Geopolitical engine unavailable this run; hedge ratio defaults to "
                   "the 50% baseline.")

    # ── HEDGE TICKET ─────────────────────────────────────────────────────────
    tk_stats = (
        f'<div style="display:flex;border-bottom:1px solid {_BRD}">'
        + _stat("THE HEDGE", f'<span style="color:{side_col}">{side} {n_hedge:,}</span>',
                f'{COMMODITY_TICKERS[name]} futures')
        + _stat("HEDGE RATIO", f'{h_ratio*100:.0f}%',
                f'{n_hedge:,} of {n_full:,.1f} contracts')
        + _stat("HEDGE NOTIONAL", _money(hedge_notl),
                f'{_money(contract_notl)} per contract')
        + _stat(f'{horizon}-MO VaR HEDGED', _money(var_hed),
                f'down {var_cut*100:.0f}% from {_money(var_un)}', _GRN, last=True)
        + '</div>')
    read_tk = (
        f'{"You buy the commodity, so a price rise hurts you; a " if is_consumer else "You sell the commodity, so a price fall hurts you; a "}'
        f'<b style="color:{side_col}">{side}</b> futures position gains exactly when your '
        f'physical hurts. Sizing <b>{n_hedge:,}</b> contracts covers <b>{h_ratio*100:.0f}%</b> '
        f'of the {qty:,.0f} {sp["unit"]}, cutting the {horizon}-month 95% VaR from '
        f'<b style="color:{_RED}">{_money(var_un)}</b> to <b style="color:{_GRN}">{_money(var_hed)}</b>. '
        f'The residual is the un-hedged {(1-h_ratio)*100:.0f}% plus an assumed {_basis*100:.0f}% '
        f'basis on the hedged slice.')
    _panel("HEDGE TICKET",
           f'{side} {n_hedge:,} &times; {COMMODITY_TICKERS[name]} '
           f'({sp["size"]:,} {sp["unit"]}/contract)',
           tk_stats +
           f'<div style="padding:.55rem .8rem;{_M}font-size:.56rem;color:#c9ccd4;'
           f'line-height:1.55">{read_tk}</div>',
           f'Contract = {sp["size"]:,} {sp["unit"]}. Contract count is rounded to whole '
           f'lots. VaR reduction assumes the future tracks your physical near 1:1; real '
           f'basis (grade, delivery point, timing) is not modelled.')

    # ── ROLL CARRY ───────────────────────────────────────────────────────────
    if cv.get("ok") and cv.get("points"):
        c_col = _GRN if carry_hedger > 0 else _RED
        verb = "EARNS" if carry_hedger > 0 else "COSTS"
        # mini curve: front + deferreds as % vs front
        bars = (f'<div style="display:flex;align-items:center;gap:8px;padding:2px 0">'
                f'<span style="{_M}font-size:.56rem;font-weight:700;color:{_TXT};min-width:70px">FRONT</span>'
                f'<span style="{_M}font-size:.52rem;color:{_MUT};min-width:70px">${px:,.2f}</span>'
                f'<div style="flex:1;height:7px;background:#141414"></div>'
                f'<span style="{_M}font-size:.52rem;color:{_MUT};min-width:52px;text-align:right">0.0%</span></div>')
        _mx = max((abs(p / px - 1) for _, p in cv["points"]), default=0.01) or 0.01
        for k, p in cv["points"]:
            d = p / px - 1.0
            w = abs(d) / _mx * 100
            bc = _ORG if d > 0 else _GRN
            bars += (f'<div style="display:flex;align-items:center;gap:8px;padding:2px 0">'
                     f'<span style="{_M}font-size:.56rem;color:{_TXT};min-width:70px">+{k}mo</span>'
                     f'<span style="{_M}font-size:.52rem;color:{_MUT};min-width:70px">${p:,.2f}</span>'
                     f'<div style="flex:1;height:7px;background:#141414"><div style="width:{w:.0f}%;'
                     f'height:7px;background:{bc}"></div></div>'
                     f'<span style="{_M}font-size:.52rem;color:{bc};min-width:52px;text-align:right">'
                     f'{d*100:+.1f}%</span></div>')
        carry_stats = (
            f'<div style="display:flex;border-bottom:1px solid {_BRD}">'
            + _stat("CURVE SHAPE", f'<span style="color:{shape_col}">{shape}</span>',
                    f'{cv["points"][-1][0]}mo slope {slope*100:+.1f}% ann')
            + _stat("ROLL CARRY", f'<span style="color:{c_col}">{verb} {abs(carry_hedger)*100:.1f}%</span>'
                    '<span style="font-size:.5rem">/yr</span>',
                    f'on the {side.lower()} hedge')
            + _stat("CARRY $/YR", f'<span style="color:{c_col}">{"+" if carry_dollars>=0 else "-"}'
                    f'{_money(abs(carry_dollars))}</span>',
                    'on hedge notional', last=True)
            + '</div>')
        read_c = (
            f'The {name} curve is <b style="color:{shape_col}">{shape.lower()}</b> '
            f'({slope*100:+.1f}%/yr front-to-{cv["points"][-1][0]}mo). Rolling a '
            f'<b style="color:{side_col}">{side.lower()}</b> hedge through it '
            f'<b style="color:{c_col}">{verb.lower()}</b> about '
            f'<b style="color:{c_col}">{abs(carry_hedger)*100:.1f}%/yr</b>, roughly '
            f'<b style="color:{c_col}">{_money(abs(carry_dollars))}</b> on this hedge '
            f'notional. '
            + ("Backwardation pays a long consumer to hold the hedge; " if (is_consumer and slope < 0)
               else "Contango charges a long consumer to keep rolling; " if (is_consumer and slope > 0)
               else "Contango pays a short producer to hold the hedge; " if (not is_consumer and slope > 0)
               else "Backwardation charges a short producer to keep rolling; " if (not is_consumer and slope < 0)
               else "A flat curve means roll is near costless; ")
            + 'this is a real, ongoing part of the hedge cost, not a one-off.')
        _panel("ROLL CARRY",
               f'read off {len(cv["points"])+1} live contract points',
               carry_stats +
               f'<div style="display:flex;gap:16px;padding:.55rem .8rem;flex-wrap:wrap">'
               f'<div style="flex:1.1;min-width:300px">'
               f'<div style="{_M}font-size:.5rem;letter-spacing:.1em;color:{_MUT};'
               f'margin-bottom:3px">FORWARD CURVE · % vs front</div>{bars}</div>'
               f'<div style="flex:1;min-width:230px;{_M}font-size:.56rem;color:#c9ccd4;'
               f'line-height:1.55">{read_c}</div></div>',
               "Curve built from live dated contracts (" +
               ", ".join(cv["tickers"][1:]) + "). Annualised as a constant-roll "
               "approximation; the real roll depends on each contract's calendar.")
    else:
        st.caption(f"Forward curve for {name} is thin on the public feed right now, so "
                   "roll carry is not estimated this run. The hedge sizing above still holds.")

    st.caption("Commodity Hedge Desk - physical exposure hedged in listed futures, "
               "geopolitics-scaled. Illustrative, not investment advice.")
    _page_footer()
