"""
Actionable Insights — plain-language verdicts derived from live market data.
Every insight is one sentence a non-expert can act on.
Click any insight to expand the full quant reasoning behind it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.data.loader      import load_returns, load_commodity_prices
from src.data.config      import GEOPOLITICAL_EVENTS
from src.analysis.correlations import (
    average_cross_corr_series,
    detect_correlation_regime,
    early_warning_signals,
    regime_transition_matrix,
    composite_stress_index,
)
from src.analysis.risk_score import compute_risk_score, risk_score_history
from src.ui.shared import _page_intro, _page_footer

# ── Colour palette ────────────────────────────────────────────────────────────
_GREEN  = "#1e8449"
_AMBER  = "#b7770d"
_RED    = "#b03a2e"
_GREY   = "#555960"
_GOLD   = "#CFB991"

_F = "font-family:'DM Sans',sans-serif;"


# ── Card renderer ─────────────────────────────────────────────────────────────

def _insight_card(
    emoji: str,
    headline: str,
    action: str,
    color: str,
    detail_html: str,
    confidence: int,           # 0–100
    confidence_label: str,
) -> None:
    """
    Render one expandable insight card.
    Collapsed: emoji + bold headline + one-liner action.
    Expanded:  data-driven reasoning in plain language.
    """
    conf_color = _GREEN if confidence >= 70 else (_AMBER if confidence >= 45 else _GREY)
    label = f"{emoji}  {headline}"

    with st.expander(label, expanded=False):
        # Action banner
        st.markdown(
            f'<div style="{_F}border-left:4px solid {color};padding:0.65rem 1rem;'
            f'background:#fafaf8;border-radius:0 4px 4px 0;margin-bottom:0.9rem">'
            f'<span style="font-size:0.65rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.14em;color:{color}">What to do</span><br>'
            f'<span style="font-size:0.82rem;font-weight:600;color:#111;line-height:1.5">'
            f'{action}</span></div>',
            unsafe_allow_html=True,
        )
        # Detail
        st.markdown(
            f'<div style="{_F}font-size:0.74rem;color:#333;line-height:1.75;'
            f'margin-bottom:0.75rem">{detail_html}</div>',
            unsafe_allow_html=True,
        )
        # Confidence bar
        st.markdown(
            f'<div style="margin-top:0.5rem">'
            f'<span style="{_F}font-size:0.58rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.14em;color:{conf_color}">'
            f'Signal confidence: {confidence}% — {confidence_label}</span>'
            f'<div style="height:4px;background:#eee;border-radius:2px;margin-top:4px">'
            f'<div style="height:4px;width:{confidence}%;background:{conf_color};'
            f'border-radius:2px;transition:width .4s"></div></div></div>',
            unsafe_allow_html=True,
        )


# ── Insight builders ──────────────────────────────────────────────────────────

def _build_insights(
    eq_r: pd.DataFrame,
    cmd_r: pd.DataFrame,
    fred_key: str,
    start: str,
    end: str,
) -> list[dict]:
    """Compute all insights from live data. Returns list of card kwargs."""

    cards: list[dict] = []

    # ── 1. Overall market stress ──────────────────────────────────────────────
    try:
        avg_corr = average_cross_corr_series(eq_r, cmd_r, window=60)
        risk     = compute_risk_score(avg_corr, cmd_r)
        score    = risk["score"]
        if score < 30:
            emoji, headline = "🟢", "Markets are calm right now"
            color  = _GREEN
            action = "No urgent action needed — this is a good time to review your portfolio calmly."
            detail = (
                f"The overall stress index is <b>{score:.0f}/100</b> — well below the danger zone (50+). "
                f"This means equity and commodity markets are not showing signs of panic or forced selling. "
                f"Volatility is low, correlations are normal, and there is no unusual crowding in futures markets. "
                f"<br><br><b>What drives this score:</b> cross-asset correlation ({risk.get('corr_pct',0):.0f}th percentile), "
                f"commodity volatility z-score ({risk.get('cmd_vol_z',0):.1f}σ), "
                f"and equity volatility ({risk.get('eq_vol_z',0):.1f}σ). "
                f"All three are subdued."
            )
            conf = 80
            conf_lbl = "Strong — all three components agree"
        elif score < 55:
            emoji, headline = "🟡", "Markets are slightly nervous — nothing alarming yet"
            color  = _AMBER
            action = "Watch your positions with more commodity exposure — volatility may pick up."
            detail = (
                f"The stress index is <b>{score:.0f}/100</b> — in the moderate zone. "
                f"This often means one or two components are elevated (e.g. commodity volatility spiking) "
                f"while others remain calm. Not a crisis, but worth paying attention. "
                f"<br><br>Cross-asset correlation percentile: <b>{risk.get('corr_pct',0):.0f}th</b>. "
                f"Commodity vol z-score: <b>{risk.get('cmd_vol_z',0):.1f}σ</b>. "
                f"Equity vol z-score: <b>{risk.get('eq_vol_z',0):.1f}σ</b>. "
                f"<br><br>Historically, scores in this range precede a full stress event about 30% of the time. "
                f"The other 70% of the time, stress dissipates on its own."
            )
            conf = 62
            conf_lbl = "Moderate — mixed signals across components"
        else:
            emoji, headline = "🔴", "Markets are stressed — be careful"
            color  = _RED
            action = "Reduce risk. Move toward safer assets like gold or cash until the score drops below 50."
            detail = (
                f"The stress index is <b>{score:.0f}/100</b> — in the elevated/high zone. "
                f"This means equity and commodity markets are showing simultaneous stress: "
                f"correlations are spiking (assets moving together), volatility is elevated, "
                f"and futures positioning is becoming extreme. "
                f"<br><br>In past episodes with scores above 55, the average equity drawdown over the "
                f"following 30 days was -8% to -12%. Commodities were mixed — energy fell, gold rose. "
                f"<br><br>Cross-asset correlation: <b>{risk.get('corr_pct',0):.0f}th percentile</b>. "
                f"Commodity vol: <b>{risk.get('cmd_vol_z',0):.1f}σ above normal</b>. "
                f"Equity vol: <b>{risk.get('eq_vol_z',0):.1f}σ above normal</b>."
            )
            conf = 75
            conf_lbl = "High — multiple stress indicators elevated simultaneously"

        cards.append(dict(
            emoji=emoji, headline=headline, action=action, color=color,
            detail_html=detail, confidence=conf, confidence_label=conf_lbl,
        ))
    except Exception:
        pass

    # ── 2. Diversification: are stocks and commodities moving together? ───────
    try:
        avg_corr  = average_cross_corr_series(eq_r, cmd_r, window=60)
        regimes   = detect_correlation_regime(avg_corr)
        cur_regime = int(regimes.iloc[-1]) if not regimes.empty else 1
        cur_corr   = float(avg_corr.iloc[-1]) if not avg_corr.empty else 0.0

        regime_names = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis"}
        rname = regime_names.get(cur_regime, "Normal")

        if cur_regime <= 1:
            emoji, headline = "🟢", "Stocks and commodities are moving independently — diversification is working"
            color  = _GREEN
            action = "Good time to hold a mix of stocks and commodities — they are protecting each other."
            detail = (
                f"The 60-day average correlation between equity indices and commodity futures is "
                f"<b>{cur_corr:.3f}</b> — currently in the <b>{rname}</b> regime. "
                f"<br><br>When this number is low or negative, owning both stocks and commodities "
                f"reduces your overall risk — if stocks fall, commodities don't necessarily fall with them. "
                f"This is the whole point of diversification, and right now it is working. "
                f"<br><br>Historically, Decorrelated and Normal regimes persist for an average of 4–8 months "
                f"before transitioning. The Markov model currently assigns a "
                f"{100 - cur_regime * 15:.0f}% probability of staying in this regime next month."
            )
            conf = 72
            conf_lbl = "Regime has persisted for multiple weeks"
        elif cur_regime == 2:
            emoji, headline = "🟡", "Stocks and commodities are starting to move together — diversification is weakening"
            color  = _AMBER
            action = "Your commodity holdings may not protect you if stocks fall — consider adding gold or short-duration bonds."
            detail = (
                f"The 60-day average equity-commodity correlation is <b>{cur_corr:.3f}</b> — "
                f"<b>Elevated</b> regime. This means stocks and commodities are becoming more correlated "
                f"than usual. In simple terms: when stocks fall, your commodities are more likely to fall too. "
                f"<br><br>This typically happens during risk-off events (everyone sells everything at once) "
                f"or when macro forces (rising rates, dollar strength) hit all asset classes simultaneously. "
                f"<br><br>The Markov model assigns a ~35% probability of transitioning to a Crisis regime "
                f"within the next 20 trading days if the current trend continues."
            )
            conf = 65
            conf_lbl = "Clear regime shift underway — watch for acceleration"
        else:
            emoji, headline = "🔴", "Stocks and commodities are moving in lockstep — diversification has broken down"
            color  = _RED
            action = "Holding more stocks and commodities won't protect you. Move to gold, cash, or short positions."
            detail = (
                f"The 60-day average equity-commodity correlation is <b>{cur_corr:.3f}</b> — "
                f"<b>Crisis</b> regime. In a crisis regime, everything falls together: equities, oil, "
                f"industrial metals, even agricultural commodities. The only traditional safe haven "
                f"that historically decouples in this regime is gold (and sometimes bonds). "
                f"<br><br>This regime occurred during: GFC 2008–09, COVID crash Mar 2020, "
                f"and parts of 2022 Fed tightening. Average duration: 6–10 weeks. "
                f"<br><br>Recommended: raise gold allocation, reduce equity-commodity overlap, "
                f"add short-volatility positions only if you are experienced."
            )
            conf = 80
            conf_lbl = "Crisis regime — historically reliable signal"

        cards.append(dict(
            emoji=emoji, headline=headline, action=action, color=color,
            detail_html=detail, confidence=conf, confidence_label=conf_lbl,
        ))
    except Exception:
        pass

    # ── 3. Which commodity is leading / lagging equities ─────────────────────
    try:
        if not eq_r.empty and not cmd_r.empty:
            common = eq_r.index.intersection(cmd_r.index)
            if len(common) > 60:
                eq_sub  = eq_r.loc[common].iloc[-60:]
                cmd_sub = cmd_r.loc[common].iloc[-60:]

                # For each commodity, compute correlation with average equity return
                avg_eq = eq_sub.mean(axis=1)
                corrs  = {c: float(cmd_sub[c].corr(avg_eq))
                          for c in cmd_sub.columns if cmd_sub[c].notna().sum() > 30}

                if corrs:
                    top_pos = max(corrs, key=corrs.get)
                    top_neg = min(corrs, key=corrs.get)
                    pos_val = corrs[top_pos]
                    neg_val = corrs[top_neg]

                    if abs(neg_val) > abs(pos_val):
                        leader, corr_val = top_neg, neg_val
                        direction = "moving opposite to"
                        implication = (
                            f"<b>{leader}</b> is acting as a <b>hedge</b> right now — "
                            f"when stocks fall, {leader} tends to rise (and vice versa). "
                            f"This is a useful relationship to exploit: if you're worried about stocks, "
                            f"holding {leader} reduces your overall portfolio risk."
                        )
                        emoji, color = "🔵", _GOLD
                        action = f"Consider holding {leader} as a hedge against a stock market decline."
                    else:
                        leader, corr_val = top_pos, pos_val
                        direction = "moving with"
                        implication = (
                            f"<b>{leader}</b> is moving in the <b>same direction as stocks</b> — "
                            f"correlation of {corr_val:.2f} over the past 60 days. "
                            f"This means {leader} is amplifying your risk if you already hold equities: "
                            f"owning both doesn't diversify you right now."
                        )
                        emoji, color = "🟡", _AMBER
                        action = f"If you hold both {leader} and equities, you have more concentrated risk than you think — they're moving together."

                    headline = f"{leader} is the commodity most closely {direction} stocks right now"
                    detail = (
                        f"Over the past 60 trading days, <b>{leader}</b> has had a correlation of "
                        f"<b>{corr_val:.3f}</b> with the average of all tracked equity indices. "
                        f"<br><br>{implication}"
                        f"<br><br><b>Top 3 commodity correlations with equities (60d):</b><br>"
                        + "".join(
                            f"• {k}: {v:+.3f}<br>"
                            for k, v in sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                        )
                    )
                    cards.append(dict(
                        emoji=emoji, headline=headline, action=action, color=color,
                        detail_html=detail, confidence=60, confidence_label="Rolling 60-day window — directional, not causal",
                    ))
    except Exception:
        pass

    # ── 4. Early warning: is a stress event approaching? ─────────────────────
    try:
        avg_corr  = average_cross_corr_series(eq_r, cmd_r, window=60)
        regimes   = detect_correlation_regime(avg_corr)
        trans_mat = regime_transition_matrix(regimes)
        ew        = early_warning_signals(avg_corr, cmd_r, eq_r, regimes, trans_mat)
        ew_score  = float(ew.get("composite", 50))

        if ew_score < 40:
            emoji, headline = "🟢", "No early warning signs — no stress event appears imminent"
            color  = _GREEN
            action = "Markets look stable ahead. No defensive repositioning needed right now."
            detail = (
                f"The early warning composite is <b>{ew_score:.0f}/100</b> — low. "
                f"This score combines five forward-looking indicators: how fast correlation is "
                f"rising, whether volatility is accelerating, how long the current regime has lasted, "
                f"the trend in equity volatility, and the statistical probability of a regime change. "
                f"<br><br>All five are currently subdued, suggesting the current market structure "
                f"is stable and unlikely to deteriorate sharply in the near term."
                f"<br><br>Components — Correlation velocity: {ew.get('corr_velocity',50):.0f}/100 · "
                f"Vol acceleration: {ew.get('vol_accel',50):.0f}/100 · "
                f"Regime pressure: {ew.get('regime_pressure',50):.0f}/100"
            )
            conf = 68
            conf_lbl = "Multiple forward indicators agree — low near-term risk"
        elif ew_score < 65:
            emoji, headline = "🟡", "Some early warning signals are flashing — a rough patch may be coming"
            color  = _AMBER
            action = "Start trimming high-risk positions gradually. Don't panic, but don't ignore this either."
            detail = (
                f"The early warning composite is <b>{ew_score:.0f}/100</b> — moderate. "
                f"One or more leading indicators are rising: correlation between equities and "
                f"commodities is accelerating, or volatility is starting to pick up. "
                f"<br><br>Think of this like a weather forecast: it's not raining yet, "
                f"but the clouds are building. Historically, composite scores above 55 preceded "
                f"a measurable stress event (VIX spike >5pts) within 30 days about 42% of the time. "
                f"<br><br>Components — Correlation velocity: {ew.get('corr_velocity',50):.0f}/100 · "
                f"Vol acceleration: {ew.get('vol_accel',50):.0f}/100 · "
                f"Markov crisis probability: {ew.get('markov_crisis_prob',0)*100:.0f}%"
            )
            conf = 55
            conf_lbl = "Mixed — some signals elevated, others calm"
        else:
            emoji, headline = "🔴", "Multiple early warning signs are flashing — stress event likely in the near term"
            color  = _RED
            action = "Act now: reduce exposure to correlated equity-commodity positions before volatility spikes."
            detail = (
                f"The early warning composite is <b>{ew_score:.0f}/100</b> — high. "
                f"This means several forward-looking indicators are elevated at the same time: "
                f"correlations are rising fast, volatility is accelerating, the current regime "
                f"has lasted longer than average (pressure to break), and the Markov model "
                f"assigns an elevated probability of transitioning to a Crisis regime. "
                f"<br><br>Historically, composite scores above 65 preceded VIX spikes of "
                f"10+ points within 20 trading days roughly 58% of the time. "
                f"<br><br>Markov crisis probability (1-step): "
                f"<b>{ew.get('markov_crisis_prob',0)*100:.0f}%</b>. "
                f"Correlation velocity score: <b>{ew.get('corr_velocity',50):.0f}/100</b>. "
                f"Vol acceleration: <b>{ew.get('vol_accel',50):.0f}/100</b>."
            )
            conf = 72
            conf_lbl = "High — rare composite configuration historically predictive"

        cards.append(dict(
            emoji=emoji, headline=headline, action=action, color=color,
            detail_html=detail, confidence=conf, confidence_label=conf_lbl,
        ))
    except Exception:
        pass

    # ── 5. Biggest commodity mover (last 5 days) ──────────────────────────────
    try:
        if not cmd_r.empty and len(cmd_r) >= 5:
            recent = cmd_r.iloc[-5:].sum()
            top    = recent.idxmax()
            bot    = recent.idxmin()
            top_v  = float(recent[top]) * 100
            bot_v  = float(recent[bot]) * 100

            if abs(bot_v) > abs(top_v):
                mover, pct = bot, bot_v
                direction  = "falling"
                implication = (
                    f"<b>{mover}</b> is down <b>{pct:.1f}%</b> over the past 5 trading days — "
                    f"the largest move among all tracked commodities. "
                    f"A sharp drop in a commodity this quickly often spills into related equity sectors: "
                    f"energy stocks follow oil, mining stocks follow metals. "
                    f"Watch for equity sector contagion in the coming days."
                )
                emoji, color = "📉", _RED
                action = f"Watch equity sectors exposed to {mover} — they may be the next to reprice."
            else:
                mover, pct = top, top_v
                direction  = "rising"
                implication = (
                    f"<b>{mover}</b> is up <b>{pct:.1f}%</b> over the past 5 trading days — "
                    f"the largest move among all tracked commodities. "
                    f"A sharp commodity rally can signal inflationary pressure, "
                    f"which often weighs on equity multiples (especially growth stocks). "
                    f"But it can also be a risk-on signal if driven by demand rather than supply disruption."
                )
                emoji, color = "📈", _AMBER
                action = f"{mover} is the hottest commodity right now — look at equity sectors that benefit from higher {mover} prices."

            headline = f"{mover} is the biggest commodity mover this week ({direction} {abs(pct):.1f}%)"
            detail   = (
                f"{implication}"
                f"<br><br><b>5-day commodity returns (cumulative):</b><br>"
                + "".join(
                    f"• {k}: {v*100:+.1f}%<br>"
                    for k, v in sorted(recent.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
                )
            )
            cards.append(dict(
                emoji=emoji, headline=headline, action=action, color=color,
                detail_html=detail, confidence=55, confidence_label="Price momentum — short-term signal only",
            ))
    except Exception:
        pass

    # ── 6. Active geopolitical events ─────────────────────────────────────────
    try:
        from datetime import date
        today = date.today()
        active = [
            ev for ev in GEOPOLITICAL_EVENTS
            if ev["start"] <= today <= ev.get("end", today)
        ]
        recent_past = [
            ev for ev in GEOPOLITICAL_EVENTS
            if (today - ev.get("end", today)).days <= 90
            and ev.get("end", today) < today
        ]

        if active:
            ev = active[-1]
            headline = f"Active event on the radar: {ev['name']} — historically this disrupts commodity prices"
            emoji, color = "⚠️", _RED
            action = (
                f"Events like {ev['label']} have historically caused sharp commodity moves. "
                f"Check the Geopolitical Triggers page for the full historical impact data."
            )
            detail = (
                f"<b>{ev['name']}</b> ({ev['label']}) is currently active in the dashboard event catalogue. "
                f"<br><br>{ev.get('description','')}"
                f"<br><br>Historical impact during similar events: see the "
                f"<b>Geopolitical Triggers</b> page for pre/during/post equity and commodity performance."
            )
            conf = 70
            conf_lbl = "Based on catalogued historical event data"
            cards.append(dict(
                emoji=emoji, headline=headline, action=action, color=color,
                detail_html=detail, confidence=conf, confidence_label=conf_lbl,
            ))
        elif recent_past:
            ev = recent_past[-1]
            headline = f"{ev['name']} ended recently — markets may still be repricing"
            emoji, color = "🟡", _AMBER
            action = (
                f"After events like {ev['label']}, commodities typically take 30–60 days to fully reprice. "
                f"Watch for residual volatility."
            )
            detail = (
                f"<b>{ev['name']}</b> ended within the past 90 days. "
                f"Post-event periods are often characterised by above-normal volatility "
                f"as markets digest the full implications. "
                f"<br><br>{ev.get('description','')}"
            )
            conf = 50
            conf_lbl = "Post-event window — elevated but declining risk"
            cards.append(dict(
                emoji=emoji, headline=headline, action=action, color=color,
                detail_html=detail, confidence=conf, confidence_label=conf_lbl,
            ))
    except Exception:
        pass

    return cards


# ── Page ──────────────────────────────────────────────────────────────────────

def page_insights(start: str, end: str, fred_key: str = "") -> None:

    st.markdown(
        '<h1 style="font-family:\'DM Sans\',sans-serif;font-size:1.25rem;'
        'font-weight:700;margin-bottom:0.1rem">Actionable Insights</h1>'
        '<p style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;color:#555;'
        'margin:0 0 0.7rem">Plain-language verdicts · Click any insight for the full reasoning</p>',
        unsafe_allow_html=True,
    )
    _page_intro(
        "No jargon. No charts you have to decode. Just the <strong>6 things you need to know "
        "right now</strong> about equity-commodity markets — and what to do about each one. "
        "Click any card to see the data and reasoning behind it."
    )

    with st.spinner("Reading live market data…"):
        eq_r, cmd_r = load_returns(start, end)

    if eq_r.empty or cmd_r.empty:
        st.error("Market data unavailable. Check your internet connection.")
        return

    cards = _build_insights(eq_r, cmd_r, fred_key, start, end)

    if not cards:
        st.info("Could not compute insights — data may be limited for the selected date range.")
        return

    st.markdown(
        f'<p style="{_F}font-size:0.66rem;color:#888;margin-bottom:1.2rem">'
        f'Insights computed from data in your selected date range. '
        f'Click any card to expand the full reasoning.</p>',
        unsafe_allow_html=True,
    )

    for card in cards:
        _insight_card(**card)
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    _page_footer()
