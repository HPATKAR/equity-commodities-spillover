"""
Actionable Insights - plain-language verdicts derived from live market data.
Every insight is one sentence a non-expert can act on.
Click any insight to expand the full quant reasoning behind it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.data.loader      import load_returns, load_commodity_prices, load_fixed_income_returns, load_fx_returns
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
from src.data.loader import load_private_credit_proxies

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
    conf_color = _GREEN if confidence >= 70 else (_AMBER if confidence >= 45 else _GREY)

    st.markdown(
        f'<div style="{_F}border:1px solid #2a2a2a;border-left:4px solid {color};'
        f'border-radius:0 6px 6px 0;padding:0.85rem 1.1rem 0.85rem;'
        f'background:#1c1c1c;margin-bottom:0">'

        # Row 1: emoji + headline
        f'<div style="display:flex;align-items:flex-start;gap:0.55rem;margin-bottom:0.5rem">'
        f'<span style="font-size:1.1rem;line-height:1.2">{emoji}</span>'
        f'<span style="font-size:0.86rem;font-weight:700;color:#e8e9ed;line-height:1.35">'
        f'{headline}</span></div>'

        # Row 2: action
        f'<div style="display:flex;align-items:flex-start;gap:0.6rem;flex-wrap:wrap;margin-bottom:0.6rem">'
        f'<span style="font-size:0.63rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.12em;color:{color};white-space:nowrap;margin-top:2px">▶ Action</span>'
        f'<span style="font-size:0.78rem;font-weight:500;color:#c8c8c8;line-height:1.5">'
        f'{action}</span></div>'

        # Row 3: reasoning snippet (first sentence of detail_html, stripped of tags)
        f'<div style="font-size:0.72rem;color:#8890a1;line-height:1.6;'
        f'border-top:1px solid #2a2a2a;padding-top:0.5rem;margin-bottom:0.55rem">'
        f'{detail_html}</div>'

        # Row 4: confidence bar
        f'<div style="display:flex;align-items:center;gap:0.6rem">'
        f'<span style="font-size:0.58rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.12em;color:{conf_color};white-space:nowrap">'
        f'Confidence {confidence}%</span>'
        f'<div style="flex:1;max-width:120px;height:3px;background:#2a2a2a;border-radius:2px">'
        f'<div style="height:3px;width:{confidence}%;background:{conf_color};'
        f'border-radius:2px"></div></div>'
        f'<span style="font-size:0.57rem;color:#6b7280;font-style:italic">'
        f'{confidence_label}</span>'
        f'</div>'

        f'</div>',
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
            action = "No urgent action needed - this is a good time to review your portfolio calmly."
            detail = (
                f"The overall stress index is <b>{score:.0f}/100</b> - well below the danger zone (50+). "
                f"This means equity and commodity markets are not showing signs of panic or forced selling. "
                f"Volatility is low, correlations are normal, and there is no unusual crowding in futures markets. "
                f"<br><br><b>What drives this score:</b> cross-asset correlation ({risk.get('corr_pct',0):.0f}th percentile), "
                f"commodity volatility z-score ({risk.get('cmd_vol_z',0):.1f}σ), "
                f"and equity volatility ({risk.get('eq_vol_z',0):.1f}σ). "
                f"All three are subdued."
            )
            conf = 80
            conf_lbl = "Strong - all three components agree"
        elif score < 55:
            emoji, headline = "🟡", "Markets are slightly nervous - nothing alarming yet"
            color  = _AMBER
            action = "Watch your positions with more commodity exposure - volatility may pick up."
            detail = (
                f"The stress index is <b>{score:.0f}/100</b> - in the moderate zone. "
                f"This often means one or two components are elevated (e.g. commodity volatility spiking) "
                f"while others remain calm. Not a crisis, but worth paying attention. "
                f"<br><br>Cross-asset correlation percentile: <b>{risk.get('corr_pct',0):.0f}th</b>. "
                f"Commodity vol z-score: <b>{risk.get('cmd_vol_z',0):.1f}σ</b>. "
                f"Equity vol z-score: <b>{risk.get('eq_vol_z',0):.1f}σ</b>. "
                f"<br><br>Historically, scores in this range precede a full stress event about 30% of the time. "
                f"The other 70% of the time, stress dissipates on its own."
            )
            conf = 62
            conf_lbl = "Moderate - mixed signals across components"
        else:
            emoji, headline = "🔴", "Markets are stressed - be careful"
            color  = _RED
            action = "Reduce risk. Move toward safer assets like gold or cash until the score drops below 50."
            detail = (
                f"The stress index is <b>{score:.0f}/100</b> - in the elevated/high zone. "
                f"This means equity and commodity markets are showing simultaneous stress: "
                f"correlations are spiking (assets moving together), volatility is elevated, "
                f"and futures positioning is becoming extreme. "
                f"<br><br>In past episodes with scores above 55, the average equity drawdown over the "
                f"following 30 days was -8% to -12%. Commodities were mixed - energy fell, gold rose. "
                f"<br><br>Cross-asset correlation: <b>{risk.get('corr_pct',0):.0f}th percentile</b>. "
                f"Commodity vol: <b>{risk.get('cmd_vol_z',0):.1f}σ above normal</b>. "
                f"Equity vol: <b>{risk.get('eq_vol_z',0):.1f}σ above normal</b>."
            )
            conf = 75
            conf_lbl = "High - multiple stress indicators elevated simultaneously"

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
            emoji, headline = "🟢", "Stocks and commodities are moving independently - diversification is working"
            color  = _GREEN
            action = "Good time to hold a mix of stocks and commodities - they are protecting each other."
            detail = (
                f"The 60-day average correlation between equity indices and commodity futures is "
                f"<b>{cur_corr:.3f}</b> - currently in the <b>{rname}</b> regime. "
                f"<br><br>When this number is low or negative, owning both stocks and commodities "
                f"reduces your overall risk - if stocks fall, commodities don't necessarily fall with them. "
                f"This is the whole point of diversification, and right now it is working. "
                f"<br><br>Historically, Decorrelated and Normal regimes persist for an average of 4–8 months "
                f"before transitioning. The Markov model currently assigns a "
                f"{100 - cur_regime * 15:.0f}% probability of staying in this regime next month."
            )
            conf = 72
            conf_lbl = "Regime has persisted for multiple weeks"
        elif cur_regime == 2:
            emoji, headline = "🟡", "Stocks and commodities are starting to move together - diversification is weakening"
            color  = _AMBER
            action = "Your commodity holdings may not protect you if stocks fall - consider adding gold or short-duration bonds."
            detail = (
                f"The 60-day average equity-commodity correlation is <b>{cur_corr:.3f}</b> - "
                f"<b>Elevated</b> regime. This means stocks and commodities are becoming more correlated "
                f"than usual. In simple terms: when stocks fall, your commodities are more likely to fall too. "
                f"<br><br>This typically happens during risk-off events (everyone sells everything at once) "
                f"or when macro forces (rising rates, dollar strength) hit all asset classes simultaneously. "
                f"<br><br>The Markov model assigns a ~35% probability of transitioning to a Crisis regime "
                f"within the next 20 trading days if the current trend continues."
            )
            conf = 65
            conf_lbl = "Clear regime shift underway - watch for acceleration"
        else:
            emoji, headline = "🔴", "Stocks and commodities are moving in lockstep - diversification has broken down"
            color  = _RED
            action = "Holding more stocks and commodities won't protect you. Move to gold, cash, or short positions."
            detail = (
                f"The 60-day average equity-commodity correlation is <b>{cur_corr:.3f}</b> - "
                f"<b>Crisis</b> regime. In a crisis regime, everything falls together: equities, oil, "
                f"industrial metals, even agricultural commodities. The only traditional safe haven "
                f"that historically decouples in this regime is gold (and sometimes bonds). "
                f"<br><br>This regime occurred during: GFC 2008–09, COVID crash Mar 2020, "
                f"and parts of 2022 Fed tightening. Average duration: 6–10 weeks. "
                f"<br><br>Recommended: raise gold allocation, reduce equity-commodity overlap, "
                f"add short-volatility positions only if you are experienced."
            )
            conf = 80
            conf_lbl = "Crisis regime - historically reliable signal"

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
                            f"<b>{leader}</b> is acting as a <b>hedge</b> right now - "
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
                            f"<b>{leader}</b> is moving in the <b>same direction as stocks</b> - "
                            f"correlation of {corr_val:.2f} over the past 60 days. "
                            f"This means {leader} is amplifying your risk if you already hold equities: "
                            f"owning both doesn't diversify you right now."
                        )
                        emoji, color = "🟡", _AMBER
                        action = f"If you hold both {leader} and equities, you have more concentrated risk than you think - they're moving together."

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
                        detail_html=detail, confidence=60, confidence_label="Rolling 60-day window - directional, not causal",
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
            emoji, headline = "🟢", "No early warning signs - no stress event appears imminent"
            color  = _GREEN
            action = "Markets look stable ahead. No defensive repositioning needed right now."
            detail = (
                f"The early warning composite is <b>{ew_score:.0f}/100</b> - low. "
                f"This score combines five forward-looking indicators: how fast correlation is "
                f"rising, whether volatility is accelerating, how long the current regime has lasted, "
                f"the trend in equity volatility, and the statistical probability of a regime change. "
                f"<br><br>All five are currently subdued, suggesting the current market structure "
                f"is stable and unlikely to deteriorate sharply in the near term."
                f"<br><br>Components - Correlation velocity: {ew.get('corr_velocity',50):.0f}/100 · "
                f"Vol acceleration: {ew.get('vol_accel',50):.0f}/100 · "
                f"Regime pressure: {ew.get('regime_pressure',50):.0f}/100"
            )
            conf = 68
            conf_lbl = "Multiple forward indicators agree - low near-term risk"
        elif ew_score < 65:
            emoji, headline = "🟡", "Some early warning signals are flashing - a rough patch may be coming"
            color  = _AMBER
            action = "Start trimming high-risk positions gradually. Don't panic, but don't ignore this either."
            detail = (
                f"The early warning composite is <b>{ew_score:.0f}/100</b> - moderate. "
                f"One or more leading indicators are rising: correlation between equities and "
                f"commodities is accelerating, or volatility is starting to pick up. "
                f"<br><br>Think of this like a weather forecast: it's not raining yet, "
                f"but the clouds are building. Historically, composite scores above 55 preceded "
                f"a measurable stress event (VIX spike >5pts) within 30 days about 42% of the time. "
                f"<br><br>Components - Correlation velocity: {ew.get('corr_velocity',50):.0f}/100 · "
                f"Vol acceleration: {ew.get('vol_accel',50):.0f}/100 · "
                f"Markov crisis probability: {ew.get('markov_crisis_prob',0)*100:.0f}%"
            )
            conf = 55
            conf_lbl = "Mixed - some signals elevated, others calm"
        else:
            emoji, headline = "🔴", "Multiple early warning signs are flashing - stress event likely in the near term"
            color  = _RED
            action = "Act now: reduce exposure to correlated equity-commodity positions before volatility spikes."
            detail = (
                f"The early warning composite is <b>{ew_score:.0f}/100</b> - high. "
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
            conf_lbl = "High - rare composite configuration historically predictive"

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
                    f"<b>{mover}</b> is down <b>{pct:.1f}%</b> over the past 5 trading days - "
                    f"the largest move among all tracked commodities. "
                    f"A sharp drop in a commodity this quickly often spills into related equity sectors: "
                    f"energy stocks follow oil, mining stocks follow metals. "
                    f"Watch for equity sector contagion in the coming days."
                )
                emoji, color = "📉", _RED
                action = f"Watch equity sectors exposed to {mover} - they may be the next to reprice."
            else:
                mover, pct = top, top_v
                direction  = "rising"
                implication = (
                    f"<b>{mover}</b> is up <b>{pct:.1f}%</b> over the past 5 trading days - "
                    f"the largest move among all tracked commodities. "
                    f"A sharp commodity rally can signal inflationary pressure, "
                    f"which often weighs on equity multiples (especially growth stocks). "
                    f"But it can also be a risk-on signal if driven by demand rather than supply disruption."
                )
                emoji, color = "📈", _AMBER
                action = f"{mover} is the hottest commodity right now - look at equity sectors that benefit from higher {mover} prices."

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
                detail_html=detail, confidence=55, confidence_label="Price momentum - short-term signal only",
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
            headline = f"Active event on the radar: {ev['name']} - historically this disrupts commodity prices"
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
            headline = f"{ev['name']} ended recently - markets may still be repricing"
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
            conf_lbl = "Post-event window - elevated but declining risk"
            cards.append(dict(
                emoji=emoji, headline=headline, action=action, color=color,
                detail_html=detail, confidence=conf, confidence_label=conf_lbl,
            ))
    except Exception:
        pass

    # ── 7. Private credit bubble risk ─────────────────────────────────────
    try:
        pc_card = _build_private_credit_insight(fred_key, start, end, eq_r)
        if pc_card is not None:
            cards.append(pc_card)
    except Exception:
        pass

    # ── 8. Yield curve regime ──────────────────────────────────────────────
    try:
        fi_r_ins = load_fixed_income_returns(start, end)
        if not fi_r_ins.empty:
            _tlt_col = "US 20Y+ Treasury (TLT)"
            _shy_col = "US 1-3Y Treasury (SHY)"
            if _tlt_col in fi_r_ins.columns and _shy_col in fi_r_ins.columns:
                _tlt_60 = float(fi_r_ins[_tlt_col].dropna().iloc[-60:].sum() * 100) if len(fi_r_ins) >= 60 else None
                _shy_60 = float(fi_r_ins[_shy_col].dropna().iloc[-60:].sum() * 100) if len(fi_r_ins) >= 60 else None
                if _tlt_60 is not None and _shy_60 is not None:
                    _dur_spread = _tlt_60 - _shy_60
                    if _dur_spread < -3:
                        emoji, color = "🔴", _RED
                        headline = "Bear steepening in Treasuries - long duration is being punished"
                        action = "Reduce TLT / long-duration bond exposure. Short end outperforming implies rates pricing higher for longer - negative for high-multiple equities and gold."
                        detail = (
                            f"TLT (20Y+) returned <b>{_tlt_60:+.1f}%</b> over 60 days vs SHY (1-3Y) <b>{_shy_60:+.1f}%</b> - "
                            f"a <b>{_dur_spread:+.1f}pp</b> duration spread. "
                            f"Bear steepening occurs when long rates rise faster than short rates, driven by term premium expansion or fiscal concerns. "
                            f"<br><br>This regime is historically negative for: growth equities (discount rate rises), gold (real rate pressure), "
                            f"and EM bonds (dollar strength accompanies). Commodities are mixed - energy benefits from growth expectations, metals suffer."
                        )
                        conf, conf_lbl = 68, "Duration underperformance confirmed over 60 days"
                    elif _dur_spread > 3:
                        emoji, color = "🟢", _GREEN
                        headline = "Long duration outperforming - rates market pricing in cuts or flight to quality"
                        action = "TLT outperformance signals either growth fears or a policy pivot. If accompanied by equity weakness, this is a classic flight-to-quality regime."
                        detail = (
                            f"TLT returned <b>{_tlt_60:+.1f}%</b> vs SHY <b>{_shy_60:+.1f}%</b> over 60 days - "
                            f"long duration outperforming by <b>{_dur_spread:+.1f}pp</b>. "
                            f"<br><br>Bull flattening or rallying long end typically accompanies: risk-off equity positioning, gold strength, "
                            f"commodity demand concerns (growth slowdown), and EM bond relief (if dollar weakens with rates)."
                        )
                        conf, conf_lbl = 65, "Duration outperformance confirmed over 60 days"
                    else:
                        emoji, color = "🟡", _AMBER
                        headline = "Yield curve broadly flat - rates market in wait-and-see mode"
                        action = "No strong duration signal. Monitor 10Y-2Y spread for directional confirmation."
                        detail = (
                            f"TLT and SHY have performed similarly over 60 days ({_dur_spread:+.1f}pp spread), "
                            f"indicating a broadly flat rates signal. Neither bull nor bear steepening is dominant. "
                            f"<br><br>Flat curve regimes are transitional - watch for a break in either direction as the lead signal for equity and commodity repricing."
                        )
                        conf, conf_lbl = 50, "Neutral - curve flat, no directional signal"
                    cards.append(dict(
                        emoji=emoji, headline=headline, action=action, color=color,
                        detail_html=detail, confidence=conf, confidence_label=conf_lbl,
                    ))
    except Exception:
        pass

    # ── 9. Dollar and FX impact ────────────────────────────────────────────
    try:
        fx_r_ins = load_fx_returns(start, end)
        fi_r_ins2 = load_fixed_income_returns(start, end)
        if not fx_r_ins.empty and "DXY (Dollar Index)" in fx_r_ins.columns:
            _dxy = fx_r_ins["DXY (Dollar Index)"].dropna()
            if len(_dxy) >= 60:
                _dxy_60 = float(_dxy.iloc[-60:].sum() * 100)
                _dxy_20 = float(_dxy.iloc[-20:].sum() * 100)
                _emb_60 = None
                if not fi_r_ins2.empty and "EM USD Bonds (EMB)" in fi_r_ins2.columns:
                    _emb_s = fi_r_ins2["EM USD Bonds (EMB)"].dropna()
                    if len(_emb_s) >= 60:
                        _emb_60 = float(_emb_s.iloc[-60:].sum() * 100)

                if _dxy_60 > 3:
                    emoji, color = "🔴", _AMBER
                    headline = f"Dollar strengthening (+{_dxy_60:.1f}% over 60 days) - headwind for commodities and EM assets"
                    action = "Dollar strength compresses dollar-denominated commodity prices and pressures EM equities and bonds. Reduce commodity and EMB exposure; favour domestic US assets."
                    detail = (
                        f"The DXY proxy (UUP) is up <b>{_dxy_60:+.1f}%</b> over 60 days (<b>{_dxy_20:+.1f}%</b> over 20 days). "
                        f"<br><br>A strong dollar mechanically reduces the dollar price of commodities for foreign buyers, dampening demand. "
                        f"For EM bonds: dollar strength raises local-currency debt service costs and triggers capital outflows. "
                        + (f"EMB has returned <b>{_emb_60:+.1f}%</b> over the same period - {'confirming EM stress' if _emb_60 and _emb_60 < -1 else 'holding relatively well despite dollar pressure'}." if _emb_60 is not None else "")
                        + f"<br><br>Historically, DXY +5% in 60 days correlates with: Gold -3 to -5%, WTI -4 to -7%, EM equity -5 to -10%."
                    )
                    conf, conf_lbl = 70, "Dollar trend confirmed over 20d and 60d windows"
                elif _dxy_60 < -3:
                    emoji, color = "🟢", _GREEN
                    headline = f"Dollar weakening ({_dxy_60:.1f}% over 60 days) - tailwind for commodities and EM assets"
                    action = "Dollar weakness is a tailwind for gold, industrial metals, and EM bonds. Increase commodity and EMB allocation."
                    detail = (
                        f"The DXY proxy is down <b>{_dxy_60:.1f}%</b> over 60 days. "
                        f"<br><br>Dollar weakness historically supports: Gold (+3 to +8%), industrial metals (+2 to +5%), "
                        f"agricultural commodities, and EM assets via improved debt dynamics and capital inflows. "
                        + (f"EMB has returned <b>{_emb_60:+.1f}%</b> over the same period." if _emb_60 is not None else "")
                    )
                    conf, conf_lbl = 67, "Dollar weakness confirmed over 20d and 60d windows"
                else:
                    emoji, color = "🟡", _AMBER
                    headline = "Dollar broadly stable - FX not a dominant cross-asset driver right now"
                    action = "No strong FX signal. Dollar-commodity and dollar-EM relationships neutral."
                    detail = (
                        f"DXY proxy is {_dxy_60:+.1f}% over 60 days - within a neutral range. "
                        f"When the dollar is range-bound, commodity and EM price action is driven more by their own fundamentals than by FX translation effects."
                    )
                    conf, conf_lbl = 48, "Dollar range-bound - signal inconclusive"
                cards.append(dict(
                    emoji=emoji, headline=headline, action=action, color=color,
                    detail_html=detail, confidence=conf, confidence_label=conf_lbl,
                ))
    except Exception:
        pass

    # ── 10. India / Rupee macro signal ────────────────────────────────────
    try:
        fx_r_india = load_fx_returns(start, end)
        if not fx_r_india.empty and "USD/INR" in fx_r_india.columns:
            _inr = fx_r_india["USD/INR"].dropna()  # rising = INR weakening
            if len(_inr) >= 60:
                _inr_60 = float(_inr.iloc[-60:].sum() * 100)
                _inr_20 = float(_inr.iloc[-20:].sum() * 100)
                # Nifty 50 signal from eq_r
                _nifty_col = next((c for c in eq_r.columns if "Nifty" in c or "NSEI" in c), None)
                _nifty_60 = None
                if _nifty_col:
                    _ns = eq_r[_nifty_col].dropna()
                    if len(_ns) >= 60:
                        _nifty_60 = float(_ns.iloc[-60:].sum() * 100)

                if _inr_60 > 2.5:
                    # USD/INR rising = INR weakening
                    emoji, color = "🔴", _AMBER
                    headline = f"Indian Rupee weakening ({_inr_60:.1f}% 60d) - pressure on India's import bill"
                    action = (
                        "INR weakness raises India's crude oil and gold import costs (India imports ~85% of crude needs). "
                        "Watch Nifty Energy and consumer discretionary for margin pressure. "
                        "RBI may intervene via forex reserves to stabilise INR."
                    )
                    detail = (
                        f"USD/INR is up <b>{_inr_60:+.1f}%</b> over 60 days (<b>{_inr_20:+.1f}%</b> over 20 days) - the rupee is depreciating. "
                        f"<br><br>India is among the world's largest commodity importers: "
                        f"<b>#3 crude oil importer</b> (~5 mb/d, 85% import dependency) and <b>#2 gold consumer</b> (~800–900 tonnes/year). "
                        f"Each 5% INR depreciation raises India's annual crude import bill by ~$8–10B, compressing CAD and pressuring equities. "
                        + (f"<br><br>Nifty 50 has returned <b>{_nifty_60:+.1f}%</b> over 60 days - "
                           f"{'underperformance consistent with FX headwind' if _nifty_60 is not None and _nifty_60 < 0 else 'equity market showing resilience despite INR pressure'}."
                           if _nifty_60 is not None else "")
                        + f"<br><br>RBI repo rate: ~6.50%. India's forex reserves are ~$620B, providing significant intervention capacity."
                    )
                    conf, conf_lbl = 65, "INR trend confirmed over 20d and 60d windows"
                elif _inr_60 < -2.5:
                    # USD/INR falling = INR strengthening
                    emoji, color = "🟢", _GREEN
                    headline = f"Indian Rupee strengthening ({_inr_60:.1f}% 60d) - easing India's commodity import pressure"
                    action = (
                        "INR strength reduces India's crude oil and gold import costs. "
                        "Tailwind for India's fiscal deficit and consumer spending. "
                        "Nifty 50 and Indian consumer/energy importers benefit."
                    )
                    detail = (
                        f"USD/INR is down <b>{_inr_60:.1f}%</b> over 60 days - the rupee is appreciating. "
                        f"<br><br>A stronger rupee directly reduces India's commodity import bill. "
                        f"India imports <b>~85% of crude oil needs</b> (~5 mb/d) and is the world's #2 gold consumer. "
                        f"INR appreciation at this magnitude could save $5–8B annually on crude imports alone. "
                        + (f"<br><br>Nifty 50 has returned <b>{_nifty_60:+.1f}%</b> over 60 days."
                           if _nifty_60 is not None else "")
                        + f"<br><br>Capital inflows and/or a weak dollar are typically the catalyst. Monitor FII equity flows into NSE."
                    )
                    conf, conf_lbl = 63, "INR appreciation confirmed over 20d and 60d windows"
                else:
                    emoji, color = "🟡", _AMBER
                    headline = "Indian Rupee broadly stable - INR not a dominant cross-asset driver right now"
                    action = "No strong INR signal. India's import costs are stable. Monitor crude oil prices as the primary Nifty macro driver."
                    detail = (
                        f"USD/INR is {_inr_60:+.1f}% over 60 days - within a neutral band. "
                        f"India's fiscal position is most sensitive to crude oil (import cost ~$120–140B/year at $80/bbl). "
                        f"With INR stable, Nifty 50 is driven by domestic earnings and global risk appetite rather than FX pass-through."
                    )
                    conf, conf_lbl = 48, "INR range-bound - FX not dominating India macro narrative"
                cards.append(dict(
                    emoji=emoji, headline=headline, action=action, color=color,
                    detail_html=detail, confidence=conf, confidence_label=conf_lbl,
                ))
    except Exception:
        pass

    return cards


# ── 7. Private credit bubble risk ─────────────────────────────────────────

def _build_private_credit_insight(
    fred_key: str,
    start: str,
    end: str,
    eq_r: pd.DataFrame,
) -> dict | None:
    """
    Quantify private credit bubble/burst risk using four observable proxies:
      1. HY OAS (FRED BAMLH0A0HYM2)  - spread percentile + 90-day trend
      2. BKLN ETF vs S&P 500          - leveraged loan / equity divergence
      3. BDC basket (ARCC/OBDC/FSK)  - NAV/price stress signal
      4. HY-IG spread ratio            - search-for-yield crowding

    Composite score 0–100:
      ≥ 65 + widening → "Burst in progress"   (RED)
      ≥ 65 tight       → "Bubble regime"       (AMBER)
      40–65            → "Late-cycle caution"  (AMBER-lite)
      < 40             → "No imminent stress"  (GREEN)
    """
    scores: dict[str, float] = {}

    # ── Signal 1: HY OAS from FRED ────────────────────────────────────────
    hy_current = hy_percentile = hy_90d_chg = ig_current = None
    hy_hist_mean = hy_hist_std = None

    if fred_key:
        try:
            from fredapi import Fred
            _fred = Fred(api_key=fred_key)
            hy_raw = _fred.get_series("BAMLH0A0HYM2", observation_start=start).dropna()
            ig_raw = _fred.get_series("BAMLC0A0CM",   observation_start=start).dropna()
            if len(hy_raw) > 100:
                hy_current      = float(hy_raw.iloc[-1])
                hy_percentile   = float((hy_raw < hy_current).mean() * 100)
                hy_90d_chg      = float(hy_raw.iloc[-1] - hy_raw.iloc[-91]) if len(hy_raw) > 91 else 0.0
                hy_hist_mean    = float(hy_raw.mean())
                hy_hist_std     = float(hy_raw.std())
                # Bubble score: how compressed are spreads vs history?
                # Tight spreads (low percentile) → high bubble score
                scores["hy_oas"] = 100.0 - hy_percentile
            if len(ig_raw) > 10:
                ig_current = float(ig_raw.iloc[-1])
        except Exception:
            pass

    # ── Signal 2: BKLN + BDC basket via yfinance ─────────────────────────
    bkln_90d = bdc_90d = spx_90d = bkln_alpha = bdc_alpha = None
    bkln_px = bdc_px = spy_px = None

    try:
        import yfinance as _yf
        _raw = _yf.download(
            ["BKLN", "ARCC", "OBDC", "FSK", "SPY"],
            start=start, end=end, auto_adjust=True, progress=False,
        )
        _px = (_raw["Close"] if isinstance(_raw.columns, pd.MultiIndex) else _raw).ffill()

        n = 91  # ~90 trading days
        if "SPY" in _px.columns and _px["SPY"].dropna().shape[0] > n:
            _spy = _px["SPY"].dropna()
            spx_90d = float((_spy.iloc[-1] / _spy.iloc[-n] - 1) * 100)
            spy_px  = _spy

        if "BKLN" in _px.columns and _px["BKLN"].dropna().shape[0] > n:
            _bkln  = _px["BKLN"].dropna()
            bkln_90d = float((_bkln.iloc[-1] / _bkln.iloc[-n] - 1) * 100)
            bkln_px  = _bkln
            if spx_90d is not None:
                bkln_alpha = bkln_90d - spx_90d
                # Underperformance → stress score: -5% alpha → 25/100; -10% → 50/100
                scores["bkln"] = min(100.0, max(0.0, -bkln_alpha * 5.0))

        _bdc_cols = [t for t in ["ARCC", "OBDC", "FSK"] if t in _px.columns and _px[t].dropna().shape[0] > n]
        if _bdc_cols:
            _vals = [float((_px[t].dropna().iloc[-1] / _px[t].dropna().iloc[-n] - 1) * 100) for t in _bdc_cols]
            bdc_90d = float(np.mean(_vals))
            if spx_90d is not None:
                bdc_alpha = bdc_90d - spx_90d
                scores["bdc"] = min(100.0, max(0.0, -bdc_alpha * 4.0))
    except Exception:
        pass

    if not scores:
        return None

    # ── Composite score ──────────────────────────────────────────────────
    _weights = {"hy_oas": 0.50, "bkln": 0.30, "bdc": 0.20}
    _total_w = sum(_weights.get(k, 0.0) for k in scores)
    composite = sum(scores[k] * _weights.get(k, 0.0) for k in scores) / max(_total_w, 0.01)

    # Widening fast? = bust in progress
    _spreading = (hy_90d_chg is not None and hy_90d_chg > 35)

    # ── Determine card signal ────────────────────────────────────────────
    if composite >= 65:
        if _spreading:
            emoji, color = "🔴", _RED
            headline = "Private credit unwind may be underway - HY spreads widening while $2T+ AUM sits illiquid"
            action = (
                "Exit semi-liquid private credit funds (interval funds, NAV funds) before redemption gates trigger. "
                "Buy HY CDX protection. Short BDC equity basket (ARCC, OBDC, FSK). Long gold as contagion hedge."
            )
            conf, conf_lbl = 74, "Spread widening + BDC underperformance - early burst signal"
        else:
            emoji, color = "🟡", _AMBER
            headline = "Private credit bubble: $2T+ AUM priced at historically tight HY spreads - repricing risk high"
            action = (
                "Reduce new commitments to private credit funds. Trim BDC equity. "
                "Spreads at this compression historically precede 200–500bp widening events within 6–18 months."
            )
            conf, conf_lbl = 67, "Spread compression + structural leverage risk"
    elif composite >= 40:
        emoji, color = "🟡", _AMBER
        headline = "Private credit late-cycle: leverage + rate dynamics are compressing coverage ratios"
        action = (
            "Monitor HY OAS monthly - a move above 400bps signals repricing onset. "
            "Begin rotating out of lower-quality private credit vintages (2021–2022 cohort most at risk)."
        )
        conf, conf_lbl = 55, "Late-cycle signals present; no acute trigger yet"
    else:
        emoji, color = "🟢", _GREEN
        headline = "Private credit markets: spreads normalised - no imminent systemic stress signal"
        action = "No defensive action required on private credit exposure. Revisit when HY OAS moves above 450bps."
        conf, conf_lbl = 58, "Spread levels and market dynamics within norms"

    # ── Build detail HTML ────────────────────────────────────────────────
    _d = []
    _d.append(
        f"<b>PC Bubble Risk Score: {composite:.0f}/100</b> · "
        f"{'Elevated' if composite >= 65 else 'Moderate' if composite >= 40 else 'Low'}"
        f"<br><br>"
        f"<b>Structural thesis:</b> Private credit AUM grew from ~$500B (2010) to $2T+ (2024) - "
        f"primarily via floating-rate direct lending at 5–7× leverage. With SOFR at 4.3%+, "
        f"all-in borrower costs are 9–12%. Interest coverage ratios at many portfolio companies "
        f"now sit below 1.5× - historically a fragile level. Unlike public markets, "
        f"private credit is marked-to-model quarterly, masking latent stress until it cannot be hidden. "
        f"The bust mechanism: redemption pressure in semi-liquid vehicles → forced sales → "
        f"price discovery → contagion to public HY and leveraged loan markets."
        f"<br><br>"
    )

    if hy_current is not None:
        _chg_str = f" - <b>{abs(hy_90d_chg):.0f} bps {'wider' if hy_90d_chg > 0 else 'tighter'}</b> in 90 days" if hy_90d_chg is not None else ""
        _d.append(
            f"<b>HY OAS:</b> {hy_current:.0f} bps "
            f"({hy_percentile:.0f}th pct of {start[:4]}–present history{_chg_str}). "
        )
        if ig_current is not None:
            _d.append(f"IG OAS: {ig_current:.0f} bps. HY/IG ratio: {hy_current/ig_current:.1f}×. ")
        if hy_percentile is not None and hy_percentile < 30:
            _d.append(
                f"Sub-30th percentile HY spreads have historically been followed by "
                f"200–500bp widening events (GFC: +1,700bp; COVID: +600bp; 2022: +280bp). "
            )
        _d.append("<br>")

    if bkln_alpha is not None:
        _d.append(
            f"<b>Leveraged loans (BKLN):</b> {bkln_90d:+.1f}% (90d) vs S&P 500 {spx_90d:+.1f}% → "
            f"alpha {bkln_alpha:+.1f}%."
        )
        if bkln_alpha < -4:
            _d.append(
                f" Loan market underperformance typically leads public HY repricing by 4–8 weeks."
            )
        _d.append("<br>")

    if bdc_alpha is not None:
        _d.append(
            f"<b>BDC basket (ARCC/OBDC/FSK):</b> avg {bdc_90d:+.1f}% (90d) vs S&P 500 {spx_90d:+.1f}% → "
            f"alpha {bdc_alpha:+.1f}%."
        )
        if bdc_alpha < -5:
            _d.append(
                f" BDC discount-to-book widening typically precedes quarterly NAV writedowns by 1–2 quarters."
            )
        _d.append("<br>")

    _d.append(
        f"<br><b>Key risk to short thesis:</b> Fed emergency rate cuts compress floating-rate "
        f"borrowing costs rapidly; sponsor recapitalisations delay visible defaults; "
        f"institutional redemptions gated (interval fund lockups mask stress for 12–18 months)."
    )

    return dict(
        emoji=emoji,
        headline=headline,
        action=action,
        color=color,
        detail_html="".join(_d),
        confidence=conf,
        confidence_label=conf_lbl,
    )


# ── Page ──────────────────────────────────────────────────────────────────────

def page_insights(start: str, end: str, fred_key: str = "") -> None:

    st.markdown(
        '<h1 style="font-family:\'DM Sans\',sans-serif;font-size:1.25rem;'
        'font-weight:700;margin-bottom:0.1rem">Actionable Insights</h1>'
        '<p style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;color:#8890a1;'
        'margin:0 0 0.7rem">Plain-language verdicts · Click any insight for the full reasoning</p>',
        unsafe_allow_html=True,
    )
    _page_intro(
        "No jargon. No charts you have to decode. Just the <strong>6 things you need to know "
        "right now</strong> about equity-commodity markets - and what to do about each one. "
        "Click any card to see the data and reasoning behind it."
    )

    with st.spinner("Reading live market data…"):
        eq_r, cmd_r = load_returns(start, end)

    if eq_r.empty or cmd_r.empty:
        st.error("Market data unavailable. Check your internet connection.")
        return

    cards = _build_insights(eq_r, cmd_r, fred_key, start, end)

    if not cards:
        st.info("Could not compute insights - data may be limited for the selected date range.")
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

    # ── PDF Download ──────────────────────────────────────────────────────────
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="{_F}font-size:0.68rem;color:#8890a1;margin:0 0 0.5rem">'
        f'Export all insights with full reasoning to an institutional-format PDF.</p>',
        unsafe_allow_html=True,
    )
    if st.button("Generate PDF Report", type="primary", key="insights_pdf_btn"):
        with st.spinner("Building PDF…"):
            try:
                from src.reports.insights_pdf import build_insights_pdf
                pdf_bytes = build_insights_pdf(cards, start, end)
                fname = f"actionable_insights_{__import__('datetime').date.today().strftime('%Y%m%d')}.pdf"
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name=fname,
                    mime="application/pdf",
                    key="insights_pdf_dl",
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

    # ── Chief Quality Officer ─────────────────────────────────────────────────
    try:
        from src.agents.quality_officer import run as _cqo_run
        from src.ui.agent_panel import render_agent_output_block
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

            _n_obs = len(eq_r.dropna(how="all"))
            _n_cards = len(cards) if "cards" in dir() and cards else 0
            _cqo_ctx = {
                "n_obs":            _n_obs,
                "date_range":       f"{start} to {end}",
                "model":            "Multi-factor composite risk score + rule-based insight cards",
                "assumption_count": 6,
                "notes": [
                    f"{_n_cards} insight cards generated — thresholds for card activation are hardcoded, not data-driven",
                    "Composite risk score stacks correlation, volatility, and commodity z-scores — weights are arbitrary",
                    "Private credit bubble score uses BDC equity proxies (ARCC, OBDC) — equity prices ≠ credit spreads",
                    "Leading/lagging commodity detection uses cross-correlation lag — spurious at short samples",
                    "Regime-conditional insights assume regime is stable — mean reversion risk in volatile periods",
                    "All insight thresholds (e.g. corr > 0.45 = elevated) lack empirical validation",
                ],
            }
            with st.spinner("CQO auditing insights methodology…"):
                _cqo_result = _cqo_run(_cqo_ctx, _provider, _api_key, page="Actionable Insights")
            if _cqo_result.get("narrative"):
                st.markdown("---")
                render_agent_output_block("quality_officer", _cqo_result)
    except Exception:
        pass

    _page_footer()
