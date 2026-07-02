"""
Page 6 - Trade Ideas
Regime-triggered + conflict-driven cross-asset trade cards.
Integrates conflict exposure scoring, scenario-aware payoff tables, QC grading,
filter sidebar, and agent debate threads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date

from src.data.loader import load_returns, load_commodity_prices
from src.data.config import GEOPOLITICAL_EVENTS, PALETTE
from src.analysis.correlations import (
    average_cross_corr_series, detect_correlation_regime, rolling_correlation,
    composite_stress_index,
)
from src.analysis.backtest import walk_forward_backtest, qc_grade_backtest
from src.ui.shared import (
    _style_fig, _chart, _page_intro, _thread, _section_note,
    _definition_block, _takeaway_block, _page_conclusion, _page_header, _page_footer,
    _insight_note,
)

_REGIME_NAMES  = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis"}
_REGIME_COLORS = {0: "#2e7d32",      1: "#555960", 2: "#e67e22",  3: "#c0392b"}


# ── Master Investor Lens ───────────────────────────────────────────────────
# Maps trade category → [manager, archetype, insight] aligned with
# fundamental/macro-driven style: Druckenmiller, Templeton, Marks, Naren, Sharma.
_MASTER_INVESTOR_LENS: dict[str, list[dict]] = {
    "Crisis Hedge": [
        {"manager": "Howard Marks",         "archetype": "CYCLE AWARE",
         "insight": "Risk control over return maximisation. Permanent capital loss is the only real risk. Being right eventually is worthless if leverage wipes you out before the thesis plays."},
        {"manager": "Stanley Druckenmiller","archetype": "MACRO TACTICIAN",
         "insight": "Ruthless capital preservation: prioritise zero-loss years above all else. Cut immediately the moment the macro thesis cracks."},
        {"manager": "Seth Klarman",         "archetype": "ABSOLUTE RETURN",
         "insight": "Cash is an asset class and an option. Liquidity evaporation during crisis is precisely when forced selling creates the asymmetric entry."},
    ],
    "Geopolitical": [
        {"manager": "John Templeton",       "archetype": "MAX PESSIMISM",
         "insight": "Buy at the point of maximum pessimism. Geopolitical panic is the entry point, not the exit. The crowd abandons assets at precisely the wrong moment."},
        {"manager": "Shankar Sharma",       "archetype": "MACRO INFLECTION",
         "insight": "Spotting inflection points early: geopolitical catalysts create the entry, but it is the fundamental macro story that drives the multi-year return."},
        {"manager": "Prem Watsa",           "archetype": "CONTRARIAN VALUE",
         "insight": "Macro-economic hedging with derivatives. Geo events create the asymmetric tail protection the Fairfax playbook demands. Never fight the geopolitical cycle."},
    ],
    "Macro": [
        {"manager": "Stanley Druckenmiller","archetype": "MACRO LIQUIDITY",
         "insight": "Liquidity drives markets above near-term earnings by an order of magnitude. Follow the money supply and central bank balance sheet — they are the real alpha signal."},
        {"manager": "S Naren",             "archetype": "DYNAMIC ALLOCATOR",
         "insight": "Dynamic asset allocation based on macro valuation metrics like Market Cap-to-GDP. Shift systematically; mean reversion is the law of financial gravity."},
        {"manager": "Shankar Sharma",       "archetype": "MACRO INFLECTION",
         "insight": "Global macro allocation based on structural data, not narrative. The largest forces — fiscal, monetary, demographic — move asset prices over years, not quarters."},
    ],
    "Growth": [
        {"manager": "Peter Lynch",          "archetype": "GARP",
         "insight": "Copper rising with equities confirms genuine demand expansion, not just liquidity fiction. Industrial metals are the purest economic truth-teller."},
        {"manager": "Rakesh Jhunjhunwala",  "archetype": "INDIA GARP",
         "insight": "When the fundamental thesis matches reality, back it with massive capital and hold through multi-year volatility. Growth plus momentum is the optimal combination."},
        {"manager": "Ramdeo Agrawal",       "archetype": "QGLP",
         "insight": "Buy right, sit tight. Low-correlation growth regimes with commodity confirmation are the multi-year compounders the QGLP framework was built to own."},
    ],
    "Private Credit": [
        {"manager": "Howard Marks",         "archetype": "CREDIT CYCLE",
         "insight": "Private credit is marked-to-model at cycle peaks. When HY OAS widens, the BDC NAV fiction unravels with a 1–2 quarter lag. Permanent capital loss is the only risk that matters."},
        {"manager": "Seth Klarman",         "archetype": "DISTRESSED",
         "insight": "Forced institutional selling driven by credit rating downgrades is the Baupost entry point. The psychology of institutional ignorance creates the asymmetric mispricing."},
        {"manager": "David Tepper",         "archetype": "DISTRESSED DEBT",
         "insight": "Invest at the absolute bottom of the capital structure. Distressed debt converts to dominant post-restructuring equity — the Appaloosa playbook from the 2009 financials trade."},
    ],
    "Fixed Income": [
        {"manager": "Howard Marks",         "archetype": "CYCLE AWARE",
         "insight": "Asymmetry of returns: capture the quality safe-haven bid while shorting the credit spread blow-out. Know where you are in the credit cycle."},
        {"manager": "S Naren",             "archetype": "DYNAMIC ALLOCATOR",
         "insight": "Shift systematically from equity to fixed income when macro valuation metrics signal cycle-peak complacency. Patience in contrariness."},
        {"manager": "Prem Watsa",           "archetype": "MACRO HEDGE",
         "insight": "Prohibiting leverage and using macro derivatives to protect against systemic credit collapse is the Fairfax doctrine. Duration safety hedges deflation scenarios."},
    ],
    "India/EM": [
        {"manager": "S Naren",             "archetype": "INDIA MACRO",
         "insight": "Patience in contrariness: invest in deeply undervalued sectors facing cyclical headwinds. India's oil import shock is a cyclical headwind — not a structural ruin."},
        {"manager": "Prashant Jain",        "archetype": "CONTRA-CYCLICAL",
         "insight": "Contra-cyclical value discipline: buy out-of-favour sectors at peak cyclical pain. Ask whether the business survives over a 10-year horizon — not 10 weeks."},
        {"manager": "John Templeton",       "archetype": "MAX PESSIMISM",
         "insight": "Maximum pessimism in EM is the entry point. INR stress and crude shock at extremes is precisely the Templeton setup. Buy when blood is in the streets."},
    ],
    "Asia Divergence": [
        {"manager": "Shankar Sharma",       "archetype": "MACRO INFLECTION",
         "insight": "Spotting inflection points early across cap ranges. China property is structural, not cyclical. Japan is the mirror image — BOJ policy normalisation is a decade-long re-rating."},
        {"manager": "Kerr Neilson",         "archetype": "GLOBAL CONTRARIAN",
         "insight": "True global sourcing agility: allocate into regions when valuations become compelling. Japan at Shiller CAPE 22x vs China property distress is a screaming divergence trade."},
        {"manager": "S Naren",             "archetype": "DYNAMIC ALLOCATOR",
         "insight": "Macro risk tracking: China leverage cycle is in systemic unwind. Mean reversion for Japan is supported by BOJ policy normalisation and structural Yen tailwind."},
    ],
    "Dollar Cycle": [
        {"manager": "Stanley Druckenmiller","archetype": "MACRO LIQUIDITY",
         "insight": "The dollar cycle is the single most powerful force for EM asset re-rating. Liquidity flows globally — when DXY peaks, EM assets inflect sharply."},
        {"manager": "John Templeton",       "archetype": "MAX PESSIMISM",
         "insight": "Buy at maximum pessimism. When EM is universally abandoned at a dollar peak, it is universally mispriced. The fundamental equation reverses."},
        {"manager": "Kerr Neilson",         "archetype": "GLOBAL CONTRARIAN",
         "insight": "True global sourcing agility: allocate freely across geographies when dollar-driven valuation dislocations create compelling entry points in EM."},
    ],
}

# ── Specific tradeable instruments for each trade ──────────────────────────
# {trade_name: {asset_name: "TICKER — description"}}
# Used for display only — backtest uses the asset name against return data columns.
_TRADE_TICKERS: dict[str, dict[str, str]] = {
    "Long Gold / Short Eurostoxx 50": {
        "Gold":          "GLD  — SPDR Gold Shares (NYSE)",
        "Eurostoxx 50":  "FEZ  — SPDR Euro Stoxx 50 ETF (NYSE)",
    },
    "Long Natural Gas / Short Nikkei 225": {
        "Natural Gas":   "UNG  — United States Natural Gas Fund (NYSE)",
        "Nikkei 225":    "EWJ  — iShares MSCI Japan ETF (NYSE)",
    },
    "Long Wheat / Long Gold / Short Emerging Markets": {
        "Wheat":         "WEAT — Teucrium Wheat Fund (NYSE)",
        "Gold":          "GLD  — SPDR Gold Shares (NYSE)",
        "Sensex":        "EEM  — iShares MSCI Emerging Markets ETF (NYSE)",
    },
    "Long Copper / Long S&P 500": {
        "Copper":        "CPER — United States Copper Index Fund (NYSE)",
        "S&P 500":       "SPY  — SPDR S&P 500 ETF Trust (NYSE)",
    },
    "Long WTI Crude / Short S&P 500 Energy-Heavy Sectors": {
        "WTI Crude Oil": "USO  — United States Oil Fund (NYSE) | XLE short for sector precision",
        "S&P 500":       "SPY  — SPDR S&P 500 ETF Trust (NYSE)",
    },
    "Long Gold, Long Silver / Short Copper, Short Shanghai Comp": {
        "Gold":          "GLD  — SPDR Gold Shares (NYSE)",
        "Silver":        "SLV  — iShares Silver Trust (NYSE)",
        "Copper":        "CPER — United States Copper Index Fund (NYSE)",
        "Shanghai Comp": "MCHI — iShares MSCI China ETF (NYSE)",
    },
    "Short BDC Basket / Long HY Credit Protection": {
        "Ares Capital (ARCC)": "ARCC — Ares Capital Corp (NASDAQ) — short",
        "Blue Owl (OBDC)":     "OBDC — Blue Owl Capital Corp (NYSE) — short",
        "Gold":                "GLD  — SPDR Gold Shares (NYSE) — long",
    },
    "Long TLT / Short HYG (Flight to Quality)": {
        "US 20Y+ Treasury (TLT)": "TLT  — iShares 20+ Year Treasury Bond ETF (NYSE)",
        "HY Corporate (HYG)":     "HYG  — iShares iBoxx $ High Yield Corporate Bond ETF (NYSE) — short",
    },
    "Long TIP / Short TLT (Inflation Breakeven Trade)": {
        "TIPS / Inflation (TIP)":  "TIP  — iShares TIPS Bond ETF (NYSE)",
        "US 20Y+ Treasury (TLT)":  "TLT  — iShares 20+ Year Treasury Bond ETF (NYSE) — short",
    },
    "Long Brent Crude / Short Nifty 50 (India Import Shock)": {
        "Brent Crude":  "BNO  — United States Brent Oil Fund (NYSE)",
        "Nifty 50":     "INDY — iShares India 50 ETF (NYSE) | NIFTYBEES.NS (NSE)",
    },
    "Long Gold / Short INR (India Geopolitical Hedge)": {
        "Gold":    "GLD  — SPDR Gold Shares (NYSE)",
        "USD/INR": "USDINR=X — Forex spot | USDINR futures on NSE",
    },
    "Long EMB / Short DXY (Dollar Debasement - EM Relief)": {
        "EM USD Bonds (EMB)":  "EMB  — iShares J.P. Morgan USD EM Bond ETF (NYSE)",
        "DXY (Dollar Index)":  "UUP  — Invesco DB US Dollar Index Bullish Fund (NYSE) — short",
        "Gold":                "GLD  — SPDR Gold Shares (NYSE)",
    },
    "Long Gold / Short TLT (Fiscal Dominance / Dollar Debasement)": {
        "Gold":                    "GLD  — SPDR Gold Shares (NYSE) | GDX for miners leverage",
        "US 20Y+ Treasury (TLT)": "TLT  — iShares 20+ Year Treasury Bond ETF (NYSE) — short",
    },
    "Long EM Asia / Short DXY (Max Pessimism EM Reversal)": {
        "Shanghai Comp":      "MCHI — iShares MSCI China ETF (NYSE) | 2800.HK Tracker Fund",
        "Sensex":             "INDA — iShares MSCI India ETF (NYSE) | NIFTYBEES.NS",
        "DXY (Dollar Index)": "UUP  — Invesco DB US Dollar Index Bullish Fund (NYSE) — short",
    },
    "Long LQD / Short HYG (Credit Cycle Peak — Quality Flight)": {
        "IG Corporate (LQD)": "LQD  — iShares iBoxx $ IG Corporate Bond ETF (NYSE)",
        "HY Corporate (HYG)": "HYG  — iShares iBoxx $ HY Corporate Bond ETF (NYSE) — short",
    },
    "Long SHY / Long Gold (Fed Pivot Front-Run)": {
        "US 1-3Y Treasury (SHY)": "SHY  — iShares 1-3 Year Treasury Bond ETF (NYSE)",
        "Gold":                    "GLD  — SPDR Gold Shares (NYSE) | GDX for leveraged exposure",
    },
    "Long Nifty 50 / Short Brent (India Rate Cut + Oil Tailwind)": {
        "Nifty 50":     "INDA — iShares MSCI India ETF (NYSE) | NIFTYBEES.NS (NSE)",
        "Brent Crude":  "BNO  — United States Brent Oil Fund (NYSE) — short",
    },
    "Short Shanghai Comp / Long Nikkei 225 (China Deflation vs Japan Reflation)": {
        "Shanghai Comp": "FXI  — iShares China Large-Cap ETF (NYSE) — short via puts",
        "Nikkei 225":    "EWJ  — iShares MSCI Japan ETF (NYSE)",
    },
}


# ── Trade idea library ─────────────────────────────────────────────────────
_TRADE_LIBRARY = [
    {
        "regime":    [2, 3],
        "trigger":   "Elevated/Crisis correlation",
        "name":      "Long Gold / Short Eurostoxx 50",
        "rationale": "When cross-asset correlation spikes, equities and commodities sell off together. "
                     "Gold decouples as safe-haven demand absorbs panic flows. "
                     "Short European equities amplifies energy-cost transmission risk.",
        "entry":     "Enter when 60d avg cross-asset corr > 0.45 and DCC(Gold/SPX) < −0.1",
        "exit":      "Close when correlation regime drops back to Normal for 5+ days",
        "risk":      "Correlation snap-back; Central bank intervention can flip gold",
        "assets":    ["Gold", "Eurostoxx 50"],
        "direction": ["Long", "Short"],
        "category":  "Crisis Hedge",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Energy supply shock",
        "name":      "Long Natural Gas / Short Nikkei 225",
        "rationale": "Japan is the world's largest LNG importer. "
                     "Natural gas supply shocks directly impair Japanese manufacturing margins "
                     "and current account. Energy spikes translate to yen weakness and equity underperformance.",
        "entry":     "Enter on Ukraine escalation OR Strait of Hormuz closure signal + NG vol spike",
        "exit":      "Peace signal, supply restoration, or NG price mean-reversion (−20% from entry)",
        "risk":      "BOJ FX intervention; domestic LNG stockpile release",
        "assets":    ["Natural Gas", "Nikkei 225"],
        "direction": ["Long", "Short"],
        "category":  "Geopolitical",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Wheat/food supply disruption (Ukraine War)",
        "name":      "Long Wheat / Long Gold / Short Emerging Markets",
        "rationale": "Food price spikes trigger inflation in EM countries with high cereal import ratios. "
                     "Combined with USD strength (gold hedge), EM equities face dual pressure from "
                     "import inflation and capital outflows.",
        "entry":     "Wheat 30d return > +15% + political instability signals in MENA/SSA",
        "exit":      "Wheat normalises to 12M average; EM carry recovery",
        "risk":      "IMF/World Bank intervention; US export restrictions",
        "assets":    ["Wheat", "Gold", "Sensex"],
        "direction": ["Long", "Long", "Short"],
        "category":  "Macro",
    },
    {
        "regime":    [0, 1],
        "trigger":   "Global growth recovery (low correlation regime)",
        "name":      "Long Copper / Long S&P 500",
        "rationale": "Copper is the premier global growth bellwether. "
                     "When correlation is low, commodities and equities price "
                     "independent fundamentals - copper rising with equities signals "
                     "genuine demand expansion, not just liquidity.",
        "entry":     "Copper 60d momentum > 0, ISM Manufacturing > 50, cross-asset corr < 0.20",
        "exit":      "Copper momentum reversal; ISM contraction; Fed pause signals",
        "risk":      "China property market collapse; USD spike from geopolitical safe-haven",
        "assets":    ["Copper", "S&P 500"],
        "direction": ["Long", "Long"],
        "category":  "Growth",
    },
    {
        "regime":    [1, 2],
        "trigger":   "Oil-equity divergence (supply shock premium)",
        "name":      "Long WTI Crude / Short S&P 500 Energy-Heavy Sectors",
        "rationale": "When oil spikes from a supply shock (not demand), "
                     "energy futures gain while broader equities face margin compression. "
                     "The long/short captures the spread between commodity producer "
                     "and equity consumer dynamics.",
        "entry":     "Brent-WTI spread widens + OPEC+ surprise cut + SPX P/E compression",
        "exit":      "Supply restoration event; recession pricing dominates oil",
        "risk":      "Demand destruction flips correlation; tech-led equity rally decouples",
        "assets":    ["WTI Crude Oil", "S&P 500"],
        "direction": ["Long", "Short"],
        "category":  "Macro",
    },
    {
        "regime":    [3],
        "trigger":   "Full crisis - all correlations elevated",
        "name":      "Long Gold, Long Silver / Short Copper, Short Shanghai Comp",
        "rationale": "Full crisis regime: precious metals outperform as industrial metals "
                     "and EM equities (particularly China) collapse under dollar strength "
                     "and risk-off flows. Gold/Silver spread also captures "
                     "the industrial-vs-monetary metals divergence.",
        "entry":     "Crisis regime active > 3 days; VIX > 35; DXY trending up",
        "exit":      "Regime drops below Elevated; Fed emergency action; VIX < 25",
        "risk":      "Chinese stimulus; commodity demand front-running recovery",
        "assets":    ["Gold", "Silver", "Copper", "Shanghai Comp"],
        "direction": ["Long", "Long", "Short", "Short"],
        "category":  "Crisis Hedge",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Private credit stress / HY spread widening + BDC underperformance",
        "name":      "Short BDC Basket / Long HY Credit Protection",
        "rationale": "Private credit ($2T+ AUM) is illiquid and marked-to-model quarterly. "
                     "When HY OAS widens >35bps in 90 days and BKLN underperforms SPY, "
                     "BDC equity (ARCC, OBDC, FSK) reprices before private marks surface - "
                     "a lagged NAV writedown is typically confirmed 1–2 quarters later. "
                     "The CDX HY 5Y index provides liquid short exposure to the same credit universe. "
                     "Gold long captures contagion into safe-haven flows as financial sector sells off.",
        "entry":     "HY OAS >350bps and rising >30bps/month; BKLN below 200d MA; "
                     "BDC basket -5% vs SPY on 60d basis; elevated/crisis equity-commodity regime",
        "exit":      "HY OAS stabilises <300bps; Fed emergency rate cut signal; "
                     "BDC premiums re-compress; credit facilities extended at par",
        "risk":      "Fed emergency cuts collapse floating-rate costs rapidly; "
                     "sponsor rescue financing delays visible defaults; "
                     "interval fund gates mask redemption pressure for 12–18 months",
        "assets":    ["Ares Capital (ARCC)", "Blue Owl (OBDC)", "Gold"],
        "direction": ["Short", "Short", "Long"],
        "category":  "Private Credit",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Flight to quality - elevated/crisis correlation regime",
        "name":      "Long TLT / Short HYG (Flight to Quality)",
        "rationale": "In elevated and crisis correlation regimes, investors rotate from credit risk to duration safety. "
                     "TLT captures the safe-haven Treasury bid while HYG shorts the credit spread widening. "
                     "The trade isolates the quality spread compression that accompanies every risk-off episode.",
        "entry":     "Crisis/Elevated regime active; HY OAS rising >30bps in 30 days; VIX >25; TLT above 200d MA",
        "exit":      "Regime drops to Normal; HY OAS stabilises; Fed pivot signal",
        "risk":      "Bear steepening (long rates rise with HY spreads simultaneously); fiscal dominance narrative",
        "assets":    ["US 20Y+ Treasury (TLT)", "HY Corporate (HYG)"],
        "direction": ["Long", "Short"],
        "category":  "Fixed Income",
    },
    {
        "regime":    [1, 2],
        "trigger":   "Inflation breakeven expansion / stagflation risk",
        "name":      "Long TIP / Short TLT (Inflation Breakeven Trade)",
        "rationale": "When commodity prices spike (energy, food) and the Fed is behind the curve, "
                     "real yields compress while nominal yields stay elevated. "
                     "TIPS outperform nominal Treasuries as breakeven inflation widens. "
                     "This trade directly monetises the commodity-to-bond inflation transmission channel.",
        "entry":     "5Y breakeven inflation rising >20bps in 60 days; WTI or wheat up >15% in 30 days; CPI surprise positive",
        "exit":      "Breakeven inflation peaks; commodity prices mean-revert; Fed delivers credible inflation response",
        "risk":      "Demand destruction flips inflation to deflation; recession pricing overrides inflation premium",
        "assets":    ["TIPS / Inflation (TIP)", "US 20Y+ Treasury (TLT)"],
        "direction": ["Long", "Short"],
        "category":  "Fixed Income",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Oil spike + INR depreciation (India crude import stress)",
        "name":      "Long Brent Crude / Short Nifty 50 (India Import Shock)",
        "rationale": "India imports ~85% of its crude oil needs (~5 mb/d). "
                     "When Brent spikes >15% in 60 days and USD/INR is depreciating simultaneously, "
                     "India's current account deficit widens sharply - historically correlating with "
                     "Nifty 50 underperformance of -6% to -12% vs global peers. "
                     "The long Brent / short Nifty trade monetises the commodity-to-EM-equity transmission channel specific to India.",
        "entry":     "Brent up >15% in 60 days AND USD/INR rising >3% in 30 days; elevated/crisis regime active",
        "exit":      "Oil supply restored; INR stabilises / RBI intervention; India CAD narrows",
        "risk":      "RBI forex reserve intervention caps INR weakness; India domestic demand surprises to upside; OPEC+ production cut reversal",
        "assets":    ["Brent Crude", "Nifty 50"],
        "direction": ["Long", "Short"],
        "category":  "India/EM",
    },
    {
        "regime":    [2, 3],
        "trigger":   "Geopolitical stress + dollar strength - India gold import sensitivity",
        "name":      "Long Gold / Short INR (India Geopolitical Hedge)",
        "rationale": "India is the world's #2 gold consumer (~800-900 tonnes/year). "
                     "During geopolitical stress (Middle East escalation, South Asia tensions), "
                     "gold demand surges from Indian households as a safe-haven AND currency hedge. "
                     "Simultaneously, USD/INR typically rises (INR weakens) under global risk-off conditions. "
                     "The long gold / short INR (via USD/INR long) trade captures both legs of this transmission.",
        "entry":     "Crisis/Elevated regime; VIX >25; USD/INR rising >2% in 20 days; Gold 20d momentum positive",
        "exit":      "Geopolitical de-escalation; gold mean-reverts -8% from peak; INR stabilises on RBI action",
        "risk":      "RBI aggressively defends INR using $620B forex reserves; Gold sell-off on Fed hawkish surprise",
        "assets":    ["Gold", "USD/INR"],
        "direction": ["Long", "Long"],
        "category":  "India/EM",
    },
    {
        "regime":    [0, 1],
        "trigger":   "Dollar weakness cycle / EM credit rally",
        "name":      "Long EMB / Short DXY (Dollar Debasement - EM Relief)",
        "rationale": "A weakening dollar reduces dollar-denominated debt service costs for EM sovereigns, "
                     "compresses EM credit spreads, and attracts capital inflows into EM assets. "
                     "EMB captures the bond price appreciation; short DXY amplifies the currency leg. "
                     "Gold is the commodity expression of the same dollar debasement theme.",
        "entry":     "DXY down >3% in 60 days; Fed on hold or cutting; EM current accounts improving; commodity prices rising",
        "exit":      "Dollar reversal; Fed hawkish pivot; EM-specific credit event; commodity demand collapse",
        "risk":      "EM-specific credit events (sovereign default, political crisis); commodity demand collapse flips EM outlook",
        "assets":    ["EM USD Bonds (EMB)", "DXY (Dollar Index)", "Gold"],
        "direction": ["Long", "Short", "Long"],
        "category":  "Fixed Income",
    },

    # ── NEW: Macro-fundamental / largest-forces style trades ─────────────────
    {
        "regime":         [1, 2, 3],
        "trigger":        "US fiscal deficit >6% GDP + Fed balance sheet monetisation + real rates collapsing",
        "name":           "Long Gold / Short TLT (Fiscal Dominance / Dollar Debasement)",
        "rationale":      (
            "When the US fiscal deficit exceeds 6% of GDP and the Fed is monetising debt (balance sheet expanding), "
            "real rates collapse and fiat currency credibility erodes. Gold — the monetary metal — outperforms nominal "
            "Treasuries structurally. Druckenmiller: 'Liquidity drives markets above all else. When the Fed prints and "
            "fiscal expands, the answer is always gold over bonds.' Watsa: macro-hedging against systemic dollar "
            "debasement. Templeton: real returns focus — nominal treasury yields minus realized inflation = deeply "
            "negative real returns. The math is unambiguous. "
            "ETFs: Long GLD (SPDR Gold Shares), Short TLT (iShares 20+ Year Treasury Bond ETF). "
            "Leverage play: Long GDX (VanEck Gold Miners) for 2–3× operating leverage to gold price."
        ),
        "entry":          "US fiscal deficit >6% GDP AND WALCL expanding >5% in 90d AND gold 60d momentum positive AND 10Y real yield (DFII10) below −0.5%",
        "exit":           "Fed credibly tightens / balance sheet shrinks; gold −12% from entry peak; deficit narrows below 4%",
        "stop":           "GLD below 200d MA for 5+ days; TLT rallies >8% in 30d (fear-driven flight)",
        "target":         "GLD +20–35% over 12 months; TLT −15–20% as nominal yields re-price fiscal risk",
        "invalidation":   "Fed delivers credible quantitative tightening >$100B/month; Congress passes credible deficit reduction",
        "holding_period": "6–18 months (structural macro thesis)",
        "assets":         ["Gold", "US 20Y+ Treasury (TLT)"],
        "direction":      ["Long", "Short"],
        "category":       "Macro",
        "investor_lens":  ["Stanley Druckenmiller", "Prem Watsa", "John Templeton"],
    },
    {
        "regime":         [0, 1],
        "trigger":        "DXY at 3-year high + EM PMIs recovering + US current account deficit widening",
        "name":           "Long EM Asia / Short DXY (Max Pessimism EM Reversal)",
        "rationale":      (
            "Templeton's core principle: buy at the point of maximum pessimism. When the dollar peaks at 3-year highs "
            "and EM assets are universally abandoned, the fundamental equation reverses. EM GDP growth premium vs DM "
            "is widening; current accounts improving; dollar debasement is structural as US fiscal position deteriorates. "
            "Kerr Neilson: 'True global sourcing agility — allocate into regions when valuations become compelling.' "
            "Druckenmiller: the dollar cycle is the most powerful single force for EM asset re-rating. "
            "Instruments: Long MCHI (iShares MSCI China ETF) + INDA (iShares MSCI India ETF); Short UUP (Invesco DB "
            "USD Index Bullish Fund). Add EEM for broad EM exposure."
        ),
        "entry":          "DXY (DX-Y.NYB) at 3Y high and declining; EM Manufacturing PMI >50 for 3 consecutive months; FII outflows reversing; US 10Y real yield peaking",
        "exit":           "DXY reversal back above prior 3Y high; EM credit event; commodity demand collapse reversal",
        "stop":           "DXY breaks above prior 3Y high with momentum; Shanghai Comp −10% from entry on fresh stimulus failure",
        "target":         "EM Asia +25–40% over 18–24 months; DXY −8–12% as US fiscal dominance weakens the dollar",
        "invalidation":   "Fed re-accelerates rate hikes; EM sovereign credit event; China hard landing confirmed",
        "holding_period": "12–24 months",
        "assets":         ["Shanghai Comp", "Sensex", "DXY (Dollar Index)"],
        "direction":      ["Long", "Long", "Short"],
        "category":       "Dollar Cycle",
        "investor_lens":  ["John Templeton", "Kerr Neilson", "Stanley Druckenmiller"],
    },
    {
        "regime":         [1, 2],
        "trigger":        "HY–IG spread compression to cycle lows + leveraged loan issuance at 3Y high + Fed at terminal rate",
        "name":           "Long LQD / Short HYG (Credit Cycle Peak — Quality Flight)",
        "rationale":      (
            "Howard Marks: 'Gauge risk as permanent loss of capital, not volatility. Calibrate aggressiveness based on "
            "where the market stands in the credit cycle.' When HY–IG spreads compress to cycle lows (<200 bps), "
            "leveraged loan issuance is at records, and the Fed has reached terminal rate — credit cycle is at peak. "
            "Investment-grade bonds offer duration safety as HY reprices default risk. Marks: 'Asymmetry of returns — "
            "capture upside while protecting downside.' Klarman: forced institutional selling driven by credit rating "
            "downgrades is the Baupost entry point. "
            "ETFs: Long LQD (iShares iBoxx $ Investment Grade Corporate Bond ETF); "
            "Short HYG (iShares iBoxx $ High Yield Corporate Bond ETF). "
            "Alternative: CDX IG index long vs CDX HY index short for pure credit spread pair."
        ),
        "entry":          "HY OAS (BAMLH0A0HYM2) − IG OAS (BAMLC0A0CM) <200 bps AND leveraged loan issuance at 3Y high AND Fed funds held >6 months at cycle high",
        "exit":           "HY spreads widen to 400 bps; credit cycle turns; recession confirmed by 2Q negative GDP",
        "stop":           "LQD −5% from entry (bear-steepening scenario); HYG −8% in risk-off panic (correlations spike — exit both legs)",
        "target":         "LQD +8–12% as rates rally; HYG −15–25% as spreads blow out on first credit event",
        "invalidation":   "Fed pivots to cuts immediately; fiscal stimulus backstops credit markets; HY default rate stays below 2%",
        "holding_period": "6–18 months",
        "assets":         ["IG Corporate (LQD)", "HY Corporate (HYG)"],
        "direction":      ["Long", "Short"],
        "category":       "Fixed Income",
        "investor_lens":  ["Howard Marks", "Seth Klarman", "Prem Watsa"],
    },
    {
        "regime":         [1, 2, 3],
        "trigger":        "Fed funds at cycle peak + unemployment +0.5% from trough + 2Y–10Y curve bull-steepening",
        "name":           "Long SHY / Long Gold (Fed Pivot Front-Run)",
        "rationale":      (
            "Druckenmiller: 'Liquidity drives markets vastly above near-term earnings. Never fight the Fed.' When the "
            "Fed is at terminal rate, unemployment has risen 0.5% from trough, and the yield curve begins "
            "bull-steepening — the next macro move is rate cuts. 2Y Treasuries front-run the cut cycle with "
            "mathematical certainty (SHY +4–6% per 100 bps cut). Gold front-runs the real yield collapse and "
            "dollar weakness that follows easing. S Naren: 'Dynamic asset allocation — shift systematically to "
            "duration and gold at the cycle peak. Market Cap-to-GDP >1.0 confirms equity risk is too high.' "
            "ETFs: Long SHY (iShares 1-3 Year Treasury Bond ETF); Long GLD (SPDR Gold Shares). "
            "Tactical overlay: GDX (VanEck Gold Miners) for leveraged gold exposure on confirmed cut signals."
        ),
        "entry":          "Fed funds held >6 months at cycle high AND unemployment +0.5% from cycle trough AND 2Y–10Y curve bull-steepening AND gold above 200d MA",
        "exit":           "Fed delivers 100 bps+ of cuts (fully priced in); gold −10% from cycle high; 2Y yields stop declining",
        "stop":           "CPI re-accelerates above 3.5%; 2Y yields break back above Fed funds rate; unemployment stops rising",
        "target":         "SHY +4–6% on 100–150 bps cut cycle; GLD +20–40% over 12 months on real yield collapse",
        "invalidation":   "No recession; unemployment reverses before Fed cuts; inflation re-accelerates to >3.5%",
        "holding_period": "6–18 months",
        "assets":         ["US 1-3Y Treasury (SHY)", "Gold"],
        "direction":      ["Long", "Long"],
        "category":       "Macro",
        "investor_lens":  ["Stanley Druckenmiller", "S Naren", "Howard Marks"],
    },
    {
        "regime":         [0, 1],
        "trigger":        "RBI rate cut cycle + Brent <$80 + India CAD improving + USD/INR stabilising",
        "name":           "Long Nifty 50 / Short Brent (India Rate Cut + Oil Tailwind)",
        "rationale":      (
            "S Naren: 'Contra-cyclical value: accumulate quality Indian equities when the macro headwinds peak.' "
            "India imports ~85% of crude oil needs. When Brent falls below $80 AND the RBI begins cutting, "
            "the India macro equation flips structurally: CAD narrows, INR stabilises, real household incomes rise, "
            "and corporate margins expand. This is the inverse of the India Import Shock trade — it fires when oil "
            "stress reverses. Prashant Jain: 'Business longevity — ask whether the company grows cash flows over 10 "
            "years. India's structural consumption story is intact below $80 oil.' "
            "ETFs: Long INDA (iShares MSCI India ETF) or NIFTYBEES.NS (NSE); "
            "Short BNO (United States Brent Oil Fund) or BZ=F futures."
        ),
        "entry":          "Brent below $80 AND declining; RBI cuts >25 bps; USD/INR <84.5 and stable; Nifty 50 P/E <22x",
        "exit":           "Oil reversal above $90; RBI pauses; India CAD widens above 3% GDP; Nifty P/E >28x",
        "stop":           "Brent spikes above $90 on new supply shock; RBI reverses cuts; INR depreciates >5% in 30d",
        "target":         "Nifty 50 +20–30% over 12 months; short Brent +10–15% as oil normalises",
        "invalidation":   "India-Pakistan escalation resumes; US recession triggers global EM selloff",
        "holding_period": "6–12 months",
        "assets":         ["Nifty 50", "Brent Crude"],
        "direction":      ["Long", "Short"],
        "category":       "India/EM",
        "investor_lens":  ["S Naren", "Prashant Jain", "Ramdeo Agrawal"],
    },
    {
        "regime":         [2, 3],
        "trigger":        "China property developer debt crisis deepening + Japan BOJ policy normalisation + Yen below ¥148",
        "name":           "Short Shanghai Comp / Long Nikkei 225 (China Deflation vs Japan Reflation)",
        "rationale":      (
            "Shankar Sharma: 'Spotting inflection points early across cap ranges. China property is structural, not "
            "cyclical — $300B+ in offshore dollar debt with demographic reversal and structural oversupply.' Japan is "
            "the mirror image: Yen weakness (USD/JPY >148) boosts export profits, domestic reflation is accelerating, "
            "and the Nikkei Shiller CAPE at ~22x is cheap vs its own 30-year history. Kerr Neilson: 'True global "
            "sourcing agility — Japan valuations are compelling. China property is a regulatory and demographic trap. "
            "Rotate with conviction.' S Naren: 'Macro risk tracking: China leverage cycle is in systemic unwind. "
            "Mean reversion for Japan is supported by BOJ policy normalisation.' "
            "ETFs: Short FXI (iShares China Large-Cap ETF) via puts or inverse ETF; Long EWJ (iShares MSCI Japan ETF). "
            "Leverage: Short MCHI for broader China exposure; add DXJ (WisdomTree Japan Hedged Equity) if hedging Yen."
        ),
        "entry":          "FXI below 200d MA AND China property PMI <45 AND USD/JPY >148 AND Japan PMI >52 AND BOJ holds rates",
        "exit":           "China stimulus package >$500B announced; BOJ hikes >50 bps driving Yen to <130; FXI rallies >15%",
        "stop":           "FXI rallies >12% on China stimulus surprise; BOJ pivot pauses",
        "target":         "FXI −20–35% on debt restructuring events; EWJ +20–30% on Yen carry + reflation trade",
        "invalidation":   "China PBOC delivers massive credit stimulus; BOJ reverses course; global recession kills Japan exports",
        "holding_period": "6–24 months",
        "assets":         ["Shanghai Comp", "Nikkei 225"],
        "direction":      ["Short", "Long"],
        "category":       "Asia Divergence",
        "investor_lens":  ["Shankar Sharma", "Kerr Neilson", "S Naren"],
    },
]

_CATEGORY_COLORS = {
    "Crisis Hedge":    "#c0392b",
    "Geopolitical":    "#e67e22",
    "Macro":           "#2980b9",
    "Growth":          "#2e7d32",
    "Private Credit":  "#8e44ad",
    "Fixed Income":    "#2980b9",
    "India/EM":        "#16a085",
    "Dollar Cycle":    "#1abc9c",
    "Asia Divergence": "#9b59b6",
}

# ── S&P 500 stock universe for AI Trade Structurer ────────────────────────────
# ~150 S&P 500 members organised by macro-relevant sector.
# At call time, _select_sectors_for_signal() picks the 3-5 most relevant sectors
# for the current regime/scenario so the AI context stays lean (~40 stocks max).
# Format: ticker → (display_name, sector)
_SP500_UNIVERSE: dict[str, tuple[str, str]] = {
    # ── Energy: Integrated, E&P, Refining, OFS ──────────────────────────────
    "XOM":  ("ExxonMobil",            "Energy"),
    "CVX":  ("Chevron",               "Energy"),
    "COP":  ("ConocoPhillips",        "Energy"),
    "EOG":  ("EOG Resources",         "Energy"),
    "OXY":  ("Occidental Petroleum",  "Energy"),
    "DVN":  ("Devon Energy",          "Energy"),
    "HES":  ("Hess",                  "Energy"),
    "MRO":  ("Marathon Oil",          "Energy"),
    "APA":  ("APA Corp",              "Energy"),
    "FANG": ("Diamondback Energy",    "Energy"),
    "PSX":  ("Phillips 66",           "Energy"),
    "VLO":  ("Valero Energy",         "Energy"),
    "MPC":  ("Marathon Petroleum",    "Energy"),
    "SLB":  ("SLB (Schlumberger)",    "Energy"),
    "HAL":  ("Halliburton",           "Energy"),
    "BKR":  ("Baker Hughes",          "Energy"),
    "KMI":  ("Kinder Morgan",         "Energy"),
    "WMB":  ("Williams Companies",    "Energy"),
    "OKE":  ("ONEOK",                 "Energy"),
    # ── Defense & Aerospace ─────────────────────────────────────────────────
    "LMT":  ("Lockheed Martin",       "Defense"),
    "RTX":  ("RTX (Raytheon)",        "Defense"),
    "NOC":  ("Northrop Grumman",      "Defense"),
    "GD":   ("General Dynamics",      "Defense"),
    "BA":   ("Boeing",                "Defense"),
    "LHX":  ("L3Harris Technologies", "Defense"),
    "HII":  ("Huntington Ingalls",    "Defense"),
    "LDOS": ("Leidos",                "Defense"),
    "TDG":  ("TransDigm Group",       "Defense"),
    "AXON": ("Axon Enterprise",       "Defense"),
    # ── Airlines ────────────────────────────────────────────────────────────
    "DAL":  ("Delta Air Lines",       "Airlines"),
    "UAL":  ("United Airlines",       "Airlines"),
    "AAL":  ("American Airlines",     "Airlines"),
    "LUV":  ("Southwest Airlines",    "Airlines"),
    "ALK":  ("Alaska Air Group",      "Airlines"),
    # ── Gold & Precious Metals Mining ───────────────────────────────────────
    "NEM":  ("Newmont",               "Gold Mining"),
    "GOLD": ("Barrick Gold",          "Gold Mining"),
    "AEM":  ("Agnico Eagle",          "Gold Mining"),
    "WPM":  ("Wheaton Precious Metals","Gold Mining"),
    # ── Industrial Metals & Mining ──────────────────────────────────────────
    "FCX":  ("Freeport-McMoRan",      "Industrial Metals"),
    "NUE":  ("Nucor Steel",           "Industrial Metals"),
    "CLF":  ("Cleveland-Cliffs",      "Industrial Metals"),
    "X":    ("US Steel",              "Industrial Metals"),
    "AA":   ("Alcoa",                 "Industrial Metals"),
    "MP":   ("MP Materials",          "Industrial Metals"),
    "SCCO": ("Southern Copper",       "Industrial Metals"),
    # ── Agriculture & Fertilizers ───────────────────────────────────────────
    "MOS":  ("Mosaic (Fertilizers)",  "Agriculture"),
    "ADM":  ("Archer-Daniels-Midland","Agriculture"),
    "BG":   ("Bunge Global",          "Agriculture"),
    "CF":   ("CF Industries",         "Agriculture"),
    "CTVA": ("Corteva Agriscience",   "Agriculture"),
    "FMC":  ("FMC Corp",              "Agriculture"),
    # ── Technology ──────────────────────────────────────────────────────────
    "AAPL": ("Apple",                 "Tech"),
    "MSFT": ("Microsoft",             "Tech"),
    "NVDA": ("NVIDIA",                "Tech"),
    "GOOGL":("Alphabet (A)",          "Tech"),
    "META": ("Meta Platforms",        "Tech"),
    "AMZN": ("Amazon",                "Tech"),
    "TSLA": ("Tesla",                 "Tech"),
    "AMD":  ("Advanced Micro Devices","Tech"),
    "INTC": ("Intel",                 "Tech"),
    "QCOM": ("Qualcomm",              "Tech"),
    "AVGO": ("Broadcom",              "Tech"),
    "CRM":  ("Salesforce",            "Tech"),
    "ORCL": ("Oracle",                "Tech"),
    "NOW":  ("ServiceNow",            "Tech"),
    "ADBE": ("Adobe",                 "Tech"),
    # ── Financials ──────────────────────────────────────────────────────────
    "JPM":  ("JPMorgan Chase",        "Financials"),
    "BAC":  ("Bank of America",       "Financials"),
    "GS":   ("Goldman Sachs",         "Financials"),
    "MS":   ("Morgan Stanley",        "Financials"),
    "WFC":  ("Wells Fargo",           "Financials"),
    "C":    ("Citigroup",             "Financials"),
    "BLK":  ("BlackRock",             "Financials"),
    "SCHW": ("Charles Schwab",        "Financials"),
    "COF":  ("Capital One",           "Financials"),
    "AXP":  ("American Express",      "Financials"),
    "BX":   ("Blackstone",            "Financials"),
    "KKR":  ("KKR & Co",              "Financials"),
    # ── Healthcare ──────────────────────────────────────────────────────────
    "UNH":  ("UnitedHealth Group",    "Healthcare"),
    "LLY":  ("Eli Lilly",             "Healthcare"),
    "JNJ":  ("Johnson & Johnson",     "Healthcare"),
    "ABBV": ("AbbVie",                "Healthcare"),
    "MRK":  ("Merck",                 "Healthcare"),
    "PFE":  ("Pfizer",                "Healthcare"),
    "TMO":  ("Thermo Fisher",         "Healthcare"),
    "ABT":  ("Abbott Laboratories",   "Healthcare"),
    "BMY":  ("Bristol-Myers Squibb",  "Healthcare"),
    "AMGN": ("Amgen",                 "Healthcare"),
    "ISRG": ("Intuitive Surgical",    "Healthcare"),
    "VRTX": ("Vertex Pharmaceuticals","Healthcare"),
    # ── Consumer Staples (safe-haven) ───────────────────────────────────────
    "PG":   ("Procter & Gamble",      "Consumer Staples"),
    "KO":   ("Coca-Cola",             "Consumer Staples"),
    "PEP":  ("PepsiCo",               "Consumer Staples"),
    "WMT":  ("Walmart",               "Consumer Staples"),
    "COST": ("Costco",                "Consumer Staples"),
    "MO":   ("Altria Group",          "Consumer Staples"),
    "PM":   ("Philip Morris",         "Consumer Staples"),
    "CL":   ("Colgate-Palmolive",     "Consumer Staples"),
    "GIS":  ("General Mills",         "Consumer Staples"),
    "KR":   ("Kroger",                "Consumer Staples"),
    # ── Consumer Discretionary ──────────────────────────────────────────────
    "MCD":  ("McDonald's",            "Consumer Discretionary"),
    "SBUX": ("Starbucks",             "Consumer Discretionary"),
    "NKE":  ("Nike",                  "Consumer Discretionary"),
    "HD":   ("Home Depot",            "Consumer Discretionary"),
    "TGT":  ("Target",                "Consumer Discretionary"),
    "F":    ("Ford Motor",            "Consumer Discretionary"),
    "GM":   ("General Motors",        "Consumer Discretionary"),
    "TJX":  ("TJX Companies",         "Consumer Discretionary"),
    "LOW":  ("Lowe's",                "Consumer Discretionary"),
    "BKNG": ("Booking Holdings",      "Consumer Discretionary"),
    "RCL":  ("Royal Caribbean",       "Consumer Discretionary"),
    "CCL":  ("Carnival Corp",         "Consumer Discretionary"),
    # ── Industrials ─────────────────────────────────────────────────────────
    "CAT":  ("Caterpillar",           "Industrials"),
    "DE":   ("Deere & Company",       "Industrials"),
    "HON":  ("Honeywell",             "Industrials"),
    "GE":   ("GE Aerospace",          "Industrials"),
    "UPS":  ("United Parcel Service", "Industrials"),
    "FDX":  ("FedEx",                 "Industrials"),
    "ETN":  ("Eaton Corp",            "Industrials"),
    "EMR":  ("Emerson Electric",      "Industrials"),
    "PCAR": ("PACCAR",                "Industrials"),
    "MMM":  ("3M",                    "Industrials"),
    # ── Utilities (safe-haven, rate-sensitive) ───────────────────────────────
    "NEE":  ("NextEra Energy",        "Utilities"),
    "DUK":  ("Duke Energy",           "Utilities"),
    "SO":   ("Southern Company",      "Utilities"),
    "D":    ("Dominion Energy",       "Utilities"),
    "EXC":  ("Exelon",                "Utilities"),
    "AEP":  ("AEP",                   "Utilities"),
    "SRE":  ("Sempra Energy",         "Utilities"),
    # ── Materials ────────────────────────────────────────────────────────────
    "LIN":  ("Linde",                 "Materials"),
    "APD":  ("Air Products",          "Materials"),
    "SHW":  ("Sherwin-Williams",      "Materials"),
    "ECL":  ("Ecolab",                "Materials"),
    "DOW":  ("Dow Inc",               "Materials"),
    "DD":   ("DuPont",                "Materials"),
    "PPG":  ("PPG Industries",        "Materials"),
    # ── Real Estate ──────────────────────────────────────────────────────────
    "AMT":  ("American Tower",        "Real Estate"),
    "PLD":  ("Prologis",              "Real Estate"),
    "EQIX": ("Equinix",               "Real Estate"),
    "SPG":  ("Simon Property Group",  "Real Estate"),
    # ── Communications ───────────────────────────────────────────────────────
    "NFLX": ("Netflix",               "Communications"),
    "DIS":  ("Walt Disney",           "Communications"),
    "T":    ("AT&T",                  "Communications"),
    "VZ":   ("Verizon",               "Communications"),
    "CMCSA":("Comcast",               "Communications"),
}

# Reverse lookup: ticker → full company name (used by the card renderer)
_TICKER_NAMES: dict[str, str] = {t: name for t, (name, _) in _SP500_UNIVERSE.items()}

# Maps signal context → which sectors to pull for the AI (keeps prompt lean)
_SECTOR_SIGNAL_MAP: dict[str, list[str]] = {
    "supply_shock":    ["Energy", "Agriculture", "Industrial Metals", "Defense"],
    "escalation":      ["Energy", "Defense", "Gold Mining", "Airlines"],
    "sanctions_shock": ["Energy", "Defense", "Financials", "Industrial Metals"],
    "shipping_shock":  ["Energy", "Industrials", "Airlines", "Consumer Discretionary"],
    "risk_off":        ["Gold Mining", "Consumer Staples", "Utilities", "Defense"],
    "crisis":          ["Gold Mining", "Defense", "Consumer Staples", "Utilities"],
    "de_escalation":   ["Airlines", "Consumer Discretionary", "Tech", "Industrials"],
    "recovery":        ["Tech", "Consumer Discretionary", "Financials", "Industrials"],
    "base":            ["Tech", "Financials", "Healthcare", "Consumer Discretionary"],
    "default":         ["Energy", "Tech", "Financials", "Defense", "Healthcare"],
}

# Always included alongside signal-driven sectors (core reference)
_ANCHOR_SECTORS = ["Energy", "Tech", "Financials"]


def _select_sectors_for_signal(regime_level: int, scenario_id: str | None) -> list[str]:
    """Pick 4-6 relevant sectors based on regime and active scenario."""
    signal_sectors = _SECTOR_SIGNAL_MAP.get(scenario_id or "default",
                                             _SECTOR_SIGNAL_MAP["default"])
    # Crisis/Elevated regime always adds safe-haven and defense
    if regime_level >= 3:
        signal_sectors = list(dict.fromkeys(
            signal_sectors + ["Gold Mining", "Defense", "Consumer Staples", "Utilities"]
        ))
    elif regime_level >= 2:
        signal_sectors = list(dict.fromkeys(signal_sectors + ["Gold Mining", "Defense"]))

    # Merge with anchor sectors (no duplicates, preserve order)
    combined = list(dict.fromkeys(signal_sectors + _ANCHOR_SECTORS))
    return combined[:7]  # cap at 7 sectors ≈ 40-50 stocks


@st.cache_data(ttl=900, show_spinner=False, max_entries=3)
def _fetch_stock_prices(sectors: tuple[str, ...] = ()) -> dict[str, float]:
    """
    Fetch latest closing prices for S&P 500 universe stocks.
    If sectors is provided, fetches only those sectors; otherwise fetches all.
    Cached 15 min.
    """
    try:
        import yfinance as yf
        sector_set = set(sectors)
        tickers = [
            t for t, (_, s) in _SP500_UNIVERSE.items()
            if not sector_set or s in sector_set
        ]
        if not tickers:
            return {}
        raw = yf.download(tickers, period="5d", progress=False, auto_adjust=True)["Close"]
        if raw.empty:
            return {}
        latest = raw.ffill().iloc[-1]
        return {str(t): round(float(v), 2) for t, v in latest.items() if not np.isnan(v)}
    except Exception:
        return {}


def _format_stock_context(prices: dict[str, float], sectors: list[str]) -> str:
    """Format stock prices compactly by sector for the AI context block."""
    if not prices:
        return ""
    sector_set = set(sectors)
    lines: list[str] = [
        "S&P 500 STOCK REFERENCE PRICES (live — use these for specific entry/target/stop):"
    ]
    by_sector: dict[str, list[str]] = {}
    for ticker, (name, sector) in _SP500_UNIVERSE.items():
        if sector_set and sector not in sector_set:
            continue
        price = prices.get(ticker)
        if price is None:
            continue
        by_sector.setdefault(sector, []).append(f"{ticker} ${price:.2f}")
    for sector in sectors:
        items = by_sector.get(sector)
        if items:
            lines.append(f"  {sector}: {', '.join(items)}")
    return "\n".join(lines)


_TI_STYLE = """<style>
/* ── Trade Ideas — Design System ───────────────────────────────────────────
   Typography scale (matches shared.py + palette.py):
     0.50rem  JetBrains Mono  uppercase labels, badges, chips, dims
     0.52rem  JetBrains Mono  header labels (slightly heavier weight labels)
     0.63rem  DM Sans / Mono  cell values, strip values, secondary data
     0.70rem  DM Sans         body / rationale (matches _page_intro scale)
     0.81rem  DM Sans bold    card trade name (primary heading in card)
     0.94rem  JetBrains Mono  KPI number (page-level metrics)
   Colors: palette.py — TEXT #e8e9ed · TEXT_SOFT #c8c8c8 · TEXT_MUTED #b8b8b8
           LABEL #8890a1 · TICK #555960 · GOLD #CFB991
           BORDER #1e1e1e · BORDER2 #2a2a2a · CARD #0d0d0d · CARD2 #141414
──────────────────────────────────────────────────────────────────────────── */

/* Card shell */
.ti-card{border:1px solid #1e1e1e;background:#0d0d0d;margin-bottom:.6rem;overflow:hidden}

/* Card header */
.ti-hdr{background:#0a0a0a;border-bottom:1px solid #1e1e1e;padding:.45rem .9rem;
  display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap}
.ti-hdr-lbl{font-family:'JetBrains Mono',monospace;font-size:0.52rem;font-weight:700;
  letter-spacing:.14em;text-transform:uppercase;line-height:1.4}

/* Badges and pills */
.ti-badges{display:flex;gap:4px;align-items:center;flex-wrap:wrap}
.ti-pill{font-family:'JetBrains Mono',monospace;font-size:0.50rem;font-weight:700;
  padding:2px 6px;color:#fff}
.ti-badge{font-family:'JetBrains Mono',monospace;font-size:0.50rem;font-weight:700;
  padding:1px 6px;letter-spacing:.08em}

/* Card body */
.ti-body{padding:.65rem .9rem}
.ti-name{font-family:'DM Sans',sans-serif;font-size:0.81rem;font-weight:700;
  color:#e8e9ed;line-height:1.3;margin-bottom:4px}
.ti-dir{font-family:'JetBrains Mono',monospace;font-size:0.52rem;
  color:#8890a1;margin-bottom:6px;line-height:1.7}
.ti-tickers{font-family:'JetBrains Mono',monospace;font-size:0.56rem;
  margin-bottom:8px;line-height:1.9;display:flex;flex-wrap:wrap;gap:14px}

/* Meta row */
.ti-meta{display:flex;gap:10px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
.ti-lbl{font-family:'JetBrains Mono',monospace;font-size:0.50rem;font-weight:700;
  letter-spacing:.12em;text-transform:uppercase;color:#555960}
.ti-conf{font-family:'JetBrains Mono',monospace;font-size:0.69rem;font-weight:700}
.ti-qc{font-family:'JetBrains Mono',monospace;font-size:0.50rem;font-weight:700;
  padding:2px 7px;color:#fff}

/* Rationale */
.ti-rationale{font-family:'DM Sans',sans-serif;font-size:0.70rem;color:#b8b8b8;
  line-height:1.68;margin-bottom:10px}

/* Entry/Exit/Risk grid + extended fields grid */
.ti-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;margin-bottom:5px}
.ti-ext-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:5px;margin-top:5px}
.ti-cell{background:#080808;border:1px solid #1a1a1a;padding:.35rem .6rem}
.ti-cell-lbl{font-family:'JetBrains Mono',monospace;font-size:0.50rem;font-weight:700;
  letter-spacing:.12em;text-transform:uppercase;margin-bottom:3px}
.ti-cell-val{font-family:'DM Sans',sans-serif;font-size:0.63rem;color:#c8c8c8;line-height:1.45}

/* P&L / Backtest strip */
.ti-strip{display:flex;gap:14px;padding:5px .9rem;align-items:center;
  flex-wrap:wrap;border-top:1px solid #1a1a1a}
.ti-strip-tag{font-family:'JetBrains Mono',monospace;font-size:0.50rem;font-weight:700;
  letter-spacing:.12em;text-transform:uppercase;min-width:56px}
.ti-strip-val{font-family:'JetBrains Mono',monospace;font-size:0.63rem;font-weight:700}
.ti-strip-dim{font-family:'JetBrains Mono',monospace;font-size:0.50rem;
  color:#555960;margin-left:auto}

/* Why chips (pass-filter reasons) */
.ti-why{display:flex;gap:4px;flex-wrap:wrap;padding:4px .9rem .45rem}
.ti-why-chip{font-family:'JetBrains Mono',monospace;font-size:0.50rem;font-weight:700;
  padding:2px 7px;border:1px solid #1a3a1a;background:#080f08;color:#27ae60}

/* QC flags */
.ti-qc-flag{font-family:'JetBrains Mono',monospace;font-size:0.50rem;font-weight:700;
  padding:2px 7px;border:1px solid #3a2000;background:#100800;color:#e67e22}

/* Asset exposure cells */
.ti-exp{display:flex;gap:5px;flex-wrap:wrap;padding:4px .9rem .5rem}
.ti-exp-cell{background:#080808;border:1px solid #1a1a1a;padding:4px 8px;min-width:80px}
.ti-exp-name{font-family:'JetBrains Mono',monospace;font-size:0.50rem;color:#555960;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:110px;margin-bottom:2px}
.ti-exp-sas{font-family:'JetBrains Mono',monospace;font-size:0.56rem;font-weight:700}

/* Page-level KPI tiles */
.ti-kpi{border:1px solid #1e1e1e;padding:.6rem .85rem;background:#0d0d0d}
.ti-kpi-lbl{font-family:'JetBrains Mono',monospace;font-size:0.50rem;font-weight:700;
  text-transform:uppercase;letter-spacing:.14em;color:#CFB991;margin-bottom:4px}
.ti-kpi-val{font-family:'JetBrains Mono',monospace;font-size:0.94rem;font-weight:700}

/* Conflict/regime geo-bar */
.ti-geo-bar{background:#080808;border:1px solid #1e1e1e;padding:.5rem 1rem;
  margin-bottom:.65rem;display:flex;align-items:center;gap:14px;flex-wrap:wrap}

/* Master Investor Lens */
.ti-lens{background:#060606;border-top:1px solid #151500;padding:.5rem .9rem .4rem}
.ti-lens-hdr{font-family:'JetBrains Mono',monospace;font-size:0.50rem;font-weight:700;
  letter-spacing:.14em;color:#CFB991;text-transform:uppercase;margin-bottom:6px}
.ti-lens-row{display:flex;gap:10px;align-items:flex-start;
  padding:4px 0;border-bottom:1px solid #111}
.ti-lens-mgr{font-family:'JetBrains Mono',monospace;font-size:0.50rem;
  font-weight:700;color:#CFB991}
.ti-lens-arch{font-family:'JetBrains Mono',monospace;font-size:0.50rem;
  color:#555960;letter-spacing:.07em}
.ti-lens-quote{font-family:'DM Sans',sans-serif;font-size:0.63rem;
  color:#888;line-height:1.55}

/* Entrance animation */
@keyframes ti-card-in{0%{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.ti-card-anim{animation:ti-card-in .45s ease-out both}
</style>"""


def _render_investor_lens_strip(col, trade: dict) -> None:
    """Slim strip showing which money managers' philosophy validates this trade and why."""
    category     = trade.get("category", "Macro")
    custom_names = trade.get("investor_lens", [])

    # Build lens list: prefer trade-specific managers, else fall back to category map
    if custom_names:
        category_pool = _MASTER_INVESTOR_LENS.get(category, []) + sum(_MASTER_INVESTOR_LENS.values(), [])
        lens_data = [l for l in category_pool if l["manager"] in custom_names][:3]
        # pad with category defaults if we got fewer than requested
        if len(lens_data) < len(custom_names):
            seen = {l["manager"] for l in lens_data}
            for l in _MASTER_INVESTOR_LENS.get(category, []):
                if l["manager"] not in seen:
                    lens_data.append(l)
                    seen.add(l["manager"])
                    if len(lens_data) >= 3:
                        break
    else:
        lens_data = _MASTER_INVESTOR_LENS.get(category, [])[:3]

    if not lens_data:
        return

    items_html = ""
    for lens in lens_data:
        mgr     = lens.get("manager", "")
        arch    = lens.get("archetype", "")
        insight = lens.get("insight", "")
        items_html += (
            f'<div class="ti-lens-row">'
            f'<div style="min-width:160px;flex-shrink:0">'
            f'<span class="ti-lens-mgr">{mgr.upper()}</span><br>'
            f'<span class="ti-lens-arch">{arch}</span>'
            f'</div>'
            f'<span class="ti-lens-quote">&ldquo;{insight}&rdquo;</span>'
            f'</div>'
        )

    col.markdown(
        f'<div class="ti-lens">'
        f'<div class="ti-lens-hdr">Master Investor Lens</div>'
        + items_html
        + '</div>',
        unsafe_allow_html=True,
    )


def _parse_holding_days(trade: dict, default: int = 30) -> int:
    """
    Derive backtest holding window from the trade's stated holding_period field.
    Uses the lower bound of the range as the conservative test horizon.
    Caps at 252 days (1 year) to keep backtest samples meaningful.
    """
    import re
    hp = trade.get("holding_period", "")
    if not hp:
        return default
    nums = re.findall(r"\d+", hp.lower())
    if not nums:
        return default
    lo = int(nums[0])
    if "month" in hp.lower():
        return max(default, min(lo * 21, 252))
    if "week" in hp.lower():
        return max(default, lo * 5)
    return default


def _compute_leg_weights(
    trade: dict,
    asset_exposure: dict | None,
) -> list[float]:
    """
    Compute conviction-weighted allocation per trade leg.
    Weights incorporate:
      - Trade-level confidence score
      - Per-asset SAS (Structural Asset Score) from exposure data
      - Direction modifier: safe-haven assets weighted higher in Crisis regime
    Returns normalised list of floats that sum to 1.0.
    """
    assets     = trade.get("assets", [])
    directions = trade.get("direction", [])
    n          = len(assets)
    if n == 0:
        return []

    confidence = float(trade.get("confidence", 0.60))
    weights: list[float] = []

    for asset, direction in zip(assets, directions):
        base = 1.0 / n   # equal-weight base
        # SAS modifier: higher exposure → more conviction on the right directional side
        if asset_exposure and asset in asset_exposure:
            sas = float(asset_exposure[asset].get("sas", 50))
            asset_dir = asset_exposure[asset].get("direction", "neutral")
            # Align SAS with trade direction: long geo-risk assets get SAS boost on long legs
            if direction.lower() == "long" and asset_dir == "long_geo_risk":
                base *= (1 + sas / 200)
            elif direction.lower() == "short" and asset_dir == "safe_haven":
                base *= (1 + sas / 200)
            elif direction.lower() == "long" and asset_dir == "safe_haven":
                base *= (1 + sas / 300)  # smaller boost — safe haven long is defensive
        # Confidence modifier
        base *= (0.5 + confidence)
        weights.append(max(base, 1e-6))

    total = sum(weights)
    return [w / total for w in weights]


@st.cache_data(show_spinner=False, max_entries=3, ttl=86400)
def _backtest_trade(
    _all_r: pd.DataFrame,
    _regimes: pd.Series,
    trade_name: str,
    trigger_regimes: list[int],
    assets: list[str],
    directions: list[str],
    holding_days: int = 30,
    leg_weights: tuple[float, ...] | None = None,
    _len_hint: int = 0,  # cache-buster: pass len(_all_r) so date-range changes bust cache
) -> dict:
    """
    Historical backtest for a single trade idea.

    Signal: every time the regime enters one of `trigger_regimes`
    (first day of that regime), enter the trade.
    Hold for `holding_days` business days.

    Portfolio P&L:
      - Equal-weight if leg_weights is None.
      - Conviction-weighted (from _compute_leg_weights) if provided.
        Weights capture confidence score × SAS modifier per leg.
        Weighted backtest shows the approach as deployed, not a naive basket.

    Returns:
      n_signals, win_rate (%), avg_return (%), sharpe, max_drawdown (%),
      available_assets, missing_assets, weighted (bool flag)
    """
    # Only backtest assets available in the returns DataFrame
    avail = [(a, d) for a, d in zip(assets, directions) if a in _all_r.columns]
    if not avail or _all_r.empty or _regimes.empty:
        return {"n_signals": 0, "error": "Insufficient data"}

    # Build leg weights aligned to available assets
    if leg_weights is not None and len(leg_weights) == len(assets):
        # Re-index weights to only available assets and renormalise
        avail_w: list[float] = []
        for a, _ in avail:
            idx = assets.index(a)
            avail_w.append(leg_weights[idx])
        total_w = sum(avail_w) or 1.0
        avail_w = [w / total_w for w in avail_w]
        is_weighted = True
    else:
        avail_w = [1.0 / len(avail)] * len(avail)
        is_weighted = False

    # Align regime to returns index.
    # limit=20: forward-fill only up to 20 trading days (intra-week alignment).
    # Dates before the regime series begins, or gaps > 20 days, receive sentinel -1
    # so they are excluded from signal generation rather than mislabelled Normal.
    reg = _regimes.reindex(_all_r.index, method="ffill", limit=20).fillna(-1).astype(int)

    # Entry signal: first day of a qualifying regime
    in_regime    = reg.isin(trigger_regimes).astype(int)
    entry_signal = (in_regime.diff() == 1)       # rising edge = regime just entered
    entry_dates  = _all_r.index[entry_signal].tolist()

    if not entry_dates:
        return {"n_signals": 0, "error": "No entry signals in history"}

    trade_returns = []
    for entry in entry_dates:
        try:
            i_start = _all_r.index.get_loc(entry)
        except KeyError:
            continue
        i_end = min(i_start + holding_days, len(_all_r) - 1)
        if i_end <= i_start:
            continue
        window = _all_r.iloc[i_start: i_end]
        # Compute conviction-weighted portfolio return
        leg_rets = []
        for (asset, direction), w in zip(avail, avail_w):
            leg_ret = (1 + window[asset]).prod() - 1   # compound return over window
            signed  = leg_ret if direction.lower() == "long" else -leg_ret
            leg_rets.append(signed * w)
        if leg_rets:
            trade_returns.append(sum(leg_rets) * 100)  # in %

    if len(trade_returns) < 3:
        return {"n_signals": len(trade_returns), "error": "Too few signals to backtest"}

    tr = np.array(trade_returns)
    wins     = (tr > 0).sum()
    win_rate = wins / len(tr) * 100
    avg_ret  = float(tr.mean())
    std_ret  = float(tr.std()) if len(tr) > 1 else 1.0
    sharpe   = avg_ret / (std_ret + 1e-8) * np.sqrt(252 / holding_days)

    # Max drawdown: waterfall of cumulative returns
    cum  = np.cumprod(1 + tr / 100)
    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak) / (peak + 1e-8) * 100
    max_dd = float(dd.min())

    return {
        "n_signals":         len(tr),
        "win_rate":          round(win_rate, 1),
        "avg_return":        round(avg_ret, 2),
        "sharpe":            round(sharpe, 2),
        "max_drawdown":      round(max_dd, 2),
        "available_assets":  [a for a, _ in avail],
        "missing_assets":    [a for a, _ in zip(assets, directions) if a not in _all_r.columns],
        "weighted":          is_weighted,
        "leg_weights":       {a: round(w, 3) for (a, _), w in zip(avail, avail_w)},
    }


@st.cache_data(show_spinner=False, max_entries=1, ttl=3600)
def _thesis_stage3_cached(
    strat_name: str,
    conflict_id: str | None,
    regime_list: tuple[int, ...],
    assets: tuple[str, ...],
    directions: tuple[str, ...],
    predicted_sign_items: tuple[tuple[str, int], ...],
    horizon_days: int,
    _all_r: pd.DataFrame,
    _regimes: pd.Series | None,
    _len_hint: int = 0,  # cache-buster: pass len(_all_r) to bust on date-range change
) -> dict:
    """
    Cached Stage 3 confirmation for one thesis strategy.
    Returns a plain dict (serialisable) so Streamlit's cache can store it.
    """
    from src.analysis.thesis_engine import (
        ThesisBlock, ThesisStrategy, SignalSpec, run_stage3,
    )

    _predicted_sign = dict(predicted_sign_items)
    _thesis = ThesisBlock(
        shock="", tps_channels=[], conflict_id=conflict_id, chokepoint=None,
        predicted_sign=_predicted_sign, horizon_days=horizon_days, persistence="",
    )
    _signal = SignalSpec(
        assets=list(assets), direction=list(directions),
        regime=list(regime_list), holding_period=horizon_days, signal_vars=[],
    )
    _tmp = ThesisStrategy(name=strat_name, category="", thesis=_thesis, signal=_signal)
    _tmp.stage1_passed = True
    _tmp.stage2_passed = True

    _tmp = run_stage3(_tmp, _all_r, _regimes)
    conf = _tmp.confirmation
    if conf is None:
        return {
            "stage_passed": False, "track": "unknown", "confirmation_score": 0.0,
            "per_leg": {}, "irf_df_records": None, "regime_stats": None,
            "rejection_reason": "Stage 3 not computed",
        }

    return {
        "stage_passed":       conf.stage_passed,
        "sign_matched":       conf.sign_matched,
        "track":              conf.track,
        "confirmation_score": conf.confirmation_score,
        "per_leg":            conf.per_leg,
        "irf_df_records":     conf.irf_df.to_dict("records") if conf.irf_df is not None else None,
        "regime_stats":       conf.regime_stats,
        "rejection_reason":   conf.rejection_reason,
    }


@st.cache_data(show_spinner=False, ttl=86400, max_entries=1)
def _run_pipeline_validator_cached(
    _all_r: pd.DataFrame,
    _regimes: "pd.Series | None",
    train_days: int = 756,
    test_days: int = 63,
    n_strategies: int = 9,
    n_random_trials: int = 500,
) -> dict:
    """
    Walk-forward pipeline validation. Returns a serialisable dict.
    Builds the validation universe internally — these theses are the
    classification test set, not a ranked catalogue.
    """
    from src.analysis.thesis_engine import ThesisStrategy, ThesisBlock, SignalSpec
    from src.analysis.pipeline_validator import walk_forward_pipeline_validation

    def _ts(name, cat, shock, channels, conflict, chokepoint, pred,
            horizon, persistence, assets, directions, regime, hold, weights=None):
        return ThesisStrategy(
            name=name, category=cat,
            thesis=ThesisBlock(
                shock=shock, tps_channels=channels,
                conflict_id=conflict, chokepoint=chokepoint,
                predicted_sign=pred, horizon_days=horizon, persistence=persistence,
            ),
            signal=SignalSpec(
                assets=assets, direction=directions,
                regime=regime, holding_period=hold,
                signal_vars=[], leg_weights=weights,
            ),
        )

    universe = [
        _ts("Long Gold / Short Eurostoxx 50", "Crisis Hedge",
            "Cross-asset correlation spike, DCC(Gold/SPX) < −0.10.",
            ["equity_sector","fx","inflation"], None, None,
            {"Gold":+1,"Eurostoxx 50":-1}, 20, "Safe-haven demand persists until regime normalises.",
            ["Gold","Eurostoxx 50"],["Long","Short"],[2,3],20),

        _ts("Long Natural Gas / Short Nikkei 225", "Geopolitical",
            "Energy supply shock: Ukraine escalation OR Hormuz closure.",
            ["oil_gas","shipping","chokepoint","fx"],"ukraine_russia","Strait of Hormuz",
            {"Natural Gas":+1,"Nikkei 225":-1}, 15, "Japan largest LNG importer; yen weakens.",
            ["Natural Gas","Nikkei 225"],["Long","Short"],[2,3],15),

        _ts("Long Wheat / Long Gold / Short Emerging Markets", "Macro",
            "Ukraine war food supply disruption; Wheat 30d return > +15%.",
            ["agriculture","inflation","fx","equity_sector"],"ukraine_russia","Black Sea Grain Corridor",
            {"Wheat":+1,"Gold":+1,"Sensex":-1}, 20, "Food inflation structural; EM CAD deteriorates.",
            ["Wheat","Gold","Sensex"],["Long","Long","Short"],[2,3],20,[0.4,0.3,0.3]),

        _ts("Long Copper / Long S&P 500", "Growth",
            "Global growth recovery: Copper 60d momentum > 0, ISM > 50.",
            ["metals","equity_sector","supply_chain"],None,None,
            {"Copper":+1,"S&P 500":+1}, 30, "Copper leads earnings by 1-2 quarters.",
            ["Copper","S&P 500"],["Long","Long"],[0,1],30),

        _ts("Long WTI Crude / Short S&P 500", "Macro",
            "Oil supply shock: OPEC+ cut or Hormuz closure → Brent-WTI spread widens.",
            ["oil_gas","chokepoint","equity_sector","inflation"],"iran_conflict","Strait of Hormuz",
            {"WTI Crude Oil":+1,"S&P 500":-1}, 20, "Equity margin compression persists ~6-8 weeks.",
            ["WTI Crude Oil","S&P 500"],["Long","Short"],[1,2],20),

        _ts("Long Gold, Long Silver / Short Copper, Short Shanghai", "Crisis Hedge",
            "Full crisis: VIX > 35, DXY trending up — monetary vs industrial metals decouple.",
            ["metals","equity_sector","fx","credit"],"taiwan_strait",None,
            {"Gold":+1,"Silver":+1,"Copper":-1,"Shanghai Comp":-1}, 10,
            "Flight from industrial metals persists until VIX < 25.",
            ["Gold","Silver","Copper","Shanghai Comp"],["Long","Long","Short","Short"],[3],10,
            [0.3,0.2,0.25,0.25]),

        _ts("Long Brent Crude / Short Nifty 50", "India/EM",
            "Brent spike > 15% in 60d AND USD/INR depreciating > 3% in 30d.",
            ["oil_gas","fx","inflation","equity_sector"],"iran_conflict","Strait of Hormuz",
            {"Brent Crude":+1,"Nifty 50":-1}, 20, "India imports ~85% crude; CAD widens sequentially.",
            ["Brent Crude","Nifty 50"],["Long","Short"],[2,3],20),

        _ts("Long Nifty 50 / Short Brent Crude", "India/EM",
            "RBI rate cut cycle: Brent below $80 and declining; INR stable.",
            ["oil_gas","fx","inflation","equity_sector"],None,None,
            {"Nifty 50":+1,"Brent Crude":-1}, 30, "Oil below $80 → India CAD improves multi-quarter.",
            ["Nifty 50","Brent Crude"],["Long","Short"],[0,1],30),

        _ts("Short Shanghai Comp / Long Nikkei 225", "Asia Divergence",
            "China property crisis deepening AND Japan BOJ normalisation.",
            ["equity_sector","credit","fx","supply_chain"],"taiwan_strait",None,
            {"Shanghai Comp":-1,"Nikkei 225":+1}, 30,
            "China property crisis structural; Japan yen weakness self-reinforcing.",
            ["Shanghai Comp","Nikkei 225"],["Short","Long"],[2,3],30),
    ]

    result = walk_forward_pipeline_validation(
        theses=universe,
        returns=_all_r,
        regimes=_regimes if _regimes is not None else pd.Series(dtype=int),
        train_days=train_days,
        test_days=test_days,
        n_strategies=n_strategies,
        n_random_trials=n_random_trials,
    )

    return {
        "admitted_vs_rejected_gap":  result.admitted_vs_rejected_gap,
        "admitted_vs_rejected_pval": result.admitted_vs_rejected_pval,
        "admitted_vs_random_gap":    result.admitted_vs_random_gap,
        "random_p_value":            result.random_p_value,
        "random_distribution":       result.random_distribution,
        "n_windows":                 result.n_windows,
        "n_theses":                  result.n_theses,
        "passed":                    result.passed(),
        "buckets": {
            k: {
                "mean":      v.mean,
                "std":       v.std,
                "n_obs":     v.n_obs,
                "n_windows": v.n_windows,
            }
            for k, v in result.buckets.items()
        },
        # Decision trace for the worked example thesis (WTI / S&P Iran/Hormuz).
        # Filtered here so the worked example display never calls a separate backtest.
        "worked_example_trace": [
            {
                "window_idx":       d.window_idx,
                "train_end":        str(d.train_end.date()),
                "test_end":         str(d.test_end.date()),
                "stage3_confirmed": d.stage3_confirmed,
                "stage3_sign":      d.stage3_sign_matched,
                "decision":         d.pipeline_decision,
                "dsr_prob":         d.dsr_prob,
                "oos_return":       d.oos_mean_return,
                "oos_n_signals":    d.oos_n_signals,
            }
            for d in result.decisions
            if d.thesis_name == "Long WTI Crude / Short S&P 500"
        ],
    }


@st.cache_data(show_spinner=False, max_entries=3, ttl=86400)
def _wf_backtest_trade(
    _all_r: pd.DataFrame,
    _avg_corr: pd.Series,
    trade_name: str,          # included in cache key
    trigger_regimes: list[int],
    assets: list[str],
    directions: list[str],
    holding_days: int = 30,
    leg_weights: tuple[float, ...] | None = None,
    avg_corr_n: int = 0,     # sentinel: busts cache when avg_corr row count changes
    n_strategies: int = 9,
    is_economic_prior: bool = True,
) -> dict:
    """Cached walk-forward backtest for a single trade card."""
    from src.analysis.backtest import _N_LIBRARY_STRATEGIES
    trade_stub = {
        "name":           trade_name,
        "assets":         assets,
        "direction":      directions,
        "regime":         trigger_regimes,
        "holding_period": holding_days,
    }
    return walk_forward_backtest(
        returns=_all_r,
        avg_corr=_avg_corr,
        trade=trade_stub,
        leg_weights=list(leg_weights) if leg_weights else None,
        n_strategies=n_strategies,
        is_economic_prior=is_economic_prior,
    )


def _render_trade_card(
    col,
    trade: dict,
    all_r_concat: pd.DataFrame,
    current: int,
    trade_idx: int,
    asset_exposure: dict | None = None,
    regimes: "pd.Series | None" = None,
    avg_corr: "pd.Series | None" = None,
    _dup_registry: "dict | None" = None,
    n_strategies: int = 9,
    is_economic_prior: bool = True,
) -> None:
    """Render a single trade card with QC grade, confidence, payoff table, and debate thread."""
    # Suppress low-confidence generated ideas to keep the page signal-to-noise high
    _conf_raw = float(trade.get("confidence", 0.60))
    if trade.get("generated") and _conf_raw < 0.55:
        return

    cat_col = _CATEGORY_COLORS.get(trade["category"], "#CFB991")

    # ── QC scoring ─────────────────────────────────────────────────────────
    try:
        from src.analysis.trade_filter import score_trade_quality
        qc = score_trade_quality(trade)
    except Exception:
        qc = {"score": 60, "grade": "B", "flags": []}

    grade       = qc["grade"]
    qc_score    = qc["score"]
    grade_color = {"A": "#27ae60", "B": "#2980b9", "C": "#e67e22", "D": "#c0392b"}.get(grade, "#8890a1")
    confidence  = float(trade.get("confidence", 0.60))
    conf_pct    = f"{confidence * 100:.0f}%"
    conf_color  = "#27ae60" if confidence >= 0.70 else "#e67e22" if confidence >= 0.55 else "#c0392b"

    # ── Badges ─────────────────────────────────────────────────────────────
    is_generated  = trade.get("generated", False)
    conflict_id   = trade.get("conflict_id")
    source_badge  = (
        '<span class="ti-badge" style="background:#1a1a2e;color:#CFB991">LIVE GEO</span>'
        if is_generated else
        '<span class="ti-badge" style="background:#1e1e1e;color:#555960">STATIC</span>'
    )
    conflict_badge = (
        f'<span class="ti-badge" style="background:#3d1a00;color:#e67e22">'
        f'{conflict_id.upper().replace("_", " ")}</span>'
        if conflict_id else ""
    )

    dir_html = " &nbsp;·&nbsp; ".join(
        f'<span style="color:{"#27ae60" if d == "Long" else "#c0392b"};font-weight:700">'
        f'{d}</span>&nbsp;{a}'
        for a, d in zip(trade["assets"], trade["direction"])
    )

    # ── Specific tradeable instruments ─────────────────────────────────────
    # Priority: static _TRADE_TICKERS lookup → tickers dict on trade → asset name fallback
    _ticker_map_static = _TRADE_TICKERS.get(trade.get("name", ""), {})
    _ticker_map_gen    = trade.get("tickers") or {}
    # For AI-structured trades tickers is a freeform string; handle both str and dict
    if isinstance(_ticker_map_gen, str):
        _ticker_map_gen = {}
    _ticker_parts = []
    for _a, _d in zip(trade.get("assets", []), trade.get("direction", [])):
        _t = _ticker_map_static.get(_a) or _ticker_map_gen.get(_a)
        if _t:
            _clr  = "#27ae60" if _d.lower() == "long" else "#c0392b"
            _full = _TICKER_NAMES.get(_t, _a)  # "ExxonMobil" or fall back to asset name
            _ticker_parts.append(
                f'<span style="color:{_clr};font-weight:700;font-size:0.50rem">{_d}</span>'
                f'&nbsp;<span style="color:#e8e9ed;font-weight:600">{_full}</span>'
                f'&nbsp;<span style="color:#8890a1">({_t})</span>'
            )
    # AI-structured trades carry tickers as a single descriptive string — surface it directly
    _ai_tickers_str = trade.get("tickers", "") if isinstance(trade.get("tickers"), str) else ""
    if not _ticker_parts and _ai_tickers_str:
        _ticker_parts = [f'<span style="color:#c8c8c8">{_ai_tickers_str}</span>']
    ticker_html = (
        '<div class="ti-tickers">'
        + "  ".join(_ticker_parts)
        + '</div>'
    ) if _ticker_parts else ""

    regime_pills = "".join(
        f'<span class="ti-pill" style="background:{_REGIME_COLORS[r]}">{_REGIME_NAMES[r]}</span>'
        for r in trade.get("regime", [])
    )

    # ── Entry / Exit / Risk cells ───────────────────────────────────────────
    def _cell(lbl: str, val: str, lbl_col: str, extra: str = "") -> str:
        return (
            f'<div class="ti-cell"{(" " + extra) if extra else ""}>'
            f'<div class="ti-cell-lbl" style="color:{lbl_col}">{lbl}</div>'
            f'<div class="ti-cell-val">{val or "—"}</div>'
            f'</div>'
        )

    grid_html = (
        '<div class="ti-grid">'
        + _cell("Entry",  trade.get("entry", "—"), "#CFB991")
        + _cell("Exit",   trade.get("exit",  "—"), "#8890a1")
        + _cell("Risk",   trade.get("risk",  "—"), "#c0392b", 'style="border-left:2px solid #220000"')
        + '</div>'
    )

    # ── AI-structured specific fields (entry_price_ref, upside_pct, stop_loss, options_structure)
    ai_price_html = ""
    _ai_entry   = trade.get("entry_price_ref", "")
    _ai_upside  = trade.get("upside_pct", "")
    _ai_stop    = trade.get("stop_loss", "")
    _ai_opts    = trade.get("options_structure", "")
    _gen_upside = trade.get("upside_pct")  # numeric from generator
    if any([_ai_entry, _ai_upside, _ai_stop, _ai_opts]):
        ai_price_html = (
            '<div class="ti-ext-grid" style="border-top:1px solid #1a1a1a;margin-top:4px">'
            + (_cell("Entry Ref",    _ai_entry,  "#CFB991") if _ai_entry  else "")
            + (_cell("Upside",       _ai_upside, "#27ae60") if _ai_upside else "")
            + (_cell("Stop-Loss",    _ai_stop,   "#c0392b") if _ai_stop   else "")
            + (_cell("Options Alt.", _ai_opts,   "#2980b9") if _ai_opts   else "")
            + '</div>'
        )
    elif isinstance(_gen_upside, (int, float)):
        ai_price_html = (
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
            'color:#8890a1;margin-top:4px">'
            f'Est. upside: <span style="color:#27ae60;font-weight:700">~{_gen_upside:.1f}%</span>'
            '</div>'
        )

    ext_html = ""
    if any(trade.get(k) for k in ["stop", "target", "invalidation", "holding_period"]):
        ext_html = (
            '<div class="ti-ext-grid">'
            + _cell("Stop",         trade.get("stop",           "—"), "#27ae60")
            + _cell("Target",       trade.get("target",         "—"), "#27ae60")
            + _cell("Invalidation", trade.get("invalidation",   "—"), "#2980b9")
            + _cell("Hold Period",  trade.get("holding_period", "—"), "#8890a1")
            + '</div>'
        )

    card_delay = f"{trade_idx * 0.06:.2f}s"

    with col:
        st.markdown(
            f'<div class="ti-card ti-card-anim" '
            f'style="border-left:3px solid {cat_col};animation-delay:{card_delay}">'
            # ── Header ──────────────────────────────────────────────────────
            f'<div class="ti-hdr">'
            f'<span class="ti-hdr-lbl" style="color:{cat_col}">'
            f'{trade["category"]} &nbsp;·&nbsp; {trade["trigger"]}</span>'
            f'<div class="ti-badges">{regime_pills} {source_badge} {conflict_badge}</div>'
            f'</div>'
            # ── Body ────────────────────────────────────────────────────────
            f'<div class="ti-body">'
            f'<div class="ti-name">{trade["name"]}</div>'
            f'<div class="ti-dir">{dir_html}</div>'
            + ticker_html
            + f'<div class="ti-meta">'
            f'<span class="ti-lbl">Confidence</span>'
            f'<span class="ti-conf" style="color:{conf_color}">{conf_pct}</span>'
            f'<span class="ti-lbl" style="margin-left:6px">QC</span>'
            f'<span class="ti-qc" style="background:{grade_color}">{grade} &middot; {qc_score}</span>'
            f'</div>'
            f'<p class="ti-rationale">{trade["rationale"]}</p>'
            + grid_html
            + ai_price_html
            + ext_html
            + '</div>'  # ti-body
            + '</div>',  # ti-card
            unsafe_allow_html=True,
        )

        # ── Projected P&L strip ───────────────────────────────────────────
        try:
            from src.analysis.profit_projection import project_trade
            _proj   = project_trade(trade, current_regime=current)
            _epnl   = _proj["expected_pnl"]
            _wpnl   = _proj["worst_case_pnl"]
            _bprob  = _proj["breakeven_prob"]
            _sharpe = _proj["sharpe_proxy"]
            _ec = "#27ae60" if _epnl >= 0 else "#c0392b"
            _wc = "#c0392b" if _wpnl < -5  else "#e67e22"
            with col:
                st.markdown(
                    f'<div class="ti-strip" style="background:#080808">'
                    f'<span class="ti-strip-tag" style="color:#CFB991">Projected</span>'
                    f'<span class="ti-strip-val" style="color:{_ec}">E[P&L]&nbsp;{_epnl:+.1f}%</span>'
                    f'<span class="ti-strip-val" style="color:{_wc}">Worst&nbsp;{_wpnl:+.1f}%</span>'
                    f'<span class="ti-strip-val" style="color:#8890a1">BE&nbsp;{_bprob * 100:.0f}%</span>'
                    f'<span class="ti-strip-val" style="color:#CFB991">Sharpe&nbsp;{_sharpe:.2f}</span>'
                    f'<span class="ti-strip-dim">model estimate</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        except Exception:
            pass

        # ── Historical backtest strip (conviction-weighted) ───────────────
        try:
            if regimes is not None and not all_r_concat.empty:
                # Compute conviction weights for this trade
                _leg_w = _compute_leg_weights(trade, asset_exposure)
                _leg_w_tuple = tuple(_leg_w) if _leg_w else None
                _bt = _backtest_trade(
                    all_r_concat,
                    regimes,
                    trade_name=trade["name"],
                    trigger_regimes=trade.get("regime", [2, 3]),
                    assets=trade.get("assets", []),
                    directions=trade.get("direction", []),
                    holding_days=_parse_holding_days(trade),
                    leg_weights=_leg_w_tuple,
                    _len_hint=len(all_r_concat),
                )
                if "error" not in _bt and _bt.get("n_signals", 0) >= 3:
                    _bt_wr = _bt["win_rate"]
                    _bt_ar = _bt["avg_return"]
                    _bt_sh = _bt["sharpe"]
                    _bt_dd = _bt["max_drawdown"]
                    _bt_n  = _bt["n_signals"]
                    _bt_wt = _bt.get("weighted", False)
                    _bt_lw = _bt.get("leg_weights", {})
                    _wr_c = "#27ae60" if _bt_wr >= 55 else "#e67e22" if _bt_wr >= 45 else "#c0392b"
                    _ar_c = "#27ae60" if _bt_ar >= 0 else "#c0392b"
                    _sh_c = "#27ae60" if _bt_sh >= 0.5 else "#e67e22" if _bt_sh >= 0 else "#c0392b"
                    _wt_label = "conviction-wtd" if _bt_wt else "equal-wtd"
                    # Leg weight tooltip string
                    _wt_str = " · ".join(
                        f"{a[:8]}:{w:.0%}" for a, w in list(_bt_lw.items())[:3]
                    ) if _bt_lw else ""
                    with col:
                        st.markdown(
                            f'<div class="ti-strip" style="background:#050a05;'
                            f'border-top:1px solid #1a2a1a">'
                            f'<span class="ti-strip-tag" style="color:#27ae60">Backtest</span>'
                            f'<span class="ti-strip-val" style="color:{_wr_c}">Win&nbsp;{_bt_wr:.0f}%</span>'
                            f'<span class="ti-strip-val" style="color:{_ar_c}">Avg&nbsp;{_bt_ar:+.1f}%</span>'
                            f'<span class="ti-strip-val" style="color:{_sh_c}">Sharpe&nbsp;{_bt_sh:.2f}</span>'
                            f'<span class="ti-strip-val" style="color:#c0392b">MaxDD&nbsp;{_bt_dd:.1f}%</span>'
                            f'<span class="ti-strip-dim">{_bt_n} signals · {_parse_holding_days(trade)}d · {_wt_label}'
                            + (f' · {_wt_str}' if _wt_str else '')
                            + '</span></div>',
                            unsafe_allow_html=True,
                        )
        except Exception:
            pass

        # ── QC flags ─────────────────────────────────────────────────────
        if qc["flags"]:
            with col:
                st.markdown(
                    '<div style="display:flex;gap:4px;flex-wrap:wrap;padding:4px .9rem">'
                    + "".join(
                        f'<span class="ti-qc-flag">⚠ {f}</span>'
                        for f in qc["flags"]
                    )
                    + '</div>',
                    unsafe_allow_html=True,
                )

        # ── Why this trade passed filters ─────────────────────────────────
        try:
            _pass_reasons: list[str] = []
            if trade.get("generated"):
                _pass_reasons.append("Live GEO signal")
            if trade.get("conflict_id"):
                _cid = trade["conflict_id"].replace("_", " ").title()
                _pass_reasons.append(f"Conflict: {_cid}")
            _pass_reasons.append(f"Regime {_REGIME_NAMES.get(current, current)}")
            if confidence >= 0.70:
                _pass_reasons.append(f"Conf {confidence * 100:.0f}%")
            _cat = trade.get("category", "")
            if _cat and _cat != "all":
                _pass_reasons.append(_cat)
            if asset_exposure:
                _trade_assets = trade.get("assets", [])
                _sas_vals = [asset_exposure[a]["sas"] for a in _trade_assets if a in asset_exposure]
                if _sas_vals:
                    _avg_sas = sum(_sas_vals) / len(_sas_vals)
                    if _avg_sas >= 60:
                        _pass_reasons.append(f"SAS {_avg_sas:.0f} — high exposure")
                    elif _avg_sas >= 35:
                        _pass_reasons.append(f"SAS {_avg_sas:.0f}")
                _hedge_scores = [asset_exposure[a]["hedge_score"] for a in _trade_assets if a in asset_exposure]
                if any(h >= 40 for h in _hedge_scores):
                    _pass_reasons.append("Hedge signal active")
                _directions = [asset_exposure[a]["direction"] for a in _trade_assets if a in asset_exposure]
                if "safe_haven" in _directions and any(d.lower() == "long" for d in trade.get("direction", [])):
                    _pass_reasons.append("Safe-haven demand")
            if _pass_reasons:
                with col:
                    st.markdown(
                        '<div class="ti-why">'
                        + "".join(
                            f'<span class="ti-why-chip">✓ {r}</span>'
                            for r in _pass_reasons
                        )
                        + '</div>',
                        unsafe_allow_html=True,
                    )
        except Exception:
            pass

        # ── Asset exposure strip ──────────────────────────────────────────
        if asset_exposure:
            try:
                _exp_items = []
                for _a, _d in zip(trade.get("assets", []), trade.get("direction", [])):
                    _ed = asset_exposure.get(_a)
                    if not _ed:
                        continue
                    _sas      = _ed["sas"]
                    _dir      = _ed["direction"]
                    _top_c    = _ed.get("top_conflict") or ""
                    _top_beta = _ed["beta"].get(_top_c, 0.0) if _top_c else 0.0
                    _dir_icon = "↑" if _dir == "long_geo_risk" else "↓" if _dir == "safe_haven" else "→"
                    _sas_col  = "#e67e22" if _sas >= 60 else "#CFB991" if _sas >= 30 else "#555960"
                    _beta_str = f"β {_top_beta:.2f}" if _top_c else ""
                    _exp_items.append(
                        f'<div class="ti-exp-cell">'
                        f'<div class="ti-exp-name">{_a[:18]}</div>'
                        f'<div style="display:flex;gap:6px;align-items:center">'
                        f'<span class="ti-exp-sas" style="color:{_sas_col}">SAS {_sas:.0f}</span>'
                        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.56rem;'
                        f'color:#8890a1">{_dir_icon}</span>'
                        + (f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
                           f'color:#555960">{_beta_str}</span>' if _beta_str else "")
                        + '</div></div>'
                    )
                if _exp_items:
                    with col:
                        st.markdown(
                            '<div class="ti-exp">' + "".join(_exp_items) + '</div>',
                            unsafe_allow_html=True,
                        )
            except Exception:
                pass

        # ── Master Investor Lens ──────────────────────────────────────────
        _render_investor_lens_strip(col, trade)

        # ── Payoff table expander ─────────────────────────────────────────
        with col:
            with st.expander(f"Scenario Payoff — {trade['name'][:40]}", expanded=False):
                try:
                    from src.analysis.profit_projection import project_trade
                    proj    = project_trade(trade)
                    p_table = proj["payoff_table"]

                    pm1, pm2, pm3, pm4 = st.columns(4)
                    pm1.metric("Exp. P&L",       f"{proj['expected_pnl']:+.1f}%")
                    pm2.metric("Worst Case",      f"{proj['worst_case_pnl']:+.1f}%")
                    pm3.metric("Breakeven Prob",  f"{proj['breakeven_prob'] * 100:.0f}%")
                    pm4.metric("Sharpe Proxy",    f"{proj['sharpe_proxy']:.2f}")

                    sc_labels = [r["label"] for r in p_table]
                    sc_pnls   = [r["expected_pnl"] for r in p_table]
                    sc_colors = ["#27ae60" if pnl >= 0 else "#c0392b" for pnl in sc_pnls]
                    fig_pf = go.Figure(go.Bar(
                        x=sc_labels, y=sc_pnls,
                        marker_color=sc_colors,
                        text=[f"{v:+.1f}%" for v in sc_pnls],
                        textposition="outside",
                    ))
                    fig_pf.update_layout(
                        template="plotly_dark", height=220,
                        title=dict(text="Expected P&L by Scenario (%)", font=dict(size=11)),
                        margin=dict(l=40, r=20, t=36, b=40),
                        yaxis=dict(title="P&L %", zeroline=True, zerolinecolor="#333", zerolinewidth=1),
                        showlegend=False,
                        plot_bgcolor="#0d0d0d", paper_bgcolor="#0d0d0d",
                    )
                    _chart(fig_pf)

                    pt_df = pd.DataFrame([{
                        "Scenario": r["label"],
                        "Prob":     f"{r['prob'] * 100:.0f}%",
                        "Exp. P&L": f"{r['expected_pnl']:+.1f}%",
                        "Vol":      f"{r['vol']:.1f}%",
                        "Wtd P&L":  f"{r['prob_weighted_pnl']:+.2f}%",
                        "Active":   "★" if r["is_current"] else "",
                    } for r in p_table])
                    st.dataframe(pt_df, width="stretch", hide_index=True)

                except Exception as exc:
                    st.caption("Payoff projection unavailable — see logs.")

        # ── Walk-forward backtest expander ────────────────────────────────
        try:
            if avg_corr is not None and not all_r_concat.empty:
                _n_strategies = 9    # strategies with all declared legs in return data (data-integrity audit)
                _leg_w_wf = _compute_leg_weights(trade, asset_exposure)
                _leg_w_wf_tuple = tuple(_leg_w_wf) if _leg_w_wf else None
                _wfbt = _wf_backtest_trade(
                    all_r_concat,
                    avg_corr,
                    trade_name=trade["name"],
                    trigger_regimes=trade.get("regime", [2, 3]),
                    assets=trade.get("assets", []),
                    directions=trade.get("direction", []),
                    holding_days=_parse_holding_days(trade),
                    leg_weights=_leg_w_wf_tuple,
                    avg_corr_n=len(avg_corr),
                    n_strategies=n_strategies,
                    is_economic_prior=is_economic_prior,
                )
                # ── Duplicate detection ───────────────────────────────────────
                # Two strategies are considered duplicates when their OOS
                # trade-return series are identical (same assets resolve, same
                # regime, same holding period). We use (n_trades, sharpe, hit_rate)
                # as a lightweight signature; a full element-wise comparison is
                # deferred to the diagnostic script.
                _dup_of: str | None = None
                if _dup_registry is not None and _wfbt.get("n_trades", 0) >= 3:
                    _dup_sig = (
                        _wfbt.get("n_trades"),
                        round(float(_wfbt.get("sharpe") or 0), 3),
                        _wfbt.get("hit_rate"),
                    )
                    if _dup_sig in _dup_registry:
                        _dup_of = _dup_registry[_dup_sig]
                    else:
                        _dup_registry[_dup_sig] = trade["name"]

                _wf_qc    = _wfbt.get("qc", {})
                _wf_grade = _wf_qc.get("grade", "D")
                _wf_score = _wf_qc.get("score", 0)
                _dsr_prob = _wf_qc.get("dsr_prob", 0.0)
                _wf_decay = _wf_qc.get("decay")
                _is_sh    = _wf_qc.get("is_sharpe")
                _pbo_val  = _wf_qc.get("pbo")   # CSCV PBO (from qc dict)
                _GC = {"A": "#27ae60", "B": "#2980b9", "C": "#e67e22",
                       "D": "#c0392b",  "F": "#6c0000"}
                _gc = _GC.get(_wf_grade, "#555960")
                _has_result   = "error" not in _wfbt and _wfbt.get("n_trades", 0) >= 3
                _missing_legs = _wfbt.get("missing_legs", [])

                _dsr_pct = f"{_dsr_prob:.0%}" if _has_result else "─"
                _pbo_pct = f"{_pbo_val:.0%}" if (_has_result and _pbo_val is not None) else None
                _wf_label = (
                    (f"Backtest (Walk-Forward OOS) — {_wf_grade} · DSR {_dsr_pct}"
                     + (f" · PBO {_pbo_pct}" if _pbo_pct else ""))
                    if _has_result else
                    ("Backtest (Walk-Forward OOS) — MISSING DATA"
                     if _missing_legs else
                     "Backtest (Walk-Forward OOS)")
                )
                with col:
                    with st.expander(_wf_label, expanded=False):
                        if _missing_legs:
                            _dropped_str = ", ".join(_missing_legs)
                            _present_str = ", ".join(
                                a for a in trade.get("assets", []) if a not in _missing_legs
                            ) or "none"
                            st.markdown(
                                f'<div style="background:#1a0000;border:1px solid #c0392b;'
                                f'border-radius:4px;padding:8px 12px;margin-bottom:8px;'
                                f'font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;color:#e74c3c">'
                                f'<b>MISSING LEGS — NOT GRADEABLE</b><br>'
                                f'Declared: {", ".join(trade.get("assets", []))}<br>'
                                f'Present in return data: {_present_str}<br>'
                                f'Absent: <b>{_dropped_str}</b><br>'
                                f'Previous behavior silently traded the subset. '
                                f'A strategy that cannot execute all declared legs '
                                f'is mislabeled and has been excluded from grading.</div>',
                                unsafe_allow_html=True,
                            )
                        elif not _has_result:
                            st.caption(_wfbt.get("error", "Backtest unavailable"))
                        else:
                            # Duplicate strategy banner
                            if _dup_of is not None:
                                st.markdown(
                                    f'<div style="background:#0d1a2a;border:1px solid #2980b9;'
                                    f'border-radius:4px;padding:6px 10px;margin-bottom:8px;'
                                    f'font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;color:#5dade2">'
                                    f'DUPLICATE DETECTED — trade-return series is identical to '
                                    f'<b>{_dup_of[:50]}</b>. '
                                    f'Non-Gold leg absent from return data. '
                                    f'Counts as ×1 unique strategy in N for DSR multiple-testing correction.</div>',
                                    unsafe_allow_html=True,
                                )

                            # LOW N warning
                            _low_n = _wf_qc.get("low_confidence", False)
                            if _low_n:
                                _n_actual = _wfbt.get("n_trades", 0)
                                st.markdown(
                                    f'<div style="background:#2a1f00;border:1px solid #e67e22;'
                                    f'border-radius:4px;padding:6px 10px;margin-bottom:8px;'
                                    f'font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;color:#e67e22">'
                                    f'LOW N — {_n_actual} trades (need ≥20). '
                                    f'Sharpe SE is too wide for A/B. Grade capped at C.</div>',
                                    unsafe_allow_html=True,
                                )

                            # ── Grade chip + DSR robustness strip ────────────────
                            _bt_cols = st.columns([1, 1, 1, 1, 1, 1])
                            _gc_display = _gc if not _low_n else "#e67e22"
                            _bt_cols[0].markdown(
                                f'<div style="text-align:center;padding:6px 0">'
                                f'<span style="font-size:1.6rem;font-weight:700;color:{_gc_display}">{_wf_grade}</span>'
                                f'<br><span style="font-size:0.60rem;color:#8890a1">DSR {_dsr_pct}'
                                + (' · LOW N' if _low_n else '')
                                + f'</span></div>',
                                unsafe_allow_html=True,
                            )
                            _bt_cols[1].metric("OOS Sharpe", f"{_wfbt['sharpe']:.2f}")
                            _bt_cols[2].metric("Max DD",     f"{_wfbt['max_drawdown']:.1f}%")
                            _bt_cols[3].metric("Hit Rate",   f"{_wfbt['hit_rate']:.0f}%")
                            _bt_cols[4].metric("Trades",     str(_wfbt["n_trades"]))
                            _bt_cols[5].metric(
                                "W/L",
                                f"{_wfbt.get('win_loss_ratio', 0):.2f}",
                                help="Avg win / |Avg loss|",
                            )

                            # ── Robustness strip: DSR, IS Sharpe, decay, PBO ─────
                            _sr_star  = _wf_qc.get("sr_star", 0.0)
                            _dec_str  = f"{_wf_decay:.0%}" if _wf_decay is not None else "n/a"
                            _is_str   = f"{_is_sh:.2f}"    if _is_sh   is not None else "n/a"
                            _dec_col  = ("#c0392b" if (_wf_decay or 0) > 0.70
                                         else "#e67e22" if (_wf_decay or 0) > 0.40
                                         else "#27ae60")
                            # PBO: green <30%, yellow 30–50%, red >50% (grade-gating threshold)
                            _pbo_col  = ("#c0392b" if (_pbo_val or 0) > 0.50
                                         else "#e67e22" if (_pbo_val or 0) > 0.30
                                         else "#27ae60")
                            _pbo_str  = f"{_pbo_val:.0%}" if _pbo_val is not None else "n/a"
                            _n_cscv   = _wfbt.get("n_cscv", 0)

                            # ── HLZ cross-check fields ─────────────────────
                            _hlz_t    = _wf_qc.get("hlz_tstat")
                            _hlz_thr  = _wf_qc.get("hlz_threshold", 0.0)
                            _hlz_pass = _wf_qc.get("hlz_pass")
                            _hlz_ag   = _wf_qc.get("hlz_agree_dsr")
                            _hlz_t_str = f"{_hlz_t:.2f}" if _hlz_t is not None else "n/a"
                            _hlz_thr_str = f"{_hlz_thr:.2f}"
                            if _hlz_pass is True:
                                _hlz_col, _hlz_verdict = "#27ae60", "PASS"
                            elif _hlz_pass is False:
                                _hlz_col, _hlz_verdict = "#c0392b", "FAIL"
                            else:
                                _hlz_col, _hlz_verdict = "#555960", "n/a"
                            _prior_tag = "THEORY" if is_economic_prior else "GRID"
                            _agree_str = ""
                            if _hlz_ag is False:
                                _agree_str = (
                                    f'<span style="background:#3d1a00;color:#e67e22;'
                                    f'border-radius:3px;padding:1px 5px;margin-left:4px;'
                                    f'font-weight:700">⚠ DSR/HLZ DISAGREE</span>'
                                )

                            st.markdown(
                                f'<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-radius:4px;'
                                f'padding:7px 10px;margin:6px 0 2px;display:flex;gap:20px;flex-wrap:wrap;'
                                f'font-family:\'JetBrains Mono\',monospace;font-size:0.62rem">'
                                f'<span style="color:#8890a1">DSR PROB '
                                f'<b style="color:{_gc_display}">{_dsr_pct}</b>'
                                f' <span style="color:#555960">(SR*={_sr_star:.3f})</span></span>'
                                f'<span style="color:#8890a1">IS SHARPE '
                                f'<b style="color:#c8c8c8">{_is_str}</b></span>'
                                f'<span style="color:#8890a1">IS→OOS DECAY '
                                f'<b style="color:{_dec_col}">{_dec_str}</b></span>'
                                f'<span style="color:#8890a1">CSCV PBO '
                                f'<b style="color:{_pbo_col}">{_pbo_str}</b>'
                                f'<span style="color:#555960"> ({_n_cscv} partitions)</span></span>'
                                f'<span style="color:#8890a1">HLZ t={_hlz_t_str} vs '
                                f'<span style="color:#555960">hurdle {_hlz_thr_str}</span> '
                                f'<b style="color:{_hlz_col}">{_hlz_verdict}</b>'
                                f'{_agree_str}</span>'
                                f'<span style="color:#555960">N={n_strategies} '
                                f'({_prior_tag}) · cross-check only, DSR gates</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                            # Avg win / avg loss detail row
                            _aw = _wfbt.get("avg_win", 0)
                            _al = _wfbt.get("avg_loss", 0)
                            _nf = _wfbt.get("n_folds", 1)
                            _od = _wfbt.get("oos_days", 0)
                            st.markdown(
                                f'<div style="display:flex;gap:16px;padding:4px 0 8px;'
                                f'font-family:\'JetBrains Mono\',monospace;font-size:0.62rem;color:#8890a1">'
                                f'<span>Avg win <b style="color:#27ae60">{_aw:+.2f}%</b></span>'
                                f'<span>Avg loss <b style="color:#c0392b">{_al:+.2f}%</b></span>'
                                f'<span>{_nf} folds · {_od}d OOS · {_wfbt["tc_bps"]}bps TC + {_wfbt["slippage_bps"]}bps slip</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                            # Equity curve
                            _eq = _wfbt.get("equity_curve")
                            if _eq is not None and len(_eq) > 2:
                                _eq_profitable = float(_eq.iloc[-1]) >= 100
                                _eq_color  = "#27ae60" if _eq_profitable else "#c0392b"
                                _eq_fill   = "rgba(39,174,96,0.10)" if _eq_profitable else "rgba(192,57,43,0.10)"
                                _fig_eq = go.Figure()
                                _fig_eq.add_trace(go.Scatter(
                                    x=list(_eq.index),
                                    y=list(_eq.values),
                                    mode="lines",
                                    line=dict(color=_eq_color, width=1.5),
                                    fill="tozeroy",
                                    fillcolor=_eq_fill,
                                    hovertemplate="%{x|%b %Y}<br>Equity: %{y:.1f}<extra></extra>",
                                    showlegend=False,
                                ))
                                _fig_eq.add_hline(y=100, line_dash="dot", line_color="#555960", line_width=1)
                                _fig_eq.update_layout(
                                    template="plotly_dark",
                                    height=160,
                                    margin=dict(l=40, r=10, t=10, b=30),
                                    yaxis=dict(title="Equity (base 100)", tickfont=dict(size=9, color="#c8c8c8")),
                                    xaxis=dict(tickfont=dict(size=9, color="#c8c8c8")),
                                    plot_bgcolor="#0d0d0d",
                                    paper_bgcolor="#0d0d0d",
                                )
                                _chart(_fig_eq)

                            # QC flags
                            _wf_flags = _wf_qc.get("flags", [])
                            if _wf_flags:
                                st.markdown(
                                    '<div style="display:flex;gap:4px;flex-wrap:wrap;padding:4px 0">'
                                    + "".join(f'<span class="ti-qc-flag">⚠ {f}</span>' for f in _wf_flags)
                                    + "</div>",
                                    unsafe_allow_html=True,
                                )
        except Exception:
            pass

        # ── Agent debate thread ───────────────────────────────────────────
        _is_geo_trade = trade.get("generated", False)
        _debate_open  = _is_geo_trade
        _debate_label = (
            f"⚡ Agent Debate — {trade['name'][:40]}"
            if _is_geo_trade
            else f"Agent Debate — {trade['name'][:40]}"
        )
        with col:
            with st.expander(_debate_label, expanded=_debate_open):
                try:
                    from src.ui.agent_panel import render_deliberation_panel
                    from src.analysis.agent_dialogue import challenge_trade, get_subject_threads
                    _trade_subject_id = trade.get("name", f"trade_{trade_idx}")
                    _stored_key = f"_debate_tid_{trade_idx}"
                    msgs = get_subject_threads(_trade_subject_id)

                    if not msgs:
                        if _is_geo_trade:
                            try:
                                _new_tid = challenge_trade(
                                    trade_id=_trade_subject_id,
                                    trade_title=trade.get("name", "Unknown Trade"),
                                    confidence=float(trade.get("confidence", 0.60)),
                                    qc_flags=list(trade.get("qc_flags", [])),
                                )
                                st.session_state[_stored_key] = _new_tid
                                msgs = get_subject_threads(_trade_subject_id)
                            except Exception:
                                pass
                        if not msgs:
                            if st.button("Run Agent Debate", key=f"debate_{trade_idx}"):
                                try:
                                    _new_tid = challenge_trade(
                                        trade_id=_trade_subject_id,
                                        trade_title=trade.get("name", "Unknown Trade"),
                                        confidence=float(trade.get("confidence", 0.60)),
                                        qc_flags=list(trade.get("qc_flags", [])),
                                    )
                                    st.session_state[_stored_key] = _new_tid
                                    st.rerun()
                                except Exception as exc:
                                    st.caption("Debate unavailable — see logs.")

                    if msgs:
                        _render_tid = st.session_state.get(_stored_key) or msgs[0]["thread_id"]
                        render_deliberation_panel(
                            thread_id=_render_tid,
                            subject_id=_trade_subject_id,
                            title="Agent Deliberation",
                            max_msgs=8,
                            show_consensus=True,
                        )
                except Exception as exc:
                    st.caption("Debate panel unavailable — see logs.")

        # ── Mini correlation chart ────────────────────────────────────────
        if len(trade["assets"]) >= 2:
            a1, a2 = trade["assets"][0], trade["assets"][1]
            if a1 in all_r_concat.columns and a2 in all_r_concat.columns:
                rc = rolling_correlation(all_r_concat[a1], all_r_concat[a2], 60)
                r_hex = int(cat_col[1:3], 16)
                g_hex = int(cat_col[3:5], 16)
                b_hex = int(cat_col[5:7], 16)
                fig_mini = go.Figure()
                fig_mini.add_trace(go.Scatter(
                    x=rc.index, y=rc.values,
                    name=f"{a1}/{a2}",
                    line=dict(color=cat_col, width=1.4),
                    fill="tozeroy",
                    fillcolor=f"rgba({r_hex},{g_hex},{b_hex},0.12)",
                ))
                fig_mini.add_hline(y=0, line=dict(color="#444", width=1, dash="dot"))
                for ev in GEOPOLITICAL_EVENTS:
                    fig_mini.add_vrect(
                        x0=str(ev["start"]), x1=str(ev["end"]),
                        fillcolor=ev["color"], opacity=0.04, layer="below", line_width=0,
                    )
                fig_mini.update_layout(
                    template="plotly_dark", height=180,
                    title=dict(text=f"60d Corr: {a1} / {a2}", font=dict(size=10)),
                    showlegend=False,
                    margin=dict(l=36, r=12, t=28, b=24),
                    xaxis=dict(rangeslider=dict(visible=False)),
                    plot_bgcolor="#0d0d0d", paper_bgcolor="#0d0d0d",
                )
                with col:
                    _chart(fig_mini)
                    _insight_note(
                        "Rolling 60-day correlation between the commodity driver and equity target. "
                        "Rising correlation = thesis strengthening. Declining = causal link breaking down."
                    )


def page_trade_ideas(start: str, end: str, fred_key: str = "") -> None:
    # ── Stale-while-revalidate: pre-populate session state from disk cache ───
    # Runs once per session. If a prior run saved results to disk, the user
    # sees them immediately without clicking "Run Validation".
    _PV_DISK_KEY = "pipeline_validation"
    _PV_SESSION_KEY = "pipeline_validation_result"
    _pv_disk_age: "str | None" = None
    if _PV_SESSION_KEY not in st.session_state:
        try:
            from src.utils.page_cache import load_cache, age_str as _age_str
            _disk_data, _disk_saved_at = load_cache(_PV_DISK_KEY)
            if _disk_data is not None:
                st.session_state[_PV_SESSION_KEY] = _disk_data
                _pv_disk_age = _age_str(_disk_saved_at)
        except Exception:
            pass

    st.markdown(_TI_STYLE, unsafe_allow_html=True)
    _page_header("Structured Trade Ideas",
                 "Step 6 of 7 · Regime-driven · Conflict-linked · 5-Stage Pipeline Validation")
    _page_intro(
        "Spillover analysis is most useful when it connects to positioning hypotheses. "
        "<strong>Each structure here is a research-oriented translation of a spillover or regime signal into an illustrative trade idea.</strong> "
        "Static library theses fire when the current regime matches their structural trigger. "
        "The five-stage pipeline (Signal → Prior Validation → Sizing → DSR gate) "
        "is walk-forward validated — the pipeline's admit/reject decisions are the deliverable, not individual trade grades."
    )
    st.markdown(
        '<div style="display:flex;gap:1rem;align-items:center;margin-bottom:.6rem;flex-wrap:wrap">'
        '<span style="font-family:\'JetBrains Mono\',monospace;font-size:.58rem;font-weight:700;'
        'letter-spacing:.12em;text-transform:uppercase;color:#8890a1">'
        'Static Library Last Reviewed</span>'
        '<span style="font-family:\'JetBrains Mono\',monospace;font-size:.58rem;font-weight:700;'
        'color:#CFB991">July 2026</span>'
        '<span style="font-family:\'JetBrains Mono\',monospace;font-size:.58rem;'
        'color:rgba(255,255,255,.2)">|</span>'
        '<span style="font-family:\'DM Sans\',sans-serif;font-size:.72rem;color:#8890a1">'
        'Structural triggers and entry/exit levels reflect research-period market conditions.</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    _ti_cr: dict = {}  # initialised here so conflict-driven block can reuse it

    # ── Geopolitical context & filter gate (shown BEFORE any trades) ──────────
    try:
        from src.analysis.conflict_model import score_all_conflicts, aggregate_portfolio_scores
        from src.analysis.scenario_state import get_scenario, get_scenario_id
        _ti_cr   = score_all_conflicts()
        _ti_agg  = aggregate_portfolio_scores(_ti_cr)
        _ti_sc   = get_scenario()
        _ti_sid  = get_scenario_id()
        _ti_cis  = _ti_agg.get("portfolio_cis", _ti_agg.get("cis", 50.0))
        _ti_tps  = _ti_agg.get("portfolio_tps", _ti_agg.get("tps", 50.0))
        _ti_top  = (_ti_agg.get("top_conflict", "-") or "-").replace("_", " ").title()
        _ti_mult = _ti_sc.get("geo_mult", 1.0)
        _ti_sc_color = _ti_sc.get("color", "#CFB991")

        if _ti_cis >= 70:    _ti_risk_color, _ti_risk_lbl = "#c0392b", "HIGH CONFLICT"
        elif _ti_cis >= 50:  _ti_risk_color, _ti_risk_lbl = "#e67e22", "ELEVATED"
        else:                _ti_risk_color, _ti_risk_lbl = "#CFB991", "MODERATE"

        st.markdown(
            f'<div class="ti-geo-bar" style="border-left:3px solid {_ti_risk_color}">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
            f'font-weight:700;color:{_ti_risk_color};letter-spacing:.14em">'
            f'■ {_ti_risk_lbl}</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.63rem;'
            f'color:#e67e22">CIS&nbsp;<b>{_ti_cis:.0f}</b></span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.63rem;'
            f'color:#CFB991">TPS&nbsp;<b>{_ti_tps:.0f}</b></span>'
            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.63rem;color:#8890a1">'
            f'Lead:&nbsp;<b style="color:{_ti_risk_color}">{_ti_top}</b></span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.63rem;'
            f'color:{_ti_sc_color};font-weight:700">'
            f'{_ti_sc.get("label", "Base").upper()}&nbsp;×{_ti_mult:.2f}</span>'
            f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.50rem;'
            f'color:#555960;margin-left:auto">'
            f'Conflict-driven ideas reflect live CIS/TPS. Set filters before reviewing.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception as _geo_err:
        st.caption("Geo context unavailable — conflict model not loaded for this session.")

    from concurrent.futures import ThreadPoolExecutor
    from src.data.loader import load_fixed_income_returns, load_fx_returns
    with st.spinner("Loading data…"):
        # load_returns must run on the main Streamlit thread so @st.cache_data
        # context is available on cold-start cache misses.  fi/fx are lighter
        # and parallelise safely because they are typically already warm.
        try:
            eq_r, cmd_r = load_returns(start, end)
        except Exception:
            eq_r, cmd_r = pd.DataFrame(), pd.DataFrame()

        # Retry once with default date range if the custom range returned empty
        # (can happen when start/end differ from warmup keys and yfinance is slow).
        if (eq_r.empty or cmd_r.empty):
            try:
                eq_r, cmd_r = load_returns()
            except Exception:
                eq_r, cmd_r = pd.DataFrame(), pd.DataFrame()

        with ThreadPoolExecutor(max_workers=2) as _ti_pool:
            _f_fi = _ti_pool.submit(load_fixed_income_returns, start, end)
            _f_fx = _ti_pool.submit(load_fx_returns, start, end)
        try:
            _fi_r = _f_fi.result()
        except Exception:
            _fi_r = pd.DataFrame()
        try:
            _fx_r = _f_fx.result()
        except Exception:
            _fx_r = pd.DataFrame()

    if eq_r.empty or cmd_r.empty:
        st.error("Market data unavailable.")
        return

    # ── Current regime ─────────────────────────────────────────────────────
    avg_corr = average_cross_corr_series(eq_r, cmd_r, window=60)
    regimes  = detect_correlation_regime(avg_corr)
    # Read attrs BEFORE any pandas operation (reindex/ffill lose attrs)
    _regime_insuf = bool(regimes.attrs.get("insufficient_data", False))
    _regime_n_obs = int(regimes.attrs.get("n_obs", 0))
    current  = int(regimes.iloc[-1]) if not regimes.empty else 1
    r_name   = _REGIME_NAMES[current]
    r_color  = _REGIME_COLORS[current]

    # Trade cards removed — the pipeline is the deliverable, not individual strategy scores.
    active_trades: list[dict] = []
    asset_exposure: dict = {}   # was populated by score_all_assets(); empty without trade cards

    # Extend returns to include Fixed Income + FX (used by Integrity Audit and Multiple Testing)
    _extra_frames: list[pd.DataFrame] = []
    if not _fi_r.empty:
        _extra_frames.append(_fi_r)
    if not _fx_r.empty:
        _extra_frames.append(_fx_r)
    all_r_concat = pd.concat([eq_r, cmd_r] + _extra_frames, axis=1)

    # Effective N for DSR and HLZ multiple-testing gates
    _RAW_N = 18
    _effective_n: int = st.session_state.get("_effective_n", 9)

    # ── Download report ─────────────────────────────────────────────────────
    _n_theses  = len(_TRADE_LIBRARY)
    _n_geo     = len(GEOPOLITICAL_EVENTS) if GEOPOLITICAL_EVENTS else 0
    _r_col_dl  = {0: "#2e7d32", 1: "#555960", 2: "#e67e22", 3: "#c0392b"}.get(current, "#555960")
    _CAT_HEX_DL = {
        "Crisis Hedge": "#c0392b", "Geopolitical": "#e67e22",
        "Macro": "#2980b9",        "Growth": "#2e7d32",
        "Dollar Cycle": "#1abc9c", "Asia Divergence": "#9b59b6",
        "Fixed Income": "#2471a3", "India/EM": "#d35400",
    }

    _REGIME_RC = {0: "#2e7d32", 1: "#6b7280", 2: "#e67e22", 3: "#c0392b"}
    _REGIME_RL = {0: "D", 1: "N", 2: "E", 3: "C"}
    _REGIME_RN = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis"}

    def _thesis_item_html(tr: dict) -> str:
        cat     = tr.get("category", "Macro")
        col     = _CAT_HEX_DL.get(cat, "#555960")
        nm      = tr["name"].split("(")[0].strip()
        dirs    = tr.get("direction", [])
        assets  = tr.get("assets", [])

        # Leg directions — all legs, asset name stripped of parentheticals
        legs = "  ".join(
            f'{"▲" if d == "Long" else "▼"} {a.split("(")[0].strip()}'
            for a, d in zip(assets, dirs)
        )

        # Regime badges: colored mini-squares D / N / E / C
        reg_html = ""
        for r in sorted(tr.get("regime", [])):
            rc = _REGIME_RC.get(r, "#555960")
            rl = _REGIME_RL.get(r, "?")
            rn = _REGIME_RN.get(r, "")
            reg_html += (
                f'<span title="{rn}" style="display:inline-flex;align-items:center;'
                f'justify-content:center;width:14px;height:14px;'
                f'background:{rc}20;border:1px solid {rc}55;border-radius:2px;'
                f'font-family:\'JetBrains Mono\',monospace;font-size:.46rem;'
                f'font-weight:700;color:{rc};flex-shrink:0">{rl}</span>'
            )

        # Holding period badge
        hold    = tr.get("holding_period", "")
        hold_h  = (
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:.47rem;'
            f'color:#CFB991;background:#CFB99112;border:1px solid #CFB99128;'
            f'border-radius:2px;padding:1px 4px;white-space:nowrap;flex-shrink:0">'
            f'⏱ {hold}</span>'
        ) if hold else ""

        # Target (optional — newer theses)
        tgt     = tr.get("target", "")
        tgt_h   = ""
        if tgt:
            tgt_short = tgt[:55] + "…" if len(tgt) > 55 else tgt
            tgt_h = (
                f'<div style="font-size:.50rem;color:#27ae6090;'
                f'font-family:\'DM Sans\',sans-serif;margin-top:2px;line-height:1.3">'
                f'▸ {tgt_short}</div>'
            )

        # Trigger (1 line, truncated)
        trig    = tr.get("trigger", "")
        trig_s  = trig[:62] + "…" if len(trig) > 62 else trig

        # Investor lens (optional — newer theses)
        lens    = tr.get("investor_lens", [])
        lens_h  = (
            f'<div style="font-size:.49rem;color:#CFB99175;'
            f'font-family:\'JetBrains Mono\',monospace;margin-top:2px;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
            f'{" · ".join(lens[:3])}</div>'
        ) if lens else ""

        return (
            f'<div style="padding:7px 8px 6px;border-bottom:1px solid #151515">'
            # Row 1: color dot + name + regime badges
            f'<div style="display:flex;align-items:flex-start;'
            f'justify-content:space-between;gap:5px;margin-bottom:3px">'
            f'<div style="display:flex;align-items:flex-start;gap:5px;min-width:0">'
            f'<span style="width:6px;height:6px;border-radius:50%;background:{col};'
            f'flex-shrink:0;margin-top:4px"></span>'
            f'<div style="font-size:.62rem;color:#dcdcdc;'
            f'font-family:\'DM Sans\',sans-serif;line-height:1.35;'
            f'word-break:break-word;font-weight:600">{nm}</div>'
            f'</div>'
            f'<div style="display:flex;gap:2px;flex-shrink:0;margin-top:1px">{reg_html}</div>'
            f'</div>'
            # Row 2: category tag + legs + hold period
            f'<div style="display:flex;align-items:center;'
            f'justify-content:space-between;gap:4px;margin-bottom:3px">'
            f'<div style="min-width:0">'
            f'<span style="font-size:.50rem;color:{col};font-family:\'JetBrains Mono\','
            f'monospace;font-weight:700;letter-spacing:.06em">{cat.upper()}</span>'
            f'<span style="font-size:.50rem;color:#555960;font-family:\'JetBrains Mono\','
            f'monospace;margin:0 4px">·</span>'
            f'<span style="font-size:.50rem;color:#555960;font-family:\'JetBrains Mono\','
            f'monospace">{legs}</span>'
            f'</div>'
            f'{hold_h}'
            f'</div>'
            # Row 3: trigger
            f'<div style="font-size:.52rem;color:#8890a1;'
            f'font-family:\'DM Sans\',sans-serif;line-height:1.3">{trig_s}</div>'
            # Row 4: target (if available)
            f'{tgt_h}'
            # Row 5: investor lens (if available)
            f'{lens_h}'
            f'</div>'
        )

    # Split 18 theses evenly across two columns
    _half = (_n_theses + 1) // 2
    _col1_html = "".join(_thesis_item_html(t) for t in _TRADE_LIBRARY[:_half])
    _col2_html = "".join(_thesis_item_html(t) for t in _TRADE_LIBRARY[_half:])

    # Count regimes across all theses for the section summary
    _cat_counts: dict = {}
    for _tr in _TRADE_LIBRARY:
        _cat_counts[_tr.get("category", "Macro")] = _cat_counts.get(_tr.get("category", "Macro"), 0) + 1
    _cat_summary = " · ".join(f"{v} {k}" for k, v in sorted(_cat_counts.items(), key=lambda x: -x[1])[:4])

    # Avg holding: count theses with holding_period
    _n_with_hold = sum(1 for t in _TRADE_LIBRARY if t.get("holding_period"))

    # Build section list for left column — each with 2 detail lines
    _dl_sections = [
        ("01", "Regime Analysis",
         f"Current regime: {r_name}",
         "60d avg |corr| · percentile bands · 10d persistence gate"),
        ("02", "Cross-Asset Heatmap",
         "8 equity indices × 8 commodity futures",
         "Full-sample Pearson · spillover magnitude ranking"),
        ("03", "Composite Stress Index",
         "0–100 blended signal",
         "Vol 45% · Corr 35% · Commodity vol 15% · Accel 5%"),
        ("04", "Commodity Performance",
         "7 key futures · indexed to base 100",
         "Last 252 trading days (~1 year) · outperformers flagged"),
        ("05", f"Trade Ideas  ·  {_n_theses} theses",
         _cat_summary,
         f"{_n_with_hold} with holding periods · regime-triggered entries"),
        ("06", f"Geopolitical Context  ·  {_n_geo} events",
         "Active + recently resolved macro events",
         "Commodity price transmission · risk premium embedding"),
        ("07", "Methodology & Data",
         "DCC-GARCH · Diebold-Yilmaz Spillover Index",
         "Granger causality · Transfer entropy · FRED · Yahoo Finance"),
    ]
    _sec_rows_html = ""
    for _sn, _st_lbl, _sd1, _sd2 in _dl_sections:
        _sec_rows_html += (
            f'<div style="display:flex;gap:8px;padding:7px 10px;'
            f'border-bottom:1px solid #151515">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:.52rem;'
            f'color:#CFB991;font-weight:700;flex-shrink:0;padding-top:2px">{_sn}</span>'
            f'<div>'
            f'<div style="font-size:.62rem;color:#e8e9ed;font-weight:600;'
            f'font-family:\'DM Sans\',sans-serif;line-height:1.3;margin-bottom:2px">{_st_lbl}</div>'
            f'<div style="font-size:.55rem;color:#9299a3;'
            f'font-family:\'DM Sans\',sans-serif;line-height:1.35">{_sd1}</div>'
            f'<div style="font-size:.52rem;color:#555960;'
            f'font-family:\'JetBrains Mono\',monospace;margin-top:2px;line-height:1.3">{_sd2}</div>'
            f'</div></div>'
        )

    st.markdown(
        f'<div style="border:1px solid #CFB99130;border-radius:6px;overflow:hidden;'
        f'margin-bottom:1.2rem;background:#080808">'
        # Gold header bar
        f'<div style="background:#CFB991;padding:6px 14px;display:flex;'
        f'justify-content:space-between;align-items:center">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:.57rem;'
        f'font-weight:700;color:#1a0f00;letter-spacing:.12em">'
        f'PURDUE · DANIELS SCHOOL OF BUSINESS</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:.53rem;'
        f'color:#4a3000;letter-spacing:.08em">A4 · PDF · 7 SECTIONS</span>'
        f'</div>'
        # Title + regime badge
        f'<div style="padding:13px 14px 10px;border-bottom:1px solid #1a1a1a;'
        f'display:flex;justify-content:space-between;align-items:flex-start">'
        f'<div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.90rem;'
        f'font-weight:700;color:#e8e9ed;letter-spacing:.03em;line-height:1.25">'
        f'EQUITY &amp; COMMODITIES<br>SPILLOVER MONITOR</div>'
        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:.64rem;'
        f'color:#8890a1;margin-top:5px">'
        f'Cross-Asset Quantitative Research · Academic Submission Format</div>'
        f'</div>'
        f'<div style="text-align:right;flex-shrink:0;margin-left:14px">'
        f'<div style="display:inline-block;background:{_r_col_dl}22;'
        f'border:1px solid {_r_col_dl}55;border-radius:3px;padding:4px 10px;'
        f'font-family:\'JetBrains Mono\',monospace;font-size:.56rem;'
        f'font-weight:700;color:{_r_col_dl};margin-bottom:6px">'
        f'REGIME: {r_name.upper()}</div>'
        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:.57rem;color:#555960">'
        f'Heramb S. Patkar<br>Jiahe Miao · Ilian Zalomai</div>'
        f'</div>'
        f'</div>'
        # Three-column body: sections | theses-col-1 | theses-col-2
        f'<div style="display:grid;grid-template-columns:30% 35% 35%;'
        f'border-bottom:1px solid #1a1a1a;align-items:start">'
        # Sections
        f'<div style="border-right:1px solid #1a1a1a">'
        f'<div style="padding:5px 10px;background:#050505;border-bottom:1px solid #1a1a1a;'
        f'font-family:\'JetBrains Mono\',monospace;font-size:.52rem;'
        f'color:#555960;letter-spacing:.10em">REPORT SECTIONS</div>'
        f'{_sec_rows_html}'
        f'</div>'
        # Theses col 1 (top half)
        f'<div style="border-right:1px solid #1a1a1a">'
        f'<div style="padding:5px 10px;background:#050505;border-bottom:1px solid #1a1a1a;'
        f'font-family:\'JetBrains Mono\',monospace;font-size:.52rem;'
        f'color:#555960;letter-spacing:.10em">'
        f'THESES REFERENCED — {_n_theses} PAIRS</div>'
        f'{_col1_html}'
        f'</div>'
        # Theses col 2 (bottom half)
        f'<div>'
        f'<div style="padding:5px 10px;background:#050505;border-bottom:1px solid #1a1a1a;'
        f'font-family:\'JetBrains Mono\',monospace;font-size:.52rem;'
        f'color:#050505;letter-spacing:.10em">&nbsp;</div>'
        f'{_col2_html}'
        f'</div>'
        f'</div>'
        # Footer bar
        f'<div style="padding:7px 14px;background:#050505;display:flex;'
        f'justify-content:space-between;align-items:center">'
        f'<span style="font-family:\'DM Sans\',sans-serif;font-size:.57rem;color:#555960">'
        f'Regime-triggered · Historical spillover patterns · Purdue MSF Research Terminal</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:.52rem;color:#CFB99170">'
        f'EDUCATIONAL USE ONLY</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.56rem;color:#8890a1;padding:6px 0 4px 0">'
        'The PDF includes all 18 theses as a reference catalogue. '
        'Active trade cards are regime-filtered through the pipeline — no active trades are pre-selected.'
        '</div>',
        unsafe_allow_html=True,
    )
    if st.button("Generate PDF Report", key="gen_report", type="primary"):
        try:
            from src.reports.report_generator import generate_report
            with st.spinner("Building report - generating charts…"):
                stress = composite_stress_index(eq_r, cmd_r, avg_corr=avg_corr)
                pdf_bytes = generate_report(
                    start=start,
                    end=end,
                    avg_corr_series=avg_corr,
                    current_regime=current,
                    regimes=regimes,
                    active_trades=active_trades,
                    all_trades=_TRADE_LIBRARY,
                    eq_r=eq_r,
                    cmd_r=cmd_r,
                    stress_series=stress,
                    geopolitical_events=GEOPOLITICAL_EVENTS,
                )
            filename = (
                f"purdue_spillover_report_"
                f"{start.replace('-','')}_to_{end.replace('-','')}.pdf"
            )
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                key="download_report",
            )
        except ImportError:
            st.error(
                "reportlab is required for PDF generation. "
                "Run: `pip install reportlab>=4.2.0`"
            )
        except Exception as exc:
            st.error("Report generation failed.")

    # ── Data Integrity Audit ────────────────────────────────────────────────
    with st.expander("Data Integrity Audit — Leg Coverage & Strategy Correlation", expanded=False):
        _DI_M = "font-family:'JetBrains Mono',monospace;"
        st.markdown(
            f'<p style="{_DI_M}font-size:0.60rem;color:#8890a1;margin-bottom:.8rem">'
            'Checks every strategy\'s declared legs against the loaded return data. '
            'Strategies with missing legs are mislabeled — their backtest results are excluded. '
            'The correlation matrix identifies hidden duplicate bets.</p>',
            unsafe_allow_html=True,
        )

        # ── Leg coverage table ──────────────────────────────────────────────
        _avail_cols = set(all_r_concat.columns)
        _audit_rows = []
        for _tr in _TRADE_LIBRARY:
            _declared  = _tr.get("assets", [])
            _present   = [a for a in _declared if a in _avail_cols]
            _dropped   = [a for a in _declared if a not in _avail_cols]
            if _dropped and not _present:
                _status = "UNGRADEABLE"
            elif _dropped:
                _status = "MISLABELED"
            else:
                _status = "OK"
            _audit_rows.append({
                "Strategy":    _tr["name"],
                "Declared":    ", ".join(_declared),
                "Present":     ", ".join(_present) if _present else "—",
                "Dropped":     ", ".join(_dropped) if _dropped else "—",
                "Status":      _status,
            })

        _n_ok  = sum(1 for r in _audit_rows if r["Status"] == "OK")
        _n_mis = sum(1 for r in _audit_rows if r["Status"] == "MISLABELED")
        _n_ung = sum(1 for r in _audit_rows if r["Status"] == "UNGRADEABLE")

        st.markdown(
            f'<p style="{_DI_M}font-size:0.65rem;color:#e8e9ed;margin-bottom:.6rem">'
            f'<b style="color:#27ae60">{_n_ok}</b> strategies have all legs · '
            f'<b style="color:#e67e22">{_n_mis}</b> mislabeled (partial data) · '
            f'<b style="color:#c0392b">{_n_ung}</b> ungradeable (0 legs)</p>',
            unsafe_allow_html=True,
        )

        _th = (f'style="color:#555960;text-align:left;padding:5px 10px;'
               f'font-family:\'JetBrains Mono\',monospace;font-size:.58rem;'
               f'letter-spacing:.10em;border-bottom:1px solid #2a2a2a;white-space:nowrap"')
        _audit_body = ""
        for _r in _audit_rows:
            _s = _r["Status"]
            _bg = "#1a0000" if _s == "UNGRADEABLE" else "#1a0d00" if _s == "MISLABELED" else "#0d0d0d"
            _sc = "#e74c3c" if _s == "UNGRADEABLE" else "#e67e22" if _s == "MISLABELED" else "#27ae60"
            _tdb = "padding:5px 10px;border-bottom:1px solid #1a1a1a;"
            _audit_body += (
                f'<tr style="background:{_bg}">'
                f'<td style="{_tdb}font-size:.63rem;color:#c8c8c8;max-width:260px">{_r["Strategy"]}</td>'
                f'<td style="{_tdb}font-size:.60rem;color:#8890a1">{_r["Declared"]}</td>'
                f'<td style="{_tdb}font-size:.60rem;color:#8890a1">{_r["Present"]}</td>'
                f'<td style="{_tdb}font-size:.60rem;color:#c0392b">{_r["Dropped"] or "—"}</td>'
                f'<td style="{_tdb}font-size:.60rem;font-weight:700;color:{_sc};'
                f'font-family:\'JetBrains Mono\',monospace">{_s}</td>'
                f'</tr>'
            )
        st.markdown(
            f'<div style="overflow-x:auto;border:1px solid #1e1e1e;border-radius:4px;'
            f'margin-bottom:.8rem">'
            f'<table style="width:100%;border-collapse:collapse;'
            f'font-family:\'DM Sans\',sans-serif">'
            f'<thead><tr style="background:#0a0a0a">'
            f'<th {_th}>STRATEGY</th><th {_th}>DECLARED LEGS</th>'
            f'<th {_th}>PRESENT</th><th {_th}>DROPPED</th><th {_th}>STATUS</th>'
            f'</tr></thead><tbody>{_audit_body}</tbody></table></div>',
            unsafe_allow_html=True,
        )

        # ── Pairwise correlation matrix ─────────────────────────────────────
        st.markdown(
            f'<p style="{_DI_M}font-size:0.60rem;color:#8890a1;margin-top:1rem;margin-bottom:.4rem">'
            'Pairwise correlation of daily equity-curve returns across all strategies '
            '(OOS walk-forward). Clusters above r ≈ 0.90 are hidden duplicates — '
            'they count as 1 distinct bet for DSR multiple-testing correction.</p>',
            unsafe_allow_html=True,
        )
        if avg_corr is not None:
            _curves: dict[str, pd.Series] = {}
            with st.spinner(f"Building equity curves for {len(_TRADE_LIBRARY)} strategies…"):
              for _tr in _TRADE_LIBRARY:
                try:
                    _lw  = _compute_leg_weights(_tr, asset_exposure or {})
                    _lwt = tuple(_lw) if _lw else None
                    _r   = _wf_backtest_trade(
                        all_r_concat, avg_corr,
                        trade_name=_tr["name"],
                        trigger_regimes=_tr.get("regime", [2, 3]),
                        assets=_tr.get("assets", []),
                        directions=_tr.get("direction", []),
                        holding_days=_parse_holding_days(_tr),
                        leg_weights=_lwt,
                        avg_corr_n=len(avg_corr),
                    )
                    _ec = _r.get("equity_curve")
                    if _ec is not None and len(_ec) > 10 and _r.get("n_trades", 0) >= 3:
                        _short_name = _tr["name"].split(" (")[0][:50]
                        _curves[_short_name] = _ec.pct_change().dropna()
                except Exception:
                    pass

            if len(_curves) >= 2:
                import plotly.express as px
                _ec_df   = pd.DataFrame(_curves).dropna(how="all")
                _corr_m  = _ec_df.corr().round(2)
                _n_names = len(_corr_m)

                # Cluster report: pairs above r = 0.90
                _clusters: list[str] = []
                _seen: set = set()
                for _i, _ni in enumerate(_corr_m.columns):
                    for _j, _nj in enumerate(_corr_m.columns):
                        if _j <= _i:
                            continue
                        _rv = float(_corr_m.loc[_ni, _nj])
                        if _rv >= 0.90:
                            _pair_key = (min(_ni, _nj), max(_ni, _nj))
                            if _pair_key not in _seen:
                                _seen.add(_pair_key)
                                _clusters.append(f"r={_rv:.2f}: **{_ni}** ↔ **{_nj}**")

                if _clusters:
                    st.markdown(
                        f'<div style="background:#1a1200;border:1px solid #e67e22;'
                        f'border-radius:4px;padding:8px 12px;margin-bottom:.6rem;'
                        f'{_DI_M}font-size:0.63rem;color:#e67e22">'
                        f'<b>HIGH-CORRELATION CLUSTERS (r ≥ 0.90) — count as 1 distinct bet each:</b><br>'
                        + "<br>".join(_clusters)
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<p style="{_DI_M}font-size:0.63rem;color:#27ae60">'
                        'No clusters above r = 0.90 detected among gradeable strategies.</p>',
                        unsafe_allow_html=True,
                    )

                import plotly.graph_objects as _go_c
                _z_arr   = _corr_m.values
                _names_x = list(_corr_m.columns)
                _names_y = list(_corr_m.index)

                # go.Heatmap (not px.imshow) so we can add per-cell annotations
                # with correct contrast: white on dark cells, near-black on light cells.
                _fig_corr = _go_c.Figure(data=_go_c.Heatmap(
                    z=_z_arr.tolist(),
                    x=_names_x,
                    y=_names_y,
                    colorscale="RdYlGn",
                    zmin=-1, zmax=1,
                    showscale=True,
                    colorbar=dict(
                        tickfont=dict(size=11, color="#c8c8c8"),
                        tickcolor="#c8c8c8",
                        outlinecolor="#080808",
                    ),
                ))

                # Per-cell text annotations — the only way to get correct contrast
                # on both dark-green (|r|>0.45 → white) and light-yellow (→ dark)
                for _ri, _rn in enumerate(_names_y):
                    for _ci, _cn in enumerate(_names_x):
                        _v   = float(_z_arr[_ri, _ci])
                        _tc  = "#ffffff" if abs(_v) > 0.45 else "#0a0a0a"
                        _fig_corr.add_annotation(
                            x=_cn, y=_rn,
                            text=f"{_v:.2f}",
                            showarrow=False,
                            font=dict(size=11, color=_tc),
                            xref="x", yref="y",
                        )

                _fig_corr.update_layout(
                    title="Strategy Pairwise Correlation (OOS Daily Equity-Curve Returns)",
                    height=max(600, _n_names * 42),
                    font=dict(family="JetBrains Mono", size=11, color="#c8c8c8"),
                    paper_bgcolor="#080808", plot_bgcolor="#080808",
                    title_font=dict(size=11, color="#8890a1"),
                    margin=dict(l=10, r=10, t=50, b=400),
                )
                _fig_corr.update_xaxes(
                    tickfont=dict(size=11, color="#c8c8c8"),
                    tickangle=-45,
                    side="bottom",
                )
                _fig_corr.update_yaxes(
                    tickfont=dict(size=11, color="#c8c8c8"),
                    autorange="reversed",
                )
                st.plotly_chart(_fig_corr, use_container_width=True)

                # Effective distinct-bet count
                _gradeable = sum(
                    1 for _tr in _TRADE_LIBRARY
                    if not [a for a in _tr.get("assets", []) if a not in _avail_cols]
                    and _tr["name"].split(" (")[0][:50] in _curves
                )
                _n_cluster_pairs = len(_seen)
                _effective_n = _gradeable - _n_cluster_pairs
                st.markdown(
                    f'<p style="{_DI_M}font-size:0.65rem;color:#e8e9ed;margin-top:.6rem">'
                    f'Gradeable strategies: <b>{_gradeable}</b> · '
                    f'Hidden duplicate pairs: <b>{_n_cluster_pairs}</b> · '
                    f'Effective distinct bets for DSR N: '
                    f'<b style="color:#CFB991">{_effective_n}</b></p>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Insufficient gradeable strategies to compute correlation matrix.")
        else:
            st.caption("avg_corr unavailable — cannot run pairwise correlation.")

    # ── Multiple Testing Report ─────────────────────────────────────────────
    # Reports effective N vs raw N, per-strategy DSR vs HLZ cross-check,
    # and flags disagreements. Opening this panel also updates session_state
    # so subsequent rerenders use the dynamic effective N in card grades.
    with st.expander(
        "Multiple Testing Report — DSR vs HLZ Cross-Check (Effective N)",
        expanded=False,
    ):
        from src.analysis.backtest import (
            compute_effective_n as _compute_eff_n,
            _N_LIBRARY_STRATEGIES as _STATIC_N,
        )
        _MT_M = "font-family:'JetBrains Mono',monospace;"
        st.markdown(
            f'<p style="{_MT_M}font-size:0.60rem;color:#8890a1;margin-bottom:.8rem">'
            'DSR is the single grading gate — it already corrects for N via the expected maximum SR under '
            'H₀ (Bailey &amp; Lopez de Prado 2014). '
            'HLZ (Harvey, Liu &amp; Zhu 2016) BHY-adjusted t-hurdle is shown as a cross-check only. '
            'Disagreements are flagged for manual review. '
            'Effective N = distinct bets after r &gt; 0.90 return-series collapse (union-find). '
            'Generated candidates use raw N (penalty for implicit grid search).</p>',
            unsafe_allow_html=True,
        )

        # ── Step 1: collect all walk-forward results from cache ─────────────
        _mt_results: dict[str, dict] = {}
        _avail_cols_mt = set(all_r_concat.columns)
        with st.spinner(f"Computing backtest results for {len(_TRADE_LIBRARY)} strategies…"):
          for _tr in _TRADE_LIBRARY:
            _missing = [a for a in _tr.get("assets", []) if a not in _avail_cols_mt]
            if _missing:
                continue   # ungradeable — skip for N computation
            try:
                _lw  = _compute_leg_weights(_tr, asset_exposure or {})
                _lwt = tuple(_lw) if _lw else None
                _r   = _wf_backtest_trade(
                    all_r_concat, avg_corr,
                    trade_name=_tr["name"],
                    trigger_regimes=_tr.get("regime", [2, 3]),
                    assets=_tr.get("assets", []),
                    directions=_tr.get("direction", []),
                    holding_days=_parse_holding_days(_tr),
                    leg_weights=_lwt,
                    avg_corr_n=len(avg_corr),
                    n_strategies=_STATIC_N,   # placeholder; will re-grade below
                    is_economic_prior=not bool(_tr.get("generated", False)),
                )
                if _r.get("n_trades", 0) >= 3 and "error" not in _r:
                    _mt_results[_tr["name"]] = _r
            except Exception:
                pass

        # ── Step 2: compute effective N and update session state ─────────────
        _eff_n, _cluster_pairs = _compute_eff_n(_mt_results, corr_threshold=0.90)
        _raw_n_gradeable       = len(_mt_results)
        if _eff_n != st.session_state.get("_effective_n"):
            st.session_state["_effective_n"] = _eff_n

        st.markdown(
            f'<div style="{_MT_M}font-size:0.70rem;color:#e8e9ed;margin-bottom:.6rem">'
            f'Raw N (declared): <b style="color:#CFB991">{_RAW_N}</b> · '
            f'Gradeable (all legs present): <b style="color:#CFB991">{_raw_n_gradeable}</b> · '
            f'Effective N (r&gt;0.90 collapse): <b style="color:#27ae60">{_eff_n}</b>'
            + (f' · <b style="color:#e67e22">{len(_cluster_pairs)} duplicate pair(s) collapsed</b>'
               if _cluster_pairs else '')
            + '</div>',
            unsafe_allow_html=True,
        )

        if _cluster_pairs:
            st.markdown(
                f'<div style="background:#1a1200;border:1px solid #e67e22;border-radius:4px;'
                f'padding:6px 10px;margin-bottom:.6rem;{_MT_M}font-size:0.62rem;color:#e67e22">'
                + "<br>".join(
                    f'r={r:.3f}: <b>{a}</b> ↔ <b>{b}</b> (count as 1 bet)'
                    for a, b, r in sorted(_cluster_pairs, key=lambda x: -x[2])
                )
                + '</div>',
                unsafe_allow_html=True,
            )

        # ── Step 3: re-grade each strategy with dynamic effective N ─────────
        _report_rows = []
        for _tr in _TRADE_LIBRARY:
            _is_gen    = bool(_tr.get("generated", False))
            _n_used    = _RAW_N if _is_gen else _eff_n
            _is_prior  = not _is_gen
            _missing   = [a for a in _tr.get("assets", []) if a not in _avail_cols_mt]
            if _missing:
                _report_rows.append({
                    "Strategy": _tr["name"][:45],
                    "Prior":    "THEORY",
                    "N used":   "—",
                    "DSR %":    "—",
                    "Grade":    "—",
                    "t-stat":   "—",
                    "HLZ hurdle": "—",
                    "HLZ":      "MISSING",
                    "Agree?":   "—",
                })
                continue
            _base = _mt_results.get(_tr["name"])
            if _base is None:
                continue
            # Re-grade with dynamic N (pure function — cheap)
            from src.analysis.backtest import qc_grade_backtest as _regrade
            _qc = _regrade(_base, n_strategies=_n_used, is_economic_prior=_is_prior)
            _hlz_p = _qc.get("hlz_pass")
            _ag    = _qc.get("hlz_agree_dsr")
            _report_rows.append({
                "Strategy":   _tr["name"][:45],
                "Prior":      "THEORY" if _is_prior else "GRID",
                "N used":     str(_n_used),
                "DSR %":      f'{_qc.get("dsr_prob", 0):.0%}',
                "Grade":      _qc.get("grade", "—"),
                "t-stat":     f'{_qc["hlz_tstat"]:.2f}' if _qc.get("hlz_tstat") is not None else "n/a",
                "HLZ hurdle": f'{_qc.get("hlz_threshold", 0):.2f}',
                "HLZ":        ("PASS" if _hlz_p is True else "FAIL" if _hlz_p is False else "n/a"),
                "Agree?":     ("✓" if _ag is True else "⚠ REVIEW" if _ag is False else "—"),
            })

        if _report_rows:
            _mt_cols = ["Strategy", "Prior", "N used", "DSR %", "Grade",
                        "t-stat", "HLZ hurdle", "HLZ", "Agree?"]
            _mt_th = (f'style="color:#555960;text-align:left;padding:5px 10px;'
                      f'font-family:\'JetBrains Mono\',monospace;font-size:.58rem;'
                      f'letter-spacing:.10em;border-bottom:1px solid #2a2a2a;'
                      f'white-space:nowrap"')
            _mt_body = ""
            for _r in _report_rows:
                _ag = _r.get("Agree?", "—")
                _gr = _r.get("Grade", "")
                _hl = _r.get("HLZ", "")
                if _ag == "⚠ REVIEW":
                    _bg = "#1a1200"; _rc = "#e67e22"
                elif _hl == "MISSING":
                    _bg = "#0d0d0d"; _rc = "#555960"
                elif _gr in ("A", "B"):
                    _bg = "#0a1a0a"; _rc = "#c8c8c8"
                elif _gr == "F":
                    _bg = "#0d0000"; _rc = "#c8c8c8"
                else:
                    _bg = "#0d0d0d"; _rc = "#c8c8c8"
                _ag_col  = "#27ae60" if _ag == "✓" else "#e67e22" if _ag == "⚠ REVIEW" else "#555960"
                _grade_colors = {"A": "#27ae60", "B": "#2980b9", "C": "#e67e22",
                                 "D": "#e67e22", "F": "#c0392b", "MT": "#9b59b6",
                                 "IE": "#8890a1"}
                def _mt_cell_color(col_name):
                    if col_name == "Grade":
                        return _grade_colors.get(_gr, _rc)
                    if col_name == "HLZ" and _r.get(col_name) == "FAIL":
                        return "#e67e22"
                    if col_name == "Agree?":
                        return _ag_col
                    return _rc
                _cells = "".join(
                    f'<td style="padding:5px 10px;border-bottom:1px solid #1a1a1a;'
                    f'font-size:.62rem;{"font-weight:700;" if _c == "Grade" else ""}'
                    f'color:{_mt_cell_color(_c)}">'
                    f'{_r.get(_c, "—")}</td>'
                    for _c in _mt_cols
                )
                _mt_body += f'<tr style="background:{_bg}">{_cells}</tr>'
            _mt_thead = "".join(f'<th {_mt_th}>{_c.upper()}</th>' for _c in _mt_cols)
            st.markdown(
                f'<div style="overflow-x:auto;border:1px solid #1e1e1e;border-radius:4px;'
                f'margin-bottom:.8rem">'
                f'<table style="width:100%;border-collapse:collapse;'
                f'font-family:\'DM Sans\',sans-serif">'
                f'<thead><tr style="background:#0a0a0a">{_mt_thead}</tr></thead>'
                f'<tbody>{_mt_body}</tbody></table></div>',
                unsafe_allow_html=True,
            )

            # Count disagreements
            _n_disagree = sum(1 for r in _report_rows if r.get("Agree?") == "⚠ REVIEW")
            _n_agree    = sum(1 for r in _report_rows if r.get("Agree?") == "✓")
            st.markdown(
                f'<p style="{_MT_M}font-size:0.65rem;color:#8890a1;margin-top:.4rem">'
                f'<b style="color:#27ae60">{_n_agree}</b> DSR/HLZ agree · '
                f'<b style="color:{"#e67e22" if _n_disagree else "#27ae60"}">'
                f'{_n_disagree}</b> disagree (manual review recommended) · '
                f'HLZ cross-check only — DSR is the binding grade criterion'
                f'</p>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No gradeable strategies to report.")

    # ── Thesis Pipeline ─────────────────────────────────────────────────────
    # Auto-expand when we have results (from disk cache or a prior run this session).
    _tp_has_results = bool(st.session_state.get(_PV_SESSION_KEY))
    with st.expander(
        "Thesis Pipeline — 5-Stage Economic Mechanism Validation",
        expanded=_tp_has_results,
    ):
        _TP_M = "font-family:'JetBrains Mono',monospace;"
        _TP_S = "font-family:'DM Sans',sans-serif;"
        _TP_GRADE_COLOR = {"A": "#27ae60", "B": "#2980b9", "C": "#e67e22",
                           "D": "#e74c3c", "F": "#c0392b",
                           "IE": "#8e44ad",
                           "MT": "#16a085"}
        import plotly.graph_objects as _go_tp

        # ── PART 1: Stage Gate Methodology ──────────────────────────────────
        st.markdown(
            f'<p style="{_TP_M}font-size:0.58rem;color:#8890a1;margin-bottom:1rem">'
            'The five-stage pipeline is the deliverable. Each gate is a binary contract: '
            'a thesis that fails any gate cannot advance. The pipeline is tested as a '
            'decision rule — does its admit/reject classification predict out-of-sample '
            'returns? Individual trade P&amp;L is not the output.</p>',
            unsafe_allow_html=True,
        )

        _stage_defs = [
            ("#CFB991", "STAGE 1 — THESIS",
             "Researcher constructs shock → TPS channel → predicted sign → holding horizon from first "
             "principles. No optimisation. The economic narrative must be stated before any data is viewed. "
             "Gate: shock is named, ≥1 TPS channel specified, predicted direction signed per leg, horizon set."),
            ("#2980b9", "STAGE 2 — SIGNAL",
             "Every declared leg must be present in the loaded return data. Any missing leg "
             "is a hard stop — the thesis cannot be tested and the researcher must revise the leg specification. "
             "Gate: all assets in return index. Fail-loud: never silently drop a leg."),
            ("#27ae60", "STAGE 3 — PRIOR-ALIGNED CONFIRMATION",
             "LP-IRF (local projection) or regime-conditional returns confirm that the data's sign "
             "matches the predicted sign from Stage 1 at the stated horizon. "
             "Outcomes: CONFIRM (sign + significance), IE (sign matched but n &lt; 20 — insufficient evidence, "
             "not a rejection), REJECT (sign contradicted). "
             "Gate: sign matched AND significant at 10%."),
            ("#e67e22", "STAGE 4 — SIZING",
             "Vol-scaled allocation: target 10% annual vol ÷ estimated strategy vol → base weight. "
             "IRF scale factor applied (larger coef at horizon → larger weight). "
             "Capped at 20% per conflict source. Gate: sizing computed; output is final weight %."),
            ("#8e44ad", "STAGE 5 — GRADE",
             "Deflated Sharpe Ratio (DSR) gate on per-trade Sharpe (de-annualised, Bailey &amp; Lopez de Prado 2014). "
             "DSR ≥ 0.50 required. If Stage 3 confirmed AND DSR &lt; 0.50 → MT (mechanism real, not tradeable: "
             "transmission genuine but too weak or already priced). "
             "If n &lt; 20 AND sign matched → IE. If sign contradicted → REJECT. "
             "Grade A/B/C/D from DSR: A ≥ 0.85, B ≥ 0.70, C ≥ 0.55, D ≥ 0.50."),
        ]
        _stage_cols = st.columns(5, gap="small")
        for _sci, (_sc, _sh, _st) in enumerate(_stage_defs):
            with _stage_cols[_sci]:
                st.markdown(
                    f'<div style="background:#080808;border:1px solid #1e1e1e;'
                    f'border-top:3px solid {_sc};border-radius:4px;padding:.6rem .7rem;height:100%">'
                    f'<div style="{_TP_M}font-size:0.56rem;letter-spacing:.12em;color:{_sc};'
                    f'margin-bottom:6px">{_sh}</div>'
                    f'<div style="{_TP_S}font-size:0.60rem;color:#a8a8b8;line-height:1.55">{_st}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── PART 2: Pipeline Validation ──────────────────────────────────────
        st.markdown(
            f'<p style="{_TP_M}font-size:0.58rem;font-weight:700;letter-spacing:.10em;'
            f'color:#CFB991;margin-bottom:.3rem">WALK-FORWARD PIPELINE VALIDATION</p>'
            f'<p style="{_TP_S}font-size:0.65rem;color:#8890a1;margin-bottom:.6rem;'
            f'line-height:1.6">'
            f'Tests the five-stage pipeline as a decision rule over a 3-year rolling training window '
            f'(756 days) with 1-quarter test steps (63 days). At each window the pipeline classifies '
            f'each thesis as admit / mt / ie / reject using only past data. The three required outputs: '
            f'(1) admitted vs rejected OOS gap, (2) admitted vs random gap (500 draws), '
            f'(3) MT and IE bucket means as labeled. The pipeline passes only if gaps 1 and 2 are positive.</p>',
            unsafe_allow_html=True,
        )

        _pv_key = _PV_SESSION_KEY

        # Warn if effective N hasn't been computed yet (MT Report expander not opened)
        if "_effective_n" not in st.session_state:
            st.markdown(
                f'<div style="{_TP_M}border:1px solid #e67e22;border-radius:4px;'
                f'padding:8px 12px;margin-bottom:8px;background:#1a1000">'
                f'<span style="font-size:0.56rem;color:#e67e22;font-weight:700">⚠ Open Multiple Testing Report first</span>'
                f'<span style="font-size:0.56rem;color:#8890a1"> — Effective N for DSR correction defaults to 9 until the MT Report '
                f'computes the true value from your backtest results. Validation run before then may apply the wrong correction.</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        _pv_col1, _pv_col2 = st.columns([1, 4])
        with _pv_col1:
            _run_pv = st.button(
                "Refresh Validation", key="run_pipeline_val", type="primary",
                help="Re-runs walk-forward validation (~2-4 min). Saves result to disk for next session.",
            )
        with _pv_col2:
            # Show staleness banner if showing data from a previous session
            if _pv_disk_age and not _run_pv:
                st.markdown(
                    f'<span style="{_TP_M}font-size:0.57rem;color:#e67e22">'
                    f'Showing cached results from {_pv_disk_age}. '
                    f'Click Refresh Validation to recompute.</span>',
                    unsafe_allow_html=True,
                )
            elif st.session_state.get(_pv_key) and not _run_pv:
                st.markdown(
                    f'<span style="{_TP_M}font-size:0.57rem;color:#555960">'
                    f'Results from this session. Click Refresh to recompute.</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span style="{_TP_M}font-size:0.57rem;color:#555960">'
                    f'First run: ~2-4 min. Result saved to disk for instant load next session.</span>',
                    unsafe_allow_html=True,
                )

        if _run_pv:
            _anim = st.empty()
            _anim.markdown("""
<style>
@keyframes _pv_scan {
  0%   { left: -30%; }
  100% { left: 110%; }
}
@keyframes _pv_pulse {
  0%,100% { opacity: .35; }
  50%      { opacity: 1;   }
}
@keyframes _pv_bar {
  0%   { width: 0%; }
  15%  { width: 22%; }
  40%  { width: 45%; }
  65%  { width: 68%; }
  85%  { width: 84%; }
  100% { width: 93%; }
}
@keyframes _pv_msg0 { 0%,18%{opacity:1} 22%,100%{opacity:0} }
@keyframes _pv_msg1 { 0%,18%{opacity:0} 22%,38%{opacity:1} 42%,100%{opacity:0} }
@keyframes _pv_msg2 { 0%,38%{opacity:0} 42%,58%{opacity:1} 62%,100%{opacity:0} }
@keyframes _pv_msg3 { 0%,58%{opacity:0} 62%,78%{opacity:1} 82%,100%{opacity:0} }
@keyframes _pv_msg4 { 0%,78%{opacity:0} 82%,100%{opacity:1} }
._pv_wrap {
  background:#080808; border:1px solid #1e1e1e; border-radius:6px;
  padding:1.2rem 1.4rem; margin:.6rem 0; position:relative; overflow:hidden;
}
._pv_title {
  font-family:'JetBrains Mono',monospace; font-size:0.50rem; letter-spacing:.18em;
  color:#CFB991; margin-bottom:1rem;
  animation: _pv_pulse 2s ease-in-out infinite;
}
._pv_stages { display:flex; gap:8px; margin-bottom:1rem; }
._pv_stage {
  flex:1; background:#0d0d0d; border:1px solid #1e1e1e; border-radius:4px;
  padding:.5rem .4rem; text-align:center; position:relative; overflow:hidden;
}
._pv_stage_lbl {
  font-family:'JetBrains Mono',monospace; font-size:0.56rem; letter-spacing:.12em;
  color:#555960; display:block; margin-bottom:4px;
}
._pv_stage_name {
  font-family:'DM Sans',sans-serif; font-size:.62rem; color:#8890a1;
}
._pv_scan_bar {
  position:absolute; top:0; left:-30%; width:30%; height:100%;
  background:linear-gradient(90deg,transparent,rgba(207,185,145,.18),transparent);
  animation: _pv_scan 2.4s ease-in-out infinite;
}
._pv_stage:nth-child(1) ._pv_scan_bar { animation-delay: 0s; }
._pv_stage:nth-child(2) ._pv_scan_bar { animation-delay: .3s; }
._pv_stage:nth-child(3) ._pv_scan_bar { animation-delay: .6s; }
._pv_stage:nth-child(4) ._pv_scan_bar { animation-delay: .9s; }
._pv_stage:nth-child(5) ._pv_scan_bar { animation-delay:1.2s; }
._pv_msgs { position:relative; height:1.1rem; margin-bottom:.9rem; }
._pv_msg {
  position:absolute; top:0; left:0; width:100%; opacity:0;
  font-family:'JetBrains Mono',monospace; font-size:.58rem; color:#8890a1;
}
._pv_msg:nth-child(1){animation:_pv_msg0 10s linear infinite}
._pv_msg:nth-child(2){animation:_pv_msg1 10s linear infinite}
._pv_msg:nth-child(3){animation:_pv_msg2 10s linear infinite}
._pv_msg:nth-child(4){animation:_pv_msg3 10s linear infinite}
._pv_msg:nth-child(5){animation:_pv_msg4 10s linear infinite}
._pv_bar_track {
  background:#111; border-radius:2px; height:3px; overflow:hidden;
}
._pv_bar_fill {
  height:100%; background:#CFB991; border-radius:2px;
  animation: _pv_bar 180s cubic-bezier(.1,.4,.3,1) forwards;
}
</style>
<div class="_pv_wrap">
  <div class="_pv_title">PIPELINE VALIDATION IN PROGRESS</div>
  <div class="_pv_stages">
    <div class="_pv_stage">
      <div class="_pv_scan_bar"></div>
      <span class="_pv_stage_lbl">S1</span>
      <span class="_pv_stage_name">Thesis</span>
    </div>
    <div class="_pv_stage">
      <div class="_pv_scan_bar"></div>
      <span class="_pv_stage_lbl">S2</span>
      <span class="_pv_stage_name">Signal</span>
    </div>
    <div class="_pv_stage">
      <div class="_pv_scan_bar"></div>
      <span class="_pv_stage_lbl">S3</span>
      <span class="_pv_stage_name">Confirm</span>
    </div>
    <div class="_pv_stage">
      <div class="_pv_scan_bar"></div>
      <span class="_pv_stage_lbl">S4</span>
      <span class="_pv_stage_name">Sizing</span>
    </div>
    <div class="_pv_stage">
      <div class="_pv_scan_bar"></div>
      <span class="_pv_stage_lbl">S5</span>
      <span class="_pv_stage_name">DSR Gate</span>
    </div>
  </div>
  <div class="_pv_msgs">
    <div class="_pv_msg">Fitting LP-IRF on training window — past data only&hellip;</div>
    <div class="_pv_msg">Stage 3 confirming sign direction per leg&hellip;</div>
    <div class="_pv_msg">Running DSR gate — deflating Sharpe by trial count&hellip;</div>
    <div class="_pv_msg">Computing OOS returns in test window&hellip;</div>
    <div class="_pv_msg">Monte Carlo random-admission baseline (500 draws)&hellip;</div>
  </div>
  <div class="_pv_bar_track"><div class="_pv_bar_fill"></div></div>
</div>
""", unsafe_allow_html=True)
            try:
                _pv = _run_pipeline_validator_cached(
                    all_r_concat, regimes,
                    train_days=756, test_days=63,
                    n_strategies=_effective_n,
                    n_random_trials=500,
                )
                st.session_state[_pv_key] = _pv
                # Persist to disk so the next session loads instantly
                try:
                    from src.utils.page_cache import save_cache as _sv
                    _sv(_PV_DISK_KEY, _pv)
                    _pv_disk_age = None  # now fresh
                except Exception:
                    pass
            except Exception as _pv_exc:
                st.error("Validation error.")
                _pv = None
            finally:
                _anim.empty()
        else:
            _pv = st.session_state.get(_pv_key)

        if _pv is not None:
            _pv_pass = _pv.get("passed", False)
            _pv_color = "#27ae60" if _pv_pass else "#c0392b"
            _pv_label = "PIPELINE PASSES" if _pv_pass else "PIPELINE FAILS (gaps not both positive)"

            st.markdown(
                f'<div style="background:#080808;border:2px solid {_pv_color};'
                f'border-radius:6px;padding:.8rem 1.2rem;margin-bottom:.8rem">'
                f'<div style="{_TP_M}font-size:0.65rem;font-weight:700;color:{_pv_color}">'
                f'{_pv_label}</div>'
                f'<div style="{_TP_M}font-size:0.58rem;color:#8890a1;margin-top:4px">'
                f'{_pv["n_windows"]} windows · {_pv["n_theses"]} theses · '
                f'756d train / 63d test</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Gap 1 and Gap 2
            _g1 = _pv.get("admitted_vs_rejected_gap")
            _g1p = _pv.get("admitted_vs_rejected_pval")
            _g2 = _pv.get("admitted_vs_random_gap")
            _g2p = _pv.get("random_p_value")

            _gap_cols = st.columns(2, gap="medium")
            for _gc_idx, (_gtitle, _gval, _gpval, _gdesc) in enumerate([
                ("GAP 1 — Admitted vs Rejected",
                 _g1, _g1p,
                 "Mean OOS return of admitted theses minus rejected theses (%). "
                 "Must be positive: the gates must discriminate."),
                ("GAP 2 — Admitted vs Random",
                 _g2, _g2p,
                 "Mean OOS return of admitted theses minus 500 random-draw baselines "
                 "(same N selected per window). Must be positive: gates must beat luck."),
            ]):
                with _gap_cols[_gc_idx]:
                    _gval_str = (f'{_gval:+.2f}%' if _gval is not None else 'n/a')
                    _gpval_str = (f'p={_gpval:.3f}' if _gpval is not None else '')
                    _gpass = _gval is not None and _gval > 0
                    _gcol  = "#27ae60" if _gpass else "#c0392b" if _gval is not None else "#555960"
                    st.markdown(
                        f'<div style="background:#090909;border:1px solid #1e1e1e;'
                        f'border-left:3px solid {_gcol};border-radius:4px;padding:.7rem 1rem">'
                        f'<div style="{_TP_M}font-size:0.56rem;letter-spacing:.12em;'
                        f'color:#8890a1;margin-bottom:5px">{_gtitle}</div>'
                        f'<div style="{_TP_M}font-size:1.3rem;font-weight:700;color:{_gcol}">'
                        f'{_gval_str}</div>'
                        f'<div style="{_TP_M}font-size:0.58rem;color:#555960;margin-top:2px">'
                        f'{_gpval_str}</div>'
                        f'<div style="{_TP_S}font-size:0.60rem;color:#8890a1;'
                        f'margin-top:6px;line-height:1.5">{_gdesc}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Bucket behavior
            st.markdown(
                f'<p style="{_TP_M}font-size:0.56rem;letter-spacing:.12em;color:#8890a1;'
                f'margin:1rem 0 .4rem">BUCKET BEHAVIOR — OOS MEAN RETURN (%)</p>',
                unsafe_allow_html=True,
            )
            _bkt_labels = {
                "admit":  ("ADMIT",  "#27ae60", "S3 confirmed + DSR ≥ 0.50"),
                "mt":     ("MT",     "#16a085", "Mechanism real, not tradeable (S3 ✓ + DSR < 0.50)"),
                "ie":     ("IE",     "#8e44ad", "Insufficient evidence (sign matched, n < 20)"),
                "reject": ("REJECT", "#c0392b", "S3 not confirmed or wrong OOS sign"),
            }
            _bkt_cols = st.columns(4, gap="small")
            for _bi, (_bkey, (_blabel, _bcol, _bdesc)) in enumerate(_bkt_labels.items()):
                with _bkt_cols[_bi]:
                    _b = _pv["buckets"].get(_bkey, {})
                    _bmean = _b.get("mean")
                    _bmean_str = (f'{_bmean:+.2f}%' if _bmean is not None else '—')
                    _bstd  = _b.get("std")
                    _bstd_str = (f'±{_bstd:.2f}%' if _bstd is not None else '')
                    _bn = _b.get("n_obs", 0)
                    st.markdown(
                        f'<div style="background:#080808;border:1px solid #1e1e1e;'
                        f'border-top:2px solid {_bcol};border-radius:4px;padding:.55rem .7rem">'
                        f'<div style="{_TP_M}font-size:0.56rem;letter-spacing:.12em;'
                        f'color:{_bcol};margin-bottom:4px">{_blabel}</div>'
                        f'<div style="{_TP_M}font-size:1.0rem;font-weight:700;color:{_bcol}">'
                        f'{_bmean_str}</div>'
                        f'<div style="{_TP_M}font-size:0.55rem;color:#555960">'
                        f'{_bstd_str}  n={_bn}</div>'
                        f'<div style="{_TP_S}font-size:0.55rem;color:#555960;'
                        f'margin-top:4px;line-height:1.4">{_bdesc}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Random distribution histogram
            _rdist = _pv.get("random_distribution", [])
            if _rdist and _g2 is not None:
                import numpy as _np_tp
                _fig_rdist = _go_tp.Figure()
                _fig_rdist.add_trace(_go_tp.Histogram(
                    x=_rdist, nbinsx=30,
                    marker_color="#2980b9", opacity=0.7,
                    name="Random admission",
                ))
                _adm_mean_oos = (_g2 + float(_np_tp.mean(_rdist))) if _rdist else None
                if _adm_mean_oos is not None:
                    _fig_rdist.add_vline(
                        x=_adm_mean_oos, line_color="#CFB991", line_dash="dash", line_width=2,
                        annotation_text="Pipeline", annotation_font=dict(size=8, color="#CFB991"),
                    )
                _fig_rdist.update_layout(
                    title="Gap 2: Pipeline vs Random Admission (500 draws)",
                    height=200, margin=dict(l=0, r=0, t=28, b=0),
                    paper_bgcolor="#080808", plot_bgcolor="#080808",
                    font=dict(family="JetBrains Mono", size=8),
                    xaxis=dict(title="Mean OOS return (%)", color="#555960", gridcolor="#1e1e1e"),
                    yaxis=dict(title="Count", color="#555960", gridcolor="#1e1e1e"),
                    title_font=dict(size=9, color="#8890a1"),
                    showlegend=False,
                )
                _fig_rdist.update_xaxes(tickfont=dict(color="#c8c8c8"))
                _fig_rdist.update_yaxes(tickfont=dict(color="#c8c8c8"))
                st.plotly_chart(_fig_rdist, use_container_width=True)

        st.markdown("---")

        # ── PART 3: Worked Example — Pipeline decision trace for one thesis ───
        st.markdown(
            f'<p style="{_TP_M}font-size:0.58rem;font-weight:700;letter-spacing:.10em;'
            f'color:#CFB991;margin-bottom:.3rem">WORKED EXAMPLE — PIPELINE DECISION TRACE</p>'
            f'<p style="{_TP_S}font-size:0.65rem;color:#8890a1;margin-bottom:.6rem;'
            f'line-height:1.6">'
            f'Thesis: Iran conflict → Strait of Hormuz → WTI crude supply shock → S&amp;P 500 margin '
            f'compression. Left: the mechanism narrative and Stage 3 confirmation result (how the pipeline '
            f'reads the data). Right: the pipeline\'s admit/reject/MT/IE decision at each walk-forward window '
            f'and the OOS return that followed — this is approach-testing, not trade performance.</p>',
            unsafe_allow_html=True,
        )

        # Worked example thesis spec (constructed inline, not from a catalogue)
        _WE_NAME    = "Long WTI Crude / Short S&P 500 (Iran / Hormuz)"
        _WE_SHOCK   = ("Iran conflict escalation → Strait of Hormuz partial or full closure → "
                       "OPEC+ supply disruption → WTI spot spike. S&amp;P 500 sectors with high "
                       "energy input costs (consumer discretionary, industrials, airlines) face "
                       "immediate margin compression. The transmission is via input-cost inflation, "
                       "not demand destruction.")
        _WE_CHANNELS = ["oil_gas", "chokepoint", "equity_sector", "inflation"]
        _WE_CONFLICT = "iran_conflict"
        _WE_CHOKEPOINT = "Strait of Hormuz"
        _WE_PRED    = {"WTI Crude Oil": +1, "S&P 500": -1}
        _WE_HORIZON = 20
        _WE_PERSIST = ("WTI price spikes from supply shocks are not quickly demand-destroyed at "
                       "short horizons — consumption is inelastic for 2-4 weeks. Equity margin "
                       "compression persists until the next earnings revision cycle (~6-8 weeks). "
                       "The Brent-WTI spread widening also signals physical tightness independent "
                       "of financial positioning.")
        _WE_ASSETS  = ["WTI Crude Oil", "S&P 500"]
        _WE_DIRS    = ["Long", "Short"]
        _WE_REGIME  = [1, 2]

        _we_col1, _we_col2 = st.columns([1, 1], gap="medium")

        with _we_col1:
            # Stage 1
            st.markdown(
                f'<div style="background:#0a0a0a;border:1px solid #1e1e1e;'
                f'border-left:3px solid #CFB991;border-radius:4px;'
                f'padding:.7rem 1rem;margin-bottom:.5rem">'
                f'<div style="{_TP_M}font-size:0.56rem;letter-spacing:.12em;'
                f'color:#CFB991;margin-bottom:6px">STAGE 1 — THESIS ✓</div>'
                f'<div style="{_TP_S}font-size:0.68rem;color:#e8e9ed;line-height:1.6">'
                f'<b>Shock:</b> {_WE_SHOCK}<br>'
                f'<b>Channels:</b> {", ".join(_WE_CHANNELS)}<br>'
                f'<b>Conflict:</b> {_WE_CONFLICT}<br>'
                f'<b>Chokepoint:</b> {_WE_CHOKEPOINT}<br>'
                f'<b>Horizon:</b> {_WE_HORIZON} trading days<br>'
                f'<b>Persistence:</b> {_WE_PERSIST}'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            # Stage 2
            _we_s2_ok = all(a in set(all_r_concat.columns) for a in _WE_ASSETS)
            _we_s2_color = "#27ae60" if _we_s2_ok else "#c0392b"
            _we_leg_str = " · ".join(
                f'<span style="color:{"#27ae60" if d=="Long" else "#c0392b"}">{d}</span> '
                f'{a} ({"+" if _WE_PRED.get(a,0)>0 else "−"}1)'
                for a, d in zip(_WE_ASSETS, _WE_DIRS)
            )
            st.markdown(
                f'<div style="background:#0a0a0a;border:1px solid #1e1e1e;'
                f'border-left:3px solid {_we_s2_color};border-radius:4px;'
                f'padding:.6rem 1rem;margin-bottom:.5rem">'
                f'<div style="{_TP_M}font-size:0.56rem;letter-spacing:.12em;'
                f'color:{_we_s2_color};margin-bottom:5px">'
                f'STAGE 2 — SIGNAL {"✓" if _we_s2_ok else "✗"}</div>'
                f'<div style="{_TP_S}font-size:0.68rem;color:#c8c8c8">{_we_leg_str}</div>'
                f'<div style="{_TP_M}font-size:0.58rem;color:{_we_s2_color};margin-top:4px">'
                f'{"All legs present in return data." if _we_s2_ok else "Missing legs — thesis untestable."}'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            # Stage 3 (using cached computation on full dataset)
            if _we_s2_ok:
                _we_pred_items = tuple(sorted(_WE_PRED.items()))
                try:
                    _we_s3d = _thesis_stage3_cached(
                        strat_name=_WE_NAME,
                        conflict_id=_WE_CONFLICT,
                        regime_list=tuple(_WE_REGIME),
                        assets=tuple(_WE_ASSETS),
                        directions=tuple(_WE_DIRS),
                        predicted_sign_items=_we_pred_items,
                        horizon_days=_WE_HORIZON,
                        _all_r=all_r_concat,
                        _regimes=regimes,
                        _len_hint=len(all_r_concat),
                    )
                except Exception as _e:
                    _we_s3d = {
                        "stage_passed": False, "sign_matched": False, "track": "error",
                        "confirmation_score": 0.0, "per_leg": {}, "irf_df_records": None,
                        "regime_stats": None, "rejection_reason": str(_e),
                    }
                _we_s3_ok   = bool(_we_s3d.get("stage_passed", False))
                _we_s3_sign = bool(_we_s3d.get("sign_matched", False))
                _we_track   = _we_s3d.get("track", "—")
                _we_score   = float(_we_s3d.get("confirmation_score", 0.0))
                _we_s3_color = "#27ae60" if _we_s3_ok else "#e67e22"
                _we_per_leg = _we_s3d.get("per_leg", {})
                _we_leg_rows = "".join(
                    f'<tr><td style="color:#c8c8c8;padding-right:8px">{_a}</td>'
                    f'<td style="color:{"#27ae60" if _v.get("matched_sign") else "#c0392b"}">'
                    f'{"✓" if _v.get("matched_sign") else "✗"} sign</td>'
                    f'<td style="color:{"#27ae60" if _v.get("significant") else "#8890a1"}">'
                    f'{"sig" if _v.get("significant") else "n.s."}</td>'
                    + (f'<td style="color:#8890a1">{_v.get("irf_coef","")}</td>'
                       if _we_track.startswith("lp") else
                       f'<td style="color:#8890a1">{_v.get("mean_ret","—")}%</td>')
                    + '</tr>'
                    for _a, _v in _we_per_leg.items()
                )
                if _we_s3_ok:
                    _we_s3_verdict = f"CONFIRMED ({_we_score:.0%} legs)"
                elif _we_s3_sign:
                    _we_s3_verdict = f"SIGN MATCHED, NOT SIGNIFICANT — IE if n &lt; 20"
                else:
                    _we_s3_verdict = "REJECTED — predicted sign not matched"
                st.markdown(
                    f'<div style="background:#0a0a0a;border:1px solid #1e1e1e;'
                    f'border-left:3px solid {_we_s3_color};border-radius:4px;'
                    f'padding:.6rem 1rem;margin-bottom:.5rem">'
                    f'<div style="{_TP_M}font-size:0.56rem;letter-spacing:.12em;'
                    f'color:{_we_s3_color};margin-bottom:5px">'
                    f'STAGE 3 — CONFIRMATION ({_we_track.upper()}) · {_we_s3_verdict}</div>'
                    + (f'<table style="{_TP_M}font-size:0.60rem;border-collapse:collapse">'
                       f'<tr><th style="color:#555960;text-align:left">Leg</th>'
                       f'<th style="color:#555960">Sign</th>'
                       f'<th style="color:#555960">Sig</th>'
                       f'<th style="color:#555960">{"IRF coef" if _we_track.startswith("lp") else "Regime ret"}</th></tr>'
                       + _we_leg_rows + '</table>'
                       if _we_leg_rows else
                       f'<div style="{_TP_S}font-size:0.63rem;color:#8890a1">'
                       f'{_we_s3d.get("rejection_reason","No result")}</div>')
                    + '</div>',
                    unsafe_allow_html=True,
                )
            else:
                _we_s3d  = {}
                _we_s3_ok = False
                _we_s3_sign = False
                _we_track = "—"
                st.markdown(
                    f'<div style="{_TP_M}font-size:0.60rem;color:#c0392b;padding:.4rem">'
                    f'Stage 3 skipped — Stage 2 failed.</div>',
                    unsafe_allow_html=True,
                )

        with _we_col2:
            # Stage 3 chart
            if _we_s2_ok and _we_s3d:
                if _we_track.startswith("lp") and _we_s3d.get("irf_df_records"):
                    import pandas as _pd_tp
                    _we_irf = _pd_tp.DataFrame(_we_s3d["irf_df_records"])
                    _fig_we = _go_tp.Figure()
                    for _ai, _a in enumerate(_we_irf["asset"].unique() if "asset" in _we_irf.columns else []):
                        _ad = _we_irf[_we_irf["asset"] == _a]
                        _col_a = ["#CFB991","#2980b9"][_ai % 2]
                        _psign = _WE_PRED.get(_a, 0)
                        _cifa  = "rgba(39,174,96,0.12)" if _psign == 1 else "rgba(231,76,60,0.12)"
                        _fig_we.add_trace(_go_tp.Scatter(
                            x=_ad["horizon"], y=_ad["ci_hi"], mode="lines",
                            line=dict(width=0), showlegend=False,
                        ))
                        _fig_we.add_trace(_go_tp.Scatter(
                            x=_ad["horizon"], y=_ad["ci_lo"], mode="lines",
                            line=dict(width=0), fill="tonexty", fillcolor=_cifa, showlegend=False,
                        ))
                        _fig_we.add_trace(_go_tp.Scatter(
                            x=_ad["horizon"], y=_ad["coef"], mode="lines+markers",
                            name=_a, line=dict(color=_col_a, width=1.5), marker=dict(size=4),
                        ))
                    _fig_we.add_hline(y=0, line_dash="dot", line_color="#555960")
                    _fig_we.add_vline(
                        x=_WE_HORIZON, line_dash="dash", line_color="#CFB991", line_width=1,
                        annotation_text=f"h={_WE_HORIZON}d",
                        annotation_font=dict(size=8, color="#CFB991"),
                    )
                    _fig_we.update_layout(
                        title="LP-IRF: Oil shock → WTI and S&P 500",
                        height=240, margin=dict(l=0, r=0, t=30, b=0),
                        paper_bgcolor="#080808", plot_bgcolor="#080808",
                        font=dict(family="JetBrains Mono", size=8),
                        legend=dict(font=dict(size=7), bgcolor="rgba(0,0,0,0)"),
                        xaxis=dict(title="Horizon (days)", color="#555960", gridcolor="#1e1e1e"),
                        yaxis=dict(title="Coef", color="#555960", gridcolor="#1e1e1e"),
                        title_font=dict(size=9, color="#8890a1"),
                    )
                    _fig_we.update_xaxes(tickfont=dict(color="#c8c8c8"))
                    _fig_we.update_yaxes(tickfont=dict(color="#c8c8c8"))
                    st.plotly_chart(_fig_we, use_container_width=True)
                elif _we_s3d.get("per_leg"):
                    _we_bar_a = list(_we_s3d["per_leg"].keys())
                    _we_bar_m = [_we_s3d["per_leg"][_a].get("mean_ret", 0) or 0 for _a in _we_bar_a]
                    _we_bar_c = ["#27ae60" if _we_s3d["per_leg"][_a].get("matched_sign") else "#c0392b"
                                 for _a in _we_bar_a]
                    _fig_we = _go_tp.Figure(data=[_go_tp.Bar(
                        x=_we_bar_a, y=_we_bar_m, marker_color=_we_bar_c,
                        text=[f'{_v:.2f}%' for _v in _we_bar_m],
                        textfont=dict(size=8, family="JetBrains Mono"),
                        textposition="outside",
                    )])
                    _fig_we.add_hline(y=0, line_color="#555960", line_dash="dot")
                    _fig_we.update_layout(
                        title="Regime-Conditional Returns in Trigger Regime",
                        height=240, margin=dict(l=0, r=0, t=30, b=0),
                        paper_bgcolor="#080808", plot_bgcolor="#080808",
                        font=dict(family="JetBrains Mono", size=8),
                        xaxis=dict(color="#555960"),
                        yaxis=dict(title="Mean return (%)", color="#555960", gridcolor="#1e1e1e"),
                        title_font=dict(size=9, color="#8890a1"),
                    )
                    _fig_we.update_xaxes(tickfont=dict(color="#c8c8c8"))
                    _fig_we.update_yaxes(tickfont=dict(color="#c8c8c8"))
                    st.plotly_chart(_fig_we, use_container_width=True)

            # Decision trace — pipeline's classification of this thesis at each window
            _we_trace = (_pv or {}).get("worked_example_trace", [])
            if _we_trace:
                import numpy as _np_we
                _dec_color = {
                    "admit":  "#27ae60",
                    "mt":     "#16a085",
                    "ie":     "#8e44ad",
                    "reject": "#c0392b",
                }
                # Decision timeline chart
                _trace_x   = [_r["test_end"] for _r in _we_trace]
                _trace_oos = [_r["oos_return"] if _r["oos_return"] is not None else 0.0
                              for _r in _we_trace]
                _trace_dec = [_r["decision"] for _r in _we_trace]
                _trace_col = [_dec_color.get(_d, "#555960") for _d in _trace_dec]
                _fig_trace = _go_tp.Figure()
                _fig_trace.add_hline(y=0, line_dash="dot", line_color="#555960", line_width=1)
                _fig_trace.add_trace(_go_tp.Bar(
                    x=_trace_x,
                    y=_trace_oos,
                    marker_color=_trace_col,
                    customdata=[[_d, f'{_r["dsr_prob"]:.0%}',
                                 _r["oos_n_signals"],
                                 '✓' if _r["stage3_confirmed"] else ('~ sign' if _r["stage3_sign"] else '✗')]
                                for _r, _d in zip(_we_trace, _trace_dec)],
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Decision: %{customdata[0]}<br>"
                        "OOS return: %{y:.2f}%<br>"
                        "DSR (train): %{customdata[1]}<br>"
                        "OOS signals: %{customdata[2]}<br>"
                        "S3: %{customdata[3]}<extra></extra>"
                    ),
                ))
                _fig_trace.update_layout(
                    title="Pipeline Decision Trace — WTI / S&P 500 across Walk-Forward Windows",
                    height=240, margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor="#080808", plot_bgcolor="#080808",
                    font=dict(family="JetBrains Mono", size=8),
                    xaxis=dict(title="Test window end", color="#555960", gridcolor="#1e1e1e",
                               tickangle=-45, tickfont=dict(size=7, color="#c8c8c8")),
                    yaxis=dict(title="OOS return (%)", color="#555960", gridcolor="#1e1e1e"),
                    title_font=dict(size=9, color="#8890a1"),
                    showlegend=False,
                )
                _fig_trace.update_xaxes(tickfont=dict(color="#c8c8c8"))
                _fig_trace.update_yaxes(tickfont=dict(color="#c8c8c8"))
                st.plotly_chart(_fig_trace, use_container_width=True)

                # Summary of decision-conditional means
                _adm_rets = [_r["oos_return"] for _r in _we_trace
                             if _r["decision"] == "admit" and _r["oos_return"] is not None]
                _rej_rets = [_r["oos_return"] for _r in _we_trace
                             if _r["decision"] == "reject" and _r["oos_return"] is not None]
                _mt_rets  = [_r["oos_return"] for _r in _we_trace
                             if _r["decision"] == "mt" and _r["oos_return"] is not None]
                _ie_rets  = [_r["oos_return"] for _r in _we_trace
                             if _r["decision"] == "ie" and _r["oos_return"] is not None]
                _summary_parts = []
                for _label, _vals, _col in [
                    ("admit",  _adm_rets, "#27ae60"),
                    ("reject", _rej_rets, "#c0392b"),
                    ("mt",     _mt_rets,  "#16a085"),
                    ("ie",     _ie_rets,  "#8e44ad"),
                ]:
                    if _vals:
                        _m = float(_np_we.mean(_vals))
                        _n = len(_vals)
                        _summary_parts.append(
                            f'<span style="color:{_col};font-weight:700">{_label.upper()}</span>'
                            f'<span style="color:#8890a1"> {_m:+.2f}% (n={_n})</span>'
                        )
                if _summary_parts:
                    st.markdown(
                        f'<div style="{_TP_M}font-size:0.60rem;margin-top:.4rem;'
                        f'padding:.5rem .8rem;background:#090909;border-radius:4px;'
                        f'border:1px solid #1e1e1e">'
                        + " &nbsp;·&nbsp; ".join(_summary_parts)
                        + f'<span style="color:#555960;font-size:0.55rem"> — mean OOS return by pipeline decision</span>'
                        + '</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f'<div style="{_TP_M}font-size:0.62rem;color:#555960;'
                    f'padding:.6rem;border:1px dashed #2a2a2a;border-radius:4px">'
                    f'Run Walk-Forward Validation above to see the pipeline decision trace '
                    f'for this thesis across all historical windows.</div>',
                    unsafe_allow_html=True,
                )

        # Individual strategy cards removed — pipeline is the deliverable, not per-thesis P&L.
        # The worked example shows the pipeline's decision trace across windows, not a trade grade.

    _section_note(
        "Trade ideas are generated from historical cross-asset patterns and regime signals. "
        "All ideas are illustrative and must be validated against current market structure, "
        "liquidity, and position sizing constraints before implementation."
    )

    _page_conclusion(
        "Framework",
        "The regime-based trade generation framework matches historical spillover patterns "
        "to current correlation regimes. Crisis regimes activate hedging and divergence plays; "
        "normal regimes favour growth-correlated long positioning. "
        "Use Granger and transfer entropy results from the Spillover page to validate lead-lag direction."
    )


    _page_footer()
