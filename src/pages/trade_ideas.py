"""
Page 6 - Trade Ideas
Regime-triggered + conflict-driven cross-asset trade cards.
Integrates conflict exposure scoring, scenario-aware payoff tables, QC grading,
filter sidebar, and agent debate threads.
"""

from __future__ import annotations

import datetime

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
         "insight": "Liquidity drives markets above near-term earnings by an order of magnitude. Follow the money supply and central bank balance sheet - they are the real alpha signal."},
        {"manager": "S Naren",             "archetype": "DYNAMIC ALLOCATOR",
         "insight": "Dynamic asset allocation based on macro valuation metrics like Market Cap-to-GDP. Shift systematically; mean reversion is the law of financial gravity."},
        {"manager": "Shankar Sharma",       "archetype": "MACRO INFLECTION",
         "insight": "Global macro allocation based on structural data, not narrative. The largest forces - fiscal, monetary, demographic - move asset prices over years, not quarters."},
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
         "insight": "Invest at the absolute bottom of the capital structure. Distressed debt converts to dominant post-restructuring equity - the Appaloosa playbook from the 2009 financials trade."},
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
         "insight": "Patience in contrariness: invest in deeply undervalued sectors facing cyclical headwinds. India's oil import shock is a cyclical headwind - not a structural ruin."},
        {"manager": "Prashant Jain",        "archetype": "CONTRA-CYCLICAL",
         "insight": "Contra-cyclical value discipline: buy out-of-favour sectors at peak cyclical pain. Ask whether the business survives over a 10-year horizon - not 10 weeks."},
        {"manager": "John Templeton",       "archetype": "MAX PESSIMISM",
         "insight": "Maximum pessimism in EM is the entry point. INR stress and crude shock at extremes is precisely the Templeton setup. Buy when blood is in the streets."},
    ],
    "Asia Divergence": [
        {"manager": "Shankar Sharma",       "archetype": "MACRO INFLECTION",
         "insight": "Spotting inflection points early across cap ranges. China property is structural, not cyclical. Japan is the mirror image - BOJ policy normalisation is a decade-long re-rating."},
        {"manager": "Kerr Neilson",         "archetype": "GLOBAL CONTRARIAN",
         "insight": "True global sourcing agility: allocate into regions when valuations become compelling. Japan at Shiller CAPE 22x vs China property distress is a screaming divergence trade."},
        {"manager": "S Naren",             "archetype": "DYNAMIC ALLOCATOR",
         "insight": "Macro risk tracking: China leverage cycle is in systemic unwind. Mean reversion for Japan is supported by BOJ policy normalisation and structural Yen tailwind."},
    ],
    "Dollar Cycle": [
        {"manager": "Stanley Druckenmiller","archetype": "MACRO LIQUIDITY",
         "insight": "The dollar cycle is the single most powerful force for EM asset re-rating. Liquidity flows globally - when DXY peaks, EM assets inflect sharply."},
        {"manager": "John Templeton",       "archetype": "MAX PESSIMISM",
         "insight": "Buy at maximum pessimism. When EM is universally abandoned at a dollar peak, it is universally mispriced. The fundamental equation reverses."},
        {"manager": "Kerr Neilson",         "archetype": "GLOBAL CONTRARIAN",
         "insight": "True global sourcing agility: allocate freely across geographies when dollar-driven valuation dislocations create compelling entry points in EM."},
    ],
}

# ── Specific tradeable instruments for each trade ──────────────────────────
# {trade_name: {asset_name: "TICKER - description"}}
# Used for display only - backtest uses the asset name against return data columns.
_TRADE_TICKERS: dict[str, dict[str, str]] = {
    "Long Gold / Short Eurostoxx 50": {
        "Gold":          "GLD - SPDR Gold Shares (NYSE)",
        "Eurostoxx 50":  "FEZ - SPDR Euro Stoxx 50 ETF (NYSE)",
    },
    "Long Natural Gas / Short Nikkei 225": {
        "Natural Gas":   "UNG - United States Natural Gas Fund (NYSE)",
        "Nikkei 225":    "EWJ - iShares MSCI Japan ETF (NYSE)",
    },
    "Long Wheat / Long Gold / Short Emerging Markets": {
        "Wheat":         "WEAT - Teucrium Wheat Fund (NYSE)",
        "Gold":          "GLD - SPDR Gold Shares (NYSE)",
        "Sensex":        "EEM - iShares MSCI Emerging Markets ETF (NYSE)",
    },
    "Long Copper / Long S&P 500": {
        "Copper":        "CPER - United States Copper Index Fund (NYSE)",
        "S&P 500":       "SPY - SPDR S&P 500 ETF Trust (NYSE)",
    },
    "Long WTI Crude / Short S&P 500 Energy-Heavy Sectors": {
        "WTI Crude Oil": "USO - United States Oil Fund (NYSE) | XLE short for sector precision",
        "S&P 500":       "SPY - SPDR S&P 500 ETF Trust (NYSE)",
    },
    "Long Gold, Long Silver / Short Copper, Short Shanghai Comp": {
        "Gold":          "GLD - SPDR Gold Shares (NYSE)",
        "Silver":        "SLV - iShares Silver Trust (NYSE)",
        "Copper":        "CPER - United States Copper Index Fund (NYSE)",
        "Shanghai Comp": "MCHI - iShares MSCI China ETF (NYSE)",
    },
    "Short BDC Basket / Long HY Credit Protection": {
        "Ares Capital (ARCC)": "ARCC - Ares Capital Corp (NASDAQ) - short",
        "Blue Owl (OBDC)":     "OBDC - Blue Owl Capital Corp (NYSE) - short",
        "Gold":                "GLD - SPDR Gold Shares (NYSE) - long",
    },
    "Long TLT / Short HYG (Flight to Quality)": {
        "US 20Y+ Treasury (TLT)": "TLT - iShares 20+ Year Treasury Bond ETF (NYSE)",
        "HY Corporate (HYG)":     "HYG - iShares iBoxx $ High Yield Corporate Bond ETF (NYSE) - short",
    },
    "Long TIP / Short TLT (Inflation Breakeven Trade)": {
        "TIPS / Inflation (TIP)":  "TIP - iShares TIPS Bond ETF (NYSE)",
        "US 20Y+ Treasury (TLT)":  "TLT - iShares 20+ Year Treasury Bond ETF (NYSE) - short",
    },
    "Long Brent Crude / Short Nifty 50 (India Import Shock)": {
        "Brent Crude":  "BNO - United States Brent Oil Fund (NYSE)",
        "Nifty 50":     "INDY - iShares India 50 ETF (NYSE) | NIFTYBEES.NS (NSE)",
    },
    "Long Gold / Short INR (India Geopolitical Hedge)": {
        "Gold":    "GLD - SPDR Gold Shares (NYSE)",
        "USD/INR": "USDINR=X - Forex spot | USDINR futures on NSE",
    },
    "Long EMB / Short DXY (Dollar Debasement - EM Relief)": {
        "EM USD Bonds (EMB)":  "EMB - iShares J.P. Morgan USD EM Bond ETF (NYSE)",
        "DXY (Dollar Index)":  "UUP - Invesco DB US Dollar Index Bullish Fund (NYSE) - short",
        "Gold":                "GLD - SPDR Gold Shares (NYSE)",
    },
    "Long Gold / Short TLT (Fiscal Dominance / Dollar Debasement)": {
        "Gold":                    "GLD - SPDR Gold Shares (NYSE) | GDX for miners leverage",
        "US 20Y+ Treasury (TLT)": "TLT - iShares 20+ Year Treasury Bond ETF (NYSE) - short",
    },
    "Long EM Asia / Short DXY (Max Pessimism EM Reversal)": {
        "Shanghai Comp":      "MCHI - iShares MSCI China ETF (NYSE) | 2800.HK Tracker Fund",
        "Sensex":             "INDA - iShares MSCI India ETF (NYSE) | NIFTYBEES.NS",
        "DXY (Dollar Index)": "UUP - Invesco DB US Dollar Index Bullish Fund (NYSE) - short",
    },
    "Long LQD / Short HYG (Credit Cycle Peak - Quality Flight)": {
        "IG Corporate (LQD)": "LQD - iShares iBoxx $ IG Corporate Bond ETF (NYSE)",
        "HY Corporate (HYG)": "HYG - iShares iBoxx $ HY Corporate Bond ETF (NYSE) - short",
    },
    "Long SHY / Long Gold (Fed Pivot Front-Run)": {
        "US 1-3Y Treasury (SHY)": "SHY - iShares 1-3 Year Treasury Bond ETF (NYSE)",
        "Gold":                    "GLD - SPDR Gold Shares (NYSE) | GDX for leveraged exposure",
    },
    "Long Nifty 50 / Short Brent (India Rate Cut + Oil Tailwind)": {
        "Nifty 50":     "INDA - iShares MSCI India ETF (NYSE) | NIFTYBEES.NS (NSE)",
        "Brent Crude":  "BNO - United States Brent Oil Fund (NYSE) - short",
    },
    "Short Shanghai Comp / Long Nikkei 225 (China Deflation vs Japan Reflation)": {
        "Shanghai Comp": "FXI - iShares China Large-Cap ETF (NYSE) - short via puts",
        "Nikkei 225":    "EWJ - iShares MSCI Japan ETF (NYSE)",
    },
}


# ── Trade idea library ─────────────────────────────────────────────────────
# IMMUTABLE SOURCE. Module globals are shared across every Streamlit session in
# a process, so the pipeline must never mutate this in place (it stamps
# is_eligible / alloc_weight / rank on each dict). page_trade_ideas() works on a
# per-run deepcopy bound to the local name _TRADE_LIBRARY; read-only consumers
# (warmup, stage-3) may reference this base directly.
_TRADE_LIBRARY_BASE = [
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
            "real rates collapse and fiat currency credibility erodes. Gold - the monetary metal - outperforms nominal "
            "Treasuries structurally. Druckenmiller: 'Liquidity drives markets above all else. When the Fed prints and "
            "fiscal expands, the answer is always gold over bonds.' Watsa: macro-hedging against systemic dollar "
            "debasement. Templeton: real returns focus - nominal treasury yields minus realized inflation = deeply "
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
            "Kerr Neilson: 'True global sourcing agility - allocate into regions when valuations become compelling.' "
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
        "name":           "Long LQD / Short HYG (Credit Cycle Peak - Quality Flight)",
        "rationale":      (
            "Howard Marks: 'Gauge risk as permanent loss of capital, not volatility. Calibrate aggressiveness based on "
            "where the market stands in the credit cycle.' When HY–IG spreads compress to cycle lows (<200 bps), "
            "leveraged loan issuance is at records, and the Fed has reached terminal rate - credit cycle is at peak. "
            "Investment-grade bonds offer duration safety as HY reprices default risk. Marks: 'Asymmetry of returns - "
            "capture upside while protecting downside.' Klarman: forced institutional selling driven by credit rating "
            "downgrades is the Baupost entry point. "
            "ETFs: Long LQD (iShares iBoxx $ Investment Grade Corporate Bond ETF); "
            "Short HYG (iShares iBoxx $ High Yield Corporate Bond ETF). "
            "Alternative: CDX IG index long vs CDX HY index short for pure credit spread pair."
        ),
        "entry":          "HY OAS (BAMLH0A0HYM2) − IG OAS (BAMLC0A0CM) <200 bps AND leveraged loan issuance at 3Y high AND Fed funds held >6 months at cycle high",
        "exit":           "HY spreads widen to 400 bps; credit cycle turns; recession confirmed by 2Q negative GDP",
        "stop":           "LQD −5% from entry (bear-steepening scenario); HYG −8% in risk-off panic (correlations spike - exit both legs)",
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
            "bull-steepening - the next macro move is rate cuts. 2Y Treasuries front-run the cut cycle with "
            "mathematical certainty (SHY +4–6% per 100 bps cut). Gold front-runs the real yield collapse and "
            "dollar weakness that follows easing. S Naren: 'Dynamic asset allocation - shift systematically to "
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
            "and corporate margins expand. This is the inverse of the India Import Shock trade - it fires when oil "
            "stress reverses. Prashant Jain: 'Business longevity - ask whether the company grows cash flows over 10 "
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
            "cyclical - $300B+ in offshore dollar debt with demographic reversal and structural oversupply.' Japan is "
            "the mirror image: Yen weakness (USD/JPY >148) boosts export profits, domestic reflation is accelerating, "
            "and the Nikkei Shiller CAPE at ~22x is cheap vs its own 30-year history. Kerr Neilson: 'True global "
            "sourcing agility - Japan valuations are compelling. China property is a regulatory and demographic trap. "
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
        from src.data.loader import _yf_download   # process-wide yfinance lock
        sector_set = set(sectors)
        tickers = [
            t for t, (_, s) in _SP500_UNIVERSE.items()
            if not sector_set or s in sector_set
        ]
        if not tickers:
            return {}
        raw = _yf_download(tickers, period="5d", progress=False, auto_adjust=True)["Close"]
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
        "S&P 500 STOCK REFERENCE PRICES (live - use these for specific entry/target/stop):"
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
/* ── Trade Ideas - Design System ───────────────────────────────────────────
   Typography scale (matches shared.py + palette.py):
     0.50rem  JetBrains Mono  uppercase labels, badges, chips, dims
     0.52rem  JetBrains Mono  header labels (slightly heavier weight labels)
     0.63rem  DM Sans / Mono  cell values, strip values, secondary data
     0.70rem  DM Sans         body / rationale (matches _page_intro scale)
     0.81rem  DM Sans bold    card trade name (primary heading in card)
     0.94rem  JetBrains Mono  KPI number (page-level metrics)
   Colors: palette.py - TEXT #e8e9ed · TEXT_SOFT #c8c8c8 · TEXT_MUTED #b8b8b8
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
                base *= (1 + sas / 300)  # smaller boost - safe haven long is defensive
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


@st.cache_data(show_spinner=False, max_entries=400, ttl=3600)
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


def _library_stage3_results(_all_r: pd.DataFrame, _regimes,
                            trades: list[dict] | None = None) -> dict[str, dict]:
    """
    Stage-3 confirmation for every thesis whose legs all exist in the loaded
    returns frame. Predicted signs derive mechanically from the legs (Long →
    +1, Short → −1). Phantom-leg entries are skipped - Stage 3 cannot run
    without data, and the leg check in annotate_eligibility() locks them with
    the specific missing legs named.

    `trades` defaults to the static library base; pass the combined
    static+live-generated set to confirm generated candidates too.

    Not cached itself: each _thesis_stage3_cached call below carries its own
    daily cache (nesting cached calls trips Streamlit's cache guard).
    """
    out: dict[str, dict] = {}
    for tr in (trades if trades is not None else _TRADE_LIBRARY_BASE):
        assets = tr.get("assets") or []
        dirs = tr.get("direction") or []
        if not assets or any(a not in _all_r.columns for a in assets):
            continue
        pred = tuple(sorted(
            (a, 1 if d == "Long" else -1) for a, d in zip(assets, dirs)
        ))
        hold = _parse_holding_days(tr, default=20)
        try:
            out[tr["name"]] = _thesis_stage3_cached(
                tr["name"], tr.get("conflict_id"), tuple(tr.get("regime", [])),
                tuple(assets), tuple(dirs), pred, hold,
                _all_r, _regimes, _len_hint=len(_all_r),
            )
        except Exception as _e:
            out[tr["name"]] = {"stage_passed": False,
                               "rejection_reason": f"Stage 3 error: {type(_e).__name__}"}
    return out


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
    Builds the validation universe internally - these theses are the
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
            "Full crisis: VIX > 35, DXY trending up - monetary vs industrial metals decouple.",
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
    # Suppress low-confidence generated CONFLICT ideas to keep signal-to-noise
    # high. The signal-ranked candidate universe (relative-value / directional /
    # safe-haven) is intentionally broad - it's meant to be browsed, and the
    # eligibility → DSR gate does the real filtering - so it is exempt.
    _conf_raw = float(trade.get("confidence", 0.60))
    if (trade.get("generated") and not trade.get("screened")
            and _conf_raw < 0.55):
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
    # AI-structured trades carry tickers as a single descriptive string - surface it directly
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
            f'<div class="ti-cell-val">{val or " - "}</div>'
            f'</div>'
        )

    grid_html = (
        '<div class="ti-grid">'
        + _cell("Entry",  trade.get("entry", " - "), "#CFB991")
        + _cell("Exit",   trade.get("exit",  " - "), "#8890a1")
        + _cell("Risk",   trade.get("risk",  " - "), "#c0392b", 'style="border-left:2px solid #220000"')
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
            + _cell("Stop",         trade.get("stop",           " - "), "#27ae60")
            + _cell("Target",       trade.get("target",         " - "), "#27ae60")
            + _cell("Invalidation", trade.get("invalidation",   " - "), "#2980b9")
            + _cell("Hold Period",  trade.get("holding_period", " - "), "#8890a1")
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
                        _pass_reasons.append(f"SAS {_avg_sas:.0f} - high exposure")
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
            with st.expander(f"Scenario Payoff - {trade['name'][:40]}", expanded=False):
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
                    st.caption("Payoff projection unavailable - see logs.")

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
                    (f"Backtest (Walk-Forward OOS) - {_wf_grade} · DSR {_dsr_pct}"
                     + (f" · PBO {_pbo_pct}" if _pbo_pct else ""))
                    if _has_result else
                    ("Backtest (Walk-Forward OOS) - MISSING DATA"
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
                                f'<b>MISSING LEGS - NOT GRADEABLE</b><br>'
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
                                    f'DUPLICATE DETECTED - trade-return series is identical to '
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
                                    f'LOW N - {_n_actual} trades (need ≥20). '
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
            f"⚡ Agent Debate - {trade['name'][:40]}"
            if _is_geo_trade
            else f"Agent Debate - {trade['name'][:40]}"
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
                                    st.caption("Debate unavailable - see logs.")

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
                    st.caption("Debate panel unavailable - see logs.")

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


# Deploy bar of Step 2's dsr_factor - single owner: trade_allocator.py.
from src.analysis.trade_allocator import DSR_DEPLOY_BAR as _DSR_DEPLOY_BAR


def _weight_earn_condition(alloc_detail: dict) -> str:
    """One-line, trade-specific statement of what must improve for a
    zero-weight eligible trade to earn allocation under Step 2's factors.
    Uses the EFFECTIVE (appetite-adjusted) deploy bar stamped on the trade."""
    from src.analysis.trade_allocator import MIN_TRADES_FOR_DSR
    n = int(alloc_detail.get("n_trades", 0))
    dsr = float(alloc_detail.get("dsr", 0.0))
    conv = float(alloc_detail.get("conviction", 0.0))
    bar = float(alloc_detail.get("deploy_bar", _DSR_DEPLOY_BAR))
    if n < MIN_TRADES_FOR_DSR:
        return f"needs ≥{MIN_TRADES_FOR_DSR} live signals (has {n})"
    if dsr <= bar:
        return f"DSR +{bar - dsr:.2f} to clear {bar:.2f} bar"
    if conv <= 0:
        return "needs Stage-3 conviction > 0"
    return "sizes on next reload"


def _portfolio_upside(book: list[dict], current_regime: int) -> dict | None:
    """Targeted upside for the CONSTRUCTED book (deployed trades only). Each
    trade's expected P&L and dispersion come from its OWN regime-conditional,
    direction-aware backtest - the mean and σ of realised returns per holding
    window - so the number is honest and per-name (a Short whose realised edge
    was negative shows negative, not a flat zero). Figures are returns on TOTAL
    book capital: a trade's weight is its share of the fully-invested equity
    sleeve. The bull/bear cone is a ±1.28σ (10th/90th-pctile) band on the BOOK
    return, using a disclosed average intra-book correlation so diversification
    is credited rather than assuming every leg moves together. Curated theses
    without a backtest sample fall back to the scenario projection. None when no
    trade is deployed."""
    import math
    deployed = [t for t in book if float(t.get("alloc_weight", 0.0)) > 0]
    if not deployed:
        return None
    from src.analysis.profit_projection import project_trade
    RHO = 0.35        # assumed average intra-book return correlation (disclosed)
    Z = 1.2816        # 10th / 90th percentile of the standard normal
    gross = sum(float(t["alloc_weight"]) for t in deployed)
    exp = be_w = 0.0
    ws_indep = 0.0    # Σ (w·σ)² - idiosyncratic-risk term
    ws_sum = 0.0      # Σ (w·σ) - common-factor term
    horizons: list[int] = []
    for t in deployed:
        w = float(t["alloc_weight"])
        det = t.get("alloc_detail") or {}
        hd = int(det.get("holding_days") or _parse_holding_days(t, default=63))
        kf = 21.0 / max(hd, 1)          # window → MONTHLY (return ×k, vol ×√k)
        if "avg_return" in det and int(det.get("n_trades", 0)) >= 1:
            mu = float(det.get("avg_return", 0.0)) * kf
            sd = max(float(det.get("ret_std", 0.0)), 0.0) * math.sqrt(kf)
        else:
            # curated thesis without a backtest sample - scenario fallback
            try:
                p = project_trade(t, holding_years=max(hd / 252.0, 0.02),
                                  current_regime=current_regime)
                _e = float(p.get("expected_pnl", 0.0))
                mu = _e * kf
                sd = (abs(float(p.get("best_case_pnl", 0.0)) - _e) / Z
                      if Z else 0.0) * math.sqrt(kf)
            except Exception:
                continue
        exp += w * mu
        ws_indep += (w * sd) ** 2
        ws_sum += w * sd
        be_w += w * (0.5 * (1.0 + math.erf(mu / (sd * math.sqrt(2.0))))
                     if sd > 1e-9 else (1.0 if mu > 0 else 0.0))
        horizons.append(max(1, round(hd / 21.0)))
    port_vol = math.sqrt(max((1.0 - RHO) * ws_indep + RHO * ws_sum ** 2, 0.0))
    return {
        "expected": exp,                       # per MONTH (horizon-normalised)
        "best": exp + Z * port_vol,
        "worst": exp - Z * port_vol,
        "annualized": exp * 12.0,
        "breakeven": (be_w / gross if gross > 1e-9 else 0.0),
        "months_lo": min(horizons) if horizons else 0,
        "months_hi": max(horizons) if horizons else 0,
        "gross": gross, "n": len(deployed),
    }


@st.cache_data(ttl=3600, show_spinner=False, max_entries=2)
def _load_stock_returns(start: str, end: str) -> pd.DataFrame:
    """Log-returns for the liquid single-stock universe (US S&P 500 + top India
    NSE + top China HK), columns keyed by DISPLAY NAME so they line up with the
    generated single-name trades' legs. Merged into the gate frame so those
    trades are eligible, backtestable and deployable - not just candidates."""
    from src.utils.artifact_cache import read_artifact, write_artifact
    _ac_key = f"stock_returns__{end}"          # end=today, so it refreshes daily
    _hit = read_artifact(_ac_key, max_age_s=3600)
    if _hit is not None:
        return _hit
    try:
        from src.analysis.trade_generator import all_stock_universe
        uni = all_stock_universe()
        if not uni:
            return pd.DataFrame()
        name_by_ticker = {tk: disp for disp, (tk, _s, _r) in uni.items()}
        import datetime as _dt
        from src.data.loader import _yf_download   # process-wide yfinance lock
        # Clamp to ~5y - backtest windows are ≤252d, so deeper history just
        # slows the 184-ticker fetch. Recent 5y is plenty for these signals.
        _floor = str(_dt.date.today() - _dt.timedelta(days=5 * 365))
        _s = _floor if start < _floor else start
        raw = _yf_download(list(name_by_ticker.keys()), start=_s, end=end,
                          auto_adjust=True, progress=False, threads=True)
        if raw.empty:
            return pd.DataFrame()
        close = raw["Close"] if "Close" in raw.columns else raw
        close = close.rename(columns=name_by_ticker)
        ret = np.log(close / close.shift(1)).dropna(how="all")
        if not ret.empty:
            write_artifact(_ac_key, ret)
        return ret
    except Exception:
        return pd.DataFrame()


def _live_generated_cached(regime: int, _start: str, _end: str) -> list:
    """Signal-ranked single-name candidates for THIS regime.

    NOT @st.cache_data on purpose: generate_signal_trades → score_all_assets
    spawns a ThreadPoolExecutor, and wrapping a threaded cached function inside
    another cached function DEADLOCKS on Streamlit's cache lock (0% CPU, loads
    forever). The heavy work (score_all_assets) carries its OWN cache, so reruns
    are still fast - only the ~0.6s trade construction repeats. Conflict-driven
    generation was dropped: it cost ~73s (LP-IRF Stage-3 on 9 theses) for at
    most one redundant gold position; the signal universe already maps stocks to
    their conflict/macro drivers, and the static library carries the conflict
    theses."""
    from src.analysis.trade_generator import generate_signal_trades
    return generate_signal_trades(regime=regime, max_trades=90)


def _attach_recent_news(feed: list[dict]) -> None:
    """Enrich each deployed trade card with recent REAL third-party coverage
    (last ~30d, source + date + link) so the desk report anchors every idea to
    live market context. Best-effort - silently skips names with no ticker or
    no recent news. Never fabricates: headlines come straight from yfinance."""
    try:
        from src.data.loader import fetch_ticker_news
        from src.analysis.trade_generator import all_stock_universe
        from src.data.config import COMMODITY_TICKERS, EQUITY_TICKERS
    except Exception:
        return
    _disp2tk = {disp: tk for disp, (tk, *_r) in all_stock_universe().items()}
    _disp2tk.update(COMMODITY_TICKERS)
    _disp2tk.update(EQUITY_TICKERS)
    # Commodities / indices → liquid ETF proxies that actually carry news.
    _NEWS_PROXY = {"Gold": "GLD", "WTI Crude Oil": "USO", "Brent Crude": "BNO",
                   "Silver": "SLV", "Copper": "CPER", "Natural Gas": "UNG",
                   "S&P 500": "SPY", "Nasdaq 100": "QQQ", "Gold Mining": "GDX",
                   "Corn": "CORN", "Wheat": "WEAT", "Soybeans": "SOYB"}
    import re as _re
    _STOP = {"the", "and", "for", "inc", "ltd", "plc", "corp", "llc", "co",
             "group", "holdings", "company", "international", "limited"}
    for _t in feed:
        try:
            _a = (_t.get("assets") or [None])[0]
            if not _a:
                continue
            _tk = _NEWS_PROXY.get(_a) or _disp2tk.get(_a)
            if not _tk:
                continue
            # Keywords for the relevance filter: distinctive name tokens + ticker
            # root, so only headlines actually about THIS name survive.
            _toks = [w.lower() for w in _re.split(r"[^A-Za-z0-9]+", _a)
                     if len(w) >= 3 and w.lower() not in _STOP]
            _tkroot = _re.split(r"[.\-]", _tk)[0].lower()
            _kw = tuple(dict.fromkeys(_toks + [_tkroot]))
            _t["recent_news"] = fetch_ticker_news(_tk, max_items=3,
                                                  max_age_days=35, keywords=_kw)
        except Exception:
            continue


@st.cache_data(show_spinner=False, ttl=7 * 86400, max_entries=400)
def _fetch_company_logo(ticker: str) -> "bytes | None":
    """Square company-logo PNG for a ticker via Financial Modeling Prep (no API
    key). Disk-cached, since logos rarely change. A genuine 404 is negatively
    cached (b'') so a logo-less name isn't refetched every report; transient
    network failures are NOT cached, so they retry next time. Returns None on any
    failure so the desk report silently degrades to no-logo."""
    if not ticker:
        return None
    from src.utils.artifact_cache import read_artifact, write_artifact
    _key = f"logo_{ticker.upper()}"
    _hit = read_artifact(_key, max_age_s=30 * 86400)
    if _hit is not None:
        return _hit or None                       # b'' sentinel = known-missing
    _png, _known_missing = None, False
    try:
        import requests
        _r = requests.get(
            f"https://financialmodelingprep.com/image-stock/{ticker.upper()}.png",
            timeout=5)
        _c = _r.content or b""
        if (_r.status_code == 200 and _c[:8] == b"\x89PNG\r\n\x1a\n"
                and len(_c) > 400):
            _png = _c
        elif _r.status_code == 404:
            _known_missing = True                 # really has no logo → cache it
    except Exception:
        _png = None                               # transient → do not cache, retry
    try:
        if _png is not None:
            write_artifact(_key, _png)
        elif _known_missing:
            write_artifact(_key, b"")
    except Exception:
        pass
    return _png


def _attach_logos(feed: list[dict]) -> None:
    """Attach a small company-logo PNG (bytes, under key 'logo_png') to each
    deployed trade whose primary leg is a single-name company, for the desk
    report. Only real companies from the single-name universe get a mark - 
    commodities, ETFs and indices are intentionally left logo-less (a company
    logo there would misrepresent the exposure). Best-effort and never raises."""
    try:
        from src.analysis.trade_generator import all_stock_universe
    except Exception:
        return
    _disp2tk = {disp: tk for disp, (tk, *_r) in all_stock_universe().items()}
    for _t in feed:
        try:
            _a = (_t.get("assets") or [None])[0]
            _tk = _disp2tk.get(_a) if _a else None
            if not _tk:
                continue
            # FMP keys most names by base symbol; try full ticker then stripped.
            _logo = _fetch_company_logo(_tk) or (
                _fetch_company_logo(_tk.split(".")[0]) if "." in _tk else None)
            if _logo:
                _t["logo_png"] = _logo
        except Exception:
            continue


# ═════════════════════════════════════════════════════════════════════════════
# Book factor & alpha decomposition
# Answers the buy-side question the pipeline validation does NOT: is the deployed
# book alpha or beta? Regresses the weight-normalised book return on a market
# factor plus market-orthogonalised thematic factors (HAC/Newey-West errors), so
# market beta, sector tilts and genuine idiosyncratic return are separated - and
# reports an effective-number-of-bets concentration read (are 8 ideas really 8?).
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600, max_entries=8)
def _load_etf_returns(ticker: str, start: str, end: str) -> "pd.Series | None":
    """Daily log-returns for one ETF factor proxy via yfinance, disk-cached.
    None on failure so the factor panel degrades to whatever loads."""
    from src.utils.artifact_cache import read_artifact, write_artifact
    _key = f"etf_ret_{ticker}_{end}"
    _hit = read_artifact(_key, max_age_s=3600)
    if _hit is not None:
        return _hit if len(_hit) else None
    try:
        import datetime as _dt
        from src.data.loader import _yf_download
        _floor = str(_dt.date.today() - _dt.timedelta(days=6 * 365))
        _s = _floor if start < _floor else start
        raw = _yf_download([ticker], start=_s, end=end, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return None
        close = raw["Close"] if "Close" in raw.columns else raw
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        ret = np.log(close / close.shift(1)).dropna()
        ret.name = ticker
        if len(ret):
            write_artifact(_key, ret)
        return ret if len(ret) else None
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3600, max_entries=2)
def _load_factor_panel(start: str, end: str) -> pd.DataFrame:
    """Daily log-return factor panel for book attribution: MKT (S&P 500), Defense
    (ITA), Energy (WTI), Gold, Rates (TLT), USD (DXY). MKT is always the first
    column. Returns whatever subset loads (empty frame if MKT is unavailable)."""
    cols: dict = {}
    try:
        from src.data.loader import load_returns as _lr
        eq, cmd = _lr()
        if "S&P 500" in eq.columns:        cols["MKT·S&P 500"] = eq["S&P 500"]
        if "WTI Crude Oil" in cmd.columns: cols["Energy·WTI"]  = cmd["WTI Crude Oil"]
        if "Gold" in cmd.columns:          cols["Gold"]        = cmd["Gold"]
    except Exception:
        pass
    try:
        from src.data.loader import (load_fixed_income_returns as _fi,
                                      load_fx_returns as _fx)
        fi = _fi()
        if "US 20Y+ Treasury (TLT)" in fi.columns:
            cols["Rates·TLT"] = fi["US 20Y+ Treasury (TLT)"]
        fx = _fx()
        if "DXY (Dollar Index)" in fx.columns:
            cols["USD·DXY"] = fx["DXY (Dollar Index)"]
    except Exception:
        pass
    _ita = _load_etf_returns("ITA", start, end)
    if _ita is not None:
        cols["Defense·ITA"] = _ita
    if "MKT·S&P 500" not in cols or len(cols) < 2:
        return pd.DataFrame()
    F = pd.concat(cols, axis=1)
    _order = ["MKT·S&P 500"] + [c for c in F.columns if c != "MKT·S&P 500"]
    return F[_order].dropna(how="all")


def _deployed_book_return(book: list, all_r_gate: pd.DataFrame):
    """(weight-normalised daily book return Series, per-position return frame L),
    or (None, None) if the deployed book is too thin. Shared by the factor
    attribution and the factor-neutral skill test so both use one book series."""
    try:
        from src.analysis.trade_allocator import trade_leg_series
    except Exception:
        return None, None
    deployed = [t for t in book if float(t.get("alloc_weight", 0.0)) > 0]
    if len(deployed) < 2:
        return None, None
    gross = sum(float(t["alloc_weight"]) for t in deployed) or 1.0
    legs: dict = {}
    wts: dict = {}
    for t in deployed:
        s = trade_leg_series(all_r_gate, t.get("assets", []), t.get("direction", []))
        if s is not None and len(s) > 120:
            nm = t.get("name", f"pos{len(legs)}")
            legs[nm] = s
            wts[nm] = float(t["alloc_weight"]) / gross
    if len(legs) < 2:
        return None, None
    L = pd.concat(legs, axis=1).dropna()
    if len(L) < 250:
        return None, None
    w = np.array([wts[c] for c in L.columns], dtype=float)
    w = w / w.sum()
    return pd.Series(L.values @ w, index=L.index, name="book"), L


def _compute_book_factor_decomp(book: list, all_r_gate: pd.DataFrame,
                                start: str, end: str) -> "dict | None":
    """Factor & alpha attribution of the deployed book. Regresses the weight-
    normalised book return on a market factor plus market-orthogonalised thematic
    factors (HAC/Newey-West), separating market beta, sector tilts and genuine
    idiosyncratic return. Returns a dict of results (or None if insufficient data)
    so both the on-screen panel and the PDF desk report render from one source."""
    try:
        import statsmodels.api as sm
    except Exception:
        return None

    book_r, L = _deployed_book_return(book, all_r_gate)
    if book_r is None:
        return None

    F = _load_factor_panel(start, end)
    if F.empty:
        return None
    df = pd.concat([book_r, F], axis=1).dropna()
    if len(df) < 250:
        return None
    y = df["book"]
    mkt = F.columns[0]
    sectors = [c for c in F.columns if c != mkt]
    _HAC = dict(cov_type="HAC", cov_kwds={"maxlags": 5})

    # CAPM / market model
    cm = sm.OLS(y, sm.add_constant(df[[mkt]])).fit(**_HAC)
    beta_mkt   = float(cm.params[mkt])
    alpha_capm = float(cm.params["const"]) * 252 * 100        # %/yr
    alpha_t    = float(cm.tvalues["const"])
    r2_mkt     = float(cm.rsquared)
    # benchmark-relative (vs S&P)
    active = y - df[mkt]
    te = float(active.std() * np.sqrt(252) * 100)             # %/yr
    ir = float((active.mean() * 252 * 100) / te) if te > 0 else float("nan")

    # Multi-factor with market-orthogonalised sector factors
    orth = {mkt: df[mkt]}
    for c in sectors:
        orth[c] = sm.OLS(df[c], sm.add_constant(df[[mkt]])).fit().resid
    XF = pd.concat([orth[mkt].rename(mkt)] +
                   [orth[c].rename(c) for c in sectors], axis=1)
    fm = sm.OLS(y, sm.add_constant(XF)).fit(**_HAC)
    r2_full    = float(fm.rsquared)
    alpha_mf   = float(fm.params["const"]) * 252 * 100
    alpha_mf_t = float(fm.tvalues["const"])
    load = {c: (float(fm.params[c]), float(fm.tvalues[c])) for c in XF.columns}

    # Effective number of bets from the correlation spectrum
    eig = np.linalg.eigvalsh(L.corr().values)
    eig = eig[eig > 1e-9]
    enb = float((eig.sum() ** 2) / np.square(eig).sum()) if eig.size else float(L.shape[1])

    # Variance ladder
    sect_share = max(0.0, r2_full - r2_mkt) * 100
    mkt_share  = r2_mkt * 100
    idio_share = max(0.0, 1.0 - r2_full) * 100

    # Verdict
    sig = abs(alpha_mf_t) >= 2.0
    if sig and alpha_mf > 0:
        verdict = "ALPHA PRESENT"
    elif r2_full >= 0.60:
        verdict = "BETA BOOK"
    elif sect_share >= 15:
        verdict = "SECTOR-TILT BOOK"
    else:
        verdict = "INCONCLUSIVE"

    # Dominant sector tilts (by |t|), significant only
    _sig_sectors = sorted(
        [(c, load[c][0], load[c][1]) for c in sectors if abs(load[c][1]) >= 2.0],
        key=lambda x: -abs(x[2]))
    lead_txt = ", ".join(
        f'{c.split("·")[0]} (b{v:+.2f}, t{tt:.1f})' for c, v, tt in _sig_sectors[:2]
    ) or "no sector loading clears t>=2"

    return {
        "n_positions": int(L.shape[1]), "obs": int(len(df)),
        "n_factors": len(sectors) + 1, "mkt_name": mkt,
        "beta_mkt": beta_mkt, "r2_mkt": r2_mkt, "r2_full": r2_full,
        "alpha_capm": alpha_capm, "alpha_capm_t": alpha_t,
        "alpha_mf": alpha_mf, "alpha_mf_t": alpha_mf_t, "sig": sig,
        "te": te, "ir": ir, "enb": enb,
        "loadings": [(c, load[c][0], load[c][1]) for c in [mkt] + sectors],
        "mkt_share": mkt_share, "sect_share": sect_share, "idio_share": idio_share,
        "verdict": verdict, "lead_txt": lead_txt,
    }


def _render_book_factor_decomp(book: list, all_r_gate: pd.DataFrame,
                               start: str, end: str) -> None:
    """On-screen factor & alpha attribution panel for the deployed book."""
    d = _compute_book_factor_decomp(book, all_r_gate, start, end)
    if not d:
        return
    beta_mkt, r2_mkt, r2_full = d["beta_mkt"], d["r2_mkt"], d["r2_full"]
    alpha_mf, alpha_mf_t, _sig = d["alpha_mf"], d["alpha_mf_t"], d["sig"]
    te, ir, enb = d["te"], d["ir"], d["enb"]
    mkt, loadings = d["mkt_name"], d["loadings"]
    mkt_share, sect_share, idio_share = d["mkt_share"], d["sect_share"], d["idio_share"]
    verdict, _lead_txt = d["verdict"], d["lead_txt"]
    vcol = {"ALPHA PRESENT": "#27ae60", "BETA BOOK": "#c0392b",
            "SECTOR-TILT BOOK": "#e67e22"}.get(verdict, "#8890a1")
    load = {c: (b, tt) for c, b, tt in loadings}
    sectors = [c for c, _, _ in loadings if c != mkt]
    n_pos, obs, n_fac = d["n_positions"], d["obs"], d["n_factors"]

    _M = "font-family:'JetBrains Mono',monospace;"
    _lbl = f"{_M}font-size:.5rem;letter-spacing:.1em;color:#8890a1"

    def _stat(lbl, val, sub, col="#e8e9ed"):
        return (f'<div style="flex:1;padding:.45rem .7rem;border-right:1px solid #1e1e1e">'
                f'<div style="{_lbl}">{lbl}</div>'
                f'<div style="{_M}font-size:1.05rem;font-weight:700;color:{col};'
                f'margin:1px 0">{val}</div>'
                f'<div style="{_M}font-size:.48rem;color:#555960">{sub}</div></div>')

    _acol = "#27ae60" if (alpha_mf > 0 and _sig) else ("#e8e9ed" if alpha_mf >= 0 else "#c0392b")
    stats = (
        f'<div style="display:flex;border-bottom:1px solid #1e1e1e">'
        + _stat("MARKET β", f"{beta_mkt:.2f}",
                f"vs S&P · R²&nbsp;{r2_mkt*100:.0f}%")
        + _stat("JENSEN α", f"{alpha_mf:+.1f}%<span style='font-size:.5rem'>/yr</span>",
                f"t&nbsp;{alpha_mf_t:.1f} · {'sig' if _sig else 'not sig'}", _acol)
        + _stat("INFO RATIO", f"{ir:.2f}",
                f"TE&nbsp;{te:.1f}%/yr vs S&P")
        + _stat("EFF. BETS", f"{enb:.1f}",
                f"of&nbsp;{n_pos} positions")
        + f'</div>')

    # factor loading bars (market + sectors), scaled to the largest |β|
    _all_f = [mkt] + sectors
    _bmax = max((abs(load[c][0]) for c in _all_f), default=1.0) or 1.0
    rows = ""
    for c in _all_f:
        b, tt = load[c]
        _w = abs(b) / _bmax * 100
        _bc = "#3a6ea5" if c == mkt else ("#27ae60" if b >= 0 else "#c0392b")
        _sg = "" if abs(tt) >= 2 else "opacity:.45;"
        rows += (
            f'<div style="display:flex;align-items:center;gap:8px;padding:1.5px 0;{_sg}">'
            f'<span style="{_M}font-size:.54rem;color:#e8e9ed;min-width:104px">{c}</span>'
            f'<div style="flex:1;height:9px;background:#141414;position:relative">'
            f'<div style="width:{_w:.0f}%;height:9px;background:{_bc}"></div></div>'
            f'<span style="{_M}font-size:.54rem;color:#e8e9ed;min-width:52px;'
            f'text-align:right">β&nbsp;{b:+.2f}</span>'
            f'<span style="{_M}font-size:.5rem;color:#8890a1;min-width:44px;'
            f'text-align:right">t&nbsp;{tt:+.1f}</span></div>')

    # variance ladder (stacked bar)
    ladder = (
        f'<div style="display:flex;height:16px;border:1px solid #1e1e1e;margin:.15rem 0 .3rem">'
        f'<div style="width:{mkt_share:.0f}%;background:#3a6ea5" title="market"></div>'
        f'<div style="width:{sect_share:.0f}%;background:#e67e22" title="sector tilts"></div>'
        f'<div style="width:{idio_share:.0f}%;background:#2b2b2b" title="idiosyncratic"></div>'
        f'</div>'
        f'<div style="{_M}font-size:.5rem;color:#8890a1;display:flex;gap:14px">'
        f'<span><span style="color:#3a6ea5">■</span> market {mkt_share:.0f}%</span>'
        f'<span><span style="color:#e67e22">■</span> sector tilts {sect_share:.0f}%</span>'
        f'<span><span style="color:#8890a1">■</span> idiosyncratic {idio_share:.0f}%</span>'
        f'</div>')

    _sig_word = "significant" if _sig else "not statistically significant"
    _skill_txt = ("evidence of selection skill beyond the factor tilts."
                  if (_sig and alpha_mf > 0) else
                  "so the book is a factor tilt, not demonstrated stock-selection alpha.")
    read = (
        f'This {n_pos}-position book carries <b style="color:#e8e9ed">'
        f'{beta_mkt:.2f} market beta</b>; <b style="color:#e8e9ed">'
        f'{r2_full*100:.0f}%</b> of its daily variance is market + sector beta '
        f'(dominated by {_lead_txt}). Jensen alpha is <b style="color:{_acol}">'
        f'{alpha_mf:+.1f}%/yr</b> and is <b>{_sig_word}</b> '
        f'(t&nbsp;{alpha_mf_t:.1f}) - {_skill_txt}'
        f' The {n_pos} positions span ~<b style="color:#e8e9ed">{enb:.1f}</b> '
        f'independent bets.')

    st.markdown(
        f'<div style="border:1px solid #1e1e1e;background:#0a0a0a;margin:.2rem 0 .8rem">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
        f'padding:.4rem .8rem;border-bottom:1px solid #1e1e1e">'
        f'<span style="{_M}font-size:.6rem;font-weight:700;letter-spacing:.14em;'
        f'color:#e8e9ed">BOOK FACTOR &amp; ALPHA DECOMPOSITION</span>'
        f'<span style="{_M}font-size:.5rem;color:#8890a1">'
        f'HAC/Newey-West · {obs} obs · alpha vs {n_fac} factors '
        f'<span style="background:{vcol};color:#000;font-weight:700;padding:1px 7px;'
        f'margin-left:6px;letter-spacing:.08em">{verdict}</span></span></div>'
        f'{stats}'
        f'<div style="display:flex;gap:16px;padding:.55rem .8rem;flex-wrap:wrap">'
        f'<div style="flex:1.3;min-width:280px">'
        f'<div style="{_lbl};margin-bottom:3px">FACTOR LOADINGS · market-orthogonalised · '
        f'faded = t&lt;2</div>{rows}</div>'
        f'<div style="flex:1;min-width:210px">'
        f'<div style="{_lbl};margin-bottom:3px">VARIANCE EXPLAINED</div>{ladder}'
        f'<div style="{_M}font-size:.56rem;color:#c9ccd4;line-height:1.5;'
        f'margin-top:.5rem">{read}</div></div>'
        f'</div>'
        f'<div style="padding:.3rem .8rem;border-top:1px solid #1e1e1e;{_M}'
        f'font-size:.48rem;color:#555960">Ex-post attribution of the current book '
        f'over the sample window · β from HAC-robust OLS · Jensen α = multi-factor '
        f'intercept (annualised) · eff. bets = participation ratio of the position '
        f'correlation spectrum · not a forward forecast</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False, ttl=7 * 86400, max_entries=2)
def _load_ff_factors(start: str, end: str) -> pd.DataFrame:
    """Daily Fama-French 5 factors + Momentum (as decimals) from the Ken French
    data library, disk-cached. Columns: MktRF, SMB, HML, RMW, CMA, RF, MOM. These
    are the academic risk factors the thematic panel omits (size, value, quality,
    investment, momentum). Empty frame on failure so the skill test degrades."""
    from src.utils.artifact_cache import read_artifact, write_artifact
    _key = f"ff5_mom_{end}"
    _hit = read_artifact(_key, max_age_s=7 * 86400)
    if _hit is not None:
        return _hit if len(_hit) else pd.DataFrame()
    import io as _io, zipfile as _zip
    _BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

    def _pull(name, cols):
        import requests
        r = requests.get(_BASE + name, timeout=25)
        r.raise_for_status()
        z = _zip.ZipFile(_io.BytesIO(r.content))
        raw = z.read(z.namelist()[0]).decode("latin-1").splitlines()
        rows, started = [], False
        for ln in raw:
            s = ln.strip()
            if len(s) >= 8 and s[:8].isdigit():
                started = True
                rows.append(s)
            elif started:
                break
        df = pd.read_csv(_io.StringIO("\n".join(rows)), header=None).dropna(axis=1, how="all")
        df.columns = ["date"] + cols[:df.shape[1] - 1]
        df["date"] = pd.to_datetime(df["date"].astype(int).astype(str), format="%Y%m%d")
        return df.set_index("date")[cols[:df.shape[1] - 1]] / 100.0

    try:
        ff5 = _pull("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
                    ["MktRF", "SMB", "HML", "RMW", "CMA", "RF"])
        mom = _pull("F-F_Momentum_Factor_daily_CSV.zip", ["MOM"])
        F = ff5.join(mom, how="inner").dropna()
    except Exception:
        return pd.DataFrame()
    if not F.empty:
        try:
            write_artifact(_key, F)
        except Exception:
            pass
    return F


def _compute_factor_neutral_skill(book: list, all_r_gate: pd.DataFrame,
                                  start: str, end: str,
                                  n_theses: int = 9) -> "dict | None":
    """Does the book's edge survive stripping the known risk factors? Regresses
    the excess book return on FF5 + Momentum (HAC), takes the residual, and asks
    whether the factor-neutral Sharpe clears a deflated-Sharpe bar. This is the
    real test of selection skill; the thematic decomposition only shows tilts."""
    try:
        import statsmodels.api as sm
    except Exception:
        return None
    book_r, L = _deployed_book_return(book, all_r_gate)
    if book_r is None:
        return None
    F = _load_ff_factors(start, end)
    if F.empty or "RF" not in F.columns:
        return None
    df = pd.concat([book_r, F], axis=1).dropna()
    if len(df) < 250:
        return None
    facs = [c for c in ["MktRF", "SMB", "HML", "RMW", "CMA", "MOM"] if c in df.columns]
    exc = df["book"] - df["RF"]                       # excess book return
    m = sm.OLS(exc, sm.add_constant(df[facs])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    resid = m.resid
    _ann = np.sqrt(252)
    raw_sharpe = float(exc.mean() / exc.std() * _ann) if exc.std() > 0 else 0.0
    res_sharpe = float(resid.mean() / resid.std() * _ann) if resid.std() > 0 else 0.0
    alpha = float(m.params["const"]) * 252 * 100      # %/yr
    alpha_t = float(m.tvalues["const"])
    r2 = float(m.rsquared)

    from src.analysis.backtest import deflated_sharpe_probability
    sr_daily = float(resid.mean() / resid.std()) if resid.std() > 0 else 0.0
    try:
        dsr, sr_star = deflated_sharpe_probability(
            sr_daily, len(resid), float(pd.Series(resid).skew()),
            float(pd.Series(resid).kurt()), n_strategies=max(1, int(n_theses)))
    except Exception:
        dsr, sr_star = float("nan"), float("nan")

    _LBL = {"MktRF": "Market", "SMB": "Size (SMB)", "HML": "Value (HML)",
            "RMW": "Profitability (RMW)", "CMA": "Investment (CMA)", "MOM": "Momentum"}
    loadings = [(_LBL.get(c, c), float(m.params[c]), float(m.tvalues[c])) for c in facs]

    if dsr >= 0.95 and res_sharpe > 0:
        verdict = "SKILL SURVIVES"
    elif dsr >= 0.75:
        verdict = "MARGINAL"
    else:
        verdict = "NO SKILL AFTER FACTORS"
    _sig = [l for l in loadings if abs(l[2]) >= 2]
    lead = max(_sig, key=lambda x: abs(x[2]))[0] if _sig else "no factor clears t>=2"
    retained = (res_sharpe / raw_sharpe * 100) if raw_sharpe > 0.05 else 0.0

    return {"obs": len(df), "raw_sharpe": raw_sharpe, "res_sharpe": res_sharpe,
            "alpha": alpha, "alpha_t": alpha_t, "r2": r2, "dsr": dsr,
            "sr_star": sr_star, "n_theses": int(n_theses), "loadings": loadings,
            "verdict": verdict, "retained": retained, "lead_factor": lead}


def _render_factor_neutral_skill(book: list, all_r_gate: pd.DataFrame,
                                 start: str, end: str, n_theses: int = 9) -> None:
    """On-screen factor-neutral skill test panel (FF5 + Momentum)."""
    d = _compute_factor_neutral_skill(book, all_r_gate, start, end, n_theses)
    if not d:
        return
    _M = "font-family:'JetBrains Mono',monospace;"
    _lbl = f"{_M}font-size:.5rem;letter-spacing:.1em;color:#8890a1"
    vcol = {"SKILL SURVIVES": "#27ae60", "MARGINAL": "#e67e22"}.get(d["verdict"], "#c0392b")
    dsr = d["dsr"]
    _dsr_txt = f"{dsr*100:.0f}%" if dsr == dsr else "n/a"      # NaN guard
    _rs_col = "#27ae60" if d["res_sharpe"] > 0.2 else ("#e8e9ed" if d["res_sharpe"] >= 0 else "#c0392b")
    _a_col = "#27ae60" if (d["alpha"] > 0 and abs(d["alpha_t"]) >= 2) else ("#e8e9ed" if d["alpha"] >= 0 else "#c0392b")

    def _stat(lbl, val, sub, col="#e8e9ed"):
        return (f'<div style="flex:1;padding:.45rem .7rem;border-right:1px solid #1e1e1e">'
                f'<div style="{_lbl}">{lbl}</div>'
                f'<div style="{_M}font-size:1.05rem;font-weight:700;color:{col};'
                f'margin:1px 0">{val}</div>'
                f'<div style="{_M}font-size:.48rem;color:#555960">{sub}</div></div>')

    stats = (
        f'<div style="display:flex;border-bottom:1px solid #1e1e1e">'
        + _stat("RAW SHARPE", f'{d["raw_sharpe"]:.2f}', "book excess, annualised")
        + _stat("FACTOR-NEUTRAL SHARPE", f'{d["res_sharpe"]:.2f}',
                f'{d["retained"]:.0f}% of raw survives', _rs_col)
        + _stat("FF5+MOM ALPHA", f'{d["alpha"]:+.1f}%<span style="font-size:.5rem">/yr</span>',
                f't {d["alpha_t"]:.1f} · R² {d["r2"]*100:.0f}%', _a_col)
        + _stat("DEFLATED SHARPE", _dsr_txt,
                f'P(skill real) · {d["n_theses"]} theses', vcol)
        + f'</div>')

    _bmax = max((abs(b) for _, b, _ in d["loadings"]), default=1.0) or 1.0
    rows = ""
    for name, b, tt in d["loadings"]:
        _w = abs(b) / _bmax * 100
        _bc = "#27ae60" if b >= 0 else "#c0392b"
        _sg = "" if abs(tt) >= 2 else "opacity:.45;"
        rows += (
            f'<div style="display:flex;align-items:center;gap:8px;padding:1.5px 0;{_sg}">'
            f'<span style="{_M}font-size:.54rem;color:#e8e9ed;min-width:118px">{name}</span>'
            f'<div style="flex:1;height:9px;background:#141414"><div style="width:{_w:.0f}%;'
            f'height:9px;background:{_bc}"></div></div>'
            f'<span style="{_M}font-size:.54rem;color:#e8e9ed;min-width:52px;'
            f'text-align:right">β&nbsp;{b:+.2f}</span>'
            f'<span style="{_M}font-size:.5rem;color:#8890a1;min-width:44px;'
            f'text-align:right">t&nbsp;{tt:+.1f}</span></div>')

    _skill = ("selection skill that survives every known factor."
              if d["verdict"] == "SKILL SURVIVES" else
              "no demonstrated selection skill once the known factors are removed.")
    read = (
        f'The book\'s raw excess Sharpe of <b style="color:#e8e9ed">{d["raw_sharpe"]:.2f}</b> '
        f'is mostly factor exposure. Neutralising the FF5 + Momentum factors leaves a '
        f'<b style="color:{_rs_col}">{d["res_sharpe"]:.2f}</b> residual Sharpe '
        f'({d["retained"]:.0f}% of the raw). The deflated-Sharpe probability of real skill, '
        f'after correcting for {d["n_theses"]} theses searched, is '
        f'<b style="color:{vcol}">{_dsr_txt}</b>. The dominant hidden exposure the thematic '
        f'panel missed is <b style="color:#e8e9ed">{d["lead_factor"]}</b>. In short: {_skill}')

    st.markdown(
        f'<div style="border:1px solid #1e1e1e;background:#0a0a0a;margin:.2rem 0 .8rem">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
        f'padding:.4rem .8rem;border-bottom:1px solid #1e1e1e">'
        f'<span style="{_M}font-size:.6rem;font-weight:700;letter-spacing:.14em;'
        f'color:#e8e9ed">FACTOR-NEUTRAL SKILL TEST</span>'
        f'<span style="{_M}font-size:.5rem;color:#8890a1">'
        f'FF5 + Momentum · {d["obs"]} obs · does edge survive the known factors?'
        f'<span style="background:{vcol};color:#000;font-weight:700;padding:1px 7px;'
        f'margin-left:6px;letter-spacing:.08em">{d["verdict"]}</span></span></div>'
        f'{stats}'
        f'<div style="display:flex;gap:16px;padding:.55rem .8rem;flex-wrap:wrap">'
        f'<div style="flex:1.2;min-width:300px">'
        f'<div style="{_lbl};margin-bottom:3px">FACTOR LOADINGS · Fama-French 5 + Momentum · '
        f'faded = t&lt;2</div>{rows}</div>'
        f'<div style="flex:1;min-width:230px">'
        f'<div style="{_M}font-size:.56rem;color:#c9ccd4;line-height:1.55">{read}</div></div>'
        f'</div>'
        f'<div style="padding:.3rem .8rem;border-top:1px solid #1e1e1e;{_M}'
        f'font-size:.48rem;color:#555960">Excess book return regressed on the Ken French '
        f'FF5 + Momentum daily factors (HAC errors). Residual Sharpe is the factor-neutral '
        f'information ratio; deflated-Sharpe probability follows Bailey and Lopez de Prado, '
        f'deflated by the thesis count. Ex-post, not a forward forecast.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _compute_rolling_exposures(book: list, all_r_gate: pd.DataFrame,
                               start: str, end: str, window: int = 126) -> "dict | None":
    """Rolling univariate factor betas of the deployed book over time, so a static
    single-window attribution is not mistaken for a stable exposure. Uses pandas
    rolling cov/var (fast, no per-window regression loop). Returns dict or None."""
    book_r, L = _deployed_book_return(book, all_r_gate)
    if book_r is None:
        return None
    F = _load_factor_panel(start, end)
    if F.empty:
        return None
    df = pd.concat([book_r, F], axis=1).dropna()
    if len(df) < window + 60:
        return None
    facs = list(F.columns)                    # market first, then thematic
    B = pd.DataFrame(
        {f: df["book"].rolling(window).cov(df[f]) / df[f].rolling(window).var()
         for f in facs}
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(B) < 20:
        return None
    summ = {f: {"current": float(B[f].iloc[-1]), "min": float(B[f].min()),
                "max": float(B[f].max()), "range": float(B[f].max() - B[f].min())}
            for f in facs}
    mkt = facs[0]
    dom = max((f for f in facs if f != mkt), key=lambda f: abs(summ[f]["current"]))
    _drift = max(summ[mkt]["range"], summ[dom]["range"])
    stability = "DRIFTING" if _drift >= 0.5 else "SHIFTING" if _drift >= 0.3 else "STABLE"
    return {"dates": list(B.index), "betas": {f: B[f].values for f in facs},
            "facs": facs, "summ": summ, "window": window, "mkt": mkt, "dom": dom,
            "stability": stability, "n": len(B)}


def _render_rolling_exposures(book: list, all_r_gate: pd.DataFrame,
                              start: str, end: str) -> None:
    """On-screen rolling factor-exposure panel: are the book's betas stable?"""
    d = _compute_rolling_exposures(book, all_r_gate, start, end)
    if not d:
        return
    import plotly.graph_objects as go
    _M = "font-family:'JetBrains Mono',monospace;"
    vcol = {"STABLE": "#27ae60", "SHIFTING": "#e67e22"}.get(d["stability"], "#c0392b")
    facs, mkt, summ = d["facs"], d["mkt"], d["summ"]

    st.markdown(
        f'<div style="border:1px solid #1e1e1e;border-bottom:none;background:#0a0a0a;'
        f'display:flex;justify-content:space-between;align-items:baseline;'
        f'padding:.4rem .8rem;margin-top:.2rem">'
        f'<span style="{_M}font-size:.6rem;font-weight:700;letter-spacing:.14em;'
        f'color:#e8e9ed">ROLLING FACTOR EXPOSURE · {d["window"]}D</span>'
        f'<span style="{_M}font-size:.5rem;color:#8890a1">'
        f'{d["n"]} windows · is the factor character stable?'
        f'<span style="background:{vcol};color:#000;font-weight:700;padding:1px 7px;'
        f'margin-left:6px;letter-spacing:.08em">{d["stability"]}</span></span></div>',
        unsafe_allow_html=True,
    )

    # chart: market + top 3 thematic factors by |current beta|
    _plot = [mkt] + sorted([f for f in facs if f != mkt],
                           key=lambda f: -abs(summ[f]["current"]))[:3]
    _cyc = ["#3a9bdc", "#27ae60", "#c0392b"]
    fig = go.Figure()
    for i, f in enumerate(_plot):
        col = "#CFB991" if f == mkt else _cyc[(i - 1) % len(_cyc)]
        fig.add_trace(go.Scatter(
            x=d["dates"], y=d["betas"][f], name=f.replace("·", " "), mode="lines",
            line=dict(width=2.4 if f == mkt else 1.7, color=col),
            hovertemplate="%{x|%b %Y}<br>β %{y:.2f}<extra>" + f.replace("·", " ") + "</extra>"))
    fig.add_hline(y=0, line=dict(color="#333", width=1))
    fig.update_layout(
        template="plotly_dark", height=290, plot_bgcolor="#0a0a0a", paper_bgcolor="#0a0a0a",
        margin=dict(l=8, r=8, t=8, b=6),
        legend=dict(orientation="h", y=1.14, x=0, font=dict(size=10, color="#c8c8c8"),
                    bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(title=dict(text="rolling β", font=dict(size=10, color="#8890a1")),
                   gridcolor="#161616", zeroline=False),
        xaxis=dict(gridcolor="#161616"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # summary rows + read
    def _drift_col(rng):
        return "#c0392b" if rng >= 0.5 else "#e67e22" if rng >= 0.3 else "#27ae60"
    _rows = ""
    for f in facs:
        s = summ[f]
        _rows += (
            f'<div style="display:flex;align-items:center;gap:8px;padding:1.5px 0">'
            f'<span style="{_M}font-size:.54rem;color:#e8e9ed;min-width:108px">{f.replace("·"," ")}</span>'
            f'<span style="{_M}font-size:.54rem;color:#e8e9ed;min-width:60px;text-align:right">'
            f'β&nbsp;{s["current"]:+.2f}</span>'
            f'<span style="{_M}font-size:.5rem;color:#8890a1;min-width:118px;text-align:right">'
            f'range&nbsp;[{s["min"]:+.2f}, {s["max"]:+.2f}]</span>'
            f'<span style="{_M}font-size:.5rem;color:{_drift_col(s["range"])};min-width:64px;'
            f'text-align:right">span&nbsp;{s["range"]:.2f}</span></div>')
    _dm = summ[d["dom"]]
    _mk = summ[mkt]
    read = (
        f'Rolling {d["window"]}-day betas show the book\'s factor character is '
        f'<b style="color:{vcol}">{d["stability"].lower()}</b>. Market beta ranged '
        f'<b style="color:#e8e9ed">{_mk["min"]:+.2f}</b> to '
        f'<b style="color:#e8e9ed">{_mk["max"]:+.2f}</b>; the '
        f'{d["dom"].replace("·"," ")} loading moved from '
        f'<b style="color:#e8e9ed">{_dm["min"]:+.2f}</b> to '
        f'<b style="color:#e8e9ed">{_dm["max"]:+.2f}</b>. A single-window attribution '
        f'averages these, so the <b>current</b> exposures are what matter for hedging '
        f'today, not the full-sample number.')
    st.markdown(
        f'<div style="border:1px solid #1e1e1e;border-top:none;background:#0a0a0a;'
        f'padding:.5rem .8rem .6rem">'
        f'<div style="display:flex;gap:16px;flex-wrap:wrap">'
        f'<div style="flex:1;min-width:280px">{_rows}</div>'
        f'<div style="flex:1;min-width:230px;{_M}font-size:.56rem;color:#c9ccd4;'
        f'line-height:1.55">{read}</div></div>'
        f'<div style="border-top:1px solid #1e1e1e;margin-top:.4rem;padding-top:.3rem;'
        f'{_M}font-size:.48rem;color:#555960">Univariate rolling beta = '
        f'cov(book, factor) / var(factor) over a trailing {d["window"]}-day window. '
        f'Gross exposure over time, not a variance decomposition.</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def _load_book_adv(tickers: tuple, end: str) -> dict:
    """Median trailing-60d dollar ADV (volume x close) per ticker via yfinance,
    disk-cached. Best-effort; tickers that fail are simply absent from the dict."""
    if not tickers:
        return {}
    from src.utils.artifact_cache import read_artifact, write_artifact
    _key = f"adv_{'_'.join(tickers)}_{end}"
    _hit = read_artifact(_key, max_age_s=3600)
    if _hit is not None:
        return _hit
    out: dict = {}
    try:
        import datetime as _dt
        from src.data.loader import _yf_download
        _s = str(_dt.date.today() - _dt.timedelta(days=140))
        raw = _yf_download(list(tickers), start=_s, end=end, auto_adjust=True, progress=False)
        close, vol = raw["Close"], raw["Volume"]
        for tk in tickers:
            try:
                c = close[tk] if hasattr(close, "columns") else close
                v = vol[tk] if hasattr(vol, "columns") else vol
                dollar = (c * v).dropna().tail(60)
                if len(dollar) >= 20:
                    out[tk] = float(dollar.median())
            except Exception:
                continue
    except Exception:
        return {}
    if out:
        try:
            write_artifact(_key, out)
        except Exception:
            pass
    return out


def _avg_holding_days(deployed: list) -> float:
    """Average holding period of the book in trading days, parsed from each
    trade's holding_period text. Defaults to ~40 days when unparseable."""
    import re
    vals = []
    for t in deployed:
        hp = str(t.get("holding_period", "")).lower()
        nums = [float(x) for x in re.findall(r"\d+\.?\d*", hp)]
        if not nums:
            continue
        mid = sum(nums) / len(nums)
        if "month" in hp:
            mid *= 21
        elif "day" in hp:
            pass
        else:                          # weeks (explicit or assumed)
            mid *= 5
        vals.append(mid)
    return float(np.mean(vals)) if vals else 40.0


def _compute_book_costs_capacity(book: list, all_r_gate: pd.DataFrame, end: str,
                                 aum: float = 250e6, participation: float = 0.15,
                                 exit_days: float = 5.0, rt_bps: float = 10.0) -> "dict | None":
    """Net-of-cost performance and liquidity capacity of the deployed book. Cost
    drag = turnover (from holding period) x a stated round-trip cost; capacity =
    days-to-exit per position at a share of ADV. ADV is best-effort market data,
    so suspiciously thin values are flagged rather than trusted."""
    book_r, L = _deployed_book_return(book, all_r_gate)
    if book_r is None:
        return None
    deployed = [t for t in book if float(t.get("alloc_weight", 0.0)) > 0]
    gross = sum(float(t["alloc_weight"]) for t in deployed) or 1.0

    g_ret = float(book_r.mean() * 252 * 100)
    g_vol = float(book_r.std() * np.sqrt(252) * 100)
    g_sharpe = g_ret / g_vol if g_vol > 0 else 0.0
    hp = _avg_holding_days(deployed)
    rt_yr = 252.0 / max(5.0, hp)                       # round-trips per year
    drag = rt_yr * rt_bps / 100.0                      # %/yr
    n_ret = g_ret - drag
    n_sharpe = n_ret / g_vol if g_vol > 0 else 0.0

    try:
        from src.analysis.trade_generator import all_stock_universe
        d2t = {disp: tk for disp, (tk, *_r) in all_stock_universe().items()}
    except Exception:
        d2t = {}
    positions = []
    for t in deployed:
        a = (t.get("assets") or [None])[0]
        positions.append({"name": a or t.get("name", "?"),
                          "ticker": d2t.get(a), "weight": float(t["alloc_weight"]) / gross})
    tks = tuple(sorted({p["ticker"] for p in positions if p["ticker"]}))
    adv = _load_book_adv(tks, end) if tks else {}

    rows = []
    for p in positions:
        a = adv.get(p["ticker"]) if p["ticker"] else None
        pos_usd = p["weight"] * aum
        dte = (pos_usd / (participation * a)) if (a and a > 0) else None
        rows.append({**p, "adv": a, "pos_usd": pos_usd, "dte": dte,
                     "suspect": bool(a is not None and a < 20e6)})
    rows.sort(key=lambda r: (r["dte"] is None, -(r["dte"] or 0)))
    measured = [r for r in rows if r["dte"] is not None and not r["suspect"]]
    binding = measured[0] if measured else None
    caps = [participation * r["adv"] * exit_days / r["weight"]
            for r in measured if r["adv"] and r["weight"] > 0]
    book_cap = min(caps) if caps else None
    worst = binding["dte"] if binding else 0.0
    n_suspect = sum(1 for r in rows if r["suspect"])
    n_priced = sum(1 for r in rows if r["dte"] is not None)

    if worst > 10:
        verdict = "CAPACITY-CONSTRAINED"
    elif worst > 3:
        verdict = "MODERATE CAPACITY"
    else:
        verdict = "LIQUID"

    return {"g_ret": g_ret, "g_vol": g_vol, "g_sharpe": g_sharpe, "n_ret": n_ret,
            "n_sharpe": n_sharpe, "drag": drag, "hp": hp, "rt_yr": rt_yr, "rt_bps": rt_bps,
            "aum": aum, "participation": participation, "exit_days": exit_days,
            "rows": rows, "binding": binding, "book_cap": book_cap, "verdict": verdict,
            "n_suspect": n_suspect, "n_priced": n_priced, "n_positions": len(rows)}


def _render_book_costs_capacity(book: list, all_r_gate: pd.DataFrame, end: str) -> None:
    """On-screen cost and capacity panel for the deployed book."""
    d = _compute_book_costs_capacity(book, all_r_gate, end)
    if not d:
        return
    _M = "font-family:'JetBrains Mono',monospace;"
    _lbl = f"{_M}font-size:.5rem;letter-spacing:.1em;color:#8890a1"
    vcol = {"LIQUID": "#27ae60", "MODERATE CAPACITY": "#e67e22"}.get(d["verdict"], "#c0392b")

    def _stat(lbl, val, sub, col="#e8e9ed"):
        return (f'<div style="flex:1;padding:.45rem .7rem;border-right:1px solid #1e1e1e">'
                f'<div style="{_lbl}">{lbl}</div>'
                f'<div style="{_M}font-size:1.05rem;font-weight:700;color:{col};'
                f'margin:1px 0">{val}</div>'
                f'<div style="{_M}font-size:.48rem;color:#555960">{sub}</div></div>')

    _cap_txt = (f'${d["book_cap"]/1e6:.0f}M' if d["book_cap"] else "n/a")
    stats = (
        f'<div style="display:flex;border-bottom:1px solid #1e1e1e">'
        + _stat("GROSS SHARPE", f'{d["g_sharpe"]:.2f}', f'{d["g_ret"]:+.1f}%/yr gross')
        + _stat("NET SHARPE", f'{d["n_sharpe"]:.2f}', f'{d["n_ret"]:+.1f}%/yr after costs')
        + _stat("COST DRAG", f'-{d["drag"]:.2f}%<span style="font-size:.5rem">/yr</span>',
                f'{d["rt_yr"]:.1f} round-trips · {d["rt_bps"]:.0f}bps', "#e67e22")
        + _stat("BOOK CAPACITY", _cap_txt,
                f'{d["exit_days"]:.0f}d exit · {d["participation"]*100:.0f}% ADV', vcol)
        + f'</div>')

    # per-position liquidity rows
    _rows = ""
    for r in d["rows"]:
        _adv = f'${r["adv"]/1e6:.0f}M' if r["adv"] else "n/a"
        _dte = f'{r["dte"]:.1f}d' if r["dte"] is not None else "-"
        _flag = ' <span style="color:#c0392b">?data</span>' if r["suspect"] else (
            ' <span style="color:#e67e22">◄ binding</span>' if (d["binding"] and r is d["binding"]) else "")
        _dcol = ("#c0392b" if (r["dte"] and r["dte"] > 10) else
                 "#e67e22" if (r["dte"] and r["dte"] > 3) else "#e8e9ed")
        _nm = (r["name"] or "?")[:22]
        _rows += (
            f'<div style="display:flex;align-items:center;gap:8px;padding:1.5px 0">'
            f'<span style="{_M}font-size:.54rem;color:#e8e9ed;min-width:150px;white-space:nowrap;'
            f'overflow:hidden;text-overflow:ellipsis">{_nm}</span>'
            f'<span style="{_M}font-size:.52rem;color:#8890a1;min-width:40px;text-align:right">'
            f'{r["weight"]*100:.0f}%</span>'
            f'<span style="{_M}font-size:.52rem;color:#8890a1;min-width:60px;text-align:right">'
            f'{_adv}</span>'
            f'<span style="{_M}font-size:.54rem;color:{_dcol};min-width:52px;text-align:right">'
            f'{_dte}</span><span style="{_M}font-size:.5rem">{_flag}</span></div>')

    _bind_name = d["binding"]["name"] if d["binding"] else "n/a"
    _bind_dte = f'{d["binding"]["dte"]:.1f}d' if d["binding"] else "n/a"
    read = (
        f'At an illustrative <b style="color:#e8e9ed">${d["aum"]/1e6:.0f}M</b> sleeve, the book '
        f'exits in a few days per name except the binding position '
        f'(<b style="color:#e8e9ed">{_bind_name}</b>, {_bind_dte}). Cost drag is modest for '
        f'liquid large-caps: ~<b style="color:#e8e9ed">{d["drag"]:.2f}%/yr</b> at '
        f'{d["rt_yr"]:.1f} round-trips and a {d["rt_bps"]:.0f}bps assumption, trimming the Sharpe '
        f'from {d["g_sharpe"]:.2f} to {d["n_sharpe"]:.2f}. Capacity is set by the least-liquid '
        f'name, not the average.')

    _caveat = (f'<div style="border-top:1px solid #1e1e1e;margin-top:.4rem;padding-top:.3rem;'
               f'{_M}font-size:.48rem;color:#555960">Round-trip cost is a stated {d["rt_bps"]:.0f}bps '
               f'assumption (half-spread + impact for liquid US large-caps); turnover from the '
               f'book\'s ~{d["hp"]:.0f}-day holding period. ADV is median trailing-60d volume x price '
               f'and best-effort market data'
               + (f'; {d["n_suspect"]} name(s) flagged ?data (implausibly thin, verify before sizing)'
                  if d["n_suspect"] else "")
               + '. Verify ADV against live venue data before sizing real capital.</div>')

    st.markdown(
        f'<div style="border:1px solid #1e1e1e;background:#0a0a0a;margin:.2rem 0 .8rem">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
        f'padding:.4rem .8rem;border-bottom:1px solid #1e1e1e">'
        f'<span style="{_M}font-size:.6rem;font-weight:700;letter-spacing:.14em;'
        f'color:#e8e9ed">COST &amp; CAPACITY</span>'
        f'<span style="{_M}font-size:.5rem;color:#8890a1">'
        f'net of costs · liquidity at {d["participation"]*100:.0f}% of ADV'
        f'<span style="background:{vcol};color:#000;font-weight:700;padding:1px 7px;'
        f'margin-left:6px;letter-spacing:.08em">{d["verdict"]}</span></span></div>'
        f'{stats}'
        f'<div style="display:flex;gap:16px;padding:.55rem .8rem;flex-wrap:wrap">'
        f'<div style="flex:1;min-width:300px">'
        f'<div style="{_lbl};margin-bottom:3px">DAYS-TO-EXIT · {d["n_priced"]}/{d["n_positions"]} '
        f'priced · at ${d["aum"]/1e6:.0f}M sleeve</div>{_rows}</div>'
        f'<div style="flex:1;min-width:230px;{_M}font-size:.56rem;color:#c9ccd4;'
        f'line-height:1.55">{read}</div></div>'
        f'{_caveat}</div>',
        unsafe_allow_html=True,
    )


def _compute_hedge_overlay(book: list, all_r_gate: pd.DataFrame, start: str, end: str,
                           borrow_bps: float = 40.0, erp: float = 5.0) -> "dict | None":
    """Turn the diagnosis into a product: the tradeable ETF basket that neutralises
    the book's systematic exposure. Sequential hedge (market via SPY, then the
    residual sector tilts via ITA/GLD/XLE/TLT/UUP), with the variance removed, the
    carry cost, the market return forgone, and the residual Sharpe (which the skill
    test already showed is ~0, so the overlay is exposure control, not alpha)."""
    try:
        import statsmodels.api as sm
    except Exception:
        return None
    book_r, L = _deployed_book_return(book, all_r_gate)
    if book_r is None:
        return None
    _ROLE = {"SPY": "market (S&P 500)", "ITA": "defense (aerospace)", "GLD": "gold",
             "XLE": "energy", "TLT": "duration (long Tsy)", "UUP": "US dollar"}
    etfs: dict = {}
    for tk in ["SPY", "ITA", "GLD", "XLE", "TLT", "UUP"]:
        s = _load_etf_returns(tk, start, end)
        if s is not None and len(s) > 250:
            etfs[tk] = s
    if "SPY" not in etfs or len(etfs) < 2:
        return None
    E = pd.concat(etfs, axis=1)
    df = pd.concat([book_r, E], axis=1).dropna()
    if len(df) < 250:
        return None
    _HAC = dict(cov_type="HAC", cov_kwds={"maxlags": 5})
    bm = sm.OLS(df["book"], sm.add_constant(df[["SPY"]])).fit(**_HAC)
    beta_m = float(bm.params["SPY"])
    mkt_hedged = df["book"] - beta_m * df["SPY"]
    sect = [c for c in E.columns if c != "SPY"]
    smf = sm.OLS(mkt_hedged, sm.add_constant(df[sect])).fit(**_HAC)
    sb = {c: (float(smf.params[c]), float(smf.tvalues[c])) for c in sect}
    fac_hedged = mkt_hedged - sum(sb[c][0] * df[c] for c in sect)

    ann = np.sqrt(252)
    vg = float(df["book"].std() * ann * 100)
    vm = float(mkt_hedged.std() * ann * 100)
    vf = float(fac_hedged.std() * ann * 100)
    var_mkt = 1 - (vm / vg) ** 2 if vg > 0 else 0.0
    var_full = 1 - (vf / vg) ** 2 if vg > 0 else 0.0
    short_total = abs(beta_m) + sum(abs(b) for b, _ in sb.values())
    carry = short_total * borrow_bps / 100.0
    beta_forgone = beta_m * erp
    res_sharpe = float(fac_hedged.mean() / fac_hedged.std() * ann) if fac_hedged.std() > 0 else 0.0

    basket = [("SPY", beta_m, _ROLE["SPY"], None)]
    for c in sect:
        b, t = sb[c]
        if abs(t) >= 2 and abs(b) >= 0.03:
            basket.append((c, b, _ROLE.get(c, c), t))
    return {"beta_mkt": beta_m, "basket": basket, "vol_gross": vg, "vol_mkt": vm,
            "vol_fac": vf, "var_mkt": var_mkt * 100, "var_full": var_full * 100,
            "carry": carry, "beta_forgone": beta_forgone, "short_total": short_total,
            "res_sharpe": res_sharpe, "borrow_bps": borrow_bps, "erp": erp,
            "obs": len(df), "n_legs": len(basket)}


def _render_hedge_overlay(book: list, all_r_gate: pd.DataFrame, start: str, end: str) -> None:
    """On-screen hedge-overlay panel: the ETF basket that neutralises the book."""
    d = _compute_hedge_overlay(book, all_r_gate, start, end)
    if not d:
        return
    _M = "font-family:'JetBrains Mono',monospace;"
    _lbl = f"{_M}font-size:.5rem;letter-spacing:.1em;color:#8890a1"
    _GOLD = "#CFB991"

    def _stat(lbl, val, sub, col="#e8e9ed"):
        return (f'<div style="flex:1;padding:.45rem .7rem;border-right:1px solid #1e1e1e">'
                f'<div style="{_lbl}">{lbl}</div>'
                f'<div style="{_M}font-size:1.05rem;font-weight:700;color:{col};'
                f'margin:1px 0">{val}</div>'
                f'<div style="{_M}font-size:.48rem;color:#555960">{sub}</div></div>')

    _rs_col = "#c0392b" if d["res_sharpe"] < 0.1 else "#e8e9ed"
    stats = (
        f'<div style="display:flex;border-bottom:1px solid #1e1e1e">'
        + _stat("VARIANCE REMOVED", f'{d["var_full"]:.0f}%',
                f'market {d["var_mkt"]:.0f}% + sectors', "#27ae60")
        + _stat("HEDGED VOL", f'{d["vol_fac"]:.1f}%',
                f'from {d["vol_gross"]:.1f}% gross')
        + _stat("CARRY COST", f'{d["carry"]:.2f}%<span style="font-size:.5rem">/yr</span>',
                f'+{d["beta_forgone"]:.1f}% beta forgone', "#e67e22")
        + _stat("RESIDUAL SHARPE", f'{d["res_sharpe"]:.2f}',
                'no alpha to unlock', _rs_col)
        + f'</div>')

    _bmax = max((abs(b) for _, b, _, _ in d["basket"]), default=1.0) or 1.0
    rows = ""
    for tk, b, role, t in d["basket"]:
        _w = abs(b) / _bmax * 100
        _side = "SHORT" if b >= 0 else "LONG"
        _sc = "#c0392b" if b >= 0 else "#27ae60"
        rows += (
            f'<div style="display:flex;align-items:center;gap:8px;padding:2px 0">'
            f'<span style="{_M}font-size:.58rem;font-weight:700;color:#e8e9ed;min-width:42px">{tk}</span>'
            f'<span style="{_M}font-size:.5rem;color:#8890a1;min-width:122px">{role}</span>'
            f'<span style="{_M}font-size:.52rem;font-weight:700;color:{_sc};min-width:44px">{_side}</span>'
            f'<div style="flex:1;height:8px;background:#141414"><div style="width:{_w:.0f}%;'
            f'height:8px;background:{_sc}"></div></div>'
            f'<span style="{_M}font-size:.56rem;color:#e8e9ed;min-width:52px;text-align:right">'
            f'{abs(b)*100:.0f}%</span></div>')

    read = (
        f'To hold this book <b style="color:#e8e9ed">factor-neutral</b>, short the basket above per '
        f'dollar of book notional. It removes <b style="color:#27ae60">{d["var_full"]:.0f}%</b> of the '
        f'book\'s variance (vol {d["vol_gross"]:.1f}% to {d["vol_fac"]:.1f}%) for about '
        f'<b style="color:#e8e9ed">{d["carry"]:.2f}%/yr</b> of borrow carry, plus roughly '
        f'<b style="color:#e8e9ed">{d["beta_forgone"]:.1f}%/yr</b> of market return you give up on the '
        f'S&amp;P hedge. What is left has a <b style="color:{_rs_col}">{d["res_sharpe"]:.2f}</b> Sharpe, '
        f'so this is <b>exposure control, not a source of return</b>. Its use: a parent portfolio can '
        f'hold the intended defense-and-gold view without adding the ~{d["beta_mkt"]:.1f} market beta it '
        f'likely already owns.')

    st.markdown(
        f'<div style="border:1px solid #1e1e1e;background:#0a0a0a;margin:.2rem 0 .8rem">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
        f'padding:.4rem .8rem;border-bottom:1px solid #1e1e1e">'
        f'<span style="{_M}font-size:.6rem;font-weight:700;letter-spacing:.14em;'
        f'color:#e8e9ed">HEDGE OVERLAY</span>'
        f'<span style="{_M}font-size:.5rem;color:#8890a1">'
        f'tradeable ETF basket · {d["obs"]} obs'
        f'<span style="background:{_GOLD};color:#000;font-weight:700;padding:1px 7px;'
        f'margin-left:6px;letter-spacing:.08em">EXPOSURE CONTROL</span></span></div>'
        f'{stats}'
        f'<div style="display:flex;gap:16px;padding:.55rem .8rem;flex-wrap:wrap">'
        f'<div style="flex:1.15;min-width:320px">'
        f'<div style="{_lbl};margin-bottom:3px">THE OVERLAY · per $1 of book notional</div>{rows}</div>'
        f'<div style="flex:1;min-width:240px;{_M}font-size:.56rem;color:#c9ccd4;'
        f'line-height:1.55">{read}</div></div>'
        f'<div style="padding:.3rem .8rem;border-top:1px solid #1e1e1e;{_M}'
        f'font-size:.48rem;color:#555960">Sequential hedge: market beta on SPY, then residual sector '
        f'tilts on ITA/GLD/XLE/TLT/UUP (HAC errors, significant legs only). Carry assumes '
        f'{d["borrow_bps"]:.0f}bps ETF borrow; beta forgone assumes a {d["erp"]:.0f}% equity risk '
        f'premium. Exposures drift, so re-estimate before trading.</div></div>',
        unsafe_allow_html=True,
    )


def _compute_hedge_oos(book: list, all_r_gate: pd.DataFrame, start: str, end: str,
                       train: int = 252, step: int = 21) -> "dict | None":
    """Walk-forward out-of-sample test of the hedge overlay. At each monthly step,
    fit the hedge ratios on a trailing `train`-day window and apply them to the
    next `step` days that were NOT used to fit. Aggregates the OOS hedged series
    and measures realised variance removed and residual market beta out of sample,
    the honest test of whether the overlay is real or an in-sample artifact."""
    try:
        import statsmodels.api as sm
    except Exception:
        return None
    book_r, L = _deployed_book_return(book, all_r_gate)
    if book_r is None:
        return None
    etfs = {tk: _load_etf_returns(tk, start, end)
            for tk in ["SPY", "ITA", "GLD", "XLE", "TLT", "UUP"]}
    etfs = {k: v for k, v in etfs.items() if v is not None and len(v) > 250}
    if "SPY" not in etfs or len(etfs) < 2:
        return None
    E = pd.concat(etfs, axis=1)
    df = pd.concat([book_r, E], axis=1).dropna()
    if len(df) < train + 3 * step:
        return None
    sect = [c for c in E.columns if c != "SPY"]

    B, M, F, SP, idx = [], [], [], [], []
    i = train
    while i + 1 < len(df):
        tr = df.iloc[i - train:i]
        oos = df.iloc[i:i + step]
        if len(oos) == 0:
            break
        try:
            bm = float(sm.OLS(tr["book"], sm.add_constant(tr[["SPY"]])).fit().params["SPY"])
            sb = sm.OLS(tr["book"] - bm * tr["SPY"], sm.add_constant(tr[sect])).fit().params.drop("const")
        except Exception:
            i += step
            continue
        B += list(oos["book"].values)
        M += list((oos["book"] - bm * oos["SPY"]).values)
        F += list((oos["book"] - bm * oos["SPY"] - oos[sect].values @ sb.values))
        SP += list(oos["SPY"].values)
        idx += list(oos.index)
        i += step
    if len(B) < 250:
        return None
    Bs, Ms, Fs, SPs = (pd.Series(x, index=idx) for x in (B, M, F, SP))
    ann = np.sqrt(252)
    _X = sm.add_constant(SPs.rename("SPY"))

    def _beta(y):
        try:
            return float(sm.OLS(y.values, _X).fit().params[1])
        except Exception:
            return float("nan")

    vol_b, vol_m, vol_f = (float(s.std() * ann * 100) for s in (Bs, Ms, Fs))
    var_mkt = 1 - (vol_m / vol_b) ** 2 if vol_b > 0 else 0.0
    var_full = 1 - (vol_f / vol_b) ** 2 if vol_b > 0 else 0.0
    beta_b, beta_m, beta_f = _beta(Bs), _beta(Ms), _beta(Fs)

    # in-sample full-hedge variance removed, for the OOS retention ratio
    bmi = float(sm.OLS(df["book"], sm.add_constant(df[["SPY"]])).fit().params["SPY"])
    sbi = sm.OLS(df["book"] - bmi * df["SPY"], sm.add_constant(df[sect])).fit().params.drop("const")
    fhi = df["book"] - bmi * df["SPY"] - df[sect].values @ sbi.values
    is_var = 1 - (fhi.std() / df["book"].std()) ** 2 if df["book"].std() > 0 else 0.0
    retention = (var_full / is_var * 100) if is_var > 0 else 0.0

    def _mdd(s):
        lvl = np.exp(s.cumsum())
        return float(((lvl - lvl.cummax()) / lvl.cummax()).min() * 100)

    dd_b, dd_f = _mdd(Bs), _mdd(Fs)
    rv_b = (Bs.rolling(63).std() * ann * 100).dropna()
    rv_f = (Fs.rolling(63).std() * ann * 100).dropna()
    rv = pd.concat([rv_b.rename("unhedged"), rv_f.rename("hedged")], axis=1).dropna()

    if var_full >= 0.25 and abs(beta_m) < 0.15:
        verdict = "HOLDS OUT OF SAMPLE"
    elif var_full >= 0.10:
        verdict = "PARTIAL"
    else:
        verdict = "DECAYS"

    return {"obs": len(Bs), "n_rebal": len(Bs) // step, "train": train, "step": step,
            "vol_b": vol_b, "vol_f": vol_f, "var_mkt": var_mkt * 100, "var_full": var_full * 100,
            "beta_b": beta_b, "beta_m": beta_m, "beta_f": beta_f, "is_var": is_var * 100,
            "retention": retention, "dd_b": dd_b, "dd_f": dd_f, "verdict": verdict,
            "rv_dates": list(rv.index), "rv_unhedged": list(rv["unhedged"].values),
            "rv_hedged": list(rv["hedged"].values),
            "span": (idx[0], idx[-1])}


def _render_hedge_oos(book: list, all_r_gate: pd.DataFrame, start: str, end: str) -> None:
    """On-screen out-of-sample hedge backtest panel."""
    d = _compute_hedge_oos(book, all_r_gate, start, end)
    if not d:
        return
    import plotly.graph_objects as go
    _M = "font-family:'JetBrains Mono',monospace;"
    _lbl = f"{_M}font-size:.5rem;letter-spacing:.1em;color:#8890a1"
    vcol = {"HOLDS OUT OF SAMPLE": "#27ae60", "PARTIAL": "#e67e22"}.get(d["verdict"], "#c0392b")

    def _stat(lbl, val, sub, col="#e8e9ed"):
        return (f'<div style="flex:1;padding:.45rem .7rem;border-right:1px solid #1e1e1e">'
                f'<div style="{_lbl}">{lbl}</div>'
                f'<div style="{_M}font-size:1.05rem;font-weight:700;color:{col};'
                f'margin:1px 0">{val}</div>'
                f'<div style="{_M}font-size:.48rem;color:#555960">{sub}</div></div>')

    _bcol = "#27ae60" if abs(d["beta_m"]) < 0.15 else "#e67e22"
    _rcol = "#27ae60" if d["retention"] >= 70 else "#e67e22" if d["retention"] >= 40 else "#c0392b"
    stats = (
        f'<div style="display:flex;border-bottom:1px solid #1e1e1e">'
        + _stat("OOS VAR REMOVED", f'{d["var_full"]:.0f}%',
                f'market leg {d["var_mkt"]:.0f}%', "#27ae60")
        + _stat("MKT BETA → 0", f'{d["beta_m"]:+.2f}',
                f'from {d["beta_b"]:+.2f} unhedged', _bcol)
        + _stat("OOS VOL", f'{d["vol_f"]:.1f}%', f'from {d["vol_b"]:.1f}% unhedged')
        + _stat("HELD OOS", f'{d["retention"]:.0f}%', 'of in-sample hedge · not overfit', _rcol)
        + f'</div>')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["rv_dates"], y=d["rv_unhedged"], name="unhedged book",
                             mode="lines", line=dict(width=1.6, color="#c0392b")))
    fig.add_trace(go.Scatter(x=d["rv_dates"], y=d["rv_hedged"], name="fully-hedged",
                             mode="lines", line=dict(width=1.8, color="#27ae60")))
    fig.update_layout(
        template="plotly_dark", height=250, plot_bgcolor="#0a0a0a", paper_bgcolor="#0a0a0a",
        margin=dict(l=8, r=8, t=6, b=6),
        legend=dict(orientation="h", y=1.16, x=0, font=dict(size=10, color="#c8c8c8"),
                    bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(title=dict(text="63d OOS vol %", font=dict(size=10, color="#8890a1")),
                   gridcolor="#161616", zeroline=False),
        xaxis=dict(gridcolor="#161616"))

    read = (
        f'Fitting the hedge on a trailing {d["train"]}-day window and applying it to the next month, '
        f'over {d["n_rebal"]} out-of-sample rebalances, removes <b style="color:#27ae60">'
        f'{d["var_full"]:.0f}%</b> of the book\'s variance out of sample ('
        f'<b style="color:#e8e9ed">{d["retention"]:.0f}%</b> of the in-sample figure) and neutralises '
        f'market beta from <b style="color:#e8e9ed">{d["beta_b"]:+.2f}</b> to '
        f'<b style="color:{_bcol}">{d["beta_m"]:+.2f}</b>. The overlay is <b>not an in-sample '
        f'artifact, it works forward</b>. Two honest caveats. The sector legs slightly over-hedge as '
        f'exposures drift (fully-hedged beta {d["beta_f"]:+.2f}), so the market leg carries the '
        f'reliable benefit. And the residual the overlay isolates has no alpha, so held on its own it '
        f'bleeds (a <b style="color:#c0392b">{d["dd_f"]:.0f}%</b> drawdown over the sample): the '
        f'overlay is exposure control <b>inside</b> a parent portfolio, not a standalone strategy. '
        f'Re-estimate monthly.')

    st.markdown(
        f'<div style="border:1px solid #1e1e1e;border-bottom:none;background:#0a0a0a;'
        f'display:flex;justify-content:space-between;align-items:baseline;padding:.4rem .8rem;'
        f'margin-top:.2rem"><span style="{_M}font-size:.6rem;font-weight:700;letter-spacing:.14em;'
        f'color:#e8e9ed">HEDGE OVERLAY · OUT-OF-SAMPLE</span>'
        f'<span style="{_M}font-size:.5rem;color:#8890a1">'
        f'walk-forward · {d["train"]}d train / {d["step"]}d test · {d["obs"]} obs'
        f'<span style="background:{vcol};color:#000;font-weight:700;padding:1px 7px;'
        f'margin-left:6px;letter-spacing:.08em">{d["verdict"]}</span></span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div style="border-left:1px solid #1e1e1e;border-right:1px solid #1e1e1e;'
                f'background:#0a0a0a">{stats}</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown(
        f'<div style="border:1px solid #1e1e1e;border-top:none;background:#0a0a0a;'
        f'padding:.1rem .8rem .55rem"><div style="{_M}font-size:.56rem;color:#c9ccd4;'
        f'line-height:1.55">{read}</div>'
        f'<div style="border-top:1px solid #1e1e1e;margin-top:.4rem;padding-top:.3rem;'
        f'{_M}font-size:.48rem;color:#555960">Hedge ratios re-fit on each trailing window and applied '
        f'only to the following out-of-sample month; the chart is rolling 63-day realised vol of the '
        f'unhedged book vs the fully-hedged overlay. Illustrative, not investment advice.</div></div>',
        unsafe_allow_html=True,
    )


def page_trade_ideas(start: str, end: str, fred_key: str = "") -> None:
    # ── Stale-while-revalidate: pre-populate session state from disk cache ───
    # Runs once per session. If a prior run saved results to disk, the user
    # sees them immediately without clicking "Run Validation".
    _PV_DISK_KEY = "pipeline_validation"
    _PV_SESSION_KEY = "pipeline_validation_result"
    _PV_SAVED_KEY = "pipeline_validation_saved_at"   # raw UTC ts, survives reruns
    _PV_STALE_HOURS = 24                             # book older than this ⇒ STALE
    _pv_disk_age: "str | None" = None
    if _PV_SESSION_KEY not in st.session_state:
        try:
            from src.utils.page_cache import load_cache, age_str as _age_str
            _disk_data, _disk_saved_at = load_cache(_PV_DISK_KEY)
            if _disk_data is not None:
                st.session_state[_PV_SESSION_KEY] = _disk_data
                # Keep the raw timestamp so the STALE gate can survive reruns
                # (without it, a rerun mislabels the disk book as 'this session').
                st.session_state[_PV_SAVED_KEY] = _disk_saved_at
                _pv_disk_age = _age_str(_disk_saved_at)
        except Exception:
            pass

    # Staleness of the currently-loaded book (from the session timestamp, which
    # survives reruns). Computed once here so the top-of-page banner and the
    # validation section below always agree. Book older than the threshold ⇒ STALE.
    _pv_saved_at = st.session_state.get(_PV_SAVED_KEY)
    _pv_has_book = bool(st.session_state.get(_PV_SESSION_KEY))
    try:
        from src.utils.page_cache import age_hours as _age_h, age_str as _age_s
        _pv_age_h   = _age_h(_pv_saved_at)
        _pv_age_lbl = _age_s(_pv_saved_at) if _pv_saved_at else ""
    except Exception:
        _pv_age_h, _pv_age_lbl = None, ""
    _pv_stale = _pv_has_book and _pv_age_h is not None and _pv_age_h > _PV_STALE_HOURS

    st.markdown(_TI_STYLE, unsafe_allow_html=True)
    _page_header("Structured Trade Ideas",
                 "Step 6 of 7 · Regime-driven · Conflict-linked · 5-Stage Pipeline Validation")
    _ti_intro_col, _ti_pdf_col = st.columns([4, 1.2], gap="medium")
    with _ti_intro_col:
        _page_intro(
            "This page is a disciplined research book <strong>and a rigorous audit of that book</strong>. "
            "It builds a regime-driven equity sleeve from a walk-forward-validated pipeline, then holds it "
            "to the standard a real desk would apply. The audit that follows (factor attribution, a "
            "factor-neutral skill test on Fama-French 5 plus Momentum, rolling exposures, and cost and "
            "capacity) reaches one verdict: <strong>the book is market and factor beta, not selection "
            "alpha</strong>. So read this as risk and hedging intelligence, not a stock-picking signal. "
            "The deliverable is knowing exactly what the book is exposed to, and what it is not."
        )
        _definition_block(
            "What this book is, and is not",
            "<strong>What it is:</strong> a cross-asset risk-monitoring and hedge-overlay tool. The four "
            "analyses below show the book carries roughly 0.7 market beta across only a handful of "
            "effective bets, no statistically significant Jensen or factor-neutral alpha, exposures that "
            "drift over time, and capacity set by its least-liquid name. That is a factor tilt, honestly "
            "measured. <strong>What it is not:</strong> an alpha engine. Two structural reasons reinforce "
            "this: the book is long-only and fully invested, so its beta and tilts are deliberate rather "
            "than skill; and the static thesis library was chosen with hindsight, so the walk-forward "
            "validation controls execution look-ahead but not thesis selection. Use the terminal to map "
            "regime and contagion risk and to size hedges, not to claim a stock-selection edge the "
            "terminal itself disproves."
        )
        _definition_block(
            "The one accountable call",
            "A research tool that hedges every claim is just a disclaimer generator, so here is the single "
            "falsifiable view this terminal will stake its name on. <strong>With cross-asset connectedness "
            "elevated and the book carrying about 0.7 market beta across only a handful of independent "
            "bets, this book will lose more than its beta implies in the next equity drawdown of 5 percent "
            "or more</strong>, because correlations spike and diversification fails exactly when it is "
            "needed most. <strong>Invalidated if</strong>, in that next drawdown, the book's peak-to-trough "
            "loss comes in below 0.7 times the S&amp;P 500's. That is testable on the next risk-off event, "
            "and the terminal is on record."
        )
    with _ti_pdf_col:
        # Desk-report PDF - fills the blank space beside the intro. The book is
        # built lower on the page, so this sets a flag and generation runs below
        # (guarded); the finished PDF is stashed and the download surfaces here.
        st.markdown('<div style="height:.2rem"></div>', unsafe_allow_html=True)
        if st.button("Generate Desk Report (PDF)", key="gen_report_top",
                     type="primary", width="stretch",
                     help="Invested book only · with recent third-party coverage per name"):
            st.session_state["_ti_pdf_pending"] = True
        if st.session_state.get("_ti_pdf_bytes"):
            st.download_button(
                "Download Desk Report", data=st.session_state["_ti_pdf_bytes"],
                file_name=st.session_state.get("_ti_pdf_name", "desk_report.pdf"),
                mime="application/pdf", key="dl_report_top",
                width="stretch")
        # Refresh/rerun of the validated book, right under the report action so it
        # is ALWAYS reachable (never gated away). When the book is > threshold old a
        # red STALE banner sits above it; otherwise a subtle "validated ago" note.
        # The button flags the deep 5-stage validation to recompute on this pass.
        if _pv_stale:
            st.markdown(
                "<div style=\"font-family:'JetBrains Mono',monospace;border:1px solid #c0392b;"
                "border-left:3px solid #c0392b;border-radius:4px;padding:6px 10px;"
                "background:#1a0808;margin:.4rem 0 .3rem\">"
                "<div style=\"font-size:0.6rem;color:#e05241;font-weight:700;"
                "letter-spacing:.08em\">⚠ STALE BOOK</div>"
                f"<div style=\"font-size:0.54rem;color:#c98b86;margin-top:2px;line-height:1.35\">"
                f"Validated {_pv_age_lbl} (&gt; {_PV_STALE_HOURS}h). Rerun before trusting "
                "the weights below.</div></div>",
                unsafe_allow_html=True,
            )
        elif _pv_has_book and _pv_age_lbl:
            st.markdown(
                "<div style=\"font-family:'JetBrains Mono',monospace;font-size:0.54rem;"
                f"color:#8890a1;margin:.4rem 0 .28rem\">Book validated {_pv_age_lbl}.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div style="height:.4rem"></div>', unsafe_allow_html=True)
        if st.button(("⚠ Rerun Validation" if _pv_stale else "Refresh Validation"),
                     key="ti_stale_rerun_top", type="primary", width="stretch",
                     help="Re-runs the 5-Stage Pipeline Validation (~2-4 min)."):
            st.session_state["_ti_force_pv"] = True
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
        st.caption("Geo context unavailable - conflict model not loaded for this session.")

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

    # Trade cards removed - the pipeline is the deliverable, not individual strategy scores.
    active_trades: list[dict] = []
    asset_exposure: dict = {}   # was populated by score_all_assets(); empty without trade cards

    # Extend returns to include Fixed Income + FX + Private Credit legs.
    # This frame is the SINGLE tradeable universe: the eligibility gate, the
    # Stage-3 runs, the allocator backtests, the constraint clusters, the
    # Integrity Audit and Multiple Testing all read the same columns.
    _PC_LEG_MAP = {"ARCC": "Ares Capital (ARCC)", "OBDC": "Blue Owl (OBDC)"}
    try:
        from src.data.loader import load_private_credit_returns
        _pc_r = (load_private_credit_returns(start, end)
                 .reindex(columns=list(_PC_LEG_MAP)).rename(columns=_PC_LEG_MAP))
    except Exception:
        _pc_r = pd.DataFrame()
    _extra_frames: list[pd.DataFrame] = []
    if not _fi_r.empty:
        _extra_frames.append(_fi_r)
    if not _fx_r.empty:
        _extra_frames.append(_fx_r)
    if not _pc_r.empty:
        _extra_frames.append(_pc_r)
    all_r_concat = pd.concat([eq_r, cmd_r] + _extra_frames, axis=1)

    # Effective N for DSR and HLZ multiple-testing gates
    _RAW_N = 18
    _effective_n: int = st.session_state.get("_effective_n", 9)

    # Per-run working copy of the trade library. The pipeline below stamps
    # is_eligible / alloc_weight / rank onto each dict; mutating the module
    # global would leak one session's state into every other session sharing
    # the process. This local rebind means every _TRADE_LIBRARY reference in
    # this function operates on the isolated copy.
    import copy as _copy
    _TRADE_LIBRARY = _copy.deepcopy(_TRADE_LIBRARY_BASE)

    # ── LIVE IDEA GENERATION - conflict-driven candidates for THIS regime ────
    # The desk is not just a static catalogue: it also generates fresh trades
    # from the current conflict / exposure / scenario / regime state and runs
    # them through the SAME eligibility → Stage-3 → DSR gate as the library.
    # Nothing is weakened - the honest trial count (n_strategies) is raised to
    # the full eligible set below, so the wider search is properly penalised by
    # the deflated Sharpe. Generated ideas only deploy if they earn it.
    _n_generated = 0
    try:
        import copy as _copy_gen
        # Signal-ranked single-name candidates for this regime (directional +
        # safe-haven, US/India/China + macro), cached so the ~8s cold scoring
        # runs once, not on every rerun. Deep-copy before mutating/appending - 
        # later steps stamp eligibility/weights in place and must never touch
        # the cached objects. Each runs the SAME eligibility → Stage-3 → sizing
        # gate; the wider search raises the deflated-Sharpe trial penalty.
        with st.spinner("Generating live single-name ideas…"):
            _gen_signal = _copy_gen.deepcopy(_live_generated_cached(current, start, end))
        _existing = {t.get("name") for t in _TRADE_LIBRARY}
        for _g in _gen_signal:
            if _g.get("name") and _g["name"] not in _existing:
                _TRADE_LIBRARY.append(_g)
                _existing.add(_g["name"])
                _n_generated += 1
    except Exception:
        _n_generated = 0

    # ── STEP 1 OF 4 - ELIGIBILITY GATE ─────────────────────────────────────
    # Eligible ⟺ every leg exists in the loaded returns frame AND the thesis
    # passed Stage-3 confirmation. Everything else is NON-ALLOCATABLE: still
    # rendered, clearly marked, hard-locked to zero weight (enforce_weight in
    # trade_filter.py is the choke point steps 2–4 must route through).
    from src.analysis.trade_filter import annotate_eligibility
    # Merge single-stock returns (US/India/China) into the gate frame so the
    # generated single-name trades are eligible + backtestable + deployable.
    try:
        with st.spinner("Loading single-stock returns (US · India · China)…"):
            _stock_r = _load_stock_returns(start, end)
    except Exception:
        _stock_r = pd.DataFrame()
    if _stock_r is not None and not _stock_r.empty:
        _all_r_gate = pd.concat([all_r_concat, _stock_r.reindex(all_r_concat.index)], axis=1)
        _all_r_gate = _all_r_gate.loc[:, ~_all_r_gate.columns.duplicated()]
    else:
        _all_r_gate = all_r_concat   # full tradeable universe - eq/cmd/FI/FX/PC
    # Loadable universe: every leg display name with ANY loader mapping,
    # loaded or not. A missing leg outside this set is STRUCTURALLY DEAD
    # (can never trade); inside it, it is merely missing from today's fetch.
    from src.data.config import FIXED_INCOME_TICKERS, FX_TICKERS
    _loadable = (set(_all_r_gate.columns) | set(FIXED_INCOME_TICKERS)
                 | set(FX_TICKERS) | set(_PC_LEG_MAP.values()))

    # ── Risk appetite → concentration ──────────────────────────────────────
    # This equity sleeve is ALWAYS fully invested; appetite controls how tightly
    # capital concentrates into the strongest risk-adjusted ideas, not a cash
    # level. Defensive spreads broadly across positive-edge names; Aggressive
    # concentrates into the best few. Ranking and the raw DSR are unaffected - 
    # only the shape of the book responds.
    from src.analysis.trade_allocator import (
        APPETITE_STOPS, effective_deploy_bar, DSR_DEPLOY_BAR, DEPLOY_BAR_FLOOR,
    )
    # Risk-appetite slider removed - the sleeve runs the AGGRESSIVE (concentrated,
    # return-forward) profile as standard: sized into the strongest ~11 names.
    _appetite_label = "Aggressive"
    _appetite = 1.0
    _eff_bar = effective_deploy_bar(_appetite)   # dsr_factor transparency only
    _concentration = 2.75
    _top_n = 11
    st.markdown(
        f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:.55rem;'
        f'color:#8890a1;margin:-.2rem 0 .5rem">BOOK PROFILE · '
        f'<b style="color:#c0392b">CONCENTRATED · RETURN-FORWARD</b> · '
        f'100% invested · &le;<b style="color:#e8e9ed">{_top_n}</b> positions · '
        f'sized into the highest-conviction names</p>',
        unsafe_allow_html=True,
    )
    try:
        with st.spinner("Running eligibility gate - leg check + Stage-3 confirmation…"):
            _s3_results = _library_stage3_results(_all_r_gate, regimes,
                                                  trades=_TRADE_LIBRARY)
    except Exception:
        _s3_results = {}
    annotate_eligibility(_TRADE_LIBRARY, _all_r_gate.columns, _s3_results,
                         loadable_universe=_loadable)
    # Honest trial count for the deflated Sharpe: the number of ELIGIBLE
    # strategies actually searched (static + generated). A wider search MUST
    # raise this so the best-of-N luck benchmark is harder, not easier - this
    # is the anti-gaming half of live generation.
    _n_eligible_total = sum(1 for _t in _TRADE_LIBRARY if _t.get("is_eligible"))
    _n_trials = max(_n_eligible_total, 9)
    # ── STEP 2 OF 4 - WEIGHT ALLOCATOR (silent: stamps alloc_weight +
    # alloc_detail; ranking and display are steps 3–4) ──────────────────────
    try:
        from src.analysis.trade_allocator import (
            build_allocation_inputs, allocate_weights,
        )
        _alloc_metrics = build_allocation_inputs(
            _TRADE_LIBRARY, _all_r_gate, regimes, _s3_results,
            n_strategies=_n_trials, deploy_bar=_eff_bar,
        )
        # Equity sleeve: always 100% invested; risk appetite now sharpens the
        # confidence tilt (Defensive spreads broadly, Aggressive concentrates in
        # the highest-DSR ideas) rather than moving capital to cash.
        allocate_weights(_TRADE_LIBRARY, _alloc_metrics,
                         concentration=_concentration, top_n=_top_n)
        # ── STEP 3 OF 4 - PORTFOLIO CONSTRAINTS (silent) ────────────────────
        # Conflict cap 40% / correlation-cluster cap 45% of gross, iterative
        # re-normalization; runtime-asserted, stamps constraint_detail.
        from src.analysis.trade_allocator import apply_portfolio_constraints
        apply_portfolio_constraints(_TRADE_LIBRARY, _all_r_gate)
    except Exception:
        for _t in _TRADE_LIBRARY:
            _t.setdefault("alloc_weight", 0.0)
    # ── STEP 4 OF 4 - RANKED BOOK · LIVE ────────────────────────────────────
    # Regenerated from current regime / Stage-3 / backtest state on every
    # load. Attractiveness = 0.50·DSR + 0.35·conviction + 0.15·constraint
    # room - raw Sharpe is displayed for transparency, never ranked on.
    _ranked_book: list[dict] = []
    try:
        from src.analysis.trade_allocator import rank_trades
        _ranked_book = rank_trades(_TRADE_LIBRARY)
    except Exception:
        for _t in _TRADE_LIBRARY:           # allocator failure ⇒ nothing sized
            _t.setdefault("alloc_weight", 0.0)
    _n_elig = sum(1 for _t in _TRADE_LIBRARY if _t.get("is_eligible"))
    _n_dead = sum(1 for _t in _TRADE_LIBRARY if _t.get("structurally_dead"))
    _n_lock = len(_TRADE_LIBRARY) - _n_elig - _n_dead
    _dead_note = (f' · <b style="color:#555960">{_n_dead} STRUCTURALLY DEAD</b> '
                  f'(no data source - excluded from the live count)'
                  if _n_dead else '')
    _gen_note = (f' · <b style="color:#2980b9">{_n_generated} LIVE-GENERATED</b> '
                 f'(signal-ranked, this regime)' if _n_generated else '')
    st.markdown(
        f'<div style="background:#080808;border:1px solid #1e1e1e;'
        f'border-left:3px solid {"#27ae60" if _n_elig else "#c0392b"};'
        f'padding:.4rem .9rem;margin:.4rem 0 .6rem;'
        f'font-family:\'JetBrains Mono\',monospace;font-size:.6rem;color:#8890a1">'
        f'ELIGIBILITY GATE · <b style="color:#27ae60">{_n_elig} ELIGIBLE</b> · '
        f'<b style="color:#c0392b">{_n_lock} NON-ALLOCATABLE</b> (zero-weight locked)'
        f'{_gen_note}{_dead_note} · DSR trial count n={_n_trials} '
        f'(deflated for the full search) · '
        f'eligible ⟺ all legs in return data AND thesis Stage-3 confirmed</div>',
        unsafe_allow_html=True,
    )

    if _ranked_book:
        _gross_bk = sum(t.get("alloc_weight", 0.0) for t in _ranked_book)
        # Equity sleeve: fully invested by design. The book is 100% cash ONLY in
        # the STRUCTURAL case where nothing is eligible this regime (no legs in
        # data / no Stage-3 confirmation) - never as a deploy-bar risk decision.
        _is_cash_book = _gross_bk <= 1e-9
        _speculative = False
        _best_dsr = max(
            (float((t.get("alloc_detail") or {}).get("dsr", 0.0))
             for t in _ranked_book), default=0.0)
        _n_deployed = sum(1 for t in _ranked_book if t.get("alloc_weight", 0.0) > 0)
        if _is_cash_book:
            st.markdown(
                f'<div style="border:1px solid #CFB991;background:#0d0b06;'
                f'padding:.65rem 1rem;margin:.2rem 0 .6rem">'
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:baseline;flex-wrap:wrap">'
                f'<span style="font-family:\'JetBrains Mono\',monospace;'
                f'font-size:.92rem;font-weight:700;letter-spacing:.06em;'
                f'color:#CFB991">EQUITY SLEEVE - NO ELIGIBLE POSITIONS</span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;'
                f'font-size:.52rem;color:#8890a1">{_appetite_label.upper()}</span></div>'
                f'<div style="font-family:\'DM Sans\',sans-serif;font-size:.63rem;'
                f'color:#8890a1;margin-top:4px">No thesis is eligible this regime - '
                f'every candidate is missing return data or failed Stage-3 '
                f'confirmation, so there is nothing to invest in. Ranked ideas below '
                f'are a watchlist. This sleeve holds cash only when it structurally '
                f'cannot invest - never as a risk call; that overlay lives at the '
                f'parent-portfolio level.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="border:1px solid #27ae60;background:#07120b;'
                f'padding:.65rem 1rem;margin:.2rem 0 .6rem">'
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:baseline;flex-wrap:wrap">'
                f'<span style="font-family:\'JetBrains Mono\',monospace;'
                f'font-size:.92rem;font-weight:700;letter-spacing:.06em;'
                f'color:#27ae60">EQUITY SLEEVE - 100% INVESTED · '
                f'{_n_deployed} POSITIONS</span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;'
                f'font-size:.52rem;color:#8890a1">CONCENTRATED · '
                f'RETURN-FORWARD</span></div>'
                f'<div style="font-family:\'DM Sans\',sans-serif;font-size:.63rem;'
                f'color:#8890a1;margin-top:4px">The equity component of a larger '
                f'portfolio - always fully invested; the cash / hedge overlay is the '
                f'parent allocation&rsquo;s call, not this sleeve&rsquo;s. Capital is '
                f'sized by <b>risk-adjusted expected return</b> (each idea&rsquo;s '
                f'own direction-aware backtested edge, shrunk by the deflated-Sharpe '
                f'luck penalty) and concentrated in the strongest {_n_deployed} of '
                f'{_n_elig} eligible ideas; the rest rank below as a zero-weight '
                f'watchlist.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Portfolio targeted upside (constructed book only) ───────────────
        _up = _portfolio_upside(_ranked_book, current)
        if _up is not None:
            _lo, _hi = _up["months_lo"], _up["months_hi"]
            _hz = (f"~{_hi}mo" if _lo == _hi else f"{_lo}–{_hi}mo") if _hi else "stated horizon"
            _exp_c = "#27ae60" if _up["expected"] >= 0 else "#c0392b"
            def _pct(v: float) -> str: return f"{v:+.1f}%"
            _tiles = "".join(
                f'<div style="flex:1;min-width:96px;border-left:2px solid {col};'
                f'padding:2px 10px">'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.46rem;'
                f'letter-spacing:.1em;color:#8890a1">{lbl}</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.9rem;'
                f'font-weight:700;color:{col}">{val}</div></div>'
                for lbl, val, col in [
                    ("ANNUALIZED E[R] (~)", _pct(_up.get("annualized", _up["expected"]*12)), _exp_c),
                    ("EXPECTED · E[R]/MO", _pct(_up["expected"]), _exp_c),
                    ("BULL /MO (90th)", _pct(_up["best"]), "#27ae60"),
                    ("BEAR /MO (10th)", _pct(_up["worst"]), "#c0392b"),
                    ("BREAKEVEN /MO", f'{_up["breakeven"]*100:.0f}%', "#CFB991"),
                ]
            )
            st.markdown(
                f'<div style="border:1px solid #1e1e1e;background:#070707;'
                f'padding:.55rem .9rem;margin:0 0 .6rem">'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.54rem;'
                f'font-weight:700;letter-spacing:.13em;color:#CFB991;margin-bottom:6px">'
                f'PORTFOLIO TARGETED UPSIDE · {_up["n"]} POSITIONS · '
                f'{_up["gross"]*100:.0f}% DEPLOYED</div>'
                f'<div style="display:flex;flex-wrap:wrap;gap:6px 4px">{_tiles}</div>'
                f'<div style="font-family:\'DM Sans\',sans-serif;font-size:.55rem;'
                f'color:#555960;margin-top:6px">Projected from each position&rsquo;s '
                f'own regime-conditional, direction-aware <b>backtest</b> (mean and '
                f'dispersion of realised returns), normalised to a common monthly '
                f'horizon and weighted by allocation on a fully-invested book. '
                f'Annualised is the monthly E[R] &times;12; the bull/bear cone is a '
                f'&plusmn;1.28&sigma; band using a disclosed 0.35 intra-book '
                f'correlation so diversification is credited. Positions held ~{_hz}. '
                f'Potential upside, not a forecast.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        _bk_rows = ""
        for _t in _ranked_book:
            _ad = _t.get("attr_detail") or {}
            _dd = _t.get("alloc_detail") or {}
            _cdt = _t.get("constraint_detail") or {}
            _inval = (_t.get("invalidation") or _t.get("exit")
                      or _t.get("risk") or " - ")
            _bk_rows += (
                f'<tr style="border-bottom:1px solid #141414">'
                f'<td style="padding:3px 8px;font-family:\'JetBrains Mono\',monospace;'
                f'font-size:.62rem;font-weight:700;color:#CFB991">#{_t["rank"]}</td>'
                f'<td style="padding:3px 8px;font-size:.64rem;color:#e8e9ed">'
                f'{_t["name"][:52]}</td>'
                f'<td style="padding:3px 8px;font-family:\'JetBrains Mono\',monospace;'
                f'font-size:.64rem;font-weight:700;text-align:right;'
                f'color:{"#27ae60" if _t.get("alloc_weight",0)>0 else "#8890a1"}">'
                f'{_t.get("alloc_weight",0)*100:5.1f}%</td>'
                f'<td style="padding:3px 8px;font-family:\'JetBrains Mono\',monospace;'
                f'font-size:.64rem;font-weight:700;text-align:right;color:#e8e9ed">'
                f'{_t.get("attractiveness",0):.3f}</td>'
                f'<td style="padding:3px 8px;font-family:\'JetBrains Mono\',monospace;'
                f'font-size:.56rem;text-align:right;color:#8890a1">'
                f'{_ad.get("dsr",0):.2f} / {_ad.get("conviction",0):.2f} / '
                f'{_ad.get("room",0):.2f}</td>'
                f'<td style="padding:3px 8px;font-family:\'JetBrains Mono\',monospace;'
                f'font-size:.56rem;text-align:right;color:'
                f'{"#CFB991" if _is_cash_book else ("#e67e22" if _cdt.get("clipped") else "#8890a1")}">'
                f'{_weight_earn_condition(_dd) if _is_cash_book else ("CLIPPED" if _cdt.get("clipped") else " - ")}</td>'
                f'<td style="padding:3px 8px;font-size:.54rem;color:#8890a1;'
                f'max-width:260px;white-space:nowrap;overflow:hidden;'
                f'text-overflow:ellipsis">{_inval[:90]}</td>'
                f'</tr>'
            )
        _hdr_cells = "".join(
            f'<th style="padding:4px 8px;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:.5rem;letter-spacing:.1em;color:#8890a1;text-align:{_a}">{_h}</th>'
            for _h, _a in [("RANK","left"),("TRADE","left"),("WEIGHT","right"),
                           ("ATTR","right"),("DSR/CONV/ROOM","right"),
                           ("TO EARN WEIGHT" if _is_cash_book else "CAPS","right"),
                           ("INVALIDATED IF","left")]
        )
        st.markdown(
            f'<div style="border:1px solid #1e1e1e;background:#0a0a0a;margin-bottom:.7rem">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
            f'padding:.4rem .8rem;border-bottom:1px solid #1e1e1e">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:.6rem;'
            f'font-weight:700;letter-spacing:.14em;color:#e8e9ed">'
            f'{"WATCHLIST · RANKED - NOT YET ALLOCATABLE" if _is_cash_book else "RANKED BOOK · LIVE"}</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:.52rem;'
            f'color:#8890a1">regenerated {datetime.datetime.now().strftime("%H:%M:%S")} · '
            f'regime {r_name.upper()} · {len(_ranked_book)} eligible · '
            f'gross {_gross_bk*100:.1f}% · cash {(1-_gross_bk)*100:.1f}%</span></div>'
            f'<table style="width:100%;border-collapse:collapse">'
            f'<thead><tr style="border-bottom:1px solid #1e1e1e">{_hdr_cells}</tr></thead>'
            f'<tbody>{_bk_rows}</tbody></table>'
            f'<div style="padding:.3rem .8rem;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:.5rem;color:#555960">attractiveness = 0.50·DSR + 0.35·conviction '
            f'+ 0.15·constraint room · raw Sharpe never ranks · zero weights = no thesis '
            f'clears the deflated-edge bar (honest cash book)</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Is this book alpha or beta? Factor & alpha attribution of the deployed
        # positions - the buy-side question the pipeline validation doesn't answer.
        try:
            _render_book_factor_decomp(_ranked_book, _all_r_gate, start, end)
        except Exception:
            pass
        # Does any edge survive stripping the known risk factors (FF5 + Momentum)?
        # The real test of selection skill, run on the factor-neutral residual.
        try:
            _n_th = st.session_state.get("_effective_n") or len(_TRADE_LIBRARY) or 9
            _render_factor_neutral_skill(_ranked_book, _all_r_gate, start, end, _n_th)
        except Exception:
            pass
        # Are those factor exposures stable, or is the static attribution a blur
        # of two different books? Rolling betas over time.
        try:
            _render_rolling_exposures(_ranked_book, _all_r_gate, start, end)
        except Exception:
            pass
        # Even if there were alpha, costs and capacity decide whether it is real
        # money: net-of-cost Sharpe, turnover drag, and days-to-exit per name.
        try:
            _render_book_costs_capacity(_ranked_book, _all_r_gate, end)
        except Exception:
            pass
        # The product the diagnosis points at: the ETF basket that neutralises the
        # book's systematic exposure, and what it removes and costs.
        try:
            _render_hedge_overlay(_ranked_book, _all_r_gate, start, end)
        except Exception:
            pass
        # Does the overlay actually work out of sample, or is it an in-sample fit?
        # Walk-forward: fit the hedge on trailing data, apply it to the next month.
        try:
            _render_hedge_oos(_ranked_book, _all_r_gate, start, end)
        except Exception:
            pass

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

        # Leg directions - all legs, asset name stripped of parentheticals
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

        # Target (optional - newer theses)
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

        # Investor lens (optional - newer theses)
        lens    = tr.get("investor_lens", [])
        lens_h  = (
            f'<div style="font-size:.49rem;color:#CFB99175;'
            f'font-family:\'JetBrains Mono\',monospace;margin-top:2px;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
            f'{" · ".join(lens[:3])}</div>'
        ) if lens else ""

        # STEP-1 eligibility verdict - every card states it; locked cards name why
        _elig = tr.get("is_eligible", False)
        _elig_reason = tr.get("eligibility_reason", "gate not run")
        if _elig:
            elig_badge = (
                f'<span style="font-family:\'JetBrains Mono\',monospace;'
                f'font-size:.44rem;font-weight:700;letter-spacing:.1em;'
                f'color:#27ae60;background:#27ae6014;border:1px solid #27ae6035;'
                f'padding:1px 5px;flex-shrink:0">ELIGIBLE</span>'
            )
            elig_reason_h = ""
        else:
            elig_badge = (
                f'<span style="font-family:\'JetBrains Mono\',monospace;'
                f'font-size:.44rem;font-weight:700;letter-spacing:.1em;'
                f'color:#e05241;background:#c0392b18;border:1px solid #c0392b40;'
                f'padding:1px 5px;flex-shrink:0">NON-ALLOCATABLE · 0 WT</span>'
            )
            elig_reason_h = (
                f'<div style="font-size:.48rem;color:#e0524190;'
                f'font-family:\'JetBrains Mono\',monospace;margin-top:2px;'
                f'line-height:1.3">⊘ {_elig_reason}</div>'
            )

        return (
            f'<div style="padding:7px 8px 6px;border-bottom:1px solid #151515;'
            f'{"opacity:.62" if not _elig else ""}">'
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
            f'<div style="display:flex;gap:4px;flex-shrink:0;margin-top:1px;'
            f'align-items:center">{elig_badge}{reg_html}</div>'
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
            f'{lens_h}{elig_reason_h}'
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

    # Build section list for left column - each with 2 detail lines
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
        f'THESES REFERENCED - {_n_theses} PAIRS</div>'
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
        'The desk report contains only the <b>invested book</b> (deployed positions), each anchored to '
        'recent third-party coverage per name. Generate it from the button at the top of the page.'
        '</div>',
        unsafe_allow_html=True,
    )
    # Generation runs here (where the book exists), triggered by the top button's
    # flag - the redundant bottom "Generate" button was removed.
    if bool(st.session_state.pop("_ti_pdf_pending", False)):
        _pdf_ok = False
        try:
            from src.reports.report_generator import generate_report
            with st.spinner("Building report - invested book + recent coverage + charts…"):
                stress = composite_stress_index(eq_r, cmd_r, avg_corr=avg_corr)
                from src.analysis.trade_allocator import desk_report_feed
                # Invested book only - desk_report_feed keeps just the deployed
                # positions (alloc_weight > 0). Enrich each with recent REAL
                # third-party coverage (source + date + link, last ~30d) so the
                # report anchors every idea to live market context.
                _feed = desk_report_feed(_ranked_book)
                _attach_recent_news(_feed)
                _attach_logos(_feed)   # company marks for single-name legs
                # Factor & alpha attribution of the deployed book → carried into
                # the report so the "alpha or beta?" verdict travels with the PDF.
                _fd = None
                _skill = None
                _roll = None
                _cost = None
                _hedge = None
                _hoos = None
                try:
                    _fd = _compute_book_factor_decomp(_ranked_book, _all_r_gate,
                                                      start, end)
                    _n_th = st.session_state.get("_effective_n") or len(_TRADE_LIBRARY) or 9
                    _skill = _compute_factor_neutral_skill(_ranked_book, _all_r_gate,
                                                           start, end, _n_th)
                    _roll = _compute_rolling_exposures(_ranked_book, _all_r_gate, start, end)
                    _cost = _compute_book_costs_capacity(_ranked_book, _all_r_gate, end)
                    _hedge = _compute_hedge_overlay(_ranked_book, _all_r_gate, start, end)
                    _hoos = _compute_hedge_oos(_ranked_book, _all_r_gate, start, end)
                except Exception:
                    _fd = _fd
                pdf_bytes = generate_report(
                    start=start,
                    end=end,
                    avg_corr_series=avg_corr,
                    current_regime=current,
                    regimes=regimes,
                    # all_trades=[] so the "other regimes" reference section stays
                    # empty (no 80-card candidate dump - the report is the book).
                    active_trades=_feed,
                    all_trades=[],
                    eq_r=eq_r,
                    cmd_r=cmd_r,
                    stress_series=stress,
                    geopolitical_events=GEOPOLITICAL_EVENTS,
                    factor_decomp=_fd,
                    skill_decomp=_skill,
                    rolling_decomp=_roll,
                    cost_decomp=_cost,
                    hedge_decomp=_hedge,
                    hedge_oos_decomp=_hoos,
                )
            st.session_state["_ti_pdf_bytes"] = pdf_bytes
            st.session_state["_ti_pdf_name"] = (
                f"desk_report_{datetime.date.today().isoformat()}_"
                f"regime_{r_name.lower()}.pdf"
            )
            _pdf_ok = True
        except ImportError:
            st.error(
                "reportlab is required for PDF generation. "
                "Run: `pip install reportlab>=4.2.0`"
            )
        except Exception:
            st.error("Report generation failed.")
        if _pdf_ok:
            st.rerun()   # surface the download at the top-right immediately
    if st.session_state.get("_ti_pdf_bytes"):
        st.download_button(
            label="Download PDF",
            data=st.session_state["_ti_pdf_bytes"],
            file_name=st.session_state.get("_ti_pdf_name", "desk_report.pdf"),
            mime="application/pdf",
            key="download_report",
        )

    # ── Data Integrity Audit ────────────────────────────────────────────────
    with st.expander("Data Integrity Audit - Leg Coverage & Strategy Correlation", expanded=False):
        _DI_M = "font-family:'JetBrains Mono',monospace;"
        st.markdown(
            f'<p style="{_DI_M}font-size:0.60rem;color:#8890a1;margin-bottom:.8rem">'
            'Checks every strategy\'s declared legs against the loaded return data. '
            'Strategies with missing legs are mislabeled - their backtest results are excluded. '
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
                "Present":     ", ".join(_present) if _present else " - ",
                "Dropped":     ", ".join(_dropped) if _dropped else " - ",
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
                f'<td style="{_tdb}font-size:.60rem;color:#c0392b">{_r["Dropped"] or " - "}</td>'
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
            '(OOS walk-forward). Clusters above r ≈ 0.90 are hidden duplicates - '
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
                        f'<b>HIGH-CORRELATION CLUSTERS (r ≥ 0.90) - count as 1 distinct bet each:</b><br>'
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

                # Per-cell text annotations - the only way to get correct contrast
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
            st.caption("avg_corr unavailable - cannot run pairwise correlation.")

    # ── Multiple Testing Report ─────────────────────────────────────────────
    # Reports effective N vs raw N, per-strategy DSR vs HLZ cross-check,
    # and flags disagreements. Opening this panel also updates session_state
    # so subsequent rerenders use the dynamic effective N in card grades.
    with st.expander(
        "Multiple Testing Report - DSR vs HLZ Cross-Check (Effective N)",
        expanded=False,
    ):
        from src.analysis.backtest import (
            compute_effective_n as _compute_eff_n,
            _N_LIBRARY_STRATEGIES as _STATIC_N,
        )
        _MT_M = "font-family:'JetBrains Mono',monospace;"
        st.markdown(
            f'<p style="{_MT_M}font-size:0.60rem;color:#8890a1;margin-bottom:.8rem">'
            'DSR is the single grading gate - it already corrects for N via the expected maximum SR under '
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
                continue   # ungradeable - skip for N computation
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
                    "N used":   " - ",
                    "DSR %":    " - ",
                    "Grade":    " - ",
                    "t-stat":   " - ",
                    "HLZ hurdle": " - ",
                    "HLZ":      "MISSING",
                    "Agree?":   " - ",
                })
                continue
            _base = _mt_results.get(_tr["name"])
            if _base is None:
                continue
            # Re-grade with dynamic N (pure function - cheap)
            from src.analysis.backtest import qc_grade_backtest as _regrade
            _qc = _regrade(_base, n_strategies=_n_used, is_economic_prior=_is_prior)
            _hlz_p = _qc.get("hlz_pass")
            _ag    = _qc.get("hlz_agree_dsr")
            _report_rows.append({
                "Strategy":   _tr["name"][:45],
                "Prior":      "THEORY" if _is_prior else "GRID",
                "N used":     str(_n_used),
                "DSR %":      f'{_qc.get("dsr_prob", 0):.0%}',
                "Grade":      _qc.get("grade", " - "),
                "t-stat":     f'{_qc["hlz_tstat"]:.2f}' if _qc.get("hlz_tstat") is not None else "n/a",
                "HLZ hurdle": f'{_qc.get("hlz_threshold", 0):.2f}',
                "HLZ":        ("PASS" if _hlz_p is True else "FAIL" if _hlz_p is False else "n/a"),
                "Agree?":     ("✓" if _ag is True else "⚠ REVIEW" if _ag is False else " - "),
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
                _ag = _r.get("Agree?", " - ")
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
                    f'{_r.get(_c, " - ")}</td>'
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
                f'HLZ cross-check only - DSR is the binding grade criterion'
                f'</p>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No gradeable strategies to report.")

    # ── Thesis Pipeline ─────────────────────────────────────────────────────
    # Auto-expand when we have results (from disk cache or a prior run this session).
    _tp_has_results = bool(st.session_state.get(_PV_SESSION_KEY))
    with st.expander(
        "Thesis Pipeline - 5-Stage Economic Mechanism Validation",
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
            'decision rule - does its admit/reject classification predict out-of-sample '
            'returns? Individual trade P&amp;L is not the output.</p>',
            unsafe_allow_html=True,
        )

        _stage_defs = [
            ("#CFB991", "STAGE 1 - THESIS",
             "Researcher constructs shock → TPS channel → predicted sign → holding horizon from first "
             "principles. No optimisation. The economic narrative must be stated before any data is viewed. "
             "Gate: shock is named, ≥1 TPS channel specified, predicted direction signed per leg, horizon set."),
            ("#2980b9", "STAGE 2 - SIGNAL",
             "Every declared leg must be present in the loaded return data. Any missing leg "
             "is a hard stop - the thesis cannot be tested and the researcher must revise the leg specification. "
             "Gate: all assets in return index. Fail-loud: never silently drop a leg."),
            ("#27ae60", "STAGE 3 - PRIOR-ALIGNED CONFIRMATION",
             "LP-IRF (local projection) or regime-conditional returns confirm that the data's sign "
             "matches the predicted sign from Stage 1 at the stated horizon. "
             "Outcomes: CONFIRM (sign + significance), IE (sign matched but n &lt; 20 - insufficient evidence, "
             "not a rejection), REJECT (sign contradicted). "
             "Gate: sign matched AND significant at 10%."),
            ("#e67e22", "STAGE 4 - SIZING",
             "Vol-scaled allocation: target 10% annual vol ÷ estimated strategy vol → base weight. "
             "IRF scale factor applied (larger coef at horizon → larger weight). "
             "Capped at 20% per conflict source. Gate: sizing computed; output is final weight %."),
            ("#8e44ad", "STAGE 5 - GRADE",
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
                f'<span style="font-size:0.56rem;color:#8890a1"> - Effective N for DSR correction defaults to 9 until the MT Report '
                f'computes the true value from your backtest results. Validation run before then may apply the wrong correction.</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Staleness (_pv_stale / _pv_age_lbl) is computed once at the top of the
        # page; the top banner's "Rerun Validation" sets _ti_force_pv, honoured here.
        _pv_col1, _pv_col2 = st.columns([1, 4])
        with _pv_col1:
            _run_pv = st.button(
                ("⚠ Rerun - Book Stale" if _pv_stale else "Refresh Validation"),
                key="run_pipeline_val", type="primary",
                help="Re-runs walk-forward validation (~2-4 min). Saves result to disk for next session.",
            ) or st.session_state.pop("_ti_force_pv", False)
        with _pv_col2:
            if _run_pv:
                st.markdown(
                    f'<span style="{_TP_M}font-size:0.57rem;color:#555960">'
                    f'Running walk-forward validation (~2-4 min)…</span>',
                    unsafe_allow_html=True,
                )
            elif _pv_stale:
                # Hard staleness flag: book past the threshold - loud, not a note.
                st.markdown(
                    f'<div style="{_TP_M}border:1px solid #c0392b;border-radius:4px;'
                    f'padding:6px 12px;background:#1a0808;display:inline-block">'
                    f'<span style="font-size:0.6rem;color:#e05241;font-weight:700;'
                    f'letter-spacing:.08em">⚠ STALE BOOK</span>'
                    f'<span style="font-size:0.57rem;color:#c98b86"> - validated '
                    f'{_pv_age_lbl} (&gt; {_PV_STALE_HOURS}h). Rerun before trusting '
                    f'these weights.</span></div>',
                    unsafe_allow_html=True,
                )
            elif _pv_age_lbl:
                st.markdown(
                    f'<span style="{_TP_M}font-size:0.57rem;color:#8890a1">'
                    f'Validated {_pv_age_lbl}. Click Refresh to recompute.</span>',
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
    <div class="_pv_msg">Fitting LP-IRF on training window - past data only&hellip;</div>
    <div class="_pv_msg">Stage 3 confirming sign direction per leg&hellip;</div>
    <div class="_pv_msg">Running DSR gate - deflating Sharpe by trial count&hellip;</div>
    <div class="_pv_msg">Computing OOS returns in test window&hellip;</div>
    <div class="_pv_msg">Monte Carlo random-admission baseline (500 draws)&hellip;</div>
  </div>
  <div class="_pv_bar_track"><div class="_pv_bar_fill"></div></div>
</div>
""", unsafe_allow_html=True)
            _pv_ok = False
            try:
                _pv = _run_pipeline_validator_cached(
                    all_r_concat, regimes,
                    train_days=756, test_days=63,
                    n_strategies=_effective_n,
                    n_random_trials=500,
                )
                st.session_state[_pv_key] = _pv
                # Mark fresh: reset the staleness clock (survives reruns).
                from datetime import datetime as _dt2, timezone as _tz2
                st.session_state[_PV_SAVED_KEY] = _dt2.now(_tz2.utc)
                # Persist to disk so the next session loads instantly
                try:
                    from src.utils.page_cache import save_cache as _sv
                    _sv(_PV_DISK_KEY, _pv)
                    _pv_disk_age = None  # now fresh
                except Exception:
                    pass
                _pv_ok = True
            except Exception as _pv_exc:
                import traceback as _tb
                _tb.print_exc()
                st.error(f"Validation error: {type(_pv_exc).__name__}: {_pv_exc}")
                _pv = None
            finally:
                _anim.empty()
            # The STALE banner at the top of the page was computed BEFORE this run
            # reset the clock, so on this pass it still reads stale. Re-render once
            # so the top banner + button recompute against the now-fresh timestamp.
            # Cheap: the validator is cached, so this rerun does NOT recompute it.
            if _pv_ok:
                st.rerun()
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
                ("GAP 1 - Admitted vs Rejected",
                 _g1, _g1p,
                 "Mean OOS return of admitted theses minus rejected theses (%). "
                 "Must be positive: the gates must discriminate."),
                ("GAP 2 - Admitted vs Random",
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
                f'margin:1rem 0 .4rem">BUCKET BEHAVIOR - OOS MEAN RETURN (%)</p>',
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
                    _bmean_str = (f'{_bmean:+.2f}%' if _bmean is not None else ' - ')
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

        # ── PART 3: Worked Example - Pipeline decision trace for one thesis ───
        st.markdown(
            f'<p style="{_TP_M}font-size:0.58rem;font-weight:700;letter-spacing:.10em;'
            f'color:#CFB991;margin-bottom:.3rem">WORKED EXAMPLE - PIPELINE DECISION TRACE</p>'
            f'<p style="{_TP_S}font-size:0.65rem;color:#8890a1;margin-bottom:.6rem;'
            f'line-height:1.6">'
            f'Thesis: Iran conflict → Strait of Hormuz → WTI crude supply shock → S&amp;P 500 margin '
            f'compression. Left: the mechanism narrative and Stage 3 confirmation result (how the pipeline '
            f'reads the data). Right: the pipeline\'s admit/reject/MT/IE decision at each walk-forward window '
            f'and the OOS return that followed - this is approach-testing, not trade performance.</p>',
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
                       "short horizons - consumption is inelastic for 2-4 weeks. Equity margin "
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
                f'color:#CFB991;margin-bottom:6px">STAGE 1 - THESIS ✓</div>'
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
                f'STAGE 2 - SIGNAL {"✓" if _we_s2_ok else "✗"}</div>'
                f'<div style="{_TP_S}font-size:0.68rem;color:#c8c8c8">{_we_leg_str}</div>'
                f'<div style="{_TP_M}font-size:0.58rem;color:{_we_s2_color};margin-top:4px">'
                f'{"All legs present in return data." if _we_s2_ok else "Missing legs - thesis untestable."}'
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
                _we_track   = _we_s3d.get("track", " - ")
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
                       f'<td style="color:#8890a1">{_v.get("mean_ret"," - ")}%</td>')
                    + '</tr>'
                    for _a, _v in _we_per_leg.items()
                )
                if _we_s3_ok:
                    _we_s3_verdict = f"CONFIRMED ({_we_score:.0%} legs)"
                elif _we_s3_sign:
                    _we_s3_verdict = f"SIGN MATCHED, NOT SIGNIFICANT - IE if n &lt; 20"
                else:
                    _we_s3_verdict = "REJECTED - predicted sign not matched"
                st.markdown(
                    f'<div style="background:#0a0a0a;border:1px solid #1e1e1e;'
                    f'border-left:3px solid {_we_s3_color};border-radius:4px;'
                    f'padding:.6rem 1rem;margin-bottom:.5rem">'
                    f'<div style="{_TP_M}font-size:0.56rem;letter-spacing:.12em;'
                    f'color:{_we_s3_color};margin-bottom:5px">'
                    f'STAGE 3 - CONFIRMATION ({_we_track.upper()}) · {_we_s3_verdict}</div>'
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
                _we_track = " - "
                st.markdown(
                    f'<div style="{_TP_M}font-size:0.60rem;color:#c0392b;padding:.4rem">'
                    f'Stage 3 skipped - Stage 2 failed.</div>',
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

            # Decision trace - pipeline's classification of this thesis at each window
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
                    title="Pipeline Decision Trace - WTI / S&P 500 across Walk-Forward Windows",
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
                        + f'<span style="color:#555960;font-size:0.55rem"> - mean OOS return by pipeline decision</span>'
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

        # Individual strategy cards removed - pipeline is the deliverable, not per-thesis P&L.
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
