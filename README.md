# Equity-Commodities Spillover Monitor

A multi-page analytical dashboard for tracking cross-asset spillover dynamics between global equity markets, commodity futures, fixed income, and FX — with geopolitical risk intelligence, forward-looking scenario analysis, and a collaborative AI agent workforce.

Built for **MGMT 69000-120: AI for Finance** · Purdue University Daniels School of Business

---

## Overview

The Spillover Monitor brings together 15 global equity indices, 17 commodity futures, fixed income ETFs, and FX pairs into a single institutional-grade workspace. It tracks how stress transmits across asset classes — not just what is happening, but *why*, *where it came from*, and *what comes next*.

An orchestrated pipeline of 7 AI agents runs in dependency order on every page load, synthesising live market data into morning briefings, trade ideas, and stress assessments — without requiring any manual input.

---

## Coverage

### Equity Markets — 15 Indices, 5 Regions
| Region | Indices |
|--------|---------|
| **USA** | S&P 500, Nasdaq 100, DJIA, Russell 2000 |
| **Europe** | Eurostoxx 50, DAX, CAC 40, FTSE 100 |
| **Japan** | Nikkei 225, TOPIX |
| **China** | Hang Seng, Shanghai Composite, CSI 300 |
| **India** | Sensex, Nifty 50 |

### Commodities — 17 Futures, 4 Groups
| Group | Contracts |
|-------|-----------|
| **Energy** | WTI Crude, Brent Crude, Natural Gas, Gasoline (RBOB), Heating Oil |
| **Precious Metals** | Gold, Silver, Platinum |
| **Industrial Metals** | Copper, Aluminum, Nickel |
| **Agriculture** | Wheat, Corn, Soybeans, Sugar #11, Coffee, Cotton |

### Fixed Income & FX
| Asset Class | Instruments |
|-------------|-------------|
| **Fixed Income** | US 20Y+ Treasury (TLT), IG Corporate (LQD), HY Corporate (HYG), EM USD Bonds (EMB), Short Treasury (SHY), TIPS (TIP) |
| **FX** | DXY Dollar Index, EUR/USD, GBP/USD, USD/JPY, USD/CNY, USD/INR |

### Implied Volatility
| Index | Description |
|-------|-------------|
| **VIX** | CBOE Equity Volatility Index |
| **OVX** | CBOE Oil Volatility Index |
| **GVZ** | CBOE Gold Volatility Index |
| **VVIX** | Volatility of VIX |

### Geopolitical Events — 13 Tracked Shocks (2008–present)
| ID | Event | Period |
|----|-------|--------|
| GFC | Global Financial Crisis | Sep 2008 – Mar 2009 |
| Arab Spring | Arab Spring & Libya Oil Shock | Jan – Oct 2011 |
| Trade War | US-China Trade War | Mar 2018 – Jan 2020 |
| Aramco | Abqaiq-Khurais Drone Attack | Sep – Oct 2019 |
| COVID-19 | COVID-19 Pandemic | Feb 2020 – Nov 2021 |
| WTI Neg | WTI Goes Negative | Apr – May 2020 |
| Ukraine | Russia-Ukraine War | Feb 2022 – present |
| Nickel | LME Nickel Short Squeeze | Mar 7–25, 2022 |
| Fed Hike | Fed Rate Hiking Cycle | Mar 2022 – Jul 2023 |
| SVB | SVB / Banking Crisis | Mar – May 2023 |
| Hamas | Israel-Hamas War & Iran Escalation | Oct 2023 – present |
| India-Pakistan | India-Pakistan Military Escalation | May 2025 – present |
| Hormuz | Iran Military Conflict & Strait of Hormuz Crisis | Jun 2025 – present |

---

## Key Features

- **Real-time correlation regimes** — four-state classification (Decorrelated → Normal → Elevated → Crisis) with Markov transition forecasts and mean days to next Crisis
- **Early Warning System** — five-component composite score (0–100) flagging conditions that historically precede regime transitions
- **Forward-looking Scenario Engine** — parametric shock simulator for oil, gold, yields, DXY, credit spreads, and geopolitical disruption; propagates shocks via OLS betas with VaR 95/99 and ES 95/99
- **Directional spillover network** — Granger causality, transfer entropy, and Diebold-Yilmaz FEVD identify who leads and who follows across all assets
- **War Impact Map** — choropleth scoring 195+ countries by equity-market exposure across three simultaneous active conflict theaters
- **Strait Watch** — live disruption monitoring for six critical maritime chokepoints (Hormuz, Suez, Panama, Bosporus, Malacca, Taiwan Strait)
- **Portfolio stress tester** — custom multi-asset allocation (indices, commodities, fixed income, individual stocks) tested against all 13 tracked events
- **Implied vol layer** — OVX, GVZ, VIX, VVIX integrated into agent context and displayed as live and historical panels
- **Automated geo-intelligence** — RSS ingestion from 7 sources (Reuters, BBC, NYT, WSJ, FT, AP) with keyword-tagged severity scoring; high-confidence events auto-routed to AI review queue
- **AI Workforce** — 7 agents in a dependency-ordered pipeline (see below); outputs feed forward across rounds
- **AI Analyst chatbot** — Claude/GPT-4o with full live dashboard context injected into every query

---

## AI Workforce

Seven agents run in three sequential rounds on every Overview page load. Each agent receives the structured outputs of its upstream peers before generating its own analysis.

| Round | Agent | Role |
|-------|-------|------|
| 1 | **Signal Auditor** | Calibrates confidence scores across all agents from Granger hit rates |
| 1 | **Macro Strategist** | Yield curve, inflation, Fed posture, GDP context |
| 1 | **Geopolitical Analyst** | Active conflict risk, sanctions, strait disruption |
| 2 | **Risk Officer** | Synthesises all Round 1 outputs into a morning briefing |
| 2 | **Commodities Specialist** | COT positioning, supply shocks, sector rotation |
| 3 | **Stress Engineer** | Scenario stress, tail risk, portfolio shock |
| 3 | **Trade Structurer** | Regime-triggered trade ideas with full peer context |

A Chief Quality Officer (CQO) runs separately on each page to audit data integrity and flag assumption violations.

Divergence detection flags disagreements between peer agents (e.g., Macro Strategist bullish on rates while Geopolitical Analyst flags energy supply disruption). All outputs are cached for 1 hour; stale agents are automatically invalidated on regime change.

---

## Page-by-Page Manual

### Overview
**Starting point for every session.**

- **KPI strip** — regime, 60d avg correlation (with 1M delta), best/worst equity, best/worst commodity
- **FI/FX context strip** — TLT 30d return, HYG vs LQD credit spread, DXY trend, equity-bond divergence, EM credit signal
- **Implied vol panel** — live OVX, GVZ, VIX, VVIX badges with freshness indicators; expandable historical chart
- **Risk Score** — composite 0–100 gauge with dynamic component weights
- **Early Warning System** — five signal cards plus composite score and historical analogues table
- **AI Workforce output** — Risk Officer morning briefing displayed inline; all other agents in collapsible expander; divergence flags highlighted
- **Activity feed** — real-time log of all agent actions, routing decisions, and escalations

---

### Scenario Engine
**Forward-looking parametric shock simulator.**

Six shock inputs: oil price change (%), gold price change (%), yield shift (bps), DXY change (%), credit spread change (bps), geopolitical disruption score (0–10).

- **Preset scenarios** — Oil Supply Shock, Safe Haven Flight, Fed Policy Error, Geopolitical Escalation, Stagflation, Risk-On Rally; one-click load
- **Propagation** — OLS betas map commodity shocks to individual equity indices; fixed sensitivity tables for yield, DXY, credit, and geo channels
- **Waterfall chart** — per-asset impact breakdown with channel attribution
- **VaR/ES table** — historical simulation at 95% and 99% confidence levels
- **AI Stress Engineer** — narrative assessment of the scenario's tail risk implications

---

### War Impact Map
**Global choropleth showing equity-market exposure to active conflicts.**

- Filter between Combined, Ukraine, Israel-Hamas, and Iran/Hormuz views
- Top 25 exposed markets ranked table with per-conflict score columns
- Tracked equity performance since conflict onset

---

### Geopolitical Triggers
**Event-window forensics for every tracked shock since 2008.**

- Normalized price performance with pre/during/post windows
- Volatility shift (annualized realized vol before vs. after)
- Correlation regime change at event onset, peak stress, and 90-day recovery
- Live Intelligence Feed — automated RSS headlines tagged by region, commodity, and severity; high-confidence events routed to pending review

---

### Correlation Analysis
Four tabs: Rolling Correlation, DCC-GARCH, Regime Detection, Regime Forecast. Full fixed income and FX pair coverage included.

**Regime Forecast** outputs Markov transition matrix, steady-state distribution, mean days to Crisis from each state, and run-length statistics.

---

### Spillover Analytics
- Granger causality grid (pairwise, 5% significance)
- Transfer entropy matrix (nonlinear information flow)
- Diebold-Yilmaz FEVD connectedness index (0–100%)
- Network graph (node size = net spillover power)
- Cross-asset section covering rates, FX, and fixed income channels

---

### Macro Intelligence Dashboard
**FRED-powered macro context with AI Macro Strategist.**

- Yield curve (10Y–2Y spread), CPI YoY, Fed Funds Rate, GDP growth, ISM PMI
- Historical series charts with recession shading
- Yield curve shape analysis and inversion tracker
- AI Macro Strategist briefing with peer context from Geopolitical Analyst and Risk Officer

---

### Commodities to Watch
**Live watchlist with COT positioning.**

- Live snapshot: price, 1d/5d return, annualized vol, regime label
- Intraday hourly view and daily chart with volatility regime bands
- COT overlay: net speculative positioning vs. commercial hedgers
- Rolling 60-day correlation vs. all equity regions

---

### Strait Watch
**Maritime chokepoint disruption monitor.**

Six straits tracked: Hormuz, Suez, Taiwan Strait, Bosporus, Malacca, Panama Canal. Each shows disruption score (0–100), vessel traffic change vs. baseline, commodity channels at risk, and estimated daily trade value at risk.

---

### Trade Ideas
**Regime-triggered cross-asset trade cards with AI structuring.**

Cards filtered by current live regime. Each includes directional thesis, entry signal, risk consideration, and regime trigger. AI Trade Structurer generates new ideas incorporating full peer agent context from macro, geo, stress, and commodities agents.

---

### Portfolio Stress Test
**Build any portfolio, test it against every tracked shock.**

- Preset allocations or custom: any mix of indices, commodity futures, fixed income ETFs, and individual S&P 500 stocks
- Per-event results: Pre/During/Post return, Max Drawdown, Sharpe Ratio
- Event Returns Heatmap, Max Drawdown bar chart, and normalized portfolio path chart

---

### Performance Review
**Signal backtesting and model validation.**

- Granger causality hit rates (directional accuracy per pair)
- Regime detection F1 score and confusion matrix
- VIX-correlation R² and model accuracy
- COT contrarian signal performance
- AI Signal Auditor calibration output

---

### Insights
**Macro research synthesis and private credit risk monitor.**

- Private credit bubble risk score with HY OAS, BKLN, BDC basket, CDX HY signals
- India macro framework: crude oil dependency, rupee transmission, Nifty 50 drivers
- Cross-asset spillover research notes

---

### AI Analyst
**Claude / GPT-4o chatbot with live dashboard context.**

Full market state injected into every query: regime, risk score, equity/commodity/FI/FX returns, implied vol levels, active geopolitical events, and AI agent outputs.

**Agent Activity Log** available at the bottom of this page — structured feed of every agent action, routing decision, and escalation from the current session.

Supports both Anthropic (Claude Sonnet) and OpenAI (GPT-4o). Configure via `.streamlit/secrets.toml` or Render Secret Files; Anthropic is preferred when both keys are present.

---

## Navigation

```
Overview
├── Macro
│   └── Macro Intelligence Dashboard
├── Analysis
│   ├── War Impact Map
│   ├── Geopolitical Triggers
│   ├── Correlation Analysis
│   └── Spillover Analytics
├── Strategy
│   ├── Trade Ideas
│   ├── Portfolio Stress Test
│   └── Scenario Engine
├── Monitor
│   ├── Commodities to Watch
│   └── Strait Watch
├── Research
│   ├── Performance Review
│   └── AI Analyst
├── Insights
└── About
    ├── Heramb S. Patkar
    ├── Jiahe Miao
    └── Ilian Zalomai
```

---

## Configuration

### API Keys

Add to `.streamlit/secrets.toml` (local) or as a **Secret File** on Render (deployed):

```toml
[keys]
fred_api_key      = "..."   # FRED macroeconomic data (optional but recommended)
anthropic_api_key = "..."   # Claude Sonnet — AI agents + analyst (preferred)
openai_api_key    = "..."   # GPT-4o fallback

[auth]
password = "..."            # Optional — set enable_auth = true in [config] to activate

[config]
enable_auth      = false    # Set true to add a password gate on app load
enable_rss       = true     # Automated geopolitical RSS ingestion
refresh_interval = 300      # Overview auto-refresh interval in seconds
```

Anthropic is used when both keys are present. FRED unlocks yield curve, CPI, Fed Funds Rate, GDP, and ISM data on the Macro Dashboard.

### Local Setup

```bash
git clone https://github.com/HPATKAR/equity-commodities-spillover.git
cd equity-commodities-spillover
pip install -r requirements.txt
streamlit run app.py
```

---

## Stack

| Layer | Libraries |
|-------|-----------|
| **UI** | Streamlit, Plotly, custom HTML/CSS |
| **Data** | yfinance, fredapi, feedparser |
| **Analytics** | pandas, numpy, statsmodels, arch, scipy, scikit-learn, networkx |
| **AI** | anthropic, openai |
| **Reports** | reportlab, Pillow |

---

## Disclaimers

This dashboard is built for **educational and research purposes only**. Nothing on this platform constitutes investment advice. All analytics are based on historical data and statistical models — past relationships do not guarantee future behavior. Market data is sourced from Yahoo Finance and FRED and may have delays or gaps. AI agent outputs are generated by third-party language models and should not be treated as financial guidance.

---

*Equity-Commodities Spillover Monitor — Purdue University | MGMT 69000-120: AI for Finance*
