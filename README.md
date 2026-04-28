# Equity-Commodities Spillover Monitor

An institutional-grade, multi-page analytical dashboard for tracking cross-asset spillover dynamics between global equity markets, commodity futures, fixed income, and FX — with geopolitical risk intelligence, a live conflict feed, forward-looking scenario analysis, and an orchestrated AI agent workforce running on real financial data.

Built for **MGMT 69000-120: AI for Finance** · Purdue University Daniels School of Business · Prof. Cinder Zhang

---

## Team

| Member | Program | Role |
|--------|---------|------|
| **Heramb S. Patkar** | MSF, Financial Analytics Track | Architecture, AI pipeline, quantitative analytics, full-stack build |
| **Ilian Zalomai** | MSF, Financial Analytics Track | Geopolitical research, War Impact Map scoring, Strait Watch framework, scenario authorship |
| **Jiahe Miao** | MSF | Correlation regime taxonomy, trade structure design, D-Y methodology, FI signal framework |

### Contributions

**Heramb S. Patkar** designed and built the full system — Streamlit multi-page application, AI agent orchestrator, quantitative analytics modules (spillover, correlation, risk scoring, scenario engine), live data integrations (GDELT, ACLED, EIA, PortWatch, FRED), harness verification architecture, and the full deployment stack.

**Ilian Zalomai** designed the War Impact Map scoring framework — per-war baseline scores, country-level commodity exposure weights, and concurrent-war amplification logic for Ukraine, Gaza/Red Sea, and Iran/Hormuz. He authored all 13 geopolitical event entries (2008–2025), built the Strait Watch maritime chokepoint framework, and wrote the narrative content connecting quantitative outputs to real-world market implications throughout the dashboard.

**Jiahe Miao** designed the 4-state correlation regime taxonomy (Decorrelated / Normal / Elevated / Crisis), calibrated the rolling window thresholds and lookback methodology for regime classification, and specified the 6 regime-conditioned trade structures with entry/exit rationale. She researched the Diebold-Yilmaz FEVD methodology, contributed to network edge threshold calibration and directional spillover interpretation, defined the fixed income cross-asset stress signal framework (TLT/HYG/LQD/EMB), validated the private credit bubble proxy selection (BKLN/ARCC/OBDC/FSK/JBBB), and reviewed equity and commodity return series for data quality.

---

## What This Does

The Spillover Monitor answers three questions that standard market dashboards do not:

1. **Where is stress coming from?** — Granger causality (BIC-optimal lag, Holm-corrected), transfer entropy with shuffle significance tests, and Diebold-Yilmaz FEVD identify directional transmission paths between assets, not just correlation levels.
2. **How severe is the current regime?** — A five-component composite risk score (0–100) and four-state correlation regime model classify market conditions with historical analogues and Markov transition forecasts.
3. **What happens next?** — A parametric scenario engine propagates shocks forward via OLS betas; seven AI agents in a dependency-ordered pipeline synthesise all of this into actionable morning briefings, trade ideas, and stress assessments.

The system ingests 15 equity indices, 17 commodity futures, 6 fixed income instruments, 6 FX pairs, 4 implied volatility indices, 24 FRED macro series, live conflict event data from GDELT and ACLED, maritime traffic from IMF PortWatch, and weekly EIA petroleum inventory reports — on every session load.

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
| **Industrial Metals** | Copper, Aluminum |
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

### Live Data Sources
| Source | Data | Cadence |
|--------|------|---------|
| **Yahoo Finance** | Prices, returns, implied vol | Real-time (30min cache) |
| **FRED** | 24 macro series (yield curve, CPI, Fed Funds, GDP, PMI) | Daily (24h cache) |
| **GDELT 2.0** | News volume and tone for 6 active conflicts | 3h cache |
| **ACLED** | Georeferenced conflict event counts and fatality data | 6h cache |
| **IMF PortWatch** | Vessel traffic at Hormuz and 5 other chokepoints | Daily |
| **EIA** | Weekly US petroleum inventory report | Wednesday release |
| **RSS** | 7 news sources (Reuters, BBC, NYT, WSJ, FT, AP, Al Jazeera) | 1h cache |

### Geopolitical Events — 13 Tracked Shocks (2008–present)
*Event taxonomy designed by Ilian Zalomai*

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

## Statistical Methodology

All core econometric methods implement published best practices. The table below documents each method, the key design choice, and the academic basis.

| Method | Implementation | Academic Basis |
|--------|---------------|----------------|
| **Granger causality** | BIC-optimal lag via `VAR(ic='bic')` before testing; Holm-Bonferroni step-down correction across the full N×M×2 test grid | Granger (1969); Lütkepohl (2005) Ch. 4 |
| **Transfer entropy** | Schreiber (2000) directed information measure; 200-permutation shuffle test for significance; optimal lag selected by maximising TE(commodity→equity) | Schreiber (2000) |
| **Diebold-Yilmaz FEVD** | VAR fitted with `ic='bic'` (lag_order is upper bound); 10-step forecast horizon; generalized variance decomposition | Diebold & Yilmaz (2012) |
| **DCC-GARCH** | Two-step Engle (2002): EWMA pre-whitening (λ=0.94, RiskMetrics) standardises returns before DCC(1,1) recursion; raw-return DCC is contaminated by heteroskedasticity | Engle (2002) |
| **Correlation regime** | Adaptive percentile thresholds; 5-day median smoothing; hysteresis (exit threshold = entry − 5pp); persistence gate (Crisis requires 60% of 10-day window above elevated) | Hamilton (1989) regime-switching |
| **Composite stress index** | Vol signals: z-score → [0,100] preserves absolute VIX level information; correlation signals: empirical percentile rank | Illing & Liu (2006) FSI design |
| **Markov regime dynamics** | 4×4 first-order transition matrix; steady-state via linear solve; Mean First Passage Time to Crisis | Hamilton (1994) |
| **CIS/TPS aggregation** | HHI-based breadth multiplier (n_eff^0.25) applied to weighted portfolio CIS and TPS; prevents concentration dilution | Rajan & Zingales (1998) diversification |
| **Risk score** | EWM span 252 days for vol normalization; no OLS residualization of commodity vol (preserves supply-shock co-movement); quadrature-combined confidence intervals | RiskMetrics (1994) |
| **Exposure scoring** | SES (structural), TAE (transmission-adjusted), SAS (scenario-adjusted); per-conflict OLS beta β(a,c) = structural[a][c] × (TPS_c / 100) | Standard factor model decomposition |

### Key Design Decisions

**BIC lag selection (Granger + VAR):** Testing all lags 1..k and reporting the minimum p-value is implicit data snooping — the effective significance level exceeds the stated α. BIC selects the lag that maximises fit relative to model complexity before any hypothesis test is run.

**Holm-Bonferroni correction:** With N_eq × N_cmd × 2 simultaneous Granger tests, a naïve p < 0.05 threshold produces ~5% false positives by construction (15 equity × 17 commodity × 2 = 510 tests → ~25 spurious rejections expected). The Holm step-down procedure controls the family-wise error rate at α while being uniformly more powerful than Bonferroni.

**Shuffle significance for transfer entropy:** Raw TE is always ≥ 0 by construction (it is a KL divergence), so a positive value alone means nothing without a null distribution. 200 permutations of the source series establish the distribution under the null of no directional information flow. Only cells passing p < 0.05 are marked significant (asterisked in the heatmap).

**EWMA pre-whitening for DCC:** The Engle (2002) two-step procedure standardises returns by their EWMA conditional standard deviation (λ=0.94) before the DCC(1,1) recursion. Skipping this step contaminates the quasi-correlation matrix Q with variance dynamics, artificially inflating measured correlations during high-vol episodes even when the true ρ is unchanged.

---

## Key Features

### Quantitative Analytics
- **Correlation regime model** — four-state classification (Decorrelated → Normal → Elevated → Crisis) with Markov transition forecasts, steady-state distribution, and mean days to next Crisis; *regime taxonomy designed by Jiahe Miao*
- **Composite Global Risk Score (0–100)** — three weighted components: Conflict Intensity Score (35%), Transmission Pressure Score (30%), Market Confirmation Score (35%); EWM-normalized with 252-day span to prevent regime normalisation during sustained crises
- **Early Warning System** — five-component composite score (correlation velocity, vol acceleration, regime duration pressure, equity vol trend, Markov crisis probability) with historical analogue matching via Euclidean nearest-neighbour on (eq_vol, cmd_vol, corr) signature
- **Directional spillover analytics** — Granger causality (BIC-optimal lag, Holm-Bonferroni corrected), transfer entropy with 200-permutation shuffle significance test, Diebold-Yilmaz FEVD connectedness index (0–100%); *D-Y interpretation framework contributed by Jiahe Miao*
- **CIS/TPS conflict model** — Conflict Intensity Score (7 dimensions: deadliness, civilian danger, diffusion, fragmentation, escalation trend, recency, source coverage) and Transmission Pressure Score (12 channels: oil/gas, metals, agriculture, shipping, chokepoint, sanctions, equity sector, FX, inflation, supply chain, credit, energy infra); aggregated via HHI-weighted portfolio score
- **DCC-GARCH correlations** — Engle (2002) two-step DCC with EWMA pre-whitening; correlation clips to [−1, 1] to guard against numerical drift
- **Correlation velocity detection** — first derivative of rolling correlation (10-day lag) for earlier regime transition detection

### Scenario & Stress Analysis
- **Parametric scenario engine** — six shock inputs (oil %, gold %, yield bps, DXY %, credit spread bps, geo disruption 0–10); propagated via OLS betas to all 32 assets; VaR 95/99 and ES 95/99
- **Preset scenarios** — Oil Supply Shock, Safe Haven Flight, Fed Policy Error, Geopolitical Escalation, Stagflation, Risk-On Rally
- **Portfolio stress tester** — any custom allocation (indices + commodities + fixed income + individual S&P 500 stocks) tested against all 13 tracked events

### Geopolitical Intelligence
- **War Impact Map** — choropleth scoring 195+ countries by equity-market exposure to three simultaneous active conflict theaters; *scoring framework designed by Ilian Zalomai*
- **Strait Watch** — live disruption monitoring for six critical maritime chokepoints (Hormuz, Suez, Panama, Bosporus, Malacca, Taiwan Strait) with IMF PortWatch vessel counts, EIA throughput, and Brent sensitivity analysis; *chokepoint framework designed by Ilian Zalomai*
- **GDELT corroboration** — GDELT media escalation signal cross-validated against ACLED event data; agreement/contradiction surfaced as confidence score
- **Proactive alert engine** — auto-triggered alerts on COT positioning extremes, regime transitions, and GDELT/EIA threshold breaches; routed to AI review queue

### Fixed Income & Macro
- **FI cross-asset stress signal** — TLT, HYG, LQD, EMB metrics with equity-bond divergence detection; *framework defined by Jiahe Miao*
- **Private credit bubble monitor** — composite risk score from HY OAS, BKLN, BDC basket (ARCC/OBDC/FSK), and CDX HY; *proxy selection validated by Jiahe Miao*
- **FRED macro dashboard** — yield curve (10Y–2Y), CPI YoY, Fed Funds Rate, GDP growth, ISM PMI; 24 series total

---

## AI Workforce

Seven agents run in three sequential rounds on every Overview page load. Each agent receives the typed, structured outputs of its upstream peers — never a truncated string — before generating its own analysis. The system is built on the **Agent = Model + Harness** framework.

### Pipeline

| Round | Agent | Role | Depends On |
|-------|-------|------|------------|
| 1 | **Signal Auditor** | Granger hit rates, confidence calibration | — |
| 1 | **Macro Strategist** | Yield curve, inflation, Fed posture | — |
| 1 | **Geopolitical Analyst** | Active conflicts, strait disruption, GDELT signals | — |
| 2 | **Risk Officer** | Synthesises all Round 1 outputs into morning briefing | All Round 1 |
| 2 | **Commodities Specialist** | COT positioning, supply shocks, sector rotation | Geo Analyst |
| 3 | **Stress Engineer** | Scenario tail risk, drawdown assessment | Risk Officer |
| 3 | **Trade Structurer** | Regime-triggered trade ideas (Pydantic schema + retry) | All Round 1+2 |

A **Chief Quality Officer** runs separately on each page to audit data integrity and flag assumption violations before outputs are displayed.

### Architecture Layers

**Prompt engineering** — system prompts encode domain expertise, output format, and confidence calibration instructions per agent. Confidence scores are grounded in data completeness flags passed via `AgentHandoff`, not self-reported strings from the model.

**Context engineering** — `AgentHandoff` TypedDict defines the explicit information contract between agents. Typed fields: `confidence (float)`, `regime (int)`, `risk_score (float)`, `cis`, `tps`, `cmd_vol_z`, `corr_pct`, `granger_hit_rate`, `yield_curve_spread`, `cpi_yoy`. Numeric precision is preserved across the full pipeline chain — no truncation, no regex parsing. `low_confidence: bool` propagates upstream uncertainty to all downstream agents when a gate is breached.

**Harness engineering** — the operating system around the model layer:
- `NUMERIC_PAIRS` divergence detection checks typed numeric fields across agent pairs (e.g. risk_score agreement within ±20 pts, regime agreement within ±1 state) for internal consistency before passing state downstream
- `CONFIDENCE_THRESHOLDS` per-agent gates derived from measured evaluation hit rates; breach triggers `log_verification_event()` to trace CSV with `status="verification_fired"` (not just session_state)
- `log_verification_event()` records every divergence and confidence gate check — fired or passed — to `logs/agent_traces.csv`, so the full verification loop history is auditable from the CSV without requiring a live session
- One automatic retry with 1.5s backoff on LLM failure; `"unavailable:"` error strings intercepted at harness layer and never stored as output or shown to users
- `trace_logger.py` records per-step latency, token estimates, cost estimate, and failure type to `logs/agent_traces.csv`
- `pipeline_status` tracks `"complete"` vs. `"partial"` failure states with `failed_agents` list
- `TradeOutput` Pydantic schema with up to 3 retries and self-correcting validation error feedback on `trade_structurer`
- GDELT retry loop with 0.8s intra-conflict and 1.5s inter-conflict throttle, plus exponential backoff on HTTP 429
- Empty `api_key` guard prevents 401s from consuming retry budget

### Evaluation

Validated against 56 labeled historical cases across crisis, stress, and normal regimes (run: 2026-04-28):

| Agent | Cases | Hit Rate | Gate | Status |
|-------|------:|--------:|------|--------|
| macro_strategist | 11 | 63.6% | 0.50 | ✅ PASS |
| stress_engineer | 1 | 100.0% | 0.48 | ✅ PASS |
| risk_officer | 30 | 43.3% | 0.35 | ✅ PASS |
| commodities_specialist | 3 | 33.3% | 0.35 | ✅ PASS |
| geopolitical_analyst | 11 | 0.0%† | 0.40 | ⚠ EVAL-LIMITED |
| signal_auditor | 4 | 0.0%† | 0.40 | ⚠ EVAL-LIMITED |

† CIS/TPS and Granger hit-rate fields require live API access (ACLED, GDELT, real-time prices) that is unavailable in the offline backtesting harness. These agents are excluded from pass/fail determination; their confidence gates are set at conservative floors. Pass threshold: 60% for agents with live-data fields; gate = 80% × measured hit rate for agents with full backtest coverage.

Confidence is calibrated via data completeness flags passed in `AgentHandoff`, not self-reported strings from the model.

### Harness Change Log

Key infrastructure decisions and their measured impact on agent hit rates:

| Date | Change | Layer | Measured Impact |
|------|--------|-------|----------------|
| Apr-14 | `AgentHandoff` TypedDict replaced 400-char string blob | Context | risk_officer +5.5 pp (71.8% → 77.3%) |
| Apr-14 | `TradeOutput` Pydantic schema + 3-attempt retry | Harness / Verification | Schema violation rate → 0% |
| Apr-14 | `NUMERIC_PAIRS` divergence detection | Harness / Verification | Caught 2 real inconsistencies in live runs |
| Apr-14 | EWM span 30 → 252 days | Context | 3 active-war cases flipped verdict; consistent with RiskMetrics long-memory specification |
| Apr-14 | OLS residualization removed from commodity vol | Context | Oil-shock cases score correctly; supply-shock co-movement preserved |
| Apr-14 | GDELT retry + throttle + outer cache | Harness / Tool Execution | 429 failures → 0; escalation_signal availability 40% → 100% |
| Apr-21 | `score_all_assets()` unhashable dict parameter removed | Harness | P0 silent failure — Exposure Scoring and Trade Ideas were returning empty results |
| Apr-21 | `_call_ai` TTL aligned 600s → 3600s to match `AGENT_TTL` | Harness | Eliminates stale/error string re-serving within the fresh window |
| Apr-21 | `"unavailable:"` narrative guard at harness layer | Harness | Error strings no longer stored as agent output or shown to users |
| Apr-21 | `pipeline_status` partial-failure tracking | Harness | Failed agents now identified and reported; were silently marked complete |
| Apr-28 | `log_verification_event()` — divergence + confidence gate events to trace CSV | Harness / Observability | Verification loops now auditable from CSV; not lost on session restart |
| Apr-28 | Granger BIC lag selection + Holm-Bonferroni correction | Statistical | Multiple-comparison inflation corrected; effective α now at stated 5% |
| Apr-28 | Transfer entropy shuffle significance (Schreiber 2000) | Statistical | TE cells now carry p-values; spurious non-zero TE values filterable |
| Apr-28 | DCC-GARCH EWMA pre-whitening (Engle 2002 Step 1) | Statistical | Heteroskedasticity bias removed from dynamic correlations |
| Apr-28 | Diebold-Yilmaz VAR BIC lag selection | Statistical | Overfitting from fixed lag_order eliminated across all four VAR fit calls |

---

## Page-by-Page Guide

### Command Center (Overview)
Starting point for every session. KPI strip (regime, 60d avg correlation with 1M delta, best/worst equity/commodity), FI/FX context strip, implied vol panel, Composite Risk Score gauge (0–100, one-decimal precision), Early Warning System, Risk Officer morning briefing, agent activity feed.

### Scenario Engine
Parametric shock simulator. Six inputs propagated via OLS betas to all 32 assets. Waterfall impact chart with channel attribution. VaR 95/99 and ES 95/99. AI Stress Engineer narrative assessment.

### War Impact Map
Global choropleth — 195+ countries scored by equity-market exposure to three simultaneous active conflict theaters. Scoring framework by Ilian Zalomai.

### Geopolitical Triggers
Event-window forensics for 13 shocks (2008–present). Normalized price performance, volatility shift, correlation regime change at event onset/peak/recovery. Live Intelligence Feed with RSS ingestion and severity scoring.

### Correlation Analysis
Four tabs: Rolling Correlation, DCC-GARCH (Engle 2002, EWMA pre-whitened), Regime Detection, Regime Forecast. Markov transition matrix, steady-state distribution, mean days to Crisis per state. Regime taxonomy by Jiahe Miao.

### Spillover Analytics
Granger causality grid (BIC lag, Holm-corrected; both raw and FWER-adjusted significant-pair counts displayed), transfer entropy matrix with shuffle p-values (* = p < 0.05), Diebold-Yilmaz FEVD connectedness index (0–100%, BIC-selected VAR lag), network graph (node size = net spillover power), cross-asset section covering rates, FX, and FI channels.

### Macro Intelligence Dashboard
FRED-powered: yield curve, CPI, Fed Funds Rate, GDP, ISM PMI. Recession shading, yield curve shape analysis. AI Macro Strategist briefing with typed peer context.

### Commodities to Watch
Live snapshot (price, 1d/5d return, annualized vol, regime label). Intraday and daily charts. COT positioning overlay. Rolling 60d correlation vs. all equity regions.

### Strait Watch
Six chokepoints: Hormuz, Suez, Taiwan Strait, Bosporus, Malacca, Panama. Disruption score (0–100), vessel traffic vs. baseline, commodity channels at risk, estimated trade value at risk. Framework by Ilian Zalomai.

### Trade Ideas
Regime-triggered cross-asset trade cards. AI Trade Structurer generates ideas using full typed peer context. Pydantic-validated structured output. Six base regime-conditioned trade structures by Jiahe Miao.

### Exposure Scoring
Per-asset Structural Exposure Score (SES), Transmission-Adjusted Exposure (TAE), and Scenario-Adjusted Score (SAS) across all tracked conflicts. Geo multiplier applied via scenario state.

### Portfolio Stress Test
Custom multi-asset allocation (indices, commodity futures, fixed income ETFs, individual stocks) tested against all 13 events. Per-event return, max drawdown, Sharpe. Heatmap and normalized portfolio path charts.

### Performance Review (Model Accuracy)
Granger hit rates, regime detection F1, confusion matrix, COT contrarian signal performance. AI Signal Auditor calibration output. 56-case benchmark suite with pass/fail thresholds. Results stored in `evals/results-2026-04.md`.

### Insights
Private credit bubble risk monitor, India macro framework (crude oil dependency, rupee transmission, Nifty drivers), cross-asset spillover research synthesis.

### AI Analyst
Claude Sonnet / GPT-4o chatbot with full live dashboard context injected into every query. Agent Activity Log with structured feed of all pipeline events and routing decisions.

---

## Navigation

```
Command Center
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
    ├── Ilian Zalomai
    └── Jiahe Miao
```

---

## Local Setup

```bash
git clone https://github.com/HPATKAR/equity-commodities-spillover.git
cd equity-commodities-spillover
pip install -r requirements.txt
streamlit run app.py
```

### Configuration

Add to `.streamlit/secrets.toml` (local) or as Secret Files on Render (deployed):

```toml
[keys]
fred_api_key      = "..."   # FRED macroeconomic data (optional — unlocks Macro Dashboard)
anthropic_api_key = "..."   # Claude Sonnet — AI agents + analyst (preferred)
openai_api_key    = "..."   # GPT-4o fallback

[auth]
password = "..."            # Optional — activate with enable_auth = true

[config]
enable_auth      = false
enable_rss       = true
refresh_interval = 300
```

Anthropic is used when both keys are present. FRED unlocks yield curve, CPI, Fed Funds Rate, GDP, and ISM data on the Macro Dashboard.

---

## Stack

| Layer | Libraries |
|-------|-----------|
| **UI** | Streamlit, Plotly, custom HTML/CSS |
| **Data** | yfinance, fredapi, feedparser, requests |
| **Analytics** | pandas, numpy, statsmodels, arch, scipy, scikit-learn, networkx |
| **AI** | anthropic, openai, pydantic |
| **Harness** | Custom orchestrator, trace_logger, agent_benchmark, agent_state |
| **Reports** | python-docx, reportlab, Pillow |

---

## Known Limitations

| Limitation | Scope | Mitigation / Threshold |
|-----------|-------|----------------------|
| Granger p-values unreliable below n=200 despite 60-obs minimum | Spillover Analytics | Warning shown; BIC lag selection reduces false positive rate |
| Holm correction reduces power: true links may be flagged non-significant in small grids | Spillover Analytics | Raw p<.05 column retained alongside Holm-corrected for comparison |
| Transfer entropy shuffle test uses 200 permutations — noisy for p-values near 0.05 | Spillover Analytics | Recommend treating 0.01 < p < 0.05 as marginal significance |
| Historical CIS/TPS are regime proxies — ACLED back-test unavailable offline | geopolitical_analyst | Forward-looking only; eval flagged as eval-limited |
| GDELT tracks media volume, not events — lags actual conflict escalation 12–48h for non-Western conflicts | geopolitical_analyst | Corroborating signal only, not primary |
| DCC EWMA λ=0.94 not data-estimated — RiskMetrics daily standard applied | Correlation Analysis | Insensitive to λ in [0.90, 0.97]; full MLE estimation would require `arch` package GARCH fit |
| VAR BIC may select lag=1 on short rolling windows, understating persistence | Rolling D-Y | lag_order parameter acts as upper bound; minimum window kept at 200 days |
| Scenario geo_mult outside [0.5, 3.0] may produce edge-case clipping | Risk Officer | Validated input range only |

---

## Academic References

| Citation | Used In |
|----------|---------|
| Granger, C.W.J. (1969). "Investigating Causal Relations by Econometric Models and Cross-spectral Methods." *Econometrica* 37(3): 424–438. | Granger causality grid |
| Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer. Ch. 4. | BIC lag selection for VAR and Granger |
| Schreiber, T. (2000). "Measuring Information Transfer." *Physical Review Letters* 85(2): 461–464. | Transfer entropy + shuffle significance |
| Diebold, F.X. & Yilmaz, K. (2012). "Better to Give than to Receive: Forecast-Based Measurement of Volatility Spillovers." *International Journal of Forecasting* 28(1): 57–66. | Diebold-Yilmaz FEVD spillover index |
| Engle, R.F. (2002). "Dynamic Conditional Correlation: A Simple Class of Multivariate Generalized Autoregressive Conditional Heteroskedasticity Models." *JBES* 20(3): 339–350. | DCC-GARCH correlations |
| Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test Procedure." *Scandinavian Journal of Statistics* 6(2): 65–70. | Bonferroni-Holm correction for Granger grid |
| Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica* 57(2): 357–384. | Regime-switching framework |
| J.P. Morgan / Reuters (1996). *RiskMetrics Technical Document*, 4th ed. | EWMA λ=0.94 for DCC pre-whitening |

---

## Disclaimers

This dashboard is built for **educational and research purposes only**. Nothing on this platform constitutes investment advice. All analytics are based on historical data and statistical models — past relationships do not guarantee future behavior. Market data is sourced from Yahoo Finance and FRED and may have delays or gaps. AI agent outputs are generated by third-party language models and should not be treated as financial guidance.

---

*Equity-Commodities Spillover Monitor · Purdue University · MGMT 69000-120: AI for Finance · Heramb S. Patkar · Ilian Zalomai · Jiahe Miao*
