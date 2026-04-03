# Contributors

**Course Project 3 - MGMT 69000-120: AI for Finance**
Purdue University - Mitchell E. Daniels, Jr. School of Business
Instructor: Prof. Cinder Zhang

---

## Team

| Name | Role | Background |
|------|------|------------|
| Heramb S. Patkar | Lead Developer & Quantitative Architect | MSF, Purdue; Equity Research, Axis Securities |
| Jiahe Miao | Quantitative Research & Trade Framework | MSF, Purdue; B.S. Information Systems, Kelley |
| Ilian Zalomai | Geopolitical Research & Scenario Design | MSF, Purdue; Deloitte; Firebird Tours |

---

## Contributions

### Heramb S. Patkar
**Project lead - full-stack development, quantitative engine, and AI system design**

Sole developer responsible for the entire codebase. Conceived the project architecture, selected the analytical stack, and built every component from data ingestion through UI rendering.

- **Application architecture**: Designed and built the complete multi-page Streamlit application from scratch - custom fixed navbar (iframe-based), institutional dark theme CSS design system, responsive layout, and all routing logic across 14 pages
- **Quantitative engine**: Implemented five independent analytical models - Diebold-Yilmaz FEVD spillover network, VAR/Granger causality with lag selection, rolling Pearson and DCC-GARCH dynamic correlation, Hidden Markov Model regime detection, and non-parametric transfer entropy estimation
- **AI agent system**: Architected and built an 8-agent AI workforce (Risk Officer, Macro Strategist, Geopolitical Analyst, Commodities Specialist, Stress Engineer, Signal Auditor, Trade Structurer, Chief Quality Officer) running a 4-round dependency-ordered orchestration pipeline; built the CQO remediation loop that automatically routes quality flags to specialist agents for active correction
- **Data infrastructure**: Built the full data ingestion layer - yfinance multi-asset downloader (50+ tickers across equities, commodities, FX, fixed income), FRED API integration for 14 macro series, real-time implied vol scraping (VIX/OVX/GVZ/VVIX), and a proactive alert engine with regime-sensitive thresholds
- **Dashboard pages**: Built all 14 pages end-to-end - Overview, Correlation Analysis, Spillover Analytics, Macro Intelligence, Geopolitical Triggers, War Impact Map, Strait Watch, Trade Ideas, Portfolio Stress Test, Scenario Engine, Commodities Watchlist, Model Accuracy, Actionable Insights, and AI Chat
- **Stress and scenario system**: Implemented historical scenario simulation across 10 crisis events with portfolio-weighted impact attribution; built the Scenario Engine with custom shock propagation and AI stress commentary
- **Strait Watch and geopolitical layers**: Built the maritime chokepoint monitoring system, Iran/Hormuz crisis tracker, and geopolitical event overlay across all charts; wrote the full geopolitical event catalog (`config.py`, 13 events spanning 2008-2025) with market-specific impact descriptions
- **Submission deliverables**: Generated the submission Word document and presentation script PDF programmatically (`generate_submission_docx.py`, `generate_script_pdf.py`, `generate_script_docx.py`)

### Jiahe Miao
**Quantitative methodology research and trade idea framework**

- Researched and specified the correlation regime framework: defined the 4-state regime taxonomy (Low/Moderate/Elevated/Crisis), threshold calibration rationale, and rolling window selection methodology
- Designed the equity-commodities trade idea logic: identified the 6 regime-conditioned trade structures, entry/exit range rationale, and correlation breakpoint triggers
- Contributed to the Fixed Income integration on the Overview page: defined the TLT/IEF/SHY/LQD/HYG/EMB metrics and their interpretation framework for cross-asset stress signaling
- Researched the Diebold-Yilmaz methodology and its application to cross-asset spillover; contributed to the network edge threshold calibration and directional spillover interpretation
- Validated the private credit bubble risk scoring methodology on the Insights page: assessed the BKLN/ARCC/OBDC/FSK/JBBB proxy selection and defined the composite score weighting logic
- Performed data quality review across equity and commodity return series; flagged and resolved TOPIX ETF proxy selection (1306.T) and Nickel ticker alignment (NILSY)

### Ilian Zalomai
**Geopolitical research, war impact framework, and macro narrative**

- Designed the War Impact Map scoring framework: defined per-war baseline scores, country-level commodity exposure weights, and the concurrent-war amplifier methodology for Ukraine, Gaza/Red Sea, and Iran/Hormuz conflicts
- Authored the geopolitical event descriptions in the dashboard (GFC, Arab Spring, US-China Trade War, Aramco Attack, COVID, WTI Negative, Ukraine War, LME Nickel Squeeze, Fed Hiking Cycle, SVB Crisis, Israel-Hamas, India-Pakistan, Iran/Hormuz)
- Researched and built the Strait Watch framework: identified the five critical maritime chokepoints (Hormuz, Malacca, Suez, Bab-el-Mandeb, Turkish Straits), sourced EIA/IEA throughput data, and defined the disruption scoring methodology
- Developed the Iran/Hormuz Crisis page content: oil and LNG transit exposure by region, energy-importing Asia risk analysis, and Brent/WTI spread interpretation
- Contributed to the geopolitical risk overlay on the Macro Dashboard: connected active conflicts to Fed policy transmission and commodity supply shocks
- Authored the narrative context sections across the dashboard (page intros, takeaway blocks, section conclusions) that connect quantitative outputs to real-world market implications
- Reviewed and stress-tested the scenario definitions in the Scenario Engine against historical market precedents

---

## Repository

GitHub: [HPATKAR/equity-commodities-spillover](https://github.com/HPATKAR/equity-commodities-spillover)

Tech stack: Python 3.11 - Streamlit - pandas - statsmodels - plotly - yfinance - fredapi - Anthropic/OpenAI APIs

~23,000 lines of Python across `src/` (pages, agents, analysis, data, UI, ingestion, reports)
