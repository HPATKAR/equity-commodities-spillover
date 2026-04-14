# Week 5 — Response to Reviewer Comments

Three-column format per professor instruction: **Comment → Action Taken → Artifact**.

---

| # | Reviewer Comment | Action Taken | Artifact |
|---|-----------------|--------------|----------|
| 1 | Peer context between agents truncated to 400 characters, discarding numeric precision | Replaced string truncation with `AgentHandoff` TypedDict carrying typed numeric fields: `risk_score`, `regime`, `cis`, `tps`, `yield_curve_spread`, `cpi_yoy`, `cmd_vol_z`, `corr_pct`, `granger_hit_rate` | `src/analysis/agent_orchestrator.py` — `AgentHandoff`, `_store_handoff`, `_get_handoff` |
| 2 | Divergence detection between agents is keyword string matching | Replaced keyword scan with numeric field comparison via `NUMERIC_PAIRS`: flags when `risk_score` diff > 20 or `regime` disagrees by ≥ 1 between peer agents | `src/analysis/agent_orchestrator.py` — `_detect_divergence`, `NUMERIC_PAIRS` |
| 3 | Agent confidence parsed from model text with regex (`CONFIDENCE: X%`) | Removed regex from all 8 agent `_call_ai()` functions. Confidence computed via `context_confidence(agent_id, context)` — data completeness × signal strength × regime clarity — then calibrated via `calibrate_confidence()` | `src/analysis/agent_state.py` — `context_confidence()`, `calibrate_confidence()`; all files in `src/agents/` |
| 4 | No agent-level benchmark — only signal-level validation exists | Built `agent_benchmark.py` with 20 real market dates (COVID crash, Lehman, Russia-Ukraine, SVB, etc.), typed field assertions per agent, `run_historical_snapshot()` loading actual yfinance price windows | `src/analysis/agent_benchmark.py`, `evals/benchmark-cases.json` |
| 5 | Confidence not defensible — uses Granger loose proxy | Per-agent posterior from benchmark back-test hit rates. `calibrate_confidence()` uses three-tier priority: (1) dynamic posteriors from `run_full_benchmark()` session results, (2) static `POSTERIOR_ACCURACY` table, (3) 0.65 fallback | `src/analysis/agent_benchmark.py` — `POSTERIOR_ACCURACY`, `compute_dynamic_posteriors()` |
| 6 | Add Pandera schema validation on `loader.py` outputs — prices, macro series, RSS rows | Added `_validate_returns()` and `_validate_prices()` to `loader.py` (non-destructive: emits `st.warning`, never crashes). Added RSS row validation loop in `geo_rss.ingest_headlines()` checking id length, title non-empty, relevance bounds, list types | `src/data/loader.py` — `_validate_returns`, `_validate_prices`; `src/ingestion/geo_rss.py` — validation block |
| 7 | No record of what each step cost or how long it took | Built `trace_logger.py`. Every `_call_ai()` in all 8 agents now calls `log_trace()` after each LLM response, appending: timestamp, agent, provider, model, prompt chars, completion chars, estimated tokens, latency ms, estimated cost USD to `logs/agent_traces.csv` | `src/analysis/trace_logger.py`; integrated in all `src/agents/*.py` |
| 8 | Prompts glued as strings with model output parsed by regex — no structured template validation | Replaced `trade_structurer` with Pydantic `TradeOutput` schema. `_call_ai()` requests JSON, validates with `TradeOutput.model_validate_json()`, retries up to 2× on `ValidationError` feeding the error back to the model. Invalid output is logged and never passed forward | `src/agents/trade_structurer.py` — `TradeOutput`, `_call_ai()` with retry loop |
| 9 | No standalone eval script or labeled benchmark cases file | Created `evals/benchmark-cases.json` (20 labeled cases as JSON), `evals/run_eval.py` (standalone runner, no Streamlit required), `evals/_eval_core.py` (quantitative model runner), `evals/results-2026-04.md` (metrics table with pass/fail) | `evals/` folder |
| 10 | Data quality report — validation rules, freshness SLAs, results | Created `week5/data-quality-report.md` documenting all validation rules, freshness thresholds per source, outlier flags, and known limitations | `week5/data-quality-report.md` |

---

## Honest Gaps

- **Granger `hit_rate` back-test**: Cannot be evaluated standalone (requires running Granger tests on historical price pairs for each snapshot window). Validated via the Model Accuracy page's live hit-rate table instead.
- **CIS/TPS back-test**: Conflict model uses manually maintained live data; historical conflict intensity cannot be reconstructed from prices. Geo assertions use VIX-regime as a proxy (documented in `evals/results-2026-04.md`).
- **LLM faithfulness metrics (Ragas/DeepEval)**: Not implemented. Would require ground-truth source documents per narrative to compute citation rate. Current proxy: narrative agents no longer append self-reported confidence — outputs are evaluated on typed numeric fields only.
