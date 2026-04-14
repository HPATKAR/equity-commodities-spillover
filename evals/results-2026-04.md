# Benchmark Results — 2026-04-14

Pass threshold: **60%** hit rate
Run command: `python evals/run_eval.py --output evals/results-2026-04.md`

> **Note:** Results below are from the quantitative model back-test (real yfinance data through `_eval_core.py`).
> Geopolitical CIS/TPS assertions use VIX-derived regime as a proxy since the conflict model
> runs on live manually maintained data and cannot be genuinely back-tested.
> `granger_hit_rate` assertions cannot be evaluated standalone (requires Granger test on
> historical pairs); those rows are marked N/A.

## Summary

| Agent | Cases | Passed | Hit Rate | Gate |
|-------|------:|-------:|---------:|------|
| risk_officer | 22 | 17 | 77.3% | ✅ PASS |
| macro_strategist | 10 | 7 | 70.0% | ✅ PASS |
| geopolitical_analyst | 12 | 9 | 75.0% | ✅ PASS |
| commodities_specialist | 5 | 4 | 80.0% | ✅ PASS |
| signal_auditor | 4 | — | N/A (Granger proxy) | ⚠ MANUAL |
| stress_engineer | 3 | 2 | 66.7% | ✅ PASS |

> **signal_auditor** `granger_hit_rate` assertions require Granger p-value computation
> over historical windows. These are validated manually against the Model Accuracy page
> hit-rate table rather than the standalone script.

## Known Limitations

1. **CIS/TPS proxy**: Geopolitical conflict intensity scores are derived from
   `risk_score × 0.85` when VIX regime ≥ 2. This is a calibrated approximation,
   not true historical conflict data.

2. **cmd_vol_z**: Computed from 20-day realized vol standardised against the
   400-day window. Actual COT-derived positioning data not available for
   historical dates.

3. **yield_curve_spread**: Approximated from TLT vs SHY 60-day cumulative
   return differential. Not a direct 10Y-2Y spread (requires FRED back-data).

4. **Regime thresholds**: 60th/80th percentile thresholds are computed from the
   same 400-day window used for assertions. In a true out-of-sample test,
   thresholds would be fixed from an earlier calibration period.

## To regenerate

```bash
cd /path/to/equity-commodities-spillover
python evals/run_eval.py --output evals/results-$(date +%Y-%m).md
```
