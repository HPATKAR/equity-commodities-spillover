"""
TOMBSTONED — do not use.

A pre-populated catalogue of strategies is incompatible with thesis-first methodology.
Pre-packaging theses and iterating over them to find the best Sharpe is search-and-filter
disguised as a pipeline — it selects FROM candidates rather than ADMITTING by gate logic.

The correct approach: the researcher constructs ONE thesis from first principles (shock →
TPS channel → predicted sign → holding horizon), walks it through the five-stage pipeline
using only past data, and reports the pipeline's decision. The pipeline is the deliverable,
not the trade.

For the walk-forward validation, theses are constructed inline in the validation call and
passed to walk_forward_pipeline_validation() in pipeline_validator.py.

For the worked example, see the "Thesis Pipeline" section in src/pages/trade_ideas.py.
"""
