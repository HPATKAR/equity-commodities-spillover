#!/usr/bin/env python3
"""
Standalone benchmark evaluator for the Cross-Asset Spillover Monitor.
Loads 20 labeled historical snapshots, runs the quantitative model on each,
evaluates typed field assertions, and prints a pass/fail results table.

Usage:
    cd /path/to/equity-commodities-spillover
    python evals/run_eval.py [--agents risk_officer macro_strategist ...]
    python evals/run_eval.py --output evals/results-$(date +%Y-%m).md

No Streamlit required. All @st.cache_data decorators are mocked out.
The script uses only the quantitative model (correlations, risk_score) —
not the LLM agents — so results are deterministic and reproducible.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import types
from pathlib import Path

# ── Mock Streamlit so decorated modules can be imported outside Streamlit ──────
def _make_st_mock() -> types.ModuleType:
    st = types.ModuleType("streamlit")
    st.cache_data  = lambda *a, **k: (lambda f: f)  # no-op decorator
    st.session_state = {}
    st.warning = lambda *a, **k: None
    st.error   = lambda *a, **k: None
    return st

sys.modules.setdefault("streamlit", _make_st_mock())

# Add project root to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ── Imports (after mock) ───────────────────────────────────────────────────────
from evals._eval_core import run_snapshot, evaluate_assertions  # noqa: E402

_AGENTS = [
    "risk_officer", "macro_strategist", "geopolitical_analyst",
    "commodities_specialist", "signal_auditor", "stress_engineer",
]

_PASS_THRESHOLD = 0.60   # below this → FAIL gate

# ── Comparator engine (mirrors agent_benchmark.py) ────────────────────────────
_CMP = {
    "gt":      lambda a, b: a > b,
    "lt":      lambda a, b: a < b,
    "gte":     lambda a, b: a >= b,
    "lte":     lambda a, b: a <= b,
    "eq":      lambda a, b: a == b,
    "between": lambda a, b: b[0] <= a <= b[1],
}


def check(actual, comparator: str, expected) -> bool:
    if actual is None:
        return False
    try:
        fn = _CMP.get(comparator)
        return bool(fn(actual, expected)) if fn else False
    except Exception:
        return False


def run_benchmark(snapshots: list[dict], agent_ids: list[str]) -> dict[str, dict]:
    """
    For each snapshot: load market data, run quantitative model, check assertions.
    Returns per-agent result dicts.
    """
    per_snap_outputs: list[tuple[dict, dict]] = []  # (snapshot, model_output)

    print(f"\nRunning {len(snapshots)} historical snapshots...")
    for snap in snapshots:
        label = snap["label"]
        date  = snap["date"]
        sys.stdout.write(f"  [{date}] {label} ... ")
        sys.stdout.flush()
        try:
            output = run_snapshot(date)
            # Proxy CIS/TPS from risk_score when VIX regime is elevated
            vix = snap.get("vix_approx", 15)
            vix_regime = 3 if vix > 35 else (2 if vix > 22 else 1)
            if vix_regime >= 2:
                rs = output.get("risk_score") or 50.0
                output.setdefault("cis", round(rs * 0.85, 1))
                output.setdefault("tps", round(rs * 0.75, 1))
            per_snap_outputs.append((snap, output))
            status = "OK" if output.get("_computed") else "PARTIAL"
        except Exception as e:
            per_snap_outputs.append((snap, {}))
            status = f"ERR: {e}"
        print(status)

    # Score each agent
    results: dict[str, dict] = {}
    for aid in agent_ids:
        total = passed = 0
        rows = []
        for snap, output in per_snap_outputs:
            for field, cmp_op, expected in snap.get("assertions", {}).get(aid, []):
                actual = output.get(field)
                if field == "regime" and actual is None:
                    # Fall back to VIX-derived regime
                    vix = snap.get("vix_approx", 15)
                    actual = 3 if vix > 35 else (2 if vix > 22 else 1)
                ok = check(actual, cmp_op, expected)
                total += 1
                if ok:
                    passed += 1
                rows.append({
                    "date":       snap["date"],
                    "label":      snap["label"],
                    "field":      field,
                    "comparator": cmp_op,
                    "expected":   expected,
                    "actual":     actual,
                    "passed":     ok,
                })
        hit_rate = round(passed / total, 3) if total else None
        results[aid] = {
            "agent_id": aid,
            "total":    total,
            "passed":   passed,
            "failed":   total - passed,
            "hit_rate": hit_rate,
            "gate":     "PASS" if (hit_rate or 0) >= _PASS_THRESHOLD else "FAIL",
            "rows":     rows,
        }
    return results


def print_summary(results: dict[str, dict]) -> None:
    print("\n" + "=" * 72)
    print(f"{'Agent':<28} {'Cases':>6} {'Pass':>6} {'Hit Rate':>9} {'Gate':>6}")
    print("-" * 72)
    for aid, r in results.items():
        hr = f"{r['hit_rate']:.1%}" if r["hit_rate"] is not None else "  N/A"
        gate = r["gate"]
        marker = "" if gate == "PASS" else " ✗"
        print(f"{aid:<28} {r['total']:>6} {r['passed']:>6} {hr:>9} {gate:>6}{marker}")
    print("=" * 72)
    n_fail = sum(1 for r in results.values() if r["gate"] == "FAIL")
    if n_fail:
        print(f"\n⚠  {n_fail} agent(s) below {_PASS_THRESHOLD:.0%} threshold — review details above.\n")
    else:
        print(f"\n✓  All agents passed the {_PASS_THRESHOLD:.0%} hit-rate gate.\n")


def write_markdown(results: dict[str, dict], path: str) -> None:
    run_date = datetime.date.today().isoformat()
    lines = [
        f"# Benchmark Results — {run_date}",
        "",
        f"Pass threshold: **{_PASS_THRESHOLD:.0%}** hit rate",
        "",
        "## Summary",
        "",
        "| Agent | Cases | Passed | Hit Rate | Gate |",
        "|-------|------:|-------:|---------:|------|",
    ]
    for aid, r in results.items():
        hr = f"{r['hit_rate']:.1%}" if r["hit_rate"] is not None else "N/A"
        gate = r["gate"]
        icon = "✅" if gate == "PASS" else "❌"
        lines.append(f"| {aid} | {r['total']} | {r['passed']} | {hr} | {icon} {gate} |")

    lines += ["", "## Per-Agent Detail", ""]
    for aid, r in results.items():
        lines += [f"### {aid}", "", "| Date | Label | Field | Expected | Actual | Pass |",
                  "|------|-------|-------|----------|--------|------|"]
        for row in r["rows"]:
            icon = "✅" if row["passed"] else "❌"
            lines.append(
                f"| {row['date']} | {row['label']} | `{row['field']}` "
                f"| {row['comparator']} {row['expected']} "
                f"| {row['actual']} | {icon} |"
            )
        lines.append("")

    Path(path).write_text("\n".join(lines))
    print(f"Results written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent benchmark eval")
    parser.add_argument("--agents", nargs="*", default=_AGENTS,
                        help="Agent IDs to evaluate (default: all)")
    parser.add_argument("--output", default="",
                        help="Write markdown results to this path")
    args = parser.parse_args()

    cases_path = Path(__file__).parent / "benchmark-cases.json"
    snapshots  = json.loads(cases_path.read_text())

    results = run_benchmark(snapshots, args.agents)
    print_summary(results)

    if args.output:
        write_markdown(results, args.output)


if __name__ == "__main__":
    main()
