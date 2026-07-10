"""
Standalone precompute — warm-before-serve.

Runs the full warm-up ONCE, synchronously, and exits. It computes and persists
every expensive shared artifact (conflict scores, asset scores, macro/single-
stock returns, market pulse, market risk, hot-stocks RSS, static back-tests) to
the artifact cache. Because the web process reads those artifacts from disk, a
process that starts AFTER this has finished serves the first request warm — no
user ever hits a cold render.

Run it (any of these):

    python -m src.precompute

Deploy pattern (Render / any container) so the web process is warm before it
serves — do BOTH:

  1. Attach a PERSISTENT DISK and point the cache at it (so artifacts survive
     redeploys and are shared between the precompute and the web process):

         ARTIFACT_CACHE_DIR=/var/data/artifacts     # your mounted disk path

  2. Warm before serving by prepending precompute to the start command:

         python -m src.precompute && streamlit run app.py --server.port $PORT ...

     The precompute writes the artifacts to the persistent disk first; Streamlit
     then starts and reads them, so its first render is warm. (Give the service a
     generous health-check grace period — first-ever precompute on an empty disk
     is the ~30-80s network pass; every run after reads disk and is seconds.)

  3. (Optional) keep the disk fresh over time with a scheduled job / cron that
     runs `python -m src.precompute` on the same persistent disk, e.g. hourly.

Idempotent and failure-safe: a failure in one source is logged and skipped, and
a bad run never leaves the app worse than a normal cold start.
"""
from __future__ import annotations

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s precompute: %(message)s",
)
_log = logging.getLogger("precompute")


def run_once() -> None:
    """Warm + persist every shared artifact, then return (no daemon timer)."""
    t0 = time.monotonic()
    _log.info("starting warm-before-serve precompute…")
    try:
        from src.warmup import _run
        _run(reschedule=False)   # reuse the exact warm-up logic, run synchronously
    except Exception as exc:      # never crash the deploy — a warm miss just = cold start
        _log.warning("precompute error (continuing): %s", exc)
    # Report which artifacts landed on disk, for deploy-log visibility.
    try:
        import datetime as _dt
        from src.utils.artifact_cache import artifact_age_s
        _end = str(_dt.date.today())
        keys = [
            "conflict_scores", "asset_scores__base", f"stock_returns__{_end}",
            "market_pulse", f"market_risk__{_end}__base", "hot_stocks",
        ]
        landed = {k: (round(a, 1) if (a := artifact_age_s(k)) is not None else None)
                  for k in keys}
        _log.info("artifacts on disk (age s): %s", landed)
    except Exception:
        pass
    _log.info("precompute complete in %.1fs", time.monotonic() - t0)


if __name__ == "__main__":
    run_once()
    sys.exit(0)
