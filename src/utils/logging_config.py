"""
Structured logging for the Cross-Asset Spillover Monitor.

Features:
- JSON-structured log records (agent_id, page, severity, message, data)
- Rotating file handler when log_file is set
- Console handler always active
- Safe to call before Streamlit context exists

Usage:
    from src.utils.logging_config import get_logger
    log = get_logger("overview")
    log.info("regime_detected", regime="Crisis", score=82.4)
    log.warning("agent_stale", agent_id="macro_strategist", age_seconds=4200)
    log.error("data_load_failed", source="yfinance", ticker="^OVX", exc=str(e))
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        }
        # Extra kwargs passed via log.info("msg", key=val) are stored in record.__dict__
        for k, v in record.__dict__.items():
            if k not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
            ):
                try:
                    json.dumps(v)   # only include JSON-serialisable extras
                    payload[k] = v
                except (TypeError, ValueError):
                    payload[k] = str(v)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class _StructuredLogger:
    """
    Thin wrapper that allows keyword-style structured logging:
        log.info("agent_complete", agent_id="risk_officer", conf=0.82)
    """

    def __init__(self, name: str) -> None:
        self._log = logging.getLogger(f"ec.{name}")

    def _emit(self, level: int, event: str, **kw: Any) -> None:
        if self._log.isEnabledFor(level):
            self._log.log(level, event, extra=kw)

    def debug(self, event: str, **kw: Any)   -> None: self._emit(logging.DEBUG,   event, **kw)
    def info(self,  event: str, **kw: Any)   -> None: self._emit(logging.INFO,    event, **kw)
    def warning(self, event: str, **kw: Any) -> None: self._emit(logging.WARNING, event, **kw)
    def error(self, event: str, **kw: Any)   -> None: self._emit(logging.ERROR,   event, **kw)


_configured = False


def _configure(level: str = "INFO", log_file: str = "") -> None:
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger("ec")
    root.setLevel(getattr(logging, level, logging.INFO))
    root.propagate = False

    fmt = _JsonFormatter()

    # Console handler
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file handler
    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            str(p), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)


def get_logger(name: str) -> _StructuredLogger:
    """Return a structured logger. Auto-configures on first call."""
    try:
        from src.utils.env_config import cfg
        _configure(cfg.log_level, cfg.log_file)
    except Exception:
        _configure()
    return _StructuredLogger(name)
