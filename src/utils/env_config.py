"""
Centralised configuration resolver.
Reads from (in priority order):
  1. Environment variables
  2. .streamlit/secrets.toml  (st.secrets)
  3. Hard-coded defaults

Usage:
    from src.utils.env_config import cfg
    cfg.fred_key          # str | ""
    cfg.anthropic_key     # str | ""
    cfg.openai_key        # str | ""
    cfg.refresh_interval  # int  seconds
    cfg.log_level         # str  "INFO" | "DEBUG" | "WARNING"
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class _Config:
    # API keys
    fred_key:      str
    anthropic_key: str
    openai_key:    str
    lseg_key:      str
    fd_key:        str

    # Refresh / caching
    refresh_interval: int   # seconds between auto-refresh on Overview
    agent_ttl:        int   # default agent output TTL (seconds)
    iv_ttl:           int   # implied vol cache TTL (seconds)

    # Feature flags
    enable_rss:       bool  # activate geo RSS ingestion
    enable_auth:      bool  # require password on app load

    # Logging
    log_level:        str   # "DEBUG" | "INFO" | "WARNING" | "ERROR"
    log_file:         str   # "" = stderr only

    @property
    def ai_provider(self) -> str | None:
        if self.anthropic_key:
            return "anthropic"
        if self.openai_key:
            return "openai"
        return None

    @property
    def ai_key(self) -> str:
        return self.anthropic_key or self.openai_key


def _s(key: str, section: str = "keys", default: str = "") -> str:
    """Resolve from env → secrets → default."""
    # 1. Environment variable (uppercase, underscores)
    env_val = os.environ.get(key.upper(), "")
    if env_val:
        return env_val
    # 2. Streamlit secrets
    try:
        import streamlit as st
        val = st.secrets.get(section, {}).get(key, "") or ""
        if val:
            return str(val)
    except Exception:
        pass
    return default


def _b(key: str, section: str = "config", default: bool = False) -> bool:
    raw = _s(key, section, str(default)).lower()
    return raw in ("1", "true", "yes", "on")


def _i(key: str, section: str = "config", default: int = 0) -> int:
    try:
        return int(_s(key, section, str(default)))
    except (ValueError, TypeError):
        return default


@lru_cache(maxsize=1)
def _build_config() -> _Config:
    return _Config(
        fred_key      = _s("fred_api_key"),
        anthropic_key = _s("anthropic_api_key"),
        openai_key    = _s("openai_api_key"),
        lseg_key      = _s("app_key", section="lseg"),
        fd_key        = _s("financial_datasets_key"),

        refresh_interval = _i("refresh_interval", default=300),   # 5 min default
        agent_ttl        = _i("agent_ttl",        default=3600),
        iv_ttl           = _i("iv_ttl",           default=900),

        enable_rss  = _b("enable_rss",  default=True),
        enable_auth = _b("enable_auth", default=False),

        log_level = _s("log_level", section="config", default="INFO").upper(),
        log_file  = _s("log_file",  section="config", default=""),
    )


# Singleton — import and use directly
cfg = _build_config()
