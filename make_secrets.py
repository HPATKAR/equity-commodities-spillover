"""Build-time script: writes .streamlit/secrets.toml from environment variables.
Run during Render build so st.secrets picks up keys set in the dashboard.
"""
import os
from pathlib import Path

KEY_MAP = {
    "anthropic_api_key":      "ANTHROPIC_API_KEY",
    "openai_api_key":         "OPENAI_API_KEY",
    "fred_api_key":           "FRED_API_KEY",
    "eia_api_key":            "EIA_API_KEY",
    "financial_datasets_key": "FINANCIAL_DATASETS_KEY",
}

LSEG_MAP = {
    "app_key": "LSEG_APP_KEY",
}

out = Path(".streamlit")
out.mkdir(exist_ok=True)

lines = ["[keys]\n"]
found = 0
for toml_key, env_key in KEY_MAP.items():
    val = os.environ.get(env_key, "").strip()
    if val:
        lines.append(f'{toml_key} = "{val}"\n')
        found += 1

lseg_vals = {k: os.environ.get(v, "").strip() for k, v in LSEG_MAP.items()}
if any(lseg_vals.values()):
    lines.append("\n[lseg]\n")
    for toml_key, val in lseg_vals.items():
        if val:
            lines.append(f'{toml_key} = "{val}"\n')
            found += 1

(out / "secrets.toml").write_text("".join(lines))
print(f"secrets.toml written — {found} key(s) configured")
