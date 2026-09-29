"""YAML configuration loader."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config" / "default.yaml"
    with open(path) as f:
        return _check(yaml.safe_load(f))


def _check(cfg: dict) -> dict:
    for c in cfg["scope"]["countries"] + cfg.get("validate", {}).get("split_real", []):
        if not isinstance(c, str):
            raise ValueError(f"country code parsed as {c!r}: quote it in the YAML")
    return cfg
