"""YAML configuration loader."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
