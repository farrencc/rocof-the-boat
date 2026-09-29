"""Atomic file writes (temp file in the same directory + os.replace), so an
interrupt mid-write never leaves a truncated checkpoint behind."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[2] / "results" / "dbz"


def _tmp(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def _commit(tmp: Path, path: Path) -> None:
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = _tmp(path)
    t.write_text(text)
    _commit(t, path)


def write_json(path, obj) -> None:
    write_text(path, json.dumps(obj, indent=1, default=_default))


def write_csv(path, df: pd.DataFrame, **kw) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = _tmp(path)
    df.to_csv(t, **kw)
    _commit(t, path)


def write_parquet(path, df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = _tmp(path)
    df.to_parquet(t)
    _commit(t, path)


def write_npz(path, **arrays) -> None:
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = _tmp(path).with_suffix(".npz")        # np.savez appends .npz otherwise
    np.savez_compressed(t, **arrays)
    _commit(t, path)


def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
