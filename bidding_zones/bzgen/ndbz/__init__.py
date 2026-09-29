"""Scenario-conditioned national dynamical bidding zones (nDBZ).

Re-zones each country on an outlier subset of the solved hours, anchored to the
static map A by a capacity-weighted Mirkin rigidity penalty.  Borders stay hard
and graphs stay national, exactly as in ``bzgen.cluster``; nothing here re-solves
the OPF or changes the static pipeline.

Configuration: ``config/ndbz.yaml`` deep-merged over ``config/default.yaml``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from bzgen import config

OVERLAY = config.ROOT / "config" / "ndbz.yaml"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(overlay: str | Path | None = OVERLAY, base: str | Path | None = None) -> dict:
    """``config/default.yaml`` (or ``base``) with the nDBZ overlay merged over it."""
    cfg = config.load(base)
    if overlay is not None:
        with open(overlay) as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def balance_floor(anneal_cfg: dict, k: int) -> float:
    """Balance-hinge floor on max(load_s, gen_s) / Ltot.

    ``balance_floor_frac_of_mean`` set (not null): that fraction of the mean zone
    share, ``frac / k`` (the European DBZ parameterisation); otherwise the static
    absolute ``balance_floor``.  0.15 / 3 = 0.05: identical to the static floor at
    k = 3.  Same semantics as ``bzgen.dbz.balance_floor``.
    """
    frac = anneal_cfg.get("balance_floor_frac_of_mean")
    if frac is None:
        return float(anneal_cfg["balance_floor"])
    return float(frac) / k


def results_dir(cfg: dict) -> Path:
    return config.ROOT / cfg["ndbz"]["results_dir"]


def figures_dir(cfg: dict) -> Path:
    return config.ROOT / cfg["ndbz"]["figures_dir"]


# --------------------------------------------------------------------------- #
# atomic writes: a cancelled run never leaves a half-written file behind
# --------------------------------------------------------------------------- #

def _atomic(path: Path, write) -> None:
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_text(path, text: str) -> None:
    _atomic(path, lambda p: Path(p).write_text(text))


def write_csv(path, df, **kw) -> None:
    _atomic(path, lambda p: df.to_csv(p, **kw))


def write_parquet(path, df) -> None:
    _atomic(path, lambda p: df.to_parquet(p))


def savefig(fig, path, **kw) -> None:
    _atomic(path, lambda p: fig.savefig(p, format=Path(path).suffix.lstrip("."), **kw))
