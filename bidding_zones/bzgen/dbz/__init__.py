"""Dynamical Bidding Zones (DBZ): zones re-optimised on one European graph with
national borders relaxed, anchored to the static map by a rigidity penalty.

A strict superset of the static pipeline (``bzgen.cluster``): the static
machinery is imported, never edited, except for the opt-in
``exclude_contiguity`` keyword of ``bzgen.cluster.anneal.initial_temperature``.
"""

from __future__ import annotations

import copy

import yaml

from bzgen import config

DBZ_YAML = config.ROOT / "config" / "dbz.yaml"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(base=None, overlay=DBZ_YAML) -> dict:
    """config/default.yaml with config/dbz.yaml deep-merged over it."""
    cfg = config.load(base)
    with open(overlay) as f:
        return config._check(_merge(cfg, yaml.safe_load(f) or {}))


def balance_floor(anneal_cfg: dict, k: int) -> float:
    """Balance-hinge floor on max(load_s, gen_s) / Ltot.

    If ``balance_floor_frac_of_mean`` is set (not null) the floor is that
    fraction of the mean zone share, ``frac / k``; otherwise the absolute
    ``balance_floor`` (the static pipeline's default, unchanged).
    """
    frac = anneal_cfg.get("balance_floor_frac_of_mean")
    if frac is None:
        return float(anneal_cfg["balance_floor"])
    return float(frac) / k
