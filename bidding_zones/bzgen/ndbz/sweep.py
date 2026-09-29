"""Scenario x lambda_rigid sweep (per-country parallel, resumable, checkpointed).

This module currently provides the per-country context the annealer needs
(``Context``, ``country_problem``); the sweep driver lands in step 5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bzgen import config
from bzgen.cluster import graphs
from bzgen.cluster import sweep as static_sweep
from bzgen.ndbz import anchor as AN
from bzgen.ndbz import balance_floor, scenarios as S

ROOT = config.ROOT
INTERIM = ROOT / "data" / "interim"


class ScenarioUndefined(LookupError):
    """The scenario selects no hours for this country (e.g. wind_surplus without wind)."""


@dataclass
class Context:
    cfg: dict
    buses: pd.DataFrame
    static_edges: pd.DataFrame
    long: pd.DataFrame           # persisted scenario tables
    L: pd.Series
    G: pd.Series
    anchors: dict
    params: dict                 # anchor configuration: alpha, lambda_b, ...


def load_context(cfg: dict) -> Context:
    buses = pd.read_csv(INTERIM / "buses.csv", index_col=0)
    static = pd.read_parquet(ROOT / "results" / "edges.parquet")
    long = pd.read_parquet(S.persisted_path(cfg))
    L, G = static_sweep.node_attributes(cfg)
    anc = AN.anchors(cfg, buses=buses, edges=static)
    return Context(cfg, buses, static, long, L, G, anc, AN.anchor_params(cfg))


def country_problem(ctx: Context, c: str, scenario: str, scope=None, clip_handling=None,
                    normalisation=None):
    """(graph, nodes, host, anchor, floor, edges) for one country and scenario, at the
    anchor configuration's alpha (same physical Hamiltonian as A)."""
    ed = S.scenario_edges(ctx.cfg, scenario, scope, clip_handling, normalisation,
                          static=ctx.static_edges, long=ctx.long)
    if not (ed.country == c).any():
        raise ScenarioUndefined(f"{scenario} is undefined for {c}")
    g, nodes, host, e = graphs.build(c, ctx.buses, ed, ctx.params["alpha"], ctx.L, ctx.G,
                                     ctx.cfg["edges"]["dc_in_energy"])
    a = ctx.anchors[c]
    assert nodes == a.nodes and host == a.host, f"{c}: node set differs from the anchor's"
    floor = balance_floor(ctx.cfg["anneal"], a.k)
    return g, nodes, host, a, floor, e
