"""The status-quo map A, per country, on the annealed node set.

A = ``results/configs/<ndbz.anchor_config_id>/labels.csv`` (default: the static
headline).  ``labels.csv`` covers every bus (3771); the annealer only sees the
buses with an intra-country edge (3763).  The 8 buses *isolated within their
country* are excluded from annealing and take their nearest same-country bus's
zone afterwards (``graphs.build`` returns that host mapping).  A is anchored on
the annealed nodes only, and the post-hoc assignment is checked to reproduce A on
the isolated buses exactly, so the scenario maps are finished the same way.

Per country: k_c = number of zones of A (asserted against ``results/sweep.csv``),
A contiguous (asserted via ``zone_components``), g_i = installed generation
capacity (MW, ``generators.csv``; an isolated bus's capacity goes to its host, as
``graphs.build`` does for load and generation), and the rigidity normaliser Z_c.
Countries with k_c = 1 are skipped (rigidity is identically 0, no move is possible).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bzgen import config
from bzgen.cluster import graphs
from bzgen.cluster.anneal import zone_components
from bzgen.ndbz import rigidity as R

RESULTS = config.ROOT / "results"
INTERIM = config.ROOT / "data" / "interim"


@dataclass
class CountryAnchor:
    country: str
    nodes: list                 # annealed buses, graph order
    host: dict                  # isolated bus -> host node
    A: np.ndarray               # int64 labels 0..k-1 on nodes
    zone_ids: list              # original zone id of each A label
    k: int
    cap: np.ndarray             # g_i, MW
    Z: float
    gbar: float
    gbar_flagged: bool
    skip: str | None = None
    info: dict = field(default_factory=dict)


def anchor_params(cfg: dict) -> dict:
    """Parameters of the anchor configuration (alpha, lambda_b, ...) from its done.json."""
    cid = cfg["ndbz"]["anchor_config_id"]
    return json.loads((RESULTS / "configs" / cid / "done.json").read_text())["params"]


def load_labels(cfg: dict) -> pd.DataFrame:
    cid = cfg["ndbz"]["anchor_config_id"]
    return pd.read_csv(RESULTS / "configs" / cid / "labels.csv")


def capacity_per_bus(buses: pd.DataFrame, gens: pd.DataFrame | None = None) -> pd.Series:
    """Installed generation capacity per bus (MW), every carrier in the fleet."""
    gens = pd.read_csv(INTERIM / "generators.csv", index_col=0) if gens is None else gens
    return gens.p_nom.groupby(gens.bus).sum().reindex(buses.index).fillna(0.0)


def attach_isolated(lab: pd.Series, host: dict) -> pd.Series:
    """The static pipeline's post-hoc assignment: an isolated bus takes its host's zone."""
    out = lab.copy()
    for b, h in host.items():
        out[b] = lab[h]
    return out


def country_anchor(c: str, labels: pd.DataFrame, buses: pd.DataFrame, edges: pd.DataFrame,
                   cap_bus: pd.Series, k_expected: int | None, g=None, nodes=None, host=None
                   ) -> CountryAnchor:
    """A for one country.  ``g, nodes, host``: a ``graphs.build`` result (built here
    from the static edge table if not given; only the node set and the topology are
    used, which do not depend on the scenario)."""
    if g is None:
        zero = pd.Series(0.0, index=buses.index)
        g, nodes, host, _ = graphs.build(c, buses, edges, 1.0, zero, zero)
    lab = labels[labels.country == c].set_index("bus").zone
    missing = [b for b in nodes if b not in lab.index]
    assert not missing, f"{c}: annealed nodes without an anchor label: {missing[:5]}"
    extra = set(lab.index) - set(nodes) - set(host)
    assert not extra, f"{c}: anchor labels on buses that are neither annealed nor isolated: {sorted(extra)[:5]}"
    # the post-hoc assignment of isolated buses reproduces A
    re = attach_isolated(lab.reindex(nodes), host)
    for b in host:
        assert re[b] == lab[b], f"{c}: isolated bus {b} not in its host's zone"
    zone_ids = sorted(lab.reindex(nodes).unique())
    k = len(zone_ids)
    assert lab.nunique() == k, f"{c}: a zone of A lives only on isolated buses"
    if k_expected is not None:
        assert k == k_expected, f"{c}: anchor has {k} zones, sweep.csv says k = {k_expected}"
    code = {z: i for i, z in enumerate(zone_ids)}
    A = lab.reindex(nodes).map(code).to_numpy(np.int64)
    cap = cap_bus.reindex(nodes).fillna(0.0).to_numpy(copy=True)
    pos = {b: i for i, b in enumerate(nodes)}
    for b, h in host.items():
        cap[pos[h]] += float(cap_bus.get(b, 0.0))
    Z, gbar, flagged = R.normaliser(cap, k)
    a = CountryAnchor(c, list(nodes), dict(host), A, zone_ids, k, cap, Z, gbar, flagged,
                      info={"n_nodes": len(nodes), "n_isolated": len(host),
                            "zone_sizes": np.bincount(A, minlength=k).tolist(),
                            "cap_total_mw": float(cap.sum()),
                            "cap_zero_share": float((cap == 0).mean())})
    if k == 1:
        a.skip = "k_c = 1: rigidity is identically zero and no move is possible"
        return a
    C = zone_components(A, k, g.ptr, g.idx)
    assert (C == 1).all(), f"{c}: anchor zone(s) not contiguous: components {C.tolist()}"
    return a


def anchors(cfg: dict, buses: pd.DataFrame | None = None, edges: pd.DataFrame | None = None,
            countries=None, graphs_of: dict | None = None) -> dict[str, CountryAnchor]:
    """A for every country of the anchor configuration."""
    buses = pd.read_csv(INTERIM / "buses.csv", index_col=0) if buses is None else buses
    edges = pd.read_parquet(RESULTS / "edges.parquet") if edges is None else edges
    labels = load_labels(cfg)
    sweep = pd.read_csv(RESULTS / "sweep.csv")
    rows = sweep[sweep.config_id == cfg["ndbz"]["anchor_config_id"]].set_index("country")
    cap_bus = capacity_per_bus(buses)
    out = {}
    for c in countries or sorted(labels.country.unique()):
        kw = {}
        if graphs_of and c in graphs_of:
            g, nodes, host = graphs_of[c][:3]
            kw = dict(g=g, nodes=nodes, host=host)
        k_exp = int(rows.loc[c, "k"]) if c in rows.index else None
        out[c] = country_anchor(c, labels, buses, edges, cap_bus, k_exp, **kw)
    return out


def summary(anc: dict[str, CountryAnchor]) -> pd.DataFrame:
    return pd.DataFrame([{"country": c, "n_nodes": a.info["n_nodes"], "n_isolated": a.info["n_isolated"],
                          "k": a.k, "zone_sizes": "/".join(map(str, a.info["zone_sizes"])),
                          "cap_total_mw": a.info["cap_total_mw"],
                          "cap_zero_share": a.info["cap_zero_share"], "gbar_mw": a.gbar,
                          "gbar_flagged": a.gbar_flagged, "Z_c": a.Z, "skip": a.skip}
                         for c, a in anc.items()]).set_index("country")
