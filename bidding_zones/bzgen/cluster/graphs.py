"""Per-country graphs for the annealer, built from the normalised edge table.

Nodes: the buses of a cluster-country that have at least one intra-country edge.
Buses whose every line crosses a border are *isolated within their country*
(8 in the current network); they cannot take part in a contiguous zone, so they
are excluded from annealing and take the zone of their nearest (geographic)
same-country bus afterwards.  Reported, not hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from bzgen.cluster.anneal import CountryGraph


def country_nodes(c: str, buses: pd.DataFrame, edges: pd.DataFrame) -> tuple[list, list]:
    e = edges[edges.country == c]
    has_edge = set(e.bus_a) | set(e.bus_b)
    allb = buses.index[buses.cluster_country == c]
    nodes = sorted(b for b in allb if b in has_edge)
    isolated = sorted(b for b in allb if b not in has_edge)
    return nodes, isolated


def build(c: str, buses: pd.DataFrame, edges: pd.DataFrame, alpha: float,
          Lbus: pd.Series, Gbus: pd.Series, dc_in_energy: bool = True):
    nodes, isolated = country_nodes(c, buses, edges)
    pos = {b: i for i, b in enumerate(nodes)}
    e = edges[(edges.country == c) & edges.bus_a.isin(pos) & edges.bus_b.isin(pos)]
    w = e.dp_n.to_numpy() - alpha * e.J_n.to_numpy()
    if not dc_in_energy:
        w = np.where(e.is_dc.to_numpy(), 0.0, w)
    ia = np.array([pos[b] for b in e.bus_a])
    ib = np.array([pos[b] for b in e.bus_b])
    L = Lbus.reindex(nodes).fillna(0.0).to_numpy(copy=True)   # pandas 3: views are read-only
    G = Gbus.reindex(nodes).fillna(0.0).to_numpy(copy=True)
    # isolated buses' load/generation counted with their host (nearest) node
    host = {}
    if isolated:
        xy = buses.loc[nodes, ["x", "y"]].to_numpy()
        t = cKDTree(np.c_[xy[:, 0] * np.cos(np.radians(xy[:, 1])), xy[:, 1]])
        for b in isolated:
            x, y = buses.at[b, "x"], buses.at[b, "y"]
            j = int(t.query([x * np.cos(np.radians(y)), y])[1])
            host[b] = nodes[j]
            L[j] += Lbus.get(b, 0.0)
            G[j] += Gbus.get(b, 0.0)
    g = CountryGraph(len(nodes), ia, ib, w, L, G)
    return g, nodes, host, e
