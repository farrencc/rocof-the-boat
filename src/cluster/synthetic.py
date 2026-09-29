"""Planted-partition benchmark for the annealer.

A random geometric graph in the unit square (a crude stand-in for a planar-ish
transmission grid) is split into ``k`` planted zones by the Voronoi cells of
``k`` random centres.  Edge statistics are planted:

* price gap: ``dp_cross`` on edges crossing a planted boundary, ``dp_in``
  inside, times lognormal noise;
* susceptance J: lognormal, independent of the zones (so J alone carries no
  information about the planted partition);

both normalised to mean 1 as in the real pipeline.  With ``w = dp~ - alpha J~``
the planted partition is (close to) the ground state when crossing gaps are
large, so recovery (adjusted Rand index) measures whether the annealer finds it.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from src.cluster.anneal import CountryGraph


def planted(n=300, k=4, radius=0.1, dp_in=0.2, dp_cross=3.0, noise=0.3, j_sigma=0.8, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2))
    G = nx.random_geometric_graph(n, radius, pos={i: tuple(p) for i, p in enumerate(xy)}, seed=seed)
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    keep = np.array(sorted(comps[0]))
    remap = {int(v): i for i, v in enumerate(keep)}
    xy = xy[keep]
    edges = np.array([(remap[a], remap[b]) for a, b in G.subgraph(keep).edges()])
    centres = rng.random((k, 2))
    truth = cKDTree(centres).query(xy)[1]
    cross = truth[edges[:, 0]] != truth[edges[:, 1]]
    dp = np.where(cross, dp_cross, dp_in) * rng.lognormal(0, noise, len(edges))
    J = rng.lognormal(0, j_sigma, len(edges))
    L = rng.exponential(1.0, len(keep))
    return {"xy": xy, "edges": edges, "truth": truth, "dp": dp / dp.mean(), "J": J / J.mean(),
            "L": L, "cross": cross}


def to_graph(p, alpha=1.0):
    w = p["dp"] - alpha * p["J"]
    return CountryGraph(len(p["xy"]), p["edges"][:, 0], p["edges"][:, 1], w, p["L"],
                        np.zeros(len(p["xy"])))
