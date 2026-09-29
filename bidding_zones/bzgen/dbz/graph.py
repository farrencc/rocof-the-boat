"""The European graph: one edge table over all buses, cross-border edges kept.

Edges
-----
Built exactly as ``bzgen.cluster.edges.edge_table`` (parallel AC lines merged,
``J = sum 1/x``, ``s_nom`` summed; HVDC links with ``J = 0``, dropped where an
AC corridor already joins the pair) except that lines and links whose two buses
lie in different cluster-countries are **kept**, and parallel HVDC links are
merged too (two cross-border corridors have two links each; the static table
never contained a parallel HVDC pair).  Extra columns:

* ``country_a``, ``country_b``: ``cluster_country`` of ``bus_a`` / ``bus_b``;
* ``is_cross``: ``country_a != country_b``;
* ``country``: the single cluster-country for intra-country edges (unchanged,
  so per-country grouping still works) and the string ``"XB"`` for every
  cross-border edge.  ``"XB"`` is never a country code; code that groups by
  ``country`` must select ``~is_cross`` first.

Normalisation (node-averaged means)
-----------------------------------
Per-country scales are computed from **intra-country edges only**, exactly as
``edges.normalise`` does: the remedy parameter (clip threshold / log1p median)
and the mean of the remedied statistic (``m_c``; for J over AC edges, ``mJ_c``).
Every edge is then normalised by the average over its two endpoints::

    theta_ij = (theta_c(i) + theta_c(j)) / 2       remedy parameter
    x_ij     = remedy(stat_ij; theta_ij)
    dp~_ij   = x_ij / ((m_c(i) + m_c(j)) / 2)

and the same for J~.  For an intra-country edge both averages are the country
value itself ((m + m) / 2 == m exactly in floating point), so intra-country
edges reproduce ``edges.normalise`` bit for bit.  Cross-border edges are only
*divided by* the scale; they never enter it.  ``rank`` remedies raise: a
within-country mid-rank has no meaning across a border.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bzgen.cluster.anneal import CountryGraph, contiguity_guarantee

XB = "XB"
STAT = {"mean": "dp_mean", "duration": "dp_duration", "quantile": "dp_quantile"}


# --------------------------------------------------------------------------- #
# edge table
# --------------------------------------------------------------------------- #

def edge_table(buses: pd.DataFrame, lines: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """European edge table (see module docstring). Intra-country rows equal
    ``bzgen.cluster.edges.edge_table``."""
    cc = buses.cluster_country
    a = np.where(lines.bus0 < lines.bus1, lines.bus0, lines.bus1)
    b = np.where(lines.bus0 < lines.bus1, lines.bus1, lines.bus0)
    ac = pd.DataFrame({"bus_a": a, "bus_b": b, "J": 1.0 / lines.x.values,
                       "s_nom": lines.s_nom.values, "n_lines": 1, "line_ids": lines.index.values})
    ac = ac.groupby(["bus_a", "bus_b"]).agg(J=("J", "sum"), s_nom=("s_nom", "sum"),
                                            n_lines=("n_lines", "sum"),
                                            line_ids=("line_ids", lambda s: ";".join(s)))
    ac["is_dc"] = False
    a = np.where(links.bus0 < links.bus1, links.bus0, links.bus1)
    b = np.where(links.bus0 < links.bus1, links.bus1, links.bus0)
    dc = pd.DataFrame({"bus_a": a, "bus_b": b, "s_nom": links.p_nom.values,
                       "n_lines": 1, "line_ids": links.index.values})
    # Parallel HVDC links are merged like parallel AC lines.  The static table never
    # needed this: its only parallel links cross a border and were dropped.  A
    # duplicated CSR neighbour breaks the annealer's contiguity counter (it assumes
    # distinct seeds), so the merge is required, not cosmetic.
    dc = dc.groupby(["bus_a", "bus_b"]).agg(s_nom=("s_nom", "sum"), n_lines=("n_lines", "sum"),
                                            line_ids=("line_ids", lambda s: ";".join(s)))
    dc["J"] = 0.0
    dc["is_dc"] = True
    dc = dc[~dc.index.isin(ac.index)]      # an AC corridor already joins the pair
    out = pd.concat([ac, dc]).reset_index()
    out["country_a"] = out.bus_a.map(cc).to_numpy()
    out["country_b"] = out.bus_b.map(cc).to_numpy()
    out["is_cross"] = out.country_a != out.country_b
    out["country"] = np.where(out.is_cross, XB, out.country_a)
    cols = ["bus_a", "bus_b", "country", "J", "s_nom", "n_lines", "line_ids", "is_dc",
            "country_a", "country_b", "is_cross"]
    return out[cols]


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #

def _remedy_param(x: np.ndarray, how: str | None, q: float) -> float:
    """The per-country parameter of a remedy, as ``edges._remedy`` computes it."""
    if not how or how == "none":
        return np.nan
    if how == "clip":
        return float(np.quantile(x, q))
    if how == "log1p":
        return float(np.median(x[x > 0])) if (x > 0).any() else 1.0
    if how == "rank":
        raise ValueError("remedy 'rank' is a within-country mid-rank and is not defined for "
                         "cross-border edges; choose none | clip | log1p for DBZ runs")
    raise ValueError(how)


def _apply_remedy(x: np.ndarray, how: str | None, theta: np.ndarray) -> np.ndarray:
    if not how or how == "none":
        return x
    if how == "clip":
        return np.minimum(x, theta)
    if how == "log1p":
        return np.log1p(x / theta)
    raise ValueError(how)


def country_scales(edges: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per cluster-country, from intra-country edges only: remedy parameters and
    normalising means for dp (all edges) and J (AC edges)."""
    e = cfg["edges"]
    stat = STAT[e["statistic"]]
    rows = []
    for c, g in edges[~edges.is_cross].groupby("country"):
        dp = g[stat].to_numpy(float)
        th_dp = _remedy_param(dp, e.get("dp_remedy"), e.get("dp_clip_q", 0.99))
        dpr = _apply_remedy(dp, e.get("dp_remedy"), th_dp)
        ac = ~g.is_dc.to_numpy()
        J = g.J.to_numpy(float)[ac]
        th_J = _remedy_param(J, e.get("J_remedy"), e.get("J_clip_q", 0.99)) if ac.any() else 1.0
        Jr = _apply_remedy(J, e.get("J_remedy"), th_J)
        rows.append({"country": c, "n_intra": len(g), "n_intra_ac": int(ac.sum()),
                     "dp_theta": th_dp, "dp_mean": float(dpr.mean()),
                     "J_theta": th_J, "J_mean": float(Jr.mean()) if ac.any() else 1.0,
                     "raw_stat_mean": float(dp.mean())})
    return pd.DataFrame(rows).set_index("country")


def normalise(edges: pd.DataFrame, cfg: dict, scales: pd.DataFrame | None = None) -> pd.DataFrame:
    """Node-averaged normalisation of every edge (intra and cross-border).

    Adds ``dp_raw, dp_n, dp_country_mean, J_n`` (as ``edges.normalise``) plus
    ``dp_scale, J_scale`` (the averaged denominators; equal to the country means
    for intra-country edges) and ``dp_theta, J_theta`` (averaged remedy parameters).
    """
    e = cfg["edges"]
    if e.get("cross_border_remedy", "average") != "average":
        raise ValueError(f"edges.cross_border_remedy={e.get('cross_border_remedy')!r}: "
                         "only 'average' is implemented")
    for r in ("dp_remedy", "J_remedy"):
        if e.get(r) == "rank":
            _remedy_param(np.zeros(1), "rank", 0.0)          # raises the explanatory error
    stat = STAT[e["statistic"]]
    sc = country_scales(edges, cfg) if scales is None else scales
    missing = (set(edges.country_a) | set(edges.country_b)) - set(sc.index)
    if missing:
        raise ValueError(f"countries with no intra-country edge, no scale defined: {sorted(missing)}")
    avg = lambda col: ((sc[col].reindex(edges.country_a).to_numpy()
                        + sc[col].reindex(edges.country_b).to_numpy()) / 2)
    out = edges.copy()
    x = out[stat].to_numpy(float)
    th = avg("dp_theta")
    xr = _apply_remedy(x, e.get("dp_remedy"), th)
    m = avg("dp_mean")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["dp_raw"] = out[stat]
        out["dp_n"] = np.where(m > 0, xr / m, 0.0)
    out["dp_country_mean"] = m
    out["dp_scale"] = m
    out["dp_theta"] = th
    ac = ~out.is_dc.to_numpy()
    thJ = avg("J_theta")
    J = out.J.to_numpy(float).copy()
    J[ac] = _apply_remedy(J[ac], e.get("J_remedy"), thJ[ac])
    mJ = avg("J_mean")
    out["J_n"] = np.where(ac, J / mJ, 0.0)
    out["J_scale"] = mJ
    out["J_theta"] = thJ
    return out


def no_signal_table(edges_n: pd.DataFrame, scales: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Static ``no_congestion_signal`` flag (intra-country edges only, unchanged)
    next to what the cross-border edges of each country carry."""
    smin = cfg["edges"].get("signal_min", 0.01)
    rows = []
    for c in scales.index:
        xb = edges_n[edges_n.is_cross & ((edges_n.country_a == c) | (edges_n.country_b == c))]
        rows.append({"country": c, "intra_edges": int(scales.at[c, "n_intra"]),
                     "intra_raw_mean": scales.at[c, "raw_stat_mean"],
                     "no_congestion_signal": bool(scales.at[c, "raw_stat_mean"] < smin),
                     "xb_edges": len(xb),
                     "xb_raw_mean": float(xb.dp_raw.mean()) if len(xb) else np.nan,
                     "xb_raw_max": float(xb.dp_raw.max()) if len(xb) else np.nan,
                     "xb_signal": bool(len(xb) and xb.dp_raw.mean() >= smin),
                     "xb_dp_n_max": float(xb.dp_n.max()) if len(xb) else np.nan})
    return pd.DataFrame(rows).set_index("country")


# --------------------------------------------------------------------------- #
# graph
# --------------------------------------------------------------------------- #

class EuropeGraph(CountryGraph):
    """CountryGraph over the whole European bus set, plus per-node installed
    capacity (rigidity weights), per-CSR-entry edge flags and the rigidity scale.

    ``Z = 2 N / K * mean(cap)``: the Mirkin penalty of one node of typical
    capacity moving between zones of typical size is O(1) after division by Z.
    """

    def __init__(self, n, edges_i, edges_j, weights, Lnode, Gnode, cap, K,
                 is_cross=None, is_dc=None):
        super().__init__(n, edges_i, edges_j, weights, Lnode, Gnode)
        self.cap = np.asarray(cap, np.float64)
        self.K = int(K)
        self.gbar = float(self.cap.mean()) if n else 0.0
        self.Z = 2.0 * n / self.K * self.gbar
        if not (np.isfinite(self.Z) and self.Z > 0):
            raise ValueError(f"rigidity scale Z={self.Z} (n={n}, K={K}, mean cap={self.gbar})")
        # no parallel edges: count_components_among needs distinct neighbour seeds
        dup = (np.diff(self.idx) == 0) & (np.diff(np.repeat(np.arange(n), np.diff(self.ptr))) == 0)
        if dup.any():
            raise ValueError(f"{int(dup.sum())} duplicate CSR entries (parallel edges): merge them")
        # per-CSR-entry flags, in the same (lexsorted) order as idx / w
        i = np.concatenate([edges_i, edges_j]).astype(np.int64)
        j = np.concatenate([edges_j, edges_i]).astype(np.int64)
        o = np.lexsort((j, i))
        f = lambda v: (np.zeros(len(o), np.bool_) if v is None
                       else np.concatenate([v, v]).astype(np.bool_)[o])
        self.is_cross = f(is_cross)
        self.is_dc = f(is_dc)
        # Workspace sanity: the BFS queue is (maxdeg, n) int64.
        qbytes = self.ws.Q.nbytes
        assert self.ws.Q.shape == (max(self.maxdeg, 1), n), self.ws.Q.shape
        assert qbytes < 64 * 2**20, f"Workspace.Q is {qbytes / 2**20:.1f} MiB"
        self.Q_bytes = qbytes

    def undirected(self):
        """(src, dst, w, is_cross, is_dc) with each undirected edge once."""
        src = np.repeat(np.arange(self.n), np.diff(self.ptr))
        und = src < self.idx
        return src[und], self.idx[und], self.w[und], self.is_cross[und], self.is_dc[und]


def node_order(buses: pd.DataFrame) -> list:
    """European node order: all buses, sorted by id.  Restricted to one country
    this is the static per-country order (``graphs.country_nodes`` sorts too)."""
    return sorted(buses.index)


def build(buses: pd.DataFrame, edges_n: pd.DataFrame, alpha: float, Lbus: pd.Series,
          Gbus: pd.Series, capbus: pd.Series, K: int, dc_in_energy: bool = True,
          nodes: list | None = None):
    """EuropeGraph over ``nodes`` (default: every bus) and the edges among them."""
    nodes = node_order(buses) if nodes is None else list(nodes)
    pos = {b: i for i, b in enumerate(nodes)}
    e = edges_n[edges_n.bus_a.isin(pos) & edges_n.bus_b.isin(pos)]
    w = e.dp_n.to_numpy() - alpha * e.J_n.to_numpy()
    if not dc_in_energy:
        w = np.where(e.is_dc.to_numpy(), 0.0, w)
    ia = np.array([pos[b] for b in e.bus_a], np.int64)
    ib = np.array([pos[b] for b in e.bus_b], np.int64)
    L = Lbus.reindex(nodes).fillna(0.0).to_numpy(copy=True)
    G = Gbus.reindex(nodes).fillna(0.0).to_numpy(copy=True)
    cap = capbus.reindex(nodes).fillna(0.0).to_numpy(copy=True)
    g = EuropeGraph(len(nodes), ia, ib, w, L, G, cap, K,
                    is_cross=e.is_cross.to_numpy(), is_dc=e.is_dc.to_numpy())
    deg = np.diff(g.ptr)
    if (deg == 0).any():
        bad = [nodes[i] for i in np.flatnonzero(deg == 0)[:10]]
        raise ValueError(f"{int((deg == 0).sum())} European nodes without any edge: {bad}")
    return g, nodes, e


def summary(g: EuropeGraph, lam_b: float, floor: float) -> dict:
    return {"N": g.n, "K": g.K, "Z": g.Z, "mean_cap_mw": g.gbar, "Ltot_mw": g.Ltot,
            "floor": floor, "maxdeg": g.maxdeg, "Q_bytes": g.Q_bytes,
            "contiguity_guarantee": contiguity_guarantee(g, lam_b)}


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #

def installed_capacity(buses: pd.DataFrame) -> pd.Series:
    """g_i: installed generation capacity per bus (MW), from data/interim/generators.csv
    (the assembled fleet; the OPF's load-shedding generators are not in it)."""
    from bzgen.cluster.sweep import INTERIM
    gens = pd.read_csv(INTERIM / "generators.csv", index_col=0)
    gens = gens[gens.carrier != "load_shedding"]
    return gens.groupby("bus").p_nom.sum().reindex(buses.index).fillna(0.0).rename("cap")


def compute_edges(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(buses, European normalised edges, country scales) from the solved prices.
    Reads data/solved (never re-solves); same shedding policy as the static gate."""
    from bzgen.cluster import edges as E
    from bzgen.cluster import prepare
    buses, lines, links, prices, sn, shed = prepare.load_solution(cfg)
    P, w, _ = prepare.shedding_policy(prices, sn, shed, buses, cfg)
    et = edge_table(buses, lines, links)
    st = E.congestion_stats(et, P, w, cfg)
    sc = country_scales(st, cfg)
    return buses, normalise(st, cfg, sc), sc
