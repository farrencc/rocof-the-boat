"""Intra-country edge table, congestion statistics, per-country normalisation.

Edges
-----
One edge per unordered bus pair inside a (cluster-)country.  Parallel AC lines
are merged: ``J = sum 1/x`` (380 kV-equivalent ohms).  HVDC links are edges
with ``J = 0`` and ``is_dc = True``.  Cross-border lines and links are dropped
entirely — they never enter the energy or the normalisation statistics.

Congestion statistics of |p_i - p_j| (all snapshot-weighted):
* ``mean``      weighted mean
* ``duration``  weighted fraction of hours with |dp| > threshold
* ``quantile``  weighted q-quantile
The statistic used in the energy is ``edges.statistic`` in the config.

Normalisation (per country): ``dp~ = dp / mean(dp)`` over all intra-country
edges; ``J~ = J / mean(J)`` over AC edges.  Optional remedies (documented in
reports/normalisation.md) are applied *before* dividing by the mean:
``clip`` (at a per-country quantile), ``log1p`` (``log(1 + J/median J)``) and
``rank`` (mid-rank / n, so values are uniform on (0, 1]).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def edge_table(buses: pd.DataFrame, lines: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    cc = buses.cluster_country
    ac = lines.assign(c0=lines.bus0.map(cc), c1=lines.bus1.map(cc))
    ac = ac[ac.c0 == ac.c1]
    a = np.where(ac.bus0 < ac.bus1, ac.bus0, ac.bus1)
    b = np.where(ac.bus0 < ac.bus1, ac.bus1, ac.bus0)
    ac = pd.DataFrame({"bus_a": a, "bus_b": b, "country": ac.c0.values,
                       "J": 1.0 / ac.x.values, "s_nom": ac.s_nom.values,
                       "n_lines": 1, "line_ids": ac.index.values})
    ac = ac.groupby(["bus_a", "bus_b"]).agg(country=("country", "first"), J=("J", "sum"),
                                            s_nom=("s_nom", "sum"), n_lines=("n_lines", "sum"),
                                            line_ids=("line_ids", lambda s: ";".join(s)))
    ac["is_dc"] = False
    dc = links.assign(c0=links.bus0.map(cc), c1=links.bus1.map(cc))
    dc = dc[dc.c0 == dc.c1]
    a = np.where(dc.bus0 < dc.bus1, dc.bus0, dc.bus1)
    b = np.where(dc.bus0 < dc.bus1, dc.bus1, dc.bus0)
    dc = pd.DataFrame({"bus_a": a, "bus_b": b, "country": dc.c0.values, "J": 0.0,
                       "s_nom": dc.p_nom.values, "n_lines": 1, "line_ids": dc.index.values,
                       "is_dc": True}).set_index(["bus_a", "bus_b"])
    dc = dc[~dc.index.isin(ac.index)]      # an AC corridor already joins the pair
    return pd.concat([ac, dc]).reset_index()


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> np.ndarray:
    """Column-wise weighted quantile of x [T x E] with weights w [T]."""
    o = np.argsort(x, axis=0)
    xs = np.take_along_axis(x, o, axis=0)
    cw = np.cumsum(w[o], axis=0)
    cw /= cw[-1]
    idx = (cw < q).sum(axis=0)
    return xs[np.minimum(idx, len(w) - 1), np.arange(x.shape[1])]


def congestion_stats(edges: pd.DataFrame, prices: pd.DataFrame, weights: pd.Series,
                     cfg: dict) -> pd.DataFrame:
    """Add dp_mean, dp_duration, dp_quantile columns."""
    w = weights.reindex(prices.index).to_numpy(float)
    P = prices.to_numpy(float)
    col = {b: i for i, b in enumerate(prices.columns)}
    ia = np.array([col[b] for b in edges.bus_a])
    ib = np.array([col[b] for b in edges.bus_b])
    D = np.abs(P[:, ia] - P[:, ib])
    ws = w.sum()
    e = cfg["edges"]
    out = edges.copy()
    out["dp_mean"] = (w[:, None] * D).sum(0) / ws
    out["dp_duration"] = (w[:, None] * (D > e["duration_threshold"])).sum(0) / ws
    out["dp_quantile"] = weighted_quantile(D, w, e["quantile"])
    return out


def _remedy(x: np.ndarray, how: str | None, q: float) -> np.ndarray:
    if not how or how == "none":
        return x
    if how == "clip":
        return np.minimum(x, np.quantile(x, q))
    if how == "log1p":
        med = np.median(x[x > 0]) if (x > 0).any() else 1.0
        return np.log1p(x / med)
    if how == "rank":
        return stats.rankdata(x) / len(x)
    raise ValueError(how)


def normalise(edges: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-country normalised dp~ and J~ (with the configured remedies)."""
    e = cfg["edges"]
    stat = {"mean": "dp_mean", "duration": "dp_duration", "quantile": "dp_quantile"}[e["statistic"]]
    out = []
    for c, g in edges.groupby("country"):
        g = g.copy()
        dp = _remedy(g[stat].to_numpy(float), e.get("dp_remedy"), e.get("dp_clip_q", 0.99))
        m = dp.mean()
        g["dp_raw"] = g[stat]
        g["dp_n"] = dp / m if m > 0 else 0.0
        g["dp_country_mean"] = m
        ac = ~g.is_dc.to_numpy()
        J = g.J.to_numpy(float).copy()
        J[ac] = _remedy(J[ac], e.get("J_remedy"), e.get("J_clip_q", 0.99))
        mj = J[ac].mean() if ac.any() else 1.0
        g["J_n"] = np.where(ac, J / mj, 0.0)
        out.append(g)
    return pd.concat(out)


def diagnostics(norm: pd.DataFrame) -> pd.DataFrame:
    """Per country: mean (should be 1), skew, top-1 % share, max of dp~ and J~ (AC)."""
    rows = []
    for c, g in norm.groupby("country"):
        r = {"country": c, "n_edges": len(g), "n_dc": int(g.is_dc.sum())}
        for name, x in (("dp", g.dp_n.to_numpy()), ("J", g.J_n[~g.is_dc].to_numpy())):
            if len(x) == 0:
                continue
            top = max(1, int(np.ceil(0.01 * len(x))))
            s = np.sort(x)[::-1]
            r[f"{name}_mean"] = x.mean()
            r[f"{name}_skew"] = stats.skew(x) if len(x) > 2 and x.std() > 0 else np.nan
            r[f"{name}_top1pct_share"] = s[:top].sum() / s.sum() if s.sum() > 0 else np.nan
            r[f"{name}_max"] = s[0]
            r[f"{name}_zero_frac"] = float((x == 0).mean())
        rows.append(r)
    return pd.DataFrame(rows).set_index("country")
