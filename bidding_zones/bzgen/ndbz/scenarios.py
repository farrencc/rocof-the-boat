"""Scenarios: outlier subsets of the solved hours, and their edge tables.

A scenario is a named set of snapshots (with their weights) **per cluster-country**.
Its edge table is the existing ``edges.congestion_stats`` evaluated on that subset
(no OPF re-solve: ``data/solved/prices.parquet`` already holds every hour).

Selectors (``SELECTORS``), each on a per-hour feature of the country
(``scope: country``) or of the whole system (``scope: europe``):

* ``baseline``        all snapshots: the set the static map was built on
* ``peak_demand``     top q of hours by total load
* ``dunkelflaute``    bottom ``vre_q`` of VRE (wind + solar) capacity factor AND top
                      ``load_q`` of load
* ``wind_surplus``    top q of hours by wind (on- + offshore) capacity factor
* ``max_dispersion``  top q of hours by the cross-sectional std of nodal prices
                      (prices after the shedding policy, i.e. what the statistic sees)

"Top q of hours" is by weight: the shortest prefix of the sorted hours whose weight
reaches ``q_eff * total weight`` with ``q_eff = max(q, min_snapshots / total weight)``.
The dunkelflaute quantiles are widened together in steps of ``widen_step`` until the
weight reaches ``min_snapshots``.  The effective parameters are recorded.

Price clip (``scenario.clip_handling``)
---------------------------------------
``as_is`` uses the prices as the static pipeline does (``prepare.shedding_policy``,
default: clipped at +-``solve.price_clip``).  ``exclude_clipped`` drops every
edge-hour with at least one endpoint at the clip from that edge's statistic and
renormalises by the remaining weight (an edge left with no hours keeps its
``as_is`` value; the count is reported).  ``unclipped`` uses the solved prices
before the clip: ``prices.parquet`` stores the OPF duals unclipped (the clip is a
post-processing step in ``prepare``), so this is what an unclipped solve gives,
without re-solving.

Normalisation (``scenario.normalisation``)
------------------------------------------
``self``      ``edges.normalise`` on the scenario's own statistics: mean 1 per country
              by construction, so a scenario's overall severity is divided out.
``baseline``  every scenario's statistic is remedied and divided with the parameters
              of the *reference* (full year, ``as_is`` - the static pipeline's own
              table): clip threshold, median, rank table and country mean all come
              from the reference.  A more congested scenario therefore has a larger
              dp~ (mean > 1).  This is intended; see tests/test_ndbz.py.

``dp_n_baseline_ownclip`` (diagnostic, never selected by ``mode``): the ``clip``
remedy at the scenario's *own* quantile, divided by the reference mean.  Under the
reference clip, an extreme scenario can pin a large share of its edges at the
reference cap (``above_ref_clip``), which ties them in the Potts term.

J~ is topology only and identical across scenarios (asserted).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

from bzgen.cluster import edges as E

STAT_COL = {"mean": "dp_mean", "duration": "dp_duration", "quantile": "dp_quantile"}
EUROPE = "EU"          # feature column for scope=europe
CLIP_HANDLING = ("as_is", "exclude_clipped", "unclipped")


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #

@dataclass
class Inputs:
    """Everything a scenario needs, already aligned on the solved snapshot index."""
    prices_raw: pd.DataFrame        # [snapshot x bus], as solved
    prices: pd.DataFrame            # [snapshot x bus], after the shedding policy
    weights: pd.Series              # [snapshot], after the shedding policy
    buses: pd.DataFrame             # needs cluster_country
    edges: pd.DataFrame             # E.edge_table(...)
    features: dict = field(default_factory=dict)   # name -> [snapshot x (country..., EU)]


def _naive_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx


def capacity_factor(gens: pd.DataFrame, prof: pd.DataFrame, carriers, group: pd.Series) -> pd.DataFrame:
    """Capacity-weighted CF per group (cluster-country) and for the whole system."""
    g = gens[gens.carrier.isin(carriers) & gens.index.isin(prof.columns) & (gens.p_nom > 0)]
    grp = g.bus.map(group)
    num = (prof[g.index] * g.p_nom.to_numpy()).T.groupby(grp.to_numpy()).sum().T
    den = g.p_nom.groupby(grp.to_numpy()).sum()
    out = num / den
    out[EUROPE] = (prof[g.index] * g.p_nom.to_numpy()).sum(axis=1) / g.p_nom.sum()
    return out


def features(buses: pd.DataFrame, prices: pd.DataFrame, nat_load: pd.DataFrame,
             gens: pd.DataFrame, prof: pd.DataFrame) -> dict:
    """Per-hour selection features, columns = cluster-countries + ``EU``."""
    cc_of_country = buses.groupby("country").cluster_country.first()
    load = nat_load.T.groupby(nat_load.columns.map(cc_of_country).to_numpy()).sum().T
    load[EUROPE] = nat_load.sum(axis=1)
    cc = buses.cluster_country.reindex(prices.columns)
    disp = prices.T.groupby(cc.to_numpy()).std(ddof=0).T
    disp[EUROPE] = prices.std(axis=1, ddof=0)
    f = {"load": load,
         "wind_cf": capacity_factor(gens, prof, ["onwind", "offwind"], buses.cluster_country),
         "vre_cf": capacity_factor(gens, prof, ["onwind", "offwind", "solar"], buses.cluster_country),
         "dispersion": disp}
    return {k: v.reindex(prices.index) for k, v in f.items()}


def load_inputs(cfg: dict) -> Inputs:
    """Read the solved year and the solve inputs (never re-solves)."""
    from bzgen.cluster import prepare
    buses, lines, links, prices_raw, sn, shed = prepare.load_solution(cfg)
    P, w, _ = prepare.shedding_policy(prices_raw, sn, shed, buses, cfg)
    raw = prices_raw.reindex(P.index)
    nat = pd.read_parquet(prepare.INTERIM / "national_load.parquet")
    prof = pd.read_parquet(prepare.INTERIM / "res_profiles.parquet")
    gens = pd.read_csv(prepare.INTERIM / "generators.csv", index_col=0)
    nat.index = _naive_utc(nat.index)
    prof.index = _naive_utc(prof.index)
    feats = features(buses, P, nat, gens, prof)
    et = E.edge_table(buses, lines, links)
    return Inputs(prices_raw=raw, prices=P, weights=w, buses=buses, edges=et, features=feats)


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #

def weighted_quantile_1d(x: np.ndarray, w: np.ndarray, q: float) -> float:
    return float(E.weighted_quantile(x[:, None], w, q)[0])


def top_share(x: pd.Series, w: pd.Series, share: float) -> pd.DatetimeIndex:
    """Shortest prefix of hours sorted by x (descending, ties by time) whose weight
    reaches ``share`` of the total."""
    x = x.dropna()
    w = w.reindex(x.index)
    order = np.lexsort((np.arange(len(x)), -x.to_numpy()))
    cw = np.cumsum(w.to_numpy()[order])
    need = share * w.sum()
    m = int(np.searchsorted(cw, need - 1e-9) + 1)
    return x.index[np.sort(order[:min(m, len(x))])]


def _q_eff(cfg, W):
    s = cfg["scenario"]
    return max(float(s["q"]), min(1.0, float(s["min_snapshots"]) / W))


def _top_q(feature):
    def sel(inp: Inputs, col: str, cfg: dict):
        x = inp.features[feature].get(col)
        if x is None or x.isna().all() or np.nanstd(x.to_numpy()) == 0:
            return None, {"status": f"undefined: no {feature} signal"}
        q = _q_eff(cfg, inp.weights.sum())
        return top_share(x, inp.weights, q), {"q_eff": q}
    return sel


def _baseline(inp: Inputs, col: str, cfg: dict):
    return inp.weights.index, {"q_eff": 1.0}


def _dunkelflaute(inp: Inputs, col: str, cfg: dict):
    d = cfg["scenario"]["dunkelflaute"]
    vre = inp.features["vre_cf"].get(col)
    load = inp.features["load"].get(col)
    if vre is None or load is None or vre.isna().all() or load.isna().all():
        return None, {"status": "undefined: no VRE capacity"}
    w = inp.weights
    need = min(float(cfg["scenario"]["min_snapshots"]), w.sum())
    f = 0.0
    while True:
        vq = min(1.0, d["vre_q"] + f)
        lq = min(1.0, d["load_q"] + f)
        tv = weighted_quantile_1d(vre.to_numpy(), w.to_numpy(), vq)
        tl = weighted_quantile_1d(load.to_numpy(), w.to_numpy(), 1.0 - lq)
        m = (vre <= tv) & (load >= tl)
        if w[m].sum() >= need or (vq >= 1.0 and lq >= 1.0):
            break
        f += d["widen_step"]
    return w.index[m.to_numpy()], {"vre_q_eff": vq, "load_q_eff": lq}


SELECTORS = {
    "baseline": _baseline,
    "peak_demand": _top_q("load"),
    "dunkelflaute": _dunkelflaute,
    "wind_surplus": _top_q("wind_cf"),
    "max_dispersion": _top_q("dispersion"),
}


@dataclass
class Scenario:
    name: str
    scope: str
    snapshots: dict                 # cluster-country -> DatetimeIndex (None = undefined)
    params: dict                    # cluster-country -> dict of effective parameters


def select(inp: Inputs, name: str, cfg: dict, scope: str | None = None) -> Scenario:
    scope = scope or cfg["scenario"]["scope"]
    if scope not in ("country", "europe"):
        raise ValueError(scope)
    sel = SELECTORS[name]
    countries = sorted(inp.edges.country.unique())
    snaps, params = {}, {}
    if scope == "europe":
        idx, p = sel(inp, EUROPE, cfg)
        for c in countries:
            snaps[c], params[c] = idx, dict(p)
    else:
        for c in countries:
            snaps[c], params[c] = sel(inp, c, cfg)
    for c, idx in snaps.items():
        if idx is None:
            continue
        w = inp.weights.reindex(idx)
        params[c].update(n_snapshots=int(len(idx)), weight=float(w.sum()),
                         weight_share=float(w.sum() / inp.weights.sum()))
        params[c].setdefault("status", "ok")
    return Scenario(name, scope, snaps, params)


# --------------------------------------------------------------------------- #
# edge statistics on a subset
# --------------------------------------------------------------------------- #

def subset(prices: pd.DataFrame, weights: pd.Series, idx) -> tuple[pd.DataFrame, pd.Series]:
    """Row subset of prices and the *same* rows of the weights, with the checks that
    catch a silently misaligned weight vector."""
    idx = pd.DatetimeIndex(idx)
    assert idx.is_unique, "duplicate snapshots in a scenario"
    assert idx.isin(weights.index).all(), "scenario snapshot without a weight"
    assert idx.isin(prices.index).all(), "scenario snapshot without prices"
    P = prices.loc[idx]
    w = weights.loc[idx]
    assert w.index.equals(P.index)
    pos = weights.index.get_indexer(idx)
    assert np.array_equal(w.to_numpy(), weights.to_numpy()[pos]), "weights subset inconsistently"
    assert np.isfinite(w.to_numpy()).all() and (w.to_numpy() > 0).all()
    return P, w


def _check_comparable(st: pd.DataFrame) -> None:
    """Every statistic is normalised by the subset's total weight, so it does not
    grow with the number of hours: duration is a share of hours in [0, 1]; mean and
    quantile are bounded by the largest |dp| of the subset."""
    d = st.dp_duration.to_numpy()
    assert np.all((d >= 0) & (d <= 1 + 1e-12)), "dp_duration is not a share of hours"


def weighted_quantile_2d(x: np.ndarray, W: np.ndarray, q: float) -> np.ndarray:
    """Column-wise weighted quantile with per-column weights W [T x E] (the
    column-weight generalisation of ``edges.weighted_quantile``)."""
    o = np.argsort(x, axis=0)
    xs = np.take_along_axis(x, o, axis=0)
    cw = np.cumsum(np.take_along_axis(W, o, axis=0), axis=0)
    tot = cw[-1].copy()
    tot[tot == 0] = np.nan
    cw = cw / tot
    idx = (cw < q).sum(axis=0)
    out = xs[np.minimum(idx, x.shape[0] - 1), np.arange(x.shape[1])]
    return np.where(np.isnan(tot), np.nan, out)


def masked_congestion_stats(edges: pd.DataFrame, prices: pd.DataFrame, weights: pd.Series,
                            excluded: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """``edges.congestion_stats`` with bus-hours flagged in ``excluded`` [T x bus]
    dropped: an edge-hour counts only if neither endpoint is excluded, and each edge
    is normalised by its own remaining weight.  NaN where no hour remains."""
    w = weights.reindex(prices.index).to_numpy(float)
    P = prices.to_numpy(float)
    X = excluded.reindex(index=prices.index, columns=prices.columns).to_numpy(bool)
    col = {b: i for i, b in enumerate(prices.columns)}
    ia = np.array([col[b] for b in edges.bus_a])
    ib = np.array([col[b] for b in edges.bus_b])
    D = np.abs(P[:, ia] - P[:, ib])
    W = w[:, None] * ~(X[:, ia] | X[:, ib])
    ws = W.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        e = cfg["edges"]
        out = edges.copy()
        out["dp_mean"] = (W * D).sum(0) / ws
        out["dp_duration"] = (W * (D > e["duration_threshold"])).sum(0) / ws
        out["dp_quantile"] = weighted_quantile_2d(D, W, e["quantile"])
    out["kept_weight_share"] = ws / w.sum()
    return out


def at_clip(prices_raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Bus-hours at (beyond) the price clip, from the unclipped prices."""
    if cfg["solve"]["shedding_policy"] != "clip":
        return pd.DataFrame(False, index=prices_raw.index, columns=prices_raw.columns)
    cap = float(cfg["solve"]["price_clip"])
    return (prices_raw >= cap) | (prices_raw <= -cap)


def scenario_stats(inp: Inputs, scn: Scenario, cfg: dict, clip_handling: str | None = None
                   ) -> pd.DataFrame:
    """Edge table with dp_mean / dp_duration / dp_quantile over each country's scenario
    hours.  Row index = ``inp.edges`` index (the static edge id)."""
    ch = clip_handling or cfg["scenario"]["clip_handling"]
    if ch not in CLIP_HANDLING:
        raise ValueError(ch)
    out = []
    for c, ec in inp.edges.groupby("country"):
        idx = scn.snapshots.get(c)
        if idx is None:
            continue
        cols = sorted(set(ec.bus_a) | set(ec.bus_b))
        P, w = subset((inp.prices_raw if ch == "unclipped" else inp.prices)[cols], inp.weights, idx)
        st = E.congestion_stats(ec, P, w, cfg)
        if ch == "exclude_clipped":
            raw, _ = subset(inp.prices_raw[cols], inp.weights, idx)
            ms = masked_congestion_stats(ec, P, w, at_clip(raw, cfg), cfg)
            empty = ms.dp_duration.isna()
            for s in STAT_COL.values():
                ms[s] = ms[s].where(~empty, st[s])
            ms["no_unclipped_hours"] = empty
            st = ms
        _check_comparable(st)
        out.append(st)
    return pd.concat(out).sort_index()


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #

def remedy_fn(ref: np.ndarray, how: str | None, q: float):
    """``edges._remedy`` with its parameters frozen on ``ref``: ``f(ref)`` equals
    ``edges._remedy(ref, how, q)`` and ``f`` can be applied to other values."""
    if not how or how == "none":
        return lambda x: np.asarray(x, float)
    if how == "clip":
        thr = np.quantile(ref, q)
        return lambda x: np.minimum(np.asarray(x, float), thr)
    if how == "log1p":
        med = np.median(ref[ref > 0]) if (ref > 0).any() else 1.0
        return lambda x: np.log1p(np.asarray(x, float) / med)
    if how == "rank":
        s = np.sort(ref)
        n = len(s)
        def f(x):
            x = np.asarray(x, float)
            lo = np.searchsorted(s, x, "left")
            hi = np.searchsorted(s, x, "right")
            return (lo + (hi - lo + 1) / 2.0) / n
        return f
    raise ValueError(how)


def normalise(st: pd.DataFrame, ref: pd.DataFrame, cfg: dict, mode: str | None = None
              ) -> pd.DataFrame:
    """Scenario edge table with ``dp_n`` / ``J_n`` in the ``edges.normalise`` layout.

    ``ref`` is the reference (full-year) statistics table, same row index.  Adds
    ``dp_n_self`` and ``dp_n_baseline`` (both always computed) and sets ``dp_n`` to
    the one selected by ``mode``; ``dp_country_mean`` is the divisor actually used.
    """
    mode = mode or cfg["scenario"]["normalisation"]
    if mode not in ("baseline", "self"):
        raise ValueError(mode)
    e = cfg["edges"]
    stat = STAT_COL[e["statistic"]]
    out = E.normalise(st, cfg).sort_index()
    ref_n = E.normalise(ref.loc[out.index], cfg).sort_index()
    assert np.array_equal(out.J_n.to_numpy(), ref_n.J_n.to_numpy()), "J~ depends on the scenario"
    out["dp_n_self"] = out.dp_n
    out["dp_country_mean_self"] = out.dp_country_mean
    base = pd.Series(np.nan, index=out.index)
    mean_base = pd.Series(np.nan, index=out.index)
    above = pd.Series(False, index=out.index)
    own = pd.Series(np.nan, index=out.index)
    for c, g in out.groupby("country"):
        r = ref.loc[g.index, stat].to_numpy(float)
        f = remedy_fn(r, e.get("dp_remedy"), e.get("dp_clip_q", 0.99))
        m = f(r).mean()
        x = g[stat].to_numpy(float)
        base[g.index] = f(x) / m if m > 0 else 0.0
        mean_base[g.index] = m
        if e.get("dp_remedy") == "clip":
            above[g.index] = x > np.quantile(r, e.get("dp_clip_q", 0.99))
            # diagnostic variant: clip at the scenario's own quantile, divide by the
            # reference mean (keeps severity without pinning edges at the reference cap)
            own[g.index] = np.minimum(x, np.quantile(x, e.get("dp_clip_q", 0.99))) / m if m > 0 else 0.0
        else:
            own[g.index] = base[g.index]
    out["dp_n_baseline"] = base
    out["dp_country_mean_baseline"] = mean_base
    out["above_ref_clip"] = above
    out["dp_n_baseline_ownclip"] = own
    if mode == "baseline":
        out["dp_n"] = out.dp_n_baseline
        out["dp_country_mean"] = out.dp_country_mean_baseline
    out["normalisation"] = mode
    return out


def reference_stats(inp: Inputs, cfg: dict) -> pd.DataFrame:
    """Full-year, ``as_is`` statistics: the static pipeline's own table."""
    return scenario_stats(inp, select(inp, "baseline", cfg, "country"), cfg, "as_is")


def edge_table(inp: Inputs, cfg: dict, name: str, scope: str | None = None,
               clip_handling: str | None = None, normalisation: str | None = None,
               ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """The scenario edge table the annealer consumes (``graphs.build`` layout)."""
    ref = reference_stats(inp, cfg) if ref is None else ref
    scn = select(inp, name, cfg, scope)
    st = scenario_stats(inp, scn, cfg, clip_handling)
    return normalise(st, ref.loc[st.index], cfg, normalisation)


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #

def clip_diagnostic(inp: Inputs, scn: Scenario, cfg: dict) -> pd.DataFrame:
    """Per country: weighted share of bus-hours at the clip, of edge-hours with at least
    one / both endpoints at the clip, and share of edges with any such hour."""
    rows = []
    for c, ec in inp.edges.groupby("country"):
        idx = scn.snapshots.get(c)
        if idx is None:
            continue
        cols = sorted(set(ec.bus_a) | set(ec.bus_b))
        raw, w = subset(inp.prices_raw[cols], inp.weights, idx)
        X = at_clip(raw, cfg).to_numpy(bool)
        ww = w.to_numpy(float)[:, None]
        col = {b: i for i, b in enumerate(cols)}
        ia = np.array([col[b] for b in ec.bus_a])
        ib = np.array([col[b] for b in ec.bus_b])
        one = X[:, ia] | X[:, ib]
        both = X[:, ia] & X[:, ib] & (np.sign(raw.to_numpy()[:, ia]) == np.sign(raw.to_numpy()[:, ib]))
        W = ww.sum()
        rows.append({"country": c, "n_edges": len(ec),
                     "bus_hours_at_clip": float((ww * X).sum() / (W * X.shape[1])),
                     "edge_hours_one_end_clipped": float((ww * one).sum() / (W * one.shape[1])),
                     "edge_hours_both_ends_clipped_same_side": float((ww * both).sum() / (W * both.shape[1])),
                     "edges_any_clipped_hour": float(one.any(axis=0).mean()),
                     "edges_clipped_ge10pct_hours": float(((ww * one).sum(0) / W >= 0.1).mean())})
    return pd.DataFrame(rows).set_index("country")


def bootstrap(inp: Inputs, scn: Scenario, cfg: dict) -> pd.DataFrame:
    """Snapshot bootstrap of the configured statistic (``as_is``).

    Per country: CI of the country mean of the statistic, the between-edge std of the
    statistic, and the median per-edge bootstrap SE.  ``thin`` = CI width /
    between-edge std above ``bootstrap.thin_ratio``: the scenario's hours are too few
    to rank its edges, so it cannot support conclusions for that country.
    """
    b = cfg["scenario"]["bootstrap"]
    e = cfg["edges"]
    stat = e["statistic"]
    rng = np.random.default_rng(b["seed"])
    lo_q, hi_q = (1 - b["ci"]) / 2, 1 - (1 - b["ci"]) / 2
    rows = []
    for c, ec in inp.edges.groupby("country"):
        idx = scn.snapshots.get(c)
        if idx is None:
            continue
        cols = sorted(set(ec.bus_a) | set(ec.bus_b))
        P, w = subset(inp.prices[cols], inp.weights, idx)
        col = {bb: i for i, bb in enumerate(cols)}
        A = P.to_numpy(float)
        D = np.abs(A[:, [col[x] for x in ec.bus_a]] - A[:, [col[x] for x in ec.bus_b]])
        ww = w.to_numpy(float)
        T = len(ww)
        point = E.congestion_stats(ec, P, w, cfg)[STAT_COL[stat]].to_numpy()
        if stat in ("duration", "mean"):
            X = (D > e["duration_threshold"]).astype(float) if stat == "duration" else D
            reps = []
            for chunk in np.array_split(np.arange(b["n"]), max(1, b["n"] // 50)):
                C = np.stack([np.bincount(rng.integers(0, T, T), minlength=T) for _ in chunk])
                CW = C * ww
                reps.append((CW @ X) / CW.sum(1, keepdims=True))
            S = np.vstack(reps)
        else:
            S = np.stack([E.weighted_quantile(D, np.bincount(rng.integers(0, T, T), minlength=T) * ww,
                                              e["quantile"]) for _ in range(min(b["n"], 200))])
        cm = S.mean(1)
        lo, hi = np.quantile(cm, [lo_q, hi_q])
        spread = float(point.std())
        rows.append({"country": c, "n_snapshots": T, "weight": float(ww.sum()),
                     "mean_stat": float(point.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
                     "between_edge_std": spread,
                     "ci_width_over_edge_std": float((hi - lo) / spread) if spread > 0 else np.inf,
                     "median_edge_se_over_edge_std": float(np.median(S.std(0)) / spread) if spread > 0 else np.inf})
    df = pd.DataFrame(rows).set_index("country")
    df["thin"] = (df.ci_width_over_edge_std > b["thin_ratio"]) | ~np.isfinite(df.ci_width_over_edge_std)
    return df


def clip_comparison(as_is: pd.DataFrame, excl: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per country: does ``exclude_clipped`` change the per-edge statistic materially?"""
    stat = STAT_COL[cfg["edges"]["statistic"]]
    t = cfg["scenario"]["clip_material"]
    rows = []
    for c, g in as_is.groupby("country"):
        a = g[stat].to_numpy(float)
        x = excl.loc[g.index, stat].to_numpy(float)
        rho = sps.spearmanr(a, x).statistic if a.std() > 0 and x.std() > 0 else np.nan
        ma, mx = a.mean(), x.mean()
        rel = (mx - ma) / ma if ma > 0 else (0.0 if mx == 0 else np.inf)
        rows.append({"country": c, "mean_as_is": ma, "mean_excl": mx, "mean_rel_change": rel,
                     "spearman": rho, "max_abs_diff": float(np.abs(a - x).max()),
                     "edges_no_unclipped_hours": int(excl.loc[g.index].get(
                         "no_unclipped_hours", pd.Series(False, index=g.index)).sum())})
    df = pd.DataFrame(rows).set_index("country")
    df["material"] = (df.spearman.fillna(1.0) < t["spearman_min"]) | (df.mean_rel_change.abs() > t["mean_rel_change"])
    return df


def snapshot_table(scns: list[Scenario], inp: Inputs) -> pd.DataFrame:
    """Long table (scope, scenario, country, snapshot, weight); baseline omitted (all)."""
    parts = []
    for s in scns:
        if s.name == "baseline":
            continue
        for c, idx in s.snapshots.items():
            if idx is None:
                continue
            parts.append(pd.DataFrame({"scope": s.scope, "scenario": s.name, "country": c,
                                       "snapshot": idx, "weight": inp.weights.reindex(idx).to_numpy()}))
    return pd.concat(parts, ignore_index=True)


def params_table(scns: list[Scenario]) -> pd.DataFrame:
    rows = []
    for s in scns:
        for c, p in s.params.items():
            rows.append({"scope": s.scope, "scenario": s.name, "country": c, **p})
    return pd.DataFrame(rows)


def dumps(obj) -> str:
    return json.dumps(obj, indent=1, default=float)
