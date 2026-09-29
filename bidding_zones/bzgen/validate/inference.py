"""Paired comparison, week-block bootstrap, and the three bands.

Pairing is by hour: the same hour has the same weather, load and fleet under
the k = 1 and the split map, so ``dDD_t = DD_split,t - DD_k1,t`` removes the
common weather signal.  Only hours in which stage B is feasible under both maps
enter (the LP-infeasible counts are reported next to every number).

Resampling is by ISO week (blocks keyed by ISO year and week), never by hour:
hourly congestion is strongly autocorrelated and an hour-level bootstrap gives
false precision.  ``validate.bootstrap.resamples`` resamples, percentile
interval at ``validate.bootstrap.ci``.  The statistic is the weighted mean of
dDD over the resampled hours scaled to a year (GWh/yr).

Bands around every headline number:
1. derating: the spread of the point estimate over ``validate.deratings``;
2. degeneracy: the spread over the headline map and its siblings;
3. sampling: the bootstrap interval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bzgen.validate.zonemap import iso_week


def paired(h_split: pd.DataFrame, h_k1: pd.DataFrame, col: str, w: pd.Series) -> pd.DataFrame:
    idx = h_split.index.intersection(h_k1.index)
    ok = h_split.stageB_ok.reindex(idx).astype(bool) & h_k1.stageB_ok.reindex(idx).astype(bool)
    idx = idx[ok.to_numpy()]
    d = (h_split.loc[idx, col] - h_k1.loc[idx, col]).rename("delta")
    wk = iso_week(pd.DatetimeIndex(idx))
    return pd.DataFrame({"delta": d, "w": w.reindex(idx).to_numpy(),
                         "block": wk.iso_year.astype(str) + "-W" + wk.iso_week.astype(str).str.zfill(2)})


def block_bootstrap(p: pd.DataFrame, cfg: dict) -> dict:
    """Point estimate and percentile CI of the annualised weighted mean of ``p.delta`` (GWh/yr)."""
    b = cfg["validate"]["bootstrap"]
    grp = p.assign(num=p.delta * p.w).groupby("block").agg(num=("num", "sum"), den=("w", "sum"))
    num, den = grp.num.to_numpy(), grp.den.to_numpy()
    scale = 8760.0 / 1e3
    point = num.sum() / den.sum() * scale if den.sum() > 0 else np.nan
    rng = np.random.default_rng(b["seed"])
    k = len(grp)
    if k < 2:
        return {"point": point, "lo": np.nan, "hi": np.nan, "n_blocks": k, "n_hours": len(p)}
    ii = rng.integers(0, k, size=(b["resamples"], k))
    stats = num[ii].sum(1) / den[ii].sum(1) * scale
    a = (1 - b["ci"]) / 2
    lo, hi = np.quantile(stats, [a, 1 - a])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "n_blocks": int(k),
            "n_hours": int(len(p)), "p_le_0": float((stats <= 0).mean())}


def bands(delta_rows: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per scoring set: headline estimate at the reference derating with the three spreads.

    ``delta_rows``: one row per (scoring, map_id, role, derating) with dDD point, lo, hi.
    """
    ref = cfg["validate"]["reference_derating"]
    out = []
    for s, g in delta_rows.groupby("scoring"):
        head = g[g.role == "headline"]
        h_ref = head[np.isclose(head.derating, ref)]
        if h_ref.empty:
            continue
        r = h_ref.iloc[0]
        sib_ref = g[np.isclose(g.derating, ref) & g.role.isin(["headline", "sibling"])]
        out.append({"scoring": s, "reference_derating": ref, "dDD_point": r.point,
                    "boot_lo": r.lo, "boot_hi": r.hi,
                    "derating_min": head.point.min(), "derating_max": head.point.max(),
                    "sibling_min": sib_ref.point.min(), "sibling_max": sib_ref.point.max(),
                    "n_maps_in_sibling_band": int(len(sib_ref)),
                    "ci_crosses_zero": bool(r.lo <= 0 <= r.hi),
                    "zero_in_sibling_band": bool(sib_ref.point.min() <= 0 <= sib_ref.point.max()),
                    "zero_in_derating_band": bool(head.point.min() <= 0 <= head.point.max())})
    return pd.DataFrame(out)


def verdict(b: pd.Series) -> str:
    """Headline sentence for one scoring set (negative dDD = splitting reduces DD)."""
    why = []
    if b.ci_crosses_zero:
        why.append("the bootstrap interval crosses zero")
    if b.zero_in_sibling_band:
        why.append("the k = 1 result (zero) lies inside the sibling band")
    if not why and b.boot_lo > 0:
        why.append("dispatch down is higher with DE split, and the interval excludes zero")
    if why:
        return ("Splitting DE does **not** demonstrably reduce dispatch down under this model: "
                + "; ".join(why) + ".")
    s = "Splitting DE reduces dispatch down under this model at the reference derating"
    if b.zero_in_derating_band:
        s += ", but not at every derating (the derating band contains zero)"
    return s + "."
