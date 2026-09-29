"""Solve the nodal-price year (or a representative subset) and persist the results.

    python -m src.solve.run            # uses solve.n_snapshots from the config

Snapshots: ``solve.n_snapshots = all`` solves all 8760 hours (weight 1 each);
an integer selects k-means representative hours (weight = cluster size).

Workers each build the PyPSA network and the LP once, then walk their block of
hours sequentially so HiGHS warm-starts from the previous hour's basis.

Persisted (``data/solved/``, gitignored; nothing downstream re-solves):
  prices.parquet        [snapshot x bus] EUR/MWh (float32)
  prices_seed2.parquet  degeneracy subsample, second cost-noise seed
  shed.parquet          long table (snapshot, bus, MW) of shedding > 0.01 MW
  gen_mean.csv          weighted mean dispatch per generator (MW)
  carrier_country.parquet  hourly dispatch per (country, carrier)
  line_stats.csv        per line: weighted mean |loading|, share of hours at limit
  link_stats.csv        same for HVDC links
  snapshots.csv         snapshot weights
"""

from __future__ import annotations

import json
import os
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

from src import config
from src.solve import opf, snapshots as snapmod
from src.solve.lp import DCOPF

SOLVED = config.ROOT / "data" / "solved"
_W = {}


def _worker_init(cfg, seed):
    os.environ["OMP_NUM_THREADS"] = "1"
    import logging
    logging.getLogger("pypsa").setLevel(logging.ERROR)
    import warnings
    warnings.filterwarnings("ignore")
    n = opf.build_network(cfg)
    rng = np.random.default_rng(seed)
    # noise in [noise/10, noise) keeps every cost clear of HiGHS's "tiny cost" range
    cn = cfg["solve"]["cost_noise"]
    costs = n.generators.mc_base.to_numpy() + rng.uniform(cn / 10, cn, len(n.generators))
    m = DCOPF(n, cfg["network"]["s_max_pu"], costs=costs)
    pmpu = n.get_switchable_as_dense("Generator", "p_max_pu")
    L = n.loads_t.p_set
    bi = pd.Series(np.arange(len(n.buses)), index=n.buses.index)
    _W.update(n=n, m=m, pmpu=pmpu, p_nom=n.generators.p_nom.to_numpy(), L=L,
              load_bus=bi[n.loads.bus].to_numpy(), nb=len(n.buses))


def _worker_solve(snaps):
    n, m = _W["n"], _W["m"]
    out = {"snaps": snaps, "price": [], "p": [], "line": [], "link": [], "status": [], "sec": [],
           "retry": []}
    for s in snaps:
        t = time.time()
        load = np.zeros(_W["nb"])
        np.add.at(load, _W["load_bus"], _W["L"].loc[s].to_numpy())
        r = m.solve(_W["pmpu"].loc[s].to_numpy() * _W["p_nom"], load)
        out["status"].append(r["status"])
        out["retry"].append(r.get("retry", ""))
        out["sec"].append(time.time() - t)
        if r["status"] != "ok":
            for k in ("price", "p", "line", "link"):
                out[k].append(None)
            continue
        out["price"].append(r["price"].astype(np.float32))
        out["p"].append(r["p"].astype(np.float32))
        out["line"].append(r["line_p"].astype(np.float32))
        out["link"].append(r["link_p"].astype(np.float32))
    return out


def select_snapshots(cfg) -> pd.DataFrame:
    nat = pd.read_parquet(opf.INTERIM / "national_load.parquet")
    idx = nat.index.tz_convert("UTC").tz_localize(None) if nat.index.tz is not None else nat.index
    ns = cfg["solve"]["n_snapshots"]
    if ns == "all":
        return pd.DataFrame({"weight": 1.0}, index=pd.Index(idx, name="snapshot"))
    gens = pd.read_csv(opf.INTERIM / "generators.csv", index_col=0)
    prof = pd.read_parquet(opf.INTERIM / "res_profiles.parquet")
    nat.index, prof.index = idx, idx
    return snapmod.representative_hours(snapmod.features(nat, gens, prof), int(ns))


def run_pool(snaps: list, cfg: dict, seed: int, workers: int, log=print) -> list:
    blocks = np.array_split(np.array(snaps, dtype=object), workers * 4)
    blocks = [list(b) for b in blocks if len(b)]
    res = []
    t0 = time.time()
    with get_context("fork").Pool(workers, initializer=_worker_init, initargs=(cfg, seed)) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker_solve, blocks)):
            res.append(r)
            log(f"  block {i + 1}/{len(blocks)} done ({sum(len(x['snaps']) for x in res)} h, "
                f"{time.time() - t0:.0f}s)")
    return res


def main():
    cfg = config.load()
    SOLVED.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    sn = select_snapshots(cfg)
    sn.to_csv(SOLVED / "snapshots.csv")
    log(f"{len(sn)} snapshots, total weight {sn.weight.sum():.0f} h")
    n = opf.build_network(cfg)
    res = run_pool(list(sn.index), cfg, cfg["anneal"]["seed"], cfg["solve"]["workers"], log)
    snaps, price, p, line, link, status, sec, retries = [], [], [], [], [], [], [], []
    for r in res:
        for j, s in enumerate(r["snaps"]):
            status.append((s, r["status"][j]))
            if r["retry"][j]:
                retries.append((str(s), r["retry"][j], r["status"][j]))
            sec.append(r["sec"][j])
            if r["status"][j] != "ok":
                continue
            snaps.append(s); price.append(r["price"][j]); p.append(r["p"][j])
            line.append(r["line"][j]); link.append(r["link"][j])
    bad = [s for s, st in status if st != "ok"]
    if bad:
        log(f"WARNING {len(bad)} snapshots not optimal: {bad[:5]}")
    order = np.argsort(np.array(snaps, dtype="datetime64[ns]"))
    idx = pd.DatetimeIndex(np.array(snaps, dtype="datetime64[ns]")[order], name="snapshot")
    P = np.vstack(price)[order]
    Gp = np.vstack(p)[order]
    F = np.vstack(line)[order]
    K = np.vstack(link)[order]
    w = sn.weight.reindex(idx).to_numpy()
    pd.DataFrame(P, index=idx, columns=n.buses.index).to_parquet(SOLVED / "prices.parquet")
    g = n.generators
    shed_cols = np.flatnonzero((g.carrier == "load_shedding").to_numpy())
    S = Gp[:, shed_cols]
    ti, gi = np.nonzero(S > 0.01)
    pd.DataFrame({"snapshot": idx[ti], "bus": g.bus.iloc[shed_cols].to_numpy()[gi],
                  "mw": S[ti, gi]}).to_parquet(SOLVED / "shed.parquet")
    pd.Series((w[:, None] * Gp).sum(0) / w.sum(), index=g.index, name="p_mean").to_csv(SOLVED / "gen_mean.csv")
    grp = g.bus.map(n.buses.country) + "|" + g.carrier
    cc = pd.DataFrame(Gp, index=idx, columns=g.index).T.groupby(grp.to_numpy()).sum().T
    cc.to_parquet(SOLVED / "carrier_country.parquet")
    lim = (n.lines.s_nom * cfg["network"]["s_max_pu"]).to_numpy()
    load = np.abs(F) / lim
    pd.DataFrame({"mean_loading": (w[:, None] * load).sum(0) / w.sum(),
                  "share_at_limit": (w[:, None] * (load > 0.999)).sum(0) / w.sum()},
                 index=n.lines.index).to_csv(SOLVED / "line_stats.csv")
    kl = np.abs(K) / n.links.p_nom.to_numpy()
    pd.DataFrame({"mean_loading": (w[:, None] * kl).sum(0) / w.sum(),
                  "share_at_limit": (w[:, None] * (kl > 0.999)).sum(0) / w.sum()},
                 index=n.links.index).to_csv(SOLVED / "link_stats.csv")
    meta = {"n_snapshots": len(idx), "not_optimal": len(bad),
            "not_optimal_snapshots": [str(b) for b in bad], "retries": retries, "solve_seconds_total": float(np.sum(sec)),
            "solve_seconds_median": float(np.median(sec)), "cost_noise": cfg["solve"]["cost_noise"]}
    # degeneracy subsample: second noise seed
    k = min(cfg["solve"]["degeneracy_sample"], len(idx))
    sub = list(pd.DatetimeIndex(np.random.default_rng(1).choice(idx, k, replace=False)).sort_values())
    log(f"degeneracy re-solve of {k} snapshots with a second noise seed")
    res2 = run_pool(sub, cfg, cfg["anneal"]["seed"] + 1, cfg["solve"]["workers"], log)
    s2, p2 = [], []
    for r in res2:
        for j, s in enumerate(r["snaps"]):
            if r["status"][j] == "ok":
                s2.append(s); p2.append(r["price"][j])
    pd.DataFrame(np.vstack(p2), index=pd.DatetimeIndex(s2, name="snapshot"),
                 columns=n.buses.index).sort_index().to_parquet(SOLVED / "prices_seed2.parquet")
    (SOLVED / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"done: {meta}")


if __name__ == "__main__":
    main()
