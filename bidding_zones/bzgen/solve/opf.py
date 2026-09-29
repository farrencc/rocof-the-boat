"""Parallel hour-by-hour DC-OPF for nodal prices (PyPSA + linopy + HiGHS).

Fixed capacities; no expansion, storage, unit commitment or CO2 constraint;
line flows limited to ``s_nom * s_max_pu``; HVDC links as lossless controllable
flows within ``+-p_nom``.  With no inter-temporal coupling every hour is an
independent LP.  Hours are grouped into small batches only to amortise model
building (a batch LP is block-diagonal, so its duals are exactly the per-hour
duals); batches are solved in parallel worker processes, one HiGHS thread each.

Load shedding: a generator at every bus with load, ``p_nom`` = that bus's peak
load, cost ``solve.load_shedding_cost``.  Ties: every generator's marginal cost
gets a seeded uniform perturbation in [0, ``cost_noise``) EUR/MWh.

Outputs (``data/solved/``, gitignored; nothing downstream re-solves):
prices.parquet [snapshot x bus], shed.parquet, gen_p.parquet, line_p.parquet,
link_p.parquet, snapshots.csv (weights), and prices_seed2.parquet for the
degeneracy subsample.
"""

from __future__ import annotations

import logging
import os
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

from bzgen import config

INTERIM = config.ROOT / "data" / "interim"
SOLVED = config.ROOT / "data" / "solved"

_NET = None


def _inputs():
    buses = pd.read_csv(INTERIM / "buses.csv", index_col=0)
    lines = pd.read_csv(INTERIM / "lines.csv", index_col=0)
    links = pd.read_csv(INTERIM / "links.csv", index_col=0)
    gens = pd.read_csv(INTERIM / "generators.csv", index_col=0)
    key = pd.read_csv(INTERIM / "load_key.csv", index_col=0).load_key
    nat = pd.read_parquet(INTERIM / "national_load.parquet")
    prof = pd.read_parquet(INTERIM / "res_profiles.parquet")
    # PyPSA snapshots must be tz-naive; everything here is UTC.
    for df in (nat, prof):
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
    return buses, lines, links, gens, key, nat, prof


def bus_load(buses, key, nat) -> pd.DataFrame:
    """Hourly load per bus [time x bus] (MW)."""
    L = nat[buses.country].to_numpy() * key.reindex(buses.index).to_numpy()[None, :]
    return pd.DataFrame(L, index=nat.index, columns=buses.index)


def build_network(cfg: dict):
    import pypsa
    buses, lines, links, gens, key, nat, prof = _inputs()
    load = bus_load(buses, key, nat)
    n = pypsa.Network()
    n.set_snapshots(nat.index)
    n.add("Carrier", list(gens.carrier.unique()) + ["load_shedding", "AC", "DC"])
    n.add("Bus", buses.index, v_nom=buses.v_nom, x=buses.x, y=buses.y, carrier="AC",
          country=buses.country)
    n.add("Line", lines.index, bus0=lines.bus0, bus1=lines.bus1, x=lines.x, r=lines.r,
          s_nom=lines.s_nom, s_max_pu=cfg["network"]["s_max_pu"])
    n.add("Link", links.index, bus0=links.bus0, bus1=links.bus1, p_nom=links.p_nom,
          p_min_pu=-1.0, efficiency=1.0, carrier="DC")
    has_load = load.columns[load.max() > 0]
    n.add("Load", has_load, bus=has_load, p_set=load[has_load])
    rng = np.random.default_rng(cfg["anneal"]["seed"])
    static = gens[~gens.index.isin(prof.columns)]
    tv = gens[gens.index.isin(prof.columns)]
    n.add("Generator", static.index, bus=static.bus, carrier=static.carrier, p_nom=static.p_nom,
          marginal_cost=static.marginal_cost, p_max_pu=static.p_max_pu.fillna(1.0))
    n.add("Generator", tv.index, bus=tv.bus, carrier=tv.carrier, p_nom=tv.p_nom,
          marginal_cost=tv.marginal_cost, p_max_pu=prof[tv.index])
    shed = [f"{b} load_shedding" for b in has_load]
    n.add("Generator", shed, bus=has_load, carrier="load_shedding",
          p_nom=load[has_load].max().to_numpy() + 1.0,
          marginal_cost=cfg["solve"]["load_shedding_cost"])
    n.generators["mc_base"] = n.generators.marginal_cost
    return n


def _init_worker(cfg):
    global _NET
    os.environ["OMP_NUM_THREADS"] = "1"
    logging.getLogger("linopy").setLevel(logging.WARNING)
    logging.getLogger("pypsa").setLevel(logging.WARNING)
    _NET = build_network(cfg)


def _solve(args):
    snaps, seed, cfg = args
    n = _NET
    rng = np.random.default_rng(seed)
    n.generators["marginal_cost"] = (n.generators.mc_base
                                     + rng.uniform(0, cfg["solve"]["cost_noise"], len(n.generators)))
    t = time.time()
    status, cond = n.optimize(snapshots=pd.DatetimeIndex(snaps), solver_name="highs", include_objective_constant=False,
                              solver_options={"threads": 1, "log_to_console": False,
                                              "output_flag": False})
    if status != "ok":
        return {"snaps": snaps, "status": f"{status}/{cond}"}
    sn = pd.DatetimeIndex(snaps)
    shed_cols = n.generators.index[n.generators.carrier == "load_shedding"]
    other = n.generators.index[n.generators.carrier != "load_shedding"]
    return {
        "snaps": snaps, "status": "ok", "seconds": time.time() - t,
        "prices": n.buses_t.marginal_price.loc[sn].copy(),
        "shed": n.generators_t.p.loc[sn, shed_cols].copy(),
        "gen_p": n.generators_t.p.loc[sn, other].astype("float32"),
        "line_p": n.lines_t.p0.loc[sn].astype("float32"),
        "link_p": n.links_t.p0.loc[sn].astype("float32"),
        "objective": float(n.objective),
    }


def solve_snapshots(snaps: pd.DatetimeIndex, cfg: dict, seed: int, workers: int, batch: int,
                    log=print) -> dict:
    chunks = [list(snaps[i:i + batch]) for i in range(0, len(snaps), batch)]
    tasks = [(c, seed + i, cfg) for i, c in enumerate(chunks)]
    out = []
    t0 = time.time()
    ctx = get_context("fork")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_solve, tasks)):
            out.append(r)
            if r["status"] != "ok":
                log(f"  batch {i}: {r['status']}  <-- FAILED")
            elif (i + 1) % max(1, len(tasks) // 10) == 0 or i + 1 == len(tasks):
                log(f"  {i + 1}/{len(tasks)} batches, {time.time() - t0:.0f}s elapsed")
    bad = [r for r in out if r["status"] != "ok"]
    if bad:
        raise RuntimeError(f"{len(bad)} batches failed: {[b['status'] for b in bad][:5]}")
    cat = lambda key: pd.concat([r[key] for r in out]).sort_index()
    return {k: cat(k) for k in ("prices", "shed", "gen_p", "line_p", "link_p")} | {
        "seconds": [r["seconds"] for r in out]}
