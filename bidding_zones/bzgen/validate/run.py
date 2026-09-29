"""Validation runner: nodal reference, maps, stage A + B per (map, derating), metrics.

    python -m bzgen.validate.run                      # everything, all hours
    python -m bzgen.validate.run --n-snapshots 200    # k-means representative hours (dev)
    python -m bzgen.validate.run --stage A            # market stage only
    python -m bzgen.validate.run --check              # PyPSA/linopy cross-check on 2 hours
    python -m bzgen.validate.run --report-only        # aggregate cached runs

Snapshots: the nodal benchmark's selection (``bzgen.solve.run.select_snapshots``);
``--n-snapshots`` overrides ``solve.n_snapshots`` for development only (the
week-block bootstrap is meaningless on weighted representative hours).

Parallelism, blocks, warm start and cost noise are the nodal benchmark's
(``bzgen.solve.run``): worker processes each build the network once, then walk
a block of hours; costs are ``mc + U[noise/10, noise)`` with ``anneal.seed``.
The nodal reference is re-solved here with that machinery because the per-hour
objective is not persisted by ``bzgen.solve.run``; its prices are checked
against ``data/solved/prices.parquet``.

Caching (``data/validate/<tag>/``, gitignored; ``tag`` = ``all`` or ``rep<N>``):
``nodal/`` once, then ``runs/<map_id>_d<derating>/`` per (map, derating) with a
``done.json``; a finished run is never recomputed, so the runner resumes where it
stopped.  Scoring sets are views on these runs: the k = 1 map needs no refit, so
its run over all hours serves both the in-sample and the out-of-sample (even
ISO weeks) scoring set.  Committed: ``results/validate/`` (maps, restarts, real
zones), ``results/validate.csv``, ``figures/validate/``, ``reports/validate.md``.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

from bzgen import config
from bzgen.solve import run as solve_run
from bzgen.validate import metrics as M
from bzgen.validate import zonemap
from bzgen.validate.market import ZonalMarket, noise_costs
from bzgen.validate.redispatch import Redispatch

ROOT = config.ROOT
INTERIM = ROOT / "data" / "interim"
SOLVED = ROOT / "data" / "solved"
DATA = ROOT / "data" / "validate"
RES = ROOT / "results" / "validate"

_V = {}


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


# --------------------------------------------------------------------------- #
# nodal reference (the benchmark's own workers)
# --------------------------------------------------------------------------- #

def _nodal_block(snaps):
    W = solve_run._W
    n, m = W["n"], W["m"]
    out = {"snaps": snaps, "obj": [], "price": [], "p": [], "status": []}
    for s in snaps:
        load = np.zeros(W["nb"])
        np.add.at(load, W["load_bus"], W["L"].loc[s].to_numpy())
        r = m.solve(W["pmpu"].loc[s].to_numpy() * W["p_nom"], load)
        out["status"].append(r["status"])
        ok = r["status"] == "ok"
        out["obj"].append(r["objective"] if ok else np.nan)
        out["price"].append(r["price"].astype(np.float32) if ok else None)
        out["p"].append(r["p"] if ok else None)
    return out


def nodal_reference(cfg, sn: pd.DataFrame, out) -> dict:
    d = out / "nodal"
    if (d / "done.json").exists():
        return json.loads((d / "done.json").read_text())
    d.mkdir(parents=True, exist_ok=True)
    seed = cfg["anneal"]["seed"]
    import pypsa  # noqa: F401  (fork after import)
    from bzgen.solve import opf
    n = opf.build_network(cfg)
    costs = noise_costs(n, cfg)
    snaps = list(sn.index)
    blocks = [list(b) for b in np.array_split(np.array(snaps, dtype=object),
                                              cfg["solve"]["workers"] * 4) if len(b)]
    res = []
    t0 = time.time()
    with get_context("fork").Pool(cfg["solve"]["workers"], initializer=solve_run._worker_init,
                                  initargs=(cfg, seed)) as pool:
        for i, r in enumerate(pool.imap_unordered(_nodal_block, blocks)):
            res.append(r)
            log(f"  nodal block {i + 1}/{len(blocks)} ({time.time() - t0:.0f}s)")
    rows = [(s, r["obj"][j], r["price"][j], r["p"][j], r["status"][j])
            for r in res for j, s in enumerate(r["snaps"])]
    rows.sort(key=lambda x: x[0])
    bad = [str(x[0]) for x in rows if x[4] != "ok"]
    if bad:
        raise RuntimeError(f"nodal reference not optimal in {len(bad)} hours: {bad[:5]}")
    idx = pd.DatetimeIndex([x[0] for x in rows], name="snapshot")
    P = np.vstack([x[2] for x in rows])
    Gp = np.vstack([x[3] for x in rows])
    g = n.generators
    obj = pd.Series([x[1] for x in rows], index=idx, name="C_nodal")
    # identical machinery => identical costs => identical objective
    chk = np.abs(Gp @ costs - obj.to_numpy()) / np.maximum(1.0, np.abs(obj.to_numpy()))
    if chk.max() > 1e-6:
        raise AssertionError(f"noise costs differ from the benchmark's (rel {chk.max():.2e})")
    meta = {"hours": len(idx)}
    pf = SOLVED / "prices.parquet"
    if pf.exists():
        ref = pd.read_parquet(pf).reindex(index=idx, columns=n.buses.index)
        dp = np.abs(ref.to_numpy() - P)
        meta["price_check_max_abs"] = float(np.nanmax(dp))
        meta["price_check_share_gt_0.01"] = float((dp > 0.01).mean())
    obj.to_frame().to_parquet(d / "cost.parquet")
    buses = pd.read_csv(INTERIM / "buses.csv", index_col=0)
    focus = buses.index[buses.cluster_country == cfg["validate"]["focus"]]
    fg = g.bus.isin(focus).to_numpy()
    pd.DataFrame(Gp[:, fg].astype(np.float32), index=idx, columns=g.index[fg]).to_parquet(d / "focus_gen_p.parquet")
    w = sn.weight.reindex(idx).to_numpy()
    odd = zonemap.iso_week(idx).iso_week.to_numpy() % 2 == 1
    pd.Series((w[:, None] * Gp).sum(0) / w.sum(), index=g.index).to_csv(d / "gen_mean_all.csv")
    if odd.any():
        pd.Series((w[odd, None] * Gp[odd]).sum(0) / w[odd].sum(), index=g.index).to_csv(d / "gen_mean_odd.csv")
    sh = (g.carrier == "load_shedding").to_numpy()
    S = Gp[:, sh]
    tol = cfg["solve"]["shed_tol_mw"]
    share = pd.Series((w[:, None] * (S > tol)).sum(0) / w.sum(), index=g.bus[sh].to_numpy())
    share.groupby(level=0).max().rename("shed_hour_share").to_csv(d / "shed_share.csv")
    pmpu = n.get_switchable_as_dense("Generator", "p_max_pu").loc[idx]
    avail = pmpu.to_numpy() * g.p_nom.to_numpy()
    res_ = fg & g.carrier.isin(cfg["validate"]["res_carriers"]).to_numpy()
    shf = fg & sh
    pd.DataFrame({"avail_res": avail[:, res_].sum(1), "spill_res": (avail - Gp)[:, res_].sum(1),
                  "shed_focus": Gp[:, shf].sum(1), "shed_total": S.sum(1)},
                 index=idx).to_parquet(d / "hourly.parquet")
    (d / "done.json").write_text(json.dumps(meta, indent=1))
    log(f"nodal reference: {meta}")
    return meta


# --------------------------------------------------------------------------- #
# stage A + B per (map, derating)
# --------------------------------------------------------------------------- #

def _init(cfg, seed, zmap, derating, pocket, focus_bus, stages):
    solve_run._worker_init(cfg, seed)
    W = solve_run._W
    n = W.pop("m").n
    costs = noise_costs(n, cfg, seed)
    zm = ZonalMarket(n, zmap, derating, cfg, costs)
    rd = Redispatch(n, zmap, cfg, costs) if "B" in stages else None
    ctx = M.Context(n, zmap, cfg, pocket, zm.nz.links, focus_bus)
    zg = zmap.reindex(n.buses.index)[n.generators.bus].to_numpy()
    zi = {z: i for i, z in enumerate(zm.zones)}
    _V.update(n=n, zm=zm, rd=rd, ctx=ctx, cfg=cfg, gen_zone=np.array([zi[z] for z in zg]),
              shed=(n.generators.carrier == "load_shedding").to_numpy(),
              res_all=n.generators.carrier.isin(cfg["validate"]["res_carriers"]).to_numpy(),
              pocket_gen=n.generators.bus.isin(pocket).to_numpy(),
              fzm=np.isin(rd.zones, list(ctx.focus_zones)) if rd is not None else None)


def _sparse(t, a, tol):
    j = np.flatnonzero(np.abs(a) > tol)
    return t, j.astype(np.int32), a[j].astype(np.float32)


def _block(snaps):
    W, V = solve_run._W, _V
    zm, rd, ctx, cfg = V["zm"], V["rd"], V["ctx"], V["cfg"]
    tol = cfg["validate"]["spill_tol_mw"]
    dt = cfg["validate"]["dual_tol"]
    Z = len(zm.zones)
    out = {"snaps": snaps, "rows": [], "pz": [], "zp": [], "np": [], "spill": [], "rd": [],
           "mu": [], "zmu": [], "shed_zone": []}
    for s in snaps:
        load = np.zeros(W["nb"])
        np.add.at(load, W["load_bus"], W["L"].loc[s].to_numpy())
        pmax = W["pmpu"].loc[s].to_numpy() * W["p_nom"]
        t = time.time()
        a = zm.solve(pmax, load)
        if a["status"] != "ok":
            raise RuntimeError(f"stage A not optimal at {s}: {a['status']} (a transport model with "
                               "load shedding is always feasible: this is a bug)")
        b = rd.solve(pmax, load, a["p"]) if rd is not None else None
        row = M.hour_metrics(ctx, pmax, a, b)
        row.update(snapshot=s, stageA_retry=a.get("retry", ""),
                   stageB_status="" if b is None else b["status"],
                   stageB_retry="" if b is None else b.get("retry", ""), seconds=time.time() - t)
        ok_b = b is not None and b["status"] == "ok"
        row["np_slack_total"] = float(b["np_slack"].sum()) if ok_b else np.nan
        row["np_slack_focus"] = float(b["np_slack"][V["fzm"]].sum()) if ok_b else np.nan
        if ok_b:
            out.setdefault("slack", []).append((s, b["np_slack"]))
        out["rows"].append(row)
        pz = np.clip(a["p"], 0.0, pmax)
        out["pz"].append(pz.astype(np.float32))
        out["zp"].append(a["price"].astype(np.float32))
        out["np"].append(a["net_position"].astype(np.float32))
        sp = np.where(V["res_all"], pmax - pz, 0.0)
        out["spill"].append(_sparse(s, sp, tol))
        out["zmu"].append(_sparse(s, a["link_mu"], dt))
        if b is not None and b["status"] == "ok":
            j = np.flatnonzero((b["up"] > tol) | (b["down"] > tol))
            out["rd"].append((s, j.astype(np.int32), b["up"][j].astype(np.float32),
                              b["down"][j].astype(np.float32)))
            mu = np.r_[b["line_mu"], b["link_mu"]]
            out["mu"].append(_sparse(s, mu, dt))
            final = pz + b["up"] - b["down"]
            sz = np.bincount(V["gen_zone"][V["shed"]], weights=final[V["shed"]], minlength=Z)
            pk = np.bincount(V["gen_zone"][V["shed"] & V["pocket_gen"]],
                             weights=final[V["shed"] & V["pocket_gen"]], minlength=Z)
            out["shed_zone"].append((s, sz, pk))
    return out


def _long(parts, cols, idx_names):
    if not parts:
        return pd.DataFrame(columns=cols)
    s = np.concatenate([np.full(len(p[1]), p[0], dtype="datetime64[ns]") for p in parts])
    j = np.concatenate([p[1] for p in parts])
    df = pd.DataFrame({"snapshot": s, cols[1]: np.asarray(idx_names)[j]})
    for k, c in enumerate(cols[2:]):
        df[c] = np.concatenate([p[2 + k] for p in parts])
    return df


def run_one(cfg, sn, out, map_id, zmap, derating, hours, pocket, focus_bus, stages, workers):
    rid = f"{map_id}_d{derating:g}"
    d = out / "runs" / rid
    done = d / "done.json"
    if done.exists():
        meta = json.loads(done.read_text())
        if set(stages) <= set(meta["stages"]) and meta["hours"] == len(hours):
            log(f"skip {rid} (done)")
            return d
    d.mkdir(parents=True, exist_ok=True)
    seed = cfg["anneal"]["seed"]
    blocks = [list(b) for b in np.array_split(np.array(list(hours), dtype=object), workers * 4) if len(b)]
    res = []
    t0 = time.time()
    with get_context("fork").Pool(workers, initializer=_init,
                                  initargs=(cfg, seed, zmap, derating, pocket, focus_bus, stages)) as pool:
        for i, r in enumerate(pool.imap_unordered(_block, blocks)):
            res.append(r)
            if (i + 1) % max(1, len(blocks) // 8) == 0 or i + 1 == len(blocks):
                log(f"  {rid}: {i + 1}/{len(blocks)} blocks ({time.time() - t0:.0f}s)")
    # static names (cheap rebuild in the parent)
    from bzgen.solve import opf
    n = opf.build_network(cfg)
    zm = ZonalMarket(n, zmap, derating, cfg, noise_costs(n, cfg))
    order = np.argsort(np.array([r["snaps"][0] for r in res], dtype="datetime64[ns]"))
    res = [res[i] for i in order]
    cat = lambda k: [x for r in res for x in r.get(k, [])]
    rows = pd.DataFrame(cat("rows")).set_index("snapshot").sort_index()
    idx = rows.index
    rows.to_parquet(d / "hourly.parquet")
    gi, zi = n.generators.index, zm.zones
    if cfg["validate"]["persist_p_zonal"]:
        pd.DataFrame(np.vstack(cat("pz")), index=idx, columns=gi).to_parquet(d / "p_zonal.parquet")
    pd.DataFrame(np.vstack(cat("zp")), index=idx, columns=zi).to_parquet(d / "zonal_prices.parquet")
    pd.DataFrame(np.vstack(cat("np")), index=idx, columns=zi).to_parquet(d / "net_positions.parquet")
    _long(cat("spill"), ["snapshot", "generator", "mw"], gi).to_parquet(d / "market_spill.parquet")
    _long(cat("zmu"), ["snapshot", "link", "mu"], zm.nz.links.index).to_parquet(d / "zonal_link_duals.parquet")
    cost_cols = ["C_market", "C_redispatch", "C_markup"]
    rows[cost_cols].to_parquet(d / "cost.parquet")
    if "B" in stages:
        _long(cat("rd"), ["snapshot", "generator", "up", "down"], gi).to_parquet(d / "redispatch.parquet")
        br = np.r_[n.lines.index.to_numpy(), n.links.index.to_numpy()]
        _long(cat("mu"), ["snapshot", "branch", "mu"], br).to_parquet(d / "branch_duals.parquet")
        tol = cfg["solve"]["shed_tol_mw"]
        inf = []
        for s, sz, pk in cat("shed_zone"):
            for k in np.flatnonzero(sz > tol):
                inf.append({"snapshot": s, "kind": "shed", "zone": zi[k], "shed_mwh": float(sz[k]),
                            "pocket_mwh": float(pk[k]), "status": "ok"})
        rz = Redispatch.zone_names(zmap, n) if "B" in stages else []
        for s, sl in cat("slack") if any("slack" in r for r in res) else []:
            for k in np.flatnonzero(sl > tol):
                inf.append({"snapshot": s, "kind": "np_slack", "zone": rz[k], "shed_mwh": float(sl[k]),
                            "pocket_mwh": np.nan, "status": "ok"})
        for s, st in rows.stageB_status.items():
            if st != "ok":
                inf.append({"snapshot": s, "kind": "lp_infeasible", "zone": "ALL",
                            "shed_mwh": np.nan, "pocket_mwh": np.nan, "status": st})
        pd.DataFrame(inf, columns=["snapshot", "kind", "zone", "shed_mwh", "pocket_mwh",
                                   "status"]).to_csv(d / "infeasible.csv", index=False)
    meta = {"map": map_id, "derating": derating, "stages": list(stages), "hours": len(idx),
            "wall_s": time.time() - t0, "n_zones": len(zi),
            "stageB_not_ok": int((rows.stageB_status != "ok").sum()) if "B" in stages else None,
            "seconds_median": float(rows.seconds.median())}
    done.write_text(json.dumps(meta, indent=1))
    log(f"{rid}: {meta}")
    return d


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def select(cfg, n_snapshots):
    c = copy.deepcopy(cfg)
    if n_snapshots is not None:
        c["solve"]["n_snapshots"] = int(n_snapshots)
    sn = solve_run.select_snapshots(c)
    sn.index = pd.DatetimeIndex(sn.index, name="snapshot")
    return sn, ("all" if c["solve"]["n_snapshots"] == "all" else f"rep{int(c['solve']['n_snapshots'])}")


def plan_runs(cfg, maps: dict, sn: pd.DataFrame, scorings) -> list[dict]:
    """Runs needed for the requested scoring sets, reference derating first (so the
    headline number is available early); k = 1 covers the union of their hours."""
    even = zonemap.iso_week(sn.index).iso_week.to_numpy() % 2 == 0
    need = {"insample": ("none", "insample"), "oos_even": ("none", "oos")}
    ref = cfg["validate"]["reference_derating"]
    ders = sorted(cfg["validate"]["deratings"], key=lambda d: (not np.isclose(d, ref), d))
    runs = []
    for der in ders:
        for mid, m in maps.items():
            if not any(m["fit"] in need[s] for s in scorings):
                continue
            if m["fit"] == "oos":
                hours = sn.index[even]
            elif m["fit"] == "none" and scorings == ["oos_even"]:
                hours = sn.index[even]
            else:
                hours = sn.index
            runs.append({"map": mid, "derating": float(der), "hours": hours})
    return runs


def scoring_views(cfg, maps, sn):
    """{scoring: {map_id: hours}}."""
    even = zonemap.iso_week(sn.index).iso_week.to_numpy() % 2 == 0
    v = {}
    if "insample" in cfg["validate"]["scoring"]:
        v["insample"] = {m: sn.index for m, x in maps.items() if x["fit"] in ("none", "insample")}
    if "oos_even" in cfg["validate"]["scoring"]:
        v["oos_even"] = {m: sn.index[even] for m, x in maps.items() if x["fit"] in ("none", "oos")}
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-snapshots", type=int)
    ap.add_argument("--stage", default="AB", choices=["A", "AB"])
    ap.add_argument("--maps", nargs="*")
    ap.add_argument("--deratings", nargs="*", type=float)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--scoring", nargs="*", choices=["oos_even", "insample"],
                    help="only these scoring sets (default: validate.scoring)")
    args = ap.parse_args()
    cfg = config.load()
    v = cfg["validate"]
    if v["alt_weather_year"]["enabled"]:
        raise NotImplementedError(
            "validate.alt_weather_year: rebuild data/interim and data/solved with weather_year = "
            f"{v['alt_weather_year']['year']} into a separate data root, then score the fixed "
            "published map on it. Not implemented in this pass (off by default).")
    if args.deratings:
        v["deratings"] = args.deratings
    workers = args.workers or cfg["solve"]["workers"]
    sn, tag = select(cfg, args.n_snapshots)
    out = DATA / tag
    out.mkdir(parents=True, exist_ok=True)
    sn.to_csv(out / "snapshots.csv")
    log(f"{len(sn)} snapshots ({tag})")
    nodal_reference(cfg, sn, out)
    buses = pd.read_csv(INTERIM / "buses.csv", index_col=0)
    RES.mkdir(parents=True, exist_ok=True)
    real, real_rep = zonemap.real_zones(buses, cfg)
    real.to_csv(RES / "real_zones.csv")
    real_rep.to_csv(RES / "real_zones_report.csv")
    maps = load_maps(cfg, log)
    if args.maps:
        maps = {k: m for k, m in maps.items() if k in args.maps}
    share = pd.read_csv(out / "nodal" / "shed_share.csv", index_col=0).shed_hour_share
    pocket = set(M.pockets(share, cfg))
    focus_bus = buses.cluster_country == v["focus"]
    zmaps = {mid: zonemap.zone_map(buses, m["labels"], cfg, real=real) for mid, m in maps.items()}
    if args.check:
        crosscheck(cfg, sn, zmaps, out)
    if not args.report_only:
        scorings = args.scoring or list(v["scoring"])
        v["scoring"] = scorings
        for r in plan_runs(cfg, maps, sn, scorings):
            run_one(cfg, sn, out, r["map"], zmaps[r["map"]], r["derating"], r["hours"], pocket,
                    focus_bus, args.stage, workers)
    from bzgen.validate import report
    report.write(cfg, sn, out, tag, maps, zmaps, pocket, buses, real_rep, args.stage)


def load_maps(cfg, log):
    """Maps under test.  The out-of-sample refit needs the full-year nodal reference
    (odd-week mean dispatch for the balance term); without it only in-sample maps exist."""
    f = DATA / "all" / "nodal" / "gen_mean_odd.csv"
    gm_odd = pd.read_csv(f, index_col=0).iloc[:, 0] if f.exists() else None
    if gm_odd is None:
        log("full-year nodal reference missing: no out-of-sample maps yet")
    return zonemap.build_maps(cfg, gm_odd, log)


def crosscheck(cfg, sn, zmaps, out):
    """Objectives of the highspy path against PyPSA/linopy on two hours (stage A and B)."""
    from bzgen.solve import opf
    from bzgen.validate.market import check_against_pypsa
    from bzgen.validate.redispatch import redispatch_pypsa
    f = out / "crosscheck.csv"
    if f.exists():
        return
    n = opf.build_network(cfg)
    costs = noise_costs(n, cfg)
    snaps = list(sn.index[:2])
    L = n.loads_t.p_set
    bi = pd.Series(np.arange(len(n.buses)), index=n.buses.index)
    lb = bi[n.loads.bus].to_numpy()

    def bus_load(s):
        x = np.zeros(len(n.buses))
        np.add.at(x, lb, L.loc[s, n.loads.index].to_numpy())
        return x
    rows = []
    mid = "is_headline" if "is_headline" in zmaps else next(iter(zmaps))
    der = cfg["validate"]["reference_derating"]
    a = check_against_pypsa(n, zmaps[mid], der, cfg, costs, snaps, bus_load)
    rows += [{"stage": "A", "snapshot": s, **r} for s, r in a.iterrows()]
    zm = ZonalMarket(n, zmaps[mid], der, cfg, costs)
    rd = Redispatch(n, zmaps[mid], cfg, costs)
    pmpu = n.get_switchable_as_dense("Generator", "p_max_pu")
    pz, ob = {}, {}
    for s in snaps:
        pmax = pmpu.loc[s].to_numpy() * n.generators.p_nom.to_numpy()
        ra = zm.solve(pmax, bus_load(s))
        pz[s] = ra["p"]
        rb = rd.solve(pmax, bus_load(s), ra["p"])
        ob[s] = rb.get("objective", np.nan)
    ref = redispatch_pypsa(n, zmaps[mid], cfg, costs, snaps,
                           pd.DataFrame(pz, index=n.generators.index).T)
    for s in snaps:
        rows.append({"stage": "B", "snapshot": s, "highspy": ob[s], "pypsa": float(ref[s]),
                     "rel_diff": abs(ob[s] - ref[s]) / max(1.0, abs(ref[s]))})
    pd.DataFrame(rows).to_csv(f, index=False)
    log(f"cross-check: {pd.DataFrame(rows)[['stage', 'rel_diff']].to_dict('records')}")


if __name__ == "__main__":
    main()
