"""Parameter sweep: one configuration at a time, all countries in parallel.

    python -m src.cluster.sweep [--no-git] [--only CID ...]

After every configuration: labels, diagnostics and figures are written, the
tidy table ``results/sweep.csv`` is regenerated, and the work is committed and
pushed, so a crash loses at most one configuration.  Resumable: a configuration
whose ``results/configs/<cid>/done.json`` exists is skipped.

Configurations are one-at-a-time variations around the baseline in
``config.sweep``.  k per country = spectral k_used + dk (clipped to >= 2).
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari

from src import config
from src.cluster import graphs
from src.cluster.anneal import anneal

ROOT = config.ROOT
RESULTS = ROOT / "results"
CONFIGS = RESULTS / "configs"
INTERIM = ROOT / "data" / "interim"
SOLVED = ROOT / "data" / "solved"


def cid_of(p: dict) -> str:
    return f"a{p['alpha']:g}_lc{p['lambda_c_initial']:g}_lb{p['lambda_b']:g}_dk{p['dk']:+d}"


def configurations(cfg) -> list[dict]:
    base = dict(cfg["sweep"]["baseline"])
    out = [dict(base, role="baseline")]
    for key, vals in cfg["sweep"]["vary"].items():
        for v in vals:
            p = dict(base, role=f"vary {key}")
            p[key] = v
            if cid_of(p) not in {cid_of(q) for q in out}:
                out.append(p)
    return out


def node_attributes(cfg) -> tuple[pd.Series, pd.Series]:
    """Time-weighted mean load and generation (MW) per bus."""
    buses = pd.read_csv(INTERIM / "buses.csv", index_col=0)
    key = pd.read_csv(INTERIM / "load_key.csv", index_col=0).load_key
    nat = pd.read_parquet(INTERIM / "national_load.parquet")
    w = pd.read_csv(SOLVED / "snapshots.csv", index_col=0, parse_dates=True).weight
    nat.index = nat.index.tz_localize(None) if nat.index.tz is not None else nat.index
    m = (nat.loc[w.index].mul(w.to_numpy(), axis=0).sum() / w.sum())
    L = key * buses.country.map(m)
    gm = pd.read_csv(SOLVED / "gen_mean.csv", index_col=0).p_mean
    gens = pd.read_csv(INTERIM / "generators.csv", index_col=0)
    G = gm.groupby(gens.bus.reindex(gm.index)).sum()
    return L.rename("L"), G.reindex(buses.index).fillna(0.0).rename("G")


# worker globals
_D = {}


def _init(cfg):
    buses = pd.read_csv(INTERIM / "buses.csv", index_col=0)
    edges = pd.read_parquet(RESULTS / "edges.parquet")
    spec = pd.read_csv(RESULTS / "spectrum.csv", index_col=0)
    L, G = node_attributes(cfg)
    _D.update(cfg=cfg, buses=buses, edges=edges, spec=spec, L=L, G=G)


def run_country(args):
    c, p = args
    cfg, buses, edges, spec = _D["cfg"], _D["buses"], _D["edges"], _D["spec"]
    a = cfg["anneal"]
    t0 = time.time()
    g, nodes, host, e = graphs.build(c, buses, edges, p["alpha"], _D["L"], _D["G"],
                                     cfg["edges"]["dc_in_energy"])
    k0 = int(spec.loc[c, "k_used"])
    k = k0 + p["dk"]
    row = {"country": c, "n_buses": len(nodes) + len(host), "n_nodes": g.n,
           "n_isolated_attached": len(host), "k_spectral": k0, "k": k,
           "elbow_convincing": bool(spec.loc[c, "convincing"])}
    if k < 2 or k > g.n:
        row["status"] = f"SKIPPED k={k} outside [2, n]"
        return row, None
    res = []
    for r in range(a["restarts"]):
        seed = (a["seed"] * 1_000_003 + hash(c) % 100_000 * 101 + r) % (2**31 - 1)
        res.append(anneal(g, k, lam_b=p["lambda_b"], lam_c0=p["lambda_c_initial"],
                          lam_c1=a["lambda_c_final"], floor=a["balance_floor"],
                          n_temps=a["n_temps"], sweeps_per_temp=a["sweeps_per_temp"],
                          t_final_ratio=a["t_final_ratio"], seed=seed,
                          quench_sweeps=a["quench_sweeps"]))
    E = np.array([x["energy"] for x in res])
    o = np.argsort(E)
    best = res[o[0]]
    scale = max(1.0, abs(E[o[0]]))
    tol = cfg["sweep"]["degeneracy_rel_tol"] * scale
    near = [i for i in o[1:] if E[i] - E[o[0]] <= tol]
    aris_near = [ari(best["labels"], res[i]["labels"]) for i in near]
    ari_2nd = ari(best["labels"], res[o[1]]["labels"]) if len(o) > 1 else np.nan
    lab = pd.Series(best["labels"], index=nodes)
    for b, h in host.items():
        lab[b] = lab[h]
    src = np.repeat(np.arange(g.n), np.diff(g.ptr))
    und = src < g.idx                      # each undirected edge once
    cut = best["labels"][src[und]] != best["labels"][g.idx[und]]
    Lz = pd.Series(g.Lnode).groupby(best["labels"]).sum()
    Gz = pd.Series(g.Gnode).groupby(best["labels"]).sum()
    share = (np.maximum(Lz, Gz) / g.Ltot).sort_values(ascending=False)
    lshare = (Lz / g.Ltot).reindex(share.index)
    sizes = pd.Series(best["labels"]).value_counts().reindex(share.index)
    row.update({
        "status": "ok", "energy": best["energy"], "potts": best["potts"],
        "balance_pen": best["balance"], "contig_excess": best["contig"],
        "contiguous": best["contig"] == 0, "lambda_c_final": best["lam_c_final"],
        "E_min": E.min(), "E_median": float(np.median(E)), "E_max": E.max(),
        "E_spread": float(E.max() - E.min()),
        "E_gap_2nd": float(E[o[1]] - E[o[0]]) if len(o) > 1 else np.nan,
        "ari_best_vs_2nd": ari_2nd,
        "n_near_degenerate": len(near),
        "ari_near_min": float(min(aris_near)) if aris_near else np.nan,
        "degenerate": bool(aris_near and min(aris_near) < cfg["sweep"]["degenerate_ari"]),
        "restarts_contiguous": int(sum(x["contig"] == 0 for x in res)),
        "zone_buses": "/".join(str(int(s)) for s in sizes),
        "zone_share_max_lg": "/".join(f"{s:.3f}" for s in share),
        "zone_load_share": "/".join(f"{s:.3f}" for s in lshare),
        "n_zones_above_floor": int((share >= a["balance_floor"]).sum()),
        "n_zones_effective": float(1.0 / (lshare ** 2).sum()),
        "frac_edges_w_positive": float((g.w[und] > 0).mean()),
        "cut_ratio": float(cut.mean()),
        "cut_weight_share_positive": float(g.w[und][cut].clip(min=0).sum() / max(g.w[und].clip(min=0).sum(), 1e-12)),
        "wall_s": time.time() - t0,
    })
    return row, lab


def git_commit(msg: str, paths: list[str]) -> None:
    subprocess.run(["git", "add", *paths], cwd=ROOT, check=True)
    r = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
    if r.returncode == 0:
        return
    body = (f"{msg}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
            "Claude-Session: https://claude.ai/code/session_01TcBF1icwEscnPGKZ1Tr2Rc")
    subprocess.run(["git", "commit", "-q", "-m", body], cwd=ROOT, check=True)
    for attempt in range(4):
        if subprocess.run(["git", "push", "-q"], cwd=ROOT).returncode == 0:
            return
        time.sleep(2 ** (attempt + 1))
    print("WARNING: git push failed after retries; commit is local", flush=True)


def rebuild_table() -> pd.DataFrame:
    rows = [pd.read_csv(f) for f in sorted(CONFIGS.glob("*/rows.csv"))]
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    df.to_csv(RESULTS / "sweep.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-git", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    cfg = config.load()
    from src.plot import figures
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    confs = configurations(cfg)
    if args.only:
        confs = [p for p in confs if cid_of(p) in args.only]
    buses = pd.read_csv(INTERIM / "buses.csv", index_col=0)
    spec = pd.read_csv(RESULTS / "spectrum.csv", index_col=0)
    countries = sorted(spec.index, key=lambda c: -int(spec.loc[c, "n_nodes"]))
    log(f"{len(confs)} configurations x {len(countries)} countries")
    with get_context("fork").Pool(cfg["solve"]["workers"], initializer=_init, initargs=(cfg,)) as pool:
        for p in confs:
            cid = cid_of(p)
            d = CONFIGS / cid
            if (d / "done.json").exists():
                log(f"skip {cid} (done)")
                continue
            d.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            out = pool.map(run_country, [(c, p) for c in countries], chunksize=1)
            rows, labs = [], []
            for row, lab in out:
                row.update(config_id=cid, role=p["role"], alpha=p["alpha"],
                           lambda_c_initial=p["lambda_c_initial"], lambda_b=p["lambda_b"],
                           dk=p["dk"])
                rows.append(row)
                if lab is not None:
                    labs.append(pd.DataFrame({"bus": lab.index, "zone": lab.values,
                                              "country": row["country"]}))
            rows = pd.DataFrame(rows)
            labels = pd.concat(labs, ignore_index=True)
            # map sensitivity: ARI against the baseline map, where available
            bfile = CONFIGS / cid_of(dict(cfg["sweep"]["baseline"])) / "labels.csv"
            if bfile.exists() and p["role"] != "baseline":
                bl = pd.read_csv(bfile).set_index("bus").zone
                cur = labels.set_index("bus").zone
                rows["ari_vs_baseline"] = [
                    ari(bl.reindex(cur[labels.set_index("bus").country == c].index),
                        cur[labels.set_index("bus").country == c])
                    if c in set(labels.country) and bl.reindex(
                        cur[labels.set_index("bus").country == c].index).notna().all() else np.nan
                    for c in rows.country]
            labels.to_csv(d / "labels.csv", index=False)
            rows.to_csv(d / "rows.csv", index=False)
            figures.config_figures(cid, labels, rows, buses)
            meta = {"params": p, "wall_s": time.time() - t0,
                    "n_ok": int((rows.status == "ok").sum()),
                    "n_non_contiguous": int((rows.status == "ok").sum() - rows.contiguous.fillna(False).sum()),
                    "n_degenerate": int(rows.degenerate.fillna(False).sum())}
            (d / "done.json").write_text(json.dumps(meta, indent=1))
            rebuild_table()
            log(f"{cid}: {meta}")
            if not args.no_git:
                git_commit(f"Sweep configuration {cid} ({p['role']})",
                           ["results", "figures/zones", "figures/europe"])
    df = rebuild_table()
    figures.sensitivity(df, cfg)
    if not args.no_git:
        git_commit("Sweep: sensitivity figures", ["results", "figures"])


if __name__ == "__main__":
    main()
