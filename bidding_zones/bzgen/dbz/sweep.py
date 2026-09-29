"""lambda_rigid sweep on the European graph: resumable, checkpointed, committed.

    python -m bzgen.dbz.sweep [--smoke] [--no-git] [--lambdas 0 1 ...] [--restarts N]
                              [--out DIR]

lambda_rigid values run serially; the worker pool maps over restarts within a
lambda.  After every lambda: labels, rows, traces, figures and the report are
written (each file atomically: temp + os.replace), ``sweep.csv`` is regenerated
from all completed lambda, ``done.json`` is written last, and the work is
committed and pushed.  A lambda is complete iff its ``done.json`` exists; an
interrupted run resumes from the first incomplete lambda.

``--smoke`` cuts n_temps / sweeps_per_temp / quench_sweeps / restarts to the
``dbz.smoke`` values and writes to ``results/dbz/smoke`` (never committed).
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari

from bzgen import config
from bzgen.cluster.anneal import contiguity_guarantee, energy_terms
from bzgen.dbz import anchor as AN
from bzgen.dbz import balance_floor, io, load_config
from bzgen.dbz import graph as DG
from bzgen.dbz.anneal import INITS, anneal_dbz
from bzgen.dbz.rigidity import rigidity_full

ROOT = config.ROOT


def lam_id(lam: float) -> str:
    return f"lr{lam:g}"


# --------------------------------------------------------------------------- #
# setup (parent process, before fork)
# --------------------------------------------------------------------------- #

_W: dict = {}


def setup(cfg: dict) -> dict:
    from bzgen.cluster.sweep import node_attributes
    from bzgen.dbz.report import anchor_params
    p = anchor_params(cfg)
    ef = io.RESULTS / "edges.parquet"
    if ef.exists():
        from bzgen.cluster.sweep import INTERIM
        buses = pd.read_csv(INTERIM / "buses.csv", index_col=0)
        en = pd.read_parquet(ef)
    else:
        buses, en, _ = DG.compute_edges(cfg)
    L, G = node_attributes(cfg)
    cap = DG.installed_capacity(buses)
    nodes = DG.node_order(buses)
    an = AN.load(cfg["dbz"]["anchor_config_id"], nodes, buses)
    K = an["K"]
    floor = balance_floor(cfg["anneal"], K)
    g, nodes, e = DG.build(buses, en, p["alpha"], L, G, cap, K, cfg["edges"]["dc_in_energy"],
                           nodes=nodes)
    contig = AN.check_contiguity(an["A"], K, g, nodes, buses)
    ep, cs, pb = energy_terms(an["A"], K, g.ptr, g.idx, g.w, g.Lnode, g.Gnode, g.Ltot, floor)
    ref = {"potts": ep, "contig": cs, "balance": pb, "physical": ep + p["lambda_b"] * pb}
    # A': the anchor with each formerly-isolated bus moved to its (foreign) neighbour's
    # zone, i.e. the nearest contiguous map to A.  Used only for the reference quench.
    pos = {b: i for i, b in enumerate(nodes)}
    Ap = an["A"].copy()
    for b in contig["isolated"]:
        i = pos[b]
        Ap[i] = an["A"][g.idx[g.ptr[i]]]
    return dict(cfg=cfg, params=p, buses=buses, g=g, nodes=nodes, anchor=an, floor=floor,
                K=K, contig=contig, anchor_energy=ref, A_prime=Ap)


def reference_quench(W: dict, lam: float) -> dict:
    """DIAGNOSTIC, not part of the sweep: a T = 0 quench (50 sweeps) started from A'.
    It shows where the anchored optimum lies; the sweep itself never starts from A."""
    res = anneal_dbz(W["g"], W["anchor"]["A"], W["K"], lam_b=W["params"]["lambda_b"],
                     floor=W["floor"], lam_rigid=lam, n_temps=1, sweeps_per_temp=0,
                     t_final_ratio=1.0, seed=0, quench_sweeps=50, init=W["A_prime"],
                     T0_band=(0.0, np.inf))
    return {"ref_physical": res["physical"], "ref_rigid": res["rigid"],
            "ref_energy": res["physical"] + lam * res["rigid"],
            "ref_transfer": AN.transfer_distance(res["labels"], W["anchor"]["A"], W["K"], W["K"])}


def _restart(args):
    lam, r, sched = args
    W = _W
    cfg, g = W["cfg"], W["g"]
    a = cfg["anneal"]
    seed = int(a["seed"]) * 1_000_003 + r      # common random numbers across lambda
    t0 = time.time()
    res = anneal_dbz(g, W["anchor"]["A"], W["K"], lam_b=W["params"]["lambda_b"],
                     floor=W["floor"], lam_rigid=lam, n_temps=sched["n_temps"],
                     sweeps_per_temp=sched["sweeps_per_temp"],
                     t_final_ratio=a["t_final_ratio"], seed=seed,
                     quench_sweeps=sched["quench_sweeps"],
                     T0_band=tuple(float(x) for x in cfg["dbz"]["T0_band"]),
                     init_mode=INITS[r % len(INITS)])
    res["wall_s"] = time.time() - t0
    res["restart"] = r
    res["seed"] = seed
    return res


# --------------------------------------------------------------------------- #
# diagnostics of one lambda
# --------------------------------------------------------------------------- #

def diagnostics(W: dict, lam: float, res: list, sched: dict) -> tuple[dict, pd.DataFrame]:
    g, A, K = W["g"], W["anchor"]["A"], W["K"]
    E = np.array([x["energy"] for x in res])
    o = np.argsort(E)
    best = res[o[0]]
    B = best["labels"]
    # sanity: incremental totals vs full recomputation
    ep, cs, pb = energy_terms(B, K, g.ptr, g.idx, g.w, g.Lnode, g.Gnode, g.Ltot, W["floor"])
    rr = rigidity_full(B, A, g.cap, K, K) / g.Z
    assert abs(ep - best["potts"]) < 1e-6 * max(1, abs(ep)), (ep, best["potts"])
    assert abs(rr - best["rigid"]) < 1e-6 * max(1, abs(rr)), (rr, best["rigid"])
    src, dst, w, isx, isdc = g.undirected()
    cut = B[src] != B[dst]
    cutA = A[src] != A[dst]
    wpos = np.clip(w, 0, None)
    Lz = np.bincount(B, weights=g.Lnode, minlength=K)
    Gz = np.bincount(B, weights=g.Gnode, minlength=K)
    share = np.maximum(Lz, Gz) / g.Ltot
    sizes = np.bincount(B, minlength=K)
    labs = [x["labels"] for x in res]
    pair = [ari(labs[i], labs[j]) for i, j in itertools.combinations(range(len(labs)), 2)]
    tr = best["trace"]
    nT = sched["n_temps"]
    early = tr[: max(1, nT // 10), 4]
    acc_best = [x["trace"][: max(1, nT // 10), 4].mean() for x in res]
    cc = W["buses"].cluster_country.reindex(W["nodes"]).to_numpy()
    zone_countries = pd.Series(cc).groupby(B).nunique()
    row = {
        "lambda_rigid": lam, "status": "ok", "K": K, "n_zones": int(len(np.unique(B))),
        "energy": best["energy"], "potts": best["potts"], "contig_excess": best["contig"],
        "contiguous": best["contig"] == 0, "balance_pen": best["balance"],
        "rigid": best["rigid"], "physical": best["physical"],
        "transfer_distance": AN.transfer_distance(B, A, K, K),
        "transfer_distance_cap_mw": AN.transfer_distance(B, A, K, K, w=g.cap),
        "transfer_share": AN.transfer_distance(B, A, K, K) / g.n,
        "ari_vs_anchor": ari(A, B),
        "zone_buses_min": int(sizes.min()), "zone_buses_median": float(np.median(sizes)),
        "zone_buses_max": int(sizes.max()),
        "zone_share_min": float(share.min()), "zone_share_max": float(share.max()),
        "n_zones_above_floor": int((share >= W["floor"]).sum()),
        "n_zones_multinational": int((zone_countries > 1).sum()),
        "cut_edges": int(cut.sum()),
        "cut_share_cross_border": float(isx[cut].mean()) if cut.any() else np.nan,
        "cut_share_dc": float(isdc[cut].mean()) if cut.any() else np.nan,
        "cut_wpos_share_cross_border": float(wpos[cut & isx].sum() / max(wpos[cut].sum(), 1e-12)),
        "cut_wpos_share_dc": float(wpos[cut & isdc].sum() / max(wpos[cut].sum(), 1e-12)),
        "cross_border_edges_cut": float(cut[isx].mean()),
        "anchor_cut_edges_kept": float((cut & cutA).sum() / max(cutA.sum(), 1)),
        "restarts": len(res), "E_min": float(E.min()), "E_median": float(np.median(E)),
        "E_max": float(E.max()), "E_spread": float(E.max() - E.min()),
        "E_gap_2nd": float(E[o[1]] - E[o[0]]) if len(o) > 1 else np.nan,
        "ari_best_vs_2nd": ari(labs[o[0]], labs[o[1]]) if len(o) > 1 else np.nan,
        "ari_restarts_mean": float(np.mean(pair)) if pair else np.nan,
        "ari_restarts_min": float(np.min(pair)) if pair else np.nan,
        "transfer_distance_restarts_median": float(np.median(
            [AN.transfer_distance(x, A, K, K) for x in labs])),
        "restarts_contiguous": int(sum(x["contig"] == 0 for x in res)),
        "T0": best["T0"], "lambda_c": best["lam_c"],
        "accept_early_best": float(early.mean()),
        "accept_early_min_restarts": float(np.min(acc_best)),
        "accept_collapse": bool(early.mean() < 0.01),
        "wall_s_restart_median": float(np.median([x["wall_s"] for x in res])),
        "n_temps": nT, "sweeps_per_temp": sched["sweeps_per_temp"],
        "quench_sweeps": sched["quench_sweeps"],
        "Z": g.Z, "floor": W["floor"],
        "init_best": best["init_mode"],
        "energy_excl_contig": best["physical"] + lam * best["rigid"],
    }
    for mode in INITS:
        Em = [x["physical"] + lam * x["rigid"] for x in res if x["init_mode"] == mode]
        row[f"E_min_{mode}"] = float(min(Em)) if Em else np.nan
    row.update(reference_quench(W, lam))
    row["ref_dominates_best"] = bool(row["ref_physical"] <= row["physical"]
                                     and row["ref_transfer"] <= row["transfer_distance"])
    row["gap_to_ref"] = row["energy_excl_contig"] - row["ref_energy"]
    lab = pd.DataFrame({"bus": W["nodes"], "zone": B, "anchor_zone": A, "country": cc})
    return row, lab


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def git_commit(msg: str, paths: list) -> None:
    subprocess.run(["git", "add", "--", *map(str, paths)], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        return
    body = (f"{msg}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
            "Claude-Session: https://claude.ai/code/session_01R66yNyEwu9TU9tmaXyXg5J")
    subprocess.run(["git", "commit", "-q", "-m", body], cwd=ROOT, check=True)
    for attempt in range(4):
        if subprocess.run(["git", "push", "-q"], cwd=ROOT).returncode == 0:
            return
        time.sleep(2 ** (attempt + 1))
    print("WARNING: git push failed after retries; commit is local", flush=True)


def rebuild_table(out: Path) -> pd.DataFrame:
    rows = [pd.read_csv(f / "rows.csv") for f in sorted(out.glob("lr*"))
            if (f / "done.json").exists()]
    df = pd.concat(rows, ignore_index=True).sort_values("lambda_rigid") if rows else pd.DataFrame()
    io.write_csv(out / "sweep.csv", df, index=False)
    return df


def schedule(cfg: dict, smoke: bool, restarts: int | None) -> dict:
    a = cfg["anneal"]
    s = cfg["dbz"]["smoke"] if smoke else a
    return {"n_temps": int(s["n_temps"]), "sweeps_per_temp": float(s["sweeps_per_temp"]),
            "quench_sweeps": float(s["quench_sweeps"]),
            "restarts": int(restarts or s["restarts"])}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-git", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--lambdas", nargs="*", type=float)
    ap.add_argument("--restarts", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    cfg = load_config()
    sched = schedule(cfg, args.smoke, args.restarts)
    out = Path(args.out) if args.out else (io.RESULTS / "smoke" if args.smoke else io.RESULTS)
    out.mkdir(parents=True, exist_ok=True)
    git = not (args.no_git or args.smoke)
    lams = args.lambdas if args.lambdas else cfg["dbz"]["lambda_rigid"]
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    todo = [l for l in lams if not (out / lam_id(l) / "done.json").exists()]
    for l in lams:
        if l not in todo:
            log(f"skip {lam_id(l)} (done)")
    if not todo:
        rebuild_table(out)
        return
    W = setup(cfg)
    _W.update(W)
    log(f"N={W['g'].n} K={W['K']} Z={W['g'].Z:.1f} floor={W['floor']:.5f} "
        f"lambda_c={contiguity_guarantee(W['g'], W['params']['lambda_b']):.1f} "
        f"schedule={sched} -> {out}")
    io.write_json(out / "anchor_energy.json", {"anchor_config_id": W["anchor"]["cid"],
                                               **W["anchor_energy"], "K": W["K"],
                                               "N": W["g"].n, "Z": W["g"].Z,
                                               "floor": W["floor"]})
    workers = args.workers or cfg["dbz"].get("workers", 4)
    abort_at = os.environ.get("DBZ_ABORT_AT")        # tests only: simulate an interruption
    with get_context("fork").Pool(min(workers, sched["restarts"])) as pool:
        for lam in todo:
            d = out / lam_id(lam)
            d.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            res = pool.map(_restart, [(lam, r, sched) for r in range(sched["restarts"])],
                           chunksize=1)
            row, lab = diagnostics(W, lam, res, sched)
            row["wall_s"] = time.time() - t0
            io.write_csv(d / "labels.csv", lab, index=False)
            if abort_at is not None and float(abort_at) == lam:
                os._exit(3)                           # dies mid-checkpoint: no done.json
            io.write_csv(d / "rows.csv", pd.DataFrame([row]), index=False)
            restarts = pd.DataFrame([{**{k: x[k] for k in ("restart", "seed", "energy", "potts",
                                                           "contig", "balance", "rigid",
                                                           "physical", "T0", "wall_s", "init_mode",
                                                           "init_transfer")},
                                      "transfer_distance": AN.transfer_distance(
                                          x["labels"], W["anchor"]["A"], W["K"], W["K"])}
                                     for x in res])
            io.write_csv(d / "restarts.csv", restarts, index=False)
            io.write_npz(d / "traces.npz", **{f"r{x['restart']}": x["trace"] for x in res})
            if not args.no_figures:
                from bzgen.dbz import plots
                plots.zone_map(W, lab, lam, row, out_dir=None if out == io.RESULTS else out)
            io.write_json(d / "done.json", {"lambda_rigid": lam, "wall_s": row["wall_s"],
                                            "schedule": sched,
                                            "anchor_config_id": W["anchor"]["cid"]})
            df = rebuild_table(out)
            if not args.no_figures:
                from bzgen.dbz import report
                report.results_stage(cfg, out, df)
            log(f"{lam_id(lam)}: E={row['energy']:.2f} phys={row['physical']:.2f} "
                f"rigid={row['rigid']:.3f} transfer={row['transfer_distance']:.0f} "
                f"accept_early={row['accept_early_best']:.3f} wall={row['wall_s']:.0f}s")
            if git:
                git_commit(f"DBZ sweep lambda_rigid={lam:g}",
                           [out, ROOT / "figures" / "dbz", ROOT / "reports" / "dbz_results.md"])
    rebuild_table(out)


if __name__ == "__main__":
    main()
