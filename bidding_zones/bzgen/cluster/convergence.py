"""Convergence check for the headline configuration -> results/convergence.json.

    python -m bzgen.cluster.convergence

For every country with >= 100 nodes: re-anneal with a 4x longer schedule
(2x temperatures, 2x sweeps per temperature) and 2x restarts, then compare
with the sweep's minimum-energy map: best energy (is the sweep's minimum
beaten?) and ARI between the two maps (is it the same map?).
"""

from __future__ import annotations

import json
from multiprocessing import get_context

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari

from bzgen import config
from bzgen.cluster import graphs, sweep
from bzgen.cluster.anneal import anneal


def _one(args):
    c, p, cfg = args
    sweep._init(cfg)
    D = sweep._D
    a = cfg["anneal"]
    g, nodes, host, e = graphs.build(c, D["buses"], D["edges"], p["alpha"], D["L"], D["G"],
                                     cfg["edges"]["dc_in_energy"])
    k = int(D["spec"].loc[c, "k_used"]) + p["dk"]
    res = [anneal(g, k, lam_b=p["lambda_b"], lam_c0=p["lambda_c_initial"], lam_c1=a["lambda_c_final"],
                  floor=a["balance_floor"], n_temps=2 * a["n_temps"],
                  sweeps_per_temp=2 * a["sweeps_per_temp"], t_final_ratio=a["t_final_ratio"],
                  seed=987_654 + 31 * r, quench_sweeps=a["quench_sweeps"])
           for r in range(2 * a["restarts"])]
    E = np.array([x["energy"] for x in res])
    best = res[int(E.argmin())]
    return c, nodes, best["labels"], float(E.min()), float(np.median(E)), float(np.ptp(E))


def main():
    cfg = config.load()
    head = cfg["sweep"].get("headline") or sweep.cid_of(cfg["sweep"]["baseline"])
    d = sweep.CONFIGS / head
    p = json.loads((d / "done.json").read_text())["params"]
    rows = pd.read_csv(d / "rows.csv").set_index("country")
    lab = pd.read_csv(d / "labels.csv").set_index("bus").zone
    cs = [c for c in rows.index if rows.loc[c, "n_nodes"] >= 100]
    with get_context("fork").Pool(cfg["solve"]["workers"]) as pool:
        out = pool.map(_one, [(c, p, cfg) for c in cs], chunksize=1)
    rep = []
    for c, nodes, labels, emin, emed, spread in out:
        r = rows.loc[c]
        rep.append({"country": c, "n_nodes": int(r.n_nodes), "sweep_E_min": float(r.E_min),
                    "long_E_min": emin, "long_E_median": emed, "long_E_spread": spread,
                    "improvement": float(r.E_min - emin),
                    "rel_improvement": float((r.E_min - emin) / abs(r.E_min)),
                    "ari_long_vs_sweep_map": float(ari(lab.reindex(nodes).to_numpy(), labels))})
    (sweep.RESULTS / "convergence.json").write_text(json.dumps(rep, indent=1))
    print(pd.DataFrame(rep).round(3).to_string())


if __name__ == "__main__":
    main()
