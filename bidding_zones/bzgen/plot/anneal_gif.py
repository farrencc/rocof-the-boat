"""Animated GIF of the DE bidding zones forming during simulated annealing.

    python -m bzgen.plot.anneal_gif [--country DE] [--restart R] [--every 2]

Re-runs one restart of the headline-configuration anneal (sweep.headline
parameters, committed edge table) one temperature at a time with the same
schedule as ``bzgen.cluster.anneal.anneal``, and draws the zone map after
every ``--every`` temperature steps plus the final zero-temperature quench.
Seeds start from the validation stage's stable (crc32) seed for that restart, but
the random stream is re-seeded at every temperature step, so the trajectory and the
final map differ from that restart's single-call result: the final frame is one
low-energy map of the near-degenerate family, not the headline map.
Writes figures/validate/de_zones_forming.gif.
"""

from __future__ import annotations

import argparse
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from bzgen import config
from bzgen.cluster import anneal as A
from bzgen.cluster import graphs
from bzgen.cluster.sweep import node_attributes
from bzgen.plot import maps as pm
from bzgen.validate.zonemap import _seed

ROOT = config.ROOT


def frames(cfg, country, restart, every):
    a = cfg["anneal"]
    cid = cfg["sweep"]["headline"]
    p = json.loads((ROOT / "results" / "configs" / cid / "done.json").read_text())["params"]
    buses = pd.read_csv(ROOT / "data" / "interim" / "buses.csv", index_col=0)
    edges = pd.read_parquet(ROOT / "results" / "edges.parquet")
    spec = pd.read_csv(ROOT / "results" / "spectrum.csv", index_col=0)
    L, G = node_attributes(cfg)
    g, nodes, host, _ = graphs.build(country, buses, edges, p["alpha"], L, G, cfg["edges"]["dc_in_energy"])
    k = int(spec.loc[country, "k_used"]) + p["dk"]
    seed = _seed(a, country, restart)
    rng = np.random.default_rng(seed)
    lam_c1 = A.contiguity_guarantee(g, p["lambda_b"]) if a["lambda_c_final"] in (None, "auto") else a["lambda_c_final"]
    labels = A.graph_voronoi(g, k, rng)
    T0 = A.initial_temperature(g, labels, k, p["lambda_b"], p["lambda_c_initial"], a["balance_floor"], rng)
    nT = a["n_temps"]
    T = T0 * a["t_final_ratio"] ** (np.arange(nT) / max(nT - 1, 1))
    lam = p["lambda_c_initial"] * (lam_c1 / p["lambda_c_initial"]) ** (np.arange(nT) / max(nT - 1, 1))
    steps = int(a["sweeps_per_temp"] * g.n)
    ws = g.ws

    def lab_series():
        s = pd.Series(labels.copy(), index=nodes)
        for b, h in host.items():
            s[b] = s[h]
        return s

    def energy(lc):
        ep, cs, pb = A.energy_terms(labels, k, g.ptr, g.idx, g.w, g.Lnode, g.Gnode, g.Ltot, a["balance_floor"])
        return ep + lc * cs + p["lambda_b"] * pb, cs

    e, cs = energy(lam[0])
    out = [(lab_series(), "initial state (graph Voronoi)", T0, lam[0], e, cs)]
    for i in range(nT):
        A._run(labels, k, g.ptr, g.idx, g.w, g.Lnode, g.Gnode, g.Ltot, a["balance_floor"], p["lambda_b"],
               lam[i:i + 1], T[i:i + 1], steps, 0, seed + i, ws.mark, ws.owner, ws.stamp, ws.Q, ws.qh,
               ws.qt, ws.parent, ws.active, ws.seeds)
        if i % every == every - 1 or i == nT - 1:
            e, cs = energy(lam[i])
            out.append((lab_series(), f"Cooling step {i + 1} of {nT}", T[i], lam[i], e, cs))
    A._run(labels, k, g.ptr, g.idx, g.w, g.Lnode, g.Gnode, g.Ltot, a["balance_floor"], p["lambda_b"],
           lam[-1:], T[-1:], 0, int(a["quench_sweeps"] * g.n), seed + nT, ws.mark, ws.owner, ws.stamp,
           ws.Q, ws.qh, ws.qt, ws.parent, ws.active, ws.seeds)
    e, cs = energy(lam_c1)
    out.append((lab_series(), "Final map", 0.0, lam_c1, e, cs))
    return out, buses, p, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="DE")
    ap.add_argument("--restart", type=int, default=1)
    ap.add_argument("--every", type=int, default=2)
    args = ap.parse_args()
    cfg = config.load()
    fr, buses, p, k = frames(cfg, args.country, args.restart, args.every)
    lines = pd.read_csv(ROOT / "data" / "interim" / "lines.csv", index_col=0)
    links = pd.read_csv(ROOT / "data" / "interim" / "links.csv", index_col=0)
    mem = sorted(buses[buses.cluster_country == args.country].country.unique())
    imgs = []
    for lab, what, T, lc, e, cs in fr:
        fig, ax = plt.subplots(figsize=(5.2, 6.0))
        pm.draw_country(ax, buses, lines, links, lab, mem)
        ax.set_title(f"{args.country}: {k} bidding zones forming ({pm.ALPHA_LABEL.lower()} = {p['alpha']:g})\n{what}",
                     fontsize=9, loc="left")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        imgs.append(Image.open(buf).convert("RGB"))
    dur = [900] + [180] * (len(imgs) - 2) + [3000]
    out = ROOT / "figures" / "validate" / f"{args.country.lower()}_zones_forming.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=dur, loop=0, optimize=True)
    print(f"{len(imgs)} frames -> {out} ({out.stat().st_size / 1e6:.1f} MB); final H = {fr[-1][4]:.3f}")


if __name__ == "__main__":
    main()
