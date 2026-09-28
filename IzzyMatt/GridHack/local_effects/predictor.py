"""Can a bus's RoCoF exposure be predicted without simulating it?

If local effects are real, the follow-on question (section 3.6) is where to put a
synchronous condenser -- and that is only practical if the answer can be read off
the network instead of found by simulating every candidate site.

The predictor is `local_inertia`: a bus's inertia averaged over its electrical
neighbourhood, weighted exp(-R_ij / lam) in effective resistance.  It costs one
pseudo-inverse of the stiffness Laplacian and no simulation at all.  Because that
Laplacian does not depend on the inertia map, the resistances are computed once
and reused for every configuration in the sweep.

The choice of target matters more than the choice of lam.  Against the worst 500 ms
window -- the metric that carries the local effect -- the predictor explains about
70% of the bus-to-bus variance.  Against the 500 ms window measured from the
event, which barely shows a local effect at all, it explains almost nothing.  Both
are reported below, because the contrast is the point: the metric you choose
decides whether there is anything to predict.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

import swing as sw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results_torus.csv")
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--periodic", action="store_true", default=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # sweeps written before the torus option carry no `periodic` column
    periodic = bool(df["periodic"].iloc[0]) if "periodic" in df.columns else False
    lat = sw.Lattice(args.L, K=3.0, periodic=periodic)
    ref = sw.build_network(lat, set(), layout="flat")
    R = sw.resistance_distance(ref)
    H_inv = float(df["H_inv"].iloc[0])

    piv = df.pivot_table(index="non_inertial", columns="bus",
                         values=["worst_window", "local_excess", "is_noninertial"])
    ni = piv["is_noninertial"].to_numpy().astype(bool)
    H = np.where(ni, H_inv, sw.H_SYNC)
    n_cfg, n_bus = ni.shape

    print(f"{args.csv}: {n_cfg} configurations x {n_bus} buses"
          f"{', torus' if lat.periodic else ''}")
    print(f"adjacent-bus effective resistance {R[0, 1]:.4f}\n")

    targets = {"worst_window": "worst 500 ms window (carries the local effect)",
               "local_excess": "500 ms from the event (barely shows one)"}
    lams = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 2.0)
    best: dict[str, tuple[float, float]] = {}

    for target, blurb in targets.items():
        y = piv[target].to_numpy()
        print(f"target: {target} -- {blurb}")
        pick = (None, 0.0)
        for lam in lams:
            W = np.exp(-R / lam)
            h = (H @ W.T) / W.sum(axis=1)
            r = stats.pearsonr(h.ravel(), y.ravel()).statistic
            print(f"    lam = {lam:<5} r = {r:+.4f}   R2 = {r * r:.3f}")
            if abs(r) > abs(pick[1]):
                pick = (lam, r)
        best[target] = pick
        print(f"    best: lam = {pick[0]}, r = {pick[1]:+.4f}, "
              f"R2 = {pick[1] ** 2:.3f}\n")

    lam, r = best["worst_window"]
    W = np.exp(-R / lam)
    h = (H @ W.T) / W.sum(axis=1)
    y = piv["worst_window"].to_numpy()
    print("the sign is the expected one: more inertia in the neighbourhood, less")
    print("RoCoF at the bus.  Most of that is carried by whether the bus is itself")
    print("a converter (see the comparison below, r = -0.82 on its own); weighting")
    print("in the neighbourhood adds a real but modest amount on top.\n")

    print("as a siting rule -- rank buses by predicted exposure, no simulation:")
    pred_worst = h.argmin(axis=1)
    rank = np.array([stats.rankdata(-y[i])[pred_worst[i]] for i in range(n_cfg)])
    print(f"    picks the true worst bus {(rank == 1).mean():.1%} of the time "
          f"(chance {1 / n_bus:.1%})")
    print(f"    its pick sits at rank {rank.mean():.2f} of {n_bus} on average")
    print(f"    its pick is in the true worst three {(rank <= 3).mean():.0%} "
          f"of the time")
    within = np.array([stats.pearsonr(h[i], y[i]).statistic
                       for i in range(n_cfg) if y[i].std() > 1e-12])
    print(f"    within-configuration r: mean {within.mean():+.3f}, "
          f"{(within < 0).mean():.0%} of configurations negative\n")

    print("against simpler predictors of the worst 500 ms window (pooled r):")
    nb = lat.A + np.eye(lat.N)
    for name, v in [("the bus's own inertia H_i", H),
                    ("mean inertia of the bus and its neighbours", (H @ nb.T) / nb.sum(1)),
                    ("number of converters within 2 hops",
                     ni.astype(float) @ (lat.dist <= 2).T)]:
        rr = stats.pearsonr(np.asarray(v, float).ravel(), y.ravel()).statistic
        print(f"    {name:<44s} r = {rr:+.4f}")
    print(f"    {'local inertia, exp(-R/lam) weighted':<44s} r = {r:+.4f}   <- best")


if __name__ == "__main__":
    main()
