"""Sweep every way of placing non-inertial generation on the toy lattice.

The question (problem sheet section 3.5) is whether inertia is only a global
quantity or whether where you put it matters locally.  The test here holds the
power injections fixed at every bus and changes only which generators are
synchronous and which are inverter-based.  Because the steady state of the swing
equation depends on P and K and not on M, every configuration in the sweep has

  * an identical pre-fault operating point (same angles, same line flows), and
  * an identical total system inertia (same count of converted generators),

so the only thing that varies is the *geography* of inertia.  On a 4x4 lattice
with 8 generators and 4 of them converted there are C(8,4) = 70 configurations,
which is small enough to enumerate exhaustively rather than sample.

Writes one tidy row per (configuration, disturbance, bus) to results.csv.
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import pandas as pd

import swing as sw


def cluster_index(lat: sw.Lattice, buses: tuple[int, ...]) -> float:
    """Mean pairwise lattice distance within a set: low = clustered, high = spread."""
    pairs = list(itertools.combinations(buses, 2))
    return float(np.mean([lat.dist[i, j] for i, j in pairs]))


def converter_density(lat: sw.Lattice, buses: tuple[int, ...]) -> np.ndarray:
    """How much inverter-based generation sits near each bus, 1/(1+d) weighted."""
    return np.array([sum(1.0 / (1.0 + lat.dist[i, g]) for g in buses)
                     for i in range(lat.N)])


def _aggregate_config(lat, net, non_inertial, events, run, ci, density, ni_set,
                      layout, damping, H_inv) -> list[dict]:
    """One row per bus, already averaged over every disturbance in the set.

    The exhaustive flat sweep is 1820 configurations x 16 disturbances x 16
    buses; writing that raw is half a million rows for a quantity the analysis
    immediately averages anyway.  Averaging here keeps the artefact exclusion
    (a bus whose own generator just tripped contributes nothing to its own mean)
    while emitting the same columns the analysis expects.
    """
    N = lat.N
    acc = {k: np.zeros(N) for k in
           ("local_excess", "abs_rocof", "worst_window", "peak_instant", "t_peak")}
    n_obs = np.zeros(N)
    coi = 0.0
    for kind, at, dP in events:
        sim = run(net, dP)
        if sim["max_angle_spread"] > 1.2:
            raise RuntimeError(f"lost synchronism: {non_inertial} {kind}@{at}")
        m = sw.metrics(sim)
        keep = np.ones(N, bool)
        if kind == "gen_trip":
            keep[at] = False
        acc["local_excess"][keep] += m["local_excess"][keep]
        acc["abs_rocof"][keep] += np.abs(m["rocof_500ms"])[keep]
        acc["worst_window"][keep] += m["worst_window"][keep]
        acc["peak_instant"][keep] += m["peak_instant"][keep]
        acc["t_peak"][keep] += m["t_peak"][keep]
        n_obs += keep
        coi += abs(m["coi_rocof_500ms"][0])
    for k in acc:
        acc[k] /= np.maximum(n_obs, 1)
    coi /= len(events)

    rows = []
    for i in range(N):
        r, c = lat.coords[i]
        rows.append({
            "non_inertial": "-".join(map(str, non_inertial)),
            "cluster_index": ci, "layout": layout, "damping": damping,
            "periodic": lat.periodic,
            "H_inv": H_inv, "bus": i, "row": r, "col": c,
            "is_gen": bool(net.P[i] > 0),
            "is_noninertial": i in ni_set,
            "dist_to_nearest_ni": min(lat.dist[i, g] for g in non_inertial),
            "converter_density": density[i],
            "coi_rocof_500ms": coi,
            "n_obs": int(n_obs[i]),
            **{k: v[i] for k, v in acc.items()},
        })
    return rows


def run_config(lat: sw.Lattice, non_inertial: tuple[int, ...], gens: np.ndarray,
               loads: np.ndarray, damping: str, H_inv: float,
               kinds: tuple[str, ...], layout: str, solver: str,
               step: float, aggregate: bool = False) -> list[dict]:
    """Every disturbance in `kinds`, for one inertia configuration."""
    net = sw.build_network(lat, set(non_inertial), layout=layout,
                           H_inv=H_inv, damping=damping)
    run = sw.simulate_linear if solver == "linear" else sw.simulate
    ni_set = set(non_inertial)
    density = converter_density(lat, non_inertial)
    ci = cluster_index(lat, non_inertial)

    events: list[tuple[str, int, np.ndarray]] = []
    if "gen_trip" in kinds:
        events += [("gen_trip", int(g), sw.gen_trip(net, int(g))) for g in gens]
    if "load_step" in kinds:
        events += [("load_step", int(b), sw.load_step(net, int(b), step))
                   for b in loads]

    if aggregate:
        return _aggregate_config(lat, net, non_inertial, events, run, ci, density,
                                 ni_set, layout, damping, H_inv)

    rows: list[dict] = []
    for kind, at, dP in events:
        sim = run(net, dP)
        if sim["max_angle_spread"] > 1.2:      # ~70 deg: no longer a small signal
            raise RuntimeError(f"lost synchronism: {non_inertial} {kind}@{at}")
        m = sw.metrics(sim)
        for i in range(lat.N):
            r, c = lat.coords[i]
            rows.append({
                "non_inertial": "-".join(map(str, non_inertial)),
                "cluster_index": ci,
                "layout": layout,
                "damping": damping,
                "H_inv": H_inv,
                "kind": kind,
                "at": at,
                "at_is_noninertial": at in ni_set,
                "bus": i, "row": r, "col": c,
                "is_gen": bool(net.P[i] > 0),
                "is_noninertial": i in ni_set,
                # A tripped generator leaves its own bus with no injection and,
                # if it was a converter, almost no inertia.  That bus is a
                # modelling artefact, not a measurement, so mark it for exclusion.
                # A load step removes nothing, so no bus needs excluding there.
                "artifact_bus": (i == at) and kind == "gen_trip",
                "dist_to_fault": lat.dist[i, at],
                "dist_to_nearest_ni": min(lat.dist[i, g] for g in non_inertial),
                "converter_density": density[i],
                "rocof_500ms": m["rocof_500ms"][i],
                "worst_window": m["worst_window"][i],
                "peak_instant": m["peak_instant"][i],
                "t_peak": m["t_peak"][i],
                "coi_rocof_500ms": m["coi_rocof_500ms"][i],
                "local_excess": m["local_excess"][i],
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--k", type=int, default=4, help="generators converted to inverter")
    ap.add_argument("--K", type=float, default=3.0, help="line coupling strength")
    ap.add_argument("--periodic", action="store_true",
                    help="wrap the lattice into a torus: every bus has degree 4, "
                         "so topology cannot masquerade as an inertia effect")
    ap.add_argument("--damping", default="uniform",
                    choices=["uniform", "proportional", "equal_ratio"])
    ap.add_argument("--H-inv", type=float, default=sw.H_INV)
    ap.add_argument("--kinds", default="gen_trip,load_step")
    ap.add_argument("--layout", default="checkerboard",
                    choices=["checkerboard", "flat"])
    ap.add_argument("--solver", default="nonlinear", choices=["nonlinear", "linear"])
    ap.add_argument("--step", type=float, default=1.0, help="load step size, p.u.")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--limit", type=int, default=0, help="stop after N configs (debug)")
    ap.add_argument("--aggregate", action="store_true",
                    help="average over disturbances before writing (big sweeps)")
    args = ap.parse_args()

    lat = sw.Lattice(args.L, K=args.K, periodic=args.periodic)
    if args.layout == "checkerboard":
        P = sw.checkerboard_power(lat)
        gens, loads = np.where(P > 0)[0], np.where(P < 0)[0]
        placeable = gens.tolist()
    else:
        # every bus carries a machine, so a converter can go anywhere and the
        # disturbance is a load step at any bus
        gens, loads = np.array([], int), np.arange(lat.N)
        placeable = list(range(lat.N))
    configs = list(itertools.combinations(placeable, args.k))
    if args.limit:
        configs = configs[:args.limit]
    kinds = tuple(args.kinds.split(","))

    print(f"L={args.L} N={lat.N} layout={args.layout} solver={args.solver}")
    print(f"{len(configs)} configurations x {args.k} converted, "
          f"damping={args.damping}, H_inv={args.H_inv}, kinds={kinds}")

    rows: list[dict] = []
    t0 = time.time()
    every = max(1, len(configs) // 20)
    for n, cfg in enumerate(configs, 1):
        rows += run_config(lat, cfg, gens, loads, args.damping, args.H_inv,
                           kinds, args.layout, args.solver, args.step,
                           args.aggregate)
        if n % every == 0 or n == len(configs):
            el = time.time() - t0
            print(f"  {n}/{len(configs)} configs, {el:.0f}s elapsed, "
                  f"{el / n * (len(configs) - n):.0f}s left", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} rows, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
