"""What clustering costs you in SNSP headroom.

The operational question behind section 3.5 is not "is there a local effect" but
"how much non-synchronous generation can this grid carry before RoCoF binds".
That is what an SNSP cap expresses, and curtailment is the instrument that
enforces it.

So this sweeps the non-synchronous share from 0 upwards, three ways at each
level: a single compact cluster of converters, a maximally spread placement, and
an ensemble of random placements as an unbiased middle case.  Unlike the earlier
sweeps, total inertia is *not* held fixed -- it falls as the share rises, which is
the whole point.  What is compared at each share is the geography.

Two disturbance regimes, because they do not agree:

  step   a discrete event -- a unit trips, a load block switches in.
  noise  continuous stochastic imbalance, the everyday regime.  The fluctuation
         is placed **at the converter buses**, because wind and solar output is
         what actually varies.  A control run with the same total fluctuation
         spread uniformly over all buses is reported alongside: clustering barely
         matters there, and the contrast localises the mechanism.  It is not that
         a low-inertia region is fragile in the abstract -- it is that co-locating
         the fluctuating sources with the missing inertia removes both the nearby
         machines that would absorb the imbalance and the spatial averaging
         between independent fluctuations.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import stochastic as st
import swing as sw

L = 8


def clustered(lat: sw.Lattice, k: int, centre: int = 0) -> set[int]:
    """The k buses nearest one point: a single compact block of wind."""
    order = sorted(range(lat.N), key=lambda b: (lat.dist[centre, b], b))
    return set(order[:k])


def spread(lat: sw.Lattice, k: int) -> set[int]:
    """k buses placed as far from each other as possible (farthest-point greedy).

    Greedy, so at some k it returns a slightly lumpy set; the random ensemble is
    the smoother comparator and the one the headroom figures are read from.
    """
    chosen = [0]
    while len(chosen) < k:
        best = max((b for b in range(lat.N) if b not in chosen),
                   key=lambda b: (min(lat.dist[b, c] for c in chosen), -b))
        chosen.append(best)
    return set(chosen[:k])


def worst_bus_step(net: sw.Network, step: float = 0.5) -> tuple[float, float]:
    """Worst bus for RoCoF and for frequency excursion, over a step at every bus.

    Both are reported because a grid code constrains both: RoCoF relays and
    loss-of-mains protection watch df/dt, under-frequency load shedding watches f.
    """
    N = net.lat.N
    rocof = np.zeros(N)
    nadir = np.zeros(N)
    for b in range(N):
        m = sw.metrics(sw.simulate_linear(net, sw.load_step(net, b, step),
                                          t_end=5.0, dt=2e-3))
        rocof += m["worst_window"]
        nadir += np.abs(m["nadir"])
    return float((rocof / N).max()), float((nadir / N).max())


def worst_bus_noise(net: sw.Network, ni: set[int], tau: float,
                    where: str) -> tuple[float, float, float]:
    """Worst-bus RoCoF sd, system-wide RoCoF sd, and worst-bus frequency sd."""
    s = st.stationary(net, st.noise_profile(net, ni, where, scale="per_bus"),
                      tau=tau)
    return (float(s["rocof_window_sd"].max()), float(s["coi_rocof_window_sd"][0]),
            float(s["freq_sd"].max()))


def evaluate(lat: sw.Lattice, ni: set[int], tau: float, droop: float) -> dict:
    net = sw.build_network(lat, ni, layout="flat", droop=droop)
    r, n = worst_bus_step(net)
    out = {"step": r, "step_freq": n}
    for where, key in (("at_converters", "noise"), ("uniform", "noise_uniform")):
        w, c, f = worst_bus_noise(net, ni, tau, where)
        out[key], out[f"{key}_coi"], out[f"{key}_freq"] = w, c, f
    return out


def headroom(share: np.ndarray, clus: np.ndarray, comp: np.ndarray,
             at: float) -> float:
    """Share a dispersed grid could run at for the same worst-bus RoCoF.

    `comp` is made monotone first: the comparator is noisy in k, and a
    non-monotone curve makes 'the share at which it reaches this level'
    ill-defined.  Taking the running maximum is the conservative reading.
    """
    mono = np.maximum.accumulate(comp)
    i = int(np.argmin(np.abs(share - at)))
    target = clus[i]
    if target <= mono[0]:
        return float(share[0])
    j = int(np.searchsorted(mono, target))
    if j >= len(mono):
        return float("nan")
    v0, v1 = mono[j - 1], mono[j]
    s0, s1 = share[j - 1], share[j]
    if v1 == v0:
        return float(s1)
    return float(s0 + (s1 - s0) * (target - v0) / (v1 - v0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=st.TAU_C)
    ap.add_argument("--droop", type=float, default=sw.DROOP_GAIN)
    ap.add_argument("--random", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="snsp.csv")
    args = ap.parse_args()

    lat = sw.Lattice(L, K=3.0, periodic=True)
    rng = np.random.default_rng(args.seed)
    ks = list(range(4, 53, 4))
    rows = []

    base_step, base_freq = worst_bus_step(sw.build_network(lat, set(), layout="flat",
                                                           droop=args.droop))
    print(f"{L}x{L} torus, governor droop {args.droop:.0f} pu/pu, noise "
          f"correlation time {args.tau} s, {args.random} random draws per level")
    print("Total inertia falls as the share rises.  'step' is normalised to the")
    print("all-synchronous grid; noise figures are raw 500 ms RoCoF sd (Hz/s).\n")
    print(f"  {'share':>6s} | {'step: clus':>10s} {'rand':>7s} {'ratio':>6s} | "
          f"{'noise: clus':>11s} {'rand':>7s} {'ratio':>6s} | {'COI':>7s}")
    print("  " + "-" * 76)

    for k in ks:
        rec = {"k": k, "share": k / lat.N}
        rec |= {f"clustered_{m}": v
                for m, v in evaluate(lat, clustered(lat, k), args.tau, args.droop).items()}
        rec |= {f"spread_{m}": v
                for m, v in evaluate(lat, spread(lat, k), args.tau, args.droop).items()}
        draws = [evaluate(lat, set(rng.choice(lat.N, k, replace=False).tolist()),
                          args.tau, args.droop) for _ in range(args.random)]
        for m in draws[0]:
            rec[f"random_{m}"] = float(np.mean([d[m] for d in draws]))
            rec[f"random_{m}_sd"] = float(np.std([d[m] for d in draws]))
        rec["clustered_step_norm"] = rec["clustered_step"] / base_step
        rec["random_step_norm"] = rec["random_step"] / base_step
        rec["clustered_step_freq_norm"] = rec["clustered_step_freq"] / base_freq
        rec["random_step_freq_norm"] = rec["random_step_freq"] / base_freq
        rows.append(rec)
        print(f"  {rec['share']:>6.1%} | {rec['clustered_step_norm']:>10.3f} "
              f"{rec['random_step_norm']:>7.3f} "
              f"{rec['clustered_step'] / rec['random_step']:>6.2f} | "
              f"{rec['clustered_noise']:>11.4f} {rec['random_noise']:>7.4f} "
              f"{rec['clustered_noise'] / rec['random_noise']:>6.2f} | "
              f"{rec['clustered_noise_coi']:>7.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    sh = df["share"].to_numpy()

    print("\nCONTROL: the same total fluctuation spread over every bus instead")
    print("of sitting on the converters -- clustering then stops mattering:\n")
    print(f"  {'share':>6s} {'clustered':>10s} {'random':>8s} {'ratio':>7s}")
    for _, r in df.iterrows():
        print(f"  {r['share']:>6.1%} {r['clustered_noise_uniform']:>10.4f} "
              f"{r['random_noise_uniform']:>8.4f} "
              f"{r['clustered_noise_uniform'] / r['random_noise_uniform']:>7.2f}")

    print("\nSNSP HEADROOM LOST TO CLUSTERING")
    print("For a clustered grid at share X, the share a randomly-sited grid could")
    print("carry for the same worst-bus RoCoF:\n")
    for metric, label in (("step", "discrete event"), ("noise", "continuous noise")):
        cl = df[f"clustered_{metric}"].to_numpy()
        rd = df[f"random_{metric}"].to_numpy()
        print(f"  {label}:")
        for at in (0.125, 0.25, 0.375, 0.5):
            eq = headroom(sh, cl, rd, at)
            i = int(np.argmin(np.abs(sh - at)))
            if np.isnan(eq):
                print(f"    clustered at {sh[i]:>5.1%}  ->  beyond the swept range "
                      f"(headroom > {sh[-1] - sh[i]:.0%})")
            else:
                print(f"    clustered at {sh[i]:>5.1%}  ->  random siting could "
                      f"carry {eq:>5.1%}   ({eq - sh[i]:+.1%})")
        print()

    print("A global SNSP limit is blind to all of it -- the system-wide RoCoF is")
    print("the same for both geographies at every share:")
    for _, r in df.iloc[[1, 3, 5, 7]].iterrows():
        print(f"    share {r['share']:>5.1%}:  COI clustered "
              f"{r['clustered_noise_coi']:.5f}   random {r['random_noise_coi']:.5f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
