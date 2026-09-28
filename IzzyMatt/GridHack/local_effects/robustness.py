"""Does the clustering result survive the modelling choices?

The headline number is the ratio between the worst bus when the converters are
clustered into one corner and the worst bus when the same converters are spread
evenly.  Several parameters in this model were picked rather than measured -- the
inertia left at a converter bus, the damping law, the line coupling, the length of
the measurement window, whether the lattice wraps.  If the result only holds at
one corner of that space it is not worth reporting, so every axis is swept here
one at a time about the default point.
"""

from __future__ import annotations

import numpy as np

import swing as sw
from designed import L, K_CONV, block, dispersed

DEFAULTS = dict(H_inv=sw.H_INV, damping="uniform", K=3.0, window=0.5, periodic=True)


def worst_and_ratio(H_inv: float, damping: str, K: float, window: float,
                    periodic: bool) -> tuple[float, float, float, float]:
    """(worst bus clustered, worst bus dispersed, their ratio, converter/sync ratio)."""
    lat = sw.Lattice(L, K=K, periodic=periodic)
    out = {}
    for tag, ni in (("block", block(lat)), ("disp", dispersed(lat))):
        net = sw.build_network(lat, ni, layout="flat", H_inv=H_inv, damping=damping)
        acc = np.zeros(lat.N)
        for b in range(lat.N):
            sim = sw.simulate_linear(net, sw.load_step(net, b, 0.5))
            acc += sw.metrics(sim, window=window)["worst_window"]
        acc /= lat.N
        mask = np.zeros(lat.N, bool)
        mask[list(ni)] = True
        out[tag] = (acc.max(), acc[mask].mean() / acc[~mask].mean())
    return out["block"][0], out["disp"][0], out["block"][0] / out["disp"][0], \
        out["block"][1]


def run(label: str, **over) -> None:
    kw = {**DEFAULTS, **over}
    b, d, ratio, conv = worst_and_ratio(**kw)
    flag = "" if ratio > 1.2 else "   <- effect largely gone"
    print(f"  {label:<34s} {b:>8.4f} {d:>9.4f} {ratio:>8.2f}x {conv:>10.2f}x{flag}")


def main() -> None:
    print(f"{L}x{L} torus, {K_CONV} converters, clustered in a {int(K_CONV ** 0.5)}x"
          f"{int(K_CONV ** 0.5)} block vs spread on a regular sublattice.")
    print("Total inertia is identical in every row of every table.\n")
    print(f"  {'variant':<34s} {'clustered':>8s} {'spread':>9s} "
          f"{'ratio':>9s} {'conv/sync':>11s}")
    print("  " + "-" * 74)

    print("  default")
    run("H_inv=0.1, uniform, K=3, 500ms")

    print("\n  inertia left at a converter bus (H, seconds)")
    for h in (0.02, 0.05, 0.5, 1.0, 2.0):
        run(f"H_inv = {h}", H_inv=h)

    print("\n  damping law")
    for dmp in ("proportional", "equal_ratio"):
        run(f"damping = {dmp}", damping=dmp)

    print("\n  line coupling strength")
    for k in (1.0, 1.5, 6.0, 12.0):
        run(f"K = {k}", K=k)

    print("\n  RoCoF measurement window")
    for w in (0.05, 0.1, 0.25, 1.0):
        run(f"window = {int(w * 1000)} ms", window=w)

    print("\n  boundary condition")
    run("open lattice (not a torus)", periodic=False)

    print("\n  under continuous stochastic forcing instead of a discrete event")
    print("  (worst-bus 500 ms RoCoF sd, Hz/s; 'uniform' is the control in which the")
    print("  same total fluctuation is spread over every bus rather than the wind)")
    import stochastic as st
    lat = sw.Lattice(L, K=3.0, periodic=True)
    print(f"    {'tau (s)':<10s} {'clustered':>10s} {'spread':>9s} {'ratio':>7s}"
          f"   |{'clustered':>10s} {'spread':>9s} {'ratio':>7s}")
    for tau in (0.2, 0.5, 1.0, 2.0, 5.0, 20.0):
        out = {}
        for where in ("at_converters", "uniform"):
            vals = []
            for mk in (block, dispersed):
                ni = mk(lat)
                net = sw.build_network(lat, ni, layout="flat", droop=sw.DROOP_GAIN)
                s = st.stationary(net, st.noise_profile(net, ni, where), tau=tau)
                vals.append(s["rocof_window_sd"].max())
            out[where] = vals
        a, b = out["at_converters"]
        c, d = out["uniform"]
        print(f"    {tau:<10.1f} {a:>10.4f} {b:>9.4f} {a / b:>7.2f}"
              f"   |{c:>10.4f} {d:>9.4f} {c / d:>7.2f}")
    print("  The clustering effect under noise is real but modest, and it needs the")
    print("  fluctuation to sit on the converters: in the uniform control it vanishes.")

    print("\n  A ratio near 1 means clustering stopped mattering.  The window sweep is")
    print("  the one that genuinely changes the answer, and that is a real property of")
    print("  the measurement, not of the grid: a long enough window averages the local")
    print("  swing away and leaves only the system-wide value.")


if __name__ == "__main__":
    main()
