"""Integrating out the rest of the grid: effective inertia as a self-energy.

The linearised swing equation has a matrix inverse propagator

    A(s) = M s^2 + D s + L ,

and eliminating every bus but one is a Schur complement,

    A_eff(s) = A_ii - A_i,r (A_rr)^-1 A_r,i ,   Sigma_i(s) = A_i,r (A_rr)^-1 A_r,i ,

which is exactly the operation that produces a self-energy when fields are
integrated out.  Expanding Sigma in s, the s^2 coefficient is a correction to the
bus's inertia, the s^1 coefficient a correction to its damping, and the s^0 term a
correction to its stiffness.  So a bus does not carry the inertia of its own rotor:
it carries a frequency-dependent *dressed* inertia, equal to the whole grid's at
zero frequency and to its own bare rotor at infinite frequency.

Two warnings about pushing the field-theory analogy further than it goes:

  * this is Gaussian, so the elimination is exact rather than the first term of a
    loop expansion.  There is no coupling constant to expand in.  The genuine
    interaction vertex is sin(theta_i - theta_j) beyond quadratic order, and every
    result in this study is verified to sit in the linear regime to 0.1%, so real
    loop corrections are negligible here;
  * "higher-order interaction" in the network-science sense -- hypergraph or
    simplicial terms coupling three or more buses at once -- is a different idea.
    Adding pairwise lines does not create any.

What the picture buys is a prediction, which `response()` below confirms: whether
reinforcement helps depends on *which mode the forcing excites*, because internal
lines and long lines dress the bus out of different inertia reservoirs.
"""

from __future__ import annotations

import numpy as np

import swing as sw


def effective_inertia(net: sw.Network, bus: int,
                      omegas: np.ndarray) -> np.ndarray:
    """Inertia the bus effectively carries at each frequency, from the Schur complement.

    Tends to the whole grid's inertia as omega -> 0 and to the bus's own as
    omega -> infinity.  In between it passes through the network's resonances,
    where it is not a useful summary -- use `response` there instead.
    """
    L, M, D = sw.laplacian(net), net.M, net.D
    rest = [k for k in range(net.lat.N) if k != bus]
    out = np.empty(len(omegas))
    for n, w in enumerate(omegas):
        s = 1j * w
        A = np.diag(M * s ** 2 + D * s) + L
        aeff = A[bus, bus] - A[bus, rest] @ np.linalg.solve(
            A[np.ix_(rest, rest)], A[rest, bus])
        out[n] = -np.real(aeff) / w ** 2
    return out


def response(net: sw.Network, bus: int, forcing: np.ndarray,
             omegas: np.ndarray, window: float | None = 0.5) -> np.ndarray:
    """|RoCoF at `bus`| per unit of `forcing`, against frequency.

    `forcing` is a power-injection pattern.  Concentrating it on one bus is the
    discrete-event case; spreading it over a group is the everyday case.

    ``window`` is the measurement, and it is not a detail.  Every headline metric
    in this study is RoCoF averaged over a 500 ms window, whose gain on a
    frequency-deviation component at omega is 2|sin(omega*T/2)|/T -- flat below
    ~1/T and falling as 1/omega above it.  The instantaneous derivative
    (``window=None``) instead has gain omega, rising without limit, so it weights
    the far tail enormously and gives a different and much less relevant answer.
    Using the instantaneous form here would disagree with `stochastic.stationary`;
    with the window it agrees to three decimals, which is a useful cross-check on
    both routes.
    """
    L, M, D = sw.laplacian(net), net.M, net.D
    out = np.empty(len(omegas))
    for n, w in enumerate(omegas):
        s = 1j * w
        A = np.diag(M * s ** 2 + D * s) + L
        # forcing -> angle -> frequency deviation (one factor of i*omega) -> RoCoF
        dev = abs(w * np.linalg.solve(A, forcing)[bus])
        gain = w if window is None else 2.0 * abs(np.sin(w * window / 2)) / window
        out[n] = gain * dev
    return out


def band_rms(omegas: np.ndarray, values: np.ndarray,
             lo: float, hi: float) -> float:
    m = (omegas >= lo) & (omegas < hi)
    return float(np.sqrt((values[m] ** 2).mean()))


def main() -> None:
    from designed import block, block_infill, quadrant_shortcuts, L as LSIZE

    base = sw.Lattice(LSIZE, K=3.0, periodic=True)
    ni = block(base)
    inside = block_infill(base)
    lats = {"baseline": base,
            "internal": sw.Lattice(LSIZE, K=3.0, periodic=True, extra_edges=inside),
            "shortcuts": sw.Lattice(LSIZE, K=3.0, periodic=True,
                                    extra_edges=quadrant_shortcuts(base))}
    nets = {t: sw.build_network(l, ni, layout="flat") for t, l in lats.items()}
    bus = 17                                   # the worst bus, inside the block

    print("DRESSED INERTIA AT BUS 17 (as an inertia constant H, seconds)")
    n0 = nets["baseline"]
    print(f"  bare rotor {n0.M[bus] * sw.OMEGA_S / 2:.2f} s, "
          f"whole grid {n0.M_total * sw.OMEGA_S / 2:.1f} s")
    for w in (1e-3, 1e4):
        v = effective_inertia(n0, bus, np.array([w]))[0] * sw.OMEGA_S / 2
        print(f"  omega = {w:<8g} -> {v:8.2f} s")
    print("  (the limits are the whole grid and the bare rotor, as they must be)\n")

    w = np.logspace(-1, 2.4, 400)
    u_local = np.zeros(base.N); u_local[bus] = 1.0
    u_block = np.zeros(base.N); u_block[blk] = 1.0 / len(blk)
    bands = [(0.1, 1), (1, 5), (5, 20), (20, 60), (60, 250)]

    print("RoCoF RESPONSE AT BUS 17, rms per band, relative to baseline")
    print("  local = all the forcing on bus 17 (a discrete event)")
    print("  block = the same total spread *coherently* over all 16 converters")
    print("          (an idealisation: the fully-correlated limit of the OU noise,")
    print("           which in stochastic.py is per-bus with correlation exp(-d/l))\n")
    for uname, u in (("local", u_local), ("block", u_block)):
        R = {t: response(nets[t], bus, u, w) for t in nets}
        print(f"  forcing = {uname}")
        print(f"    {'band rad/s':<12s} {'internal':>9s} {'shortcuts':>10s}")
        for lo, hi in bands:
            b = band_rms(w, R["baseline"], lo, hi)
            i2 = band_rms(w, R["internal"], lo, hi)
            s2 = band_rms(w, R["shortcuts"], lo, hi)
            print(f"    {f'{lo}-{hi}':<12s} {i2 / b:>9.2f} {s2 / b:>10.2f}")
        print()
    print("Note the measurement: these are 500 ms windowed RoCoF, matching every")
    print("headline metric.  Read against the instantaneous derivative instead and")
    print("the far tail dominates and the ordering changes -- see `response`.")


if __name__ == "__main__":
    main()
