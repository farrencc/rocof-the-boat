"""Checks that the fast path is telling the truth, and that the model is sane.

Run this before believing anything in the sweeps.  The linearised solver carries
the exhaustive 1820-configuration run, so it has to be shown to agree with the
nonlinear equation it stands in for, on the actual disturbance sizes used.
"""

from __future__ import annotations

import time

import numpy as np

import swing as sw

FAIL = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAIL.append(name)


def main() -> None:
    lat = sw.Lattice(4, K=3.0)

    print("steady state and construction")
    a = sw.build_network(lat, set())
    b = sw.build_network(lat, {0, 2, 5, 8})
    check("operating point does not depend on the inertia map",
          np.allclose(a.theta0, b.theta0, atol=1e-12),
          f"max diff {np.abs(a.theta0 - b.theta0).max():.1e}")
    check("power balances at the operating point",
          abs(a.P.sum()) < 1e-12)
    P, th = a.P, a.theta0
    resid = P - (lat.Kij * np.sin(th[:, None] - th[None, :])).sum(axis=1)
    check("steady state solves the power flow", np.abs(resid).max() < 1e-9,
          f"max residual {np.abs(resid).max():.1e}")
    check("total inertia is the same for every configuration with the same k",
          abs(sw.build_network(lat, {0, 2, 5, 8}).M_total
              - sw.build_network(lat, {0, 7, 10, 15}).M_total) < 1e-15)
    flat = sw.build_network(lat, {0, 1, 4, 5}, layout="flat")
    check("flat layout has a trivial operating point",
          np.allclose(flat.theta0, 0.0, atol=1e-12))

    print("\nconservation and known limits")
    net = sw.build_network(lat, {0, 2, 5, 8})
    dP = sw.gen_trip(net, 13)
    sim = sw.simulate(net, dP)
    coi0 = sim["rocof_coi"][0]
    expect = dP.sum() / net.M_total / (2 * np.pi)
    check("initial COI RoCoF equals dP / M_total",
          abs(coi0 - expect) < 1e-6 * abs(expect),
          f"{coi0:.6f} vs {expect:.6f} Hz/s")
    check("coupling terms cancel out of the COI",
          abs(sim["rocof_coi"][0] - expect) < 1e-9)
    check("system stays synchronised over the run",
          sim["max_angle_spread"] < 1.0,
          f"max relative angle {sim['max_angle_spread']:.3f} rad")

    print("\nlinear solver vs the nonlinear equation it replaces")
    # Tolerances are absolute, in Hz/s.  Judging local_excess against its own
    # range would be misleading: it is a difference of two numbers near 1 Hz/s,
    # so a 0.4% error on each shows up as several percent of the difference.
    # What the sweeps need from the linear solver is (a) an absolute error small
    # against the 1 Hz/s grid-code scale and (b) the bus ordering preserved,
    # since every conclusion drawn from it is comparative.
    from scipy import stats
    cases = [("checkerboard, gen trip (as used in the nonlinear sweep)",
              "checkerboard", lambda n: sw.gen_trip(n, 13)),
             ("flat, 0.5 pu load step (as used in the linear sweep)",
              "flat", lambda n: sw.load_step(n, 1, 0.5))]
    for tag, layout, mk in cases:
        ni = {0, 1, 4, 5} if layout == "flat" else {0, 2, 5, 8}
        net = sw.build_network(lat, ni, layout=layout)
        dP = mk(net)
        t0 = time.time(); nl = sw.simulate(net, dP); t_nl = time.time() - t0
        t0 = time.time(); li = sw.simulate_linear(net, dP); t_li = time.time() - t0
        mn, ml = sw.metrics(nl), sw.metrics(li)
        for key in ("rocof_500ms", "worst_window", "local_excess"):
            err = np.abs(mn[key] - ml[key]).max()
            check(f"{tag}: {key} within 0.01 Hz/s", err < 0.01,
                  f"max abs err {err:.5f} Hz/s")
        rho = stats.spearmanr(mn["local_excess"], ml["local_excess"]).statistic
        check(f"{tag}: bus ordering preserved", rho > 0.99, f"Spearman {rho:.4f}")
        print(f"       nonlinear {t_nl:.2f}s, linear {t_li:.3f}s "
              f"({t_nl / max(t_li, 1e-6):.0f}x faster)")

    print("\n  nonlinear correction vs disturbance size (should fall away)")
    net = sw.build_network(lat, {0, 1, 4, 5}, layout="flat")
    errs = []
    for size in (1.0, 0.5, 0.25):
        dP = sw.load_step(net, 1, size)
        e = np.abs(sw.metrics(sw.simulate(net, dP))["rocof_500ms"]
                   - sw.metrics(sw.simulate_linear(net, dP))["rocof_500ms"]).max()
        errs.append(e)
        print(f"       {size:4.2f} pu step: {e:.5f} Hz/s")
    check("error shrinks with disturbance size (it is amplitude, not a bug)",
          errs[0] > errs[1] > errs[2])

    print("\nlinearity of the response (justifies using the linear solver)")
    net = sw.build_network(lat, {0, 2, 5, 8})
    full = sw.metrics(sw.simulate(net, sw.gen_trip(net, 13)))["rocof_500ms"]
    half = sw.metrics(sw.simulate(net, 0.5 * sw.gen_trip(net, 13)))["rocof_500ms"]
    ratio = full / half
    check("halving the disturbance halves the response",
          np.abs(ratio - 2.0).max() < 0.02,
          f"ratio {ratio.mean():.4f} +- {ratio.std():.4f}")

    print("\nsymmetry: a homogeneous lattice must give a symmetric answer")
    net = sw.build_network(lat, set(), layout="flat")
    m = sw.metrics(sw.simulate_linear(net, sw.load_step(net, 0, 1.0)))
    g = np.abs(m["rocof_500ms"]).reshape(4, 4)
    check("uniform inertia, corner disturbance -> response symmetric about the diagonal",
          np.allclose(g, g.T, atol=1e-9),
          f"max asymmetry {np.abs(g - g.T).max():.1e}")
    # and with a symmetric converter placement, the symmetry must survive
    net = sw.build_network(lat, {0, 3, 12, 15}, layout="flat")
    m = sw.metrics(sw.simulate_linear(net, sw.load_step(net, 5, 1.0)))
    g = np.abs(m["rocof_500ms"]).reshape(4, 4)
    check("converters at all four corners -> response still symmetric",
          np.allclose(g, g.T, atol=1e-9),
          f"max asymmetry {np.abs(g - g.T).max():.1e}")

    print("\nthe null: uniform inertia must produce no spatial structure at all")
    for per in (False, True):
        lt = sw.Lattice(4, periodic=per)
        net = sw.build_network(lt, set(), layout="flat")
        acc = np.zeros(lt.N)
        for b in range(lt.N):
            acc += sw.metrics(sw.simulate_linear(net, sw.load_step(net, b, 0.5)))["local_excess"]
        acc /= lt.N
        check(f"{'torus' if per else 'open lattice'}: flat field when every bus has "
              f"the same inertia", np.ptp(acc) < 1e-12,
              f"spread {np.ptp(acc):.1e} Hz/s")
    lt = sw.Lattice(4, periodic=True)
    check("torus is vertex-transitive (every bus degree 4)",
          len(set(lt.A.sum(1))) == 1, f"degrees {sorted(set(lt.A.sum(1).astype(int)))}")

    print("\nprimary frequency response")
    lt = sw.Lattice(6, K=3.0, periodic=True)
    ni = {0, 1, 6, 7}
    plain = sw.build_network(lt, ni, layout="flat")
    gov = sw.build_network(lt, ni, layout="flat", droop=sw.DROOP_GAIN)
    dP = sw.load_step(plain, 3, 0.5)
    a = sw.metrics(sw.simulate_linear(plain, dP, t_end=20.0, dt=2e-3))
    check("droop = 0 leaves the pure swing equation untouched",
          np.abs(a["rocof_500ms"]
                 - sw.metrics(sw.simulate(plain, dP))["rocof_500ms"]).max() < 0.01)
    g = sw.simulate_linear(gov, dP, t_end=60.0, dt=4e-3)
    f = g["freq_coi"]
    # with droop, frequency settles at dP / (sum(D) + sum(droop)/omega_s)
    want = dP.sum() / (gov.D.sum() + gov.droop.sum() / sw.OMEGA_S) / (2 * np.pi)
    check("governor settles the system at the droop offset",
          abs(f[-1] - want) < 0.02 * abs(want),
          f"{f[-1]:.5f} vs {want:.5f} Hz")
    check("governor produces a nadir, then recovers",
          f.min() < f[-1] - 1e-4,
          f"nadir {1000 * f.min():.1f} mHz, settles {1000 * f[-1]:.1f} mHz")
    b = sw.metrics(sw.simulate_linear(gov, dP, t_end=20.0, dt=2e-3))
    # The defensible claim is about the window measured *from the event*: with a
    # 5 s governor lag, almost no extra MW have arrived by 500 ms, so that number
    # is governor-independent.  It is not a claim about `worst_window`, which
    # scans the whole run -- and without a governor the run never arrests, so its
    # late windows are picking up the unphysical tail.  The governor makes that
    # metric more trustworthy, not less, so it is reported rather than asserted.
    rel = (np.abs(b["rocof_500ms"] - a["rocof_500ms"]).max()
           / np.abs(a["rocof_500ms"]).max())
    check("RoCoF over the first 500 ms is governor-independent", rel < 0.02,
          f"changes by {rel:.2%}")
    wrel = np.abs(b["worst_window"] - a["worst_window"]).max() / a["worst_window"].max()
    print(f"       (worst-window-anywhere moves {wrel:.1%}: without a governor the "
          f"tail\n        never arrests, so late windows are not comparable)")

    print("\nstochastic forcing: Lyapunov solution vs brute-force simulation")
    import stochastic as stoch
    lt = sw.Lattice(4, K=3.0, periodic=True)
    ni = {0, 1, 4, 5}
    net = sw.build_network(lt, ni, layout="flat", droop=sw.DROOP_GAIN)
    sig = stoch.noise_profile(net, ni, "uniform")
    an = stoch.stationary(net, sig, tau=1.0)
    mc = stoch.montecarlo_check(net, sig, tau=1.0, t_end=8000.0, seed=7)
    for key in ("freq_sd", "rocof_window_sd"):
        r = (an[key] / mc[key]).mean()
        check(f"{key} matches Monte Carlo within 5%", abs(r - 1.0) < 0.05,
              f"analytic/MC = {r:.4f}")
    check("noise profiles carry the same total variance",
          abs((stoch.noise_profile(net, ni, "uniform") ** 2).sum()
              - (stoch.noise_profile(net, ni, "at_converters") ** 2).sum()) < 1e-12)
    flat_net = sw.build_network(lt, set(), layout="flat", droop=sw.DROOP_GAIN)
    s = stoch.stationary(flat_net, stoch.noise_profile(flat_net, set(), "uniform"),
                         tau=1.0)
    check("uniform inertia + uniform noise -> every bus identical",
          np.ptp(s["rocof_window_sd"]) < 1e-12,
          f"spread {np.ptp(s['rocof_window_sd']):.1e} Hz/s")

    print("\nspatial correlation of the imbalance")
    lt = sw.Lattice(6, K=3.0, periodic=True)
    ni = {0, 1, 6, 7}
    net = sw.build_network(lt, ni, layout="flat", droop=sw.DROOP_GAIN)
    sig = stoch.noise_profile(net, ni, "at_converters")
    g = np.zeros(lt.N, bool)
    g[list(ni)] = True
    check("zero correlation length reproduces independent noise exactly",
          abs(stoch.separation_sd(net, sig, g, 1.0, stoch.correlation(lt, 0.0))
              - stoch.separation_sd(net, sig, g, 1.0)) < 1e-12)
    seps = [stoch.separation_sd(net, sig, g, 1.0, stoch.correlation(lt, c))
            for c in (0.0, 1.0, 4.0)]
    check("correlated fluctuation drives the pocket harder than independent",
          seps[0] < seps[1] < seps[2],
          f"{[round(1000 * s, 1) for s in seps]} mHz")

    print("\nwhat the governor can and cannot reach")
    lt = sw.Lattice(6, K=3.0, periodic=True)
    ni = {0, 1, 6, 7}
    out = []
    for droop in (0.0, 100.0):
        n2 = sw.build_network(lt, ni, layout="flat", droop=droop)
        s2 = stoch.noise_profile(n2, ni, "at_converters")
        S, _, (a, w, p, x) = stoch.covariance(n2, s2, 1.0)
        c = np.zeros(S.shape[0])
        c[w] = n2.M / n2.M_total
        out.append((np.sqrt(c @ S @ c) / (2 * np.pi),
                    stoch.separation_sd(n2, s2, g, 1.0)))
    check("droop strongly damps the system-wide frequency",
          out[1][0] < 0.75 * out[0][0],
          f"{1000 * out[0][0]:.1f} -> {1000 * out[1][0]:.1f} mHz")
    check("droop leaves the local separation essentially untouched",
          abs(out[1][1] / out[0][1] - 1.0) < 0.02,
          f"{1000 * out[0][1]:.2f} -> {1000 * out[1][1]:.2f} mHz")
    lo = sw.build_network(lt, ni, layout="flat", H_inv=0.1, droop=sw.DROOP_GAIN)
    hi = sw.build_network(lt, ni, layout="flat", H_inv=4.0, droop=sw.DROOP_GAIN)
    sl = stoch.noise_profile(lo, ni, "at_converters")
    check("inertia, unlike droop, does reduce the local separation",
          stoch.separation_sd(hi, sl, g, 1.0) < 0.75 * stoch.separation_sd(lo, sl, g, 1.0),
          f"{1000 * stoch.separation_sd(lo, sl, g, 1.0):.2f} -> "
          f"{1000 * stoch.separation_sd(hi, sl, g, 1.0):.2f} mHz")

    print("\nnon-local links (the shortcut experiment)")
    from designed import block, field, quadrant_shortcuts
    lt = sw.Lattice(8, K=3.0, periodic=True)
    ed = quadrant_shortcuts(lt)
    sc = sw.Lattice(8, K=3.0, periodic=True, extra_edges=ed)
    ni = block(lt)
    # hop distance must stay the plain-lattice closed form when nothing is added
    man = np.array([[sum(min(abs(x - y), 8 - abs(x - y))
                         for x, y in zip(lt.coords[i], lt.coords[j]))
                     for j in range(lt.N)] for i in range(lt.N)], float)
    check("hop distance equals the wrapped Manhattan distance on a plain torus",
          np.array_equal(lt.dist, man))
    check("48 distinct shortcut links", len(set(map(frozenset, ed))) == 48,
          f"{len(ed)} listed")
    deg = sc.A.sum(axis=1)
    others = [i for i in range(sc.N) if i not in ni]
    check("every converter gains 3 links, every other bus exactly 1",
          set(deg[list(ni)]) == {7.0} and set(deg[others]) == {5.0},
          f"converters {sorted(set(deg[list(ni)].astype(int)))}, "
          f"others {sorted(set(deg[others].astype(int)))}")
    check("the shortcuts halve the graph diameter",
          lt.dist.max() == 8 and sc.dist.max() == 4,
          f"{lt.dist.max():.0f} -> {sc.dist.max():.0f} hops")
    a = sw.build_network(lt, ni, layout="flat", droop=sw.DROOP_GAIN)
    b = sw.build_network(sc, ni, layout="flat", droop=sw.DROOP_GAIN)
    check("adding lines changes neither the operating point nor total inertia",
          np.allclose(a.theta0, b.theta0, atol=1e-12)
          and abs(a.M_total - b.M_total) < 1e-15)
    # The *initial* COI RoCoF is dP/M_total exactly, whatever the graph: the
    # coupling terms cancel in the inertia-weighted sum.  Measured over 500 ms it
    # is not exactly topology-free, because by then damping and governor have
    # acted on the individual bus frequencies, and those do depend on the graph.
    # So the strong claim holds at t=0 and the 500 ms figure only nearly holds.
    dP = sw.load_step(a, 5, 0.5)
    c0 = [sw.simulate_linear(n, dP, t_end=0.1)["rocof_coi"][0] for n in (a, b)]
    check("initial system-wide RoCoF is exactly graph-independent",
          abs(c0[0] - c0[1]) < 1e-12, f"{c0[0]:.8f} vs {c0[1]:.8f} Hz/s")
    fa, fb = field(a), field(b)
    check("system-wide RoCoF and nadir barely move (no inertia was added)",
          abs(fa["coi"][0] / fb["coi"][0] - 1) < 0.005
          and abs(fa["coi_nadir"][0] / fb["coi_nadir"][0] - 1) < 0.005,
          f"COI {fa['coi'][0]:.5f} -> {fb['coi'][0]:.5f} Hz/s "
          f"({100 * (fb['coi'][0] / fa['coi'][0] - 1):+.2f}%), "
          f"{fa['coi_nadir'][0]*1000:.2f} -> {fb['coi_nadir'][0]*1000:.2f} mHz")
    check("but they do lower the worst bus",
          fb["worst_window"].max() < fa["worst_window"].max(),
          f"{fa['worst_window'].max():.4f} -> {fb['worst_window'].max():.4f} Hz/s")

    print("\n" + ("all checks passed" if not FAIL else f"FAILURES: {FAIL}"))
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
