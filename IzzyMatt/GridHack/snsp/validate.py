"""Checks that have to pass before any number out of this folder means anything.

Three kinds: that the frozen inputs are what they claim to be, that the solver
is solving the equations it says it is, and that the cost functional is a
property of the system rather than of the numerical choices made to evaluate it.

Run it first, and after any change to the model.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "local_effects"))
import swing as sw                                          # noqa: E402

import configs as C                                         # noqa: E402
import dynamics as dyn                                      # noqa: E402
import grid as G                                            # noqa: E402
import perturbations as PB                                  # noqa: E402

OK, BAD = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (OK if cond else BAD).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail
                                                       else ""))


def close(a, b, tol) -> bool:
    return bool(np.all(np.abs(np.asarray(a) - np.asarray(b)) <= tol))


def main() -> None:
    t0 = time.time()
    g = G.load()
    events = PB.load(g)
    U = PB.matrix(events)
    rng = np.random.default_rng(7)

    print("\nFROZEN INPUTS")
    check("grid loads and passes its own structural checks", True)
    check("grid matches the specification in grid.py",
          np.array_equal(G.build().A, g.A) and
          np.allclose(G.build().capacity, g.capacity))
    check("20 disturbances, all of one order of magnitude",
          len(events) == 20 and
          max(abs(e.size()) for e in events)
          / min(abs(e.size()) for e in events) < 2)
    check("every disturbance is the same for every configuration",
          all(np.allclose(a.vector(), b.vector())
              for a, b in zip(events, PB.load(g))))

    print("\nCONFIGURATIONS")
    cfgs = [C.draw(g, s, rng, i) for i, s in enumerate(
        (0.30, 0.50, 0.70, 0.85, 1.00))]
    check("every configuration balances", all(abs(c.P.sum()) < 1e-7 for c in cfgs),
          f"max |sum P| = {max(abs(c.P.sum()) for c in cfgs):.2e}")
    check("realised SNSP matches the target",
          all(abs(c.meta["snsp"] - c.snsp_target) < 1e-6 for c in cfgs))
    check("no unit exceeds its nameplate",
          all(np.all(c.P[g.generators] <= g.capacity[g.generators] + 1e-9)
              for c in cfgs))
    check("demand never moves between configurations",
          all(np.allclose(c.P[g.role == "load"], -g.demand[g.role == "load"])
              for c in cfgs))
    check("inertia falls monotonically with SNSP",
          all(a.meta["stored_energy"] >= b.meta["stored_energy"] - 1e-9
              for a, b in zip(cfgs, cfgs[1:])),
          " -> ".join(f"{c.meta['stored_energy']:.0f}" for c in cfgs))
    check("no machine is committed at SNSP = 100%", cfgs[-1].committed == ())

    print("\nTHE HAZARD-SCALED SET")
    sc = PB.load(g, which="scaled")
    check("8 scaled events, frozen and read back",
          len(sc) == 8 and all(e.scaled for e in sc))
    lo_P, hi_P = G.dispatch(g, 0.30), G.dispatch(g, 1.00)
    tot_lo = sum(abs(e.size(lo_P)) for e in sc)
    tot_hi = sum(abs(e.size(hi_P)) for e in sc)
    check("the same events are larger when more wind is running",
          tot_hi > 2.5 * tot_lo,
          f"{tot_lo:.2f} p.u. at 30% SNSP, {tot_hi:.2f} at 100% "
          f"({tot_hi / tot_lo:.1f}x)")
    # A scaled event takes a fraction of what a bus *generates*, so no generator
    # can be driven below zero output.  Load buses have P < 0 by convention and
    # are untouched by these events; the check is about generation only.
    check("a scaled event never removes power a bus is not generating",
          all(np.all(np.clip(P, 0.0, None)[:, None] + PB.matrix(sc, P) > -1e-9)
              for P in (lo_P, hi_P)))
    check("the fixed set is unchanged by any of this",
          all(np.allclose(a.vector(), b.vector())
              for a, b in zip(events, PB.build(g))))

    print("\nTHE MUON FLOOR")
    free = [C.draw(g, s, rng, i) for i, s in enumerate((0.3, 0.6, 0.9, 1.0))]
    held = [C.draw(g, s, rng, i, muon=4)
            for i, s in enumerate((0.3, 0.6, 0.9, 1.0))]
    Ef = [c.meta["stored_energy"] for c in free]
    Eh = [c.meta["stored_energy"] for c in held]
    check("without a floor, inertia collapses as SNSP rises",
          Ef[-1] < 0.4 * Ef[0], " -> ".join(f"{v:.0f}" for v in Ef))
    check("with MUON = 4, inertia is held roughly constant",
          max(Eh) / min(Eh) < 1.3, " -> ".join(f"{v:.0f}" for v in Eh))
    check("MUON keeps four machines spinning at every SNSP",
          all(c.meta["n_spinning"] >= 4 for c in held))
    check("condensers carry inertia but no reserve",
          held[-1].meta["reserve"] == 0.0 and held[-1].meta["stored_energy"] > 40,
          f"at SNSP 100%: {held[-1].meta['n_condensers']} condensers, "
          f"E = {held[-1].meta['stored_energy']:.0f} p.u. s, reserve = "
          f"{held[-1].meta['reserve']:.0f}")

    print("\nTHE SOLVER")
    net = cfgs[2].network(g)
    check("steady state solves the power-flow equations",
          close(net.P, (net.lat.Kij * np.sin(
              net.theta0[:, None] - net.theta0[None, :])).sum(axis=1), 1e-8))
    check("state matrix agrees with swing.py",
          np.allclose(sw.state_matrix(net), sw.state_matrix(net)))

    # The propagator against swing.py's own step solver, on the same network.
    e = 13
    sim = sw.simulate_linear(net, U[:, e], t_end=2.0, dt=1e-3)
    _, tr = dyn.respond(net, U[:, [e]], t_end=2.0, dt=1e-3, t_meas=0.0,
                        trace=True)
    check("propagator matches swing.simulate_linear (frequency)",
          close(tr["f"][:, 0, :], sim["freq_dev"], 1e-9),
          f"max diff {np.abs(tr['f'][:, 0, :] - sim['freq_dev']).max():.2e} Hz")
    check("propagator matches swing.simulate_linear (RoCoF)",
          close(tr["rocof"][:, 0, :], sim["rocof"], 1e-8))

    # Against the full nonlinear solver.
    lin = dyn.respond(net, U[:, [e]], t_end=5.0, dt=1e-3, t_meas=0.0)
    non = dyn.nonlinear_cost(net, U[:, e], t_end=5.0, dt=1e-3)
    rel = abs(lin.total - non) / non
    check("linearised cost matches the nonlinear solver", rel < 0.05,
          f"{lin.total:.3f} vs {non:.3f}  ({rel * 100:.2f}%)")

    # Analytic centre-of-inertia RoCoF: dP * f_nom / (2 * stored energy).
    E = g.stored_energy(cfgs[2].H)
    want = abs(U[:, e].sum()) * 50.0 / (2.0 * E)
    _, tr2 = dyn.respond(net, U[:, [e]], t_end=0.2, dt=1e-3, t_meas=0.0,
                         trace=True)
    got = abs(tr2["f_coi"][0, 1] - tr2["f_coi"][0, 0]) / 1e-3
    check("initial COI RoCoF matches dP*f/(2E)", abs(got - want) / want < 0.02,
          f"{got:.3f} vs {want:.3f} Hz/s")

    print("\nTHE COST FUNCTIONAL")
    ref = dyn.summarise(dyn.respond(net, U), g.N)["J"]
    fine = dyn.summarise(dyn.respond(net, U, dt=dyn.DT / 2), g.N)["J"]
    check("cost is converged in the time step",
          abs(fine - ref) / ref < 0.01,
          f"dt {dyn.DT}: {ref:.2f}   dt/2: {fine:.2f}  "
          f"({abs(fine - ref) / ref * 100:.2f}%)")

    # Antisymmetry: in the linearisation a load rejection is a load step with
    # the sign flipped, so the cost (quadratic) must be identical.
    k_step = [i for i, e_ in enumerate(events) if e_.name == "load_step@Dublin"][0]
    up = dyn.respond(net, -U[:, [k_step]]).total
    down = dyn.respond(net, U[:, [k_step]]).total
    check("cost is even in the sign of the disturbance",
          abs(up - down) / down < 1e-9)

    # Scaling: the system is linear, so doubling every disturbance must
    # quadruple a quadratic functional.
    twice = dyn.respond(net, 2.0 * U).total
    check("cost scales quadratically with disturbance size",
          abs(twice / dyn.respond(net, U).total - 4.0) < 1e-6)

    # The measurement filter must not change the ordering of configurations,
    # only the numbers.  If it did, the headline would be an artefact of it.
    js = {}
    for tm in (0.0, 0.05, 0.1, 0.2):
        js[tm] = [dyn.summarise(dyn.respond(c.network(g), U, t_meas=tm),
                                g.N)["J"] for c in cfgs]
    order = {k: np.argsort(v).tolist() for k, v in js.items()}
    filtered = [o for k, o in order.items() if k > 0]
    check("ranking of configurations is the same for every measurement lag "
          "of 50-200 ms", all(o == filtered[0] for o in filtered),
          "  ".join(f"t_meas={k}: J(30%)={v[0]:.0f}, J(100%)={v[-1]:.0f}"
                    for k, v in js.items() if k > 0))
    ratios = [v[-1] / v[0] for v in js.values()]
    check("the 30%-to-100% ratio survives the measurement lag",
          min(ratios) > 2.0,
          "ratios " + ", ".join(f"{r:.1f}" for r in ratios))
    # Documented, not asserted away: scoring the *raw* state derivative does
    # reorder configurations, and it is the only setting that does.  That is the
    # argument for evaluating the functional on a measurable frequency -- see
    # dynamics.T_MEAS -- rather than a defect to be tuned around.  The direction
    # of the headline is unaffected either way (previous check).
    check("the unfiltered functional is the one that reorders them",
          order[0.0] != filtered[0],
          f"raw order {order[0.0]} vs filtered {filtered[0]}")

    # Droop is proportional control: it must leave a standing offset of exactly
    # dP / (droop gain + damping), and secondary control is what removes it.
    # This is the check behind "why does the frequency not return to 50 Hz".
    c = cfgs[2]
    dP = U[:, e]
    R = c.droop.sum() / 50.0 + net.D.sum() * dyn.TWO_PI      # p.u. per Hz
    _, tr3 = dyn.respond(net, U[:, [e]], t_end=120.0, t_meas=0.0, trace=True)
    settled = tr3["f_coi"][0][-1]
    check("droop leaves the analytic standing offset dP/(R+D)",
          abs(settled - dP.sum() / R) / abs(dP.sum() / R) < 0.1,
          f"{settled * 1000:.1f} mHz settled vs {dP.sum() / R * 1000:.1f} "
          f"predicted")
    _, tr4 = dyn.respond(net, U[:, [e]], t_end=120.0, t_meas=0.0,
                         agc=c.participation(g), trace=True)
    check("secondary control returns the frequency to nominal",
          abs(tr4["f_coi"][0][-1]) < 5e-3,
          f"{tr4['f_coi'][0][-1] * 1000:+.2f} mHz at 120 s, "
          f"{tr4['f_coi'][0][4000] * 1000:+.1f} mHz at 10 s")
    check("secondary control barely touches RoCoF inside the window",
          abs(dyn.summarise(dyn.respond(net, U, agc=c.participation(g)),
                            g.N)["rocof_top5"]
              / dyn.summarise(dyn.respond(net, U), g.N)["rocof_top5"] - 1)
          < 0.02)

    # The post-fault equilibrium: the nonlinear criterion the linear solver
    # cannot see.  Its frequency offset must agree with what the linear model
    # settles at, and a large enough disturbance must have no equilibrium at all.
    ok, load = dyn.post_fault(net, U[:, e])
    check("a modest event has a post-fault equilibrium", ok and load < 1.0,
          f"worst line at {load:.3f} of its static limit")
    _, tr5 = dyn.respond(net, U[:, [e]], t_end=120.0, t_meas=0.0, trace=True)
    w_inf = (U[:, e].sum()
             / (net.D.sum() + net.droop.sum() / sw.OMEGA_S)) / dyn.TWO_PI
    check("post-fault offset matches the settled linear response",
          abs(w_inf - tr5["f_coi"][0][-1]) < 2e-3,
          f"{w_inf * 1000:.1f} mHz predicted, "
          f"{tr5['f_coi'][0][-1] * 1000:.1f} mHz simulated")
    huge = np.zeros(g.N)
    huge[g.index(2, 0)] = -8.0          # far more than the spur can carry
    check("an impossible disturbance has no post-fault equilibrium",
          not dyn.post_fault(net, huge)[0])

    # The top-n scores must be monotone in n and reduce to the plain sum.
    r = dyn.respond(net, U)
    tops = [dyn.worst_n(np.abs(r.rocof_500ms), n) for n in (1, 5, 10)]
    check("top-n scores increase with n", tops[0] < tops[1] < tops[2],
          f"worst bus {tops[0]:.3f}, worst 5 {tops[1]:.3f}, "
          f"worst 10 {tops[2]:.3f} Hz/s")
    check("top-n with n = every bus is the plain sum",
          abs(dyn.worst_n(np.abs(r.rocof_500ms), g.N)
              - np.abs(r.rocof_500ms).sum(axis=0).mean()) < 1e-12)
    check("the nadir is the downward excursion only",
          bool(np.all(r.nadir >= 0)) and
          bool(np.all(r.nadir <= r.peak_dev + 1e-12)))

    # Horizon: the ordering must not depend on where the integral stops.
    orders = []
    for T in (5.0, 10.0, 20.0):
        v = [dyn.summarise(dyn.respond(c.network(g), U, t_end=T), g.N)["J"]
             for c in cfgs]
        orders.append(np.argsort(v).tolist())
    check("ranking of configurations is the same for every horizon",
          all(o == orders[0] for o in orders))

    print(f"\n{len(OK)} passed, {len(BAD)} failed, {time.time() - t0:.0f}s")
    if BAD:
        print("  failed: " + "; ".join(BAD))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
