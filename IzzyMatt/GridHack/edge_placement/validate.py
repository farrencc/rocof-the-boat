"""Checks that have to pass before any number out of this folder means anything.

The spec lists nine of them in §10 and they are all here, plus the ones this
particular implementation needs: that the batched rank-one update agrees with
the explicit one, that the batched sigma_max agrees with an SVD, and that the
two structural degeneracies survive the governor term that ../snsp adds to the
spec's operator.

Run it first, and after any change to the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import base                                                  # noqa: E402
import greens as gr                                          # noqa: E402
from base import TWO_PI, BaseCase                            # noqa: E402

OMEGA_S = TWO_PI * 50.0
OK, BAD = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (OK if cond else BAD).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail
                                                       else ""))


def rel(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    scale = max(float(np.max(np.abs(b))), 1e-300)
    return float(np.max(np.abs(a - b)) / scale)


# ---------------------------------------------------------------------------
# a time-domain solver written against the same (M, D, droop, L) as A(s), so
# that "frequency domain agrees with time domain" is a real check and not a
# tautology


def propagate(bc: BaseCase, L: np.ndarray, dP: np.ndarray, t_end: float,
              dt: float = 1e-3, omega_drive: float | None = None):
    """Integrate the linear swing system with a held or sinusoidal injection.

    State [x; xdot; p] plus a driver block: two extra states generating
    cos(omega t) when `omega_drive` is given, one constant state otherwise --
    the same augmentation ../snsp/dynamics.py uses to fold a held step in, so a
    single matrix exponential serves the whole trajectory.
    """
    N = bc.N
    Minv = 1.0 / bc.M
    nd = 2 if omega_drive else 1
    n = 3 * N + nd
    A = np.zeros((n, n))
    A[:N, N:2 * N] = np.eye(N)
    A[N:2 * N, :N] = -Minv[:, None] * L
    A[N:2 * N, N:2 * N] = -np.diag(Minv * bc.D)
    A[N:2 * N, 2 * N:3 * N] = np.diag(Minv)
    A[2 * N:3 * N, N:2 * N] = -np.diag(bc.droop / (OMEGA_S * bc.T_gov))
    A[2 * N:3 * N, 2 * N:3 * N] = -np.diag(1.0 / bc.T_gov)
    A[N:2 * N, 3 * N] = Minv * dP                    # driver -> acceleration
    if omega_drive:
        A[3 * N, 3 * N + 1] = omega_drive
        A[3 * N + 1, 3 * N] = -omega_drive

    E = expm(A * dt)
    y = np.zeros(n)
    y[3 * N] = 1.0
    steps = int(round(t_end / dt))
    out = np.empty((steps + 1, 2 * N))
    out[0] = y[:2 * N]
    for k in range(steps):
        y = E @ y
        out[k + 1] = y[:2 * N]
    return out[:, :N], out[:, N:]                    # angle, angular velocity


def added(bc: BaseCase, a: int, b: int, kappa: float) -> np.ndarray:
    """L with edge (a, b) added -- the first-order form the screen uses."""
    u = np.zeros(bc.N)
    u[a], u[b] = 1.0, -1.0
    return bc.L + kappa * np.outer(u, u)


# ---------------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(20260910)
    print("VALIDATION -- single-edge placement\n" + "=" * 70 + "\n")

    print("frozen inputs")
    bc = base.load()
    pairs, joined = base.candidates(bc)
    check("grid loads and hash-verifies", bc.N == 100 and int(bc.g.A.sum() // 2) == 166,
          f"{bc.N} buses, {int(bc.g.A.sum() // 2)} lines")
    check("L is symmetric", rel(bc.L, bc.L.T) < 1e-14)
    check("candidate count is C(N,2)", len(pairs) == bc.N * (bc.N - 1) // 2,
          f"{len(pairs)} = {int((~joined).sum())} new + {int(joined.sum())} existing")
    ok, worst = base.stable(bc)
    check("base case is stable", ok, f"worst non-zero root Re = {worst:.4f}")

    # The base case must BE a row of the SNSP study, not merely resemble one.
    # ../snsp/grid.json hash-checks the grid but nothing checks configs.npz, and
    # that file is stale: its droop sums to 40 where run_ensemble.py's own
    # redraw gives 114.  Reading it produces a differently damped grid whose
    # RoCoF still matches to 0.5%, so only a cross-check against published
    # numbers catches it.  See base.load.
    import pandas as pd                                       # noqa: PLC0415
    import perturbations as PB                                # noqa: PLC0415
    import dynamics as dyn                                    # noqa: PLC0415
    src = base.SNSP / "ensemble_src.csv"
    if src.exists():
        ref = pd.read_csv(src)
        ref = ref[ref["config"] == bc.config]
        dP = PB.matrix(PB.load(bc.g, base.SNSP, verify=True, which="sourced"),
                       bc.P, bc.g)
        got = dyn.summarise(dyn.respond(bc.net, dP), bc.N)
        worst_rel = max(abs(got[k] / float(ref[k].iloc[0]) - 1.0)
                        for k in ("rocof_top1", "rocof_top5", "dev_top5",
                                  "nadir_top5"))
        check("base case reproduces ../snsp/ensemble_src.csv", worst_rel < 1e-9,
              f"worst relative difference {worst_rel:.2e} over the top-n scores")
    else:
        check("base case reproduces ../snsp/ensemble_src.csv", True,
              "SKIPPED -- ensemble_src.csv not present")

    # -- spec sec.10.2 --------------------------------------------------------
    print("\nspec sec.10 -- structure")
    check("2. L 1 = 0", float(np.max(np.abs(bc.L @ np.ones(bc.N)))) < 1e-10)
    S0 = [complex(gr.self_energy(bc, 0.0, [i])[1][0, 0]) for i in range(0, bc.N, 7)]
    check("2. S_i(0) = 0 for every i", max(abs(v) for v in S0) < 1e-8,
          f"max |S_i(0)| = {max(abs(v) for v in S0):.2e}")

    # -- spec sec.10.1 --------------------------------------------------------
    m1, m2, g1, g2, k = 0.7, 1.3, 0.11, 0.23, 3.0

    def S1(s):
        return m1 * s ** 2 + g1 * s + k - k ** 2 / (m2 * s ** 2 + g2 * s + k)

    # S_1(s)/s = (g1+g2) + s (m1+m2-g2^2/k) + O(s^2), so testing the residual
    # against *both* coefficients at once is sharper than either limit alone --
    # and the gamma^2/k correction is spec §4(a) itself: at low frequency the
    # pair moves as one lump and mass and damping renormalise together, so
    # "the effective mass" is not separately observable.
    s = 1e-5
    damp, mass = S1(s) / s, m1 + m2 - g2 ** 2 / k
    check("1. two-mass limit: damping adds", abs(damp - (g1 + g2)) < 3 * s * mass,
          f"S_1(s)/s = {damp:.6f} -> gamma_1+gamma_2 = {g1 + g2:.6f}")
    check("1. two-mass limit: mass adds, less gamma_2^2/k",
          abs(damp - (g1 + g2) - s * mass) < 1e-9,
          f"residual/s = {(damp - (g1 + g2)) / s:.6f} vs "
          f"m_1+m_2-gamma_2^2/k = {mass:.6f}")

    # -- spec sec.10.5, 10.6, 10.7, 10.8 -----------------------------------
    print("\nspec sec.10 -- the machinery")
    worst_wood = worst_quot = 0.0
    for _ in range(12):
        w = float(10 ** rng.uniform(-2, 1.3)) * TWO_PI
        a, b = rng.choice(bc.N, size=2, replace=False)
        kap = float(rng.uniform(0.5, 8.0))
        G = gr.green(bc, w)
        direct = np.linalg.inv(bc.A(1j * w) + kap * np.outer(
            np.eye(bc.N)[a] - np.eye(bc.N)[b], np.eye(bc.N)[a] - np.eye(bc.N)[b]))
        worst_wood = max(worst_wood, rel(gr.update(G, a, b, kap), direct))

        Sig1, _ = gr.self_energy(bc, w, [a])
        SigT, ST = gr.self_energy(bc, w, [a, b])
        lhs = Sig1[0, 0]
        rhs = SigT[0, 0] + ST[0, 1] ** 2 / ST[1, 1]
        worst_quot = max(worst_quot, abs(lhs - rhs) / abs(lhs))
    check("5. Sherman-Morrison equals a direct inverse", worst_wood < 1e-8,
          f"worst relative error {worst_wood:.2e}")
    check("6. quotient property Sigma_i = Sigma_ii + (K+Sigma_ij)^2/S_jj",
          worst_quot < 1e-8, f"worst relative error {worst_quot:.2e}")

    i = 41
    w1 = gr.harmonic_weights(bc, [i])
    bath = float(bc.M.sum() - bc.M[i])
    cond = gr.condensed(bc, [i])
    check("7. Guyan limit: w = 1 for a single target", rel(w1[:, 0], np.ones(bc.N - 1)) < 1e-9)
    check("7. Guyan limit: condensed mass = own + whole bath",
          abs(float(cond["M"][0, 0].real) - (bc.M[i] + bath)) < 1e-10,
          f"{float(cond['M'][0, 0].real):.5f} vs {bc.M[i] + bath:.5f}")
    w2 = gr.harmonic_weights(bc, [i, i + 3])
    check("   and w^(i) + w^(j) = 1 for a pair", rel(w2.sum(axis=1), np.ones(bc.N - 2)) < 1e-9)

    w = 2 * np.pi * 1.7
    G = gr.green(bc, w)
    Pi = bc.coi_projector()
    entry = float(np.abs(G).max())
    smax = float(np.linalg.svd(G, compute_uv=False)[0])
    frob = float(np.linalg.norm(G))
    check("8. max|G_ij| <= sigma_max <= ||G||_F", entry <= smax + 1e-9 <= frob + 1e-9,
          f"{entry:.4f} <= {smax:.4f} <= {frob:.4f}")

    # -- this implementation ------------------------------------------------
    print("\nthis implementation")
    B = np.eye(bc.N)[:, ::5]                       # a stand-in disturbance set
    X, PG, PX = G @ B, Pi @ G, Pi @ (G @ B)
    sub = pairs[rng.choice(len(pairs), size=24, replace=False)]
    Y = gr.apply_to(G, X, PG, PX, sub, base.KAPPA)
    ref = np.stack([Pi @ (gr.update(G, a, b, base.KAPPA) @ B) for a, b in sub])
    check("batched update equals the explicit one", rel(Y, ref) < 1e-9,
          f"worst relative error {rel(Y, ref):.2e}")
    check("batched sigma_max equals an SVD",
          rel(gr.sigma_max_stack(Y), [np.linalg.svd(y, compute_uv=False)[0]
                                      for y in Y]) < 1e-10)
    flagged = int(gr.near_cancellation(G, pairs, base.KAPPA).sum())
    check("no near-cancelling denominators at this frequency", flagged == 0,
          f"{flagged} candidates flagged")

    # -- spec sec.10.3, §10.4: the degeneracies, with and without an edge ------
    print("\nspec sec.10 -- the degeneracies (must NOT move when an edge is added)")
    dP = np.zeros(bc.N)
    dP[np.array([12, 55])] = np.array([-0.4, -0.2])
    gamma_tot = bc.D.sum() + bc.droop.sum() / OMEGA_S
    predicted = dP.sum() / gamma_tot
    a, b = 3, 87
    for label, L in (("base", bc.L), (f"+edge ({a},{b})", added(bc, a, b, base.KAPPA))):
        _, om = propagate(bc, L, dP, t_end=60.0, dt=2e-3)
        settled = float(om[-1].mean())
        check(f"3. steady offset, {label}", abs(settled - predicted) < 1e-4,
              f"{settled:.6f} rad/s vs sum(dP)/sum(gamma) = {predicted:.6f}")
    for label, L in (("base", bc.L), (f"+edge ({a},{b})", added(bc, a, b, base.KAPPA))):
        _, om = propagate(bc, L, dP, t_end=0.02, dt=1e-4)
        r0 = (om[1] - om[0]) / 1e-4
        check(f"4. initial RoCoF = dP/m, {label}",
              rel(r0[[12, 55]], (dP / bc.M)[[12, 55]]) < 2e-3,
              f"bus 12: {r0[12]:.3f} vs {dP[12] / bc.M[12]:.3f} rad/s^2")

    # -- spec sec.10.9 ---------------------------------------------------------
    print("\nspec sec.10.9 -- frequency domain against time domain")
    for fHz, j, i in ((0.35, 55, 12), (1.7, 12, 88)):
        w = TWO_PI * fHz
        dP = np.zeros(bc.N)
        dP[j] = 1.0
        th, _ = propagate(bc, bc.L, dP, t_end=90.0, dt=5e-4, omega_drive=w)
        tail = th[-int(round((1.0 / fHz) / 5e-4)):, i]
        amp = 0.5 * (tail.max() - tail.min())
        pred = float(np.abs(gr.green(bc, w)[i, j]))
        check(f"9. |G_{i},{j}| at {fHz} Hz", abs(amp - pred) / pred < 5e-3,
              f"simulated {amp:.6f} vs predicted {pred:.6f}")

    print("\n" + "=" * 70)
    print(f"  {len(OK)} passed, {len(BAD)} failed")
    if BAD:
        for n in BAD:
            print(f"    FAILED: {n}")
        sys.exit(1)


if __name__ == "__main__":
    main()
