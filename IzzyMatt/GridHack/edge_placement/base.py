"""The base case: one frozen operating point of the frozen SNSP grid.

This investigation adds **one edge** to a network and asks whether the network
got better.  For that comparison to mean anything, everything except the edge
has to be nailed down first -- and that is all this module does.  It adds no
edges and it scores nothing.

WHAT IS HELD FIXED, AND WHY EACH ONE MATTERS

  the grid          The 10x10 caricature of the all-island system frozen by
                    ../snsp/grid.py, loaded through `G.load(verify=True)` so a
                    quietly edited network can never contaminate a comparison.
                    166 lines, so C(100,2) - 166 = 4784 pairs are *not* joined
                    and could be.
  the dispatch      P is taken from the frozen ensemble (../snsp/configs.npz)
                    and never moves.  This is the fixed-P design the
                    local-effects study turned on: the steady state of the swing
                    equation depends on P and K but never on M, so holding P
                    fixed and changing only the topology leaves the added edge
                    as the single variable.
  inertia, damping  M, D, droop and T_gov come from the same frozen Config, so
                    an added edge cannot smuggle in a commitment change.

WHAT MOVES WHEN AN EDGE IS ADDED

  theta*            The operating angles shift, because the power has another
                    path.  The linear screen neglects this -- the rank-one
                    update is first-order correct -- and the shortlist is
                    re-solved nonlinearly with the edge present (spec §9).
  L                 The synchronising Laplacian, L_ij = K_ij cos(theta*_i -
                    theta*_j).  Note the cosine: the correct entries are the
                    synchronising coefficients at the operating point, not the
                    bare K_ij.

THE OPERATOR

The SNSP model carries a governor state at every committed machine, so the
plain second-order form of the spec (M s^2 + Gamma s + L) is not quite the
operator here.  Eliminating p from

    M x'' + D x' + L x = dP + p ,      T p' = -p - droop x' / omega_s

gives, in Laplace,

    A(s) = M s^2 + [ D + droop / (omega_s (1 + T s)) ] s + L ,

i.e. exactly the spec's operator with a *frequency-dependent* diagonal damping.
Nothing downstream cares: A(s) is still complex symmetric and still
diagonal-plus-Laplacian, so the Sherman-Morrison screen and the Schur
self-energies go through unchanged.  Both of the spec's structural degeneracies
survive the change, and both are asserted in `validate.py`:

  - the steady-state offset (§5a) becomes sum(dP) / sum(D + droop/omega_s),
    still independent of topology, so still not available to optimise;
  - the initial RoCoF (§5b) is still dP_j / m_j, since the governor term is
    O(1/s) at large s.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SNSP = HERE.parent / "snsp"
LOCAL = HERE.parent / "local_effects"
for _p in (SNSP, LOCAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import configs as C                                          # noqa: E402
import dynamics as dyn                                       # noqa: E402
import grid as G                                             # noqa: E402
import swing as sw                                           # noqa: E402

TWO_PI = 2.0 * np.pi

#: The operating point the headline runs on.  `--config` picks any of the 240
#: and `--snsp` picks the first draw at a level.  The reference is deliberately
#: mid-sweep: at 30% SNSP so many machines are on that nothing is fragile, and
#: at 100% there is no governor left and every candidate looks alike.
REFERENCE_SNSP = 0.70

#: Strength of the edge we are allowed to add, p.u.  The frozen grid rates its
#: circuits at 8.0 (backbone), 4.0 (main) and 2.5 (spur), so the default buys
#: one new single circuit and the sweep spans the three classes.
KAPPA = 2.5
KAPPA_SWEEP = (2.5, 4.0, 8.0)

#: Measurement, not raw state.  ../snsp evaluates its cost on a frequency passed
#: through a 100 ms first-order lag -- the window a RoCoF relay is specified
#: over -- because an injection step landing on a near-massless converter bus
#: produces an instantaneous df/dt of order 250 Hz/s that no relay or PLL ever
#: reports.  The same lag is applied here.
T_MEAS = 0.1

#: RoCoF is measured over a sliding window, never as an instantaneous
#: derivative.  The window has gain 2|sin(omega T / 2)| / T, NOT omega: the
#: instantaneous form weights the far tail of the band and reorders which
#: reinforcement looks best.  This is the spec's §5(b) decision, taken the
#: windowed way rather than the band-limited way, and it is the convention the
#: rest of this folder is written to.
T_ROCOF = 0.5

#: The band the *solver* sweeps, Hz.  Wide on purpose: `sweep.py` stores the
#: whole sigma_max(omega) curve, so the band actually scored is a reprocessing
#: choice made afterwards by `physical_band` and swept by `robustness.py`.
#:
#: The band that is scored is the consequential choice, and it is the one this
#: study got wrong on its first pass.  The base case has 200 modes spanning
#: 0.042 to 35.6 Hz, but only about 1% sit below 1.8 Hz: 93 of the 100 buses
#: carry M ~ 6e-4 p.u.s, the small regularising mass ../snsp puts on an
#: algebraic bus, and the crowd above ~2.5 Hz is those buses ringing against
#: their own circuits at sqrt(K/M).  A real converter bus has neither mass nor
#: such a mode.  A time-domain study barely notices them; an H-infinity peak
#: seeks out the worst frequency in whatever band it is handed, and lands
#: squarely on one.  See `artefact_floor`.
BAND_HZ = (0.01, 20.0)


@dataclass
class BaseCase:
    """A frozen operating point, plus everything A(s) needs."""

    g: G.ToyGrid
    config: int
    snsp: float
    P: np.ndarray
    H: np.ndarray
    droop: np.ndarray
    T_gov: np.ndarray
    theta0: np.ndarray = field(init=False)
    M: np.ndarray = field(init=False)
    D: np.ndarray = field(init=False)
    L: np.ndarray = field(init=False)
    net: sw.Network = field(init=False)

    def __post_init__(self) -> None:
        self.net = dyn.build_network(self.g, self.P, self.H, self.droop,
                                     self.T_gov)
        self.theta0 = self.net.theta0
        self.M = self.net.M
        self.D = self.net.D
        self.L = sw.laplacian(self.net)

    # -- the operator ------------------------------------------------------
    def gamma(self, s):
        """Diagonal damping at complex frequency s, governor included."""
        return self.D + self.droop / (sw.OMEGA_S * (1.0 + self.T_gov * s))

    def A(self, s) -> np.ndarray:
        """A(s) = M s^2 + Gamma(s) s + L -- complex symmetric, not Hermitian."""
        return np.diag(self.M * s ** 2 + self.gamma(s) * s) + self.L

    @property
    def N(self) -> int:
        return int(self.g.N)

    @property
    def stored_energy(self) -> float:
        return float(self.g.stored_energy(self.H))

    @property
    def steady_offset(self) -> float:
        """Hz per p.u. of imbalance, once everything has settled.

        Topology-independent by construction (spec §5a), and reported here so
        that no one is tempted to read a change in it off a candidate.
        """
        return 1.0 / (self.D.sum() + self.droop.sum() / sw.OMEGA_S) / TWO_PI

    # -- projection and measurement ----------------------------------------
    def coi_projector(self) -> np.ndarray:
        """Pi = I - 1 1^T M / (1^T M 1): deviation from the centre of inertia.

        The uniform mode is untouchable by any added edge (§5a) and otherwise
        dominates sigma_max at small omega, adding a near-constant to every
        candidate's score and compressing exactly the differences we are trying
        to resolve.
        """
        m = self.M
        return np.eye(self.N) - np.outer(np.ones(self.N), m) / m.sum()

    @staticmethod
    def weight_freq(omega):
        """|s / (2 pi (1 + s T_MEAS))|: angle response to measured frequency."""
        s = 1j * np.asarray(omega, float)
        return np.abs(s / (TWO_PI * (1.0 + s * T_MEAS)))

    @staticmethod
    def weight_rocof(omega):
        """The same, then differenced over T_ROCOF: extra gain 2|sin(wT/2)|/T.

        Both weights are scalars times the identity, so they factor straight out
        of the singular value: sigma_max(Omega Pi G B) = |Omega| sigma_max(Pi G
        B).  One sigma_max sweep therefore serves both objectives, and what
        separates them is entirely *where in the band* a candidate's peak sits.
        """
        w = np.asarray(omega, float)
        return (BaseCase.weight_freq(w)
                * np.abs(2.0 * np.sin(w * T_ROCOF / 2.0)) / T_ROCOF)


def load(config: int | None = None, snsp: float | None = None) -> BaseCase:
    """The frozen grid and one configuration of the SNSP ensemble.

    The configuration is obtained by **re-drawing** the ensemble with
    `C.ensemble(g)`, which is what ../snsp/run_ensemble.py itself does, and not
    by reading the frozen `configs.npz`.  The two disagree: the saved npz
    carries a droop vector summing to 40 where the redraw gives 114, with P, H
    and T_gov identical.  Every published SNSP result comes from the redraw, so
    reading the npz silently produces a *differently damped* grid -- one whose
    RoCoF matches to 0.5% (inertia is untouched) but whose frequency excursions
    are 21% larger and whose integral of f^2 is 2.5x larger.

    Droop enters this study's operator directly, through
    Gamma(s) = D + droop/(omega_s(1+Ts)), so it sets the damping of the very
    resonance the H-infinity peak is taken at.  Getting it from the stale file
    is not a bookkeeping detail here.

    `validate.py` asserts that the base case reproduces the corresponding row of
    `../snsp/ensemble_src.csv` exactly, which is the check that would have caught
    it: the grid is hash-verified on load but `configs.npz` is not.
    """
    g = G.load(SNSP, verify=True)
    cfgs = C.ensemble(g)
    if config is None:
        want = REFERENCE_SNSP if snsp is None else snsp
        hit = [i for i, c in enumerate(cfgs) if np.isclose(c.snsp_target, want)]
        if not hit:
            raise ValueError(f"no ensemble stratum at snsp={want}")
        config = int(hit[0])
    c = cfgs[int(config)]
    return BaseCase(g=g, config=int(config),
                    snsp=float(G.snsp_of(g, c.P)), P=c.P, H=c.H,
                    droop=c.droop, T_gov=c.T_gov)


# ---------------------------------------------------------------------------
# the modes, and the band they set


def modes(bc: BaseCase) -> np.ndarray:
    """Damped natural frequencies of the base case, Hz, slowest first.

    Taken from the full 3N x 3N state matrix rather than the quadratic pencil,
    so the governor's own slow mode is in there too.  The zero mode (a uniform
    phase shift) and the purely real roots are dropped.
    """
    lam = np.linalg.eigvals(sw.state_matrix(bc.net))
    f = np.abs(lam.imag) / TWO_PI
    return np.sort(f[f > 1e-9])


def stable(bc: BaseCase, tol: float = 1e-9) -> tuple[bool, float]:
    """Are all roots of det A(s) = 0 in the open left half-plane, bar the zero?"""
    lam = np.linalg.eigvals(sw.state_matrix(bc.net))
    worst = float(np.sort(lam.real)[-2])       # the last one is the zero mode
    return worst < -tol, worst


def artefact_floor(bc: BaseCase, scale: float = 0.5,
                   tol: float = 0.10) -> tuple[float, float]:
    """(highest physical mode, lowest regularisation mode), Hz.

    ../snsp gives an algebraic bus a small regularising mass, H = 0.1-0.2 s
    against 4 s for a machine, so that the swing equations stay an ODE.  That
    invents modes: 93 of the 100 buses here carry M ~ 6e-4 p.u.s, and they ring
    against their own circuits at sqrt(K/M).  A time-domain study integrating
    over ten seconds barely notices -- there is little energy in them.  An
    H-infinity peak notices completely, because it seeks out the single worst
    frequency in the band, and on this grid that frequency is one of these.

    The test is direct: a regularisation mode scales as 1/sqrt(M) and runs away
    when the regularising mass is reduced, while a physical mode stays put and
    converges.  Halving H on the light buses moves 2.542 Hz to 3.365 Hz (+32%)
    but 1.825 Hz only to 1.881 Hz (+3%).  So a mode is called physical if it has
    a partner within `tol` after the halving, and the lowest mode without one is
    where the model stops describing the grid.
    """
    f0 = modes(bc)
    H = bc.H.copy()
    light = H < 1.0
    H[light] *= scale
    net = dyn.build_network(bc.g, bc.P, H, bc.droop, bc.T_gov)
    lam = np.linalg.eigvals(sw.state_matrix(net))
    f1 = np.sort(np.abs(lam.imag) / TWO_PI)
    f1 = f1[f1 > 1e-9]

    moved = np.array([np.min(np.abs(f1 - f)) / f > tol for f in f0])
    if not moved.any():
        return float(f0[-1]), float("inf")
    first = int(np.argmax(moved))
    physical = f0[:first]
    return float(physical[-1] if len(physical) else f0[0]), float(f0[first])


def physical_band(bc: BaseCase) -> tuple[float, float]:
    """The band over which this model is describing the grid rather than itself.

    Capped at the geometric mean of the highest physical mode and the lowest
    regularisation mode, which keeps the former's resonance peak inside the band
    and leaves the latter's outside.
    """
    phys, art = artefact_floor(bc)
    return BAND_HZ[0], float(np.sqrt(phys * art)) if np.isfinite(art) else BAND_HZ[1]


def band(bc: BaseCase, lo: float = BAND_HZ[0], hi: float = BAND_HZ[1],
         n: int = 160) -> np.ndarray:
    """A log grid of omega over the band, in rad/s.

    A fixed grid systematically *under*-estimates an H-infinity peak, so this is
    a starting grid only: `rank.py` refines adaptively around every local
    maximum it finds.
    """
    return TWO_PI * np.geomspace(lo, hi, n)


# ---------------------------------------------------------------------------
# the candidates


def candidates(bc: BaseCase, include_existing: bool = True):
    """Every pair (a, b) with a < b, and whether that pair is already joined.

    New pairs are the decision variable proper.  Existing pairs are kept as a
    *comparison class*: reinforcing a circuit that is already there is the other
    thing an operator can buy with the same money, and the rank-one update
    handles it identically -- it is the same u = e_a - e_b.
    """
    a, b = np.triu_indices(bc.N, k=1)
    joined = bc.g.A[a, b] > 0
    if include_existing:
        return np.stack([a, b], axis=1), joined
    keep = ~joined
    return np.stack([a[keep], b[keep]], axis=1), joined[keep]


def with_edge(bc: BaseCase, a: int, b: int, kappa: float = KAPPA) -> BaseCase:
    """The same operating point on a grid that has the edge (a, b) in it.

    This is the honest version of a candidate, and it is not what the screen
    evaluates.  The screen adds kappa u u^T to L and stops, which is exact for
    the *operator* but holds theta* fixed; in truth the power now has another
    path, the angles shift, and L is rebuilt from the new cosines.  So this
    re-solves the load flow with the edge present and lets everything follow.

    P, H, droop and T_gov do not move -- the dispatch is still the frozen one,
    which is the whole controlled-experiment design.  What changes is theta*,
    and through it every synchronising coefficient in the network, not just the
    two ends of the new line.
    """
    A = bc.g.A.copy()
    K = bc.g.Kij.copy()
    A[a, b] = A[b, a] = 1.0
    K[a, b] = K[b, a] = K[a, b] + kappa
    g = dataclasses.replace(bc.g, A=A, Kij=K,
                            line_class={**bc.g.line_class, (min(a, b), max(a, b)): "added"})
    return BaseCase(g=g, config=bc.config, snsp=bc.snsp, P=bc.P, H=bc.H,
                    droop=bc.droop, T_gov=bc.T_gov)


def edge_length(bc: BaseCase, pairs: np.ndarray) -> np.ndarray:
    """Lattice distance between the ends -- a proxy for what a circuit costs.

    Distances are lattice squares, not kilometres, exactly as in ../snsp.  The
    point of carrying it is that a candidate joining opposite corners of the
    grid is not a comparable purchase to one joining neighbours, and the Pareto
    front should be allowed to say so.
    """
    xy = np.asarray([bc.g.coords[i] for i in range(bc.N)], float)
    d = xy[pairs[:, 0]] - xy[pairs[:, 1]]
    return np.hypot(d[:, 0], d[:, 1])


# ---------------------------------------------------------------------------
# freeze


def digest(bc: BaseCase) -> str:
    h = hashlib.sha256()
    for a in (bc.P, bc.H, bc.droop, bc.T_gov, bc.L):
        h.update(np.ascontiguousarray(a, dtype=float).tobytes())
    return h.hexdigest()[:16]


def save(bc: BaseCase, out: Path = HERE, tag: str = "") -> dict:
    suffix = f"_{tag}" if tag else ""
    np.savez_compressed(out / f"base{suffix}.npz", P=bc.P, H=bc.H,
                        droop=bc.droop, T_gov=bc.T_gov, theta0=bc.theta0,
                        M=bc.M, D=bc.D, L=bc.L)
    grid_json = SNSP / "grid.json"
    meta = {"config": bc.config, "snsp": bc.snsp, "N": bc.N,
            "lines": int(bc.g.A.sum() // 2), "kappa": KAPPA,
            "t_meas": T_MEAS, "t_rocof": T_ROCOF,
            "stored_energy_pu_s": bc.stored_energy,
            "steady_offset_hz_per_pu": bc.steady_offset,
            "digest": digest(bc),
            "grid_json": json.loads(grid_json.read_text()) if
            grid_json.exists() else None}
    (out / f"base{suffix}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def report(bc: BaseCase) -> str:
    f = modes(bc)
    ok, worst = stable(bc)
    pairs, joined = candidates(bc)
    w = band(bc)
    gap = bc.net.max_angle_gap()
    return "\n".join([
        "BASE CASE FOR SINGLE-EDGE PLACEMENT", "=" * 70, "",
        f"  grid                  {bc.N} buses, {int(bc.g.A.sum() // 2)} lines",
        f"  configuration         #{bc.config}    SNSP {bc.snsp:.1%}",
        f"  machines synchronised {int((bc.H > 1.0).sum())}",
        f"  stored energy         {bc.stored_energy:8.2f} p.u.s",
        f"  total damping         {bc.D.sum():8.3f} p.u./(rad/s)",
        f"  droop, total          {bc.droop.sum():8.3f} p.u.",
        f"  steady offset         {bc.steady_offset:8.4f} Hz per p.u."
        "   (topology-independent)",
        f"  max angle gap         {gap:8.4f} rad",
        f"  min cos(angle gap)    {np.cos(gap):8.4f}", "",
        f"  stable                {'yes' if ok else 'NO'}"
        f"   worst non-zero root Re = {worst:.4f}",
        f"  modes                 {len(f)}, from {f[0]:.4f} to {f[-1]:.3f} Hz",
        f"  band swept            {w[0] / TWO_PI:.5f} to {w[-1] / TWO_PI:.3f} Hz"
        f"  ({len(w)} log-spaced points)", "",
        f"  candidate pairs       {len(pairs)}"
        f"   ({int((~joined).sum())} new, {int(joined.sum())} reinforcements)",
        f"  kappa                 {KAPPA} p.u."
        f"   (spur {G.K_BY_CLASS['spur']}, main {G.K_BY_CLASS['main']},"
        f" backbone {G.K_BY_CLASS['backbone']})",
        f"  digest                {digest(bc)}", ""])


def main() -> None:
    ap = argparse.ArgumentParser(description="freeze the base case")
    ap.add_argument("--config", type=int, default=None)
    ap.add_argument("--snsp", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()

    bc = load(config=args.config, snsp=args.snsp)
    text = report(bc)
    print(text)
    if args.freeze:
        meta = save(bc, tag=args.tag)
        suffix = f"_{args.tag}" if args.tag else ""
        (HERE / f"base{suffix}_report.txt").write_text(text)
        print(f"  frozen -> base{suffix}.npz / .json    digest {meta['digest']}")


if __name__ == "__main__":
    main()
