"""The dynamical Green's function, the rank-one update, and the self-energies.

Three things live here, in the order the spec uses them.

1. G(s) = A(s)^-1, the transfer function from an injection at j to the phase at
   i.  A(s) is complex **symmetric**, not Hermitian, so nothing here reaches for
   a Cholesky factor or assumes real eigenvalues.

2. The Sherman-Morrison update.  Adding an edge (a, b) of strength kappa is a
   rank-one update of L and hence of A:

       A -> A + kappa u u^T ,     u = e_a - e_b ,

       G_new = G - kappa (G u)(u^T G) / (1 + kappa z_ab) ,
       z_ab  = u^T G u = G_aa - G_ab - G_ba + G_bb ,

   z_ab being a complex, frequency-dependent effective impedance between a and b
   -- the dynamical generalisation of effective resistance.  Because u is a
   difference of two basis vectors, every candidate reduces to two columns of G
   and four of its entries, so **one factorisation per frequency serves all 4950
   pairs**.  That is the whole computational argument of the spec, and on this
   grid it is strong enough that the shortlist stage can be skipped: see
   `apply_to` and `sigma_max_stack` below.

3. The block Schur complement and its self-energy, which the spec is clear about
   -- machinery and post-hoc diagnosis, never the objective.  §4 of the spec
   lists four separate reasons why maximising a min effective mass optimises the
   wrong thing; the one that bites hardest here is that the effective coupling
   sits in the *numerator* of G_ij, so a rule of "make the effective mass large"
   would actively select the worst edges off-diagonal.  Nothing in this module
   is used to rank candidates.  `explain()` is called on the winner, after the
   fact.

A NOTE ON WHAT IS CHEAP HERE

N = 100, so a candidate costs almost nothing and there is no need to screen.
Per frequency: one O(N^3) inverse, then for every one of the 4950 pairs a rank-
one update of the N x n_B matrix Pi G B (2 000 flops) and the largest singular
value of the result, taken as sqrt of the top eigenvalue of its n_B x n_B Gram
matrix.  With the 20 frozen ../snsp events as B that is a 20 x 20 Hermitian
eigenproblem per candidate.  The spec's Frobenius screen is still implemented
(`frobenius_stack`) because it is nearly free and because §7.3's argument -- that
it is a screen and not a criterion -- is worth being able to *show* on this grid
rather than merely cite.
"""

from __future__ import annotations

import numpy as np

from base import BaseCase


# ---------------------------------------------------------------------------
# the Green's function


def green(bc: BaseCase, omega: float) -> np.ndarray:
    """G(i omega) = A(i omega)^-1, dense and complex symmetric."""
    return np.linalg.inv(bc.A(1j * omega))


def impedance(G: np.ndarray, a, b):
    """z_ab = G_aa - G_ab - G_ba + G_bb, for scalar or array index sets."""
    d = np.diag(G)
    return d[a] + d[b] - G[a, b] - G[b, a]


def update(G: np.ndarray, a: int, b: int, kappa: float) -> np.ndarray:
    """G_new for one candidate, formed explicitly.  For tests and diagnosis."""
    u = np.zeros(G.shape[0])
    u[a], u[b] = 1.0, -1.0
    Gu = G @ u
    return G - kappa * np.outer(Gu, u @ G) / (1.0 + kappa * impedance(G, a, b))


def apply_to(G: np.ndarray, X: np.ndarray, PG: np.ndarray, PX: np.ndarray,
             pairs: np.ndarray, kappa: float) -> np.ndarray:
    """Pi G_new B for every candidate at once.

    `X = G B` and `PX = Pi G B` and `PG = Pi G` are precomputed once per
    frequency.  Then, per candidate,

        Pi G_new B = Pi G B - c (Pi G u)(u^T G B) ,   c = kappa/(1 + kappa z),

    where `Pi G u` is a difference of two columns of `Pi G` and `u^T G B` is a
    difference of two *rows* of `G B` -- the latter because G is symmetric, so
    u^T G = (G u)^T.

    Returns an array of shape (n_pairs, N, n_B).
    """
    a, b = pairs[:, 0], pairs[:, 1]
    z = impedance(G, a, b)
    denom = 1.0 + kappa * z
    c = kappa / denom
    U = (PG[:, a] - PG[:, b]).T                     # (n_pairs, N)
    V = X[a, :] - X[b, :]                           # (n_pairs, n_B)
    return PX[None, :, :] - c[:, None, None] * U[:, :, None] * V[:, None, :]


def near_cancellation(G: np.ndarray, pairs: np.ndarray, kappa: float,
                      tol: float = 1e-6) -> np.ndarray:
    """Candidates where 1 + kappa z_ab is near zero: flag, never divide blindly.

    The spec's warning at §6.  A vanishing denominator means the added edge
    almost exactly cancels a mode of the base network; the update is then
    numerically meaningless and the candidate needs solving directly.
    """
    a, b = pairs[:, 0], pairs[:, 1]
    return np.abs(1.0 + kappa * impedance(G, a, b)) < tol


# ---------------------------------------------------------------------------
# the norms


def sigma_max_stack(Y: np.ndarray) -> np.ndarray:
    """Largest singular value of each matrix in a stack (n, N, m).

    Via the m x m Gram matrix Y^H Y, which is Hermitian positive semidefinite
    and small when B is a disturbance set rather than the identity.  The
    complex symmetry of A does not help here: sigma_max needs the SVD (or this
    Hermitian eigenproblem), never the eigenvalues of G.
    """
    Gram = np.conjugate(np.swapaxes(Y, -1, -2)) @ Y
    lam = np.linalg.eigvalsh(Gram)[..., -1]
    return np.sqrt(np.clip(lam, 0.0, None))


def frobenius_stack(Y: np.ndarray) -> np.ndarray:
    """||.||_F of each matrix in a stack -- the spec's screen, not its criterion."""
    return np.sqrt(np.einsum("nij,nij->n", Y, np.conjugate(Y)).real)


def row_sum_max(Y: np.ndarray) -> np.ndarray:
    """max_i sum_j |Y_ij|: the induced l-infinity norm.

    The exact worst case when each disturbance is individually bounded -- which
    is arguably the more physical box for a grid than a bounded-energy pattern.
    Non-smooth and more conservative, and free once Y is in hand.
    """
    return np.abs(Y).sum(axis=-1).max(axis=-1)


# ---------------------------------------------------------------------------
# Schur complements and self-energies -- diagnosis only (spec §3, §4, stage 4)


def self_energy(bc: BaseCase, omega: float, T) -> tuple[np.ndarray, np.ndarray]:
    """(Sigma_T, S_T) at i omega for a target set T.

    Sigma_T = A_Tr A_rr^-1 A_rT and S_T = A_TT - Sigma_T, so that
    G_TT = S_T^-1 exactly.  For |T| = 2 the off-diagonal of S_T carries the
    renormalised coupling K_eff_ij = K_ij + Sigma_ij, frequency dependent and
    complex even though the bare coupling is real and static.
    """
    T = np.atleast_1d(np.asarray(T, int))
    r = np.setdiff1d(np.arange(bc.N), T)
    A = bc.A(1j * omega)
    Arr = A[np.ix_(r, r)]
    Sigma = A[np.ix_(T, r)] @ np.linalg.solve(Arr, A[np.ix_(r, T)])
    return Sigma, A[np.ix_(T, T)] - Sigma


def harmonic_weights(bc: BaseCase, T) -> np.ndarray:
    """The constraint modes w^(i) = L_rr^-1 K_ri, one column per node of T.

    The potential on the bath when node i of T is held at 1 and the others at 0.
    For |T| = 2 they sum to the all-ones vector, and for |T| = 1, w = 1 and the
    condensed mass is the whole bath mass -- both asserted in `validate.py`.
    """
    T = np.atleast_1d(np.asarray(T, int))
    r = np.setdiff1d(np.arange(bc.N), T)
    Lrr = bc.L[np.ix_(r, r)]
    return np.linalg.solve(Lrr, -bc.L[np.ix_(r, T)])


def condensed(bc: BaseCase, T) -> dict[str, np.ndarray]:
    """Guyan / Craig-Bampton condensation onto T: M_eff, Gamma_eff, L_eff.

    The s -> 0 expansion of S_T.  The spec's trap applies and is worth repeating
    where it can be read next to the code: the expansion has a finite radius of
    convergence set by the first Dirichlet mode of the bath, which on a real
    network lies *inside* the band of interest.  These numbers are reported for
    the winning edge as an explanation of where its effect came from; they are
    never evaluated as a band-wide constant and never optimised.
    """
    T = np.atleast_1d(np.asarray(T, int))
    r = np.setdiff1d(np.arange(bc.N), T)
    w = harmonic_weights(bc, T)
    m, d = bc.M, bc.D + bc.droop / (2.0 * np.pi * 50.0)
    Meff = np.diag(m[T]) + w.T @ (m[r][:, None] * w)
    Geff = np.diag(d[T]) + w.T @ (d[r][:, None] * w)
    Leff = bc.L[np.ix_(T, T)] + bc.L[np.ix_(T, r)] @ w
    return {"M": Meff, "Gamma": Geff, "L": Leff, "w": w}
