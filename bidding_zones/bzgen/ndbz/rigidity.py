"""Rigidity penalty anchoring the scenario map B to the static map A (per country).

    H_rigid = (1 / Z_c) * sum_i g_i |BZ_B(i) symdiff BZ_A(i)|

The symmetric difference counts every annealed node of the country; the outer sum
is weighted by g_i = installed generation capacity, so load-only buses carry no
weight of their own but are still counted inside everybody else's term.  It is a
capacity-weighted Mirkin (pair-counting) penalty: label-invariant, so B's labels
never have to be matched to A's.  (Each node's term is symmetric in A and B, but
with g_i >= 0, many zero, it is not a metric: always a *penalty*, never a distance.)

Primitives: ``bzgen.dbz.rigidity`` when it is importable (the European variant,
shared implementation), otherwise the identical fallback ``_rigidity_core``.
Only the normalisation constant is national:

    Z_c = (2 n_c / k_c) * gbar_c        gbar_c = mean of g_i over the annealed nodes

so a typical single-node move costs O(1) in every country (a move p -> q at B = A
changes the sum by ~ G_p + G_q + g_i (n_p + n_q - 1), i.e. ~ Z_c (1 + g_i / gbar_c)).
If gbar_c = 0 the country falls back to gbar_c = 1 and is flagged.
"""

from __future__ import annotations

import numpy as np

try:                                                     # shared with the European DBZ
    from bzgen.dbz.rigidity import (build_tables, fragment_clear, fragment_profile,
                                    rigidity_apply, rigidity_apply_fragment, rigidity_delta,
                                    rigidity_delta_fragment, rigidity_from_tables, rigidity_full)
    SOURCE = "bzgen.dbz.rigidity"
except ImportError:
    from bzgen.ndbz._rigidity_core import (build_tables, fragment_clear, fragment_profile,
                                           rigidity_apply, rigidity_apply_fragment, rigidity_delta,
                                           rigidity_delta_fragment, rigidity_from_tables,
                                           rigidity_full)
    SOURCE = "bzgen.ndbz._rigidity_core"

__all__ = ["build_tables", "fragment_clear", "fragment_profile", "rigidity_apply",
           "rigidity_apply_fragment", "rigidity_delta", "rigidity_delta_fragment",
           "rigidity_from_tables", "rigidity_full", "normaliser", "SOURCE"]


def normaliser(cap: np.ndarray, k: int) -> tuple[float, float, bool]:
    """(Z_c, gbar_c, flagged): Z_c = (2 n / k) gbar, gbar = 1 if the country has no
    generation on any annealed node (flagged)."""
    cap = np.asarray(cap, float)
    n = len(cap)
    gbar = float(cap.mean()) if n else 0.0
    flagged = not gbar > 0
    if flagged:
        gbar = 1.0
    return 2.0 * n / k * gbar, gbar, flagged


def single_move_deltas(A: np.ndarray, cap: np.ndarray, k: int, ptr, idx, Z: float,
                       B: np.ndarray | None = None) -> np.ndarray:
    """Normalised dH_rigid of every boundary single-node move (i -> each different
    neighbouring label) from labelling B (default: A itself).  For reports/tests."""
    B = A.copy() if B is None else B
    n, g, nB, gB = build_tables(B, A, cap, k, k)
    out = []
    for i in range(len(B)):
        seen = set()
        for e in range(ptr[i], ptr[i + 1]):
            q = int(B[idx[e]])
            if q != B[i] and q not in seen:
                seen.add(q)
                out.append(rigidity_delta(int(B[i]), q, int(A[i]), float(cap[i]), n, g, nB, gB) / Z)
    return np.array(out)
