"""Fallback copy of the capacity-weighted Mirkin primitives.

Used by ``bzgen.ndbz.rigidity`` only when ``bzgen.dbz.rigidity`` (the European
DBZ variant, developed on a parallel branch) is not importable.  It is the same
code with the same API, so once ``bzgen.dbz`` is merged the import switches to it
and this file can be deleted.  ``tests/test_ndbz.py`` runs the rigidity tests
against whichever implementation is active *and* against this fallback.

All functions work with the unnormalised sum ``Z * H_rigid``::

    Z * H_rigid = sum_i g_i |BZ_B(i) symdiff BZ_A(i)|
                = sum_{p,a} g[p,a] (nB[p] + nA[a] - 2 n[p,a])

A single-node or fragment move changes only rows p and q of the tables.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def build_tables(B, A, cap, KB, KA):
    """(n, g, nB, gB) for labelling B against anchor A."""
    n = np.zeros((KB, KA), np.int64)
    g = np.zeros((KB, KA))
    nB = np.zeros(KB, np.int64)
    gB = np.zeros(KB)
    for i in range(B.shape[0]):
        p = B[i]
        a = A[i]
        n[p, a] += 1
        g[p, a] += cap[i]
        nB[p] += 1
        gB[p] += cap[i]
    return n, g, nB, gB


@njit(cache=True)
def rigidity_from_tables(n, g, nB, gB):
    """Z * H_rigid from the tables (full O(K_B K_A) double sum)."""
    KB, KA = n.shape
    nA = np.zeros(KA, np.int64)
    for p in range(KB):
        for a in range(KA):
            nA[a] += n[p, a]
    s = 0.0
    for p in range(KB):
        for a in range(KA):
            if g[p, a] != 0.0:
                s += g[p, a] * (nB[p] + nA[a] - 2 * n[p, a])
    return s


@njit(cache=True)
def rigidity_full(B, A, cap, KB, KA):
    """Reference: Z * H_rigid by building the tables from scratch."""
    n, g, nB, gB = build_tables(B, A, cap, KB, KA)
    return rigidity_from_tables(n, g, nB, gB)


@njit(cache=True)
def rigidity_delta(p, q, a, c, n, g, nB, gB):
    """Exact change of Z * H_rigid when one node (anchor zone a, capacity c)
    moves p -> q.  Tables unchanged."""
    if p == q:
        return 0.0
    d_row = (-gB[p] - c * nB[p] + c) + (gB[q] + c * nB[q] + c)
    d_x = (-g[p, a] - c * n[p, a] + c) + (g[q, a] + c * n[q, a] + c)
    return d_row - 2.0 * d_x


@njit(cache=True)
def rigidity_apply(p, q, a, c, n, g, nB, gB):
    n[p, a] -= 1
    n[q, a] += 1
    g[p, a] -= c
    g[q, a] += c
    nB[p] -= 1
    nB[q] += 1
    gB[p] -= c
    gB[q] += c


@njit(cache=True)
def fragment_profile(members, A, cap, fa, fn, fg):
    """Aggregate a fragment per A-zone: the m distinct zones in fa[:m], with
    counts fn[a] and capacities fg[a] (fn, fg must be zero on entry for those a).
    Returns (m, N_F, G_F)."""
    m = 0
    GF = 0.0
    for v in members:
        a = A[v]
        if fn[a] == 0:
            fa[m] = a
            m += 1
        fn[a] += 1
        fg[a] += cap[v]
        GF += cap[v]
    return m, members.shape[0], GF


@njit(cache=True)
def fragment_clear(fa, m, fn, fg):
    for t in range(m):
        fn[fa[t]] = 0
        fg[fa[t]] = 0.0


@njit(cache=True)
def rigidity_delta_fragment(p, q, fa, m, fn, fg, NF, GF, n, g, nB, gB):
    """Exact change of Z * H_rigid when a whole fragment moves p -> q."""
    if p == q:
        return 0.0
    d_row = (-gB[p] * NF - GF * nB[p] + GF * NF) + (gB[q] * NF + GF * nB[q] + GF * NF)
    d_x = 0.0
    for t in range(m):
        a = fa[t]
        d_x += (-g[p, a] * fn[a] - fg[a] * n[p, a] + fg[a] * fn[a]
                + g[q, a] * fn[a] + fg[a] * n[q, a] + fg[a] * fn[a])
    return d_row - 2.0 * d_x


@njit(cache=True)
def rigidity_apply_fragment(p, q, fa, m, fn, fg, NF, GF, n, g, nB, gB):
    for t in range(m):
        a = fa[t]
        n[p, a] -= fn[a]
        n[q, a] += fn[a]
        g[p, a] -= fg[a]
        g[q, a] += fg[a]
    nB[p] -= NF
    nB[q] += NF
    gB[p] -= GF
    gB[q] += GF
