"""Potts-model simulated annealing with incremental energy evaluation (numba).

Energy, for labels sigma and fixed k::

    H = sum_(i,j) w_ij delta(s_i, s_j)                 w_ij = dp~_ij - alpha J~_ij
        + lambda_c sum_s (C_s - 1)                    C_s = #connected components of zone s
        + lambda_b sum_s hinge(s)                     hinge = ((f - m_s)/f)^2 if m_s < f else 0
                                                      m_s = max(load_s, gen_s) / national load

The graph (CSR ``ptr, idx, w``) is the intra-country graph: merged AC lines
plus HVDC links (both count as adjacency for C_s).

Incremental cost of a single-node move i: a -> b
* Potts term: O(deg i).
* Balance term: O(1) (zone load/generation sums are maintained).
* Contiguity: joining b changes C_b by ``1 - R_b`` where R_b is the number of
  distinct b-components among i's b-neighbours; leaving a changes C_a by
  ``R_a - 1`` where R_a is the number of components the a-neighbours fall into
  once i is removed.  Both R are found by an *interleaved* BFS from the
  neighbours, restricted to the zone, that stops as soon as all seeds have
  merged into one group or all but one group are exhausted.  This is O(deg)
  whenever the neighbours reconnect locally, and otherwise bounded by (deg x)
  the size of the smaller fragments; it is never a full recomputation.  (Exact
  contiguity accounting cannot be strictly O(deg) in general.)

Moves that would empty a zone are rejected, so k is fixed.
Proposals: a uniformly random *boundary* node, moved to the label of a
uniformly random differently-labelled neighbour.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #

@njit(cache=True)
def hinge(L, G, Ltot, floor):
    share = max(L, G) / Ltot
    if share >= floor:
        return 0.0
    x = (floor - share) / floor
    return x * x


@njit(cache=True)
def _find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


@njit(cache=True)
def count_components_among(seeds, m, z, excl, labels, ptr, idx, mark, owner, stamp,
                           Q, qh, qt, parent, active):
    """Number of connected components of {v : labels[v] == z, v != excl} that
    contain the m (distinct) seed nodes.  Interleaved BFS, early exit."""
    if m == 0:
        return 0
    if m == 1:
        return 1
    for t in range(m):
        s = seeds[t]
        parent[t] = t
        mark[s] = stamp
        owner[s] = t
        Q[t, 0] = s
        qh[t] = 0
        qt[t] = 1
    n_roots = m
    while True:
        if n_roots == 1:
            return 1
        for t in range(m):
            active[t] = False
        n_active = 0
        for t in range(m):
            if qh[t] < qt[t]:
                r = _find(parent, t)
                if not active[r]:
                    active[r] = True
                    n_active += 1
        if n_active <= 1:
            return n_roots
        for t in range(m):
            if qh[t] < qt[t]:
                v = Q[t, qh[t]]
                qh[t] += 1
                for e in range(ptr[v], ptr[v + 1]):
                    u = idx[e]
                    if u == excl or labels[u] != z:
                        continue
                    if mark[u] != stamp:
                        mark[u] = stamp
                        owner[u] = t
                        Q[t, qt[t]] = u
                        qt[t] += 1
                    else:
                        ra = _find(parent, t)
                        rb = _find(parent, owner[u])
                        if ra != rb:
                            parent[rb] = ra
                            n_roots -= 1


@njit(cache=True)
def zone_components(labels, k, ptr, idx):
    """Full recomputation of C_s for every zone (BFS). Used for init and tests."""
    n = labels.shape[0]
    seen = np.zeros(n, np.bool_)
    C = np.zeros(k, np.int64)
    stack = np.empty(n, np.int64)
    for s in range(n):
        if seen[s]:
            continue
        z = labels[s]
        C[z] += 1
        seen[s] = True
        top = 0
        stack[top] = s
        top += 1
        while top > 0:
            top -= 1
            v = stack[top]
            for e in range(ptr[v], ptr[v + 1]):
                u = idx[e]
                if not seen[u] and labels[u] == z:
                    seen[u] = True
                    stack[top] = u
                    top += 1
    return C


@njit(cache=True)
def energy_terms(labels, k, ptr, idx, w, Lnode, Gnode, Ltot, floor):
    """Return (potts, sum_s (C_s - 1), sum_s hinge_s) by full recomputation."""
    n = labels.shape[0]
    ep = 0.0
    for i in range(n):
        for e in range(ptr[i], ptr[i + 1]):
            j = idx[e]
            if j > i and labels[j] == labels[i]:
                ep += w[e]
    C = zone_components(labels, k, ptr, idx)
    zL = np.zeros(k)
    zG = np.zeros(k)
    for i in range(n):
        zL[labels[i]] += Lnode[i]
        zG[labels[i]] += Gnode[i]
    pb = 0.0
    for s in range(k):
        pb += hinge(zL[s], zG[s], Ltot, floor)
    return ep, float(C.sum() - k), pb


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #

class Workspace:
    """Scratch arrays for the BFS; allocated once per graph."""

    def __init__(self, n, maxdeg):
        self.mark = np.zeros(n, np.int64)
        self.owner = np.zeros(n, np.int64)
        self.Q = np.zeros((max(maxdeg, 1), n), np.int64)
        self.qh = np.zeros(max(maxdeg, 1), np.int64)
        self.qt = np.zeros(max(maxdeg, 1), np.int64)
        self.parent = np.zeros(max(maxdeg, 1), np.int64)
        self.active = np.zeros(max(maxdeg, 1), np.bool_)
        self.seeds = np.zeros(max(maxdeg, 1), np.int64)
        self.stamp = np.zeros(1, np.int64)


@njit(cache=True)
def move_delta(i, b, labels, ptr, idx, w, Lnode, Gnode, zL, zG, Ltot, floor,
               mark, owner, stamp, Q, qh, qt, parent, active, seeds):
    """(dPotts, dC_a, dC_b, dHinge) for moving node i to zone b. State unchanged."""
    a = labels[i]
    dp = 0.0
    for e in range(ptr[i], ptr[i + 1]):
        lj = labels[idx[e]]
        if lj == b:
            dp += w[e]
        elif lj == a:
            dp -= w[e]
    dh = (hinge(zL[a] - Lnode[i], zG[a] - Gnode[i], Ltot, floor)
          + hinge(zL[b] + Lnode[i], zG[b] + Gnode[i], Ltot, floor)
          - hinge(zL[a], zG[a], Ltot, floor) - hinge(zL[b], zG[b], Ltot, floor))
    # leaving a
    m = 0
    for e in range(ptr[i], ptr[i + 1]):
        j = idx[e]
        if labels[j] == a:
            seeds[m] = j
            m += 1
    stamp[0] += 1
    Ra = count_components_among(seeds, m, a, i, labels, ptr, idx, mark, owner, stamp[0],
                                Q, qh, qt, parent, active)
    # joining b
    m = 0
    for e in range(ptr[i], ptr[i + 1]):
        j = idx[e]
        if labels[j] == b:
            seeds[m] = j
            m += 1
    stamp[0] += 1
    Rb = count_components_among(seeds, m, b, i, labels, ptr, idx, mark, owner, stamp[0],
                                Q, qh, qt, parent, active)
    return dp, Ra - 1, 1 - Rb, dh


@njit(cache=True)
def _label_components(labels, ptr, idx, comp, stack):
    """comp[v] = component id (within its zone); returns number of components."""
    n = labels.shape[0]
    for v in range(n):
        comp[v] = -1
    nc = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        z = labels[s]
        comp[s] = nc
        top = 0
        stack[top] = s
        top += 1
        while top > 0:
            top -= 1
            v = stack[top]
            for e in range(ptr[v], ptr[v + 1]):
                u = idx[e]
                if comp[u] < 0 and labels[u] == z:
                    comp[u] = nc
                    stack[top] = u
                    top += 1
        nc += 1
    return nc


@njit(cache=True)
def fragment_pass(labels, k, ptr, idx, w, Lnode, Gnode, zL, zG, cnt, Ltot, floor, lam_b,
                  lam_c, T, comp, stack):
    """Cluster moves: every non-largest connected piece of a zone is offered, as a
    whole, to its best adjacent zone and accepted by the Metropolis rule.

    Moving fragment F from a to b: Potts changes by the weight of the F-b edges
    (F's internal edges stay intra-zone and F has no edge to the rest of a),
    C_a drops by 1 and C_b changes by 1 - (#b-components F touches).  Costs one
    O(n) component labelling per accepted move; this is a separate move type,
    not the single-flip path.  Returns (dPotts, dCsum, dHinge, n_accepted).
    """
    n = labels.shape[0]
    tot_dp = 0.0
    tot_dc = 0
    tot_dh = 0.0
    n_acc = 0
    dpb = np.zeros(k)
    for it in range(n):          # each accepted move removes >= 1 fragment
        nc = _label_components(labels, ptr, idx, comp, stack)
        size = np.zeros(nc, np.int64)
        czone = np.zeros(nc, np.int64)
        for v in range(n):
            size[comp[v]] += 1
            czone[comp[v]] = labels[v]
        main = -np.ones(k, np.int64)
        for c in range(nc):
            z = czone[c]
            if main[z] < 0 or size[c] > size[main[z]]:
                main[z] = c
        touched = np.zeros(nc, np.bool_)
        tlist = np.empty(nc, np.int64)
        order = np.random.permutation(nc)
        moved = False
        for c in order:
            a = czone[c]
            if main[a] == c:
                continue
            members = np.flatnonzero(comp == c)
            FL = 0.0
            FG = 0.0
            for v in members:
                FL += Lnode[v]
                FG += Gnode[v]
            for z in range(k):
                dpb[z] = 0.0
            for v in members:
                for e in range(ptr[v], ptr[v + 1]):
                    u = idx[e]
                    if labels[u] != a:
                        dpb[labels[u]] += w[e]
            best_b = -1
            best_dE = 1e300
            best_dp = 0.0
            best_dc = 0
            best_dh = 0.0
            for b in range(k):
                if b == a:
                    continue
                nt = 0
                for v in members:
                    for e in range(ptr[v], ptr[v + 1]):
                        u = idx[e]
                        if labels[u] == b and not touched[comp[u]]:
                            touched[comp[u]] = True
                            tlist[nt] = comp[u]
                            nt += 1
                for t in range(nt):
                    touched[tlist[t]] = False
                if nt == 0:
                    continue
                dc = -1 + (1 - nt)
                dh = (hinge(zL[a] - FL, zG[a] - FG, Ltot, floor)
                      + hinge(zL[b] + FL, zG[b] + FG, Ltot, floor)
                      - hinge(zL[a], zG[a], Ltot, floor) - hinge(zL[b], zG[b], Ltot, floor))
                dE = dpb[b] + lam_c * dc + lam_b * dh
                if dE < best_dE:
                    best_dE = dE
                    best_b = b
                    best_dp = dpb[b]
                    best_dc = dc
                    best_dh = dh
            if best_b < 0:
                continue
            if T > 0.0:
                ok = best_dE <= 0.0 or np.random.random() < math.exp(-best_dE / T)
            else:
                ok = best_dE < 0.0
            if not ok:
                continue
            b = best_b
            for v in members:
                labels[v] = b
            zL[a] -= FL
            zG[a] -= FG
            zL[b] += FL
            zG[b] += FG
            cnt[a] -= members.shape[0]
            cnt[b] += members.shape[0]
            tot_dp += best_dp
            tot_dc += best_dc
            tot_dh += best_dh
            n_acc += 1
            moved = True
            break                # component ids are stale now: relabel
        if not moved:
            break
    return tot_dp, tot_dc, tot_dh, n_acc


@njit(cache=True)
def _rebuild_boundary(labels, ptr, idx, diff, bnd, pos):
    n = labels.shape[0]
    nb = 0
    for i in range(n):
        diff[i] = 0
        pos[i] = -1
        for e in range(ptr[i], ptr[i + 1]):
            if labels[idx[e]] != labels[i]:
                diff[i] += 1
        if diff[i] > 0:
            bnd[nb] = i
            pos[i] = nb
            nb += 1
    return nb


@njit(cache=True)
def _run(labels, k, ptr, idx, w, Lnode, Gnode, Ltot, floor, lam_b, lam_c_sched, T_sched,
         steps_per_T, quench_steps, seed, mark, owner, stamp, Q, qh, qt, parent, active, seeds):
    np.random.seed(seed)
    n = labels.shape[0]
    zL = np.zeros(k)
    zG = np.zeros(k)
    cnt = np.zeros(k, np.int64)
    for i in range(n):
        zL[labels[i]] += Lnode[i]
        zG[labels[i]] += Gnode[i]
        cnt[labels[i]] += 1
    ep, csum, pb = energy_terms(labels, k, ptr, idx, w, Lnode, Gnode, Ltot, floor)
    # boundary bookkeeping
    diff = np.zeros(n, np.int64)
    for i in range(n):
        for e in range(ptr[i], ptr[i + 1]):
            if labels[idx[e]] != labels[i]:
                diff[i] += 1
    bnd = np.empty(n, np.int64)
    pos = -np.ones(n, np.int64)
    nb = 0
    for i in range(n):
        if diff[i] > 0:
            bnd[nb] = i
            pos[i] = nb
            nb += 1
    comp = np.empty(n, np.int64)
    stack = np.empty(n, np.int64)
    nT = T_sched.shape[0]
    trace = np.zeros((nT + 1, 5))
    acc_total = 0
    for ti in range(nT + 1):
        if ti < nT:
            T = T_sched[ti]
            lam_c = lam_c_sched[ti]
            nsteps = steps_per_T
        else:
            T = 0.0
            lam_c = lam_c_sched[nT - 1]
            nsteps = quench_steps
        acc = 0
        for step in range(nsteps):
            if nb == 0:
                break
            i = bnd[np.random.randint(nb)]
            a = labels[i]
            if cnt[a] == 1:
                continue
            r = np.random.randint(diff[i])
            b = -1
            for e in range(ptr[i], ptr[i + 1]):
                lj = labels[idx[e]]
                if lj != a:
                    if r == 0:
                        b = lj
                        break
                    r -= 1
            # threshold for Metropolis: accept iff dE <= thr
            if T > 0.0:
                u = np.random.random()
                thr = -T * math.log(u) if u > 0.0 else 1e300
            else:
                thr = -1e-12
            # cheap lower bound before the BFS
            dp = 0.0
            mb = 0
            ma = 0
            for e in range(ptr[i], ptr[i + 1]):
                lj = labels[idx[e]]
                if lj == b:
                    dp += w[e]
                    mb += 1
                elif lj == a:
                    dp -= w[e]
                    ma += 1
            dh = (hinge(zL[a] - Lnode[i], zG[a] - Gnode[i], Ltot, floor)
                  + hinge(zL[b] + Lnode[i], zG[b] + Gnode[i], Ltot, floor)
                  - hinge(zL[a], zG[a], Ltot, floor) - hinge(zL[b], zG[b], Ltot, floor))
            dca_min = -1 if ma == 0 else 0
            dcb_min = 1 - mb
            if dp + lam_b * dh + lam_c * (dca_min + dcb_min) > thr:
                continue
            dp2, dca, dcb, dh2 = move_delta(i, b, labels, ptr, idx, w, Lnode, Gnode, zL, zG,
                                            Ltot, floor, mark, owner, stamp, Q, qh, qt,
                                            parent, active, seeds)
            dE = dp2 + lam_c * (dca + dcb) + lam_b * dh2
            if dE > thr:
                continue
            # apply
            labels[i] = b
            zL[a] -= Lnode[i]
            zG[a] -= Gnode[i]
            zL[b] += Lnode[i]
            zG[b] += Gnode[i]
            cnt[a] -= 1
            cnt[b] += 1
            ep += dp2
            csum += dca + dcb
            pb += dh2
            acc += 1
            # boundary update
            d_i = 0
            for e in range(ptr[i], ptr[i + 1]):
                j = idx[e]
                lj = labels[j]
                if lj != b:
                    d_i += 1
                old = diff[j]
                if lj == a:
                    diff[j] += 1
                elif lj == b:
                    diff[j] -= 1
                if old == 0 and diff[j] > 0:
                    bnd[nb] = j
                    pos[j] = nb
                    nb += 1
                elif old > 0 and diff[j] == 0:
                    p = pos[j]
                    last = bnd[nb - 1]
                    bnd[p] = last
                    pos[last] = p
                    pos[j] = -1
                    nb -= 1
            old = diff[i]
            diff[i] = d_i
            if old == 0 and d_i > 0:
                bnd[nb] = i
                pos[i] = nb
                nb += 1
            elif old > 0 and d_i == 0:
                p = pos[i]
                last = bnd[nb - 1]
                bnd[p] = last
                pos[last] = p
                pos[i] = -1
                nb -= 1
        # fragment (cluster) moves once per temperature step
        if csum > 0:
            fdp, fdc, fdh, nacc = fragment_pass(labels, k, ptr, idx, w, Lnode, Gnode, zL, zG,
                                                cnt, Ltot, floor, lam_b, lam_c, T, comp, stack)
            if nacc > 0:
                ep += fdp
                csum += fdc
                pb += fdh
                nb = _rebuild_boundary(labels, ptr, idx, diff, bnd, pos)
        acc_total += acc
        trace[ti, 0] = T
        trace[ti, 1] = lam_c
        trace[ti, 2] = ep + lam_c * csum + lam_b * pb
        trace[ti, 3] = csum
        trace[ti, 4] = acc / max(nsteps, 1)
    return ep, csum, pb, trace


# --------------------------------------------------------------------------- #
# python-facing API
# --------------------------------------------------------------------------- #

class CountryGraph:
    """CSR graph + node attributes for one country."""

    def __init__(self, n, edges_i, edges_j, weights, Lnode, Gnode):
        self.n = n
        i = np.concatenate([edges_i, edges_j]).astype(np.int64)
        j = np.concatenate([edges_j, edges_i]).astype(np.int64)
        ww = np.concatenate([weights, weights]).astype(np.float64)
        o = np.lexsort((j, i))
        i, j, ww = i[o], j[o], ww[o]
        self.ptr = np.zeros(n + 1, np.int64)
        np.add.at(self.ptr, i + 1, 1)
        self.ptr = np.cumsum(self.ptr)
        self.idx = j
        self.w = ww
        self.Lnode = np.asarray(Lnode, np.float64)
        self.Gnode = np.asarray(Gnode, np.float64)
        self.Ltot = float(self.Lnode.sum())
        deg = np.diff(self.ptr)
        self.maxdeg = int(deg.max()) if n else 0
        self.ws = Workspace(n, self.maxdeg)


def graph_voronoi(g: CountryGraph, k: int, rng: np.random.Generator) -> np.ndarray:
    """Contiguous random initial partition: multi-source BFS from k random seeds."""
    labels = -np.ones(g.n, np.int64)
    seeds = rng.choice(g.n, size=k, replace=False)
    frontier = [[int(s)] for s in seeds]
    for z, s in enumerate(seeds):
        labels[s] = z
    while any(frontier):
        order = rng.permutation(k)
        for z in order:
            if not frontier[z]:
                continue
            nxt = []
            for v in frontier[z]:
                for e in range(g.ptr[v], g.ptr[v + 1]):
                    u = g.idx[e]
                    if labels[u] < 0:
                        labels[u] = z
                        nxt.append(int(u))
            frontier[z] = nxt
    # disconnected leftovers (should not occur on a connected graph)
    if (labels < 0).any():
        labels[labels < 0] = rng.integers(0, k, int((labels < 0).sum()))
    return labels


def initial_temperature(g: CountryGraph, labels, k, lam_b, lam_c, floor, rng, n_samples=400,
                        target_accept=0.6, exclude_contiguity: bool = False):
    """T0 such that a typical uphill move is accepted with ~target probability.

    ``exclude_contiguity=True`` evaluates the sampled dE with lambda_c = 0, i.e. on
    the physical terms only.  Needed when lambda_c is pinned at the contiguity
    guarantee (a huge number on a large graph), which would otherwise dominate the
    median and blow T0 up.  Default False: the static pipeline is unchanged.
    """
    if exclude_contiguity:
        lam_c = 0.0
    ws = g.ws
    zL = np.zeros(k)
    zG = np.zeros(k)
    np.add.at(zL, labels, g.Lnode)
    np.add.at(zG, labels, g.Gnode)
    ups = []
    for _ in range(n_samples):
        i = int(rng.integers(g.n))
        nbr = g.idx[g.ptr[i]:g.ptr[i + 1]]
        other = [int(labels[j]) for j in nbr if labels[j] != labels[i]]
        if not other:
            continue
        b = other[int(rng.integers(len(other)))]
        dp, dca, dcb, dh = move_delta(i, b, labels, g.ptr, g.idx, g.w, g.Lnode, g.Gnode, zL, zG,
                                      g.Ltot, floor, ws.mark, ws.owner, ws.stamp, ws.Q, ws.qh,
                                      ws.qt, ws.parent, ws.active, ws.seeds)
        dE = dp + lam_c * (dca + dcb) + lam_b * dh
        if dE > 0:
            ups.append(dE)
    if not ups:
        return 1.0
    return float(np.median(ups) / -math.log(target_accept))


def contiguity_guarantee(g: CountryGraph, lam_b: float) -> float:
    """A lambda_c above any possible non-contiguity energy change.

    Dissolving a fragment changes the Potts term by at most sum |w| (over
    undirected edges) and the balance term by at most 2 lambda_b, while it
    lowers sum(C_s - 1) by at least 1.  With lambda_c above this bound the
    T = 0 fragment pass always accepts, so the final state is contiguous.
    """
    return float(np.abs(g.w).sum() / 2 + 2 * lam_b + 1.0)


def anneal(g: CountryGraph, k: int, lam_b: float, lam_c0: float, lam_c1, floor: float,
           n_temps: int, sweeps_per_temp: float, t_final_ratio: float, seed: int,
           init: np.ndarray | None = None, quench_sweeps: float = 20.0):
    rng = np.random.default_rng(seed)
    if lam_c1 is None or lam_c1 == "auto":
        lam_c1 = contiguity_guarantee(g, lam_b)
    labels = graph_voronoi(g, k, rng) if init is None else init.copy()
    T0 = initial_temperature(g, labels, k, lam_b, lam_c0, floor, rng)
    T = T0 * t_final_ratio ** (np.arange(n_temps) / max(n_temps - 1, 1))
    lam_c = lam_c0 * (lam_c1 / lam_c0) ** (np.arange(n_temps) / max(n_temps - 1, 1))
    steps = int(sweeps_per_temp * g.n)
    ws = g.ws
    ep, csum, pb, trace = _run(labels, k, g.ptr, g.idx, g.w, g.Lnode, g.Gnode, g.Ltot, floor,
                               lam_b, lam_c, T, steps, int(quench_sweeps * g.n), seed,
                               ws.mark, ws.owner, ws.stamp, ws.Q, ws.qh, ws.qt, ws.parent,
                               ws.active, ws.seeds)
    return {"labels": labels, "potts": ep, "contig": csum, "balance": pb,
            "energy": ep + lam_c1 * csum + lam_b * pb, "T0": T0, "trace": trace,
            "lam_c_final": lam_c1}
