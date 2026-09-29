"""European annealer: the static Potts annealer plus the rigidity penalty.

    H = sum w_ij delta(s_i, s_j) + lambda_c sum_s (C_s - 1) + lambda_b sum_s hinge_s
        + lambda_rigid * H_rigid(B; A)

Everything that is agnostic to the graph's extent is imported from
``bzgen.cluster.anneal``.  ``_run`` and ``fragment_pass`` are ``@njit`` and the
rigidity term must be inside their loops, so they are forked here as
``_run_dbz`` / ``fragment_pass_dbz``; the originals are untouched.

Schedule differences from the static runs (the DBZ spec, §4.5):
* K fixed from the anchor; the empty-zone rejection (``cnt[a] == 1``) conserves it.
* lambda_c constant at ``contiguity_guarantee`` for the whole schedule, so
  contiguity-breaking moves are rejected by the cheap lower bound before the BFS.
* T0 calibrated on every term except contiguity (Potts + balance + rigidity;
  ``initial_temperature_dbz``, which equals ``initial_temperature(...,
  exclude_contiguity=True)`` at lambda_rigid = 0).  lambda_c is excluded because
  at the contiguity guarantee it would blow T0 up; rigidity is included because
  otherwise it is frozen from the first step at large lambda_rigid.
* Initial labels: ``graph_voronoi`` run separately on each connected component of
  the European graph (IE is an island: GB is out of scope), with as many zones
  per component as the anchor has there.  A zone never crosses components
  (moves only take a neighbour's label), so this fixes each component's zone count.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from bzgen.cluster.anneal import (CountryGraph, _label_components, _rebuild_boundary,
                                  contiguity_guarantee, energy_terms, graph_voronoi, hinge,
                                  initial_temperature, move_delta)
from bzgen.dbz.rigidity import (build_tables, fragment_clear, fragment_profile,
                                rigidity_apply, rigidity_apply_fragment, rigidity_delta,
                                rigidity_delta_fragment, rigidity_from_tables)


# --------------------------------------------------------------------------- #
# forks of bzgen.cluster.anneal._run / fragment_pass
# --------------------------------------------------------------------------- #

@njit(cache=True)
def fragment_pass_dbz(labels, k, ptr, idx, w, Lnode, Gnode, zL, zG, cnt, Ltot, floor, lam_b,
                      lam_c, T, comp, stack, A, cap, lam_rz, tn, tg, tnB, tgB, fa, fn, fg):
    """FORK of bzgen.cluster.anneal.fragment_pass.  Added: the rigidity change of
    moving the whole fragment (``rigidity_delta_fragment``, weighted by
    lam_rz = lambda_rigid / Z) enters every candidate's dE, and accepted moves
    update the contingency tables.  Returns (dPotts, dCsum, dHinge, dRigid_raw, n_acc)."""
    n = labels.shape[0]
    tot_dp = 0.0
    tot_dc = 0
    tot_dh = 0.0
    tot_dr = 0.0
    n_acc = 0
    dpb = np.zeros(k)
    for it in range(n):
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
            m, NF, GF = fragment_profile(members, A, cap, fa, fn, fg)
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
            best_dr = 0.0
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
                dr = rigidity_delta_fragment(a, b, fa, m, fn, fg, NF, GF, tn, tg, tnB, tgB)
                dE = dpb[b] + lam_c * dc + lam_b * dh + lam_rz * dr
                if dE < best_dE:
                    best_dE = dE
                    best_b = b
                    best_dp = dpb[b]
                    best_dc = dc
                    best_dh = dh
                    best_dr = dr
            ok = False
            if best_b >= 0:
                if T > 0.0:
                    ok = best_dE <= 0.0 or np.random.random() < math.exp(-best_dE / T)
                else:
                    ok = best_dE < 0.0
            if not ok:
                fragment_clear(fa, m, fn, fg)
                continue
            b = best_b
            rigidity_apply_fragment(a, b, fa, m, fn, fg, NF, GF, tn, tg, tnB, tgB)
            fragment_clear(fa, m, fn, fg)
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
            tot_dr += best_dr
            n_acc += 1
            moved = True
            break
        if not moved:
            break
    return tot_dp, tot_dc, tot_dh, tot_dr, n_acc


@njit(cache=True)
def _run_dbz(labels, k, ptr, idx, w, Lnode, Gnode, Ltot, floor, lam_b, lam_c_sched, T_sched,
             steps_per_T, quench_steps, seed, mark, owner, stamp, Q, qh, qt, parent, active,
             seeds, A, KA, cap, lam_rz):
    """FORK of bzgen.cluster.anneal._run.  Added: the rigidity term.  Contingency
    tables of labels vs the anchor A are built at the start; every single-flip
    proposal adds lam_rz * rigidity_delta (O(1), exact) to dE — inside the cheap
    lower bound too, so rigidity-rejected moves skip the BFS; accepted moves update
    the tables; fragment moves use fragment_pass_dbz; the running rigidity total is
    re-synchronised from the tables after every temperature.  ``trace`` gains a
    column 5: the rigidity total (unnormalised, Z * H_rigid)."""
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
    tn, tg, tnB, tgB = build_tables(labels, A, cap, k, KA)
    pr = rigidity_from_tables(tn, tg, tnB, tgB)
    fa = np.zeros(KA, np.int64)
    fn = np.zeros(KA, np.int64)
    fg = np.zeros(KA)
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
    trace = np.zeros((nT + 1, 6))
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
            if T > 0.0:
                u = np.random.random()
                thr = -T * math.log(u) if u > 0.0 else 1e300
            else:
                thr = -1e-12
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
            dr = rigidity_delta(a, b, A[i], cap[i], tn, tg, tnB, tgB)
            dca_min = -1 if ma == 0 else 0
            dcb_min = 1 - mb
            if dp + lam_b * dh + lam_rz * dr + lam_c * (dca_min + dcb_min) > thr:
                continue
            dp2, dca, dcb, dh2 = move_delta(i, b, labels, ptr, idx, w, Lnode, Gnode, zL, zG,
                                            Ltot, floor, mark, owner, stamp, Q, qh, qt,
                                            parent, active, seeds)
            dE = dp2 + lam_c * (dca + dcb) + lam_b * dh2 + lam_rz * dr
            if dE > thr:
                continue
            labels[i] = b
            zL[a] -= Lnode[i]
            zG[a] -= Gnode[i]
            zL[b] += Lnode[i]
            zG[b] += Gnode[i]
            cnt[a] -= 1
            cnt[b] += 1
            rigidity_apply(a, b, A[i], cap[i], tn, tg, tnB, tgB)
            ep += dp2
            csum += dca + dcb
            pb += dh2
            pr += dr
            acc += 1
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
        if csum > 0:
            fdp, fdc, fdh, fdr, nacc = fragment_pass_dbz(
                labels, k, ptr, idx, w, Lnode, Gnode, zL, zG, cnt, Ltot, floor, lam_b, lam_c,
                T, comp, stack, A, cap, lam_rz, tn, tg, tnB, tgB, fa, fn, fg)
            if nacc > 0:
                ep += fdp
                csum += fdc
                pb += fdh
                pr += fdr
                nb = _rebuild_boundary(labels, ptr, idx, diff, bnd, pos)
        pr = rigidity_from_tables(tn, tg, tnB, tgB)      # re-sync (no drift)
        trace[ti, 0] = T
        trace[ti, 1] = lam_c
        trace[ti, 2] = ep + lam_c * csum + lam_b * pb + lam_rz * pr
        trace[ti, 3] = csum
        trace[ti, 4] = acc / max(nsteps, 1)
        trace[ti, 5] = pr
    return ep, csum, pb, pr, trace


# --------------------------------------------------------------------------- #
# python-facing API
# --------------------------------------------------------------------------- #

def components(g: CountryGraph) -> np.ndarray:
    src = np.repeat(np.arange(g.n), np.diff(g.ptr))
    m = coo_matrix((np.ones(len(src)), (src, g.idx)), shape=(g.n, g.n))
    return connected_components(m, directed=False)[1]


def zones_per_component(comp: np.ndarray, A: np.ndarray) -> dict:
    """Anchor zones per component: each anchor zone counted in the component that
    holds most of its nodes (the formerly-isolated singleton fragments aside)."""
    out = {}
    for z in np.unique(A):
        c = np.bincount(comp[A == z]).argmax()
        out[int(c)] = out.get(int(c), 0) + 1
    return out


def component_voronoi(g: CountryGraph, comp: np.ndarray, kc: dict,
                      rng: np.random.Generator) -> np.ndarray:
    """graph_voronoi on each connected component, kc[c] zones in component c;
    labels are made global (consecutive blocks)."""
    labels = -np.ones(g.n, np.int64)
    off = 0
    for c in sorted(kc):
        nodes = np.flatnonzero(comp == c)
        pos = -np.ones(g.n, np.int64)
        pos[nodes] = np.arange(len(nodes))
        src = np.repeat(np.arange(g.n), np.diff(g.ptr))
        m = (comp[src] == c) & (src < g.idx)
        sub = CountryGraph(len(nodes), pos[src[m]], pos[g.idx[m]], g.w[m],
                           g.Lnode[nodes], g.Gnode[nodes])
        labels[nodes] = graph_voronoi(sub, kc[c], rng) + off
        off += kc[c]
    assert (labels >= 0).all()
    return labels


def initial_temperature_dbz(g, labels, k, lam_b, floor, rng, A, KA, lam_rigid,
                            n_samples=400, target_accept=0.6):
    """T0 on every term but contiguity: Potts + lambda_b balance + lambda_rigid rigidity.

    Mirrors ``bzgen.cluster.anneal.initial_temperature(..., exclude_contiguity=True)``
    (same sampling, same random draws), with the rigidity change of each sampled
    move added; at lambda_rigid = 0 the two are identical (tested).  Including the
    rigidity term is what keeps the high-temperature phase active for it: otherwise
    at large lambda_rigid a typical move costs far more than T0 and the rigidity
    landscape is frozen from the first step.
    """
    ws = g.ws
    zL = np.zeros(k)
    zG = np.zeros(k)
    np.add.at(zL, labels, g.Lnode)
    np.add.at(zG, labels, g.Gnode)
    tn, tg, tnB, tgB = build_tables(labels, np.asarray(A, np.int64), g.cap, k, KA)
    lam_rz = lam_rigid / g.Z
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
        dE = dp + lam_b * dh
        if lam_rz != 0.0:
            dE += lam_rz * rigidity_delta(int(labels[i]), b, int(A[i]), float(g.cap[i]),
                                          tn, tg, tnB, tgB)
        if dE > 0:
            ups.append(dE)
    if not ups:
        return 1.0
    return float(np.median(ups) / -math.log(target_accept))


def anchor_seeded_voronoi(g: CountryGraph, A: np.ndarray, K: int,
                          rng: np.random.Generator) -> np.ndarray:
    """graph_voronoi growth from one uniformly random seed *inside each anchor zone*
    (the formerly-isolated singletons are never chosen when the zone has other
    nodes).  Not B = A: seeds are random and zones grow by BFS, so the start is a
    random contiguous partition with the anchor's number of zones per region."""
    seeds = np.empty(K, np.int64)
    for z in range(K):
        cand = np.flatnonzero(A == z)
        nb_same = np.array([(A[g.idx[g.ptr[v]:g.ptr[v + 1]]] == z).any() for v in cand])
        if nb_same.any():
            cand = cand[nb_same]
        seeds[z] = rng.choice(cand)
    labels = -np.ones(g.n, np.int64)
    labels[seeds] = np.arange(K)
    frontier = [[int(s)] for s in seeds]
    while any(frontier):
        for z in rng.permutation(K):
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
    assert (labels >= 0).all()
    return labels


INITS = ("voronoi", "anchor_seeded")


def anneal_dbz(g, A: np.ndarray, KA: int, lam_b: float, floor: float, lam_rigid: float,
               n_temps: int, sweeps_per_temp: float, t_final_ratio: float, seed: int,
               quench_sweeps: float = 20.0, T0_band=(1e-3, 1e3), init: np.ndarray | None = None,
               init_mode: str = "voronoi"):
    """One restart.  ``g`` is an EuropeGraph (needs ``cap``, ``Z``).  ``init_mode``:
    ``voronoi`` (graph_voronoi per connected component, random seeds) or
    ``anchor_seeded`` (graph_voronoi growth from one random seed per anchor zone)."""
    rng = np.random.default_rng(seed)
    K = KA
    comp = components(g)
    kc = zones_per_component(comp, A)
    if sum(kc.values()) != K:
        raise ValueError(f"zones per component {kc} do not sum to K={K}")
    if init is not None:
        labels = init.copy()
    elif init_mode == "voronoi":
        labels = component_voronoi(g, comp, kc, rng)
    elif init_mode == "anchor_seeded":
        labels = anchor_seeded_voronoi(g, A, K, rng)
    else:
        raise ValueError(init_mode)
    from bzgen.dbz.anchor import transfer_distance
    init_transfer = transfer_distance(labels, A, K, K)
    lam_c = contiguity_guarantee(g, lam_b)
    T0 = initial_temperature_dbz(g, labels, K, lam_b, floor, rng, A, KA, lam_rigid)
    if not (np.isfinite(T0) and T0_band[0] < T0 < T0_band[1]):
        raise RuntimeError(f"T0 = {T0} outside the sane band {T0_band}: refusing to anneal")
    T = T0 * t_final_ratio ** (np.arange(n_temps) / max(n_temps - 1, 1))
    lam_c_sched = np.full(n_temps, lam_c)
    steps = int(sweeps_per_temp * g.n)
    ws = g.ws
    lam_rz = float(lam_rigid) / g.Z
    ep, csum, pb, pr, trace = _run_dbz(
        labels, K, g.ptr, g.idx, g.w, g.Lnode, g.Gnode, g.Ltot, floor, lam_b, lam_c_sched, T,
        steps, int(quench_sweeps * g.n), seed, ws.mark, ws.owner, ws.stamp, ws.Q, ws.qh, ws.qt,
        ws.parent, ws.active, ws.seeds, np.asarray(A, np.int64), int(KA),
        np.asarray(g.cap, np.float64), lam_rz)
    rig = pr / g.Z
    return {"labels": labels, "potts": ep, "contig": csum, "balance": pb, "rigid": rig,
            "rigid_raw": pr, "energy": ep + lam_c * csum + lam_b * pb + lam_rigid * rig,
            "physical": ep + lam_b * pb, "T0": T0, "trace": trace, "lam_c": lam_c,
            "lam_rigid": lam_rigid, "init_mode": init_mode if init is None else "given",
            "init_transfer": init_transfer}
