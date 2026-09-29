"""Per-country annealer for scenario re-zoning: the static annealer + rigidity + frames.

    H = sum_ij w_ij delta(s_i, s_j) + lambda_c sum_s (C_s - 1) + lambda_b sum_s hinge(s)
        + lambda_rigid * H_rigid(B; A)            H_rigid = (1/Z_c) sum_i g_i |B(i) symdiff A(i)|

``_run_ndbz`` and ``fragment_pass_ndbz`` are forks of ``bzgen.cluster.anneal._run``
and ``fragment_pass`` (both ``@njit``, so the new term has to live inside the hot
loop).  The originals are untouched.  Everything else is imported: ``hinge``,
``move_delta``, ``energy_terms``, ``Workspace``, ``graph_voronoi``,
``contiguity_guarantee``, ``initial_temperature``.

Nothing added here draws a random number, so at lambda_rigid = 0 with the static
schedule the fork follows the static RNG stream exactly and returns the static
result bit for bit (tests/test_ndbz.py), and frame capture on/off is bit-identical.

Scenario schedule (``mode="ndbz"``):
* lambda_c flat at ``contiguity_guarantee(g, lambda_b)`` for the whole schedule: A is
  contiguous and so must B be; the cheap lower-bound pre-check then rejects
  contiguity-breaking moves before the BFS.
* T0 calibrated with lambda_c = 0 (``initial_temperature_ndbz``: Potts + balance +
  rigidity; at lambda_rigid = 0 it *is* ``initial_temperature(...,
  exclude_contiguity=True)``, tested).  With the guarantee in the sampled dE the
  median uphill move would be ~ the guarantee (FR: ~1000 against a converged energy
  of ~ -1300) and the whole schedule would run at effectively infinite temperature.
  The rigidity term *is* included: calibrating on Potts + balance alone left the
  schedule far too cold for lambda_rigid = 10 (IE, baseline: frozen at acceptance
  6 % from the first temperature, E = -0.2 against -60.8 at B = A).  T0 outside
  ``T0_band`` fails loudly.
* initial state ``graph_voronoi`` (never B = A): zones form under the scenario,
  anchored but not seeded.
"""

from __future__ import annotations

import math
import zlib

import numpy as np
from numba import njit

from bzgen.cluster.anneal import (CountryGraph, contiguity_guarantee, energy_terms,
                                  graph_voronoi, hinge, initial_temperature, move_delta,
                                  _label_components, _rebuild_boundary)
from bzgen.ndbz.rigidity import (build_tables, fragment_clear, fragment_profile,
                                 rigidity_apply, rigidity_apply_fragment, rigidity_delta,
                                 rigidity_delta_fragment, rigidity_from_tables)

META = ("T", "lam_c", "potts", "contig", "balance", "rigid", "accept_rate")


@njit(cache=True)
def fragment_pass_ndbz(labels, k, ptr, idx, w, Lnode, Gnode, zL, zG, cnt, Ltot, floor, lam_b,
                       lam_c, T, comp, stack, A, cap, rho, tn, tg, tnB, tgB, fa, fn, fg):
    """Fork of ``bzgen.cluster.anneal.fragment_pass``.

    Added: the rigidity change of moving the whole fragment (``rho`` = lambda_rigid / Z_c
    times the exact table-differenced ``rigidity_delta_fragment``) enters the
    Metropolis dE and the choice of the best target zone, and accepted moves update the
    contingency tables.  No extra random numbers are drawn.
    Returns (dPotts, dCsum, dHinge, n_accepted); the caller re-synchronises the
    rigidity total from the tables.
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
            m_f, NF, GF = fragment_profile(members, A, cap, fa, fn, fg)
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
                dE += rho * rigidity_delta_fragment(a, b, fa, m_f, fn, fg, NF, GF,
                                                    tn, tg, tnB, tgB)
                if dE < best_dE:
                    best_dE = dE
                    best_b = b
                    best_dp = dpb[b]
                    best_dc = dc
                    best_dh = dh
            if best_b < 0:
                fragment_clear(fa, m_f, fn, fg)
                continue
            if T > 0.0:
                ok = best_dE <= 0.0 or np.random.random() < math.exp(-best_dE / T)
            else:
                ok = best_dE < 0.0
            if not ok:
                fragment_clear(fa, m_f, fn, fg)
                continue
            b = best_b
            rigidity_apply_fragment(a, b, fa, m_f, fn, fg, NF, GF, tn, tg, tnB, tgB)
            fragment_clear(fa, m_f, fn, fg)
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
def block_pass_ndbz(labels, k, ptr, idx, w, Lnode, Gnode, zL, zG, cnt, Ltot, floor, lam_b,
                    lam_c, T, comp, stack, A, cap, rho, tn, tg, tnB, tgB, fa, fn, fg, bcomp,
                    inx, seen):
    """Anchor-block cluster moves (new; no static counterpart).  Run only when
    lambda_rigid > 0.

    Under a pair-counting (Mirkin) penalty a *connected block* of nodes that A puts
    in zone a but B puts in zone p is a local minimum of single-node moves: moving
    one node out breaks its agreement with every node left behind.  The block can
    only move as a whole, and ``fragment_pass`` never offers it because it is still
    attached to its zone.  Here every connected component X of {B = p, A = a} with
    2 <= |X| < |p| is offered, whole, to its best adjacent zone q and accepted by the
    Metropolis rule on the exact dE: Potts (edges X-q gained, X-(p minus X) lost),
    contiguity (the B-component containing X splits into R pieces without it:
    dC_p = R - 1; X touches nt q-components: dC_q = 1 - nt), balance, rigidity
    (fragment table differencing).  Returns (dPotts, dCsum, dHinge, n_accepted).
    """
    n = labels.shape[0]
    tot_dp = 0.0
    tot_dc = 0
    tot_dh = 0.0
    n_acc = 0
    dpb = np.zeros(k)
    key = labels * k + A
    nbk = _label_components(key, ptr, idx, bcomp, stack)
    _label_components(labels, ptr, idx, comp, stack)
    ncomp_max = n
    touched = np.zeros(ncomp_max, np.bool_)
    tlist = np.empty(ncomp_max, np.int64)
    order = np.random.permutation(nbk)
    for v in range(n):
        seen[v] = 0             # stamps restart at 1 on every call
    stamp = 0
    for bi in order:
        members = np.flatnonzero(bcomp == bi)
        m = members.shape[0]
        if m < 2:
            continue
        p = labels[members[0]]
        if labels[members[0]] * k + A[members[0]] != key[members[0]]:
            continue            # moved as part of an earlier block (cannot happen, defensive)
        if m >= cnt[p]:
            continue            # would empty zone p (k is fixed)
        for v in members:
            inx[v] = True
        FL = 0.0
        FG = 0.0
        lost = 0.0
        for z in range(k):
            dpb[z] = 0.0
        for v in members:
            FL += Lnode[v]
            FG += Gnode[v]
            for e in range(ptr[v], ptr[v + 1]):
                u = idx[e]
                if inx[u]:
                    continue
                if labels[u] == p:
                    lost += w[e]
                else:
                    dpb[labels[u]] += w[e]
        # leaving p: pieces of X's B-component once X is removed
        cX = comp[members[0]]
        stamp += 1
        R = 0
        for v in members:
            for e in range(ptr[v], ptr[v + 1]):
                s0 = idx[e]
                if inx[s0] or labels[s0] != p or seen[s0] == stamp:
                    continue
                R += 1
                seen[s0] = stamp
                top = 0
                stack[top] = s0
                top += 1
                while top > 0:
                    top -= 1
                    x = stack[top]
                    for e2 in range(ptr[x], ptr[x + 1]):
                        y = idx[e2]
                        if labels[y] == p and not inx[y] and seen[y] != stamp:
                            seen[y] = stamp
                            stack[top] = y
                            top += 1
        # pieces of cX not adjacent to X cannot exist (cX is connected through X)
        dca = R - 1
        m_f, NF, GF = fragment_profile(members, A, cap, fa, fn, fg)
        best_b = -1
        best_dE = 1e300
        best_dp = 0.0
        best_dc = 0
        best_dh = 0.0
        for b in range(k):
            if b == p:
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
            dc = dca + (1 - nt)
            dh = (hinge(zL[p] - FL, zG[p] - FG, Ltot, floor)
                  + hinge(zL[b] + FL, zG[b] + FG, Ltot, floor)
                  - hinge(zL[p], zG[p], Ltot, floor) - hinge(zL[b], zG[b], Ltot, floor))
            dpp = dpb[b] - lost
            dE = dpp + lam_c * dc + lam_b * dh + rho * rigidity_delta_fragment(
                p, b, fa, m_f, fn, fg, NF, GF, tn, tg, tnB, tgB)
            if dE < best_dE:
                best_dE = dE
                best_b = b
                best_dp = dpp
                best_dc = dc
                best_dh = dh
        ok = False
        if best_b >= 0:
            if T > 0.0:
                ok = best_dE <= 0.0 or np.random.random() < math.exp(-best_dE / T)
            else:
                ok = best_dE < 0.0
        if ok:
            b = best_b
            rigidity_apply_fragment(p, b, fa, m_f, fn, fg, NF, GF, tn, tg, tnB, tgB)
            for v in members:
                labels[v] = b
                key[v] = b * k + A[v]
            zL[p] -= FL
            zG[p] -= FG
            zL[b] += FL
            zG[b] += FG
            cnt[p] -= m
            cnt[b] += m
            tot_dp += best_dp
            tot_dc += best_dc
            tot_dh += best_dh
            n_acc += 1
            _label_components(labels, ptr, idx, comp, stack)
        fragment_clear(fa, m_f, fn, fg)
        for v in members:
            inx[v] = False
    return tot_dp, tot_dc, tot_dh, n_acc


@njit(cache=True)
def _capture(frames, meta, f, labels, T, lam_c, ep, csum, pb, er_norm, acc_rate):
    for i in range(labels.shape[0]):
        frames[f, i] = labels[i]
    meta[f, 0] = T
    meta[f, 1] = lam_c
    meta[f, 2] = ep
    meta[f, 3] = csum
    meta[f, 4] = pb
    meta[f, 5] = er_norm
    meta[f, 6] = acc_rate


@njit(cache=True)
def _run_ndbz(labels, k, ptr, idx, w, Lnode, Gnode, Ltot, floor, lam_b, lam_c_sched, T_sched,
              steps_per_T, quench_steps, seed, mark, owner, stamp, Q, qh, qt, parent, active,
              seeds, A, cap, lam_rigid, Z, frames, frame_meta, capture, quench_stride, blocks):
    """Fork of ``bzgen.cluster.anneal._run``.

    Added: (1) the rigidity term lambda_rigid * H_rigid, evaluated per proposal by
    exact row-differencing of the (k x k) contingency tables (``rigidity_delta``),
    included in the cheap lower bound (it can be negative) and in the Metropolis dE,
    tables updated on acceptance, total re-synchronised after fragment moves;
    (2) frame capture: when ``capture``, ``labels`` and ``frame_meta`` (T, lambda_c,
    Potts, contiguity excess, balance, H_rigid, acceptance rate) are written for the
    initial state, at the end of every temperature step, every ``quench_stride``
    steps of the quench and once at the very end.  No extra random numbers are drawn.
    (3) when ``blocks`` and lambda_rigid > 0: ``block_pass_ndbz`` once per
    temperature step, after the fragment pass (draws random numbers, so only there).
    trace columns: T, lambda_c, E, csum, single-flip acceptance, H_rigid, cumulative
    accepted block moves.
    Returns (potts, csum, balance, H_rigid (normalised), trace, n_frames).
    """
    np.random.seed(seed)
    n = labels.shape[0]
    rho = lam_rigid / Z
    zL = np.zeros(k)
    zG = np.zeros(k)
    cnt = np.zeros(k, np.int64)
    for i in range(n):
        zL[labels[i]] += Lnode[i]
        zG[labels[i]] += Gnode[i]
        cnt[labels[i]] += 1
    ep, csum, pb = energy_terms(labels, k, ptr, idx, w, Lnode, Gnode, Ltot, floor)
    tn, tg, tnB, tgB = build_tables(labels, A, cap, k, k)
    er = rigidity_from_tables(tn, tg, tnB, tgB)
    fa = np.zeros(k, np.int64)
    fn = np.zeros(k, np.int64)
    fg = np.zeros(k)
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
    bcomp = np.empty(n, np.int64)
    inx = np.zeros(n, np.bool_)
    seen = np.zeros(n, np.int64)
    n_block = 0
    nT = T_sched.shape[0]
    trace = np.zeros((nT + 1, 7))
    nf = 0
    if capture:                                   # frame 0: the initial (graph-Voronoi) state
        _capture(frames, frame_meta, nf, labels, T_sched[0], lam_c_sched[0], ep, csum, pb,
                 er / Z, 0.0)
        nf += 1
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
            if capture and ti == nT and step > 0 and step % quench_stride == 0:
                _capture(frames, frame_meta, nf, labels, T, lam_c, ep, csum, pb, er / Z,
                         acc / step)
                nf += 1
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
            # cheap lower bound before the BFS (rigidity is exact and O(1): included)
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
            if dp + lam_b * dh + lam_c * (dca_min + dcb_min) + rho * dr > thr:
                continue
            dp2, dca, dcb, dh2 = move_delta(i, b, labels, ptr, idx, w, Lnode, Gnode, zL, zG,
                                            Ltot, floor, mark, owner, stamp, Q, qh, qt,
                                            parent, active, seeds)
            dE = dp2 + lam_c * (dca + dcb) + lam_b * dh2 + rho * dr
            if dE > thr:
                continue
            # apply
            rigidity_apply(a, b, A[i], cap[i], tn, tg, tnB, tgB)
            er += dr
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
            fdp, fdc, fdh, nacc = fragment_pass_ndbz(labels, k, ptr, idx, w, Lnode, Gnode, zL,
                                                     zG, cnt, Ltot, floor, lam_b, lam_c, T,
                                                     comp, stack, A, cap, rho, tn, tg, tnB,
                                                     tgB, fa, fn, fg)
            if nacc > 0:
                ep += fdp
                csum += fdc
                pb += fdh
                nb = _rebuild_boundary(labels, ptr, idx, diff, bnd, pos)
        # anchor-block moves once per temperature step (lambda_rigid > 0 only, so the
        # lambda_rigid = 0 path is the static annealer's, bit for bit)
        if rho != 0.0 and blocks:
            bdp, bdc, bdh, bacc = block_pass_ndbz(labels, k, ptr, idx, w, Lnode, Gnode, zL, zG,
                                                  cnt, Ltot, floor, lam_b, lam_c, T, comp,
                                                  stack, A, cap, rho, tn, tg, tnB, tgB, fa, fn,
                                                  fg, bcomp, inx, seen)
            if bacc > 0:
                ep += bdp
                csum += bdc
                pb += bdh
                nb = _rebuild_boundary(labels, ptr, idx, diff, bnd, pos)
            n_block += bacc
        er = rigidity_from_tables(tn, tg, tnB, tgB)    # re-sync (exact; O(k^2))
        acc_total += acc
        trace[ti, 0] = T
        trace[ti, 1] = lam_c
        trace[ti, 2] = ep + lam_c * csum + lam_b * pb + rho * er
        trace[ti, 3] = csum
        trace[ti, 4] = acc / max(nsteps, 1)
        trace[ti, 5] = er / Z
        trace[ti, 6] = n_block
        if capture:
            _capture(frames, frame_meta, nf, labels, T, lam_c, ep, csum, pb, er / Z,
                     acc / max(nsteps, 1))
            nf += 1
    return ep, csum, pb, er / Z, trace, nf


# --------------------------------------------------------------------------- #
# python-facing API
# --------------------------------------------------------------------------- #

def initial_temperature_ndbz(g: CountryGraph, labels, k, lam_b, floor, rng, A, cap, rho,
                             n_samples=400, target_accept=0.6):
    """``initial_temperature(..., exclude_contiguity=True)`` plus the rigidity term:
    the same sampled moves (same RNG use), dE = dPotts + lam_b dHinge + rho dRigid."""
    ws = g.ws
    zL = np.zeros(k)
    zG = np.zeros(k)
    np.add.at(zL, labels, g.Lnode)
    np.add.at(zG, labels, g.Gnode)
    tables = build_tables(np.asarray(labels, np.int64), A, cap, k, k)
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
        dE = dp + 0.0 * (dca + dcb) + lam_b * dh
        if rho != 0.0:
            dE += rho * rigidity_delta(int(labels[i]), b, int(A[i]), float(cap[i]), *tables)
        if dE > 0:
            ups.append(dE)
    if not ups:
        return 1.0
    return float(np.median(ups) / -math.log(target_accept))


def country_seed(c: str, base_seed: int, offset: int, restart: int) -> int:
    """Deterministic restart seed.  The static sweep used Python's ``hash(c)``, which
    is randomised per process; this uses CRC32 so a cell is reproducible.  Restart r
    of every nDBZ cell uses ``offset + r`` (``rigidity.seed_offset`` = 10000), distinct
    from the seeds of the run that produced A; the same seeds are reused across
    scenarios and lambda_rigid (common random numbers)."""
    return (base_seed * 1_000_003 + zlib.crc32(c.encode()) % 100_000 * 101 + offset + restart) % (2**31 - 1)


def n_frames(n_temps: int, quench_steps: int, quench_stride: int) -> int:
    """Frames written by ``_run_ndbz``: the initial state, one per temperature, one every
    ``quench_stride`` quench steps (strictly inside the quench), one at the end."""
    return 1 + n_temps + max(quench_steps - 1, 0) // quench_stride + 1


def anneal_ndbz(g: CountryGraph, k: int, A: np.ndarray, cap: np.ndarray, Z: float,
                lam_rigid: float, lam_b: float, floor: float, n_temps: int,
                sweeps_per_temp: float, t_final_ratio: float, seed: int,
                quench_sweeps: float = 20.0, mode: str = "ndbz", lam_c0: float | None = None,
                capture: bool = False, quench_frames: int = 20,
                T0_band: tuple = (1e-3, 1e3), init: np.ndarray | None = None,
                blocks: bool = True) -> dict:
    """Anneal one country.

    ``mode="ndbz"`` (the scenario runs): flat lambda_c at the contiguity guarantee,
    T0 on the physical terms.  ``mode="static"``: the static schedule (geometric
    lambda_c ramp from ``lam_c0``, T0 with lambda_c0 in the sampled dE); with
    ``lam_rigid = 0`` it reproduces ``bzgen.cluster.anneal.anneal`` exactly (a test
    fixture, not an experiment).
    """
    rng = np.random.default_rng(seed)
    lam_c1 = contiguity_guarantee(g, lam_b)
    labels = graph_voronoi(g, k, rng) if init is None else init.copy()
    if mode == "ndbz":
        T0 = initial_temperature_ndbz(g, labels, k, lam_b, floor, rng, np.asarray(A, np.int64),
                                      np.asarray(cap, np.float64), lam_rigid / Z)
        lam_c = np.full(n_temps, lam_c1)
        if not (np.isfinite(T0) and T0_band[0] < T0 < T0_band[1]):
            raise RuntimeError(f"T0 = {T0!r} outside the sane band {T0_band}: the schedule "
                               "would be meaningless (see bzgen/ndbz/anneal.py)")
    elif mode == "static":
        T0 = initial_temperature(g, labels, k, lam_b, lam_c0, floor, rng)
        lam_c = lam_c0 * (lam_c1 / lam_c0) ** (np.arange(n_temps) / max(n_temps - 1, 1))
    else:
        raise ValueError(mode)
    T = T0 * t_final_ratio ** (np.arange(n_temps) / max(n_temps - 1, 1))
    steps = int(sweeps_per_temp * g.n)
    qsteps = int(quench_sweeps * g.n)
    stride = max(1, qsteps // max(quench_frames, 1))
    nf_max = n_frames(n_temps, qsteps, stride) if capture else 0
    frames = np.zeros((nf_max, g.n), np.int32)
    meta = np.zeros((nf_max, len(META)))
    ws = g.ws
    ep, csum, pb, hr, trace, nf = _run_ndbz(
        labels, k, g.ptr, g.idx, g.w, g.Lnode, g.Gnode, g.Ltot, floor, lam_b, lam_c, T, steps,
        qsteps, seed, ws.mark, ws.owner, ws.stamp, ws.Q, ws.qh, ws.qt, ws.parent, ws.active,
        ws.seeds, np.asarray(A, np.int64), np.asarray(cap, np.float64), float(lam_rigid),
        float(Z), frames, meta, bool(capture), int(stride), bool(blocks))
    assert nf == nf_max
    return {"labels": labels, "potts": ep, "contig": csum, "balance": pb, "rigid": hr,
            "energy": ep + lam_c1 * csum + lam_b * pb + lam_rigid * hr,
            "physical": ep + lam_b * pb, "T0": T0, "trace": trace, "lam_c_final": lam_c1,
            "frames": frames, "frame_meta": meta, "quench_stride": stride}


def reference_quench(g: CountryGraph, k: int, A: np.ndarray, cap: np.ndarray, Z: float,
                     lam_rigid: float, lam_b: float, floor: float, quench_sweeps: float = 20.0,
                     seed: int = 0, blocks: bool = True) -> dict:
    """Diagnostic: a T = 0 quench from A itself under the cell's Hamiltonian.  A point
    near the anchored optimum; if it beats every restart, the annealer did not find
    the optimum of that cell (reported, never used as a result)."""
    lam_c1 = contiguity_guarantee(g, lam_b)
    labels = np.asarray(A, np.int64).copy()
    ws = g.ws
    ep, csum, pb, hr, trace, nf = _run_ndbz(
        labels, k, g.ptr, g.idx, g.w, g.Lnode, g.Gnode, g.Ltot, floor, lam_b,
        np.array([lam_c1]), np.array([0.0]), 0, int(quench_sweeps * g.n), seed, ws.mark,
        ws.owner, ws.stamp, ws.Q, ws.qh, ws.qt, ws.parent, ws.active, ws.seeds,
        np.asarray(A, np.int64), np.asarray(cap, np.float64), float(lam_rigid), float(Z),
        np.zeros((0, g.n), np.int32), np.zeros((0, len(META))), False, 1, bool(blocks))
    return {"labels": labels, "potts": ep, "contig": csum, "balance": pb, "rigid": hr,
            "energy": ep + lam_c1 * csum + lam_b * pb + lam_rigid * hr,
            "physical": ep + lam_b * pb}
