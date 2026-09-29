"""Dispatch-down evaluator for an N x K binary (possibly overlapping) membership.

A reimplementation of the ``evaluate_assignment`` closure inside upstream's
``make_emulator``; the only change is that group membership is an N x K boolean
matrix instead of one integer label per node. Everything else mirrors upstream:

  per weighted snapshot, up to ``max_group_actions`` times:
    loading = |state_flow| / state_limits; worst = argmax; stop if <= 1 (+1e-8)
    for each group g (in column order) with available = sum(dispatch[members]) > EPS:
        w   = dispatch[members] / available
        S_g = w @ node_state_sensitivity[members, worst]
        req = _required_reduction(state_flow[worst], state_limits[worst], S_g)
    pick the smallest finite req (upstream's sequential scan: a later group wins
    only if strictly smaller by >1e-12, so exact ties go to the lowest index)
    delta = min(available, req * 1.000001 + 1e-7)
    state_flow += delta * (w @ node_state_sensitivity[members, :])
    dispatch[members] -= delta * w; clip at 0
  total_dd = pre_dd_weighted + sum(weights * network_dd); pct = 100 * total_dd / potential

Overlap needs no special handling. ``w`` is recomputed from the CURRENT
dispatch on every action, so a node in two groups that are both called is cut
twice, the second time pro-rata off what the first cut left. It can never go
below zero: the group's own ``delta <= available`` bound means each member's cut
``delta * w_i <= dispatch_i``, and the clip at zero catches float round-off,
exactly as upstream does.

The inner loop is compiled with numba. It also records which screened state is
binding (the argmax at the top of each relief iteration while loading > 1) and
the node-level network curtailment, which the anchor analysis and the
full-network recheck need.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

try:  # the package is imported as ``src.*`` by tests and as top-level by scripts
    from . import upstream as up
except ImportError:  # pragma: no cover
    import upstream as up

EPS = up.EPS


@njit(cache=True)
def _required_reduction_nb(flow, limit, sensitivity):
    # Same branches and tolerances as upstream all_island_annealer_api._required_reduction.
    if flow > limit + 1e-9:
        if sensitivity >= -1e-12:
            return math.inf
        return (flow - limit) / (-sensitivity)
    if flow < -limit - 1e-9:
        if sensitivity <= 1e-12:
            return math.inf
        return (-limit - flow) / sensitivity
    return 0.0


@njit(cache=True)
def _evaluate_kernel(member, weights, state_base, pre_dispatch, nss, limits, max_actions, kind):
    """member: N x K bool. Returns per-snapshot results plus binding diagnostics."""
    n_snap = weights.shape[0]
    n_node, n_group = member.shape
    n_state = limits.shape[0]
    network_dd = np.zeros(n_snap)
    conv_mw = np.zeros(n_snap)  # conventional down-regulation (kind==1), not dispatch-down
    secure = np.ones(n_snap, dtype=np.bool_)
    worst_loading = np.zeros(n_snap)
    actions = np.zeros(n_snap, dtype=np.int64)
    # binding[j, 0/1]: case-weighted count of relief iterations whose top-of-iteration
    # worst state was j with positive (0) / negative (1) flow and loading > 1. This
    # includes the final iteration in which no group could relieve j (loop breaks).
    binding = np.zeros((n_state, 2))
    # relief[j, g]: case-weighted MW curtailed by group g while j was the worst state.
    relief = np.zeros((n_state, n_group))
    curtail = np.zeros((n_snap, n_node))

    # Precompute member index lists (CSR style) so iteration order is node order.
    counts = np.zeros(n_group, dtype=np.int64)
    for g in range(n_group):
        for i in range(n_node):
            if member[i, g]:
                counts[g] += 1
    ptr = np.zeros(n_group + 1, dtype=np.int64)
    for g in range(n_group):
        ptr[g + 1] = ptr[g] + counts[g]
    idx_all = np.empty(ptr[n_group], dtype=np.int64)
    for g in range(n_group):
        c = ptr[g]
        for i in range(n_node):
            if member[i, g]:
                idx_all[c] = i
                c += 1

    for s in range(n_snap):
        if weights[s] <= 0:
            continue
        flow = state_base[s].copy()
        dispatch = pre_dispatch[s].copy()
        used = 0
        for _it in range(max_actions):
            worst = 0
            worst_ratio = -1.0
            for j in range(n_state):
                r = abs(flow[j]) / limits[j]
                if r > worst_ratio:  # strict '>' == np.argmax first-max semantics
                    worst_ratio = r
                    worst = j
            if worst_ratio <= 1.0 + 1e-8:
                break
            if flow[worst] >= 0:
                binding[worst, 0] += weights[s]
            else:
                binding[worst, 1] += weights[s]

            best_g = -1
            best_req = 0.0
            best_avail = 0.0
            for g in range(n_group):
                a, b = ptr[g], ptr[g + 1]
                if a == b:
                    continue
                available = 0.0
                for q in range(a, b):
                    available += dispatch[idx_all[q]]
                if available <= EPS:
                    continue
                sens = 0.0
                for q in range(a, b):
                    i = idx_all[q]
                    sens += (dispatch[i] / available) * nss[i, worst]
                req = _required_reduction_nb(flow[worst], limits[worst], sens)
                if not np.isfinite(req):
                    continue
                if best_g < 0 or req < best_req - 1e-12:
                    best_g = g
                    best_req = req
                    best_avail = available
                # abs(req-best_req) <= 1e-12 with a higher g never replaces: groups are
                # scanned in increasing index, so ties keep the lowest index (upstream rule).

            if best_g < 0:
                break
            delta = min(best_avail, best_req * 1.000001 + 1e-7)
            if delta <= EPS:
                break
            a, b = ptr[best_g], ptr[best_g + 1]
            # group_state_sens = w @ nss[members, :], then flow += delta * it (upstream order)
            gs = np.zeros(n_state)
            for q in range(a, b):
                i = idx_all[q]
                wi = dispatch[i] / best_avail
                for j in range(n_state):
                    gs[j] += wi * nss[i, j]
            for j in range(n_state):
                flow[j] += delta * gs[j]
            for q in range(a, b):
                i = idx_all[q]
                wi = dispatch[i] / best_avail
                new = dispatch[i] - delta * wi
                if new < 0.0:
                    new = 0.0
                curtail[s, i] += dispatch[i] - new
                if kind[i] == 0:
                    network_dd[s] += dispatch[i] - new
                else:
                    conv_mw[s] += dispatch[i] - new
                dispatch[i] = new
            relief[worst, best_g] += weights[s] * delta
            used += 1

        wl = 0.0
        for j in range(n_state):
            r = abs(flow[j]) / limits[j]
            if r > wl:
                wl = r
        worst_loading[s] = wl
        secure[s] = wl <= 1.0 + 1e-6
        actions[s] = used
    return network_dd, secure, worst_loading, actions, binding, relief, curtail, conv_mw


@dataclass
class EvalResult:
    pct: float                    # dispatch-down % of weighted renewable potential
    secure: np.ndarray            # snapshot -> screened-secure after relief
    worst_loading: np.ndarray     # snapshot -> max screened loading after relief
    network_dd_mw: np.ndarray     # snapshot -> network (constraint) curtailment MW
    actions: np.ndarray           # snapshot -> number of group actions used
    binding: np.ndarray           # state x {pos, neg}: case-weighted binding-iteration counts
    relief_mw: np.ndarray         # state x group: case-weighted MW curtailed while state was worst
    curtail_mw: np.ndarray        # snapshot x node network curtailment MW
    conv_mw: np.ndarray | None = None  # snapshot -> conventional down-regulation MW
    conv_pct: float = 0.0         # weighted conventional redispatch, % of renewable potential


class Evaluator:
    """Frozen evaluator: ``Evaluator(cases)(M)`` -> EvalResult for membership M."""

    def __init__(self, cases, max_group_actions: int = 24):
        self.cases = cases
        self.max_group_actions = int(max_group_actions)
        self.weights = np.ascontiguousarray(cases.case_weights, dtype=float)
        self.potential_weighted = float(np.sum(self.weights * cases.renewable_potential_mw.sum(axis=1)))
        self.pre_dd_weighted = float(np.sum(self.weights * cases.pre_network_dispatch_down_mw.sum(axis=1)))
        self._state_base = np.ascontiguousarray(cases.state_base_flows_mw, dtype=float)
        self._pre_dispatch = np.ascontiguousarray(cases.pre_network_dispatch_mw, dtype=float)
        self._nss = np.ascontiguousarray(cases.node_state_sensitivity, dtype=float)
        self._limits = np.ascontiguousarray(cases.state_limits_mw, dtype=float)
        self.n_node = self._nss.shape[0]
        self._kind = np.ascontiguousarray(getattr(cases, "node_kind", np.zeros(self.n_node, np.int64)), dtype=np.int64)
        self.calls = 0

    def __call__(self, membership: np.ndarray) -> EvalResult:
        M = np.ascontiguousarray(membership, dtype=np.bool_)
        if M.ndim != 2 or M.shape[0] != self.n_node:
            raise ValueError(f"membership must be {self.n_node} x K, got {M.shape}")
        self.calls += 1
        net_dd, secure, worst, actions, binding, relief, curtail, conv = _evaluate_kernel(
            M, self.weights, self._state_base, self._pre_dispatch, self._nss, self._limits,
            self.max_group_actions, self._kind)
        total_dd = self.pre_dd_weighted + float(np.sum(self.weights * net_dd))
        pct = 100.0 * total_dd / max(self.potential_weighted, EPS)
        conv_pct = 100.0 * float(np.sum(self.weights * conv)) / max(self.potential_weighted, EPS)
        return EvalResult(float(pct), secure, worst, net_dd, actions, binding, relief, curtail, conv, conv_pct)

    def new_failures(self, result: EvalResult, baseline_secure: np.ndarray) -> int:
        """Snapshots secure at baseline but insecure now (upstream security_guard)."""
        return int(np.sum(baseline_secure & ~result.secure & (self.weights > 0)))


def labels_to_membership(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exclusive integer labels -> N x K bool, columns in ascending label order
    (the order upstream's ``np.unique`` scan uses)."""
    labels = np.asarray(labels, dtype=int)
    uniq = np.unique(labels)
    M = labels[:, None] == uniq[None, :]
    return M, uniq


def wdt_membership(template) -> tuple[np.ndarray, np.ndarray]:
    """True multi-membership WDT groups from the CSV ``wdt_groups`` column."""
    import ast
    sets = [tuple(int(x) for x in ast.literal_eval(str(v))) for v in template["wdt_groups"]]
    uniq = np.array(sorted({g for t in sets for g in t}), dtype=int)
    col = {g: k for k, g in enumerate(uniq)}
    M = np.zeros((len(sets), len(uniq)), dtype=bool)
    for i, t in enumerate(sets):
        for g in t:
            M[i, col[g]] = True
    return M, uniq


@dataclass
class Member:
    """One frozen case of an ensemble, with its baseline-security reference."""
    seed: int
    thermal_scale: float
    cases: object
    evaluator: Evaluator
    baseline_secure: np.ndarray | None = None
    baseline_pct: float | None = None


class EnsembleEvaluator:
    """Evaluate one membership over several frozen cases.

    D(G) is the unweighted mean of the members' dispatch-down %; the upstream
    security guard is applied per member against that member's baseline
    (a new insecure snapshot in ANY member makes the candidate infeasible).
    """

    def __init__(self, members: list[Member]):
        self.members = members

    @property
    def calls(self) -> int:
        return self.members[0].evaluator.calls if self.members else 0

    def set_baseline(self, membership: np.ndarray) -> None:
        for m in self.members:
            r = m.evaluator(membership)
            m.baseline_secure = r.secure.copy()
            m.baseline_pct = r.pct

    def __call__(self, membership: np.ndarray, keep_results: bool = False) -> dict:
        pcts, fails, sec, results, conv = [], [], [], [], []
        for m in self.members:
            r = m.evaluator(membership)
            pcts.append(r.pct)
            conv.append(r.conv_pct)
            fails.append(m.evaluator.new_failures(r, m.baseline_secure) if m.baseline_secure is not None else 0)
            sec.append(100.0 * float(np.sum(m.evaluator.weights * r.secure)))
            if keep_results:
                results.append(r)
        out = dict(D=float(np.mean(pcts)), D_members=np.array(pcts), C=float(np.mean(conv)), new_failures=int(np.sum(fails)),
                   new_failures_members=np.array(fails), security_pct=float(np.mean(sec)),
                   security_members=np.array(sec))
        out["feasible"] = out["new_failures"] == 0
        if keep_results:
            out["results"] = results
        return out


def build_ensemble(grid, seeds, thermal_scales, runs: int = 10_000, builder=None) -> EnsembleEvaluator:
    """builder(grid, seed, thermal_scale) -> cases; default is upstream's _build_frozen_cases."""
    members = []
    for seed in seeds:
        for ts in thermal_scales:
            cases = (builder(grid, seed, ts) if builder is not None
                     else up.build_cases(grid, seed=seed, thermal_scale=ts, runs=runs))
            members.append(Member(int(seed), float(ts), cases, Evaluator(cases)))
    return EnsembleEvaluator(members)
