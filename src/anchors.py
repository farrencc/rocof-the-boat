"""Step 2: choose anchor branches from binding frequency across a high-risk ensemble.

Pipeline (decisions agreed with the project owner are marked [decision]):

1. Ensemble: ANCHOR_SEEDS x THERMAL_SCALES frozen cases (seed drives both the
   synthetic weather and the case-weight sampling upstream).
2. Run the baseline grouping [decision: true multi-membership WDT ``wdt_groups``]
   through the evaluator. A screened (monitor, outage) state "binds" in a relief
   iteration when it is the argmax loading at the top of the iteration and that
   loading is > 1. Counts are case-weighted and summed over the ensemble.
3. Relievability screen [decision]: a monitored branch is dropped if, in the
   majority of its (case-weighted) overload events, curtailing EVERY helpful-sign
   renewable node in full could not bring it inside its limit. Such a branch
   (e.g. 458 Poolbeg S - Carrickmines PST under the Inchicore-Irishtown outage)
   dominates the raw binding counts because upstream's greedy loop stops on it,
   but no grouping can relieve it, so it cannot anchor a group.
4. p_e = binding counts per MONITORED branch (summed over its outage variants and
   both directions), normalised to sum to 1 over the relievable branches.
   K = min k with the top-k cumulative p_e >= threshold [decision: 0.95; the
   original 0.5 gives K=1 because one line dominates]. K for 0.3..0.95 is reported.
5. Per anchor: dominant overload direction s_k from binding events; flag if the
   minority direction exceeds 20%. Oriented sensitivity [decision: binding-
   weighted mean over the anchor's outage variants]:
       sigma_ik = -s_k * sum_v b_v nss[i, v] / sum_v b_v
   so sigma > 0 means "curtailing node i helps relieve anchor k". Sign
   consistency across variants is reported.
6. Initialisation: group k = nodes with sigma_ik > 0 AND |sigma_ik| above the
   75th percentile of |sigma_.k| over all nodes (shift-factor clustering).
7. Guard anchors [decision]: if the initial grouping violates the security guard
   (a snapshot secure under the baseline becomes insecure), find the monitored
   branches that are the worst post-relief state in those snapshots. Any such
   branch that step 3 excluded as unrelievable is added back as an extra anchor,
   seeded with ALL its helpful-sign nodes. (26-county: 458 Poolbeg S - CKMN is
   unrelievable in general, but in marginal snapshots (90-103% loading) the
   baseline does relieve it, and the three p_e anchors' groups are 100% wrong-sign
   for it. A dedicated 458 group restores feasibility.) Guard anchors do not
   count toward the p_e/K rule.

node_state_sensitivity depends only on topology and the balance bus, so a given
(monitor, outage) key has identical sensitivities in every ensemble member
(checked in tests); only screen membership and flows vary per case.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import upstream as up
    from .evaluator import EnsembleEvaluator, build_ensemble, wdt_membership
except ImportError:  # pragma: no cover
    import upstream as up
    from evaluator import EnsembleEvaluator, build_ensemble, wdt_membership

ANCHOR_SEEDS = (11, 23, 42, 76, 101)
THERMAL_SCALES = (1.0, 0.95)
VALIDATION_SEEDS = (2001, 2002, 2003, 2004, 2005)
K_THRESHOLD = 0.95
RELIEVABLE_MIN_FRACTION = 0.5
BIDIRECTIONAL_FLAG = 0.20
INIT_PERCENTILE = 75.0
REPORT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


@dataclass
class AnchorSet:
    scope: str
    branches: list[int]                 # monitored branch index e_k
    orientation: list[int]              # s_k (+1: overloads positive)
    sigma: np.ndarray                   # N x K oriented sensitivity
    init_membership: np.ndarray         # N x K bool
    table: pd.DataFrame                 # per-branch stats (all monitored branches)
    variants: pd.DataFrame              # per (monitor, outage) stats
    anchor_report: list[dict] = field(default_factory=list)
    n_pe_anchors: int = 0               # anchors chosen by the p_e rule; the rest are guard anchors
    guard_log: list[dict] = field(default_factory=list)
    k_by_threshold: dict = field(default_factory=dict)
    k_by_threshold_literal: dict = field(default_factory=dict)


def _k_for(p_sorted: np.ndarray, thr: float) -> int:
    cum = np.cumsum(p_sorted)
    return int(np.argmax(cum >= thr - 1e-12) + 1) if len(cum) else 0


def collect_state_stats(ens: EnsembleEvaluator, membership: np.ndarray):
    """Per (monitor, outage) key: binding counts, pre-relief overloads, ideal relievability."""
    rows = {}
    nss_by_key = {}
    for mem in ens.members:
        c = mem.cases
        r = mem.evaluator(membership)
        w = c.case_weights
        for j, (m, o) in enumerate(zip(c.state_monitor.tolist(), c.state_outage.tolist())):
            key = (int(m), int(o))
            nss = c.node_state_sensitivity[:, j]
            nss_by_key.setdefault(key, nss.copy())
            f = c.state_base_flows_mw[:, j]
            L = c.state_limits_mw[j]
            exc = np.abs(f) - L
            over = exc > 0
            # Ideal group for this state and snapshot: every node whose curtailment moves
            # the flow toward zero, curtailed in full (DC linear, so relief adds up).
            help_mw = pre = 0.0
            relievable = 0.0
            if over.any():
                sgn = np.sign(f[over])
                disp = c.pre_network_dispatch_mw[over]
                max_rel = np.sum(disp * np.maximum(0.0, -nss[None, :] * sgn[:, None]), axis=1)
                relievable = float(np.sum(w[over] * (max_rel >= exc[over])))
            d = rows.setdefault(key, dict(monitor=key[0], outage=key[1], bind_pos=0.0, bind_neg=0.0,
                                          over_w=0.0, over_pos_w=0.0, relievable_w=0.0, n_members=0))
            d["bind_pos"] += float(r.binding[j, 0])
            d["bind_neg"] += float(r.binding[j, 1])
            d["over_w"] += float(np.sum(w[over]))
            d["over_pos_w"] += float(np.sum(w[over & (f > 0)]))
            d["relievable_w"] += relievable
            d["n_members"] += 1
    var = pd.DataFrame(rows.values())
    var["bind"] = var.bind_pos + var.bind_neg
    return var, nss_by_key


def select_anchors(grid, ens: EnsembleEvaluator | None = None, baseline: np.ndarray | None = None,
                   threshold: float = K_THRESHOLD, add_guard_anchors: bool = True) -> AnchorSet:
    if baseline is None:
        baseline, _ = wdt_membership(grid.template)
    if ens is None:
        ens = build_ensemble(grid, ANCHOR_SEEDS, THERMAL_SCALES)
        ens.set_baseline(baseline)
    var, nss_by_key = collect_state_stats(ens, baseline)

    br = grid.model.branches.reset_index(drop=True)
    station = grid.network.buses.set_index("bus")["station"].astype(str)
    tab = var.groupby("monitor")[["bind_pos", "bind_neg", "bind", "over_w", "over_pos_w", "relievable_w"]].sum()
    tab["n_variants"] = var.groupby("monitor").size()
    tab["relievable_frac"] = np.where(tab.over_w > 0, tab.relievable_w / tab.over_w.where(tab.over_w > 0, 1), 1.0)
    tab["relievable"] = tab.relievable_frac >= RELIEVABLE_MIN_FRACTION
    tab["p_literal"] = tab.bind / tab.bind.sum()
    rel_bind = tab.bind.where(tab.relievable, 0.0)
    tab["p_e"] = rel_bind / rel_bind.sum()
    tab["frac_pos_binding"] = np.where(tab.bind > 0, tab.bind_pos / tab.bind.where(tab.bind > 0, 1), np.nan)
    tab["frac_pos_overload"] = np.where(tab.over_w > 0, tab.over_pos_w / tab.over_w.where(tab.over_w > 0, 1), np.nan)
    tab["name"] = [f"{station.get(br.loc[m, 'bus0'], br.loc[m, 'bus0'])} - {station.get(br.loc[m, 'bus1'], br.loc[m, 'bus1'])}"
                   for m in tab.index]
    tab["branch_id"] = [br.loc[m, "branch"] for m in tab.index]
    tab["s_nom_mva"] = [float(br.loc[m, "s_nom_mva"]) for m in tab.index]
    tab = tab.sort_values(["p_e", "bind"], ascending=False)

    p_sorted = tab.p_e.to_numpy()
    K = _k_for(p_sorted, threshold)
    anchors = [int(m) for m in tab.index[:K]]
    lit_sorted = np.sort(tab.p_literal.to_numpy())[::-1]

    N = len(grid.template)
    cols = [_anchor_column(grid, tab, var, nss_by_key, m, k, INIT_PERCENTILE) for k, m in enumerate(anchors)]
    a = AnchorSet(
        scope=grid.scope, branches=anchors, orientation=[c[3]["orientation"] for c in cols],
        sigma=np.column_stack([c[0] for c in cols]) if cols else np.zeros((N, 0)),
        init_membership=np.column_stack([c[1] for c in cols]) if cols else np.zeros((N, 0), bool),
        table=tab, variants=var, anchor_report=[c[3] for c in cols],
        k_by_threshold={t: _k_for(p_sorted, t) for t in REPORT_THRESHOLDS},
        k_by_threshold_literal={t: _k_for(lit_sorted, t) for t in REPORT_THRESHOLDS},
        n_pe_anchors=K,
    )
    if add_guard_anchors:
        _add_guard_anchors(grid, ens, a, nss_by_key)
    return a


def _worst_failing_monitors(ens: EnsembleEvaluator, M: np.ndarray) -> dict[int, int]:
    """Monitored branch that is the worst post-relief state in each new-failure snapshot."""
    out: dict[int, int] = {}
    ev = ens(M, keep_results=True)
    for mem, r in zip(ens.members, ev["results"]):
        c = mem.cases
        post = c.state_base_flows_mw + r.curtail_mw @ c.node_state_sensitivity
        for s in np.flatnonzero(mem.baseline_secure & ~r.secure):
            j = int(np.argmax(np.abs(post[s]) / c.state_limits_mw))
            m = int(c.state_monitor[j])
            out[m] = out.get(m, 0) + 1
    return out


def _add_guard_anchors(grid, ens, a: AnchorSet, nss_by_key, max_rounds: int = 3) -> None:
    for _ in range(max_rounds):
        fails = _worst_failing_monitors(ens, a.init_membership)
        entry = dict(branches=list(a.branches), new_failures_by_worst_monitor={str(k): v for k, v in fails.items()})
        a.guard_log.append(entry)
        cand = [m for m, _ in sorted(fails.items(), key=lambda kv: -kv[1])
                if m not in a.branches and not bool(a.table.loc[m, "relievable"])]
        if not fails or not cand:
            break
        for m in cand:
            k = len(a.branches)
            sig, init, _, rep = _anchor_column(grid, a.table, a.variants, nss_by_key, m, k, None)
            rep["guard_anchor"] = True
            rep["guard_trigger_failures"] = int(fails[m])
            a.branches.append(m); a.orientation.append(rep["orientation"]); a.anchor_report.append(rep)
            a.sigma = np.column_stack([a.sigma, sig])
            a.init_membership = np.column_stack([a.init_membership, init])
    a.guard_log.append(dict(branches=list(a.branches), final_new_failures=int(ens(a.init_membership)["new_failures"])))


def _anchor_column(grid, tab, var, nss_by_key, m: int, k: int, init_percentile):
    """sigma column, init column, orientation and report for anchor branch m.

    init_percentile=None seeds the group with ALL helpful-sign nodes (guard anchors).
    """
    row = tab.loc[m]
    s_k = 1 if row.bind_pos >= row.bind_neg else -1
    minority = float(min(row.bind_pos, row.bind_neg) / row.bind) if row.bind > 0 else 0.0
    v = var[var.monitor == m].sort_values("bind", ascending=False)
    wts = v.bind.to_numpy()
    keys = [(int(a_), int(b_)) for a_, b_ in zip(v.monitor, v.outage)]
    mats = np.stack([nss_by_key[kk] for kk in keys], axis=1)  # N x variants
    mean_nss = mats @ wts / wts.sum() if wts.sum() > 0 else mats.mean(axis=1)
    sigma = -s_k * mean_nss
    ref = np.sign(mean_nss)
    nz = np.abs(mean_nss) > 1e-6
    agree = [(float(np.mean(np.sign(mats[nz, q]) == ref[nz])), float(wts[q])) for q in range(mats.shape[1])]
    all_consistent = float(np.mean(np.all(np.sign(mats[nz]) == ref[nz, None], axis=1)))
    corr_min = float(np.min([np.corrcoef(mats[:, q], mean_nss)[0, 1] for q in range(mats.shape[1])]))
    mag = np.abs(sigma)
    if init_percentile is None:
        init = sigma > 0
    else:
        init = (sigma > 0) & (mag > np.percentile(mag, init_percentile))
    mec = grid.template.mec_mw.to_numpy()
    rep = dict(
        k=k, monitor=int(m), branch_id=row.branch_id, name=row["name"], s_nom_mva=float(row.s_nom_mva),
        p_e=float(row.p_e), p_literal=float(row.p_literal), relievable_frac=float(row.relievable_frac),
        orientation=s_k, frac_pos_binding=float(row.frac_pos_binding), frac_pos_overload=float(row.frac_pos_overload),
        minority_direction_frac=minority, bidirectional_flag=bool(minority > BIDIRECTIONAL_FLAG),
        n_variants=int(len(keys)), n_binding_variants=int(np.sum(wts > 0)),
        top_variant_outage=int(keys[0][1]), top_variant_share=float(wts[0] / wts.sum()) if wts.sum() > 0 else 0.0,
        sign_agreement_min=float(min(a_ for a_, _ in agree)),
        sign_agreement_binding_weighted=float(sum(a_ * b_ for a_, b_ in agree) / max(sum(b_ for _, b_ in agree), 1e-12)),
        nodes_sign_consistent_all_variants=all_consistent, corr_min_variant_vs_mean=corr_min,
        n_helpful=int(np.sum(sigma > 0)), n_init=int(init.sum()), init_mec_mw=float(mec[init].sum()),
        guard_anchor=False,
    )
    return sigma, init, s_k, rep


def save_anchors(a: AnchorSet, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    a.table.to_csv(out_dir / "branch_binding.csv")
    a.variants.sort_values("bind", ascending=False).to_csv(out_dir / "state_binding.csv", index=False)
    np.savez(out_dir / "anchors.npz", branches=np.array(a.branches), orientation=np.array(a.orientation),
             sigma=a.sigma, init_membership=a.init_membership)
    meta = dict(scope=a.scope, branches=a.branches, orientation=a.orientation,
                threshold=K_THRESHOLD, relievable_min_fraction=RELIEVABLE_MIN_FRACTION,
                anchor_seeds=list(ANCHOR_SEEDS), thermal_scales=list(THERMAL_SCALES),
                k_by_threshold={str(k): v for k, v in a.k_by_threshold.items()},
                k_by_threshold_literal={str(k): v for k, v in a.k_by_threshold_literal.items()},
                n_pe_anchors=a.n_pe_anchors, guard_log=a.guard_log, anchors=a.anchor_report)
    (out_dir / "anchors.json").write_text(json.dumps(meta, indent=2, default=float))


def load_anchors(out_dir: Path) -> dict:
    z = np.load(out_dir / "anchors.npz")
    meta = json.loads((out_dir / "anchors.json").read_text())
    meta.update(sigma=z["sigma"], init_membership=z["init_membership"].astype(bool))
    return meta
