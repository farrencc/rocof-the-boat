"""Frozen cases v2: renewable + conventional nodes, SNSP cap.

Upstream's ``_build_frozen_cases`` does not return conventional dispatch, so
conventional units cannot be group members with it. This module rebuilds the
frozen cases in our own code, following the same dispatch sequence, and extends them:

1. Demand and fixed HVDC schedules per bus, conventional units at p_min, and
   renewables at synthetic potential (upstream's weather generator, same seed).
2. Deficit: conventional units are filled in merit order (cost asc, size desc);
   any remainder goes to the balance bus as "shortage".
   Surplus: renewables are cut pro-rata (pre-network dispatch-down); any
   remainder goes to the balance bus.
3. **SNSP cap** (new, optional): SNSP = (renewable output + HVDC import) /
   (demand + HVDC export). If it exceeds ``snsp_limit``, renewables are cut
   pro-rata until it equals the limit, and the MW is replaced by conventional
   headroom in merit order (then by the balance bus). The cut counts as
   pre-network dispatch-down.
4. PTDF (balance bus = largest conventional unit), frozen flows, and upstream's
   LODF and 90% N-1 screen.
5. **Nodes** = the 174 renewable farms, followed (optionally) by each conventional
   unit except the balance unit. A node's action is "reduce 1 MW, replaced by the
   slack". ``slack="balance"`` (upstream) puts the replacement on the balance bus:
   sensitivity ``H[:, balance] - H[:, bus]``. ``slack="distributed"`` [decision]
   shares it across every conventional unit in proportion to p_nom (a fixed
   participation vector a): sensitivity ``H[:, conv] @ a - H[:, bus]``. The
   distributed slack exists because single-bus replacement exceeded Great
   Island's 464.5 MW rating in ~45% of snapshots once conventional plant joined
   groups, and overloaded its exit lines. (A down-regulated unit keeps its own
   share a_j of the replacement; that is small and ignored.) For a conventional unit the "dispatch" fed to the
   relief loop is its DOWN-ROOM (dispatch - p_min), so pro-rata cuts stop at p_min.

With ``snsp_limit=None, include_conventional=False`` the arrays are identical
to upstream's ``_build_frozen_cases`` (tests/test_cases_v2.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from . import upstream as up
except ImportError:  # pragma: no cover
    import upstream as up

EPS = up.EPS
SNSP_LIMIT = 0.75


@dataclass(frozen=True)
class CasesV2:
    # Same fields the evaluator uses from upstream _FrozenCases (node axis = all nodes)
    renewable_potential_mw: np.ndarray       # snapshot x renewable node
    pre_network_dispatch_mw: np.ndarray      # snapshot x node (conventional: down-room)
    pre_network_dispatch_down_mw: np.ndarray  # snapshot x renewable node
    branch_flows_mw: np.ndarray
    case_weights: np.ndarray
    node_state_sensitivity: np.ndarray       # node x state
    state_base_flows_mw: np.ndarray
    state_limits_mw: np.ndarray
    state_monitor: np.ndarray
    state_outage: np.ndarray
    state_lodf_coeff: np.ndarray
    balance_bus: str
    shortage_mw: np.ndarray
    # Extensions
    node_kind: np.ndarray                    # 0 renewable, 1 conventional
    node_bus: np.ndarray
    conv_table: pd.DataFrame                 # conventional units that are nodes
    conv_dispatch_mw: np.ndarray             # snapshot x all conventional units (merit order)
    snsp: np.ndarray                         # snapshot, after the cap
    snsp_uncapped: np.ndarray
    snsp_curtail_mw: np.ndarray              # snapshot, renewable MW cut by the cap
    demand_mw: np.ndarray                    # snapshot total
    replacement_ptdf: np.ndarray             # branch: flow per MW of replacement injection
    slack: str


def conventional_fleet(grid) -> pd.DataFrame:
    gens = grid.extra.generators
    sel = (gens["bus"].astype(str).isin(set(grid.model.bus_ids))
           & ~gens["carrier"].str.lower().isin(["wind", "solar", "load shedding", "import", "export"])
           & (gens["p_nom_mw"] > 0))
    conv = gens[sel].sort_values(["marginal_cost", "p_nom_mw"], ascending=[True, False], kind="stable")
    conv = conv.reset_index(drop=True).copy()
    conv["p_min_mw"] = np.clip(conv["p_min_pu"].to_numpy(float), 0.0, 1.0) * conv["p_nom_mw"].to_numpy(float)
    station = grid.network.buses.set_index("bus")["station"].astype(str)
    conv["station"] = conv["bus"].astype(str).map(station)
    return conv


def build_cases_v2(grid, seed: int, thermal_scale: float = 1.0, runs: int = 10_000,
                   snsp_limit: float | None = SNSP_LIMIT, include_conventional: bool = True,
                   security_screen_threshold_pct: float = 90.0, max_security_states: int = 800,
                   include_n1: bool = True, slack: str = "distributed") -> CasesV2:
    model, network, extra, nodes = grid.model, grid.network, grid.extra, grid.template
    n_snap = len(network.load_profile)
    n_bus = len(model.bus_ids)
    bidx = model.bus_index
    ren_bus = np.array([bidx[b] for b in nodes["bus"].astype(str)])
    potential = up._synthetic_renewable_potential(nodes, n_snap, seed)

    demand = np.zeros((n_snap, n_bus))
    load_bus = network.loads.set_index("load")["bus"].astype(str).to_dict()
    for lid in network.load_profile.columns:
        b = load_bus.get(str(lid))
        if b in bidx:
            demand[:, bidx[b]] += network.load_profile[str(lid)].to_numpy(float)

    fixed = np.zeros((n_snap, n_bus))
    hvdc_import = np.zeros(n_snap)
    hvdc_export = np.zeros(n_snap)
    for lk in extra.links.itertuples(index=False):
        p = float(np.clip(lk.p_set_mw, -abs(lk.p_nom_mw), abs(lk.p_nom_mw)))
        b0, b1 = str(lk.bus0), str(lk.bus1)
        in0, in1 = b0 in bidx, b1 in bidx
        if in0:
            fixed[:, bidx[b0]] -= p
        if in1:
            fixed[:, bidx[b1]] += p * float(lk.efficiency)
        if in1 and not in0:          # boundary link flowing into scope
            v = p * float(lk.efficiency)
            hvdc_import += max(v, 0.0); hvdc_export += max(-v, 0.0)
        elif in0 and not in1:
            hvdc_export += max(p, 0.0); hvdc_import += max(-p, 0.0)

    conv = conventional_fleet(grid)
    pmax = conv["p_nom_mw"].to_numpy(float)
    pmin = conv["p_min_mw"].to_numpy(float)
    conv_bus = np.array([bidx[str(b)] for b in conv["bus"]])
    bal_row = int(np.argmax(pmax))
    balance_bus = str(conv.iloc[bal_row]["bus"])
    bal = bidx[balance_bus]

    inj = fixed - demand
    np.add.at(inj.T, ren_bus, potential.T)
    cd = np.repeat(pmin[None, :], n_snap, axis=0)
    np.add.at(inj.T, conv_bus, np.repeat(pmin[:, None], n_snap, axis=1))

    def fill(s, need):
        """Merit-order conventional uplift; returns MW not covered."""
        for j in range(len(pmax)):
            add = min(need, max(0.0, pmax[j] - cd[s, j]))
            if add > 0:
                inj[s, conv_bus[j]] += add
                cd[s, j] += add
                need -= add
            if need <= EPS:
                break
        return need

    pre_dd = np.zeros_like(potential)
    shortage = np.zeros(n_snap)
    for s in range(n_snap):
        residual = -float(inj[s].sum())
        if residual > EPS:
            left = fill(s, residual)
            if left > EPS:
                inj[s, bal] += left
                shortage[s] = left
        elif residual < -EPS:
            total = float(potential[s].sum())
            cut = min(-residual, total)
            if cut > EPS and total > EPS:
                pre_dd[s] = potential[s] * (cut / total)
                np.subtract.at(inj[s], ren_bus, pre_dd[s])
            imb = float(inj[s].sum())
            if abs(imb) > 1e-7:
                inj[s, bal] -= imb

    tot_dem = demand.sum(axis=1)
    ren_out = (potential - pre_dd).sum(axis=1)
    snsp_uncapped = (ren_out + hvdc_import) / np.maximum(tot_dem + hvdc_export, EPS)
    snsp_cut = np.zeros(n_snap)
    if snsp_limit is not None:
        for s in range(n_snap):
            allowed = snsp_limit * (tot_dem[s] + hvdc_export[s]) - hvdc_import[s]
            excess = ren_out[s] - max(allowed, 0.0)
            if excess > EPS:
                avail = potential[s] - pre_dd[s]
                frac = excess / max(avail.sum(), EPS)
                cut = avail * frac
                pre_dd[s] += cut
                np.subtract.at(inj[s], ren_bus, cut)
                left = fill(s, excess)
                if left > EPS:
                    inj[s, bal] += left
                snsp_cut[s] = excess
    pre_dispatch = np.maximum(0.0, potential - pre_dd)
    snsp = (pre_dispatch.sum(axis=1) + hvdc_import) / np.maximum(tot_dem + hvdc_export, EPS)

    H = model.ptdf(balance_bus)
    flows = inj @ H.T
    limits = model.branches["s_nom_mva"].to_numpy(float) * float(thermal_scale)
    lodf, valid = up._build_lodf(model, H)
    mon, out, coeff, state_limits, _ = up._screen_security_states(
        flows, limits, lodf, valid, threshold=security_screen_threshold_pct / 100.0,
        max_states=int(max_security_states), include_n1=bool(include_n1))
    state_base = flows[:, mon].copy()
    has_out = out >= 0
    if has_out.any():
        state_base[:, has_out] += flows[:, out[has_out]] * coeff[has_out][None, :]

    node_bus_idx = list(ren_bus)
    kinds = [0] * len(ren_bus)
    dispatch_cols = [pre_dispatch]
    conv_nodes = conv.drop(index=bal_row).copy() if include_conventional else conv.iloc[0:0].copy()
    if include_conventional:
        keep = [j for j in range(len(conv)) if j != bal_row]
        node_bus_idx += [conv_bus[j] for j in keep]
        kinds += [1] * len(keep)
        dispatch_cols.append(np.maximum(0.0, cd[:, keep] - pmin[keep][None, :]))
    node_bus_idx = np.array(node_bus_idx)
    if slack == "balance":
        repl = H[:, bal]
    elif slack == "distributed":
        part = pmax / pmax.sum()
        repl = H[:, conv_bus] @ part
    else:
        raise ValueError(slack)
    bns = repl[None, :] - H[:, node_bus_idx].T              # node x branch
    nss = bns[:, mon].copy()
    if has_out.any():
        nss[:, has_out] += bns[:, out[has_out]] * coeff[has_out][None, :]

    rng = np.random.default_rng(int(seed))
    sampled = rng.integers(0, n_snap, size=max(1, int(runs)))
    counts = np.bincount(sampled, minlength=n_snap).astype(float)

    return CasesV2(
        renewable_potential_mw=potential, pre_network_dispatch_mw=np.hstack(dispatch_cols),
        pre_network_dispatch_down_mw=pre_dd, branch_flows_mw=flows, case_weights=counts / counts.sum(),
        node_state_sensitivity=nss, state_base_flows_mw=state_base, state_limits_mw=state_limits,
        state_monitor=mon, state_outage=out, state_lodf_coeff=coeff, balance_bus=balance_bus,
        shortage_mw=shortage, node_kind=np.array(kinds), node_bus=np.array(model.bus_ids)[node_bus_idx],
        conv_table=conv_nodes.reset_index(drop=True), conv_dispatch_mw=cd, snsp=snsp,
        snsp_uncapped=snsp_uncapped, snsp_curtail_mw=snsp_cut, demand_mw=tot_dem,
        replacement_ptdf=repl, slack=slack,
    )


def node_table(grid, cases: CasesV2) -> pd.DataFrame:
    """Unified node table: renewable farms then conventional units, with a 'capacity' column
    (MEC for farms, down-room p_nom - p_min for conventional units) and coordinates."""
    t = grid.template
    ren = pd.DataFrame(dict(name=t["name"], bus=t["bus"].astype(str), kind="renewable",
                            technology=t["technology"], capacity_mw=t["mec_mw"].astype(float),
                            latitude=t["latitude"].astype(float), longitude=t["longitude"].astype(float)))
    buses = grid.network.buses.set_index("bus")
    c = cases.conv_table
    conv = pd.DataFrame(dict(name=c["station"] + " (" + c["generator"].astype(str) + ")", bus=c["bus"].astype(str),
                             kind="conventional", technology=c["carrier"],
                             capacity_mw=(c["p_nom_mw"] - c["p_min_mw"]).astype(float),
                             latitude=[float(buses.loc[b, "lat"]) for b in c["bus"].astype(str)],
                             longitude=[float(buses.loc[b, "lon"]) for b in c["bus"].astype(str)]))
    out = pd.concat([ren, conv], ignore_index=True)
    out.insert(0, "node", np.arange(len(out)))
    return out
