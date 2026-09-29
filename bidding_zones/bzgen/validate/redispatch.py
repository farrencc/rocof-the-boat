"""Stage B: nodal redispatch of the zonal market schedule, net positions fixed.

Full nodal European network, identical to the nodal benchmark
(``bzgen.solve.opf.build_network``: KVL on, flows within ``s_nom * s_max_pu``,
HVDC links free within ``+-p_nom``).  The market schedule ``p_zonal`` enters as
a fixed injection; every generator (load shedding included, at the same cost as
everywhere else) becomes two redispatch units:

* up:   0 <= u_g <= p_nom p_max_pu(t) - p_zonal_g,   cost  (mc_g + markup) u_g
* down: -p_zonal_g <= d_g <= 0,                      cost  (mc_g - markup) d_g

The down unit follows PyPSA's sign convention (``p_min_pu = -1``,
``p_max_pu = 0``): backing a unit down by |d| saves ``mc |d|`` and costs
``markup |d|``, i.e. ``(-mc + markup)`` per MWh backed down, which is the
brief's ``marginal_cost = -mc + markup`` expressed per MWh of down-regulation.

Net positions are fixed: for every zone z and hour, ``sum_{g in z} (u_g + d_g) = 0``
(sum of up equals sum of down), so each zone's net injection equals its market
net position.  No counter-trading variant.

**Soft net positions (deviation from the brief, for a quick first result).**
With hard equalities most hours proved LP-infeasible: the zonal market
schedules exchanges that the meshed grid cannot deliver, even with load
shedding.  Each net-position row therefore carries two slack columns
(``s+ - s-``) at ``validate.np_slack_penalty`` EUR/MWh, above every generator
cost and below the load-shedding cost: net positions hold wherever redispatch
can deliver them, and the undeliverable part is reported as ``np_slack`` (MWh),
never hidden.  Setting the penalty to ``null`` restores the hard equalities.

Production solves use ``bzgen.solve.lp.DCOPF`` (highspy, warm-started; the
nodal benchmark's solver) because ``n.optimize()`` costs ~40 s per hour outside
the solver on this network (PLAN.md changelog).  :func:`redispatch_pypsa` is
the same model built with PyPSA/linopy, the net-position rows added through
``n.model.add_constraints``; it cross-checks the objective on sample hours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bzgen.solve.lp import DCOPF


def split_network(n, cfg: dict):
    """Static PyPSA network with every generator split into ``<g>|up`` and ``<g>|down``."""
    import pypsa
    nb = pypsa.Network()
    nb.add("Carrier", list(n.carriers.index))
    nb.add("Bus", n.buses.index, v_nom=n.buses.v_nom, x=n.buses.x, y=n.buses.y, carrier="AC",
           country=n.buses.country)
    L = n.lines
    nb.add("Line", L.index, bus0=L.bus0, bus1=L.bus1, x=L.x, r=L.r, s_nom=L.s_nom,
           s_max_pu=cfg["network"]["s_max_pu"])
    K = n.links
    nb.add("Link", K.index, bus0=K.bus0, bus1=K.bus1, p_nom=K.p_nom, p_min_pu=-1.0,
           efficiency=1.0, carrier="DC")
    g = n.generators
    nb.add("Generator", g.index + "|up", bus=g.bus.to_numpy(), carrier=g.carrier.to_numpy(),
           p_nom=g.p_nom.to_numpy())
    nb.add("Generator", g.index + "|down", bus=g.bus.to_numpy(), carrier=g.carrier.to_numpy(),
           p_nom=g.p_nom.to_numpy(), p_min_pu=-1.0, p_max_pu=0.0)
    return nb


class Redispatch:
    """Stage-B LP for one zone map, built once, solved hour by hour."""

    def __init__(self, n, zmap: pd.Series, cfg: dict, costs: np.ndarray):
        self.G = len(n.generators)
        mk = cfg["validate"]["markup"]
        self.costs = costs
        self.markup = mk
        self.nb = split_network(n, cfg)
        self.m = DCOPF(self.nb, cfg["network"]["s_max_pu"],
                       costs=np.r_[costs + mk, costs - mk])
        zg = n.generators.bus.map(zmap.reindex(n.buses.index)).to_numpy()
        self.zones = np.array(sorted(set(zg)))
        rows = []
        for z in self.zones:
            gi = np.flatnonzero(zg == z)
            rows.append(np.r_[gi, self.G + gi])
        self.m.add_gen_rows(rows, np.zeros(len(rows)), np.zeros(len(rows)))
        self.penalty = cfg["validate"].get("np_slack_penalty")
        if self.penalty is not None:
            Z = len(rows)
            self.m.add_cols_on_extra_rows(np.full(2 * Z, float(self.penalty)),
                                          np.r_[np.arange(Z), np.arange(Z)],
                                          np.r_[np.ones(Z), -np.ones(Z)])
        bi = pd.Series(np.arange(len(n.buses)), index=n.buses.index)
        self.gen_bus = bi[n.generators.bus].to_numpy()
        self.nbus = len(n.buses)

    @staticmethod
    def zone_names(zmap: pd.Series, n) -> np.ndarray:
        """Net-position row order (as in ``np_slack``)."""
        return np.array(sorted(set(n.generators.bus.map(zmap.reindex(n.buses.index)))))

    def solve(self, pmax: np.ndarray, bus_load: np.ndarray, p_zonal: np.ndarray) -> dict:
        pz = np.clip(p_zonal, 0.0, pmax)
        inj = np.zeros(self.nbus)
        np.add.at(inj, self.gen_bus, pz)
        hi = np.r_[np.maximum(pmax - pz, 0.0), np.zeros(self.G)]
        lo = np.r_[np.zeros(self.G), -pz]
        r = self.m.solve(hi, bus_load - inj, pmin=lo, duals=True)
        if r["status"] != "ok":
            return {"status": r["status"], "retry": r.get("retry", "")}
        u, d = r["p"][:self.G], r["p"][self.G:]
        ex = r.get("extra_x", np.zeros(0))
        Z = len(self.zones)
        slack = ex[:Z] + ex[Z:2 * Z] if len(ex) else np.zeros(Z)
        return {"status": "ok", "np_slack": slack, "retry": r.get("retry", ""), "up": u, "down": -d,
                "line_mu": r["line_mu"], "link_mu": r["link_mu"], "np_dual": r["extra_dual"],
                "line_p": r["line_p"], "link_p": r["link_p"],
                "cost_phys": float(self.costs @ (u + d)),
                "cost_markup": float(self.markup * (u.sum() - d.sum())),
                "objective": r["objective"], "price": r["price"]}


def redispatch_pypsa(n, zmap: pd.Series, cfg: dict, costs: np.ndarray, snaps,
                     p_zonal: pd.DataFrame) -> pd.Series:
    """Same stage-B model in PyPSA/linopy; net positions added with
    ``n.model.add_constraints``.  Returns the objective per snapshot."""
    snaps = pd.DatetimeIndex(snaps)
    nb = split_network(n, cfg)
    nb.set_snapshots(snaps)
    g = n.generators
    mk = cfg["validate"]["markup"]
    pmax = n.get_switchable_as_dense("Generator", "p_max_pu").loc[snaps] * g.p_nom
    pz = p_zonal.loc[snaps, g.index].clip(lower=0.0).clip(upper=pmax)
    pn = g.p_nom.replace(0.0, 1.0)
    up_pu = ((pmax - pz).clip(lower=0.0) / pn)
    dn_pu = -(pz / pn)
    up_pu.columns = g.index + "|up"
    dn_pu.columns = g.index + "|down"
    nb.generators_t.p_max_pu = up_pu
    nb.generators_t.p_min_pu = dn_pu
    nb.generators["marginal_cost"] = np.r_[costs + mk, costs - mk]
    load = n.loads_t.p_set.loc[snaps, n.loads.index].T.groupby(n.loads.bus.to_numpy()).sum().T
    inj = pz.T.groupby(g.bus.to_numpy()).sum().T
    net = load.reindex(columns=n.buses.index, fill_value=0.0) - inj.reindex(columns=n.buses.index,
                                                                              fill_value=0.0)
    net.columns = n.buses.index + " net"
    nb.add("Load", n.buses.index + " net", bus=n.buses.index, p_set=net)
    nb.optimize.create_model()
    m = nb.model
    P = m.variables["Generator-p"]
    zg = g.bus.map(zmap.reindex(n.buses.index))
    pen = cfg["validate"].get("np_slack_penalty")
    slacks = []
    for z in sorted(zg.unique()):
        names = list(g.index[zg == z] + "|up") + list(g.index[zg == z] + "|down")
        lhs = P.sel(name=names).sum("name")
        if pen is not None:
            sp = m.add_variables(lower=0, coords=[pd.Index(snaps, name="snapshot")], name=f"np_slack_up_{z}")
            sm = m.add_variables(lower=0, coords=[pd.Index(snaps, name="snapshot")], name=f"np_slack_dn_{z}")
            lhs = lhs + sp - sm
            slacks += [sp, sm]
        m.add_constraints(lhs == 0, name=f"net_position_{z}")
    if slacks:
        m.objective = m.objective + sum(float(pen) * v.sum() for v in slacks)
    nb.optimize.solve_model(solver_name="highs", solver_options={"threads": 1, "output_flag": False})
    obj = (nb.generators_t.p * nb.generators.marginal_cost).sum(axis=1)
    for v in slacks:
        obj = obj + float(pen) * v.solution.to_pandas().reindex(obj.index)
    return obj
