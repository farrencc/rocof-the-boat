"""Stage A: zonal day-ahead clearing as an ATC transport model.

The nodal network (``bzgen.solve.opf.build_network``, unchanged) is rebuilt
with buses collapsed to zones:

* one ``Bus`` per zone; every generator (load shedding included, at
  ``solve.load_shedding_cost``) and every load re-pointed to its zone's bus,
  **identity preserved** (stage B needs per-unit ``p_zonal``).  This is why
  ``get_clustering_from_busmap`` is not used: it aggregates generators;
* intra-zone AC lines deleted (copper plate);
* inter-zone AC lines collapsed to one ``Link`` per zone pair,
  ``p_nom = derating * sum(s_nom * s_max_pu)`` over the cut branches,
  ``p_min_pu = -1``, ``efficiency = 1``.  A ``Link``, not a ``Line``: no KVL —
  this is a market model, not a physical one;
* HVDC links kept as they are, not derated (a link whose two ends fall in the
  same zone is dropped: it would be a self-loop).

The LP is solved by the same machinery as the nodal benchmark
(``bzgen.solve.lp.DCOPF``: with no lines the angle block is empty and it is a
transport model) with the same seeded cost noise, the same snapshots and the
same worker/block parallelism (``bzgen.validate.run``).

Relaxation property: at ``derating = 1`` every nodal-feasible dispatch is
zonal-feasible (sum the nodal flows over each cut), so ``C_market <= C_nodal``
hour by hour.  At ``derating < 1`` this does **not** hold in general (the ATC is
smaller than what the physical cut can carry) and is reported, not asserted.
``C_market(k=1) <= C_market(split)`` holds at every derating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bzgen.solve.lp import DCOPF


def noise_costs(n, cfg: dict, seed: int | None = None) -> np.ndarray:
    """Marginal cost + the nodal benchmark's seeded noise, in ``n.generators`` order.
    Identical to ``bzgen.solve.run._worker_init``."""
    seed = cfg["anneal"]["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    cn = cfg["solve"]["cost_noise"]
    return n.generators.mc_base.to_numpy() + rng.uniform(cn / 10, cn, len(n.generators))


def interzone_links(lines: pd.DataFrame, zmap: pd.Series, derating: float,
                    s_max_pu: float) -> pd.DataFrame:
    """One row per zone pair joined by AC lines: bus0/bus1 = zones, p_nom, cut lines."""
    z0 = lines.bus0.map(zmap).to_numpy()
    z1 = lines.bus1.map(zmap).to_numpy()
    cut = z0 != z1
    a = np.where(z0 < z1, z0, z1)[cut]
    b = np.where(z0 < z1, z1, z0)[cut]
    cap = (lines.s_nom.to_numpy() * s_max_pu)[cut]
    df = pd.DataFrame({"bus0": a, "bus1": b, "cap": cap, "line": lines.index[cut]})
    g = df.groupby(["bus0", "bus1"]).agg(sum_cap=("cap", "sum"), n_lines=("line", "size"),
                                         lines=("line", lambda s: ";".join(s))).reset_index()
    g["p_nom"] = derating * g.sum_cap
    g.index = "AC " + g.bus0 + "--" + g.bus1
    return g


def collapse(n, zmap: pd.Series, derating: float, cfg: dict, snapshots=None):
    """The zonal PyPSA network.  ``snapshots``: time series are attached only for these
    (the solver takes per-hour bounds directly; the series are for the PyPSA cross-check)."""
    import pypsa
    zones = sorted(zmap.unique())
    nz = pypsa.Network()
    if snapshots is not None:
        nz.set_snapshots(pd.DatetimeIndex(snapshots))
    nz.add("Carrier", list(n.carriers.index))
    nz.add("Bus", zones, carrier="AC")
    g = n.generators
    if snapshots is not None:
        pmpu = n.get_switchable_as_dense("Generator", "p_max_pu").loc[snapshots]
    else:
        pmpu = g.p_max_pu
    nz.add("Generator", g.index, bus=g.bus.map(zmap).to_numpy(), carrier=g.carrier,
           p_nom=g.p_nom, marginal_cost=g.marginal_cost, p_max_pu=pmpu)
    nz.generators["mc_base"] = g.mc_base
    if snapshots is not None:
        nz.add("Load", n.loads.index, bus=n.loads.bus.map(zmap).to_numpy(),
               p_set=n.loads_t.p_set.loc[snapshots, n.loads.index])
    ac = interzone_links(n.lines, zmap, derating, cfg["network"]["s_max_pu"])
    nz.add("Link", ac.index, bus0=ac.bus0, bus1=ac.bus1, p_nom=ac.p_nom, p_min_pu=-1.0,
           efficiency=1.0, carrier="AC")
    k = n.links
    kz0, kz1 = k.bus0.map(zmap), k.bus1.map(zmap)
    keep = kz0 != kz1
    nz.add("Link", k.index[keep], bus0=kz0[keep], bus1=kz1[keep], p_nom=k.p_nom[keep],
           p_min_pu=-1.0, efficiency=1.0, carrier="DC")
    info = {"ac_links": ac, "dc_dropped_intrazone": list(k.index[~keep]),
            "n_zones": len(zones)}
    return nz, info


class ZonalMarket:
    """Stage-A LP for one (zone map, derating), built once, solved hour by hour."""

    def __init__(self, n, zmap: pd.Series, derating: float, cfg: dict, costs: np.ndarray):
        self.zmap = zmap.reindex(n.buses.index)
        self.nz, self.info = collapse(n, self.zmap, derating, cfg)
        if not (self.nz.generators.index == n.generators.index).all():
            raise AssertionError("zonal generators must keep identity and order")
        self.m = DCOPF(self.nz, cfg["network"]["s_max_pu"], costs=costs)
        zi = pd.Series(np.arange(len(self.nz.buses)), index=self.nz.buses.index)
        self.bus_zone = zi[self.zmap.to_numpy()].to_numpy()        # nodal bus -> zone row
        self.zones = self.nz.buses.index
        self.costs = costs

    def zone_load(self, bus_load: np.ndarray) -> np.ndarray:
        out = np.zeros(len(self.zones))
        np.add.at(out, self.bus_zone, bus_load)
        return out

    def solve(self, pmax: np.ndarray, bus_load: np.ndarray) -> dict:
        zl = self.zone_load(bus_load)
        r = self.m.solve(pmax, zl, duals=True)
        if r["status"] != "ok":
            return r
        gen_z = np.zeros(len(self.zones))
        np.add.at(gen_z, self.m.n.generators.bus.map(
            pd.Series(np.arange(len(self.zones)), index=self.zones)).to_numpy(), r["p"])
        r["net_position"] = gen_z - zl
        r["cost"] = float(self.costs @ r["p"])
        return r


def check_against_pypsa(n, zmap, derating, cfg, costs, snaps, bus_load_fn) -> pd.DataFrame:
    """Solve a few hours with ``pypsa.Network.optimize`` on the collapsed network and
    compare objectives with the highspy path (same costs)."""
    zm = ZonalMarket(n, zmap, derating, cfg, costs)
    nz, _ = collapse(n, zm.zmap, derating, cfg, snapshots=snaps)
    nz.generators["marginal_cost"] = costs
    nz.optimize(solver_name="highs", include_objective_constant=False,
                solver_options={"threads": 1, "log_to_console": False, "output_flag": False})
    pmpu = n.get_switchable_as_dense("Generator", "p_max_pu")
    rows = []
    for s in snaps:
        r = zm.solve(pmpu.loc[s].to_numpy() * n.generators.p_nom.to_numpy(), bus_load_fn(s))
        c_py = float((nz.generators_t.p.loc[s] * costs).sum())
        rows.append({"snapshot": s, "highspy": r["cost"], "pypsa": c_py,
                     "rel_diff": abs(r["cost"] - c_py) / max(1.0, abs(c_py))})
    return pd.DataFrame(rows).set_index("snapshot")
