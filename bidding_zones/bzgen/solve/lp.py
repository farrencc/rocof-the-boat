"""DC-OPF LP assembled from a PyPSA network and solved hour by hour with highspy.

Why not ``n.optimize()`` per hour: on this network (3.8k buses, 6.4k lines,
16k generators incl. shedding) one ``optimize`` call spends ~150 s outside the
solver (model building, KVL cycle construction, solution assignment) and
~18 s in HiGHS for 4 hours.  The hour-by-hour LP has fixed structure, so it is
built once from the PyPSA network components and only bounds change per hour;
HiGHS then re-optimises from the previous basis (dual simplex warm start).
The formulation is PyPSA's linear power flow in angle form:

    min  sum_g c_g p_g
    s.t. sum_{g at n} p_g - sum_l A_nl F_l - sum_k A_nk f_k = d_n      (dual = LMP_n)
         F_l = (theta_bus0 - theta_bus1) / x_pu,l,  |F_l| <= s_nom s_max_pu
         0 <= p_g <= p_nom p_max_pu(t),  |f_k| <= p_nom,k
         theta = 0 at one reference bus per synchronous area

with ``x_pu = x / v_nom^2`` (PyPSA's per-unit convention, 1 MVA base).  Prices
are checked against ``n.optimize()`` on sample hours (``check_against_pypsa``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


class DCOPF:
    def __init__(self, n, s_max_pu: float, costs: np.ndarray | None = None):
        import highspy

        self.n = n
        buses = n.buses.index
        self.buses = buses
        bi = pd.Series(np.arange(len(buses)), index=buses)
        g = n.generators
        self.gens = g.index
        L = n.lines
        K = n.links
        N, G, NL, NK = len(buses), len(g), len(L), len(K)
        self.N, self.G, self.NL, self.NK = N, G, NL, NK
        gb = bi[g.bus].to_numpy()
        l0 = bi[L.bus0].to_numpy()
        l1 = bi[L.bus1].to_numpy()
        k0 = bi[K.bus0].to_numpy()
        k1 = bi[K.bus1].to_numpy()
        v = n.buses.v_nom[L.bus0].to_numpy()
        # Susceptance in PyPSA per-unit (1 MVA base).  The angle columns are
        # scaled by S = median(b): phi = S theta, so the matrix holds b/S
        # (~1e-3..1e2) instead of b (~1e2..3e6), which HiGHS otherwise fails on.
        b_pu = 1.0 / (L.x.to_numpy() / v ** 2)
        self.S = float(np.median(b_pu))
        self.bl = b_pu / self.S
        self.l0, self.l1 = l0, l1
        self.Fmax = (L.s_nom * s_max_pu).to_numpy()
        # columns: [p_g (G) | theta (N) | f_k (NK)]
        # balance rows: +p_g ; line flow F_l = b (th0 - th1) enters bus0 with -1, bus1 with +1
        rows, cols, vals = [], [], []
        rows += list(gb); cols += list(range(G)); vals += [1.0] * G
        for a, b, bl in ((l0, l1, self.bl),):
            # -F at bus0: -b th0 + b th1 ; +F at bus1: +b th0 - b th1
            rows += list(a); cols += list(G + a); vals += list(-bl)
            rows += list(a); cols += list(G + b); vals += list(bl)
            rows += list(b); cols += list(G + a); vals += list(bl)
            rows += list(b); cols += list(G + b); vals += list(-bl)
        rows += list(k0); cols += list(G + N + np.arange(NK)); vals += [-1.0] * NK
        rows += list(k1); cols += list(G + N + np.arange(NK)); vals += [1.0] * NK
        Abal = sp.coo_matrix((vals, (rows, cols)), shape=(N, G + N + NK))
        # line rows: b (th0 - th1)
        r = np.repeat(np.arange(NL), 2)
        c = np.c_[G + l0, G + l1].ravel()
        vv = np.c_[self.bl, -self.bl].ravel()
        Aline = sp.coo_matrix((vv, (r, c)), shape=(NL, G + N + NK))
        A = sp.vstack([Abal, Aline]).tocsc()
        A.sum_duplicates()
        self.ncol = G + N + NK
        self.cost = np.r_[g.marginal_cost.to_numpy() if costs is None else costs,
                          np.zeros(N + NK)]
        # reference bus per synchronous area: AC components
        adj = sp.coo_matrix((np.ones(NL), (l0, l1)), shape=(N, N))
        ncomp, lab = sp.csgraph.connected_components(adj, directed=False)
        ref = np.array([np.flatnonzero(lab == c)[0] for c in range(ncomp)])
        self.n_areas = ncomp
        col_lo = np.r_[np.zeros(G), np.full(N, -1e30), -K.p_nom.to_numpy()]
        col_hi = np.r_[g.p_nom.to_numpy(), np.full(N, 1e30), K.p_nom.to_numpy()]
        col_lo[G + ref] = 0.0
        col_hi[G + ref] = 0.0
        row_lo = np.r_[np.zeros(N), -self.Fmax]
        row_hi = np.r_[np.zeros(N), self.Fmax]
        lp = highspy.HighsLp()
        lp.num_col_ = self.ncol
        lp.num_row_ = N + NL
        lp.col_cost_ = self.cost
        lp.col_lower_ = col_lo
        lp.col_upper_ = col_hi
        lp.row_lower_ = row_lo
        lp.row_upper_ = row_hi
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr
        lp.a_matrix_.index_ = A.indices
        lp.a_matrix_.value_ = A.data
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.passModel(lp)
        self.h = h
        self.highspy = highspy
        self.p_nom = g.p_nom.to_numpy()
        self.gidx = np.arange(G, dtype=np.int32)
        self.bidx = np.arange(N, dtype=np.int32)

    def set_costs(self, costs: np.ndarray) -> None:
        self.h.changeColsCost(self.G, self.gidx, costs.astype(float))

    def solve(self, pmax: np.ndarray, load: np.ndarray) -> dict:
        """pmax: MW upper bound per generator; load: MW per bus. Returns prices etc."""
        h = self.h
        h.changeColsBounds(self.G, self.gidx, np.zeros(self.G), pmax.astype(float))
        h.changeRowsBounds(self.N, self.bidx, load.astype(float), load.astype(float))
        h.run()
        st = h.getModelStatus()
        retry = ""
        if st != self.highspy.HighsModelStatus.kOptimal:
            # 1) cold restart (drop the warm basis); 2) interior point + crossover
            h.clearSolver()
            h.run()
            st = h.getModelStatus()
            retry = "cold"
            if st != self.highspy.HighsModelStatus.kOptimal:
                h.setOptionValue("solver", "ipm")
                h.clearSolver()
                h.run()
                st = h.getModelStatus()
                h.setOptionValue("solver", "choose")
                retry = "ipm"
        if st != self.highspy.HighsModelStatus.kOptimal:
            return {"status": h.modelStatusToString(st), "retry": retry}
        sol = h.getSolution()
        x = np.asarray(sol.col_value)
        y = np.asarray(sol.row_dual)
        th = x[self.G:self.G + self.N]
        return {"status": "ok", "retry": retry, "price": y[:self.N].copy(), "p": x[:self.G].copy(),
                "line_p": self.bl * (th[self.l0] - th[self.l1]),
                "link_p": x[self.G + self.N:].copy(),
                "objective": h.getInfo().objective_function_value}
