"""Dispatch down, attribution, cost gap and load pockets.

Dispatch down (focus cluster-country, i.e. DE incl. LU and DE offshore wind)::

    DD_RES    = market_spill[RES] + redispatch_down[RES]
    DD_RES_TH = DD_RES + redispatch_down[thermal]

``RES`` = ``validate.res_carriers``; thermal = every other carrier except load
shedding.  Market spill of a RES unit = available energy minus its zonal
schedule.  Dispatch down here is **entirely constraint-driven**: a DC-OPF has
no SNSP or inertia limit, so there is no global pro-rata curtailment component.
Load shedding is reported separately everywhere and never merged into DD.

Attribution (per hour, of DD_RES)
---------------------------------
* market spill — over the zonal links (inter-zone ATC and HVDC) touching a
  focus zone whose stage-A shadow price exceeds ``validate.dual_tol``;
* redispatch down — over the nodal branches (AC lines and HVDC links) touching
  a focus bus whose stage-B shadow price exceeds ``validate.dual_tol``;
each weighted by ``|mu| * binding capacity``.  Buckets: ``internal`` (both ends
in the same zone), ``border_intra`` (two different focus zones), ``border_cross``
(one end abroad), ``pocket`` (a branch incident to a chronic load-pocket bus:
excluded from the congestion statistics, kept as its own bucket so that the
buckets always sum to DD_RES), ``unattributed`` (no binding branch that hour).
Zonal links are border by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATS = ("internal", "border_intra", "border_cross", "pocket")
BUCKETS = CATS + ("unattributed",)


def attribute(dd: float, mu: np.ndarray, cap: np.ndarray, cat: np.ndarray, tol: float) -> dict:
    """Split ``dd`` over binding branches by |mu| * cap.  ``cat``: index into CATS, -1 = ignore."""
    out = dict.fromkeys(BUCKETS, 0.0)
    if dd <= 0:
        return out
    m = (np.abs(mu) > tol) & (cat >= 0)
    w = np.abs(mu[m]) * cap[m]
    if w.sum() <= 0:
        out["unattributed"] = dd
        return out
    s = np.bincount(cat[m], weights=w, minlength=len(CATS)) / w.sum()
    for i, c in enumerate(CATS):
        out[c] = dd * s[i]
    return out


def branch_categories(b0: pd.Series, b1: pd.Series, zmap: pd.Series, focus_bus: pd.Series,
                      pocket: set) -> np.ndarray:
    """Category code per branch (bus-level ends)."""
    f0 = focus_bus.reindex(b0).fillna(False).to_numpy(bool)
    f1 = focus_bus.reindex(b1).fillna(False).to_numpy(bool)
    z0, z1 = zmap.reindex(b0).to_numpy(), zmap.reindex(b1).to_numpy()
    cat = np.full(len(b0), -1, dtype=np.int64)
    touch = f0 | f1
    cat[touch & f0 & f1 & (z0 == z1)] = 0
    cat[touch & f0 & f1 & (z0 != z1)] = 1
    cat[touch & (f0 ^ f1)] = 2
    pk = np.array([a in pocket or b in pocket for a, b in zip(b0, b1)], dtype=bool)
    cat[touch & pk] = 3
    return cat


def zone_link_categories(bus0: pd.Series, bus1: pd.Series, focus_zones: set) -> np.ndarray:
    f0 = bus0.isin(focus_zones).to_numpy()
    f1 = bus1.isin(focus_zones).to_numpy()
    cat = np.full(len(bus0), -1, dtype=np.int64)
    cat[f0 & f1] = 1
    cat[f0 ^ f1] = 2
    return cat


class Context:
    """Per (map) static arrays for :func:`hour_metrics`."""

    def __init__(self, n, zmap: pd.Series, cfg: dict, pocket: set, zonal_links: pd.DataFrame,
                 focus_bus: pd.Series):
        """``focus_bus``: bool per bus, True in the focus cluster-country."""
        v = cfg["validate"]
        g = n.generators
        focus_bus = focus_bus.reindex(n.buses.index).fillna(False).astype(bool)
        gf = focus_bus.reindex(g.bus).to_numpy()
        self.res = gf & g.carrier.isin(v["res_carriers"]).to_numpy()
        self.shed = (g.carrier == "load_shedding").to_numpy()
        self.th = gf & ~self.res & ~self.shed
        self.focus_gen = gf
        self.focus_shed = gf & self.shed
        self.pocket_shed = self.shed & g.bus.isin(pocket).to_numpy()
        self.tol = v["dual_tol"]
        self.line_cat = branch_categories(n.lines.bus0, n.lines.bus1, zmap, focus_bus, pocket)
        self.link_cat = branch_categories(n.links.bus0, n.links.bus1, zmap, focus_bus, pocket)
        self.line_cap = (n.lines.s_nom * cfg["network"]["s_max_pu"]).to_numpy()
        self.link_cap = n.links.p_nom.to_numpy()
        fz = set(zmap[focus_bus.reindex(zmap.index).to_numpy()].unique())
        self.focus_zones = fz
        self.zl_cat = zone_link_categories(zonal_links.bus0, zonal_links.bus1, fz)
        self.zl_cap = zonal_links.p_nom.to_numpy()


def hour_metrics(ctx: Context, pmax: np.ndarray, ra: dict, rb: dict | None) -> dict:
    """Scalars for one hour.  ``ra``: stage-A result, ``rb``: stage-B result (None/!ok = infeasible)."""
    pz = np.clip(ra["p"], 0.0, pmax)
    spill = pmax - pz
    out = {"avail_res": float(pmax[ctx.res].sum()),
           "spill_res": float(spill[ctx.res].sum()),
           "C_market": ra["cost"],
           "shed_market_focus": float(pz[ctx.focus_shed].sum()),
           "shed_market_total": float(pz[ctx.shed].sum())}
    a_sp = attribute(out["spill_res"], ra["link_mu"], ctx.zl_cap, ctx.zl_cat, ctx.tol)
    ok = rb is not None and rb.get("status") == "ok"
    out["stageB_ok"] = ok
    if not ok:
        for k in ("down_res", "down_th", "up_focus", "down_focus", "up_total", "down_total",
                  "C_redispatch", "C_markup", "shed_focus", "shed_total", "shed_pocket",
                  "DD_RES", "DD_RES_TH"):
            out[k] = np.nan
        for b in BUCKETS:
            out[f"attr_{b}"] = np.nan
        return out
    u, d = rb["up"], rb["down"]
    final = pz + u - d
    out.update({"down_res": float(d[ctx.res].sum()), "down_th": float(d[ctx.th].sum()),
                "up_focus": float(u[ctx.focus_gen].sum()),
                "down_focus": float(d[ctx.focus_gen].sum()),
                "up_total": float(u.sum()), "down_total": float(d.sum()),
                "C_redispatch": rb["cost_phys"], "C_markup": rb["cost_markup"],
                "shed_focus": float(final[ctx.focus_shed].sum()),
                "shed_total": float(final[ctx.shed].sum()),
                "shed_pocket": float(final[ctx.pocket_shed].sum())})
    out["DD_RES"] = out["spill_res"] + out["down_res"]
    out["DD_RES_TH"] = out["DD_RES"] + out["down_th"]
    mu = np.r_[rb["line_mu"], rb["link_mu"]]
    cap = np.r_[ctx.line_cap, ctx.link_cap]
    cat = np.r_[ctx.line_cat, ctx.link_cat]
    a_rd = attribute(out["down_res"], mu, cap, cat, ctx.tol)
    for b in BUCKETS:
        out[f"attr_{b}"] = a_sp[b] + a_rd[b]
    return out


def pockets(shed_share: pd.Series, cfg: dict) -> pd.Index:
    """Buses shedding in at least ``validate.pocket_hour_share`` of nodal hours."""
    return shed_share.index[shed_share >= cfg["validate"]["pocket_hour_share"]]


def summarise(h: pd.DataFrame, w: pd.Series, nodal_cost: pd.Series) -> dict:
    """Annualised aggregates of an hourly table (weights ``w``; feasible stage-B hours only
    for the stage-B quantities, all hours for stage A).  GWh/yr, EUR/yr."""
    w = w.reindex(h.index)
    scale = 8760.0 / w.sum()
    ok = h.stageB_ok.astype(bool)
    wo = w[ok]
    sc_ok = 8760.0 / wo.sum() if wo.sum() > 0 else np.nan

    def tot(col, mask=None, s=scale):
        x = h[col] if mask is None else h.loc[mask, col]
        return float((x * w.reindex(x.index)).sum() * s)

    row = {"hours": int(len(h)), "hours_stageB_ok": int(ok.sum()),
           "lp_infeasible_hours": int((~ok).sum()),
           "avail_res_GWh": tot("avail_res", ok, sc_ok) / 1e3,
           "spill_res_GWh": tot("spill_res", ok, sc_ok) / 1e3,
           "down_res_GWh": tot("down_res", ok, sc_ok) / 1e3,
           "down_th_GWh": tot("down_th", ok, sc_ok) / 1e3,
           "DD_RES_GWh": tot("DD_RES", ok, sc_ok) / 1e3,
           "DD_RES_TH_GWh": tot("DD_RES_TH", ok, sc_ok) / 1e3,
           "redispatch_up_focus_GWh": tot("up_focus", ok, sc_ok) / 1e3,
           "redispatch_down_focus_GWh": tot("down_focus", ok, sc_ok) / 1e3,
           "shed_focus_GWh": tot("shed_focus", ok, sc_ok) / 1e3,
           "shed_pocket_GWh": tot("shed_pocket", ok, sc_ok) / 1e3,
           "shed_total_GWh": tot("shed_total", ok, sc_ok) / 1e3,
           "C_market_EUR": tot("C_market", ok, sc_ok),
           "C_redispatch_EUR": tot("C_redispatch", ok, sc_ok),
           "C_markup_EUR": tot("C_markup", ok, sc_ok)}
    if "np_slack_total" in h:
        tol = 1.0
        row["np_slack_total_GWh"] = tot("np_slack_total", ok, sc_ok) / 1e3
        row["np_slack_focus_GWh"] = tot("np_slack_focus", ok, sc_ok) / 1e3
        row["np_undeliverable_hours"] = int((h.loc[ok, "np_slack_total"] > tol).sum())
        row["np_undeliverable_hours_focus"] = int((h.loc[ok, "np_slack_focus"] > tol).sum())
    cn = float((nodal_cost.reindex(h.index[ok]) * wo).sum() * sc_ok)
    row["C_nodal_EUR"] = cn
    row["C_total_EUR"] = row["C_market_EUR"] + row["C_redispatch_EUR"]
    row["efficiency_gap"] = (row["C_total_EUR"] - cn) / cn if cn else np.nan
    row["DD_share_of_avail_res"] = row["DD_RES_GWh"] / row["avail_res_GWh"] if row["avail_res_GWh"] else np.nan
    att = {b: tot(f"attr_{b}", ok, sc_ok) / 1e3 for b in BUCKETS}
    for b in BUCKETS:
        row[f"attr_{b}_GWh"] = att[b]
    cong = att["internal"] + att["border_intra"] + att["border_cross"]
    row["internal_share"] = att["internal"] / cong if cong > 0 else np.nan
    row["border_share"] = (att["border_intra"] + att["border_cross"]) / cong if cong > 0 else np.nan
    return row
