"""Run every configuration against every disturbance, and score it.

This is the experiment: for each of the ensemble's operating states, integrate
the swing equations through the same fixed disturbance set, evaluate

    J = sum_buses sum_events int_0^T [ tau^2 (df/dt)^2 + f^2 ] dt

and write one row per configuration.  Nothing about the network changes between
rows -- only which machines are running and what everything is producing.

The analysis at the end asks the question the study was set up for, in three
parts, because they have different answers:

  1. Does J rise with SNSP?  (Pearson and Spearman, plus the shape of the rise.)
  2. How much of J does SNSP explain, and how much is left over at fixed SNSP?
     The residual is the part that depends on *which* machines are on and where
     the wind is -- geography, not share.
  3. Which is the operative variable?  SNSP is a proxy; the regression puts it
     against system inertia, primary reserve and distance-to-the-nearest-machine
     to see which one the cost actually follows.

`--tag` names the output files, so variants (FFR on, different tau, different
horizon) sit side by side with the baseline rather than overwriting it.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import configs as C
import dynamics as dyn
import grid as G
import perturbations as PB

HERE = Path(__file__).resolve().parent

#: Grid-code-shaped thresholds, used only to report an exceedance count
#: alongside the smooth cost: a bus-event is "at risk" if its 500 ms RoCoF or
#: its frequency excursion passes these.
ROCOF_LIMIT, DEV_LIMIT = 0.5, 0.5      # Hz/s, Hz
#: The all-island RoCoF limit the grid code is written against, and the first
#: stage of under-frequency load shedding.  Thresholds, not smooth costs: a
#: linear model scales, but crossing a threshold does not, so these are where
#: making a disturbance larger can change which configuration is worse.
ROCOF_CODE, UFLS_LIMIT = 1.0, 0.8      # Hz/s, Hz below nominal
#: The per-bus-per-event readings saved alongside each ensemble, by their name
#: on `dynamics.Response`.  Every `*_top{n}` column in the CSV is a reduction of
#: one of these over the bus axis (`dynamics.worst_n`), so keeping them makes
#: any other n, or any per-event view of the same score, a reprocess of the npz
#: rather than another pass through the solver.
FIELDS = ("rocof_500ms", "peak_rocof", "peak_dev", "nadir")


def run(g: G.ToyGrid, cfgs: list[C.Config], events: list[PB.Event], **kw
        ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray,
                   dict[str, np.ndarray]]:
    """Score every configuration against every event.

    With a hazard-scaled event set the injection matrix is rebuilt for each
    configuration, because the megawatts an event removes depend on what that
    configuration has running.  With the fixed set it is built once.

    Alongside J summed over buses (`by_event`), the raw per-bus-per-event
    *readings* are kept: the 500 ms RoCoF, the peak RoCoF, the absolute
    frequency deviation and the nadir, each one number per bus per event.

    They are kept rather than the scores derived from them because `summarise`
    collapses each field to a single number -- worst five buses, averaged over
    the disturbance set -- and that collapse is not invertible.  Saving only the
    collapsed scores means every new question about this family (a different n,
    a different field, one scenario at a time rather than the average) needs the
    swing equations solved again, for readings that were computed and thrown
    away.  Four (n_bus, n_event) float64 fields cost about 15 MB per ensemble
    and turn all of those into a reprocess.
    """
    scaled = any(e.scaled for e in events)
    U_fixed = None if scaled else PB.matrix(events)
    rows, by_bus, by_event, sizes = [], [], [], []
    fields: dict[str, list] = {k: [] for k in FIELDS}
    t0 = time.time()
    for n, c in enumerate(cfgs):
        U = PB.matrix(events, c.P) if scaled else U_fixed
        net = c.network(g, damping=kw.get("damping", "rating"))
        resp = dyn.respond(net, U, t_end=kw.get("t_end", dyn.T_END),
                           dt=kw.get("dt", dyn.DT), tau=kw.get("tau", dyn.TAU),
                           t_meas=kw.get("t_meas", dyn.T_MEAS),
                           agc=c.participation(g) if kw.get("agc") else None)
        row = {"config": c.index, "snsp_target": c.snsp_target}
        row.update(c.meta)
        row.update(dyn.summarise(resp, g.N))
        row["share_rocof_over"] = float(
            (np.abs(resp.rocof_500ms) > ROCOF_LIMIT).mean())
        row["share_dev_over"] = float((resp.peak_dev > DEV_LIMIT).mean())
        row["worst_dev"] = float(resp.peak_dev.max())

        # Threshold-shaped readings: a linear model scales, but a limit does
        # not, so these can reorder configurations where the smooth cost cannot.
        row["share_rocof_over_code"] = float(
            (np.abs(resp.rocof_500ms) > ROCOF_CODE).mean())
        row["share_ufls"] = float((resp.nadir > UFLS_LIMIT).mean())
        row["events_ufls"] = float((resp.nadir.max(axis=0) > UFLS_LIMIT).mean())

        # And the nonlinear one: is there a post-event equilibrium at all?
        feas = [dyn.post_fault(net, U[:, k]) for k in range(U.shape[1])]
        row["share_infeasible"] = float(np.mean([not ok for ok, _ in feas]))
        loads = [ld for ok, ld in feas if ok]
        row["worst_post_fault_loading"] = float(max(loads)) if loads else np.inf

        row["hazard_pu"] = float(np.abs(U.sum(axis=0)).sum())
        row["committed"] = "|".join(str(b) for b in c.committed)
        row["condensers"] = "|".join(str(b) for b in c.condensers)
        rows.append(row)
        by_bus.append(resp.by_bus)
        by_event.append(resp.by_event)
        for name in FIELDS:
            fields[name].append(getattr(resp, name))
        sizes.append(np.abs(U.sum(axis=0)))
        if (n + 1) % 20 == 0:
            done = (n + 1) / len(cfgs)
            el = time.time() - t0
            print(f"  {n + 1:4d}/{len(cfgs)}  {el:5.0f}s elapsed, "
                  f"{el / done - el:5.0f}s left", flush=True)
    return (pd.DataFrame(rows), np.array(by_bus), np.array(by_event),
            np.array(sizes), {k: np.array(v) for k, v in fields.items()})


def _fit(x: np.ndarray, y: np.ndarray, deg: int = 1) -> tuple[np.ndarray, float]:
    """Least-squares polynomial and its R^2."""
    p = np.polyfit(x, y, deg)
    resid = y - np.polyval(p, x)
    return p, 1.0 - resid.var() / y.var()


def analyse(df: pd.DataFrame, events: list[PB.Event],
            by_event: np.ndarray) -> str:
    x, y = df["snsp"].to_numpy(), df["J"].to_numpy()
    t = df["snsp_target"].to_numpy()
    out = ["SNSP AGAINST THE COST OF THE FREQUENCY RESPONSE", "=" * 74, "",
           f"  {len(df)} configurations x {len(events)} disturbances, "
           f"J = sum_bus sum_event int [tau^2 (df/dt)^2 + f^2] dt", ""]

    r, p = stats.pearsonr(x, y)
    rho, prho = stats.spearmanr(x, y)
    lin, r2_lin = _fit(x, y, 1)
    quad, r2_quad = _fit(x, y, 2)
    out += ["1. Is there a correlation?", "-" * 74,
            f"  Pearson  r = {r:+.3f}  (p = {p:.2e})",
            f"  Spearman rho = {rho:+.3f}  (p = {prho:.2e})",
            f"  linear fit    J = {lin[0]:.2f} * SNSP + {lin[1]:.2f}"
            f"   R^2 = {r2_lin:.3f}",
            f"  quadratic fit                              R^2 = {r2_quad:.3f}"
            f"   (curvature {quad[0]:+.1f})",
            f"  J at SNSP {t.min() * 100:.0f}% vs {t.max() * 100:.0f}%: "
            f"{y[t == t.min()].mean():.1f} -> {y[t == t.max()].mean():.1f} "
            f"({y[t == t.max()].mean() / y[t == t.min()].mean():.2f}x)"]

    # The relationship is convex, so a Pearson r on the raw scale understates a
    # monotone rise and is dominated by the top end.  Two extra readings:
    # the same correlation with the SNSP = 100% stratum dropped (that stratum is
    # qualitatively different -- no machine is synchronised at all, so there is
    # no primary response of any kind), and the fit in log J, where a constant
    # proportional rise per point of SNSP would be a straight line.
    keep = t < t.max()
    r_ex = stats.pearsonr(x[keep], y[keep])[0]
    rho_ex = stats.spearmanr(x[keep], y[keep])[0]
    logp, r2_log = _fit(x, np.log(y), 1)
    out += [f"  excluding the SNSP = 100% stratum: r = {r_ex:+.3f}, "
            f"rho = {rho_ex:+.3f}",
            f"  log fit  log J = {logp[0]:.2f} * SNSP + {logp[1]:.2f}"
            f"   R^2 = {r2_log:.3f}",
            f"  i.e. J multiplies by {np.exp(logp[0] * 0.1):.2f} per 10 points "
            f"of SNSP, doubling every {np.log(2) / logp[0] * 100:.0f} points",
            ""]

    out += ["2. What do the two terms do separately?", "-" * 74,
            f"  {'quantity':26s} {'30% SNSP':>10s} {'100% SNSP':>10s} "
            f"{'ratio':>7s} {'r vs SNSP':>10s}"]
    t = df["snsp_target"].to_numpy()
    lo, hi = t == t.min(), t == t.max()
    for key, label in (("J", "J total"),
                       ("J_rocof", "  gradient term (RoCoF)"),
                       ("J_freq", "  field term (deviation)"),
                       ("J_coi", "J system-wide (COI)"),
                       ("J_local", "J local (J - COI)"),
                       ("J_ex_self", "J excluding hit bus"),
                       ("worst_rocof_500ms", "worst 500 ms RoCoF Hz/s"),
                       ("worst_dev", "worst |f| Hz"),
                       ("share_rocof_over", "share over 0.5 Hz/s"),
                       ("share_rocof_over_code", "share over 1 Hz/s (code)"),
                       ("share_ufls", "share past UFLS (-0.8 Hz)"),
                       ("share_infeasible", "no post-event equilibrium"),
                       ("worst_post_fault_loading", "worst line, post-event"),
                       ("hazard_pu", "total imbalance applied p.u."),
                       ("stored_energy", "stored energy p.u. s"),
                       ("reserve", "primary reserve")):
        v = df[key].to_numpy()
        rr = stats.pearsonr(x, v)[0] if v.std() > 1e-12 else float("nan")
        a, b = v[lo].mean(), v[hi].mean()
        # A ratio against a base of zero is not a large number, it is undefined:
        # several of these scores are exactly zero at 30% SNSP because nothing
        # crosses the threshold there at all.
        ratio = f"{b / a:7.2f}" if a > 1e-6 else ("      -" if b <= 1e-9
                                                 else "    inf")
        out.append(f"  {label:26s} {a:10.3f} {b:10.3f} {ratio} {rr:+10.3f}")
    out.append("")

    out += ["3. How much does SNSP not explain?", "-" * 74]
    resid = y - np.polyval(quad, x)
    within = df.groupby("snsp_target")["J"].agg(["mean", "std", "min", "max"])
    out += [f"  residual sd about the quadratic fit: {resid.std():.2f} "
            f"({resid.std() / y.mean() * 100:.1f}% of mean J)",
            f"  spread within an SNSP level (mean of max/min): "
            f"{(within['max'] / within['min']).mean():.2f}x",
            f"  the widest level: {(within['max'] / within['min']).max():.2f}x "
            f"at SNSP {within.index[(within['max'] / within['min']).argmax()] * 100:.0f}%",
            ""]
    out += [f"  {'SNSP':>6s} {'mean J':>9s} {'sd':>7s} {'min':>9s} {'max':>9s}"]
    for s, rr in within.iterrows():
        out.append(f"  {s * 100:5.0f}% {rr['mean']:9.2f} {rr['std']:7.2f} "
                   f"{rr['min']:9.2f} {rr['max']:9.2f}")
    out.append("")

    out += ["4. Which variable is the cost actually following?", "-" * 74,
            f"  {'predictor':30s} {'r':>8s} {'R^2 alone':>10s} "
            f"{'R^2 with SNSP':>14s}"]
    base = _fit(x, y, 2)[1]
    for key, label in (("snsp", "SNSP"),
                       ("stored_energy", "system inertia (sum H S)"),
                       ("reserve", "primary reserve (sum droop)"),
                       ("n_committed", "machines synchronised"),
                       ("machine_dist_mean", "mean hops to a machine"),
                       ("coastal_wind_share", "share of wind on the coast"),
                       ("radial_wind_pu", "output on radial farms")):
        v = df[key].to_numpy(float)
        ok = np.isfinite(v)
        if v[ok].std() == 0:
            continue
        rr = stats.pearsonr(v[ok], y[ok])[0]
        alone = _fit(v[ok], y[ok], 2)[1]
        both = _multi_r2(np.column_stack([x[ok], v[ok]]), y[ok])
        out.append(f"  {label:30s} {rr:+8.3f} {alone:10.3f} {both:14.3f}")
    out += ["", f"  (SNSP alone, quadratic: R^2 = {base:.3f})", ""]

    out += ["5. Do other cost functions agree?", "-" * 74,
            "  J and its two terms are integrals over the whole grid and the "
            "whole window.",
            "  The top-n scores are the other shape a limit can take: the "
            "500 ms RoCoF, or the",
            "  downward nadir, at the n worst buses, added up and averaged "
            "over events.", "",
            f"  {'score':28s} {'30%':>9s} {'100%':>9s} {'ratio':>7s} "
            f"{'rho vs SNSP':>12s} {'rho vs J':>9s}"]
    for key, label in (("J", "J  (both terms)"),
                       ("J_rocof", "J gradient term only"),
                       ("J_freq", "J field term only"),
                       ("rocof_top1", "500 ms RoCoF, worst bus"),
                       ("rocof_top5", "500 ms RoCoF, worst 5"),
                       ("rocof_top10", "500 ms RoCoF, worst 10"),
                       ("coi_rocof_500ms", "  the same, system-wide"),
                       ("nadir_top1", "nadir, worst bus"),
                       ("nadir_top5", "nadir, worst 5"),
                       ("nadir_top10", "nadir, worst 10"),
                       ("coi_nadir_mean", "  the same, system-wide"),
                       ("rocof_local_excess", "worst bus / system, RoCoF"),
                       ("nadir_local_excess", "worst bus / system, nadir")):
        if key not in df:
            continue
        v = df[key].to_numpy(float)
        out.append(f"  {label:28s} {v[lo].mean():9.3f} {v[hi].mean():9.3f} "
                   f"{v[hi].mean() / max(v[lo].mean(), 1e-12):7.2f} "
                   f"{stats.spearmanr(x, v)[0]:+12.3f} "
                   f"{stats.spearmanr(y, v)[0]:+9.3f}")
    out.append("")

    out += ["6. Which disturbances cost the most?", "-" * 74,
            f"  {'event':28s} {'kind':13s} {'mean J':>9s} "
            f"{'30%':>8s} {'100%':>8s} {'ratio':>7s}"]
    order = np.argsort(-by_event.mean(axis=0))
    for k in order:
        e = events[k]
        a, b = by_event[lo, k].mean(), by_event[hi, k].mean()
        out.append(f"  {e.name:28s} {e.kind:13s} "
                   f"{by_event[:, k].mean():9.3f} {a:8.3f} {b:8.3f} "
                   f"{b / a:7.2f}")
    return "\n".join(out) + "\n"


def _multi_r2(X: np.ndarray, y: np.ndarray) -> float:
    """R^2 of a quadratic-in-each-column least-squares fit."""
    cols = [np.ones(len(y))]
    for j in range(X.shape[1]):
        cols += [X[:, j], X[:, j] ** 2]
    A = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return 1.0 - (y - A @ beta).var() / y.var()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--ffr", type=float, default=C.FFR_DROOP,
                    help="droop gain given to the converters (0 = none)")
    ap.add_argument("--damping", default="rating",
                    choices=["rating", "equal_ratio", "uniform", "proportional"],
                    help="see dynamics.damping_vector; only 'rating' is "
                         "calibrated for this grid")
    ap.add_argument("--tau", type=float, default=dyn.TAU)
    ap.add_argument("--t-end", type=float, default=dyn.T_END)
    ap.add_argument("--dt", type=float, default=dyn.DT)
    ap.add_argument("--t-meas", type=float, default=dyn.T_MEAS)
    ap.add_argument("--events", default="sourced",
                    choices=list(PB.SETS),
                    help="which frozen disturbance set: 'sourced' (each event "
                         "scales with its own source -- the headline set), "
                         "'fixed' (hazard held constant, the vulnerability-only "
                         "control) or 'scaled' (wind-driven events only)")
    ap.add_argument("--muon", type=int, default=0,
                    help="minimum units on: keep at least this many stations "
                         "synchronised, as condensers if the energy balance "
                         "does not need them")
    ap.add_argument("--agc", action="store_true",
                    help="add secondary control (integral action on the system "
                         "frequency, shared over the committed machines), which "
                         "is what actually returns the frequency to 50 Hz")
    ap.add_argument("--draws", type=int, default=C.DRAWS_PER_LEVEL)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--analyse-only", action="store_true",
                    help="re-derive the report from the saved CSV and npz "
                         "without re-solving anything")
    args = ap.parse_args()

    g = G.load()
    events = PB.load(g, which=args.events)
    tag = f"_{args.tag}" if args.tag else ""

    if args.analyse_only:
        df = pd.read_csv(HERE / f"ensemble{tag}.csv")
        by_event = np.load(HERE / f"ensemble{tag}_nodes.npz")["by_event"]
        text = analyse(df, events, by_event)
        print(text)
        (HERE / f"ensemble{tag}_report.txt").write_text(text)
        print(f"rewrote ensemble{tag}_report.txt")
        return
    cfgs = C.ensemble(g, draws=args.draws, seed=args.seed, ffr=args.ffr,
                      muon=args.muon)
    print(f"{len(cfgs)} configurations, {len(events)} {args.events} "
          f"disturbances, muon={args.muon}, "
          f"tau={args.tau} s, T={args.t_end} s, dt={args.dt} s, "
          f"t_meas={args.t_meas} s, ffr={args.ffr}, damping={args.damping}, "
          f"agc={'on' if args.agc else 'off'}")

    df, by_bus, by_event, sizes, fields = run(
        g, cfgs, events, damping=args.damping, tau=args.tau, t_end=args.t_end,
        dt=args.dt, t_meas=args.t_meas, agc=args.agc)
    df.to_csv(HERE / f"ensemble{tag}.csv", index=False, lineterminator="\n")
    np.savez_compressed(HERE / f"ensemble{tag}_nodes.npz",
                        by_bus=by_bus, by_event=by_event, sizes=sizes,
                        snsp=df["snsp"].to_numpy(), **fields)

    text = analyse(df, events, by_event)
    print("\n" + text)
    (HERE / f"ensemble{tag}_report.txt").write_text(text)
    print(f"wrote ensemble{tag}.csv, ensemble{tag}_nodes.npz, "
          f"ensemble{tag}_report.txt")


if __name__ == "__main__":
    main()
