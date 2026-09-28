"""Does the Green's function predict risk *across operating states*?

Everything else in this folder asks a difference question: an edge changes the
cost by so much, does it change the risk by so much.  The answer was no for
RoCoF.  That is not the same question as whether the cost is a good predictor of
risk at all, and the two can have different answers -- a quantity can order a
population correctly and still be useless for ranking small perturbations of one
member of it.

So: **no edges anywhere**.  The grid is the frozen one, untouched.  What varies
is the operating state -- all 240 balanced configurations of the ../snsp
ensemble, whose commitment, inertia, droop and dispatch differ enormously (system
inertia runs from about 11 to 55 p.u.s).  For each one, build its Green's
function and reduce it to a scalar; then plot that against the risk that
configuration actually incurs, taken from the published `ensemble_src.csv`.

THE COSTS

  C_H2      the Frobenius norm of Omega_f Pi G B, integrated over the band --
            the band-limited H2 norm.  This is the quantity the design spec
            dismisses as "a screen, not a criterion" (its section 7.3), on the
            grounds that an average over disturbance directions hedges against
            nothing in the worst case.  That argument is about worst-case
            robustness; it says nothing about whether the norm *predicts*, which
            is what is measured here.
  C_freq    max over the band of |Omega_f| sigma_max(Pi G B) -- the H-infinity
            peak, carried for comparison.
  C_rocof   the same with the windowed-RoCoF weight.

Each configuration gets **its own** band, from `base.physical_band`, because the
regularisation floor moves with the commitment: a state with more machines
synchronised has physical modes that a state with fewer does not.  A single
fixed band would admit artefacts for some configurations and cut physical modes
from others.  `--band` overrides this with a common one, and the report prints
both so the choice can be seen rather than trusted.

THE RISKS

`rocof_top5` and `dev_top5` from `ensemble_src.csv`, the worst-five-bus scores
that ../snsp draws its conclusions on.  Nothing is re-solved in the time domain:
those numbers are the published result and `validate.py` already asserts this
folder reproduces them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import base                                                  # noqa: E402
import nonlinear as nl                                       # noqa: E402
from base import BaseCase                                    # noqa: E402

sys.path.insert(0, str(base.SNSP))
import configs as C                                          # noqa: E402
import grid as G                                             # noqa: E402
import perturbations as PB                                   # noqa: E402

COSTS = ("C_H2", "C_freq", "C_rocof")
RISKS = ("rocof_top5", "dev_top5")
#: Carried for context: these are ../snsp's own predictors, and the point of
#: showing them is that a Green's-function norm has to beat them to be worth
#: computing.
REFERENCE = ("snsp", "stored_energy")


def run(which: str = "sourced", n_freq: int = 140,
        band: tuple[float, float] | None = None) -> tuple[pd.DataFrame, dict]:
    g = G.load(base.SNSP, verify=True)
    cfgs = C.ensemble(g)
    events = PB.load(g, base.SNSP, verify=True, which=which)
    risk = pd.read_csv(base.SNSP / "ensemble_src.csv").set_index("config")

    rows = []
    t0 = time.perf_counter()
    for k, c in enumerate(cfgs):
        bc = BaseCase(g=g, config=k, snsp=float(G.snsp_of(g, c.P)), P=c.P,
                      H=c.H, droop=c.droop, T_gov=c.T_gov)
        bnd = band or base.physical_band(bc)
        cost = nl.exact_cost(bc, PB.matrix(events, c.P, g), bnd, n_freq)
        row = {"config": k, "snsp": bc.snsp,
               "stored_energy": float(g.stored_energy(c.H)),
               "n_committed": len(c.committed),
               "band_hi": bnd[1], "f_peak": cost["f_peak"]}
        row.update({m: cost[m] for m in COSTS})
        row.update({m: float(risk.loc[k, m]) for m in RISKS})
        rows.append(row)
    df = pd.DataFrame(rows)
    meta = {"events": which, "n_freq": n_freq, "n_configs": len(df),
            "band": "per-configuration" if band is None else list(band),
            "seconds": time.perf_counter() - t0}
    return df, meta


def scores(df: pd.DataFrame, x: str, y: str) -> dict:
    """Spearman, and R^2 of a straight line fitted in log-log."""
    rho = stats.spearmanr(df[x], df[y]).statistic
    lx, ly = np.log10(df[x]), np.log10(df[y])
    r2 = stats.linregress(lx, ly).rvalue ** 2
    return {"spearman": rho, "r2_loglog": r2}


def report(df: pd.DataFrame, meta: dict) -> str:
    out = ["THE GREEN'S FUNCTION AS A PREDICTOR OF RISK", "=" * 74, "",
           "  no edges anywhere -- the frozen grid, 240 operating states", "",
           f"  events          {meta['events']}",
           f"  band            {meta['band']}"
           f"   ({df['band_hi'].min():.3f} to {df['band_hi'].max():.3f} Hz"
           " across the ensemble)",
           f"  frequencies     {meta['n_freq']} per configuration",
           f"  elapsed         {meta['seconds']:.0f} s", "",
           "  Spearman rho, and R^2 of a log-log straight line:", "",
           f"      {'predictor':16s}" + "".join(
               f"{r:>26s}" for r in RISKS), "",
           f"      {'':16s}" + "".join(f"{'rho':>12s}{'R2':>14s}"
                                       for _ in RISKS), ""]
    for p in COSTS + REFERENCE:
        line = f"      {p:16s}"
        for r in RISKS:
            s = scores(df, p, r)
            line += f"{s['spearman']:12.3f}{s['r2_loglog']:14.3f}"
        out.append(line)
    out += ["",
            "  spread of each quantity over the ensemble (max/min):", ""]
    for p in COSTS + REFERENCE + RISKS:
        out.append(f"      {p:16s} {df[p].max() / df[p].min():8.2f}x")
    best = max(COSTS, key=lambda p: abs(scores(df, p, "rocof_top5")["spearman"]))
    out += ["",
            f"  best Green's-function predictor of rocof_top5: {best}"
            f"  (rho {scores(df, best, 'rocof_top5')['spearman']:+.3f}"
            f", vs {scores(df, 'stored_energy', 'rocof_top5')['spearman']:+.3f}"
            " for system inertia)", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="the Green's function as a predictor of risk, no edges")
    ap.add_argument("--events", default="sourced")
    ap.add_argument("--n-freq", type=int, default=140)
    ap.add_argument("--band", type=float, nargs=2, default=None,
                    help="a common band in Hz; default is per-configuration")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    df, meta = run(which=args.events, n_freq=args.n_freq,
                   band=tuple(args.band) if args.band else None)
    text = report(df, meta)
    print(text)
    suffix = f"_{args.tag}" if args.tag else ""
    df.to_csv(HERE / f"predictor{suffix}.csv", index=False, lineterminator="\n")
    (HERE / f"predictor{suffix}_report.txt").write_text(text)
    pd.Series(meta).to_json(HERE / f"predictor{suffix}.json", indent=2)
    print(f"  written -> predictor{suffix}.csv / .json / _report.txt")


if __name__ == "__main__":
    main()
