"""Put one edge in the grid and re-run the whole SNSP ensemble through it.

Everything else in this folder asks "which edge, at this operating point".  This
asks the complementary question, and it is the one that shows what an edge is
worth: take a single chosen edge, build it once, and re-run **all 240 balanced
operating states** of ../snsp against the same frozen disturbance set.

The result is two swarms on the same axes -- the ensemble as it is, and the
ensemble with one more circuit in it -- so the benefit is read as a shift of a
whole population rather than as one arrow on one point.  It also answers a
question the single-configuration study cannot: an edge is a permanent asset and
the grid runs at many operating points, so a candidate that wins at 70% SNSP has
to be shown not to lose at 40%.

WHAT IS AND IS NOT RECOMPUTED

The "before" swarm is read from ../snsp/ensemble_src.csv rather than recomputed.
That file is the published result, and `validate.py` already asserts that this
folder reproduces its rows to machine precision; `--check` re-verifies a sample
of them here before the comparison is drawn, so the two swarms are known to come
from the same code path.  Only the "after" swarm is solved, which halves the run.

The dispatch P is untouched, as everywhere else in this folder: each
configuration keeps its own commitment, its own inertia and its own wind
pattern, and the only thing that differs between the two swarms is one circuit.
theta* is re-solved per configuration, because the edge moves it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import base                                                  # noqa: E402

sys.path.insert(0, str(base.SNSP))
import configs as C                                          # noqa: E402
import dynamics as dyn                                       # noqa: E402
import grid as G                                             # noqa: E402
import perturbations as PB                                   # noqa: E402

#: The scores carried through, in the order they are reported.  The first two
#: are the vertical axes of ../snsp/figures/fig11_topbus_vs_predictor.png.
SCORES = ("rocof_top5", "dev_top5", "rocof_top1", "dev_top1", "nadir_top5")


def grid_with_edge(g: G.ToyGrid, a: int, b: int, kappa: float) -> G.ToyGrid:
    """The frozen grid with one more circuit in it."""
    import dataclasses
    A, K = g.A.copy(), g.Kij.copy()
    A[a, b] = A[b, a] = 1.0
    K[a, b] = K[b, a] = K[a, b] + kappa
    return dataclasses.replace(
        g, A=A, Kij=K,
        line_class={**g.line_class, (min(a, b), max(a, b)): "added"})


def run(a: int, b: int, kappa: float = base.KAPPA, which: str = "sourced",
        check: int = 5) -> tuple[pd.DataFrame, dict]:
    g = G.load(base.SNSP, verify=True)
    ge = grid_with_edge(g, a, b, kappa)
    cfgs = C.ensemble(g)
    events = PB.load(g, base.SNSP, verify=True, which=which)
    before = pd.read_csv(base.SNSP / "ensemble_src.csv").set_index("config")

    # Same code path as the published run?  Check a sample before trusting it.
    worst = 0.0
    for k in np.linspace(0, len(cfgs) - 1, check).astype(int):
        c = cfgs[int(k)]
        s = dyn.summarise(dyn.respond(c.network(g),
                                      PB.matrix(events, c.P, g)), g.N)
        worst = max(worst, max(abs(s[m] / before.loc[int(k), m] - 1.0)
                               for m in SCORES))
    if worst > 1e-9:
        raise RuntimeError(f"cannot reproduce ensemble_src.csv (worst {worst:.2e})"
                           " -- do not trust the 'before' swarm")

    rows = []
    t0 = time.perf_counter()
    for c in cfgs:
        dP = PB.matrix(events, c.P, g)          # the hazard follows the dispatch,
        s = dyn.summarise(dyn.respond(c.network(ge), dP), g.N)   # not the topology
        row = {"config": c.index, "snsp_target": c.snsp_target,
               "snsp": float(G.snsp_of(g, c.P)),
               "stored_energy": float(g.stored_energy(c.H)),
               "n_committed": len(c.committed)}
        for m in SCORES:
            row[m] = s[m]
            row[f"r_{m}"] = s[m] / before.loc[c.index, m]
        rows.append(row)
    after = pd.DataFrame(rows)
    meta = {"edge": [int(a), int(b)], "kappa": kappa, "events": which,
            "n_configs": len(after), "seconds": time.perf_counter() - t0,
            "reproduction_error": worst}
    return after, meta


def report(after: pd.DataFrame, meta: dict) -> str:
    a, b = meta["edge"]
    out = ["THE WHOLE ENSEMBLE, WITH ONE MORE CIRCUIT IN IT", "=" * 76, "",
           f"  edge {a}-{b}, kappa {meta['kappa']}, events {meta['events']}",
           f"  {meta['n_configs']} configurations re-solved,"
           f" {meta['seconds']:.0f} s",
           f"  'before' swarm reproduces ../snsp/ensemble_src.csv to"
           f" {meta['reproduction_error']:.1e}", "",
           "  risk with the edge, relative to the same configuration without it:",
           "",
           f"      {'measure':12s} {'mean':>8s} {'median':>8s} {'best':>8s}"
           f" {'worst':>8s}  {'improved':>10s}", ""]
    for m in SCORES:
        r = after[f"r_{m}"]
        out += [f"      {m:12s} {r.mean():8.4f} {r.median():8.4f} {r.min():8.4f}"
                f" {r.max():8.4f}  {int((r < 1).sum()):5d}/{len(r)}"]
    out += ["", "  by SNSP band (mean ratio):", "",
            f"      {'SNSP':>10s}" + "".join(f"{m:>13s}" for m in SCORES[:3]), ""]
    for lo, hi in ((0.30, 0.50), (0.50, 0.70), (0.70, 0.85), (0.85, 1.01)):
        k = (after["snsp_target"] >= lo) & (after["snsp_target"] < hi)
        if k.any():
            out += [f"      {100 * lo:3.0f}-{100 * hi:3.0f}%"
                    + "".join(f"{after.loc[k, f'r_{m}'].mean():13.4f}"
                              for m in SCORES[:3])]
    out += ["",
            "  the edge is chosen at one operating point; these rows are whether"
            " it survives the others.", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="the ensemble, with one edge added")
    ap.add_argument("--edge", type=int, nargs=2, default=None,
                    help="bus pair; default is the best by rocof_top5 in "
                         "nonlinear.csv")
    ap.add_argument("--kappa", type=float, default=base.KAPPA)
    ap.add_argument("--events", default="sourced")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.edge is None:
        nl = pd.read_csv(HERE / "nonlinear.csv")
        best = nl.loc[nl["r_R_rocof_top5"].idxmin()]
        args.edge = [int(best["a"]), int(best["b"])]
    after, meta = run(*args.edge, kappa=args.kappa, which=args.events)
    text = report(after, meta)
    print(text)
    suffix = f"_{args.tag}" if args.tag else ""
    after.to_csv(HERE / f"ensemble_edge{suffix}.csv", index=False,
                 lineterminator="\n")
    (HERE / f"ensemble_edge{suffix}_report.txt").write_text(text)
    pd.Series(meta).to_json(HERE / f"ensemble_edge{suffix}.json", indent=2)
    print(f"  written -> ensemble_edge{suffix}.csv / .json / _report.txt")


if __name__ == "__main__":
    main()
