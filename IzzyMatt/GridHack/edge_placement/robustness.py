"""Is the ranking a property of the grid, or of the choices made to score it?

Everything here is a reprocess of `sweep_curves.npz` -- no solving -- which is
why the curves are stored in the first place.  Four axes, and the first one is
not a formality:

  the band            The scoring band is the choice this study got wrong on its
                      first pass.  ../snsp regularises an algebraic bus with a
                      small mass, which invents modes above ~2.5 Hz, and an
                      H-infinity peak seeks out the worst frequency in whatever
                      band it is given.  Scored to 20 Hz, the baseline peak lands
                      on a regularisation mode and the winner is chosen to damp
                      something that does not exist.  `base.physical_band` cuts
                      the band where the modes stop converging as the
                      regularising mass is reduced.  The 5 Hz and 20 Hz rows are
                      kept as the control that shows the difference.
  the measurement lag ../snsp checks its ranking is unchanged at 50, 100 and
                      200 ms and reports that the unfiltered version is the only
                      one that reorders it.  Same check here.
  the RoCoF window    A window of T is blind to exactly 1/T Hz and its gain
                      2|sin(wT/2)|/T varies steeply nearby, so a ranking taken
                      near a null is fragile by construction.  This is the axis
                      most likely to move.
  the frequency grid  Re-scoring on every second stored point estimates whether
                      the H-infinity peaks are resolved.

The comparison is always against the folder's default scoring, over the new
edges only, and is reported three ways: Spearman on the whole ranking, overlap
of the top ten, and whether the winner survives.  A high Spearman with a changed
winner is a real outcome and worth seeing -- it means the field is flat near the
top, which is itself an answer to "how much does the choice of edge matter".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import base                                                  # noqa: E402
import sweep                                                 # noqa: E402


def compare(ref: pd.DataFrame, alt: pd.DataFrame, key: str) -> dict:
    """How much did the ranking move?"""
    a, b = ref[~ref["existing"]], alt[~alt["existing"]]
    rho = stats.spearmanr(a[key], b[key]).statistic
    ta, tb = set(a.nsmallest(10, key).index), set(b.nsmallest(10, key).index)
    win_a, win_b = a[key].idxmin(), b[key].idxmin()
    return {"spearman": rho, "top10": len(ta & tb),
            "winner": f"{int(alt.loc[win_b, 'a'])}-{int(alt.loc[win_b, 'b'])}",
            "same_winner": win_a == win_b,
            "best": float(b[key].min()),
            "peak_hz": float(b.loc[win_b, "f_peak_freq" if key == "r_C_freq"
                                   else "f_peak_rocof"])}


def variants(bc, cur) -> list[tuple[str, pd.DataFrame, dict]]:
    phys = base.physical_band(bc)
    out = [(f"default: band {phys[0]:g}-{phys[1]:.3g} Hz, lag 100 ms, window 500 ms",
            *sweep.score(bc, cur))]
    for hi in (1.0, 5.0, 20.0):
        out.append((f"band 0.01-{hi:g} Hz",
                    *sweep.score(bc, cur, band=(0.01, hi))))
    for lag in (0.05, 0.2):
        out.append((f"measurement lag {lag * 1e3:.0f} ms",
                    *sweep.score(bc, cur, t_meas=lag)))
    for tw in (0.25, 1.0):
        out.append((f"RoCoF window {tw * 1e3:.0f} ms",
                    *sweep.score(bc, cur, t_rocof=tw)))
    half = dict(cur)
    for k in ("sigma", "fro", "lin"):
        half[k] = cur[k][:, ::2]
    half["omega"], half["ref"] = cur["omega"][::2], cur["ref"][:, ::2]
    out.append(("every second frequency point", *sweep.score(bc, half)))
    return out


def report(bc, cur) -> str:
    vs = variants(bc, cur)
    ref_df, ref_meta = vs[0][1], vs[0][2]
    out = ["ROBUSTNESS OF THE RANKING", "=" * 88, "",
           f"  reprocessed from {cur['meta']['n_freq']} solved frequencies;"
           " no re-solving", "",
           f"  reference: {vs[0][0]}", ""]
    for key, label in (("r_C_freq", "FREQUENCY DEVIATION"),
                       ("r_C_rocof", "WINDOWED ROCOF")):
        out += [f"  {label}", "",
                f"      {'variant':<44s} {'rho':>7s} {'top10':>6s}"
                f" {'winner':>9s} {'best':>7s} {'peak Hz':>8s}", ""]
        for name, df, _ in vs:
            c = compare(ref_df, df, key)
            mark = " " if c["same_winner"] else "*"
            out += [f"      {name:<44s} {c['spearman']:7.3f} {c['top10']:5d}/10"
                    f" {c['winner']:>9s}{mark}{c['best']:7.4f}"
                    f" {c['peak_hz']:8.3f}"]
        out += ["", "      * winner differs from the reference", ""]

    out += ["  where the baseline's own peak sits, band by band", ""]
    for name, _, meta in vs:
        r = meta["baseline"]
        out += [f"      {name:<44s} C_freq peak {r['f_peak_freq']:7.3f} Hz,"
                f"  C_rocof peak {r['f_peak_rocof']:7.3f} Hz"]
    phys, art = base.artefact_floor(bc)
    out += ["", f"  highest physical mode {phys:.3f} Hz;"
            f"  lowest regularisation mode {art:.3f} Hz", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="robustness of the ranking")
    ap.add_argument("--tag", default="")
    ap.add_argument("--config", type=int, default=None)
    args = ap.parse_args()
    bc = base.load(config=args.config)
    cur = sweep.load_curves(tag=args.tag)
    text = report(bc, cur)
    print(text)
    suffix = f"_{args.tag}" if args.tag else ""
    (HERE / f"robustness{suffix}_report.txt").write_text(text)
    print(f"  written -> robustness{suffix}_report.txt")


if __name__ == "__main__":
    main()
