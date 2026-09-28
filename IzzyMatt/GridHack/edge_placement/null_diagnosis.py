"""Why C_rocof does not rank RoCoF risk -- and what happens when the RoCoF
window's blind spot is taken out of the band.

THE CLAIM UNDER TEST

`fig7_cost_vs_risk` reports Spearman +0.14 between C_rocof and measured RoCoF
risk, i.e. essentially nothing, over 45 candidates.  Two things were wrong with
that number and only one of them is a sampling problem:

  the sample   The 45 were drawn from the top and bottom of each ranking, which
               is a design for spanning the *risk*, not a sample entitled to a
               correlation.  `nonlinear.py --n-uniform/--n-strata` fixes that.

  the cost     A 500 ms RoCoF window has gain 2|sin(wT/2)|/T, which is exactly
               ZERO at f = 1/T = 2.000 Hz.  The scored band runs to 2.155 Hz, so
               the null sits INSIDE it, and the baseline C_rocof peaks at
               1.814 Hz -- already on the shoulder, where the gain has fallen to
               1.15 on its way to 0.

               Adding a line stiffens the network, so an edge pushes that mode
               UP, toward the null.  The ten `top C_rocof` candidates all move
               it to ~1.99 Hz, where the window gain is 0.06 instead of 1.15.
               Their weighted peak then falls back onto a lesser feature at
               1.3121 Hz and C_rocof reports a 5x improvement.

               The question this file settles is whether that 5x is REAL.  A
               500 ms RoCoF genuinely cannot see a 2 Hz oscillation -- it
               averages over exactly one period -- so the null is a property of
               the measurement, not a modelling error, and the cost is not
               simply lying.  What it is doing is making the score
               hypersensitive: a 10% shift in one mode's frequency moves the
               cost by 5x.  A real disturbance excites a broad band and a real
               grid rings at more than one mode, so the time domain cannot
               deliver anything like that.  A cost that swings 5x over a few
               percent of real risk cannot rank -- not because it is wrong at
               any one candidate, but because its dynamic range is dominated by
               a term the risk does not share.

               If that diagnosis is right, the cure is not a different band but
               a different NORM: an H-infinity peak samples the weight at one
               frequency and inherits the null's full leverage, while an H2
               integral of the same weighted curve averages the null away.
               `C_H2rocof` -- the RoCoF weight under an integral -- is already
               computed for all 4784 candidates, so this costs nothing to test.

THE TEST

Re-score every candidate on a band capped BELOW the null and correlate both
versions against the risk that was actually measured.  Capping at 1.90 Hz is not
a fitted choice: 1.827 Hz is the highest physical mode, so nothing real is being
discarded, and 2.0 Hz is the null.  Everything between them is headroom the band
never needed.

Curves for all 4784 candidates at every frequency are already on disk, so every
re-score here is free -- no solve is repeated.

Predictors are the SCREENED costs throughout, because the screen is what is
known before any re-solve and so what a shortlist would really be built from;
it tracks the exact cost at Spearman +0.93 (`nonlinear_report.txt` section 1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import base                                                  # noqa: E402
import nonlinear as nlmod                                    # noqa: E402
import sweep                                                 # noqa: E402

#: Cap the band just above the highest physical mode (1.827 Hz) and below the
#: 500 ms window's null (2.000 Hz).
CAP_HZ = 1.90

#: A candidate whose weighted RoCoF peak has fallen this far below the baseline
#: peak (1.814 Hz) is not reporting the dominant resonance any more -- the null
#: has eaten it and the peak has dropped back onto a lesser feature.
FLED_BELOW_HZ = 1.60


def boot_ci(x, y, n_boot: int = 4000, seed: int = 7):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = stats.spearmanr(x, y).statistic
    if len(x) < 30:
        return r, np.nan, np.nan
    rng = np.random.default_rng(seed)
    xy = np.column_stack([x, y])
    bs = [stats.spearmanr(*xy[rng.integers(0, len(xy), len(xy))].T).statistic
          for _ in range(n_boot)]
    return r, *np.percentile(bs, [2.5, 97.5])


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "wide"
    nl = pd.read_csv(HERE / f"nonlinear_{tag}.csv")
    bc = base.load(config=128)
    cur = sweep.load_curves()

    full, meta = sweep.score(bc, cur)
    capped, _ = sweep.score(bc, cur, band=(meta["band_hz"][0], CAP_HZ))
    band_top = meta["band_hz"][1]

    cols = ["a", "b", "r_C_rocof", "r_C_freq", "r_C_H2", "r_C_H2rocof",
            "r_C_Linf", "f_peak_rocof", "f_peak_freq"]
    j = nl.merge(full[cols], on=["a", "b"], how="left")
    j = j.merge(capped[["a", "b", "r_C_rocof", "f_peak_rocof"]], on=["a", "b"],
                how="left", suffixes=("", "_cap"))

    unb = j["source"].isin(nlmod.UNBIASED)
    y = "r_R_rocof_top5"
    L = []
    p = L.append

    p("WHY C_rocof DOES NOT RANK RoCoF RISK")
    p("=" * 78)
    p("")
    p(f"  {len(j)} candidates re-solved in the time domain"
      f"  ({int(unb.sum())} of them a plain random draw)")
    p(f"  band as scored   {meta['band_hz'][0]:g} to {band_top:.4g} Hz")
    p(f"  window null      {1 / base.T_ROCOF:.3f} Hz   (T_rocof"
      f" = {base.T_ROCOF} s)  -- INSIDE the band")
    p(f"  capped band      {meta['band_hz'][0]:g} to {CAP_HZ:g} Hz"
      f"   (above the 1.827 Hz top physical mode, below the null)")
    p("")

    p("  1. THE HONEST CORRELATION, on the random draw only")
    p("")
    p(f"      {'predictor':28s} {'rho':>8s}  {'95% CI':>18s}   {'n':>4s}")
    p("")
    for lab, c in (("C_rocof   peak, RoCoF weight", "r_C_rocof"),
                   ("C_rocof   same, band to 1.90", "r_C_rocof_cap"),
                   ("C_freq    peak, no null in wt", "r_C_freq"),
                   ("C_Linf    peak, row-sum norm", "r_C_Linf"),
                   ("C_H2      INTEGRAL, freq weight", "r_C_H2"),
                   ("C_H2rocof INTEGRAL, RoCoF weight", "r_C_H2rocof")):
        d = j.loc[unb & j[c].notna()]
        r, lo, hi = boot_ci(d[c], d[y])
        ci = f"[{lo:+.3f}, {hi:+.3f}]" if np.isfinite(lo) else ""
        p(f"      {lab:28s} {r:+8.3f}  {ci:>18s}   {len(d):4d}")
    p("")

    # -- who actually fled the null -------------------------------------------
    fled = j["f_peak_rocof"] < FLED_BELOW_HZ
    p("  2. WHICH CANDIDATES HID IN THE NULL")
    p("")
    p(f"      A candidate whose weighted peak sits below {FLED_BELOW_HZ} Hz is")
    p("      no longer reporting the dominant resonance at all.")
    p("")
    p(f"      {'group':34s} {'n':>5s} {'mean C_rocof':>13s}"
      f" {'mean C_freq':>12s} {'mean risk':>10s}")
    p("")
    for lab, k in (("peak fled the null", fled),
                   ("peak stayed put", ~fled)):
        d = j[k]
        if len(d):
            p(f"      {lab:34s} {len(d):5d} {d['r_C_rocof'].mean():13.3f}"
              f" {d['r_C_freq'].mean():12.3f} {d[y].mean():10.4f}")
    p("")
    p("      C_rocof says the fled group is far better and C_freq -- whose")
    p("      weight has no null -- says it is barely better.  The gap between")
    p("      those two columns is the size of the artefact.")
    p("")

    p("  3. THE CORRELATION WITH THE ARTEFACT REMOVED")
    p("")
    p(f"      {'sample':40s} {'rho':>8s}  {'95% CI':>18s}  {'n':>4s}")
    p("")
    for lab, k, c in (
            ("random draw, as scored", unb, "r_C_rocof"),
            ("random draw, peak-stayed-put only", unb & ~fled, "r_C_rocof"),
            ("random draw, capped band", unb, "r_C_rocof_cap"),
            ("all rows, as scored", pd.Series(True, index=j.index),
             "r_C_rocof"),
            ("all rows, capped band", pd.Series(True, index=j.index),
             "r_C_rocof_cap")):
        d = j.loc[k & j[c].notna()]
        if len(d) < 10:
            continue
        r, lo, hi = boot_ci(d[c], d[y])
        ci = f"[{lo:+.3f}, {hi:+.3f}]" if np.isfinite(lo) else ""
        p(f"      {lab:40s} {r:+8.3f}  {ci:>18s}  {len(d):4d}")
    p("")

    p("  4. DOES THE CAPPED COST STILL FIND THE GOOD EDGES?")
    p("")
    best = j.nsmallest(10, y)[["a", "b", "source", "r_C_rocof",
                               "r_C_rocof_cap", y]]
    n = len(j)
    best = best.assign(
        rank_as_scored=j["r_C_rocof"].rank().loc[best.index].astype(int),
        rank_capped=j["r_C_rocof_cap"].rank().loc[best.index].astype(int))
    p(f"      the ten best edges by MEASURED RoCoF risk, and where each cost"
      f" ranks them (of {n})")
    p("")
    p(f"      {'edge':>9s} {'risk':>8s} {'C_rocof':>9s} {'rank':>6s}"
      f" {'C_rocof*':>9s} {'rank*':>6s}    * = capped band")
    p("")
    for r_ in best.itertuples():
        p(f"      {int(r_.a):3d}-{int(r_.b):<5d} {getattr(r_, y):8.4f}"
          f" {r_.r_C_rocof:9.3f} {r_.rank_as_scored:6d}"
          f" {r_.r_C_rocof_cap:9.3f} {r_.rank_capped:6d}")
    p("")
    for c, lab in (("r_C_rocof", "as scored"), ("r_C_rocof_cap", "capped"),
                   ("r_C_H2rocof", "H2rocof")):
        top20 = set(map(tuple, j.nsmallest(20, c)[["a", "b"]].to_numpy()))
        hit = set(map(tuple, j.nsmallest(20, y)[["a", "b"]].to_numpy()))
        p(f"      top-20 overlap with the true top 20, {lab:10s}:"
          f" {len(top20 & hit):2d}/20")
    p("")

    text = "\n".join(L)
    print(text)
    (HERE / "null_diagnosis_report.txt").write_text(text)
    j.to_csv(HERE / "null_diagnosis.csv", index=False, lineterminator="\n")
    print("\n  written -> null_diagnosis_report.txt / null_diagnosis.csv")


if __name__ == "__main__":
    main()
