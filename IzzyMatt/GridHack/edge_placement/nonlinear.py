"""Stage 5: re-solve the shortlist honestly, then measure the actual risk.

TWO DIFFERENT QUANTITIES, AND THEY MUST NOT SHARE A LETTER

  the cost, C     A property of the Green's function: an H-infinity peak of
                  sigma_max(Pi G B) over a band, weighted by a measurement.
                  It is cheap, it ranks all 4950 candidates in one pass, and it
                  is *not* a risk.  It is a quantity we reason physically ought
                  to correlate with risk, and this module is where that
                  reasoning gets tested rather than assumed.

  the risk, R     What the grid actually does.  Integrate the full response to
                  the frozen disturbance set and measure it the way ../snsp
                  measures it -- the 500 ms RoCoF and the peak frequency
                  excursion at the *worst five buses*, averaged over events.
                  These are the vertical axes of ../snsp's own
                  fig11_topbus_vs_predictor, i.e. the quantities that study drew
                  its conclusions on, and they are the ones an added edge has to
                  be judged against.

`rocof_top5` and `dev_top5` are the headline pair.  `top1` is carried because a
grid code binds on a single worst bus, and the two need not agree: a redistributing
intervention can lower the five-bus sum while raising the single worst.

WHY THE SHORTLIST HAS TO BE RE-SOLVED AT ALL

The screen adds kappa u u^T to L and stops, which is exact for the operator but
holds theta* fixed.  In truth the power has another path, the angles shift, and
every synchronising coefficient moves with them: on this base case one edge
changes **422 entries of L**, not the 4 the rank-one form touches, and moves
theta* by up to 0.5 rad.  So the screen's ranking is a hypothesis, and both the
exact linear cost and the true risk are computed here.

Candidates are taken from the top of each ranking, plus the *worst* by C_freq
and a random sample, so the comparison spans the range rather than only the part
every score already agrees is good.
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
import greens as gr                                          # noqa: E402
import sweep                                                 # noqa: E402
from base import BaseCase, TWO_PI                             # noqa: E402

sys.path.insert(0, str(base.SNSP))
import dynamics as dyn                                       # noqa: E402

#: The risk measures, in the order they are reported.  The first two are the
#: vertical axes of ../snsp/figures/fig11_topbus_vs_predictor.png.
RISKS = ("rocof_top5", "dev_top5", "rocof_top1", "dev_top1", "nadir_top5", "J")


def exact_cost(bc: BaseCase, B: np.ndarray, band: tuple[float, float],
               n_freq: int = 140) -> dict:
    """C_freq, C_rocof and C_H2 for a grid, solved directly -- no rank-one update."""
    Pi = bc.coi_projector()
    w = TWO_PI * np.geomspace(band[0], band[1], n_freq)
    sig = np.empty(n_freq)
    fro = np.empty(n_freq)
    for k, wk in enumerate(w):
        Y = Pi @ (gr.green(bc, float(wk)) @ B)
        sig[k] = np.linalg.svd(Y, compute_uv=False)[0]
        fro[k] = float(np.linalg.norm(Y))
    s = 1j * w
    wf = np.abs(s / (TWO_PI * (1.0 + s * base.T_MEAS)))
    wr = wf * np.abs(2.0 * np.sin(w * base.T_ROCOF / 2.0)) / base.T_ROCOF
    dw = np.gradient(w) / np.pi
    return {"C_freq": float((sig * wf).max()), "C_rocof": float((sig * wr).max()),
            "C_H2": float(np.sqrt((dw * (wf * fro) ** 2).sum())),
            "f_peak": float(w[(sig * wf).argmax()] / TWO_PI)}


def risk(bc: BaseCase, dP: np.ndarray) -> dict:
    """What the grid actually does, measured the way ../snsp measures it."""
    r = dyn.respond(bc.net, dP)
    out = dyn.summarise(r, bc.N)
    out["J"] = r.total
    ok, worst = dyn.post_fault(bc.net, dP[:, int(np.argmax(np.abs(dP).sum(0)))])
    out["post_fault_ok"] = bool(ok)
    out["worst_line_load"] = float(worst)
    return out


def shortlist(df: pd.DataFrame, n_top: int = 12, n_worst: int = 8,
              n_random: int = 12, n_uniform: int = 0, n_strata: int = 0,
              n_bins: int = 10, seed: int = 20260910) -> pd.DataFrame:
    """Which candidates get re-solved, and -- just as important -- why.

    The first four sources are a *design*, not a sample: they deliberately take
    the extremes of each ranking so the re-solve spans the range.  That is the
    right choice for "how much risk can an edge buy", and the wrong one for
    "does the cost predict the risk", because a Spearman over a deliberately
    bimodal draw estimates nothing about the 4784.  Two honest samples are
    therefore carried alongside:

      uniform     A plain random draw.  This is the only group whose rank
                  correlation is an unbiased estimate of the population's, and
                  it is the one the correlation should be quoted on.
      stratified  Equal numbers per bin of the *screened* C_rocof, which fills
                  in the middle of the cost axis that the design leaves empty
                  (the top/worst groups put 27 of 45 in the bottom decile).
                  Its rho is inflated by construction -- it is for coverage of
                  the scatter and for within-bin comparisons, not for a number.

    Binning on the screened cost, not the exact one, is deliberate: the screen
    is what is known for all 4784 before any re-solve, so the strata are a
    genuine a-priori design rather than a post-hoc selection on the answer.
    """
    new = df[~df["existing"]]
    picks: dict[int, str] = {}
    for key, label in (("r_C_freq", "top C_freq"), ("r_C_rocof", "top C_rocof"),
                       ("r_C_H2", "top C_H2")):
        for i in new.nsmallest(n_top, key).index:
            picks.setdefault(i, label)
    for i in new.nlargest(n_worst, "r_C_freq").index:
        picks.setdefault(i, "worst C_freq")
    rng = np.random.default_rng(seed)
    for i in rng.choice(new.index, size=n_random, replace=False):
        picks.setdefault(int(i), "random")
    # Separate streams, so adding one sample never re-rolls another and the
    # original 45 reproduce exactly.
    if n_uniform:
        for i in np.random.default_rng(seed + 1).choice(
                new.index, size=n_uniform, replace=False):
            picks.setdefault(int(i), "uniform")
    if n_strata:
        rs = np.random.default_rng(seed + 2)
        q = np.quantile(new["r_C_rocof"], np.linspace(0.0, 1.0, n_bins + 1))
        which = np.digitize(new["r_C_rocof"], q[1:-1])
        for k in range(n_bins):
            pool = new.index[which == k]
            if len(pool):
                for i in rs.choice(pool, size=min(n_strata, len(pool)),
                                   replace=False):
                    picks.setdefault(int(i), "stratified")
    out = df.loc[list(picks)].copy()
    out["source"] = [picks[i] for i in out.index]
    return out


#: Sources that are plain random draws from the 4784, and so the only ones a
#: population rank correlation may be quoted on.  `random` and `uniform` differ
#: only by which rng stream drew them.
UNBIASED = ("random", "uniform")


def run(bc: BaseCase, df: pd.DataFrame, meta: dict, n_top: int = 12,
        n_uniform: int = 0, n_strata: int = 0
        ) -> tuple[pd.DataFrame, dict]:
    B, _ = sweep.disturbances(bc, meta["events"])
    band = tuple(meta["band_hz"])
    kappa = meta["kappa"]

    ref_c, ref_r = exact_cost(bc, B, band), risk(bc, B)
    rows = []
    t0 = time.perf_counter()
    picks = shortlist(df, n_top=n_top, n_uniform=n_uniform, n_strata=n_strata)
    for n_done, r in enumerate(picks.itertuples()):
        if n_done and n_done % 50 == 0:
            el = time.perf_counter() - t0
            print(f"    {n_done}/{len(picks)}  {el:.0f} s elapsed,"
                  f" ~{el / n_done * (len(picks) - n_done):.0f} s left",
                  flush=True)
        a, b = int(r.a), int(r.b)
        bce = base.with_edge(bc, a, b, kappa)
        c, rk = exact_cost(bce, B, band), risk(bce, B)
        row = {"a": a, "b": b, "source": r.source, "length": r.length,
               "role_a": r.role_a, "role_b": r.role_b,
               "screen_C_freq": r.r_C_freq, "screen_C_rocof": r.r_C_rocof,
               "screen_C_H2": r.r_C_H2,
               "theta_shift": float(np.abs(bce.theta0 - bc.theta0).max()),
               "post_fault_ok": rk["post_fault_ok"],
               "worst_line_load": rk["worst_line_load"]}
        for k in ("C_freq", "C_rocof", "C_H2"):
            row[f"exact_{k}"] = c[k] / ref_c[k]
        for k in RISKS:
            row[f"R_{k}"] = rk[k]
            row[f"r_R_{k}"] = rk[k] / ref_r[k]
        rows.append(row)
    out = pd.DataFrame(rows)
    info = {"seconds": time.perf_counter() - t0, "n": len(out),
            "baseline_cost": ref_c,
            "baseline_risk": {k: ref_r[k] for k in RISKS},
            "band_hz": list(band), "kappa": kappa, "events": meta["events"],
            "config": meta["config"], "snsp": meta["snsp"]}
    return out, info


def report(bc: BaseCase, df: pd.DataFrame, info: dict) -> str:
    def rho(x, y):
        return stats.spearmanr(df[x], df[y]).statistic

    ref = info["baseline_risk"]
    out = ["STAGE 5 -- NONLINEAR RE-SOLVE, AND THE RISK IT ACTUALLY BUYS",
           "=" * 82, "",
           f"  {info['n']} candidates re-solved with the edge present,"
           f" {info['seconds']:.0f} s",
           f"  config #{info['config']}, SNSP {info['snsp']:.1%},"
           f" band {info['band_hz'][0]:g}-{info['band_hz'][1]:.4g} Hz,"
           f" kappa {info['kappa']}, events {info['events']}",
           f"  theta* shift, max over candidates:"
           f" {df['theta_shift'].max():.4f} rad", "",
           "  risk with no edge at all  (../snsp measures, sourced set):", ""]
    units = {"rocof_top5": "Hz/s", "rocof_top1": "Hz/s", "dev_top5": "Hz",
             "dev_top1": "Hz", "nadir_top5": "Hz", "J": "Hz^2 s"}
    for k in RISKS:
        out += [f"      {k:12s} {ref[k]:10.4f} {units[k]}"]
    out += ["",
            "  1. does the exact cost agree with the screened cost?", "",
            f"      Spearman, screen vs exact   C_freq  "
            f"{rho('screen_C_freq', 'exact_C_freq'):+.3f}",
            f"                                  C_rocof "
            f"{rho('screen_C_rocof', 'exact_C_rocof'):+.3f}",
            f"                                  C_H2    "
            f"{rho('screen_C_H2', 'exact_C_H2'):+.3f}", "",
            "  2. does the cost predict the RISK?"
            "   (Spearman over the re-solved candidates)", "",
            f"      {'':12s}" + "".join(f"{k:>13s}" for k in RISKS), ""]
    for c in ("exact_C_freq", "exact_C_rocof", "exact_C_H2"):
        out += [f"      {c[6:]:12s}"
                + "".join(f"{rho(c, 'r_R_' + k):+13.3f}" for k in RISKS)]
    out += ["",
            "  2b. ON WHICH SAMPLE?  the same rho, split by how the candidate"
            " was drawn", "",
            "      Only the `unbiased` row estimates the population's rank"
            " correlation.",
            "      The `design` row is a bimodal draw and its rho is an"
            " artefact of that draw.", "",
            f"      {'sample':22s} {'n':>4s}"
            + "".join(f"{c:>10s}" for c in ("C_freq", "C_rocof", "C_H2"))
            + f"   {'C_rocof vs dev':>15s}", ""]
    groups = [("unbiased (random draw)", df["source"].isin(UNBIASED)),
              ("stratified on cost", df["source"] == "stratified"),
              ("design (top/worst)",
               ~df["source"].isin(UNBIASED + ("stratified",))),
              ("all rows pooled", pd.Series(True, index=df.index))]
    for name, k in groups:
        d = df[k]
        if len(d) < 5:
            continue
        cells = "".join(
            f"{stats.spearmanr(d[c], d['r_R_rocof_top5']).statistic:+10.3f}"
            for c in ("exact_C_freq", "exact_C_rocof", "exact_C_H2"))
        dv = stats.spearmanr(d["exact_C_rocof"], d["r_R_dev_top5"]).statistic
        out += [f"      {name:22s} {len(d):4d}{cells}   {dv:+15.3f}"]
    ub = df[df["source"].isin(UNBIASED)]
    if len(ub) >= 30:
        # A bootstrap interval, because the question is whether the weak rho is
        # weak or merely unresolved -- a point estimate cannot answer that.
        rng = np.random.default_rng(7)
        bs = np.array([
            stats.spearmanr(*np.transpose(
                ub[["exact_C_rocof", "r_R_rocof_top5"]].to_numpy()[
                    rng.integers(0, len(ub), len(ub))])).statistic
            for _ in range(4000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        r_ub = stats.spearmanr(ub["exact_C_rocof"],
                               ub["r_R_rocof_top5"]).statistic
        out += ["",
                f"      C_rocof vs rocof_top5 on the unbiased sample:"
                f" rho = {r_ub:+.3f}",
                f"      95% bootstrap interval  [{lo:+.3f}, {hi:+.3f}]"
                f"   (n = {len(ub)}, 4000 resamples)",
                f"      -> {'EXCLUDES' if lo > 0 or hi < 0 else 'includes'}"
                f" zero"]

    out += ["",
            "  3. how much risk does the best edge actually buy?", "",
            f"      {'measure':12s} {'best':>9s} {'worst':>9s}"
            f" {'spread':>9s}   {'best edge':>10s}", ""]
    for k in RISKS:
        col = f"r_R_{k}"
        b = df.loc[df[col].idxmin()]
        out += [f"      {k:12s} {df[col].min():9.4f} {df[col].max():9.4f}"
                f" {df[col].max() - df[col].min():9.4f}"
                f"   {int(b.a):3d}-{int(b.b):<3d}"]
    out += ["",
            f"      candidates that raise rocof_top5: "
            f"{int((df['r_R_rocof_top5'] > 1).sum())} of {len(df)}",
            f"      candidates that raise dev_top5  : "
            f"{int((df['r_R_dev_top5'] > 1).sum())} of {len(df)}",
            f"      sum against max: Spearman(rocof_top5, rocof_top1) = "
            f"{rho('r_R_rocof_top5', 'r_R_rocof_top1'):+.3f}", ""]

    out += ["  best ten by rocof_top5  (1.000 = no edge)", "",
            f"      {'edge':>9s} {'source':>14s} {'len':>5s}"
            f" {'rocof_t5':>9s} {'dev_t5':>8s} {'rocof_t1':>9s}"
            f" {'dev_t1':>8s} {'J':>8s} {'screen':>7s}", ""]
    for r in df.nsmallest(10, "r_R_rocof_top5").itertuples():
        out += [f"      {r.a:3d}-{r.b:<5d} {r.source:>14s} {r.length:5.1f}"
                f" {r.r_R_rocof_top5:9.4f} {r.r_R_dev_top5:8.4f}"
                f" {r.r_R_rocof_top1:9.4f} {r.r_R_dev_top1:8.4f}"
                f" {r.r_R_J:8.4f} {r.screen_C_freq:7.4f}"]
    out += ["",
            f"  post-fault equilibrium exists for all: "
            f"{bool(df['post_fault_ok'].all())}"
            f"   (worst line at {df['worst_line_load'].max():.3f} of its limit)",
            ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="nonlinear re-solve of the shortlist")
    ap.add_argument("--tag", default="",
                    help="which sweep curves to read (and, by default, the "
                         "output name)")
    ap.add_argument("--out-tag", default=None,
                    help="write under this name instead; the curves still come "
                         "from --tag, so a wider sample does not need a re-solve")
    ap.add_argument("--config", type=int, default=None)
    ap.add_argument("--n-top", type=int, default=12)
    ap.add_argument("--n-uniform", type=int, default=0,
                    help="extra plain-random candidates -- the only sample a "
                         "population rank correlation may be quoted on")
    ap.add_argument("--n-strata", type=int, default=0,
                    help="extra candidates per decile of the screened C_rocof, "
                         "to cover the middle of the cost axis")
    args = ap.parse_args()

    bc = base.load(config=args.config)
    suffix = f"_{args.out_tag or args.tag}" if (args.out_tag or args.tag) else ""
    cur = sweep.load_curves(tag=args.tag)
    df, meta = sweep.score(bc, cur)
    out, info = run(bc, df, meta, n_top=args.n_top,
                    n_uniform=args.n_uniform, n_strata=args.n_strata)
    text = report(bc, out, info)
    print(text)
    out.to_csv(HERE / f"nonlinear{suffix}.csv", index=False, lineterminator="\n")
    (HERE / f"nonlinear{suffix}_report.txt").write_text(text)
    pd.Series(info).to_json(HERE / f"nonlinear{suffix}.json", indent=2)
    print(f"  written -> nonlinear{suffix}.csv / .json / _report.txt")


if __name__ == "__main__":
    main()
