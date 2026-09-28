"""Score every candidate edge over the band.

Stages 1 and 2 of the spec collapsed into one pass, for the reason set out in
the README: at N = 100 with B a 20-column disturbance set, sigma_max is cheap
enough that there is nothing to shortlist.  Per frequency this does one O(N^3)
inverse and then, for each of the 4950 pairs, a rank-one update of the 100 x 20
matrix Pi G B and the top eigenvalue of its 20 x 20 Gram matrix.

WHAT IS STORED, AND WHY IT IS THE WHOLE CURVE

The solve writes `sweep_curves.npz`: sigma_max, the Frobenius norm and the
row-sum norm of Pi G_new B for **every candidate at every frequency**, plus the
same three for the base network.  Roughly 30 MB, and it is what makes this
folder usable: the band, the measurement lag, the RoCoF window and the choice
of norm are then all *reprocessing*, not another pass through the solver.  Given
that the first run of this study discovered its own band was wrong (see
`base.artefact_floor`), that is not a convenience.

Four scores come out of the reprocessing:

    C_freq     max over the band of |Omega_f| sigma_max(Pi G_new B)
    C_rocof    the same with the windowed-RoCoF weight
    H2         the band-limited H2 norm -- the spec's Frobenius screen, kept
               *not* as a filter but so that §7.3's argument can be shown on
               this grid rather than cited: if H2 and sigma_max disagree about
               the winner, that disagreement is the result
    C_Linf       max_i sum_j |.|, the exact worst case when each bus's dP is
               individually bounded

Because both measurement weights are scalars times the identity, one
sigma_max(omega) curve serves both objectives (see base.weight_rocof), so
C_freq and C_rocof differ only in *where in the band* they take their maximum.

ADAPTIVE REFINEMENT

A fixed frequency grid systematically under-estimates an H-infinity peak.  All
candidates share one grid, so refinement is done on the union of the peaks:
every distinct argmax of sigma_max -- taken over the whole grid and again over
the physical sub-band, since the two need not coincide -- is bracketed by its
neighbours and extra points inserted there.  Refining around sigma_max rather
than around a weighted peak keeps the stored curve neutral about the weighting,
which is the point of storing it; the weights are smooth on the scale of a
resonance, so a weighted peak never lands far from an unweighted one.
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
import greens as gr                                          # noqa: E402
from base import BaseCase, TWO_PI                            # noqa: E402

sys.path.insert(0, str(base.SNSP))
import perturbations as PB                                   # noqa: E402

CHUNK = 512
SCORES = ("C_freq", "C_rocof", "C_H2", "C_H2rocof", "C_Linf")


def disturbances(bc: BaseCase, which: str = "sourced"):
    """B: the disturbance patterns actually credited.

    ``sourced`` / ``fixed`` / ``scaled`` are the three frozen ../snsp sets, so an
    answer here is directly comparable with that study.  ``identity`` is the
    control: every bus perturbed on its own, i.e. the "worst case over
    everything" end of spec §7.4, against which the frozen sets measure how much
    the answer depends on crediting only realistic patterns.
    """
    if which == "identity":
        return np.eye(bc.N), [f"bus{i}" for i in range(bc.N)]
    events = PB.load(bc.g, base.SNSP, verify=True, which=which)
    return PB.matrix(events, bc.P, bc.g), [e.name for e in events]


# ---------------------------------------------------------------------------
# the solve


def norms_at(bc: BaseCase, B: np.ndarray, pairs: np.ndarray, kappa: float,
             omega: float):
    """(sigma_max, Frobenius, row-sum) for every candidate and for no edge."""
    Pi = bc.coi_projector()
    G = gr.green(bc, omega)
    X = G @ B
    PG, PX = Pi @ G, Pi @ X
    sig, fro, lin = (np.empty(len(pairs)) for _ in range(3))
    for lo in range(0, len(pairs), CHUNK):
        sl = slice(lo, lo + CHUNK)
        Y = gr.apply_to(G, X, PG, PX, pairs[sl], kappa)
        sig[sl] = gr.sigma_max_stack(Y)
        fro[sl] = gr.frobenius_stack(Y)
        lin[sl] = gr.row_sum_max(Y)
    Y0 = PX[None]
    ref = (gr.sigma_max_stack(Y0)[0], gr.frobenius_stack(Y0)[0],
           gr.row_sum_max(Y0)[0])
    return sig, fro, lin, ref, gr.near_cancellation(G, pairs, kappa)


def refinement_points(evaluated: np.ndarray, peaks: np.ndarray,
                      n_sub: int = 4) -> np.ndarray:
    """Extra frequencies bracketing every distinct peak in `peaks`."""
    ev = np.unique(evaluated)
    out = []
    for w in np.unique(peaks):
        i = int(np.searchsorted(ev, w))
        lo, hi = ev[max(i - 1, 0)], ev[min(i + 1, len(ev) - 1)]
        if hi > lo:
            out.append(np.geomspace(lo, hi, n_sub + 2)[1:-1])
    if not out:
        return np.empty(0)
    new = np.unique(np.concatenate(out))
    return new[~np.isin(new, ev)]


def solve(bc: BaseCase, which: str = "sourced", kappa: float = base.KAPPA,
          n_freq: int = 400, refine: int = 2, lo: float = base.BAND_HZ[0],
          hi: float = base.BAND_HZ[1]) -> dict:
    """Sweep the whole band once and keep the curves."""
    B, names = disturbances(bc, which)
    pairs, joined = base.candidates(bc)
    omegas = base.band(bc, lo, hi, n_freq)

    cols: dict[float, tuple] = {}
    t0 = time.perf_counter()
    flagged = np.zeros(len(pairs), bool)

    def evaluate(ws) -> None:
        for w in ws:
            sig, fro, lin, ref, flag = norms_at(bc, B, pairs, kappa, float(w))
            cols[float(w)] = (sig, fro, lin, ref)
            flagged[:] |= flag

    evaluate(omegas)
    # Refine inside the scored band only.  The wider bands exist as controls,
    # and they are wrong for a structural reason -- their peak sits on a
    # regularisation mode -- which no amount of resolution changes, so spending
    # frequencies to pin those peaks down buys nothing.  Refining everywhere is
    # what made the first attempt at this run take an hour.
    phys_hi = base.physical_band(bc)[1] * TWO_PI
    for _ in range(refine):
        w_ev = np.array(sorted(cols))
        inside = w_ev <= phys_hi
        if inside.sum() < 3:
            break
        S = np.stack([cols[w][0] for w in w_ev[inside]], axis=1)
        extra = refinement_points(w_ev, w_ev[inside][S.argmax(axis=1)], n_sub=3)
        if not len(extra):
            break
        evaluate(extra)
    elapsed = time.perf_counter() - t0

    w_ev = np.array(sorted(cols))
    return {"omega": w_ev,
            "sigma": np.stack([cols[w][0] for w in w_ev], 1).astype(np.float32),
            "fro": np.stack([cols[w][1] for w in w_ev], 1).astype(np.float32),
            "lin": np.stack([cols[w][2] for w in w_ev], 1).astype(np.float32),
            "ref": np.array([cols[w][3] for w in w_ev], float).T,
            "pairs": pairs, "existing": joined, "flagged": flagged,
            "meta": {"events": which, "n_events": int(B.shape[1]),
                     "kappa": kappa, "solve_band_hz": [lo, hi],
                     "n_freq": int(len(w_ev)), "n_coarse": n_freq,
                     "config": bc.config, "snsp": bc.snsp,
                     "digest": base.digest(bc), "seconds": elapsed,
                     "event_names": names}}


# ---------------------------------------------------------------------------
# the reprocessing


def score(bc: BaseCase, cur: dict, band: tuple[float, float] | None = None,
          t_meas: float = base.T_MEAS, t_rocof: float = base.T_ROCOF
          ) -> tuple[pd.DataFrame, dict]:
    """Turn stored curves into scores, for a given band and measurement."""
    band = band or base.physical_band(bc)
    w = cur["omega"]
    keep = (w >= band[0] * TWO_PI) & (w <= band[1] * TWO_PI)
    if keep.sum() < 3:
        raise ValueError(f"band {band} contains only {keep.sum()} grid points")
    w = w[keep]

    s = 1j * w
    wf = np.abs(s / (TWO_PI * (1.0 + s * t_meas)))
    wr = wf * np.abs(2.0 * np.sin(w * t_rocof / 2.0)) / t_rocof
    dw = np.gradient(w) / np.pi                # 2 x dw/2pi: the negative half

    sig, fro, lin = (cur[k][:, keep].astype(float) for k in ("sigma", "fro", "lin"))
    ref_sig, ref_fro, ref_lin = cur["ref"][:, keep]

    def peak(curve, weight):
        v = curve * weight
        i = v.argmax(axis=-1)
        return v.max(axis=-1), w[i] / TWO_PI

    jf, at_f = peak(sig, wf)
    jr, at_r = peak(sig, wr)
    ref_jf, ref_at_f = peak(ref_sig, wf)
    ref_jr, ref_at_r = peak(ref_sig, wr)
    pairs = cur["pairs"]
    df = pd.DataFrame({
        "a": pairs[:, 0], "b": pairs[:, 1], "existing": cur["existing"],
        "length": base.edge_length(bc, pairs),
        "role_a": [str(bc.g.role[i]) for i in pairs[:, 0]],
        "role_b": [str(bc.g.role[i]) for i in pairs[:, 1]],
        "C_freq": jf, "C_rocof": jr, "f_peak_freq": at_f, "f_peak_rocof": at_r,
        "C_H2": np.sqrt((dw * (wf * fro) ** 2).sum(axis=-1)),
        "C_H2rocof": np.sqrt((dw * (wr * fro) ** 2).sum(axis=-1)),
        "C_Linf": (lin * wf).max(axis=-1), "flagged": cur["flagged"]})
    ref = {"C_freq": float(ref_jf), "C_rocof": float(ref_jr),
           "C_H2": float(np.sqrt((dw * (wf * ref_fro) ** 2).sum())),
           "C_H2rocof": float(np.sqrt((dw * (wr * ref_fro) ** 2).sum())),
           "C_Linf": float((ref_lin * wf).max()),
           "f_peak_freq": float(ref_at_f), "f_peak_rocof": float(ref_at_r)}
    for c in SCORES:
        df[f"r_{c}"] = df[c] / ref[c]
    meta = dict(cur["meta"], band_hz=list(band), t_meas=t_meas,
                t_rocof=t_rocof, n_band_points=int(keep.sum()), baseline=ref)
    return df, meta


def save_curves(cur: dict, out: Path = HERE, tag: str = "") -> None:
    suffix = f"_{tag}" if tag else ""
    np.savez_compressed(out / f"sweep_curves{suffix}.npz", omega=cur["omega"],
                        sigma=cur["sigma"], fro=cur["fro"], lin=cur["lin"],
                        ref=cur["ref"], pairs=cur["pairs"],
                        existing=cur["existing"], flagged=cur["flagged"],
                        meta=np.array(repr(cur["meta"])))


def load_curves(out: Path = HERE, tag: str = "") -> dict:
    suffix = f"_{tag}" if tag else ""
    z = dict(np.load(out / f"sweep_curves{suffix}.npz", allow_pickle=True))
    z["meta"] = eval(str(z["meta"]))          # noqa: S307 -- our own repr
    return z


# ---------------------------------------------------------------------------


def report(bc: BaseCase, df: pd.DataFrame, meta: dict) -> str:
    ref = meta["baseline"]
    new = df[~df["existing"]]
    phys, art = base.artefact_floor(bc)
    out = ["SINGLE-EDGE SWEEP", "=" * 78, "",
           f"  operating point   config #{meta['config']}, SNSP {meta['snsp']:.1%}",
           f"  disturbances      {meta['events']}, {meta['n_events']} columns",
           f"  kappa             {meta['kappa']} p.u.",
           f"  band scored       {meta['band_hz'][0]:g} to {meta['band_hz'][1]:.4g} Hz"
           f"   ({meta['n_band_points']} of {meta['n_freq']} solved points)",
           f"  highest physical mode {phys:.3f} Hz;"
           f" lowest regularisation mode {art:.3f} Hz",
           f"  measurement       lag {meta['t_meas']} s,"
           f" RoCoF window {meta['t_rocof']} s",
           f"  candidates        {len(df)} ({len(new)} new,"
           f" {int(df['existing'].sum())} reinforcements)",
           f"  flagged           {int(df['flagged'].sum())}"
           "  (near-cancelling denominator)", "",
           "  no edge at all:", "",
           f"      C_freq  {ref['C_freq']:10.4f}   peak at"
           f" {ref['f_peak_freq']:7.3f} Hz",
           f"      C_rocof {ref['C_rocof']:10.4f}   peak at"
           f" {ref['f_peak_rocof']:7.3f} Hz",
           f"      C_H2 {ref['C_H2']:10.4f}",
           f"      C_Linf    {ref['C_Linf']:10.4f}", ""]

    for key, label in (("r_C_freq", "frequency deviation"),
                       ("r_C_rocof", "windowed RoCoF"),
                       ("r_C_H2", "H2 (the screen)")):
        out += [f"  best ten by {label}  ({key}, 1.000 = no edge):", ""]
        out += [f"      {int(r.a):3d}-{int(r.b):<3d} {r.role_a:>5s}/{r.role_b:<5s}"
                f" len {r.length:4.1f}   {getattr(r, key):.4f}"
                f"   [freq {r.r_C_freq:.4f}  rocof {r.r_C_rocof:.4f}"
                f"  H2 {r.r_C_H2:.4f}]"
                for r in new.sort_values(key).head(10).itertuples()] + [""]

    out += ["  Braess: candidates that make it worse", ""]
    for key, label in (("r_C_freq", "C_freq "), ("r_C_rocof", "C_rocof"),
                       ("r_C_H2", "H2     ")):
        n = int((new[key] > 1.0).sum())
        out += [f"      by {label}  {n:4d} of {len(new)}"
                f"  ({n / len(new):6.1%}),  worst {new[key].max():.4f}"]
    out += [""]

    rho = new[[f"r_{c}" for c in SCORES]].corr(method="spearman")
    out += ["  do the scores agree?  (Spearman over the new edges)", "",
            "      " + "  ".join(f"{c[2:]:>10s}" for c in rho.columns)]
    for name, row in rho.iterrows():
        out += ["      " + "  ".join(f"{v:10.4f}" for v in row) + f"   {name[2:]}"]
    top = {c: set(new.nsmallest(20, f"r_{c}").index) for c in SCORES}
    out += ["",
            f"      top-20 overlap  C_freq vs C_rocof: "
            f"{len(top['C_freq'] & top['C_rocof'])}/20",
            f"      top-20 overlap  C_freq vs H2     : "
            f"{len(top['C_freq'] & top['C_H2'])}/20", ""]

    best = new.loc[new["r_C_freq"].idxmin()]
    ex = df[df["existing"]].nsmallest(1, "r_C_freq")
    out += ["  the winner", "",
            f"      buses {int(best.a)} ({best.role_a}) - {int(best.b)}"
            f" ({best.role_b}),  lattice length {best.length:.2f}",
            f"      C_freq  x{best.r_C_freq:.4f}   C_rocof x{best.r_C_rocof:.4f}"
            f"   H2 x{best.r_C_H2:.4f}   C_Linf x{best.r_C_Linf:.4f}",
            f"      rank by C_rocof "
            f"{int((new['r_C_rocof'] < best.r_C_rocof).sum()) + 1} of {len(new)}",
            f"      best reinforcement of an existing circuit:"
            f" x{float(ex['r_C_freq'].iloc[0]):.4f}"
            f"  (buses {int(ex['a'].iloc[0])}-{int(ex['b'].iloc[0])})", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="score every candidate edge")
    ap.add_argument("--events", default="sourced",
                    choices=("sourced", "fixed", "scaled", "identity"))
    ap.add_argument("--kappa", type=float, default=base.KAPPA)
    ap.add_argument("--config", type=int, default=None)
    ap.add_argument("--snsp", type=float, default=None)
    ap.add_argument("--n-freq", type=int, default=400)
    ap.add_argument("--refine", type=int, default=2)
    ap.add_argument("--band", type=float, nargs=2, default=None,
                    help="scoring band in Hz; default is base.physical_band")
    ap.add_argument("--t-meas", type=float, default=base.T_MEAS)
    ap.add_argument("--t-rocof", type=float, default=base.T_ROCOF)
    ap.add_argument("--tag", default="")
    ap.add_argument("--analyse-only", action="store_true",
                    help="re-score stored curves without re-solving")
    args = ap.parse_args()

    bc = base.load(config=args.config, snsp=args.snsp)
    suffix = f"_{args.tag}" if args.tag else ""
    if args.analyse_only:
        cur = load_curves(tag=args.tag)
    else:
        cur = solve(bc, which=args.events, kappa=args.kappa,
                    n_freq=args.n_freq, refine=args.refine)
        save_curves(cur, tag=args.tag)
    df, meta = score(bc, cur, band=tuple(args.band) if args.band else None,
                     t_meas=args.t_meas, t_rocof=args.t_rocof)
    text = report(bc, df, meta)
    print(text)
    df.to_csv(HERE / f"sweep{suffix}.csv", index=False, lineterminator="\n")
    (HERE / f"sweep{suffix}_report.txt").write_text(text)
    pd.Series(meta).to_json(HERE / f"sweep{suffix}.json", indent=2)
    print(f"  written -> sweep{suffix}.csv / .json / _report.txt"
          f"  (curves in sweep_curves{suffix}.npz)")


if __name__ == "__main__":
    main()
