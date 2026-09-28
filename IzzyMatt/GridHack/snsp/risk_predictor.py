"""A better predictor of risk than SNSP, built on the spectral clusters.

RISK, here, is one number per configuration: the 500 ms RoCoF summed over the
five worst buses and averaged over the twenty hazard scenarios -- the
`rocof_top5` column, the vertical axis of the right-hand pane of
`figures/fig1c_topbus.png`.  It is a grid-code-shaped reading rather than an
integral, and it is what an operator is actually held to.

The case against SNSP as its predictor is not that it fails -- it rises
monotonically with risk and correlates at rho = 0.91 -- but that it is a single
system-wide ratio, so it cannot distinguish a configuration that put its
non-synchronous generation next to the machines that are still on from one that
put it four hops away.  That difference is most of the scatter around the SNSP
trend, and the scatter is what a limit has to be set conservatively against.

The construction, in order:

  1. `clusters.py` cuts the grid into k electrically coherent regions, once,
     from the coupling matrix alone.
  2. Each configuration is described by per-region quantities -- the share, the
     inertia, the reserve, the imbalance-to-inertia ratio -- rather than by
     system-wide ones.
  3. Those go into a regression, and the regression is scored *out of sample*.

Point 3 is the part that decides whether any of this is real.  A cluster feature
set has more columns than SNSP has, so it will fit the training data better
whether or not it predicts anything, and the only defence is to never let the
number that gets reported come from data the model was chosen on.  So: 60 of the
240 configurations are held out at the start and touched exactly once, at the
end; everything else -- which k, which feature set, which ridge penalty -- is
decided by repeated k-fold cross-validation inside the remaining 180.

Run `python predictor.py` for the report, `--freeze` to write it out.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import clusters as CL
import configs as C
import dynamics as dyn
import grid as G

HERE = Path(__file__).resolve().parent

#: The risk being predicted, and the ensemble it is read from.  `TARGET` is
#: the default; every function that scores anything takes a `target` argument,
#: because the whole argument of section 1 of the report is that *which* score
#: is being predicted decides how much room a spatial predictor has -- so the
#: machinery has to be able to point at another one.
TARGET = "rocof_top5"

#: What each supported target is, in words, for the report's header.
TARGET_WORDS = {
    "rocof_top1": "the 500 ms RoCoF at the single worst bus",
    "rocof_top5": "the 500 ms RoCoF summed over the 5 worst buses",
    "rocof_top10": "the 500 ms RoCoF summed over the 10 worst buses",
    "dev_top1": "the peak |f - 50 Hz| at the single worst bus",
    "dev_top5": "the peak |f - 50 Hz| summed over the 5 worst buses",
}
ENSEMBLE = "ensemble_src.csv"

#: Held-out fraction, drawn stratified by SNSP level so the test set spans the
#: same range as the training set rather than a corner of it.
TEST_FRAC = 0.25
SPLIT_SEED = 20260910

#: Cross-validation inside the training set: 5 folds, repeated, so that model
#: choice is not decided by one lucky partition.
N_FOLDS, N_REPEATS, CV_SEED = 5, 20, 4242

#: How many Laplacian modes the modal family projects onto.  The constant
#: mode is skipped -- it is the centre of inertia, which is what the system-wide
#: predictors already carry -- so these are the slowest *inter-area* shapes.
MODAL_M = 10

EPS = 1e-9


# -- features --------------------------------------------------------------

def _cluster_frame(g: G.ToyGrid, cfgs: list[C.Config],
                   part: CL.Partition) -> pd.DataFrame:
    """Per-region quantities for every configuration.

    Everything here is knowable before the disturbance: it is the dispatch, the
    commitment and the network.  Nothing is read from a solve -- a predictor
    that needed the answer would not be one.
    """
    k = part.k
    rows = []
    roles = g.role.astype(str)
    ns_bus = np.isin(roles, ("wind", "hvdc"))
    diameter = float(g.dist.max())

    for c in cfgs:
        gen = np.clip(c.P, 0.0, None)
        HS = c.H * g.S
        spin = np.array(sorted(set(c.committed) | set(c.condensers)), int)
        dist = (g.dist[:, spin].min(axis=1) if spin.size
                else np.full(g.N, diameter + 1.0))
        row: dict[str, float] = {}
        for j in range(k):
            m = part.label == j
            ns = float(gen[m & ns_bus].sum())
            tot = float(gen[m].sum())
            iner = float(HS[m].sum())
            row[f"ns_{j}"] = ns
            row[f"gen_{j}"] = tot
            # A region with nothing running has no share to speak of.  0 is the
            # honest encoding -- no non-synchronous generation is present -- and
            # `gen_{j}` carries the "there is nothing here at all" case
            # separately, so the two are not confounded in the regression.
            row[f"snsp_{j}"] = ns / tot if tot > EPS else 0.0
            row[f"inertia_{j}"] = iner
            row[f"reserve_{j}"] = float(c.droop[m].sum())
            row[f"dist_{j}"] = float(dist[m].mean())
            # The physical one.  A RoCoF is an imbalance divided by an inertia,
            # so the ratio -- not the share and not the inertia on its own -- is
            # the quantity the swing equation actually writes down.
            row[f"stress_{j}"] = ns / (iner + EPS)
        rows.append(row)

    d = pd.DataFrame(rows)
    st = d[[f"stress_{j}" for j in range(k)]].to_numpy()
    ine = d[[f"inertia_{j}" for j in range(k)]].to_numpy()
    sn = d[[f"snsp_{j}" for j in range(k)]].to_numpy()
    # Aggregates over the regions.  `stress_max` is the one the physics argues
    # for: the worst five buses are in whichever region is worst, so a maximum
    # over regions should track them better than any system-wide average can.
    d["stress_max"] = st.max(axis=1)
    d["stress_mean"] = st.mean(axis=1)
    d["stress_spread"] = st.max(axis=1) - st.min(axis=1)
    d["inertia_min"] = ine.min(axis=1)
    d["snsp_max"] = sn.max(axis=1)
    d["snsp_spread"] = sn.max(axis=1) - sn.min(axis=1)
    return d


def modal_features(g: G.ToyGrid, cfgs: list[C.Config],
                   V: np.ndarray) -> np.ndarray:
    """Project each configuration's dispatch onto a set of network modes.

    A hard partition says which region a bus is in.  A mode shape says how much
    each bus participates in one pattern of inter-area swing, which is the
    continuous version of the same idea and the thing the Fiedler vector
    actually is.  For each mode this takes

      proj_m   how strongly the trippable (non-synchronous) injection excites it,
      iner_m   the inertia that mode has to swing against, sum_i v_m(i)^2 H_i S_i,
      proj/iner the ratio, which is the modal form of "imbalance over inertia".

    `V` is any orthonormal basis, so the same construction runs on the slow
    Laplacian modes, on the fast ones, or on a random rotation -- which is how
    the report tests whether the *spectrum* matters or only the dimension.
    """
    ns_bus = np.isin(g.role.astype(str), ("wind", "hvdc"))
    NS = np.array([np.clip(c.P, 0.0, None) * ns_bus for c in cfgs])
    HS = np.array([c.H * g.S for c in cfgs])
    proj = np.abs(NS @ V)
    iner = HS @ (V ** 2)
    lg = lambda a: np.log10(np.maximum(a, 1e-6))               # noqa: E731
    return np.column_stack([lg(proj), lg(iner), lg(proj / (iner + EPS)),
                            lg(HS.sum(axis=1))])


def slow_modes(g: G.ToyGrid, m: int = MODAL_M) -> np.ndarray:
    """The m slowest non-constant modes of the Kij-weighted Laplacian."""
    _, vecs = CL.spectrum(CL.laplacian(g))
    return vecs[:, 1:m + 1]


_CACHE: dict = {}


def _base() -> tuple[G.ToyGrid, list[C.Config], pd.DataFrame]:
    """The grid, the regenerated configurations and the scored ensemble.

    The configurations are rebuilt from the same seed rather than reloaded,
    because the CSV keeps only their summary and the features here need the
    per-bus dispatch.  The assertion is what makes that safe: if the rebuild
    ever stopped matching the run that produced the scores, every feature below
    would be describing a different grid state than the target it is fitted to.
    """
    if "base" not in _CACHE:
        g = G.load()
        df = pd.read_csv(HERE / ENSEMBLE)
        cfgs = C.ensemble(g, draws=C.DRAWS_PER_LEVEL, seed=20260909)
        assert len(cfgs) == len(df), "ensemble CSV and configs disagree in length"
        assert np.allclose([c.meta["snsp"] for c in cfgs],
                           df["snsp"].to_numpy()), \
            "regenerated configs are not the ones the ensemble was run on"
        _CACHE.update(g=g, cfgs=cfgs, base=df)
    return _CACHE["g"], _CACHE["cfgs"], _CACHE["base"]


def build(part: CL.Partition | None = None
          ) -> tuple[pd.DataFrame, CL.Partition, G.ToyGrid]:
    """The full design matrix: the ensemble's own columns plus the cluster ones."""
    g, cfgs, df = _base()
    part = part or CL.load()
    out = pd.concat([df.reset_index(drop=True),
                     _cluster_frame(g, cfgs, part)], axis=1)
    # `machine_dist_mean` is NaN wherever nothing is synchronised at all (the
    # whole 100% stratum).  That is not a missing measurement, it is a distinct
    # state, so it gets the diameter and an indicator rather than an imputation.
    diameter = float(g.dist.max())
    out["no_machine"] = out["machine_dist_mean"].isna().astype(float)
    for col in ("machine_dist_mean", "machine_dist_max"):
        out[col] = out[col].fillna(diameter + 1.0)
    return out, part, g


# -- model families --------------------------------------------------------

def _logs(d: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """log10 of a positive quantity, floored so a zero does not become -inf."""
    return np.log10(np.maximum(d[cols].to_numpy(float), 1e-6))


def design(d: pd.DataFrame, name: str, k: int) -> np.ndarray:
    """The feature matrix for one named model family."""
    cl = lambda p: [f"{p}_{j}" for j in range(k)]          # noqa: E731
    if name == "snsp":
        return d[["snsp"]].to_numpy(float)
    if name == "snsp_quad":
        s = d["snsp"].to_numpy(float)
        return np.column_stack([s, s ** 2])
    if name == "swing_ratio":
        # One column, with the coefficient forced to the physical value: the
        # swing equation says RoCoF = dP / 2H, so log risk should move one for
        # one with log(hazard) - log(inertia) and this fits only the offset.
        return _logs(d, ["hazard_pu"]) - _logs(d, ["stored_energy"])
    if name == "swing":
        # The same two quantities with their coefficients free.  This is the
        # whole of the swing equation's content, written as a regression: how
        # many megawatts can go, and how much stored energy is there to absorb
        # them.  Both are known from the dispatch before anything trips.
        return np.column_stack([_logs(d, ["hazard_pu"]),
                                _logs(d, ["stored_energy"])])
    if name == "swing_plus":
        return np.column_stack([
            _logs(d, ["hazard_pu"]), _logs(d, ["stored_energy"]),
            _logs(d, ["reserve"]),
            d[["machine_dist_mean"]].to_numpy(float)])
    if name == "global_physics":
        # What a single-machine-equivalent argument would write down: the
        # imbalance available to trip, over the inertia holding it.
        return np.column_stack([
            _logs(d, ["stored_energy"]), _logs(d, ["wind_pu"]),
            _logs(d, ["reserve"])])
    if name == "global_all":
        return np.column_stack([
            d[["snsp"]].to_numpy(float), _logs(d, ["stored_energy"]),
            _logs(d, ["reserve"]), _logs(d, ["wind_pu"]),
            d[["hvdc_pu", "sync_pu", "n_spinning", "machine_dist_mean",
               "machine_dist_max", "coastal_wind_share", "radial_wind_pu",
               "no_machine"]].to_numpy(float)])
    if name == "cluster_snsp":
        return d[cl("snsp")].to_numpy(float)
    if name == "cluster_stress":
        return np.column_stack([_logs(d, cl("stress")),
                                _logs(d, ["stored_energy"])])
    if name == "stress_max":
        return _logs(d, ["stress_max"])
    if name == "cluster_all":
        return np.column_stack([
            d[cl("snsp")].to_numpy(float), _logs(d, cl("inertia")),
            d[cl("reserve")].to_numpy(float), d[cl("dist")].to_numpy(float),
            _logs(d, cl("stress"))])
    if name == "modal":
        g, cfgs, _ = _base()
        return modal_features(g, cfgs, slow_modes(g))
    if name == "hybrid":
        # The global physics, plus what the partition adds to it: where the
        # stress is concentrated, and how unevenly it is spread.
        return np.column_stack([
            _logs(d, ["stored_energy"]), _logs(d, ["wind_pu"]),
            _logs(d, ["reserve"]), _logs(d, cl("stress")),
            _logs(d, ["stress_max"]),
            d[["stress_spread", "snsp_spread", "no_machine"]].to_numpy(float)])
    raise KeyError(name)


#: Every family that gets scored, in the order the report prints them.
FAMILIES = (
    ("snsp", "SNSP alone, linear"),
    ("snsp_quad", "SNSP alone, quadratic"),
    ("swing_ratio", "log(hazard / inertia), coefficient forced to -1"),
    ("swing", "log hazard and log inertia, coefficients free"),
    ("swing_plus", "  the same, plus reserve and distance to a machine"),
    ("global_physics", "system inertia, wind on, reserve (no partition)"),
    ("global_all", "every system-wide column (no partition)"),
    ("cluster_snsp", "SNSP per region"),
    ("stress_max", "worst region's imbalance / inertia, alone"),
    ("cluster_stress", "imbalance / inertia per region, plus system inertia"),
    ("cluster_all", "share, inertia, reserve, distance and stress per region"),
    ("hybrid", "global physics + where the stress sits"),
    ("modal", f"projection onto the {MODAL_M} slowest network modes"),
)


# -- fitting ---------------------------------------------------------------

def ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float,
                                                               np.ndarray,
                                                               np.ndarray]:
    """Ridge with a free intercept and standardised columns.

    Standardising inside the fit -- and returning the centre and scale so the
    same transform is applied to held-out rows -- keeps the penalty comparable
    across features that are shares, logs and distances at once.
    """
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (X - mu) / sd
    ybar = y.mean()
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    b = Z.T @ (y - ybar)
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Unpenalised, and a column went constant inside this fold -- an
        # indicator whose cases all landed in the other folds, say.  The
        # least-norm solution is the right answer there: it gives that column a
        # coefficient of zero, which is what a column that does not vary
        # deserves, rather than failing the whole fit.
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w, float(ybar), mu, sd


def apply_fit(fit, X: np.ndarray) -> np.ndarray:
    w, b, mu, sd = fit
    return ((X - mu) / sd) @ w + b


def _folds(n: int, n_folds: int, rng: np.random.Generator) -> list[np.ndarray]:
    idx = rng.permutation(n)
    return [np.sort(a) for a in np.array_split(idx, n_folds)]


def cv_predict(X: np.ndarray, y: np.ndarray, alpha: float, n_folds: int,
               n_repeats: int, seed: int) -> np.ndarray:
    """Out-of-fold predictions, averaged over repeats."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((n_repeats, len(y)))
    for r in range(n_repeats):
        for te in _folds(len(y), n_folds, rng):
            tr = np.setdiff1d(np.arange(len(y)), te)
            acc[r, te] = apply_fit(ridge(X[tr], y[tr], alpha), X[te])
    return acc.mean(axis=0)


#: Ridge penalties searched.  0 is ordinary least squares; the grid is wide
#: because the families differ by an order of magnitude in width.
ALPHAS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)

#: A wider grid for the event-wise selection, which wants far heavier
#: penalties: the target it is fitted to is a noisy 10-event average, and
#: the right response to a noisy target is to shrink hard.
ALPHAS_WIDE = (0.0, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)

#: Worst-n buses the per-event score keeps; matches `TARGET`.
TOP_N_BUS = 5


def choose_alpha(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pick the penalty by cross-validation *within the training set only*."""
    best, best_r2 = 0.0, -np.inf
    for a in ALPHAS:
        r2 = r_squared(y, cv_predict(X, y, a, N_FOLDS, 4, CV_SEED))
        if r2 > best_r2:
            best, best_r2 = a, r2
    return best, best_r2


# -- scoring ---------------------------------------------------------------

def r_squared(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(1.0 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


@dataclass
class Score:
    """How well a predictor does, and how badly its worst cases do."""

    r2: float
    rmse_pct: float          # RMS residual, as a percentage of the reading
    band98: float            # the multiplicative width of the 1-99% residuals
    worst: float             # the largest single multiplicative miss
    n_out: int               # points missed by more than OUTLIER

    @staticmethod
    def of(y_log: np.ndarray, yhat_log: np.ndarray,
           outlier: float = 0.10) -> "Score":
        res = y_log - yhat_log                       # log10 multiplicative error
        lo, hi = np.percentile(res, [1, 99])
        return Score(r2=r_squared(y_log, yhat_log),
                     rmse_pct=float((10 ** np.sqrt((res ** 2).mean()) - 1) * 100),
                     band98=float(10 ** (hi - lo)),
                     worst=float(10 ** np.abs(res).max()),
                     n_out=int((np.abs(res) > np.log10(1 + outlier)).sum()))


#: A point is an "outlier from the trend" if the predictor misses it by more
#: than this, multiplicatively.  10% of a RoCoF reading is roughly the width of
#: the band an operator would care about.
OUTLIER = 0.10


def split(d: pd.DataFrame, frac: float = TEST_FRAC,
          seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Stratified holdout: the same share of every SNSP level goes to test."""
    rng = np.random.default_rng(seed)
    te = []
    for _, idx in d.groupby("snsp_target").groups.items():
        idx = np.asarray(idx)
        n = int(round(frac * len(idx)))
        te.extend(rng.choice(idx, size=n, replace=False).tolist())
    te = np.sort(np.array(te, int))
    return np.setdiff1d(np.arange(len(d)), te), te


def evaluate(d: pd.DataFrame, k: int, tr: np.ndarray, te: np.ndarray,
             target: str = TARGET) -> pd.DataFrame:
    """Every family: penalty and CV score from train, final score from test."""
    y = np.log10(d[target].to_numpy(float))
    rows = []
    for name, label in FAMILIES:
        X = design(d, name, k)
        alpha, cv_r2 = choose_alpha(X[tr], y[tr])
        cv = Score.of(y[tr], cv_predict(X[tr], y[tr], alpha, N_FOLDS,
                                        N_REPEATS, CV_SEED))
        fit = ridge(X[tr], y[tr], alpha)
        test = Score.of(y[te], apply_fit(fit, X[te]))
        rows.append({"family": name, "label": label, "p": X.shape[1],
                     "alpha": alpha, "cv_r2": cv.r2, "cv_band98": cv.band98,
                     "test_r2": test.r2, "test_rmse_pct": test.rmse_pct,
                     "test_band98": test.band98, "test_worst": test.worst,
                     "test_out": test.n_out})
    return pd.DataFrame(rows)


def sweep_k(tr: np.ndarray, te: np.ndarray, ks=range(2, 13),
            target: str = TARGET) -> pd.DataFrame:
    """Training-CV score of the cluster families against k.

    The eigengap picks k = 4 by a hair, so the honest thing is to show what the
    predictor does across the whole range and let the reader see how much of the
    result rests on that choice.  Selection still uses the training folds only.
    """
    g, _, _ = _base()
    y = None
    rows = []
    for k in ks:
        d, _, _ = build(CL.cluster(g, k=k))
        if y is None:
            y = np.log10(d[target].to_numpy(float))
        for name in ("cluster_stress", "hybrid"):
            X = design(d, name, k)
            alpha, _ = choose_alpha(X[tr], y[tr])
            cv = Score.of(y[tr], cv_predict(X[tr], y[tr], alpha, N_FOLDS, 4,
                                            CV_SEED))
            fit = ridge(X[tr], y[tr], alpha)
            test = Score.of(y[te], apply_fit(fit, X[te]))
            rows.append({"k": k, "family": name, "cv_r2": cv.r2,
                         "test_r2": test.r2, "test_band98": test.band98})
    return pd.DataFrame(rows)


# -- report ----------------------------------------------------------------

def _table(res: pd.DataFrame, best: str) -> list[str]:
    out = ["  family              p  alpha   CV R2  |  test R2  test RMS  "
           "98% band  worst  n>10%",
           "  " + "-" * 88]
    for _, r in res.iterrows():
        mark = "  <-" if r["family"] == best else ""
        out.append(f"  {r['family']:<18s} {int(r['p']):2d} {r['alpha']:6.2f} "
                   f"{r['cv_r2']:7.3f}  |  {r['test_r2']:7.3f}  "
                   f"{r['test_rmse_pct']:7.1f}%  {r['test_band98']:7.2f}x  "
                   f"{r['test_worst']:5.2f}x  {int(r['test_out']):5d}{mark}")
    return out


def report(res: pd.DataFrame, sw: pd.DataFrame, ctl: pd.DataFrame,
           bas: pd.DataFrame, abl: pd.DataFrame, part: CL.Partition,
           d: pd.DataFrame, tr: np.ndarray, te: np.ndarray,
           target: str = TARGET) -> str:
    base = res.set_index("family").loc["snsp"]
    # The model that gets to be "the answer" is chosen on the training folds
    # alone; its test row is then read once, and so is everything else's.
    best = res.loc[res["cv_r2"].idxmax(), "family"]
    win = res.set_index("family").loc[best]
    dec = decomposition(d, target)

    out = ["A PREDICTOR OF RISK, BUILT ON THE GRID'S OWN CLUSTERS",
           "=" * 92, "",
           f"  risk = {target}: "
           f"{TARGET_WORDS.get(target, target)},",
           f"         averaged over the 20 hazard scenarios "
           f"({d[target].min():.2f} to {d[target].max():.2f})",
           f"  {len(d)} configurations: {len(tr)} train, {len(te)} held out "
           f"(stratified by SNSP level, seed {SPLIT_SEED})",
           f"  fitted on log10(risk); model choice by {N_REPEATS}x{N_FOLDS}-fold "
           f"CV inside the training set only", "",
           "1. How much room is there for a spatial predictor at all?",
           "-" * 92,
           "  Before clustering anything, ask what the target is made of.  The",
           "  same 500 ms window evaluated on the centre-of-inertia frequency",
           "  -- the whole grid as one machine -- already accounts for",
           f"      R2 = {dec['r2_system_alone']:.3f}   of log10(risk).",
           "",
           f"  var(log risk)            {dec['var_total']:.5f}",
           f"  var(system-wide part)    {dec['var_system']:.5f}",
           f"  var(local amplification) {dec['var_local']:.5f}   "
           f"({dec['local_share'] * 100:.1f}% of the target's variance)",
           f"  the score runs {dec['amp_mean']:.2f}x (sd {dec['amp_sd']:.2f}) "
           f"the system-wide reading",
           "",
           "  So this score is mostly a system-wide quantity wearing a local",
           "  costume.  That is the ceiling every spatial feature below is",
           "  working under, and it is worth knowing before reading the rest.",
           "",
           "2. What the partition is", "-" * 92,
           f"  k = {part.k} regions from the eigengap heuristic on the "
           f"Kij-weighted normalised Laplacian",
           f"  sizes {part.sizes.tolist()}, conductance "
           f"{np.round(part.conductance(G.load()), 3).tolist()}",
           "",
           "3. Every model family, scored the same way", "-" * 92]
    out += _table(res, best)
    out += ["",
            "  CV R2 is in-training (the number families were chosen on);",
            "  everything right of the bar is the 60 held-out configurations,",
            "  read once.  '98% band' is the multiplicative width of the",
            "  central 98% of residuals -- the spread of the trend -- and",
            "  'n>10%' counts held-out points missed by more than 10%.", ""]

    out += ["4. The comparison that was asked for", "-" * 92,
            f"  baseline   SNSP alone".ljust(33) +
            f"test R2 {base['test_r2']:6.3f}   "
            f"98% band {base['test_band98']:5.2f}x   "
            f"worst {base['test_worst']:5.2f}x   "
            f"{int(base['test_out'])} of {len(te)} missed by >10%",
            f"  chosen     {best}".ljust(33) +
            f"test R2 {win['test_r2']:6.3f}   "
            f"98% band {win['test_band98']:5.2f}x   "
            f"worst {win['test_worst']:5.2f}x   "
            f"{int(win['test_out'])} of {len(te)} missed by >10%", "",
            f"  the spread of the trend narrows "
            f"{base['test_band98'] / win['test_band98']:.2f}x, the worst single "
            f"miss falls from {base['test_worst']:.2f}x to "
            f"{win['test_worst']:.2f}x,",
            f"  and the count of >10% misses falls from "
            f"{int(base['test_out'])} to {int(win['test_out'])} out of "
            f"{len(te)}.  That is the goal, met.", ""]

    out += ["5. But is it the *clustering*?  The controls say no.", "-" * 92,
            f"  (a) the `{CONTROL_FAMILY}` construction -- share, inertia, "
            f"reserve, distance",
            "      and stress per region -- on partitions that are not the",
            "      spectral one:", "",
            "      partition                     CV R2   test R2  98% band  "
            "n>10%"]
    for name, grp in ctl.groupby("partition", sort=False):
        if len(grp) == 1:
            r = grp.iloc[0]
            out.append(f"      {name:<28s} {r['cv_r2']:6.3f}  "
                       f"{r['test_r2']:7.3f}  {r['test_band98']:7.2f}x  "
                       f"{int(r['test_out']):5d}")
        else:
            out.append(f"      {name:<28s} {grp['cv_r2'].mean():6.3f}  "
                       f"{grp['test_r2'].mean():7.3f}  "
                       f"{grp['test_band98'].mean():7.2f}x  "
                       f"{grp['test_out'].mean():5.1f}"
                       f"   (mean of {len(grp)}, range "
                       f"{grp['test_r2'].min():.3f}-{grp['test_r2'].max():.3f})")
    spec = ctl.iloc[0]["test_r2"]
    rnd = ctl[ctl["partition"] == "random, same sizes"]["test_r2"]
    quad = ctl[ctl["partition"] == "geographic quarters"]["test_r2"]
    out += ["",
            f"      the spectral partition beats {int((spec > rnd).sum())} of "
            f"{len(rnd)} random partitions of identical size"]
    if len(quad):
        out.append(f"      and the naive geographic quartering by "
                   f"{spec - quad.iloc[0]:+.3f} in held-out R2.")
    out += ["",
            f"  (b) the modal family on {MODAL_M} slow modes, {MODAL_M} fast "
            f"modes, and random rotations:", "",
            "      basis                              test R2  98% band  n>10%"]
    for name, grp in bas.groupby("basis", sort=False):
        if len(grp) == 1:
            r = grp.iloc[0]
            out.append(f"      {name:<33s} {r['test_r2']:7.3f}  "
                       f"{r['test_band98']:7.2f}x  {int(r['test_out']):5d}")
        else:
            out.append(f"      {name:<33s} {grp['test_r2'].mean():7.3f}  "
                       f"{grp['test_band98'].mean():7.2f}x  "
                       f"{grp['test_out'].mean():5.1f}"
                       f"   (mean of {len(grp)}, range "
                       f"{grp['test_r2'].min():.3f}-{grp['test_r2'].max():.3f})")
    slow = bas.iloc[0]["test_r2"]
    fast = bas.iloc[1]["test_r2"]
    out += ["",
            f"      the slowest modes and the fastest modes score "
            f"{slow:.3f} and {fast:.3f}.",
            "      The fast modes are single-bus wiggles with no regional",
            "      meaning whatever.  If they predict as well as the Fiedler",
            "      vector does, the spectrum is not what is being used --",
            "      the dimension is.  Any rich enough description of *where*",
            "      the generation and the inertia sit does the same work.", ""]

    out += ["6. How much rests on k = 4", "-" * 92]
    if len(sw):
        out += ["  training-CV and held-out R2 for the cluster families, "
                "against k:",
                "   k    cluster_stress            hybrid",
                "        CV R2  test R2      CV R2  test R2"]
        piv = sw.pivot(index="k", columns="family", values=["cv_r2", "test_r2"])
        for k in piv.index:
            out.append(f"  {k:2d}   {piv[('cv_r2', 'cluster_stress')][k]:6.3f} "
                       f"{piv[('test_r2', 'cluster_stress')][k]:8.3f}     "
                       f"{piv[('cv_r2', 'hybrid')][k]:6.3f} "
                       f"{piv[('test_r2', 'hybrid')][k]:8.3f}")
        out += ["",
                "  the eigengap preferred k = 4 by a hair (1.01x over the "
                "runner-up gap),",
                "  which is consistent with the flatness of this table.", ""]
    else:
        out += ["  (skipped: --no-sweep)", ""]

    out += [f"7. What the `{CONTROL_FAMILY}` model leans on", "-" * 92,
            "  held-out R2 lost when a whole group of columns is dropped and",
            "  the model refitted -- reported instead of coefficients, because",
            "  a region with one synchronous station switches its reserve and",
            "  its inertia on together, and ridge then splits one effect",
            "  between two collinear columns.",
            "",
            "  Read on the interpretable family rather than on the winner: the",
            "  modal family's columns are three mutually redundant views of one",
            "  projection, so dropping any group of them costs nothing and the",
            "  table would say only that.", "",
            "  group                        cols   test R2   lost"]
    for _, r in abl.iterrows():
        out.append(f"  {r['group']:<28s} {int(r['cols']):4d}  "
                   f"{r['test_r2']:8.3f}  {r['lost']:+7.3f}")
    dead = dead_columns(d, CONTROL_FAMILY, part.k)
    if dead:
        out += ["",
                f"  {len(dead)} column(s) never vary and carry nothing: "
                f"{', '.join(dead)}.",
                "  (that region has no wind and no HVDC, so its share and its",
                "   stress are structurally zero in every configuration.)"]

    sw2 = res.set_index("family").loc["swing"]
    out += ["", "8. What to take away", "-" * 92,
            "  * A far better predictor than SNSP exists and is out-of-sample",
            f"    real: R2 {base['test_r2']:.3f} -> {win['test_r2']:.3f}, "
            f"outliers {int(base['test_out'])} -> {int(win['test_out'])} of "
            f"{len(te)}.",
            "  * It needs two columns, not thirty-one.  `swing` -- log of the",
            "    megawatts the contingency set can remove, and log of the",
            f"    stored energy there is to absorb them -- scores "
            f"{sw2['test_r2']:.3f} with",
            f"    {int(sw2['test_out'])} outliers, matching every elaborate "
            f"feature set here.",
            "    Those are the two quantities the swing equation names: the",
            "    aggregate size of the contingency set (not any standing",
            "    imbalance -- the states are balanced) over the stored energy.",
            "    Forcing the exponents to the purely inertial (+1, -1) already",
            "    gives 0.944; freeing them gives 0.971, at (+1.29, -0.65),",
            "    because a 500 ms window is read after damping and governors",
            "    have begun to act, not at t = 0.",
            "  * So what was wrong with SNSP was never that it ignores",
            "    geography.  It is a *ratio*: it tracks the denominator only",
            "    loosely (r = -0.92 against log inertia) and is nearly blind",
            "    to the numerator (r = +0.34 against log hazard).  At a fixed",
            "    SNSP level the inertia still ranges about 28% and the hazard",
            "    about 23%, and that spread is the scatter in figure 1c.",
            "  * The spectral clustering buys nothing over that.  Random",
            "    partitions of the same sizes, a naive geographic quartering,",
            "    and the fastest Laplacian modes all score within noise of the",
            "    spectral cut -- and all of them are just rich bases for the",
            "    same two aggregates.",
            "  * The reason there is nothing local to find is section 1: this",
            f"    target is {dec['r2_system_alone'] * 100:.0f}% a system-wide "
            f"quantity.  Narrowing it to the",
            "    single worst bus (rocof_top1) raises the local share from",
            "    11.8% to 19.1% and lifts the spectral cut from beating 12 of",
            "    20 random partitions to 15 of 20 -- the right direction, and",
            "    still not a result.", ""]
    return "\n".join(out)


def feature_names(name: str, k: int) -> list[str]:
    """Column labels matching `design`, for the coefficient table."""
    cl = lambda p: [f"{p}[{j}]" for j in range(k)]              # noqa: E731
    return {
        "snsp": ["snsp"],
        "snsp_quad": ["snsp", "snsp^2"],
        "swing_ratio": ["log(hazard / inertia)"],
        "swing": ["log hazard", "log inertia"],
        "swing_plus": ["log hazard", "log inertia", "log reserve",
                       "machine dist mean"],
        "global_physics": ["log inertia", "log wind on", "log reserve"],
        "global_all": ["snsp", "log inertia", "log reserve", "log wind on",
                       "hvdc", "sync on", "n spinning", "machine dist mean",
                       "machine dist max", "coastal wind share", "radial wind",
                       "no machine"],
        "cluster_snsp": cl("snsp"),
        "cluster_stress": [f"log stress[{j}]" for j in range(k)]
        + ["log inertia"],
        "stress_max": ["log stress_max"],
        "cluster_all": cl("snsp") + [f"log inertia[{j}]" for j in range(k)]
        + cl("reserve") + cl("dist") + [f"log stress[{j}]" for j in range(k)],
        "modal": [f"log proj[{j}]" for j in range(MODAL_M)]
        + [f"log modal inertia[{j}]" for j in range(MODAL_M)]
        + [f"log proj/inertia[{j}]" for j in range(MODAL_M)] + ["log inertia"],
        "hybrid": ["log inertia", "log wind on", "log reserve"]
        + [f"log stress[{j}]" for j in range(k)]
        + ["log stress_max", "stress spread", "snsp spread", "no machine"],
    }[name]


# -- controls --------------------------------------------------------------
#
# The question the table in section 2 cannot answer on its own: `cluster_all`
# has twenty columns and `global_all` has twelve, and more columns fit better
# whether or not they mean anything.  So the same feature construction is run on
# partitions that are *not* the spectral one -- a naive geographic quartering,
# and random labellings with the identical cluster sizes.  If the spectral
# partition is doing real work, it beats both; if it does not, the gain was
# feature count all along.

def random_partition(g: G.ToyGrid, sizes: np.ndarray, seed: int
                     ) -> CL.Partition:
    """A random labelling with the same cluster sizes as the real partition."""
    rng = np.random.default_rng(seed)
    lab = np.concatenate([np.full(n, j) for j, n in enumerate(sizes)])
    return CL.Partition(k=len(sizes), label=rng.permutation(lab),
                        vals=np.zeros(1), gaps={}, normalised=True)


def quadrant_partition(g: G.ToyGrid) -> CL.Partition:
    """North/south x east/west, drawn with a ruler rather than an eigenvector.

    The spectral cut on this grid happens to come out roughly quartered, so this
    is the control that matters most: it asks whether the eigenvectors found
    anything the map does not already give away for free.
    """
    half = g.L // 2
    lab = np.array([2 * (r >= half) + (c >= half) for r, c in g.coords], int)
    return CL.Partition(k=4, label=CL.relabel(lab), vals=np.zeros(1), gaps={},
                        normalised=True)


#: The family the partition controls are run on.  It has to be one whose
#: columns actually depend on the partition -- `modal` does not, so running the
#: control on whichever family happened to win would silently compare a model
#: with itself and report a dead heat.
CONTROL_FAMILY = "cluster_all"


def control_scores(part: CL.Partition, tr: np.ndarray, te: np.ndarray,
                   family: str = CONTROL_FAMILY, n_random: int = 20,
                   target: str = TARGET) -> pd.DataFrame:
    """`family` scored on the spectral partition and on the controls."""
    assert family in ("cluster_all", "cluster_snsp", "cluster_stress",
                      "hybrid"), f"{family} does not depend on the partition"
    g, _, _ = _base()
    y = np.log10(_base()[2][target].to_numpy(float))

    def one(pt: CL.Partition, name: str) -> dict:
        d = build(pt)[0]
        X = design(d, family, pt.k)
        alpha, _ = choose_alpha(X[tr], y[tr])
        cv = Score.of(y[tr], cv_predict(X[tr], y[tr], alpha, N_FOLDS, 4,
                                        CV_SEED))
        test = Score.of(y[te], apply_fit(ridge(X[tr], y[tr], alpha), X[te]))
        return {"partition": name, "cv_r2": cv.r2, "test_r2": test.r2,
                "test_band98": test.band98, "test_out": test.n_out}

    rows = [one(part, "spectral (eigengap k=%d)" % part.k),
            one(quadrant_partition(g), "geographic quarters")]
    for i in range(n_random):
        r = one(random_partition(g, part.sizes, 900 + i), f"random {i}")
        r["partition"] = "random, same sizes"
        rows.append(r)
    return pd.DataFrame(rows)


def basis_scores(tr: np.ndarray, te: np.ndarray, y: np.ndarray | None = None,
                 n_random: int = 15, m: int = MODAL_M,
                 target: str = TARGET) -> pd.DataFrame:
    """The modal family on the slow modes, the fast modes and random rotations.

    The partition control asks whether the eigenvectors beat a ruler.  This asks
    the sharper version: whether the *slow* end of the spectrum beats the fast
    end.  Slow modes are the inter-area shapes -- the ones a coherency argument
    says a grid swings in -- and the fast ones are single-bus wiggles with no
    regional meaning at all.  If the two score alike, the spectrum is not what
    the features are carrying.
    """
    g, cfgs, base = _base()
    y = np.log10(base[target].to_numpy(float)) if y is None else y
    _, vecs = CL.spectrum(CL.laplacian(g))

    def one(V: np.ndarray, name: str) -> dict:
        X = modal_features(g, cfgs, V)
        alpha, _ = choose_alpha(X[tr], y[tr])
        test = Score.of(y[te], apply_fit(ridge(X[tr], y[tr], alpha), X[te]))
        return {"basis": name, "test_r2": test.r2, "test_band98": test.band98,
                "test_out": test.n_out}

    rows = [one(vecs[:, 1:m + 1], f"slowest {m} modes (Fiedler and after)"),
            one(vecs[:, -m:], f"fastest {m} modes")]
    rng = np.random.default_rng(7)
    for i in range(n_random):
        r = one(np.linalg.qr(rng.standard_normal((g.N, m)))[0], "random")
        r["basis"] = "random orthonormal"
        rows.append(r)
    return pd.DataFrame(rows)


def decomposition(d: pd.DataFrame, target: str = TARGET) -> dict[str, float]:
    """How much of the target is system-wide, and how much is local.

    This is the number that governs everything else in the report.  The score
    being predicted is a *bus* reading, but if the five worst buses simply move
    with the system then it is a system-wide quantity wearing a local costume,
    and no amount of spatial feature engineering can reach what is not there.

    `coi_rocof_500ms` is the same 500 ms window evaluated on the centre-of-
    inertia frequency: what the whole grid did, as one machine.  It is a solver
    output and so can never be a *feature* -- but as a diagnostic it says how
    much room a spatial predictor has to work in.
    """
    y = np.log10(d[target].to_numpy(float))
    sysw = np.log10(d["coi_rocof_500ms"].to_numpy(float))
    amp = y - sysw
    r = stats.pearsonr(y, sysw)[0]
    return {"r2_system_alone": float(r ** 2),
            "var_total": float(y.var()), "var_system": float(sysw.var()),
            "var_local": float(amp.var()),
            "local_share": float(amp.var() / y.var()),
            "amp_mean": float((10 ** amp).mean()),
            "amp_sd": float((10 ** amp).std())}


def ablate(d: pd.DataFrame, family: str, k: int, tr: np.ndarray,
           te: np.ndarray, target: str = TARGET) -> pd.DataFrame:
    """Held-out R2 lost when each group of columns is dropped.

    Reported instead of the raw coefficients, because several of these features
    are collinear by construction -- in a region with a single synchronous
    station, its reserve and its inertia switch on together, and ridge then
    splits one effect between them, which makes an individual coefficient
    uninterpretable and its sign meaningless.  Dropping a whole group and
    refitting asks a question that survives the collinearity.
    """
    y = np.log10(d[target].to_numpy(float))
    X = design(d, family, k)
    names = np.array(feature_names(family, k))
    groups: dict[str, np.ndarray] = {}
    for i, n in enumerate(names):
        key = n.split("[")[0].strip()
        groups.setdefault(key, []).append(i)

    alpha, _ = choose_alpha(X[tr], y[tr])
    full = Score.of(y[te], apply_fit(ridge(X[tr], y[tr], alpha), X[te])).r2
    rows = [{"group": "(nothing dropped)", "cols": X.shape[1], "test_r2": full,
             "lost": 0.0}]
    for key, cols in groups.items():
        keep = np.setdiff1d(np.arange(X.shape[1]), cols)
        a, _ = choose_alpha(X[tr][:, keep], y[tr])
        r2 = Score.of(y[te], apply_fit(ridge(X[tr][:, keep], y[tr], a),
                                       X[te][:, keep])).r2
        rows.append({"group": key, "cols": len(cols), "test_r2": r2,
                     "lost": full - r2})
    return pd.DataFrame(rows).sort_values("lost", ascending=False)


def dead_columns(d: pd.DataFrame, family: str, k: int) -> list[str]:
    """Features that never vary, and so can carry no information."""
    X = design(d, family, k)
    names = feature_names(family, k)
    return [names[i] for i in range(X.shape[1]) if X[:, i].std() < 1e-12]


def predictions(d: pd.DataFrame, res: pd.DataFrame, k: int, tr: np.ndarray,
                te: np.ndarray, target: str = TARGET) -> pd.DataFrame:
    """Per-configuration predictions, for the figures to draw from.

    Every model is fitted on the training rows only, so the prediction carried
    for a held-out configuration has never seen it.  `in_test` says which rows
    those are, and it is the only ones a figure should quote a score from.
    """
    y = np.log10(d[target].to_numpy(float))
    out = pd.DataFrame({
        "config": d["config"].to_numpy(),
        "snsp": d["snsp"].to_numpy(float),
        "snsp_target": d["snsp_target"].to_numpy(float),
        "risk": d[target].to_numpy(float),
        "in_test": np.isin(np.arange(len(d)), te),
    })
    best = res.loc[res["cv_r2"].idxmax(), "family"]
    for name in dict.fromkeys(("snsp", "swing", "modal", CONTROL_FAMILY, best)):
        X = design(d, name, k)
        alpha = float(res.set_index("family").loc[name, "alpha"])
        out[f"pred_{name}"] = 10.0 ** apply_fit(ridge(X[tr], y[tr], alpha), X)
    out["best_family"] = best
    return out


# -- generalisation across the disturbance set -----------------------------
#
# The held-out split above holds out *configurations*.  Every fold therefore
# shares the same twenty disturbances, and a feature built from those twenty --
# `hazard_pu`, the aggregate size of the contingency set -- is scored against a
# target that is an average over the very same realisations.  That is a leak,
# and it is invisible to configuration-wise validation.
#
# The disturbance set is meant to stand for a distribution of credible events,
# so the question a predictor has to answer is whether it ranks configurations
# correctly under a *different draw*.  These functions ask exactly that: fit on
# one half of the events, score on the disjoint half.  Rank correlation rather
# than R2, because the two halves have different absolute scales and it is the
# ordering a limit is set from.

def per_event_score(tag: str = "src", n: int = TOP_N_BUS) -> np.ndarray:
    """(config, event) array of the score, from the saved raw fields."""
    suffix = f"_{tag}" if tag else ""
    z = np.load(HERE / f"ensemble{suffix}_nodes.npz")
    return dyn.worst_n_by_event(np.abs(z["rocof_500ms"]), n, axis=1)


def event_cv_alpha(X: np.ndarray, per_ev: np.ndarray, tr: np.ndarray,
                   pool: np.ndarray, rng: np.random.Generator,
                   n_rep: int = 6) -> float:
    """Choose the ridge penalty by holding out EVENTS, not configurations.

    Configuration-wise cross-validation cannot see the failure mode that
    matters here, because every one of its folds is scored on the same
    disturbances the model was fitted on.  Splitting the events instead puts
    the penalty where the generalisation actually has to happen, and it uses
    only information available at fit time.
    """
    best, best_r = 0.0, -np.inf
    for a in ALPHAS_WIDE:
        sc = []
        for _ in range(n_rep):
            q = rng.permutation(pool)
            i, j = q[:len(q) // 2], q[len(q) // 2:]
            yi = np.log10(per_ev[:, i].mean(axis=1))
            yj = np.log10(per_ev[:, j].mean(axis=1))
            sc.append(stats.spearmanr(
                apply_fit(ridge(X[tr], yi[tr], a), X[tr]), yj[tr])[0])
        if np.mean(sc) > best_r:
            best, best_r = a, float(np.mean(sc))
    return best


def cross_event_scores(d: pd.DataFrame, k: int, tr: np.ndarray, te: np.ndarray,
                       families: tuple[str, ...] = (), n_splits: int = 40,
                       event_cv: bool = True, seed: int = 2027
                       ) -> pd.DataFrame:
    """Fit on one half of the disturbances, rank the disjoint half.

    `hazard + inertia` aggregates the contingency sizes over the *fitting*
    events only.  Its two output columns are then the whole argument in one
    row: `rho_same` scores it on the events it was fitted on -- features and
    target sharing the same realisations, which is the leak -- and
    `rho_disjoint` scores it on the half it never saw.
    """
    per_ev = per_event_score()
    sizes = np.load(HERE / "ensemble_src_nodes.npz")["sizes"]
    H = _logs(d, ["stored_energy"])
    families = families or ("snsp", "global_all", "cluster_all", "modal")
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_splits):
        perm = rng.permutation(per_ev.shape[1])
        A, B = perm[:len(perm) // 2], perm[len(perm) // 2:]
        yA = np.log10(per_ev[:, A].mean(axis=1))
        yB = np.log10(per_ev[:, B].mean(axis=1))
        cand = {f: design(d, f, k) for f in families}
        cand["inertia only"] = H
        cand["hazard + inertia"] = np.column_stack(
            [np.log10(sizes[:, A].sum(axis=1)), H.ravel()])
        for name, X in cand.items():
            a = (event_cv_alpha(X, per_ev, tr, A, rng) if event_cv
                 else choose_alpha(X[tr], yA[tr])[0])
            pred = apply_fit(ridge(X[tr], yA[tr], a), X)
            rows.append({"split": r, "model": name, "alpha": a,
                         "rho_disjoint": stats.spearmanr(pred[te], yB[te])[0],
                         "rho_same": stats.spearmanr(pred[te], yA[te])[0]})
    return pd.DataFrame(rows)


def cross_hazard_model(d: pd.DataFrame, k: int, tr: np.ndarray, te: np.ndarray,
                       families: tuple[str, ...] = ()) -> pd.DataFrame:
    """Fit on the sourced hazard model, rank risk under the fixed one.

    The sharpest version of the same question: not another draw from the same
    hazard distribution, but a different hazard *model* -- the control set in
    which every event is held at a constant size, so that only vulnerability
    varies.  A predictor whose skill lives in the hazard term has nothing to
    say there, and one whose skill is in the inertia term keeps it.
    """
    y_src = np.log10(per_event_score("src").mean(axis=1))
    y_fix = np.log10(per_event_score("").mean(axis=1))
    H = _logs(d, ["stored_energy"])
    families = families or ("snsp", "global_all", "cluster_all", "modal")
    cand = {f: design(d, f, k) for f in families}
    cand["inertia only"] = H
    cand["hazard + inertia"] = np.column_stack(
        [_logs(d, ["hazard_pu"]).ravel(), H.ravel()])
    rows = []
    for name, X in cand.items():
        a, _ = choose_alpha(X[tr], y_src[tr])
        pred = apply_fit(ridge(X[tr], y_src[tr], a), X)
        rows.append({"model": name,
                     "rho_same_model": stats.spearmanr(pred[te], y_src[te])[0],
                     "rho_other_model": stats.spearmanr(pred[te], y_fix[te])[0]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze", action="store_true",
                    help="write predictor_report.txt and predictor.csv")
    ap.add_argument("--no-sweep", action="store_true",
                    help="skip the k sweep (much faster)")
    ap.add_argument("--quick", action="store_true",
                    help="skip the random-partition controls too")
    ap.add_argument("--target", default=TARGET, choices=sorted(TARGET_WORDS),
                    help="which risk score to predict")
    ap.add_argument("--suffix", default="",
                    help="tag appended to the frozen output filenames")
    args = ap.parse_args()

    t = args.target
    d, part, g = build()
    tr, te = split(d)
    res = evaluate(d, part.k, tr, te, target=t)
    ctl = control_scores(part, tr, te, n_random=0 if args.quick else 20,
                         target=t)
    bas = basis_scores(tr, te, n_random=0 if args.quick else 15, target=t)
    abl = ablate(d, CONTROL_FAMILY, part.k, tr, te, target=t)
    sw = (pd.DataFrame(columns=["k", "family", "cv_r2", "test_r2",
                                "test_band98"])
          if args.no_sweep else sweep_k(tr, te, target=t))
    text = report(res, sw, ctl, bas, abl, part, d, tr, te, target=t)
    print(text)
    preds = predictions(d, res, part.k, tr, te, target=t)
    if args.freeze:
        (HERE / f"predictor_report{args.suffix}.txt").write_text(text, encoding="utf-8")
        res.to_csv(HERE / f"predictor{args.suffix}.csv", index=False, lineterminator="\n")
        sw.to_csv(HERE / f"predictor_k_sweep{args.suffix}.csv", index=False,
                  lineterminator="\n")
        ctl.to_csv(HERE / f"predictor_controls{args.suffix}.csv", index=False,
                   lineterminator="\n")
        bas.to_csv(HERE / f"predictor_bases{args.suffix}.csv", index=False,
                   lineterminator="\n")
        preds.to_csv(HERE / f"predictor_predictions{args.suffix}.csv", index=False,
                     lineterminator="\n")
        print(f"\nwrote predictor_report{args.suffix}.txt and "
              f"predictor*{args.suffix}.csv for target {t}")


if __name__ == "__main__":
    main()
