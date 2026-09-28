"""Turn the sweep into an answer about local effects.

Three questions, in order:

  1. Is there a local effect at all -- does RoCoF measured at a bus differ from
     the system-wide (centre-of-inertia) RoCoF that a global inertia argument
     would predict?
  2. Does the geography of inertia change that local effect, given that total
     inertia and the pre-fault operating point are held fixed by construction?
  3. Specifically: does clustering the inverter-based generators raise RoCoF near
     them, relative to spreading the same number of them out?

Faults are averaged over a fixed, spatially balanced set (every generator trip,
every load step), identical in every configuration, so the excitation cannot
itself imprint a spatial pattern.  The faulted bus is dropped from bus-level
aggregates: when the tripped unit is a converter, that bus is left nearly
massless with no injection, which is a modelling artefact rather than a result.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats


def is_aggregated(df: pd.DataFrame) -> bool:
    """True if experiments.py already averaged over disturbances for us."""
    return "abs_rocof" in df.columns


def per_bus_field(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over disturbances of each per-bus metric, one row per (config, bus)."""
    if is_aggregated(df):
        return df
    clean = df[~df["artifact_bus"]]
    g = clean.groupby(["non_inertial", "cluster_index", "bus", "row", "col",
                       "is_gen", "is_noninertial", "converter_density",
                       "dist_to_nearest_ni"], as_index=False)
    out = g.agg(local_excess=("local_excess", "mean"),
                abs_rocof=("rocof_500ms", lambda s: s.abs().mean()),
                worst_window=("worst_window", "mean"),
                peak_instant=("peak_instant", "mean"),
                t_peak=("t_peak", "mean"))
    return out


def per_config(field: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-bus field to one row per configuration."""
    recs = []
    for (ni, ci), sub in field.groupby(["non_inertial", "cluster_index"]):
        # how strongly the local effect tracks nearness to the converters
        r = stats.pearsonr(sub["converter_density"], sub["local_excess"])
        recs.append({
            "non_inertial": ni,
            "cluster_index": ci,
            "field_std": sub["local_excess"].std(ddof=0),
            "field_range": sub["local_excess"].max() - sub["local_excess"].min(),
            "worst_bus_rocof": sub["abs_rocof"].max(),
            "worst_bus_excess": sub["local_excess"].max(),
            "mean_excess_at_ni": sub.loc[sub["is_noninertial"], "local_excess"].mean(),
            "r_density_excess": r.statistic,
            "p_density_excess": r.pvalue,
        })
    return pd.DataFrame(recs).sort_values("cluster_index").reset_index(drop=True)


def report_correlation(cfg: pd.DataFrame, col: str, label: str) -> None:
    r = stats.pearsonr(cfg["cluster_index"], cfg[col])
    rho = stats.spearmanr(cfg["cluster_index"], cfg[col])
    print(f"  {label:<34s} r = {r.statistic:+.3f} (p = {r.pvalue:.1e})"
          f"   rho = {rho.statistic:+.3f}")


def grid(field: pd.DataFrame, ni: str, col: str, L: int) -> np.ndarray:
    sub = field[field["non_inertial"] == ni].sort_values("bus")
    return sub[col].to_numpy().reshape(L, L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--L", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    L = args.L
    coi = df["coi_rocof_500ms"].abs().mean()

    print("=" * 74)
    print(f"{args.csv}: {df['non_inertial'].nunique()} configurations, "
          f"{len(df)} bus-observations")
    print(f"total inertia and pre-fault operating point identical across all of them")
    print(f"system-wide (COI) RoCoF over 500 ms: {coi:.3f} Hz/s -- "
          f"this is what a purely global inertia argument predicts at every bus")

    # ---- Q1: is there a local effect at all? -----------------------------
    print("\n[1] IS THE EFFECT LOCAL AT ALL?")
    if is_aggregated(df):
        ex = df["local_excess"]
        print(f"  per-bus |RoCoF(500ms)| spans {df['abs_rocof'].min():.3f} "
              f"to {df['abs_rocof'].max():.3f} Hz/s about a COI of {coi:.3f}")
        print("  (this file is pre-averaged over disturbances, so the "
              "distance-to-fault\n   breakdown is only available in the "
              "un-aggregated checkerboard run)")
    else:
        clean = df[~df["artifact_bus"]]
        ex = clean["local_excess"]
        print(f"  per-bus |RoCoF(500ms)| spans {clean['rocof_500ms'].abs().min():.3f} "
              f"to {clean['rocof_500ms'].abs().max():.3f} Hz/s about a COI of {coi:.3f}")
        d = clean.groupby("dist_to_fault")["local_excess"].mean()
        tp = clean.groupby("dist_to_fault")["t_peak"].mean()
    print(f"  local excess: mean {ex.mean():+.3f}, sd {ex.std():.3f}, "
          f"range {ex.min():+.3f} to {ex.max():+.3f} Hz/s "
          f"({100 * ex.abs().max() / coi:.0f}% of the global value)")
    if not is_aggregated(df):
        print("  local excess vs hops from the disturbance:")
        print("   ", "  ".join(f"{int(k)}:{v:+.3f}" for k, v in d.items()))
        print("  arrival of the peak swing (ms) vs hops -- the effect propagates:")
        print("   ", "  ".join(f"{int(k)}:{1000 * v:.0f}" for k, v in tp.items()))

    # ---- Q2/Q3: does the geography matter? -------------------------------
    field = per_bus_field(df)
    cfg = per_config(field)

    print("\n[2] DOES THE GEOGRAPHY OF INERTIA MATTER? "
          "(clustered = low cluster index)")
    print(f"  cluster index spans {cfg['cluster_index'].min():.2f} (most clustered)"
          f" to {cfg['cluster_index'].max():.2f} (most spread)")
    report_correlation(cfg, "field_std", "spatial sd of local excess")
    report_correlation(cfg, "field_range", "spatial range of local excess")
    report_correlation(cfg, "worst_bus_rocof", "worst bus |RoCoF(500ms)|")
    report_correlation(cfg, "worst_bus_excess", "worst bus local excess")
    report_correlation(cfg, "mean_excess_at_ni", "mean excess at converter buses")

    lo = cfg[cfg["cluster_index"] <= cfg["cluster_index"].quantile(0.25)]
    hi = cfg[cfg["cluster_index"] >= cfg["cluster_index"].quantile(0.75)]
    print(f"\n  most-clustered quartile (n={len(lo)}) vs most-spread quartile "
          f"(n={len(hi)}):")
    for col, lab in [("worst_bus_rocof", "worst bus |RoCoF| (Hz/s)"),
                     ("field_std", "spatial sd of excess (Hz/s)"),
                     ("worst_bus_excess", "worst bus excess (Hz/s)"),
                     ("mean_excess_at_ni", "excess at converter buses (Hz/s)")]:
        t = stats.mannwhitneyu(lo[col], hi[col])
        print(f"    {lab:<32s} clustered {lo[col].mean():+.4f}  "
              f"spread {hi[col].mean():+.4f}   p = {t.pvalue:.1e}")

    print("\n[3] IS THE EXTRA RoCoF *AT* THE CONVERTERS?")
    print("  within-configuration correlation across buses between how much "
          "inverter\n  generation is nearby and the local excess:")
    print(f"    mean r = {cfg['r_density_excess'].mean():+.3f}, "
          f"{(cfg['p_density_excess'] < 0.05).sum()}/{len(cfg)} configurations "
          f"significant at p<0.05")
    print(f"    clustered quartile mean r = {lo['r_density_excess'].mean():+.3f}, "
          f"spread quartile mean r = {hi['r_density_excess'].mean():+.3f}")
    byd = field.groupby("dist_to_nearest_ni")["local_excess"].agg(["mean", "count"])
    print("  local excess vs hops to the nearest converter bus:")
    for k, row in byd.iterrows():
        print(f"    {int(k)} hops: {row['mean']:+.4f} Hz/s  (n={int(row['count'])})")

    # ---- the two extreme configurations, drawn out ------------------------
    most_c = cfg.iloc[0]
    most_s = cfg.iloc[-1]
    print("\n[4] THE TWO EXTREMES, SIDE BY SIDE")
    for tag, row in (("CLUSTERED", most_c), ("SPREAD", most_s)):
        ni = row["non_inertial"]
        mask = grid(field, ni, "is_noninertial", L).astype(int)
        exc = grid(field, ni, "local_excess", L)
        print(f"\n  {tag}: converters at buses {ni} "
              f"(cluster index {row['cluster_index']:.2f})")
        print("    converter map            local excess (Hz/s), fault-averaged")
        for r in range(L):
            left = " ".join("C" if v else "." for v in mask[r])
            right = " ".join(f"{v:+.3f}" for v in exc[r])
            print(f"    {left:<24s} {right}")
        print(f"    worst bus |RoCoF| {row['worst_bus_rocof']:.4f} Hz/s, "
              f"spatial sd {row['field_std']:.4f}, "
              f"r(density, excess) = {row['r_density_excess']:+.3f}")

    diff = most_c["worst_bus_rocof"] - most_s["worst_bus_rocof"]
    print(f"\n  clustering the same 4 converters raises the worst bus by "
          f"{diff:+.4f} Hz/s ({100 * diff / most_s['worst_bus_rocof']:+.2f}%)")

    cfg.to_csv(args.csv.replace(".csv", "_config.csv"), index=False)
    wrote = ["_config.csv"]
    if not is_aggregated(df):
        # for a pre-aggregated sweep the per-bus table *is* the input file, and
        # writing it back out just duplicates several megabytes
        field.to_csv(args.csv.replace(".csv", "_field.csv"), index=False)
        wrote.append("_field.csv")
    print(f"\nwrote {' and '.join(wrote)} alongside {args.csv}")


if __name__ == "__main__":
    main()
