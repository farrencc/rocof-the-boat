"""All figures. Colours: validated reference categorical palette, fixed order."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#9a9993"
TERM_LABEL = {"DD": "T1 dispatch-down D(G)", "S": "T2 mean sigma", "V": "T3 variance",
              "N": "T4 size sum n_k^2", "P": "T5 wrong-sign hinge"}

plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titlecolor": INK,
    "lines.linewidth": 1.6, "figure.facecolor": "white", "savefig.bbox": "tight",
})


# ---------------------------------------------------------------- anchors
def anchors_plot(setup_dir: Path, out_path: Path) -> Path:
    meta = json.loads((setup_dir / "anchors.json").read_text())
    tab = pd.read_csv(setup_dir / "branch_binding.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for col, lab, c in (("p_e", "relievable branches only (used)", SERIES[0]),
                        ("p_literal", "literal: all branches incl. unrelievable", SERIES[1])):
        p = np.sort(tab[col].to_numpy())[::-1]
        p = p[p > 0]
        x = np.arange(1, len(p) + 1)
        ax.step(np.r_[0, x], np.r_[0, np.cumsum(p)], where="post", color=c, label=lab)
    K = meta["n_pe_anchors"]
    ax.axhline(meta["threshold"], color=MUTED, ls="--", lw=1)
    ax.axhline(0.5, color=MUTED, ls=":", lw=1)
    ax.axvline(K, color=SERIES[0], ls="--", lw=1)
    ax.annotate(f"K={K} at {meta['threshold']}", (K, meta["threshold"]), xytext=(6, -14),
                textcoords="offset points", color=INK2)
    names = [a["name"] for a in meta["anchors"][:K]]
    ax.set_title("Cumulative binding probability p_e by monitored branch")
    ax.set_xlabel("number of branches (sorted by p_e)")
    ax.set_ylabel("cumulative p_e")
    ax.set_xlim(0, max(8, K + 3))
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.02, 0.97, "anchors: " + "; ".join(names), transform=ax.transAxes, va="top", fontsize=7, color=INK2)

    ax = axes[1]
    for key, lab, c in (("k_by_threshold", "relievable only", SERIES[0]),
                        ("k_by_threshold_literal", "literal", SERIES[1])):
        d = {float(k): v for k, v in meta[key].items()}
        t = sorted(d)
        ax.plot(t, [d[x] for x in t], marker="o", ms=5, color=c, label=lab, drawstyle="steps-post")
    ax.set_title("K vs cumulative-p_e threshold")
    ax.set_xlabel("threshold")
    ax.set_ylabel("K")
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------- per config
def config_plots(out: Path, title: str = "") -> None:
    h = pd.read_csv(out / "history.csv")
    if h.empty:
        return
    x = h["blackbox_calls"]
    # Energy trace: E and D and every normalised term, each on its own axis (no dual axes).
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    axes = axes.ravel()
    axes[0].plot(x, h.current_energy, color=SERIES[0], alpha=0.6, label="current E")
    axes[0].plot(x, h.best_energy, color=SERIES[1], label="best E")
    axes[0].set_ylabel("E (normalised)")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Hamiltonian E")
    axes[1].plot(x, h.current_D, color=SERIES[0], alpha=0.6, label="D of current")
    axes[1].plot(x, h.best_E_D, color=SERIES[1], label="D of best-E grouping")
    axes[1].plot(x, h.best_D, color=SERIES[2], label="best D seen (feasible)")
    axes[1].set_ylabel("dispatch-down %")
    axes[1].set_title("raw D(G)  (always shown separately from E)")
    axes[1].legend(fontsize=7)
    for ax, t, c in zip(axes[2:], ["S", "V", "N", "P"], SERIES[3:]):
        ax.plot(x, h[f"raw_{t}"], color=c)
        ax.set_title(TERM_LABEL[t] + " (raw, current state)")
    for ax in axes[4:]:
        ax.set_xlabel("blackbox calls")
    fig.suptitle(title, fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "energy_trace.png", dpi=130)
    plt.close(fig)

    # Upstream plot_hist.py three-panel style.
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})
    ax = axes[0]
    ax.plot(x, h.current_energy, color=SERIES[0], alpha=0.5, lw=1, label="current energy")
    ax.plot(x, h.best_energy, color=SERIES[7], lw=2, label="best-so-far energy")
    imp = h[h.new_bests > 0]
    ax.scatter(imp.blackbox_calls, imp.best_energy, color=SERIES[7], s=16, zorder=5, label="new best found")
    ax.set_ylabel("E (normalised)")
    ax.set_title(title or "Annealing run diagnostics")
    ax.legend(fontsize=8)
    axes[1].plot(x, h.temperature, color=SERIES[3])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("temperature (log)")
    axes[2].plot(x, h.acceptance_rate, color=SERIES[2], label="accepted")
    n_prop = (h.blackbox_calls.diff().fillna(h.blackbox_calls.iloc[0])).clip(lower=1)
    axes[2].plot(x, h.infeasible_proposals / n_prop,
        color=SERIES[7], lw=1, alpha=0.7, label="infeasible (guard) proposals")
    axes[2].set_ylabel("fraction of proposals")
    axes[2].set_xlabel("blackbox calls (ensemble evaluations)")
    axes[2].set_ylim(0, 1)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "annealing_diagnostics.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- sweep level
def pareto_plot(summary: pd.DataFrame, out_path: Path, baseline: dict | None = None) -> Path:
    """D(G) of the best-E grouping vs each raw structural term across the lambda sweep."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, t, c in zip(axes, ["S", "V", "N", "P"], SERIES[3:]):
        ax.scatter(summary[f"raw_{t}"], summary["D"], s=40, color=SERIES[0], edgecolor="white",
                   linewidth=1.5, zorder=3, label="best-E grouping")
        ax.scatter(summary[f"bestD_raw_{t}"], summary["bestD_D"], s=22, marker="s", color=SERIES[1],
                   edgecolor="white", linewidth=1, zorder=2, label="best-D grouping")
        if baseline is not None:
            ax.scatter([baseline[f"raw_{t}"]], [baseline["D"]], marker="*", s=120, color=INK, zorder=4,
                       label="initial (shift-factor) grouping")
        for _, r in summary.iterrows():
            ax.annotate(r["config_id"].split("_", 1)[0], (r[f"raw_{t}"], r["D"]), fontsize=6, color=INK2,
                        xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel(TERM_LABEL[t])
    axes[0].set_ylabel("training-ensemble D(G) %")
    axes[0].legend(fontsize=7)
    fig.suptitle("Pareto view across the lambda sweep (raw terms)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def validation_plot(val: pd.DataFrame, out_path: Path, label_opt: str) -> Path:
    """Baseline (WDT) vs optimised D on fresh seeds, with pp and % improvement."""
    val = val.copy()
    val["case"] = val.seed.astype(str) + "\nts=" + val.thermal_scale.astype(str)
    cases = val.case.tolist()
    xx = np.arange(len(cases))
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    w = 0.26
    for off, col, lab, c in ((-w, "D_wdt", "baseline WDT (multi-membership)", SERIES[0]),
                             (0, "D_excl", "WDT exclusive (upstream)", SERIES[6]),
                             (w, "D_opt", label_opt, SERIES[1])):
        ax.bar(xx + off, val[col], width=w - 0.02, color=c, label=lab, edgecolor="white", linewidth=1)
    ax.set_ylabel("dispatch-down %")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.set_ylim(0, val[["D_wdt", "D_excl", "D_opt"]].to_numpy().max() * 1.25)
    ax.set_title("Fresh validation seeds: baseline vs optimised dispatch-down")
    ax = axes[1]
    imp = val.D_wdt - val.D_opt
    ax.bar(xx, imp, color=[SERIES[2] if v >= 0 else SERIES[7] for v in imp], edgecolor="white")
    for i, (v, r) in enumerate(zip(imp, 100 * imp / val.D_wdt)):
        ax.annotate(f"{v:+.2f}pp\n{r:+.1f}%", (i, v), ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=7, color=INK2)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylabel("improvement vs WDT (pp)")
    ax.set_xticks(xx, cases, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def group_map(template: pd.DataFrame, M: np.ndarray, anchor_names: list[str], rings, out_path: Path,
              title: str, anchor_xy=None) -> Path:
    lon, lat = template.longitude.to_numpy(float), template.latitude.to_numpy(float)
    mec = template.mec_mw.to_numpy(float)
    size = 8 + 120 * mec / mec.max()
    K = M.shape[1]
    ncol = K + 1
    fig, axes = plt.subplots(1, ncol, figsize=(3.3 * ncol, 4.4))

    def backdrop(ax):
        for r in rings:
            ax.plot(r[:, 0], r[:, 1], color=MUTED, lw=0.6)
        ax.set_aspect(1 / np.cos(np.deg2rad(53.4)))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)

    ax = axes[0]
    backdrop(ax)
    nmem = M.sum(1)
    ax.scatter(lon[nmem == 0], lat[nmem == 0], s=size[nmem == 0], facecolor="none", edgecolor=MUTED, lw=0.6,
               label="uncovered")
    for k in range(K):
        sel = M[:, k] & (nmem == 1)
        ax.scatter(lon[sel], lat[sel], s=size[sel], color=SERIES[k], edgecolor="white", lw=0.8,
                   label=f"g{k}: {anchor_names[k]}")
    ov = nmem > 1
    ax.scatter(lon[ov], lat[ov], s=size[ov], color=INK, edgecolor="white", lw=0.8, marker="D",
               label="in >1 group (overlap)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, loc="lower right")
    for k in range(K):
        ax = axes[k + 1]
        backdrop(ax)
        ax.scatter(lon, lat, s=size, facecolor="none", edgecolor=GRID, lw=0.6)
        sel = M[:, k]
        ax.scatter(lon[sel], lat[sel], s=size[sel], color=SERIES[k], edgecolor="white", lw=0.8)
        if anchor_xy is not None:
            (x0, y0), (x1, y1) = anchor_xy[k]
            ax.plot([x0, x1], [y0, y1], color=INK, lw=3)
            ax.scatter([x0, x1], [y0, y1], color=INK, s=12, zorder=5)
        ax.set_title(f"g{k} anchor: {anchor_names[k]}\n{int(sel.sum())} nodes, {mec[sel].sum():.0f} MW MEC",
                     fontsize=8)
    fig.text(0.01, 0.01, "marker area ~ MEC", fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def sigma_distributions(sigma, mec, groupings: dict[str, np.ndarray], anchor_names, out_path: Path) -> Path:
    """Per anchor: MEC-weighted distribution of sigma_ik over the members of each grouping's group k."""
    K = sigma.shape[1]
    fig, axes = plt.subplots(1, K, figsize=(3.6 * K, 3.6))
    axes = np.atleast_1d(axes)
    for k, ax in enumerate(axes):
        lo, hi = sigma[:, k].min(), sigma[:, k].max()
        bins = np.linspace(lo, hi, 30)
        for j, (lab, M) in enumerate(groupings.items()):
            sel = M[:, k] if M.shape[1] > k else np.zeros(len(mec), bool)
            if not sel.any():
                continue
            ax.hist(sigma[sel, k], bins=bins, weights=mec[sel], histtype="step", lw=1.8, color=SERIES[j],
                    label=f"{lab} (n={int(sel.sum())})")
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title(f"g{k}: {anchor_names[k]}", fontsize=8)
        ax.set_xlabel("oriented sigma_ik (>0 helps)")
        ax.legend(fontsize=6)
    axes[0].set_ylabel("MEC-weighted count (MW)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
