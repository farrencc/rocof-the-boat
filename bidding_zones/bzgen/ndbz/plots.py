"""Figures for the nDBZ reports (static PNGs under figures/ndbz/)."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from bzgen.ndbz import savefig

# categorical order (fixed, never cycled) and ramps: see the dataviz reference palette
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f7f9fc", "#cde2fb", "#86b6ef", "#3987e5",
                                                     "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f19a9a", "#b3261e"])


def scenario_colour(names: list[str]) -> dict:
    return {n: SERIES[i % len(SERIES)] for i, n in enumerate(names)}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=7)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def heatmap(df: pd.DataFrame, fn, title: str, cbar: str, fmt: str = "{:.2f}", cmap=SEQ,
            norm=None, vmin=None, vmax=None, flag: pd.DataFrame | None = None):
    """Rows = scenarios, columns = countries. ``flag`` marks cells with a dot outline."""
    fig, ax = plt.subplots(figsize=(0.42 * df.shape[1] + 2.2, 0.45 * df.shape[0] + 1.6))
    im = ax.imshow(df.to_numpy(float), aspect="auto", cmap=cmap, norm=norm,
                   vmin=None if norm else vmin, vmax=None if norm else vmax)
    ax.set_xticks(range(df.shape[1]), df.columns, fontsize=7, rotation=90, color=INK2)
    ax.set_yticks(range(df.shape[0]), df.index, fontsize=8, color=INK)
    arr = df.to_numpy(float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = arr[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=6, color=INK2)
                continue
            rgba = im.cmap(im.norm(v))
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=5.5,
                    color="white" if lum < 0.5 else INK)
            if flag is not None and bool(flag.iloc[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.4))
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(cbar, fontsize=7, color=INK2)
    cb.ax.tick_params(labelsize=6, colors=INK2)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    savefig(fig, fn, dpi=150)
    plt.close(fig)


def severity_heatmap(df: pd.DataFrame, fn, title: str):
    lo, hi = np.nanmin(df.to_numpy()), np.nanmax(df.to_numpy())
    norm = TwoSlopeNorm(vcenter=1.0, vmin=min(lo, 0.99), vmax=max(hi, 1.01))
    heatmap(df, fn, title, "mean dp̃ (baseline normalisation; 1 = full-year level)", cmap=DIV, norm=norm)


def dpn_ecdf(tables: dict, countries: list[str], fn, title: str):
    """ECDF of dp~ per country (rows) under baseline vs self normalisation (columns),
    one line per scenario.  ``tables``: scenario -> normalised edge table."""
    names = list(tables)
    col = scenario_colour(names)
    fig, axes = plt.subplots(len(countries), 2, figsize=(8.0, 1.9 * len(countries)), squeeze=False,
                             sharex="row")
    for r, c in enumerate(countries):
        for k, (mode, key) in enumerate((("baseline", "dp_n_baseline"), ("self", "dp_n_self"))):
            ax = axes[r, k]
            _style(ax)
            for n in names:
                t = tables[n]
                x = np.sort(t.loc[t.country == c, key].to_numpy(float))
                if len(x) == 0:
                    continue
                ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", lw=1.6, color=col[n],
                        label=n)
            ax.axvline(1.0, color=INK2, lw=0.7, ls=":")
            ax.set_title(f"{c} — {mode} normalisation", fontsize=8, color=INK, loc="left")
            ax.set_ylim(0, 1.02)
            if r == len(countries) - 1:
                ax.set_xlabel("dp̃", fontsize=7, color=INK2)
            if k == 0:
                ax.set_ylabel("share of edges ≤ x", fontsize=7, color=INK2)
    axes[0, 1].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.suptitle(title, fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    savefig(fig, fn, dpi=150)
    plt.close(fig)


def bars(values: pd.Series, fn, title: str, ylabel: str, log: bool = False, note: str = ""):
    """Single-series bar chart (one hue, no legend: the title names the series)."""
    fig, ax = plt.subplots(figsize=(0.28 * len(values) + 1.8, 3.0))
    _style(ax)
    ax.grid(axis="x", visible=False)
    ax.bar(np.arange(len(values)), values.to_numpy(float), width=0.7, color=SERIES[0], lw=0)
    ax.set_xticks(np.arange(len(values)), values.index, fontsize=7, rotation=90, color=INK2)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel, fontsize=7, color=INK2)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    if note:
        ax.text(0.0, -0.32, note, transform=ax.transAxes, fontsize=6.5, color=INK2, va="top")
    fig.tight_layout()
    savefig(fig, fn, dpi=150)
    plt.close(fig)
