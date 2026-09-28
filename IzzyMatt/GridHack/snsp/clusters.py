"""Spectral clustering of the frozen grid, by electrical coupling.

SNSP is a single number for the whole system, and that is its weakness as a risk
predictor: it says how much of the energy is non-synchronous, and nothing about
*where*.  Two configurations with identical SNSP can put the non-synchronous
generation next to the machines that are still on, or a long way from them, and
the local-effects study says those are not the same grid to be a bus in.

This module cuts the grid into electrically coherent pieces so that the share
can be asked per piece.  The partition is a property of the network alone -- the
coupling matrix `Kij`, which never varies in this study -- so it is computed once
and frozen, and every configuration is then described by the same k regions.
That is deliberate: a partition that moved with the dispatch would make the
per-cluster features of two configurations incomparable, which is the one thing
a predictor built on them needs them to be.

The method is the standard one (Ng, Jordan and Weiss):

  1. Build the symmetric normalised Laplacian  L = I - D^-1/2 K D^-1/2  from the
     coupling matrix, weighting each edge by its synchronising coefficient, so a
     400 kV backbone tie (K = 8) counts for more than a radial spur (K = 2.5).
  2. Take the k eigenvectors of the k smallest eigenvalues.  The first is
     constant; the second is the Fiedler vector, and the rest continue the
     sequence of increasingly fine cuts.
  3. Row-normalise that embedding and k-means it.

`k` itself comes from the eigengap heuristic: the spectrum of a graph with k
well-separated pieces has k small eigenvalues and then a jump, so the k that
maximises lambda_{k+1} - lambda_k is the number of pieces the graph actually
has.  On this grid that is a real but not overwhelming preference -- a 10x10
mesh is not a barbell, and the report prints the whole gap profile rather than
just the winner so the margin is visible.

Run `python clusters.py --freeze` to write `clusters.json` and the report.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2

import grid as G

HERE = Path(__file__).resolve().parent

#: The range of k the eigengap heuristic is allowed to choose from.  k = 1 is
#: excluded because it is the trivial partition and its "gap" is just the
#: algebraic connectivity, which is always there; the upper end is set well
#: above any plausible answer so the choice is the data's and not the range's.
K_MIN, K_MAX = 2, 12

#: k-means restarts.  The spectral embedding is low-dimensional and well
#: separated, but k-means is still only locally optimal, so it is run many times
#: from different starts and the lowest-distortion labelling wins.
N_INIT = 200
SEED = 20260910


def laplacian(g: G.ToyGrid, normalised: bool = True) -> np.ndarray:
    """The weighted Laplacian of the grid, edges weighted by `Kij`.

    `Kij` is the synchronising coefficient of each line -- the stiffness of the
    spring between two buses in the swing equations -- so this is the Laplacian
    of the network as the dynamics actually see it, not of its wiring diagram.

    Normalised (the default) because the buses have very different degrees: a
    radial stub has one line and a backbone node has six, and the combinatorial
    Laplacian's cuts follow that degree imbalance rather than the structure.
    """
    W = g.Kij
    d = W.sum(axis=1)
    if not normalised:
        return np.diag(d) - W
    dm = 1.0 / np.sqrt(d)
    return np.eye(g.N) - (dm[:, None] * W * dm[None, :])


def spectrum(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues and eigenvectors of a symmetric Laplacian, ascending."""
    vals, vecs = np.linalg.eigh(L)
    return vals, vecs


def eigengaps(vals: np.ndarray, k_min: int = K_MIN,
              k_max: int = K_MAX) -> dict[int, float]:
    """lambda_{k+1} - lambda_k for each admissible k, 1-indexed eigenvalues.

    With `vals` ascending and 0-indexed, the k-th smallest eigenvalue is
    `vals[k - 1]`, so the gap that follows a k-piece partition is
    `vals[k] - vals[k - 1]`.
    """
    return {k: float(vals[k] - vals[k - 1]) for k in range(k_min, k_max + 1)}


def choose_k(vals: np.ndarray, k_min: int = K_MIN, k_max: int = K_MAX) -> int:
    """The eigengap heuristic: the k with the largest jump after it."""
    gaps = eigengaps(vals, k_min, k_max)
    return max(gaps, key=gaps.get)


def embed(vecs: np.ndarray, k: int) -> np.ndarray:
    """The k lowest eigenvectors, row-normalised onto the unit sphere.

    Row-normalising is what makes k-means the right thing to run afterwards: it
    removes the per-bus scale that the normalised Laplacian introduces, leaving
    only the direction, which is what carries the cluster identity.
    """
    X = vecs[:, :k]
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def kmeans(X: np.ndarray, k: int, seed: int = SEED,
           n_init: int = N_INIT) -> np.ndarray:
    """k-means with many restarts, lowest within-cluster distortion wins."""
    rng = np.random.default_rng(seed)
    best, best_d = None, np.inf
    for _ in range(n_init):
        c, lab = kmeans2(X, k, minit="++", seed=int(rng.integers(1 << 31)))
        if len(np.unique(lab)) < k:            # a start that collapsed
            continue
        d = float(((X - c[lab]) ** 2).sum())
        if d < best_d:
            best, best_d = lab, d
    if best is None:
        raise RuntimeError(f"k-means never produced {k} non-empty clusters")
    return relabel(best)


def relabel(lab: np.ndarray) -> np.ndarray:
    """Canonical cluster ids: by first appearance, so a rerun names them alike."""
    order = {}
    for v in lab:
        order.setdefault(int(v), len(order))
    return np.array([order[int(v)] for v in lab], int)


@dataclass
class Partition:
    """A frozen cut of the grid into electrically coherent regions."""

    k: int
    label: np.ndarray            # cluster id per bus
    vals: np.ndarray             # the Laplacian spectrum that produced it
    gaps: dict[int, float]
    normalised: bool

    @property
    def sizes(self) -> np.ndarray:
        return np.bincount(self.label, minlength=self.k)

    def members(self, c: int) -> np.ndarray:
        return np.flatnonzero(self.label == c)

    def cut_weight(self, g: G.ToyGrid) -> float:
        """Total coupling on edges that cross a cluster boundary, p.u."""
        cross = self.label[:, None] != self.label[None, :]
        return float(np.triu(g.Kij * cross, 1).sum())

    def conductance(self, g: G.ToyGrid) -> np.ndarray:
        """Per-cluster conductance: boundary coupling over total coupling.

        The standard quality reading for a spectral cut, and the one that says
        whether a cluster is really a piece of the network or just a label: 0 is
        an island, 1 is a set of buses with nothing holding them together.
        """
        out = np.zeros(self.k)
        for c in range(self.k):
            m = self.label == c
            vol = g.Kij[m].sum()
            out[c] = g.Kij[np.ix_(m, ~m)].sum() / max(vol, 1e-12)
        return out


def cluster(g: G.ToyGrid, k: int | None = None, normalised: bool = True,
            seed: int = SEED) -> Partition:
    """Spectrally cluster the grid; `k` from the eigengap heuristic if None."""
    L = laplacian(g, normalised=normalised)
    vals, vecs = spectrum(L)
    gaps = eigengaps(vals)
    if k is None:
        k = max(gaps, key=gaps.get)
    return Partition(k=k, label=kmeans(embed(vecs, k), k, seed=seed),
                     vals=vals, gaps=gaps, normalised=normalised)


# -- freeze / load ---------------------------------------------------------

def freeze(p: Partition, g: G.ToyGrid, path: Path | None = None) -> Path:
    path = path or (HERE / "clusters.json")
    path.write_text(json.dumps({
        "k": p.k,
        "normalised": p.normalised,
        "label": p.label.tolist(),
        "eigenvalues": p.vals[:K_MAX + 1].tolist(),
        "gaps": {str(k): v for k, v in p.gaps.items()},
        "sizes": p.sizes.tolist(),
        "conductance": p.conductance(g).tolist(),
        # The grid this partition was cut from.  A partition frozen against a
        # different network is not a partition of this one.
        "grid_sha256": json.loads(
            (HERE / "grid.json").read_text(encoding="utf-8"))["sha256"],
    }, indent=2), encoding="utf-8")
    return path


def load(path: Path | None = None) -> Partition:
    path = path or (HERE / "clusters.json")
    d = json.loads(path.read_text(encoding="utf-8"))
    return Partition(k=d["k"], label=np.array(d["label"], int),
                     vals=np.array(d["eigenvalues"], float),
                     gaps={int(k): v for k, v in d["gaps"].items()},
                     normalised=d["normalised"])


# -- report ----------------------------------------------------------------

def report(g: G.ToyGrid, p: Partition) -> str:
    out = ["SPECTRAL CLUSTERING OF THE FROZEN GRID",
           "=" * 74, ""]
    kind = "normalised" if p.normalised else "combinatorial"
    out += [f"  {kind} Laplacian of the coupling matrix Kij, "
            f"{g.N} buses, {len(g.edges)} lines", ""]

    out += ["1. The spectrum and the eigengap heuristic",
            "-" * 74,
            "  k   lambda_k   gap to lambda_(k+1)"]
    for k in sorted(p.gaps):
        mark = "   <- chosen" if k == p.k else ""
        out.append(f"  {k:2d}   {p.vals[k - 1]:8.5f}   {p.gaps[k]:10.5f}{mark}")
    best = sorted(p.gaps.values(), reverse=True)
    out += ["",
            f"  the chosen k = {p.k} wins by {best[0] / best[1]:.2f}x over the "
            f"runner-up gap",
            "  (a mesh is not a barbell: read this as a preference, not a "
            "partition that was hiding in the graph)", ""]

    out += ["2. The clusters", "-" * 74,
            "  id  buses  conductance  wind  hvdc  sync  city   demand   "
            "capacity  rows      cols"]
    cond = p.conductance(g)
    roles = g.role.astype(str)
    for c in range(p.k):
        m = p.members(c)
        rr = [g.coords[i][0] for i in m]
        cc = [g.coords[i][1] for i in m]
        n = {r: int((roles[m] == r).sum()) for r in ("wind", "hvdc", "sync",
                                                     "city")}
        out.append(f"  {c:2d}  {len(m):5d}  {cond[c]:11.3f}  {n['wind']:4d}  "
                   f"{n['hvdc']:4d}  {n['sync']:4d}  {n['city']:4d}  "
                   f"{g.demand[m].sum():7.3f}  {g.capacity[m].sum():8.3f}  "
                   f"{min(rr)}-{max(rr):<7d} {min(cc)}-{max(cc)}")
    out += ["",
            f"  coupling cut by the partition: {p.cut_weight(g):.1f} p.u. of "
            f"{np.triu(g.Kij, 1).sum():.1f} total "
            f"({p.cut_weight(g) / np.triu(g.Kij, 1).sum() * 100:.1f}%)", ""]

    out += ["3. The map", "-" * 74, ""]
    for r in range(g.L):
        row = "   "
        for c in range(g.L):
            i = g.index(r, c)
            row += f" {p.label[i]}"
        out.append(row)
    out += ["",
            "  rows run north 0 -> south 9, columns west 0 -> east 9;",
            "  the west coast is column 0 and the radial stubs are rows "
            f"{', '.join(str(r) for r in G.STUB_ROWS)}.", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, default=None,
                    help="force a number of clusters instead of the eigengap")
    ap.add_argument("--combinatorial", action="store_true",
                    help="use the unnormalised Laplacian (for the sweep)")
    ap.add_argument("--freeze", action="store_true",
                    help="write clusters.json and clusters_report.txt")
    args = ap.parse_args()

    g = G.load()
    p = cluster(g, k=args.k, normalised=not args.combinatorial)
    text = report(g, p)
    print(text)
    if args.freeze:
        freeze(p, g)
        (HERE / "clusters_report.txt").write_text(text, encoding="utf-8")
        print("wrote clusters.json, clusters_report.txt")


if __name__ == "__main__":
    main()
