"""Representative hours by k-means on national load / wind / solar features.

Features per hour: for every country, load / mean load, capacity-weighted wind
CF (onshore + offshore generators) and solar CF; each feature z-scored.  The
representative of a cluster is the member hour nearest its centroid (a
medoid-like choice that keeps every representative a real, internally
consistent hour).  **Weight = cluster size in hours**; weights sum to the
number of hours in the year and are carried through every time average.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def features(nat: pd.DataFrame, gens: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for c in nat.columns:
        cols[f"{c}_load"] = nat[c] / nat[c].mean()
        for car, carriers in (("wind", ["onwind", "offwind"]), ("solar", ["solar"])):
            g = gens[(gens.country == c) & gens.carrier.isin(carriers) & gens.index.isin(prof.columns)]
            if g.p_nom.sum() > 0:
                cols[f"{c}_{car}"] = (prof[g.index] * g.p_nom).sum(1) / g.p_nom.sum()
    X = pd.DataFrame(cols)
    X = X.loc[:, X.std() > 0]
    return (X - X.mean()) / X.std()


def representative_hours(X: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    km = KMeans(n_clusters=n, n_init=4, random_state=seed).fit(X.to_numpy())
    lab = km.labels_
    d = np.linalg.norm(X.to_numpy() - km.cluster_centers_[lab], axis=1)
    rows = []
    for c in range(n):
        m = np.flatnonzero(lab == c)
        rep = m[np.argmin(d[m])]
        rows.append({"snapshot": X.index[rep], "weight": float(len(m)), "cluster": c})
    return pd.DataFrame(rows).sort_values("snapshot").set_index("snapshot")
