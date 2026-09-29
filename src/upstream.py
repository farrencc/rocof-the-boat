"""Thin adapter over the read-only upstream simulator.

This is the ONLY module in rocof-the-boat that imports from ``external/``.
Upstream (ciangregg/EIEG_Hackathon26) has no LICENSE, so nothing is copied from
it; it is cloned by ``scripts/setup_upstream.sh`` and imported via ``sys.path``.
If upstream moves or renames a private helper, fix it here only.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPO_ROOT / "external" / "EIEG_Hackathon26"
SIM_DIR = UPSTREAM_ROOT / "NI_Grid_Simulator"

if not SIM_DIR.is_dir():
    raise ImportError(
        f"Upstream simulator not found at {SIM_DIR}. Run scripts/setup_upstream.sh first."
    )
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

import all_island_annealer_api as _api  # noqa: E402

read_network_nc = _api.read_network_nc
DCGridModel = _api.DCGridModel
_read_extra = _api._read_extra
_subset_network = _api._subset_network
_build_node_table = _api._build_node_table
_build_frozen_cases = _api._build_frozen_cases
_screen_security_states = _api._screen_security_states
_required_reduction = _api._required_reduction
_node_groups = _api._node_groups
_build_lodf = _api._build_lodf
_scope_value = _api._scope_value
make_emulator = _api.make_emulator
EPS = _api.EPS

DATA_DIR = SIM_DIR / "data"
NETWORK_FILE = DATA_DIR / "SV2024_all-island.nc"
NODE_FILES = {
    "26": DATA_DIR / "annealer_nodes_26_counties_sv2024_wdt_exclusive.csv",
    "32": DATA_DIR / "annealer_nodes_32_counties_sv2024_wdt_exclusive.csv",
}

# Upstream make_emulator defaults, reproduced so our construction matches it.
DEFAULTS = dict(
    include_n1=True,
    security_screen_threshold_pct=90.0,
    max_security_states=800,
    max_group_actions=24,
)


@dataclass
class Grid:
    """Topology-level objects: independent of weather seed and thermal scale."""
    scope: str
    extra: object
    network: object
    model: object
    template: pd.DataFrame


_GRID_CACHE: dict[str, Grid] = {}


def load_grid(scope: str = "26") -> Grid:
    """Replicate the first half of ``make_emulator``: network, model, node table."""
    scope = _scope_value(scope)
    if scope in _GRID_CACHE:
        return _GRID_CACHE[scope]
    extra = _read_extra(NETWORK_FILE)
    full = read_network_nc(NETWORK_FILE)
    network, _selected = _subset_network(full, extra, scope)
    model = DCGridModel(network)
    template = _build_node_table(network, NODE_FILES[scope], scope)
    grid = Grid(scope, extra, network, model, template)
    _GRID_CACHE[scope] = grid
    return grid


def build_cases(grid: Grid, seed: int, thermal_scale: float = 1.0, runs: int = 10_000,
                include_n1: bool = True, security_screen_threshold_pct: float = 90.0,
                max_security_states: int = 800):
    """Replicate the second half of ``make_emulator``: the frozen ``_FrozenCases``.

    Note: upstream uses the SAME ``seed`` for synthetic weather and for sampling
    case weights, so "weather seed" and "simulation seed" are one knob here.
    """
    return _build_frozen_cases(
        grid.network, grid.model, grid.extra, grid.template,
        runs=int(runs), seed=int(seed), thermal_scale=float(thermal_scale),
        include_n1=bool(include_n1),
        security_screen_threshold_pct=float(security_screen_threshold_pct),
        max_security_states=int(max_security_states),
    )


def ptdf(grid: Grid, balance_bus: str) -> np.ndarray:
    """Branch x bus PTDF for the given slack (topology only)."""
    return grid.model.ptdf(balance_bus)


def lodf(grid: Grid, H: np.ndarray):
    """Branch x outage LODF and valid-outage mask, via upstream's helper."""
    return _build_lodf(grid.model, H)


def branch_limits(grid: Grid, thermal_scale: float) -> np.ndarray:
    return grid.model.branches["s_nom_mva"].to_numpy(float) * float(thermal_scale)


def exclusive_group_ids(grid: Grid) -> np.ndarray:
    """Upstream's scalar group per node, exactly as make_emulator reads it."""
    return _node_groups(grid.template, grid.template)


LAND_BOUNDARY_GEOJSON = SIM_DIR / "ireland_land_boundary.geojson"


def land_outlines() -> list[np.ndarray]:
    """Ireland coastline rings (lon, lat) from upstream's geojson, for map backdrops."""
    import json
    if not LAND_BOUNDARY_GEOJSON.exists():
        return []
    gj = json.loads(LAND_BOUNDARY_GEOJSON.read_text())
    rings = []

    def add(geom):
        t, c = geom["type"], geom["coordinates"]
        polys = [c] if t == "Polygon" else c if t == "MultiPolygon" else []
        for poly in polys:
            for ring in poly:
                rings.append(np.asarray(ring, float)[:, :2])

    for f in gj.get("features", [gj]):
        add(f.get("geometry", f))
    return rings
