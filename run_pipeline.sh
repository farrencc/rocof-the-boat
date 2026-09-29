#!/usr/bin/env bash
# Full pipeline, in order. Every stage caches/persists; re-running skips completed work
# where it can (downloads, sweep configurations).
set -euo pipefail
. .venv/bin/activate
python -m src.data.topology            # PyPSA-Eur OSM network (Zenodo, md5-verified)
python -m src.data.weather fetch       # ERA5 via Open-Meteo (slow: rate limits)
python -m src.report data              # assemble network + loads + fleet -> reports/data.md
python -m src.solve.run                # 8760 h DC-OPF -> data/solved/
python -m src.cluster.prepare          # edge stats + normalisation gate -> reports/normalisation.md
python -m src.report solve             # -> reports/solve.md
python -m src.cluster.kselect          # Laplacian spectra -> reports/spectrum.md
python -m src.cluster.sweep            # parameter sweep, commits after each configuration
