#!/usr/bin/env bash
# Full pipeline, in order. Every stage caches/persists; re-running skips completed work
# where it can (downloads, sweep configurations).
set -euo pipefail
cd "$(dirname "$0")"                      # run from bidding_zones/
if [ -f .venv/bin/activate ]; then . .venv/bin/activate
elif [ -f ../.venv/bin/activate ]; then . ../.venv/bin/activate; fi
python -m bzgen.data.topology            # PyPSA-Eur OSM network (Zenodo, md5-verified)
python -m bzgen.data.weather fetch       # ERA5 via Open-Meteo (slow: rate limits)
python -m bzgen.report data              # assemble network + loads + fleet -> reports/data.md
python -m bzgen.solve.run                # 8760 h DC-OPF -> data/solved/
python -m bzgen.cluster.prepare          # edge stats + normalisation gate -> reports/normalisation.md
python -m bzgen.report solve             # -> reports/solve.md
python -m bzgen.cluster.kselect          # Laplacian spectra -> reports/spectrum.md
python -m bzgen.cluster.sweep            # parameter sweep, commits after each configuration
python -m bzgen.validate.run             # zonal market + redispatch, DE split vs k=1 -> reports/validate.md
