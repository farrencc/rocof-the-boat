# rocof-the-boat — Potts-model bidding-zone generation (research PoC)

> ## ⚠ Read this first: nodal prices are a proxy
>
> Zones here are drawn from **nodal prices**: prices that would clear under
> *nodal* pricing on a DC approximation of the European grid. If you imposed
> the zones we produce, dispatch would change, so congestion would change,
> so the prices would change. The prices we cluster on are **not** the prices
> that would clear under our zones. Closing that loop (re-solving under the
> proposed zones and iterating) is future work. This proof of concept does not
> do it.

## What this is

Candidate bidding-zone boundaries built from first principles, as described
in Regulation (EU) 2019/943 Art. 14 (zones based on long-term structural
congestion):

1. Assemble a nodal European network (PyPSA-Eur's prebuilt OSM grid, EU +
   NO + CH + Western Balkans, GB excluded) with a 2025 fleet and demand level.
2. Solve DC optimal power flow hour by hour for nodal prices.
3. For every intra-country line, compute a congestion statistic of the price gap
   and the line's structural coupling (susceptance).
4. Partition each country independently by minimising a Potts energy with
   simulated annealing.

See `PLAN.md` for the full plan and `reports/` for the diagnostics at each gate.

## Status

Work in progress. See `PLAN.md` changelog.

## Layout

```
src/data/      fetchers, caching, provenance      data/PROVENANCE.md
src/network/   assembly, simplification
src/solve/     parallel DC-OPF
src/cluster/   edge stats, spectrum, annealer
src/plot/      figures
config/        YAML parameter sets
results/ figures/ reports/   committed outputs
data/cache/    downloads (gitignored)
```

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt     # requirements-lock.txt for the full resolved set
pytest tests/
```
