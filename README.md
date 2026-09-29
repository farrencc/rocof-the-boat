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

Candidate bidding-zone boundaries built from first principles, in the spirit of
Regulation (EU) 2019/943 Art. 14 (zones based on long-term structural congestion):

1. **Network**: PyPSA-Eur's prebuilt OSM grid (v0.7, Zenodo 18619025, md5-verified):
   EU members present in the data, plus NO, CH and the Western Balkans. GB is excluded,
   and so is Northern Ireland, which the source codes as GB. Simplified the way
   PyPSA-Eur does it: 380 kV-equivalent reactances, transformers contracted,
   same-country stubs absorbed. 3771 buses.
2. **Inputs, 2025**: OPSD 2019 hourly load shape scaled by Eurostat 2025 consumption;
   GDP/population NUTS3 load key; powerplantmatching fleet with wind/solar reconciled
   to Eurostat capacity; ERA5 wind and irradiance via Open-Meteo at ~320 points; onshore
   wind calibrated to Eurostat-observed 2019 capacity factors.
3. **Nodal prices**: DC-OPF for all 8760 hours of weather year 2019. Every hour is an
   independent LP, solved with HiGHS and cross-checked against `pypsa.Network.optimize()`.
4. **Edge statistics**: for every intra-country line, a persistence statistic of the price
   gap (default: share of hours with |Δp| > €1/MWh) and its susceptance J = 1/x, each
   normalised by the country mean.
5. **Zones**: each country is partitioned independently by simulated annealing of

   ```
   H = Σ_(i,j) (Δp̃_ij − α J̃_ij) δ(σ_i, σ_j) + λ_c Σ_s (C_s − 1) + λ_b Σ_s hinge_s
   ```

   with k fixed per country (Laplacian eigengap, or a flagged fallback), contiguity
   counting HVDC links as adjacency, and a small balance hinge on max(load, generation)
   share.

Deliverable: one minimum-energy map per parameter set (`results/`, `figures/`), with
restart spread and degeneracy reported alongside it.

## Results in one paragraph

Headline map (α = 2): [`figures/europe/a2_lc0.1_lb0.5_dk+0.png`](figures/europe/a2_lc0.1_lb0.5_dk+0.png).
Full discussion: [`reports/results.md`](reports/results.md).

The pipeline runs end to end on real data. It produces contiguous zones for 31
countries under 17 parameter sets. **The central finding is negative.** For the large
countries (FR, ES, DE, IT, PL, SE) the minimum-energy map is not identifiable:
structurally different partitions (ARI 0.1–0.3 between them) lie within about 1 % of the
best energy found, and a 4× longer anneal finds a different map. α trades interdigitated
zones (small α; about half of all edges are repulsive at α = 1) against microzones
(large α, fixed k). No Laplacian eigengap is statistically meaningful, so every k is a
default. Small countries give reproducible maps, but k there is a default too. The
underlying reason: nodal price differences in a meshed network form regional gradients,
not steps at congested lines, so edge-wise |Δp| does not localise a boundary. These maps
are a demonstration of the method and its failure modes, not candidate zones for use.

## Where to look

| file | what |
|---|---|
| `PLAN.md` | the plan, and a changelog of every deviation and why |
| `data/PROVENANCE.md` | URL, access date and hash of every input |
| `reports/data.md` | data assembly; **all fallbacks and gaps flagged at the top** |
| `reports/solve.md` | OPF: shedding, prices, solver retries |
| `reports/normalisation.md` | the normalisation gate: distributions, remedy, degeneracy |
| `reports/spectrum.md` | Laplacian spectra, eigengap, the k used per country |
| `reports/annealer_validation.md` | annealer validated on planted partitions |
| `reports/results.md` | results, degeneracy, convergence check, sensitivity |
| `results/sweep.csv` | one row per country × configuration |
| `figures/` | maps, spectra, diagnostics, sensitivity |

## Limitations (read before using any map)

1. **Nodal prices are a proxy** (above). This is the most important limitation.
2. **Weather and load shape are from 2019**, combined with a 2025 fleet and a 2025 demand level.
3. **No storage, so hydro is crude.** Reservoirs dispatch at a water value with no energy
   limit, and pumped storage and batteries are absent. Prices in NO, SE, CH and AT are
   therefore crude.
4. **The grid is truncated at 220 kV.** The 110 kV networks that also feed cities are
   missing, which produces chronic "load pockets" that shed load in most hours (Munich,
   Paris, Oslo, Stockholm, Cádiz and others; see `reports/solve.md`). Prices are clipped
   at ±€500/MWh before statistics are computed.
5. **Normalisation is per country, so α is not comparable across countries.** Because
   both terms have mean 1 in every country, the energy always rewards some split, and
   k = 1 is effectively unreachable. This is a modelling choice, not a result.
6. **The balance term has no basis in Art. 14.** It is a proxy for the bidding zone
   review's market-liquidity indicator, and Art. 14 explicitly excludes zone size as a
   splitting criterion.
7. **Some countries are missing or merged.** GB and NI are dropped along with their
   interconnectors, as are UA and MD. CY and MT have no topology in the source. LU and ME
   are merged into DE and BA.
8. **Marginal costs are the author's estimates** of 2025 levels, not sourced data (`config/default.yaml`).
9. **k is usually a default, not a finding.** The Laplacian elbow is weak or absent for
   most countries (`reports/spectrum.md`).

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt     # requirements-lock.txt for the full resolved set
pytest tests/                       # energy, incremental update, contiguity, planted recovery
./run_pipeline.sh                   # every stage caches; the sweep resumes where it stopped
```

## Layout

```
src/data/      fetchers, caching, provenance        src/solve/    DC-OPF (highspy), runner
src/network/   simplification, load key, fleet      src/cluster/  edge stats, spectrum, annealer, sweep
src/plot/      maps and figures                     config/       default.yaml (every parameter)
results/ figures/ reports/   committed outputs      data/cache/   downloads (gitignored)
```
