# PLAN — Potts-model bidding-zone generation (proof of concept)

This is the working plan, written before any code. It will be amended in place
(with a dated changelog at the bottom) when reality disagrees with it.

## 0. Reconnaissance already done (read-only, before writing this)

| What | Finding |
|---|---|
| Compute | 4 cores, 15 GB RAM, ~29 GB free disk, Python 3.11. |
| Topology | PyPSA-Eur's own `data/versions.csv` (master) lists the prebuilt OSM network: **`osm` v0.7, primary `https://zenodo.org/records/18619025/files`**. Zenodo API confirms record 18619025 = *"Prebuilt Electricity Network for PyPSA-Eur based on OpenStreetMap Data"*, v0.7, 2026-02-12, Xiong/Fioriti/Neumann/Riepin/Brown, 220–750 kV. Files: `buses.csv` (0.8 MB), `lines.csv` (19.8 MB), `links.csv`, `converters.csv`, `transformers.csv`, `map.html`, with md5 checksums. The `data.pypsa.org` archive mirror is **blocked by this session's egress policy** (CONNECT rejected); Zenodo is reachable, so the mirror is not needed. |
| Weather | `archive-api.open-meteo.com` reachable. The very first request got **HTTP 429**, a retry succeeded → the free tier is rate-limited and the egress IP may be shared. Fetching must be paced and resumable. |
| Other hosts | OPSD (`data.open-power-system-data.org`), Eurostat API, GISCO, `raw.githubusercontent.com` (powerplantmatching data), PyPI: reachable. ENTSO-E transparency, Ember, JRC data portal: **not reachable** (not needed). |

## 1. Open scope questions (defaults chosen; please redirect if wrong)

1. **Northern Ireland.** In the PyPSA-Eur data NI buses carry country code `GB`.
   "Exclude Great Britain" is ambiguous for NI (UK, but in the all-island SEM with IE).
   **Default: drop everything coded `GB`, including NI.** Ireland then has no AC
   neighbour and its HVDC links to GB are dropped; IE is partitioned alone.
   Alternative: keep NI and merge it into IE as the SEM. Say if you prefer that.
2. **Cyprus and Malta** are EU members but (I expect) absent from the PyPSA-Eur
   network. If so they are reported as "no topology" and skipped — not invented.
3. **Microstate merge target.** LU, ME, LI, AD, MC, SM, VA (and any country with
   fewer than `min_buses` buses after simplification, default 5) are merged into the
   neighbour to which they have the largest total cross-border susceptance. The
   merged country is then partitioned as part of that neighbour. This is the one
   deliberate exception to "a zone never spans two countries", as the brief implies.
4. **GB interconnectors** (to FR, BE, NL, NO, DK, IE) are dropped with GB. This removes
   GB import/export from the OPF; noted as a limitation.

## 2. Phases

Each phase ends with a commit + push to `claude/serene-darwin-sizwdl`, and a
short status message in chat. Existing `IzzyMatt/` content is not touched.

### Phase 0 — scaffolding
- Directory layout from the brief (`src/{data,network,solve,cluster,plot}`, `config/`,
  `results/`, `figures/`, `reports/`, `data/cache/` gitignored, `tests/`).
- `requirements.txt` with **pinned** versions (pypsa, linopy, highspy, pandas, numpy,
  scipy, geopandas, shapely, pyproj, networkx, numba, matplotlib, pyyaml, requests,
  powerplantmatching, pytest), installed into a venv; `pip freeze` recorded.
- `config/default.yaml` holding every tunable (no magic constants in code).
- `src/data/cache.py`: one `fetch(url, name)` helper — on-disk cache under
  `data/cache/`, SHA-256 of every file, and an append/update into
  `data/PROVENANCE.md` (URL, access date, size, hash, notes).
- README skeleton with the **nodal-price-proxy limitation at the top**.

### Phase 1 — data acquisition (`src/data/`)
1. **Topology**: the six Zenodo files above; verify each md5 against the Zenodo API
   record before use. Stop and report on mismatch.
2. **NUTS3 geometry**: GISCO NUTS 2021 (the version Eurostat's regional
   population/GDP table is coded in — to be verified from the data, not assumed).
   Country outlines (NUTS0) for maps from the same source.
3. **Population + GDP per NUTS3**: Eurostat `nama_10r_3popgdp` (population) and
   `nama_10r_3gdp` (GDP), latest common year. Non-EU countries without NUTS3 data
   (e.g. BA, XK) → buses there get equal weights, **flagged in the report**.
4. **Hourly load**: OPSD `time_series` 60-min single-index CSV (latest release), weather
   year **2019** (latest full pre-COVID year in OPSD; see §4). Countries with no OPSD
   series get a neighbour's normalised profile scaled to their own annual total —
   **loud flag** in `reports/data.md`. If OPSD fails entirely → documented synthetic
   profile, flagged everywhere downstream (not expected).
5. **2025 scaling**: Eurostat `nrg_cb_e` annual electricity consumption; factor =
   (latest available year, ideally 2025) / 2019 from the *same* series so definitional
   differences with OPSD's load cancel. Year actually used is recorded per country.
6. **Generation fleet**: `powerplantmatching` prepared dataset (pinned release), filtered to
   plants operating in 2025. Because powerplantmatching is known to under-cover wind and
   (especially) solar, national wind/solar totals are checked against Eurostat
   `nrg_inf_epcrw` (installed renewable capacity); where Eurostat is higher, the gap is
   distributed spatially (onshore wind ∝ existing ppm wind at the bus, else ∝ area; solar
   ∝ population). The decision and per-country numbers go in `reports/data.md`.
   Marginal costs from a hardcoded, cited fuel-price × heat-rate table in the config.
7. **Weather**: Open-Meteo ERA5 archive (`models=era5`), hourly `wind_speed_100m` and
   `shortwave_radiation` for 2019 on a regular grid (target ~250–350 points, 1.0–1.25°,
   kept only where buses are within reach, offshore included). One cached JSON per point,
   resumable, paced well under the free-tier limits (≈26 weighted calls per point-year),
   exponential backoff on 429/5xx. **This runs in the background from early Phase 1**
   because it is the slowest input (expected 1–3 h).
   - Wind CF: documented generic 3 MW-class power curve (cut-in 3, rated 12, cut-out
     25 m/s) with a flat availability/wake loss; offshore uses the same curve.
   - PV CF: `PR × GHI / 1000` with a stated performance ratio (horizontal irradiance, no
     transposition — flagged as a simplification).
   - Annual mean CFs per country reported against typical published values as a sanity
     check (not a calibration).

### Phase 2 — network assembly and simplification (`src/network/`)
- Drop GB buses (and incident branches), out-of-scope countries, under-construction assets.
- Emulate PyPSA-Eur `simplify_network`: convert every line's reactance to a 380 kV
  equivalent (`x·(380/v)²`), contract transformers (merge the buses they join),
  contract AC–DC converters so HVDC links run AC-bus to AC-bus, merge parallel
  lines (susceptances and ratings add), iteratively absorb degree-1 stubs **within a
  country only** (load/generation moved to the neighbour).
- Report per stage: buses/lines/links removed and surviving, per country.
- If `lines.csv` lacks `x`/`s_nom`, derive them from PyPSA standard line types exactly as
  PyPSA-Eur's `base_network` does — marked as inference.
- `s_max_pu` (N−1 proxy) from config, default 0.7 as in PyPSA-Eur.
- Loads to buses ∝ `w_pop·pop + w_gdp·gdp` of the bus's NUTS3 region (point-in-polygon;
  a NUTS3 region's value is split equally among its buses; regions with no bus hand their
  weight to the nearest bus). Plants → nearest bus in the same country.

### Phase 3 — nodal prices (`src/solve/`)
- PyPSA DC-OPF (linopy + HiGHS), fixed capacities, no storage/UC/CO₂, `s_nom·s_max_pu`
  flow limits. Every hour is an independent LP; hours are grouped into small batches
  (e.g. 24 per LP) purely for model-building efficiency and solved in parallel on 4
  cores. Output: per-bus prices, dispatch, line flows → `data/solved/` (parquet),
  plus a committed summary.
- **Snapshots**: start with ~300 representative hours chosen by k-medoids on
  standardised national load / wind / solar features of 2019; **weight = cluster size**.
  Weights are carried through every average from day one. Scale up (more medoids, or
  all 8760 h with weight 1) if time allows.
- **Hydro without storage** (unavoidable given "no storage"): run-of-river at a flat
  capacity factor, reservoir hydro dispatchable at a documented water value, pumped
  storage excluded. This matters for NO/CH/AT/SE and is flagged as a key limitation.
- **Load shedding**: a shedding generator at every bus at a high cost (config). Snapshots
  with any shedding are counted and reported; **default: exclude them** from the edge
  statistics (with their weight) rather than clip, because clipping still leaves an
  arbitrary price level in the statistic. Sensitivity with clipping is available.
- **Tie-breaking and dual degeneracy**: a tiny seeded random perturbation of marginal
  costs (as PyPSA-Eur does) breaks exact ties. Degeneracy is then *measured*: a
  subsample of snapshots is re-solved with a different perturbation seed; bus-snapshots
  whose price moves by more than a tolerance are counted as dual-degenerate. The fraction
  is reported per country; not averaged away.

### Phase 4 — edge statistics + **normalisation checkpoint (gate)** (`src/cluster/`)
- Intra-country edges only (cross-border dropped entirely, including from statistics).
  Parallel lines merged. HVDC links: in the adjacency/contiguity graph always; in the
  energy with their Δp and **J = 0** (config: `dc_in_energy`), since a controllable link
  has no susceptance. J statistics are over AC edges only.
- Δp statistics (all weighted): mean |Δp|, **duration** = weighted fraction of hours with
  |Δp| > threshold, and a weighted high quantile. **Default = duration**, because Art. 14
  speaks of *structural/persistent* congestion, which the mean conflates with rare spikes.
  Threshold from config (default €1/MWh, above the perturbation noise floor).
- J = 1/x (380 kV-equivalent). Normalise each quantity by its per-country mean.
- **Gate**: per-country report + plots of the distributions of Δp̃ and J̃: mean (must be
  1), skew, top-1 % share. If dominated by outliers, apply a documented remedy (expected:
  log-transform or rank/quantile-clip J; clip Δp̃ at a high quantile), re-check, and
  write `reports/normalisation.md`. Countries whose nodal prices show essentially no
  intra-country divergence are flagged "no congestion signal" — clustering there is
  driven by J and the balance term only, and the report will say so.
- **Commit, push and report in chat before Phase 5.** Proceed with the documented remedy
  unless the finding implies a scope change, in which case stop and ask.

### Phase 5 — spectral k (`src/cluster/spectrum.py`)
- Normalised Laplacian `I − D^-½ W D^-½` of the intra-country graph with W = J̃ (post
  remedy); HVDC edges weighted at the country's median J̃ so islands are attached.
- Full spectrum stored and plotted for every country. Elbow via largest eigengap
  among the first `k_max` eigenvalues; report location, gap magnitude, and a
  "convincing" flag (largest gap ≥ `gap_ratio`× the next largest). Otherwise a documented
  fallback k, stated as such in the report.

### Phase 6 — annealer (`src/cluster/anneal.py`, numba)
- H exactly as in the brief. Potts and balance terms: O(degree) / O(1) incremental.
- **Contiguity** `C_s`: maintained incrementally. Joining zone b: ΔC_b = 1 − (#distinct
  b-components among i's b-neighbours). Leaving zone a: ΔC_a = (#components the a-neighbours
  fall into once i is removed) − 1, computed by an **interleaved local BFS from the
  neighbours with early termination**. This is O(degree) whenever the neighbours reconnect
  locally, and bounded by the smaller split fragment otherwise; it is never a full
  recomputation. (Stated honestly: exact contiguity cannot be strictly O(degree).)
- Proposals: uniformly random **boundary** node, target label drawn from its neighbours'
  labels (HVDC neighbours included).
- **Empty zones: moves that would empty a zone are rejected** (k stays fixed).
- Balance: hinge on `max(load_s, gen_s)/national_load` below floor (default 5 %),
  quadratic, small λ_b. Documented as a proxy for the BZR liquidity indicator with no
  basis in Art. 14.
- λ_c ramped up as T falls; final zero-temperature quench; final contiguity reported.
  Energies compared only at equal final λ_c.
- Initial states: graph-Voronoi growth from k random seeds (contiguous by construction).
  R restarts per country-configuration; keep the minimum; record energy spread and the
  pairwise adjusted Rand index between the best restarts (degeneracy report).
- **Validation first**: planted-partition synthetic graph (known zones, planted Δp/J),
  recovery measured by ARI; plus pytest for H, incremental-vs-full agreement (random move
  sequences), and the contiguity counter against networkx.

### Phase 7 — sweeps (`src/cluster/sweep.py`)
- One-at-a-time around a baseline rather than full factorial (to keep commits and time
  bounded): α ∈ {0.25, 0.5, 0.75, 1, 1.5, 2, 3}; λ_c,final ∈ 3 values; λ_b ∈ 3 values;
  k ∈ {k*−1, k*, k*+1, k*+2}. Each configuration runs all countries in parallel, then
  writes results, diagnostics and figures, appends to `results/sweep.csv`, **commits and
  pushes** before the next. Resumable: configurations with existing outputs are skipped.
- `sweep.csv` columns: config id, params, country, n_buses, k, final energy, restart
  min/median/max, restart ARI, zone sizes (buses and load share), contiguity satisfied,
  wall time.

### Phase 8 — figures + write-up
- Per-country zone maps per configuration (zones as dissolved bus-Voronoi cells clipped to
  the country outline; lines drawn; HVDC distinct), stitched European headline map,
  Laplacian spectra with elbow, normalisation diagnostics, α sensitivity (#zones, energy),
  nodal price map (weighted mean price and duration of divergence).
- README: method, all limitations (nodal-price proxy first), how to reproduce.

## 3. Principles
- No fabricated data or URLs. Anything inferred is marked `INFERENCE:` in code and listed
  in the relevant report.
- Failures are loud: data gaps, synthetic fills, meaningless elbows, and non-contiguous
  results are written into the reports and the sweep table, not only into comments.
- Nothing downstream re-solves the OPF; nothing re-downloads.

## 4. Known modelling limitations (to carry into the README)
1. Nodal prices are a proxy — the zones we draw would change the dispatch that produced them.
2. Weather/load year 2019 with a 2025 fleet and 2025 demand level (demand shape is 2019).
3. No storage / hydro reservoir energy limits → hydro-dominated prices are crude.
4. GB (and, by default, NI) dropped with its interconnectors.
5. Per-country normalisation → α not comparable across countries; k = 1 effectively unreachable.
6. Representative-hour subsampling approximates duration statistics.

## Changelog
- 2026-09-29 — initial plan.
- 2026-09-29 — amendments while building (each is also documented where it applies):
  - **Weather**: 303→323 points (hex grid 150 km, not 1°) to stay inside Open-Meteo's
    free quota (≈27 weighted calls per point-year). The egress IP pool is shared: many
    requests get "daily limit exceeded" 429s from one IP and succeed on another, so the
    fetcher retries after 30 s instead of sleeping an hour. YAML parsed `NO` as `false`
    (Norway silently dropped) — caught, codes quoted, a config check added.
  - **Load key**: PyPSA-Eur's weights are GDP 0.6 / population 0.4 (my first draft had
    them swapped). Population from `demo_r_pjanaggr3` (covers CH/NO/AL), NUTS 2024.
  - **Hydro**: capping reservoirs at their annual-average CF made winter peaks infeasible
    (NO short by 6.5 GW). Reservoirs are now dispatchable to 90 % at the water value, with
    no energy limit (no storage); annual output reported against Eurostat.
  - **Solver**: one `n.optimize()` call took ~150 s outside HiGHS for 4 hours. The hourly
    LP is now assembled once from the PyPSA network and solved with highspy, warm-started,
    ~0.1 s/hour; prices and objective checked identical to `n.optimize()`. Consequence:
    **all 8760 hours are solved (weight 1)**; the k-means subset code stays available.
    Angle variables are rescaled (b/median b) — HiGHS failed on the raw [1, 3e6] range;
    failures fall back to cold restart then IPM, and every retry is logged.
  - **Load shedding** occurs in ~97 % of hours at a handful of chronic "load pockets"
    (single 220 kV buses inside dense NUTS3 regions: Munich, Paris, Oslo, Stockholm, Cádiz,
    Nice, Dublin) — a 220 kV-truncation + allocation artefact. Voronoi-overlap allocation
    was tested and moved rather than removed them. Snapshot exclusion is therefore
    infeasible; **policy = clip prices at ±500 EUR/MWh** (reports/solve.md).
  - **Annealer**: single flips froze with locked-in fragments; added **fragment (cluster)
    moves** and an automatic λ_c,final above any possible Potts+balance change, so final
    states are contiguous by construction. λ_c,initial is what the sweep varies.
    Validated on planted partitions (reports/annealer_validation.md).
  - **Eigengap**: the plain largest gap is biased to k_max on near-planar graphs; the
    elbow now uses gap / median(neighbouring gaps), and k ≤ max(2, n/4).
  - **Energy sign structure** (dev finding, to be confirmed on real data): with J̃
    strongly skewed, most edges have w = Δp̃ − αJ̃ > 0 at α = 1, so the Potts model is
    antiferromagnetic on most edges and minimisers are contiguous but interdigitated. The
    α range is extended to {…, 5, 10}; share of w>0 edges and cut ratio are recorded.
