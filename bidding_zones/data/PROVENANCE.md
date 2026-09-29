# Data provenance

Generated from `data/provenance.json` by `bzgen/data/cache.py`. Every external
input is listed with its URL, access date and SHA-256 of the cached file.

| key | URL | accessed | bytes | sha256 | notes |
|---|---|---|---|---|---|
| `electricitymaps_zones` | https://raw.githubusercontent.com/electricitymaps/electricitymaps-contrib/809539d5c76de19291ff6cb7766aca0f583dd826/geo/world.geojson | 2026-09-29 | 2568315 | `d1bcf7d3c68ba8ab…` | Electricity Maps electricitymaps-contrib geo/world.geojson at commit 809539d5c76de19291ff6cb7766aca0f583dd826; curated open zone geometry, NOT an ENTSO-E publication (fallback: no authoritative bidding-zone polygons reachable) |
| `eurostat_demo_r_pjanaggr3` | https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/demo_r_pjanaggr3?format=TSV&compressed=true | 2026-09-29 | 2019779 | `eacde2505c811c0a…` | Eurostat population on 1 January by broad age group, sex and NUTS 3, SDMX TSV |
| `eurostat_nama_10r_3gdp` | https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nama_10r_3gdp?format=TSV&compressed=true | 2026-09-29 | 743572 | `3cccb6e2f2f66958…` | Eurostat nama_10r_3gdp, SDMX TSV (full dataset) |
| `eurostat_nama_10r_3popgdp` | https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nama_10r_3popgdp?format=TSV&compressed=true | 2026-09-29 | 123058 | `a56a5993001e1057…` | Eurostat nama_10r_3popgdp, SDMX TSV (full dataset) |
| `eurostat_nrg_bal_peh` | https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nrg_bal_peh?format=TSV&compressed=true | 2026-09-29 | 2115156 | `0cb72a9d18303b03…` | Eurostat gross and net production of electricity and derived heat by type of plant and operator, GWh |
| `eurostat_nrg_cb_e` | https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nrg_cb_e?format=TSV&compressed=true | 2026-09-29 | 184396 | `e1248d9b3d9b1644…` | Eurostat nrg_cb_e, SDMX TSV (full dataset) |
| `eurostat_nrg_inf_epcrw` | https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nrg_inf_epcrw?format=TSV&compressed=true | 2026-09-29 | 53081 | `57d23c9515412954…` | Eurostat nrg_inf_epcrw, SDMX TSV (full dataset) |
| `gisco_countries_2024` | https://gisco-services.ec.europa.eu/distribution/v2/countries/geojson/CNTR_RG_10M_2024_4326.geojson | 2026-09-29 | 3846978 | `be561885519d4bd4…` | GISCO countries 2024, 1:10M, EPSG:4326 |
| `gisco_nuts2021_l3` | https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_01M_2021_4326_LEVL_3.geojson | 2026-09-29 | 28176705 | `554336d416056abc…` | GISCO NUTS 2021 level 3, 1:1M, EPSG:4326 |
| `gisco_nuts2024_l0_10m` | https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_10M_2024_4326_LEVL_0.geojson | 2026-09-29 | 565367 | `d320443cd7fa6116…` | GISCO NUTS 2024 level 0, 1:10M, EPSG:4326 (maps only) |
| `gisco_nuts2024_l3` | https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_01M_2024_4326_LEVL_3.geojson | 2026-09-29 | 27590155 | `2f298a6546e02212…` | GISCO NUTS 2024 level 3, 1:1M, EPSG:4326 |
| `open_meteo_era5_2019` | https://archive-api.open-meteo.com/v1/archive | 2026-09-29 |  | `76a227047f3dc319…` | 323/323 points, models=era5, hourly wind_speed_100m,shortwave_radiation, 2019; sha256 over concatenated per-point JSON (sorted lat,lon); points in data/weather_points.csv |
| `opsd_time_series_60min` | https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv | 2026-09-29 | 130339665 | `6a7f2bc571314cbf…` | OPSD Time series package, latest release, 60-min single-index CSV |
| `osm_buses.csv` | https://zenodo.org/api/records/18619025/files/buses.csv/content | 2026-09-29 | 805469 | `f50c75ea0339f557…` | PyPSA-Eur OSM network v0.7, Zenodo 18619025; md5 031c30f04dc24e99210b3f5ad8a01c94 verified |
| `osm_converters.csv` | https://zenodo.org/api/records/18619025/files/converters.csv/content | 2026-09-29 | 9288 | `7564f11e58604a01…` | PyPSA-Eur OSM network v0.7, Zenodo 18619025; md5 fe13a4f336fdc273c808cdfc485b4753 verified |
| `osm_lines.csv` | https://zenodo.org/api/records/18619025/files/lines.csv/content | 2026-09-29 | 19806938 | `d3ee4f45a0d9889e…` | PyPSA-Eur OSM network v0.7, Zenodo 18619025; md5 04d1e7cb33835de9c7d8be4716c6a8bf verified |
| `osm_links.csv` | https://zenodo.org/api/records/18619025/files/links.csv/content | 2026-09-29 | 322916 | `2c7b4769fecc68bb…` | PyPSA-Eur OSM network v0.7, Zenodo 18619025; md5 2ddd515a30cfc19205e6681def579fbd verified |
| `osm_record_18619025` | https://zenodo.org/api/records/18619025 | 2026-09-29 |  | `` | Zenodo metadata: 'Prebuilt Electricity Network for PyPSA-Eur based on OpenStreetMap Data' v0.7, published 2026-02-12 |
| `osm_transformers.csv` | https://zenodo.org/api/records/18619025/files/transformers.csv/content | 2026-09-29 | 121972 | `65ac747f623da832…` | PyPSA-Eur OSM network v0.7, Zenodo 18619025; md5 b6730b52097d2691d49231bdd53c7bd4 verified |
| `powerplantmatching` | https://raw.githubusercontent.com/PyPSA/powerplantmatching/v0.8.0/powerplants.csv | 2026-09-29 | 20984705 | `eb4fceb8285322dd…` | powerplantmatching 0.8.0 prepared (matched) dataset, tag v0.8.0 |
