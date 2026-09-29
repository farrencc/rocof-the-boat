"""PyPSA-Eur prebuilt OSM network (Zenodo), verified against the record's md5.

The record ID is not guessed: it is the ``osm`` v0.7 *primary* entry in
PyPSA-Eur's ``data/versions.csv`` (master, read 2026-09-29), and the Zenodo
API confirms its title ("Prebuilt Electricity Network for PyPSA-Eur based on
OpenStreetMap Data") and version (0.7) before any file is used.
"""

from __future__ import annotations

import requests

from src import config
from src.data.cache import FetchError, fetch, record

ZENODO_API = "https://zenodo.org/api/records/{rid}"
EXPECTED_TITLE = "Prebuilt Electricity Network for PyPSA-Eur based on OpenStreetMap Data"


def fetch_osm(cfg: dict | None = None) -> dict:
    cfg = cfg or config.load()
    rid = cfg["sources"]["osm_zenodo_record"]
    meta = requests.get(ZENODO_API.format(rid=rid), timeout=60).json()
    title = meta["metadata"]["title"]
    version = meta["metadata"].get("version")
    if title != EXPECTED_TITLE:
        raise FetchError(f"Zenodo record {rid} is '{title}', not the PyPSA-Eur OSM network")
    record(f"osm_record_{rid}", ZENODO_API.format(rid=rid),
           notes=f"Zenodo metadata: '{title}' v{version}, "
                 f"published {meta['metadata']['publication_date']}")
    files = {f["key"]: f for f in meta["files"]}
    paths = {}
    for name in cfg["sources"]["osm_files"]:
        if name not in files:
            raise FetchError(f"{name} missing from Zenodo record {rid}")
        f = files[name]
        algo, digest = f["checksum"].split(":")
        assert algo == "md5"
        paths[name] = fetch(f["links"]["self"], f"osm/{name}", key=f"osm_{name}",
                            expect_md5=digest,
                            notes=f"PyPSA-Eur OSM network v{version}, Zenodo {rid}; "
                                  f"md5 {digest} verified")
    return paths


if __name__ == "__main__":
    for k, v in fetch_osm().items():
        print(k, v)
