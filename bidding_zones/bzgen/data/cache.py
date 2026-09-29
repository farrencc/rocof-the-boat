"""Download-once cache with provenance.

Every external file enters the project through :func:`fetch`.  It is written
under ``data/cache/`` (gitignored) and never downloaded twice; each fetch
records URL, access date, size and SHA-256 in ``data/provenance.json``, from
which ``data/PROVENANCE.md`` (committed) is regenerated.

Large families of small files (the Open-Meteo weather points) are recorded as
one aggregate provenance entry by their fetcher rather than one row per file;
see :func:`record`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
PROV_JSON = ROOT / "data" / "provenance.json"
PROV_MD = ROOT / "data" / "PROVENANCE.md"

USER_AGENT = "rocof-the-boat bidding-zone research PoC (python-requests)"


class FetchError(RuntimeError):
    """A source could not be obtained. Never swallowed into a fallback silently."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_prov() -> dict:
    if PROV_JSON.exists():
        return json.loads(PROV_JSON.read_text())
    return {}


def record(key: str, url: str, path: Path | None = None, notes: str = "",
           sha: str | None = None, size: int | None = None) -> None:
    """Add or update one provenance entry and regenerate PROVENANCE.md."""
    prov = _load_prov()
    if path is not None and path.exists():
        sha = sha or sha256(path)
        size = size if size is not None else path.stat().st_size
    prov[key] = {
        "url": url,
        "accessed": dt.date.today().isoformat(),
        "file": str(path.relative_to(ROOT)) if path is not None else "",
        "bytes": size,
        "sha256": sha,
        "notes": notes,
    }
    PROV_JSON.parent.mkdir(parents=True, exist_ok=True)
    PROV_JSON.write_text(json.dumps(prov, indent=1, sort_keys=True))
    _write_md(prov)


def _write_md(prov: dict) -> None:
    lines = [
        "# Data provenance",
        "",
        "Generated from `data/provenance.json` by `bzgen/data/cache.py`. Every external",
        "input is listed with its URL, access date and SHA-256 of the cached file.",
        "",
        "| key | URL | accessed | bytes | sha256 | notes |",
        "|---|---|---|---|---|---|",
    ]
    for k in sorted(prov):
        e = prov[k]
        sha = (e.get("sha256") or "")[:16] + ("…" if e.get("sha256") else "")
        lines.append(f"| `{k}` | {e['url']} | {e['accessed']} | {e.get('bytes') or ''} "
                     f"| `{sha}` | {e.get('notes', '')} |")
    PROV_MD.write_text("\n".join(lines) + "\n")


def fetch(url: str, name: str, key: str | None = None, notes: str = "",
          params: dict | None = None, timeout: int = 300, retries: int = 5,
          expect_md5: str | None = None) -> Path:
    """Return the cached path for ``url``, downloading it once if needed.

    ``expect_md5`` (e.g. from a Zenodo record) is verified on every call; a
    mismatch raises rather than proceeding with an unexpected file.
    """
    path = CACHE / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        delay = 2.0
        for attempt in range(retries):
            try:
                with requests.get(url, params=params, stream=True, timeout=timeout,
                                  headers={"User-Agent": USER_AGENT}) as r:
                    if r.status_code in (429, 500, 502, 503, 504):
                        raise requests.HTTPError(f"HTTP {r.status_code}")
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                tmp.rename(path)
                break
            except (requests.RequestException, OSError) as e:
                if attempt == retries - 1:
                    raise FetchError(f"could not fetch {url}: {e}") from e
                time.sleep(delay)
                delay *= 2
    if expect_md5 is not None:
        got = md5(path)
        if got != expect_md5:
            raise FetchError(f"md5 mismatch for {name}: expected {expect_md5}, got {got}. "
                             "Refusing to use an unexpected file.")
    full = url if not params else requests.Request("GET", url, params=params).prepare().url
    prov = _load_prov()
    k = key or name
    if k not in prov or prov[k].get("sha256") != sha256(path):
        record(k, full, path, notes=notes)
    return path
