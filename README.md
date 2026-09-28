# rocof-the-boat

Does **SNSP** or **ΔP / H_COI** better predict frequency-security violations on the all-island Irish system?

This repository holds the dispatch ensemble and the frequency layer built to answer that question. Start with [`docs/ENSEMBLE.md`](docs/ENSEMBLE.md). It records which inputs are real (the TYTFS 2024 network) and which are stand-ins (synthetic weather, no validation against history). §4 explains why the RoCoF comparison is currently circular.

```bash
pip install -r requirements.txt
python ensemble.py time                     # one scenario, timed
python ensemble.py run --n 50 --workers 4   # smoke-test ensemble -> data/ensemble/
python frequency.py data/ensemble/2024_seed42_n50
python -m pytest -q
```

## Where the code comes from

`psse.py`, `pypsa_net.py`, `synthetic.py` and `profiles.py` are copied unchanged from
[farrencc/Hackathons `grid_TF_Wind/`](https://github.com/farrencc/Hackathons/tree/main/grid_TF_Wind), the TPSA & TFWind 2026 pipeline, along with the TYTFS v35 cases and the geocoding tables they read. The original development history is in
[farrencc/Hackathons#3](https://github.com/farrencc/Hackathons/pull/3).
The TYTFS files are EirGrid's; see `data/TYTFS2024_studyfiles/DISCLAIMER - read me.txt`.
