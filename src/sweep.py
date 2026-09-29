"""Step 5: coupling-constant sweep with checkpointing.

    python src/sweep.py --scope 26 [--workers 4] [--only c00_dd_only,c01_base] [--no-commit]

* Problem setup (training ensemble, WDT-multi baseline, anchors, normaliser) is
  built once and cached under results/setup_<scope>/.
* Each configuration runs in a worker process. After every temperature step the
  annealer checkpoints to results/<config_id>/checkpoint.npz; on restart it resumes.
* After every completed configuration the worker writes final artefacts and
  plots, then the parent process git-commits results/<config_id>/.
* A crash in one configuration is logged to results/<config_id>/error.log and
  the sweep moves on.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import upstream as up  # noqa: E402
import anchors as A  # noqa: E402
from evaluator import build_ensemble, wdt_membership, labels_to_membership  # noqa: E402
from hamiltonian import Hamiltonian, Normaliser, TERMS, structural_terms, group_stats  # noqa: E402
from anneal import AnnealConfig, anneal  # noqa: E402

REPO = SRC.parent
RESULTS = REPO / "results"
FIGURES = REPO / "figures"

# lambda order: (DD, S, V, N, P). All terms are z-scored (see hamiltonian.py), so
# lambda = 1 weights one random-membership standard deviation of that term.
BASE = (1.0, 0.5, 0.5, 0.5, 0.5)


# v2 lambda order: (DD, S, V, N, P, C). Base relaxes the size coupling to 0.1 (with a hard
# 5-node floor) and prices conventional redispatch at 0.5.
BASE_V2 = (1.0, 0.5, 0.5, 0.1, 0.5, 0.5)
V2_MIN_GROUP_SIZE = 5
V2_THERMAL_SCALES = (1.0, 0.95, 0.90)


def sweep_configs_v2() -> list[tuple[str, tuple]]:
    cfgs = [("d00_dd_only", (1.0, 0, 0, 0, 0, 0)), ("d01_dd_conv", (1.0, 0, 0, 0, 0, 1.0)),
            ("d02_base", BASE_V2)]
    n = 3
    grid = {"N": (0.0, 0.05, 0.25), "C": (0.0, 2.0), "S": (0.0, 2.0), "V": (0.0, 2.0), "P": (0.0, 2.0)}
    for t, vals in grid.items():
        j = TERMS.index(t)
        for v in vals:
            lam = list(BASE_V2)
            lam[j] = v
            cfgs.append((f"d{n:02d}_{t}{v:g}", tuple(lam)))
            n += 1
    cfgs.append((f"d{n:02d}_structural_only", (0.0, 1.0, 1.0, 0.1, 1.0, 1.0)))
    return cfgs


def sweep_configs(version: str = "v1") -> list[tuple[str, tuple]]:
    if version == "v2":
        return sweep_configs_v2()
    cfgs = [("c00_dd_only", (1.0, 0, 0, 0, 0)), ("c01_base", BASE)]
    n = 2
    for j, t in enumerate(TERMS):
        if t == "DD":
            vals = (0.25, 4.0)
        else:
            vals = (0.0, 2.0)
        for v in vals:
            lam = list(BASE)
            lam[j] = v
            cfgs.append((f"c{n:02d}_{t}{v:g}", tuple(lam)))
            n += 1
    corners = [
        ("structural_only", (0.0, 1.0, 1.0, 1.0, 1.0)),
        ("all_high", (1.0, 2.0, 2.0, 2.0, 2.0)),
        ("hinge_hard", (1.0, 0.5, 0.5, 0.5, 5.0)),
        ("mean_var", (1.0, 2.0, 2.0, 0.0, 0.0)),
        ("size_hinge", (1.0, 0.0, 0.0, 2.0, 2.0)),
        ("dd_heavy", (4.0, 1.0, 1.0, 1.0, 1.0)),
    ]
    for name, lam in corners:
        cfgs.append((f"c{n:02d}_{name}", lam))
        n += 1
    return cfgs


class Problem:
    """Everything a configuration needs, built once per scope."""

    def __init__(self, scope: str = "26", version: str = "v1"):
        self.scope = scope
        self.version = version
        self.grid = up.load_grid(scope)
        self.M_wdt, self.wdt_ids = wdt_membership(self.grid.template)
        self.M_excl, self.excl_ids = labels_to_membership(up.exclusive_group_ids(self.grid))
        if version == "v1":
            self.root = RESULTS
            setup = RESULTS / f"setup_{scope}"
            self.min_size = 1
            self.ens = build_ensemble(self.grid, A.ANCHOR_SEEDS, A.THERMAL_SCALES)
            self.mec = self.grid.template.mec_mw.to_numpy(float)
            self.nodes = None
        else:
            from cases import build_cases_v2, node_table
            self.root = RESULTS / f"v2_{scope}"
            setup = self.root / "setup"
            self.min_size = V2_MIN_GROUP_SIZE
            self.thermal_scales = V2_THERMAL_SCALES
            self.builder = lambda g, s, ts: build_cases_v2(g, s, ts)
            self.ens = build_ensemble(self.grid, A.ANCHOR_SEEDS, V2_THERMAL_SCALES, builder=self.builder)
            self.nodes = node_table(self.grid, self.ens.members[0].cases)
            self.mec = self.nodes.capacity_mw.to_numpy(float)
            pad = len(self.nodes) - self.M_wdt.shape[0]   # WDT groups contain no conventional units
            self.M_wdt = np.vstack([self.M_wdt, np.zeros((pad, self.M_wdt.shape[1]), bool)])
            self.M_excl = np.vstack([self.M_excl, np.zeros((pad, self.M_excl.shape[1]), bool)])
        self.setup = setup
        self.ens.set_baseline(self.M_wdt)
        if (setup / "anchors.npz").exists():
            meta = A.load_anchors(setup)
            self.anchor_meta = meta
            self.sigma, self.M0 = meta["sigma"], meta["init_membership"]
        else:
            if version == "v1":
                a = A.select_anchors(self.grid, self.ens, self.M_wdt)
            else:
                a = A.select_anchors(self.grid, self.ens, self.M_wdt, min_share=0.01,
                                     min_size=self.min_size, repair=True)
            A.save_anchors(a, setup)
            self.anchor_meta = A.load_anchors(setup)
            self.sigma, self.M0 = a.sigma, a.init_membership
        npath = setup / "normaliser.npz"
        if npath.exists():
            z = np.load(npath)
            self.norm = Normaliser(z["offset"], z["scale"], z["probe"])
        else:
            self.norm = Hamiltonian.build_normaliser(self.ens, self.sigma, self.mec, self.M0,
                                                     min_size=self.min_size)
            np.savez(npath, offset=self.norm.offset, scale=self.norm.scale, probe=self.norm.probe,
                     terms=np.array(TERMS[:len(self.norm.offset)]))
            (setup / "normaliser.json").write_text(json.dumps(
                dict(terms=TERMS, offset_at_init=self.norm.offset.tolist(), probe_sd=self.norm.scale.tolist(),
                     probe_mean=self.norm.probe.mean(0).tolist(), n_probe=len(self.norm.probe)), indent=2))

    def hamiltonian(self, lambdas) -> Hamiltonian:
        return Hamiltonian(self.ens, self.sigma, self.mec, lambdas, self.norm)

    def describe(self, M) -> dict:
        ev = self.ens(M)
        return dict(D=ev["D"], C=ev.get("C", 0.0), feasible=ev["feasible"], new_failures=ev["new_failures"],
                    security_pct=ev["security_pct"],
                    raw_terms=dict(zip(TERMS[1:], structural_terms(M, self.sigma, self.mec).tolist())),
                    sizes=M.sum(0).tolist(), covered=int((M.sum(1) > 0).sum()), overlap=int((M.sum(1) > 1).sum()))


_PROBLEM: Problem | None = None


def run_config(config_id: str, lambdas, cfg: AnnealConfig) -> dict:
    """Worker entry point. Never raises: errors are logged and reported."""
    out = _PROBLEM.root / config_id
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    try:
        P = _PROBLEM
        H = P.hamiltonian(lambdas)
        res = anneal(H, P.M0, cfg, out, log=log)
        final = dict(config_id=config_id, lambdas=dict(zip(TERMS, map(float, lambdas))), anneal=asdict(cfg),
                     T_start=res["T_start"], T_end=res["T_end"], steps=res["steps"], calls=res["calls"],
                     runtime_s=time.time() - t0)
        for tag, key in (("best_E", "best"), ("best_D", "bestD")):
            rec = res[key]
            M = res[f"{key}_M"]
            final[tag] = dict(E=rec["E"], D=rec["D"], security_pct=rec["security_pct"],
                              C=float(rec["raw"][5]) if len(rec["raw"]) > 5 else 0.0,
                              raw=dict(zip(TERMS, rec["raw"].tolist())), z=dict(zip(TERMS, rec["z"].tolist())),
                              groups=group_stats(M, P.sigma, P.mec), sizes=M.sum(0).tolist(),
                              covered=int((M.sum(1) > 0).sum()), overlap=int((M.sum(1) > 1).sum()))
        np.savez(out / "final.npz", best_E_M=res["best_M"], best_D_M=res["bestD_M"], last_M=res["M"])
        (out / "summary.json").write_text(json.dumps(final, indent=2, default=float))
        try:
            import plots
            plots.config_plots(out, title=f"{config_id}  lambda{tuple(TERMS[:len(lambdas)])}={tuple(lambdas)}")
        except Exception:  # plots must not lose a finished run
            log("plotting failed:\n" + traceback.format_exc())
        log(f"[{config_id}] done in {time.time() - t0:.1f}s")
        return dict(config_id=config_id, ok=True, runtime_s=time.time() - t0)
    except Exception:
        tb = traceback.format_exc()
        (out / "error.log").write_text(tb)
        log(f"[{config_id}] FAILED:\n{tb}")
        return dict(config_id=config_id, ok=False, runtime_s=time.time() - t0, error=tb.splitlines()[-1])
    finally:
        logf.close()


def _run_star(a):
    return run_config(*a)


def git_commit(paths: list[Path], message: str) -> None:
    rel = [str(p.relative_to(REPO)) for p in paths if p.exists()]
    if not rel:
        return
    subprocess.run(["git", "add", "--", *rel], cwd=REPO, check=False)
    msg = (message + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_01BaMkaLKS5dfPs9eFSt4qPp")
    r = subprocess.run(["git", "commit", "-q", "-m", msg, "--", *rel], cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0 and "nothing to commit" not in (r.stdout + r.stderr):
        print(f"git commit failed: {r.stderr.strip()}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="26")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--alpha", type=float, default=AnnealConfig.alpha)
    ap.add_argument("--sweeps", type=int, default=AnnealConfig.sweeps_per_temp)
    ap.add_argument("--no-commit", action="store_true")
    ap.add_argument("--version", default="v2", choices=["v1", "v2"])
    args = ap.parse_args(argv)

    global _PROBLEM
    t = time.time()
    _PROBLEM = P = Problem(args.scope, args.version)
    print(f"setup {time.time() - t:.1f}s; anchors={P.anchor_meta['branches']} "
          f"init sizes={P.M0.sum(0).tolist()}", flush=True)
    if not args.no_commit:
        git_commit([P.setup], f"Sweep setup ({args.version}, {args.scope}-county): anchors and normaliser")

    cfgs = sweep_configs(args.version)
    if args.only:
        keep = set(args.only.split(","))
        cfgs = [c for c in cfgs if c[0] in keep]
    todo = [(cid, lam) for cid, lam in cfgs if not (P.root / cid / "summary.json").exists()]
    print(f"{len(cfgs)} configs, {len(todo)} to run", flush=True)
    cfg = AnnealConfig(alpha=args.alpha, sweeps_per_temp=args.sweeps, min_group_size=P.min_size)

    t = time.time()
    ctx = mp.get_context("fork")  # workers inherit the built Problem
    status = []
    with ctx.Pool(args.workers) as pool:
        for r in pool.imap_unordered(_run_star, [(cid, lam, cfg) for cid, lam in todo]):
            status.append(r)
            cid = r["config_id"]
            print(f"== {cid}: {'ok' if r['ok'] else 'FAILED ' + r.get('error', '')} "
                  f"({r.get('runtime_s', 0):.0f}s)", flush=True)
            if not args.no_commit:
                git_commit([P.root / cid], f"Sweep {args.version} {args.scope}-county: {cid} "
                                           f"({'complete' if r['ok'] else 'failed'})")
    print(f"sweep wall time {time.time() - t:.0f}s", flush=True)
    (P.root / f"sweep_status_{args.scope}.json").write_text(json.dumps(status, indent=2))
    return status


if __name__ == "__main__":
    main()
