"""Step 4: Metropolis annealer over an N x K binary membership matrix.

Adapted from upstream ``annealer+blackbox_26+32.py``: same temperature
calibration (``estimate_temperature_range_blackbox``), same Metropolis rule and
geometric cooling. Differences:

* state is the N x K bool matrix M (groups may overlap; nodes may be uncovered);
* move set: with probability ``p_flip`` flip one M[i, k] (add/remove node i to/from
  group k); otherwise move a node from group k1 to group k2 (it leaves k1 and
  joins k2). Group sizes are NOT conserved. Any move that would empty a group is
  rejected and redrawn (it never reaches the black box);
* best-E and best-D groupings are tracked separately, because once E != D the
  lowest-energy grouping is not the lowest-dispatch-down grouping;
* checkpoint after EVERY temperature step (history CSV row + checkpoint.npz
  with current/best matrices, energies, term breakdown and RNG state); a run
  resumes from ``checkpoint.npz`` if it exists.

Temperature calibration: T_start = -median|dE| / ln(0.5), T_end = -p5|dE| / ln(1e-3)
from a random walk of probe moves through finite states. Because T is set from
percentiles of dE of the COMPOSITE energy, an unnormalised term (e.g. T4 = sum
n_k^2, O(10^3)) would dominate dE and silently set the whole schedule, making the
other lambdas irrelevant. This is why hamiltonian.py z-scores every term first.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

try:
    from .hamiltonian import TERMS
except ImportError:  # pragma: no cover
    from hamiltonian import TERMS


@dataclass
class AnnealConfig:
    alpha: float = 0.98
    sweeps_per_temp: int = 25
    n_temperature_samples: int = 40
    hot_accept_prob: float = 0.5
    cold_accept_prob: float = 1e-3
    p_flip: float = 0.5
    min_group_size: int = 1             # hard floor: moves that shrink a group below it are redrawn
    seed: int = 0
    max_temperature_steps: int = 2000   # safety cap on the geometric schedule


def propose(M: np.ndarray, rng: np.random.Generator, p_flip: float, min_size: int = 1) -> np.ndarray:
    N, K = M.shape
    floor = max(1, int(min_size))
    sizes = M.sum(axis=0)
    while True:
        if K == 1 or rng.random() < p_flip:
            i, k = int(rng.integers(N)), int(rng.integers(K))
            if M[i, k] and sizes[k] <= floor:
                continue  # would empty group k / break the minimum size
            C = M.copy()
            C[i, k] = not C[i, k]
            return C
        k1, k2 = (int(x) for x in rng.choice(K, size=2, replace=False))
        if sizes[k1] <= floor:
            continue  # would empty k1 / break the minimum size
        cand = np.flatnonzero(M[:, k1] & ~M[:, k2])
        if len(cand) == 0:
            continue
        i = int(rng.choice(cand))
        C = M.copy()
        C[i, k1] = False
        C[i, k2] = True
        return C


def estimate_temperature_range(H, M, E, rng, cfg: AnnealConfig):
    """Upstream estimate_temperature_range_blackbox for the N x K state.

    Random walk of probe moves; +inf (security-guard) candidates are rejected and
    do not contribute a dE. Returns (T_start, T_end, n_calls).
    """
    dEs = []
    for _ in range(cfg.n_temperature_samples):
        C = propose(M, rng, cfg.p_flip, cfg.min_group_size)
        rec = H(C)
        if np.isfinite(rec["E"]):
            dE = abs(rec["E"] - E)
            if dE > 0 and np.isfinite(dE):
                dEs.append(dE)
            M, E = C, rec["E"]
    if not dEs:
        return 1e-3, 1e-6, cfg.n_temperature_samples
    dEs = np.array(dEs)
    T_start = -np.percentile(dEs, 50) / np.log(cfg.hot_accept_prob)
    T_end = -np.percentile(dEs, 5) / np.log(cfg.cold_accept_prob)
    return float(T_start), float(T_end), cfg.n_temperature_samples


HISTORY_FIELDS = (
    ["temperature_step", "temperature", "current_energy", "best_energy", "current_D", "best_E_D", "best_D",
     "accepted", "acceptance_rate", "new_bests", "infeasible_proposals", "blackbox_calls", "security_pct"]
    + [f"raw_{t}" for t in TERMS] + [f"z_{t}" for t in TERMS] + ["sizes", "n_covered", "n_overlap"]
)


def _rng_state_json(rng):
    return json.dumps(rng.bit_generator.state)


def _atomic_savez(path: Path, **arrays):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _pack(rec):
    return np.concatenate([[rec["E"], rec["D"], rec["security_pct"], float(rec["feasible"])], rec["raw"], rec["z"]])


def _unpack(v):
    v = np.asarray(v, float)
    n = (len(v) - 4) // 2
    return dict(E=float(v[0]), D=float(v[1]), security_pct=float(v[2]), feasible=bool(v[3]),
                raw=v[4:4 + n], z=v[4 + n:4 + 2 * n])


def anneal(H, M0: np.ndarray, cfg: AnnealConfig, out_dir: Path, log=print) -> dict:
    """Run (or resume) one annealing configuration, checkpointing in ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "checkpoint.npz"
    hist_path = out_dir / "history.csv"
    rng = np.random.default_rng(cfg.seed)

    if ckpt.exists():
        z = np.load(ckpt, allow_pickle=False)
        if not np.allclose(z["lambdas"], H.lam) or json.loads(str(z["config"])) != asdict(cfg):
            raise RuntimeError(f"{ckpt} was written for a different config; delete it to restart")
        rng.bit_generator.state = json.loads(str(z["rng_state"]))
        M, best_M, bestD_M = z["M"].astype(bool), z["best_M"].astype(bool), z["bestD_M"].astype(bool)
        cur, best, bestD = _unpack(z["cur"]), _unpack(z["best"]), _unpack(z["bestD"])
        E, best_E = cur["E"], best["E"]
        T, T_start, T_end = float(z["T"]), float(z["T_start"]), float(z["T_end"])
        step, calls, done = int(z["step"]), int(z["calls"]), bool(z["done"])
        # Drop history rows written after the checkpoint (crash between the two writes).
        if hist_path.exists():
            with open(hist_path) as f:
                rows = [r for r in csv.DictReader(f) if int(r["temperature_step"]) <= step]
            with open(hist_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
                w.writeheader()
                w.writerows(rows)
        log(f"[{out_dir.name}] resumed at step {step}, T={T:.4g}, calls={calls}, best E={best_E:.4f}")
    else:
        M = M0.astype(bool).copy()
        cur = H(M)
        E = cur["E"]
        if not np.isfinite(E):
            raise RuntimeError("initial grouping is infeasible under the security guard")
        best_M, best, best_E = M.copy(), cur, E
        bestD_M, bestD = M.copy(), cur
        calls = 1
        T_start, T_end, n = estimate_temperature_range(H, M.copy(), E, rng, cfg)
        calls += n
        T, step, done = T_start, 0, False
        with open(hist_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=HISTORY_FIELDS).writeheader()
        log(f"[{out_dir.name}] E0={E:.4f} D0={cur['D']:.4f}  T_start={T_start:.4g} T_end={T_end:.4g} alpha={cfg.alpha}")

    def save_ckpt():
        _atomic_savez(
            ckpt, M=M, best_M=best_M, bestD_M=bestD_M, cur=_pack(cur), best=_pack(best), bestD=_pack(bestD),
            T=T, T_start=T_start, T_end=T_end, step=step, calls=calls, done=done,
            rng_state=np.array(_rng_state_json(rng)), lambdas=H.lam, config=np.array(json.dumps(asdict(cfg))),
            norm_offset=H.norm.offset, norm_scale=H.norm.scale)

    if step == 0 and not done:
        save_ckpt()

    while not done and T > T_end and step < cfg.max_temperature_steps:
        step += 1
        accepted = improved = infeasible = 0
        for _ in range(cfg.sweeps_per_temp):
            C = propose(M, rng, cfg.p_flip, cfg.min_group_size)
            rec = H(C)
            calls += 1
            if not rec["feasible"]:
                infeasible += 1
                continue  # E = +inf: exp(-inf/T) = 0, never accepted
            if rec["D"] < bestD["D"]:
                bestD_M, bestD = C.copy(), rec
            dE = rec["E"] - E
            if dE < 0 or rng.random() < np.exp(-dE / T):
                M, E, cur = C, rec["E"], rec
                accepted += 1
                if E < best_E:
                    best_M, best_E, best = M.copy(), E, rec
                    improved += 1
        sizes = M.sum(axis=0)
        row = dict(temperature_step=step, temperature=T, current_energy=E, best_energy=best_E,
                   current_D=cur["D"], best_E_D=best["D"], best_D=bestD["D"], accepted=accepted,
                   acceptance_rate=accepted / cfg.sweeps_per_temp, new_bests=improved,
                   infeasible_proposals=infeasible, blackbox_calls=calls, security_pct=cur["security_pct"],
                   sizes="|".join(map(str, sizes.tolist())), n_covered=int((M.sum(1) > 0).sum()),
                   n_overlap=int((M.sum(1) > 1).sum()))
        row.update({f"raw_{t}": v for t, v in zip(TERMS, cur["raw"])})
        row.update({f"z_{t}": v for t, v in zip(TERMS, cur["z"])})
        with open(hist_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=HISTORY_FIELDS).writerow(row)
        T *= cfg.alpha
        if not (T > T_end and step < cfg.max_temperature_steps):
            done = True
        save_ckpt()
        if step % 25 == 0 or done:
            log(f"[{out_dir.name}] step {step:4d} T={T:.4g} E={E:.4f} D={cur['D']:.4f} best E={best_E:.4f} "
                f"(D={best['D']:.4f}) best D={bestD['D']:.4f} acc={accepted}/{cfg.sweeps_per_temp} "
                f"infeas={infeasible} calls={calls} sizes={sizes.tolist()}")

    return dict(best_M=best_M, best=best, bestD_M=bestD_M, bestD=bestD, M=M, cur=cur,
                T_start=T_start, T_end=T_end, steps=step, calls=calls)
