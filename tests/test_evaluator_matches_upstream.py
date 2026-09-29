"""Gate for the whole project: our N x K evaluator, fed the membership matrix of
upstream's exclusive ``group`` column, must reproduce ``make_emulator``'s
``dispatch_down`` on the same node table, seed and scope to < 1e-6 pp."""
import numpy as np
import pytest

import upstream as up
from evaluator import Evaluator, labels_to_membership

CASES = [("26", 42, 1.0), ("26", 76, 1.0), ("26", 7, 0.95)]


@pytest.mark.parametrize("scope,seed,thermal_scale", CASES)
def test_exclusive_membership_matches_make_emulator(scope, seed, thermal_scale):
    dispatch_down, nodes = up.make_emulator(scope=scope, runs=10_000, seed=seed,
                                            thermal_scale=thermal_scale)
    ref = dispatch_down(nodes)

    grid = up.load_grid(scope)
    assert grid.template["node_id"].tolist() == nodes["node_id"].tolist()
    cases = up.build_cases(grid, seed=seed, thermal_scale=thermal_scale, runs=10_000)
    # Our own construction must be bit-identical to the emulator's frozen cases.
    for f in ("state_base_flows_mw", "node_state_sensitivity", "case_weights",
              "pre_network_dispatch_mw", "state_limits_mw"):
        np.testing.assert_array_equal(getattr(cases, f), getattr(dispatch_down.cases, f))

    M, _ = labels_to_membership(up.exclusive_group_ids(grid))
    res = Evaluator(cases)(M)
    assert abs(res.pct - dispatch_down.baseline_dispatch_down_pct) < 1e-6
    if np.isfinite(ref):
        assert abs(res.pct - ref) < 1e-6
    sec = 100.0 * np.sum(cases.case_weights * res.secure)
    assert abs(sec - dispatch_down.baseline_security_pass_pct) < 1e-9
    worst = 100.0 * np.max(res.worst_loading[cases.case_weights > 0])
    assert abs(worst - dispatch_down.baseline_worst_screened_loading_pct) < 1e-6


def test_relabelled_candidate_matches_make_emulator():
    """A non-baseline exclusive candidate (labels permuted + nodes moved) also matches,
    including the security-guard verdict."""
    dispatch_down, nodes = up.make_emulator(scope="26", runs=10_000, seed=42)
    grid = up.load_grid("26")
    cases = up.build_cases(grid, seed=42)
    ev = Evaluator(cases)
    base = ev(labels_to_membership(up.exclusive_group_ids(grid))[0])
    rng = np.random.default_rng(0)
    labels = up.exclusive_group_ids(grid).copy()
    for trial in range(8):
        i, j = rng.integers(0, len(labels), size=2)
        labels[i], labels[j] = labels[j], labels[i]
        cand = nodes.copy()
        cand["groups"] = [(int(g),) for g in labels]
        ref = dispatch_down(cand)
        res = ev(labels_to_membership(labels)[0])
        assert abs(res.pct - dispatch_down.last_raw_dispatch_down_pct) < 1e-6
        assert ev.new_failures(res, base.secure) == dispatch_down.last_new_security_failures
        assert np.isinf(ref) == (ev.new_failures(res, base.secure) > 0)
