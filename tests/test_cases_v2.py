"""Our case builder, with SNSP and conventional nodes switched off, equals upstream's."""
import numpy as np
import pytest

import upstream as up
from cases import build_cases_v2


@pytest.mark.parametrize("seed,ts", [(42, 1.0), (11, 0.95), (2003, 0.9)])
def test_v2_reproduces_upstream(seed, ts):
    g = up.load_grid("26")
    a = up.build_cases(g, seed=seed, thermal_scale=ts)
    b = build_cases_v2(g, seed=seed, thermal_scale=ts, snsp_limit=None, include_conventional=False,
                       slack="balance")
    for f in ("renewable_potential_mw", "pre_network_dispatch_mw", "pre_network_dispatch_down_mw",
              "case_weights", "state_limits_mw", "state_monitor", "state_outage"):
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f), err_msg=f)
    for f in ("branch_flows_mw", "state_base_flows_mw", "node_state_sensitivity"):
        np.testing.assert_allclose(getattr(a, f), getattr(b, f), rtol=0, atol=1e-8, err_msg=f)


def test_snsp_cap_and_conventional_nodes():
    g = up.load_grid("26")
    c = build_cases_v2(g, seed=42, thermal_scale=1.0)
    assert np.all(c.snsp <= 0.75 + 1e-9)
    n_ren = len(g.template)
    assert (c.node_kind[:n_ren] == 0).all() and (c.node_kind[n_ren:] == 1).all()
    assert np.all(c.pre_network_dispatch_mw >= 0)
    assert c.node_state_sensitivity.shape[0] == len(c.node_kind)
