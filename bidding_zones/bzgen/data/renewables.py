"""Weather -> capacity factors.

Wind (onshore and offshore alike): a generic 3 MW-class power curve, cubic
between cut-in and rated speed, flat to cut-out, applied to ERA5 100 m wind
speed, times ``(1 - losses)`` for wakes and availability.  No hub-height
correction (100 m is a typical modern hub height), no spatial smoothing of the
curve.  ERA5 is known to under-state onshore wind speed variability in complex
terrain; no bias correction is applied (PyPSA-Eur's atlite does not bias-
correct by default either).  The annual means are reported as a sanity check.

PV: ``CF = PR * GHI / 1000``, clipped to [0, 1].  Horizontal irradiance, no
transposition to a tilted plane and no temperature derate — both flagged as
simplifications (they roughly cancel for south-facing fixed tilt in central
Europe; INFERENCE).  Open-Meteo's ``shortwave_radiation`` at timestamp t is
the mean over the preceding hour, so it is shifted back one hour to match
the OPSD convention (timestamp = start of the hour).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wind_cf(speed: pd.DataFrame, w: dict) -> pd.DataFrame:
    v = speed.to_numpy(dtype=float)
    ci, r, co = w["cut_in"], w["rated"], w["cut_out"]
    cf = np.where(v < ci, 0.0, np.where(v < r, (v**3 - ci**3) / (r**3 - ci**3),
                                         np.where(v < co, 1.0, 0.0)))
    return pd.DataFrame(cf * (1 - w["losses"]), index=speed.index, columns=speed.columns)


def pv_cf(ghi: pd.DataFrame, p: dict) -> pd.DataFrame:
    g = ghi.shift(-1).fillna(0.0)
    return (p["performance_ratio"] * g / 1000.0).clip(0.0, 1.0)
