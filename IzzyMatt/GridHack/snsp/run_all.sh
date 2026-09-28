#!/bin/sh
# Everything in this folder, in order.  ~75 minutes end to end.
PY=../repo/grid_TF_Wind/participant-kit/.venv/Scripts/python.exe

$PY grid.py --freeze --steady --verify   # the toy grid, frozen and checked
$PY perturbations.py --freeze            # all three disturbance sets, frozen
$PY configs.py --freeze                  # the ensemble of operating states
$PY validate.py                          # 40 checks -- run before believing anything

# the headline: every event scaled to its own source, plus the two control knobs
$PY run_ensemble.py --tag src                                  # ~6 min
$PY run_ensemble.py --ffr 20 --tag src_ffr                     # converter FFR
$PY run_ensemble.py --agc     --tag src_agc                    # secondary control

# finding 11: hazard model x minimum-units floor
$PY run_ensemble.py --events fixed  --tag ""
$PY run_ensemble.py --events scaled --tag scaled
$PY run_ensemble.py --events fixed  --muon 4 --tag muon
$PY run_ensemble.py --events scaled --muon 4 --tag scaled_muon

# finding 12's sensitivity: a trip that takes the whole station
# (edit perturbations.LARGEST_UNIT to 0 and re-freeze, or use the saved
#  ensemble_srcuncapped*.csv already in the folder)

$PY figures.py                           # figures/*.png       (sourced family)
$PY figures.py --family fixed            # figures/fixed_hazard/*.png
