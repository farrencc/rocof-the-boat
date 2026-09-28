#!/bin/sh
# Everything in this folder, in order.  ~25 minutes end to end.
PY=../repo/grid_TF_Wind/participant-kit/.venv/Scripts/python.exe

$PY validate.py              # 24 checks -- run before believing anything
$PY base.py --freeze         # base.npz / base.json, hash-checked against ../snsp

$PY sweep.py                 # the solve + the scores        (~7 min)
$PY robustness.py            # band / lag / window / grid     (reprocess, seconds)
$PY nonlinear.py             # stage 5: re-solve, then time domain  (~3 min)
$PY ensemble_edge.py         # the winner, through all 240 states     (~4 min)
$PY figures.py               # figures/*.png

# the axes not yet run -- each is one flag, and only --snsp needs a re-solve
# $PY sweep.py --kappa 8.0    --tag k8
# $PY sweep.py --events identity --tag bI
# $PY sweep.py --snsp 0.40    --tag s40
#
# re-scoring an existing solve costs nothing:
# $PY sweep.py --analyse-only --band 0.01 5 --tag wide
