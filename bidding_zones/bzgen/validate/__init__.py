"""Validation stage: does splitting DE into zones reduce dispatch down?

zonemap     bus -> zone (DE map under test, real foreign zones), map refits
market      stage A: zonal clearing, ATC transport model
redispatch  stage B: nodal redispatch with net positions fixed
metrics     dispatch down, attribution, cost gap, load pockets
inference   paired week-block bootstrap, bands
run         orchestration, caching, resume, report
"""
