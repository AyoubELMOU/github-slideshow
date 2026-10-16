PY ?= python3

.PHONY: test verify all report-fast

test:            ## fast regression tests (FreeFEM + analytical)
	$(PY) -m pytest -q tests

verify:          ## verification + validation, no PINN / mesh / sensitivity studies (~10 min)
	$(PY) scripts/run_all.py --skip-pinn --skip-slow

all:             ## everything incl. PINN forward/inverse training (~35 min on CPU)
	$(PY) scripts/run_all.py
