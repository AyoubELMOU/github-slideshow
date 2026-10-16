PY ?= python3

.PHONY: test verify all report-fast

test:            ## fast regression tests (FreeFEM + analytical)
	$(PY) -m pytest -q tests

verify:          ## verification + validation, no PINN (~2 min)
	$(PY) scripts/run_all.py --skip-pinn

all:             ## everything incl. PINN forward/inverse training (~10 min on CPU)
	$(PY) scripts/run_all.py
