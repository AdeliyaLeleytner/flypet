.PHONY: check test data smoke
PYTHON ?= python

check:
	ruff check .
	ruff format --check .
	$(PYTHON) -m compileall -q flypet scripts tests deploy release

test:
	$(PYTHON) -m unittest discover -s tests -v
	node --test tests/*.test.cjs

data:
	$(PYTHON) scripts/fetch_data.py brain

smoke:
	$(PYTHON) scripts/flytalk/smoke_test.py --workers 2
