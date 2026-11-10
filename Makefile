# grace. The pipeline is `python main.py --stage <stage> --dataset <name>`;
# these targets are shortcuts.
.DEFAULT_GOAL := help
PYTHON ?= $(if $(wildcard venv/bin/python),venv/bin/python,python)
DATA ?= synthetic_test
PORT ?= 8000

.PHONY: help test format generate run dashboard

## help: this list
help:
	@grep -hE '^## ' $(MAKEFILE_LIST) | sed 's/^## //'

## test: the pytest suite
test:
	$(PYTHON) -m pytest -q tests

## format: black and isort over the repo
format:
	$(PYTHON) -m black .
	$(PYTHON) -m isort --profile black .

## generate: the synthetic dataset (ROWS=<n> for a smaller one)
generate:
	$(PYTHON) generate_synthetic.py $(if $(ROWS),--rows $(ROWS),)

## run: every stage in order on DATA (default synthetic_test)
run:
	$(PYTHON) main.py --stage clean --dataset $(DATA)
	$(PYTHON) main.py --stage prepare --dataset $(DATA)
	$(PYTHON) main.py --stage regions --dataset $(DATA)
	$(PYTHON) main.py --stage quality_report --dataset $(DATA)

## dashboard: the read-only dashboard on 127.0.0.1:PORT (default 8000)
dashboard:
	$(PYTHON) -m uvicorn dashboard.app:app --port $(PORT)
