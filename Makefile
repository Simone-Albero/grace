# grace. The pipeline is `python main.py --stage <stage> --dataset <name>`;
# these targets are shortcuts.
.DEFAULT_GOAL := help
PYTHON ?= $(if $(wildcard venv/bin/python),venv/bin/python,python)
METHOD ?= default
PORT ?= 8000

.PHONY: help format run dashboard

## help: this list
help:
	@grep -hE '^## ' $(MAKEFILE_LIST) | sed 's/^## //'

## format: black and isort over the repo
format:
	$(PYTHON) -m black .
	$(PYTHON) -m isort --profile black .

## run: every stage in order on DATA (a name of configs/datasets/ or a yaml) with METHOD (default default)
run:
	$(if $(DATA),,$(error DATA=<name|yaml> is required))
	$(PYTHON) main.py --stage clean --dataset $(DATA) --method $(METHOD)
	$(PYTHON) main.py --stage prepare --dataset $(DATA) --method $(METHOD)
	$(PYTHON) main.py --stage cells --dataset $(DATA) --method $(METHOD)
	$(PYTHON) main.py --stage quality_report --dataset $(DATA) --method $(METHOD)

## dashboard: the read-only dashboard on 127.0.0.1:PORT (default 8000)
dashboard:
	$(PYTHON) -m uvicorn dashboard.app:app --port $(PORT)
