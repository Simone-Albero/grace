# grace. The pipeline is `python main.py --stage <stage> --dataset <name>`;
# these targets are shortcuts.
.DEFAULT_GOAL := help
PYTHON ?= $(if $(wildcard venv/bin/python),venv/bin/python,python)
METHOD ?= default
PORT ?= 8000
AWAKE ?= $(if $(shell command -v caffeinate),caffeinate -i,)  # keeps a Mac awake while a long target runs: its
#   seconds are of the run, not of a sleep

.PHONY: help format run dashboard paper

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

## paper: every experiment of the paper on the pocs, in order (hours): the pocs, every grace run, DataBadSmells,
##        the truth of the corrected release, then every figure; each step resumes where it stopped, and runs
##        under caffeinate -i where there is one (AWAKE=), so that a sleep does not enter the seconds measured
paper:
	$(AWAKE) $(PYTHON) -m experiments.poc.generate experiments/poc/cic_2018.yaml
	$(AWAKE) $(PYTHON) -m experiments.runs.generate experiments/runs/cic_2018.yaml
	$(AWAKE) $(PYTHON) -m experiments.databadsmells.generate experiments/databadsmells/cic_2018.yaml
	$(AWAKE) $(PYTHON) -m experiments.rivals.generate experiments/rivals/cic_2018.yaml
	$(AWAKE) $(PYTHON) -m experiments.relabel.generate experiments/relabel/cic_2018.yaml
	for e in databadsmells rivals regions relabel sensitivity sample; do \
		$(AWAKE) $(PYTHON) -m experiments.$$e.run experiments/$$e/cic_2018.yaml || exit 1; \
	done

## dashboard: the read-only dashboard on 127.0.0.1:PORT (default 8000)
dashboard:
	$(PYTHON) -m uvicorn dashboard.app:app --port $(PORT)
