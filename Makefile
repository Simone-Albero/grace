# grace. The pipeline is `python -m stages <target…> [key=value…] [--sweep k=v1,v2…] [--force]`;
# these targets are shortcuts.
.DEFAULT_GOAL := help
PYTHON ?= $(if $(wildcard venv/bin/python),venv/bin/python,python)

VERIFY := data=synthetic_test root=resources/verify
MODE ?= quick
ROOT ?= resources
PORT ?= 8000
CLASSIFIERS := logistic_regression,decision_tree,naive_bayes,hist_gradient_boosting,lda,linear_svc,xgboost,knn,random_forest,mlp
comma := ,
space := $(subst ,, )
DATASETS ?= $(subst $(space),$(comma),$(sort $(basename $(notdir $(wildcard configs/data/*.yaml)))))
SWEEP_quick := --sweep classifier=random_forest,mlp
SWEEP_full := --sweep classifier=$(CLASSIFIERS)
SWEEP = $(or $(SWEEP_$(MODE)),$(error MODE must be quick or full, not '$(MODE)'))

.PHONY: help test format smoke run sweep generate dashboard

## help: this list
help:
	@grep -hE '^## ' $(MAKEFILE_LIST) | sed 's/^## //'

## test: the pytest suite, about 4-5 minutes
test:
	$(PYTHON) -m pytest -q tests

## format: black and isort over the repo, as the style requires
format:
	$(PYTHON) -m black .
	$(PYTHON) -m isort --profile black .

## smoke: every stage on synthetic_test into resources/verify (own DB), MODE=quick|full; the second pass must be fully cached
smoke:
	$(PYTHON) -m stages render complexity $(VERIFY) $(SWEEP)
	$(PYTHON) -m stages render complexity $(VERIFY) $(SWEEP) 2> resources/verify/cached.log || { cat resources/verify/cached.log; exit 1; }
	! grep "running execution" resources/verify/cached.log

## run: the pipeline up to render, without complexity, for DATA=<dataset> CLASSIFIER=<classifier> into ROOT (default resources)
run:
	$(if $(DATA),,$(error DATA=<dataset> is required, e.g. make run DATA=nb15_v2 CLASSIFIER=mlp))
	$(if $(CLASSIFIER),,$(error CLASSIFIER=<classifier> is required, e.g. make run DATA=nb15_v2 CLASSIFIER=mlp))
	$(PYTHON) -m stages render data=$(DATA) classifier=$(CLASSIFIER) root=$(ROOT)

## sweep: run for every dataset of configs/data and every classifier into ROOT, past the failures, which it lists at the end (DATASETS=a,b and CLASSIFIERS=c,d narrow it)
sweep:
	@failed=; \
	for data in $(subst $(comma),$(space),$(DATASETS)); do \
	  for classifier in $(subst $(comma),$(space),$(CLASSIFIERS)); do \
	    $(PYTHON) -m stages render data=$$data classifier=$$classifier root=$(ROOT) \
	      || failed="$$failed $$data/$$classifier"; \
	  done; \
	done; \
	test -z "$$failed" || { echo "failed:$$failed"; exit 1; }

## generate: the synthetic dataset (ROWS=<n> for a smaller one)
generate:
	$(PYTHON) -m pipeline.generate_synthetic $(if $(ROWS),--rows $(ROWS),)

## dashboard: dashboard (the results of every stage, by experiment) of ROOT/grace.db (default resources), created if missing, on 127.0.0.1:PORT (default 8000); remote host: ssh -L PORT:127.0.0.1:PORT <host>
dashboard:
	$(PYTHON) -m dashboard --root $(ROOT) --port $(PORT)
