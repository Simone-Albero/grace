# grace. The pipeline is `python -m stages <target…> [key=value…] [--sweep k=v1,v2…] [--force]`;
# these targets are shortcuts.
.DEFAULT_GOAL := help
PYTHON ?= $(if $(wildcard venv/bin/python),venv/bin/python,python)

VERIFY := data=synthetic_test root=resources/verify
MODE ?= quick
ROOT ?= resources
PORT ?= 8000
CLASSIFIERS := logistic_regression,decision_tree,naive_bayes,hist_gradient_boosting,lda,linear_svc,xgboost,knn,random_forest,mlp
SWEEP_quick := --sweep classifier=random_forest,mlp
SWEEP_full := --sweep classifier=$(CLASSIFIERS) --sweep clustering=kmeans,hdbscan,birch,spectral --sweep distance=euclidean,cosine
SWEEP = $(or $(SWEEP_$(MODE)),$(error MODE must be quick or full, not '$(MODE)'))

.PHONY: help test format smoke generate dashboard

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
	$(PYTHON) -m stages render partitions $(VERIFY) $(SWEEP)
	$(PYTHON) -m stages render partitions $(VERIFY) $(SWEEP) 2> resources/verify/cached.log || { cat resources/verify/cached.log; exit 1; }
	! grep "running execution" resources/verify/cached.log

## generate: the synthetic dataset (ROWS=<n> for a smaller one)
generate:
	$(PYTHON) -m pipeline.generate_synthetic $(if $(ROWS),--rows $(ROWS),)

## dashboard: dashboard (results, presets, jobs) of ROOT/grace.db (default resources), created if missing, on 127.0.0.1:PORT (default 8000); remote host: ssh -L PORT:127.0.0.1:PORT <host>
dashboard:
	$(PYTHON) -m dashboard --root $(ROOT) --port $(PORT)
