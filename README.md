# grace

Where a trained classifier fails, estimated without the labels of the rows in question: each class of
train is clustered into regions, every val and test row is routed to the nearest region without its
label, each region is described by 45 data-complexity descriptors, and a Random Forest learns each
region's test failure rate from them, next to two confidence baselines (MCP and ATC).

```
make generate                                                    # resources/raw_data/synthetic/synthetic_test.csv
make run DATA=synthetic_test MODEL=random_forest                 # the six stages below, in order
python main.py --stage split --dataset synthetic_test
python main.py --stage regions --dataset synthetic_test          # after split
python main.py --stage complexity --dataset synthetic_test       # after regions
python main.py --stage train --dataset synthetic_test --model random_forest      # after split
python main.py --stage failures --dataset synthetic_test --model random_forest   # after regions, train
python main.py --stage regress --dataset synthetic_test --model random_forest    # after complexity, failures
make dashboard                                                   # http://127.0.0.1:8000
```

Python 3.12, `pip install -r requirements.txt`, every command from the repo root (the paths in the configs are
relative to it). `--dataset`, `--model` and `--method` take a name of `configs/datasets/`, `configs/models/` and
`configs/method/` or the path of a yaml; `--method` defaults to `default`, and only train, failures and regress take
a `--model`. `make help` lists the targets; `make test` runs the suite. `resources/` is not tracked: a fresh clone has
no data, and the synthetic generator is the only dataset that works out of the box.

Stages (each writes its files into the dataset's `output_dir` and its rows into the DB):
- `split`: reads the configured columns of the CSV, drops rows with a missing or infinite number and the classes
  under `min_class_count`, splits stratified by class (40 / 15 / 45 % train / val / test in every shipped yaml) and
  prepares every split with a preprocessor fitted on train (signed log1p and robust scaling of the numbers, top-N
  plus hashed codes for the categories). Writes `{train,val,test}.parquet`, `classes.json`, `preprocessor.joblib`.
- `regions`: builds the space every distance is measured in (the prepared numbers plus a one-hot block of the
  `space.top_k` most frequent codes of each categorical column) and the k-NN graph of a sample of train, then
  clusters each class of train on its own. The granularity is the candidate of least loss: the kDN hardness spread
  a region hides plus the sampling noise of the failure rate its test rows will give. Small clusters and noise
  merge into the nearest region of their class. Val and test rows go to the nearest region centre of any class.
  Writes `regions.parquet`, `assignments.parquet`, `space.joblib`, `graph.npz`. Reads no val or test label.
- `complexity`: 45 descriptors (feature overlap F1-F4, neighbourhood N1-N4, network density and hubs,
  dimensionality T2-T4, geometry, linearity L1-L3; `src/complexity/`) of every region and every class, on its train
  rows against the `complexity.top_k_clusters` nearest populations of other classes. Writes `complexity.parquet`.
  Reads train only.
- `train`: fits the model with the fixed params of its yaml on train (undersampled to the smallest class with
  `balance: undersample`) and predicts val and test. Writes `models/<model>.joblib`,
  `pred_<model>_{val,test}.parquet`. Reads the val and test labels only to score the predictions.
- `failures`: counts the test errors of each region (its failure rate) and the two baselines on the same rows: MCP,
  the mean of 1 − confidence, and ATC, the share of rows under a confidence threshold fitted on val. A region with
  fewer than `regressor.min_eval_support` test rows is not *used*: no predictor is scored on it. Writes
  `failures_<model>.parquet`. Reads the test labels (the rates) and the val labels (the ATC threshold).
- `regress`: a Random Forest from each used region's descriptors and its class's (`region_<m>` and `class_<m>` for
  the `regressor.measures`) to its test failure rate, under nested cross-validation (a randomized search inside each
  outer fold), so every region's estimate comes from a forest that never saw it; TreeSHAP of that same forest
  explains it. A final refit on every region is kept. The estimates are scored (Spearman, R², MAE, MSE) against the
  observed rates next to MCP and ATC. Writes `regress_<model>_estimates.parquet`,
  `regress_<model>_contributions.parquet`, `regress_<model>.joblib`. Reads the test labels through the rates of
  `failures`.

No classifier trains on val. The test labels fit nothing but the regressor, which learns each region's rate from the
other regions' rates and is scored out of fold.

Measured on synthetic_test (69,500 rows, 9 classes after the rare one is dropped) with random_forest: about a
minute and a half for `make run`, test accuracy 0.85, 107 regions, all used; Spearman with the observed rates
RF 0.81, MCP 0.83, ATC 0.89.

Configs:
- `configs/datasets/`: one yaml per dataset: `name`, `path`, `label_col`, `num_cols`, `cat_cols`, `split`,
  `min_class_count`, `top_n`, `hash_buckets`, `seed`, `output_dir` and `db_path`. Four network datasets
  (`nb15_v2`, `cic_2018_v2`, `ton_iot_v2`, `bot_iot_v2`), five general benchmarks (`bank_marketing`, `covertype`,
  `letter_recognition`, `statlog_landsat_satellite`, `thyroid_disease`) and `synthetic_test`; their CSVs are not in
  the repo.
- `configs/models/`: ten classifiers, `name` (a key of `MODELS` in `src/models.py`) and `params`.
- `configs/method/default.yaml`: the method, shared by every dataset and model: `distance` (`euclidean` or
  `cosine`), `balance`, `space`, `graph`, `clustering` (`name`: `kmeans`, `birch`, `spectral` or `hdbscan`),
  `complexity` and `regressor`, each key commented. Another method is a copy of it passed as `--method <path>`.

`src/config.py` loads the three into frozen dataclasses and validates them before any stage reads data.

Outputs: the files of a dataset in its `output_dir` (`resources/processed/<dataset>/`), every dataset's runs in one
SQLite file, `db_path` (`resources/processed/grace.db`). Table `runs` holds one row per stage run (stage, dataset,
model, the whole config as JSON, status `running`, `ok` or `failed`, start and end); every other table carries the
`run_id` of the run that wrote it and is created by its first insert: `split_summary`, `dropped_classes`,
`split_class_counts`, `features` (split); `regions`, `class_clusterings`, `granularity_candidates`, `region_stats`
(regions); `complexity`; `model_metrics`, `confusion`, `train_stats` (train); `failures`, `baseline_scores`,
`failure_stats` (failures); `estimates`, `contributions`, `regressor_folds`, `importances`, `regress_scores`
(regress). There are no migrations: after a change of a table, delete the DB.

Dashboard: `make dashboard` (`PORT=` to change it) or `python -m uvicorn dashboard.app:app` serves a read-only page
on 127.0.0.1, one tab per stage, for the datasets of `configs/datasets/` and the models of `configs/models/`. Each
tab shows the latest `ok` run of its stage on that dataset (and model), and says which stage to run when there is
none. In regress, a click on a region shows its TreeSHAP contributions and its descriptors against its class and
the dataset. The URL hash keeps tab, dataset and model.

Layout:
- `main.py`: the CLI, one stage per call inside a logged run
- `stages/`: one file per stage, `run(cfg, conn, run_id)`; `stages/paths.py` names every file a stage writes
- `src/`: config and DB logging, then the method on arrays and frames: data, prepare, space, graph, clustering,
  complexity/, models, baselines, regressor, evaluation
- `dashboard/`: FastAPI `app.py`, read-only `queries.py`, `static/` (one page, hand-made SVG charts)
- `generate_synthetic.py`: the synthetic dataset, a ladder of sub-groups per class that overlap their rivals by
  known margins (the `true_subgroup` column records it; no stage reads it)
- `tests/`: the suite; the session fixture runs every stage once on a 12,000-row synthetic CSV

To add:
- a dataset: put the CSV under `resources/raw_data/`, copy a yaml of `configs/datasets/`, set `name`, `path`, the
  columns and `output_dir`, and pass its name to `--dataset`.
- a model: an entry of `MODELS` in `src/models.py`, a factory and its categorical strategy (`onehot`, `drop_cat`,
  `native_sklearn`, `native_xgb`, or None for the codes as numbers), and `configs/models/<name>.yaml`.
- a clustering: a function `fit_<name>(X, *, <params>, max_fit_samples, random_state)` returning a label per row
  (-1 for noise) in `ALGORITHMS` in `src/clustering.py`. Its keyword-only parameters become the allowed
  `clustering.params`; a list value is swept and the candidate of least loss kept. If it takes `n_clusters`, the
  stage sweeps it per class and the yaml must not set it.
- a descriptor: one more key returned by a family function of `src/complexity/` and listed in `MEASURES` in
  `src/complexity/measure.py`; the regressor sees it when `regressor.measures` names it, or is null.

Known limits: every rerun of a stage overwrites its files, so the files hold only the latest run of each stage (and
model), and nothing records which run wrote a stage's inputs: after changing the method or rerunning a stage,
rerun everything after it (`make run`). Two method yamls on one dataset share its `output_dir`. A failed rerun can
leave files that disagree with the latest `ok` run the dashboard shows. A run killed by anything but Ctrl-C stays
`running`. The classifiers are not tuned. The k-NN searches and the regressor's search use every core.
