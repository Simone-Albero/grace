# grace

Where a trained classifier fails, estimated without the labels of the rows in question: each class of
train is clustered into regions, every val and test row is routed to the nearest region without its
label, each region is described by 45 data-complexity descriptors, and a Random Forest learns each
region's test failure rate from them, next to three baselines (MCP, ATC and a model of the classifier's errors)
and to the mean of its estimate with each of them.

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
- `split`: reads the configured columns of the CSV, drops the classes under `min_class_count` and the configured
  columns that hold a single value, and splits stratified by class (40 / 15 / 45 % train / val / test in every
  shipped yaml). No row is dropped for a missing or infinite value and nothing is scaled or encoded: the classifier
  and the region space each fit their own on train. Writes `{train,val,test}.parquet` (the kept columns as read, the
  class as its code) and `classes.json`.
- `regions`: builds the space every distance is measured in, fitted on train (the numbers as signed log1p with robust
  scaling, a missing or infinite one on the train median; each categorical column coded, its `top_n` most frequent
  values on their own and the rest hashed into `hash_buckets`, then one-hot over its `space.top_k` most frequent
  codes) and the k-NN graph of a sample of train, then clusters each class of train on its own. The granularity is
  the candidate of least loss: the kDN hardness spread a region hides plus the sampling noise of the failure rate its
  test rows will give. Small clusters and noise merge into the nearest region of their class. Val and test rows go to
  the nearest region centre of any class. Writes `regions.parquet`, `assignments.parquet`, `space.joblib`,
  `graph.npz`. Reads no val or test label.
- `complexity`: 45 descriptors (feature overlap F1-F4, neighbourhood N1-N4, network density and hubs,
  dimensionality T2-T4, geometry, linearity L1-L3; `src/complexity/`) of every region and every class, on its train
  rows against the `complexity.top_k_clusters` nearest populations of other classes. Writes `complexity.parquet`.
  Reads train only.
- `train`: fits the model with the fixed params of its yaml on at most `max_per_class` train rows per class (a seeded
  draw; all of train when null), behind a preprocessing fitted on the same rows (z-scored numbers, a one-hot of the
  `top_n` most frequent values of each categorical column plus one for the rest, and the train median for a missing
  number when the classifier cannot read it), and predicts val and test whole. Writes `models/<model>.joblib`,
  `pred_<model>_{val,test}.parquet`. Reads the val and test labels only to score the predictions.
- `failures`: counts the test errors of each region (its failure rate) and the three baselines on the same rows:
  MCP, the mean of 1 − confidence; ATC, the share of rows under a confidence threshold fitted on val; and meta, the
  mean P(error) of a HistGradientBoostingClassifier (`error_model`) fitted on val (a uniform sample of at most
  `max_samples` rows) to predict the classifier's errors from the row's columns, the classifier's probabilities,
  its confidence and its predicted class, as the post-hoc error predictors of the literature do. A region with fewer than `regressor.min_eval_support` test rows
  is not *used*: no predictor is scored on it. Writes `failures_<model>.parquet`. Reads the test labels (the rates)
  and the val labels (the ATC threshold and the errors meta learns).
- `regress`: a Random Forest from each used region's descriptors and its class's (`region_<m>` and `class_<m>` for
  the `regressor.measures`) to its test failure rate, under nested cross-validation (a randomized search inside each
  outer fold), so every region's estimate comes from a forest that never saw it; TreeSHAP of that same forest
  explains it. A final refit on every region is kept. The estimates are scored (Spearman, R², MAE, MSE) against the
  observed rates next to each baseline and to `rf+<baseline>`, the mean of the estimate and that baseline. Writes `regress_<model>_estimates.parquet`,
  `regress_<model>_contributions.parquet`, `regress_<model>.joblib`. Reads the test labels through the rates of
  `failures`.

The studied classifier never trains on val; only meta, the model of its errors, does. The test labels fit nothing but the regressor, which learns each region's rate from the
other regions' rates and is scored out of fold.

Measured on synthetic_test (69,500 rows, 9 classes after the rare one is dropped) with random_forest: about a
minute and a half for `make run`, test accuracy 0.89, 56 regions, all used; Spearman with the observed rates
RF 0.83, MCP 0.86, ATC 0.89.

Configs:
- `configs/datasets/`: one yaml per dataset: `name`, `path`, `label_col`, `num_cols`, `cat_cols`, `split`,
  `min_class_count`, `max_per_class`, `top_n`, `hash_buckets`, `seed`, `output_dir` and `db_path`. Four network
  datasets (`nb15_v2`, `cic_2018_v2`, `ton_iot_v2`, `bot_iot_v2`), five general benchmarks (`bank_marketing`,
  `covertype`, `letter_recognition`, `statlog_landsat_satellite`, `thyroid_disease`) and `synthetic_test`; their CSVs
  are not in the repo.
- `configs/models/`: ten classifiers, `name` (a key of `MODELS` in `src/models.py`) and `params`.
- `configs/method/default.yaml`: the method, shared by every dataset and model: `distance` (`euclidean` or
  `cosine`), `space`, `graph`, `clustering` (`name`: `kmeans`, `birch`, `spectral` or `hdbscan`),
  `complexity`, `regressor` and `error_model`, each key commented. Another method is a copy of it passed as `--method <path>`.

`src/config.py` loads the three into frozen dataclasses and validates them before any stage reads data.

Outputs: the files of a dataset in its `output_dir` (`resources/processed/<dataset>/`), every dataset's runs in one
SQLite file, `db_path` (`resources/processed/grace.db`). Table `runs` holds one row per stage run (stage, dataset,
model, the whole config as JSON, status `running`, `ok` or `failed`, start and end); every other table carries the
`run_id` of the run that wrote it and is created by its first insert: `split_summary`, `dropped_classes`,
`dropped_columns`, `split_class_counts`, `features` (split); `regions`, `class_clusterings`,
`granularity_candidates`, `region_stats` (regions); `complexity`; `model_metrics`, `confusion`, `train_stats`
(train); `failures`, `baseline_scores`, `failure_stats` (failures); `estimates`, `contributions`, `regressor_folds`,
`importances`, `regress_scores` (regress). There are no migrations: after a change of a table, delete the DB.

Dashboard: `make dashboard` (`PORT=` to change it) or `python -m uvicorn dashboard.app:app` serves a read-only page
on 127.0.0.1, one tab per stage, for the datasets of `configs/datasets/` and the models of `configs/models/`. Each
tab shows the latest `ok` run of its stage on that dataset (and model), and says which stage to run when there is
none. In regress, a click on a region shows its TreeSHAP contributions and its descriptors against its class and
the dataset. The URL hash keeps tab, dataset and model.

Layout:
- `main.py`: the CLI, one stage per call inside a logged run
- `stages/`: one file per stage, `run(cfg, conn, run_id)`; `stages/paths.py` names every file a stage writes
- `src/`: config and DB logging, then the method on arrays and frames: data, preprocessing, space, graph, clustering,
  complexity/, models, baselines, regressor, evaluation
- `dashboard/`: FastAPI `app.py`, read-only `queries.py`, `static/` (one page, hand-made SVG charts)
- `generate_synthetic.py`: the synthetic dataset, a ladder of sub-groups per class that overlap their rivals by
  known margins (the `true_subgroup` column records it; no stage reads it)
- `tests/`: the suite; the session fixture runs every stage once on a 12,000-row synthetic CSV

To add:
- a dataset: put the CSV under `resources/raw_data/`, copy a yaml of `configs/datasets/`, set `name`, `path`, the
  columns and `output_dir`, and pass its name to `--dataset`.
- a model: an entry of `MODELS` in `src/models.py`, a factory and whether the classifier reads a missing value
  (`nan_ok`; otherwise the pipeline imputes the train median), and `configs/models/<name>.yaml`.
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
