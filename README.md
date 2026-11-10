# grace

Data quality through the geometry of the data: each class is clustered into regions, and each region and each class
is described by 45 data-complexity descriptors.

```
make generate                                                    # resources/raw_data/synthetic/synthetic_test.csv
make run DATA=synthetic_test                                     # the four stages below, in order
python main.py --stage clean --dataset synthetic_test
python main.py --stage prepare --dataset synthetic_test          # after clean
python main.py --stage regions --dataset synthetic_test          # after prepare
python main.py --stage complexity --dataset synthetic_test       # after regions
make dashboard                                                   # http://127.0.0.1:8000
```

Python 3.12, `pip install -r requirements.txt`, every command from the repo root (the paths in the configs are
relative to it). `--dataset` and `--method` take a name of `configs/datasets/` and `configs/method/` or the path of a
yaml; `--method` defaults to `default`. `make help` lists the targets; `make test` runs the suite. `resources/` is not
tracked: a fresh clone has no data, and the synthetic generator is the only dataset that works out of the box.

Stages (each writes its files into the dataset's `output_dir` and its rows into the DB):
- `clean`: reads the configured columns of the CSV, keeps one copy of the rows equal on every column (the class
  included: equal features under different classes stay apart) and records their copies in a numeric feature,
  `n_duplicates` (1 for a row without copies), then drops the classes under `min_class_count` distinct rows and the
  configured columns that hold a single value, and codes the class. No row is dropped for a missing or infinite
  value and nothing is scaled or encoded: stage prepare does it. Writes `data.parquet` (the kept columns as read,
  `n_duplicates`, the class as its code; a row's position is its id downstream) and `classes.json`.
- `prepare`: the features every distance is measured on (`src/prepare.py`). Each number, `n_duplicates` included,
  becomes sign(x)·log1p(|x|), then is standardized (mean 0, standard deviation 1; a missing or infinite one lands on
  the mean, and the column `n_missing`, always present and unscaled, counts how many numbers of the row did; it is a
  diagnostic, no distance sees it). Each categorical column gets an integer code, its `top_n` most frequent values (a
  missing value counts as one) their own and the rest a shared "other", then a one-hot slot per code. Its active slot
  weighs sqrt(`prepare.cat_cost` / g), g = 1 - Σ p² over the shares of its slots (the chance two rows differ on it):
  on average over pairs of rows a categorical column then adds `cat_cost` times what a standardized number adds, and
  a mismatch on a concentrated column costs more. Writes `prepared.parquet` (the prepared columns, Float64, then the
  class code; same rows as `data.parquet`).
- `regions`: builds the points of the space (`graph.space_points`): the prepared numbers, at unit length under
  `cosine`, then the categorical slots unchanged, `n_missing` left out. Every distance from there on, here and in
  `complexity`, is euclidean on them. Builds the k-NN graph of a sample of them, then clusters each class on its
  own. The granularity is the candidate of least loss: the kDN hardness spread a region hides plus the sampling
  noise of its mean hardness. Small clusters and noise merge into the nearest region of their class. Writes
  `regions.parquet`, `assignments.parquet`, `graph.npz`.
- `complexity`: 45 descriptors (feature overlap F1-F4, neighbourhood N1-N4, network density and hubs,
  dimensionality T2-T4, geometry, linearity L1-L3; `src/complexity/`) of every region and every class, on its rows
  against the `complexity.top_k_clusters` nearest populations of other classes. Writes `complexity.parquet`.

Configs:
- `configs/datasets/`: one yaml per dataset: `name`, `path`, `label_col`, `num_cols`, `cat_cols`,
  `min_class_count`, `top_n`, `seed`, `output_dir` and `db_path`. Four network datasets (`nb15_v2`,
  `cic_2018_v2`, `ton_iot_v2`, `bot_iot_v2`), five general benchmarks (`bank_marketing`, `covertype`,
  `letter_recognition`, `statlog_landsat_satellite`, `thyroid_disease`) and `synthetic_test`; their CSVs are not in
  the repo.
- `configs/method/default.yaml`: the method, shared by every dataset: `distance` (`euclidean` or `cosine`), `prepare`,
  `graph`, `clustering` (`name`: `kmeans`, `birch`, `spectral` or `hdbscan`) and `complexity`, each key commented.
  Another method is a copy of it passed as `--method <path>`.

`src/config.py` loads both into frozen dataclasses and validates them before any stage reads data.

Outputs: the files of a dataset in its `output_dir` (`resources/processed/<dataset>/`), every dataset's runs in one
SQLite file, `db_path` (`resources/processed/grace.db`). Table `runs` holds one row per stage run (stage, dataset,
the whole config as JSON, status `running`, `ok` or `failed`, start and end); every other table carries the `run_id`
of the run that wrote it and is created by its first insert: `clean_summary`, `dropped_classes`, `dropped_columns`,
`class_counts`, `features` (clean); `prepare_summary`, `prepared_columns` (prepare); `regions`,
`class_clusterings`, `granularity_candidates`, `region_stats` (regions); `complexity`. There are no migrations: after
a change of a table, delete the DB.

Dashboard: `make dashboard` (`PORT=` to change it) or `python -m uvicorn dashboard.app:app` serves a read-only page
on 127.0.0.1, one tab per stage, for the datasets of `configs/datasets/`. Each tab shows the latest `ok` run of its
stage on that dataset, and says which stage to run when there is none. The URL hash keeps tab and dataset.

Layout:
- `main.py`: the CLI, one stage per call inside a logged run
- `stages/`: one file per stage, `run(cfg, conn, run_id)`; `stages/paths.py` names every file a stage writes
- `src/`: config and DB logging, then the method on arrays and frames: data, prepare, graph, clustering, complexity/
- `dashboard/`: FastAPI `app.py`, read-only `queries.py`, `static/` (one page, hand-made SVG charts)
- `generate_synthetic.py`: the synthetic dataset, a ladder of sub-groups per class that overlap their rivals by
  known margins (the `true_subgroup` column records it; no stage reads it)
- `tests/`: the suite; the session fixture runs every stage once on a 12,000-row synthetic CSV

To add:
- a dataset: put the CSV under `resources/raw_data/`, copy a yaml of `configs/datasets/`, set `name`, `path`, the
  columns and `output_dir`, and pass its name to `--dataset`.
- a clustering: a function `fit_<name>(X, *, <params>, max_fit_samples, random_state)` returning a label per row
  (-1 for noise) in `ALGORITHMS` in `src/clustering.py`. Its keyword-only parameters become the allowed
  `clustering.params`; a list value is swept and the candidate of least loss kept. If it takes `n_clusters`, the
  stage sweeps it per class and the yaml must not set it.
- a descriptor: one more key returned by a family function of `src/complexity/` and listed in `MEASURES` in
  `src/complexity/measure.py`.

Known limits: every rerun of a stage overwrites its files, so the files hold only the latest run of each stage, and
nothing records which run wrote a stage's inputs: after changing the method or rerunning a stage, rerun everything
after it (`make run`). Two method yamls on one dataset share its `output_dir`. A failed rerun can leave files that
disagree with the latest `ok` run the dashboard shows. A run killed by anything but Ctrl-C stays `running`. The k-NN
searches use every core.
