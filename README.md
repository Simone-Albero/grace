# grace

Data quality through the geometry of the data: each class is clustered into regions, and the regions and the pairs
of neighbouring regions of different classes get a quality report after the data design smells of Flood et al.
(EuroS&P 2024): copies, label conflicts, overlap, single-feature separation and shortcuts, poor diversity.

```
make generate                                                    # resources/raw_data/synthetic/synthetic_test.csv
make run DATA=synthetic_test                                     # the four stages below, in order
python main.py --stage clean --dataset synthetic_test
python main.py --stage prepare --dataset synthetic_test          # after clean
python main.py --stage regions --dataset synthetic_test          # after prepare
python main.py --stage quality_report --dataset synthetic_test   # after regions
make dashboard                                                   # http://127.0.0.1:8000
```

Python 3.12, `pip install -r requirements.txt`, every command from the repo root (the paths in the configs are
relative to it). `--dataset` and `--method` take a name of `configs/datasets/` and `configs/method/` or the path of a
yaml; `--method` defaults to `default`. `make help` lists the targets; `make test` runs the suite. `resources/` is not
tracked: a fresh clone has no data, and the synthetic generator is the only dataset that works out of the box.

Stages (each writes its files into the dataset's `output_dir` and its rows into the DB):
- `clean`: reads the configured columns of the CSV, keeps one copy of the rows equal on every column (the class
  included: equal features under different classes stay apart) and records their copies in `n_duplicates` (1 for a row
  without copies), the row's weight in stage regions and no feature, then drops the classes under `min_class_count`
  distinct rows and the configured columns that hold a single value, and codes the class. No row is dropped for a
  missing or infinite value and nothing is scaled or encoded: stage prepare does it. Writes `data.parquet` (the kept
  columns as read, `n_duplicates`, the class as its code; a row's position is its id downstream) and `classes.json`.
- `prepare`: the features every distance is measured on (`src/prepare.py`). Each number becomes sign(x)·log1p(|x|),
  then is standardized (mean 0, standard deviation 1; a missing or infinite one lands on the mean, and the column
  `n_missing`, always present and unscaled, counts how many numbers of the row did; it is a diagnostic, no distance
  sees it). Each categorical column gets an integer code, its `top_n` most frequent values (a missing value counts as
  one) their own and the rest a shared "other", then a one-hot slot per code. Its active slot weighs
  sqrt(`prepare.cat_cost` / g), g = 1 - Σ p² over the shares of its slots (the chance two rows differ on it): on
  average over pairs of rows a categorical column then adds `cat_cost` times what a standardized number adds, and a
  mismatch on a concentrated column costs more. Writes `prepared.parquet` (the prepared columns, Float64, then the
  class code; same rows as `data.parquet`).
- `regions`: builds the points of the space (`graph.space_points`): the prepared numbers, at unit length under
  `cosine`, then the categorical slots unchanged, `n_missing` left out. Every distance from there on, here and in
  `quality_report`, is euclidean on them. Draws a sample of them as the reference rows of the kDN hardness
  (`graph.max_samples`), then splits each class on its own
  (`src/clustering.py`): a Ward tree of its distinct rows (at most `clustering.max_fit_samples`), each weighted by
  `n_duplicates`, so it is the Ward tree of the rows as read; then the cut of least loss, branch by branch, the loss
  being the kDN hardness spread a region hides plus the sampling noise of its mean hardness, measured on the sample.
  Each region costs `region_cost` times the loss of its class as one region, so a split is kept only when it lowers
  the loss by at least that share. A branch under `min_cluster_floor` distinct rows of the sample is pruned. Its rows,
  and the rows outside the sample of the tree, take the majority region of their `vote_k` nearest labelled rows. Past
  `max_regions` the cost of a region rises until the class fits. Last, where the classes overlap, judged on the
  regions alone: the separation of two regions is the distance between their centroids in standard deviations of
  their rows (weighted) along the line between them. Two regions of different classes overlap under
  `overlap_separation`; single linkage on these pairs groups the regions into overlap zones, and a region close to no
  other class is in none.
  Writes `regions.parquet` (with `n_read`, the rows of the CSV) and `assignments.parquet`.
- `quality_report`: the quality of the regions and of the pairs of regions of different classes (`src/quality.py`),
  after the smells of Flood et al. but per region rather than per class. The pairs measured are those whose
  separation is under `quality.candidate_separation` (in the full space or without `shortcut_cols`), those that share
  label conflicts, and each region's nearest rival; s only chooses them, every verdict comes from the measures. Per
  pair, on at most `quality.max_rows_per_region` distinct rows of each region: kDN and ENN (`quality.k` neighbours
  inside the pair), N1 (rows on the border of the pair's spanning tree), F3 read as separation (the share of rows one
  feature alone tells apart, the feature, and how many features do) and L2 (error of a linear SVM: high with a low
  overlap, the boundary is curved), all again without `shortcut_cols`. Label conflicts: rows equal on every feature as
  read under different classes (and equal on every feature but `shortcut_cols`). Per region: the copies clean
  removed (`n_read - n_rows`), the mean cosine similarity of `quality.diversity_pairs` pairs of its rows on the points
  scaled into [0, 1] (pdd) and the share of those pairs over 0.95 (tc). Per class, the roll-up of its regions.
  `quality.highlight` only says from where the dashboard marks a value, and which pairs count as shortcuts.

Configs:
- `configs/datasets/`: one yaml per dataset: `name`, `path`, `label_col`, `num_cols`, `cat_cols`,
  `min_class_count`, `top_n`, `seed`, `output_dir`, `db_path` and, optional, `shortcut_cols` (the columns suspected of
  shortcuts, which the quality report leaves out to measure the pairs again). Four network datasets (`nb15_v2`,
  `cic_2018_v2`, `ton_iot_v2`, `bot_iot_v2`), five general benchmarks (`bank_marketing`, `covertype`,
  `letter_recognition`, `statlog_landsat_satellite`, `thyroid_disease`) and `synthetic_test`; their CSVs are not in
  the repo.
- `configs/method/default.yaml`: the method, shared by every dataset: `distance` (`euclidean` or `cosine`), `prepare`,
  `graph`, `clustering` (the weighted Ward regions) and `quality` (the report), each key commented.
  Another method is a copy of it passed as `--method <path>`.

`src/config.py` loads both into frozen dataclasses and validates them before any stage reads data.

Outputs: the files of a dataset in its `output_dir` (`resources/processed/<dataset>/`), every dataset's runs in one
SQLite file, `db_path` (`resources/processed/grace.db`). Table `runs` holds one row per stage run (stage, dataset, the
whole config as JSON, status `running`, `ok` or `failed`, start and end); every other table carries the `run_id` of the
run that wrote it and is created by its first insert: `clean_summary`, `dropped_classes`, `dropped_columns`,
`class_counts`, `features` (clean); `prepare_summary`, `prepared_columns` (prepare); `regions` (with the mean and
standard deviation of the kDN hardness of each region), `region_tree` (the tree each class's cut keeps, each node with
the distinct and CSV rows below it), `class_clusterings` (with the loss over that of the class as one region),
`granularity_candidates`, `overlap_zones` (the tree of each
overlap zone, each merge at the separation of the pair that made it), `region_overlaps` (every pair of regions of
different classes under `overlap_separation`), `region_stats` (regions); `quality_pairs`, `quality_regions`,
`quality_classes`, `label_conflict_groups` (every row in a label conflict, with its group), `quality_stats`
(quality_report). There are no migrations: after a change of a table, delete the DB.

Dashboard: `make dashboard` (`PORT=` to change it) or `python -m uvicorn dashboard.app:app` serves a read-only page
on 127.0.0.1, one tab per stage, for the datasets of `configs/datasets/`. Each tab shows the latest `ok` run of its
stage on that dataset, and says which stage to run when there is none. The regions tab draws the tree of each class;
a click on a node maps its rows (at most 2,000 per region, 20,000 in all) on their first two principal components,
computed on the click. Below it, the overlap zones show where the classes overlap; a click on one of their nodes maps
its rows by class and names its branch in each class tree. The quality tab shows the smells per class, the pairs of
regions with their diagnoses (filtered by diagnosis) and the regions. The URL hash keeps tab and dataset, and on the
regions tab class, node and overlap node.

Layout:
- `main.py`: the CLI, one stage per call inside a logged run
- `stages/`: one file per stage, `run(cfg, conn, run_id)`; `stages/paths.py` names every file a stage writes
- `src/`: config and DB logging, then the method on arrays and frames: data, prepare, graph, clustering, quality
- `dashboard/`: FastAPI `app.py`, read-only `queries.py`, `static/` (one page, hand-made SVG charts)
- `generate_synthetic.py`: the synthetic dataset, a ladder of sub-groups per class that overlap their rivals by
  known margins (the `true_subgroup` column records it; no stage reads it)
- `tests/`: the suite; the session fixture runs every stage once on a 12,000-row synthetic CSV

To add:
- a dataset: put the CSV under `resources/raw_data/`, copy a yaml of `configs/datasets/`, set `name`, `path`, the
  columns and `output_dir`, and pass its name to `--dataset`.
- a pair measure: a function of `src/quality.py` called in `_pair_measures` of `stages/quality_report.py`, its key
  listed in `MEASURES` there.

Known limits: every rerun of a stage overwrites its files, so the files hold only the latest run of each stage, and
nothing records which run wrote a stage's inputs: after changing the method or rerunning a stage, rerun everything
after it (`make run`). Two method yamls on one dataset share its `output_dir`. A failed rerun can leave files that
disagree with the latest `ok` run the dashboard shows. A run killed by anything but Ctrl-C stays `running`. The k-NN
searches use every core.
