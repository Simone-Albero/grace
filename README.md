# grace

Data quality through the geometry of the data: each class is clustered into regions, and the regions and the pairs
of neighbouring regions of different classes get a quality report after the data design smells of Flood et al.
(EuroS&P 2024): null and invalid values, copies, label conflicts, overlap, single-feature separation and shortcuts,
poor diversity. Every region gets a status: solid, or the problems it has.

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
- `clean`: reads the configured columns of the CSV (`path`; a glob such as `resources/raw_data/cic_2018/*.csv` reads
  every file it matches, in sorted order, as one dataset: the files may differ in their other columns, and a header
  line a file repeats inside is dropped; the `sources` table logs each file's rows and dropped headers), keeps one copy of the rows equal on every column (the class
  included: equal features under different classes stay apart) and records their copies in `n_duplicates` (1 for a row
  without copies), the row's weight in stage regions and no feature, then drops the classes under `min_class_count`
  distinct rows and the configured columns that hold a single value, and codes the class. A numeric cell whose text is
  not a number becomes null like an empty one, but is marked first, so the two stay apart (also as copies) and the
  quality report can tell them. No row is dropped for a missing or infinite value and nothing is scaled or encoded:
  stage prepare does it. Writes `data.parquet` (the kept columns as read, `n_duplicates`, the class as its code; a
  row's position is its id downstream), `classes.json` and `unparsable.parquet` (row and column of every cell that was
  not a number).
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
  `quality_report`, is euclidean on them. Then splits each class on its own, by geometry alone (`src/clustering.py`):
  - two disjoint halves of its distinct rows (at most `clustering.max_fit_samples` each): a Ward tree of half A, each
    row weighted by `n_duplicates` (so it is the Ward tree of the rows as read), proposes the splits; half B, each row
    attached to the node of its nearest row of A, judges them. Ward always splits where a split looks best on the rows
    it was built on, so a split is never judged on them;
  - the separation of the two sides of a split is the distance between their centroids (on A) in standard deviations
    of B's rows along the line between them, Ashman's D over √2, on the numeric columns only (a categorical value is a
    constant block of slots: two values always look far apart) and each distinct row counted once (a row of many
    copies is not a tight group), less two standard errors (delta method). It measures how far apart two groups are;
    it is not a test, since B's rows are themselves sorted by the split (Gao, Bien & Witten, JASA 2024): a split of
    one group already gives about 1.5–2.5, hence `split_separation` well above it;
  - the tree is cut from the leaves up wherever a split reaches `split_separation` (so a gap hidden inside a larger
    branch is still found), then the pieces are sewn back, the least separated pair of neighbouring pieces first
    (each against its 10 nearest), while they are under `split_separation` on the whole pieces (Hennig 2010): what is
    left are the regions. A branch under `min_cluster_floor` rows of B is pruned. Its rows, and the rows outside the
    halves, take the majority region of their `vote_k` nearest labelled rows. Past `max_regions` the pieces keep
    being sewn until the class fits;
  - the regions are made again `stability_repeats` times on other halves, and each region gets its stability: the
    Jaccard of its rows with its most similar new region (Hennig 2007), stable from 0.75, doubtful from 0.6,
    unreliable under it. A stable region is not proven, an unstable one is suspect;
  - the regions are numbered by a Ward tree of their centroids, which the dashboard draws.

  Limits: the standard error takes the rows as independent (flows of one session are not); Ward in many dimensions
  misses groups only a few sd apart; a single row of many copies is never a region of its own (the quality report
  counts the copies); the categorical columns weigh in every distance but not in the separation.
  Last, where the classes overlap, judged on the regions alone: the separation of two regions is the distance between
  their centroids in standard deviations of their rows (weighted) along the line between them. Two regions of
  different classes overlap under `overlap_separation`; single linkage on these pairs groups the regions into overlap
  zones, and a region close to no other class is in none.
  Writes `regions.parquet` (with `n_read`, the rows of the CSV) and `assignments.parquet`.
- `quality_report`: the quality of the regions, of the pairs of regions of different classes and of the classes
  (`src/quality.py`), after the smells of Flood et al. but per region. The pairs measured are those whose separation
  is under `quality.candidate_separation`, those that share label conflicts, and each region's nearest rival, each on
  at most `quality.max_rows_per_region` distinct rows per region, unweighted. Every pair measure is per side (`_a`,
  `_b`), its summary the mean of the two, so a large region does not hide a small one (Barella et al. 2021):
  - `nn_error`: the 1-NN error, the share of a side's rows whose nearest row (leave-one-out) is of the other side; rows
    tied at the least distance share the vote, so the error does not depend on the order of the rows.
    `bayes_error_low` is the Cover &
    Hart lower bound of the Bayes error it implies, a guide on a finite sample. It replaces kDN, ENN and N1, which
    said the same (Spearman 0.9-0.97 on cic) and depended on ties.
  - `single_column_acc`: the balanced accuracy of the best rule on a single column, `single_column` (OneR, Holte
    1993; `n_single_columns` counts the columns that reach `thresholds.single_column_acc`): a threshold on a number,
    each value to its side on a categorical (its slots read back as one value). It is Ho & Basu's F3 (Lorena et al.'s
    1 - F3) made robust: F3 bounds the columns by min and max, so one row out of place halves it and a binary column
    split 99/1 against 1/99 gets 0. Its scale starts at 0.5, not 0.
  - `shortcut_columns`: each side is split into halves; on the first, up to `quality.max_group` columns are removed
    one at a time, each the one that raises the 1-NN error most (ties to the larger margin d_own / (d_own + d_other));
    on the second, the errors are measured (`nn_error_without`, `nn_error_rise`), so choosing does not inflate them.
    Verdict, with tau = `thresholds.nn_error`: `label_conflict` (error >= tau and rows equal across the two classes),
    `class_overlap` (error >= tau and no single column
    reaches 1 - tau: the 1-NN, swamped by the other columns, misses a column that alone separates), `shortcut`
    (removing the first columns raises the error by tau or more, up to tau at least), `robust` otherwise. L2, the
    error of a linear SVM, is gone: on its own training rows it is 0 whenever a side has fewer rows than dimensions,
    and cross-validated it added nothing to the 1-NN error.
  Per region: its copies (`n_duplicates` = `n_read - n_rows`), its rows in a label conflict (rows equal on every
  feature as read under another class), its rows with a null or invalid value by kind (empty, unparsable, NaN, inf),
  its rival (the pair where its own side errs most) and its pairs by verdict, then its problems: `missing` (any null or
  invalid value), `duplicates` (`dup_share` >= `thresholds.dup_share`), `label_conflict`, `class_overlap`, `shortcut` (a pair of
  that verdict); none makes it solid. Per class: the roll-up, the share of its CSV rows in solid regions and in each
  problem, and its diversity: the mean cosine similarity of `quality.diversity_pairs` pairs of its rows on the points
  scaled into [0, 1] (pdd) and the share over 0.95 (tc), problems `low_diversity` and `collapse` from
  `thresholds.pdd` and `thresholds.tc`. They are judged per class only: a region is alike by construction.

Configs:
- `configs/datasets/`: one yaml per dataset: `name`, `path`, `label_col`, `num_cols`, `cat_cols`,
  `min_class_count`, `top_n`, `seed`, `output_dir`, `db_path`. Four network datasets (`nb15_v2`,
  `cic_2018_v2`, `ton_iot_v2`, `bot_iot_v2`), five general benchmarks (`bank_marketing`, `covertype`,
  `letter_recognition`, `statlog_landsat_satellite`, `thyroid_disease`) and `synthetic_test`; their CSVs are not in
  the repo.
- `configs/method/default.yaml`: the method, shared by every dataset: `distance` (`euclidean` or `cosine`), `prepare`,
  `clustering` (the regions) and `quality` (the report), each key commented.
  Another method is a copy of it passed as `--method <path>`.

`src/config.py` loads both into frozen dataclasses and validates them before any stage reads data.

Outputs: the files of a dataset in its `output_dir` (`resources/processed/<dataset>/`), every dataset's runs in one
SQLite file, `db_path` (`resources/processed/grace.db`). Table `runs` holds one row per stage run (stage, dataset, the
whole config as JSON, status `running`, `ok` or `failed`, start and end); every other table carries the `run_id` of the
run that wrote it and is created by its first insert: `clean_summary` (with the null and invalid cells read, by
kind), `dropped_classes`, `dropped_columns`,
`class_counts`, `features` (clean); `prepare_summary`, `prepared_columns` (prepare); `regions` (with the stability of
each region), `region_tree` (the Ward tree of each class's regions, each node with the distinct and CSV rows below it),
`class_clusterings` (with the pieces cut before sewing, the separation the regions keep and the mean stability),
`granularity_candidates` (the regions of each class at other thresholds), `overlap_zones` (the tree of each
overlap zone, each merge at the separation of the pair that made it), `region_overlaps` (every pair of regions of
different classes under `overlap_separation`), `region_stats` (regions); `quality_pairs`, `quality_regions`,
`quality_classes`, `region_missing` (the null and invalid cells by region, column and kind),
`label_conflict_groups` (every row in a label conflict, with its group), `quality_stats` (with the share of the CSV
rows in solid regions and in each problem) (quality_report). There are no migrations: a new column is added to its
table, an old one stays, empty for the new runs; to drop the old ones, delete the DB.

Dashboard: `make dashboard` (`PORT=` to change it) or `python -m uvicorn dashboard.app:app` serves a read-only page
on 127.0.0.1, one tab per stage, for the datasets of `configs/datasets/`. Each tab shows the latest `ok` run of its
stage on that dataset, and says which stage to run when there is none. The regions tab draws the tree of each class;
a click on a node maps its rows (at most 2,000 per region, 20,000 in all) on their first two principal components,
computed on the click. Below it, the overlap zones show where the classes overlap; a click on one of their nodes maps
its rows by class and names its branch in each class tree. The quality tab has a summary line, a "How to read" box
with every definition and threshold (each column header also explains itself on hover) and two panels. Classes: one
row per class with its shares of solid, copied, conflicting and null rows, nn_error, best single column, pdd, tc and
problems. Regions: filtered by class and by problem, one column per check saying what was found (which null cells,
the copies, how many rows identical to which classes, the worst neighbour with its verdict and the pairs by verdict).
A click on a region opens its detail (`/quality/region`, `/quality/map`): its neighbours of other classes with every
pair measure from its side, their nn_error with every column and without the shortcut columns, a map of its rows and
its neighbours' by class, the heaviest groups of rows identical to rows of other classes with their values, and its
null cells by column. The URL hash keeps tab and dataset, on the regions tab class, node and overlap node, on the
quality tab panel and region.

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
- a pair measure: a function of `src/quality.py` called in `_pair_measures` of `stages/quality_report.py`, per side
  (`_a`, `_b`) and their mean.

Known limits: every rerun of a stage overwrites its files, so the files hold only the latest run of each stage, and
nothing records which run wrote a stage's inputs: after changing the method or rerunning a stage, rerun everything
after it (`make run`). Two method yamls on one dataset share its `output_dir`. A failed rerun can leave files that
disagree with the latest `ok` run the dashboard shows. A run killed by anything but Ctrl-C stays `running`. The k-NN
searches use every core.
