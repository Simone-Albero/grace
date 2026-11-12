# grace

Data quality through the geometry of the data: each class is cut into cells of about the same size, every cell is
checked for the defects of a dataset after the data design smells of Flood et al. (EuroS&P 2024) (invalid values,
duplicates, label conflicts, class overlap and shortcuts), and neighbouring cells with the same defects are merged into
regions for reading. Each defect has a one-line definition, and the report has a single threshold, tau, the share of a
cell's rows that may be confused with another class's.

```
make generate                                                    # resources/raw_data/synthetic/synthetic_test.csv
make run DATA=synthetic_test                                     # the four stages below, in order
python main.py --stage clean --dataset synthetic_test
python main.py --stage prepare --dataset synthetic_test          # after clean
python main.py --stage cells --dataset synthetic_test            # after prepare
python main.py --stage quality_report --dataset synthetic_test   # after cells
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
  without copies), the row's weight in stage cells and no feature, then drops the classes under `min_class_count`
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
- `cells`: builds the points of the space (`graph.space_points`): the prepared numbers, at unit length under
  `cosine`, then the categorical slots unchanged, `n_missing` left out. Every distance from there on, here and in
  `quality_report`, is euclidean on them. Then cuts each class on its own, by geometry alone, into cells
  (`src/cells.py`):
  - a Ward tree of a uniform sample of its distinct rows (at most `cells.max_fit_samples`), each row weighted by
    `n_duplicates` (so it is the Ward tree of the rows as read); every row of the class joins its nearest sampled row,
    so each node of the tree holds a known number of the class's rows;
  - from the root, a node of more than `cells.max_rows` rows is split into its two children, until every node left
    is a cell; a child under `cells.min_rows` rows is no cell of its own, its rows join the nearest cell;
  - the cells are the units every defect is measured on. Their size limit is the same for every class, so two cells
    compared are sampled alike (with cells of different sizes, measured on the same number of rows, the larger one's
    rows would look rarer than they are where the two mix); and bounded, so a defect touching a share of a cell's rows
    is not diluted in a larger group: the limit is the resolution of the report (with 1000 rows and tau = 0.1, a
    defect of about 100 rows is seen);
  - the cut is the tree of the cells: sibling cells are the most alike, and the report merges them up this tree.

  Limits: Ward in many dimensions joins groups only a few sd apart into one node, so a cell may straddle them; a
  single row of many copies counts once (the quality report counts the copies); a class of fewer rows than
  `max_rows` is a single cell, so nothing is localized inside it.
  Writes `cells.parquet` (with `n_read`, the rows of the CSV), `assignments.parquet` (the cell of every row) and
  `cell_tree.parquet` (the cut of every class).
- `quality_report`: the defects of every cell (`src/quality.py`), after the smells of Flood et al. but per cell, then
  of the regions the cells are merged into. The rivals of a cell are the cells of other classes that hold the nearest
  row of another class to one of its rows (searched from at most `quality.rival_rows` rows per cell): no threshold
  picks them. Each pair of a cell and a rival is measured on at most `quality.max_rows_per_cell` distinct rows per
  cell, unweighted, per side (`_a`, `_b`):
  - `nn_error`: the 1-NN error, the share of a side's rows whose nearest row (leave-one-out) is of the other side; rows
    tied at the least distance share the vote. `bayes_error_low` is the Cover & Hart lower bound of the Bayes error it
    implies, a guide on a finite sample;
  - `single_column_acc`: the per-side accuracy of the best rule on a single column, `single_column` (OneR, Holte
    1993): a threshold on a number, each value to its side on a categorical (its slots read back as one value);
  - `confusion`: the per-side error of the better of the two (the lower mean over the sides). Any classifier errs at
    least as much as the best one can; the 1-NN alone overstates the error when columns that do not separate swamp
    the distance, and a single column misses a separation that needs several;
  - `key_columns`: each side is split into halves; on the first, up to `quality.max_group` columns are removed one at
    a time, each the one that raises the 1-NN error most (ties to the larger margin d_own / (d_own + d_other)); on the
    second, the confusion is measured after each removal (`confusion_without`, the single column then sought among
    the columns left), so choosing does not inflate it.

  A cell, or a region, has a defect when (`quality.tau` = 0.1):

  | defect | when | why |
  |---|---|---|
  | `invalid_values` | a row holds an empty, unparsable, NaN or infinite cell | by definition |
  | `duplicates` | a row comes more than once in the CSV | copies can fall on both sides of a train/test split |
  | `label_conflict` | a row equals a row of another class on every feature | no classifier can tell them apart |
  | `class_overlap` | against a rival cell, its confusion is tau or more | even the better classifier mixes the two |
  | `shortcut` | against a rival cell its confusion is under tau, but removing at most `max_group` key columns brings it to tau | the separation rests on those few columns |

  The first three are facts of single rows: counted in a group of any size, they are never diluted. The last two are
  measured between cells, at the resolution of the cells. The cells of a class are then merged into regions up their
  tree: a region is the largest branch whose cells all have the same classes of `class_overlap` and of `shortcut`,
  and its row defects are counted over its rows. Measuring the merged regions instead would dilute again: on
  cic_2018_v2 it loses 1.9% of the rows with `class_overlap` and 2.5% with `shortcut` (8% and 12% on the synthetic
  dataset), so the regions only gather what the cells found. Per cell (`quality_cells`) and per region
  (`quality_regions`): the copies, the rows in a label conflict and the classes they conflict with, the rows with a
  null or invalid value by kind, the classes of `class_overlap` and of `shortcut`, the highest confusion with a rival
  and its class, and the `defects`; per cell also the rival under tau whose confusion rises most when columns are
  removed (`shortcut_rival`, `shortcut_columns`, `confusion_without`). Per class (`quality_classes`) and over the
  dataset (`quality_stats`): the share of the CSV rows with each defect and with none. A row has `invalid_values`,
  `duplicates` (all copies of a row but the first) and `label_conflict` itself, `class_overlap` and `shortcut` through
  its cell: a region flagged for one conflicting row does not count its other rows as conflicting.

Configs:
- `configs/datasets/`: one yaml per dataset: `name`, `path`, `label_col`, `num_cols`, `cat_cols`,
  `min_class_count`, `top_n`, `seed`, `output_dir`, `db_path`. Four network datasets (`nb15_v2`,
  `cic_2018_v2`, `ton_iot_v2`, `bot_iot_v2`), five general benchmarks (`bank_marketing`, `covertype`,
  `letter_recognition`, `statlog_landsat_satellite`, `thyroid_disease`) and `synthetic_test`; their CSVs are not in
  the repo.
- `configs/method/default.yaml`: the method, shared by every dataset: `distance` (`euclidean` or `cosine`), `prepare`,
  `cells` and `quality` (the report), each key commented.
  Another method is a copy of it passed as `--method <path>`.

`src/config.py` loads both into frozen dataclasses and validates them before any stage reads data.

Outputs: the files of a dataset in its `output_dir` (`resources/processed/<dataset>/`), every dataset's runs in one
SQLite file, `db_path` (`resources/processed/grace.db`). Table `runs` holds one row per stage run (stage, dataset, the
whole config as JSON, status `running`, `ok` or `failed`, start and end); every other table carries the `run_id` of the
run that wrote it and is created by its first insert: `clean_summary` (with the null and invalid cells read, by
kind), `dropped_classes`, `dropped_columns`,
`class_counts`, `features` (clean); `prepare_summary`, `prepared_columns` (prepare); `cells`, `cell_tree` (the cut of
each class, each node with the distinct and CSV rows below it), `class_cells`, `cell_stats` (cells); `quality_pairs`
(every pair of a cell and a rival), `quality_cells`, `quality_regions`, `quality_classes`, `cell_missing` (the null and
invalid cells by cell, column and kind), `label_conflict_groups` (every row in a label conflict, with its group),
`quality_stats` (with tau and the share of the CSV rows with each defect and with none)
(quality_report). There are no migrations: a new column is added to its table, an old one stays, empty for the new
runs; to drop the old ones, delete the DB.

Dashboard: `make dashboard` (`PORT=` to change it) or `python -m uvicorn dashboard.app:app` serves a read-only page
on 127.0.0.1, one tab per stage, for the datasets of `configs/datasets/`. Each tab shows the latest `ok` run of its
stage on that dataset, and says which stage to run when there is none. The cells tab draws the cut of each class, at
most 63 nodes at a time: from the top, the node with the most rows is opened first; a node left closed is drawn a
level down on a click, and a path leads back up. A click on a node maps its rows (at most 2,000 per cell, 20,000 in
all) on their first two principal components, computed on the click. A map draws each colour (a class, or a subtree) as the density of its rows, each sampled row
counting for the rows of its cell it stands for: curves enclosing 25, 50, 75 and 95% of them, smoothed by a
Gaussian as wide as the median square around a row holding n^(-1/3)/π of its colour's rows (Scott's bandwidth for a
Gaussian cloud, but local), and their density along each axis in strips above and on the right; a switch adds the
rows as points. The quality tab has a summary line, a "How to read" box with every definition (each column header also
explains itself on hover) and two panels. Classes: per class, its cells and regions and the share of its CSV rows
with each defect and with none. Regions: filtered by class and by defect, one column per defect saying what was found
(which null cells, the copies, how many rows identical to which classes, the classes of `class_overlap` and of
`shortcut`). A click on a region opens its detail (`/quality/region`, `/quality/map`): its cells, every pair of one of
them and a rival with every measure from its side, the confusion of the most confused pairs with every column and
after each key column removed, a map of its cells and their rivals by class, the heaviest groups of rows identical to
rows of other classes with their values, and its null cells by column. The URL hash keeps tab and dataset, on the
cells tab class and node, on the quality tab panel and region.

Layout:
- `main.py`: the CLI, one stage per call inside a logged run
- `stages/`: one file per stage, `run(cfg, conn, run_id)`; `stages/paths.py` names every file a stage writes
- `src/`: config and DB logging, then the method on arrays and frames: data, prepare, graph, cells, quality
- `dashboard/`: FastAPI `app.py`, read-only `queries.py`, `static/` (one page, hand-made SVG charts)
- `generate_synthetic.py`: the synthetic dataset, a ladder of sub-groups per class that overlap their rivals by
  known margins (the `true_subgroup` column records it; no stage reads it)
- `tests/`: the suite; the session fixture runs every stage once on a 12,000-row synthetic CSV

To add:
- a dataset: put the CSV under `resources/raw_data/`, copy a yaml of `configs/datasets/`, set `name`, `path`, the
  columns and `output_dir`, and pass its name to `--dataset`.
- a pair measure: a function of `src/quality.py` called in `_pair_measures` of `stages/quality_report.py`, per side
  (`_a`, `_b`) and their mean.
- a defect: its rule in `defects_of` (and in `rival_defects` if it is measured against a rival) and its name in
  `DEFECTS` (`src/quality.py`), and its line in the table above: one definition, and tau as its threshold if it needs
  one.

Known limits: every rerun of a stage overwrites its files, so the files hold only the latest run of each stage, and
nothing records which run wrote a stage's inputs: after changing the method or rerunning a stage, rerun everything
after it (`make run`). Two method yamls on one dataset share its `output_dir`. A failed rerun can leave files that
disagree with the latest `ok` run the dashboard shows. A run killed by anything but Ctrl-C stays `running`. The k-NN
searches use every core.
