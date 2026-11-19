# grace

Data quality through the geometry of the data, independent of any model: each class is cut into the smallest cells a
measure makes sense on, every cell is checked for the defects of a dataset after the data design smells of Flood et
al. (EuroS&P 2024) and diagnosed by its relation to other cells (against the cells of another class: a suspect label,
class overlap or a shortcut; against the other cells of its class: redundancy or an atypical provenance), and the cells
of a class merge into regions where they do not contradict each other, the units the report is read in. Each defect
has a one-line definition, the report has a single threshold, tau, and every measure is read with its error of
sampling at one confidence.

```
make generate                                                    # resources/raw_data/synthetic/synthetic_test.csv
make run DATA=synthetic_test                                     # the four stages below, in order (METHOD=<name|yaml>)
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
  distinct rows, samples every class over `max_class_rows` distinct rows down to a uniform sample of that many (each
  row kept with its copies counted; the shares of a sampled class are estimates on the sample) and drops the
  configured columns that hold a single value, and codes the class. The rows are never held whole: the CSVs are read
  twice in streaming, first for a key of each row (two 64-bit hashes of its columns, which stand for the row: two
  distinct rows share one with a chance of about 1e-23 at 6e7 rows), from which the copies, the rare classes and the
  sample are found, then for the rows kept only. A numeric cell whose text is
  not a number becomes null like an empty one, but is marked first, so the two stay apart (also as copies) and the
  quality report can tell them. No row is dropped for a missing or infinite value and nothing is scaled or encoded:
  stage prepare does it. Writes `data.parquet` (the kept columns as read, `n_duplicates`, the class as its code; a
  row's position is its id downstream), `classes.json` and `unparsable.parquet` (row and column of every cell that was
  not a number). When the dataset names its provenance columns, it also writes `provenance.parquet`: for every row of
  the CSV that is a copy of a kept row, in reading order, the row of `data.parquet` it is a copy of and its source and destination
  address and port and its time, as read (a copy is equal on the features, not on where it comes from: each keeps its
  own); `provenance_columns` logs, per column, its empty values, the times `time_format` does not read and the
  distinct values.
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
  - from the root, a node is split into its two children while it holds at least 2 · `min_rows` rows, until every
    node left is a cell; a child under `min_rows` rows is no cell of its own, its rows join the nearest cell. A cell
    so holds from `min_rows` to under twice it, but for the small branches it gathers. `min_rows` is no key of the
    method: it is what `quality.tau` and `quality.confidence` ask for, twice the rows a share must be measured on to
    be able to fall short of tau (`quality.decisive_rows`, z² (1 - tau) / tau rows; half of a cell's rows choose the
    rule and half are measured, and a cell is never sampled down). Under it a cell could never be told to lack a defect, whatever it holds: 50 rows at
    tau 0.1 and 0.95, 32 at tau 0.15. A smaller tau or a higher confidence so asks for larger cells;
  - the cells are the units every defect is measured on, as small as the measures allow, so a defect touching a share
    of a cell's rows is not diluted in a larger group. The regions they merge into (below) set the granularity of the
    report: large where a class is uniform, small where it is not;
  - the cut is the tree of the cells: sibling cells are the most alike, and the report merges them up this tree.

  Limits: Ward in many dimensions joins groups only a few sd apart into one node, so a cell may straddle them; a
  single row of many copies counts once (the quality report counts the copies); a class of fewer than 2 · `min_rows`
  rows is a single cell; a class of more rows than `max_fit_samples` · 2 · `min_rows` is cut at most into its sampled
  rows.
  Writes `cells.parquet` (with `n_read`, the rows of the CSV), `assignments.parquet` (the cell of every row) and
  `cell_tree.parquet` (the cut of every class).
- `quality_report`: the defects of every cell (`src/quality.py`), after the smells of Flood et al. but per cell, then
  of the regions the cells are merged into. The rival cells of a cell are the cells of other classes that hold the
  nearest row of another class to one of its rows (searched from at most `quality.rival_rows` rows per cell): no
  threshold picks them. A rival class is the class of some of them. A cell is measured once against each rival class:
  its rows against the rows of its rival cells of that class pooled, so that rows confused with a class spread over
  several of its cells are counted together (against each cell alone, a cell of rows relabelled from a class cut into
  nine cells was confused at most 0.19 on the injection experiment). The cell is measured on all its rows,
  so that `min_rows` holds what it promises (half of them are scored, enough to decide a defect); the rival side is
  sampled down to `quality.max_rows_per_cell`, which in many dimensions barely moves the nearest rival row (thinning
  it by r multiplies that distance by r^(-1/d), 0.97 at r = 0.1 and d = 70). Distinct rows, unweighted; each side is
  split into halves, one to choose on and one to measure on, so that choosing does not inflate what is measured. Read
  on the cell's side:
  - `nn_error`: the 1-NN error, the share of the cell's measured rows whose nearest row (leave-one-out) is of the rival
    class; rows tied at the least distance share the vote. `bayes_error_low` is the Cover & Hart lower bound of the
    Bayes error it implies, a guide on a finite sample;
  - `single_column_acc`: the accuracy on the cell's measured rows of the best rule on a single column, `single_column`
    (OneR, Holte 1993), chosen on the other half: a threshold on a number, each value to its side on a categorical
    (its slots read back as one value; a value not seen when choosing goes to the cell's side);
  - `confusion`: the error of the better of the two, the one whose worse side errs less (a rule that sends nearly
    every row to the cell's side, right on it and wrong on the rival's, is so no better than the 1-NN). Any classifier
    errs at least as much as the best one can; the 1-NN alone overstates the error when columns that do not separate
    swamp the distance, and a single column misses a separation that needs several;
  - `key_columns`: on the choosing half, up to `quality.max_group` columns are removed one at a time, each the one that
    raises the 1-NN error most (ties to the larger margin d_own / (d_own + d_other)); on the measured half, the
    confusion is measured after each removal (`confusion_without`, the single column then sought among the columns
    left). A shortcut is read on the rise, `confusion_without` less `confusion`, not on its value: the share of the
    cell's rows that those columns alone keep apart, a share of the same rows, read with its own interval. So a cell
    already near tau is no shortcut for a hair of a rise, and a cell far from its rival is one when a few columns
    alone hold them apart.

  Every measure of a cell is an estimate on few rows, so it is read with its interval at `quality.confidence` (0.95,
  each bound one-sided): Wilson's for a share of rows, the normal one for a mean. Against a threshold a cell has the
  defect when the interval lies at or past it, has it not when the interval lies short of it, and is undecided
  otherwise (`undecided`). A small cell is so judged only on strong evidence, at any granularity.

  A cell has a defect when (`quality.tau` = 0.1, the threshold the run logs its verdicts at; see "tau, after the run"
  below):

  | defect | relation | when | why |
  |---|---|---|---|
  | `invalid_values` | the row | a row holds an empty, unparsable, NaN or infinite cell | by definition |
  | `duplicates` | the row | a row comes more than once in the CSV | copies can fall on both sides of a train/test split |
  | `label_conflict` | the row | a row equals a row of another class on every feature | no classifier can tell them apart |
  | `suspect_label` | cell and rival class | its confusion is at least one half | the cell lies inside the other class, as rows under a wrong label would |
  | `class_overlap` | cell and rival class | its confusion is tau or more, and it is no `suspect_label` | even the better classifier mixes the two |
  | `shortcut` | cell and rival class | its confusion is under tau, but removing at most `max_group` key columns adds tau of its rows to it | the separation rests on those few columns |
  | `redundancy` | cell and its class | its variance (the mean squared distance of its rows to their centre) is at most tau times the typical one of a cell of its class (the median) | near copies of one another, as one session repeated |
  | `atypical_provenance` | cell and its class | for one key of where its rows come from (`src_ip`, `src_port` or the part of its class's time), at least tau of its CSV rows come from one value more than in the typical cell of its class | one recording, which a classifier can learn by heart, or traffic of another kind under the class's label |

  The first three are facts of single rows: counted in a group of any size, they are never diluted, and need no
  interval. The next three compare a cell with a rival class, the last two with the other cells of its class; a
  dataset without provenance does not measure `atypical_provenance` (its shares are null). One half, the majority, is
  the other threshold of the report: it is no choice.

  The cells of a class are then merged into regions up their tree: a region is the largest branch whose cells do not
  contradict each other, no defect against one target (a class, a key) being established in one cell and ruled out in
  another (a cell with no rival cell of a class rules out the rival defects against it); an undecided cell never
  splits a branch. A region is named by the node of the tree at its top (so the same branch has the same name at any
  tau), has the defects its cells have, and its row defects are counted over its rows. Measuring the merged regions
  instead would dilute again, so the regions only gather what the cells found. Per cell (`quality_cells`) and per
  region (`quality_regions`): the copies, the rows in a label conflict and the classes they conflict with, the rows
  with a null or invalid value by kind, the classes of `suspect_label` (`label_with`), `class_overlap` and `shortcut`
  and the keys of `atypical_provenance` (`provenance_with`), the highest confusion with a rival class, its interval and
  that class, the variance over the typical one (`variance_ratio`; per region the lowest), and the `defects`; per cell
  also its rival cells (`n_rivals`), the rival class it does not overlap whose confusion rises most when columns are
  removed (`shortcut_class`, `shortcut_columns`, `confusion_without`) and the defects it is `undecided` on. Per class
  (`quality_classes`) and over the dataset (`quality_stats`): the share of the CSV rows with each defect and with none,
  and per defect the cells undecided on it. A row has `invalid_values`, `duplicates` (all copies of a row but the first)
  and `label_conflict` itself, the others through its cell: a region flagged for one conflicting row does not count its
  other rows as conflicting.

  Provenance (`src/provenance.py`), when the dataset names its columns: per cell, over its CSV rows (each copy with its
  own, an empty value left out), for each of `src_ip`, `src_port`, `dst_ip` and `dst_port`, and of `time` (below), the
  most frequent value and its share (`<key>_top`, `<key>_top_share`), the distinct values (`<key>_n_values`) and the
  effective number (`<key>_eff_n`, exp of the Shannon entropy of their shares, Hill's number of order 1: as many
  equally frequent values have the same entropy), and the excess (`<key>_excess`, `<key>_excess_value`, with the share
  of that value in the cell and the rows it is a share of, `<key>_excess_share` and `<key>_n`): the most the cell's
  share of one of its values exceeds that value's typical share in its class, the median of its shares in the n cells
  of the class (the ceil(n/2)-th smallest, a cell without it counting 0, every cell alike). The time is a key as the
  others: the CSV rows of a class, in order of time, are cut into as many parts of equal rows as the class has cells,
  and the part a row falls in is its value (rows of one time share a part); a cell recorded at once lies in one part.
  Also for the time, the window holding the central `quality.time_share` (0.9) of the cell's rows, from the 5% to the
  95% quantile (`time_from`, `time_to`, `time_window` in seconds), so a few stray rows do not widen it, and its share
  of the same window of its class's rows (`time_window_share`), a description. `atypical_provenance` is the excess of
  `src_ip`, `src_port` or the time (`provenance.ORIGIN`, where the rows come from and when) reaching tau, read with
  the interval of the share: against the class, not as a share alone, so a class that all comes from one host (an
  attack from one machine) has none, and a host in a class of many does; against the typical cell, not the class's
  mean, as a part of the class from one host would lower the mean share of every other host, and the cells of the
  rest would rise over it. The destination and its port are measured but judged on no defect: they follow the
  service, so a cell of one service is the rule (on the injection experiment nearly every clean BENIGN cell had one
  destination port, and the typical cell of a class of many services has none).

Configs:
- `configs/datasets/`: one yaml per dataset: `name`, `path`, `label_col`, `num_cols`, `cat_cols`,
  `min_class_count`, `top_n`, `seed`, `output_dir`, `db_path`, optionally `max_class_rows` (the most distinct rows a
  class keeps, null or absent for all; at least `min_class_count`), and optionally `provenance`: the columns of where each
  row comes from, never features (`src_ip`, `src_port`, `dst_ip`, `dst_port`, `time` with its `time_format`, the
  chrono format polars reads, e.g. `"%Y-%m-%d %H:%M:%S%.f"`; each optional, a missing one fails clean as a missing
  feature does; a column named both as provenance and as a feature fails the config). The destination port is no
  feature: it follows the service, and the cells would form around it. The NetFlow v2 datasets have no time column. Four network datasets (`nb15_v2`,
  `cic_2018_v2`, `ton_iot_v2`, `bot_iot_v2`), five general benchmarks (`bank_marketing`, `covertype`,
  `letter_recognition`, `statlog_landsat_satellite`, `thyroid_disease`) and `synthetic_test`; their CSVs are not in
  the repo.
- `configs/method/default.yaml`: the method, shared by every dataset: `distance` (`euclidean` or `cosine`), `prepare`,
  `cells` and `quality` (the report), each key commented. The size of a cell is no key of it: `quality.tau` and
  `quality.confidence` set it.
  Another method is a copy of it passed as `--method <path>`.

`src/config.py` loads both into frozen dataclasses and validates them before any stage reads data.

Outputs: the files of a dataset in its `output_dir` (`resources/processed/<dataset>/`), every dataset's runs in one
SQLite file, `db_path` (`resources/processed/grace.db`). Table `runs` holds one row per stage run (stage, dataset, the
whole config as JSON, status `running`, `ok` or `failed`, start and end); every other table carries the `run_id` of the
run that wrote it and is created by its first insert: `clean_summary` (with the null and invalid cells read, by
kind), `dropped_classes`, `dropped_columns`,
`class_counts`, `features`, `sources`, `provenance_columns` (clean); `prepare_summary`, `prepared_columns` (prepare); `cells`, `cell_tree` (the cut of
each class, each node with the distinct and CSV rows below it), `class_cells`, `cell_stats` (cells); `quality_rivals`
(every cell against each rival class: its rival cells of that class, the rows measured on each side, the measures of
the cell's side), `quality_cells`, `quality_regions`, `quality_classes`, `cell_missing` (the null and
invalid cells by cell, column and kind), `label_conflict_groups` (every row in a label conflict, with its group),
`quality_stats` (with tau and the confidence, the pairs of rival cells, the measures against a rival class and those
taken on a sample of the cell's rows, the cells undecided on each defect, and the share of the CSV rows with each
defect and with none; `atypical_provenance` null without a provenance)
(quality_report). There are no migrations: a new column is added to its table, an old one stays, empty for the new
runs; to drop the old ones, delete the DB.

**tau, after the run.** What the run measures does not depend on tau: the confusion of each cell with each rival
class with every column and without the key ones, with the rows it is a share of (`quality_rivals`), the variance of
each cell and the typical one of its class, the excess of each provenance key of each cell, the counts, and
`clear_rows` per cell (its distinct rows with no null or invalid value and in no label conflict, which the share with
no defect needs). tau and the confidence only compare them. `src/report.py` (`judge`) turns these measures into the
verdicts at any tau: the defects of the cells, the regions they merge into, the shares per class and over the dataset.
The stage calls it once, at `quality.tau` and `quality.confidence`, and logs the result (`quality_rivals.region` and
`rival_regions`, `quality_cells` from `region` and the class lists on, `quality_regions`, `quality_classes`, the
shares and undecided counts of `quality_stats`); the dashboard calls it per request, with the `tau` of the request
and the run's confidence. At the run's tau it gives the logged tables, and a test checks it. A quality_report run
older than the last cells run is not judged (its cells are not those of the tree): the dashboard asks to run it
again. The description of the time window does not depend on tau either (`quality.time_share`).

Dashboard: `make dashboard` (`PORT=` to change it) or `python -m uvicorn dashboard.app:app` serves a read-only page
on 127.0.0.1 for the datasets of `configs/datasets/`, in three tabs, each topped by a table of the hyperparameters its
run used (`HYPERPARAMETERS` of `dashboard/queries.py`, read from the run's config). Each shows the latest `ok` run of
its stages on that dataset, and says which stage to run when there is none. Counts are of CSV rows (copies included,
as a classifier is trained on them); the distinct rows are on hover.

- quality (quality_report) calls *cell* what the stage calls a region, and never shows the leaves inside it. A slider
  over the matrix sets τ (0.005 to 0.5): every number of the quality and cells tabs, the regions and what the detail
  of a cell compares with τ are judged again at it, in the same code as the stage's (`?tau=` on `/quality`,
  `/quality/region`, `/quality/map`, `/quality/profile` and `/cells`; the run's own when absent), and "back to the
  run's τ" undoes it. A matrix
  of the classes, the most CSV rows first, then the whole dataset, by their clear ratio (the share of their CSV rows
  with no defect) and their share with each defect, darker as the share grows (a defect the dataset does not measure
  is left out); a click on a share lists its cells, the most CSV rows first, each defect with what was found. A click
  on a cell opens one section per defect, named by it: its null and invalid values by column; its groups of rows
  identical to rows of other classes (class a, class b, their CSV rows, the values); for `suspect_label` and
  `class_overlap`, feature by feature, the difference of the mean numeric features from the rows near it of each class
  it has the defect against (its rival leaves of that class; logged and standardized, so in standard deviations of the
  dataset; `/quality/profile`) and the most frequent categorical values; for `shortcut`, per class, the confusion with
  every column and without the shortcut columns; for `redundancy`, its variance over the typical one of its class; for
  `atypical_provenance`, per key the value, its share and excess, and the time window. A confusion is shown with its
  interval. Then a map of the cell and the rival leaves of the classes picked (`/quality/map`), and, closed, its rival
  classes (nn_error and the confusion as the key columns are removed) and its provenance (per column its distinct
  values, the most frequent and the CSV rows per repeated value). A rival class is measured by its most confused leaf
  of the cell. A glossary closes the tab.
- cells (cells) calls the cells of the stage *leaves*: per class its leaves and how many distinct rows they hold
  (median, least and most), then the cut of a class, at most 63 nodes at a time: from the top, the node with the most
  rows is opened first; a node left closed is drawn a level down on a click, and a path leads back up. A click on a
  node maps its rows (at most 2,000 per leaf, 20,000 in all) on their first two principal components, computed on the
  click. The tree and the map are coloured by subtree, or, once quality_report judged the leaves (`/cells` carries
  their region and defects), by whether a leaf has a defect; a mapped leaf links to its cell in quality.
- data (clean and prepare): the rows read, the copies, the rows kept, the classes and the features; per class its
  distinct and CSV rows; the classes and columns removed; the provenance columns.

A map draws each colour (a cell, a subtree, or the leaves with and without a defect) as the density of its rows, each
sampled row counting for the rows of its leaf it stands for: curves enclosing 25, 50, 75 and 95% of them, smoothed by
a Gaussian as wide as the median square around a row holding n^(-1/3)/π of its colour's rows (Scott's bandwidth for
a Gaussian cloud, but local), and their density along each axis in strips above and on the right; a switch adds the
rows as points. The URL hash keeps tab and dataset, on the quality tab the cell, on the cells tab class and node, and
a τ other than the run's (`?tau=0.15` after them); a link to the former clean and prepare tabs opens data.

Layout:
- `main.py`: the CLI, one stage per call inside a logged run
- `stages/`: one file per stage, `run(cfg, conn, run_id)`; `stages/paths.py` names every file a stage writes
- `src/`: config and DB logging, then the method on arrays and frames: data, prepare, graph, cells, quality,
  provenance, and report (the verdicts at a tau, from what quality_report measured)
- `dashboard/`: FastAPI `app.py`, read-only `queries.py`, `static/` (one page, hand-made SVG charts)
- `generate_synthetic.py`: the synthetic dataset, a ladder of sub-groups per class that overlap their rivals by
  known margins (the `true_subgroup` column records it; no stage reads it), and the provenance of every row, with two
  planted artefacts: class_5 from a single host (typical of it: no `atypical_provenance`), the canonical sub-group of
  class_4 from one host on one source port within 5 minutes (its cells, and in class_4 only they, have
  `atypical_provenance` on `src_ip`)
- `tests/`: the suite; the session fixture runs every stage once on a 12,000-row synthetic CSV

To add:
- a dataset: put the CSV under `resources/raw_data/`, copy a yaml of `configs/datasets/`, set `name`, `path`, the
  columns and `output_dir`, and pass its name to `--dataset`.
- a measure against a rival class: a function of `src/quality.py` called in `_measure` of
  `stages/quality_report.py`, logged in `_rival_row` on the cell's side.
- a defect: its name in `DEFECTS` (`src/quality.py`), its verdict (a function of `src/quality.py` that reads the
  measure with its interval, `verdict`) in `rival_defects` if it is measured against a rival class, else in `_entries`
  of `src/report.py`, and its line in the table above: one definition, and tau as its threshold if it needs one.

Known limits: every rerun of a stage overwrites its files, so the files hold only the latest run of each stage, and
nothing records which run wrote a stage's inputs: after changing the method or rerunning a stage, rerun everything
after it (`make run`). Two method yamls on one dataset share its `output_dir`. A failed rerun can leave files that
disagree with the latest `ok` run the dashboard shows. A run killed by anything but Ctrl-C stays `running`. The k-NN
searches use every core. Under `distance: cosine` the numbers are scaled to unit length on every column, so removing
a key column subtracts its part of a distance measured with it: the shortcut is then an approximation. A run logged
before cells were measured against rival classes (a `quality_pairs` table) or without a confidence must be run again
for the dashboard. The excess of a cell is the highest over its values, read with the interval of that value only:
over many cells and keys a few cells pass tau by chance. The time parts of a class are as many as its cells: a part
of the class recorded at once but larger than a cell spans several parts, and its cells are not atypical in time.
