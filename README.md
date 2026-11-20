# grace

Data quality through the geometry of the data, independent of any model: each class is cut into the smallest cells a
measure makes sense on, every cell is checked for the defects of a dataset after the data design smells of Flood et
al. (EuroS&P 2024) and diagnosed by its relation to other cells (against another class: a suspect label, class
overlap or a shortcut; within a large and tight branch of its class: repetition; against the scenario the dataset's
documentation declares: off_scenario), and the cells
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
  one) their own and the rest a shared "other", then a one-hot slot per code. Its active slot weighs sqrt(1 / g),
  g = 1 - Σ p² over the shares of its slots (the chance two rows differ on it): on average over pairs of rows a
  categorical column then adds what a standardized number adds, and a mismatch on a concentrated column costs more. Writes `prepared.parquet` (the prepared columns, Float64, then the
  class code; same rows as `data.parquet`).
- `cells`: builds the points of the space (`graph.space_points`): the prepared numbers, then the categorical slots,
  `n_missing` left out. Every distance from there on, here and in `quality_report`, is euclidean on them. Then cuts each class on its own, by geometry alone, into cells
  (`src/cells.py`):
  - a Ward tree of a uniform sample of its distinct rows (at most `cells.max_fit_samples`), each counted once whatever
    its copies (they are a defect of their own, `duplicates`, and no measure of a cell weighs them); every row of the
    class joins its nearest sampled row, so each node of the tree holds a known number of the class's rows;
  - from the root, a node is split into its two children while it holds at least 2 · `min_rows` rows, until every
    node left is a cell; a child under `min_rows` rows is no cell of its own, its rows join the nearest cell. A cell
    so holds from `min_rows` to under twice it, but for the small branches it gathers. `min_rows` is no key of the
    method: it is what `quality.tau` and `quality.confidence` ask for, the rows on which a share tau away from a
    threshold of the report, tau or one half, is decided against it (`quality.resolution_rows`): a cell is then
    undecided only within tau of a threshold. With the normal interval that is z² s (1 - s) / tau² for a share s,
    at most z² / (4 tau²) near one half, where a share is the least certain: 68 rows at tau 0.1 and 0.95, 31 at
    tau 0.15, 271 at tau 0.05. A smaller tau or a higher confidence so asks for larger cells. On the cells of
    cic_2018_broken_poc read against the labels of the corrected release, 68 rows find the relabelled rows of
    Infilteration as well as 50 and better than 25, whose cells leave more of them undecided;
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
  of the regions the cells are merged into. Every distinct row is looked at once against the whole dataset
  (`quality.nearest_rivals`, euclidean and exact, a kd-tree per class): its nearest row of another class, and its
  nearest row of its own class, itself left out. The row is confused with the class of its nearest rival row when that
  row is nearer than its nearest own row, the leave-one-out 1-NN on every column and every row; at the same distance
  it counts one half. The rival classes of a cell are the classes of its rows' nearest rival rows, its rival cells the
  cells those rows lie in: no threshold picks them. Against each rival class (`quality.rival_shares`), read on the
  cell's side:
  - `rival_share`: the share of the cell's distinct rows confused with that class. The rival class is whole, so rows
    confused with a class spread over several of its cells are counted together, and no row is sampled away on
    either side. The shares of a cell over its rival classes sum to at most one. (It replaces the confusion of the
    earlier runs, the better of the 1-NN and of a one-column rule chosen on the cell, on a sample of 500 rows of its
    rival cells: on cic_2018_broken_poc, Infilteration against Benign, the rule chosen on a compact cell of about 90
    rows separated it in 73 cells of 102, and the confusion read 0.21 where the 1-NN on the two classes whole reads 0.72
    and the nearest neighbours test of Flood et al. 0.78.)
  - `rules`: per column, the rule of the two classes (`quality.class_rule`, on all the distinct rows of the cell's class
    and of the rival class: a threshold either way of highest balanced accuracy, exact, or each categorical value to
    the side where it is the larger share), read on the cell (`quality.rule_errors`): the share of the cell's rows it
    sends to the rival's side and the share of the rival class's rows it sends to the cell's, kept when both are under
    one half (`n_rival_rows`, the distinct rows of the rival class, is what the second is a share of). The rule is that
    of the classes, not of the cell: a cell is small and compact, and on some of many columns it lies in a tail of any
    class, so a rule chosen for it alone separates nearly every cell (on cic_2018_broken, 99% of the pairs apart had
    one, 50% ten). This is the single-feature efficacy of Flood et al., per cell and against every rival class,
    without a model.

  Every share of a cell's rows is an estimate on few rows, so it is read with its interval at `quality.confidence`
  (0.95, each bound one-sided, Wilson's). Against a threshold a cell has the defect when the interval lies at or past
  it, has it not when the interval lies short of it, and is undecided otherwise (`undecided`). A small cell is so
  judged only on strong evidence, at any granularity.

  A cell has a defect when (`quality.tau` = 0.1, the threshold the run logs its verdicts at; see "tau, after the run"
  below):

  | defect | relation | when | why |
  |---|---|---|---|
  | `invalid_values` | the row | a row holds an empty, unparsable, NaN or infinite cell | by definition |
  | `duplicates` | the row | a row comes more than once in the CSV | copies can fall on both sides of a train/test split |
  | `label_conflict` | the row | a row equals a row of another class on every feature | no classifier can tell them apart |
  | `suspect_label` | cell and rival class | its `rival_share` is at least one half | the cell lies inside the other class, as rows under a wrong label would |
  | `class_overlap` | cell and rival class | its `rival_share` is tau or more, and it is no `suspect_label` | the nearest row of a share of its rows is of the other class |
  | `shortcut` | cell and rival class | the rule of its class and the rival class on one column leaves under tau of its rows and under tau of the rival class's rows on the wrong side | one column alone tells the two classes apart where the cell lies, an artefact of the capture rather than of the attack |
  | `repetition` | cell and its class's tree | it lies in a branch of its class's tree that holds at least tau of the class's CSV rows and whose variance is at most tau times the dataset's (`quality.dataset_variance`, the mean squared distance of every distinct row to their centroid; `quality.branch_spread`, `quality.repetition_verdicts`) | a large part of its class, the class or one of its modes, repeated in little space: the repetitiveness of Flood et al. |
  | `off_scenario` | cell and the scenario | at least tau of its judged CSV rows break the rule of its class on some key of the scenario (`dst_port`, `src_ip`, `time`) | traffic the documentation does not describe under that label |

  The first three are facts of single rows: counted in a group of any size, they are never diluted, and need no
  interval. The next three compare a cell with a rival class (a rule is a shortcut when the upper bounds of both its
  errors lie under tau, none when the lower bound of either reaches it, whatever the rival share: a cell may overlap
  a class and still lie where one column tells the two apart); the last two with a reference of its own: the
  dataset, the scenario. A dataset without a scenario does not measure `off_scenario` (its shares are
  null). One half, the majority, is the other threshold of the report: it is no choice.

  Repetitiveness depends on the scale it is looked at: at the scale of a cell every cell is a dense spot, so a cell's
  variance measures the density of its neighbourhood, not whether its class repeats itself; at the scale of a class,
  a class of two tight modes far apart (HOIC in cic_2018_broken) looks spread around a centre between them. The tree
  of a class holds every scale in between, so `repetition` reads its branches: a branch is large when it holds at
  least tau of its class's CSV rows, and tight when its variance is at most tau times the dataset's; a cell has
  repetition when a large branch above it is tight (`repetition_branch`, the largest). It is the one verdict read
  without an interval: a large branch holds many rows, and its variance, a mean of squared distances, is no share.
  The variance of a branch is exact, from the sums of its cells (rows, sum, squared norms) added up the tree.

  The cells of a class are then merged into regions up their tree: a region is the largest branch whose cells do not
  contradict each other, no defect against one target (a class, or the cell's own reference) being established in one cell and ruled out in
  another (a cell with no rival cell of a class rules out the rival defects against it); an undecided cell never
  splits a branch. A region is named by the node of the tree at its top (so the same branch has the same name at any
  tau), has the defects its cells have, and its row defects are counted over its rows. Measuring the merged regions
  instead would dilute again, so the regions only gather what the cells found. Per cell (`quality_cells`) and per
  region (`quality_regions`): the copies, the rows in a label conflict and the classes they conflict with, the rows
  with a null or invalid value by kind, the classes of `suspect_label` (`label_with`), `class_overlap` and `shortcut`,
  the highest `rival_share`, its interval and that rival class, and the `defects`; per cell also its rival cells (`n_rivals`), the
  widest branch of repetition it lies in, its share of its class and its variance over the dataset's
  (`repetition_branch`, `repetition_share`, `repetition_ratio`; per region its widest cell's), the rival class of its
  shortcut whose best rule errs least (`shortcut_class`), the columns whose rule is a shortcut
  against it (`shortcut_columns`, the best first) and the worse error of the best (`shortcut_error`), and the defects
  it is `undecided` on. Per class
  (`quality_classes`) and over the dataset (`quality_stats`): the share of the CSV rows with each defect and with none,
  and per defect the cells undecided on it. A row has `invalid_values`, `duplicates` (all copies of a row but the first)
  and `label_conflict` itself, the others through its cell: a region flagged for one conflicting row does not count its
  other rows as conflicting.

  Provenance (`src/provenance.py`), when the dataset names its columns: per cell, over its CSV rows (each copy with its
  own, an empty value left out), for each of `src_ip`, `src_port`, `dst_ip` and `dst_port`, the most frequent value and
  its share (`<key>_top`, `<key>_top_share`), the distinct values (`<key>_n_values`) and the effective number
  (`<key>_eff_n`, exp of the Shannon entropy of their shares, Hill's number of order 1), a description. With a
  scenario, per key it rules on, the share of the cell's judged CSV rows that break it (`scenario_<key>`, null with
  none judged) and those rows (`scenario_<key>_n`), and the same on any key (`scenario`, `scenario_n`,
  `provenance.any_breach`: a row breaks the scenario when it breaks a key, and is judged when a key judges it), the
  one `off_scenario` reads; `provenance.scenario_breaches`: a row of the benign class breaks
  nothing; a row of another class breaks `dst_port` when its destination is a background port, `src_ip` when its
  source is none of its class's hosts, `time` when its time lies in none of its class's windows (closed). A key the
  scenario does not rule on for a class, or a value unknown, leaves the row unjudged.

Configs:
- `configs/datasets/`: one yaml per dataset: `name`, `path`, `label_col`, `num_cols`, `cat_cols`,
  `min_class_count`, `top_n`, `seed`, `output_dir`, `db_path`, optionally `max_class_rows` (the most distinct rows a
  class keeps, null or absent for all; at least `min_class_count`), optionally `provenance`: the columns of where each
  row comes from (`src_ip`, `src_port`, `dst_ip`, `dst_port`, `time` with its `time_format`, the chrono format polars
  reads, e.g. `"%Y-%m-%d %H:%M:%S%.f"`; each optional, a missing one fails clean as a missing feature does; the source
  and the time are never features, a column named both fails the config; the destination may be a feature too: its
  port names the service an attack targets, as DataBadSmells reads it), and optionally `scenario`: what the dataset's
  documentation says of its classes (`benign`, the class of the background traffic; `background_ports`, the ports of
  background services, which needs `provenance.dst_port`; `attacks`, per class `src_ip`, its hosts, and `time`, its
  windows `[start, end]` in ISO and in the clock of the CSV's time column, each needing its provenance column). The
  scenario is written per dataset, in its own values: `cic_2018_broken` writes its times on 12 hours without AM/PM, so
  it declares the ports only; a class it names that the data lacks fails quality_report. The NetFlow v2 datasets have
  no time column. Four network datasets (`nb15_v2`,
  `cic_2018_v2`, `ton_iot_v2`, `bot_iot_v2`), five general benchmarks (`bank_marketing`, `covertype`,
  `letter_recognition`, `statlog_landsat_satellite`, `thyroid_disease`) and `synthetic_test`; their CSVs are not in
  the repo.
- `configs/method/default.yaml`: the method, shared by every dataset: `cells` (`max_fit_samples`, a bound of
  computation) and `quality` (`tau` and `confidence`), each key commented. The size of a cell is no key of it: `quality.tau` and
  `quality.confidence` set it.
  Another method is a copy of it passed as `--method <path>`.

`src/config.py` loads both into frozen dataclasses and validates them before any stage reads data.

Outputs: the files of a dataset in its `output_dir` (`resources/processed/<dataset>/`), every dataset's runs in one
SQLite file, `db_path` (`resources/processed/grace.db`). Table `runs` holds one row per stage run (stage, dataset, the
whole config as JSON, status `running`, `ok` or `failed`, start and end); every other table carries the `run_id` of the
run that wrote it and is created by its first insert: `clean_summary` (with the null and invalid cells read, by
kind), `dropped_classes`, `dropped_columns`,
`class_counts`, `features`, `sources`, `provenance_columns` (clean); `prepare_summary`, `prepared_columns` (prepare); `cells`, `cell_tree` (the cut of
each class, each node with the distinct and CSV rows below it), `class_cells`, `cell_stats` (cells); `quality_branches`
(the cut of each class again, each node with its distinct and CSV rows and the variance of its rows), `quality_rivals`
(every cell against each rival class: its rival cells of that class, its `rival_share`, the distinct rows of the rival
class and the rules of the two classes), `quality_cells`, `quality_regions`, `quality_classes`, `cell_missing` (the null and
invalid cells by cell, column and kind), `label_conflict_groups` (every row in a label conflict, with its group),
`quality_stats` (with tau and the confidence, the measures against a rival class, the cells undecided on each defect, and the share of the CSV rows with each
defect and with none; `off_scenario` null without a scenario)
(quality_report). There are no migrations: a new column is added to its table, an old one stays, empty for the new
runs; to drop the old ones, delete the DB.

**tau, after the run.** What the run measures does not depend on tau: the `rival_share` of each cell against each
rival class, and the errors of the rules of the two classes on every column
(`quality_rivals`), the variance of each branch of the tree of each class (`quality_branches`, which `judge` takes for the tree) and of the dataset, the share of
its CSV rows that break the scenario on any key, the counts, and
`clear_rows` per cell (its distinct rows with no null or invalid value and in no label conflict, which the share with
no defect needs). tau and the confidence only compare them. `src/report.py` (`judge`) turns these measures into the
verdicts at any tau: the defects of the cells, the regions they merge into, the shares per class and over the dataset.
The stage calls it once, at `quality.tau` and `quality.confidence`, and logs the result (`quality_rivals.region` and
`rival_regions`, `quality_cells` from `region` and the class lists on, `quality_regions`, `quality_classes`, the
shares and undecided counts of `quality_stats`); the dashboard calls it per request, with the `tau` of the request
and the run's confidence. At the run's tau it gives the logged tables, and a test checks it. A quality_report run
older than the last cells run is not judged (its cells are not those of the tree): the dashboard asks to run it
again.

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
  dataset; `/quality/profile`) and the most frequent categorical values; for `shortcut`, per class, the columns whose
  rule is under τ on both sides, with their error on the cell and on the rival class; for `repetition`, the widest
  branch it lies in, its share of the class and its variance over the dataset's; for `off_scenario`, the share of its
  CSV rows off the scenario, on any key and per key. A rival share is shown with its interval. Then a map of the cell and the rival
  leaves of the classes picked (`/quality/map`), and, closed, its rival classes (the rival share and the best
  one-column rule) and its provenance (per column its distinct values, the most frequent and the CSV rows per
  repeated value; the first and last time). A rival class is measured by its leaf of highest rival share
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
  planted artefacts: class_5 from a single host, the canonical sub-group of class_4 from one host on one source port
  within 5 minutes; `synthetic_test.yaml` declares the scenario (class_4's hosts and hour), so the cells of that
  sub-group, and in class_4 only they, have `off_scenario` on `src_ip`
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
searches use every core. A run logged before cells were measured against rival classes (a
`quality_pairs` table), without a confidence, without the rules of a shortcut or before `off_scenario` was read on
any key (the `scenario` column) must be run again for the dashboard.
The rule of two classes is chosen on all their rows, the cell's among them: for a class of a single cell it is chosen
on the cell, and its error there is optimistic. A cell has a shortcut when some column passes, read with its own
interval: over many columns a few pass by chance. The scenario is only as good as the documentation it is copied
from.
