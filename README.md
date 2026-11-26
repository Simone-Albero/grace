# grace

grace finds the defects of a labelled tabular dataset (mostly network intrusion detection datasets) without training
any model. It cuts each class into a tree of small cells of nearby rows, judges every branch of the tree for seven
defects (after the data design smells of Flood et al., EuroS&P 2024), and merges the cells of a class into regions:
the largest branches whose verdicts stay those of their parts. You read the report in a dashboard.

## Install

Python 3.12. Run every command from the repo root: the paths in the configs are relative to it.

```
python3.12 -m venv venv
venv/bin/pip install -r requirements.txt
```

## Run

```
make run DATA=cic_2018        # the four stages in order
make dashboard                # http://127.0.0.1:8000 (PORT=<n> to change it)
```

`DATA` is a name of `configs/datasets/` or the path of a yaml. `METHOD=<name|yaml>` picks another method
(default `configs/method/default.yaml`). One stage at a time:

```
python main.py --stage clean          --dataset cic_2018
python main.py --stage prepare        --dataset cic_2018
python main.py --stage cells          --dataset cic_2018
python main.py --stage quality_report --dataset cic_2018
```

Each stage reads what the one before it wrote. After you change a config or rerun a stage, rerun every stage after it.

## Add a dataset

1. Put the CSV under `resources/raw_data/` (not tracked by git).
2. Write `configs/datasets/<name>.yaml`:

```yaml
name: my_dataset
path: resources/raw_data/my_dataset/*.csv  # one file or a glob, read as one dataset
label_col: Label
num_cols: [Flow Duration, Total Fwd Packet]  # numeric features
cat_cols: [Dst Port, Protocol]               # categorical features
min_class_count: 50       # classes with fewer distinct rows are dropped
max_class_rows: 10000000  # optional: larger classes are sampled down to this many distinct rows
top_n: 32                 # categorical values with their own one-hot slot; the rest share one
seed: 42
output_dir: resources/processed/my_dataset
db_path: resources/processed/grace.db

significant_digits: 10    # optional: round every number to 10 significant digits as it is read, so that files
                          #   writing the same value with more or fewer digits give the same row

provenance:               # optional: where each row comes from (never features, except dst_port)
  src_ip: Src IP          # a column missing from a file is read as null there
  src_port: Src Port
  dst_ip: Dst IP
  dst_port: Dst Port
  time: Timestamp
  time_format: "%Y-%m-%d %H:%M:%S%.f"
  time_afternoon_before: 8  # optional: times on 12 hours without AM/PM; an hour under 8 gets 12 hours
```

3. `make run DATA=my_dataset`.

The yamls in `configs/datasets/` are further examples.

## Method parameters

`configs/method/default.yaml`, shared by every dataset:

| key | default | meaning |
|---|---|---|
| `quality.tau` | 0.1 | the one threshold of the report (see the table below) |
| `quality.confidence` | 0.95 | level of the Wilson interval every share of a cell is read with |
| `cells.max_fit_samples` | 20000 | most rows of a class the Ward tree is built on (a bound of computation) |

tau and the confidence also set the size of the cells, z² / (4 tau²) rows: 68 at the defaults. A smaller tau or a
higher confidence asks for larger cells, so another tau means running `cells` and `quality_report` again. The
dashboard shows the verdicts of the run, at its tau.

## How it works

| stage | what it does | writes |
|---|---|---|
| `clean` | reads the CSV, keeps one copy of each row and counts its copies, drops rare classes and constant columns, samples classes over `max_class_rows` | `data.parquet`, `provenance.parquet` |
| `prepare` | numbers to sign(x)·log1p(\|x\|), then standardized; categoricals to weighted one-hot | `prepared.parquet` |
| `cells` | a Ward tree per class, split while a node holds at least twice the cell size | `cells.parquet`, `assignments.parquet`, `cell_tree.parquet` |
| `quality_report` | measures every cell and every branch of the trees, judges every branch at tau and merges the cells into regions | tables in the DB |

A row is confused with another class when its nearest row in the dataset belongs to that class. The `rival_share`
of a group of rows against a class is the share of its rows confused with it. Every share is read with its interval
on the group's rows: the group has a defect (yes), has it not (no), or is undecided when the interval straddles the
threshold.

Regions: a branch of a class's tree merges its two children when every verdict they decide (yes or no) stays the same
on their rows together; a region is the highest branch merged, and its verdicts are read on all its rows. An
undecided verdict never stops a merge.

Provenance, when the dataset names it, supports a verdict and never decides one: for each key and for the time, how
often a row shares it with its nearest rival row, against its nearest row of its own class (the control) and chance,
and the hosts of each region.

## Defects

| family | defect | a region has it when | why it matters |
|---|---|---|---|
| structural | `invalid_values` | at least tau of its rows hold an empty, unparsable, NaN or infinite value | by definition |
| structural | `duplicates` | at least tau of its CSV rows repeat a row (the interval on its distinct rows) | copies can fall on both sides of a train/test split |
| structural | `label_conflict` | at least tau of its rows equal a row of another class on every feature | no classifier can tell them apart |
| overlap | `minority_overlap` | its `rival_share` against a class is at least one half | its class is the minority where it lies: wrong labels, or a true minority |
| overlap | `majority_overlap` | its `rival_share` is at least tau, and it has no `minority_overlap` against that class | part of it is mixed with the other class |
| generalization | `single_feature_cut` | under tau of its rival class's rows fall inside its values on one column | one column alone separates them: an artefact or a legitimate signal |
| generalization | `low_variability` | it lies in a branch holding at least tau of its class's CSV rows, no more spread than the median cell | near copies of one flow: a model sees the same rows in training and test |

The names describe the geometry, not the cause: whether a `minority_overlap` is a wrong label or a true minority,
whether a cut is an artefact, the provenance and the domain say.

## Results

The dashboard (`make dashboard`) has three tabs for the datasets of `configs/datasets/`. The quality tab shows per
class the share of rows with each defect, and per region the evidence of each defect; in this tab a "cell" is a
region. The cells tab shows the tree of each class and a map of its leaves (the stage's cells). The data
tab shows what clean and prepare kept and dropped. Each tab shows the latest successful run and the parameters it
used.

The parquet files of each dataset are in its `output_dir`. A row's position in `data.parquet` is its id in every other
file.

The database is one SQLite file (`db_path`) holding every run. `runs` lists every stage run with its config and
status. The measures are in `quality_cells` (per cell), `quality_rivals` (per cell and rival class) and
`quality_branches` (per branch); the verdicts in `quality_regions`, `quality_classes` and `quality_stats` (the whole
dataset). Every table carries a `run_id`.

## Extending

- A measure: a function in `src/quality.py`, called in `stages/quality_report.py`. A measure of a cell that adds up
  goes in `quality_cells` (`src/report.judge` sums it up the tree); one that does not is measured on every branch, in
  `quality_branches`. A measure must not depend on tau: tau is applied only in `src/report.judge`.
- A defect: its name in a family of `src/quality.py` (`DEFECTS`), its verdict in `node_verdicts`, its columns in the
  region rows of `src/report.judge`, and a line in the table above.

## Limits

- A rerun overwrites the files of its stage, and nothing records which run wrote a stage's inputs.
- Two method yamls on the same dataset share its `output_dir`.
- A run killed by anything but Ctrl-C stays `running` in the DB.
- The nearest-row searches use every core; a large class needs memory (cic_2018 samples Benign to 10M rows on 16 GB).
- Every share treats a region's distinct rows as independent trials; rows that share their nearest rival row are
  not quite, so the intervals are a little narrower than true. The copies of a row count once.
- `single_feature_cut` reads a column's [min, max]: one extreme row widens it, so it reports less, never more.
- `low_variability` has no interval, and its reference, the median cell, depends on tau and on the classes of the
  dataset. In a class of fewer than 1/tau cells one cell is a large branch.
- `cic_2018_broken`: the IPs are in one file only, and the times are on 12 hours, read with `time_afternoon_before`
  (the rows at 20:xx local read as 08:xx).
