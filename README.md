# grace

grace finds the defects of a labelled tabular dataset (mostly network intrusion detection datasets) without training
any model. It cuts each class into small cells of nearby rows, checks every cell for eight defects (after the data
design smells of Flood et al., EuroS&P 2024), and merges the cells of a class into regions that share the same
diagnosis. You read the report in a dashboard.

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

provenance:               # optional: where each row comes from (never features, except dst_port)
  src_ip: Src IP
  src_port: Src Port
  dst_ip: Dst IP
  dst_port: Dst Port
  time: Timestamp
  time_format: "%Y-%m-%d %H:%M:%S%.f"

scenario:                 # optional, needs provenance: what the dataset's documentation says
  benign: BENIGN                    # the background class
  background_ports: [53, 123]       # ports no attack targets
  attacks:
    DoS:
      src_ip: [10.0.0.1, 10.0.0.2]                            # the attacker hosts
      time: [["2024-01-01 10:00:00", "2024-01-01 11:00:00"]]  # windows, in the CSV's clock
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
higher confidence asks for larger cells. In the dashboard you can change tau after the run without rerunning anything.

## How it works

| stage | what it does | writes |
|---|---|---|
| `clean` | reads the CSV, keeps one copy of each row and counts its copies, drops rare classes and constant columns, samples classes over `max_class_rows` | `data.parquet`, `provenance.parquet` |
| `prepare` | numbers to sign(x)·log1p(\|x\|), then standardized; categoricals to weighted one-hot | `prepared.parquet` |
| `cells` | a Ward tree per class, split while a node holds at least twice the cell size | `cells.parquet`, `assignments.parquet`, `cell_tree.parquet` |
| `quality_report` | measures every cell against the other classes, then judges the defects at tau and merges the cells into regions | tables in the DB |

A row is confused with another class when its nearest row in the dataset belongs to that class. A cell's
`rival_share` against a class is the share of its rows confused with it. Every share is read with its interval: a
cell has a defect (yes), has it not (no), or is undecided when the interval straddles the threshold.

Regions: the cells of a class merge up their tree as long as no cell has a defect another one rules out. An undecided
cell never splits a region.

## Defects

| defect | a cell has it when | why it matters |
|---|---|---|
| `invalid_values` | a row holds an empty, unparsable, NaN or infinite value | by definition |
| `duplicates` | a row appears more than once in the CSV | copies can fall on both sides of a train/test split |
| `label_conflict` | a row equals a row of another class on every feature | no classifier can tell them apart |
| `suspect_label` | its `rival_share` against a class is at least one half | the cell lies inside the other class |
| `class_overlap` | its `rival_share` is at least tau (and it is not a `suspect_label`) | part of it is mixed with the other class |
| `shortcut` | a one-column rule of the two classes puts less than tau of each on the wrong side | one column alone separates them: likely a capture artefact |
| `repetition` | it lies in a branch of its class holding at least tau of the class, with at most tau times the dataset's variance | a large part of the class repeated in little space |
| `off_scenario` | at least tau of its rows break the scenario (wrong host, time window or a background port) | traffic the documentation does not describe under that label; needs a `scenario` |

The first three are facts of single rows and are counted exactly. The others are judged per cell.

## Results

- **Dashboard** (`make dashboard`): three tabs for the datasets of `configs/datasets/`. *quality*: per class the share
  of rows with each defect, a τ slider, and per region the evidence of each defect. In this tab a "cell" is a
  region. *cells*: the tree of each class and a map of its leaves (the stage's cells). *data*: what clean and prepare
  kept and dropped. Each tab shows the latest successful run and the parameters it used.
- **Files**: the parquet files of each dataset in its `output_dir`. A row's position in `data.parquet` is its id in
  every other file.
- **Database**: every run in one SQLite file (`db_path`). `runs` lists every stage run with its config and status.
  The verdicts are in `quality_cells` (per cell), `quality_regions`, `quality_classes` and `quality_stats` (the whole
  dataset). Every table carries a `run_id`.

## Extending

- A measure against another class: a function in `src/quality.py`, called in `_measure` of
  `stages/quality_report.py`. A measure must not depend on tau: tau is applied only in `src/report.judge`.
- A defect: its name in `DEFECTS` (`src/quality.py`), its verdict in `rival_defects` (against another class) or in
  `_entries` of `src/report.py`, and a line in the table above.

## Limits

- A rerun overwrites the files of its stage, and nothing records which run wrote a stage's inputs.
- Two method yamls on the same dataset share its `output_dir`.
- A run killed by anything but Ctrl-C stays `running` in the DB.
- The nearest-row searches use every core; a large class needs memory (cic_2018 samples Benign to 10M rows on 16 GB).
- Across many columns, a few may pass as a `shortcut` by chance.
- `off_scenario` is only as good as the documentation the scenario is copied from.
