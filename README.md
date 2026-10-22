# grace

A framework for finding out **where** a trained classifier fails, and **why**, without needing labelled test data for the regions in question.

It divides a dataset into regions, describes each region by how it sits relative to the classes competing with it, and learns how those descriptions relate to the mistakes a classifier actually made. The result is an error-rate estimate for any region, including regions the framework has never seen, together with a ranking of the data properties that drive the risk.

The estimate belongs to the model you point it at. It is fitted on that model's own errors, so it describes how *that* classifier copes with the shape of your data rather than how difficult the data is in the abstract. This makes it useful for identifying unreliable regions before you have the labels to prove they are unreliable, for deciding where to gather more data or review labels, and for monitoring a model after deployment.

Everything is tabular and everything is driven by configuration: ten classifiers (scikit-learn, XGBoost, PyTorch), four clustering algorithms, two distances, configurations for the four public network-security datasets and five general benchmarks, and a synthetic dataset that exercises the whole pipeline.

## Quickstart

Python 3.12 on Linux or macOS (run locks use `fcntl`). Run every command from the repository root: the paths in the configuration are relative to it.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

make generate                     # resources/raw_data/synthetic/synthetic_test.csv (ROWS=<n> for fewer rows)
python -m stages render render_regress data=synthetic_test classifier=random_forest
make dashboard                    # http://127.0.0.1:8000
```

On a remote machine, forward the port first: `ssh -L 8000:127.0.0.1:8000 <host>`, then open http://127.0.0.1:8000.

The default `data` in [configs/config.yaml](configs/config.yaml) is `cic_2018_v2` (a 3.2 GB CSV), so always pass `data=`. `resources/` is not tracked by git: a fresh clone carries no data, and the synthetic generator is the only dataset that works out of the box.

## Stages and DAG

`python -m stages <target…>` runs the targets and every stage they read. Each stage is one module in [stages/](stages/) that declares `CONFIG` (the config keys its outputs depend on; `config_keys(cfg)` instead when they depend on the config, as in `classify`: ML or DL), `INPUTS` (the stages it reads), `FILES` (raw files whose size and mtime count) and `OUTPUTS` (the files it must write), reads only its inputs' artifacts and DB rows, and writes only its own.

```
split → prepare → regions → classify → failures → baselines → render
                     └→ complexity → regress → render_regress
```

`render` needs no complexity and no regressor, so a run can stop after the baselines.

| Stage | Reads | Config keys | Files written | DB tables written |
|---|---|---|---|---|
| `split` | | `data.{path,label_col,num_cols,cat_cols,min_class_count,train_frac,val_frac,test_frac}`, `seed` | `{train,val,test}.parquet` | `features`, `metrics` |
| `prepare` | split | `data.top_n`, `data.hash_buckets` | `{train,val,test}.parquet`, `preprocessor.joblib` | `classes` |
| `regions` | split, prepare | `space`, `graph`, `clustering`, `distance`, `seed` | `space.joblib`, `graph.npz`, `centroids.parquet`, `assignments.parquet` | `regions`, `class_clusterings`, `metrics` |
| `classify` | split, prepare | `classifier`, `fit.balance`, `fit.n_samples`, `seed`, plus `grid_search` (ML) or `loss`, `optimizer`, `scheduler`, `fit.training`, `fit.validation`, `data.top_n`, `data.hash_buckets` (DL) | `predictions.parquet`, `model/` | `class_metrics`, `classifier_grid`, `history`, `metrics` |
| `failures` | prepare, regions, classify | none | none | `failures`, `metrics` |
| `baselines` | prepare, regions, classify, failures | `failure_regressor.min_eval_support` | none | `baselines`, `metrics` |
| `render` | prepare, regions, classify, failures, baselines | `figure_format` | `figures/` | `figures`, `metrics` |
| `complexity` | prepare, regions | `complexity`, `graph.k`, `distance`, `seed` | none | `complexity`, `metrics` |
| `regress` | regions, complexity, failures, baselines | `failure_regressor`, `seed` | `model.joblib` | `estimates`, `regressor_folds`, `importances`, `metrics` |
| `render_regress` | regions, complexity, failures, baselines, regress | `figure_format` | `figures/` | `figures`, `metrics` |

- **split** drops rows with missing or infinite values, drops the classes with fewer than `min_class_count` rows, and splits the rest stratified by class into train, validation and test.
- **prepare** is preprocessing only. A signed log and a robust scaler for the numerical columns, a top-N plus hash encoder for the categorical ones, and the integer class labels are all fitted on train and applied to the three splits.
- **regions** builds the euclidean space regions live in (the prepared numerical features plus a one-hot block for the `space.top_k` most frequent codes of each categorical feature), a k-nearest-neighbour graph of a uniform sample of train, and clusters each class of train on its own. The granularity (number of clusters, or the density-based parameters) is the candidate of the clustering grid with the least loss: the hardness spread a region's mean hides plus the sampling noise of its test failure rate (see [Method](#method)); the hardness of a row being the share of its `clustering.hardness_k` nearest training neighbours that belong to another class (kDN). Undersized clusters and noise points merge into the nearest region of the same class. Validation and test rows are then routed to the nearest region centre **without their labels**.
- **classify** trains one classifier on train (undersampled when `fit.balance` says so) and predicts validation and test. An ML classifier with a `grid` is tuned by cross-validated grid search and refit with the winner; the MLP trains with early stopping on validation.
- **failures** counts the classifier's errors on the test rows routed to each region. Failure rates are measured on test only.
- **baselines** predicts each region's failure rate from the classifier's probabilities alone: MCP (the mean of 1 minus the maximum class probability of the region's test rows) and ATC (the share of the region's test rows under a confidence threshold fitted on validation). It scores both against the observed rates on the regions with at least `failure_regressor.min_eval_support` test rows (the *used* regions, the ones the regressor is scored on too) and fails if there is none.
- **render** draws the static figures of the classifier and the baselines: confusion matrix, F1 per class, the loss of a DL classifier, MCP and ATC against the observed rates, and the observed failure rate of the regions of each class (box and strip plot, a point's area is its test rows).
- **complexity** describes every region and every class on its train rows (see [Method](#method)).
- **regress** predicts each region's failure rate with a Random Forest on its descriptors and its class's, under nested cross-validation (a randomized search inside each outer fold) and refits the forest on all used regions. Its log sets the scores against those of the baselines, which it reads, not recomputes.
- **render_regress** draws the regressor's figures: predicted against observed rates, the scores of the three predictors, the forest's feature importances and the distribution of the most important descriptors.

## CLI

```
python -m stages <target…> [key=value…] [--sweep key=v1,v2…] [--force]
```

- Targets run with every upstream stage. `group=option` (for example `data=nb15_v2`, `classifier=knn`) picks a YAML file of that group; any other `key=value` overrides a value with a dotted key (`seed=7`, `graph.k=15`).
- `--sweep` runs the cartesian product of its values, the first sweep outermost. Every combination is validated before the first one runs. Values are split on the commas outside square brackets, so a list value can be swept too: `--sweep 'clustering.params.min_samples=[1,5],[15,50]'` (quote it: the shell would expand the brackets).
- `--force` re-runs the named targets only; their upstream stays cached.
- SIGTERM and Ctrl-C mark the run and the running execution `failed`. A run killed outright (SIGKILL, power loss) is marked `failed` at the first of these moments when no other run is live: the next run on the root, the dashboard's start, or the end of a dashboard job that was killed. A live run holds a shared lock on `runs.lock`, and rows still `running` while nobody holds it are dead. A SIGKILL of the CLI can leave joblib (loky) worker processes behind: they finish their queued fits, hold their memory for minutes and a few never exit; when no run is live, `pkill -f loky` removes them. A dashboard job has its own process group, which the dashboard kills whole after the cancel grace period.
- A database error (locked, read-only, full disk, corrupted) ends with a message naming `<root>/grace.db` and, when it is known, what to do (restore a copy, or move it away together with `artifacts/` and `jobs/` to start a new one, wait for the process that locks it, make the root writable, free disk space). A full disk while a stage writes an artifact ends with the OS error.

An execution is reused when a done execution of the same stage has the same key and all its output files exist with the size they were written with (`sizes.json` in its folder, without which the stage runs again; the content is not checked, so a same-size edit goes unnoticed: use `--force <stage>`). The key is the sha256 of the stage's config slice, the ids of the input executions it read, and the size and mtime of the raw CSV (`split`). `runtime` and `root` are in no key. Code changes are not detected: use `--force`.

```bash
python -m stages render render_regress data=synthetic_test classifier=random_forest
python -m stages render data=synthetic_test classifier=knn  # without the regressor
python -m stages render render_regress data=synthetic_test --sweep classifier=knn,xgboost,mlp --sweep distance=euclidean,cosine
python -m stages regions data=synthetic_test clustering=hdbscan distance=cosine
python -m stages regions data=synthetic_test clustering=hdbscan 'clustering.params.min_samples=[5,15]'
python -m stages regions data=synthetic_test clustering=hdbscan --sweep 'clustering.params.min_samples=[1,5],[15,50]'
python -m stages classify data=synthetic_test classifier=xgboost --force
python -m stages render render_regress data=synthetic_test root=resources/other runtime.n_jobs=16
```

## Configuration

[configs/](configs/) holds one folder per group, one YAML per option: `data` (10), `classifier` (10), `clustering` (4), `space`, `graph`, `complexity`, `failure_regressor`, `fit`, `grid_search`, `loss`, `optimizer`, `scheduler`. [configs/config.yaml](configs/config.yaml) chooses the option of each group (`groups`) and holds `seed`, `distance` (`euclidean` or `cosine`), `figure_format` (`pdf` or `png`), `root`, `runtime.*` and the `clustering` keys every algorithm shares (`max_fit_samples`, `max_regions`, `min_cluster_floor`, `hardness_k`). `runtime` says how fast stages compute, never what: `n_jobs` (the classifier grid search, the failure regressor search, the k-NN searches of regions and complexity, and the fit of random_forest, knn and xgboost), `device`, `num_workers` and `pin_memory` (DL only).

`stages/config.py` loads the YAML files, applies the overrides and validates the whole config in one place, so a bad key, type or range of the pipeline's own settings fails before anything runs, with the valid values in the message. Values passed through to a library as they are (`classifier.params` and `grid` values, `optimizer.params`, `scheduler.params`, `clustering.params` values, `failure_regressor.params`) are checked by that library when their stage runs. So are the columns against the CSV (split) and the length of a `loss.params.class_weight` list (classify, which knows the classes). The config is strict: an unknown key fails. Add a new key (including `classifier.params.*`, `clustering.params.*` and `failure_regressor.params.*`, which cannot be introduced from the command line) to a YAML first.

## Outputs

Everything lives under the root (`resources` by default, `root=` to change it):

- `grace.db`: one SQLite database for every configuration.
- `artifacts/<stage>/<execution id>/`: the files of each execution. The paths stored in the DB are relative to the root, so a root can be moved or copied.
- `jobs/<id>.log`: the log of each dashboard job; `jobs/.lock`, held by the running dashboard; `runs.lock`, held (shared) by every live run.

There are no migrations: after a schema change, delete the DB and its artifacts.

| Tables | Content |
|---|---|
| `runs`, `executions`, `run_stages` | bookkeeping: one row per run (one per sweep combination) and per stage execution (config, inputs, status, error, duration, directory); which execution each run used and whether it was cached |
| `metrics`, `features`, `classes` | scalar results of an execution; feature names and kinds; class names and split sizes |
| `regions`, `class_clusterings` | each region's class and sizes per split; per class, the chosen number of clusters and its loss |
| `complexity` | one row per region or class and descriptor |
| `class_metrics`, `classifier_grid`, `history` | per-class precision, recall and F1; the grid-search candidates; the loss of a DL classifier |
| `failures`, `baselines`, `estimates` | per region, test rows, errors and failure rate; whether it is scored, its MCP and its ATC; the out-of-fold Random Forest prediction |
| `regressor_folds`, `importances` | the outer folds (fold -1 is the refit); the mean feature importances |
| `figures` | the figure files of a render or render_regress execution |
| `presets`, `jobs` | dashboard presets and their background jobs |

For example, the Spearman correlation of the three predictors with the observed failure rates, per run:

```sql
SELECT r.id, json_extract(r.config, '$.groups.classifier') AS classifier,
       MAX(CASE WHEN m.name = 'regressor_spearman' THEN m.value END) AS rf,
       MAX(CASE WHEN m.name = 'mcp_spearman' THEN m.value END) AS mcp,
       MAX(CASE WHEN m.name = 'atc_spearman' THEN m.value END) AS atc
FROM runs r
JOIN run_stages s ON s.run_id = r.id AND s.stage IN ('regress', 'baselines')
JOIN metrics m ON m.execution_id = s.execution_id
WHERE r.status = 'done'
GROUP BY r.id;
```

## Dashboard

`make dashboard ROOT=<root> PORT=<port>` serves `<root>/grace.db` (created if missing) on 127.0.0.1, with no authentication: any local user can reach it. Requests must name `127.0.0.1` or `localhost` as host, not `[::1]` (a guard against DNS rebinding) and POSTs must come from its own pages. One dashboard per root: a second one on the same root exits at once with an error (it would close the first one's jobs). It is built with FastAPI, Jinja2, HTMX and Plotly.js from a CDN, so it needs no build step.

- **Results** (`/`, JSON at `/api/results`): one row per done failures execution, with the run that also holds a done regress (else a done baselines, else the latest) and that run's baselines and regress executions; the MCP and ATC columns are empty without baselines and the regressor's without regress. A sweep over `failure_regressor` keys shares one failures execution, so only its newest run is listed (see the Runs page for the others). Filterable by data, classifier, clustering, distance and seed, sortable.
- **Comparisons** (`/compare`): interactive charts of any metric across configurations, and of the three predictors.
- **Runs** and run detail: stage status and cache hits, metrics, the regressor's folds, per-class scores, the figures of both render stages. **Regions** and region detail: a sortable, filterable table, a scatter of any two columns, and the descriptors of one region.
- **Presets**: one `python -m stages` call (targets, overrides, sweeps, force) without the root, validated when saved and again when launched.
- **Jobs**: launching a preset queues a background job. Jobs run one at a time with a live log. Cancelling sends SIGTERM and, after 20 s, SIGKILL; jobs left open by a restart are closed (a job process may outlive a dashboard that was killed: see its log), and a job killed by the SIGKILL has its run and executions marked failed (at the first moment no other run is live; see the CLI section).

## Method

Three steps, together with the classifier whose errors they are fitted on. The method assumes nothing about the classifier, the clustering algorithm or the number of classes; it requires only tabular features.

1. **Identify regions.** Divide each class into compact sub-regions, using the training data alone. A finer partition hides less of the spread of its rows' hardness inside each region's average, but leaves each region fewer of the test rows its failure rate will be measured on, so the rate carries more sampling noise. The candidate with the smallest sum of the two, weighted by region size, is kept.
2. **Describe each region.** Score how separable it is from the rival regions nearest to it, and do the same at class level. These descriptors are drawn from the data alone; nothing about the classifier enters here.
3. **Learn the mapping.** A Random Forest regresses each region's observed failure rate on its descriptors under nested cross-validation, reporting the correlation together with the descriptors that mattered. The tree-based estimator is deliberate: every risk estimate arrives with the properties that produced it. MCP and ATC, which read the classifier's probabilities instead, are the only baselines.

The classifier contributes exactly one thing to step 3, the failure rate it achieved in each region. That is what ties the mapping to the model rather than to the dataset.

- Every distance (clustering, routing, hardness, the neighbour graph and every descriptor) is measured in one space. Under `distance=cosine` every row is scaled to unit length first. The classifier keeps its own encoding of the categorical features: it is the object under study, not part of the geometry.
- Regions are built from the training split alone, each inside one class. Every test row is routed to the region whose centre lies nearest **without reference to its label**, as a row would be at inference time, and a region's failure rate counts the classifier's mistakes on the test rows routed to it, whatever their class. This keeps the correlation a genuine prediction rather than a restatement of labels already known.
- Failure rates are counted on the test split alone: training rows fall back into their own class's region more often than new data would, and would flatter the regions. No test row's features or label shape the classifier, the graph, the regions or the descriptors (only the test split's size enters the granularity loss); the ATC threshold is fitted on validation.
- The analysis works per region rather than per class, since one class usually occupies several separate areas of the space. Class-level descriptors are kept as a coarse reference.

How the data is divided into regions matters: count-based algorithms (`kmeans`, `birch`, `spectral`) suit a continuous difficulty gradient, while `hdbscan` looks for genuine gaps and finds far fewer regions in data without them.

### What the descriptors measure

Five families, all measured in that space on each region's own training rows. The neighbourhood and network families read the training graph: its nodes are a uniform sample of train, the whole split up to `graph.max_samples` rows, joined by their `graph.k` nearest-neighbour graph and a spanning tree over it. Above that size a region is measured on at most `complexity.max_sample_per_region` of its rows and a class on `complexity.max_sample_per_class`. A rival is a region of another class. The feature and neighbourhood families and `network_density` set a region against the `complexity.top_k_clusters` rival regions whose centres lie nearest to it, reported as min / mean / max across them.

| Family | Keys | What it captures |
|---|---|---|
| **F**: feature | `f1`–`f4` | how well individual features separate the region from its rivals |
| **N**: neighbourhood | `n1`–`n4` | how many of a region's neighbours belong to another class |
| **ND**: network | `network_density`, `cls_coef`, `hub` | how many neighbour links cross into a rival region |
| **T**: dimensionality | `t2`–`t4` | how many features there are relative to rows, and how many of them matter |
| **G**: geometry | `max_dispersion`, `p95_dispersion`, `dist_to_nearest_rival`, `p5_silhouette`, `frac_at_risk` | how widely the region is spread, and how close the nearest rival lies |

The regressor sees every key twice, as `region_<key>` and as `class_<key>` for the region's class. Regions with fewer than `failure_regressor.min_eval_support` test rows are left out of its dataset.

### Inside the synthetic dataset

Each class consists of a clean core together with a *difficulty ladder* running towards every class it is meant to overlap with (`_OVERLAPS` in [generate_synthetic.py](stages/generate_synthetic.py)). Every rung sits a fixed distance from the boundary between the two classes, and both sides face one another symmetrically, so the best error rate any classifier could reach on a rung is known in advance:

| Sub-group | Distance from boundary | Best possible error | Rows |
|---|---|---|---|
| `canonical` | none | ≈ 0 % | 23,966 |
| `clear` | 1.80 σ | ≈ 4 % | 19,316 |
| `evasive` | 0.90 σ | ≈ 18 % | 16,560 |
| `mimicry` | 0.25 σ | ≈ 40 % | 9,658 |

The file has 69,500 rows, 10 classes, 20 numerical and 2 categorical features; the 500-row class falls under `min_class_count` and is dropped, leaving 9. The two categorical features fade out along the same ladder, so they cease to help exactly where the numbers begin to overlap. Each row's intended difficulty is recorded in the `true_subgroup` column so that you can check afterwards; the pipeline never reads it, and `split` keeps only the columns the configuration lists. Accuracy settles at about 0.85 by design: the hard rungs are genuinely hard, and that spread is what makes a per-region correlation meaningful.

## Datasets

[configs/data/](configs/data/) has one YAML per dataset: `path`, `label_col`, `num_cols`, `cat_cols`, `train_frac`, `val_frac`, `test_frac`, `min_class_count`, `top_n` and `hash_buckets`. The four network datasets (`nb15_v2`, `cic_2018_v2`, `ton_iot_v2`, `bot_iot_v2`) are read from `resources/raw_data/dataset_v2/`, the five general benchmarks (`bank_marketing`, `covertype`, `letter_recognition`, `statlog_landsat_satellite`, `thyroid_disease`) from `resources/raw_data/<name>/`, and the synthetic one from `resources/raw_data/synthetic/`. The benchmark CSVs are not in the repository.

To add a dataset, place the CSV under `resources/raw_data/`, copy a YAML from `configs/data/`, set `path` and the columns, and run with `data=<name>`.

Every shipped configuration splits 40 % train, 5 % validation, 55 % test. The test split is kept large because it is the only one the regressor's target, every region's failure rate, is measured on. Measured on `unsw_nb15_v2` (2.39 million rows) with random_forest and `runtime.n_jobs=16`: about 6.5 minutes end to end, peak memory about 10 GB.

## Extending

- **A classifier**: an entry in `CLASSIFIERS` in [src/ml.py](src/ml.py) (the estimator and the categorical strategy in front of it: a key of `_STRATEGIES`, or None to pass the prepared frame as it is) and a `configs/classifier/<name>.yaml` with `kind`, `name`, `params` and `grid`. If it takes `n_jobs`, add it to `PARALLEL` in the same file.
- **A DL model**: an entry in `MODELS` in [src/dl.py](src/dl.py), a class whose constructor takes `num_classes`, `num_numerical_features`, `cardinalities` and the YAML `params`, and whose `forward(x_num, x_cat)` returns logits, each row's independent of the others in eval mode (prediction runs in chunks); and a YAML with `kind: dl`.
- **A clustering**: a function `fit(X, *, …, max_fit_samples, random_state) -> labels` (-1 = noise) in `ALGORITHMS` in [src/clustering.py](src/clustering.py) and a `configs/clustering/<name>.yaml` with `name` and `params` (`{}` for none). If it takes `n_clusters`, the pipeline derives it per class from the data and sweeps it; it must not appear in the YAML. List-valued parameters are swept and the candidate of least loss kept.
- **A descriptor**: return one more key from a family function in [src/complexity/](src/complexity/); it reaches the complexity table, the regressor (as `region_<key>` and `class_<key>`) and the dashboard by itself. A new family is one call, merged into the returned dict, in `src/complexity/measure.py` (and, optionally, a label in `src/plot/style.py`).
- **A stage**: a module in `stages/` with `CONFIG` (or `config_keys`), `INPUTS`, `FILES`, `OUTPUTS` and `run(ctx)`; insert it into `STAGES` in [stages/runner.py](stages/runner.py) after its inputs and add its tables to `SCHEMA` in [stages/db.py](stages/db.py). Its tables start with `execution_id INTEGER NOT NULL REFERENCES executions (id)`, filled by the runner. Read upstream rows through `ctx.read`, `ctx.features`, `ctx.labels` and `ctx.split_rows`, and upstream files from `ctx.inputs[<stage>].dir`; write rows only through `ctx.write`, `ctx.save_metrics` and, for figures, `ctx.save_figures` (the stage must declare `figure_format` in `CONFIG`). Scalar results reach the Results and Compare pages through `METRICS` in [dashboard/queries.py](dashboard/queries.py).

## Development

```bash
make help                    # the targets
make test                    # the pytest suite, about 4-5 minutes
make smoke MODE=quick        # every stage on synthetic_test into resources/verify, with its own DB
```

`make smoke` runs the pipeline twice; the second pass must be fully cached. `MODE=full` sweeps all classifiers, clusterings and distances. Code style: PEP 8 with Google docstrings (Args/Returns/Raises), black and isort (`make format` runs both with profile black; 88 columns; 72 for comments and docstrings), absolute imports; `src/` is pure (no config loading, no I/O, no paths) and the stages do the I/O.

## Repository layout

```
configs/                 YAML groups and config.yaml
src/                     pure topic modules: data, space, graph, clustering, complexity/,
                         ml, dl, regressor, baselines, scores, plot/ …
stages/                  one module per stage, runner (DAG, cache), config, db, CLI
dashboard/               FastAPI app, queries, presets and jobs, templates
tests/                   end-to-end, cache, DB, routing and dashboard tests, and src/
                         functions on toy data
stages/generate_synthetic.py  the synthetic dataset
Makefile                 shortcuts
```
