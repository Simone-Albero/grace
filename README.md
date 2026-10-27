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
python -m stages render data=synthetic_test classifier=random_forest
make dashboard                    # http://127.0.0.1:8000
```

On a remote machine, forward the port first: `ssh -L 8000:127.0.0.1:8000 <host>`, then open http://127.0.0.1:8000.

The default `data` in [configs/config.yaml](configs/config.yaml) is `cic_2018_v2` (a 3.2 GB CSV), so always pass `data=`. `resources/` is not tracked by git: a fresh clone carries no data, and the synthetic generator is the only dataset that works out of the box.

## Stages and DAG

`python -m stages <target…>` runs the targets and every stage they read. Each stage is one module in [stages/](stages/) that declares `CONFIG` (the config keys its outputs depend on; `config_keys(cfg)` instead when they depend on the config, as in `classify`: ML or DL), `INPUTS` (the stages it reads; `inputs(cfg)` instead when they depend on the config, as in `render`), `FILES` (raw files whose size and mtime count) and `OUTPUTS` (the files it must write), reads only its inputs' artifacts and DB rows, and writes only its own.

```
split → prepare → space → regions → failures → baselines → render
          │         │                 ↑                      ↑
          │         └────────┐        │                      │
          └→ classify ───────┼────────┘                      │
              │              ↓                               │
              └→ crossfit → tree → complexity → regress ─────┘
```

`render` reads `tree`, `complexity` and `regress` only to draw the regressor's figures, which `regressor_figures` (true by default) turns on. With `regressor_figures=false` it reads none of them, so a run can stop after the baselines without running `crossfit` or `tree`.

| Stage | Reads | Config keys | Files written | DB tables written |
|---|---|---|---|---|
| `split` | | `data.{path,label_col,num_cols,cat_cols,min_class_count,train_frac,val_frac,test_frac}`, `seed` | `{train,val,test}.parquet` | `features`, `metrics` |
| `prepare` | split | `data.top_n`, `data.hash_buckets` | `{train,val,test}.parquet`, `preprocessor.joblib` | `classes` |
| `space` | split, prepare | `space`, `graph`, `distance`, `seed` | `space.joblib`, `graph.npz`, `lda.joblib`, `z.parquet` | `metrics` |
| `regions` | split, prepare, space | `clustering`, `distance`, `seed` | `centroids.parquet`, `assignments.parquet` | `regions`, `class_clusterings`, `metrics` |
| `classify` | split, prepare | `classifier`, `data.max_per_class`, `seed`, plus `grid_search` (ML) or `loss`, `optimizer`, `scheduler`, `fit.training`, `data.top_n`, `data.hash_buckets` (DL) | `predictions.parquet`, `model/` | `class_metrics`, `classifier_grid`, `history`, `metrics` |
| `crossfit` | split, prepare, classify | `classifier`, `data.max_per_class`, `seed`, `crossfit.n_folds`, plus `loss`, `optimizer`, `scheduler`, `fit.training`, `data.top_n`, `data.hash_buckets` (DL) | `oof.parquet` | `metrics` |
| `tree` | prepare, space, classify, crossfit | `tree`, `seed` | `tree.joblib`, `rows.parquet` | `tree_nodes`, `metrics` |
| `failures` | prepare, regions, classify | none | none | `failures`, `metrics` |
| `baselines` | prepare, regions, classify, failures | `failure_regressor.min_eval_support` | none | `baselines`, `metrics` |
| `complexity` | prepare, space, tree | `complexity`, `graph.k`, `distance`, `seed` | none | `complexity`, `metrics` |
| `regress` | tree, complexity | `failure_regressor`, `seed` | `model.joblib` | `estimates`, `regressor_folds`, `importances`, `metrics` |
| `render` | prepare, regions, classify, failures, baselines, plus tree, complexity and regress when `regressor_figures` is true | `figure_format`, `regressor_figures` | `figures/` | `figures`, `metrics` |

- **split** drops rows with missing or infinite values, drops the classes with fewer than `min_class_count` rows, and splits the rest stratified by class into train, validation and test.
- **prepare** is preprocessing only. A signed log and a robust scaler for the numerical columns, a top-N plus hash encoder for the categorical ones, and the integer class labels are all fitted on train and applied to the three splits.
- **space** builds the euclidean space every distance is measured in (the prepared numerical features plus a one-hot block for the `space.top_k` most frequent codes of each categorical feature) and a k-nearest-neighbour graph of a uniform sample of train, at most `graph.max_samples` rows. It also fits a linear discriminant analysis (LDA) on the train labels in that space and projects every row of the three splits with it: `z.parquet` holds, for each split and row position (`row`), one column per discriminant direction. Validation and test rows are projected from their features alone. Its metrics are the rows of the graph and the number of directions.
- **regions** clusters each class of train on its own, in the space and with the graph of `space`. The granularity (number of clusters, or the density-based parameters) is the candidate of the clustering grid with the least loss: the hardness spread a region's mean hides plus the sampling noise of its failure rate on validation (see [Method](#method)); the hardness of a row being the share of its `clustering.hardness_k` nearest training neighbours that belong to another class (kDN). Undersized clusters and noise points merge into the nearest region of the same class. Validation and test rows are then routed to the nearest region centre **without their labels**.
- **classify** trains one classifier on train and predicts validation and test. Each class of train is capped at `data.max_per_class` rows: a larger class is subsampled, a smaller one kept whole, and `null` keeps every row. An ML classifier with a `grid` is tuned by cross-validated grid search and refit with the winner; the MLP stops early on a holdout of its training rows, stratified by class (`fit.training.early_stopping.holdout`), so no classifier learns from a validation label.
- **crossfit** predicts every train row with a model that never saw it. It splits train into `crossfit.n_folds` stratified folds and, for each, trains the classifier on the other folds as classify trains it (the same `data.max_per_class` cap; an ML classifier with the parameters classify's grid search chose, without searching again) and predicts the fold. `oof.parquet` holds, for each train row by its position (`row`), the fold that held it out, the predicted class and one probability per class; its metrics are the out-of-fold error rate and the mean number of rows a fold's model fit on. It fails when a class has fewer train rows than folds.
- **tree** divides the rows into the leaves of a tree grown in the LDA space of `space`, one root per predicted class. Train rows start at the root of their out-of-fold class (from `crossfit`), and every node splits its rows in two with 2-means while both sides keep at least `tree.n_min` rows; a row goes to the nearer of the two centres, so validation and test rows descend from the root of the class `classify` predicted for them, **without their labels**. The tree is then cut where a deeper split would add more sampling noise to a leaf's validation rate than it removes spread of the out-of-fold errors it hides. Each leaf's failure rate is its validation rate shrunk, by a Fay-Herriot model, toward a line in its smoothed out-of-fold rate; a leaf without a validation row takes the line. Each row's probability `q` then moves its leaf rate `q_leaf` on the log-odds scale by its disagreement `d`, the share of its `tree.k_neighbours` nearest train rows labelled otherwise than its prediction (standardized on train), with a slope `beta` fitted on validation. `rows.parquet` holds, for each split and row, its leaf, `d`, `q_leaf` and `q`; `tree_nodes` holds every node of the grown tree. No test label is read.
- **failures** counts the classifier's errors on the validation rows and on the test rows routed to each region. Each region gets a validation rate, the estimate, and a test rate, the ground truth every baseline is scored against; a rate is empty when no row of its split fell in the region. No rate is measured on train.
- **baselines** predicts each region's failure rate three ways, without the descriptors: `val`, the region's own validation rate (the global validation error rate for a region with no validation row), and, from the classifier's probabilities alone, MCP (the mean of 1 minus the maximum class probability of the region's test rows) and ATC (the share of the region's test rows under a confidence threshold fitted on validation). It scores all three against the test rates on the regions with at least `failure_regressor.min_eval_support` test rows (the *used* regions) and fails if there is none.
- **complexity** describes every leaf of the tree and every class on their train rows (see [Method](#method)).
- **regress** learns each leaf's validation failure rate with a Random Forest on its descriptors and those of the class its root predicts, on every leaf with a validation row, each weighted by how many it has, under nested cross-validation (a randomized search inside each outer fold), and refits the forest on all of them. Those leaves are predicted out of fold, and the predictions are scored against their validation rates (`oof_spearman`, `oof_r2`, `oof_mae`, `oof_mse`); a leaf with no validation row is predicted by the refit, which never saw it. It reads no test label.
- **render** draws the static figures of the classifier and the baselines: confusion matrix, F1 per class, the loss of a DL classifier, the val rate, MCP and ATC against the test rates, and the test rate of the regions of each class (box and strip plot, a point's area is its test rows). With `regressor_figures` it also draws the regressor's: its out-of-fold predictions against the leaves' validation rates (`regress/predicted_vs_val`), the forest's feature importances (`regress/feature_importances`) and, for each of the four most important descriptors, its distribution over the leaves binned by validation rate (`descriptors/<feature>`).

## CLI

```
python -m stages <target…> [key=value…] [--sweep key=v1,v2…] [--force]
```

- Targets run with every upstream stage. `group=option` (for example `data=nb15_v2`, `classifier=knn`) picks a YAML file of that group; any other `key=value` overrides a value with a dotted key (`seed=7`, `graph.k=15`).
- `--sweep` runs the cartesian product of its values, the first sweep outermost. Every combination is validated before the first one runs. Values are split on the commas outside square brackets, so a list value can be swept too: `--sweep 'clustering.params.min_samples=[1,5],[15,50]'` (quote it: the shell would expand the brackets).
- `--force` re-runs the named targets only; their upstream stays cached.
- SIGTERM and Ctrl-C mark the run and the running execution `failed`. A run killed outright (SIGKILL, power loss) is marked `failed` at the first of these moments when no other run is live: the next run on the root or the dashboard's start. A live run holds a shared lock on `runs.lock`, and rows still `running` while nobody holds it are dead. A SIGKILL of the CLI can leave joblib (loky) worker processes behind: they finish their queued fits, hold their memory for minutes and a few never exit; when no run is live, `pkill -f loky` removes them.
- A database error (locked, read-only, full disk, corrupted) ends with a message naming `<root>/grace.db` and, when it is known, what to do (restore a copy, or move it away together with `artifacts/` to start a new one, wait for the process that locks it, make the root writable, free disk space). A full disk while a stage writes an artifact ends with the OS error.

An execution is reused when a done execution of the same stage has the same key and all its output files exist with the size they were written with (`sizes.json` in its folder, without which the stage runs again; the content is not checked, so a same-size edit goes unnoticed: use `--force <stage>`). The key is the sha256 of the stage's config slice, the ids of the input executions it read, and the size and mtime of the raw CSV (`split`). `runtime` and `root` are in no key. Code changes are not detected: use `--force`.

```bash
python -m stages render data=synthetic_test classifier=random_forest
python -m stages render data=synthetic_test classifier=knn regressor_figures=false  # without the regressor
python -m stages render data=synthetic_test --sweep classifier=knn,xgboost,mlp --sweep distance=euclidean,cosine
python -m stages regions data=synthetic_test clustering=hdbscan distance=cosine
python -m stages regions data=synthetic_test clustering=hdbscan 'clustering.params.min_samples=[5,15]'
python -m stages regions data=synthetic_test clustering=hdbscan --sweep 'clustering.params.min_samples=[1,5],[15,50]'
python -m stages classify data=synthetic_test classifier=xgboost --force
python -m stages render data=synthetic_test root=resources/other runtime.n_jobs=16
```

## Configuration

[configs/](configs/) holds one folder per group, one YAML per option: `data` (10), `classifier` (10), `clustering` (4), `space`, `graph`, `complexity`, `failure_regressor`, `fit`, `grid_search`, `loss`, `optimizer`, `scheduler`, `crossfit`, `tree`. [configs/config.yaml](configs/config.yaml) chooses the option of each group (`groups`) and holds `seed`, `distance` (`euclidean` or `cosine`), `figure_format` (`pdf` or `png`), `regressor_figures` (whether `render` also draws the regressor's figures), `root`, `runtime.*` and the `clustering` keys every algorithm shares (`max_fit_samples`, `max_regions`, `min_cluster_floor`, `hardness_k`). `runtime` says how fast stages compute, never what: `n_jobs` (the classifier grid search, the failure regressor search, the k-NN searches of space, tree and complexity, and the fit of random_forest, knn and xgboost), `device`, `num_workers` and `pin_memory` (DL only).

`pipeline/config.py` loads the YAML files, applies the overrides and validates the whole config in one place, so a bad key, type or range of the pipeline's own settings fails before anything runs, with the valid values in the message. Values passed through to a library as they are (`classifier.params` and `grid` values, `optimizer.params`, `scheduler.params`, `clustering.params` values, `failure_regressor.params`) are checked by that library when their stage runs. So are the columns against the CSV (split) and the length of a `loss.params.class_weight` list (classify and crossfit, which know the classes). The config is strict: an unknown key fails. Add a new key (including `classifier.params.*`, `clustering.params.*` and `failure_regressor.params.*`, which cannot be introduced from the command line) to a YAML first.

## Outputs

Everything lives under the root (`resources` by default, `root=` to change it):

- `grace.db`: one SQLite database for every configuration.
- `artifacts/<stage>/<execution id>/`: the files of each execution. The paths stored in the DB are relative to the root, so a root can be moved or copied.
- `runs.lock`: held (shared) by every live run.

There are no migrations: after a schema change, delete the DB and its artifacts.

| Tables | Content |
|---|---|
| `runs`, `executions`, `run_stages` | bookkeeping: one row per run (one per sweep combination) and per stage execution (config, inputs, status, error, duration, directory); which execution each run used and whether it was cached |
| `metrics`, `features`, `classes` | scalar results of an execution; feature names and kinds; class names and split sizes |
| `regions`, `class_clusterings` | each region's class and sizes per split; per class, the chosen number of clusters and its loss |
| `complexity` | one row per leaf or class and descriptor |
| `class_metrics`, `classifier_grid`, `history` | per-class precision, recall and F1; the grid-search candidates; the loss of a DL classifier |
| `failures`, `baselines` | per region, validation and test rows, errors and failure rates; whether it is scored, its val rate, MCP and ATC |
| `tree_nodes` | per node of the tree: parent, predicted class, depth, train rows and their out-of-fold errors, cost, whether it is a leaf, internal or pruned, validation rows and errors, and a leaf's failure rate and the weight of its own validation rate |
| `estimates`, `regressor_folds`, `importances` | per leaf, the Random Forest prediction and the outer fold that held the leaf out (empty when the refit predicted it); the outer folds with the leaves each held out (fold -1 is the refit, with the leaves it trained on); the mean feature importances |
| `figures` | the figure files of a render execution |

For example, per run, the Spearman correlation of the three baselines with the regions' test failure rates, and that of the regressor's out-of-fold predictions with the leaves' validation rates:

```sql
SELECT r.id, json_extract(r.config, '$.groups.classifier') AS classifier,
       MAX(CASE WHEN m.name = 'oof_spearman' THEN m.value END) AS rf_oof,
       MAX(CASE WHEN m.name = 'val_spearman' THEN m.value END) AS val,
       MAX(CASE WHEN m.name = 'mcp_spearman' THEN m.value END) AS mcp,
       MAX(CASE WHEN m.name = 'atc_spearman' THEN m.value END) AS atc
FROM runs r
JOIN run_stages s ON s.run_id = r.id AND s.stage IN ('regress', 'baselines')
JOIN metrics m ON m.execution_id = s.execution_id
WHERE r.status = 'done'
GROUP BY r.id;
```

## Dashboard

`make dashboard ROOT=<root> PORT=<port>` serves the results in `<root>/grace.db` (created if missing) on 127.0.0.1, with no authentication: any local user can reach it. Requests must name `127.0.0.1` or `localhost` as host, not `[::1]` (a guard against DNS rebinding). It only reads: the DB, opened read-only, and two artifact files for the confusion matrix (the test labels in prepare's `test.parquet` and classify's `predictions.parquet`). At its start it marks `failed` the runs a killed process left `running` (see the CLI section). It is built with FastAPI, Jinja2, HTMX and Plotly.js from a CDN, so it needs no build step. Its colors follow the system's light or dark theme, and a page that shows a `running` run refreshes itself every 5 s.

An experiment is the runs whose configs are equal once `seed`, `root`, `runtime`, `figure_format` and `regressor_figures` are set aside; each of its seeds is represented by that seed's latest run, and its metrics are the mean ± standard deviation over the seeds.

- **Experiments** (`/`, JSON at `/api/experiments`): one row per experiment with its seeds, the state of their stages and the headline metrics, filterable by data, classifier, clustering and distance, sortable.
- **Experiment**: its seeds and their runs, the headline metrics across the seeds, figures per metric family and a table of every metric.
- **Run**: tabs Overview, Data, Classifier (with the confusion matrix, or a note naming the two files it looked for when they are missing or disagree in length), Regions (the number of regions per class, their train rows and the clustering of each class), Failures and Predictors (the scores of the three baselines, their estimates against the test rates, and the regressor's feature importances and outer folds). A tab whose stages have no done execution in the run is greyed out. The figures of `render` are not shown. The Failures tab plots each region's test rate against its validation rate, each class's rate on both splits and the distribution of the region rates; it bins the regions by failure rate (0 on its own, then up to 5, 10, 20, 30, 50 and 100%) on validation and on test, counting them and averaging their rows in that split; it boxes each class's validation − test gap, region by region; and it lists the 15 regions of highest test rate.
- **Compare** (`/compare`): any metric and the three baselines' scores across experiments, one value per seed, filterable and grouped by any of data, classifier, clustering and distance.

## Method

Three steps, together with the classifier whose errors they are fitted on. The method assumes nothing about the classifier, the clustering algorithm or the number of classes; it requires only tabular features.

1. **Identify regions.** Two divisions of the data coexist, both built from the training data. The *regions* divide each class into compact sub-regions. A finer partition hides less of the spread of its rows' hardness inside each region's average, but leaves each region fewer of the validation rows its failure rate will be estimated on, so the estimate carries more sampling noise. The candidate with the smallest sum of the two, weighted by region size, is kept. The *leaves* divide the rows by the class the classifier predicts for them: a tree of 2-means splits in the LDA projection of the space, cut by the same trade-off, with the spread of train's out-of-fold errors in place of hardness. The regions carry the failure rates the baselines are scored on; the leaves are what the descriptors and the regressor work on.
2. **Describe each leaf.** A leaf holds train rows of several classes, so each of its (leaf, class) populations is scored on how separable it is from the nearest populations of another class, and the leaf takes the mean of its populations' scores, weighted by their rows. The same is done at class level. The descriptors read only the features and labels of train; the classifier enters only through the leaves they are measured on.
3. **Learn the mapping.** A Random Forest learns each leaf's validation failure rate from its descriptors under nested cross-validation, and reports the correlation of its out-of-fold predictions with those rates together with the descriptors that mattered. The tree-based estimator is deliberate: every risk estimate arrives with the properties that produced it. The baselines, scored against the regions' test rates, are MCP and ATC, which read the classifier's probabilities instead, and the region's own validation rate.

The classifier contributes its predictions to step 1, which decide the root each row descends from and where the tree is cut, and to step 3 the failure rate it achieved in each leaf. That is what ties the mapping to the model rather than to the dataset.

- Every distance (clustering, routing, hardness, the neighbour graph and every descriptor) is measured in one space. Under `distance=cosine` every row is scaled to unit length first. The classifier keeps its own encoding of the categorical features: it is the object under study, not part of the geometry. The tree and each row's disagreement with its nearest train rows work in the LDA projection of that space, with the euclidean distance.
- Regions are built from the training split alone, each inside one class. Every validation and test row is routed to the region whose centre lies nearest **without reference to its label**, as a row would be at inference time, and a region's failure rate on a split counts the classifier's mistakes on that split's rows routed to it, whatever their class. Likewise, every validation and test row descends the tree from the root of its predicted class, without its label. This keeps the correlation a genuine prediction rather than a restatement of labels already known.
- Failure rates are never counted on train's own predictions: training rows fall back into their own class's region more often than new data would, and would flatter the regions. The tree reads train's out-of-fold errors from `crossfit` instead, made by models that never saw those rows. Validation is the label budget: each region's and each leaf's rate is estimated there (the val-rate baseline, the tree's leaf rates and the regressor's target), and the ATC threshold and the tree's per-row correction are fitted there. Test is ground truth only, the rates the baselines are scored against. No validation label and no test row's features or label shape the classifier, the graph, the regions, the tree's splits and cut or the descriptors (only the validation split's size enters the granularity loss and the cut).
- The analysis works per region or leaf rather than per class, since one class usually occupies several separate areas of the space. Class-level descriptors are kept as a coarse reference.

How the data is divided into regions matters: count-based algorithms (`kmeans`, `birch`, `spectral`) suit a continuous difficulty gradient, while `hdbscan` looks for genuine gaps and finds far fewer regions in data without them.

### What the descriptors measure

Five families, all measured in that space on train rows, (leaf, class) population by population; a leaf's value of a key is the mean over its populations, weighted by their rows. The neighbourhood and network families read the training graph: its nodes are a uniform sample of train, the whole split up to `graph.max_samples` rows, joined by their `graph.k` nearest-neighbour graph and a spanning tree over it. Above that size a population is measured on at most `complexity.max_sample_per_population` of its rows and a class on `complexity.max_sample_per_class`. A population with fewer than `complexity.min_population` train rows is not measured and is nobody's rival; a rival is a population of another class. The feature and neighbourhood families and `network_density` set a population against the `complexity.top_k_clusters` rival populations whose centres lie nearest to it, reported as min / mean / max across them.

| Family | Keys | What it captures |
|---|---|---|
| **F**: feature | `f1`–`f4` | how well individual features separate the population from its rivals |
| **N**: neighbourhood | `n1`–`n4` | how many of a population's neighbours belong to another class |
| **ND**: network | `network_density`, `cls_coef`, `hub` | how many neighbour links cross into a rival population |
| **T**: dimensionality | `t2`–`t4` | how many features there are relative to rows, and how many of them matter |
| **G**: geometry | `max_dispersion`, `p95_dispersion`, `dist_to_nearest_rival`, `p5_silhouette`, `frac_at_risk` | how widely the population is spread, and how close the nearest rival lies |

The regressor sees every key twice, as `leaf_<key>` and as `class_<key>` for the class the leaf's root predicts. Its dataset is every leaf with at least one validation row, each weighted by its validation rows.

### Inside the synthetic dataset

Each class consists of a clean core together with a *difficulty ladder* running towards every class it is meant to overlap with (`_OVERLAPS` in [generate_synthetic.py](pipeline/generate_synthetic.py)). Every rung sits a fixed distance from the boundary between the two classes, and both sides face one another symmetrically, so the best error rate any classifier could reach on a rung is known in advance:

| Sub-group | Distance from boundary | Best possible error | Rows |
|---|---|---|---|
| `canonical` | none | ≈ 0 % | 23,966 |
| `clear` | 1.80 σ | ≈ 4 % | 19,316 |
| `evasive` | 0.90 σ | ≈ 18 % | 16,560 |
| `mimicry` | 0.25 σ | ≈ 40 % | 9,658 |

The file has 69,500 rows, 10 classes, 20 numerical and 2 categorical features; the smallest class has 500 rows, well above `min_class_count`, so all 10 are kept. The two categorical features fade out along the same ladder, so they cease to help exactly where the numbers begin to overlap. Each row's intended difficulty is recorded in the `true_subgroup` column so that you can check afterwards; the pipeline never reads it, and `split` keeps only the columns the configuration lists. Accuracy settles at about 0.85 by design: the hard rungs are genuinely hard, and that spread is what makes a per-region correlation meaningful.

## Datasets

[configs/data/](configs/data/) has one YAML per dataset: `path`, `label_col`, `num_cols`, `cat_cols`, `train_frac`, `val_frac`, `test_frac`, `min_class_count` (20 in every shipped YAML), `max_per_class` (the cap on each class of train that classify and crossfit fit on, `null` for none), `top_n` and `hash_buckets`. The four network datasets (`nb15_v2`, `cic_2018_v2`, `ton_iot_v2`, `bot_iot_v2`) are read from `resources/raw_data/dataset_v2/`, the five general benchmarks (`bank_marketing`, `covertype`, `letter_recognition`, `statlog_landsat_satellite`, `thyroid_disease`) from `resources/raw_data/<name>/`, and the synthetic one from `resources/raw_data/synthetic/`. The benchmark CSVs are not in the repository.

To add a dataset, place the CSV under `resources/raw_data/`, copy a YAML from `configs/data/`, set `path` and the columns, and run with `data=<name>`.

Every shipped configuration splits 40 % train, 15 % validation, 45 % test. No classifier trains on validation: the validation labels are the budget every region's and leaf's failure rate is estimated on (the val-rate baseline, the tree's leaf rates and the regressor's target), and they fit the ATC threshold and the tree's per-row correction. The test split is held out as ground truth: its labels score the classifier and the baselines, and fit nothing. Measured on `nb15_v2` (2.39 million rows) with random_forest and `runtime.n_jobs=16`: about 6.5 minutes end to end, peak memory about 10 GB.

## Extending

- **A classifier**: an entry in `CLASSIFIERS` in [src/ml.py](src/ml.py) (the estimator and the categorical strategy in front of it: a key of `_STRATEGIES`, or None to pass the prepared frame as it is) and a `configs/classifier/<name>.yaml` with `kind`, `name`, `params` and `grid`. If it takes `n_jobs`, add it to `PARALLEL` in the same file.
- **A DL model**: an entry in `MODELS` in [src/dl.py](src/dl.py), a class whose constructor takes `num_classes`, `num_numerical_features`, `cardinalities` and the YAML `params`, and whose `forward(x_num, x_cat)` returns logits, each row's independent of the others in eval mode (prediction runs in chunks); and a YAML with `kind: dl`.
- **A clustering**: a function `fit(X, *, …, max_fit_samples, random_state) -> labels` (-1 = noise) in `ALGORITHMS` in [src/clustering.py](src/clustering.py) and a `configs/clustering/<name>.yaml` with `name` and `params` (`{}` for none). If it takes `n_clusters`, the pipeline derives it per class from the data and sweeps it; it must not appear in the YAML. List-valued parameters are swept and the candidate of least loss kept.
- **A descriptor**: return one more key from a family function in [src/complexity/](src/complexity/); it reaches the complexity table, the regressor (as `leaf_<key>` and `class_<key>`) and its feature importances on the dashboard by itself. A new family is one call, merged into the returned dict, in `src/complexity/measure.py` (and, optionally, a label in `src/plot/style.py`).
- **A stage**: a module in `stages/` with `CONFIG` (or `config_keys`), `INPUTS` (or `inputs`), `FILES`, `OUTPUTS` and `run(ctx)`; insert it into `STAGES` in [pipeline/runner.py](pipeline/runner.py) after its inputs and add its tables to `SCHEMA` in [pipeline/db.py](pipeline/db.py). Its tables start with `execution_id INTEGER NOT NULL REFERENCES executions (id)`, filled by the runner. Read upstream rows through `ctx.read`, `ctx.features`, `ctx.labels` and `ctx.split_rows`, and upstream files from `ctx.inputs[<stage>].dir`; write rows only through `ctx.write`, `ctx.save_metrics` and, for figures, `ctx.save_figures` (the stage must declare `figure_format` in `CONFIG`). Add it to `STAGE_ORDER` in [dashboard/queries.py](dashboard/queries.py) too (a test keeps it equal to `STAGES`); then every metric it saves is listed on the Experiment and Compare pages by itself. A metric becomes a column of the Experiments page through `HEADLINE` in [dashboard/summary.py](dashboard/summary.py); what a run page shows is built per tab in [dashboard/tabs.py](dashboard/tabs.py) (`TABS` and `build`), with a `run_<tab>.html` template.

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
src/                     pure topic modules: data, space, graph, clustering, tree,
                         complexity/, ml, dl, regressor, baselines, scores, plot/ …
stages/                  one module per stage, the stage contract (Context), CLI
pipeline/                runner (DAG, cache), config, db, the synthetic dataset generator
dashboard/               read-only FastAPI app: queries, experiments, run tabs, charts,
                         templates, static
tests/                   end-to-end, cache, DB, routing and dashboard tests, and src/
                         functions on toy data
Makefile                 shortcuts
```
