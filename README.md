# grace

A framework for finding out **where** a trained classifier fails, and **why**, without needing labelled test data for the regions in question.

It divides a dataset into regions by the mistakes a classifier actually made, and gives every region an error-rate estimate and every row a failure probability, including regions that hold no labelled validation row. Each region can also be described by how it sits relative to the classes competing with it.

The estimate belongs to the model you point it at. It is fitted on that model's own errors, so it describes how *that* classifier copes with the shape of your data rather than how difficult the data is in the abstract. This makes it useful for identifying unreliable regions before you have the labels to prove they are unreliable, for deciding where to gather more data or review labels, and for monitoring a model after deployment.

Everything is tabular and everything is driven by configuration: ten classifiers (scikit-learn, XGBoost, PyTorch), configurations for the four public network-security datasets and five general benchmarks, and a synthetic dataset that exercises the whole pipeline.

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

`python -m stages <target…>` runs the targets and every stage they read. Each stage is one module in [stages/](stages/) that declares `CONFIG` (the config keys its outputs depend on; `config_keys(cfg)` instead when they depend on the config, as in `classify`: ML or DL), `INPUTS` (the stages it reads; `inputs(cfg)` instead when they depend on the config), `FILES` (raw files whose size and mtime count) and `OUTPUTS` (the files it must write), reads only its inputs' artifacts and DB rows, and writes only its own.

```
split → prepare → space ─────┐
          │                  ↓
          └→ classify ────→ tree → complexity
                 │           │
                 ↓           ↓
             baselines ──→ evaluate
                               ↓
                            render
```

The drawing shows the main flow; the table lists every stage each one reads. No stage reads `complexity`, so it runs only when a target names it: `python -m stages render complexity` runs every stage, `python -m stages render` all but `complexity`.

| Stage | Reads | Config keys | Files written | DB tables written |
|---|---|---|---|---|
| `split` | | `data.{path,label_col,num_cols,cat_cols,min_class_count,train_frac,val_frac,test_frac}`, `seed` | `{train,val,test}.parquet` | `features`, `metrics` |
| `prepare` | split | `data.top_n`, `data.hash_buckets` | `{train,val,test}.parquet`, `preprocessor.joblib` | `classes` |
| `space` | split, prepare | `space` | `space.joblib`, `lda.joblib`, `z.parquet` | `metrics` |
| `classify` | split, prepare | `classifier`, `data.max_per_class`, `seed`, plus `grid_search` (ML) or `loss`, `optimizer`, `scheduler`, `fit.training`, `data.top_n`, `data.hash_buckets` (DL) | `predictions.parquet`, `model/` | `class_metrics`, `classifier_grid`, `history`, `metrics` |
| `tree` | prepare, space, classify | `tree`, `seed` | `tree.joblib`, `rows.parquet` | `tree_nodes`, `metrics` |
| `complexity` | prepare, space, tree | `complexity`, `graph`, `seed` | none | `complexity`, `metrics` |
| `baselines` | prepare, space, classify | `seed` | `scores.parquet` | `metrics` |
| `evaluate` | prepare, classify, tree, baselines | none | none | `leaf_tests`, `metrics` |
| `render` | prepare, classify, evaluate | `figure_format` | `figures/` | `figures`, `metrics` |

- **split** drops rows with missing or infinite values, drops the classes with fewer than `min_class_count` rows, and splits the rest stratified by class into train, validation and test.
- **prepare** is preprocessing only. A signed log and a robust scaler for the numerical columns, a top-N plus hash encoder for the categorical ones, and the integer class labels are all fitted on train and applied to the three splits.
- **space** builds the euclidean space every distance is measured in (the prepared numerical features plus a one-hot block for the `space.top_k` most frequent codes of each categorical feature). It also fits a linear discriminant analysis (LDA) on the train labels in that space and projects every row of the three splits with it: `z.parquet` holds, for each split and row position (`row`), one column per discriminant direction. Validation and test rows are projected from their features alone. Its metric is the number of directions.
- **classify** trains one classifier on train and predicts validation and test. Each class of train is capped at `data.max_per_class` rows: a larger class is subsampled, a smaller one kept whole, and `null` keeps every row. An ML classifier with a `grid` is tuned by cross-validated grid search and refit with the winner; the MLP stops early on a holdout of its training rows, stratified by class (`fit.training.early_stopping.holdout`), so no classifier learns from a validation label.
- **tree** divides the validation and test rows into the leaves of a tree grown in the LDA space of `space` plus one axis, one root per predicted class, from the predictions of `classify`. It first halves validation at random, stratified by class and seeded by `seed` (`halve` in [src/data.py](src/data.py)): the tree is grown and cut on one half, the *grown* half, and its leaf rates are estimated on the other, the *estimation* half. It fails when a class has fewer than two validation rows. The extra axis is each row's disagreement `d`, the share of its `tree.k_neighbours` nearest train rows in the LDA space labelled otherwise than its predicted class, standardized with the mean and standard deviation of the grown half and multiplied by `tree.d_weight` times the spread of the LDA space on train (the square root of the summed variances of its directions); `tree.d_weight=0` grows the tree on the LDA space alone. The rows of the grown half start at the root of their predicted class, and every node splits its rows in two with 2-means while both sides keep at least `tree.n_min` rows; a row goes to the nearer of the two centres, so the estimation half and the test rows descend from the root of their predicted class **without their labels**. The tree is then cut where a deeper split would add more sampling noise to a leaf's estimated rate than it removes spread of the grown half's errors it hides; the noise of a node is counted on the estimation rows that descend to it, without reading their labels. Each leaf's failure rate is its estimation-half rate shrunk, by a Fay-Herriot model, toward a line in its smoothed grown-half rate; a leaf no estimation row reaches takes the line. Each row's probability `q` then moves its leaf rate `q_leaf` on the log-odds scale by the classifier's confidence: `q = expit(logit(q_leaf) + beta * s)`, where `s` is the logit of 1 minus the row's top class probability, standardized with the mean and standard deviation of the grown half, and the slope `beta` is fitted on the estimation half. `rows.parquet` holds, for each validation and test row (`split`, `row`), whether it is in the grown half (`grow`), its leaf, `d`, `confidence` (the top class probability), `q_leaf` and `q`; `tree_nodes` holds every node of the grown tree, with its grown rows and errors and its estimation rows and errors. Its metrics include the size and error rate of each half (`n_grow`, `grow_error_rate`, `n_estimate`, `estimate_error_rate`) and the leaves no estimation row reaches (`n_leaves_without_estimate`). No test label is read.
- **complexity** describes every leaf of the tree and every class on the grown half of validation, with its true labels, in a k-nearest-neighbour graph of its own over those rows (a uniform sample of at most `graph.max_samples` of them, see [Method](#method)); its metrics include the rows of that graph (`n_graph_rows`).
- **baselines** gives every validation and test row four failure probabilities, built from the classifier's outputs without the descriptors and without reading a test label: `val_rate`, the global validation failure rate; `confidence`, 1 minus the row's top class probability, calibrated on validation by isotonic regression; `binned`, the validation failure rate of the row's bin, the bins dividing each predicted class by confidence into equal shares of its validation rows (one bin per 150 validation rows, at least one per class); and `gbm`, a histogram gradient boosting classifier trained on the classifier's errors on the grown half of validation (the same halves as `tree`) from the row's LDA coordinates, its top class probability and its predicted class, then calibrated isotonically on the estimation half. `scores.parquet` holds the four, for each split and row; its metrics are the validation error rate, the number of bins and the rows the gbm trained on.
- **evaluate** is the only stage that judges the failure predictors against the test labels. For every test row it compares six probabilities with whether the classifier failed on it: `tree` (the row's `q`), `tree_leaf` (its leaf's `q_leaf`) and the four of `baselines`. Each predictor gets four scores, next to `test_error_rate` and `n_test`: its mean squared error against the failures divided by the one of the constant `val_rate` (`<predictor>_mse_ratio`: below 1 beats the constant); the same ratio over up to 10 bins of equal size by the predictor's probability (equal probabilities share a bin), where each bin's gap is its mean probability against its failure rate, weighted by its rows, and the reference is `val_rate`'s gap on the same bins (`<predictor>_binned_mse_ratio`: a row's own chance drops out but each bin's sampling noise stays, so compare predictors within a run); its AUROC, the chance that a failed row gets a higher probability than a correct one (`<predictor>_auroc`: 0.5 is a random order); and its average precision divided by the test failure rate (`<predictor>_ap_lift`: 1 is a random order, 1 / failure rate the best). A score is NaN when it cannot be taken: the AUROC when every test row failed or none did, the average precision ratio when none did, and either MSE ratio when `val_rate`'s own error, plain or binned, is 0. `leaf_tests` holds, for each leaf, its test rows, the classifier's errors on them, its test rate and the mean `q` of those rows.
- **render** draws the static figures: confusion matrix, F1 per class and the loss of a DL classifier (`classification/…`), and the four scores of the predictors from `evaluate` (`predictors/mse_ratio`, `predictors/binned_mse_ratio`, `predictors/auroc`, `predictors/ap_lift`).

## CLI

```
python -m stages <target…> [key=value…] [--sweep key=v1,v2…] [--force]
```

- Targets run with every upstream stage. `group=option` (for example `data=nb15_v2`, `classifier=knn`) picks a YAML file of that group; any other `key=value` overrides a value with a dotted key (`seed=7`, `graph.k=15`).
- `--sweep` runs the cartesian product of its values, the first sweep outermost. Every combination is validated before the first one runs. Values are split on the commas outside square brackets, so a list value can be swept too: `--sweep 'classifier.grid.n_estimators=[100,300],[500]'` (quote it: the shell would expand the brackets).
- `--force` re-runs the named targets only; their upstream stays cached.
- SIGTERM and Ctrl-C mark the run and the running execution `failed`. A run killed outright (SIGKILL, power loss) is marked `failed` at the first of these moments when no other run is live: the next run on the root or the dashboard's start. A live run holds a shared lock on `runs.lock`, and rows still `running` while nobody holds it are dead. A SIGKILL of the CLI can leave joblib (loky) worker processes behind: they finish their queued fits, hold their memory for minutes and a few never exit; when no run is live, `pkill -f loky` removes them.
- A database error (locked, read-only, full disk, corrupted) ends with a message naming `<root>/grace.db` and, when it is known, what to do (restore a copy, or move it away together with `artifacts/` to start a new one, wait for the process that locks it, make the root writable, free disk space). A full disk while a stage writes an artifact ends with the OS error.

An execution is reused when a done execution of the same stage has the same key and all its output files exist with the size they were written with (`sizes.json` in its folder, without which the stage runs again; the content is not checked, so a same-size edit goes unnoticed: use `--force <stage>`). The key is the sha256 of the stage's config slice, the ids of the input executions it read, and the size and mtime of the raw CSV (`split`). `runtime` and `root` are in no key. Code changes are not detected: use `--force`.

```bash
python -m stages render data=synthetic_test classifier=random_forest
python -m stages render complexity data=synthetic_test classifier=knn  # complexity runs only when named
python -m stages render data=synthetic_test --sweep classifier=knn,xgboost,mlp --sweep seed=1,2,3
python -m stages evaluate data=synthetic_test tree.n_min=50
python -m stages complexity data=synthetic_test complexity.top_k_clusters=5
python -m stages classify data=synthetic_test classifier=random_forest --sweep 'classifier.grid.n_estimators=[100,300],[500]'
python -m stages classify data=synthetic_test classifier=xgboost --force
python -m stages render data=synthetic_test root=resources/other runtime.n_jobs=16
```

## Configuration

[configs/](configs/) holds one folder per group, one YAML per option: `data` (10), `classifier` (10), `space`, `graph`, `complexity`, `fit`, `grid_search`, `loss`, `optimizer`, `scheduler`, `tree`. [configs/config.yaml](configs/config.yaml) chooses the option of each group (`groups`) and holds `seed`, `figure_format` (`pdf` or `png`), `root` and `runtime.*`. `runtime` says how fast stages compute, never what: `n_jobs` (the classifier grid search, the k-NN searches of tree and complexity, and the fit of random_forest, knn and xgboost), `device`, `num_workers` and `pin_memory` (DL only).

`pipeline/config.py` loads the YAML files, applies the overrides and validates the whole config in one place, so a bad key, type or range of the pipeline's own settings fails before anything runs, with the valid values in the message. Values passed through to a library as they are (`classifier.params` and `grid` values, `optimizer.params`, `scheduler.params`) are checked by that library when their stage runs. So are the columns against the CSV (split) and the length of a `loss.params.class_weight` list (classify, which knows the classes). The config is strict: an unknown key fails. Add a new key (including `classifier.params.*`, which cannot be introduced from the command line) to a YAML first.

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
| `complexity` | one row per leaf or class and descriptor |
| `class_metrics`, `classifier_grid`, `history` | per-class precision, recall and F1; the grid-search candidates; the loss of a DL classifier |
| `tree_nodes` | per node of the tree: parent, predicted class, depth, rows of the grown half of validation and the classifier's errors on them, cost, whether it is a leaf, internal or pruned, rows of the estimation half and errors, and a leaf's failure rate and the weight of its own estimation-half rate |
| `leaf_tests` | per leaf, the test rows that reach it, the classifier's errors on them, its test rate and the mean probability `q` of those rows |
| `figures` | the figure files of a render execution |

For example, per run, the MSE ratio of the tree's per-row probability and of two baselines on the test rows, and the tree's AUROC on the test failures:

```sql
SELECT r.id, json_extract(r.config, '$.groups.classifier') AS classifier,
       MAX(CASE WHEN m.name = 'tree_mse_ratio' THEN m.value END) AS tree,
       MAX(CASE WHEN m.name = 'confidence_mse_ratio' THEN m.value END) AS confidence,
       MAX(CASE WHEN m.name = 'gbm_mse_ratio' THEN m.value END) AS gbm,
       MAX(CASE WHEN m.name = 'tree_auroc' THEN m.value END) AS tree_auroc
FROM runs r
JOIN run_stages s ON s.run_id = r.id AND s.stage = 'evaluate'
JOIN metrics m ON m.execution_id = s.execution_id
WHERE r.status = 'done'
GROUP BY r.id;
```

## Dashboard

`make dashboard ROOT=<root> PORT=<port>` serves the results in `<root>/grace.db` (created if missing) on 127.0.0.1, with no authentication: any local user can reach it. Requests must name `127.0.0.1` or `localhost` as host, not `[::1]` (a guard against DNS rebinding). It only reads: the DB, opened read-only, two artifact files for the confusion matrix (the test labels in prepare's `test.parquet` and classify's `predictions.parquet`) and two for the map of a class's leaves (the validation rows of space's `z.parquet`, and their half, leaf and disagreement `d` in tree's `rows.parquet`). At its start it marks `failed` the runs a killed process left `running` (see the CLI section). It is built with FastAPI, Jinja2, HTMX and Plotly.js from a CDN, so it needs no build step. Its colors follow the system's light or dark theme, and a page that shows a `running` run refreshes itself every 5 s.

An experiment is the runs whose configs are equal once `seed`, `root`, `runtime` and `figure_format` are set aside; each of its seeds is represented by that seed's latest run, and its metrics are the mean ± standard deviation over the seeds.

- **Experiments** (`/`, JSON at `/api/experiments`): one row per experiment with its seeds, the state of their stages and the headline metrics, filterable by data and classifier, sortable.
- **Experiment**: its seeds and their runs, the headline metrics across the seeds, figures per metric family and a table of every metric.
- **Run**: tabs Overview, Data, Classifier, Tree and Predictors. A tab whose stages have no done execution in the run is greyed out. The figures of `render` are not shown. The Classifier tab shows first, across the full width, the confusion matrix (or a note naming the two files it looked for when they are missing or disagree in length), then the classifier's other figures and a table of each class's train rows, its train rows after the `data.max_per_class` cap and its test rows, beside its precision, recall and F1 on test, with a note on which of those rows the classifier was fitted on. The Tree tab shows the tree's metrics as tiles, how many leaves hold how many rows of the grown half, the estimation half and test (in power-of-two bins), each leaf's estimate against its raw estimation-half rate and its test rate against its estimate, the tree of one predicted class at a time, picked from a menu (a marker per node that was not pruned, a square for a leaf, sized by its grown rows, with its counts and rates on the grown half, the estimation half and test on hover) and under it a map of the class's leaves (a random sample of each leaf's grown rows as points in the plane of the first two principal components of the class's grown rows, either in the LDA space z or in z plus the tree's disagreement axis `s_d`, the standardized `d` weighted as the tree grew on it, picked from a third menu shown only when the tree has that axis (its metric `d_scale` is not 0), up to 1,000 rows per leaf, thinned to the same count per leaf so that at most 8,000 are drawn, with the view framed on the 1st to 99th percentile of each axis and the chart's autoscale showing all; or a note naming the two files it looked for when they are missing or disagree in validation length), both colored by the measure picked from a second menu (purity, the share of the node's grown rows whose label is the class; the grow rate, on the grown half; the estimate rate, on the estimation half; and, when the run includes `evaluate`, the test rate) on a diverging scale (centred on its value at the top of the tree drawn: blue is better than that value and red worse, a purity being worse below it and a rate above, the ends of the scale are the 2nd and 98th percentiles of the leaves, the distance from the centre is square-rooted so that the many leaves close to it still show their side, and the colorbar is labelled with the values, `top` marking the centre), the root of each predicted class (its grown rows and the classifier's errors among them, its leaves and the depth of its cut), and a sortable table of the leaves; each leaf links to its own page (`/runs/<run id>/leaves/<node id>`) with its counts and rates, its path from the root and, when the run includes `complexity`, its complexity descriptors beside those of its class. Clicking a node of the tree that splits cuts the tree and the map down to its subtree: the tree is drawn from that node, with the color scale centred on its value, and the map is refitted to the plane of the first two principal components of the subtree's grown rows in the space picked; a path above the tree leads back to the class or to any node between it and the one clicked, a Reset button brings back the whole class, and a class picked from the menu starts whole. The Predictors tab shows the four scores of the six predictors of `evaluate` on the test rows, the best of each in bold, with what each score means and a chart per score.
- **Compare** (`/compare`): any metric and one score of the six predictors (AUROC unless another is picked) across experiments, one value per seed, filterable and grouped by data or classifier.

## Method

Two steps, together with the classifier whose errors they are fitted on. The method assumes nothing about the classifier or the number of classes; it requires only tabular features.

1. **Identify regions.** The regions are the leaves of a tree that divides the rows by the class the classifier predicts for them: one root per predicted class, split by 2-means in the LDA projection of the space plus one axis, the share of each row's nearest train rows whose label differs from its predicted class. The space, the projection and those neighbours are built from the training data; the tree itself is grown on one half of the validation split, the grown half. A deeper split hides less of the spread of the classifier's errors on the grown half inside each leaf's average, but leaves each leaf fewer rows of the other half, the estimation half, on which its failure rate is estimated, so the estimate carries more sampling noise; the tree is cut where the second outweighs the first. Each row's failure probability is its leaf's estimation-half rate, shrunk toward the leaf's grown-half rate, then corrected by the classifier's confidence in the row's prediction.
2. **Describe each leaf.** A leaf holds rows of several classes, so each of its (leaf, class) populations is scored on how separable it is from the nearest populations of another class, and the leaf takes the mean of its populations' scores, weighted by their rows. The same is done at class level. The descriptors read only the features and labels of the grown half of validation; the classifier enters only through the leaves they are measured on. They are computed by `complexity`, which runs only when a target names it.

On the test rows, `evaluate` then scores the tree's per-row probability and its leaf rate against four baselines built without the tree: the global validation failure rate, the calibrated confidence, the validation rate of the row's confidence bin within its predicted class, and a gradient boosting model of the classifier's errors on the grown half.

The classifier contributes four things to step 1: its predictions, which decide the root each row descends from and, against the labels of the nearest train rows, the tree's disagreement axis; its errors on the grown half of validation, which decide where the tree is cut; its errors on the estimation half, which set each leaf's failure rate; and its class probabilities, which correct each row's rate. That is what ties the estimate to the model rather than to the dataset.

- Every distance of the descriptors and their neighbour graph is measured in one euclidean space. The classifier keeps its own encoding of the categorical features: it is the object under study, not part of the geometry. Each row's nearest train rows, behind its disagreement, are found in the LDA projection of that space, and the tree splits that projection with the disagreement as one more axis, both with the euclidean distance.
- The tree is grown from the grown half of validation alone, its rows starting at the root of their predicted class; their labels enter only through the classifier's errors, which decide the cut and the line the leaf rates are shrunk toward. Every row of the estimation half and of test descends it from the root of its predicted class **without reference to its label**, as a row would at inference time, and a leaf's failure rate on a set of rows counts the classifier's mistakes on those that reach it, whatever their class. This keeps every score a genuine prediction rather than a restatement of labels already known.
- Failure rates are never counted on the classifier's predictions for its own training rows: it gets those right more often than new data, which would flatter every rate. Besides the classifier, train fits only the space, the LDA and the nearest neighbours behind the disagreement `d`; every error the method learns from is made by the one model of `classify` on validation rows it never trained on. Validation is the label budget, halved at random within each class (seeded by `seed`): the grown half grows and cuts the tree and trains the `gbm` baseline; the estimation half estimates each leaf's rate, fits the tree's per-row correction and calibrates `gbm`; the `val_rate`, `confidence` and `binned` baselines use all of validation. Test is ground truth only, read by `evaluate` to score the predictors (and by `classify` and `render` for the classifier's own scores). No test row's features or label shape the classifier, the tree's splits and cut or the descriptors, and no label of the estimation half shapes the tree's splits or cut (only how many of its rows reach each node enters the cut).
- The analysis works per leaf rather than per class, since one class usually occupies several separate areas of the space. Class-level descriptors are kept as a coarse reference.

### What the descriptors measure

Five families, all measured in that space on the rows of the grown half of validation, (leaf, class) population by population; a leaf's value of a key is the mean over its populations, weighted by their rows. The neighbourhood and network families read the graph `complexity` builds: its nodes are a uniform sample of the grown half, all of it up to `graph.max_samples` rows, joined by their `graph.k` nearest-neighbour graph and a spanning tree over it. Above that size a population is measured on at most `complexity.max_sample_per_population` of its rows and a class on `complexity.max_sample_per_class`. A population with fewer than `complexity.min_population` rows is not measured and is nobody's rival; a rival is a population of another class. The feature and neighbourhood families and `network_density` set a population against the `complexity.top_k_clusters` rival populations whose centres lie nearest to it, reported as min / mean / max across them.

| Family | Keys | What it captures |
|---|---|---|
| **F**: feature | `f1`–`f4` | how well individual features separate the population from its rivals |
| **N**: neighbourhood | `n1`–`n4` | how many of a population's neighbours belong to another class |
| **ND**: network | `network_density`, `cls_coef`, `hub` | how many neighbour links cross into a rival population |
| **T**: dimensionality | `t2`–`t4` | how many features there are relative to rows, and how many of them matter |
| **G**: geometry | `max_dispersion`, `p95_dispersion`, `dist_to_nearest_rival`, `p5_silhouette`, `frac_at_risk` | how widely the population is spread, and how close the nearest rival lies |

### Inside the synthetic dataset

Each class consists of a clean core together with a *difficulty ladder* running towards every class it is meant to overlap with (`_OVERLAPS` in [generate_synthetic.py](pipeline/generate_synthetic.py)). Every rung sits a fixed distance from the boundary between the two classes, and both sides face one another symmetrically, so the best error rate any classifier could reach on a rung is known in advance:

| Sub-group | Distance from boundary | Best possible error | Rows |
|---|---|---|---|
| `canonical` | none | ≈ 0 % | 23,966 |
| `clear` | 1.80 σ | ≈ 4 % | 19,316 |
| `evasive` | 0.90 σ | ≈ 18 % | 16,560 |
| `mimicry` | 0.25 σ | ≈ 40 % | 9,658 |

The file has 69,500 rows, 10 classes, 20 numerical and 2 categorical features; the smallest class has 500 rows, well above `min_class_count`, so all 10 are kept. The two categorical features fade out along the same ladder, so they cease to help exactly where the numbers begin to overlap. Each row's intended difficulty is recorded in the `true_subgroup` column so that you can check afterwards; the pipeline never reads it, and `split` keeps only the columns the configuration lists. Accuracy settles at about 0.85 by design: the hard rungs are genuinely hard, and that spread is what makes a per-leaf correlation meaningful.

## Datasets

[configs/data/](configs/data/) has one YAML per dataset: `path`, `label_col`, `num_cols`, `cat_cols`, `train_frac`, `val_frac`, `test_frac`, `min_class_count` (20 in every shipped YAML), `max_per_class` (the cap on each class of train that classify fits on, `null` for none), `top_n` and `hash_buckets`. The four network datasets (`nb15_v2`, `cic_2018_v2`, `ton_iot_v2`, `bot_iot_v2`) are read from `resources/raw_data/dataset_v2/`, the five general benchmarks (`bank_marketing`, `covertype`, `letter_recognition`, `statlog_landsat_satellite`, `thyroid_disease`) from `resources/raw_data/<name>/`, and the synthetic one from `resources/raw_data/synthetic/`. The benchmark CSVs are not in the repository.

To add a dataset, place the CSV under `resources/raw_data/`, copy a YAML from `configs/data/`, set `path` and the columns, and run with `data=<name>`.

Every shipped configuration splits 40 % train, 30 % validation, 30 % test. No classifier trains on validation: its labels are the tree's budget, one half cutting the tree it grows and the other estimating the leaf rates (see [Method](#method)), and they fit the tree's per-row correction and the baselines. The test split is held out as ground truth: its labels score the classifier and every failure predictor, and fit nothing. Measured on `nb15_v2` (2.39 million rows) with random_forest and `runtime.n_jobs=16`: about 6.5 minutes end to end, peak memory about 10 GB.

## Extending

- **A classifier**: an entry in `CLASSIFIERS` in [src/ml.py](src/ml.py) (the estimator and the categorical strategy in front of it: a key of `_STRATEGIES`, or None to pass the prepared frame as it is) and a `configs/classifier/<name>.yaml` with `kind`, `name`, `params` and `grid`. If it takes `n_jobs`, add it to `PARALLEL` in the same file.
- **A DL model**: an entry in `MODELS` in [src/dl.py](src/dl.py), a class whose constructor takes `num_classes`, `num_numerical_features`, `cardinalities` and the YAML `params`, and whose `forward(x_num, x_cat)` returns logits, each row's independent of the others in eval mode (prediction runs in chunks); and a YAML with `kind: dl`.
- **A descriptor**: return one more key from a family function in [src/complexity/](src/complexity/); it reaches the complexity table and the leaf page of the dashboard by itself. A new family is one call, merged into the returned dict, in `src/complexity/measure.py`.
- **A stage**: a module in `stages/` with `CONFIG` (or `config_keys`), `INPUTS` (or `inputs`), `FILES`, `OUTPUTS` and `run(ctx)`; insert it into `STAGES` in [pipeline/runner.py](pipeline/runner.py) after its inputs and add its tables to `SCHEMA` in [pipeline/db.py](pipeline/db.py). Its tables start with `execution_id INTEGER NOT NULL REFERENCES executions (id)`, filled by the runner. Read upstream rows through `ctx.read`, `ctx.features`, `ctx.labels` and `ctx.split_rows`, and upstream files from `ctx.inputs[<stage>].dir`; write rows only through `ctx.write`, `ctx.save_metrics` and, for figures, `ctx.save_figures` (the stage must declare `figure_format` in `CONFIG`). Add it to `STAGE_ORDER` in [dashboard/queries.py](dashboard/queries.py) too (a test keeps it equal to `STAGES`); then every metric it saves is listed on the Experiment and Compare pages by itself. A metric becomes a column of the Experiments page through `HEADLINE` in [dashboard/summary.py](dashboard/summary.py); what a run page shows is built per tab in [dashboard/tabs.py](dashboard/tabs.py) (`TABS` and `build`), with a `run_<tab>.html` template.

## Development

```bash
make help                    # the targets
make test                    # the pytest suite, about 4-5 minutes
make smoke MODE=quick        # every stage on synthetic_test into resources/verify, with its own DB
make run DATA=nb15_v2 CLASSIFIER=mlp   # the pipeline up to render, without complexity, into resources
make sweep                   # the same for every dataset of configs/data and every classifier
```

`make smoke` runs the pipeline twice; the second pass must be fully cached. `MODE=full` sweeps all ten classifiers. `make run` takes `DATA`, `CLASSIFIER` and `ROOT` (default `resources`). `make sweep` runs one `render` per dataset and classifier, goes on past a failed one (a dataset whose raw file is missing, say) and lists the failures at the end with a non-zero exit; `DATASETS=a,b` and `CLASSIFIERS=c,d` narrow it. Code style: PEP 8 with Google docstrings (Args/Returns/Raises), black and isort (`make format` runs both with profile black; 88 columns; 72 for comments and docstrings), absolute imports; `src/` is pure (no config loading, no I/O, no paths) and the stages do the I/O.

## Repository layout

```
configs/                 YAML groups and config.yaml
src/                     pure topic modules: data, space, graph, grouping, tree,
                         complexity/, ml, dl, baselines, scores, plot/
stages/                  one module per stage, the stage contract (Context), CLI
pipeline/                runner (DAG, cache), config, db, the synthetic dataset generator
dashboard/               read-only FastAPI app: queries, experiments, run tabs, charts,
                         templates, static
tests/                   end-to-end, cache, DB, routing and dashboard tests, and src/
                         functions on toy data
Makefile                 shortcuts
```
