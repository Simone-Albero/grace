"""Per-class clustering at the granularity of least hardness loss."""

__all__ = [
    "ALGORITHMS",
    "KdnNodes",
    "assign_nearest_centroid",
    "cluster_class",
    "cluster_size_balance",
    "compute_centroids",
    "subsample_indices",
]

import itertools
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from inspect import signature

import hdbscan
import numpy as np
from sklearn.cluster import Birch, KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from src.space import l2_normalize

logger = logging.getLogger(__name__)

_ASSIGN_ROWS = 50_000
_PREDICT_CHUNK = 200_000
_GRANULARITY_LEVELS = 7


@dataclass(frozen=True)
class KdnNodes:
    """The graph nodes in the clustering space.

    The nodes are a uniform sample of train, where kDN counts
    neighbours.

    Attributes:
        X: The node points, shape (n_nodes, d).
        y: The class of every node.
        row_ids: The train row position of every node.
    """

    X: np.ndarray
    y: np.ndarray
    row_ids: np.ndarray


def _compute_kdn(
    X: np.ndarray, *, row_ids: np.ndarray, label: object, nodes: KdnNodes, k: int
) -> np.ndarray:
    """Return the share of each row's k nearest nodes of another class.

    The row's own node is excluded.
    """
    effective_k = min(k, nodes.X.shape[0] - 1)
    if effective_k < 1:
        raise ValueError(
            f"_compute_kdn: {nodes.X.shape[0]} node(s) leave no "
            "neighbour once a row's own copy is excluded."
        )
    index = NearestNeighbors(n_neighbors=effective_k + 1).fit(nodes.X)
    _, neighbor_positions = index.kneighbors(X)
    neighbor_ids = nodes.row_ids[neighbor_positions]
    neighbor_labels = nodes.y[neighbor_positions]

    self_mask = neighbor_ids == row_ids[:, np.newaxis]
    has_self = self_mask.any(axis=1)
    # Drop the row's own node when it has one, else the farthest
    # neighbour, so every row keeps exactly `effective_k` votes.
    drop = np.where(has_self, self_mask.argmax(axis=1), effective_k)
    keep = np.ones((len(X), effective_k + 1), dtype=bool)
    keep[np.arange(len(X)), drop] = False

    kept_labels = neighbor_labels[keep].reshape(len(X), effective_k)
    return (kept_labels != label).sum(axis=1) / effective_k


def cluster_size_balance(labels: np.ndarray) -> float:
    """Return the normalized entropy of the cluster sizes.

    Args:
        labels: The cluster label of every row.

    Returns:
        A value in [0, 1], 1 for equal sizes and 0 below two clusters.
    """
    _, counts = np.unique(labels, return_counts=True)
    n_clusters = counts.size
    if n_clusters < 2:
        return 0.0
    shares = counts / counts.sum()
    entropy = float(-(shares * np.log(shares)).sum())
    return entropy / float(np.log(n_clusters))


def subsample_indices(
    n_rows: int, *, max_samples: int, random_state: int
) -> np.ndarray:
    """Return the indices of a random subsample of `n_rows` rows.

    Args:
        n_rows: The number of rows to sample from.
        max_samples: The most rows kept; all of them when `n_rows` is
            not above it.
        random_state: The seed of the draw.

    Returns:
        The indices, ascending when all rows are kept.
    """
    if n_rows <= max_samples:
        return np.arange(n_rows)
    rng = np.random.default_rng(random_state)
    return rng.choice(n_rows, size=max_samples, replace=False)


def _subsample_features(
    X: np.ndarray,
    *,
    max_samples: int,
    random_state: int,
) -> np.ndarray:
    """Return a random subsample of X, unchanged when small enough."""
    rows = subsample_indices(
        X.shape[0], max_samples=max_samples, random_state=random_state
    )
    return X[rows]


def assign_nearest_centroid(
    X: np.ndarray,
    centroids: Mapping[int, np.ndarray],
    *,
    metric: str,
) -> np.ndarray:
    """Assign each row to the nearest centroid.

    Args:
        X: The points, shape (n, d).
        centroids: The centroid vector of every cluster id.
        metric: A `pairwise_distances` metric, such as `euclidean`.

    Returns:
        The cluster id of every row.
    """
    items = [(int(label), centroid) for label, centroid in centroids.items()]
    centroid_ids = np.array([label for label, _ in items], dtype=np.int64)
    centroid_matrix = np.array(
        [np.asarray(centroid, dtype=np.float64) for _, centroid in items],
        dtype=np.float64,
    )
    assigned = np.empty(len(X), dtype=np.int64)
    for start in range(0, len(X), _ASSIGN_ROWS):
        batch = X[start : start + _ASSIGN_ROWS]
        distances = pairwise_distances(batch, centroid_matrix, metric=metric)
        assigned[start : start + _ASSIGN_ROWS] = centroid_ids[distances.argmin(axis=1)]
    return assigned


def compute_centroids(
    X: np.ndarray, labels: np.ndarray, *, metric: str
) -> dict[int, np.ndarray]:
    """Return the mean feature vector of every label.

    Args:
        X: The points, shape (n, d).
        labels: The cluster label of every row.
        metric: `cosine` renormalizes each mean to a unit direction.

    Returns:
        The centroid by label.
    """
    centroids = {}
    for label in np.unique(labels):
        centroid = X[labels == label].mean(axis=0)
        # Under cosine a region's centre is a direction: the mean of the
        # unit vectors it was clustered on, renormalized.
        if metric == "cosine":
            centroid = l2_normalize(centroid[np.newaxis])[0]
        centroids[int(label)] = centroid
    return centroids


def _merge_small_clusters(
    X: np.ndarray, labels: np.ndarray, *, min_size: float, metric: str
) -> tuple[np.ndarray, int, int]:
    """Fold noise and small clusters into the nearest survivor.

    Every cluster under `min_size` rows is small.

    Returns:
        The merged labels, the number of clusters absorbed and the
        number of rows moved.
    """
    ids, counts = np.unique(labels[labels != -1], return_counts=True)
    small_ids = ids[counts < min_size]
    absorbed = np.isin(labels, small_ids) | (labels == -1)
    n_merged_clusters = int(small_ids.size)
    n_merged = int(absorbed.sum())
    if not absorbed.any():
        return labels, 0, 0

    survivor_mask = ~absorbed
    if not survivor_mask.any():
        # No cluster reaches the floor: the class becomes one region.
        return (
            np.zeros(labels.shape[0], dtype=labels.dtype),
            n_merged_clusters,
            n_merged,
        )

    centroids = compute_centroids(
        X[survivor_mask], labels[survivor_mask], metric=metric
    )
    merged = labels.copy()
    merged[absorbed] = assign_nearest_centroid(X[absorbed], centroids, metric=metric)
    return merged, n_merged_clusters, n_merged


def _fit_hdbscan(
    X: np.ndarray,
    *,
    min_cluster_size: int = 50,
    min_samples: int | None = None,
    cluster_selection_method: str = "leaf",
    max_fit_samples: int,
    random_state: int,
) -> np.ndarray:
    """Fit HDBSCAN (Euclidean) and label every row, noise as -1.

    Above `max_fit_samples` rows the model is fitted on a subsample and
    the rest is labelled by `hdbscan.approximate_predict`.

    Args:
        X: The points, shape (n, d).
        min_cluster_size: The smallest cluster HDBSCAN keeps.
        min_samples: The neighbours behind a core distance; `None` uses
            `min_cluster_size`.
        cluster_selection_method: `leaf` or `eom`.
        max_fit_samples: The most rows the model is fitted on.
        random_state: The seed of the subsample.

    Returns:
        The labels, shape (n,).
    """
    n_rows = X.shape[0]

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        metric="euclidean",
        prediction_data=True,
    )

    if n_rows > max_fit_samples:
        X_sub = _subsample_features(
            X, max_samples=max_fit_samples, random_state=random_state
        )
        clusterer.fit(X_sub)
        labels = np.concatenate(
            [
                hdbscan.approximate_predict(
                    clusterer, X[start : start + _PREDICT_CHUNK]
                )[0]
                for start in range(0, n_rows, _PREDICT_CHUNK)
            ]
        )
    else:
        clusterer.fit(X)
        labels = clusterer.labels_

    return labels


def _clamp_clusters(n_clusters: int, n_rows: int, max_fit_samples: int) -> int:
    """Bound `n_clusters` by the rows the model is fitted on.

    The sweep scored its grid on those rows, so a larger count was never
    tried.
    """
    return max(2, min(int(n_clusters), min(n_rows, max_fit_samples) - 1))


def _fit_kmeans(
    X: np.ndarray,
    *,
    n_clusters: int = 8,
    max_fit_samples: int,
    random_state: int,
) -> np.ndarray:
    """Fit K-means on at most `max_fit_samples` rows, label every row.

    Args:
        X: The points, shape (n, d).
        n_clusters: The number of clusters, clamped to the rows fitted.
        max_fit_samples: The most rows the model is fitted on.
        random_state: The seed of the subsample and the k-means start.

    Returns:
        The labels, shape (n,).
    """
    n_rows = X.shape[0]
    n_clusters = _clamp_clusters(n_clusters, n_rows, max_fit_samples)
    X = np.ascontiguousarray(X, dtype=np.float64)
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    if n_rows > max_fit_samples:
        X_sub = _subsample_features(
            X, max_samples=max_fit_samples, random_state=random_state
        )
        model.fit(X_sub)
        return model.predict(X)
    return model.fit_predict(X)


def _fit_birch(
    X: np.ndarray,
    *,
    n_clusters: int = 8,
    threshold: float = 0.5,
    branching_factor: int = 50,
    max_fit_samples: int,
    random_state: int,
) -> np.ndarray:
    """Fit BIRCH with `n_clusters` (agglomerative on CF-tree leaves).

    Args:
        X: The points, shape (n, d).
        n_clusters: The number of clusters, clamped to the rows fitted.
        threshold: The radius a subcluster may span in the CF tree.
        branching_factor: The most subclusters per CF-tree node.
        max_fit_samples: The most rows the model is fitted on.
        random_state: The seed of the subsample.

    Returns:
        The labels, shape (n,).
    """
    n_rows = X.shape[0]
    n_clusters = _clamp_clusters(n_clusters, n_rows, max_fit_samples)
    X = np.ascontiguousarray(X, dtype=np.float64)
    clusterer = Birch(
        threshold=threshold,
        branching_factor=branching_factor,
        n_clusters=n_clusters,
    )
    if n_rows > max_fit_samples:
        X_sub = _subsample_features(
            X, max_samples=max_fit_samples, random_state=random_state
        )
        clusterer.fit(X_sub)
        labels = clusterer.predict(X)
    else:
        clusterer.fit(X)
        labels = clusterer.labels_
    return labels


def _fit_spectral(
    X: np.ndarray,
    *,
    n_clusters: int = 8,
    n_neighbors: int = 10,
    max_fit_samples: int,
    random_state: int,
) -> np.ndarray:
    """Cluster spectrally on the k-NN graph, labelling every row.

    Above `max_fit_samples` rows the model is fitted on a subsample and
    the rest takes the label of its nearest sampled row.

    Args:
        X: The points, shape (n, d).
        n_clusters: The number of clusters, clamped to the rows fitted.
        n_neighbors: The neighbours of the affinity graph.
        max_fit_samples: The most rows the model is fitted on.
        random_state: The seed of the subsample, the ARPACK start vector
            and the k-means label assignment.

    Returns:
        The labels, shape (n,).
    """
    n_rows = X.shape[0]
    n_clusters = _clamp_clusters(n_clusters, n_rows, max_fit_samples)
    X = np.ascontiguousarray(X, dtype=np.float64)
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=int(n_neighbors),
        assign_labels="kmeans",
        random_state=random_state,
        eigen_solver="arpack",
    )
    if n_rows <= max_fit_samples:
        return model.fit_predict(X)

    X_sub = _subsample_features(
        X, max_samples=max_fit_samples, random_state=random_state
    )
    sub_labels = model.fit_predict(X_sub)
    nearest = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(X_sub)
    _, nearest_sample = nearest.kneighbors(X, n_neighbors=1, return_distance=True)
    return sub_labels[nearest_sample.ravel()]


ALGORITHMS: dict[str, Callable[..., np.ndarray]] = {
    "kmeans": _fit_kmeans,
    "hdbscan": _fit_hdbscan,
    "birch": _fit_birch,
    "spectral": _fit_spectral,
}


def _granularity_loss(
    labels: np.ndarray,
    hardness: np.ndarray,
    *,
    n_class: int,
    eval_rows_per_train_row: float,
) -> float:
    """Return the squared error of reading hardness off region rates.

    The loss is the sum of two terms: `within` is the hardness variance
    a region's mean hides, `noise` the sampling noise of a rate
    measured on the test rows the region will receive, at its
    full-class size. A finer partition lowers the first and raises the
    second.
    """
    ids, counts = np.unique(labels, return_counts=True)
    scale = n_class / labels.shape[0]
    weight = counts / counts.sum()
    groups = [hardness[labels == cluster] for cluster in ids]
    mean_hardness = np.array([group.mean() for group in groups])
    variance = np.array(
        [group.var(ddof=1) if group.size > 1 else np.nan for group in groups]
    )
    # A one-row region stands for about `scale` rows of the class whose
    # spread is unknown, not zero: it takes the spread of the regions
    # that have one.
    known = ~np.isnan(variance)
    pooled = (
        np.average(variance[known], weights=counts[known])
        if known.any()
        else hardness.var(ddof=1)
    )
    within = float((weight * np.where(known, variance, pooled)).sum())
    noise = float(
        (
            weight
            * np.maximum(mean_hardness * (1 - mean_hardness), 1e-3)
            / np.maximum(counts * scale * eval_rows_per_train_row, 1e-9)
        ).sum()
    )
    return within + noise


def _score_candidates(
    X_sub: np.ndarray,
    fit: Callable[..., np.ndarray],
    candidates: Sequence[Mapping[str, object]],
    *,
    label: object,
    hardness: np.ndarray,
    n_class: int,
    scaled_floor: float,
    eval_rows_per_train_row: float,
    metric: str,
) -> list[tuple[dict[str, object], np.ndarray]]:
    """Fit and score every candidate on the subsample.

    A candidate whose fit fails is skipped and logged, except on a
    TypeError, which is a configuration error and stops the sweep.

    Returns:
        The summary and merged labels of each candidate that fitted.

    Raises:
        RuntimeError: If every candidate failed to fit.
    """
    scored: list[tuple[dict[str, object], np.ndarray]] = []
    failures: list[tuple[Mapping[str, object], Exception]] = []

    for combo in tqdm(candidates, desc="Granularity"):
        try:
            labels = fit(X_sub, **combo)
        except TypeError:
            # A call the algorithm cannot take is a configuration error,
            # not a degenerate candidate: stop the sweep, do not pick
            # among the survivors.
            raise
        except Exception as exc:
            failures.append((combo, exc))
            continue

        labels, n_merged_clusters, n_merged = _merge_small_clusters(
            X_sub, labels, min_size=scaled_floor, metric=metric
        )
        loss = _granularity_loss(
            labels,
            hardness,
            n_class=n_class,
            eval_rows_per_train_row=eval_rows_per_train_row,
        )
        entry = {
            "combo": combo,
            "n_clusters": int(np.unique(labels).size),
            "n_merged_clusters": n_merged_clusters,
            "n_merged": n_merged,
            "loss": loss,
        }
        scored.append((entry, labels))

    if failures:
        first_combo, first_exc = failures[0]
        logger.warning(
            "_score_candidates: %d of %d candidates failed and were skipped "
            "(first: %s, %r)",
            len(failures),
            len(failures) + len(scored),
            first_combo,
            first_exc,
        )

    if not scored:
        raise RuntimeError(
            f"All {len(candidates)} candidate(s) failed to cluster class {label}."
        ) from failures[0][1]
    return scored


def _select_granularity(
    X: np.ndarray,
    fit: Callable[..., np.ndarray],
    candidates: Sequence[Mapping[str, object]],
    *,
    row_ids: np.ndarray,
    label: object,
    nodes: KdnNodes,
    hardness_k: int,
    eval_rows_per_train_row: float,
    min_cluster_floor: int,
    max_clusters: int,
    metric: str,
    max_fit_samples: int,
    random_state: int,
) -> tuple[dict[str, object], np.ndarray | None]:
    """Return the scored entry of least loss and its labels.

    The candidates are keyword arguments of `fit`. The labels are None
    when the class was subsampled, since they then cover only the
    sample.
    """
    scored_rows = subsample_indices(
        X.shape[0], max_samples=max_fit_samples, random_state=random_state
    )
    X_sub = X[scored_rows]
    n_class = X.shape[0]
    scale = n_class / X_sub.shape[0]
    scaled_floor = min_cluster_floor / scale
    hardness = _compute_kdn(
        X_sub, row_ids=row_ids[scored_rows], label=label, nodes=nodes, k=hardness_k
    )

    scored = _score_candidates(
        X_sub,
        fit,
        candidates,
        label=label,
        hardness=hardness,
        n_class=n_class,
        scaled_floor=scaled_floor,
        eval_rows_per_train_row=eval_rows_per_train_row,
        metric=metric,
    )

    eligible = [
        (entry, labels)
        for entry, labels in scored
        if entry["n_clusters"] <= max_clusters
    ]
    if not eligible:
        raise ValueError(
            f"No candidate partition of max_clusters={max_clusters} or fewer survives "
            "merging undersized regions. Raise clustering.max_regions or tighten the "
            "clustering grid."
        )

    best_entry, best_labels = min(eligible, key=lambda pair: pair[0]["loss"])

    # Without subsampling, the sweep's fit of the winner is the refit.
    subsampled = n_class > max_fit_samples
    return best_entry, None if subsampled else best_labels


def _n_clusters_grid(n_class: int, target_size: int, max_n_clusters: int) -> list[int]:
    """Return `n_clusters` candidates, from a geometric band of sizes.

    The band starts at `target_size` rows per cluster and doubles at
    each level.
    """
    sizes = (target_size * (2**level) for level in range(_GRANULARITY_LEVELS))
    candidates = {min(max_n_clusters, max(2, round(n_class / size))) for size in sizes}
    return sorted(candidates)


def cluster_class(
    X: np.ndarray,
    *,
    row_ids: np.ndarray,
    label: int,
    algorithm: str,
    params: Mapping[str, object],
    nodes: KdnNodes,
    max_fit_samples: int,
    max_clusters: int,
    min_cluster_floor: int,
    hardness_k: int,
    eval_rows_per_train_row: float,
    metric: str,
    random_state: int,
) -> tuple[np.ndarray, int, int, float]:
    """Cluster one class at the granularity of least loss.

    List-valued `params` are swept and the others fixed; `n_clusters`
    is swept too when the algorithm takes it.

    Args:
        X: The class's points, shape (n, d).
        row_ids: The class's row positions in the train split.
        label: The class label.
        algorithm: The key of `ALGORITHMS`.
        params: The algorithm's keyword parameters.
        nodes: The graph nodes the kDN hardness is counted on.
        max_fit_samples: The most rows a candidate is fitted on.
        max_clusters: The most clusters the class may keep.
        min_cluster_floor: The finest candidate is about the class size
            over this; also the merge floor for undersized clusters.
        hardness_k: The neighbours behind the kDN hardness.
        eval_rows_per_train_row: Evaluation rows per train row.
        metric: `euclidean` or `cosine`.
        random_state: The seed.

    Returns:
        The cluster labels, the merged cluster count, the merged row
        count and the loss of the chosen granularity.

    Raises:
        ValueError: If no candidate of at most `max_clusters` clusters
            survives the merge of undersized clusters.
        RuntimeError: If every candidate fails to fit.
    """
    fit_fn = ALGORITHMS[algorithm]
    fixed = {
        name: value for name, value in params.items() if not isinstance(value, list)
    }
    grid = {name: value for name, value in params.items() if isinstance(value, list)}
    if "n_clusters" in signature(fit_fn).parameters:
        max_n_clusters = max(2, min(max_fit_samples // min_cluster_floor, max_clusters))
        grid["n_clusters"] = _n_clusters_grid(
            X.shape[0], min_cluster_floor, max_n_clusters
        )
    fit = partial(
        fit_fn, max_fit_samples=max_fit_samples, random_state=random_state, **fixed
    )
    candidates = [
        dict(zip(grid, values)) for values in itertools.product(*grid.values())
    ]
    best, best_labels = _select_granularity(
        X,
        fit,
        candidates,
        row_ids=row_ids,
        label=label,
        nodes=nodes,
        hardness_k=hardness_k,
        eval_rows_per_train_row=eval_rows_per_train_row,
        min_cluster_floor=min_cluster_floor,
        max_clusters=max_clusters,
        metric=metric,
        max_fit_samples=max_fit_samples,
        random_state=random_state,
    )
    if best_labels is not None:
        return best_labels, best["n_merged_clusters"], best["n_merged"], best["loss"]
    refit_labels = fit(X, **best["combo"])
    # Refit on the full class: the sweep's merge counts described the
    # scored subsample, at a floor scaled down to match it, so they
    # don't describe this.
    final_labels, n_merged_clusters, n_merged = _merge_small_clusters(
        X, refit_labels, min_size=min_cluster_floor, metric=metric
    )
    return final_labels, n_merged_clusters, n_merged, best["loss"]
