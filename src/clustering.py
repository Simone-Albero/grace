"""Regions: each class clustered at its least-loss granularity.

A finer partition hides less of the spread of its rows' hardness inside
each region's mean, but leaves each region fewer rows to estimate that
mean on. The granularity loss adds the two; the candidate of least loss
wins. Hardness is kDN: the share of a row's nearest graph
nodes of another class.
"""

from __future__ import annotations

__all__ = [
    "ALGORITHMS",
    "assign_nearest",
    "candidate_grid",
    "centroids",
    "cluster_class",
    "fit_hdbscan",
    "fit_kmeans",
    "granularity_loss",
    "kdn",
    "merge_small_clusters",
    "n_clusters_grid",
    "takes_n_clusters",
    "tunable_params",
]

import inspect
import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import hdbscan
import numpy as np
from sklearn.cluster import Birch, KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from src.graph import sample_rows

FitFn = Callable[..., np.ndarray]
RESERVED = ("n_clusters", "max_fit_samples", "random_state")
GRANULARITY_LEVELS = 7
_ASSIGN_ROWS = 50_000  # rows assigned at once


def fit_kmeans(
    X: np.ndarray,
    *,
    n_clusters: int = 8,
    max_fit_samples: int,
    random_state: int,
) -> np.ndarray:
    """Return KMeans labels, fitted on at most max_fit_samples rows.

    Args:
        X: The points, shape (n, d).
        n_clusters: Clamped to [2, min(n, max_fit_samples) - 1].
        max_fit_samples: The most rows the fit sees (graph.sample_rows).
        random_state: The seed of the sample and of KMeans.

    Returns:
        The label of every row, >= 0.
    """
    n_clusters = _clamp(n_clusters, len(X), max_fit_samples)
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    if len(X) <= max_fit_samples:
        return model.fit_predict(X)
    return model.fit(X[sample_rows(len(X), max_fit_samples, random_state)]).predict(X)


def fit_birch(
    X: np.ndarray,
    *,
    n_clusters: int = 8,
    threshold: float = 0.5,
    branching_factor: int = 50,
    max_fit_samples: int,
    random_state: int,
) -> np.ndarray:
    """Return Birch labels, fitted on at most max_fit_samples rows.

    Args:
        X: The points, shape (n, d).
        n_clusters: Clamped as in fit_kmeans.
        threshold: Birch's subcluster radius.
        branching_factor: Birch's most subclusters per node.
        max_fit_samples: The most rows the fit sees.
        random_state: The seed of the sample.

    Returns:
        The label of every row, >= 0.
    """
    n_clusters = _clamp(n_clusters, len(X), max_fit_samples)
    model = Birch(
        threshold=threshold, branching_factor=branching_factor, n_clusters=n_clusters
    )
    if len(X) <= max_fit_samples:
        return model.fit_predict(X)
    return model.fit(X[sample_rows(len(X), max_fit_samples, random_state)]).predict(X)


def fit_spectral(
    X: np.ndarray,
    *,
    n_clusters: int = 8,
    n_neighbors: int = 10,
    max_fit_samples: int,
    random_state: int,
) -> np.ndarray:
    """Return spectral labels, fitted on at most max_fit_samples rows.

    The affinity is a k-NN graph. Rows outside the fitted sample take
    the label of their nearest sampled row.

    Args:
        X: The points, shape (n, d).
        n_clusters: Clamped as in fit_kmeans.
        n_neighbors: Neighbours of the affinity graph.
        max_fit_samples: The most rows the fit sees.
        random_state: The seed of the sample and of the embedding.

    Returns:
        The label of every row, >= 0.
    """
    n_clusters = _clamp(n_clusters, len(X), max_fit_samples)
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        assign_labels="kmeans",
        eigen_solver="arpack",
        random_state=random_state,
    )
    if len(X) <= max_fit_samples:
        return model.fit_predict(X)
    sample = X[sample_rows(len(X), max_fit_samples, random_state)]
    labels = model.fit_predict(sample)
    nearest = NearestNeighbors(n_neighbors=1).fit(sample).kneighbors(X)[1]
    return labels[nearest.ravel()]


def fit_hdbscan(
    X: np.ndarray,
    *,
    min_cluster_size: int = 50,
    min_samples: int | None = None,
    cluster_selection_method: str = "leaf",
    max_fit_samples: int,
    random_state: int,
) -> np.ndarray:
    """Return HDBSCAN labels, fitted on at most max_fit_samples rows.

    Rows outside the fitted sample are labelled by approximate_predict.

    Args:
        X: The points, shape (n, d).
        min_cluster_size: HDBSCAN's smallest cluster.
        min_samples: HDBSCAN's core neighbours; None is
            min_cluster_size.
        cluster_selection_method: "leaf" or "eom".
        max_fit_samples: The most rows the fit sees.
        random_state: The seed of the sample.

    Returns:
        The label of every row; -1 is noise.
    """
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        prediction_data=True,
    )
    if len(X) <= max_fit_samples:
        return model.fit_predict(X)
    model.fit(X[sample_rows(len(X), max_fit_samples, random_state)])
    return hdbscan.approximate_predict(model, X)[0]


ALGORITHMS: dict[str, FitFn] = {
    "kmeans": fit_kmeans,
    "birch": fit_birch,
    "spectral": fit_spectral,
    "hdbscan": fit_hdbscan,
}


def tunable_params(algorithm: FitFn) -> tuple[str, ...]:
    """Return the parameters of a fit function a config may set.

    Args:
        algorithm: A value of ALGORITHMS.

    Returns:
        Its keyword-only parameter names except RESERVED, in signature
        order.
    """
    return tuple(
        name
        for name, parameter in inspect.signature(algorithm).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY and name not in RESERVED
    )


def takes_n_clusters(algorithm: FitFn) -> bool:
    """Return whether a fit function takes n_clusters.

    Args:
        algorithm: A value of ALGORITHMS.

    Returns:
        Whether n_clusters is one of its parameters.
    """
    return "n_clusters" in inspect.signature(algorithm).parameters


def kdn(neighbor_labels: np.ndarray, label: int) -> np.ndarray:
    """Return the kDN hardness of rows of one class.

    Args:
        neighbor_labels: The class of each row's nearest graph nodes,
            shape (n, k), k >= 1.
        label: The class of the rows.

    Returns:
        The share of each row's neighbours of another class, in [0, 1].
    """
    return (neighbor_labels != label).mean(axis=1)


def centroids(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean point of every label.

    Args:
        X: The points, shape (n, d).
        labels: The label of every row; -1 rows are ignored.

    Returns:
        The labels, ascending, and their centroids, shape (m, d).
    """
    ids = np.unique(labels[labels != -1])
    centers = np.stack([X[labels == i].mean(axis=0) for i in ids])
    return ids, centers


def merge_small_clusters(
    X: np.ndarray, labels: np.ndarray, min_size: float
) -> np.ndarray:
    """Move noise and small clusters to the nearest surviving centroid.

    A cluster is small with fewer than min_size rows.

    Ids are not renumbered. When no cluster has min_size rows, every row
    gets label 0.

    Args:
        X: The points, shape (n, d).
        labels: The label of every row; -1 is noise.
        min_size: The fewest rows a cluster keeps.

    Returns:
        The new labels, all >= 0.
    """
    ids, counts = np.unique(labels[labels != -1], return_counts=True)
    absorbed = np.isin(labels, ids[counts < min_size]) | (labels == -1)
    if not absorbed.any():
        return labels.copy()
    if absorbed.all():
        return np.zeros_like(labels)
    survivors, centers = centroids(X[~absorbed], labels[~absorbed])
    merged = labels.copy()
    merged[absorbed] = survivors[assign_nearest(X[absorbed], centers)]
    return merged


def granularity_loss(labels: np.ndarray, hardness: np.ndarray, scale: float) -> float:
    """Return the loss of a partition of a class's scored rows.

    With w_c = n_c / N over the clusters c of the N scored rows, h_c
    their mean hardness and v_c its variance (ddof=1; the n_c-weighted
    mean of the known v_c for a single-row cluster, or of all rows when
    none is known):
        within = Σ w_c · v_c
        noise = Σ w_c · max(h_c (1 - h_c), 1e-3)
                / max(n_c · scale, 1e-9)
    and the loss is within + noise. n_c · scale estimates the rows of
    the region in the whole class.

    Args:
        labels: The final label of every scored row, all >= 0.
        hardness: The kDN of every scored row.
        scale: Rows of the class per scored row (n_class / N).

    Returns:
        The loss, >= 0.
    """
    clusters, sizes = np.unique(labels, return_counts=True)
    weights = sizes / len(labels)
    means = np.array([hardness[labels == c].mean() for c in clusters])
    variances = np.array(
        [
            hardness[labels == c].var(ddof=1) if n > 1 else np.nan
            for c, n in zip(clusters, sizes)
        ]
    )
    known = ~np.isnan(variances)
    if known.any():
        pooled = np.average(variances[known], weights=sizes[known])
    else:
        pooled = hardness.var(ddof=1) if len(hardness) > 1 else 0.0
    within = np.sum(weights * np.where(known, variances, pooled))
    spread = np.maximum(means * (1 - means), 1e-3)
    noise = np.sum(weights * spread / np.maximum(sizes * scale, 1e-9))
    return float(within + noise)


def n_clusters_grid(n_rows: int, floor: float, max_n: int) -> list[int]:
    """Return the n_clusters candidates of a class.

    Args:
        n_rows: The rows of the class.
        floor: The rows of the finest cluster.
        max_n: The most clusters.

    Returns:
        Sorted distinct values min(max_n, max(2, round(n_rows / (floor ·
        2**l)))) for l in range(GRANULARITY_LEVELS).
    """
    return sorted(
        {
            min(max_n, max(2, round(n_rows / (floor * 2**level))))
            for level in range(GRANULARITY_LEVELS)
        }
    )


def candidate_grid(
    algorithm: FitFn, params: Mapping[str, Any], n_clusters: Sequence[int]
) -> list[dict[str, Any]]:
    """Return every parameter combination to try.

    A list-valued param is swept, any other value fixed; when algorithm
    takes n_clusters, it is swept last over n_clusters.

    Args:
        algorithm: The fit function.
        params: The configured params.
        n_clusters: The n_clusters candidates (ignored when algorithm
            takes none).

    Returns:
        The combinations, in product order of params then n_clusters.
    """
    grid = {k: v if isinstance(v, list) else [v] for k, v in params.items()}
    if takes_n_clusters(algorithm):
        grid["n_clusters"] = list(n_clusters)
    return [
        dict(zip(grid, combination))
        for combination in itertools.product(*grid.values())
    ]


@dataclass(frozen=True)
class Candidate:
    """One scored combination of the grid of a class.

    Attributes:
        params: The combination.
        n_clusters: The clusters after the merge.
        loss: Its granularity loss.
    """

    params: dict[str, Any]
    n_clusters: int
    loss: float


@dataclass(frozen=True)
class ClassClustering:
    """The chosen partition of one class.

    Attributes:
        labels: The label of every row of the class, >= 0, gaps
            allowed.
        params: The winning combination.
        loss: Its loss on the scored rows.
        candidates: Every combination that fitted, in grid order.
    """

    labels: np.ndarray
    params: dict[str, Any]
    loss: float
    candidates: list[Candidate]


def cluster_class(
    X: np.ndarray,
    scored: np.ndarray,
    hardness: np.ndarray,
    *,
    algorithm: FitFn,
    params: Mapping[str, Any],
    max_fit_samples: int,
    min_cluster_floor: int,
    max_clusters: int,
    seed: int,
) -> ClassClustering:
    """Cluster one class at the granularity of least loss.

    Every combination of candidate_grid (n_clusters from n_clusters_grid
    with max_n = max(2, min(max_fit_samples // min_cluster_floor,
    max_clusters))) is fitted on X[scored], merged with floor
    min_cluster_floor · len(scored) / len(X), and scored by
    granularity_loss. A fit that raises anything but TypeError is
    skipped. The least loss among those with at most max_clusters
    clusters wins (ties: the first). When scored is every row, its
    labels are kept; otherwise algorithm is refit on X and merged with
    min_cluster_floor.

    Args:
        X: The points of the class, shape (n, d).
        scored: The rows the candidates are scored on (graph.sample_rows
            of n with max_fit_samples).
        hardness: The kDN of every scored row.
        algorithm: A value of ALGORITHMS.
        params: Its configured params.
        max_fit_samples: Passed to algorithm.
        min_cluster_floor: The merge floor on the whole class.
        max_clusters: The most clusters of the class.
        seed: Passed to algorithm as random_state.

    Returns:
        The chosen partition.

    Raises:
        RuntimeError: When every combination fails.
        ValueError: When no combination has at most max_clusters
            clusters.
    """
    n_rows = len(X)
    max_n = max(2, min(max_fit_samples // min_cluster_floor, max_clusters))
    grid = candidate_grid(
        algorithm, params, n_clusters_grid(n_rows, min_cluster_floor, max_n)
    )
    sample = X[scored]
    scale = n_rows / len(scored)
    candidates, candidate_labels = [], []
    for combination in grid:
        try:
            labels = algorithm(
                sample,
                max_fit_samples=max_fit_samples,
                random_state=seed,
                **combination,
            )
        # A TypeError is a bad param: a config error, which every other
        # candidate would repeat, not a candidate that failed to fit.
        except TypeError:
            raise
        except Exception as e:
            print(f"[regions] skipped {combination}: {e}", flush=True)
            continue
        labels = merge_small_clusters(sample, labels, min_cluster_floor / scale)
        loss = granularity_loss(labels, hardness, scale)
        candidates.append(Candidate(combination, len(np.unique(labels)), loss))
        candidate_labels.append(labels)
    if not candidates:
        raise RuntimeError(f"every one of the {len(grid)} clustering candidates failed")
    allowed = [
        i
        for i, candidate in enumerate(candidates)
        if candidate.n_clusters <= max_clusters
    ]
    if not allowed:
        raise ValueError(
            f"no candidate has at most max_clusters={max_clusters} clusters, "
            f"least: {min(c.n_clusters for c in candidates)}"
        )
    best = min(allowed, key=lambda i: candidates[i].loss)
    chosen = candidates[best]
    if len(scored) == n_rows:
        labels = candidate_labels[best]
    else:
        labels = algorithm(
            X, max_fit_samples=max_fit_samples, random_state=seed, **chosen.params
        )
        labels = merge_small_clusters(X, labels, min_cluster_floor)
    return ClassClustering(labels, chosen.params, chosen.loss, candidates)


def assign_nearest(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return the index of the nearest center of every point.

    Args:
        points: Shape (n, d).
        centers: Shape (m, d).

    Returns:
        Indices into centers (ties: the first), int64.
    """
    # In chunks: the rows against every centre may not fit in memory.
    return np.concatenate(
        [
            pairwise_distances(points[start : start + _ASSIGN_ROWS], centers)
            .argmin(axis=1)
            .astype(np.int64)
            for start in range(0, len(points), _ASSIGN_ROWS)
        ]
        or [np.empty(0, dtype=np.int64)]
    )


def _clamp(n_clusters: int, n_rows: int, max_fit_samples: int) -> int:
    """Bound n_clusters by the rows the model is fitted on."""
    return max(2, min(int(n_clusters), min(n_rows, max_fit_samples) - 1))
