"""Every complexity family for every population (region or class)."""

from __future__ import annotations

__all__ = ["MEASURES", "compute_population_complexity", "nearest_rivals"]

from collections.abc import Mapping, Sequence
from itertools import chain

import numpy as np
from sklearn.metrics import pairwise_distances

from src.complexity.common import MeasuredSample, summary_keys
from src.complexity.dimensionality import compute_t_measures
from src.complexity.feature import F_KEYS, compute_f_measures
from src.complexity.geometry import GEOMETRY_KEYS, compute_geometry_measures
from src.complexity.linearity import L_KEYS, compute_l_measures
from src.complexity.neighborhood import N_KEYS, compute_n_measures
from src.complexity.network import compute_network_measures
from src.graph import Graph

MEASURES: tuple[str, ...] = (
    *chain.from_iterable(map(summary_keys, F_KEYS)),
    *chain.from_iterable(map(summary_keys, N_KEYS)),
    *summary_keys("network_density"),
    "cls_coef",
    "hub",
    "rival_classes",
    "rival_entropy",
    "t2",
    "t3",
    "t4",
    "anisotropy",
    *GEOMETRY_KEYS,
    *L_KEYS,
)


def nearest_rivals(
    ids: Sequence[int], classes: np.ndarray, distances: np.ndarray, top_k: int
) -> dict[int, list[int]]:
    """Return each population's `top_k` nearest rivals by centroid.

    Args:
        ids: The population ids, in the row order of `distances`.
        classes: The class of every population in `ids`.
        distances: The centroid distances, shape (n, n).
        top_k: The most rivals kept per population.

    Returns:
        The rival ids of every population, nearest first.
    """
    rivals: dict[int, list[int]] = {}
    for i, population_id in enumerate(ids):
        rival_positions = np.where(classes != classes[i])[0]
        order = np.argsort(distances[i, rival_positions])
        rivals[population_id] = [ids[int(rival_positions[j])] for j in order[:top_k]]
    return rivals


def compute_population_complexity(
    graph: Graph,
    graph_population: np.ndarray,
    sample: MeasuredSample,
    *,
    population_class: Mapping[int, int],
    centroids: Mapping[int, np.ndarray],
    sizes: Mapping[int, int],
    top_k_clusters: int,
    metric: str,
    silhouette_max_samples: int,
    silhouette_min_per_cluster: int,
    random_state: int,
) -> dict[int, dict[str, float | None]]:
    """Return every complexity family for each population.

    Args:
        graph: The graph of the train rows.
        graph_population: The population id of every graph node.
        sample: The sampled rows with their graph neighbours.
        population_class: The class of every population id.
        centroids: The centroid of every population id.
        sizes: The train rows of every population id.
        top_k_clusters: The nearest rivals per population.
        metric: The distance of the clustering space.
        silhouette_max_samples: The silhouette subsample cap.
        silhouette_min_per_cluster: The silhouette floor per cluster.
        random_state: The seed of the silhouette subsample and of the
            points the L3 measure interpolates.

    Returns:
        The F, N, ND, T, G and L measures by population id, ascending.
    """
    ids = list(population_class)
    centroid_matrix = np.stack(
        [
            np.asarray(centroids[population_id], dtype=np.float64)
            for population_id in ids
        ]
    )
    classes = np.array(
        [population_class[population_id] for population_id in ids], dtype=np.int64
    )
    centroid_distances = pairwise_distances(centroid_matrix, metric=metric)
    np.fill_diagonal(centroid_distances, np.inf)
    rivals = nearest_rivals(ids, classes, centroid_distances, top_k=top_k_clusters)

    feature_overlap = compute_f_measures(
        sample.X, sample.population, rivals, metric=metric
    )
    neighborhood = compute_n_measures(sample, graph_population, graph.mst, rivals)
    network = compute_network_measures(
        sample,
        graph_population,
        graph.knn_idx,
        rivals,
        population_class=population_class,
        graph_class=np.array(
            [population_class[population_id] for population_id in graph_population]
        ),
    )
    dimensionality = compute_t_measures(
        sample.X, sample.population, sizes, metric=metric
    )
    geometry = compute_geometry_measures(
        sample.X,
        sample.population,
        ids=ids,
        rivals=rivals,
        centroid_matrix=centroid_matrix,
        classes=classes,
        centroid_distances=centroid_distances,
        metric=metric,
        silhouette_max_samples=silhouette_max_samples,
        silhouette_min_per_cluster=silhouette_min_per_cluster,
        random_state=random_state,
    )
    linearity = compute_l_measures(
        sample.X, sample.population, rivals, metric=metric, random_state=random_state
    )

    return {
        population_id: {
            **feature_overlap[population_id],
            **neighborhood[population_id],
            **network[population_id],
            **dimensionality[population_id],
            **geometry[population_id],
            **linearity[population_id],
        }
        for population_id in sorted(population_class)
    }
