"""Every complexity family for every population."""

__all__ = ["compute_population_complexity"]

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.metrics import pairwise_distances

from src.complexity.common import MeasuredSample
from src.complexity.dimensionality import compute_t_measures
from src.complexity.feature import compute_f_measures
from src.complexity.geometry import compute_geometry_measures
from src.complexity.neighborhood import compute_n_measures
from src.complexity.network import compute_network_measures
from src.graph import RowGraph


def _nearest_rivals(
    ids: Sequence[int], classes: np.ndarray, distances: np.ndarray, *, top_k: int
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
    graph: RowGraph,
    graph_population: np.ndarray,
    sample: MeasuredSample,
    *,
    population_class: Mapping[int, int],
    centroids: Mapping[int, np.ndarray],
    sizes: Mapping[int, int],
    top_k_clusters: int,
    silhouette_max_samples: int,
    silhouette_min_per_cluster: int,
    random_state: int,
) -> dict[int, dict[str, float | None]]:
    """Return every complexity family for each population.

    Args:
        graph: The graph of the rows.
        graph_population: The population id of every graph node.
        sample: The sampled rows with their graph neighbours.
        population_class: The class of every population id.
        centroids: The centroid of every population id.
        sizes: The rows of every population id.
        top_k_clusters: The nearest rivals per population.
        silhouette_max_samples: The silhouette subsample cap.
        silhouette_min_per_cluster: The silhouette floor per cluster.
        random_state: The seed of the silhouette subsample.

    Returns:
        The F, N, ND, T and G measures by population id, ascending.
    """
    ids = list(population_class)
    matrix = np.stack(
        [
            np.asarray(centroids[population_id], dtype=np.float64)
            for population_id in ids
        ]
    )
    classes = np.array(
        [population_class[population_id] for population_id in ids], dtype=np.int64
    )
    centroid_distances = pairwise_distances(matrix)
    np.fill_diagonal(centroid_distances, np.inf)
    rivals = _nearest_rivals(ids, classes, centroid_distances, top_k=top_k_clusters)

    feature_overlap = compute_f_measures(sample.X, sample.population, rivals)
    neighborhood = compute_n_measures(sample, graph_population, graph.mst_edges, rivals)
    network = compute_network_measures(sample, graph_population, graph.knn_idx, rivals)
    dimensionality = compute_t_measures(sample.X, sample.population, sizes)
    geometry = compute_geometry_measures(
        sample.X,
        sample.population,
        ids=ids,
        centroid_matrix=matrix,
        classes=classes,
        centroid_distances=centroid_distances,
        silhouette_max_samples=silhouette_max_samples,
        silhouette_min_per_cluster=silhouette_min_per_cluster,
        random_state=random_state,
    )

    return {
        population_id: {
            **feature_overlap[population_id],
            **neighborhood[population_id],
            **network[population_id],
            **dimensionality[population_id],
            **geometry[population_id],
        }
        for population_id in sorted(population_class)
    }
