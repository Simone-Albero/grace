"""Density, clustering coefficient and hubs."""

__all__ = ["compute_network_measures"]

from collections.abc import Mapping, Sequence

import numpy as np
from tqdm import tqdm

from src.complexity.common import MeasuredSample, summarize


def _clustering_coefficient(
    neighbors: np.ndarray, in_population: np.ndarray, knn_idx: np.ndarray
) -> float:
    """Return the mean share of linked same-population neighbours."""
    coefficients = np.zeros(neighbors.shape[0], dtype=np.float64)
    for i, own_mask in enumerate(in_population):
        own_neighbors = neighbors[i, own_mask]
        if own_neighbors.size < 2:
            continue
        triangles = int(np.isin(knn_idx[own_neighbors], own_neighbors).sum())
        coefficients[i] = triangles / (own_neighbors.size * (own_neighbors.size - 1))
    return float(coefficients.mean())


def compute_network_measures(
    sample: MeasuredSample,
    graph_population: np.ndarray,
    knn_idx: np.ndarray,
    rivals: Mapping[int, Sequence[int]],
) -> dict[int, dict[str, float | None]]:
    """Return density, clustering coefficient and hubs per population.

    `knn_idx` is the graph's own k-NN: hubs are read off it, the rest
    off the graph neighbours of the population's sampled rows.

    Args:
        sample: The sampled rows with their graph neighbours.
        graph_population: The population id of every graph node.
        knn_idx: The graph's k-NN, shape (nodes, k).
        rivals: The nearest rival ids of every population.

    Returns:
        By population id, the min, mean and max density over its
        rivals, "cls_coef" and "hub" (None without a graph node).
    """
    k = knn_idx.shape[1]
    in_degree = np.bincount(knn_idx.ravel(), minlength=knn_idx.shape[0])

    result: dict[int, dict[str, float | None]] = {}
    for population_id in tqdm(rivals, desc="ND measures", unit="pop", leave=False):
        neighbors = sample.neighbors[sample.population == population_id]
        in_population = (graph_population == population_id)[neighbors]
        density = [
            float((graph_population == rival)[neighbors].sum())
            / (neighbors.shape[0] * k)
            for rival in rivals[population_id]
        ]
        node_in_degrees = in_degree[graph_population == population_id]
        result[population_id] = {
            **summarize("network_density", density),
            "cls_coef": _clustering_coefficient(neighbors, in_population, knn_idx),
            # None when the population has no graph node to average.
            "hub": float(node_in_degrees.mean()) if node_in_degrees.size else None,
        }
    return result
