"""Network measures of the graph neighbourhoods of a population.

With k the graph's neighbours per node:
- network_density_{min,mean,max}: per rival, the neighbour entries of
  the population's sampled rows that are rival nodes / (rows · k);
- cls_coef: mean over the sampled rows of Σ_{u in S} |knn(u) ∩ S| / (|S|
  (|S| - 1)), S the row's neighbours in its own population (0 when |S| <
  2, still averaged);
- hub: mean in-degree in the graph's k-NN of the population's graph
  nodes (None without a graph node);
- rival_classes: distinct other classes among the neighbours of the
  sampled rows;
- rival_entropy: Shannon entropy (nats) of those other-class counts, 0
  if none.
"""

from __future__ import annotations

__all__ = ["compute_network_measures"]

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import entropy

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


def _rival_class_diversity(
    neighbor_class: np.ndarray, own_class: int
) -> dict[str, float]:
    """Return the rival classes of the neighbours and their entropy."""
    rival_neighbor_class = neighbor_class[neighbor_class != own_class]
    counts = np.unique(rival_neighbor_class, return_counts=True)[1]
    return {
        "rival_classes": float(counts.size),
        "rival_entropy": float(entropy(counts)) if counts.size else 0.0,
    }


def compute_network_measures(
    sample: MeasuredSample,
    graph_population: np.ndarray,
    knn_idx: np.ndarray,
    rivals: Mapping[int, Sequence[int]],
    *,
    population_class: Mapping[int, int],
    graph_class: np.ndarray,
) -> dict[int, dict[str, float | None]]:
    """Return density, clustering coefficient, hubs and rival classes.

    `knn_idx` is the graph's own k-NN: hubs are read off it, the rest
    off the graph neighbours of the population's sampled rows.

    Args:
        sample: The sampled rows with their graph neighbours.
        graph_population: The population id of every graph node.
        knn_idx: The graph's k-NN, shape (nodes, k).
        rivals: The nearest rival ids of every population.
        population_class: The class of every population id.
        graph_class: The class of every graph node.

    Returns:
        By population id, the min, mean and max density over its
        rivals, "cls_coef", "hub" (None without a graph node), then
        "rival_classes", the count of other classes among the
        neighbours, and "rival_entropy", the Shannon entropy (nats) of
        their mix; both 0 without such a neighbour.
    """
    k = knn_idx.shape[1]
    in_degree = np.bincount(knn_idx.ravel(), minlength=knn_idx.shape[0])

    result: dict[int, dict[str, float | None]] = {}
    for population_id in rivals:
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
            **_rival_class_diversity(
                graph_class[neighbors], population_class[population_id]
            ),
        }
    return result
