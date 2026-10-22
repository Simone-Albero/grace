"""N1-N4 neighbourhood measures."""

__all__ = ["compute_n_measures"]

from collections.abc import Mapping, Sequence

import numpy as np
from tqdm import tqdm

from src.complexity.common import MeasuredSample, summarize

_EPS = 1e-8
_N_KEYS = ("n1", "n2", "n3", "n4")


def _n1_pair(
    own_mask: np.ndarray, rival_mask: np.ndarray, mst_edges: np.ndarray
) -> float:
    """Return N1, the share of own nodes on an MST edge to the rival."""
    n_own = int(own_mask.sum())
    if mst_edges.shape[0] == 0:
        return 0.0
    source, target = mst_edges[:, 0], mst_edges[:, 1]
    boundary_source = source[own_mask[source] & rival_mask[target]]
    boundary_target = target[own_mask[target] & rival_mask[source]]
    if boundary_source.size == 0 and boundary_target.size == 0:
        return 0.0
    return float(
        np.unique(np.concatenate([boundary_source, boundary_target])).size / n_own
    )


def _n2_pair(
    neighbors: np.ndarray,
    neighbor_dist: np.ndarray,
    in_own: np.ndarray,
    in_rival: np.ndarray,
) -> float:
    """Return N2, the mean own/(own+rival) neighbour distance ratio."""
    valid = in_own.any(axis=1) & in_rival.any(axis=1)
    if not valid.any():
        return 0.0
    rows = np.arange(neighbors.shape[0])
    own_dist = neighbor_dist[rows, in_own.argmax(axis=1)]
    rival_dist = neighbor_dist[rows, in_rival.argmax(axis=1)]
    ratios = own_dist[valid] / (own_dist[valid] + rival_dist[valid] + _EPS)
    return float(ratios.mean())


def _n3_pair(
    neighbors: np.ndarray,
    rival_mask: np.ndarray,
    in_own: np.ndarray,
    in_rival: np.ndarray,
) -> float:
    """Return N3, the 1-NN error rate over own and rival neighbours."""
    in_either = in_own | in_rival
    valid = in_either.any(axis=1)
    if not valid.any():
        return 0.0
    rows = np.arange(neighbors.shape[0])
    nearest = neighbors[rows, in_either.argmax(axis=1)]
    misclassified = rival_mask[nearest] & valid
    return float(misclassified.sum() / valid.sum())


def _n4_pair(in_own: np.ndarray, in_rival: np.ndarray) -> float:
    """Return N4, the k-NN majority-vote error over own and rival."""
    own_votes = in_own.sum(axis=1)
    rival_votes = in_rival.sum(axis=1)
    valid = (own_votes + rival_votes) > 0
    if not valid.any():
        return 0.0
    return float(((rival_votes > own_votes) & valid).sum() / valid.sum())


def _rival_measures(
    neighbors: np.ndarray,
    neighbor_dist: np.ndarray,
    own_mask: np.ndarray,
    rival_mask: np.ndarray,
    mst_edges: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return N1-N4 of a population against one rival."""
    in_own = own_mask[neighbors]
    in_rival = rival_mask[neighbors]
    n1 = _n1_pair(own_mask, rival_mask, mst_edges)
    n2 = _n2_pair(neighbors, neighbor_dist, in_own, in_rival)
    n3 = _n3_pair(neighbors, rival_mask, in_own, in_rival)
    n4 = _n4_pair(in_own, in_rival)
    return n1, n2, n3, n4


def _pair_measures(
    neighbors: np.ndarray,
    neighbor_dist: np.ndarray,
    own_mask: np.ndarray,
    rival_masks: Sequence[np.ndarray],
    mst_edges: np.ndarray,
) -> dict[str, list[float]]:
    """Return N1-N4 of a population against each of its rivals."""
    values_by_key: dict[str, list[float]] = {key: [] for key in _N_KEYS}
    for rival_mask in rival_masks:
        n1, n2, n3, n4 = _rival_measures(
            neighbors, neighbor_dist, own_mask, rival_mask, mst_edges
        )
        # No graph node in the population: no MST edge to count.
        if own_mask.any():
            values_by_key["n1"].append(n1)
        values_by_key["n2"].append(n2)
        values_by_key["n3"].append(n3)
        values_by_key["n4"].append(n4)
    return values_by_key


def compute_n_measures(
    sample: MeasuredSample,
    graph_population: np.ndarray,
    mst_edges: np.ndarray,
    rivals: Mapping[int, Sequence[int]],
) -> dict[int, dict[str, float | None]]:
    """Return N1-N4 per population against its nearest rivals.

    N1 counts graph nodes on the MST; N2-N4 read the graph neighbours of
    the population's own sampled rows.

    Args:
        sample: The sampled rows with their graph neighbours.
        graph_population: The population id of every graph node.
        mst_edges: The graph's minimum spanning tree, shape (m, 2).
        rivals: The nearest rival ids of every population.

    Returns:
        By population id, the min, mean and max of each measure over
        its rivals.
    """
    result: dict[int, dict[str, float | None]] = {}
    for population_id in tqdm(rivals, desc="N measures", unit="pop", leave=False):
        in_population = sample.population == population_id
        pair_values = _pair_measures(
            sample.neighbors[in_population],
            sample.neighbor_dist[in_population],
            graph_population == population_id,
            [graph_population == rival for rival in rivals[population_id]],
            mst_edges,
        )
        row: dict[str, float | None] = {}
        for key in _N_KEYS:
            row |= summarize(key, pair_values[key])
        result[population_id] = row

    return result
