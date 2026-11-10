"""Geometry: spread of a population and how close its rivals lie.

- max_dispersion, p95_dispersion: max and 95th percentile of the
  distances of the sampled rows from the population's centroid;
- dist_to_nearest_rival: least centroid distance to any population of
  another class (None if none);
- p5_silhouette, frac_at_risk: 5th percentile and share below 0 of the
  population's silhouette scores, computed jointly over all populations
  on a subsample (silhouette_max_samples rows, at least
  silhouette_min_per_cluster per population);
- mahalanobis: least, over rivals with 2+ rows each side, of the
  distance between the two means under the Ledoit-Wolf covariance of the
  stacked centred rows;
- rival_reach: largest, over rivals, of the 95th percentile of the own
  rows' projection on the axis to the rival centroid, divided by the
  squared axis length.
"""

from __future__ import annotations

__all__ = ["GEOMETRY_KEYS", "compute_geometry_measures"]

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import pairwise_distances, silhouette_samples

GEOMETRY_KEYS = (
    "max_dispersion",
    "p95_dispersion",
    "dist_to_nearest_rival",
    "p5_silhouette",
    "frac_at_risk",
    "mahalanobis",
    "rival_reach",
)


def _approx_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    max_samples: int,
    min_per_cluster: int,
    random_state: int,
) -> np.ndarray | None:
    """Return silhouette scores of a stratified subsample.

    The rows outside the subsample are NaN. The result is None below
    two labels, or when scikit-learn rejects the subsample.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return None

    n_rows = len(X)
    if n_rows <= max_samples:
        rows = np.arange(n_rows)
    else:
        rng = np.random.default_rng(random_state)
        # The floor holds even past the cap: a silhouette tail read off
        # a handful of rows per cluster is noise, so many clusters grow
        # the subsample instead.
        picked_parts: list[np.ndarray] = []
        for label in unique_labels:
            members = np.where(labels == label)[0]
            take = min(len(members), min_per_cluster)
            picked_parts.append(rng.choice(members, size=take, replace=False))
        floor_rows = np.concatenate(picked_parts)
        remaining = max_samples - len(floor_rows)
        if remaining > 0:
            pool = np.setdiff1d(np.arange(n_rows), floor_rows)
            extra = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
            rows = np.concatenate([floor_rows, extra])
        else:
            rows = floor_rows

    try:
        scores = silhouette_samples(X[rows], labels[rows])
    except ValueError:
        # scikit-learn rejects a label count outside [2, n_samples - 1].
        return None

    scores_by_row = np.full(n_rows, np.nan)
    scores_by_row[rows] = scores
    return scores_by_row


def _dispersion(samples: np.ndarray, centroid: np.ndarray) -> tuple[float, float]:
    """Return the max and 95th-percentile distance to the centroid."""
    distances = pairwise_distances(samples, centroid.reshape(1, -1)).ravel()
    return float(np.max(distances)), float(np.percentile(distances, 95))


def _nearest_rival(
    distances_to_centroids: np.ndarray, own_class: int, classes: np.ndarray
) -> float | None:
    """Return the distance to the closest centroid of another class."""
    rival_distances = distances_to_centroids[classes != own_class]
    finite = rival_distances[np.isfinite(rival_distances)]
    if finite.size == 0:
        return None
    return float(np.min(finite))


def _mahalanobis(own_rows: np.ndarray, rival_rows: np.ndarray) -> float:
    """Return the Mahalanobis distance between the two means.

    The covariance is pooled from both populations, each centred on
    its own mean, and shrunk by Ledoit-Wolf: a region holds no more
    rows than a few times the features, too few for the raw estimate
    to be invertible.
    """
    own_mean, rival_mean = own_rows.mean(axis=0), rival_rows.mean(axis=0)
    centered = np.vstack([own_rows - own_mean, rival_rows - rival_mean])
    covariance = LedoitWolf(assume_centered=True).fit(centered).covariance_
    mean_difference = own_mean - rival_mean
    # lstsq, not solve: a feature constant in both populations leaves
    # the covariance singular when the shrinkage comes out as zero.
    solved = np.linalg.lstsq(covariance, mean_difference, rcond=None)[0]
    return float(np.sqrt(max(float(mean_difference @ solved), 0.0)))


def _rival_reach(
    own_rows: np.ndarray, centroid: np.ndarray, rival_centroid: np.ndarray
) -> float | None:
    """Return the p95 reach of the rows along the axis to the rival.

    It is a share of the centroid distance, None when the centroids
    coincide.
    """
    axis = rival_centroid - centroid
    squared_distance = float(axis @ axis)
    if squared_distance == 0.0:
        return None
    return float(np.percentile((own_rows - centroid) @ axis, 95) / squared_distance)


def _rival_separation(
    rows_by_population: Mapping[int, np.ndarray],
    centroid_of: Mapping[int, np.ndarray],
    population_id: int,
    rivals: Sequence[int],
) -> dict[str, float | None]:
    """Return the worst Mahalanobis distance and reach of the rivals."""
    own_rows = rows_by_population[population_id]
    mahalanobis_distances, reaches = [], []
    for rival in rivals:
        rival_rows = rows_by_population[rival]
        if len(own_rows) >= 2 and len(rival_rows) >= 2:
            mahalanobis_distances.append(_mahalanobis(own_rows, rival_rows))
        reach = _rival_reach(own_rows, centroid_of[population_id], centroid_of[rival])
        if reach is not None:
            reaches.append(reach)
    return {
        "mahalanobis": min(mahalanobis_distances) if mahalanobis_distances else None,
        "rival_reach": max(reaches) if reaches else None,
    }


def compute_geometry_measures(
    X: np.ndarray,
    population: np.ndarray,
    *,
    ids: Sequence[int],
    rivals: Mapping[int, Sequence[int]],
    centroid_matrix: np.ndarray,
    classes: np.ndarray,
    centroid_distances: np.ndarray,
    silhouette_max_samples: int,
    silhouette_min_per_cluster: int,
    random_state: int,
) -> dict[int, dict[str, float | None]]:
    """Return dispersion, rival separation and silhouette tail.

    Every distance is euclidean on the points of the space
    (graph.space_points), as the graph measures them.

    Args:
        X: The sampled rows, shape (n, d).
        population: The population id of every row of `X`.
        ids: The population ids, in the row order of `centroid_matrix`,
            `classes` and `centroid_distances`.
        rivals: The nearest rival ids of every population.
        centroid_matrix: The centroids, shape (len(ids), d).
        classes: The class of every population in `ids`.
        centroid_distances: The centroid distances, diagonal inf.
        silhouette_max_samples: The silhouette subsample cap.
        silhouette_min_per_cluster: The silhouette floor per cluster.
        random_state: The seed of the silhouette subsample.

    Returns:
        By population id: "max_dispersion", "p95_dispersion",
        "dist_to_nearest_rival", "p5_silhouette", "frac_at_risk", then
        "mahalanobis", the least over the rivals with two rows or more,
        and "rival_reach", the most over the rivals; None without one.
    """
    silhouettes = _approx_silhouette(
        X,
        population,
        max_samples=silhouette_max_samples,
        min_per_cluster=silhouette_min_per_cluster,
        random_state=random_state,
    )

    rows_by_population = {
        int(population_id): X[population == population_id]
        for population_id in np.unique(population)
    }
    centroid_of = dict(zip(ids, centroid_matrix))

    result: dict[int, dict[str, float | None]] = {}

    for i, population_id in enumerate(ids):
        in_population = population == population_id
        max_dispersion, p95_dispersion = _dispersion(
            X[in_population], centroid_matrix[i]
        )
        dist_to_nearest_rival = _nearest_rival(
            centroid_distances[i], classes[i], classes
        )

        if silhouettes is None:
            p5_silhouette, frac_at_risk = None, None
        else:
            own_silhouettes = silhouettes[in_population]
            finite_silhouettes = own_silhouettes[np.isfinite(own_silhouettes)]
            if finite_silhouettes.size == 0:
                p5_silhouette, frac_at_risk = None, None
            else:
                p5_silhouette = float(np.percentile(finite_silhouettes, 5))
                frac_at_risk = float(np.mean(finite_silhouettes < 0))

        result[population_id] = {
            "max_dispersion": max_dispersion,
            "p95_dispersion": p95_dispersion,
            "dist_to_nearest_rival": dist_to_nearest_rival,
            "p5_silhouette": p5_silhouette,
            "frac_at_risk": frac_at_risk,
            **_rival_separation(
                rows_by_population, centroid_of, population_id, rivals[population_id]
            ),
        }

    return result
