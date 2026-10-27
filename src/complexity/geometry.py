"""Dispersion, rival separation and silhouette tail."""

__all__ = ["compute_geometry_measures"]

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_samples


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


def compute_geometry_measures(
    X: np.ndarray,
    population: np.ndarray,
    *,
    ids: Sequence[int],
    centroid_matrix: np.ndarray,
    classes: np.ndarray,
    centroid_distances: np.ndarray,
    silhouette_max_samples: int,
    silhouette_min_per_cluster: int,
    random_state: int,
) -> dict[int, dict[str, float | None]]:
    """Return dispersion, rival separation and silhouette tail.

    Args:
        X: The sampled rows, shape (n, d).
        population: The population id of every row of `X`.
        ids: The population ids, in the row order of the next three.
        centroid_matrix: The centroids, shape (len(ids), d).
        classes: The class of every population in `ids`.
        centroid_distances: The centroid distances, diagonal inf.
        silhouette_max_samples: The silhouette subsample cap.
        silhouette_min_per_cluster: The silhouette floor per cluster.
        random_state: The seed of the silhouette subsample.

    Returns:
        By population id: "max_dispersion", "p95_dispersion",
        "dist_to_nearest_rival", "p5_silhouette" and "frac_at_risk".
    """
    silhouettes = _approx_silhouette(
        X,
        population,
        max_samples=silhouette_max_samples,
        min_per_cluster=silhouette_min_per_cluster,
        random_state=random_state,
    )

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
        }

    return result
