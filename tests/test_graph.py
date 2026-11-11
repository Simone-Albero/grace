"""src.graph on hand-made points."""

import numpy as np

from src.graph import sample_rows, space_points


def test_space_points_scale_only_the_numbers_for_the_metric():
    numbers = np.array([[3.0, 4.0], [0.0, 0.0]])
    slots = np.array([[0.0, 2.0], [2.0, 0.0]])
    euclidean = space_points(numbers, slots, "euclidean")
    np.testing.assert_array_equal(euclidean, np.hstack([numbers, slots]))
    cosine = space_points(numbers, slots, "cosine")
    np.testing.assert_allclose(cosine[:, :2], [[0.6, 0.8], [0.0, 0.0]])
    np.testing.assert_array_equal(cosine[:, 2:], slots)
    without_slots = space_points(numbers, np.empty((2, 0)), "cosine")
    assert without_slots.shape == (2, 2) and without_slots.dtype == np.float64


def test_sample_rows():
    np.testing.assert_array_equal(sample_rows(10, 20, 0), np.arange(10))
    rows = sample_rows(100, 10, 0)
    assert len(rows) == 10 == len(set(rows.tolist()))
    assert rows.tolist() == sorted(rows.tolist())
    assert rows.max() < 100
    np.testing.assert_array_equal(rows, sample_rows(100, 10, 0))
