"""src.graph on hand-made points."""

import numpy as np

from src.graph import sample_rows, space_points


def test_space_points_are_the_numbers_then_the_slots():
    numbers = np.array([[3.0, 4.0], [0.0, 0.0]])
    slots = np.array([[0.0, 2.0], [2.0, 0.0]])
    points = space_points(numbers, slots)
    np.testing.assert_array_equal(points, np.hstack([numbers, slots]))
    without_slots = space_points(numbers, np.empty((2, 0)))
    assert without_slots.shape == (2, 2) and without_slots.dtype == np.float64


def test_sample_rows():
    np.testing.assert_array_equal(sample_rows(10, 20, 0), np.arange(10))
    rows = sample_rows(100, 10, 0)
    assert len(rows) == 10 == len(set(rows.tolist()))
    assert rows.tolist() == sorted(rows.tolist())
    assert rows.max() < 100
    np.testing.assert_array_equal(rows, sample_rows(100, 10, 0))
