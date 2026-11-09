"""src.data and src.prepare on hand-made frames."""

import hashlib
import math

import numpy as np
import polars as pl
import pytest

from src.data import SPLITS, drop_incomplete, drop_rare_classes, stratified_split
from src.prepare import (
    HASH_SALT,
    LABEL,
    Preprocessor,
    TopNHashEncoder,
    hash_bucket,
    signed_log1p,
)


def test_drop_incomplete_drops_nulls_nan_and_inf():
    df = pl.DataFrame(
        {
            "a": [1.0, float("nan"), float("inf"), 2.0, 3.0, -float("inf")],
            "c": ["x", "y", "z", None, "w", "v"],
            "y": ["p", "p", "q", "q", "q", "p"],
        }
    )
    kept = drop_incomplete(df, ["a"])
    assert kept["a"].to_list() == [1.0, 3.0]
    assert kept["c"].to_list() == ["x", "w"]


def test_drop_rare_classes_keeps_order_and_the_class_at_the_floor():
    df = pl.DataFrame({"y": ["a", "c", "b", "a", "b", "a"], "v": list(range(6))})
    kept, dropped = drop_rare_classes(df, "y", 2)
    assert kept["v"].to_list() == [0, 2, 3, 4, 5]
    assert dropped == ["c"]


def _frame(counts: dict[str, int]) -> pl.DataFrame:
    labels = []
    for label, n_rows in counts.items():
        labels += [label] * n_rows
    return pl.DataFrame({"y": labels, "v": list(range(len(labels)))})


def test_stratified_split_is_a_stratified_partition():
    df = _frame({"a": 600, "b": 400})
    shares = {"train": 0.4, "val": 0.15, "test": 0.45}
    parts = stratified_split(df, "y", shares, seed=1)
    assert tuple(parts) == SPLITS
    values = pl.concat([parts[name]["v"] for name in SPLITS]).to_list()
    assert sorted(values) == list(range(1000))
    for name, part in parts.items():
        assert part["v"].to_list() == sorted(part["v"].to_list())
        for label, n in (("a", 600), ("b", 400)):
            got = (part["y"] == label).sum()
            assert abs(got - shares[name] * n) <= 1, (name, label, got)


def test_stratified_split_depends_on_the_seed_only():
    df = _frame({"a": 300, "b": 200})
    shares = {"train": 0.5, "val": 0.2, "test": 0.3}
    first = stratified_split(df, "y", shares, seed=3)
    again = stratified_split(df, "y", shares, seed=3)
    other = stratified_split(df, "y", shares, seed=4)
    assert first["train"].equals(again["train"])
    assert not first["train"].equals(other["train"])


def test_signed_log1p():
    x = np.array([-3.0, 0.0, math.e - 1])
    np.testing.assert_allclose(signed_log1p(x), [-math.log(4), 0.0, 1.0])


def test_hash_bucket_is_the_salted_blake2b():
    digest = hashlib.blake2b(f"{HASH_SALT}|col|val".encode(), digest_size=8)
    expected = int.from_bytes(digest.digest(), "little") % 7
    assert hash_bucket("col", "val", 7) == expected
    assert 0 <= hash_bucket("other", "val", 7) < 7


def test_top_n_hash_encoder_codes():
    train = pl.Series("c", ["a", "a", "b", "b", "c", None, "a"])
    encoder = TopNHashEncoder("c", top_n=2, hash_buckets=5).fit(train)
    codes = encoder.transform(pl.Series("c", ["a", "b", "c", None, "zzz"]))
    assert codes.dtype == np.int32
    assert codes.tolist() == [
        1,
        2,
        3 + hash_bucket("c", "c", 5),
        0,
        3 + hash_bucket("c", "zzz", 5),
    ]


def test_top_n_hash_encoder_ties_and_no_buckets():
    encoder = TopNHashEncoder("c", top_n=1, hash_buckets=0)
    encoder.fit(pl.Series("c", ["b", "a"]))
    assert encoder.transform(pl.Series("c", ["a", "b"])).tolist() == [1, 0]


@pytest.fixture
def raw() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "n": [0.0, 1.0, 2.0, 3.0, 4.0],
            "c": ["x", "x", "y", "z", "x"],
            "y": ["dog", "cat", "dog", "cat", "dog"],
        }
    )


def test_preprocessor_columns_codes_and_scaling(raw):
    prep = Preprocessor(["n"], ["c"], "y", top_n=1, hash_buckets=0).fit(raw)
    out = prep.transform(raw)
    assert out.columns == ["n", "c", LABEL]
    assert prep.classes_ == ("cat", "dog")
    assert out[LABEL].to_list() == [1, 0, 1, 0, 1]
    assert out["c"].to_list() == [1, 1, 0, 0, 1]
    logged = np.log1p(raw["n"].to_numpy())
    q25, q50, q75 = np.percentile(logged, [25, 50, 75])
    np.testing.assert_allclose(out["n"].to_numpy(), (logged - q50) / (q75 - q25))


def test_preprocessor_rejects_an_unseen_class(raw):
    prep = Preprocessor(["n"], ["c"], "y", top_n=1, hash_buckets=0).fit(raw)
    with pytest.raises(ValueError, match="bird"):
        prep.transform(raw.with_columns(pl.lit("bird").alias("y")))
