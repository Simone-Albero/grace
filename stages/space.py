"""The space distances are measured in, and every row's LDA projection.

Artifacts: the fitted space, the LDA fitted on the train labels and
every row projected by it (`z.parquet`, one column per discriminant
direction). DB rows written: metric n_components.
"""

__all__ = ["CONFIG", "INPUTS", "FILES", "OUTPUTS", "run"]

from collections.abc import Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.space import Space
from stages import LABEL, SPLITS, Context

CONFIG = ("space",)
# Split gives the feature kinds; prepare, the splits.
INPUTS = ("split", "prepare")
FILES = ()
OUTPUTS = ("space.joblib", "lda.joblib", "z.parquet")
_EMBED_ROWS = 500_000


def _project(
    lda: LinearDiscriminantAnalysis, space: Space, rows: pd.DataFrame
) -> np.ndarray:
    """Return the rows' discriminant coordinates, embedded in chunks."""
    return np.vstack(
        [
            lda.transform(
                space.embed(rows.iloc[start : start + _EMBED_ROWS]).astype(np.float64)
            )
            for start in range(0, len(rows), _EMBED_ROWS)
        ]
    ).astype(np.float32)


def _projection_table(projected: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Return the projected rows of every split in one table."""
    table = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": split,
                    "row": np.arange(len(projected[split]), dtype=np.int32),
                    **{f"z_{i}": column for i, column in enumerate(projected[split].T)},
                }
            )
            for split in SPLITS
        ],
        ignore_index=True,
    )
    table["split"] = pd.Categorical(table["split"], categories=SPLITS)
    return table


def run(ctx: Context) -> None:
    """Fit the space and the LDA on train, project all.

    Args:
        ctx: The stage context.
    """
    cfg = ctx.cfg
    num_cols, cat_cols = ctx.features()
    prepare_dir = ctx.inputs["prepare"].dir
    train = pd.read_parquet(prepare_dir / "train.parquet")

    space = Space.fit(
        train,
        num_cols=num_cols,
        cat_cols=cat_cols,
        top_k=cfg.space.top_k,
        cat_cost=cfg.space.cat_cost,
    )

    lda = LinearDiscriminantAnalysis().fit(
        space.embed(train).astype(np.float64), train[LABEL].to_numpy()
    )
    # Features only: no label of val or test is read.
    projected = {"train": _project(lda, space, train)} | {
        split: _project(
            lda,
            space,
            pd.read_parquet(
                prepare_dir / f"{split}.parquet", columns=num_cols + cat_cols
            ),
        )
        for split in SPLITS[1:]
    }

    joblib.dump(space, ctx.out / "space.joblib")
    joblib.dump(lda, ctx.out / "lda.joblib")
    _projection_table(projected).to_parquet(ctx.out / "z.parquet", index=False)
    ctx.save_metrics({"n_components": projected["train"].shape[1]})
