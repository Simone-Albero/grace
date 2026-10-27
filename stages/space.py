"""The space distances are measured in, and the k-NN graph of train.

Artifacts: the fitted space, the graph arrays (the rows of the graph,
its k-NN lists and its minimum spanning tree), the LDA fitted on the
train labels and every row projected by it (`z.parquet`, one column per
discriminant direction). DB rows written: metrics n_graph_rows and
n_components.
"""

__all__ = ["CONFIG", "INPUTS", "FILES", "OUTPUTS", "run"]

from collections.abc import Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.graph import build_train_graph, subsample_indices
from src.space import Space
from stages import LABEL, SPLITS, Context

CONFIG = ("space", "graph", "seed")
# Split gives the feature kinds; prepare, the splits.
INPUTS = ("split", "prepare")
FILES = ()
OUTPUTS = ("space.joblib", "graph.npz", "lda.joblib", "z.parquet")
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
    """Fit the space and the LDA on train, graph a sample, project all.

    Args:
        ctx: The stage context.
    """
    cfg = ctx.cfg
    num_cols, cat_cols = ctx.features()
    prepare_dir = ctx.inputs["prepare"].dir
    train = pd.read_parquet(prepare_dir / "train.parquet")

    graph_rows = np.sort(
        subsample_indices(
            len(train), max_samples=cfg.graph.max_samples, random_state=cfg.seed
        )
    )
    space = Space.fit(
        train,
        num_cols=num_cols,
        cat_cols=cat_cols,
        top_k=cfg.space.top_k,
        cat_cost=cfg.space.cat_cost,
    )
    train_graph = build_train_graph(
        space.embed(train.iloc[graph_rows]),
        k=cfg.graph.k,
        n_jobs=ctx.runtime.n_jobs,
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
    np.savez(
        ctx.out / "graph.npz",
        rows=graph_rows,
        knn_idx=train_graph.knn_idx,
        knn_dist=train_graph.knn_dist,
        mst=train_graph.mst_edges,
    )
    ctx.save_metrics(
        {"n_graph_rows": len(graph_rows), "n_components": projected["train"].shape[1]}
    )
