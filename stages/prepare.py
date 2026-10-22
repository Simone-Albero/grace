"""The encoders fit on train and applied to every split.

DB rows written: `classes` (class id, name and size of each split).
"""

__all__ = ["CONFIG", "INPUTS", "FILES", "OUTPUTS", "run"]

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder

from src.data import build_preprocessor
from stages import LABEL, SPLITS, Context

CONFIG = ("data.top_n", "data.hash_buckets")
INPUTS = ("split",)
FILES = ()
OUTPUTS = (*(f"{name}.parquet" for name in SPLITS), "preprocessor.joblib")


def run(ctx: Context) -> None:
    """Write the preprocessed splits and the fitted preprocessor.

    Args:
        ctx: The stage context.
    """
    num_cols, cat_cols = ctx.features()
    raw = {
        name: pd.read_parquet(ctx.inputs["split"].dir / f"{name}.parquet")
        for name in SPLITS
    }

    preprocessor = build_preprocessor(
        num_cols,
        cat_cols,
        top_n=ctx.cfg.data.top_n,
        hash_buckets=ctx.cfg.data.hash_buckets,
    )
    label_encoder = LabelEncoder().fit(raw["train"][LABEL])
    # A context, not set_config: the pandas output stays local.
    with sklearn.config_context(transform_output="pandas"):
        preprocessor.fit(raw["train"])
        encoded = {
            name: preprocessor.transform(frame)
            .reset_index(drop=True)
            .assign(**{LABEL: label_encoder.transform(frame[LABEL])})
            for name, frame in raw.items()
        }

    for name, frame in encoded.items():
        frame.to_parquet(ctx.out / f"{name}.parquet", index=False)
    joblib.dump(preprocessor, ctx.out / "preprocessor.joblib")

    n_classes = len(label_encoder.classes_)
    ctx.write(
        "classes",
        pd.DataFrame(
            {
                "class_id": range(n_classes),
                "name": [str(name) for name in label_encoder.classes_],
                **{
                    f"n_{name}": np.bincount(frame[LABEL], minlength=n_classes)
                    for name, frame in encoded.items()
                },
            }
        ),
    )
