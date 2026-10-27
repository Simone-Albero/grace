"""scikit-learn and xgboost classifiers: pipeline, search, predict."""

__all__ = [
    "CLASSIFIERS",
    "PARALLEL",
    "CalibratedLinearSVC",
    "CappedCategoryEncoder",
    "build_pipeline",
    "classifier_params",
    "grid_search",
    "predict",
]

import tempfile
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class CalibratedLinearSVC(BaseEstimator, ClassifierMixin):
    """A LinearSVC in CalibratedClassifierCV, for `predict_proba`.

    Attributes:
        C: The inverse regularization strength of the LinearSVC.
        max_iter: The most iterations of the LinearSVC.
        random_state: The seed of the LinearSVC.
        classes_: The class labels, set by `fit`.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 2000,
        random_state: int | None = None,
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CalibratedLinearSVC":
        """Fit the LinearSVC and its sigmoid calibration.

        Args:
            X: The training features.
            y: The training labels.

        Returns:
            The fitted classifier.
        """
        base = LinearSVC(
            C=self.C, max_iter=self.max_iter, random_state=self.random_state
        )
        self._calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        self._calibrated.fit(X, y)
        self.classes_ = self._calibrated.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted class labels.

        Args:
            X: The features.

        Returns:
            The label of every row.
        """
        return self._calibrated.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the calibrated class probabilities.

        Args:
            X: The features.

        Returns:
            The probabilities, shape (n, n_classes).
        """
        return self._calibrated.predict_proba(X)


class CappedCategoryEncoder(BaseEstimator, TransformerMixin):
    """A cast of columns to pandas Categorical, keeping common values.

    Attributes:
        max_cardinality: The most categories kept per column; None keeps
            all.
        categories_: By column, the kept categories, set by `fit`.
        feature_names_in_: The fitted column names, set by `fit`.
    """

    def __init__(self, max_cardinality: int | None = 255) -> None:
        self.max_cardinality = max_cardinality

    def fit(self, X: pd.DataFrame, y: object = None) -> "CappedCategoryEncoder":
        """Learn the retained categories of every column.

        Args:
            X: The categorical features.
            y: The labels, ignored.

        Returns:
            The fitted encoder.
        """
        self.categories_: dict[str, pd.Index] = {}
        for col in X.columns:
            counts = X[col].value_counts()
            keep = (
                len(counts)
                if self.max_cardinality is None
                else min(self.max_cardinality, len(counts))
            )
            self.categories_[col] = counts.nlargest(keep).index
        self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Cast every column to its kept categories, the rest to NaN.

        Args:
            X: The categorical features, with the fitted columns.

        Returns:
            A copy of `X` with Categorical columns.
        """
        result = X.copy()
        for col in result.columns:
            kept = self.categories_[col]
            result[col] = pd.Categorical(
                result[col].where(result[col].isin(kept)), categories=kept
            )
        return result

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return the output feature names, equal to the input ones.

        Args:
            input_features: Ignored.

        Returns:
            The fitted column names.
        """
        return self.feature_names_in_.copy()


# Each categorical strategy: its encoder (None drops the columns) and,
# for a classifier that reads pandas categories itself, the extra
# classifier params.
_STRATEGIES: dict[
    str, tuple[Callable[[], TransformerMixin] | None, dict[str, object] | None]
] = {
    "drop_cat": (None, None),
    "onehot": (
        lambda: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        None,
    ),
    "native_sklearn": (
        # HistGradientBoosting takes at most 255 categories per
        # feature.
        lambda: CappedCategoryEncoder(max_cardinality=255),
        {"categorical_features": "from_dtype"},
    ),
    "native_xgb": (
        lambda: CappedCategoryEncoder(max_cardinality=None),
        {"enable_categorical": True, "tree_method": "hist"},
    ),
}

# Each classifier with its categorical strategy; None passes the
# frame as it is.
CLASSIFIERS: dict[str, tuple[type[BaseEstimator], str | None]] = {
    "logistic_regression": (LogisticRegression, "onehot"),
    "lda": (LinearDiscriminantAnalysis, "onehot"),
    "linear_svc": (CalibratedLinearSVC, "onehot"),
    "knn": (KNeighborsClassifier, "onehot"),
    "decision_tree": (DecisionTreeClassifier, None),
    "random_forest": (RandomForestClassifier, None),
    "hist_gradient_boosting": (HistGradientBoostingClassifier, "native_sklearn"),
    "xgboost": (XGBClassifier, "native_xgb"),
    "naive_bayes": (GaussianNB, "drop_cat"),
}

# The classifiers that get `runtime.n_jobs`; logistic_regression's
# n_jobs is deprecated in scikit-learn and has no effect.
PARALLEL = frozenset({"random_forest", "knn", "xgboost"})


def classifier_params(
    name: str, params: Mapping[str, object], *, seed: int, n_jobs: int
) -> dict[str, object]:
    """Return an ML classifier's `params` with its seed and jobs set.

    Args:
        name: A key of `CLASSIFIERS`.
        params: The classifier's own parameters.
        seed: The `random_state` of a classifier that takes one and was
            not given one.
        n_jobs: The parallel jobs of a classifier in `PARALLEL`.

    Returns:
        A copy of `params`.
    """
    params = dict(params)
    if "random_state" in CLASSIFIERS[name][0]().get_params():
        params.setdefault("random_state", seed)
    if name in PARALLEL:
        params["n_jobs"] = n_jobs
    return params


def build_pipeline(
    name: str,
    params: dict[str, object],
    num_cols: list[str],
    cat_cols: list[str],
) -> Pipeline:
    """Return the named classifier behind its categorical strategy.

    Args:
        name: A key of `CLASSIFIERS`.
        params: The classifier's own parameters; they win over the
            ones its strategy sets.
        num_cols: The numerical feature names.
        cat_cols: The categorical feature names.

    Returns:
        The unfitted pipeline, steps "pre" and "clf".
    """
    estimator, strategy = CLASSIFIERS[name]
    if strategy is None:
        return Pipeline([("pre", "passthrough"), ("clf", estimator(**params))])
    encoder, native_params = _STRATEGIES[strategy]
    transformers = [("num", "passthrough", num_cols)]
    if encoder is not None:
        transformers.append(("cat", encoder(), cat_cols))
    preprocessor = ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=native_params is None
    )
    if native_params is not None:
        # A categorical dtype survives only in a DataFrame.
        preprocessor.set_output(transform="pandas")
        params = native_params | params
    return Pipeline([("pre", preprocessor), ("clf", estimator(**params))])


def _strip_clf_prefix(params: Mapping[str, object]) -> dict[str, object]:
    """Drop the pipeline step prefix from the parameter names."""
    return {key.removeprefix("clf__"): value for key, value in params.items()}


def grid_search(
    name: str,
    params: dict[str, object],
    grid: Mapping[str, Sequence[object]],
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    num_cols: list[str],
    cat_cols: list[str],
    scoring: str,
    cv: int,
    max_samples: int | None,
    n_jobs: int,
    random_state: int,
) -> tuple[Pipeline, dict[str, object]]:
    """Search the classifier's grid by CV, then refit on all the data.

    Args:
        name: A key of `CLASSIFIERS`.
        params: The classifier's own parameters.
        grid: The values to try, by classifier parameter.
        X: The training features.
        y: The training labels.
        num_cols: The numerical feature names.
        cat_cols: The categorical feature names.
        scoring: The scikit-learn scorer name.
        cv: The number of folds.
        max_samples: The most rows the search sees; None for all.
        n_jobs: The parallel jobs of the search.
        random_state: The seed of the search subsample.

    Returns:
        The refit pipeline and a summary with "best_params",
        "best_score" and "cv_results", the latter one dict per
        candidate with "params", "mean_test_score" and "std_test_score".
    """
    prefixed_grid = {f"clf__{key}": values for key, values in grid.items()}

    if max_samples is not None and len(X) > max_samples:
        _, search_positions = train_test_split(
            np.arange(len(X)),
            test_size=max_samples,
            stratify=y,
            random_state=random_state,
        )
        # A class too thin to fill every fold keeps all its rows, or a
        # fit of the search would miss a class the pipeline needs.
        labels = np.asarray(y)
        n_kept_by_class = pd.Series(labels[search_positions]).value_counts()
        thin_classes = [
            label for label in np.unique(labels) if n_kept_by_class.get(label, 0) < cv
        ]
        search_positions = np.union1d(
            search_positions, np.flatnonzero(np.isin(labels, thin_classes))
        )
        X_search, y_search = X.iloc[search_positions], labels[search_positions]
    else:
        X_search, y_search = X, y

    # The search parallelizes over candidates, so each fit is
    # single-threaded; the winner is then refit once with the caller's
    # params, its n_jobs included.
    search_params = {**params, "n_jobs": 1} if "n_jobs" in params else params
    with tempfile.TemporaryDirectory() as cache_dir:
        base_pipeline = build_pipeline(name, search_params, num_cols, cat_cols)
        base_pipeline.memory = cache_dir
        search = GridSearchCV(
            base_pipeline,
            param_grid=prefixed_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=False,
        )
        search.fit(X_search, y_search)

    best_params = _strip_clf_prefix(search.best_params_)
    best_pipeline = build_pipeline(name, {**params, **best_params}, num_cols, cat_cols)
    best_pipeline.fit(X, y)

    cv_results = [
        {
            "params": _strip_clf_prefix(candidate_params),
            "mean_test_score": float(mean_score),
            "std_test_score": float(std_score),
        }
        for candidate_params, mean_score, std_score in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"],
        )
    ]
    summary = {
        "best_params": best_params,
        "best_score": float(search.best_score_),
        "cv_results": cv_results,
    }
    return best_pipeline, summary


def predict(model: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Predict the class and class probabilities of every row.

    Args:
        model: A fitted pipeline from `build_pipeline`.
        X: The features.

    Returns:
        The predicted class ids and the probabilities, shape
        (n, n_classes).
    """
    # One pass answers both, so y_pred always agrees with the y_proba
    # published beside it. On a tie, exact or within rounding, argmax
    # takes the lowest class id; knn's own predict() with distance
    # weights can pick another.
    y_proba = model.predict_proba(X)
    y_pred = model.classes_[np.argmax(y_proba, axis=1)]
    return y_pred, y_proba
