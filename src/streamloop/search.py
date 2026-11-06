"""Broad random search, then a grid around the region that search liked.

Both searches fit only the training split. ``refit=True`` trains the
mean-F2 winner. This module never calls ``fit`` on that winner again.
"""

from streamloop import _threads  # noqa: F401

import pandas as pd
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from streamloop.util import plain, same_param

# Cartesian product is 5*4*4*4*4 = 1280. Random search draws a sample;
# the grid below is a local subset, not this whole product.
BROAD_SPACE = {
    "model__learning_rate": [0.03, 0.05, 0.1, 0.15, 0.25],
    "model__max_depth": [3, 5, 8, None],
    "model__max_iter": [50, 100, 200, 300],
    "model__min_samples_leaf": [10, 20, 40, 80],
    "model__l2_regularization": [0.0, 0.1, 1.0, 10.0],
}

# Caps keep the follow-up grid local: at most 3*3*2*2*2 = 72 candidates.
DEFAULT_CAPS = {
    "model__learning_rate": 3,
    "model__max_depth": 3,
    "model__max_iter": 2,
    "model__min_samples_leaf": 2,
    "model__l2_regularization": 2,
}


def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2, zero_division=0)


def f2_scorer():
    return make_scorer(f2_score)


def make_cv(n_splits, random_state):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def space_size(space):
    size = 1
    for values in space.values():
        size *= len(list(values))
    return int(size)


def catalog_index(catalog, value):
    for index, item in enumerate(catalog):
        if same_param(item, value):
            return index
    raise KeyError(f"{value!r} is not in the declared search catalog {list(catalog)!r}")


def narrow_param_grid(cv_results, broad_space=None, top_k=5, caps=None):
    """Build a local grid from the best random-search rows.

    For each hyperparameter the region is the span of values in the top
    rows, plus the catalog neighbors of the best value. A per-parameter
    cap then keeps the values closest to that best value. The best value
    itself is always kept, so the grid can recover the random-search winner.
    """
    broad_space = broad_space or BROAD_SPACE
    caps = caps or DEFAULT_CAPS
    frame = pd.DataFrame(cv_results)
    if frame.empty:
        raise ValueError("cv_results_ is empty.")
    if "mean_test_score" not in frame.columns or "std_test_score" not in frame.columns:
        raise ValueError("cv_results_ is missing mean_test_score or std_test_score.")
    ranked = frame.sort_values(
        ["mean_test_score", "std_test_score"],
        ascending=[False, True],
        kind="mergesort",
    )
    top = ranked.head(min(top_k, len(ranked)))
    grid = {}
    for name, catalog in broad_space.items():
        column = f"param_{name}"
        if column not in frame.columns:
            raise ValueError(f"cv_results_ is missing {column}.")
        catalog = list(catalog)
        best_idx = catalog_index(catalog, plain(top.iloc[0][column]))
        observed = [catalog_index(catalog, plain(value)) for value in top[column].tolist()]
        chosen = set(range(min(observed), max(observed) + 1))
        if best_idx - 1 >= 0:
            chosen.add(best_idx - 1)
        if best_idx + 1 < len(catalog):
            chosen.add(best_idx + 1)
        cap = int(caps.get(name, 3))
        if cap < 1:
            raise ValueError(f"Cap for {name} must be >= 1.")
        ordered = sorted(chosen, key=lambda index: (abs(index - best_idx), index))
        picked = []
        for index in ordered:
            if index not in picked:
                picked.append(index)
            if len(picked) >= cap:
                break
        if best_idx not in picked:
            picked = [best_idx] + [index for index in picked if index != best_idx]
            picked = picked[:cap]
        grid[name] = [catalog[index] for index in sorted(picked)]
    return grid


def grid_contains_params(grid, params):
    for key, value in params.items():
        if key not in grid:
            return False
        if not any(same_param(value, candidate) for candidate in grid[key]):
            return False
    return True


def _fit_train_only(search, X_train, y_train, n_full):
    if len(X_train) >= n_full:
        raise RuntimeError("Refusing to fit a search on the full dataset.")
    if len(X_train) != len(y_train):
        raise RuntimeError("Training features and labels have different lengths.")
    search.fit(X_train, y_train)
    return search


def run_random_search(pipeline, X_train, y_train, config, space=None, n_full=None):
    space = space or BROAD_SPACE
    n_full = len(X_train) + 1 if n_full is None else n_full
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=space,
        n_iter=config.n_iter,
        scoring=f2_scorer(),
        refit=True,
        cv=make_cv(config.cv, config.random_state),
        n_jobs=config.n_jobs,
        random_state=config.random_state,
        error_score="raise",
        return_train_score=False,
    )
    return _fit_train_only(search, X_train, y_train, n_full)


def run_grid_search(pipeline, param_grid, X_train, y_train, config, n_full=None):
    n_full = len(X_train) + 1 if n_full is None else n_full
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=f2_scorer(),
        refit=True,
        cv=make_cv(config.cv, config.random_state),
        n_jobs=config.n_jobs,
        error_score="raise",
        return_train_score=False,
    )
    return _fit_train_only(search, X_train, y_train, n_full)


def candidates_from_cv(cv_results):
    frame = pd.DataFrame(cv_results)
    fold_columns = [column for column in frame.columns if column.startswith("split") and column.endswith("_test_score")]
    fold_columns = sorted(fold_columns, key=lambda column: int(column.split("_", 1)[0].replace("split", "")))
    param_columns = [column for column in frame.columns if column.startswith("param_")]
    rows = []
    for index, row in frame.iterrows():
        params = {column[len("param_") :]: plain(row[column]) for column in param_columns}
        rows.append(
            {
                "index": int(index),
                "mean_f2": float(row["mean_test_score"]),
                "std_f2": float(row["std_test_score"]),
                "folds": [float(row[column]) for column in fold_columns],
                "params": params,
            }
        )
    return rows


def choose_candidate(candidates, best_index, near_tie):
    """Pick the most stable candidate inside a pre-set band of the best mean.

    The band is ``near_tie`` F2 points under the ``refit=True`` champion.
    Lower fold standard deviation wins. Equal mean and equal spread keep
    the champion so its already-refit object can be reused.
    """
    by_index = {candidate["index"]: candidate for candidate in candidates}
    if best_index not in by_index:
        raise ValueError(f"best_index {best_index} is not in cv_results_.")
    champion = by_index[best_index]
    band = [candidate for candidate in candidates if candidate["mean_f2"] >= champion["mean_f2"] - near_tie]
    if not band:
        raise RuntimeError("The near-tie band is empty.")
    chosen = sorted(band, key=lambda candidate: (candidate["std_f2"], -candidate["mean_f2"], candidate["index"]))[0]
    same_mean = abs(chosen["mean_f2"] - champion["mean_f2"]) <= 1e-12
    same_std = abs(chosen["std_f2"] - champion["std_f2"]) <= 1e-12
    if same_mean and same_std:
        chosen = champion
    return chosen, champion, band


def select_final_estimator(grid_search, template, X_train, y_train, near_tie):
    """Return the model to score once on the test set.

    When the stability rule agrees with ``best_index_``, the object
    ``refit=True`` already trained is returned untouched. Otherwise a new
    pipeline is fit once on the training split. ``best_estimator_.fit``
    is never called.
    """
    candidates = candidates_from_cv(grid_search.cv_results_)
    chosen, champion, band = choose_candidate(candidates, int(grid_search.best_index_), near_tie)
    from streamloop.util import params_equal

    used_refit = params_equal(chosen["params"], champion["params"])
    decision = {
        "rule": (
            f"Entre los candidatos de la rejilla a como máximo {near_tie:g} de F2 "
            "de validación de la mejor media, se elige la menor desviación entre pliegues. "
            "Si empatan, gana la media más alta y, si media y desviación coinciden, "
            "el campeón que ya reentrenó refit=True."
        ),
        "near_tie": float(near_tie),
        "band_size": len(band),
        "chosen": chosen,
        "champion": champion,
        "used_refit_best_estimator": used_refit,
    }
    if used_refit:
        decision["how"] = (
            "Se entrega grid.best_estimator_, el objeto que ya entrenó refit=True. "
            "No se volvió a llamar fit sobre ese objeto."
        )
        return grid_search.best_estimator_, decision

    def _refuse_refit(*args, **kwargs):
        raise RuntimeError("Refusing to call fit on best_estimator_ after refit=True.")

    grid_search.best_estimator_.fit = _refuse_refit
    from sklearn.base import clone

    estimator = clone(template)
    estimator.set_params(**chosen["params"])
    estimator.fit(X_train, y_train)
    decision["how"] = (
        "La regla de estabilidad eligió un candidato distinto del campeón de refit. "
        "Se ajustó un pipeline nuevo una sola vez sobre el entrenamiento con esos "
        "hiperparámetros. No se llamó fit sobre best_estimator_."
    )
    return estimator, decision
