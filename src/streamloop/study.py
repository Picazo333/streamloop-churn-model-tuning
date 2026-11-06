"""Run the baseline, the random search, and the grid in that order.

The test set is scored in exactly two places: after the default pipeline,
and after the final tuned model is chosen.
"""

from streamloop import _threads  # noqa: F401

import time
from dataclasses import dataclass

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from streamloop.data import positive_rate, split_features_target
from streamloop.metrics import METRIC_KEYS, EvaluationLedger
from streamloop.pipeline import build_pipeline, fit_baseline, parameter_overrides
from streamloop.search import (
    BROAD_SPACE,
    candidates_from_cv,
    grid_contains_params,
    narrow_param_grid,
    run_grid_search,
    run_random_search,
    select_final_estimator,
    space_size,
)
from streamloop.util import NEAR_TIE_F2, SCORING_REASON, SCORING_REASON_ES, format_params, plain


@dataclass(frozen=True)
class StudyConfig:
    random_state: int = 42
    test_size: float = 0.2
    cv: int = 5
    n_iter: int = 24
    n_jobs: int = -1
    near_tie: float = NEAR_TIE_F2
    model_kwargs: dict | None = None


PRODUCTION_CONFIG = StudyConfig()


def split_train_test(X, y, test_size, random_state):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def _search_record(search, space, n_iter=None):
    best_index = int(search.best_index_)
    candidates = candidates_from_cv(search.cv_results_)
    champion = candidates[best_index]
    return {
        "n_jobs": int(search.n_jobs),
        "refit": True,
        "cv_splits": int(search.n_splits_),
        "n_candidates": len(candidates),
        "n_iter": n_iter,
        "space_size": space_size(space) if space is not None else None,
        "best_index": best_index,
        "best_params": {key: plain(value) for key, value in search.best_params_.items()},
        "best_mean_f2": champion["mean_f2"],
        "best_std_f2": champion["std_f2"],
        "best_folds": champion["folds"],
        "candidates": candidates,
    }


def run_study(frame, config=PRODUCTION_CONFIG, dataset_meta=None, space=None, caps=None):
    """Fit on the training split only. Score the test split twice."""
    started = time.perf_counter()
    timings = {}
    space = space or BROAD_SPACE
    meta = dict(dataset_meta or {})
    meta.setdefault("source", "in-memory")
    meta.setdefault("rows", int(len(frame)))
    if "blank_total_charges" not in meta and "TotalCharges" in frame.columns:
        meta["blank_total_charges"] = int(frame["TotalCharges"].astype(str).str.strip().eq("").sum())

    mark = time.perf_counter()
    X, y = split_features_target(frame)
    if "Churn" in X.columns or "customerID" in y.index.names:
        raise RuntimeError("Feature and target split is inconsistent.")
    n_full = int(len(X))
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, config.test_size, config.random_state
    )
    if len(X_train.index.intersection(X_test.index)):
        raise RuntimeError("Train and test indices overlap.")
    if len(X_train) + len(X_test) != n_full:
        raise RuntimeError("The split dropped or duplicated rows.")
    timings["split_sec"] = time.perf_counter() - mark

    ledger = EvaluationLedger()

    mark = time.perf_counter()
    baseline = fit_baseline(X_train, y_train, config.model_kwargs)
    baseline_metrics = ledger.evaluate("baseline", baseline, X_test, y_test)
    timings["baseline_sec"] = time.perf_counter() - mark

    mark = time.perf_counter()
    random_search = run_random_search(
        build_pipeline(config.model_kwargs),
        X_train,
        y_train,
        config,
        space=space,
        n_full=n_full,
    )
    timings["random_search_sec"] = time.perf_counter() - mark
    random_record = _search_record(random_search, space, n_iter=config.n_iter)

    mark = time.perf_counter()
    param_grid = narrow_param_grid(random_search.cv_results_, broad_space=space, caps=caps)
    if not grid_contains_params(param_grid, random_search.best_params_):
        raise RuntimeError("The narrowed grid dropped the best random-search parameters.")
    if space_size(param_grid) > space_size(space):
        raise RuntimeError("The grid is larger than the broad space.")
    timings["narrow_sec"] = time.perf_counter() - mark

    mark = time.perf_counter()
    grid_search = run_grid_search(
        build_pipeline(config.model_kwargs),
        param_grid,
        X_train,
        y_train,
        config,
        n_full=n_full,
    )
    timings["grid_search_sec"] = time.perf_counter() - mark
    grid_record = _search_record(grid_search, param_grid)
    grid_record["param_grid"] = {key: [plain(value) for value in values] for key, values in param_grid.items()}

    mark = time.perf_counter()
    final_estimator, decision = select_final_estimator(
        grid_search,
        build_pipeline(config.model_kwargs),
        X_train,
        y_train,
        config.near_tie,
    )
    tuned_metrics = ledger.evaluate("final", final_estimator, X_test, y_test)
    timings["select_and_final_sec"] = time.perf_counter() - mark

    if ledger.events != ["baseline", "final"]:
        raise RuntimeError(f"Test ledger is {ledger.events}, expected exactly baseline then final.")

    comparison = {}
    for key in METRIC_KEYS:
        baseline_value = baseline_metrics[key]
        tuned_value = tuned_metrics[key]
        comparison[key] = {
            "baseline": baseline_value,
            "tuned": tuned_value,
            "delta": tuned_value - baseline_value,
        }

    timings["total_sec"] = time.perf_counter() - started
    final_params = parameter_overrides(final_estimator)
    # random_state is a reproducibility control. Tuned knobs are the search keys.
    tuned_knobs = {key: plain(value) for key, value in decision["chosen"]["params"].items()}

    return {
        "dataset": meta,
        "split": {
            "random_state": config.random_state,
            "test_size": config.test_size,
            "stratified": True,
            "n_full": n_full,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "full_positive_rate": positive_rate(y),
            "train_positive_rate": positive_rate(y_train),
        },
        "scoring": {
            "name": "f2",
            "beta": 2,
            "greater_is_better": True,
            "reason": SCORING_REASON,
            "reason_es": SCORING_REASON_ES,
        },
        "baseline": {
            "estimator": "HistGradientBoostingClassifier",
            "hyperparameters": "sklearn defaults except the overrides below",
            "overrides": parameter_overrides(baseline),
            "metrics": baseline_metrics,
        },
        "random_search": random_record,
        "grid_search": grid_record,
        "decision": decision,
        "tuned": {
            "estimator": "HistGradientBoostingClassifier",
            "hyperparameters": tuned_knobs,
            "overrides_vs_defaults": final_params,
            "metrics": tuned_metrics,
            "delivered_via": decision["how"],
            "params_text": format_params(tuned_knobs),
        },
        "comparison": comparison,
        "ledger": list(ledger.events),
        "timings_sec": {key: round(value, 3) for key, value in timings.items()},
        "versions": {
            "python_packages": {
                "scikit-learn": sklearn.__version__,
                "pandas": pd.__version__,
            }
        },
        "config": {
            "random_state": config.random_state,
            "cv": config.cv,
            "cv_kind": "StratifiedKFold(shuffle=True)",
            "n_iter": config.n_iter,
            "n_jobs": config.n_jobs,
            "near_tie": config.near_tie,
            "broad_space_size": space_size(space),
            "grid_size": space_size(param_grid),
        },
    }
