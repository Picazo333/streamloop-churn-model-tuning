"""Discipline tests for the StreamLoop tuning study.

The end-to-end case uses a tiny in-memory table. It does not download the
IBM file and it does not stand in for those customers.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from streamloop.data import (
    EXPECTED_COLUMNS,
    EXPECTED_ROWS,
    EXPECTED_SHA256,
    OFFICIAL_URL,
    assert_official_payload,
    positive_rate,
    split_features_target,
    validate_telco_frame,
)
from streamloop.metrics import EvaluationLedger, classification_metrics
from streamloop.pipeline import (
    PrepareColumns,
    build_pipeline,
    fit_baseline,
    parameter_overrides,
)
from streamloop.report import render_report, replace_marked_block, write_deliverables
from streamloop.search import (
    BROAD_SPACE,
    DEFAULT_CAPS,
    choose_candidate,
    f2_score,
    f2_scorer,
    narrow_param_grid,
    space_size,
)
from streamloop.study import PRODUCTION_CONFIG, StudyConfig, run_study, split_train_test
from streamloop.util import SCORING_REASON, params_equal, same_param

TINY_SPACE = {
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, None],
    "model__max_iter": [12],
    "model__min_samples_leaf": [20],
    "model__l2_regularization": [0.0],
}
TINY_CONFIG = StudyConfig(
    cv=2,
    n_iter=2,
    n_jobs=1,
    model_kwargs={"max_iter": 12, "learning_rate": 0.1},
)


def synthetic_frame(n=160, seed=0):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(20, 120, n)
    total = [f"{value:.2f}" for value in monthly * np.maximum(tenure, 1)]
    total[0] = " "
    yes = rng.random(n) < 0.35
    yes[:2] = [True, False]
    # Keep both classes large enough for a stratified split and 2-fold CV.
    yes[:50] = True
    yes[50:100] = False
    frame = pd.DataFrame(
        {
            "customerID": [f"id-{i}" for i in range(n)],
            "gender": rng.choice(["Female", "Male"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                n,
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": np.where(yes, "Yes", "No"),
        }
    )
    frame.loc[3, "Partner"] = None
    return frame


def test_official_dataset_contract():
    assert OFFICIAL_URL == (
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
        "master/data/Telco-Customer-Churn.csv"
    )
    assert EXPECTED_ROWS == 7043
    assert EXPECTED_SHA256 == "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
    assert EXPECTED_COLUMNS[0] == "customerID"
    assert EXPECTED_COLUMNS[-1] == "Churn"
    with pytest.raises(ValueError):
        assert_official_payload(b"not-the-ibm-file")
    with pytest.raises(ValueError):
        validate_telco_frame(synthetic_frame())


def test_target_split_and_unknown_label():
    frame = synthetic_frame()
    X, y = split_features_target(frame)
    assert "Churn" not in X.columns
    assert "customerID" in X.columns
    assert set(y.unique()) <= {0, 1}
    assert positive_rate(y) == pytest.approx(float((frame["Churn"] == "Yes").mean()))
    broken = frame.copy()
    broken.loc[0, "Churn"] = "Maybe"
    with pytest.raises(ValueError):
        split_features_target(broken)


def test_split_is_disjoint_before_any_fit():
    frame = synthetic_frame()
    X, y = split_features_target(frame)
    X_train, X_test, y_train, y_test = split_train_test(X, y, 0.2, 42)
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_prepare_drops_id_and_coerces_blank_total_charges():
    frame = synthetic_frame()
    X, _ = split_features_target(frame)
    prepared = PrepareColumns().fit_transform(X)
    assert "customerID" not in prepared.columns
    assert "Churn" not in prepared.columns
    assert pd.api.types.is_numeric_dtype(prepared["TotalCharges"])
    assert prepared["TotalCharges"].isna().sum() == 1


def test_imputer_statistics_come_from_the_fit_rows_only():
    frame = synthetic_frame(n=160, seed=1)
    X, y = split_features_target(frame)
    X = X.copy()
    X.loc[X.index[:100], "TotalCharges"] = "10"
    X.loc[X.index[0], "TotalCharges"] = " "
    X.loc[X.index[100:], "TotalCharges"] = "999"
    pipeline = build_pipeline({"max_iter": 5, "max_depth": 3})
    pipeline.fit(X.iloc[:100], y.iloc[:100])
    imputer = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"]
    total_index = ["tenure", "MonthlyCharges", "TotalCharges"].index("TotalCharges")
    assert imputer.statistics_[total_index] == pytest.approx(10.0)
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    assert names
    assert all("customerID" not in name for name in names)
    assert [step[0] for step in pipeline.steps] == ["prepare", "preprocess", "model"]


def test_baseline_defaults_except_random_state():
    frame = synthetic_frame(n=120, seed=2)
    X, y = split_features_target(frame)
    X_train, _, y_train, _ = split_train_test(X, y, 0.25, 42)
    pipeline = fit_baseline(X_train, y_train, None)
    assert parameter_overrides(pipeline) == {"random_state": 42}


def test_scorer_is_f2_not_accuracy():
    y_true = np.array([1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0])
    assert f2_score(y_true, y_pred) == pytest.approx(fbeta_score(y_true, y_pred, beta=2))
    assert f2_score(y_true, y_pred) != pytest.approx(accuracy_score(y_true, y_pred))

    class _Constant(ClassifierMixin, BaseEstimator):
        classes_ = np.array([0, 1])

        def fit(self, X, y):
            return self

        def predict(self, X):
            return y_pred

        def __sklearn_is_fitted__(self):
            return True

    score = f2_scorer()(_Constant(), np.zeros((len(y_true), 1)), y_true)
    assert score == pytest.approx(f2_score(y_true, y_pred))
    assert "accuracy" in SCORING_REASON
    assert "not used" in SCORING_REASON


def test_narrow_grid_stays_local_and_keeps_the_best_value():
    cv_results = {
        "mean_test_score": [0.50, 0.80, 0.79, 0.40],
        "std_test_score": [0.01, 0.02, 0.03, 0.00],
        "param_model__learning_rate": [0.25, 0.10, 0.15, 0.03],
        "param_model__max_depth": [3, 3, 3, 3],
        "param_model__max_iter": [50, 50, 50, 50],
        "param_model__min_samples_leaf": [10, 10, 10, 10],
        "param_model__l2_regularization": [0.0, 0.0, 0.0, 0.0],
    }
    grid = narrow_param_grid(cv_results, broad_space=BROAD_SPACE, top_k=2, caps=DEFAULT_CAPS)
    assert grid["model__learning_rate"] == [0.05, 0.1, 0.15]
    assert 0.25 not in grid["model__learning_rate"]
    assert 0.03 not in grid["model__learning_rate"]
    assert space_size(grid) < space_size(BROAD_SPACE)
    assert space_size(BROAD_SPACE) == 1280
    cap_product = 1
    for value in DEFAULT_CAPS.values():
        cap_product *= value
    assert cap_product == 3 * 3 * 2 * 2 * 2


def test_choose_candidate_trades_a_small_mean_gap_for_stability():
    candidates = [
        {"index": 0, "mean_f2": 0.70, "std_f2": 0.05, "params": {"model__max_depth": 3}},
        {"index": 1, "mean_f2": 0.695, "std_f2": 0.01, "params": {"model__max_depth": 5}},
        {"index": 2, "mean_f2": 0.60, "std_f2": 0.001, "params": {"model__max_depth": 8}},
    ]
    chosen, champion, band = choose_candidate(candidates, best_index=0, near_tie=0.01)
    assert champion["index"] == 0
    assert chosen["index"] == 1
    assert [row["index"] for row in band] == [0, 1]

    far = [
        {"index": 0, "mean_f2": 0.70, "std_f2": 0.05, "params": {"a": 1}},
        {"index": 1, "mean_f2": 0.68, "std_f2": 0.001, "params": {"a": 2}},
    ]
    chosen, _, _ = choose_candidate(far, best_index=0, near_tie=0.01)
    assert chosen["index"] == 0

    tied = [
        {"index": 0, "mean_f2": 0.70, "std_f2": 0.02, "params": {"a": 1}},
        {"index": 1, "mean_f2": 0.70, "std_f2": 0.02, "params": {"a": 2}},
    ]
    chosen, champion, _ = choose_candidate(tied, best_index=1, near_tie=0.01)
    assert chosen["index"] == champion["index"] == 1


def test_params_equal_accepts_numpy_scalars():
    assert same_param(np.float64(0.1), 0.1)
    assert same_param(np.int64(100), 100)
    assert same_param(None, None)
    assert not same_param(None, 0)
    assert params_equal({"model__max_depth": np.int64(3)}, {"model__max_depth": 3})


def test_production_config_matches_the_brief():
    assert PRODUCTION_CONFIG.n_jobs == -1
    assert PRODUCTION_CONFIG.cv == 5
    assert PRODUCTION_CONFIG.n_iter == 24
    assert PRODUCTION_CONFIG.random_state == 42
    assert PRODUCTION_CONFIG.test_size == 0.2
    assert PRODUCTION_CONFIG.model_kwargs is None
    assert PRODUCTION_CONFIG.near_tie == 0.01
    assert all("random_state" not in name for name in BROAD_SPACE)
    source = open("src/streamloop/search.py", encoding="utf-8").read()
    assert "refit=True" in source
    assert "n_jobs=config.n_jobs" in source
    assert 'scoring="accuracy"' not in source
    assert "scoring='accuracy'" not in source


def test_ledger_rejects_a_third_test_touch():
    ledger = EvaluationLedger()

    class _Model:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    X = pd.DataFrame({"a": [1, 2, 3, 4]})
    y = np.array([0, 1, 0, 1])
    model = _Model()
    ledger.evaluate("baseline", model, X, y)
    ledger.evaluate("final", model, X, y)
    with pytest.raises(RuntimeError):
        ledger.evaluate("final", model, X, y)
    assert ledger.events == ["baseline", "final"]


def test_end_to_end_study_keeps_the_test_set_out_of_search(monkeypatch, tmp_path):
    frame = synthetic_frame()
    seen = []

    random_fit = RandomizedSearchCV.fit
    grid_fit = GridSearchCV.fit

    def spy_random(self, X, y=None, **kwargs):
        seen.append(("random", len(X)))
        return random_fit(self, X, y, **kwargs)

    def spy_grid(self, X, y=None, **kwargs):
        seen.append(("grid", len(X)))
        return grid_fit(self, X, y, **kwargs)

    monkeypatch.setattr(RandomizedSearchCV, "fit", spy_random)
    monkeypatch.setattr(GridSearchCV, "fit", spy_grid)

    calls = {"scorer": 0}
    from streamloop import search as search_module

    real_scorer = search_module.f2_scorer

    def spy_scorer():
        calls["scorer"] += 1
        return real_scorer()

    monkeypatch.setattr(search_module, "f2_scorer", spy_scorer)

    result = run_study(
        frame,
        config=TINY_CONFIG,
        dataset_meta={"source": "synthetic-test", "rows": len(frame), "blank_total_charges": 1},
        space=TINY_SPACE,
    )
    assert result["ledger"] == ["baseline", "final"]
    assert [name for name, _ in seen] == ["random", "grid"]
    assert {rows for _, rows in seen} == {result["split"]["n_train"]}
    assert result["split"]["n_train"] != result["split"]["n_full"]
    assert result["split"]["n_train"] + result["split"]["n_test"] == result["split"]["n_full"]
    assert calls["scorer"] == 2
    assert result["random_search"]["n_jobs"] == 1
    assert result["grid_search"]["n_jobs"] == 1
    assert result["random_search"]["refit"] is True
    assert result["grid_search"]["refit"] is True
    assert set(result["baseline"]["metrics"]) == set(result["tuned"]["metrics"])
    assert "accuracy" not in result["baseline"]["metrics"]
    assert result["scoring"]["beta"] == 2
    assert result["config"]["grid_size"] <= result["config"]["broad_space_size"]
    assert result["decision"]["used_refit_best_estimator"] in {True, False}

    written = write_deliverables(result, tmp_path)
    report = open(written["report"], encoding="utf-8").read()
    assert "RandomizedSearchCV" in report
    assert "GridSearchCV" in report
    assert "beta=2" in report
    assert "refit=True" in report
    assert f"{result['baseline']['metrics']['f2']:.4f}" in report
    assert f"{result['tuned']['metrics']['f2']:.4f}" in report
    assert "best_estimator_" in report
    assert (tmp_path / "docs" / "figures" / "baseline_vs_tuned.png").is_file()
    assert (tmp_path / "docs" / "figures" / "grid_fold_stability.png").is_file()
    assert (tmp_path / "docs" / "figures" / "random_search_scores.png").is_file()
    metrics_text = (tmp_path / "results" / "metrics.json").read_text(encoding="utf-8")
    assert "id-0" not in metrics_text
    rendered = render_report(result)
    assert "F2" in rendered


def test_replace_marked_block():
    text = "before\n<!-- METRICS_TABLE_START -->\nold\n<!-- METRICS_TABLE_END -->\nafter\n"
    updated = replace_marked_block(text, "METRICS_TABLE", "| a |")
    assert "old" not in updated
    assert "| a |" in updated
    assert updated.startswith("before\n")


def test_classification_metrics_counts_balance():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["true_positive"] + metrics["false_negative"] == metrics["support_churn"]
    assert metrics["support_churn"] + metrics["support_stay"] == 4
