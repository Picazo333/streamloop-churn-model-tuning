"""Test-set scoring. Callers may use this only for the baseline and the final model."""

from streamloop import _threads  # noqa: F401

from sklearn.metrics import confusion_matrix, precision_score, recall_score

from streamloop.search import f2_score

METRIC_KEYS = ("f2", "recall_churn", "precision_churn")


class EvaluationLedger:
    """Counts the two allowed test-set evaluations and rejects a third."""

    def __init__(self):
        self.events = []

    def evaluate(self, name, estimator, X_test, y_test):
        if name not in {"baseline", "final"}:
            raise ValueError(f"Unexpected test evaluation {name!r}.")
        if name in self.events or len(self.events) >= 2:
            raise RuntimeError(f"The test set was already used ({self.events}); refusing {name!r}.")
        predictions = estimator.predict(X_test)
        self.events.append(name)
        return classification_metrics(y_test, predictions)


def classification_metrics(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (int(value) for value in matrix.ravel())
    return {
        "f2": float(f2_score(y_true, y_pred)),
        "recall_churn": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision_churn": float(precision_score(y_true, y_pred, zero_division=0)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "support_churn": true_positive + false_negative,
        "support_stay": true_negative + false_positive,
    }
