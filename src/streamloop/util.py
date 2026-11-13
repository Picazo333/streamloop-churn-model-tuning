"""Small type helpers shared by the search, the study, and the report."""

import math

import numpy as np

SCORING_REASON = (
    "StreamLoop loses more money when a customer churns unnoticed than when "
    "a retention offer reaches someone who would have stayed. The search "
    "therefore maximizes F2 (F-beta with beta=2) on the positive class "
    "Churn=Yes. F2 weights recall higher than precision. Sklearn's default "
    "accuracy is not used: it prices both mistakes the same and is flattered "
    "by the larger group of customers who stay."
)

SCORING_REASON_ES = (
    "StreamLoop pierde más dinero cuando un cliente cancela sin que nadie lo "
    "detecte que cuando una oferta de retención llega a alguien que se habría "
    "quedado. La búsqueda maximiza F2 (F-beta con beta=2) sobre la clase "
    "positiva Churn=Yes. F2 pesa más el recall que la precision. No se usa el "
    "accuracy por defecto de scikit-learn: trata los dos errores igual y queda "
    "inflado por el grupo más grande, el de quienes no cancelan."
)

NEAR_TIE_F2 = 0.01


def plain(value):
    """Convert numpy scalars to JSON-safe Python values. Leave other objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    return value


def same_param(left, right):
    """Compare hyperparameter values, including None and numpy scalars."""
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
        return isinstance(left, (bool, np.bool_)) and isinstance(right, (bool, np.bool_)) and bool(left) is bool(right)
    if isinstance(left, (int, float, np.integer, np.floating)) and isinstance(
        right, (int, float, np.integer, np.floating)
    ):
        return abs(float(left) - float(right)) <= 1e-9
    return left == right


def params_equal(left, right):
    if set(left) != set(right):
        return False
    return all(same_param(left[key], right[key]) for key in left)


def format_value(value):
    value = plain(value)
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def format_params(params):
    parts = []
    for key in sorted(params):
        short = key.replace("model__", "")
        parts.append(f"{short}={format_value(params[key])}")
    return ", ".join(parts)
