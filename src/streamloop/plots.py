"""Charts for the baseline comparison and the fold spread of the grid."""

from streamloop import _threads  # noqa: F401

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from streamloop.metrics import METRIC_KEYS
from streamloop.util import format_value

INK = "#18202B"
MUTED = "#505E70"
ACTION = "#B53A17"
INFO = "#204EA0"
SUCCESS = "#236344"
CANVAS = "#F7F8FA"
SURFACE = "#FFFFFF"

METRIC_LABELS = {
    "f2": "F2",
    "recall_churn": "Recall (churn)",
    "precision_churn": "Precision (churn)",
}


def _style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": CANVAS,
            "axes.facecolor": SURFACE,
            "axes.titleweight": "bold",
            "axes.grid": False,
        }
    )


def _finish(figure, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def save_plots(result, directory):
    directory = Path(directory)
    _style()
    paths = {
        "baseline_vs_tuned": _baseline_vs_tuned(result, directory / "baseline_vs_tuned.png"),
        "grid_fold_stability": _stability(result, directory / "grid_fold_stability.png"),
        "random_search_scores": _random_scores(result, directory / "random_search_scores.png"),
    }
    return {key: str(path) for key, path in paths.items()}


def _baseline_vs_tuned(result, path):
    labels = [METRIC_LABELS[key] for key in METRIC_KEYS]
    baseline = [result["comparison"][key]["baseline"] for key in METRIC_KEYS]
    tuned = [result["comparison"][key]["tuned"] for key in METRIC_KEYS]
    positions = list(range(len(labels)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    bars_base = axis.bar(
        [p - width / 2 for p in positions],
        baseline,
        width=width,
        color=INFO,
        label="Baseline (default hyperparameters)",
    )
    bars_tuned = axis.bar(
        [p + width / 2 for p in positions],
        tuned,
        width=width,
        color=ACTION,
        label="Tuned model",
    )
    axis.set_ylim(0, 1)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Score on the held-out test set")
    axis.set_title("Test scores: baseline vs tuned model")
    axis.legend(frameon=False)
    for bars in (bars_base, bars_tuned):
        for bar in bars:
            height = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(height + 0.03, 0.97),
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
            )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    return _finish(figure, path)


def _compact_params(params):
    def value(name):
        return format_value(params[f"model__{name}"])

    return (
        f"lr {value('learning_rate')}, depth {value('max_depth')}, "
        f"iter {value('max_iter')}, leaf {value('min_samples_leaf')}, l2 {value('l2_regularization')}"
    )


def _stability(result, path):
    candidates = list(result["grid_search"]["candidates"])
    if len(candidates) > 12:
        best = sorted(candidates, key=lambda row: row["mean_f2"], reverse=True)[:11]
        chosen = result["decision"]["chosen"]
        indexes = {row["index"] for row in best}
        if chosen["index"] not in indexes:
            best.append(chosen)
        candidates = best
    candidates = sorted(candidates, key=lambda row: row["mean_f2"])
    labels = []
    means = []
    stds = []
    colors = []
    chosen_index = result["decision"]["chosen"]["index"]
    for row in candidates:
        labels.append(f"{row['index']}: {_compact_params(row['params'])}")
        means.append(row["mean_f2"])
        stds.append(row["std_f2"])
        colors.append(SUCCESS if row["index"] == chosen_index else INFO)
    figure, axis = plt.subplots(figsize=(10.4, max(4.8, 0.42 * len(labels) + 1.4)))
    for position, (mean, std, color) in enumerate(zip(means, stds, colors)):
        axis.errorbar(
            mean,
            position,
            xerr=std,
            fmt="o",
            color=color,
            ecolor=MUTED,
            capsize=3,
            markersize=6,
        )
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=8)
    low = min(mean - std for mean, std in zip(means, stds)) - 0.01
    high = max(mean + std for mean, std in zip(means, stds)) + 0.01
    axis.set_xlim(max(0, low), min(1, high))
    axis.set_xlabel("Mean validation F2, with the fold standard deviation")
    axis.set_title("Grid candidates near the top: mean F2 and fold spread")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.plot([], [], "o", color=SUCCESS, label="Delivered candidate")
    axis.plot([], [], "o", color=INFO, label="Other candidates shown")
    axis.legend(frameon=False, loc="lower right")
    return _finish(figure, path)


def _random_scores(result, path):
    rows = sorted(result["random_search"]["candidates"], key=lambda row: row["mean_f2"])
    scores = [row["mean_f2"] for row in rows]
    best = result["random_search"]["best_mean_f2"]
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    positions = list(range(len(scores)))
    colors = [SUCCESS if abs(score - best) <= 1e-12 else ACTION for score in scores]
    axis.scatter(scores, positions, c=colors, s=36, zorder=3)
    axis.axvline(best, color=INK, linestyle="--", linewidth=1.0, label=f"Best draw ({best:.3f})")
    axis.set_yticks([])
    axis.set_xlabel("Mean validation F2")
    axis.set_ylabel("Draws sorted from lower to higher mean")
    axis.set_title(f"Broad random search: {len(scores)} draws")
    axis.legend(frameon=False, loc="lower right")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if scores:
        pad = 0.015
        axis.set_xlim(max(0, min(scores) - pad), min(1, max(scores) + pad))
    return _finish(figure, path)
