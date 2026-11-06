"""Write tuning_report.md, the metric tables, and results/metrics.json."""

from streamloop import _threads  # noqa: F401

import json
from pathlib import Path

from streamloop.metrics import METRIC_KEYS
from streamloop.plots import save_plots
from streamloop.util import SCORING_REASON_ES, format_params, format_value, plain

METRIC_LABELS = {
    "f2": "F2",
    "recall_churn": "Recall (churn)",
    "precision_churn": "Precision (churn)",
}


def write_deliverables(result, root):
    root = Path(root)
    figures = save_plots(result, root / "docs" / "figures")
    payload = _public_result(result)
    payload["figures"] = {
        "baseline_vs_tuned": "docs/figures/baseline_vs_tuned.png",
        "grid_fold_stability": "docs/figures/grid_fold_stability.png",
        "random_search_scores": "docs/figures/random_search_scores.png",
    }
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_path = root / "tuning_report.md"
    report_path.write_text(render_report(payload), encoding="utf-8")
    _replace_readme_markers(root / "README.md", payload)
    _replace_readme_markers(root / "README.es.md", payload)
    return {
        "report": str(report_path),
        "metrics": str(metrics_path),
        "figures": figures,
    }


def comparison_table(result):
    lines = [
        "| Metric | Baseline | Tuned | Delta (tuned − baseline) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in METRIC_KEYS:
        row = result["comparison"][key]
        lines.append(
            f"| {METRIC_LABELS[key]} | {row['baseline']:.4f} | {row['tuned']:.4f} | {row['delta']:+.4f} |"
        )
    return "\n".join(lines)


def render_report(result):
    baseline = result["baseline"]["metrics"]
    tuned = result["tuned"]["metrics"]
    decision = result["decision"]
    chosen = decision["chosen"]
    champion = decision["champion"]
    dataset = result["dataset"]
    split = result["split"]
    random_search = result["random_search"]
    grid = result["grid_search"]
    source = dataset.get("url") or dataset.get("source")
    sha = dataset.get("sha256", "not pinned (in-memory frame)")
    blank = dataset.get("blank_total_charges", "n/a")
    fold_text = ", ".join(f"{score:.4f}" for score in chosen["folds"])
    champion_folds = ", ".join(f"{score:.4f}" for score in champion["folds"])
    direction = _direction(result["comparison"]["f2"]["delta"])
    lines = [
        "# Informe de ajuste: clasificador de cancelación de StreamLoop",
        "",
        "Este informe compara la línea base con hiperparámetros por defecto y el modelo elegido tras una búsqueda aleatoria seguida de una rejilla. Las cifras salen de `results/metrics.json`, generado por `python main.py`.",
        "",
        "## Resumen",
        "",
        comparison_table(result),
        "",
        (
            f"{direction} "
            f"La línea base acierta {baseline['true_positive']} de {baseline['support_churn']} cancelaciones "
            f"y dispara {baseline['false_positive']} alertas de más. "
            f"El modelo final acierta {tuned['true_positive']} de {tuned['support_churn']} "
            f"y dispara {tuned['false_positive']} alertas de más. "
            "Esa comparación no se usó para elegir hiperparámetros: el test set ya estaba cerrado."
        ),
        "",
        "Accuracy no es la métrica de selección. Con los conteos anteriores se puede reconstruir, y no es lo que la búsqueda maximizó.",
        "",
        "## Por qué F2",
        "",
        SCORING_REASON_ES,
        "",
        "El parámetro `scoring` de `RandomizedSearchCV` y de `GridSearchCV` es esa F2 (`beta=2`), no el accuracy por defecto de scikit-learn. `greater_is_better` queda en verdadero.",
        "",
        "## Protocolo",
        "",
        f"- Dataset: `{source}`. SHA-256: `{sha}`. Filas: {dataset.get('rows', split['n_full'])}. `TotalCharges` en blanco (conteo sobre el archivo, no una estadística del imputer): {blank}.",
        f"- División estratificada {int((1 - split['test_size']) * 100)}/{int(split['test_size'] * 100)} con `random_state={split['random_state']}` **antes** de ajustar cualquier modelo: {split['n_train']} entrenamiento / {split['n_test']} test. Tasa de cancelación en el archivo: {split['full_positive_rate']:.4f}. En entrenamiento: {split['train_positive_rate']:.4f}.",
        "- El identificador `customerID` se elimina dentro del `Pipeline`. `TotalCharges` se convierte a número dentro del `Pipeline`. La mediana de los numéricos y la moda de las categorías se ajustan solo con las filas que ve cada `fit`.",
        "- Clasificador de partida: `HistGradientBoostingClassifier`. La arquitectura no es el objeto de la evaluación; el proceso de búsqueda sí.",
        f"- Validación cruzada: `{result['config']['cv_kind']}`, {result['config']['cv']} pliegues, `random_state={split['random_state']}`. La misma especificación se usa en la búsqueda aleatoria y en la rejilla.",
        f"- `n_jobs={result['config']['n_jobs']}` en ambas búsquedas. `refit=True` en ambas.",
        "- La búsqueda aleatoria ve solo el entrenamiento. La rejilla ve solo el entrenamiento. El test set se puntúa exactamente dos veces: línea base y modelo final. Ledger: "
        + ", ".join(result["ledger"])
        + ".",
        "- No se llama `fit` sobre `best_estimator_` después de la búsqueda.",
        "",
        "## Línea base",
        "",
        f"Estimador: `{result['baseline']['estimator']}`. Hiperparámetros: defaults de scikit-learn, salvo `{result['baseline']['overrides']}`.",
        "",
        "`random_state=42` fija la semilla; no entra en el espacio de búsqueda. El resto de hiperparámetros del modelo de la línea base son los de un constructor vacío de scikit-learn.",
        "",
        f"Test — F2 {baseline['f2']:.4f}, recall {baseline['recall_churn']:.4f}, precision {baseline['precision_churn']:.4f}.",
        "",
        "## Búsqueda aleatoria",
        "",
        (
            f"`RandomizedSearchCV` extrajo {random_search['n_iter']} combinaciones de un espacio de "
            f"{random_search['space_size']}. "
            f"`n_jobs={random_search['n_jobs']}`, `refit=True`, CV de {random_search['cv_splits']} pliegues."
        ),
        "",
        (
            f"Mejor media de validación: F2 {random_search['best_mean_f2']:.4f} "
            f"± {random_search['best_std_f2']:.4f}. Parámetros: `{format_params(random_search['best_params'])}`."
        ),
        "",
        "Esas medias de validación, no el test set, deciden qué región merece una rejilla.",
        "",
        "## Rejilla acotada",
        "",
        (
            f"La rejilla se arma con los mejores resultados de `cv_results_` de la búsqueda aleatoria: "
            f"el rango de valores del top y los vecinos de catálogo del mejor valor, con un tope por hiperparámetro. "
            f"Quedan {grid['n_candidates']} candidatos (producto declarado {grid['space_size']}), "
            f"frente a {random_search['space_size']} del espacio amplio. "
            f"El mejor punto aleatorio sigue dentro de la rejilla."
        ),
        "",
        "Espacio de la rejilla:",
        "",
        "```",
        _format_grid(grid["param_grid"]),
        "```",
        "",
        (
            f"`GridSearchCV` con `n_jobs={grid['n_jobs']}`, `refit=True`, "
            f"los mismos pliegues. Mejor media según el ranking de scikit-learn (índice {grid['best_index']}): "
            f"F2 {grid['best_mean_f2']:.4f} ± {grid['best_std_f2']:.4f}."
        ),
        "",
        "## Variación entre folds y modelo final",
        "",
        decision["rule"],
        "",
        (
            f"Campeón de `refit=True` (índice {champion['index']}): media {champion['mean_f2']:.4f}, "
            f"desviación {champion['std_f2']:.4f}. Folds: {champion_folds}. "
            f"Parámetros: `{format_params(champion['params'])}`."
        ),
        "",
        (
            f"Candidato elegido (índice {chosen['index']}): media {chosen['mean_f2']:.4f}, "
            f"desviación {chosen['std_f2']:.4f}. Folds: {fold_text}. "
            f"Parámetros: `{format_params(chosen['params'])}`."
        ),
        "",
        (
            f"Diferencia de media (elegido − campeón): {chosen['mean_f2'] - champion['mean_f2']:+.4f}. "
            f"Diferencia de desviación: {chosen['std_f2'] - champion['std_f2']:+.4f}. "
            f"Tamaño de la banda de empate: {decision['band_size']} candidatos. "
            f"`used_refit_best_estimator`: {str(decision['used_refit_best_estimator']).lower()}."
        ),
        "",
        decision["how"],
        "",
        "Lectura de `cv_results_`: no se entrega el número de la media sin la desviación entre pliegues. Si la regla se queda con el campeón, es porque dentro de la banda su spread ya es el mejor (o empata). Si se queda con otro, se sacrifica como máximo "
        f"{result['config']['near_tie']:g} de F2 de validación a cambio de menos variación entre pliegues, y ese otro se entrena una vez sobre el entrenamiento.",
        "",
        "### Candidatos de la rejilla",
        "",
        _candidate_table(grid["candidates"], chosen["index"], champion["index"]),
        "",
        "## Comparación en el test",
        "",
        "Las mismas métricas que la línea base, en una sola pasada sobre el test set para el modelo final.",
        "",
        comparison_table(result),
        "",
        "## Hiperparámetros finales",
        "",
        "```",
        format_params(result["tuned"]["hyperparameters"]),
        "```",
        "",
        f"Entrega: {result['tuned']['delivered_via']}",
        "",
        "## Figuras",
        "",
        "![Test scores: baseline vs tuned model](docs/figures/baseline_vs_tuned.png)",
        "",
        "![Grid candidates: mean F2 and fold spread](docs/figures/grid_fold_stability.png)",
        "",
        "![Broad random search: validation F2](docs/figures/random_search_scores.png)",
        "",
        "## Qué queda fuera de esta corrida",
        "",
        "- Ni `RandomizedSearchCV` ni `GridSearchCV` ven el dataset completo ni el test set.",
        "- El CSV oficial no se versiona. `python main.py` lo vuelve a descargar y comprueba su SHA-256.",
        "- No se guarda un modelo serializado: el enunciado no pide el binario, y `python main.py` regenera métricas e informe.",
        "",
        f"Tiempos (segundos): `{result['timings_sec']}`.",
        "",
        f"Versiones: scikit-learn {result['versions']['python_packages']['scikit-learn']}, pandas {result['versions']['python_packages']['pandas']}.",
        "",
    ]
    return "\n".join(lines)


def render_summary(result):
    comparison = result["comparison"]
    return "\n".join(
        [
            "StreamLoop churn tuning",
            f"Dataset rows: {result['dataset'].get('rows', result['split']['n_full'])}  source: {result['dataset'].get('url', result['dataset'].get('source'))}",
            f"SHA-256: {result['dataset'].get('sha256', 'n/a')}",
            (
                f"Split: {result['split']['n_train']} train / {result['split']['n_test']} test, "
                f"stratified, random_state={result['split']['random_state']}"
            ),
            "Scoring: F2 (beta=2). A missed churn costs more than an unneeded retention offer.",
            (
                f"Baseline test F2: {comparison['f2']['baseline']:.4f}  "
                f"recall: {comparison['recall_churn']['baseline']:.4f}  "
                f"precision: {comparison['precision_churn']['baseline']:.4f}"
            ),
            (
                f"Random search: {result['random_search']['n_iter']} draws of {result['random_search']['space_size']}, "
                f"best CV F2 {result['random_search']['best_mean_f2']:.4f} ± {result['random_search']['best_std_f2']:.4f}, "
                f"n_jobs={result['random_search']['n_jobs']}"
            ),
            (
                f"Grid: {result['grid_search']['n_candidates']} candidates, "
                f"best CV F2 {result['grid_search']['best_mean_f2']:.4f} ± {result['grid_search']['best_std_f2']:.4f}"
            ),
            f"Selected: {result['tuned']['params_text']}",
            f"Delivery: {result['decision']['how']}",
            (
                f"Final test F2: {comparison['f2']['tuned']:.4f}  "
                f"recall: {comparison['recall_churn']['tuned']:.4f}  "
                f"precision: {comparison['precision_churn']['tuned']:.4f}  "
                f"delta F2: {comparison['f2']['delta']:+.4f}"
            ),
            f"Test-set evaluations: {', '.join(result['ledger'])}",
            "Wrote tuning_report.md, results/metrics.json, docs/figures/",
            "",
        ]
    )


def replace_marked_block(text, name, body):
    start = f"<!-- {name}_START -->"
    end = f"<!-- {name}_END -->"
    if start not in text or end not in text:
        raise ValueError(f"Missing markers for {name}.")
    pre, rest = text.split(start, 1)
    _, post = rest.split(end, 1)
    return f"{pre}{start}\n{body.rstrip()}\n{end}{post}"


def _replace_readme_markers(path, result):
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if "<!-- METRICS_TABLE_START -->" not in text:
        return
    text = replace_marked_block(text, "METRICS_TABLE", comparison_table(result))
    if path.name.endswith(".es.md"):
        delivered = (
            "grid.best_estimator_ de refit=True, sin un segundo fit"
            if result["decision"]["used_refit_best_estimator"]
            else "un solo fit sobre el entrenamiento del candidato más estable; no se llamó best_estimator_.fit"
        )
        decision = (
            f"Hiperparámetros finales: `{result['tuned']['params_text']}`. "
            f"Entrega: {delivered}. "
            f"F2 de validación del candidato entregado: {result['decision']['chosen']['mean_f2']:.4f} "
            f"± {result['decision']['chosen']['std_f2']:.4f}."
        )
    else:
        delivered = (
            "grid.best_estimator_ from refit=True, without a second fit"
            if result["decision"]["used_refit_best_estimator"]
            else "one training-split fit of the more stable grid candidate; best_estimator_.fit was not called"
        )
        decision = (
            f"Final hyperparameters: `{result['tuned']['params_text']}`. "
            f"Delivered via {delivered}. "
            f"Validation F2 of the delivered candidate: {result['decision']['chosen']['mean_f2']:.4f} "
            f"± {result['decision']['chosen']['std_f2']:.4f}."
        )
    if "<!-- DECISION_START -->" in text:
        text = replace_marked_block(text, "DECISION", decision)
    path.write_text(text, encoding="utf-8")


def _public_result(result):
    """Drop nothing row-level: the study result is already aggregate."""
    return json.loads(json.dumps(result, default=plain, allow_nan=False))


def _direction(delta):
    if delta > 1e-12:
        return (
            f"En el test set, el F2 ajustado supera al de la línea base por {delta:.4f}."
        )
    if delta < -1e-12:
        return (
            "En el test set, el F2 ajustado queda por debajo del de la línea base "
            f"por {abs(delta):.4f}."
        )
    return "En el test set, el F2 ajustado empata con el de la línea base."


def _format_grid(param_grid):
    lines = []
    for key in sorted(param_grid):
        values = ", ".join(format_value(value) for value in param_grid[key])
        lines.append(f"{key.replace('model__', '')}: {values}")
    return "\n".join(lines)


def _candidate_table(candidates, chosen_index, champion_index):
    ranked = sorted(candidates, key=lambda row: (-row["mean_f2"], row["std_f2"], row["index"]))
    lines = [
        "| Index | Role | Mean F2 | Fold std | Folds | Hyperparameters |",
        "| ---: | --- | ---: | ---: | --- | --- |",
    ]
    for row in ranked:
        roles = []
        if row["index"] == champion_index:
            roles.append("refit champion")
        if row["index"] == chosen_index:
            roles.append("delivered")
        role = ", ".join(roles) if roles else ""
        folds = ", ".join(f"{score:.3f}" for score in row["folds"])
        lines.append(
            f"| {row['index']} | {role} | {row['mean_f2']:.4f} | {row['std_f2']:.4f} | {folds} | `{format_params(row['params'])}` |"
        )
    return "\n".join(lines)
