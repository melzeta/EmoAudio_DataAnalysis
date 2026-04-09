import math

from evaluation.constants import EMOTION_COLUMNS


def _to_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def enrich_metrics(metrics_summary: dict, prediction_rows: list[dict]) -> dict:
    metrics = metrics_summary.get("metrics", {}) if metrics_summary else {}
    overall = dict(metrics.get("overall", {}))
    per_emotion = {emotion: dict(values) for emotion, values in metrics.get("per_emotion", {}).items()}

    if not prediction_rows:
        return {"overall": overall, "per_emotion": per_emotion}

    vector_cosines = []
    top_hits = []

    for row in prediction_rows:
        pred_values = {emotion: _to_float(row.get(f"pred_{emotion}")) for emotion in EMOTION_COLUMNS}
        true_values = {emotion: _to_float(row.get(f"true_{emotion}")) for emotion in EMOTION_COLUMNS}

        top_hits.append(1.0 if max(pred_values, key=pred_values.get) == max(true_values, key=true_values.get) else 0.0)

        pred_norm = math.sqrt(sum(value * value for value in pred_values.values()))
        true_norm = math.sqrt(sum(value * value for value in true_values.values()))
        if pred_norm == 0 or true_norm == 0:
            vector_cosines.append(0.0)
        else:
            vector_cosines.append(
                sum(pred_values[emotion] * true_values[emotion] for emotion in EMOTION_COLUMNS)
                / (pred_norm * true_norm)
            )

    overall.setdefault("top_emotion_accuracy", sum(top_hits) / len(top_hits))
    overall.setdefault("mean_vector_cosine_similarity", sum(vector_cosines) / len(vector_cosines))

    for emotion in EMOTION_COLUMNS:
        emotion_metrics = per_emotion.setdefault(emotion, {})
        hits = []
        for row in prediction_rows:
            pred_label = _to_float(row.get(f"pred_{emotion}")) >= 0.5
            true_label = _to_float(row.get(f"true_{emotion}")) >= 0.5
            hits.append(1.0 if pred_label == true_label else 0.0)
        emotion_metrics.setdefault("accuracy_at_0_5", sum(hits) / len(hits) if hits else 0.0)

    return {"overall": overall, "per_emotion": per_emotion}
