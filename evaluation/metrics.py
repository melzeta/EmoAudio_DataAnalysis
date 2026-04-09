import math

from evaluation.constants import EMOTION_COLUMNS


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def _pearson(xs: list, ys: list):
    if len(xs) < 2 or len(ys) < 2:
        return None
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    x_deltas = [value - x_mean for value in xs]
    y_deltas = [value - y_mean for value in ys]
    denominator = math.sqrt(sum(delta ** 2 for delta in x_deltas) * sum(delta ** 2 for delta in y_deltas))
    if denominator == 0:
        return None
    numerator = sum(x_delta * y_delta for x_delta, y_delta in zip(x_deltas, y_deltas))
    return numerator / denominator


def _average_ranks(values: list) -> list:
    sorted_pairs = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(sorted_pairs):
        end = start
        while end + 1 < len(sorted_pairs) and sorted_pairs[end + 1][1] == sorted_pairs[start][1]:
            end += 1
        average_rank = (start + end + 2) / 2.0
        for index in range(start, end + 1):
            original_position = sorted_pairs[index][0]
            ranks[original_position] = average_rank
        start = end + 1
    return ranks


def _spearman(xs: list, ys: list):
    if len(xs) < 2 or len(ys) < 2:
        return None
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _mae(xs: list, ys: list) -> float:
    return _mean([abs(x_value - y_value) for x_value, y_value in zip(xs, ys)])


def _rmse(xs: list, ys: list) -> float:
    return math.sqrt(_mean([(x_value - y_value) ** 2 for x_value, y_value in zip(xs, ys)]))


def _argmax_emotion(prefix: str, row: dict) -> str:
    return max(EMOTION_COLUMNS, key=lambda emotion: row[f"{prefix}{emotion}"])


def evaluate_predictions(prediction_rows: list) -> dict:
    per_emotion = {}
    flat_true = []
    flat_pred = []

    for emotion in EMOTION_COLUMNS:
        true_values = [row[f"true_{emotion}"] for row in prediction_rows]
        pred_values = [row[f"pred_{emotion}"] for row in prediction_rows]
        per_emotion[emotion] = {
            "pearson": _pearson(pred_values, true_values),
            "spearman": _spearman(pred_values, true_values),
            "mae": _mae(pred_values, true_values),
            "rmse": _rmse(pred_values, true_values),
        }
        flat_true.extend(true_values)
        flat_pred.extend(pred_values)

    top_emotion_accuracy = _mean(
        [
            1.0 if _argmax_emotion("pred_", row) == _argmax_emotion("true_", row) else 0.0
            for row in prediction_rows
        ]
    ) if prediction_rows else 0.0

    vector_cosines = []
    for row in prediction_rows:
        pred_vector = [row[f"pred_{emotion}"] for emotion in EMOTION_COLUMNS]
        true_vector = [row[f"true_{emotion}"] for emotion in EMOTION_COLUMNS]
        pred_norm = math.sqrt(sum(value * value for value in pred_vector))
        true_norm = math.sqrt(sum(value * value for value in true_vector))
        if pred_norm == 0 or true_norm == 0:
            vector_cosines.append(0.0)
        else:
            vector_cosines.append(
                sum(pred_value * true_value for pred_value, true_value in zip(pred_vector, true_vector))
                / (pred_norm * true_norm)
            )

    return {
        "n_test_samples": len(prediction_rows),
        "overall": {
            "pearson": _pearson(flat_pred, flat_true),
            "spearman": _spearman(flat_pred, flat_true),
            "mae": _mae(flat_pred, flat_true),
            "rmse": _rmse(flat_pred, flat_true),
            "top_emotion_accuracy": top_emotion_accuracy,
            "mean_vector_cosine_similarity": _mean(vector_cosines),
        },
        "per_emotion": per_emotion,
    }

