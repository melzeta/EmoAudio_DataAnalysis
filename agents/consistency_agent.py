import csv
import math
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = ROOT_DIR / "data" / "annotations"
EMOTION_COLUMNS = [
    "amusement",
    "anger",
    "awe",
    "contentment",
    "disgust",
    "excitement",
    "fear",
    "sadness",
]
MODEL_NAMES = ["deepseek", "gemini", "mistral"]


def _load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _flatten_values(rows: list[dict]) -> list[float]:
    return [float(row[emotion]) for row in rows for emotion in EMOTION_COLUMNS]


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _cosine_similarity(xs: list[float], ys: list[float]) -> float:
    numerator = sum(x_value * y_value for x_value, y_value in zip(xs, ys))
    x_norm = math.sqrt(sum(value * value for value in xs))
    y_norm = math.sqrt(sum(value * value for value in ys))
    if x_norm == 0 or y_norm == 0:
        return 0.0
    return numerator / (x_norm * y_norm)


def run(fold_number) -> dict:
    issues = []
    fold_dir = ANNOTATIONS_DIR / f"fold_{fold_number}"
    model_rows = {}
    for model_name in MODEL_NAMES:
        path = fold_dir / f"{model_name}.csv"
        if not path.exists():
            issues.append(f"Missing annotation file: {path.relative_to(ROOT_DIR)}")
            continue
        model_rows[model_name] = _load_rows(path)

    for model_name, rows in model_rows.items():
        values = _flatten_values(rows)
        if _stddev(values) < 0.02:
            issues.append(f"Overall standard deviation too low for {model_name} in fold {fold_number}")

        for emotion in EMOTION_COLUMNS:
            column_values = [float(row[emotion]) for row in rows]
            if _stddev(column_values) == 0.0:
                issues.append(f"Zero variance detected for {model_name} column {emotion} in fold {fold_number}")

    for left_index, left_model in enumerate(MODEL_NAMES):
        for right_model in MODEL_NAMES[left_index + 1:]:
            if left_model not in model_rows or right_model not in model_rows:
                continue
            left_values = _flatten_values(model_rows[left_model])
            right_values = _flatten_values(model_rows[right_model])
            if left_values and right_values and _cosine_similarity(left_values, right_values) > 0.98:
                issues.append(
                    f"Pairwise cosine similarity above 0.98 for {left_model} vs {right_model} in fold {fold_number}"
                )

    return {
        "agent": "consistency",
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "fold": fold_number,
    }
