import csv
import math
from pathlib import Path
from evaluation.utils import read_json, write_json

from evaluation import fold_orchestrator

try:
    import krippendorff
except ImportError:  # pragma: no cover - optional dependency fallback
    krippendorff = None


ROOT_DIR = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = ROOT_DIR / "data" / "annotations"
GROUND_TRUTH_PATH = ROOT_DIR / "data" / "song_emotion_ground_truth.csv"
LLM_ANALYSIS_DIR = ROOT_DIR / "state" / "llm_analysis"
EMOTION_COLUMNS = fold_orchestrator.EMOTION_COLUMNS
ANNOTATORS = ["human_test", "human_consensus", "deepseek", "gemini", "mistral", "ground_truth"]


def fold_metrics_path(fold_number: int) -> Path:
    return LLM_ANALYSIS_DIR / f"fold_{fold_number}_metrics.json"


def aggregate_metrics_path() -> Path:
    return LLM_ANALYSIS_DIR / "aggregate_metrics.json"


def _load_annotation_csv(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["filename"]: {emotion: float(row[emotion]) for emotion in EMOTION_COLUMNS}
        for row in rows
    }


def _load_ground_truth() -> dict:
    with GROUND_TRUTH_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        fold_orchestrator._normalize_song_key(row["filename"]): {
            emotion: float(row[emotion]) for emotion in EMOTION_COLUMNS
        }
        for row in rows
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rmse(values: list[float]) -> float:
    return math.sqrt(_mean(values))


def _pearson(xs: list[float], ys: list[float]):
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


def _average_ranks(values: list[float]) -> list[float]:
    sorted_pairs = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(sorted_pairs):
        end = start
        while end + 1 < len(sorted_pairs) and sorted_pairs[end + 1][1] == sorted_pairs[start][1]:
            end += 1
        average_rank = (start + end + 2) / 2.0
        for index in range(start, end + 1):
            ranks[sorted_pairs[index][0]] = average_rank
        start = end + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]):
    if len(xs) < 2 or len(ys) < 2:
        return None
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _cosine(left: dict, right: dict) -> float:
    xs = [left[emotion] for emotion in EMOTION_COLUMNS]
    ys = [right[emotion] for emotion in EMOTION_COLUMNS]
    numerator = sum(x_value * y_value for x_value, y_value in zip(xs, ys))
    x_norm = math.sqrt(sum(value * value for value in xs))
    y_norm = math.sqrt(sum(value * value for value in ys))
    if x_norm == 0 or y_norm == 0:
        return 0.0
    return numerator / (x_norm * y_norm)


def _top_emotion_accuracy(reference_rows: dict, predicted_rows: dict) -> float:
    shared_keys = sorted(set(reference_rows) & set(predicted_rows))
    if not shared_keys:
        return 0.0
    correct = 0
    for song_key in shared_keys:
        reference_top = max(EMOTION_COLUMNS, key=lambda emotion: reference_rows[song_key][emotion])
        predicted_top = max(EMOTION_COLUMNS, key=lambda emotion: predicted_rows[song_key][emotion])
        correct += 1 if reference_top == predicted_top else 0
    return correct / len(shared_keys)


def _krippendorff_alpha_manual(rows_by_source: list[list[float]]) -> float | None:
    if not rows_by_source:
        return None
    values = [value for row in rows_by_source for value in row]
    if len(values) < 2:
        return None
    grand_mean = _mean(values)
    denominator = sum((value - grand_mean) ** 2 for value in values)
    if denominator == 0:
        return 1.0

    disagreement = 0.0
    pair_count = 0
    for index, row in enumerate(rows_by_source):
        for other_row in rows_by_source[index + 1:]:
            for left_value, right_value in zip(row, other_row):
                disagreement += (left_value - right_value) ** 2
                pair_count += 1
    if pair_count == 0:
        return None
    return 1.0 - ((disagreement / pair_count) / (denominator / len(values)))


def _krippendorff_alpha(rows_by_source: list[list[float]]) -> float | None:
    if not rows_by_source:
        return None
    if krippendorff is not None:
        try:
            return krippendorff.alpha(reliability_data=rows_by_source, level_of_measurement="interval")
        except Exception:
            pass
    return _krippendorff_alpha_manual(rows_by_source)


def compute_metrics(reference_rows: dict, predicted_rows: dict) -> dict:
    shared_keys = sorted(set(reference_rows) & set(predicted_rows))
    if not shared_keys:
        return {
            "n_songs": 0,
            "mae": {"overall": None, "per_emotion": {emotion: None for emotion in EMOTION_COLUMNS}},
            "rmse": {"overall": None},
            "pearson": {"overall": None, "per_emotion": {emotion: None for emotion in EMOTION_COLUMNS}},
            "spearman": {"overall": None, "per_emotion": {emotion: None for emotion in EMOTION_COLUMNS}},
            "cosine_similarity": {"mean_per_song": None},
            "top_emotion_accuracy": None,
            "krippendorff_alpha": None,
        }

    flat_reference = [reference_rows[song_key][emotion] for song_key in shared_keys for emotion in EMOTION_COLUMNS]
    flat_predicted = [predicted_rows[song_key][emotion] for song_key in shared_keys for emotion in EMOTION_COLUMNS]

    per_emotion_mae = {}
    per_emotion_pearson = {}
    per_emotion_spearman = {}
    squared_errors = []
    song_cosines = []

    for emotion in EMOTION_COLUMNS:
        ref_values = [reference_rows[song_key][emotion] for song_key in shared_keys]
        pred_values = [predicted_rows[song_key][emotion] for song_key in shared_keys]
        per_emotion_mae[emotion] = _mean(
            [abs(reference_rows[song_key][emotion] - predicted_rows[song_key][emotion]) for song_key in shared_keys]
        )
        per_emotion_pearson[emotion] = _pearson(ref_values, pred_values)
        per_emotion_spearman[emotion] = _spearman(ref_values, pred_values)

    for song_key in shared_keys:
        squared_errors.extend(
            (reference_rows[song_key][emotion] - predicted_rows[song_key][emotion]) ** 2
            for emotion in EMOTION_COLUMNS
        )
        song_cosines.append(_cosine(reference_rows[song_key], predicted_rows[song_key]))

    alpha = _krippendorff_alpha(
        [[reference_rows[song_key][emotion] for song_key in shared_keys for emotion in EMOTION_COLUMNS]]
        + [[predicted_rows[song_key][emotion] for song_key in shared_keys for emotion in EMOTION_COLUMNS]]
    )

    return {
        "n_songs": len(shared_keys),
        "mae": {
            "overall": _mean([abs(left - right) for left, right in zip(flat_reference, flat_predicted)]),
            "per_emotion": per_emotion_mae,
        },
        "rmse": {"overall": _rmse(squared_errors)},
        "pearson": {"overall": _pearson(flat_reference, flat_predicted), "per_emotion": per_emotion_pearson},
        "spearman": {"overall": _spearman(flat_reference, flat_predicted), "per_emotion": per_emotion_spearman},
        "cosine_similarity": {"mean_per_song": _mean(song_cosines)},
        "top_emotion_accuracy": _top_emotion_accuracy(reference_rows, predicted_rows),
        "krippendorff_alpha": alpha,
    }


def _load_fold_annotators(fold_number: int) -> dict:
    fold_dir = ANNOTATIONS_DIR / f"fold_{fold_number}"
    annotators = {
        annotator: _load_annotation_csv(fold_dir / f"{annotator}.csv")
        for annotator in ["human_test", "human_consensus", "deepseek", "gemini", "mistral"]
    }
    ground_truth = _load_ground_truth()
    shared_keys = None
    for annotator_rows in annotators.values():
        if shared_keys is None:
            shared_keys = set(annotator_rows)
        else:
            shared_keys &= set(annotator_rows)
    shared_keys = shared_keys or set()
    shared_keys &= set(ground_truth)

    trimmed = {
        annotator: {song_key: rows[song_key] for song_key in sorted(shared_keys)}
        for annotator, rows in annotators.items()
    }
    trimmed["ground_truth"] = {song_key: ground_truth[song_key] for song_key in sorted(shared_keys)}
    return trimmed


def compute_fold_metrics(fold_number: int) -> dict:
    annotators = _load_fold_annotators(fold_number)
    comparisons = {}
    for reference_name in ANNOTATORS:
        comparisons[reference_name] = {}
        for predicted_name in ANNOTATORS:
            comparisons[reference_name][predicted_name] = compute_metrics(
                annotators.get(reference_name, {}),
                annotators.get(predicted_name, {}),
            )
    return {
        "fold": fold_number,
        "annotators": annotators,
        "comparisons": comparisons,
    }


def persist_fold_metrics(fold_number: int) -> dict:
    metrics = compute_fold_metrics(fold_number)
    write_json(fold_metrics_path(fold_number), metrics)
    return metrics


def load_saved_fold_metrics(fold_number: int) -> dict | None:
    return read_json(fold_metrics_path(fold_number), default=None)


def load_or_compute_fold_metrics(fold_number: int) -> dict:
    saved = load_saved_fold_metrics(fold_number)
    if saved is not None:
        return saved
    return persist_fold_metrics(fold_number)


def compute_all_folds_metrics() -> dict:
    fold_results = []
    for row in fold_orchestrator.get_fold_status():
        if row["status"] == "completed":
            fold_results.append(compute_fold_metrics(row["fold"]))

    aggregate = {}
    if fold_results:
        for reference_name in ANNOTATORS:
            aggregate[reference_name] = {}
            for predicted_name in ANNOTATORS:
                pair_results = [fold["comparisons"][reference_name][predicted_name] for fold in fold_results]
                aggregate[reference_name][predicted_name] = {
                    "mae_overall_mean": _mean(
                        [result["mae"]["overall"] for result in pair_results if result["mae"]["overall"] is not None]
                    ),
                    "rmse_overall_mean": _mean(
                        [result["rmse"]["overall"] for result in pair_results if result["rmse"]["overall"] is not None]
                    ),
                    "pearson_overall_mean": _mean(
                        [
                            result["pearson"]["overall"]
                            for result in pair_results
                            if result["pearson"]["overall"] is not None
                        ]
                    ),
                    "spearman_overall_mean": _mean(
                        [
                            result["spearman"]["overall"]
                            for result in pair_results
                            if result["spearman"]["overall"] is not None
                        ]
                    ),
                    "cosine_similarity_mean": _mean(
                        [
                            result["cosine_similarity"]["mean_per_song"]
                            for result in pair_results
                            if result["cosine_similarity"]["mean_per_song"] is not None
                        ]
                    ),
                    "top_emotion_accuracy_mean": _mean(
                        [
                            result["top_emotion_accuracy"]
                            for result in pair_results
                            if result["top_emotion_accuracy"] is not None
                        ]
                    ),
                    "krippendorff_alpha_mean": _mean(
                        [
                            result["krippendorff_alpha"]
                            for result in pair_results
                            if result["krippendorff_alpha"] is not None
                        ]
                    ),
                }
    return {
        "folds": fold_results,
        "aggregate": aggregate,
    }


def persist_all_folds_metrics() -> dict:
    results = compute_all_folds_metrics()
    write_json(aggregate_metrics_path(), results)
    return results


def load_saved_all_folds_metrics() -> dict | None:
    return read_json(aggregate_metrics_path(), default=None)


def load_or_compute_all_folds_metrics() -> dict:
    saved = load_saved_all_folds_metrics()
    if saved is not None:
        return saved
    return persist_all_folds_metrics()
