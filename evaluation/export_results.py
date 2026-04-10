import csv
import json
from pathlib import Path

from evaluation.metrics_llm import ANNOTATORS, EMOTION_COLUMNS, compute_all_folds_metrics


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def export_fold_metrics_csv(output_path) -> Path:
    output_path = Path(output_path)
    _ensure_parent(output_path)
    results = compute_all_folds_metrics()
    fieldnames = [
        "fold",
        "reference_annotator",
        "predicted_annotator",
        "mae",
        "rmse",
        "pearson",
        "spearman",
        "cosine_similarity",
        "top_emotion_accuracy",
        "krippendorff_alpha",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for fold_result in results["folds"]:
            fold_number = fold_result["fold"]
            for reference_name in ANNOTATORS:
                for predicted_name in ANNOTATORS:
                    metrics = fold_result["comparisons"][reference_name][predicted_name]
                    writer.writerow(
                        {
                            "fold": fold_number,
                            "reference_annotator": reference_name,
                            "predicted_annotator": predicted_name,
                            "mae": metrics["mae"]["overall"],
                            "rmse": metrics["rmse"]["overall"],
                            "pearson": metrics["pearson"]["overall"],
                            "spearman": metrics["spearman"]["overall"],
                            "cosine_similarity": metrics["cosine_similarity"]["mean_per_song"],
                            "top_emotion_accuracy": metrics["top_emotion_accuracy"],
                            "krippendorff_alpha": metrics["krippendorff_alpha"],
                        }
                    )
    return output_path


def export_per_emotion_csv(output_path) -> Path:
    output_path = Path(output_path)
    _ensure_parent(output_path)
    results = compute_all_folds_metrics()
    fieldnames = ["fold", "annotator", "emotion", "mae", "pearson", "spearman"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for fold_result in results["folds"]:
            fold_number = fold_result["fold"]
            for annotator in ANNOTATORS:
                metrics = fold_result["comparisons"]["human_test"][annotator]
                for emotion in EMOTION_COLUMNS:
                    writer.writerow(
                        {
                            "fold": fold_number,
                            "annotator": annotator,
                            "emotion": emotion,
                            "mae": metrics["mae"]["per_emotion"][emotion],
                            "pearson": metrics["pearson"]["per_emotion"][emotion],
                            "spearman": metrics["spearman"]["per_emotion"][emotion],
                        }
                    )
    return output_path


def export_summary_json(output_path) -> Path:
    output_path = Path(output_path)
    _ensure_parent(output_path)
    results = compute_all_folds_metrics()
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results["aggregate"], handle, indent=2, sort_keys=True)
    return output_path
