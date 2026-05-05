import csv
import time
from pathlib import Path

from agents import supervisor
from annotation.llm_clients import call_deepseek, call_gpt_oss
from annotation.prompt_builder import build_prompt


EMOTION_ORDER = [
    "amusement",
    "anger",
    "awe",
    "contentment",
    "disgust",
    "excitement",
    "fear",
    "sadness",
]

OUTPUT_MODELS = {
    "deepseek": call_deepseek,
    "gpt_oss": call_gpt_oss,
}


def _intended_emotion_from_filename(filename: str) -> str:
    parts = filename.replace("\\", "/").split("/")
    if len(parts) > 1:
        return parts[0]
    return "unknown"


def _load_existing_filenames(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["filename"] for row in csv.DictReader(handle)}


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filename", "resolved_model", *EMOTION_ORDER]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def annotate_songs(
    songs: list[dict],
    fold_number: int,
    annotations_root: Path | None = None,
    raw_outputs_root: Path | None = None,
    report_dir: Path | None = None,
    user_folds_path: Path | None = None,
    user_responses_path: Path | None = None,
    expected_filenames: list[str] | None = None,
) -> None:
    annotations_root = annotations_root or (Path("data") / "annotations")
    raw_outputs_root = raw_outputs_root or (Path("data") / "raw_outputs")
    output_dir = annotations_root / f"fold_{fold_number}"
    raw_output_dir = raw_outputs_root / f"fold_{fold_number}"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    existing_by_model = {
        model_name: _load_existing_filenames(output_dir / f"{model_name}.csv")
        for model_name in OUTPUT_MODELS
    }

    total = len(songs)
    for index, song in enumerate(songs, start=1):
        filename = song["filename"]
        ground_truth = {emotion: float(song[emotion]) for emotion in EMOTION_ORDER}
        intended_emotion = song.get("intended_emotion") or _intended_emotion_from_filename(filename)
        prompt = build_prompt(
            filename,
            intended_emotion,
            ground_truth,
            song.get("few_shot_examples", []),
        )

        print(f"[fold {fold_number}] Processing {index}/{total}: {filename}")
        for model_name, caller in OUTPUT_MODELS.items():
            if filename in existing_by_model[model_name]:
                print(f"  - {model_name}: skipped (already saved)")
                continue

            raw_output_path = raw_output_dir / f"{model_name}_{index:03d}.json"
            result, resolved_model = caller(prompt, output_path=raw_output_path)
            row = {
                "filename": filename,
                "resolved_model": resolved_model,
                **{emotion: result[emotion] for emotion in EMOTION_ORDER},
            }
            _append_row(output_dir / f"{model_name}.csv", row)
            existing_by_model[model_name].add(filename)
            print(f"  - {model_name}: saved via {resolved_model}")
            time.sleep(1.5)

    try:
        report = supervisor.run(
            fold_number,
            annotations_dir=annotations_root,
            report_dir=report_dir,
            user_folds_path=user_folds_path,
            user_responses_path=user_responses_path,
            expected_filenames=expected_filenames,
        )
    except Exception as exc:
        raise RuntimeError(f"Annotation supervisor failed for fold {fold_number}: {exc}") from exc

    if report.get("overall") != "pass":
        raise RuntimeError(f"Annotation supervisor reported failure for fold {fold_number}: {report}")
