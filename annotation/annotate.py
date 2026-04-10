import csv
from pathlib import Path

from agents import supervisor
from annotation.llm_clients import call_deepseek, call_gemini, call_mistral
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
    "gemini": call_gemini,
    "mistral": call_mistral,
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
    fieldnames = ["filename", *EMOTION_ORDER]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def annotate_songs(songs: list[dict], fold_number: int) -> None:
    output_dir = Path("data/annotations") / f"fold_{fold_number}"
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_by_model = {
        model_name: _load_existing_filenames(output_dir / f"{model_name}.csv")
        for model_name in OUTPUT_MODELS
    }

    total = len(songs)
    for index, song in enumerate(songs, start=1):
        filename = song["filename"]
        ground_truth = {emotion: float(song[emotion]) for emotion in EMOTION_ORDER}
        intended_emotion = song.get("intended_emotion") or _intended_emotion_from_filename(filename)
        prompt = build_prompt(filename, intended_emotion, ground_truth)

        print(f"[fold {fold_number}] Processing {index}/{total}: {filename}")
        for model_name, caller in OUTPUT_MODELS.items():
            if filename in existing_by_model[model_name]:
                print(f"  - {model_name}: skipped (already saved)")
                continue

            result = caller(prompt)
            row = {"filename": filename, **{emotion: result[emotion] for emotion in EMOTION_ORDER}}
            _append_row(output_dir / f"{model_name}.csv", row)
            existing_by_model[model_name].add(filename)
            print(f"  - {model_name}: saved")

    try:
        report = supervisor.run(fold_number)
    except Exception as exc:
        raise RuntimeError(f"Annotation supervisor failed for fold {fold_number}: {exc}") from exc

    if report.get("overall") != "pass":
        raise RuntimeError(f"Annotation supervisor reported failure for fold {fold_number}: {report}")
