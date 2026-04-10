import csv
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
USER_RESPONSES_PATH = ROOT_DIR / "data" / "user_emotion_responses.json"
USER_FOLDS_PATH = ROOT_DIR / "state" / "user_folds.json"
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


def _load_expected_song_count(fold_number: int) -> tuple[int | None, list[str]]:
    if not USER_FOLDS_PATH.exists():
        return None, [f"Missing fold assignments: {USER_FOLDS_PATH.relative_to(ROOT_DIR)}"]

    with USER_FOLDS_PATH.open("r", encoding="utf-8") as handle:
        fold_assignments = json.load(handle)

    fold_info = fold_assignments.get("folds", {}).get(str(fold_number))
    if not fold_info:
        return None, [f"Fold {fold_number} not found in {USER_FOLDS_PATH.relative_to(ROOT_DIR)}"]

    test_users = set(fold_info.get("test_users", []))
    with USER_RESPONSES_PATH.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    expected_songs = set()
    for user_id, user_info in raw_data.get("userData", {}).items():
        if user_id not in test_users:
            continue
        for response in user_info.get("emotionResponses", []):
            song_path = response.get("song")
            if song_path:
                expected_songs.add(song_path.replace("\\", "/").split("/", 1)[-1])

    return len(expected_songs), []


def _check_csv(path: Path, expected_rows: int) -> list[str]:
    issues = []
    if not path.exists():
        return [f"Missing annotation file: {path.relative_to(ROOT_DIR)}"]

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    fieldnames = rows[0].keys() if rows else []
    expected_columns = {"filename", *EMOTION_COLUMNS}
    missing_columns = expected_columns - set(fieldnames)
    if missing_columns:
        issues.append(f"Missing columns in {path.relative_to(ROOT_DIR)}: {sorted(missing_columns)}")

    if len(rows) != expected_rows:
        issues.append(
            f"Row count mismatch in {path.relative_to(ROOT_DIR)}: expected {expected_rows}, found {len(rows)}"
        )

    for row_index, row in enumerate(rows, start=1):
        for emotion in EMOTION_COLUMNS:
            value = row.get(emotion)
            if value in [None, ""]:
                issues.append(f"Null value in {path.relative_to(ROOT_DIR)} row {row_index} column {emotion}")
                continue
            try:
                numeric_value = float(value)
            except ValueError:
                issues.append(f"Non-float value in {path.relative_to(ROOT_DIR)} row {row_index} column {emotion}")
                continue
            if not 0.0 <= numeric_value <= 1.0:
                issues.append(
                    f"Out-of-range value in {path.relative_to(ROOT_DIR)} row {row_index} column {emotion}: {value}"
                )

    return issues


def run(fold_number) -> dict:
    expected_rows, issues = _load_expected_song_count(fold_number)
    if expected_rows is None:
        return {"agent": "quality", "status": "fail", "issues": issues, "fold": fold_number}

    fold_dir = ANNOTATIONS_DIR / f"fold_{fold_number}"
    for model_name in MODEL_NAMES:
        issues.extend(_check_csv(fold_dir / f"{model_name}.csv", expected_rows))

    return {
        "agent": "quality",
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "fold": fold_number,
    }
