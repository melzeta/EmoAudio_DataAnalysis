import csv
import json
from pathlib import Path

from annotation.annotate import annotate_songs
from evaluation.fold_users import N_FOLDS, USER_FOLDS_PATH, build_user_folds
from evaluation.utils import utc_now


ROOT_DIR = Path(__file__).resolve().parent.parent
USER_RESPONSES_PATH = ROOT_DIR / "data" / "user_emotion_responses.json"
GROUND_TRUTH_PATH = ROOT_DIR / "data" / "song_emotion_ground_truth.csv"
STATE_DIR = ROOT_DIR / "state"
WORKFLOW_STATE_PATH = STATE_DIR / "fold_workflow.json"
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


def _normalize_song_key(value: str) -> str:
    return value.replace("\\", "/").removeprefix("songs/")


def _read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _create_initial_state(user_folds: dict) -> dict:
    return {
        "prepared": True,
        "prepared_at": utc_now(),
        "current_review_fold": 0,
        "last_completed_fold": 0,
        "last_reviewed_fold": 0,
        "folds": {
            str(fold_index): {
                "fold_index": fold_index,
                "status": "pending",
                "reviewed": False,
                "approved_to_proceed": False,
                "train_count": user_folds["folds"][str(fold_index)]["train_count"],
                "test_count": user_folds["folds"][str(fold_index)]["test_count"],
                "summary_path": str(STATE_DIR / f"fold_{fold_index}_summary.json"),
            }
            for fold_index in range(1, N_FOLDS + 1)
        },
    }


def _load_state() -> dict:
    return _read_json(WORKFLOW_STATE_PATH, default={})


def _save_state(state: dict) -> None:
    _write_json(WORKFLOW_STATE_PATH, state)


def _assert_can_run_fold(state: dict, fold_number: int) -> None:
    if not state.get("prepared"):
        raise RuntimeError("Folds have not been prepared yet.")
    if str(fold_number) not in state.get("folds", {}):
        raise RuntimeError(f"Fold {fold_number} is not defined.")

    current = state["folds"][str(fold_number)]
    if current["status"] == "completed":
        raise RuntimeError(f"Fold {fold_number} has already been completed.")

    if fold_number > 1:
        previous = state["folds"][str(fold_number - 1)]
        if not previous["reviewed"]:
            raise RuntimeError(
                f"Fold {fold_number - 1} has not been reviewed. Approve it before running fold {fold_number}."
            )
        if not previous["approved_to_proceed"]:
            raise RuntimeError(
                f"Fold {fold_number - 1} has not been approved to proceed. Fold {fold_number} remains locked."
            )


def _mark_fold_completed(state: dict, fold_number: int) -> dict:
    fold_state = state["folds"][str(fold_number)]
    fold_state["status"] = "completed"
    fold_state["reviewed"] = False
    fold_state["approved_to_proceed"] = False
    fold_state["completed_at"] = utc_now()
    state["current_review_fold"] = fold_number
    state["last_completed_fold"] = max(state.get("last_completed_fold", 0), fold_number)
    return state


def _load_ground_truth_by_key() -> dict:
    with GROUND_TRUTH_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        _normalize_song_key(row["filename"]): {
            "filename": _normalize_song_key(row["filename"]),
            **{emotion: float(row[emotion]) for emotion in EMOTION_COLUMNS},
        }
        for row in rows
    }


def _load_user_responses() -> dict:
    with USER_RESPONSES_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_fold_assignments() -> dict:
    if not USER_FOLDS_PATH.exists():
        return build_user_folds()
    return _read_json(USER_FOLDS_PATH, default={})


def prepare_folds() -> dict:
    user_folds = build_user_folds()
    state = _create_initial_state(user_folds)
    _save_state(state)
    return user_folds


def approve_fold(fold_number: int) -> dict:
    state = _load_state()
    if not state.get("prepared"):
        raise RuntimeError("Folds have not been prepared yet.")
    fold_state = state["folds"][str(fold_number)]
    if fold_state["status"] != "completed":
        raise RuntimeError(f"Fold {fold_number} has not been completed yet.")

    fold_state["reviewed"] = True
    fold_state["approved_to_proceed"] = True
    fold_state["reviewed_at"] = utc_now()
    fold_state["approved_at"] = utc_now()
    state["last_reviewed_fold"] = max(state.get("last_reviewed_fold", 0), fold_number)
    _save_state(state)
    return state


def _build_song_payloads(test_users: set[str]) -> tuple[list[dict], list[str]]:
    raw_data = _load_user_responses()
    ground_truth_by_key = _load_ground_truth_by_key()
    song_payloads = {}

    for user_id, user_info in raw_data.get("userData", {}).items():
        if user_id not in test_users:
            continue
        for response in user_info.get("emotionResponses", []):
            song_path = response.get("song")
            if not song_path:
                continue
            song_key = _normalize_song_key(song_path)
            if song_key not in ground_truth_by_key:
                continue
            if song_key not in song_payloads:
                song_payloads[song_key] = {
                    "filename": ground_truth_by_key[song_key]["filename"],
                    "intended_emotion": song_key.split("/")[0] if "/" in song_key else "unknown",
                    **{emotion: ground_truth_by_key[song_key][emotion] for emotion in EMOTION_COLUMNS},
                }

    return [song_payloads[key] for key in sorted(song_payloads)], sorted(song_payloads)


def run_fold(fold_number: int) -> dict:
    state = _load_state()
    if not state.get("prepared"):
        prepare_folds()
        state = _load_state()

    _assert_can_run_fold(state, fold_number)
    fold_assignments = _load_fold_assignments()
    fold_info = fold_assignments["folds"][str(fold_number)]
    test_users = set(fold_info["test_users"])
    songs, song_keys = _build_song_payloads(test_users)

    annotate_songs(songs, fold_number)

    summary = {
        "fold": fold_number,
        "timestamp": utc_now(),
        "test_users": sorted(test_users),
        "songs_annotated": song_keys,
        "song_count": len(song_keys),
        "annotation_dir": str(ANNOTATIONS_DIR / f"fold_{fold_number}"),
    }
    _write_json(STATE_DIR / f"fold_{fold_number}_summary.json", summary)

    state = _mark_fold_completed(state, fold_number)
    _save_state(state)
    return summary


def get_fold_status() -> list[dict]:
    state = _load_state()
    if not state.get("prepared"):
        return [
            {
                "fold": fold_index,
                "status": "pending",
                "approved_to_proceed": False,
                "song_count": 0,
                "timestamp": None,
            }
            for fold_index in range(1, N_FOLDS + 1)
        ]

    statuses = []
    for fold_index in range(1, N_FOLDS + 1):
        fold_state = state["folds"][str(fold_index)]
        summary = _read_json(STATE_DIR / f"fold_{fold_index}_summary.json", default={})
        statuses.append(
            {
                "fold": fold_index,
                "status": fold_state["status"],
                "reviewed": fold_state["reviewed"],
                "approved_to_proceed": fold_state["approved_to_proceed"],
                "song_count": summary.get("song_count", 0),
                "timestamp": summary.get("timestamp") or fold_state.get("completed_at"),
            }
        )
    return statuses


def get_next_runnable_fold() -> int | None:
    state = _load_state()
    if not state.get("prepared"):
        return 1
    for fold_index in range(1, N_FOLDS + 1):
        fold_state = state["folds"][str(fold_index)]
        if fold_state["status"] != "completed":
            try:
                _assert_can_run_fold(state, fold_index)
            except RuntimeError:
                return None
            return fold_index
    return None
