import csv
import hashlib
import json
import math
from pathlib import Path

from annotation.annotate import annotate_songs
from annotation.llm_clients import get_run_mode
from evaluation.fold_users_5k import N_FOLDS, USER_FOLDS_PATH, WORKFLOW_ROOT, build_user_folds
from evaluation.utils import utc_now


ROOT_DIR = Path(__file__).resolve().parent.parent
USER_RESPONSES_PATH = ROOT_DIR / "data" / "user_emotion_responses.json"
GROUND_TRUTH_PATH = ROOT_DIR / "data" / "song_emotion_ground_truth.csv"
STATE_DIR = WORKFLOW_ROOT / "state"
WORKFLOW_STATE_PATH = STATE_DIR / "fold_workflow.json"
DATA_DIR = WORKFLOW_ROOT / "data"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
RAW_OUTPUTS_DIR = DATA_DIR / "raw_outputs"
LLM_ANALYSIS_DIR = STATE_DIR / "llm_analysis"
PROMPT_CONTEXTS_DIR = LLM_ANALYSIS_DIR / "prompt_contexts"
REPORTS_DIR = STATE_DIR / "agent_reports"
PROMPT_METHOD = "few_shot_in_context_learning_5k_fold"
MAX_FEW_SHOT_EXAMPLES = 6
MAX_SAME_EMOTION_EXAMPLES = 3
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


def _write_annotation_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filename", *EMOTION_COLUMNS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _analysis_metrics_path(fold_number: int) -> Path:
    return LLM_ANALYSIS_DIR / f"fold_{fold_number}_metrics.json"


def _prompt_contexts_path(fold_number: int) -> Path:
    return PROMPT_CONTEXTS_DIR / f"fold_{fold_number}_prompt_contexts.json"


def _annotation_dir(fold_number: int) -> Path:
    return ANNOTATIONS_DIR / f"fold_{fold_number}"


def _annotation_manifest_path(fold_number: int) -> Path:
    return _annotation_dir(fold_number) / "run_manifest.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _source_file_metadata() -> dict:
    return {
        "user_responses": {
            "path": str(USER_RESPONSES_PATH),
            "sha256": _sha256_file(USER_RESPONSES_PATH),
        },
        "ground_truth": {
            "path": str(GROUND_TRUTH_PATH),
            "sha256": _sha256_file(GROUND_TRUTH_PATH),
        },
    }


def _vector_cosine(left: dict, right: dict) -> float:
    numerator = sum(float(left[emotion]) * float(right[emotion]) for emotion in EMOTION_COLUMNS)
    left_norm = math.sqrt(sum(float(left[emotion]) ** 2 for emotion in EMOTION_COLUMNS))
    right_norm = math.sqrt(sum(float(right[emotion]) ** 2 for emotion in EMOTION_COLUMNS))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _load_annotation_manifest(fold_number: int) -> dict | None:
    path = _annotation_manifest_path(fold_number)
    if not path.exists():
        return None
    return _read_json(path, default={})


def _write_annotation_manifest(fold_number: int, payload: dict) -> Path:
    path = _annotation_manifest_path(fold_number)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _has_existing_annotation_outputs(fold_number: int) -> bool:
    fold_dir = _annotation_dir(fold_number)
    if not fold_dir.exists():
        return False
    return any(path.suffix.lower() == ".csv" for path in fold_dir.glob("*.csv"))


def _assert_annotation_storage_is_compatible(fold_number: int) -> None:
    manifest = _load_annotation_manifest(fold_number)
    current_mode = get_run_mode()
    current_sources = _source_file_metadata()

    if manifest is None:
        if _has_existing_annotation_outputs(fold_number):
            raise RuntimeError(
                f"Fold {fold_number} already has annotation CSVs but no run manifest. "
                f"Clean {(_annotation_dir(fold_number)).relative_to(ROOT_DIR)} before running this fold again."
            )
        return

    existing_mode = manifest.get("run_mode")
    if existing_mode and existing_mode != current_mode:
        raise RuntimeError(
            f"Fold {fold_number} already contains {existing_mode} annotations. "
            f"Current mode is {current_mode}. Clean fold artifacts before re-running."
        )

    existing_sources = manifest.get("source_files", {})
    if existing_sources and existing_sources != current_sources:
        raise RuntimeError(
            f"Fold {fold_number} was created from different source data. "
            "Clean the fold artifacts before running it against the current data files."
        )

    existing_prompt_method = manifest.get("prompt_method")
    if existing_prompt_method and existing_prompt_method != PROMPT_METHOD:
        raise RuntimeError(
            f"Fold {fold_number} was created with prompt method '{existing_prompt_method}', "
            f"but the current workflow expects '{PROMPT_METHOD}'. Clean the fold artifacts before re-running."
        )


def _prepare_annotation_run_manifest(fold_number: int, test_users: set[str]) -> None:
    manifest = _load_annotation_manifest(fold_number) or {}
    if manifest.get("status") == "completed":
        return

    _write_annotation_manifest(
        fold_number,
        {
            **manifest,
            "fold": fold_number,
            "status": "running",
            "run_mode": get_run_mode(),
            "prompt_method": PROMPT_METHOD,
            "max_few_shot_examples": MAX_FEW_SHOT_EXAMPLES,
            "max_same_emotion_examples": MAX_SAME_EMOTION_EXAMPLES,
            "source_files": _source_file_metadata(),
            "test_users": sorted(test_users),
            "started_at": manifest.get("started_at") or utc_now(),
            "updated_at": utc_now(),
        },
    )


def _persist_fold_artifacts(
    fold_number: int,
    song_keys: list[str],
    test_users: set[str],
    baseline_counts: dict,
    train_song_profile_count: int,
    prompt_context_path: Path,
    prompt_contexts: dict,
) -> dict:
    from evaluation.metrics_llm_5k import aggregate_metrics_path, persist_all_folds_metrics, persist_fold_metrics

    metrics = persist_fold_metrics(fold_number)
    aggregate = persist_all_folds_metrics()
    source_files = _source_file_metadata()
    report_path = REPORTS_DIR / f"fold_{fold_number}_report.json"
    existing_manifest = _load_annotation_manifest(fold_number) or {}
    manifest_payload = {
        **existing_manifest,
        "fold": fold_number,
        "status": "completed",
        "saved_at": utc_now(),
        "updated_at": utc_now(),
        "run_mode": get_run_mode(),
        "prompt_method": PROMPT_METHOD,
        "max_few_shot_examples": MAX_FEW_SHOT_EXAMPLES,
        "max_same_emotion_examples": MAX_SAME_EMOTION_EXAMPLES,
        "source_files": source_files,
        "test_users": sorted(test_users),
        "train_song_profile_count": train_song_profile_count,
        "song_count": len(song_keys),
        "songs_annotated": song_keys,
        "annotation_files": {
            annotator: str(_annotation_dir(fold_number) / f"{annotator}.csv")
            for annotator in ["deepseek", "gpt_oss", "human_test", "human_consensus"]
        },
        "prompt_contexts_path": str(prompt_context_path),
        "average_few_shot_examples_per_song": round(
            sum(context["few_shot_example_count"] for context in prompt_contexts.values()) / len(prompt_contexts),
            3,
        ) if prompt_contexts else 0.0,
        "agent_report_path": str(report_path),
        "fold_metrics_path": str(_analysis_metrics_path(fold_number)),
        "aggregate_metrics_path": str(aggregate_metrics_path()),
        "metric_song_count": metrics["comparisons"]["human_test"]["ground_truth"]["n_songs"],
        **baseline_counts,
    }
    manifest_path = _write_annotation_manifest(fold_number, manifest_payload)

    summary = {
        "fold": fold_number,
        "timestamp": utc_now(),
        "run_mode": get_run_mode(),
        "prompt_method": PROMPT_METHOD,
        "max_few_shot_examples": MAX_FEW_SHOT_EXAMPLES,
        "max_same_emotion_examples": MAX_SAME_EMOTION_EXAMPLES,
        "test_users": sorted(test_users),
        "train_song_profile_count": train_song_profile_count,
        "songs_annotated": song_keys,
        "song_count": len(song_keys),
        "annotation_dir": str(_annotation_dir(fold_number)),
        "annotation_manifest_path": str(manifest_path),
        "prompt_contexts_path": str(prompt_context_path),
        "average_few_shot_examples_per_song": round(
            sum(context["few_shot_example_count"] for context in prompt_contexts.values()) / len(prompt_contexts),
            3,
        ) if prompt_contexts else 0.0,
        "agent_report_path": str(report_path),
        "fold_metrics_path": str(_analysis_metrics_path(fold_number)),
        "aggregate_metrics_path": str(aggregate_metrics_path()),
        "source_files": source_files,
        "metric_song_count": metrics["comparisons"]["human_test"]["ground_truth"]["n_songs"],
        "aggregate_fold_count": len(aggregate.get("folds", [])),
        **baseline_counts,
    }
    _write_json(STATE_DIR / f"fold_{fold_number}_summary.json", summary)
    return summary


def _create_initial_state(user_folds: dict) -> dict:
    return {
        "prepared": True,
        "prepared_at": utc_now(),
        "current_review_fold": 0,
        "last_completed_fold": 0,
        "last_reviewed_fold": 0,
        "user_count": user_folds.get("user_count", 0),
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


def _build_user_song_profiles(user_ids: set[str]) -> dict:
    raw_data = _load_user_responses()
    ground_truth_by_key = _load_ground_truth_by_key()
    grouped = {}

    for user_id, user_info in raw_data.get("userData", {}).items():
        if user_id not in user_ids:
            continue

        for response in user_info.get("emotionResponses", []):
            song_path = response.get("song")
            emotion_values = response.get("emotionValues")
            if not song_path or not emotion_values:
                continue

            song_key = _normalize_song_key(song_path)
            if song_key not in ground_truth_by_key:
                continue

            bucket = grouped.setdefault(
                song_key,
                {
                    "filename": song_key,
                    "intended_emotion": song_key.split("/")[0] if "/" in song_key else "unknown",
                    "ground_truth_vector": {
                        emotion: float(ground_truth_by_key[song_key][emotion])
                        for emotion in EMOTION_COLUMNS
                    },
                    "responses": {emotion: [] for emotion in EMOTION_COLUMNS},
                },
            )

            for emotion in EMOTION_COLUMNS:
                bucket["responses"][emotion].append(float(emotion_values[emotion]))

    profiles = {}
    for song_key, payload in grouped.items():
        profiles[song_key] = {
            "filename": payload["filename"],
            "intended_emotion": payload["intended_emotion"],
            "ground_truth_vector": payload["ground_truth_vector"],
            "train_user_average": {
                emotion: sum(payload["responses"][emotion]) / len(payload["responses"][emotion])
                for emotion in EMOTION_COLUMNS
            },
            "num_ratings": len(next(iter(payload["responses"].values()), [])),
        }

    return profiles


def _select_few_shot_examples(
    target_song_key: str,
    intended_emotion: str,
    target_ground_truth: dict,
    train_profiles: dict,
) -> list[dict]:
    candidates = []
    for song_key, profile in train_profiles.items():
        if song_key == target_song_key:
            continue

        similarity = _vector_cosine(target_ground_truth, profile["ground_truth_vector"])
        candidates.append(
            {
                **profile,
                "song_key": song_key,
                "similarity": similarity,
                "same_emotion": profile["intended_emotion"] == intended_emotion,
            }
        )

    candidates.sort(
        key=lambda item: (
            1 if item["same_emotion"] else 0,
            item["similarity"],
            item["num_ratings"],
            item["filename"],
        ),
        reverse=True,
    )

    selected = []
    same_emotion_count = 0
    for candidate in candidates:
        if len(selected) >= MAX_FEW_SHOT_EXAMPLES:
            break
        if candidate["same_emotion"] and same_emotion_count >= MAX_SAME_EMOTION_EXAMPLES:
            continue

        selected.append(
            {
                "filename": candidate["filename"],
                "intended_emotion": candidate["intended_emotion"],
                "ground_truth_vector": candidate["ground_truth_vector"],
                "train_user_average": candidate["train_user_average"],
                "num_ratings": candidate["num_ratings"],
                "similarity": round(candidate["similarity"], 6),
            }
        )
        if candidate["same_emotion"]:
            same_emotion_count += 1

    return selected


def prepare_folds() -> dict:
    user_folds = build_user_folds()
    state = _create_initial_state(user_folds)
    state["run_mode"] = get_run_mode()
    state["source_files"] = _source_file_metadata()
    _save_state(state)
    return user_folds


def total_folds() -> int:
    return N_FOLDS


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


def _build_song_payloads(test_users: set[str], train_profiles: dict) -> tuple[list[dict], list[str], dict]:
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
                ground_truth_vector = {
                    emotion: ground_truth_by_key[song_key][emotion]
                    for emotion in EMOTION_COLUMNS
                }
                song_payloads[song_key] = {
                    "filename": ground_truth_by_key[song_key]["filename"],
                    "intended_emotion": song_key.split("/")[0] if "/" in song_key else "unknown",
                    "few_shot_examples": _select_few_shot_examples(
                        song_key,
                        song_key.split("/")[0] if "/" in song_key else "unknown",
                        ground_truth_vector,
                        train_profiles,
                    ),
                    **ground_truth_vector,
                }

    prompt_contexts = {
        key: {
            "filename": song_payloads[key]["filename"],
            "intended_emotion": song_payloads[key]["intended_emotion"],
            "few_shot_example_count": len(song_payloads[key]["few_shot_examples"]),
            "few_shot_examples": song_payloads[key]["few_shot_examples"],
        }
        for key in sorted(song_payloads)
    }
    return [song_payloads[key] for key in sorted(song_payloads)], sorted(song_payloads), prompt_contexts


def _average_song_vectors(user_ids: set[str] | None = None) -> dict:
    raw_data = _load_user_responses()
    grouped = {}
    for user_id, user_info in raw_data.get("userData", {}).items():
        if user_ids is not None and user_id not in user_ids:
            continue
        for response in user_info.get("emotionResponses", []):
            song_path = response.get("song")
            emotion_values = response.get("emotionValues")
            if not song_path or not emotion_values:
                continue
            song_key = _normalize_song_key(song_path)
            bucket = grouped.setdefault(song_key, {emotion: [] for emotion in EMOTION_COLUMNS})
            for emotion in EMOTION_COLUMNS:
                bucket[emotion].append(float(emotion_values[emotion]))

    return {
        song_key: {
            emotion: sum(values[emotion]) / len(values[emotion])
            for emotion in EMOTION_COLUMNS
        }
        for song_key, values in grouped.items()
        if all(values[emotion] for emotion in EMOTION_COLUMNS)
    }


def _export_human_baselines(fold_number: int, song_keys: list[str], test_users: set[str]) -> dict:
    output_dir = ANNOTATIONS_DIR / f"fold_{fold_number}"
    consensus_rows = _average_song_vectors()
    test_rows = _average_song_vectors(test_users)

    human_consensus = [
        {"filename": song_key, **{emotion: consensus_rows[song_key][emotion] for emotion in EMOTION_COLUMNS}}
        for song_key in song_keys
        if song_key in consensus_rows
    ]
    human_test = [
        {"filename": song_key, **{emotion: test_rows[song_key][emotion] for emotion in EMOTION_COLUMNS}}
        for song_key in song_keys
        if song_key in test_rows
    ]

    _write_annotation_csv(output_dir / "human_consensus.csv", human_consensus)
    _write_annotation_csv(output_dir / "human_test.csv", human_test)
    return {
        "human_consensus_count": len(human_consensus),
        "human_test_count": len(human_test),
    }


def _write_prompt_contexts(fold_number: int, payload: dict) -> Path:
    path = _prompt_contexts_path(fold_number)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_fold(fold_number: int) -> dict:
    state = _load_state()
    if not state.get("prepared"):
        prepare_folds()
        state = _load_state()

    _assert_can_run_fold(state, fold_number)
    _assert_annotation_storage_is_compatible(fold_number)
    fold_assignments = _load_fold_assignments()
    fold_info = fold_assignments["folds"][str(fold_number)]
    test_users = set(fold_info["test_users"])
    train_users = set(fold_info["train_users"])
    train_profiles = _build_user_song_profiles(train_users)
    songs, song_keys, prompt_contexts = _build_song_payloads(test_users, train_profiles)
    _prepare_annotation_run_manifest(fold_number, test_users)
    prompt_context_path = _write_prompt_contexts(
        fold_number,
        {
            "fold": fold_number,
            "prompt_method": PROMPT_METHOD,
            "max_few_shot_examples": MAX_FEW_SHOT_EXAMPLES,
            "max_same_emotion_examples": MAX_SAME_EMOTION_EXAMPLES,
            "train_song_profile_count": len(train_profiles),
            "contexts": prompt_contexts,
        },
    )

    annotate_songs(
        songs,
        fold_number,
        annotations_root=ANNOTATIONS_DIR,
        raw_outputs_root=RAW_OUTPUTS_DIR,
        report_dir=REPORTS_DIR,
        user_folds_path=USER_FOLDS_PATH,
        user_responses_path=USER_RESPONSES_PATH,
        expected_filenames=song_keys,
    )
    baseline_counts = _export_human_baselines(fold_number, song_keys, test_users)
    summary = _persist_fold_artifacts(
        fold_number,
        song_keys,
        test_users,
        baseline_counts,
        len(train_profiles),
        prompt_context_path,
        prompt_contexts,
    )

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
