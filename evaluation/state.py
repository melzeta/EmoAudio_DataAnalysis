from pathlib import Path

from evaluation.constants import MANUAL_CV_STATE_PATH, N_FOLDS, RANDOM_SEED
from evaluation.utils import read_json, utc_now, write_json


def create_initial_state(manifest: dict, eligible_sample_count: int, duplicate_rows_removed: int) -> dict:
    return {
        "prepared": True,
        "prepared_at": utc_now(),
        "random_seed": RANDOM_SEED,
        "eligible_sample_count": eligible_sample_count,
        "duplicate_rows_removed": duplicate_rows_removed,
        "current_review_fold": 0,
        "last_completed_fold": 0,
        "last_reviewed_fold": 0,
        "folds": {
            str(fold_index): {
                "fold_index": fold_index,
                "status": "pending",
                "reviewed": False,
                "approved_to_proceed": False,
                "train_count": manifest["folds"][str(fold_index)]["train_count"],
                "test_count": manifest["folds"][str(fold_index)]["test_count"],
                "results_dir": str(Path("results") / f"fold_{fold_index}"),
            }
            for fold_index in range(1, N_FOLDS + 1)
        },
    }


def load_state() -> dict:
    return read_json(MANUAL_CV_STATE_PATH, default={})


def save_state(state: dict) -> None:
    write_json(MANUAL_CV_STATE_PATH, state)


def assert_can_run_fold(state: dict, fold_index: int) -> None:
    if not state.get("prepared"):
        raise RuntimeError("Folds have not been prepared yet.")

    if fold_index < 1 or str(fold_index) not in state.get("folds", {}):
        raise RuntimeError(f"Fold {fold_index} is not defined.")

    fold_state = state["folds"][str(fold_index)]
    if fold_state["status"] == "completed":
        raise RuntimeError(f"Fold {fold_index} has already been completed.")

    if fold_index > 1:
        previous_fold = state["folds"][str(fold_index - 1)]
        if not previous_fold["reviewed"]:
            raise RuntimeError(
                f"Fold {fold_index - 1} has not been marked as reviewed. Approval is required before running Fold {fold_index}."
            )
        if not previous_fold["approved_to_proceed"]:
            raise RuntimeError(
                f"Fold {fold_index - 1} has not been approved to proceed. Fold {fold_index} remains locked."
            )


def mark_fold_completed(state: dict, fold_index: int) -> dict:
    fold_state = state["folds"][str(fold_index)]
    fold_state["status"] = "completed"
    fold_state["reviewed"] = False
    fold_state["approved_to_proceed"] = False
    fold_state["completed_at"] = utc_now()
    state["current_review_fold"] = fold_index
    state["last_completed_fold"] = max(state.get("last_completed_fold", 0), fold_index)
    return state


def mark_fold_reviewed(state: dict, fold_index: int, approve_next: bool = False) -> dict:
    fold_state = state["folds"][str(fold_index)]
    if fold_state["status"] != "completed":
        raise RuntimeError(f"Fold {fold_index} has not been completed yet.")

    fold_state["reviewed"] = True
    fold_state["reviewed_at"] = utc_now()
    if approve_next:
        fold_state["approved_to_proceed"] = True
        fold_state["approved_at"] = utc_now()
    state["last_reviewed_fold"] = max(state.get("last_reviewed_fold", 0), fold_index)
    return state

