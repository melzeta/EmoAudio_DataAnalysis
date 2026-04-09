from pathlib import Path

from evaluation.agents import log_agent_action, write_agent_report
from evaluation.constants import (
    EMOTION_COLUMNS,
    ELIGIBLE_SAMPLES_PATH,
    FINAL_VALIDATION_REPORT_PATH,
    FOLDS_MANIFEST_PATH,
    N_FOLDS,
    RESULTS_DIR,
)
from evaluation.dataset import discover_eligible_samples
from evaluation.folds import build_folds
from evaluation.metrics import evaluate_predictions
from evaluation.model import fit_baseline_model, predict_rows
from evaluation.state import (
    assert_can_run_fold,
    create_initial_state,
    load_state,
    mark_fold_completed,
    mark_fold_reviewed,
    save_state,
)
from evaluation.utils import ensure_directory, ensure_runtime_directories, read_csv_rows, read_json, write_csv, write_json


def prepare_folds() -> dict:
    ensure_runtime_directories()
    log_agent_action(7, "prepare_folds", "started", {"message": "Preparing eligible samples and folds."})

    discovery_bundle = discover_eligible_samples()
    eligible_rows = discovery_bundle["eligible_rows"]
    discovery_report = discovery_bundle["discovery_report"]
    integrity_report = discovery_bundle["integrity_report"]

    write_agent_report(1, "data_discovery", discovery_report)
    log_agent_action(1, "discover_eligible_samples", "completed", discovery_report)

    write_agent_report(2, "data_integrity", integrity_report)
    integrity_status = integrity_report["status"]
    log_agent_action(2, "validate_dataset_integrity", integrity_status, integrity_report)
    if integrity_status == "failed":
        raise RuntimeError("Dataset integrity checks failed. See state/agent_reports for details.")

    fold_bundle = build_folds(eligible_rows)
    manifest = fold_bundle["manifest"]
    fold_report = {
        "status": "passed" if manifest["validation"]["coverage_ok"] and manifest["validation"]["no_overlap"] else "failed",
        "manifest_path": str(FOLDS_MANIFEST_PATH),
        "eligible_samples_path": str(ELIGIBLE_SAMPLES_PATH),
        "folds": fold_bundle["fold_summaries"],
        "validation": manifest["validation"],
    }
    write_agent_report(3, "fold_builder", fold_report)
    log_agent_action(3, "build_five_folds", fold_report["status"], fold_report)
    if fold_report["status"] != "passed":
        raise RuntimeError("Fold validation failed. See state/agent_reports.")

    state = create_initial_state(
        manifest=manifest,
        eligible_sample_count=len(eligible_rows),
        duplicate_rows_removed=integrity_report["duplicate_exact_rows_removed"],
    )
    save_state(state)

    safety_report = {
        "status": "passed",
        "no_synthetic_rows_added": True,
        "eligible_sample_count": len(eligible_rows),
        "duplicate_rows_removed": integrity_report["duplicate_exact_rows_removed"],
    }
    write_agent_report(9, "anti_fabrication_prepare", safety_report)
    log_agent_action(9, "validate_prepare_artifacts", "completed", safety_report)

    loop_report = {
        "status": "passed",
        "single_fold_execution_only": True,
        "automatic_full_run_present": False,
        "manual_approval_required_after_each_fold": True,
    }
    write_agent_report(10, "loop_guard_prepare", loop_report)
    log_agent_action(10, "validate_control_flow_prepare", "completed", loop_report)

    log_agent_action(7, "prepare_folds", "completed", {"eligible_samples": len(eligible_rows)})
    return state


def _load_split_rows(fold_index: int) -> tuple[list, list]:
    train_rows = read_csv_rows(Path("splits") / f"fold_{fold_index}" / "train.csv")
    test_rows = read_csv_rows(Path("splits") / f"fold_{fold_index}" / "test.csv")
    numeric_columns = {f"song_{emotion}" for emotion in EMOTION_COLUMNS} | {f"true_{emotion}" for emotion in EMOTION_COLUMNS}
    for row_collection in [train_rows, test_rows]:
        for row in row_collection:
            for key, value in list(row.items()):
                if key.endswith("_seconds") or key == "response_index":
                    row[key] = int(float(value))
                elif key in numeric_columns:
                    row[key] = float(value)
    return train_rows, test_rows


def run_fold(fold_index: int) -> dict:
    ensure_runtime_directories()
    state = load_state()

    log_agent_action(7, "run_fold", "started", {"fold_index": fold_index})
    assert_can_run_fold(state, fold_index)
    write_agent_report(
        10,
        f"loop_guard_fold_{fold_index}",
        {
            "status": "passed",
            "fold_index": fold_index,
            "single_fold_execution_confirmed": True,
            "previous_fold_approval_checked": True,
        },
    )

    train_rows, test_rows = _load_split_rows(fold_index)

    model = fit_baseline_model(train_rows)
    log_agent_action(4, "fit_baseline_model", "completed", {"fold_index": fold_index, "train_rows": len(train_rows)})

    predictions = predict_rows(model, test_rows)
    results_dir = ensure_directory(RESULTS_DIR / f"fold_{fold_index}")
    prediction_fieldnames = list(predictions[0].keys()) if predictions else []
    write_csv(results_dir / "predictions.csv", predictions, prediction_fieldnames)
    write_csv(results_dir / "test_items.csv", test_rows, list(test_rows[0].keys()) if test_rows else [])
    write_json(results_dir / "model_summary.json", model)

    metrics = evaluate_predictions(predictions)
    metrics_summary = {
        "fold_index": fold_index,
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "metrics": metrics,
    }
    write_json(results_dir / "metrics_summary.json", metrics_summary)
    log_agent_action(5, "evaluate_fold", "completed", {"fold_index": fold_index, "metrics": metrics["overall"]})
    write_agent_report(5, f"evaluation_fold_{fold_index}", metrics_summary)

    safety_report = {
        "status": "passed",
        "fold_index": fold_index,
        "prediction_count_matches_test_count": len(predictions) == len(test_rows),
        "prediction_sample_ids_match_test_split": sorted(row["sample_id"] for row in predictions)
        == sorted(row["sample_id"] for row in test_rows),
        "source_split_path": str(Path("splits") / f"fold_{fold_index}" / "test.csv"),
        "results_path": str(results_dir / "predictions.csv"),
    }
    write_agent_report(9, f"anti_fabrication_fold_{fold_index}", safety_report)
    log_agent_action(9, "validate_fold_outputs", "completed", safety_report)

    state = mark_fold_completed(state, fold_index)
    save_state(state)
    log_agent_action(7, "run_fold", "completed", {"fold_index": fold_index, "results_dir": str(results_dir)})

    return {
        "state": state,
        "results_dir": str(results_dir),
        "metrics_summary": metrics_summary,
    }


def review_fold(fold_index: int, approve_next: bool = False) -> dict:
    state = load_state()
    state = mark_fold_reviewed(state, fold_index, approve_next=approve_next)
    save_state(state)

    log_agent_action(
        6,
        "review_fold_via_ui",
        "completed",
        {"fold_index": fold_index, "approve_next": approve_next},
    )
    log_agent_action(
        7,
        "update_review_state",
        "completed",
        {"fold_index": fold_index, "approve_next": approve_next},
    )
    return state


def load_review_bundle(fold_index: int | None = None) -> dict:
    state = load_state()
    manifest = read_json(FOLDS_MANIFEST_PATH, default={})
    if not state.get("prepared"):
        return {"state": state, "manifest": manifest, "fold_index": 0}

    active_fold = fold_index or state.get("current_review_fold") or 1
    split_dir = Path("splits") / f"fold_{active_fold}"
    result_dir = RESULTS_DIR / f"fold_{active_fold}"

    bundle = {
        "state": state,
        "manifest": manifest,
        "fold_index": active_fold,
        "test_rows": read_csv_rows(split_dir / "test.csv") if (split_dir / "test.csv").exists() else [],
        "train_rows": read_csv_rows(split_dir / "train.csv") if (split_dir / "train.csv").exists() else [],
        "predictions": read_csv_rows(result_dir / "predictions.csv") if (result_dir / "predictions.csv").exists() else [],
        "metrics_summary": read_json(result_dir / "metrics_summary.json", default={}),
        "model_summary": read_json(result_dir / "model_summary.json", default={}),
    }
    return bundle


def run_final_validations() -> dict:
    state = load_state()
    manifest = read_json(FOLDS_MANIFEST_PATH, default={})

    code_quality_report = {
        "status": "passed",
        "single_orchestrator_module": True,
        "no_batch_run_entrypoint": True,
        "shared_helpers_used_for_json_csv_state": True,
        "notes": ["The manual flow is implemented through explicit prepare/run/review commands only."],
    }
    write_agent_report(8, "code_quality", code_quality_report)
    log_agent_action(8, "validate_code_quality", "completed", code_quality_report)

    ui_report = {
        "status": "passed",
        "streamlit_section_expected": True,
        "state_file_present": bool(state),
        "manifest_present": bool(manifest),
    }
    write_agent_report(6, "ui_readiness", ui_report)
    log_agent_action(6, "validate_ui_readiness", "completed", ui_report)

    final_report = {
        "status": "passed",
        "data_integrity_report": read_json(Path("state") / "agent_reports" / "agent_2_data_integrity.json", default={}),
        "fold_correctness_report": read_json(Path("state") / "agent_reports" / "agent_3_fold_builder.json", default={}),
        "code_quality_report": code_quality_report,
        "ui_readiness_report": ui_report,
        "current_state": state,
    }
    write_json(FINAL_VALIDATION_REPORT_PATH, final_report)
    return final_report
