import json
from pathlib import Path

from agents import consistency_agent, quality_agent, security_agent


ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "state" / "agent_reports"


def _inline_self_check(fold_number: int, annotations_dir: Path | None = None) -> dict:
    return security_agent.run(
        fold_number,
        files_to_check=["agents/supervisor.py"],
        annotations_dir=annotations_dir,
    )


def _record_agent_result(results: list[dict], agent_result: dict) -> None:
    print(f"[{agent_result['agent']}] {agent_result['status'].upper()}")
    if agent_result["issues"]:
        for issue in agent_result["issues"]:
            print(f"  - {issue}")
    results.append(agent_result)


def run(
    fold_number,
    annotations_dir: Path | None = None,
    report_dir: Path | None = None,
    user_folds_path: Path | None = None,
    user_responses_path: Path | None = None,
    expected_filenames: list[str] | None = None,
) -> dict:
    report_dir = report_dir or REPORTS_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    results = []
    security_result = security_agent.run(fold_number, annotations_dir=annotations_dir)
    _record_agent_result(results, security_result)
    _record_agent_result(results, _inline_self_check(fold_number, annotations_dir=annotations_dir))

    if security_result["status"] == "pass":
        quality_result = quality_agent.run(
            fold_number,
            annotations_dir=annotations_dir,
            user_folds_path=user_folds_path,
            user_responses_path=user_responses_path,
            expected_filenames=expected_filenames,
        )
        _record_agent_result(results, quality_result)
        _record_agent_result(results, _inline_self_check(fold_number, annotations_dir=annotations_dir))
    else:
        quality_result = None

    if quality_result and quality_result["status"] == "pass":
        consistency_result = consistency_agent.run(fold_number, annotations_dir=annotations_dir)
        _record_agent_result(results, consistency_result)
        _record_agent_result(results, _inline_self_check(fold_number, annotations_dir=annotations_dir))

    overall = "pass" if all(result["status"] == "pass" for result in results) else "fail"
    report = {
        "overall": overall,
        "fold": fold_number,
        "agents": results,
    }

    report_path = report_dir / f"fold_{fold_number}_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if overall != "pass":
        raise RuntimeError(f"Supervisor checks failed for fold {fold_number}: {json.dumps(report, indent=2)}")

    return report
