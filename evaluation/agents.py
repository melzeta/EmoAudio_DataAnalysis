from evaluation.constants import AGENT_LOG_DIR, AGENT_REPORT_DIR
from evaluation.utils import append_jsonl, utc_now, write_json


AGENT_REGISTRY = {
    1: "Data Discovery Agent",
    2: "Data Integrity Agent",
    3: "Fold Builder Agent",
    4: "Training/Inference Agent",
    5: "Evaluation Agent",
    6: "Streamlit UI Agent",
    7: "Orchestration Agent",
    8: "Code Quality Agent",
    9: "Safety & Anti-Fabrication Agent",
    10: "Loop & Logic Validator Agent",
}


def log_agent_action(agent_id: int, action: str, status: str, details: dict) -> None:
    append_jsonl(
        AGENT_LOG_DIR / f"agent_{agent_id}.jsonl",
        {
            "timestamp": utc_now(),
            "agent_id": agent_id,
            "agent_name": AGENT_REGISTRY[agent_id],
            "action": action,
            "status": status,
            "details": details,
        },
    )


def write_agent_report(agent_id: int, report_name: str, payload: dict) -> None:
    write_json(
        AGENT_REPORT_DIR / f"agent_{agent_id}_{report_name}.json",
        {
            "timestamp": utc_now(),
            "agent_id": agent_id,
            "agent_name": AGENT_REGISTRY[agent_id],
            **payload,
        },
    )

