import csv
import json
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from annotation.llm_clients import USE_MOCK
from evaluation import fold_orchestrator


ROOT_DIR = Path(__file__).resolve().parent.parent
USER_RESPONSES_PATH = ROOT_DIR / "data" / "user_emotion_responses.json"
GROUND_TRUTH_PATH = ROOT_DIR / "data" / "song_emotion_ground_truth.csv"
USER_FOLDS_PATH = ROOT_DIR / "state" / "user_folds.json"
REPORTS_DIR = ROOT_DIR / "state" / "agent_reports"
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


def _normalize_song_key(value: str) -> str:
    return value.replace("\\", "/").removeprefix("songs/")


def _load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_annotation_csv(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["filename"]: {emotion: float(row[emotion]) for emotion in EMOTION_COLUMNS}
        for row in rows
    }


def _load_ground_truth() -> dict:
    with GROUND_TRUTH_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        _normalize_song_key(row["filename"]): {emotion: float(row[emotion]) for emotion in EMOTION_COLUMNS}
        for row in rows
    }


def _load_human_averages(fold_number: int) -> dict:
    fold_assignments = _load_json(USER_FOLDS_PATH, default={})
    fold_info = fold_assignments.get("folds", {}).get(str(fold_number), {})
    test_users = set(fold_info.get("test_users", []))
    raw_data = _load_json(USER_RESPONSES_PATH, default={})

    grouped = {}
    for user_id, user_info in raw_data.get("userData", {}).items():
        if user_id not in test_users:
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


def _vector(values: dict) -> list[float]:
    return [float(values[emotion]) for emotion in EMOTION_COLUMNS]


def _mean_vector(rows: dict) -> dict:
    if not rows:
        return {emotion: 0.0 for emotion in EMOTION_COLUMNS}
    return {
        emotion: sum(row[emotion] for row in rows.values()) / len(rows)
        for emotion in EMOTION_COLUMNS
    }


def _mae(left: dict, right: dict) -> float:
    return sum(abs(left[emotion] - right[emotion]) for emotion in EMOTION_COLUMNS) / len(EMOTION_COLUMNS)


def _cosine(left: dict, right: dict) -> float:
    xs = _vector(left)
    ys = _vector(right)
    numerator = sum(x_value * y_value for x_value, y_value in zip(xs, ys))
    x_norm = math.sqrt(sum(value * value for value in xs))
    y_norm = math.sqrt(sum(value * value for value in ys))
    if x_norm == 0 or y_norm == 0:
        return 0.0
    return numerator / (x_norm * y_norm)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pearson(xs: list[float], ys: list[float]):
    if len(xs) < 2 or len(ys) < 2:
        return None
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    x_deltas = [value - x_mean for value in xs]
    y_deltas = [value - y_mean for value in ys]
    denominator = math.sqrt(sum(delta ** 2 for delta in x_deltas) * sum(delta ** 2 for delta in y_deltas))
    if denominator == 0:
        return None
    numerator = sum(x_delta * y_delta for x_delta, y_delta in zip(x_deltas, y_deltas))
    return numerator / denominator


def _average_ranks(values: list[float]) -> list[float]:
    sorted_pairs = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(sorted_pairs):
        end = start
        while end + 1 < len(sorted_pairs) and sorted_pairs[end + 1][1] == sorted_pairs[start][1]:
            end += 1
        average_rank = (start + end + 2) / 2.0
        for index in range(start, end + 1):
            ranks[sorted_pairs[index][0]] = average_rank
        start = end + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]):
    if len(xs) < 2 or len(ys) < 2:
        return None
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _top_emotion_accuracy(reference_rows: dict, predicted_rows: dict) -> float:
    shared_keys = sorted(set(reference_rows) & set(predicted_rows))
    if not shared_keys:
        return 0.0
    matches = 0
    for song_key in shared_keys:
        ref_top = max(EMOTION_COLUMNS, key=lambda emotion: reference_rows[song_key][emotion])
        pred_top = max(EMOTION_COLUMNS, key=lambda emotion: predicted_rows[song_key][emotion])
        matches += 1 if ref_top == pred_top else 0
    return matches / len(shared_keys)


def _per_emotion_mae(reference_rows: dict, predicted_rows: dict) -> dict:
    scores = {}
    shared_keys = sorted(set(reference_rows) & set(predicted_rows))
    for emotion in EMOTION_COLUMNS:
        values = [abs(reference_rows[song_key][emotion] - predicted_rows[song_key][emotion]) for song_key in shared_keys]
        scores[emotion] = _mean(values)
    return scores


def _krippendorff_alpha(rows_by_source: list[list[float]]) -> float | None:
    if not rows_by_source:
        return None
    values = [value for row in rows_by_source for value in row]
    if len(values) < 2:
        return None
    grand_mean = _mean(values)
    denominator = sum((value - grand_mean) ** 2 for value in values)
    if denominator == 0:
        return 1.0

    disagreement = 0.0
    pair_count = 0
    for index, row in enumerate(rows_by_source):
        for other_row in rows_by_source[index + 1:]:
            for left_value, right_value in zip(row, other_row):
                disagreement += (left_value - right_value) ** 2
                pair_count += 1
    if pair_count == 0:
        return None
    return 1.0 - ((disagreement / pair_count) / (denominator / len(values)))


def _load_fold_bundle(fold_number: int) -> dict:
    human_rows = _load_human_averages(fold_number)
    ground_truth_rows = _load_ground_truth()
    model_rows = {
        model_name: _load_annotation_csv(ANNOTATIONS_DIR / f"fold_{fold_number}" / f"{model_name}.csv")
        for model_name in MODEL_NAMES
    }
    song_keys = sorted(
        set(human_rows)
        & set(ground_truth_rows)
        & set.intersection(*(set(model_rows[model_name]) for model_name in MODEL_NAMES))
    ) if all(model_rows[model_name] for model_name in MODEL_NAMES) else []
    return {
        "human": {song_key: human_rows[song_key] for song_key in song_keys},
        "ground_truth": {song_key: ground_truth_rows[song_key] for song_key in song_keys},
        "models": {
            model_name: {song_key: model_rows[model_name][song_key] for song_key in song_keys}
            for model_name in MODEL_NAMES
        },
        "song_keys": song_keys,
    }


def _model_metrics(reference_rows: dict, predicted_rows: dict) -> dict:
    shared_keys = sorted(set(reference_rows) & set(predicted_rows))
    flat_reference = [reference_rows[song_key][emotion] for song_key in shared_keys for emotion in EMOTION_COLUMNS]
    flat_predicted = [predicted_rows[song_key][emotion] for song_key in shared_keys for emotion in EMOTION_COLUMNS]

    song_cosines = [_cosine(reference_rows[song_key], predicted_rows[song_key]) for song_key in shared_keys]
    aggregate_rows = [_vector(reference_rows[song_key]) for song_key in shared_keys]
    predicted_aggregate_rows = [_vector(predicted_rows[song_key]) for song_key in shared_keys]
    alpha = _krippendorff_alpha(aggregate_rows + predicted_aggregate_rows) if shared_keys else None

    return {
        "mae": _mean(
            [_mae(reference_rows[song_key], predicted_rows[song_key]) for song_key in shared_keys]
        ),
        "cosine_similarity": _mean(song_cosines),
        "pearson": _pearson(flat_reference, flat_predicted),
        "spearman": _spearman(flat_reference, flat_predicted),
        "top_emotion_accuracy": _top_emotion_accuracy(reference_rows, predicted_rows),
        "krippendorff_alpha": alpha,
        "per_emotion_mae": _per_emotion_mae(reference_rows, predicted_rows),
    }


def _render_badge(label: str, color: str) -> None:
    st.markdown(
        f"<span style='display:inline-block;padding:0.25rem 0.6rem;border-radius:999px;background:{color};"
        f"color:white;font-weight:600;'>{label}</span>",
        unsafe_allow_html=True,
    )


def _render_radar(title: str, model_vector: dict, human_vector: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=_vector(model_vector),
            theta=EMOTION_COLUMNS,
            fill="toself",
            name=title,
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=_vector(human_vector),
            theta=EMOTION_COLUMNS,
            fill="toself",
            name="Human Average",
            opacity=0.5,
        )
    )
    fig.update_layout(height=350, margin=dict(l=30, r=30, t=40, b=30), showlegend=True)
    return fig


def _render_overview() -> None:
    st.subheader("Fold Status")
    statuses = fold_orchestrator.get_fold_status()
    st.dataframe(pd.DataFrame(statuses), hide_index=True, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Active Models**")
        st.write(", ".join(MODEL_NAMES))
    with col2:
        st.markdown("**Execution Mode**")
        _render_badge("MOCK" if USE_MOCK else "LIVE", "#d97706" if USE_MOCK else "#15803d")

    state = fold_orchestrator._load_state()
    pending_approval = None
    if state.get("prepared"):
        for fold_index in range(1, fold_orchestrator.N_FOLDS + 1):
            fold_state = state["folds"][str(fold_index)]
            if fold_state["status"] == "completed" and not fold_state["approved_to_proceed"]:
                pending_approval = fold_index
                break

    controls = st.columns(2)
    next_fold = fold_orchestrator.get_next_runnable_fold()
    with controls[0]:
        if st.button("Run Next Fold", use_container_width=True):
            try:
                fold_to_run = next_fold or 1
                with st.spinner(f"Running fold {fold_to_run}"):
                    if not state.get("prepared"):
                        fold_orchestrator.prepare_folds()
                        fold_to_run = fold_orchestrator.get_next_runnable_fold() or 1
                    fold_orchestrator.run_fold(fold_to_run)
                st.success(f"Fold {fold_to_run} completed.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    with controls[1]:
        if pending_approval:
            if st.button(f"Approve Fold {pending_approval}", use_container_width=True):
                try:
                    fold_orchestrator.approve_fold(pending_approval)
                    st.success(f"Fold {pending_approval} approved.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        else:
            st.caption("No completed fold is waiting for approval.")


def _completed_folds() -> list[int]:
    return [row["fold"] for row in fold_orchestrator.get_fold_status() if row["status"] == "completed"]


def _render_fold_comparison() -> None:
    completed_folds = _completed_folds()
    if not completed_folds:
        st.info("No completed folds available yet.")
        return

    selected_fold = st.selectbox("Select Fold", completed_folds)
    bundle = _load_fold_bundle(selected_fold)
    if not bundle["song_keys"]:
        st.warning("The selected fold does not have a full set of annotation files yet.")
        return

    human_vector = _mean_vector(bundle["human"])
    cols = st.columns(3)
    metric_rows = []
    for index, model_name in enumerate(MODEL_NAMES):
        model_vector = _mean_vector(bundle["models"][model_name])
        metrics = _model_metrics(bundle["human"], bundle["models"][model_name])
        metric_rows.append(
            {
                "model": model_name,
                "mae_vs_human": metrics["mae"],
                "cosine_vs_human": metrics["cosine_similarity"],
                "pearson_vs_human": metrics["pearson"],
                "spearman_vs_human": metrics["spearman"],
                "top_emotion_accuracy": metrics["top_emotion_accuracy"],
                "krippendorff_alpha": metrics["krippendorff_alpha"],
            }
        )
        with cols[index]:
            st.plotly_chart(
                _render_radar(model_name.capitalize(), model_vector, human_vector),
                use_container_width=True,
            )

    st.subheader("Metrics vs Human Average")
    st.dataframe(pd.DataFrame(metric_rows), hide_index=True, use_container_width=True)


def _render_cross_model_analysis() -> None:
    completed_folds = _completed_folds()
    if not completed_folds:
        st.info("No completed folds available yet.")
        return

    aggregate_human = {}
    aggregate_ground_truth = {}
    aggregate_models = {model_name: {} for model_name in MODEL_NAMES}
    for fold_number in completed_folds:
        bundle = _load_fold_bundle(fold_number)
        for song_key in bundle["song_keys"]:
            aggregate_human[f"{fold_number}:{song_key}"] = bundle["human"][song_key]
            aggregate_ground_truth[f"{fold_number}:{song_key}"] = bundle["ground_truth"][song_key]
            for model_name in MODEL_NAMES:
                aggregate_models[model_name][f"{fold_number}:{song_key}"] = bundle["models"][model_name][song_key]

    summary_rows_human = []
    summary_rows_ground_truth = []
    per_emotion_rows = []
    pairwise_rows = []

    for model_name in MODEL_NAMES:
        human_metrics = _model_metrics(aggregate_human, aggregate_models[model_name])
        truth_metrics = _model_metrics(aggregate_ground_truth, aggregate_models[model_name])
        summary_rows_human.append({"model": model_name, **{k: v for k, v in human_metrics.items() if k != "per_emotion_mae"}})
        summary_rows_ground_truth.append(
            {"model": model_name, **{k: v for k, v in truth_metrics.items() if k != "per_emotion_mae"}}
        )
        for emotion, score in human_metrics["per_emotion_mae"].items():
            per_emotion_rows.append({"model": model_name, "emotion": emotion, "mae": score})

    for left_model in MODEL_NAMES:
        for right_model in MODEL_NAMES:
            pairwise_rows.append(
                {
                    "left": left_model,
                    "right": right_model,
                    "cosine": _cosine(
                        _mean_vector(aggregate_models[left_model]),
                        _mean_vector(aggregate_models[right_model]),
                    ),
                }
            )

    st.subheader("Aggregate Metrics vs Human Average")
    st.dataframe(pd.DataFrame(summary_rows_human), hide_index=True, use_container_width=True)

    st.subheader("Aggregate Metrics vs Expert Ground Truth")
    st.dataframe(pd.DataFrame(summary_rows_ground_truth), hide_index=True, use_container_width=True)

    bar_fig = px.bar(
        pd.DataFrame(per_emotion_rows),
        x="emotion",
        y="mae",
        color="model",
        barmode="group",
        title="MAE per Emotion per Model",
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    heatmap_frame = pd.DataFrame(pairwise_rows).pivot(index="left", columns="right", values="cosine")
    heatmap_fig = px.imshow(
        heatmap_frame,
        text_auto=".3f",
        color_continuous_scale="Blues",
        title="Pairwise Cosine Similarity Between Models",
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)


def _report_status(report: dict) -> str:
    return report.get("overall") or report.get("status") or "unknown"


def _render_agent_reports() -> None:
    report_paths = sorted(REPORTS_DIR.glob("*.json")) if REPORTS_DIR.exists() else []
    if not report_paths:
        st.info("No agent reports available yet.")
        return

    selected = st.selectbox("Select Report", report_paths, format_func=lambda path: path.name)
    report = _load_json(selected, default={})
    status = _report_status(report)
    _render_badge(status.upper(), "#15803d" if status == "pass" else "#b91c1c")
    st.json(report)


def render(page_name: str):
    st.header("LLM Analysis")

    if page_name == "Overview":
        _render_overview()
    elif page_name == "Fold Comparison":
        _render_fold_comparison()
    elif page_name == "Cross-Model Analysis":
        _render_cross_model_analysis()
    elif page_name == "Agent Reports":
        _render_agent_reports()
    else:
        st.info("Unknown LLM Analysis page.")
