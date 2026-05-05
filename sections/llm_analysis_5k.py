import json
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from annotation.llm_clients import USE_MOCK
from evaluation import fold_orchestrator_5k as fold_orchestrator
from evaluation.metrics_llm_5k import EMOTION_COLUMNS, load_or_compute_fold_metrics


ROOT_DIR = fold_orchestrator.WORKFLOW_ROOT
REPORTS_DIR = fold_orchestrator.REPORTS_DIR
DISPLAY_ANNOTATORS = ["deepseek", "gpt_oss"]
LINE_SOURCES = ["held_out_users", "deepseek", "gpt_oss", "human_consensus"]
SOURCE_LABELS = {
    "held_out_users": "Held-out Users",
    "deepseek": "DeepSeek Chat",
    "gpt_oss": "GPT-4o Mini",
    "human_consensus": "All-User Consensus",
    "ground_truth": "Musicologist Vector",
}
SOURCE_COLORS = {
    "held_out_users": "#111827",
    "deepseek": "#2563eb",
    "gpt_oss": "#059669",
    "human_consensus": "#7c3aed",
    "ground_truth": "#f59e0b",
}


def _load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _render_badge(label: str, color: str) -> None:
    st.markdown(
        f"<span style='display:inline-block;padding:0.25rem 0.6rem;border-radius:999px;background:{color};"
        f"color:white;font-weight:600;'>{label}</span>",
        unsafe_allow_html=True,
    )


def _completed_folds() -> list[int]:
    return [row["fold"] for row in fold_orchestrator.get_fold_status() if row["status"] == "completed"]


def _fold_summary(fold_number: int) -> dict:
    return _load_json(fold_orchestrator.STATE_DIR / f"fold_{fold_number}_summary.json", default={})


def _prompt_contexts(fold_number: int) -> dict:
    path = fold_orchestrator.PROMPT_CONTEXTS_DIR / f"fold_{fold_number}_prompt_contexts.json"
    return _load_json(path, default={})


def _raw_output_dir(fold_number: int) -> Path:
    return fold_orchestrator.RAW_OUTPUTS_DIR / f"fold_{fold_number}"


def _report_status(report: dict) -> str:
    return report.get("overall") or report.get("status") or "unknown"


def _vector_cosine(left: dict, right: dict) -> float:
    numerator = sum(float(left[emotion]) * float(right[emotion]) for emotion in EMOTION_COLUMNS)
    left_norm = math.sqrt(sum(float(left[emotion]) ** 2 for emotion in EMOTION_COLUMNS))
    right_norm = math.sqrt(sum(float(right[emotion]) ** 2 for emotion in EMOTION_COLUMNS))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _mean_vector(rows: dict) -> dict:
    if not rows:
        return {emotion: 0.0 for emotion in EMOTION_COLUMNS}
    return {
        emotion: sum(row[emotion] for row in rows.values()) / len(rows)
        for emotion in EMOTION_COLUMNS
    }


def _line_projection_frame(fold_metrics: dict) -> pd.DataFrame:
    annotators = fold_metrics.get("annotators", {})
    held_out_rows = annotators.get("human_test", {})
    if not held_out_rows:
        return pd.DataFrame()

    reference_axis = _mean_vector(held_out_rows)
    ordered_song_keys = sorted(
        held_out_rows,
        key=lambda song_key: _vector_cosine(held_out_rows[song_key], reference_axis),
        reverse=True,
    )

    source_to_rows = {
        "held_out_users": held_out_rows,
        "deepseek": annotators.get("deepseek", {}),
        "gpt_oss": annotators.get("gpt_oss", {}),
        "human_consensus": annotators.get("human_consensus", {}),
    }

    line_rows = []
    for song_index, song_key in enumerate(ordered_song_keys, start=1):
        for source in LINE_SOURCES:
            source_rows = source_to_rows.get(source, {})
            if song_key not in source_rows:
                continue
            line_rows.append(
                {
                    "song_index": song_index,
                    "song_key": song_key,
                    "song_name": song_key.split("/")[-1],
                    "source": source,
                    "source_label": SOURCE_LABELS[source],
                    "projection_score": _vector_cosine(source_rows[song_key], reference_axis),
                }
            )

    return pd.DataFrame(line_rows)


def _projection_chart(fold_metrics: dict, fold_number: int):
    df_line = _line_projection_frame(fold_metrics)
    if df_line.empty:
        return None

    fig = px.line(
        df_line,
        x="song_index",
        y="projection_score",
        color="source_label",
        hover_data={"song_name": True, "song_index": True, "projection_score": ":.3f", "source_label": False},
        color_discrete_map={SOURCE_LABELS[key]: SOURCE_COLORS[key] for key in SOURCE_COLORS},
        markers=True,
        title=f"Fold {fold_number}: 1D Projection Against Held-out User Axis",
    )
    fig.update_layout(
        xaxis_title="Song Order Within Held-out Fold",
        yaxis_title="1D Projection Score",
        legend_title="Vector Source",
        height=430,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig


def _fold_metric_rows(fold_metrics: dict) -> pd.DataFrame:
    rows = []
    for annotator in DISPLAY_ANNOTATORS + ["human_consensus", "ground_truth"]:
        metrics = fold_metrics["comparisons"]["human_test"][annotator]
        rows.append(
            {
                "Vector Source": SOURCE_LABELS.get(annotator, annotator),
                "MAE vs Held-out Users": metrics["mae"]["overall"],
                "RMSE vs Held-out Users": metrics["rmse"]["overall"],
                "Cosine vs Held-out Users": metrics["cosine_similarity"]["mean_per_song"],
                "Top Emotion Accuracy": metrics["top_emotion_accuracy"],
                "Krippendorff Alpha": metrics["krippendorff_alpha"],
            }
        )
    return pd.DataFrame(rows)


def _render_overview() -> None:
    st.subheader("Fold Workflow")
    st.caption(
        "Method: 5-fold cross-validation over all users. "
        "Users are stratified by gender and age range, shuffled with seed 42, then partitioned into 5 nearly even folds."
    )

    state = fold_orchestrator._load_state()
    status_rows = pd.DataFrame(fold_orchestrator.get_fold_status())
    st.dataframe(status_rows, hide_index=True, use_container_width=True)

    completed_folds = _completed_folds()
    pending_approval = None
    if state.get("prepared"):
        for fold_index in range(1, fold_orchestrator.total_folds() + 1):
            fold_state = state["folds"][str(fold_index)]
            if fold_state["status"] == "completed" and not fold_state["approved_to_proceed"]:
                pending_approval = fold_index
                break

    headline = st.columns(5)
    with headline[0]:
        st.metric("Mode", "MOCK" if USE_MOCK else "LIVE")
    with headline[1]:
        st.metric("Completed Folds", len(completed_folds))
    with headline[2]:
        st.metric("Total Folds", fold_orchestrator.total_folds() or "-")
    with headline[3]:
        st.metric("Users", state.get("user_count", 0))
    with headline[4]:
        next_fold = fold_orchestrator.get_next_runnable_fold()
        st.metric("Next Runnable Fold", next_fold or "-")

    controls = st.columns(2)
    with controls[0]:
        if st.button("Run Next Fold", use_container_width=True, key="fivek_run_next_fold"):
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
            if st.button(
                f"Approve Fold {pending_approval}",
                use_container_width=True,
                key="fivek_approve_fold",
            ):
                try:
                    fold_orchestrator.approve_fold(pending_approval)
                    st.success(f"Fold {pending_approval} approved.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        else:
            st.caption("No completed fold is waiting for approval.")

    if completed_folds:
        st.subheader("Saved Fold Outputs")
        for fold_number in completed_folds:
            summary = _fold_summary(fold_number)
            report = _load_json(REPORTS_DIR / f"fold_{fold_number}_report.json", default={})
            with st.container():
                st.markdown(f"### Fold {fold_number}: held-out user slice")
                meta = st.columns(6)
                with meta[0]:
                    st.metric("Test Users", len(summary.get("test_users", [])))
                with meta[1]:
                    st.metric("Songs", summary.get("song_count", 0))
                with meta[2]:
                    st.metric("Metric Songs", summary.get("metric_song_count", 0))
                with meta[3]:
                    st.metric("Train Song Profiles", summary.get("train_song_profile_count", 0))
                with meta[4]:
                    st.metric("Avg Few-shot Examples", summary.get("average_few_shot_examples_per_song", 0.0))
                with meta[5]:
                    _render_badge(
                        _report_status(report).upper(),
                        "#15803d" if _report_status(report) == "pass" else "#b91c1c",
                    )
                st.caption(
                    "Artifacts are stored under `llm analysis 5 k fold/`. "
                    f"Prompt method: {summary.get('prompt_method', 'unknown')}."
                )


def _render_fold_review() -> None:
    completed_folds = _completed_folds()
    if not completed_folds:
        st.info("No completed folds available yet.")
        return

    st.subheader("Per-Fold Review")
    st.caption(
        "Each fold holds out one user group as test and uses the remaining four groups as training."
    )

    for fold_number in completed_folds:
        summary = _fold_summary(fold_number)
        prompt_contexts = _prompt_contexts(fold_number)
        fold_metrics = load_or_compute_fold_metrics(fold_number)
        annotators = fold_metrics.get("annotators", {})
        if not annotators.get("human_test"):
            st.warning(f"Fold {fold_number} has no saved held-out user vectors yet.")
            continue

        st.markdown(f"### Fold {fold_number}: stratified held-out user fold")
        summary_cols = st.columns(5)
        with summary_cols[0]:
            st.metric("Test Users", len(summary.get("test_users", [])))
        with summary_cols[1]:
            st.metric("Songs", summary.get("song_count", 0))
        with summary_cols[2]:
            st.metric("Train Song Profiles", summary.get("train_song_profile_count", 0))
        with summary_cols[3]:
            st.metric("Prompt Examples / Song", summary.get("average_few_shot_examples_per_song", 0.0))
        with summary_cols[4]:
            st.metric("Metric Songs", summary.get("metric_song_count", 0))

        figure = _projection_chart(fold_metrics, fold_number)
        if figure is not None:
            st.plotly_chart(figure, use_container_width=True)

        st.dataframe(_fold_metric_rows(fold_metrics), hide_index=True, use_container_width=True)

        if prompt_contexts:
            example_counts = [
                context["few_shot_example_count"]
                for context in prompt_contexts.get("contexts", {}).values()
            ]
            if example_counts:
                st.caption(
                    f"Saved prompt contexts: {len(example_counts)} target songs, "
                    f"min {min(example_counts)}, max {max(example_counts)}, "
                    f"avg {sum(example_counts)/len(example_counts):.2f} few-shot examples."
                )
        st.divider()


def _render_agent_reports() -> None:
    report_paths = sorted(REPORTS_DIR.glob("*.json")) if REPORTS_DIR.exists() else []
    if not report_paths:
        st.info("No agent reports available yet.")
        return

    selected_path = st.selectbox(
        "Select Report",
        report_paths,
        format_func=lambda path: path.name,
        key="fivek_report_selector",
    )
    report = _load_json(selected_path, default={})
    status = _report_status(report)
    _render_badge(status.upper(), "#15803d" if status == "pass" else "#b91c1c")
    st.json(report)


def _comparison_table_frame(fold_number: int, song_index: int) -> pd.DataFrame:
    rows_by_emotion = {emotion: {"Emotion": emotion} for emotion in EMOTION_COLUMNS}
    for model_name in DISPLAY_ANNOTATORS:
        raw_path = _raw_output_dir(fold_number) / f"{model_name}_{song_index:03d}.json"
        raw_payload = _load_json(raw_path, default={})
        parsed_payload = raw_payload.get("parsed_payload", {})
        confidence = parsed_payload.get("confidence", {})
        for emotion in EMOTION_COLUMNS:
            rows_by_emotion[emotion][f"{SOURCE_LABELS[model_name]} Score"] = parsed_payload.get(emotion)
            rows_by_emotion[emotion][f"{SOURCE_LABELS[model_name]} Confidence"] = (
                confidence.get(emotion) if isinstance(confidence, dict) else "N/A"
            )

    if not rows_by_emotion:
        return pd.DataFrame()
    return pd.DataFrame([rows_by_emotion[emotion] for emotion in EMOTION_COLUMNS])


def _render_model_comparison() -> None:
    completed_folds = _completed_folds()
    if not completed_folds:
        st.info("No completed folds available yet.")
        return

    st.subheader("Model Comparison")
    st.caption(
        "Comparison table for one saved target song from the 5-fold workflow. "
        "Raw JSON is loaded from `llm analysis 5 k fold/data/raw_outputs`."
    )

    fold_number = st.selectbox("Select Fold", completed_folds, key="fivek_comparison_fold")
    summary = _fold_summary(fold_number)
    songs = summary.get("songs_annotated", [])
    if not songs:
        st.info("No saved songs for this fold yet.")
        return

    selected_song = st.selectbox("Select Song", songs, key="fivek_comparison_song")
    song_index = songs.index(selected_song) + 1

    table = _comparison_table_frame(fold_number, song_index)
    if table.empty:
        st.warning("No raw model outputs found for this fold/song.")
        return

    st.dataframe(table, hide_index=True, use_container_width=True)


def render(page_name: str, title: str = "Analysis 5 k fold", caption: str | None = None):
    st.header(title)
    st.caption(
        caption
        or (
            "Separate artifact root: `llm analysis 5 k fold/`. "
            "This workflow runs true 5-fold cross-validation over the full user set."
        )
    )

    overview_tab, review_tab, comparison_tab, reports_tab = st.tabs(
        ["Overview", "Fold Review", "Model Comparison", "Agent Reports"]
    )
    with overview_tab:
        _render_overview()
    with review_tab:
        _render_fold_review()
    with comparison_tab:
        _render_model_comparison()
    with reports_tab:
        _render_agent_reports()
