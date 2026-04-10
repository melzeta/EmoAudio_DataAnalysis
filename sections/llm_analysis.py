import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from annotation.llm_clients import USE_MOCK
from evaluation import fold_orchestrator
from evaluation.export_results import (
    export_fold_metrics_csv,
    export_per_emotion_csv,
    export_summary_json,
)
from evaluation.metrics_llm import ANNOTATORS, EMOTION_COLUMNS, compute_all_folds_metrics, compute_fold_metrics


ROOT_DIR = Path(__file__).resolve().parent.parent
USER_FOLDS_PATH = ROOT_DIR / "state" / "user_folds.json"
REPORTS_DIR = ROOT_DIR / "state" / "agent_reports"
EXPORTS_DIR = ROOT_DIR / "data" / "exports"
DISPLAY_ANNOTATORS = ["deepseek", "gemini", "mistral", "human_test", "human_consensus"]


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


def _demographics_frame() -> pd.DataFrame:
    folds = _load_json(USER_FOLDS_PATH, default={})
    users = folds.get("users", [])
    return pd.DataFrame(users)


def _demographic_chart(df: pd.DataFrame, column: str, title: str):
    if df.empty or column not in df.columns:
        st.info(f"No data for {column}.")
        return
    counts = df[column].fillna("N/A").astype(str).value_counts().reset_index()
    counts.columns = [column, "count"]
    fig = px.bar(counts, x=column, y="count", title=title)
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def _render_overview() -> None:
    st.subheader("Fold Status")
    st.dataframe(pd.DataFrame(fold_orchestrator.get_fold_status()), hide_index=True, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Active Models**")
        st.write(", ".join(["deepseek", "gemini", "mistral"]))
    with col2:
        st.markdown("**Execution Mode**")
        _render_badge("MOCK" if USE_MOCK else "LIVE", "#d97706" if USE_MOCK else "#15803d")

    df_users = _demographics_frame()
    st.subheader("Demographics Summary")
    demographics_cols = st.columns(3)
    with demographics_cols[0]:
        _demographic_chart(df_users, "gender", "Gender Distribution")
    with demographics_cols[1]:
        _demographic_chart(df_users, "age_range", "Age Range Distribution")
    with demographics_cols[2]:
        _demographic_chart(df_users, "nationality", "Nationality Distribution")

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


def _mean_vector(rows: dict) -> dict:
    if not rows:
        return {emotion: 0.0 for emotion in EMOTION_COLUMNS}
    return {
        emotion: sum(row[emotion] for row in rows.values()) / len(rows)
        for emotion in EMOTION_COLUMNS
    }


def _render_fold_comparison() -> None:
    completed_folds = _completed_folds()
    if not completed_folds:
        st.info("No completed folds available yet.")
        return

    fold_number = st.selectbox("Select Fold", completed_folds)
    fold_metrics = compute_fold_metrics(fold_number)
    annotators = fold_metrics["annotators"]
    if not annotators.get("human_test"):
        st.warning("Fold data is incomplete for comparison.")
        return

    fig = go.Figure()
    for annotator in DISPLAY_ANNOTATORS:
        vector = _mean_vector(annotators.get(annotator, {}))
        fig.add_trace(
            go.Scatterpolar(
                r=[vector[emotion] for emotion in EMOTION_COLUMNS],
                theta=EMOTION_COLUMNS,
                fill="toself",
                name=annotator,
                opacity=0.45,
            )
        )
    fig.update_layout(height=520, margin=dict(l=30, r=30, t=40, b=30), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    metric_rows = []
    for annotator in DISPLAY_ANNOTATORS:
        metrics = fold_metrics["comparisons"]["human_test"][annotator]
        metric_rows.append(
            {
                "annotator": annotator,
                "mae_vs_human_test": metrics["mae"]["overall"],
                "rmse_vs_human_test": metrics["rmse"]["overall"],
                "pearson_vs_human_test": metrics["pearson"]["overall"],
                "spearman_vs_human_test": metrics["spearman"]["overall"],
                "cosine_vs_human_test": metrics["cosine_similarity"]["mean_per_song"],
                "top_emotion_accuracy": metrics["top_emotion_accuracy"],
                "krippendorff_alpha": metrics["krippendorff_alpha"],
            }
        )
    st.subheader("Metrics vs Human Test Average")
    st.dataframe(pd.DataFrame(metric_rows), hide_index=True, use_container_width=True)


def _render_cross_model_analysis() -> None:
    completed_folds = _completed_folds()
    if not completed_folds:
        st.info("No completed folds available yet.")
        return

    results = compute_all_folds_metrics()
    aggregate_rows = []
    per_emotion_rows = []
    pairwise_rows = []

    for annotator in DISPLAY_ANNOTATORS + ["ground_truth"]:
        metrics = results["aggregate"].get("human_test", {}).get(annotator, {})
        aggregate_rows.append({"annotator": annotator, **metrics})

    for fold_result in results["folds"]:
        fold_number = fold_result["fold"]
        for annotator in ["deepseek", "gemini", "mistral"]:
            metrics = fold_result["comparisons"]["human_test"][annotator]
            for emotion in EMOTION_COLUMNS:
                per_emotion_rows.append(
                    {
                        "fold": fold_number,
                        "annotator": annotator,
                        "emotion": emotion,
                        "mae": metrics["mae"]["per_emotion"][emotion],
                    }
                )

        for left in DISPLAY_ANNOTATORS:
            for right in DISPLAY_ANNOTATORS:
                pairwise_rows.append(
                    {
                        "left": left,
                        "right": right,
                        "cosine": fold_result["comparisons"][left][right]["cosine_similarity"]["mean_per_song"],
                    }
                )

    st.subheader("Aggregate Metrics vs Human Test")
    st.dataframe(pd.DataFrame(aggregate_rows), hide_index=True, use_container_width=True)

    bar_fig = px.bar(
        pd.DataFrame(per_emotion_rows),
        x="emotion",
        y="mae",
        color="annotator",
        barmode="group",
        title="MAE per Emotion per Annotator",
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    heatmap_df = pd.DataFrame(pairwise_rows).groupby(["left", "right"], as_index=False)["cosine"].mean()
    heatmap_pivot = heatmap_df.pivot(index="left", columns="right", values="cosine")
    heatmap_fig = px.imshow(
        heatmap_pivot,
        text_auto=".3f",
        color_continuous_scale="Blues",
        title="Pairwise Cosine Similarity Heatmap",
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)

    st.subheader("Export Results")
    if st.button("Export Metrics", use_container_width=True):
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fold_csv = export_fold_metrics_csv(EXPORTS_DIR / f"fold_metrics_{timestamp}.csv")
        emotion_csv = export_per_emotion_csv(EXPORTS_DIR / f"per_emotion_{timestamp}.csv")
        summary_json = export_summary_json(EXPORTS_DIR / f"summary_{timestamp}.json")
        st.session_state["llm_exports"] = [fold_csv, emotion_csv, summary_json]
        st.success("Export complete.")

    export_paths = st.session_state.get("llm_exports", [])
    for path in export_paths:
        with path.open("rb") as handle:
            st.download_button(
                label=f"Download {path.name}",
                data=handle.read(),
                file_name=path.name,
                mime="application/octet-stream",
                key=f"download_{path.name}",
            )


def _report_status(report: dict) -> str:
    return report.get("overall") or report.get("status") or "unknown"


def _render_agent_reports() -> None:
    report_paths = sorted(REPORTS_DIR.glob("*.json")) if REPORTS_DIR.exists() else []
    if not report_paths:
        st.info("No agent reports available yet.")
        return

    selected_path = st.selectbox("Select Report", report_paths, format_func=lambda path: path.name)
    report = _load_json(selected_path, default={})
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
