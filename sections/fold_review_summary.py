import pandas as pd
import streamlit as st

from components.fold_metrics import enrich_metrics
from evaluation.constants import N_FOLDS
from evaluation.orchestrator import load_review_bundle


def _render_metrics_explanation() -> None:
    st.markdown("**Metrics Explanation**")
    st.caption("MAE -> Mean Absolute Error. Measures average prediction error. Lower is better.")
    st.caption("RMSE -> Root Mean Squared Error. Penalizes larger errors more heavily.")
    st.caption("Pearson -> Measures linear correlation between predictions and ground truth.")
    st.caption("Spearman -> Measures rank correlation between predictions and ground truth.")
    st.caption("Top Emotion Accuracy -> Checks whether the highest-scoring predicted emotion matches the highest-scoring true emotion.")
    st.caption("Cosine Similarity -> Measures how similar the full predicted emotion vector is to the true vector. Higher is better.")
    st.caption("Per-Emotion Accuracy -> Reported as accuracy@0.5, where each emotion score is converted to above/below 0.5 before comparison.")


def _render_fold_metrics(metrics_summary: dict, prediction_rows: list[dict]) -> None:
    enriched_metrics = enrich_metrics(metrics_summary, prediction_rows)
    overall = enriched_metrics.get("overall", {})

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("MAE", f"{overall.get('mae', 0):.4f}")
    col2.metric("RMSE", f"{overall.get('rmse', 0):.4f}")
    col3.metric("Pearson", f"{(overall.get('pearson') or 0):.4f}")
    col4.metric("Spearman", f"{(overall.get('spearman') or 0):.4f}")
    col5.metric("Top Emotion Accuracy", f"{overall.get('top_emotion_accuracy', 0):.4f}")
    col6.metric("Cosine Similarity", f"{overall.get('mean_vector_cosine_similarity', 0):.4f}")

    _render_metrics_explanation()

    with st.expander("Per Emotion Metrics"):
        per_emotion = enriched_metrics.get("per_emotion", {})
        if per_emotion:
            st.dataframe(pd.DataFrame(per_emotion).T, use_container_width=True)
        else:
            st.info("No per-emotion metrics available.")


def _render_prediction_archive(bundle: dict) -> None:
    with st.expander("Current Asset Prediction vs Ground Truth"):
        if bundle.get("predictions"):
            st.markdown("**Predictions**")
            st.dataframe(pd.DataFrame(bundle["predictions"]), use_container_width=True)
        else:
            st.info("No predictions saved for this fold.")

        if bundle.get("test_rows"):
            st.markdown("**Test Items**")
            st.dataframe(pd.DataFrame(bundle["test_rows"]), use_container_width=True)
        else:
            st.info("No test items saved for this fold.")

        model_summary = bundle.get("model_summary", {})
        if model_summary:
            st.markdown("**Model Summary**")
            st.json(model_summary, expanded=False)


def render() -> None:
    st.header("Fold Review Summary")
    st.caption("Read-only archival view of all fold results. Use Manual Fold Review for review controls.")

    state = load_review_bundle().get("state", {})
    if not state.get("prepared"):
        st.info("No fold state found yet. Run `python3 -m evaluation.cli prepare` first.")
        return

    for fold_index in range(1, N_FOLDS + 1):
        bundle = load_review_bundle(fold_index)
        fold_state = state.get("folds", {}).get(str(fold_index), {})
        executed = bool(bundle.get("metrics_summary")) and bool(bundle.get("predictions"))

        with st.expander(f"Fold {fold_index}", expanded=(fold_index == state.get("current_review_fold", 1))):
            with st.expander("Fold Split Summary"):
                st.write(
                    {
                        "train_count": fold_state.get("train_count"),
                        "test_count": fold_state.get("test_count"),
                        "reviewed": fold_state.get("reviewed"),
                        "approved_to_proceed": fold_state.get("approved_to_proceed"),
                    }
                )

            if not executed:
                st.info("Fold not yet executed")
                continue

            st.subheader("Fold Metrics")
            _render_fold_metrics(bundle.get("metrics_summary", {}), bundle.get("predictions", []))
            _render_prediction_archive(bundle)
