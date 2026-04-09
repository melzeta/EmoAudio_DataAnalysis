import pandas as pd
import streamlit as st

from components.fold_metrics import enrich_metrics
from evaluation.orchestrator import load_review_bundle, review_fold


def render():
    st.header("Manual 5-Fold Review")

    bundle = load_review_bundle()
    state = bundle.get("state", {})

    if not state.get("prepared"):
        st.info("No manual fold state found yet. Run `python3 -m evaluation.cli prepare` first.")
        return

    current_review_fold = state.get("current_review_fold", 1) or 1
    fold_options = [int(key) for key in state.get("folds", {}).keys()]
    selected_fold = st.selectbox(
        "Fold to inspect",
        options=fold_options,
        index=max(0, fold_options.index(current_review_fold) if current_review_fold in fold_options else 0),
    )

    bundle = load_review_bundle(selected_fold)
    fold_state = state["folds"][str(selected_fold)]

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Review Fold", current_review_fold)
    col2.metric("Selected Fold", selected_fold)
    col3.metric("Fold Status", fold_state["status"])

    st.subheader("Fold Split Summary")
    st.write(
        {
            "train_count": fold_state["train_count"],
            "test_count": fold_state["test_count"],
            "reviewed": fold_state["reviewed"],
            "approved_to_proceed": fold_state["approved_to_proceed"],
        }
    )

    if bundle["test_rows"]:
        st.subheader("Current Test Set")
        st.dataframe(pd.DataFrame(bundle["test_rows"]), use_container_width=True)

    if bundle["predictions"]:
        st.subheader("Predictions vs Ground Truth")
        st.dataframe(pd.DataFrame(bundle["predictions"]), use_container_width=True)

    metrics_summary = bundle.get("metrics_summary", {})
    if metrics_summary:
        st.subheader("Fold Metrics")
        enriched_metrics = enrich_metrics(metrics_summary, bundle.get("predictions", []))
        overall = enriched_metrics.get("overall", {})
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Overall MAE", f"{overall.get('mae', 0):.4f}")
        col2.metric("Overall RMSE", f"{overall.get('rmse', 0):.4f}")
        col3.metric("Overall Pearson", f"{(overall.get('pearson') or 0):.4f}")
        col4.metric("Overall Spearman", f"{(overall.get('spearman') or 0):.4f}")
        col5.metric("Top Emotion Accuracy", f"{overall.get('top_emotion_accuracy', 0):.4f}")
        col6.metric("Cosine Similarity", f"{overall.get('mean_vector_cosine_similarity', 0):.4f}")

        st.markdown("**Per-emotion metrics**")
        st.caption("Per-Emotion Accuracy is reported as accuracy@0.5, where predicted and true scores are both thresholded at 0.5.")
        per_emotion = enriched_metrics.get("per_emotion", {})
        st.dataframe(pd.DataFrame(per_emotion).T, use_container_width=True)

    if fold_state["status"] != "completed":
        st.warning("This fold has not been executed yet. Run it from the CLI before review.")
        return

    button_col1, button_col2 = st.columns(2)
    with button_col1:
        if st.button("Mark Fold Reviewed", key=f"review_{selected_fold}"):
            review_fold(selected_fold, approve_next=False)
            st.rerun()

    with button_col2:
        next_label = "Approve Next Fold" if selected_fold < max(fold_options) else "Approve Final Fold Completion"
        if st.button(next_label, key=f"approve_{selected_fold}"):
            review_fold(selected_fold, approve_next=True)
            st.rerun()

    if fold_state["reviewed"]:
        st.success(f"Fold {selected_fold} has been marked as reviewed.")
    if fold_state["approved_to_proceed"] and selected_fold < max(fold_options):
        st.success(f"Fold {selected_fold} has been approved. Fold {selected_fold + 1} can now be run.")
