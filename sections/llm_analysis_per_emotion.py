import pandas as pd
import plotly.express as px
import streamlit as st

from evaluation import fold_orchestrator_5k


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
MODEL_COLUMNS = ["deepseek", "gpt_oss"]


def _completed_folds() -> list[int]:
    return [row["fold"] for row in fold_orchestrator_5k.get_fold_status() if row["status"] == "completed"]


def _annotation_dir(fold_number: int):
    return fold_orchestrator_5k.ANNOTATIONS_DIR / f"fold_{fold_number}"


def _load_fold_prediction_rows(fold_number: int) -> pd.DataFrame:
    fold_dir = _annotation_dir(fold_number)
    consensus_path = fold_dir / "human_consensus.csv"
    if not consensus_path.exists():
        return pd.DataFrame()

    frames = []
    df_consensus = pd.read_csv(consensus_path)[["filename", *EMOTION_COLUMNS]]
    for model_name in MODEL_COLUMNS:
        model_path = fold_dir / f"{model_name}.csv"
        if not model_path.exists():
            continue

        df_model = pd.read_csv(model_path)
        merged = df_model[["filename", *EMOTION_COLUMNS]].merge(
            df_consensus,
            on="filename",
            how="inner",
            suffixes=("_model", "_consensus"),
        )
        if merged.empty:
            continue

        for emotion in EMOTION_COLUMNS:
            merged[f"{emotion}_error"] = merged[f"{emotion}_model"] - merged[f"{emotion}_consensus"]
            merged[f"{emotion}_squared_distance"] = (
                merged[f"{emotion}_error"]
            ) ** 2

        merged["fold"] = fold_number
        merged["model"] = model_name
        frames.append(merged)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_all_prediction_rows() -> pd.DataFrame:
    completed_folds = _completed_folds()
    frames = [_load_fold_prediction_rows(fold_number) for fold_number in completed_folds]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _distance_frames(df_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_predictions.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary_rows = []
    per_fold_rows = []
    for model_name in sorted(df_predictions["model"].unique()):
        model_rows = df_predictions[df_predictions["model"] == model_name]
        for emotion in EMOTION_COLUMNS:
            summary_rows.append(
                {
                    "model": model_name,
                    "emotion": emotion,
                    "mean_squared_distance": model_rows[f"{emotion}_squared_distance"].mean(),
                    "matched_rows": len(model_rows),
                }
            )
        for fold_number in sorted(model_rows["fold"].unique()):
            fold_rows = model_rows[model_rows["fold"] == fold_number]
            for emotion in EMOTION_COLUMNS:
                per_fold_rows.append(
                    {
                        "fold": fold_number,
                        "model": model_name,
                        "emotion": emotion,
                        "mean_squared_distance": fold_rows[f"{emotion}_squared_distance"].mean(),
                        "matched_rows": len(fold_rows),
                    }
                )

    df_summary = pd.DataFrame(summary_rows).sort_values(
        ["model", "mean_squared_distance"],
        ascending=[True, False],
    )
    df_per_fold = pd.DataFrame(per_fold_rows).sort_values(
        ["fold", "model", "emotion"],
        ascending=[True, True, True],
    )
    return df_summary, df_per_fold


def _error_frames(df_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_predictions.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary_rows = []
    per_fold_rows = []
    for model_name in sorted(df_predictions["model"].unique()):
        model_rows = df_predictions[df_predictions["model"] == model_name]
        for emotion in EMOTION_COLUMNS:
            summary_rows.append(
                {
                    "model": model_name,
                    "emotion": emotion,
                    "mean_error": model_rows[f"{emotion}_error"].mean(),
                    "matched_rows": len(model_rows),
                }
            )
        for fold_number in sorted(model_rows["fold"].unique()):
            fold_rows = model_rows[model_rows["fold"] == fold_number]
            for emotion in EMOTION_COLUMNS:
                per_fold_rows.append(
                    {
                        "fold": fold_number,
                        "model": model_name,
                        "emotion": emotion,
                        "mean_error": fold_rows[f"{emotion}_error"].mean(),
                        "matched_rows": len(fold_rows),
                    }
                )

    df_summary = pd.DataFrame(summary_rows).sort_values(
        ["model", "emotion"],
        ascending=[True, True],
    )
    df_per_fold = pd.DataFrame(per_fold_rows).sort_values(
        ["fold", "model", "emotion"],
        ascending=[True, True, True],
    )
    return df_summary, df_per_fold


def _hardest_emotions_frame(df_summary: pd.DataFrame) -> pd.DataFrame:
    if df_summary.empty:
        return pd.DataFrame()

    hardest = (
        df_summary.groupby("emotion", as_index=False)["mean_squared_distance"]
        .mean()
        .rename(columns={"mean_squared_distance": "average_model_error"})
        .sort_values("average_model_error", ascending=False)
    )
    return hardest


def _emotion_data_volume_frame(df_predictions: pd.DataFrame) -> pd.DataFrame:
    if df_predictions.empty:
        return pd.DataFrame()

    frame = df_predictions.copy()
    frame["intended_emotion"] = frame["filename"].str.split("/").str[0]
    return (
        frame.groupby("intended_emotion", as_index=False)
        .size()
        .rename(columns={"intended_emotion": "emotion", "size": "matched_rows"})
        .sort_values("matched_rows", ascending=False)
    )


def _user_bias_frame(df_predictions: pd.DataFrame) -> pd.DataFrame:
    if df_predictions.empty:
        return pd.DataFrame()

    consensus_columns = [f"{emotion}_consensus" for emotion in EMOTION_COLUMNS]
    unique_rows = df_predictions[["filename", *consensus_columns]].drop_duplicates(subset=["filename"])
    bias_rows = [
        {
            "emotion": emotion,
            "average_consensus_score": unique_rows[f"{emotion}_consensus"].mean(),
        }
        for emotion in EMOTION_COLUMNS
    ]
    return pd.DataFrame(bias_rows).sort_values("average_consensus_score", ascending=False)


def _render_context_charts(df_summary: pd.DataFrame, df_predictions: pd.DataFrame) -> None:
    hardest = _hardest_emotions_frame(df_summary)
    volume = _emotion_data_volume_frame(df_predictions)
    bias = _user_bias_frame(df_predictions)

    st.markdown("**Emotions consistently hardest across models.**")
    st.caption(
        "These charts separate three different explanations: model difficulty, data volume, and user-consensus bias."
    )

    chart_columns = st.columns(3)

    with chart_columns[0]:
        if not hardest.empty:
            hardest_chart = px.pie(
                hardest,
                names="emotion",
                values="average_model_error",
                title="Hardest Across Models",
                hole=0.45,
            )
            hardest_chart.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=10), legend_title="Emotion")
            st.plotly_chart(hardest_chart, use_container_width=True)
            st.caption(
                "Share of average model error by emotion, using mean squared distance averaged across models."
            )

    with chart_columns[1]:
        if not volume.empty:
            volume_chart = px.pie(
                volume,
                names="emotion",
                values="matched_rows",
                title="Matched Data Volume",
                hole=0.45,
            )
            volume_chart.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=10), legend_title="Emotion")
            st.plotly_chart(volume_chart, use_container_width=True)
            st.caption(
                "How many matched prediction rows come from each intended-emotion class in the completed 5k folds."
            )

    with chart_columns[2]:
        if not bias.empty:
            bias_chart = px.pie(
                bias,
                names="emotion",
                values="average_consensus_score",
                title="User Consensus Bias",
                hole=0.45,
            )
            bias_chart.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=10), legend_title="Emotion")
            st.plotly_chart(bias_chart, use_container_width=True)
            st.caption(
                "Relative share of average human-consensus intensity by emotion across unique songs."
            )

    with st.expander("Show context tables", expanded=False):
        context_columns = st.columns(3)
        with context_columns[0]:
            st.dataframe(
                hardest,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "emotion": "Emotion",
                    "average_model_error": st.column_config.NumberColumn(
                        "Avg Model Error",
                        format="%.6f",
                    ),
                },
            )
        with context_columns[1]:
            st.dataframe(
                volume,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "emotion": "Emotion",
                    "matched_rows": "Matched Rows",
                },
            )
        with context_columns[2]:
            st.dataframe(
                bias,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "emotion": "Emotion",
                    "average_consensus_score": st.column_config.NumberColumn(
                        "Avg Consensus Score",
                        format="%.6f",
                    ),
                },
            )


def _render_model_predictions(df_predictions: pd.DataFrame) -> None:
    st.divider()
    st.subheader("Model Predictions")
    st.caption(
        "Each row compares one saved model prediction against the human consensus for the same song in the 5k-fold workflow."
    )

    available_models = sorted(df_predictions["model"].unique())
    selected_model = st.selectbox("Select Model", available_models, key="per_emotion_model_selector")
    model_rows = df_predictions[df_predictions["model"] == selected_model].copy()

    songs = sorted(model_rows["filename"].unique())
    selected_song = st.selectbox("Select Song", songs, key="per_emotion_song_selector")
    song_rows = model_rows[model_rows["filename"] == selected_song]
    selected_row = song_rows.iloc[0]

    detail_rows = []
    for emotion in EMOTION_COLUMNS:
        detail_rows.append(
            {
                "Emotion": emotion,
                "Model Score": selected_row[f"{emotion}_model"],
                "Consensus Score": selected_row[f"{emotion}_consensus"],
                "Error": selected_row[f"{emotion}_error"],
                "Squared Distance": selected_row[f"{emotion}_squared_distance"],
            }
        )
    df_detail = pd.DataFrame(detail_rows)

    chart = px.bar(
        df_detail,
        x="Emotion",
        y=["Model Score", "Consensus Score"],
        barmode="group",
        title=f"{selected_model} vs Human Consensus: {selected_song}",
    )
    chart.update_layout(height=420, yaxis_title="Score", xaxis_title="Emotion")
    st.plotly_chart(chart, use_container_width=True)
    st.markdown(
        """
            **How to read this graph**

            - **X-axis:** the eight emotion categories for the selected song.
            - **Y-axis:** emotion scores on a `0–1` scale. Higher values indicate stronger emotional intensity.
            - **Good vs bad result:** small gaps between model and consensus bars indicate close agreement; large gaps indicate stronger disagreement, especially for dominant emotions.
            - **Data source:** model scores come from annotation CSVs such as `deepseek.csv` and `gpt_oss.csv`, compared against `human_consensus.csv` from the same fold.
            - **Computation:** the chart directly plots `model_score` and `consensus_score` side by side after matching rows by `filename`.
        """
    )

    error_chart = px.bar(
        df_detail,
        x="Emotion",
        y="Error",
        color="Error",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title=f"{selected_model} Signed Error vs Human Consensus: {selected_song}",
    )
    error_chart.add_hline(y=0, line_dash="dash", line_color="black")
    error_chart.update_layout(height=360, yaxis_title="Error", xaxis_title="Emotion")
    st.plotly_chart(error_chart, use_container_width=True)
    st.markdown(
        """
            **How to read this graph**

            - **X-axis:** the eight emotion categories for the selected song and model.
            - **Y-axis:** signed prediction error computed as `model_score - consensus_score`. A value of `0` means perfect agreement with human consensus.
            - **Good vs bad result:** bars close to `0` indicate small disagreement; large positive values indicate overestimation, while large negative values indicate underestimation.
            - **Data source:** scores are loaded from the saved 5k-fold prediction CSVs and matched with `human_consensus.csv` using `filename`.
            - **Computation:** for each emotion, the graph plots `error = model_score - consensus_score`, preserving whether the model is too high or too low.
        """
    )

    with st.expander("Show per-song emotion detail table", expanded=False):
        st.dataframe(
            df_detail,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Model Score": st.column_config.NumberColumn("Model Score", format="%.4f"),
                "Consensus Score": st.column_config.NumberColumn("Consensus Score", format="%.4f"),
                "Error": st.column_config.NumberColumn("Error", format="%.6f"),
                "Squared Distance": st.column_config.NumberColumn("Squared Distance", format="%.6f"),
            },
        )

    st.subheader("Saved Prediction Rows")
    summary_columns = ["fold", "model", "filename", *[f"{emotion}_model" for emotion in EMOTION_COLUMNS]]
    renamed_columns = {
        "fold": "Fold",
        "model": "Model",
        "filename": "Filename",
        **{f"{emotion}_model": emotion for emotion in EMOTION_COLUMNS},
    }
    with st.expander("Show saved prediction rows table", expanded=False):
        st.dataframe(
            model_rows[summary_columns].rename(columns=renamed_columns),
            hide_index=True,
            use_container_width=True,
        )


def render() -> None:
    st.header("Per-emotion Analysis")
    st.caption(
        "Uses the saved `Analysis 5 k fold` artifacts. For each model and each emotion, this page computes "
        "mean signed error as `model_score - human_consensus_score` across all matched rows."
    )

    df_predictions = _load_all_prediction_rows()
    df_summary, df_per_fold = _distance_frames(df_predictions)
    df_error_summary, df_error_per_fold = _error_frames(df_predictions)
    if df_summary.empty:
        st.info("No completed 5k-fold outputs with matching model CSVs and `human_consensus.csv` were found.")
        return

    st.subheader("Mean Squared Distance by Emotion")
    chart = px.bar(
        df_summary,
        x="emotion",
        y="mean_squared_distance",
        color="model",
        barmode="group",
        title="Per-Emotion Mean Squared Distance by Model",
    )
    chart.update_layout(
        xaxis_title="Emotion",
        yaxis_title="Mean Squared Distance",
        height=420,
    )
    st.plotly_chart(chart, use_container_width=True)

    chart_fixed = px.bar(
        df_summary,
        x="emotion",
        y="mean_squared_distance",
        color="model",
        barmode="group",
        title="Per-Emotion Mean Squared Distance by Model (Y-axis 0 to 1)",
    )
    chart_fixed.update_layout(
        xaxis_title="Emotion",
        yaxis_title="Mean Squared Distance",
        height=420,
    )
    chart_fixed.update_yaxes(range=[0, 1])
    st.plotly_chart(chart_fixed, use_container_width=True)
    st.markdown(
        """
            **How to read this graph**

            - **X-axis:** the eight emotion categories, grouped by model.
            - **Y-axis:** mean squared distance between model predictions and human consensus.
            - **Good vs bad result:** lower values indicate closer agreement; higher values indicate larger disagreement.
            - **Data source:** model scores from completed 5k-fold prediction CSVs are matched with `human_consensus.csv` using `filename`.
            - **Computation:** for each emotion, the graph computes `(model_score - consensus_score)^2`, then averages the result across all matched rows.
        """
    )

    _render_context_charts(df_summary, df_predictions)

    st.divider()

    st.subheader("Mean Signed Error by Emotion")
    st.caption(
        "Values above 0 indicate overestimation versus consensus. "
        "Values below 0 indicate underestimation. The zero line represents perfect consensus."
    )
    error_chart = px.bar(
        df_error_summary,
        x="emotion",
        y="mean_error",
        color="model",
        barmode="group",
        title="Per-Emotion Mean Signed Error by Model",
    )
    error_chart.add_hline(y=0, line_dash="dash", line_color="black")
    error_chart.update_layout(
        xaxis_title="Emotion",
        yaxis_title="Mean Error",
        height=420,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    st.plotly_chart(error_chart, use_container_width=True)
    st.markdown(
        """
            **How to read this graph**

            - **X-axis:** the eight emotion categories used in the dataset and evaluation workflow.
            - **Y-axis:** mean signed error on the original `0–1` emotion scale.
            - **Good vs bad result:** bars close to `0` indicate little systematic bias; large positive values indicate overestimation, while large negative values indicate underestimation.
            - **Data source:** scores come from completed 5k-fold model prediction CSVs matched with `human_consensus.csv` using `filename`.
            - **Computation:** for each matched row, the graph computes `error = model_score - consensus_score`, then averages these values across rows to produce `mean_error` for each emotion and model.
        """
    )

    with st.expander("Show mean signed error table", expanded=False):
        st.dataframe(
            df_error_summary,
            hide_index=True,
            use_container_width=True,
            column_config={
                "model": "Model",
                "emotion": "Emotion",
                "mean_error": st.column_config.NumberColumn(
                    "Mean Error",
                    format="%.6f",
                ),
                "matched_rows": "Matched Rows",
            },
        )

    with st.expander("Show mean squared distance table", expanded=False):
        st.dataframe(
            df_summary,
            hide_index=True,
            use_container_width=True,
            column_config={
                "model": "Model",
                "emotion": "Emotion",
                "mean_squared_distance": st.column_config.NumberColumn(
                    "Mean Squared Distance",
                    format="%.6f",
                ),
                "matched_rows": "Matched Rows",
            },
        )

    if not df_per_fold.empty:
        st.divider()
        st.subheader("Per-Fold Breakdown")
        with st.expander("Show per-fold mean squared distance table", expanded=False):
            st.dataframe(
                df_per_fold,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "fold": "Fold",
                    "model": "Model",
                    "emotion": "Emotion",
                    "mean_squared_distance": st.column_config.NumberColumn(
                        "Mean Squared Distance",
                        format="%.6f",
                    ),
                    "matched_rows": "Matched Rows",
                },
            )

    if not df_error_per_fold.empty:
        st.subheader("Per-Fold Signed Error Breakdown")
        with st.expander("Show per-fold mean signed error table", expanded=False):
            st.dataframe(
                df_error_per_fold,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "fold": "Fold",
                    "model": "Model",
                    "emotion": "Emotion",
                    "mean_error": st.column_config.NumberColumn(
                        "Mean Error",
                        format="%.6f",
                    ),
                    "matched_rows": "Matched Rows",
                },
            )

    if not df_predictions.empty:
        _render_model_predictions(df_predictions)
