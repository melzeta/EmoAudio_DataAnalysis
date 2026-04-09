import streamlit as st

from config.settings import (
    set_app_config,
    EMOTIONS_LIST,
    EMOTION_COLORS,
    CHART_LAYOUT,
)

from data.loaders import (
    load_and_process_data,
    load_original_emotions,
)

from sections.overview import render as render_overview
from sections.spider_charts import render as render_spider
from sections.fold_review import render as render_fold_review
from sections.fold_review_summary import render as render_fold_review_summary


def main():
    set_app_config()

    df_users, df_responses = load_and_process_data()
    original_emotions = load_original_emotions()

    st.sidebar.title("🎵 Analisi Emozioni")
    menu = st.sidebar.radio(
        "Sezioni:",
        [
            "Panoramica Dataset",
            "🕷️ Spider Charts",
            "Similarity Analysis",
            "Manual Fold Review",
            "Fold Review Summary",
        ],
    )

    # Routing
    if menu == "Panoramica Dataset":
        render_overview(df_users, df_responses, CHART_LAYOUT)
    elif menu == "🕷️ Spider Charts":
        render_spider(df_responses, EMOTIONS_LIST, EMOTION_COLORS, original_emotions)
    elif menu == "Similarity Analysis":
        import sections.similarity_analysis as similarity_analysis
        similarity_analysis.show(df_responses)
    elif menu == "Manual Fold Review":
        render_fold_review()
    elif menu == "Fold Review Summary":
        render_fold_review_summary()


if __name__ == "__main__":
    main()
