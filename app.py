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

from pages.overview import render as render_overview
from pages.spider_charts import render as render_spider


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
            "Similarity Analysis"
        ],
    )

    # Routing
    if menu == "Panoramica Dataset":
        overview.show()
    elif menu == "Spider Charts":
        spider_charts.show()
    elif menu == "Similarity Analysis":
        import pages.similarity_analysis as similarity_analysis
        similarity_analysis.show()


if __name__ == "__main__":
    main()