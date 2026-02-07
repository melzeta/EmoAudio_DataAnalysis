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
        ],
    )

    # Routing
    if menu == "Panoramica Dataset":
        render_overview(df_users, df_responses, CHART_LAYOUT)

    elif menu == "🕷️ Spider Charts":
        render_spider(
            df_responses,
            EMOTIONS_LIST,
            EMOTION_COLORS,
            original_emotions,
        )


if __name__ == "__main__":
    main()