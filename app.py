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
from sections.llm_analysis import render as render_llm_analysis


def _activate_main_nav():
    st.session_state["active_nav"] = "main"


def _activate_llm_nav():
    st.session_state["active_nav"] = "llm"


def main():
    set_app_config()

    if "active_nav" not in st.session_state:
        st.session_state["active_nav"] = "main"

    df_users, df_responses = load_and_process_data()
    original_emotions = load_original_emotions()

    st.sidebar.title("🎵 Analisi Emozioni")
    menu = st.sidebar.radio(
        "Sezioni:",
        [
            "Panoramica Dataset",
            "🕷️ Spider Charts",
            "Similarity Analysis",
        ],
        key="main_menu",
        on_change=_activate_main_nav,
    )
    with st.sidebar.expander("🤖 LLM Analysis", expanded=False):
        llm_page = st.radio(
            "LLM Analysis Pages",
            [
                "Overview",
                "Fold Comparison",
                "Cross-Model Analysis",
                "Agent Reports",
            ],
            key="llm_analysis_page",
            label_visibility="collapsed",
            on_change=_activate_llm_nav,
        )

    # Routing
    if st.session_state.get("active_nav") == "llm":
        render_llm_analysis(llm_page)
    elif menu == "Panoramica Dataset":
        render_overview(df_users, df_responses, CHART_LAYOUT)
    elif menu == "🕷️ Spider Charts":
        render_spider(df_responses, EMOTIONS_LIST, EMOTION_COLORS, original_emotions)
    elif menu == "Similarity Analysis":
        import sections.similarity_analysis as similarity_analysis
        similarity_analysis.show(df_responses)


if __name__ == "__main__":
    main()
