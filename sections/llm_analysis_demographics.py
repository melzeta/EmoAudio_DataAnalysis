import streamlit as st

import sections.llm_analysis as base_llm_analysis


def render() -> None:
    st.header("Analysis by Demographics")
    st.caption(
        "This section uses the original LLM analysis data, but presents it in the same "
        "tabbed structure as `Analysis > 2 Users`."
    )

    overview_tab, review_tab, comparison_tab, reports_tab = st.tabs(
        ["Overview", "Fold Review", "Model Comparison", "Agent Reports"]
    )
    with overview_tab:
        base_llm_analysis._render_overview()
    with review_tab:
        base_llm_analysis._render_fold_review()
    with comparison_tab:
        base_llm_analysis._render_model_comparison()
    with reports_tab:
        base_llm_analysis._render_agent_reports()
