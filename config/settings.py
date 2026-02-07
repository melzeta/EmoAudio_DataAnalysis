import streamlit as st

# ============================================================================
# EMOTIONS
# ============================================================================

EMOTIONS_LIST = [
    "amusement",
    "anger",
    "sadness",
    "contentment",
    "disgust",
    "awe",
    "fear",
    "excitement",
]

EMOTION_COLORS = {
    "amusement": "#F9E264",
    "anger": "#D20101",
    "sadness": "#2A3CBD",
    "contentment": "#A8E6A3",
    "disgust": "#DF1FDB",
    "awe": "#9DBCF5",
    "fear": "#08811E",
    "excitement": "#F88F68",
}

# ============================================================================
# CHART LAYOUT
# ============================================================================

CHART_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#333"),
    margin=dict(l=40, r=40, t=40, b=40),
)

# ============================================================================
# PAGE CONFIG
# ============================================================================

def set_app_config():
    st.set_page_config(
        page_title="Music Emotion Dashboard",
        layout="wide",
    )