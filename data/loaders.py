import json
import pandas as pd
import streamlit as st


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_and_process_data(json_path: str = "data.json"):
    """Load and process user emotion response data from JSON file"""
    with open(json_path, "r") as f:
        data = json.load(f)

    user_rows = []
    response_rows = []

    for user_id, user_info in data.get("userData", {}).items():
        user_rows.append({
            "user_id": user_id,
            "gender": user_info.get("gender", "N/A"),
            "age": user_info.get("age", "N/A"),
            "num_responses": len(user_info.get("emotionResponses", [])),
        })

        for resp in user_info.get("emotionResponses", []):
            if "emotionValues" in resp:
                path_parts = resp["song"].split("/")
                intended = path_parts[1] if len(path_parts) > 1 else "unknown"

                row = resp["emotionValues"].copy()
                row.update({
                    "user_id": user_id,
                    "song_path": resp["song"],
                    "intended_emotion": intended,
                    "time_spent": resp.get("timeSpentSeconds", 0),
                })
                response_rows.append(row)

    return pd.DataFrame(user_rows), pd.DataFrame(response_rows)


@st.cache_data
def load_original_emotions(csv_path: str = "song_emotions.csv"):
    """Load original emotion values for songs from CSV file"""
    try:
        df_original = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.warning(
            "⚠️ File song_emotions.csv non trovato. Gli spider charts mostreranno solo i dati degli utenti."
        )
        return {}

    original_dict = {}
    for _, row in df_original.iterrows():
        filename = row["filename"]
        original_dict[filename] = {
            "amusement": row["amusement"],
            "anger": row["anger"],
            "awe": row["awe"],
            "contentment": row["contentment"],
            "disgust": row["disgust"],
            "excitement": row["excitement"],
            "fear": row["fear"],
            "sadness": row["sadness"],
        }

    return original_dict