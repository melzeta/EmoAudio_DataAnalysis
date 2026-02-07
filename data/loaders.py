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


# ============================================================================
# SIMILARITY CALCULATION FUNCTIONS
# ============================================================================

def load_original_emotions_by_filename(csv_path: str = "song_emotions.csv"):
    """
    Load original emotion values using only filename (not full path).
    Returns dict: {filename: {emotion: value}}
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.warning("⚠️ File song_emotions.csv non trovato.")
        return {}
    
    original_data = {}
    for _, row in df.iterrows():
        # Estrai solo il nome del file (ultima parte dopo \ o /)
        filename = row['filename'].replace('\\', '/').split('/')[-1]
        
        emotions = {
            'amusement': row['amusement'],
            'anger': row['anger'],
            'awe': row['awe'],
            'contentment': row['contentment'],
            'disgust': row['disgust'],
            'excitement': row['excitement'],
            'fear': row['fear'],
            'sadness': row['sadness']
        }
        original_data[filename] = emotions
    
    return original_data


def calculate_similarity_top3(original_emotions, user_emotions_avg):
    """
    Calculate similarity score (inner product) between original and user emotions
    using only the top 3 emotions from original values.
    
    Args:
        original_emotions: dict {emotion: value} - original values
        user_emotions_avg: dict {emotion: value} - user averages
    
    Returns:
        (similarity_score: float, top3_emotions: list)
    """
    if not original_emotions or not user_emotions_avg:
        return None, []
    
    # Sort original emotions by value (descending) and take top 3
    sorted_emotions = sorted(original_emotions.items(), key=lambda x: x[1], reverse=True)
    top3_emotions = [e[0] for e in sorted_emotions[:3]]
    
    # Inner product: sum of products for top 3 emotions
    similarity = sum(
        original_emotions[emotion] * user_emotions_avg.get(emotion, 0)
        for emotion in top3_emotions
    )
    
    return similarity, top3_emotions