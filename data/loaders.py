import json
import math
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


def _l2_normalize_vector(vec):
    if not vec:
        return {}, 0.0

    norm = math.sqrt(sum(value * value for value in vec.values()))
    if norm == 0:
        return {key: 0.0 for key in vec}, 0.0

    return {key: value / norm for key, value in vec.items()}, norm


def calculate_similarity_top3(original_emotions, user_emotions_avg):
    """Calculate similarity score using L2-normalized top-3 emotions."""
    if not original_emotions or not user_emotions_avg:
        return None, []

    original_norm, _ = _l2_normalize_vector(original_emotions)
    user_norm, _ = _l2_normalize_vector(user_emotions_avg)

    sorted_emotions = sorted(original_norm.items(), key=lambda x: x[1], reverse=True)
    top3_emotions = [e[0] for e in sorted_emotions[:3]]

    original_sub_norm, _ = _l2_normalize_vector({e: original_norm[e] for e in top3_emotions})
    user_sub_norm, _ = _l2_normalize_vector({e: user_norm.get(e, 0.0) for e in top3_emotions})

    similarity = sum(
        original_sub_norm.get(emotion, 0.0) * user_sub_norm.get(emotion, 0.0)
        for emotion in top3_emotions
    )

    return similarity, top3_emotions


def calculate_similarity_top3_with_flag(original_emotions, user_emotions):
    """Calculate similarity score and flag details for top-3 emotions."""
    if not original_emotions or not user_emotions:
        return None, [], False, []

    original_norm, _ = _l2_normalize_vector(original_emotions)
    user_norm, _ = _l2_normalize_vector(user_emotions)

    sorted_emotions = sorted(original_norm.items(), key=lambda x: x[1], reverse=True)
    top3_emotions = [e[0] for e in sorted_emotions[:3]]

    original_sub_norm, _ = _l2_normalize_vector({e: original_norm[e] for e in top3_emotions})
    user_sub_norm, _ = _l2_normalize_vector({e: user_norm.get(e, 0.0) for e in top3_emotions})

    similarity = sum(
        original_sub_norm.get(emotion, 0.0) * user_sub_norm.get(emotion, 0.0)
        for emotion in top3_emotions
    )

    threshold = min((original_norm.get(emotion, 0.0) for emotion in top3_emotions), default=0.0)
    flag_details = [
        {
            "emotion": emotion,
            "user_value": value,
            "threshold": threshold,
        }
        for emotion, value in user_norm.items()
        if emotion not in top3_emotions and value > threshold
    ]
    flag_details.sort(key=lambda item: item["user_value"], reverse=True)

    return similarity, top3_emotions, len(flag_details) > 0, flag_details