EMOTION_ORDER = [
    "amusement",
    "anger",
    "awe",
    "contentment",
    "disgust",
    "excitement",
    "fear",
    "sadness",
]


def build_prompt(song_filename: str, intended_emotion: str, ground_truth_vector: dict) -> str:
    _ = song_filename
    values = ", ".join(
        f"{emotion}={float(ground_truth_vector[emotion])}"
        for emotion in EMOTION_ORDER
    )
    return (
        "You are an expert music psychologist. "
        f"A music clip is categorised as primarily evoking {intended_emotion}. "
        f"Musicologists have rated it on 8 emotions (0 to 1 scale): {values}. "
        "Based on your knowledge of how humans perceive music emotion, predict how an average listener "
        "would rate this same clip on all 8 emotions. Return only a JSON object with exactly these keys: "
        "amusement, anger, awe, contentment, disgust, excitement, fear, sadness. "
        "All values must be floats between 0 and 1. No explanation, no markdown, just the JSON object."
    )
