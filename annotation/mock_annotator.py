import hashlib
import random


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

MODEL_SEEDS = {
    "gpt4": 101,
    "claude": 202,
    "gemini": 303,
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def annotate_with_mock(ground_truth_vector: dict, model_name: str, song_key: str = "") -> dict:
    if model_name not in MODEL_SEEDS:
        raise ValueError(f"Unsupported mock model: {model_name}")

    seed_material = f"{model_name}:{song_key}:{sorted(ground_truth_vector.items())}"
    seed_digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    rng = random.Random(MODEL_SEEDS[model_name] + int(seed_digest[:8], 16))

    return {
        emotion: round(_clamp(float(ground_truth_vector[emotion]) + rng.uniform(-0.08, 0.08)), 6)
        for emotion in EMOTION_ORDER
    }
