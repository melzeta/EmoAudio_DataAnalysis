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
    "deepseek": 111,
    "gemini": 222,
    "mistral": 333,
}
MODEL_BIASES = {
    "deepseek": {
        "amusement": 0.035,
        "anger": -0.02,
        "awe": 0.01,
        "contentment": -0.01,
        "disgust": 0.02,
        "excitement": 0.04,
        "fear": -0.015,
        "sadness": -0.02,
    },
    "gemini": {
        "amusement": -0.015,
        "anger": 0.03,
        "awe": 0.045,
        "contentment": 0.015,
        "disgust": -0.025,
        "excitement": 0.01,
        "fear": 0.03,
        "sadness": -0.015,
    },
    "mistral": {
        "amusement": -0.03,
        "anger": 0.015,
        "awe": -0.02,
        "contentment": 0.035,
        "disgust": 0.01,
        "excitement": -0.025,
        "fear": 0.04,
        "sadness": 0.03,
    },
}
MODEL_SCALES = {
    "deepseek": {
        "amusement": 1.18,
        "anger": 0.88,
        "awe": 1.02,
        "contentment": 0.9,
        "disgust": 1.08,
        "excitement": 1.16,
        "fear": 0.86,
        "sadness": 0.84,
    },
    "gemini": {
        "amusement": 0.92,
        "anger": 1.12,
        "awe": 1.2,
        "contentment": 1.05,
        "disgust": 0.9,
        "excitement": 1.02,
        "fear": 1.18,
        "sadness": 0.9,
    },
    "mistral": {
        "amusement": 0.86,
        "anger": 1.08,
        "awe": 0.9,
        "contentment": 1.22,
        "disgust": 1.04,
        "excitement": 0.9,
        "fear": 1.14,
        "sadness": 1.2,
    },
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
        emotion: round(
            _clamp(
                (float(ground_truth_vector[emotion]) * MODEL_SCALES[model_name][emotion])
                + MODEL_BIASES[model_name][emotion]
                + rng.uniform(-0.12, 0.12)
            ),
            6,
        )
        for emotion in EMOTION_ORDER
    }
