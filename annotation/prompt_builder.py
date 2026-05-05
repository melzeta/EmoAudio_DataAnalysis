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


def _format_vector(values: dict) -> str:
    return ", ".join(
        f"{emotion}={float(values[emotion])}"
        for emotion in EMOTION_ORDER
    )


def _format_training_examples(training_examples: list[dict]) -> str:
    if not training_examples:
        return "No training examples were available for this fold."

    lines = []
    for index, example in enumerate(training_examples, start=1):
        lines.append(
            f"Example {index}: "
            f"song={example['filename']}; "
            f"intended_emotion={example['intended_emotion']}; "
            f"musicologist_vector=({_format_vector(example['ground_truth_vector'])}); "
            f"train_user_average=({_format_vector(example['train_user_average'])}); "
            f"train_ratings={int(example['num_ratings'])}"
        )
    return "\n".join(lines)


def build_prompt(
    song_filename: str,
    intended_emotion: str,
    ground_truth_vector: dict,
    training_examples: list[dict],
) -> str:
    values = _format_vector(ground_truth_vector)
    examples_block = _format_training_examples(training_examples)
    example_count = len(training_examples)
    return (
        "You are an expert music psychologist. "
        "You are doing 5-fold evaluation with few-shot in-context learning. "
        "The examples below come only from the training portion of the fold (roughly 80% of users). "
        "Learn the mapping between musicologist emotion vectors and average listener responses from these examples, "
        "then predict the held-out test-song listener response for the target clip. "
        "Do not copy any example mechanically; infer the relationship from the training examples.\n\n"
        f"Training examples available: {example_count}\n"
        f"{examples_block}\n\n"
        "Target song:\n"
        f"song={song_filename}; intended_emotion={intended_emotion}; musicologist_vector=({values})\n\n"
        "Return only a JSON object with exactly these keys: "
        "amusement, anger, awe, contentment, disgust, excitement, fear, sadness, confidence. "
        "The first 8 keys must be floats between 0 and 1. "
        "The 'confidence' key must be an object with the same 8 emotion keys, each also a float between 0 and 1, "
        "representing your confidence in that emotion score. "
        "No explanation, no markdown, just the JSON object."
    )
