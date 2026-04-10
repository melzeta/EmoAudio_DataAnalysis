import json
import os
import re

from annotation.mock_annotator import annotate_with_mock

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional at CLI runtime
    st = None


USE_MOCK = True

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


def _extract_ground_truth_from_prompt(prompt: str) -> dict:
    values = {}
    for emotion in EMOTION_ORDER:
        match = re.search(rf"{emotion}=([0-9]*\.?[0-9]+)", prompt)
        if not match:
            raise ValueError(f"Malformed prompt: missing score for '{emotion}'.")
        values[emotion] = float(match.group(1))
    return values


def _extract_song_key_from_prompt(prompt: str) -> str:
    match = re.search(r"primarily evoking ([^.]+)\.", prompt)
    return match.group(1) if match else "unknown"


def _validate_response(payload: dict, provider_name: str) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(f"{provider_name} returned a non-object response.")

    missing = [emotion for emotion in EMOTION_ORDER if emotion not in payload]
    extra = [key for key in payload if key not in EMOTION_ORDER]
    if missing or extra:
        raise ValueError(
            f"{provider_name} response keys invalid. Missing: {missing or 'none'}. Extra: {extra or 'none'}."
        )

    validated = {}
    for emotion in EMOTION_ORDER:
        try:
            value = float(payload[emotion])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{provider_name} returned a non-numeric value for '{emotion}'.") from exc
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{provider_name} returned out-of-range value for '{emotion}': {value}.")
        validated[emotion] = value
    return validated


def _get_api_key(secret_name: str) -> str:
    if st is not None:
        try:
            return st.secrets[secret_name]
        except Exception:
            pass

    api_key = os.environ.get(secret_name)
    if api_key:
        return api_key
    raise RuntimeError(
        f"Missing {secret_name}. Provide it via Streamlit secrets or the {secret_name} environment variable."
    )


def _mock_response(prompt: str, model_name: str) -> dict:
    ground_truth = _extract_ground_truth_from_prompt(prompt)
    song_key = _extract_song_key_from_prompt(prompt)
    return annotate_with_mock(ground_truth, model_name=model_name, song_key=song_key)


def _call_openrouter(prompt: str, model_name: str) -> dict:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The 'openai' package is required for OpenRouter calls.") from exc

    client = OpenAI(
        api_key=_get_api_key("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content if response.choices else ""
    if not content:
        raise ValueError(f"OpenRouter returned an empty response for model '{model_name}'.")

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenRouter returned malformed JSON for model '{model_name}': {content}") from exc
    return _validate_response(payload, model_name)


def call_deepseek(prompt: str) -> dict:
    if USE_MOCK:
        return _mock_response(prompt, "deepseek")
    return _call_openrouter(prompt, "deepseek/deepseek-chat:free")


def call_gemini(prompt: str) -> dict:
    if USE_MOCK:
        return _mock_response(prompt, "gemini")
    return _call_openrouter(prompt, "google/gemini-2.0-flash-exp:free")


def call_mistral(prompt: str) -> dict:
    if USE_MOCK:
        return _mock_response(prompt, "mistral")
    return _call_openrouter(prompt, "mistralai/mistral-7b-instruct:free")
