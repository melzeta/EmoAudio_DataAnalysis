import json
import os
import re

import streamlit as st

from annotation.mock_annotator import annotate_with_mock


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


def _extract_intended_emotion(prompt: str) -> str:
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


def _mock_response(prompt: str, model_name: str) -> dict:
    ground_truth = _extract_ground_truth_from_prompt(prompt)
    song_key = _extract_intended_emotion(prompt)
    return annotate_with_mock(ground_truth, model_name=model_name, song_key=song_key)


def _get_api_key(secret_name: str) -> str:
    try:
        return st.secrets[secret_name]
    except Exception:
        api_key = os.environ.get(secret_name)
        if api_key:
            return api_key
    raise RuntimeError(
        f"Missing {secret_name}. Provide it via Streamlit secrets or the {secret_name} environment variable."
    )


def call_gpt4(prompt: str) -> dict:
    if USE_MOCK:
        return _mock_response(prompt, "gpt4")

    api_key = _get_api_key("OPENAI_API_KEY")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The 'openai' package is required for GPT-4 calls.") from exc

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model="gpt-4o",
        input=prompt,
    )
    text = response.output_text
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenAI returned malformed JSON: {text}") from exc
    return _validate_response(payload, "OpenAI")


def call_claude(prompt: str) -> dict:
    if USE_MOCK:
        return _mock_response(prompt, "claude")

    api_key = _get_api_key("ANTHROPIC_API_KEY")

    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("The 'anthropic' package is required for Claude calls.") from exc

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    text_blocks = [block.text for block in response.content if getattr(block, "type", "") == "text"]
    text = "".join(text_blocks).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Anthropic returned malformed JSON: {text}") from exc
    return _validate_response(payload, "Anthropic")


def call_gemini(prompt: str) -> dict:
    if USE_MOCK:
        return _mock_response(prompt, "gemini")

    api_key = _get_api_key("GOOGLE_API_KEY")

    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError("The 'google-generativeai' package is required for Gemini calls.") from exc

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    text = (response.text or "").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini returned malformed JSON: {text}") from exc
    return _validate_response(payload, "Gemini")
