import json
import os
import re
import time
from pathlib import Path

from annotation.mock_annotator import annotate_with_mock

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional at CLI runtime
    st = None


USE_MOCK = False
# Paid "budget-king" models: stable and cheap enough for this evaluation workflow.
DEEPSEEK_MODEL = "deepseek/deepseek-chat"
GEMINI_MODEL = "google/gemma-3-12b-it:free"
GPT_OSS_MODEL = "openai/gpt-4o-mini"
ALLOW_MODEL_FALLBACKS = False
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_COMPLETION_TOKENS = 300

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
EMOTION_KEY_ALIASES = {
    "excitation": "excitement",
}


def get_run_mode() -> str:
    return "mock" if USE_MOCK else "live"


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


def _canonicalize_emotion_keys(payload: dict, provider_name: str) -> dict:
    canonical = {}
    for key, value in payload.items():
        normalized_key = EMOTION_KEY_ALIASES.get(key, key)
        if normalized_key in canonical and canonical[normalized_key] != value:
            raise ValueError(
                f"{provider_name} returned conflicting values for '{normalized_key}' via alias '{key}'."
            )
        canonical[normalized_key] = value
    return canonical


def _validate_response(payload: dict, provider_name: str) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(f"{provider_name} returned a non-object response.")

    payload = _canonicalize_emotion_keys(payload, provider_name)

    missing = [emotion for emotion in EMOTION_ORDER if emotion not in payload]
    extra = [key for key in payload if key not in EMOTION_ORDER and key != "confidence"]
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

    confidence_payload = payload.get("confidence")
    if confidence_payload is not None:
        if not isinstance(confidence_payload, dict):
            raise ValueError(f"{provider_name} returned a non-object 'confidence' field.")
        confidence_payload = _canonicalize_emotion_keys(confidence_payload, provider_name)
        missing_conf = [emotion for emotion in EMOTION_ORDER if emotion not in confidence_payload]
        extra_conf = [key for key in confidence_payload if key not in EMOTION_ORDER]
        if missing_conf or extra_conf:
            raise ValueError(
                f"{provider_name} confidence keys invalid. Missing: {missing_conf or 'none'}. "
                f"Extra: {extra_conf or 'none'}."
            )

        validated_confidence = {}
        for emotion in EMOTION_ORDER:
            try:
                value = float(confidence_payload[emotion])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{provider_name} returned a non-numeric confidence for '{emotion}'.") from exc
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{provider_name} returned out-of-range confidence for '{emotion}': {value}.")
            validated_confidence[emotion] = value
        validated["confidence"] = validated_confidence

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


def _extract_json_text(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return stripped[start : end + 1]
    return stripped


def _save_raw_output(
    output_path: Path | None,
    provider_name: str,
    requested_model: str,
    resolved_model: str,
    raw_content: str,
    parsed_payload: dict,
) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "provider_name": provider_name,
                "requested_model": requested_model,
                "resolved_model": resolved_model,
                "raw_content": raw_content,
                "parsed_payload": parsed_payload,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _call_openrouter(
    prompt: str,
    model_name: str,
    provider_name: str,
    output_path: Path | None = None,
) -> tuple[dict, str]:
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
        temperature=DEFAULT_TEMPERATURE,
        max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    )
    content = response.choices[0].message.content if response.choices else ""
    if not content:
        raise ValueError(f"OpenRouter returned an empty response for model '{model_name}'.")

    try:
        payload = json.loads(_extract_json_text(content))
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenRouter returned malformed JSON for model '{model_name}': {content}") from exc
    resolved_model = getattr(response, "model", None) or model_name
    _save_raw_output(output_path, provider_name, model_name, resolved_model, content, payload)
    return _validate_response(payload, model_name), resolved_model


def _is_429_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True

    response = getattr(exc, "response", None)
    if response is not None:
        try:
            if getattr(response, "status_code", None) == 429:
                return True
        except Exception:
            pass

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error_payload = body.get("error", {})
        if error_payload.get("code") == 429:
            return True

    message = str(exc)
    return "429" in message


def _is_model_unavailable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {400, 404}:
        return True

    response = getattr(exc, "response", None)
    if response is not None:
        try:
            if getattr(response, "status_code", None) in {400, 404}:
                return True
        except Exception:
            pass

    body = getattr(exc, "body", None)
    message_parts = [str(exc)]
    if isinstance(body, dict):
        error_payload = body.get("error", {})
        code = error_payload.get("code")
        if code in {400, 404}:
            return True
        nested_message = error_payload.get("message")
        if nested_message:
            message_parts.append(str(nested_message))

    message = " ".join(message_parts).lower()
    unavailable_markers = [
        "no endpoints found",
        "not a valid model id",
        "unknown model",
        "model not found",
    ]
    return any(marker in message for marker in unavailable_markers)


def _call_openrouter_with_retry(
    prompt: str,
    model_name: str,
    provider_name: str,
    output_path: Path | None = None,
) -> tuple[dict, str]:
    delay_seconds = 1.0
    last_exc = None
    for attempt in range(1, 6):
        try:
            return _call_openrouter(prompt, model_name, provider_name, output_path=output_path)
        except Exception as exc:
            if not _is_429_error(exc):
                raise
            last_exc = exc
            if attempt == 5:
                break
            print(
                f"[{provider_name}] rate-limited on attempt {attempt}/5 for {model_name}. "
                f"Retrying in {delay_seconds:.0f}s."
            )
            time.sleep(delay_seconds)
            delay_seconds *= 2

    raise RuntimeError(
        f"{provider_name} failed after 5 retries due to repeated 429 rate limits for model '{model_name}'."
    ) from last_exc


def _call_openrouter_with_model_fallbacks(
    prompt: str,
    model_names: list[str],
    provider_name: str,
    output_path: Path | None = None,
) -> tuple[dict, str]:
    last_exc = None
    for index, model_name in enumerate(model_names):
        try:
            if index > 0:
                print(f"[{provider_name}] falling back to {model_name}")
            return _call_openrouter_with_retry(prompt, model_name, provider_name, output_path=output_path)
        except Exception as exc:
            last_exc = exc
            if not _is_model_unavailable_error(exc) or index == len(model_names) - 1:
                raise

    raise RuntimeError(
        f"{provider_name} failed for all configured models: {', '.join(model_names)}"
    ) from last_exc


def _call_configured_model(
    prompt: str,
    primary_model: str,
    provider_name: str,
    output_path: Path | None = None,
    fallback_models: list[str] | None = None,
) -> tuple[dict, str]:
    if ALLOW_MODEL_FALLBACKS and fallback_models:
        return _call_openrouter_with_model_fallbacks(
            prompt,
            [primary_model, *fallback_models],
            provider_name,
            output_path=output_path,
        )
    return _call_openrouter_with_retry(prompt, primary_model, provider_name, output_path=output_path)


def call_deepseek(prompt: str, output_path: Path | None = None) -> tuple[dict, str]:
    if USE_MOCK:
        return _mock_response(prompt, "deepseek"), "mock/deepseek"
    return _call_configured_model(prompt, DEEPSEEK_MODEL, "deepseek", output_path=output_path)


def call_gemini(prompt: str, output_path: Path | None = None) -> tuple[dict, str]:
    if USE_MOCK:
        return _mock_response(prompt, "gemini"), "mock/gemini"
    return _call_configured_model(prompt, GEMINI_MODEL, "gemini", output_path=output_path)


def call_gpt_oss(prompt: str, output_path: Path | None = None) -> tuple[dict, str]:
    if USE_MOCK:
        return _mock_response(prompt, "gpt_oss"), "mock/gpt_oss"
    return _call_configured_model(prompt, GPT_OSS_MODEL, "gpt_oss", output_path=output_path)
