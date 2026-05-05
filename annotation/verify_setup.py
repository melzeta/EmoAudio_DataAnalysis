import os
import time
from pathlib import Path
import tomllib

from annotation.llm_clients import (
    DEEPSEEK_MODEL,
    GPT_OSS_MODEL,
    USE_MOCK,
    call_deepseek,
    call_gpt_oss,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
STREAMLIT_SECRETS_PATH = ROOT_DIR / ".streamlit" / "secrets.toml"
TEST_PROMPT = (
    "Return only a JSON object with exactly these keys: amusement, anger, awe, contentment, "
    "disgust, excitement, fear, sadness. All values must be floats between 0 and 1. "
    "Use these values exactly: amusement=0.11, anger=0.22, awe=0.33, contentment=0.44, "
    "disgust=0.55, excitement=0.66, fear=0.77, sadness=0.88."
)
VERIFY_COOLDOWN_SECONDS = int(os.environ.get("VERIFY_COOLDOWN_SECONDS", "10"))


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def _load_streamlit_secrets(path: Path) -> None:
    if not path.exists():
        return
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    for key, value in data.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)


def _run_check(label: str, caller) -> bool:
    try:
        result, resolved_model = caller(TEST_PROMPT)
        print(f"[PASS] {label}: resolved_model={resolved_model}, keys={sorted(result.keys())}")
        return True
    except Exception as exc:
        print(f"[FAIL] {label}: {exc}")
        return False


def _run_all_checks() -> list[bool]:
    checks = [
        ("deepseek", call_deepseek),
        ("gpt_oss", call_gpt_oss),
    ]
    results = []
    for index, (label, caller) in enumerate(checks):
        results.append(_run_check(label, caller))
        if index < len(checks) - 1:
            print(f"Cooling down for {VERIFY_COOLDOWN_SECONDS}s to avoid free-tier rate limits...")
            time.sleep(VERIFY_COOLDOWN_SECONDS)
    return results


def main() -> int:
    _load_dotenv(ENV_PATH)
    _load_streamlit_secrets(STREAMLIT_SECRETS_PATH)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    print(f"OPENROUTER_API_KEY set: {'yes' if api_key else 'no'}")
    print(f"USE_MOCK = {USE_MOCK}")
    print(f"DEEPSEEK_MODEL = {DEEPSEEK_MODEL}")
    print(f"GPT_OSS_MODEL = {GPT_OSS_MODEL}")

    if not api_key:
        print("OPENROUTER_API_KEY is missing.")
        return 1
    if USE_MOCK:
        print("USE_MOCK must be False for real-model verification.")
        return 1

    results = _run_all_checks()

    if all(results):
        print("All model checks passed.")
        return 0

    print("One or more model checks failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
