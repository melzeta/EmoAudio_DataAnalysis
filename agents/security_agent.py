import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
ANNOTATION_DIR = ROOT_DIR / "annotation"
AGENTS_DIR = ROOT_DIR / "agents"
SECTIONS_DIR = ROOT_DIR / "sections"
DATA_ANNOTATIONS_DIR = ROOT_DIR / "data" / "annotations"

KEY_PATTERN = re.compile(
    "|".join(
        [
            re.escape("sk" + "-"),
            re.escape("AI" + "za"),
            re.escape("sk" + "-ant" + "-"),
            r"Bearer\s+[A-Za-z0-9]",
            re.escape("sk" + "-or" + "-"),
        ]
    ),
    re.IGNORECASE,
)
PROMPT_INJECTION_PATTERN = re.compile(
    "|".join(
        [
            "ignore previous " + "instructions",
            "dis" + "regard",
            "you are " + "now",
            "new " + "instructions",
        ]
    ),
    re.IGNORECASE,
)
MAX_CSV_SIZE_BYTES = 500 * 1024
ALLOWED_ANNOTATION_SUFFIXES = {".csv", ".gitkeep"}
IGNORED_DIR_NAMES = {"__pycache__"}
IGNORED_SUFFIXES = {".pyc"}


def _check_file(path: Path, issues: list[str]) -> None:
    if any(part in IGNORED_DIR_NAMES for part in path.parts) or path.suffix.lower() in IGNORED_SUFFIXES:
        return
    if not path.is_file():
        issues.append(f"Missing file: {path.relative_to(ROOT_DIR)}")
        return

    if path.suffix.lower() == ".csv" and path.stat().st_size > MAX_CSV_SIZE_BYTES:
        issues.append(f"CSV exceeds 500KB: {path.relative_to(ROOT_DIR)}")

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        issues.append(f"File is not valid UTF-8 text: {path.relative_to(ROOT_DIR)}")
        return

    if KEY_PATTERN.search(content):
        issues.append(f"Potential credential pattern found in {path.relative_to(ROOT_DIR)}")
    if PROMPT_INJECTION_PATTERN.search(content):
        issues.append(f"Prompt injection phrase found in {path.relative_to(ROOT_DIR)}")


def _iter_default_files(fold_number: int) -> list[Path]:
    files = []
    for directory in [AGENTS_DIR, ANNOTATION_DIR, SECTIONS_DIR, DATA_ANNOTATIONS_DIR / f"fold_{fold_number}"]:
        if directory.exists():
            files.extend(
                sorted(
                    path
                    for path in directory.rglob("*")
                    if path.is_file()
                    and not any(part in IGNORED_DIR_NAMES for part in path.parts)
                    and path.suffix.lower() not in IGNORED_SUFFIXES
                )
            )
    return files


def _check_annotations_tree(issues: list[str]) -> None:
    if not DATA_ANNOTATIONS_DIR.exists():
        return

    for path in DATA_ANNOTATIONS_DIR.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIR_NAMES for part in path.parts) or path.suffix.lower() in IGNORED_SUFFIXES:
            continue
        if path.suffix.lower() not in ALLOWED_ANNOTATION_SUFFIXES:
            issues.append(f"Disallowed file in data/annotations: {path.relative_to(ROOT_DIR)}")


def run(fold_number, files_to_check=None) -> dict:
    issues = []
    if files_to_check:
        paths = [ROOT_DIR / relative_path for relative_path in files_to_check]
    else:
        paths = _iter_default_files(fold_number)

    for path in paths:
        _check_file(path, issues)

    _check_annotations_tree(issues)

    return {
        "agent": "security",
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "fold": fold_number,
    }
