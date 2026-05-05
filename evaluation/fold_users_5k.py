import json
import random
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
USER_RESPONSES_PATH = ROOT_DIR / "data" / "user_emotion_responses.json"
WORKFLOW_ROOT = ROOT_DIR / "llm analysis 5 k fold"
STATE_DIR = WORKFLOW_ROOT / "state"
USER_FOLDS_PATH = STATE_DIR / "user_folds.json"
N_FOLDS = 5
SEED = 42


def _user_payloads(raw_data: dict) -> list[dict]:
    users = []
    for user_id, user_info in sorted(raw_data.get("userData", {}).items()):
        demographics = user_info.get("demographics", {})
        users.append(
            {
                "user_id": user_id,
                "gender": demographics.get("gender", user_info.get("gender", "N/A")),
                "age_range": demographics.get("age_range", user_info.get("age", "N/A")),
                "nationality": demographics.get("nationality", "N/A"),
                "music_genres": demographics.get("music_genres", []),
            }
        )
    return users


def _stratified_shuffle(users: list[dict], seed: int = SEED) -> list[dict]:
    rows_by_stratum = defaultdict(list)
    for payload in users:
        stratum = f"{payload['gender']}|{payload['age_range']}"
        rows_by_stratum[stratum].append(payload)

    rng = random.Random(seed)
    for stratum_users in rows_by_stratum.values():
        rng.shuffle(stratum_users)

    ordered_strata = sorted(rows_by_stratum)
    stratified_users = []
    while any(rows_by_stratum[stratum] for stratum in ordered_strata):
        for stratum in ordered_strata:
            if rows_by_stratum[stratum]:
                stratified_users.append(rows_by_stratum[stratum].pop())
    return stratified_users


def build_user_folds() -> dict:
    with USER_RESPONSES_PATH.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    users = _user_payloads(raw_data)
    shuffled_users = _stratified_shuffle(users, seed=SEED)
    fold_boundaries = {
        1: (0, 11),
        2: (11, 22),
        3: (22, 33),
        4: (33, 44),
        5: (44, len(shuffled_users)),
    }

    result = {
        "seed": SEED,
        "n_folds": N_FOLDS,
        "users": users,
        "user_count": len(users),
        "fold_boundaries": {
            str(fold_index): {"start": start, "end": end}
            for fold_index, (start, end) in fold_boundaries.items()
        },
        "folds": {},
    }
    all_user_ids = [user["user_id"] for user in shuffled_users]
    for fold_index in range(1, N_FOLDS + 1):
        start, end = fold_boundaries[fold_index]
        test_users = all_user_ids[start:end]
        train_users = [user_id for user_id in all_user_ids if user_id not in set(test_users)]
        result["folds"][str(fold_index)] = {
            "fold_index": fold_index,
            "start_index": start,
            "end_index_exclusive": end,
            "train_users": sorted(train_users),
            "test_users": sorted(test_users),
            "train_count": len(train_users),
            "test_count": len(test_users),
        }

    USER_FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_FOLDS_PATH.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
