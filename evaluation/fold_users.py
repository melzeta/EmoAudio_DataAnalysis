import json
import random
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
USER_RESPONSES_PATH = ROOT_DIR / "data" / "user_emotion_responses.json"
USER_FOLDS_PATH = ROOT_DIR / "state" / "user_folds.json"
N_FOLDS = 5
SEED = 42


def build_user_folds() -> dict:
    with USER_RESPONSES_PATH.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    users = []
    rows_by_stratum = defaultdict(list)
    for user_id, user_info in sorted(raw_data.get("userData", {}).items()):
        demographics = user_info.get("demographics", {})
        payload = {
            "user_id": user_id,
            "gender": demographics.get("gender", "N/A"),
            "age_range": demographics.get("age_range", "N/A"),
            "nationality": demographics.get("nationality", "N/A"),
            "music_genres": demographics.get("music_genres", []),
        }
        users.append(payload)
        stratum = f"{payload['gender']}|{payload['age_range']}"
        rows_by_stratum[stratum].append(payload)

    rng = random.Random(SEED)
    fold_tests = {fold_index: [] for fold_index in range(1, N_FOLDS + 1)}
    for stratum in sorted(rows_by_stratum):
        stratum_users = list(rows_by_stratum[stratum])
        rng.shuffle(stratum_users)
        for index, user_payload in enumerate(stratum_users):
            fold_index = (index % N_FOLDS) + 1
            fold_tests[fold_index].append(user_payload["user_id"])

    result = {
        "seed": SEED,
        "n_folds": N_FOLDS,
        "users": users,
        "folds": {},
    }
    all_user_ids = sorted(user["user_id"] for user in users)
    for fold_index in range(1, N_FOLDS + 1):
        test_users = sorted(fold_tests[fold_index])
        train_users = sorted(user_id for user_id in all_user_ids if user_id not in set(test_users))
        result["folds"][str(fold_index)] = {
            "fold_index": fold_index,
            "train_users": train_users,
            "test_users": test_users,
            "train_count": len(train_users),
            "test_count": len(test_users),
        }

    USER_FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_FOLDS_PATH.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
