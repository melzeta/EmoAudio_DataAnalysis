import csv
import json
from collections import Counter

from evaluation.constants import EMOTION_COLUMNS, GROUND_TRUTH_PATH, USER_RESPONSES_PATH


def normalize_song_key(value: str) -> str:
    return value.replace("\\", "/").split("/")[-1]


def load_ground_truth() -> dict:
    rows = []
    by_song_key = {}
    with GROUND_TRUTH_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized_key = normalize_song_key(row["filename"])
            payload = {
                "filename": row["filename"],
                "song_key": normalized_key,
            }
            for emotion in EMOTION_COLUMNS:
                payload[emotion] = float(row[emotion])
            rows.append(payload)
            by_song_key[normalized_key] = payload
    return {"rows": rows, "by_song_key": by_song_key}


def _required_emotions_present(values: dict) -> bool:
    return all(emotion in values for emotion in EMOTION_COLUMNS)


def discover_eligible_samples() -> dict:
    with USER_RESPONSES_PATH.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    ground_truth = load_ground_truth()
    ground_truth_by_song = ground_truth["by_song_key"]

    discovery = {
        "input_files": {
            "user_responses": str(USER_RESPONSES_PATH),
            "ground_truth": str(GROUND_TRUTH_PATH),
        },
        "top_level_keys": sorted(raw_data.keys()),
        "registered_users": len(raw_data.get("userData", {})),
        "raw_response_rows": 0,
        "eligible_rows_before_deduplication": 0,
        "eligible_rows_after_deduplication": 0,
        "ineligible_missing_song": 0,
        "ineligible_missing_emotion_values": 0,
        "ineligible_missing_required_emotions": 0,
        "ineligible_missing_ground_truth": 0,
        "unique_song_count": 0,
        "intended_emotion_distribution": {},
    }
    integrity = {
        "status": "passed",
        "duplicate_exact_rows_removed": 0,
        "duplicate_groups": [],
        "out_of_range_values": [],
        "song_match_failures": [],
        "notes": [],
    }

    dedup_seen = {}
    eligible_rows = []
    intended_counter = Counter()

    for user_id, user_info in raw_data.get("userData", {}).items():
        responses = user_info.get("emotionResponses", [])
        for response_index, response in enumerate(responses):
            discovery["raw_response_rows"] += 1

            song_path = response.get("song")
            if not song_path:
                discovery["ineligible_missing_song"] += 1
                continue

            emotion_values = response.get("emotionValues")
            if not emotion_values:
                discovery["ineligible_missing_emotion_values"] += 1
                continue

            if not _required_emotions_present(emotion_values):
                discovery["ineligible_missing_required_emotions"] += 1
                continue

            song_key = normalize_song_key(song_path)
            ground_truth_row = ground_truth_by_song.get(song_key)
            if ground_truth_row is None:
                discovery["ineligible_missing_ground_truth"] += 1
                integrity["song_match_failures"].append(song_path)
                continue

            out_of_range = [
                {
                    "sample_hint": f"{user_id}:{song_path}:{response_index}",
                    "emotion": emotion,
                    "value": emotion_values[emotion],
                }
                for emotion in EMOTION_COLUMNS
                if float(emotion_values[emotion]) < 0.0 or float(emotion_values[emotion]) > 1.0
            ]
            integrity["out_of_range_values"].extend(out_of_range)

            intended_emotion = "unknown"
            song_parts = song_path.replace("\\", "/").split("/")
            if len(song_parts) > 1:
                intended_emotion = song_parts[1]

            dedup_key = (
                user_id,
                song_path,
                tuple(float(emotion_values[emotion]) for emotion in EMOTION_COLUMNS),
            )
            if dedup_key in dedup_seen:
                integrity["duplicate_exact_rows_removed"] += 1
                integrity["duplicate_groups"].append(
                    {
                        "sample_id_kept": dedup_seen[dedup_key],
                        "duplicate_user_id": user_id,
                        "duplicate_song_path": song_path,
                        "duplicate_response_index": response_index,
                    }
                )
                continue

            sample_id = f"{user_id}__{song_key.replace('.', '_')}__r{response_index:03d}"
            dedup_seen[dedup_key] = sample_id

            row = {
                "sample_id": sample_id,
                "user_id": user_id,
                "song_path": song_path,
                "song_key": song_key,
                "ground_truth_filename": ground_truth_row["filename"],
                "response_index": response_index,
                "intended_emotion": intended_emotion,
                "time_spent_seconds": response.get("timeSpentSeconds", 0),
            }
            for emotion in EMOTION_COLUMNS:
                row[f"song_{emotion}"] = float(ground_truth_row[emotion])
                row[f"true_{emotion}"] = float(emotion_values[emotion])

            eligible_rows.append(row)
            intended_counter[intended_emotion] += 1

    if integrity["song_match_failures"] or integrity["out_of_range_values"]:
        integrity["status"] = "failed"
    elif integrity["duplicate_exact_rows_removed"]:
        integrity["status"] = "warning"
        integrity["notes"].append("Exact duplicate evaluated rows were removed before fold creation.")

    discovery["eligible_rows_before_deduplication"] = len(eligible_rows) + integrity["duplicate_exact_rows_removed"]
    discovery["eligible_rows_after_deduplication"] = len(eligible_rows)
    discovery["unique_song_count"] = len({row["song_path"] for row in eligible_rows})
    discovery["intended_emotion_distribution"] = dict(sorted(intended_counter.items()))

    return {
        "eligible_rows": eligible_rows,
        "discovery_report": discovery,
        "integrity_report": integrity,
    }

