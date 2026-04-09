import random
from collections import defaultdict
from pathlib import Path

from evaluation.constants import ELIGIBLE_SAMPLES_PATH, FOLDS_MANIFEST_PATH, N_FOLDS, RANDOM_SEED, SPLITS_DIR
from evaluation.utils import write_csv, write_json


def _fold_dir(fold_index: int) -> Path:
    return SPLITS_DIR / f"fold_{fold_index}"


def build_folds(eligible_rows: list, seed: int = RANDOM_SEED) -> dict:
    rows_by_stratum = defaultdict(list)
    for row in eligible_rows:
        rows_by_stratum[row["intended_emotion"]].append(row)

    rng = random.Random(seed)
    fold_assignments = {fold_index: [] for fold_index in range(1, N_FOLDS + 1)}

    for stratum in sorted(rows_by_stratum):
        rows = sorted(rows_by_stratum[stratum], key=lambda item: item["sample_id"])
        rng.shuffle(rows)
        for index, row in enumerate(rows):
            fold_index = (index % N_FOLDS) + 1
            fold_assignments[fold_index].append(row["sample_id"])

    sample_lookup = {row["sample_id"]: row for row in eligible_rows}
    all_test_ids = []
    fold_summaries = []

    manifest = {
        "seed": seed,
        "n_folds": N_FOLDS,
        "eligible_sample_count": len(eligible_rows),
        "folds": {},
    }

    eligible_fieldnames = list(eligible_rows[0].keys()) if eligible_rows else []
    write_csv(ELIGIBLE_SAMPLES_PATH, eligible_rows, eligible_fieldnames)

    for fold_index in range(1, N_FOLDS + 1):
        test_ids = sorted(fold_assignments[fold_index])
        train_ids = sorted(sample_lookup.keys() - set(test_ids))
        all_test_ids.extend(test_ids)

        train_rows = [sample_lookup[sample_id] for sample_id in train_ids]
        test_rows = [sample_lookup[sample_id] for sample_id in test_ids]

        fold_dir = _fold_dir(fold_index)
        write_csv(fold_dir / "train.csv", train_rows, eligible_fieldnames)
        write_csv(fold_dir / "test.csv", test_rows, eligible_fieldnames)

        manifest["folds"][str(fold_index)] = {
            "fold_index": fold_index,
            "train_count": len(train_rows),
            "test_count": len(test_rows),
            "train_path": str(fold_dir / "train.csv"),
            "test_path": str(fold_dir / "test.csv"),
        }
        fold_summaries.append(manifest["folds"][str(fold_index)])

    coverage_ok = sorted(all_test_ids) == sorted(sample_lookup.keys())
    no_overlap = sum(len(values) for values in fold_assignments.values()) == len(sample_lookup)

    manifest["validation"] = {
        "coverage_ok": coverage_ok,
        "no_overlap": no_overlap,
        "all_test_sample_ids_sorted": sorted(all_test_ids),
    }
    write_json(FOLDS_MANIFEST_PATH, manifest)

    return {
        "manifest": manifest,
        "fold_summaries": fold_summaries,
    }

