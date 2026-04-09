import argparse
import json

from evaluation.orchestrator import load_review_bundle, prepare_folds, review_fold, run_final_validations, run_fold
from evaluation.state import load_state


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual 5-fold evaluation workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare", help="Prepare eligible samples and 5 deterministic folds")

    run_parser = subparsers.add_parser("run-fold", help="Run exactly one fold")
    run_parser.add_argument("--fold", type=int, required=True, help="Fold index to run")

    review_parser = subparsers.add_parser("review", help="Mark a completed fold as reviewed")
    review_parser.add_argument("--fold", type=int, required=True, help="Fold index to review")
    review_parser.add_argument("--approve-next", action="store_true", help="Approve the next fold to run")

    bundle_parser = subparsers.add_parser("show-fold", help="Print the review bundle for one fold")
    bundle_parser.add_argument("--fold", type=int, required=False, help="Fold index to inspect")

    subparsers.add_parser("status", help="Show current manual CV state")
    subparsers.add_parser("validate", help="Run validation reports")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        print(json.dumps(prepare_folds(), indent=2))
    elif args.command == "run-fold":
        print(json.dumps(run_fold(args.fold), indent=2))
    elif args.command == "review":
        print(json.dumps(review_fold(args.fold, approve_next=args.approve_next), indent=2))
    elif args.command == "show-fold":
        print(json.dumps(load_review_bundle(args.fold), indent=2))
    elif args.command == "status":
        print(json.dumps(load_state(), indent=2))
    elif args.command == "validate":
        print(json.dumps(run_final_validations(), indent=2))


if __name__ == "__main__":
    main()

