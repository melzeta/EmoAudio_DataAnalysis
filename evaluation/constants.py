from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
USER_RESPONSES_PATH = DATA_DIR / "user_emotion_responses.json"
GROUND_TRUTH_PATH = DATA_DIR / "song_emotion_ground_truth.csv"

SPLITS_DIR = ROOT_DIR / "splits"
RESULTS_DIR = ROOT_DIR / "results"
STATE_DIR = ROOT_DIR / "state"
AGENT_LOG_DIR = STATE_DIR / "agent_logs"
AGENT_REPORT_DIR = STATE_DIR / "agent_reports"

MANUAL_CV_STATE_PATH = STATE_DIR / "manual_cv_state.json"
FINAL_VALIDATION_REPORT_PATH = STATE_DIR / "final_validation_report.json"
ELIGIBLE_SAMPLES_PATH = SPLITS_DIR / "eligible_samples.csv"
FOLDS_MANIFEST_PATH = SPLITS_DIR / "folds_manifest.json"

RANDOM_SEED = 42
N_FOLDS = 5

EMOTION_COLUMNS = [
    "amusement",
    "anger",
    "sadness",
    "contentment",
    "disgust",
    "awe",
    "fear",
    "excitement",
]

