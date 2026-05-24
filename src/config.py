from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ECG_PROCESSED_DIR = PROCESSED_DIR / "ECG"
EEG_PROCESSED_DIR = PROCESSED_DIR / "EEG"

ANNOTATIONS_DIR = (
    DATA_DIR
    / "mesa"
    / "polysomnography"
    / "annotations-events-nsrr"
)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FEATURES_DIR = OUTPUTS_DIR / "features"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"

ECG_FEATURES_PATH = FEATURES_DIR / "ecg_features.csv"
EEG_FEATURES_PATH = FEATURES_DIR / "eeg_features.csv"
ECG_SLEEP_ONLY_FEATURES_PATH = FEATURES_DIR / "ecg_features_sleep_only.csv"
EEG_SLEEP_ONLY_FEATURES_PATH = FEATURES_DIR / "eeg_features_sleep_only.csv"

ECG_MODEL_PATH = MODELS_DIR / "lightgbm_ecg.pkl"
EEG_MODEL_PATH = MODELS_DIR / "lightgbm_eeg.pkl"
COMBINED_MODEL_PATH = MODELS_DIR / "lightgbm_combined.pkl"
ECG_SLEEP_ONLY_MODEL_PATH = MODELS_DIR / "lightgbm_ecg_sleep_only.pkl"
EEG_SLEEP_ONLY_MODEL_PATH = MODELS_DIR / "lightgbm_eeg_sleep_only.pkl"
COMBINED_SLEEP_ONLY_MODEL_PATH = MODELS_DIR / "lightgbm_combined_sleep_only.pkl"

ECG_METRICS_PATH = OUTPUTS_DIR / "metrics_ecg.json"
EEG_METRICS_PATH = OUTPUTS_DIR / "metrics_eeg.json"
COMBINED_METRICS_PATH = OUTPUTS_DIR / "metrics_combined.json"
ECG_SLEEP_ONLY_METRICS_PATH = OUTPUTS_DIR / "metrics_ecg_sleep_only.json"
EEG_SLEEP_ONLY_METRICS_PATH = OUTPUTS_DIR / "metrics_eeg_sleep_only.json"
COMBINED_SLEEP_ONLY_METRICS_PATH = OUTPUTS_DIR / "metrics_combined_sleep_only.json"

ECG_CONFUSION_MATRIX_PATH = FIGURES_DIR / "ecg_confusion_matrix.png"
EEG_CONFUSION_MATRIX_PATH = FIGURES_DIR / "eeg_confusion_matrix.png"
COMBINED_CONFUSION_MATRIX_PATH = FIGURES_DIR / "combined_confusion_matrix.png"
ECG_SLEEP_ONLY_CONFUSION_MATRIX_PATH = FIGURES_DIR / "ecg_confusion_matrix_sleep_only.png"
EEG_SLEEP_ONLY_CONFUSION_MATRIX_PATH = FIGURES_DIR / "eeg_confusion_matrix_sleep_only.png"
COMBINED_SLEEP_ONLY_CONFUSION_MATRIX_PATH = FIGURES_DIR / "combined_confusion_matrix_sleep_only.png"

DEFAULT_ECG_FS = 256.0
DEFAULT_EEG_FS = 256.0
WINDOW_SECONDS = 30.0
MIN_APNEA_OVERLAP_SECONDS = 1.0
RANDOM_STATE = 42
SLEEP_ONLY = False

APNEA_EVENT_NAMES = {
    "hypopnea",
    "obstructive apnea",
    "central apnea",
    "mixed apnea",
}
