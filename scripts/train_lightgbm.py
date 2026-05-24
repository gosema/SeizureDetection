import os
import sys
import argparse
from pathlib import Path

if __name__ == "__main__" and not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.modeling import train_and_save, train_combined_and_save


def print_mean_metrics(modality, metrics):
    print(f"Mean {modality} metrics: {metrics['mean']}")
    print(f"Std {modality} metrics: {metrics['std']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ECG, EEG, and combined LightGBM models.")
    parser.add_argument(
        "--sleep-only",
        action="store_true",
        default=config.SLEEP_ONLY,
        help="Train on sleep-only feature CSVs and save sleep-only outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.sleep_only:
        ecg_features_path = config.ECG_SLEEP_ONLY_FEATURES_PATH
        eeg_features_path = config.EEG_SLEEP_ONLY_FEATURES_PATH
        ecg_model_path = config.ECG_SLEEP_ONLY_MODEL_PATH
        eeg_model_path = config.EEG_SLEEP_ONLY_MODEL_PATH
        combined_model_path = config.COMBINED_SLEEP_ONLY_MODEL_PATH
        ecg_metrics_path = config.ECG_SLEEP_ONLY_METRICS_PATH
        eeg_metrics_path = config.EEG_SLEEP_ONLY_METRICS_PATH
        combined_metrics_path = config.COMBINED_SLEEP_ONLY_METRICS_PATH
        ecg_figure_path = config.ECG_SLEEP_ONLY_CONFUSION_MATRIX_PATH
        eeg_figure_path = config.EEG_SLEEP_ONLY_CONFUSION_MATRIX_PATH
        combined_figure_path = config.COMBINED_SLEEP_ONLY_CONFUSION_MATRIX_PATH
    else:
        ecg_features_path = config.ECG_FEATURES_PATH
        eeg_features_path = config.EEG_FEATURES_PATH
        ecg_model_path = config.ECG_MODEL_PATH
        eeg_model_path = config.EEG_MODEL_PATH
        combined_model_path = config.COMBINED_MODEL_PATH
        ecg_metrics_path = config.ECG_METRICS_PATH
        eeg_metrics_path = config.EEG_METRICS_PATH
        combined_metrics_path = config.COMBINED_METRICS_PATH
        ecg_figure_path = config.ECG_CONFUSION_MATRIX_PATH
        eeg_figure_path = config.EEG_CONFUSION_MATRIX_PATH
        combined_figure_path = config.COMBINED_CONFUSION_MATRIX_PATH

    print("Training ECG model with 5-fold GroupKFold...")
    ecg_metrics = train_and_save(
        ecg_features_path,
        ecg_model_path,
        ecg_metrics_path,
        ecg_figure_path,
        "ECG LightGBM Confusion Matrix",
        modality_name="ECG",
    )
    print_mean_metrics("ECG", ecg_metrics)

    print("Training EEG model with 5-fold GroupKFold...")
    eeg_metrics = train_and_save(
        eeg_features_path,
        eeg_model_path,
        eeg_metrics_path,
        eeg_figure_path,
        "EEG LightGBM Confusion Matrix",
        modality_name="EEG",
    )
    print_mean_metrics("EEG", eeg_metrics)

    print("Training combined ECG+EEG model with 5-fold GroupKFold...")
    combined_metrics = train_combined_and_save(
        ecg_features_path,
        eeg_features_path,
        combined_model_path,
        combined_metrics_path,
        combined_figure_path,
        "Combined ECG+EEG LightGBM Confusion Matrix",
        modality_name="Combined",
    )
    print_mean_metrics("combined ECG+EEG", combined_metrics)


if __name__ == "__main__":
    main()
