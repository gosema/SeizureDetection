import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.annotations import load_apnea_events_for_patient
from src.data_loader import load_processed_directory
from src.features import build_feature_table


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ECG and EEG window features.")
    parser.add_argument(
        "--sleep-only",
        action="store_true",
        default=config.SLEEP_ONLY,
        help="Remove Wake windows and save separate sleep-only feature CSVs.",
    )
    return parser.parse_args()


def _load_events(records):
    patient_ids = sorted({record["patient_id"] for record in records})
    return {
        patient_id: load_apnea_events_for_patient(patient_id)
        for patient_id in patient_ids
    }


def _extract_modality(modality, input_dir, output_path, default_fs, sleep_only=False):
    print(f"Loading processed {modality.upper()} files from {input_dir}")
    records = load_processed_directory(input_dir, modality=modality, default_fs=default_fs)
    print(f"Loaded {len(records)} {modality.upper()} records")

    if not records:
        print(f"Warning: no {modality.upper()} records found; skipping")
        return

    events_by_patient = _load_events(records)
    features = build_feature_table(records, events_by_patient, modality=modality)
    if sleep_only:
        before_count = len(features)
        features = features[features["is_sleep"] == 1].copy()
        print(
            f"Removed {before_count - len(features)} Wake windows "
            f"for {modality.upper()} sleep-only features"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    print(
        f"Saved {len(features)} {modality.upper()} windows "
        f"for {features['patient_id'].nunique()} patients to {output_path}"
    )


def main():
    args = parse_args()
    ecg_output_path = (
        config.ECG_SLEEP_ONLY_FEATURES_PATH
        if args.sleep_only
        else config.ECG_FEATURES_PATH
    )
    eeg_output_path = (
        config.EEG_SLEEP_ONLY_FEATURES_PATH
        if args.sleep_only
        else config.EEG_FEATURES_PATH
    )

    _extract_modality(
        "ecg",
        config.ECG_PROCESSED_DIR,
        ecg_output_path,
        config.DEFAULT_ECG_FS,
        sleep_only=args.sleep_only,
    )
    _extract_modality(
        "eeg",
        config.EEG_PROCESSED_DIR,
        eeg_output_path,
        config.DEFAULT_EEG_FS,
        sleep_only=args.sleep_only,
    )


if __name__ == "__main__":
    main()
