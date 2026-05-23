import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import data_loader
import fourier
import pickle

# We obtain in order the data using data_loader, filter the signals with preprocessing and obtain the fourier transform for both EEG and ECG.
def process_data(edf_path, xml_path, eeg_ch, ecg_ch):
    
    loader = data_loader.DataLoader()
    
    # Load with mne and normalize
    raw = loader.load_and_standardize(edf_path, [eeg_ch], [ecg_ch])

    # Apply band-pass filter with signal
    import preprocessing

    preprocessor = preprocessing.SignalPreprocessor()
    raw = preprocessor.apply_filters(raw)
    
    # Split the raw signal into EEG and ECG
    raw_eeg, raw_ecg = loader.split_raw(raw, [eeg_ch], [ecg_ch])

    # Obtain fourier transform for both EEG and ECG
    xf, psd = fourier.process_signal_fourier(raw_eeg, raw_eeg.info['sfreq'], type="EEG")
    lfhf = fourier.process_signal_fourier(raw_ecg, raw_ecg.info['sfreq'], type="ECG")
    
    # Cargamos las anotaciones desde el archivo XML y las superponemos a la señal temporal limpia
    raw = loader.load_annotations(raw, xml_path)

    return raw, xf, psd, lfhf

def save_results(raw, xf, psd, lfhf, ecg_dir, eeg_dir, patient_id, eeg_ch, ecg_ch, overwrite=False):
    # asegurar imports arriba: import numpy as np; import pickle
    ecg_dir.mkdir(parents=True, exist_ok=True)
    eeg_dir.mkdir(parents=True, exist_ok=True)

    # comprobar y extraer canales (copias para no modificar `raw`)
    eeg_exist = [ch for ch in [eeg_ch] if ch in raw.ch_names]
    ecg_exist = [ch for ch in [ecg_ch] if ch in raw.ch_names]

    raw_eeg = raw.copy().pick_channels(eeg_exist) if eeg_exist else None 
    raw_ecg = raw.copy().pick_channels(ecg_exist) if ecg_exist else None
    
    if raw_eeg is None:
        print(f"Warning: No EEG channel found for {patient_id}.")
    if raw_ecg is None:
        print(f"Warning: No ECG channel found for {patient_id}.")

    # obtener arrays NumPy
    eeg_signal = raw_eeg.get_data() if raw_eeg is not None else None  # shape: (n_channels, n_samples)
    ecg_signal = raw_ecg.get_data()[0] if raw_ecg is not None else None  # primer canal ECG

    # Extraer anotaciones desde raw_eeg (que ya las tiene copiadas)
    annotations = None
    if raw_eeg is not None and raw_eeg.annotations is not None and len(raw_eeg.annotations) > 0:
        annotations = {
            'onset': raw_eeg.annotations.onset.tolist(),
            'duration': raw_eeg.annotations.duration.tolist(),
            'description': list(raw_eeg.annotations.description),
            'orig_time': raw_eeg.annotations.orig_time.isoformat() if raw_eeg.annotations.orig_time else None
        }
    
    ecg_path = ecg_dir / f"{patient_id}_ecg.pkl"
    eeg_path = eeg_dir / f"{patient_id}_eeg.pkl"

    # guardar ECG + lfhf (pickle preserva np.inf)
    if overwrite or not ecg_path.exists():
        with open(ecg_path, "wb") as f:
            pickle.dump({"ecg_signal": ecg_signal, "lfhf": lfhf, "annotations": annotations}, f)
        print(f"Saved ECG: {ecg_path}")
    else:
        print(f"Skipped existing ECG: {ecg_path}")

    # guardar EEG + xf + psd
    if overwrite or not eeg_path.exists():
        with open(eeg_path, "wb") as f:
            pickle.dump({"eeg_signal": eeg_signal, "xf": xf, "psd": psd, "annotations": annotations}, f)
        print(f"Saved EEG: {eeg_path}")
    else:
        print(f"Skipped existing EEG: {eeg_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Process MESA EDF files into ECG and EEG pickle files.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing processed pickle files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N candidate patients.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parent

    patient_dir = (repo_root / "data" / "mesa" / "polysomnography" / "edfs").resolve()
    annotations_dir = (repo_root / "data" / "mesa" / "polysomnography" / "annotations-events-nsrr").resolve()
    ecg_dir = (repo_root / "data" / "processed" / "ECG").resolve()
    eeg_dir = (repo_root / "data" / "processed" / "EEG").resolve()

    EEG_channel = "EEG3"
    ECG_channel = "EKG"

    edf_files = sorted(patient_dir.glob("*.edf"))
    candidates = []
    missing_xml = []

    for edf_file in edf_files:
        xml_file = annotations_dir / (edf_file.stem + "-nsrr.xml")
        if xml_file.exists():
            candidates.append((edf_file, xml_file))
        else:
            missing_xml.append(edf_file.stem)

    total_matching_xml = len(candidates)
    if args.limit is not None:
        candidates = candidates[:args.limit]

    processed_successfully = 0
    skipped_existing = 0
    failed = []

    for index, (edf_file, xml_file) in enumerate(candidates, start=1):
        patient_id = edf_file.stem
        ecg_path = ecg_dir / f"{patient_id}_ecg.pkl"
        eeg_path = eeg_dir / f"{patient_id}_eeg.pkl"

        print(f"Processing {index}/{len(candidates)}: {patient_id}")

        if not args.overwrite and ecg_path.exists() and eeg_path.exists():
            print(f"Skipped {patient_id}: outputs already exist")
            skipped_existing += 1
            continue

        try:
            raw, xf, psd, lfhf = process_data(edf_file, xml_file, EEG_channel, ECG_channel)
            save_results(
                raw,
                xf,
                psd,
                lfhf,
                ecg_dir,
                eeg_dir,
                patient_id,
                EEG_channel,
                ECG_channel,
                overwrite=args.overwrite,
            )
            processed_successfully += 1
        except Exception as exc:
            print(f"Failed {patient_id}: {exc}")
            failed.append(patient_id)

    print()
    print("Summary")
    print(f"Total EDF files found: {len(edf_files)}")
    print(f"Patients with matching XML: {total_matching_xml}")
    print(f"Processed successfully: {processed_successfully}")
    print(f"Skipped because outputs already existed: {skipped_existing}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed patient IDs: {', '.join(failed)}")
    if missing_xml:
        print(f"Patients missing XML: {len(missing_xml)}")


if __name__ == "__main__":
    main()
