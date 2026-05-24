import pickle
import re
from pathlib import Path

import numpy as np
import xml.etree.ElementTree as ET

from src import config

try:
    import mne
except ImportError:
    mne = None


SIGNAL_KEYS = (
    "signal",
    "data",
    "ecg",
    "eeg",
    "values",
    "ecg_signal",
    "eeg_signal",
)

FS_KEYS = ("fs", "sfreq", "sampling_frequency", "sample_rate")


def extract_patient_id(file_path):
    """Extract ids such as mesa-sleep-0028 from processed filenames."""
    match = re.search(r"(mesa-sleep-\d+)", Path(file_path).name)
    if not match:
        raise ValueError(f"Could not extract patient_id from {file_path}")
    return match.group(1)


def _pick_signal_from_dict(data, modality=None):
    keys = list(SIGNAL_KEYS)
    if modality:
        keys.insert(0, f"{modality.lower()}_signal")
        keys.insert(1, modality.lower())

    for key in keys:
        if key in data:
            return data[key]
    return None


def _pick_fs_from_dict(data):
    for key in FS_KEYS:
        if key in data and data[key] is not None:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                return None
    return None


def to_1d_array(signal):
    arr = np.asarray(signal, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        raise ValueError("signal is scalar")
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        raise ValueError("signal is empty")
    return arr


def load_processed_pickle(file_path, modality=None, default_fs=None):
    """
    Load a processed ECG/EEG pickle and return patient_id, signal and fs.

    The processed files in student projects often differ a little in shape and
    keys, so this loader accepts either a raw array-like object or a dictionary
    with common signal and sampling-frequency keys.
    """
    file_path = Path(file_path)
    patient_id = extract_patient_id(file_path)
    if default_fs is None:
        default_fs = (
            config.DEFAULT_ECG_FS
            if str(modality).lower() == "ecg"
            else config.DEFAULT_EEG_FS
        )

    with file_path.open("rb") as f:
        obj = pickle.load(f)

    fs = None
    annotations = None
    if isinstance(obj, dict):
        signal = _pick_signal_from_dict(obj, modality=modality)
        fs = _pick_fs_from_dict(obj)
        annotations = obj.get("annotations")
    else:
        signal = obj

    if signal is None:
        raise ValueError("could not find a signal array in pickle")

    signal = to_1d_array(signal)
    if fs is None:
        fs = float(default_fs)

    return {
        "patient_id": patient_id,
        "signal": signal,
        "fs": fs,
        "annotations": annotations,
        "path": file_path,
    }


def load_processed_directory(directory, modality, default_fs=None):
    records = []
    for file_path in sorted(Path(directory).glob("*.pkl")):
        try:
            records.append(
                load_processed_pickle(
                    file_path,
                    modality=modality,
                    default_fs=default_fs,
                )
            )
        except Exception as exc:
            print(f"Warning: skipping {file_path}: {exc}")
    return records


class DataLoader:
    # A target sampling rate is defined since the data belongs to different hospitals that may record at different speeds.
    def __init__(self, target_fs=256.0):
        self.target_fs = target_fs
    
    # The raw signal of a sample is taken and cleaning steps are applied
    # TODO: LOOP FOR ALL PATIENT SAMPLES
    def load_and_standardize(self, edf_path, eeg_channels, ecg_channels):
        if mne is None:
            raise ImportError("mne is required for EDF loading")

        print(f"Processing: {edf_path}...")
        
        # Channels to keep helps reduce memory usage
        channels_to_keep = eeg_channels + ecg_channels
        
        # Load the EDF file with MNE
        raw = mne.io.read_raw_edf(edf_path, include=channels_to_keep, preload=True, verbose=False)
        
        # Filter only the channels of interest (EEG and ECG)
        channels_to_keep = eeg_channels + ecg_channels
        
        # Verify which channels actually exist in the EDF to avoid errors
        existing_channels = [ch for ch in channels_to_keep if ch in raw.ch_names]
        raw.pick(existing_channels)

        # The frequency is standardized to the target frequency; if it does not match, the signal is resampled.
        if abs(raw.info["sfreq"] - self.target_fs) > 1e-6:
            raw.resample(self.target_fs)

        # We use CAR (Common Average Reference), which calculates the mean of all 
        # EEG electrodes and subtracts it, mitigating the physical differences in the montages.
        eeg_existentes = [ch for ch in eeg_channels if ch in raw.ch_names]
        if eeg_existentes:
            # We tell MNE which channels are the EEG channels
            raw.set_channel_types({ch: 'eeg' for ch in eeg_existentes})
            raw.set_eeg_reference('average', projection=False, verbose=False)

        # Normalize Amplitudes (using Z-score normalization (mean 0, standard deviation 1)).
        data = raw.get_data() # Returns a NumPy array (channels x samples)
        # Normalize channel by channel
        for i in range(data.shape[0]):
            media = np.mean(data[i, :])
            std = np.std(data[i, :])
            if std > 0: # Prevent division by zero if a channel is "flat"
                data[i, :] = (data[i, :] - media) / std
                
        # Reassign the normalized data to the raw object
        raw._data = data

        return raw

    # Reads the XML file in NSRR format from polysomnography/annotations-events-nsrr and overlays it on the clean temporal waveform
    def load_annotations(self, raw, xml_path):
        if mne is None:
            raise ImportError("mne is required for MNE annotations")

        try:
            # Parse the XML file
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"    [ERROR] Could not read the XML file: {e}")
            return raw
            
        onsets = []
        durations = []
        descriptions = []
        
        # Search for all labeled events in the file
        # In the NSRR format, the tag is usually <ScoredEvent>
        for event in root.iter('ScoredEvent'):
            # Different databases use 'Name' or 'EventConcept' for the name
            name_node = event.find('EventConcept')
            if name_node is None:
                name_node = event.find('EventConcept')
                
            start_node = event.find('Start')
            duration_node = event.find('Duration')
            
            # If the event has a name, onset, and duration, we extract it
            if name_node is not None and start_node is not None and duration_node is not None:
                desc = name_node.text
                try:
                    start = float(start_node.text)
                    duration = float(duration_node.text)
                    
                    onsets.append(start)
                    durations.append(duration)
                    descriptions.append(desc)
                except ValueError:
                    # Ignore any corrupted text that is not a number
                    continue
                    
        if len(onsets) > 0:
            # Create the MNE annotations object
            # The orig_time from the EDF file is used so the times match exactly
            annotations = mne.Annotations(
                onset=onsets, 
                duration=durations, 
                description=descriptions,
                orig_time=raw.info['meas_date']
            )
            
            # Attach the labels to the physiological signal
            raw.set_annotations(annotations)
                        
        return raw
    
    def split_raw(self, raw, eeg_channels, ecg_channels):
        eeg_exist = [ch for ch in eeg_channels if ch in raw.ch_names]
        ecg_exist = [ch for ch in ecg_channels if ch in raw.ch_names]

        raw_eeg = raw.copy().pick_channels(eeg_exist)
        raw_ecg = raw.copy().pick_channels(ecg_exist)

        return raw_eeg, raw_ecg
