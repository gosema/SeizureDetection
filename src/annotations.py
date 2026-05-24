import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from src import config


def _text_or_none(node, names):
    for name in names:
        child = node.find(name)
        if child is not None and child.text is not None:
            return child.text.strip()
    return None


def _is_positive_event(name):
    if not name:
        return False
    parts = [part.strip().lower() for part in name.split("|")]
    return any(part in config.APNEA_EVENT_NAMES for part in parts)


def find_annotation_file(patient_id, annotations_dir=config.ANNOTATIONS_DIR):
    annotations_dir = Path(annotations_dir)
    candidates = sorted(annotations_dir.glob(f"{patient_id}*.xml"))
    candidates += sorted(annotations_dir.glob(f"{patient_id}*.sml"))
    return candidates[0] if candidates else None


def parse_apnea_events(annotation_path):
    """
    Parse positive apnea/hypopnea events from an NSRR XML/SML file.

    Returns a list of (start, end) intervals in seconds.
    """
    annotation_path = Path(annotation_path)
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    events = []
    for event in root.iter("ScoredEvent"):
        name = _text_or_none(event, ("EventConcept", "Name"))
        if not _is_positive_event(name):
            continue

        start_text = _text_or_none(event, ("Start",))
        duration_text = _text_or_none(event, ("Duration",))
        if start_text is None or duration_text is None:
            continue

        try:
            start = float(start_text)
            duration = float(duration_text)
        except ValueError:
            continue

        if duration <= 0:
            continue
        events.append((start, start + duration))

    return events


def load_apnea_events_for_patient(patient_id, annotations_dir=config.ANNOTATIONS_DIR):
    annotation_file = find_annotation_file(patient_id, annotations_dir)
    if annotation_file is None:
        print(f"Warning: no annotation file found for {patient_id}")
        return []

    try:
        return parse_apnea_events(annotation_file)
    except Exception as exc:
        print(f"Warning: could not parse annotations for {patient_id}: {exc}")
        return []


def _normalize_stage_name(description):
    if not description:
        return None

    text = str(description).strip().lower()
    parts = [part.strip() for part in text.split("|")]

    if "wake" in text or "0" in parts:
        return "W"
    if "stage 1" in text or "n1" in parts or "1" in parts:
        return "N1"
    if "stage 2" in text or "n2" in parts or "2" in parts:
        return "N2"
    if "stage 3" in text or "stage 4" in text or "n3" in parts or "3" in parts or "4" in parts:
        return "N3"
    if "rem sleep" in text or text == "rem" or "rem" in parts or "5" in parts:
        return "REM"
    return None


def extract_sleep_stage_events(events):
    """
    Extract sleep-stage intervals from a processed pickle annotation dict.

    Returns tuples of (start, end, stage), where stage is one of W, N1, N2,
    N3, or REM.
    """
    if not events:
        return []

    onsets = events.get("onset", [])
    durations = events.get("duration", [])
    descriptions = events.get("description", [])

    stage_events = []
    for onset, duration, description in zip(onsets, durations, descriptions):
        stage = _normalize_stage_name(description)
        if stage is None:
            continue

        try:
            start = float(onset)
            duration = float(duration)
        except (TypeError, ValueError):
            continue

        if duration <= 0:
            continue
        stage_events.append((start, start + duration, stage))

    return stage_events


def stage_for_window(window_start, window_end, stage_events):
    best_stage = "UNKNOWN"
    best_overlap = 0.0

    for event_start, event_end, stage in stage_events:
        overlap = min(window_end, event_end) - max(window_start, event_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_stage = stage

    return best_stage


def label_windows(window_starts, window_ends, events, min_overlap_seconds=None):
    if min_overlap_seconds is None:
        min_overlap_seconds = config.MIN_APNEA_OVERLAP_SECONDS

    labels = []
    for start, end in zip(window_starts, window_ends):
        label = 0
        for event_start, event_end in events:
            overlap = min(end, event_end) - max(start, event_start)
            if overlap >= min_overlap_seconds:
                label = 1
                break
        labels.append(label)
    return np.asarray(labels, dtype=int)
