"""
End-to-end preprocessing for a single source record: decode -> detect ->
track -> align -> write crops to disk -> return processed manifest rows.

This is the piece the CLI (and later, dataset training code) calls per file.
"""
import csv
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2

from ai_engine.config import PreprocessingConfig
from ai_engine.datasets.manifest import SourceRecord
from ai_engine.preprocessing.align import align_face
from ai_engine.preprocessing.face_detector import FaceDetector
from ai_engine.preprocessing.tracker import FaceTracker
from ai_engine.preprocessing.video_io import load_image, sample_frames
from ai_engine.utils.io_utils import ensure_dir, stable_id
from ai_engine.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedRecord:
    crop_id: str
    crop_path: str
    source_id: str
    source_file: str
    label: str
    split: str
    media_type: str
    frame_index: int
    track_id: int
    bbox: str            # "x,y,w,h" — kept as a string for simple CSV round-tripping


def process_source_record(
    record: SourceRecord,
    config: PreprocessingConfig,
    detector: FaceDetector,
) -> list[ProcessedRecord]:
    source_path = Path(record.file_path)
    out_dir = ensure_dir(config.processed_data_dir / record.split / record.label / record.source_id)

    processed: list[ProcessedRecord] = []

    if record.media_type == "video":
        frames = sample_frames(source_path, config.frames_per_video)
        if not frames:
            logger.warning(f"No frames extracted from {source_path}; skipping.")
            return processed

        tracker = FaceTracker(iou_threshold=config.iou_match_threshold, max_gap=config.max_track_gap)
        for sampled in frames:
            detections = detector.detect(sampled.image)
            assigned = tracker.update(sampled.frame_index, detections)
            for track_id, det in assigned.items():
                crop = align_face(sampled.image, det, config.aligned_size, config.face_margin)
                if crop is None:
                    continue
                processed.append(
                    _save_crop(crop, out_dir, record, sampled.frame_index, track_id, det.bbox)
                )
    else:
        image = load_image(source_path)
        if image is None:
            return processed
        detections = detector.detect(image)
        for track_id, det in enumerate(detections):
            crop = align_face(image, det, config.aligned_size, config.face_margin)
            if crop is None:
                continue
            processed.append(_save_crop(crop, out_dir, record, 0, track_id, det.bbox))

    if not processed:
        logger.warning(f"No faces detected in {source_path} (label={record.label}).")

    return processed


def _save_crop(crop, out_dir: Path, record: SourceRecord, frame_index: int, track_id: int, bbox) -> ProcessedRecord:
    crop_id = stable_id(record.source_id, str(frame_index), str(track_id))
    crop_filename = f"f{frame_index:06d}_t{track_id:03d}_{crop_id}.jpg"
    crop_path = out_dir / crop_filename
    cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return ProcessedRecord(
        crop_id=crop_id,
        crop_path=str(crop_path),
        source_id=record.source_id,
        source_file=record.file_path,
        label=record.label,
        split=record.split,
        media_type=record.media_type,
        frame_index=frame_index,
        track_id=track_id,
        bbox=",".join(str(v) for v in bbox),
    )


def run_pipeline(records: list[SourceRecord], config: PreprocessingConfig) -> list[ProcessedRecord]:
    detector = FaceDetector(min_face_size=config.min_face_size)
    all_processed: list[ProcessedRecord] = []

    for i, record in enumerate(records, start=1):
        logger.info(f"[{i}/{len(records)}] Processing {record.file_path} (label={record.label}, split={record.split})")
        try:
            all_processed.extend(process_source_record(record, config, detector))
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to process {record.file_path}: {exc}")

    _write_processed_manifest(all_processed, config.processed_manifest_path)
    return all_processed


def _write_processed_manifest(records: list[ProcessedRecord], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = list(asdict(records[0]).keys()) if records else [
        "crop_id", "crop_path", "source_id", "source_file", "label",
        "split", "media_type", "frame_index", "track_id", "bbox",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
    logger.info(f"Wrote processed manifest ({len(records)} face crops) to {path}")
