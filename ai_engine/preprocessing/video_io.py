"""
Video decoding via OpenCV. Kept minimal and dependency-light (cv2 + numpy
only) since this runs on every raw video in the dataset and needs to be fast
and robust to slightly malformed files (common in scraped datasets).
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ai_engine.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SampledFrame:
    frame_index: int          # index in the original video
    timestamp_sec: float
    image: np.ndarray          # BGR, HxWx3


def sample_frames(video_path: Path, n_frames: int) -> list[SampledFrame]:
    """Evenly samples up to n_frames frames across the full duration of the video.

    Even spacing (rather than e.g. the first N frames) matters for deepfake
    detection: swap artifacts, blending seams, and physiological signals like
    blink rate are not uniform across a clip, so sampling only the start
    biases the dataset.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path}")
        return []

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        if total_frames <= 0:
            logger.warning(f"Video reports 0 frames, skipping: {video_path}")
            return []

        n_to_sample = min(n_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, num=n_to_sample, dtype=int)
        indices = sorted(set(indices.tolist()))  # dedupe for very short videos

        sampled: list[SampledFrame] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            sampled.append(SampledFrame(frame_index=idx, timestamp_sec=idx / fps, image=frame))

        return sampled
    finally:
        cap.release()


def load_image(image_path: Path) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning(f"Could not read image: {image_path}")
    return image
