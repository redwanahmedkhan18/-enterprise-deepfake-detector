"""
Face alignment for the crop stage of the pipeline.

Implementation note: this does margin-padded bbox cropping + square resize,
not full landmark-based similarity-transform alignment (eyes-horizontal,
canonical landmark positions). That's a deliberate scope cut for Phase 2 —
it requires a landmark model (e.g. 5-point or 68-point) which is a separate
model dependency from the face *detector*. The interface below
(`align_face`) is the seam: swap its body for a landmark-based warp later
without touching any caller.
"""
import cv2
import numpy as np

from ai_engine.preprocessing.face_detector import FaceDetection


def _expand_bbox(det: FaceDetection, margin: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    pad_w = int(det.w * margin)
    pad_h = int(det.h * margin)

    x1 = max(0, det.x - pad_w)
    y1 = max(0, det.y - pad_h)
    x2 = min(img_w, det.x + det.w + pad_w)
    y2 = min(img_h, det.y + det.h + pad_h)
    return x1, y1, x2, y2


def align_face(
    image_bgr: np.ndarray,
    detection: FaceDetection,
    output_size: int = 224,
    margin: float = 0.35,
) -> np.ndarray | None:
    """Crops the face region (with margin) and resizes to a square output_size x output_size.
    Returns None if the resulting crop is degenerate (shouldn't normally happen given
    upstream size filtering, but guards against edge-of-frame boxes)."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = _expand_bbox(detection, margin, w, h)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
