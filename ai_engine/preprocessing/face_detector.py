"""
Face detection, behind a small backend interface so this can start with a
zero-dependency default (OpenCV's bundled Haar cascade — no model download,
works anywhere cv2 is installed) and later swap in a stronger detector
(RetinaFace, MTCNN, YOLO-face) for production accuracy without touching
any calling code.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from ai_engine.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class FaceDetection:
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def area(self) -> int:
        return self.w * self.h


class FaceDetectorBackend(ABC):
    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        """Returns all faces detected in a single BGR image."""
        raise NotImplementedError


class HaarCascadeFaceDetector(FaceDetectorBackend):
    """Default backend. Ships inside opencv-python, no download required.
    Good enough for pipeline development/testing; swap for RetinaFaceDetector
    or similar before training a production model — Haar cascades miss
    profile faces, low light, and small/occluded faces that matter for
    deepfake robustness."""

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5, min_size: int = 30):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # improves detection under uneven lighting
        boxes = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size),
        )
        return [FaceDetection(x=int(x), y=int(y), w=int(w), h=int(h)) for (x, y, w, h) in boxes]


class FaceDetector:
    """Facade used by the rest of the pipeline. Filters out detections below
    the configured minimum size so tiny false-positive boxes don't pollute
    the dataset."""

    def __init__(self, backend: FaceDetectorBackend | None = None, min_face_size: int = 40):
        self.backend = backend or HaarCascadeFaceDetector()
        self.min_face_size = min_face_size

    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        detections = self.backend.detect(image_bgr)
        return [d for d in detections if min(d.w, d.h) >= self.min_face_size]
