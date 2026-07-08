"""
Lightweight IoU tracker: associates face detections across consecutive
sampled frames of the same video so the temporal branch can later pull
"track A, frames 3-10" as a coherent clip of the same face, and so the
physiological branch (blink rate, head pose over time) has a consistent
identity to analyze rather than independent per-frame detections.

Not a full Kalman/SORT implementation on purpose — frames are sparsely
sampled (not every frame of the video), so a lightweight greedy IoU match
per sampled frame is appropriate; swap in SORT/DeepSORT if you later sample
densely (e.g. every frame) for finer-grained temporal modeling.
"""
from dataclasses import dataclass, field

from ai_engine.preprocessing.face_detector import FaceDetection


def _iou(a: FaceDetection, b: FaceDetection) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h

    inter_x1, inter_y1 = max(a.x, b.x), max(a.y, b.y)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    union_area = a.area() + b.area() - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


@dataclass
class Track:
    track_id: int
    last_detection: FaceDetection
    last_frame_index: int
    frames_since_seen: int = 0
    history: list[tuple[int, FaceDetection]] = field(default_factory=list)  # (frame_index, detection)


class FaceTracker:
    def __init__(self, iou_threshold: float = 0.3, max_gap: int = 5):
        self.iou_threshold = iou_threshold
        self.max_gap = max_gap
        self._tracks: list[Track] = []
        self._next_id = 0

    def update(self, frame_index: int, detections: list[FaceDetection]) -> dict[int, FaceDetection]:
        """Call once per sampled frame, in increasing frame_index order.
        Returns {track_id: detection} for detections matched (or newly created) this frame."""
        assigned: dict[int, FaceDetection] = {}
        unmatched_detections = list(detections)

        # Greedy match: for each live track, find the best remaining IoU match.
        for track in self._tracks:
            if not unmatched_detections:
                break
            best_iou, best_det = 0.0, None
            for det in unmatched_detections:
                score = _iou(track.last_detection, det)
                if score > best_iou:
                    best_iou, best_det = score, det

            if best_det is not None and best_iou >= self.iou_threshold:
                track.last_detection = best_det
                track.last_frame_index = frame_index
                track.frames_since_seen = 0
                track.history.append((frame_index, best_det))
                assigned[track.track_id] = best_det
                unmatched_detections.remove(best_det)

        # Any detection that didn't match an existing track starts a new one.
        for det in unmatched_detections:
            track = Track(track_id=self._next_id, last_detection=det, last_frame_index=frame_index)
            track.history.append((frame_index, det))
            self._tracks.append(track)
            assigned[track.track_id] = det
            self._next_id += 1

        # Age out tracks that haven't been seen recently.
        for track in self._tracks:
            if track.last_frame_index != frame_index:
                track.frames_since_seen += 1
        self._tracks = [t for t in self._tracks if t.frames_since_seen <= self.max_gap]

        return assigned

    def all_tracks(self) -> list[Track]:
        return self._tracks
