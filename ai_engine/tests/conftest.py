"""
Generates a small synthetic "raw dataset" (crude drawn faces, not real
people/footage) once per test session, instead of committing binary video
fixtures to the repo. This keeps the repo text-only and makes the fixture's
structure legible instead of opaque.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixture_raw"


def _make_face_frame(cx: int, cy: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 60, (240, 320, 3), dtype=np.uint8)
    cv2.circle(img, (cx, cy), 60, (200, 180, 160), -1)
    cv2.circle(img, (cx - 20, cy - 15), 8, (20, 20, 20), -1)
    cv2.circle(img, (cx + 20, cy - 15), 8, (20, 20, 20), -1)
    cv2.ellipse(img, (cx, cy + 25), (25, 10), 0, 0, 180, (60, 40, 40), 2)
    return img


def _make_video(path: Path, n_frames: int = 20) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (320, 240))
    for i in range(n_frames):
        cx = 160 + int(10 * np.sin(i / 3))
        writer.write(_make_face_frame(cx, 120, seed=i))
    writer.release()


@pytest.fixture(scope="session", autouse=True)
def fixture_raw_dataset():
    (FIXTURE_DIR / "real").mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / "fake").mkdir(parents=True, exist_ok=True)

    for person in range(4):
        for clip in range(2):
            _make_video(FIXTURE_DIR / "real" / f"person{person:02d}_clip{clip:02d}.mp4")
            _make_video(FIXTURE_DIR / "fake" / f"person{person:02d}_clip{clip:02d}_swap.mp4")

    for person in range(2):
        img = _make_face_frame(160, 120, seed=100 + person)
        cv2.imwrite(str(FIXTURE_DIR / "real" / f"person{person:02d}_photo.jpg"), img)

    yield FIXTURE_DIR
