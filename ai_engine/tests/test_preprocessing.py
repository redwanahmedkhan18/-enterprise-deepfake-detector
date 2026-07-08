"""
Tests for Phase 2 (dataset management & preprocessing). Uses the synthetic
fixture data under ai_engine/tests/fixture_raw (generated deterministically,
not committed as binary — see conftest.py) so these run without any real
face dataset.

Covers: manifest building, leakage-safe splitting, the IoU tracker in
isolation, and the full pipeline end-to-end.
"""
from pathlib import Path

import pytest

from ai_engine.config import PreprocessingConfig
from ai_engine.datasets.manifest import build_and_split
from ai_engine.datasets.stats import summarize
from ai_engine.preprocessing.face_detector import FaceDetection
from ai_engine.preprocessing.pipeline import run_pipeline
from ai_engine.preprocessing.tracker import FaceTracker

FIXTURE_RAW = Path(__file__).parent / "fixture_raw"


@pytest.fixture(scope="module")
def config(tmp_path_factory) -> PreprocessingConfig:
    out = tmp_path_factory.mktemp("dataset_out")
    return PreprocessingConfig(
        raw_data_dir=FIXTURE_RAW,
        processed_data_dir=out / "processed",
        manifest_path=out / "manifests" / "source_manifest.csv",
        processed_manifest_path=out / "manifests" / "processed_manifest.csv",
        frames_per_video=8,
    )


def test_manifest_discovers_all_fixture_files(config):
    records = build_and_split(config)
    assert len(records) == 18  # 16 videos + 2 photos, from fixture generation
    assert all(r.split in ("train", "val", "test") for r in records)


def test_split_is_group_leakage_safe(config):
    records = build_and_split(config)
    group_to_splits: dict[str, set] = {}
    for r in records:
        group_to_splits.setdefault(r.group_key, set()).add(r.split)
    leaked = {g: s for g, s in group_to_splits.items() if len(s) > 1}
    assert not leaked, f"Groups spanning multiple splits: {leaked}"


def test_split_is_deterministic(config):
    records_a = build_and_split(config)
    records_b = build_and_split(config)
    splits_a = {r.source_id: r.split for r in records_a}
    splits_b = {r.source_id: r.split for r in records_b}
    assert splits_a == splits_b


class TestFaceTracker:
    def test_same_position_stays_one_track(self):
        tracker = FaceTracker(iou_threshold=0.3, max_gap=5)
        det = FaceDetection(x=100, y=100, w=50, h=50)

        r0 = tracker.update(0, [det])
        r1 = tracker.update(1, [det])

        assert list(r0.keys()) == list(r1.keys()), "Same bbox across frames should keep the same track id"

    def test_far_apart_detections_become_separate_tracks(self):
        tracker = FaceTracker(iou_threshold=0.3, max_gap=5)
        det_a = FaceDetection(x=0, y=0, w=50, h=50)
        det_b = FaceDetection(x=500, y=500, w=50, h=50)

        result = tracker.update(0, [det_a, det_b])
        assert len(result) == 2
        assert len(set(result.keys())) == 2

    def test_track_expires_after_max_gap(self):
        tracker = FaceTracker(iou_threshold=0.3, max_gap=2)
        det = FaceDetection(x=100, y=100, w=50, h=50)
        tracker.update(0, [det])

        for frame_idx in range(1, 6):  # gap of several frames, no detections
            tracker.update(frame_idx, [])

        assert len(tracker.all_tracks()) == 0, "Track should have expired after max_gap frames unseen"


def test_full_pipeline_end_to_end(config):
    records = build_and_split(config)
    processed = run_pipeline(records, config)

    assert len(processed) > 0, "Pipeline should detect at least some faces in the synthetic fixture"
    assert all(Path(p.crop_path).exists() for p in processed), "Every manifest row must have a real file on disk"
    assert all(p.label in ("real", "fake") for p in processed)

    report = summarize(config.processed_manifest_path)
    assert report["leakage_check"] == "PASS"
