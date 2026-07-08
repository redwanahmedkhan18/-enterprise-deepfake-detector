"""
Configuration for dataset management & preprocessing.

Kept dependency-free (stdlib dataclasses) so this module can be imported by
CLI scripts, tests, and the eventual training code without pulling in heavy
ML libraries just to read a config value.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PreprocessingConfig:
    # --- Sampling ---
    frames_per_video: int = 32          # how many frames to sample per source video
    min_face_size: int = 40             # px, skip detections smaller than this (likely noise)
    face_margin: float = 0.35           # fraction of bbox size added as crop padding on each side
    aligned_size: int = 224             # output crop is a square aligned_size x aligned_size image

    # --- Tracking ---
    max_track_gap: int = 5              # frames a track can go undetected before being closed
    iou_match_threshold: float = 0.3    # min IoU to associate a detection with an existing track

    # --- Splitting ---
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42

    # --- I/O ---
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    manifest_path: Path = Path("data/manifests/source_manifest.csv")
    processed_manifest_path: Path = Path("data/manifests/processed_manifest.csv")

    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    video_extensions: tuple = (".mp4", ".mov", ".avi", ".mkv", ".webm")

    def __post_init__(self):
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, (
            "train/val/test ratios must sum to 1.0"
        )
        self.raw_data_dir = Path(self.raw_data_dir)
        self.processed_data_dir = Path(self.processed_data_dir)
        self.manifest_path = Path(self.manifest_path)
        self.processed_manifest_path = Path(self.processed_manifest_path)


@dataclass
class AugmentationConfig:
    """Perturbations that simulate real-world degradation deepfakes travel through
    (social media re-compression, resizing, etc.) so the classifier doesn't
    overfit to pristine-source artifacts."""
    horizontal_flip_p: float = 0.5
    jpeg_compression_p: float = 0.3
    jpeg_quality_range: tuple = (40, 90)
    gaussian_blur_p: float = 0.2
    brightness_contrast_p: float = 0.3
    downscale_p: float = 0.2
    downscale_range: tuple = (0.5, 0.9)
