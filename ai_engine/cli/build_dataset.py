"""
CLI to build/process a deepfake-detection dataset from a raw directory.

Usage:
    python -m ai_engine.cli.build_dataset \\
        --raw-dir data/raw \\
        --processed-dir data/processed \\
        --frames-per-video 32

Expects data/raw/<real|fake>/<file>.{mp4,jpg,...}. See docstring in
ai_engine/datasets/manifest.py for the full expected layout and the
rationale behind the leakage-safe split.
"""
import argparse
from pathlib import Path

from ai_engine.config import PreprocessingConfig
from ai_engine.datasets.manifest import build_and_split
from ai_engine.preprocessing.pipeline import run_pipeline
from ai_engine.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deepfake-detection dataset from raw media.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--frames-per-video", type=int, default=32)
    parser.add_argument("--aligned-size", type=int, default=224)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only build+split the source manifest; skip face detection/cropping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = PreprocessingConfig(
        raw_data_dir=args.raw_dir,
        processed_data_dir=args.processed_dir,
        manifest_path=args.manifest_dir / "source_manifest.csv",
        processed_manifest_path=args.manifest_dir / "processed_manifest.csv",
        frames_per_video=args.frames_per_video,
        aligned_size=args.aligned_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.seed,
    )

    logger.info("Step 1/2: building source manifest...")
    records = build_and_split(config)

    if args.manifest_only:
        logger.info("`--manifest-only` set; skipping face detection/cropping.")
        return

    logger.info("Step 2/2: running detection/tracking/alignment pipeline...")
    processed = run_pipeline(records, config)
    logger.info(f"Done. {len(processed)} face crops written to {config.processed_data_dir}")


if __name__ == "__main__":
    main()
