"""
Source manifest: one row per raw source file (image or video), before any
preprocessing. This is the ground truth of "what data do we have and what's
it labeled".

Expected raw layout (flexible: only top-level label folder name matters):

    data/raw/
        real/
            person01_clip01.mp4
            person02_photo01.jpg
            ...
        fake/
            person01_clip01_faceswap.mp4
            ...

Design decisions worth flagging:
- Splitting is done at the SOURCE level (one video -> one split), never at
  the frame level. Frame-level random splitting is a classic deepfake-
  detection leakage bug: frames from the same clip end up in both train and
  test, so the model partially memorizes identities/compression artifacts
  instead of learning to generalize.
- Splitting is also stratified by an inferred "identity/group" key when the
  filename encodes one (e.g. "person01_clip01.mp4" -> group "person01"), so
  the same subject doesn't appear in both train and test either. Falls back
  to per-file grouping if no pattern is detected.
"""
import csv
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from ai_engine.config import PreprocessingConfig
from ai_engine.utils.io_utils import ensure_dir, iter_files_with_extensions, stable_id
from ai_engine.utils.logging_utils import get_logger

logger = get_logger(__name__)

_GROUP_PATTERN = re.compile(r"^([a-zA-Z]+\d+)")  # e.g. "person01" from "person01_clip01.mp4"


@dataclass
class SourceRecord:
    source_id: str
    file_path: str
    label: str          # "real" | "fake"
    media_type: str      # "image" | "video"
    group_key: str        # identity/grouping key used for leakage-safe splitting
    split: str = ""       # filled in by split_manifest()


def _infer_group_key(file_path: Path) -> str:
    match = _GROUP_PATTERN.match(file_path.stem)
    return match.group(1) if match else file_path.stem


def build_source_manifest(config: PreprocessingConfig) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    if not config.raw_data_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {config.raw_data_dir}. "
            f"Expected subfolders like 'real/' and 'fake/' inside it."
        )

    label_dirs = [d for d in config.raw_data_dir.iterdir() if d.is_dir()]
    if not label_dirs:
        raise ValueError(f"No label subfolders found under {config.raw_data_dir}")

    for label_dir in label_dirs:
        label = label_dir.name.lower()
        all_extensions = config.image_extensions + config.video_extensions
        for file_path in iter_files_with_extensions(label_dir, all_extensions):
            media_type = "video" if file_path.suffix.lower() in config.video_extensions else "image"
            source_id = stable_id(str(file_path), label)
            group_key = _infer_group_key(file_path)
            records.append(
                SourceRecord(
                    source_id=source_id,
                    file_path=str(file_path),
                    label=label,
                    media_type=media_type,
                    group_key=f"{label}::{group_key}",
                )
            )

    logger.info(f"Discovered {len(records)} source files across {len(label_dirs)} label(s).")
    return records


def split_manifest(records: list[SourceRecord], config: PreprocessingConfig) -> list[SourceRecord]:
    """Deterministic, group-aware split. Same group_key always lands in the same split."""
    import random

    groups: dict[str, list[SourceRecord]] = {}
    for r in records:
        groups.setdefault(r.group_key, []).append(r)

    group_keys = sorted(groups.keys())
    rng = random.Random(config.split_seed)
    rng.shuffle(group_keys)

    n = len(group_keys)
    n_train = int(n * config.train_ratio)
    n_val = int(n * config.val_ratio)

    train_keys = set(group_keys[:n_train])
    val_keys = set(group_keys[n_train:n_train + n_val])
    test_keys = set(group_keys[n_train + n_val:])

    for key, group_records in groups.items():
        split = "train" if key in train_keys else "val" if key in val_keys else "test"
        for r in group_records:
            r.split = split

    counts = {s: sum(1 for r in records if r.split == s) for s in ("train", "val", "test")}
    logger.info(f"Split {n} group(s) into {len(train_keys)}/{len(val_keys)}/{len(test_keys)} "
                f"(train/val/test groups) -> {counts} source files.")
    return records


def write_manifest(records: list[SourceRecord], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = list(asdict(records[0]).keys()) if records else [
        "source_id", "file_path", "label", "media_type", "group_key", "split"
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
    logger.info(f"Wrote source manifest ({len(records)} rows) to {path}")


def load_manifest(path: Path) -> list[SourceRecord]:
    with open(path, newline="") as f:
        return [SourceRecord(**row) for row in csv.DictReader(f)]


def build_and_split(config: PreprocessingConfig) -> list[SourceRecord]:
    records = build_source_manifest(config)
    records = split_manifest(records, config)
    write_manifest(records, config.manifest_path)
    return records
