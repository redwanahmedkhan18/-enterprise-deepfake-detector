"""
PyTorch Dataset classes that read the *processed* manifest (output of
ai_engine.preprocessing.pipeline) and serve tensors to the spatial and
temporal branches described in the architecture doc.

torch/cv2 are imported at module scope here since these classes are useless
without them anyway (unlike manifest.py, which stays lightweight for CLI/
tooling use).
"""
import csv
from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

LABEL_TO_IDX = {"real": 0, "fake": 1}


def _load_processed_manifest(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class FaceCropDataset(Dataset):
    """Spatial branch: one aligned face crop per sample.
    Also the natural dataset for the frequency branch (FFT/DCT/wavelet),
    since those operate on single images too — same crops, different
    downstream transform in the model, not in this Dataset."""

    def __init__(self, manifest_path: Path, split: str, transform: Callable | None = None):
        rows = _load_processed_manifest(manifest_path)
        self.rows = [r for r in rows if r["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows found for split='{split}' in {manifest_path}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = cv2.imread(row["crop_path"])
        if image is None:
            raise FileNotFoundError(f"Missing processed crop: {row['crop_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label = LABEL_TO_IDX[row["label"]]
        return image, torch.tensor(label, dtype=torch.long)


class FaceTrackClipDataset(Dataset):
    """Temporal branch: a fixed-length clip of consecutive frames from the
    same tracked face (same source video + track_id), ordered by frame_index.
    This is what VideoMAE/TimeSformer-style models in the architecture doc
    consume.

    Tracks shorter than clip_len are handled by looping (repeating frames),
    which is a simple, common strategy — swap for padding + attention mask
    if the temporal model should distinguish real vs. padded frames."""

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        clip_len: int = 8,
        transform: Callable | None = None,
    ):
        rows = _load_processed_manifest(manifest_path)
        rows = [r for r in rows if r["split"] == split]
        if not rows:
            raise ValueError(f"No rows found for split='{split}' in {manifest_path}")

        self.clip_len = clip_len
        self.transform = transform

        # Group into (source_id, track_id) -> sorted frames
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in rows:
            groups[(r["source_id"], r["track_id"])].append(r)

        self.tracks: list[list[dict]] = []
        for group_rows in groups.values():
            group_rows.sort(key=lambda r: int(r["frame_index"]))
            self.tracks.append(group_rows)

        if not self.tracks:
            raise ValueError(f"No tracks found for split='{split}' in {manifest_path}")

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, idx: int):
        track = self.tracks[idx]

        # Loop frames to reach clip_len if the track is shorter than requested.
        indices = [i % len(track) for i in range(self.clip_len)]
        frames = []
        for i in indices:
            row = track[i]
            image = cv2.imread(row["crop_path"])
            if image is None:
                raise FileNotFoundError(f"Missing processed crop: {row['crop_path']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform is not None:
                image = self.transform(image=image)["image"]
            else:
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            frames.append(image)

        clip = torch.stack(frames, dim=0)  # (T, C, H, W)
        label = LABEL_TO_IDX[track[0]["label"]]
        return clip, torch.tensor(label, dtype=torch.long)
