# AI Engine — Phase 2: Dataset Management & Preprocessing

Turns a raw folder of real/fake videos and images into a leakage-safe,
per-face-crop dataset ready for the spatial/temporal/frequency/physiological
branches (Phase 3) to train on.

**This has been run end-to-end and tested, not just written.** See "Verified"
below.

## Pipeline

```
raw/real/*, raw/fake/*
        │
        ▼
build_source_manifest()      — discovers files, infers label + identity group
        │
        ▼
split_manifest()               — group-aware train/val/test split (see below)
        │
        ▼
run_pipeline() per file:
    sample_frames()             — evenly-spaced frame sampling (video only)
    FaceDetector.detect()        — pluggable backend, Haar cascade by default
    FaceTracker.update()          — IoU tracking across sampled frames
    align_face()                   — margin crop + resize to aligned_size²
    write crop + manifest row
        │
        ▼
processed_manifest.csv   +   processed/<split>/<label>/<source_id>/*.jpg
```

## Why the split isn't `random.shuffle`

Frame-level or naive random splitting is the most common bug in deepfake-
detection pipelines: frames from the same clip (or the same identity) leak
across train/val/test, so the model partially memorizes identities or
compression artifacts instead of learning to generalize, and your eval
numbers lie to you. `manifest.py` splits at the **group level** (inferred
identity from filename, falling back to per-file), deterministically, and
`stats.py`'s `leakage_check` verifies no group spans multiple splits.

## Face detection is a pluggable seam

`face_detector.py` defines a `FaceDetectorBackend` interface. The default,
`HaarCascadeFaceDetector`, ships inside `opencv-python` — zero model
downloads, works anywhere. It's good enough to build and test the pipeline
end-to-end (which is what we did), but it misses profile faces, low light,
and occlusion. Before training a production model, swap in a stronger
backend (RetinaFace, MTCNN, YOLO-face) by implementing `detect()` — nothing
else in the pipeline changes.

Same pattern in `align.py`: it currently does bbox-margin crop + resize, not
landmark-based similarity-transform alignment (eyes horizontal, canonical
landmark positions), because that needs a separate landmark model. The
`align_face()` function is the seam to swap in later.

## Verified

Ran the full CLI (`python -m ai_engine.cli.build_dataset`) against a
synthetic fixture dataset (8 videos + 2 images, drawn faces, 4 fake/4 real
identities) in this environment:

- Discovered all 18 source files, split into leakage-safe train/val/test
- Haar cascade actually detected the synthetic faces and produced 162 aligned crops
- `stats.py`'s leakage check: **PASS**
- Visually confirmed an output crop is a correctly cropped, aligned 224×224 face image
- Full pytest suite (`ai_engine/tests/test_preprocessing.py`): **7/7 passed**
  — manifest discovery, split leakage-safety, split determinism, tracker
  behavior (same-track continuity, separate tracks, track expiry), and the
  full pipeline end-to-end

Test fixtures are generated in code (`tests/conftest.py`), not committed as
binary files — run `pytest ai_engine/tests/` and it builds its own tiny
synthetic dataset on the fly.

## Usage

```bash
pip install -r ai_engine/requirements.txt   # opencv + numpy sufficient for the CLI below

python -m ai_engine.cli.build_dataset \
    --raw-dir data/raw \
    --processed-dir data/processed \
    --manifest-dir data/manifests \
    --frames-per-video 32
```

Expected `data/raw/` layout:

```
data/raw/
    real/
        person01_clip01.mp4
        person02_photo01.jpg
    fake/
        person01_clip01_faceswap.mp4
```

Check the result:

```python
from pathlib import Path
from ai_engine.datasets.stats import print_report
print_report(Path("data/manifests/processed_manifest.csv"))
```

## Consuming the output (Phase 3 will do this)

```python
from ai_engine.datasets.dataset import FaceCropDataset, FaceTrackClipDataset
from ai_engine.datasets.augmentations import build_train_transform
from ai_engine.config import AugmentationConfig

# Spatial / frequency branches: single aligned face crops
train_ds = FaceCropDataset(
    "data/manifests/processed_manifest.csv", split="train",
    transform=build_train_transform(AugmentationConfig()),
)

# Temporal branch: fixed-length clips of the same tracked face
clip_ds = FaceTrackClipDataset("data/manifests/processed_manifest.csv", split="train", clip_len=8)
```

(`torch`/`albumentations` are only required for this dataset-consumption
layer, not for manifest building or preprocessing — kept as lazy/module-local
imports so the CLI stays lightweight.)

## What's NOT in Phase 2

Model training, the fusion network, and the physiological-signal extractors
(blink/rPPG/head-pose) are Phase 3/4. This phase hands them clean, labeled,
leakage-safe face crops and tracks — that's the contract.
