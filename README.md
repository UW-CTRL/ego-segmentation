# Egocentric Segmentation

Egocentric Segmentation is a Python package for gaze-guided object segmentation in first-person video.
It combines:

- YOLOv5 for object detection
- SAM2 for prompt-based video segmentation and propagation
- gaze coordinates (from NPZ or CSV) to trigger segmentation targets

The main entry point is `run_segmentation_pipeline`, which returns:

- frame-level object metadata (`objects_key`)
- frame-level propagated masks (`video_segments`)

## Features

- Device-aware setup (CUDA, MPS, or CPU)
- Automatic model loading (SAM2 + YOLOv5)
- Support for two input modes:
  - NPZ mode: scene frames + gaze in one file
  - Video mode: video file + gaze CSV
- Per-frame object lifecycle tracking (first seen / last seen)
- Temporal cleanup for stale objects via `elapse_threshold`

## Project Structure

```text
.
├── egosegmentation/
│   ├── __init__.py
│   ├── core.py
│   ├── models.py
│   └── utils.py
├── requirements.txt
└── setup.py
```

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd ego-segmentation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Install as a package

```bash
pip install -e .
```

## Requirements

Core dependencies include:

- numpy
- torch
- Pillow
- matplotlib
- opencv-python
- sam2

See `requirements.txt` for the full list.

## Quick Start

```python
from egosegmentation.core import run_segmentation_pipeline

objects_key, video_segments = run_segmentation_pipeline(
    sam2_checkpoint="/path/to/sam2_checkpoint.pt",
    model_cfg="/path/to/model_config.yaml",
    detection_threshold=0.3,
    elapse_threshold=10,
    data_npz_path="/path/to/data.npz",   # Use NPZ mode
    video_path=None,
    csv_path=None,
)
```

## Input Modes

### Mode A: NPZ input

Provide `data_npz_path` and keep `video_path` and `csv_path` as `None`.

Expected NPZ keys:

- `scene`: array of frames (`H x W x C` per frame)
- `gaze`: array of gaze coordinates (`N x 2`, columns = x, y)

### Mode B: Video + CSV input

Set `data_npz_path=None` and provide both:

- `video_path`: input video file
- `csv_path`: gaze file with two columns per row: `x,y`

## Output Format

### `objects_key`

Dictionary keyed by frame index, then object id. Typical fields:

- `First Seen (Time)`
- `First Seen (Frame)`
- `Last Seen (Time)`
- `Last Seen (Frame)`
- `object_type`
- `BBox` as `(x, y, w, h)`
- `Mask` (when available)

### `video_segments`

Dictionary keyed by frame index, then object id, containing segmentation masks (NumPy arrays).

## Core API

### `run_segmentation_pipeline(...)`

Runs the full workflow:

1. set up device and models
2. load frames and gaze
3. detect objects and match to gaze
4. add SAM2 prompts and propagate masks
5. return metadata and segmentation maps

Main arguments:

- `sam2_checkpoint`: path to SAM2 weights
- `model_cfg`: SAM2 model config
- `detection_threshold`: IoU threshold for gaze-object association
- `elapse_threshold`: frame count before stale object removal
- `data_npz_path` or (`video_path` + `csv_path`)

## Notes

- YOLO class filtering is currently set to person and common animal classes.
- The confidence threshold is set to `0.90` in model setup.
- Temporary extracted frame files are automatically cleaned up at the end of pipeline execution.

## Troubleshooting

- Verify PyTorch device support (`cuda`, `mps`, or `cpu`) is available in your environment.
- Ensure SAM2 checkpoints/config paths are correct.
- Ensure gaze coordinates align with frame resolution.
- If running in a clean environment, install all dependencies from `requirements.txt` first.

## License

Add your project license information here.
