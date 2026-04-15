# video-keyframe-extract

A lightweight, opinionated keyframe extraction pipeline for short-form videos.

Designed for high-throughput scenarios (e.g. street-view clips), it combines scene cut detection and CLIP-based semantic filtering to extract representative frames.

## Features

- **Hard-cut detection**: Detects scene boundaries using PySceneDetect (content-based)
- **Semantic-diff sampling**: Uses CLIP ViT-B/32 to capture frames when visual semantics change
- **Max-interval fallback**: Ensures coverage by forcing a capture after a configurable interval
- **Per-video cap**: Prevents over-extraction by limiting total captures per video

## Requirements

- Docker (with NVIDIA GPU support recommended)
- NVIDIA GPU (tested on RTX 3060 12GB; CPU fallback is supported but slower)

## Quick Start

```bash
# Build
docker build -t video-extract:latest .

# Small batch test (first 5 videos)
docker run --rm -it --gpus=all --ipc=host -v "${PWD}/:/app" video-extract:latest \
    python extract.py --limit 5

# Full run
docker run --rm -it --gpus=all --ipc=host -v "${PWD}/:/app" video-extract:latest \
    python extract.py
```

## Directory Structure

```
├── Dockerfile
├── requirements.txt
├── extract.py          # Main script
├── video/              # Input videos
└── output/             # Output images ({video_index}_{image_index}.jpg)
```

## Output Rules

- Videos are sorted alphabetically by filename; indices are 3-digit zero-padded (`001`, `002`, ...)
- Image indices are 3-digit zero-padded (`001_001.jpg`, `001_002.jpg`, ...)

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cut-threshold` | `27.0` | PySceneDetect hard-cut sensitivity (lower = more sensitive) |
| `--semantic-threshold` | `0.12` | CLIP semantic distance threshold (lower = captures more easily) |
| `--sample-fps` | `2.0` | Sampling rate (frames per second) |
| `--max-interval` | `6.0` | Max-interval fallback (seconds); force capture if exceeded |
| `--max-captures` | `15` | Max captures per video |
| `--jpeg-quality` | `95` | Output JPEG quality |
| `--start-index` | `1` | Starting video index number (for incremental updates) |
| `--limit` | `0` | Only process first N videos (0 = all) |

### Incremental Updates

When adding new videos to an existing batch, use `--start-index` to continue numbering from where you left off:

```bash
# First batch: 50 videos → output 001_*.jpg ~ 050_*.jpg
python extract.py

# Add more videos, continue from index 51
python extract.py --start-index 51
```

### Tuning Examples

```bash
# Too few captures -> lower semantic threshold
python extract.py --semantic-threshold 0.08

# Too many captures -> raise semantic threshold
python extract.py --semantic-threshold 0.18

# Single-take video missing frames -> shorten max interval
python extract.py --max-interval 3.0
```
