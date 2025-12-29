"""Dataset Setup Guide - Public Person Detection Datasets"""

# Dataset Download and Setup Guide

## Overview

This project provides dataset preparation scripts for COCO dataset:

1. **download_coco.py** - Downloads COCO 2017 dataset and converts to YOLO format

## Dataset Options

| Dataset | Time | Data Size | Purpose |
|---------|------|-----------|---------|
| COCO val2017 | 30-60 min | 5K images (~1GB) | Recommended for testing |
| COCO train2017 | 2-4 hours | 118K images (~20GB) | Production training |

## Quick Start (Recommended ⭐)

Download COCO val2017 (5K images, ~1GB):

```bash
# Download COCO val2017
python scripts/download_coco.py --output data/coco --split val2017

# Train
python -m src.train --data data/coco/dataset.yaml --epochs 50

# Or optimize with RL
python -m src.train_rl --data data/coco/dataset.yaml --iterations 5
```

**Total Time**: 1-4 hours (depending on hardware)
**GPU**: Recommended for faster training

## Detailed Usage

### download_coco.py

Downloads COCO dataset and converts to YOLO format.

```bash
# Basic usage (recommended, downloads val2017)
python scripts/download_coco.py

# Custom output directory
python scripts/download_coco.py --output data/my_coco

# Download train2017 (approximately 118K images, needs more disk space)
python scripts/download_coco.py --split train2017

# Convert only existing data (skip download step)
python scripts/download_coco.py --convert-only
```

**Output Structure:**
```
data/coco/
├── images/
│   └── val2017/           # COCO images
├── labels/
│   └── val2017/           # YOLO format annotations (class x_center y_center width height)
├── annotations/           # Original COCO JSON
└── dataset.yaml           # YOLO dataset config
```

**Download Sizes:**
- val2017: ~1GB images + 250MB annotations
- train2017: ~20GB images + 850MB annotations

## Dataset Validation

Validate dataset setup:

```python
# Check dataset
from pathlib import Path
import yaml

# Load configuration
with open('data/coco/dataset.yaml') as f:
    config = yaml.safe_load(f)

# Check images
train_imgs = list(Path(config['path']) / config['train']).glob('*')
print(f"Train images: {len(train_imgs)}")

val_imgs = list(Path(config['path']) / config['val']).glob('*')
print(f"Val images: {len(val_imgs)}")

# Check annotations
train_labels = list(Path(config['path']) / 'labels' / 'train' / '*.txt')
print(f"Train labels: {len(train_labels)}")
```

## Training Commands

```bash
# Standard training (100 epochs)
python -m src.train \
    --data data/coco/dataset.yaml \
    --epochs 100

# RL optimization (recommended)
python -m src.train_rl \
    --data data/coco/dataset.yaml \
    --iterations 10
```

## FAQ

### Q: Slow network, how to speed up download?
A: Download images only:
```bash
python scripts/download_coco.py --split val2017
```

### Q: Limited disk space?
A: Use val2017 (5K images, ~1GB) instead of train2017 (118K images, ~20GB)

### Q: Support for other datasets?
A: Manual conversion supported - just follow this format:
```
your_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/  (YOLO format)
│   ├── train/
│   └── val/
└── dataset.yaml
```

## Recommended Workflows

### First Time (Quick Validation)
```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download test data
python scripts/download_coco.py --output data/coco --split val2017

# 4. Quick training validation
python -m src.train --data data/coco/dataset.yaml --epochs 10

# 5. Check results
ls results/train/
```

### Stage Two (Performance Optimization)
```bash
# 1. RL hyperparameter optimization
python -m src.train_rl --data data/coco/dataset.yaml --iterations 10

# 2. View best hyperparameters
cat results/rl/rl_optimization_results.json

# 3. Train final model with optimal hyperparameters
python -m src.train --data data/coco/dataset.yaml --epochs 100
```

## Performance Expectations

| Dataset | Model | mAP@50 | Training Time |
|---------|-------|--------|---------------|
| 50 images | yolo11n | 0.45-0.55 | 10 minutes |
| 50 images + RL | yolo11n | 0.55-0.65 | 20 minutes |
| 5K images | yolo11m | 0.65-0.75 | 2 hours |
| 5K images + RL | yolo11m | 0.75-0.85 | 4 hours |

## File Structure

```
scripts/
└── download_coco.py       # Download COCO dataset

data/
├── coco/                  # COCO dataset
└── sample/                # Project samples
```

## Next Steps

1. ✅ Prepare dataset
2. ⬜ Run training
3. ⬜ Validate performance
4. ⬜ Deploy model

See [QUICK_START.md](../QUICK_START.md) for complete workflow.
