# Multi-Task Passenger Detection

Unified framework for person detection + 5 attribute classification:
- **Gender**: male, female, other
- **Age Group**: 0-2, 3-5, 6-12, 13-18, 19-30, 31-45, 46-60, 60+
- **Height Range**: very_short, short, average, tall, very_tall
- **BMI Category**: underweight, normal, overweight, obese
- **Clothing**: casual, formal, sports, traditional, work, summer, winter, other, unknown, multiple

## Core Modules

| Module | Purpose |
|--------|---------|
| `src/models/multitask_model.py` | Model architecture |
| `src/utils/multitask_data.py` | Data loader |
| `src/train_multitask.py` | Training script |
| `src/inference_multitask.py` | Inference engine |

## Quick Start

### Data Format

Create `annotations.json` with structure:

```json
{
  "image.jpg": {
    "boxes": [[x1, y1, x2, y2]],
    "genders": ["male"],
    "age_groups": ["25-30"],
    "height_ranges": ["average"],
    "bmi_categories": ["normal"],
    "clothings": ["casual"]
  }
}
```

### 1. Validate Data

```bash
python scripts/data_converter.py validate \
    --annotations data/train/annotations.json \
    --images data/train/images
```

### 2. Check Dataset Stats

```bash
python scripts/data_converter.py summary \
    --annotations data/train/annotations.json \
    --images data/train/images
```

### 3. Train Model

```bash
python -m src.train_multitask \
    --train-images data/train/images \
    --train-annotations data/train/annotations.json \
    --val-images data/val/images \
    --val-annotations data/val/annotations.json \
    --epochs 50 --batch-size 16 --device cuda:0
```

Training outputs to: `results/multitask/checkpoints/`

### 4. Run Inference

```bash
# Single image
python -m src.inference_multitask \
    --checkpoint results/multitask/checkpoints/best_model.pt \
    --image test.jpg --visualize

# Batch
python -m src.inference_multitask \
    --checkpoint results/multitask/checkpoints/best_model.pt \
    --image-dir test_images/ --output predictions.json
```

## Python API

### Data Loading
```python
from src.utils.multitask_data import PassengerAttributeDataset

dataset = PassengerAttributeDataset(
    image_dir="data/train/images",
    annotations_file="data/train/annotations.json"
)
sample = dataset[0]
```

### Training
```python
from src.train_multitask import MultiTaskTrainer

trainer = MultiTaskTrainer(output_dir="results/multitask")
trainer.train(
    train_image_dir="data/train/images",
    train_annotations="data/train/annotations.json",
    val_image_dir="data/val/images",
    val_annotations="data/val/annotations.json"
)
```

### Inference
```python
from src.inference_multitask import MultiTaskInferenceEngine
import json

engine = MultiTaskInferenceEngine("checkpoint.pt")
results = engine.infer("image.jpg")
print(json.dumps(results, indent=2))
```

## Output Format

```json
{
  "image_path": "test.jpg",
  "original_size": [1920, 1080],
  "detections": [
    {
      "box": {"x1": 100, "y1": 150, "x2": 300, "y2": 450},
      "detection_confidence": 0.95,
      "attributes": {
        "gender": {"class": "male", "confidence": 0.92},
        "age_group": {"class": "25-30", "confidence": 0.87},
        "height_range": {"class": "average", "confidence": 0.89},
        "bmi_category": {"class": "normal", "confidence": 0.84},
        "clothing": {"class": "casual", "confidence": 0.91}
      }
    }
  ]
}
```

## Model Details

**Architecture**: YOLO11m + 5 parallel classification heads

**Loss**: 50% detection + 10% each attribute

**Config** (src/config.py):
```python
TRAIN_CONFIG = {
    "epochs": 100,
    "batch_size": 16,
    "lr0": 0.01,
    "momentum": 0.937,
    "weight_decay": 5e-4,
}
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM | Reduce batch size: `--batch-size 8` |
| Low accuracy | Increase epochs: `--epochs 100` or lower LR: `--lr 0.0001` |
| Import errors | Run from project root: `cd d:/Github/EV-PassengerDection-RL` |
| Data format errors | Validate with: `python scripts/data_converter.py validate ...` |

## Directory Structure

```
data/
├── train/
│   ├── images/
│   └── annotations.json
├── val/
│   ├── images/
│   └── annotations.json

results/
└── multitask/
    ├── checkpoints/
    │   ├── best_model.pt
    │   └── checkpoint_*.pt
    └── logs/
```

## Examples

Run all examples:
```bash
python examples_multitask.py
```

Shows:
1. Data loading
2. DataLoader batching
3. Model creation
4. Loss function setup
5. Training workflow
6. Inference setup
7. Data validation
8. Format conversion

## Next Steps

1. ✅ Framework ready
2. 📊 Prepare your labeled data
3. ✔️ Validate format with `data_converter.py`
4. 🚀 Train with `train_multitask.py`
5. 🎯 Inference with `inference_multitask.py`
