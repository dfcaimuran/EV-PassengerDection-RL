# Examples

Multi-task passenger detection framework examples.

## Quick Start

Run all examples:
```bash
cd d:\Github\EV-PassengerDection-RL
python examples/run_all_examples.py
```

## Individual Examples

### 1. Data Loading
```bash
python examples/example_data_loading.py
```
Shows how to load and inspect the dataset.

### 2. Model Architecture
```bash
python examples/example_model.py
```
Creates the model and tests forward pass.

### 3. Training Setup
```bash
python examples/example_training.py
```
Shows how to configure and start training.

### 4. Inference Setup
```bash
python examples/example_inference.py
```
Shows how to run inference on images.

## Data Format

Before running examples, create your data structure:

```
data/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   └── ...
│   └── annotations.json
├── val/
│   ├── images/
│   └── annotations.json
```

### annotations.json Format

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

## Full Workflow

1. **Prepare data** with proper annotations
2. **Validate format**:
   ```bash
   python scripts/data_converter.py validate \
       --annotations data/train/annotations.json \
       --images data/train/images
   ```
3. **Train model**:
   ```bash
   python -m src.train_multitask \
       --train-images data/train/images \
       --train-annotations data/train/annotations.json \
       --val-images data/val/images \
       --val-annotations data/val/annotations.json \
       --epochs 50 --batch-size 16 --device cuda:0
   ```
4. **Run inference**:
   ```bash
   python -m src.inference_multitask \
       --checkpoint results/multitask/checkpoints/best_model.pt \
       --image test.jpg --visualize
   ```

## See Also

- [MULTITASK_GUIDE.md](../MULTITASK_GUIDE.md) - Full documentation
- [README.md](../README.md) - Project overview
