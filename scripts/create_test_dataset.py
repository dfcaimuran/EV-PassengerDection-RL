"""
Create a small test dataset for quick validation.

Usage:
    python scripts/create_test_dataset.py --num-images 50
"""

import os
import json
import random
import argparse
from pathlib import Path
import shutil


def create_test_dataset(output_dir: str = "data/test_dataset", num_images: int = 50):
    """Create a simple test dataset with YOLO annotations.
    
    Args:
        output_dir: Output directory
        num_images: Number of images to generate
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create directories
    for subset in ["train", "val"]:
        (output_path / "images" / subset).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / subset).mkdir(parents=True, exist_ok=True)
    
    # Try to get COCO val2017 samples if available
    # Check both possible paths
    coco_images = Path("data/coco/val2017") if Path("data/coco/val2017").exists() else Path("data/coco/images/val2017")
    coco_labels = Path("data/coco/labels/val2017")
    
    if coco_images.exists() and coco_labels.exists():
        print(f"Using COCO dataset samples from {coco_images}")
        
        # Get all COCO images with labels
        image_files = list(coco_images.glob("*.jpg"))
        if not image_files:
            image_files = list(coco_images.glob("*.png"))
        
        # Shuffle and select
        random.shuffle(image_files)
        selected = image_files[:num_images]
        
        # Split: 80% train, 20% val
        split_idx = int(len(selected) * 0.8)
        train_images = selected[:split_idx]
        val_images = selected[split_idx:]
        
        # Copy train set
        print(f"Copying {len(train_images)} training images...")
        for img_file in train_images:
            img_dst = output_path / "images" / "train" / img_file.name
            label_src = coco_labels / f"{img_file.stem}.txt"
            label_dst = output_path / "labels" / "train" / f"{img_file.stem}.txt"
            
            if img_file.exists():
                shutil.copy(img_file, img_dst)
            if label_src.exists():
                shutil.copy(label_src, label_dst)
        
        # Copy val set
        print(f"Copying {len(val_images)} validation images...")
        for img_file in val_images:
            img_dst = output_path / "images" / "val" / img_file.name
            label_src = coco_labels / f"{img_file.stem}.txt"
            label_dst = output_path / "labels" / "val" / f"{img_file.stem}.txt"
            
            if img_file.exists():
                shutil.copy(img_file, img_dst)
            if label_src.exists():
                shutil.copy(label_src, label_dst)
        
        print(f"✓ Created {len(train_images)} train + {len(val_images)} val images")
    
    else:
        print(f"Note: COCO dataset not found at {coco_images}")
        print("Please download COCO first:")
        print("  python scripts/download_coco.py --output data/coco --split val2017")
        return
    
    # Create dataset.yaml
    yaml_content = f"""# Test dataset for quick validation
path: {output_path.absolute()}
train: images/train
val: images/val

nc: 1
names:
  - person
"""
    
    yaml_file = output_path / "dataset.yaml"
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✓ Setup complete!")
    print(f"Dataset: {output_path.absolute()}")
    print(f"Dataset YAML: {yaml_file}")
    print(f"\nQuick test commands:")
    print(f"  # Fast training (10 epochs)")
    print(f"  python -m src.train --data {output_dir}/dataset.yaml --epochs 10")
    print(f"\n  # RL optimization (2 iterations)")
    print(f"  python -m src.train_rl --data {output_dir}/dataset.yaml --iterations 2")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create small test dataset for quick validation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/test_dataset",
        help="Output directory"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help="Number of images (default: 50)"
    )
    
    args = parser.parse_args()
    
    create_test_dataset(args.output, args.num_images)


if __name__ == "__main__":
    main()
