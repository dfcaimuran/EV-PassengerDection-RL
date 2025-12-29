"""Example 1: Data loading and inspection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.multitask_data import PassengerAttributeDataset


def main():
    """Load and inspect dataset."""
    print("\n" + "="*60)
    print("Example 1: Data Loading")
    print("="*60)
    
    # Create dataset
    dataset = PassengerAttributeDataset(
        image_dir="data/train/images",
        annotations_file="data/train/annotations.json",
        image_size=640,
        split="train"
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} images")
    
    # Load a single sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample structure:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Number of people: {len(sample['boxes'])}")
        print(f"  Image file: {sample['image_file']}")
        print(f"\nSample keys: {list(sample.keys())}")
    else:
        print("No images found in dataset")


if __name__ == "__main__":
    main()
