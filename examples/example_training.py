"""Example 3: Training setup."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.train_multitask import MultiTaskTrainer


def main():
    """Setup training."""
    print("\n" + "="*60)
    print("Example 3: Training Setup")
    print("="*60)
    
    # Create trainer
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trainer = MultiTaskTrainer(
        output_dir="results/multitask",
        device=device
    )
    
    print(f"✓ Trainer created")
    print(f"\nConfiguration:")
    print(f"  Device: {trainer.device}")
    print(f"  Output directory: {trainer.output_dir}")
    
    print("\n" + "-"*60)
    print("To start training, run:")
    print("-"*60)
    print("""
python -m src.train_multitask \\
    --train-images data/train/images \\
    --train-annotations data/train/annotations.json \\
    --val-images data/val/images \\
    --val-annotations data/val/annotations.json \\
    --epochs 50 \\
    --batch-size 16 \\
    --lr 0.001 \\
    --device cuda:0
    """)
    
    print("\nTraining parameters:")
    print("  --epochs: Number of training epochs (default: 50)")
    print("  --batch-size: Batch size (default: 16)")
    print("  --lr: Learning rate (default: 0.001)")
    print("  --output: Output directory (default: results/multitask)")
    print("  --device: Device to use (default: cuda:0)")


if __name__ == "__main__":
    main()
