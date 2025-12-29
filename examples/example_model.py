"""Example 2: Model architecture and forward pass."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models.multitask_model import MultiTaskDetectionModel, MultiTaskLoss


def main():
    """Create model and test forward pass."""
    print("\n" + "="*60)
    print("Example 2: Model Architecture")
    print("="*60)
    
    # Create model
    model = MultiTaskDetectionModel(yolo_model_name="yolo11m")
    model.eval()
    
    print(f"✓ Model created successfully")
    print(f"Model type: {type(model).__name__}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✓ Forward pass successful")
    print(f"\nOutput tensors:")
    for key, val in output.items():
        print(f"  {key}: {val.shape}")
    
    # Show loss function
    print("\n" + "-"*60)
    print("Multi-Task Loss Configuration")
    print("-"*60)
    
    criterion = MultiTaskLoss(
        detection_weight=0.5,
        gender_weight=0.1,
        age_weight=0.1,
        height_weight=0.1,
        bmi_weight=0.1,
        clothing_weight=0.1,
    )
    
    print(f"Loss function: {type(criterion).__name__}")
    print(f"\nLoss weights:")
    print(f"  Detection: 50%")
    print(f"  Gender: 10%")
    print(f"  Age: 10%")
    print(f"  Height: 10%")
    print(f"  BMI: 10%")
    print(f"  Clothing: 10%")


if __name__ == "__main__":
    main()
