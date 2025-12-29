"""Example script demonstrating multi-task learning framework usage."""

import sys
import json
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.utils.multitask_data import PassengerAttributeDataset, create_multitask_dataloader
from src.models.multitask_model import MultiTaskDetectionModel, MultiTaskLoss
from src.train_multitask import MultiTaskTrainer
from src.inference_multitask import MultiTaskInferenceEngine


def example_1_data_loading():
    """Example 1: Load and inspect dataset."""
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
        print(f"\nSample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Number of people: {len(sample['boxes'])}")
        print(f"Image file: {sample['image_file']}")


def example_2_dataloader():
    """Example 2: Create dataloader for batching."""
    print("\n" + "="*60)
    print("Example 2: DataLoader")
    print("="*60)
    
    # Create dataloader
    dataloader = create_multitask_dataloader(
        image_dir="data/train/images",
        annotations_file="data/train/annotations.json",
        batch_size=4,
        num_workers=0,
        image_size=640,
        split="train",
        shuffle=True
    )
    
    print(f"✓ DataLoader created with {len(dataloader)} batches")
    
    # Get first batch
    for batch in dataloader:
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Image batch shape: {batch['image'].shape}")
        print(f"Boxes shape: {batch['boxes'].shape}")
        print(f"Gender labels shape: {batch['gender'].shape}")
        break


def example_3_model_creation():
    """Example 3: Create and inspect model."""
    print("\n" + "="*60)
    print("Example 3: Model Architecture")
    print("="*60)
    
    # Create model
    model = MultiTaskDetectionModel(yolo_model_name="yolo11m")
    model.eval()
    
    print(f"✓ Model created successfully")
    print(f"Model type: {type(model).__name__}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✓ Forward pass successful")
    print(f"Output keys: {output.keys()}")
    for key, val in output.items():
        print(f"  {key}: {val.shape}")


def example_4_loss_function():
    """Example 4: Setup multi-task loss."""
    print("\n" + "="*60)
    print("Example 4: Multi-Task Loss")
    print("="*60)
    
    # Create loss function
    criterion = MultiTaskLoss(
        detection_weight=0.5,
        gender_weight=0.1,
        age_weight=0.1,
        height_weight=0.1,
        bmi_weight=0.1,
        clothing_weight=0.1,
    )
    
    print(f"✓ Loss function created")
    print(f"Loss type: {type(criterion).__name__}")
    print(f"\nLoss weights:")
    print(f"  Detection: 50%")
    print(f"  Gender: 10%")
    print(f"  Age: 10%")
    print(f"  Height: 10%")
    print(f"  BMI: 10%")
    print(f"  Clothing: 10%")


def example_5_training_setup():
    """Example 5: Setup training."""
    print("\n" + "="*60)
    print("Example 5: Training Setup")
    print("="*60)
    
    # Create trainer
    trainer = MultiTaskTrainer(
        output_dir="results/example_training",
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"✓ Trainer created")
    print(f"Device: {trainer.device}")
    print(f"Output directory: {trainer.output_dir}")
    
    print("\nTo start training, run:")
    print("python -m src.train_multitask \\")
    print("    --train-images data/train/images \\")
    print("    --train-annotations data/train/annotations.json \\")
    print("    --val-images data/val/images \\")
    print("    --val-annotations data/val/annotations.json \\")
    print("    --epochs 50 \\")
    print("    --batch-size 16 \\")
    print("    --lr 0.001")


def example_6_inference_setup():
    """Example 6: Setup inference."""
    print("\n" + "="*60)
    print("Example 6: Inference Setup")
    print("="*60)
    
    checkpoint_path = "results/multitask/checkpoints/best_model.pt"
    
    if Path(checkpoint_path).exists():
        print(f"✓ Checkpoint found: {checkpoint_path}")
        
        # Create inference engine
        engine = MultiTaskInferenceEngine(checkpoint_path)
        
        print(f"✓ Inference engine created")
        
        print("\nTo run inference on an image:")
        print("results = engine.infer('path/to/image.jpg')")
        print("print(json.dumps(results, indent=2))")
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("Please train a model first using the training script.")
        print("\nInference usage example:")
        print("from src.inference_multitask import MultiTaskInferenceEngine")
        print("engine = MultiTaskInferenceEngine('path/to/checkpoint.pt')")
        print("results = engine.infer('image.jpg')")


def example_7_data_validation():
    """Example 7: Validate data format."""
    print("\n" + "="*60)
    print("Example 7: Data Validation")
    print("="*60)
    
    annotation_file = "data/train/annotations.json"
    
    if Path(annotation_file).exists():
        print(f"✓ Annotation file found: {annotation_file}")
        
        # Load and check
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        print(f"  Total images: {len(data)}")
        
        total_people = sum(len(img.get('boxes', [])) for img in data.values())
        print(f"  Total people: {total_people}")
        
        # Check attribute distribution
        print("\nAttribute distribution (first image):")
        if data:
            first_img_name = list(data.keys())[0]
            first_img = data[first_img_name]
            num_people = len(first_img.get('boxes', []))
            
            print(f"  Image: {first_img_name}")
            print(f"  People: {num_people}")
            print(f"  Genders: {first_img.get('genders', [])}")
            print(f"  Age groups: {first_img.get('age_groups', [])}")
            print(f"  Heights: {first_img.get('height_ranges', [])}")
            print(f"  BMIs: {first_img.get('bmi_categories', [])}")
            print(f"  Clothings: {first_img.get('clothings', [])}")
    else:
        print(f"✗ Annotation file not found: {annotation_file}")
        print("\nTo validate your data, use:")
        print("python scripts/data_converter.py validate \\")
        print("    --annotations path/to/annotations.json \\")
        print("    --images path/to/images")


def example_8_conversion():
    """Example 8: Data format conversion."""
    print("\n" + "="*60)
    print("Example 8: Data Format Conversion")
    print("="*60)
    
    print("If you have data in YOLO format (txt files), convert it:")
    print("\npython scripts/data_converter.py convert \\")
    print("    --from-format yolo \\")
    print("    --input path/to/yolo/labels \\")
    print("    --output annotations.json")
    
    print("\nNote: YOLO conversion fills attributes with default values.")
    print("You must manually update them with actual values.")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Multi-Task Learning Framework - Usage Examples")
    print("="*70)
    
    examples = [
        ("Data Loading", example_1_data_loading),
        ("DataLoader Batch", example_2_dataloader),
        ("Model Architecture", example_3_model_creation),
        ("Loss Function", example_4_loss_function),
        ("Training Setup", example_5_training_setup),
        ("Inference Setup", example_6_inference_setup),
        ("Data Validation", example_7_data_validation),
        ("Format Conversion", example_8_conversion),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...\n")
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n✗ Error in {name}:")
            print(f"  {type(e).__name__}: {e}")
    
    print("\n" + "="*70)
    print("Examples Complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Prepare your data with annotations.json")
    print("2. Validate with: python scripts/data_converter.py validate --annotations ... --images ...")
    print("3. Train with: python -m src.train_multitask --train-images ... --train-annotations ...")
    print("4. Inference with: python -m src.inference_multitask --checkpoint ... --image ...")
    print("\nFor more details, see MULTITASK_GUIDE.md")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
