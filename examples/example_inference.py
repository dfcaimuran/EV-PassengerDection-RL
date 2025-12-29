"""Example 4: Inference setup."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference_multitask import MultiTaskInferenceEngine


def main():
    """Setup inference."""
    print("\n" + "="*60)
    print("Example 4: Inference Setup")
    print("="*60)
    
    checkpoint_path = "results/multitask/checkpoints/best_model.pt"
    
    if Path(checkpoint_path).exists():
        print(f"✓ Checkpoint found: {checkpoint_path}")
        
        # Create inference engine
        engine = MultiTaskInferenceEngine(checkpoint_path)
        
        print(f"✓ Inference engine initialized")
        
        print("\n" + "-"*60)
        print("Inference usage examples:")
        print("-"*60)
        
        print("""
# Single image
results = engine.infer('path/to/image.jpg')
print(json.dumps(results, indent=2))

# With visualization
engine.visualize('path/to/image.jpg', results, output_path='output.jpg')

# Batch inference
all_results = engine.infer_batch('path/to/images/', output_file='predictions.json')
        """)
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("\nTo use inference:")
        print("1. First train a model: python -m src.train_multitask ...")
        print("2. Then run inference: python -m src.inference_multitask ...")
        
        print("\n" + "-"*60)
        print("Inference command:")
        print("-"*60)
        print("""
# Single image
python -m src.inference_multitask \\
    --checkpoint results/multitask/checkpoints/best_model.pt \\
    --image test.jpg \\
    --visualize

# Batch
python -m src.inference_multitask \\
    --checkpoint results/multitask/checkpoints/best_model.pt \\
    --image-dir test_images/ \\
    --output predictions.json
        """)


if __name__ == "__main__":
    main()
