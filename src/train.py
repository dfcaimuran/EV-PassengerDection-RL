"""YOLOv11 fine-tuning for passenger detection."""

import argparse
from pathlib import Path
from ultralytics import YOLO
from src.config import YOLO_CONFIG, TRAIN_CONFIG, RESULTS_DIR, MODELS_DIR


def train(data_yaml: str, output_dir: str = RESULTS_DIR, epochs: int = None):
    """Train YOLOv11 on passenger detection dataset.
    
    Args:
        data_yaml: Path to dataset YAML file
        output_dir: Output directory for results
        epochs: Number of training epochs (overrides config)
    """
    # Create model
    model = YOLO(f"{YOLO_CONFIG['model']}.pt")
    
    # Training configuration
    train_config = TRAIN_CONFIG.copy()
    train_config["device"] = YOLO_CONFIG["device"]
    train_config["project"] = output_dir
    train_config["name"] = "passenger_detection"
    
    # Override epochs if provided
    if epochs is not None:
        train_config["epochs"] = epochs
    
    print(f"Starting YOLOv11 training...")
    print(f"Model: {YOLO_CONFIG['model']}")
    print(f"Data: {data_yaml}")
    print(f"Epochs: {train_config['epochs']}")
    
    # Train model
    results = model.train(
        data=data_yaml,
        **train_config
    )
    
    return results


def export_model(weights_path: str, export_format: str = "onnx"):
    """Export trained model to different formats.
    
    Args:
        weights_path: Path to trained weights
        export_format: Export format (onnx, torchscript, tflite, etc.)
    """
    model = YOLO(weights_path)
    
    export_path = model.export(format=export_format)
    print(f"Model exported to: {export_path}")
    
    return export_path


def validate_model(weights_path: str, data_yaml: str):
    """Validate model on validation set.
    
    Args:
        weights_path: Path to trained weights
        data_yaml: Path to dataset YAML
    """
    model = YOLO(weights_path)
    
    results = model.val(data=data_yaml, device=YOLO_CONFIG["device"])
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 for passenger detection")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument("--output", type=str, default=RESULTS_DIR, help="Output directory")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--validate", action="store_true", help="Validate after training")
    parser.add_argument("--export", type=str, help="Export format (onnx, torchscript, etc.)")
    parser.add_argument("--weights", type=str, help="Path to weights for export/validate")
    
    args = parser.parse_args()
    
    if args.weights:
        if args.validate:
            print("Validating model...")
            validate_model(args.weights, args.data)
        if args.export:
            print(f"Exporting model to {args.export}...")
            export_model(args.weights, args.export)
    else:
        # Train model
        train(args.data, args.output, args.epochs)


if __name__ == "__main__":
    main()
