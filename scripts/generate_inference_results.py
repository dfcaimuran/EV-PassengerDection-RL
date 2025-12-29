#!/usr/bin/env python
"""Generate inference results visualization on COCO val dataset."""

import os
from pathlib import Path
from ultralytics import YOLO
import shutil

def generate_inference_results():
    """Run inference on COCO val2017 images and save visualization."""
    
    # Path to trained model
    best_model = Path("results/coco_full/passenger_detection/weights/best.pt")
    
    if not best_model.exists():
        print(f"✗ Model not found: {best_model}")
        return
    
    # Input and output paths
    data_dir = Path("data/coco/images/val2017")
    output_dir = Path("results/coco_full/passenger_detection/inference_results")
    
    if not data_dir.exists():
        print(f"✗ Dataset not found: {data_dir}")
        print("Please download COCO dataset first: python scripts/download_coco.py")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Generating Inference Results")
    print(f"{'='*60}")
    print(f"Model: {best_model}")
    print(f"Input: {data_dir}")
    print(f"Output: {output_dir}")
    
    # Load model
    model = YOLO(str(best_model))
    
    # Run inference on first 6 images for visualization
    image_files = sorted(list(data_dir.glob("*.jpg")))[:6]
    print(f"\nProcessing {len(image_files)} images for visualization...")
    
    # Save predictions directly to results directory
    temp_output = Path("runs/detect/inference_temp")
    results = model.predict(
        source=[str(img) for img in image_files],
        save=True,
        save_dir=str(temp_output),
        conf=0.25,
        iou=0.45,
        imgsz=640,
        verbose=False
    )
    
    # Count detections
    total_detections = sum(len(r.boxes) for r in results)
    
    # Copy results to main output directory
    predict_dir = temp_output / "predict"
    if predict_dir.exists():
        # Get up to 6 best images (sorted by filename)
        detection_images = sorted(list(predict_dir.glob("*.jpg")))[:6]
        
        print(f"\n✓ Inference complete!")
        print(f"  Total detections: {total_detections}")
        print(f"  Saving {len(detection_images)} sample detection images...")
        
        for i, img_path in enumerate(detection_images, 1):
            dest = output_dir / f"inference_{i}.jpg"
            shutil.copy(img_path, dest)
            print(f"  ✓ {dest.name}")
        
        # Cleanup
        import shutil as sh
        sh.rmtree(temp_output, ignore_errors=True)
        
        print(f"\n✓ Inference results saved to: {output_dir.resolve()}")
    else:
        print(f"✗ Prediction directory not found at {predict_dir}")

if __name__ == "__main__":
    generate_inference_results()
