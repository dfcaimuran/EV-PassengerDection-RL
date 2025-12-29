#!/usr/bin/env python
"""Generate inference results visualization on COCO val dataset."""

import os
from pathlib import Path
from ultralytics import YOLO
import shutil
import cv2

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
    
    # Get all COCO images and randomly select 20
    all_images = sorted(list(data_dir.glob("*.jpg")))
    import random
    # random.seed(42)  # Comment out for true randomness on each run
    image_files = random.sample(all_images, min(20, len(all_images)))
    image_files = sorted(image_files)  # Sort for consistent order
    
    print(f"\nProcessing {len(image_files)} randomly selected images for visualization...")
    
    # Run inference
    results = model.predict(
        source=[str(img) for img in image_files],
        save=False,  # Don't auto-save
        conf=0.25,
        iou=0.45,
        imgsz=640,
        verbose=False
    )
    
    # Count detections
    total_detections = sum(len(r.boxes) for r in results)
    
    # Manually save annotated images
    for i, result in enumerate(results, 1):
        # Get the annotated image
        im_array = result.plot()  # RGB array
        
        # Convert RGB to BGR for cv2.imwrite
        im_bgr = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
        
        # Add detection count in bottom-right corner
        num_detections = len(result.boxes)
        text = f"Detections: {num_detections}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 2
        text_color = (0, 255, 0)  # Green in BGR
        
        # Get text size to position it correctly
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        h, w = im_bgr.shape[:2]
        x = w - text_size[0] - 15  # 15 pixels from right edge
        y = h - 15  # 15 pixels from bottom edge
        
        # Add background rectangle for better visibility
        cv2.rectangle(im_bgr, (x - 5, y - text_size[1] - 5), (w - 10, h - 5), (0, 0, 0), -1)
        # Put text
        cv2.putText(im_bgr, text, (x, y), font, font_scale, text_color, font_thickness)
        
        # Save to output directory
        output_path = output_dir / f"inference_{i}.png"
        cv2.imwrite(str(output_path), im_bgr)
        print(f"  ✓ Saved: inference_{i}.png ({num_detections} detections)")
    
    print(f"\n✓ Inference complete!")
    print(f"  Total detections: {total_detections}")
    print(f"  Saved {len(results)} detection images")
    print(f"  Location: {output_dir.resolve()}")

if __name__ == "__main__":
    generate_inference_results()
