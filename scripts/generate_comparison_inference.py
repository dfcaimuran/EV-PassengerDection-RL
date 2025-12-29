#!/usr/bin/env python
"""Generate inference results with both pretrained and trained models on same images."""

import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import random
import json

def generate_comparison_inference():
    """Run inference with both models on same 20 images."""
    
    # Paths
    pretrained_model = Path("yolo11m.pt")
    trained_model = Path("results/coco_full/passenger_detection/weights/best.pt")
    data_dir = Path("data/coco/images/val2017")
    output_dir = Path("results/coco_full/passenger_detection/inference_results")
    
    # Check models
    if not pretrained_model.exists():
        print(f"✗ Pretrained model not found: {pretrained_model}")
        return
    
    if not trained_model.exists():
        print(f"✗ Trained model not found: {trained_model}")
        return
    
    if not data_dir.exists():
        print(f"✗ Dataset not found: {data_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Generating Comparison Inference Results")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    
    # Get all COCO images and randomly select 20
    all_images = sorted(list(data_dir.glob("*.jpg")))
    random.seed(42)  # Same seed for consistency
    image_files = random.sample(all_images, min(20, len(all_images)))
    image_files = sorted(image_files)
    
    # Save image list for reference
    image_list_path = output_dir / "inference_images.json"
    with open(image_list_path, 'w') as f:
        json.dump([img.name for img in image_files], f, indent=2)
    
    print(f"\nSelected images saved to: {image_list_path}")
    
    # Load models
    print("\nLoading models...")
    pretrained = YOLO(str(pretrained_model))
    trained = YOLO(str(trained_model))
    
    # Run inference with pretrained model
    print(f"\nRunning inference with pretrained YOLOv11m...")
    results_pretrained = pretrained.predict(
        source=[str(img) for img in image_files],
        save=False,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        verbose=False
    )
    
    # Run inference with trained model
    print(f"Running inference with trained model...")
    results_trained = trained.predict(
        source=[str(img) for img in image_files],
        save=False,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        verbose=False
    )
    
    # Save results from both models
    print(f"\nSaving results...")
    
    total_det_pretrained = 0
    total_det_trained = 0
    
    # Save pretrained model results
    for i, result in enumerate(results_pretrained, 1):
        im_array = result.plot()
        im_bgr = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
        
        # Add detection count
        num_detections = len(result.boxes)
        total_det_pretrained += num_detections
        text = f"Detections: {num_detections} (Pretrained)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 2
        text_color = (0, 255, 255)  # Yellow in BGR
        
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        h, w = im_bgr.shape[:2]
        x = w - text_size[0] - 15
        y = h - 15
        
        cv2.rectangle(im_bgr, (x - 5, y - text_size[1] - 5), (w - 10, h - 5), (0, 0, 0), -1)
        cv2.putText(im_bgr, text, (x, y), font, font_scale, text_color, font_thickness)
        
        output_path = output_dir / f"pretrained_inference_{i}.png"
        cv2.imwrite(str(output_path), im_bgr)
        print(f"  ✓ Saved: pretrained_inference_{i}.png ({num_detections} detections)")
    
    # Save trained model results
    for i, result in enumerate(results_trained, 1):
        im_array = result.plot()
        im_bgr = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
        
        # Add detection count
        num_detections = len(result.boxes)
        total_det_trained += num_detections
        text = f"Detections: {num_detections} (Trained)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 2
        text_color = (0, 255, 0)  # Green in BGR
        
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        h, w = im_bgr.shape[:2]
        x = w - text_size[0] - 15
        y = h - 15
        
        cv2.rectangle(im_bgr, (x - 5, y - text_size[1] - 5), (w - 10, h - 5), (0, 0, 0), -1)
        cv2.putText(im_bgr, text, (x, y), font, font_scale, text_color, font_thickness)
        
        output_path = output_dir / f"trained_inference_{i}.png"
        cv2.imwrite(str(output_path), im_bgr)
        print(f"  ✓ Saved: trained_inference_{i}.png ({num_detections} detections)")
    
    print(f"\n✓ Comparison inference complete!")
    print(f"  Pretrained YOLOv11m: {total_det_pretrained} detections across 20 images")
    print(f"  Trained Model: {total_det_trained} detections across 20 images")
    print(f"  Difference: {total_det_trained - total_det_pretrained:+d} detections")
    print(f"  Location: {output_dir.resolve()}")


if __name__ == "__main__":
    generate_comparison_inference()
