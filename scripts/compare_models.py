#!/usr/bin/env python
"""Compare trained model vs original YOLOv11m pretrained model."""

import time
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from tabulate import tabulate

def compare_models():
    """Compare original YOLOv11m vs trained model on COCO val2017."""
    
    # Model paths
    pretrained_model = Path("yolo11m.pt")
    trained_model = Path("results/coco_full/passenger_detection/weights/best.pt")
    
    if not pretrained_model.exists():
        print(f"✗ Pretrained model not found: {pretrained_model}")
        return
    
    if not trained_model.exists():
        print(f"✗ Trained model not found: {trained_model}")
        return
    
    # Dataset
    data_dir = Path("data/coco/images/val2017")
    if not data_dir.exists():
        print(f"✗ Dataset not found: {data_dir}")
        return
    
    # Get test images (use 50 images for comparison)
    all_images = sorted(list(data_dir.glob("*.jpg")))[:50]
    image_paths = [str(img) for img in all_images]
    
    print(f"\n{'='*80}")
    print("Model Comparison: Original YOLOv11m vs Trained Model")
    print(f"{'='*80}")
    print(f"Test images: {len(image_paths)}")
    print(f"Dataset: COCO val2017\n")
    
    # Load models
    print("Loading models...")
    pretrained = YOLO(str(pretrained_model))
    trained = YOLO(str(trained_model))
    
    # Run inference on pretrained
    print(f"\nRunning inference on pretrained YOLOv11m...")
    start_time = time.time()
    results_pretrained = pretrained.predict(
        source=image_paths,
        save=False,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        verbose=False
    )
    time_pretrained = time.time() - start_time
    
    # Run inference on trained
    print(f"Running inference on trained model...")
    start_time = time.time()
    results_trained = trained.predict(
        source=image_paths,
        save=False,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        verbose=False
    )
    time_trained = time.time() - start_time
    
    # Calculate statistics
    detections_pretrained = sum(len(r.boxes) for r in results_pretrained)
    detections_trained = sum(len(r.boxes) for r in results_trained)
    
    # Calculate confidence scores
    conf_pretrained = []
    conf_trained = []
    
    for r in results_pretrained:
        if len(r.boxes) > 0:
            conf_pretrained.extend(r.boxes.conf.cpu().numpy().tolist())
    
    for r in results_trained:
        if len(r.boxes) > 0:
            conf_trained.extend(r.boxes.conf.cpu().numpy().tolist())
    
    avg_conf_pretrained = np.mean(conf_pretrained) if conf_pretrained else 0
    avg_conf_trained = np.mean(conf_trained) if conf_trained else 0
    
    # Prepare comparison table
    comparison_data = [
        ["Metric", "Pretrained YOLOv11m", "Trained Model", "Improvement"],
        ["-" * 20, "-" * 22, "-" * 22, "-" * 22],
        [
            "Total Detections",
            f"{detections_pretrained}",
            f"{detections_trained}",
            f"{detections_trained - detections_pretrained:+d} ({(detections_trained/max(detections_pretrained, 1) - 1)*100:+.1f}%)"
        ],
        [
            "Detections per Image",
            f"{detections_pretrained/len(image_paths):.2f}",
            f"{detections_trained/len(image_paths):.2f}",
            f"{(detections_trained - detections_pretrained)/len(image_paths):+.2f}"
        ],
        [
            "Avg Confidence",
            f"{avg_conf_pretrained:.4f}",
            f"{avg_conf_trained:.4f}",
            f"{avg_conf_trained - avg_conf_pretrained:+.4f}"
        ],
        [
            "Total Inference Time",
            f"{time_pretrained:.2f}s",
            f"{time_trained:.2f}s",
            f"{time_trained - time_pretrained:+.2f}s"
        ],
        [
            "Time per Image",
            f"{time_pretrained/len(image_paths):.4f}s",
            f"{time_trained/len(image_paths):.4f}s",
            f"{(time_trained - time_pretrained)/len(image_paths):+.4f}s"
        ],
    ]
    
    print("\n" + tabulate(comparison_data, headers="firstrow", tablefmt="grid"))
    
    print(f"\n{'='*80}")
    print("Summary:")
    print(f"{'='*80}")
    print(f"✓ Pretrained Model: {detections_pretrained} detections at {avg_conf_pretrained:.4f} avg confidence")
    print(f"✓ Trained Model: {detections_trained} detections at {avg_conf_trained:.4f} avg confidence")
    
    if detections_trained > detections_pretrained:
        improvement = ((detections_trained / detections_pretrained) - 1) * 100
        print(f"\n✅ Trained model detects {improvement:.1f}% MORE people than pretrained!")
    elif detections_trained < detections_pretrained:
        reduction = ((detections_pretrained / detections_trained) - 1) * 100
        print(f"\n⚠️  Trained model detects {reduction:.1f}% FEWER people (may indicate overfitting to COCO)")
    else:
        print(f"\n→ Models detect same number of people")
    
    if avg_conf_trained > avg_conf_pretrained:
        conf_gain = (avg_conf_trained - avg_conf_pretrained) * 100
        print(f"✅ Trained model has {conf_gain:.2f}% higher average confidence")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    compare_models()
