"""YOLO11 detection inference."""

from ultralytics import YOLO
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np


class PassengerDetector:
    """YOLOv11 based passenger detection."""

    def __init__(self, model_path: str = "yolo11m.pt", device: int = 0):
        """Initialize detector with YOLOv11 model.
        
        Args:
            model_path: Path to YOLO weights or model ID
            device: GPU device ID (-1 for CPU)
        """
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, image_path: str) -> List[Dict[str, Any]]:
        """Detect passengers in image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of detections with bounding boxes and confidence
        """
        results = self.model(image_path, device=self.device, conf=0.5, iou=0.45)
        detections = []
        
        for result in results:
            for box in result.boxes:
                detection = {
                    "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    "confidence": float(box.conf[0]),
                    "class": int(box.cls[0]),
                }
                detections.append(detection)
        
        return detections

    def detect_batch(self, image_dir: str) -> Dict[str, List[Dict]]:
        """Detect passengers in multiple images.
        
        Args:
            image_dir: Directory with images
            
        Returns:
            Dictionary mapping image paths to detections
        """
        from src.utils.data_utils import load_images
        
        image_paths = load_images(image_dir)
        results = {}
        
        for img_path in image_paths:
            results[img_path] = self.detect(img_path)
        
        return results

