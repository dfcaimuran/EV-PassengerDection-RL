"""Simple inference pipeline using YOLOv11."""

from typing import List, Dict, Any, Optional
from src.models.detector import PassengerDetector
from src.utils.data_utils import load_images
import os


def predict_image(image_path: str, model_path: str = "yolo11m.pt") -> List[Dict[str, Any]]:
    """Predict on single image.
    
    Args:
        image_path: Path to image
        model_path: Path to YOLO weights
    """
    detector = PassengerDetector(model_path)
    return detector.detect(image_path)


def predict_directory(input_dir: str, model_path: str = "yolo11m.pt") -> Dict[str, List[Dict]]:
    """Predict on directory of images.
    
    Args:
        input_dir: Directory with images
        model_path: Path to YOLO weights
    """
    detector = PassengerDetector(model_path)
    return detector.detect_batch(input_dir)

