"""Simple visualization utilities for YOLO detections."""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


def draw_bbox(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    label: str = "",
    confidence: Optional[float] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding box on image.
    
    Args:
        image: Input image
        bbox: [x1, y1, x2, y2]
        label: Label text
        confidence: Confidence score
        color: BGR color
        thickness: Line thickness
    """
    x1, y1, x2, y2 = [int(x) for x in bbox]
    
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    if label or confidence is not None:
        text = label
        if confidence is not None:
            text += f" {confidence:.2f}"
        
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(image, (x1, y1 - text_size[1] - 4), 
                     (x1 + text_size[0], y1), color, -1)
        cv2.putText(image, text, (x1, y1 - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


def plot_detections(
    image_path: str,
    detections: List[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Visualize detections on image.
    
    Args:
        image_path: Path to image
        detections: List of detections from YOLO
        save_path: Path to save result
    """
    image = cv2.imread(image_path)
    
    for det in detections:
        bbox = det.get("bbox")
        conf = det.get("confidence", 0)
        image = draw_bbox(image, bbox, confidence=conf)
    
    if save_path:
        cv2.imwrite(save_path, image)
    
    return image

