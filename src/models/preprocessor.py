"""Image preprocessing utilities (YOLO handles most of this)."""

import cv2
import numpy as np
from typing import Tuple


class ImagePreprocessor:
    """Simple image utilities."""

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """Load image from file."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return image

    @staticmethod
    def resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image."""
        return cv2.resize(image, size)

    @staticmethod
    def to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert BGR to RGB."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

