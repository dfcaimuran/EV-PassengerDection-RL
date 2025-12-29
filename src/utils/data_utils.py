"""Data loading and processing utilities."""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np


def load_images(directory: str) -> List[str]:
    """
    Load all image paths from directory.

    Args:
        directory: Directory path

    Returns:
        List of image file paths
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = []

    for file in os.listdir(directory):
        if Path(file).suffix.lower() in valid_extensions:
            image_paths.append(os.path.join(directory, file))

    return sorted(image_paths)


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save detection results to JSON file.

    Args:
        results: List of detection results
        output_path: Path to save JSON file
    """
    # Make directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(input_path: str) -> List[Dict[str, Any]]:
    """
    Load detection results from JSON file.

    Args:
        input_path: Path to JSON file

    Returns:
        List of detection results
    """
    with open(input_path, "r") as f:
        return json.load(f)


def create_data_splits(
    image_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1
) -> Dict[str, List[str]]:
    """
    Create train/val/test splits.

    Args:
        image_dir: Directory with images
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set

    Returns:
        Dict with 'train', 'val', 'test' keys
    """
    images = load_images(image_dir)
    np.random.shuffle(images)

    n = len(images)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    return {
        "train": images[:train_size],
        "val": images[train_size : train_size + val_size],
        "test": images[train_size + val_size :],
    }
