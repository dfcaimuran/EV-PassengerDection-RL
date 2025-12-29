"""Data loading utilities for multi-task learning with attributes."""

import os
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import Dataset, DataLoader


class PassengerAttributeDataset(Dataset):
    """Dataset for passenger detection with attribute labels.
    
    Expected directory structure:
    data/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── annotations/
    │   ├── train_attributes.json
    │   ├── val_attributes.json
    │   └── test_attributes.json
    └── dataset.yaml
    """
    
    def __init__(
        self,
        image_dir: str,
        annotations_file: str,
        image_size: int = 640,
        split: str = "train",
    ):
        """Initialize dataset.
        
        Args:
            image_dir: Directory containing images
            annotations_file: JSON file with bounding boxes and attributes
            image_size: Target image size
            split: "train", "val", or "test"
        """
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.split = split
        
        # Load annotations
        self.annotations = self._load_annotations(annotations_file)
        self.image_files = list(self.annotations.keys())
        
        # Attribute class mappings
        self.gender_classes = ["male", "female", "other"]
        self.age_classes = ["0-2", "3-5", "6-12", "13-18", "19-30", "31-45", "46-60", "60+"]
        self.height_classes = ["very_short", "short", "average", "tall", "very_tall"]
        self.bmi_classes = ["underweight", "normal", "overweight", "obese"]
        self.clothing_classes = [
            "casual", "formal", "sports", "traditional", "work",
            "summer", "winter", "other", "unknown", "multiple"
        ]
    
    def _load_annotations(self, annotations_file: str) -> Dict[str, Dict]:
        """Load annotations from JSON file.
        
        Args:
            annotations_file: Path to JSON file
        
        Returns:
            Dictionary mapping image filenames to annotations
        """
        if not os.path.exists(annotations_file):
            print(f"Warning: Annotations file not found: {annotations_file}")
            return {}
        
        with open(annotations_file, "r") as f:
            return json.load(f)
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get single sample.
        
        Returns:
            Dictionary with:
                - image: Tensor of shape (3, H, W)
                - boxes: List of bounding boxes
                - gender: List of gender labels
                - age_group: List of age group labels
                - height_range: List of height labels
                - bmi_category: List of BMI labels
                - clothing: List of clothing labels
        """
        image_file = self.image_files[idx]
        image_path = self.image_dir / image_file
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load image: {image_path}")
            return self._get_empty_sample()
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        
        # Load annotations
        ann = self.annotations.get(image_file, {})
        
        boxes = ann.get("boxes", [])  # List of [x1, y1, x2, y2]
        genders = ann.get("genders", [])
        ages = ann.get("age_groups", [])
        heights = ann.get("height_ranges", [])
        bmis = ann.get("bmi_categories", [])
        clothings = ann.get("clothings", [])
        
        # Convert class labels to indices
        gender_indices = [self.gender_classes.index(g) if g in self.gender_classes else 2 
                         for g in genders]
        age_indices = [self.age_classes.index(a) if a in self.age_classes else 0 
                      for a in ages]
        height_indices = [self.height_classes.index(h) if h in self.height_classes else 2 
                         for h in heights]
        bmi_indices = [self.bmi_classes.index(b) if b in self.bmi_classes else 1 
                      for b in bmis]
        clothing_indices = [self.clothing_classes.index(c) if c in self.clothing_classes else 8 
                           for c in clothings]
        
        return {
            "image": image,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "gender": torch.tensor(gender_indices, dtype=torch.long),
            "age_group": torch.tensor(age_indices, dtype=torch.long),
            "height_range": torch.tensor(height_indices, dtype=torch.long),
            "bmi_category": torch.tensor(bmi_indices, dtype=torch.long),
            "clothing": torch.tensor(clothing_indices, dtype=torch.long),
            "image_file": image_file,
        }
    
    def _get_empty_sample(self) -> Dict[str, Any]:
        """Return empty sample (for error handling)."""
        return {
            "image": torch.zeros(3, self.image_size, self.image_size),
            "boxes": torch.zeros(0, 4),
            "gender": torch.zeros(0, dtype=torch.long),
            "age_group": torch.zeros(0, dtype=torch.long),
            "height_range": torch.zeros(0, dtype=torch.long),
            "bmi_category": torch.zeros(0, dtype=torch.long),
            "clothing": torch.zeros(0, dtype=torch.long),
            "image_file": "unknown",
        }


def create_multitask_dataloader(
    image_dir: str,
    annotations_file: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 640,
    split: str = "train",
    shuffle: bool = True,
) -> DataLoader:
    """Create dataloader for multi-task learning.
    
    Args:
        image_dir: Directory with images
        annotations_file: JSON file with labels
        batch_size: Batch size
        num_workers: Number of workers
        image_size: Target image size
        split: "train", "val", or "test"
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader instance
    """
    dataset = PassengerAttributeDataset(
        image_dir=image_dir,
        annotations_file=annotations_file,
        image_size=image_size,
        split=split,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and split == "train",
        num_workers=num_workers,
        drop_last=split == "train",
    )


# Example JSON annotation format
ANNOTATION_FORMAT = {
    "image1.jpg": {
        "boxes": [
            [100, 150, 200, 300],  # [x1, y1, x2, y2]
            [250, 100, 350, 350],
        ],
        "genders": ["male", "female"],
        "age_groups": ["25-30", "35-40"],
        "height_ranges": ["tall", "average"],
        "bmi_categories": ["normal", "overweight"],
        "clothings": ["casual", "formal"],
    }
}
