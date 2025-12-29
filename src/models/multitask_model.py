"""Multi-task learning model for passenger detection and attribute classification."""

import torch
import torch.nn as nn
from ultralytics import YOLO
from typing import Dict, List, Tuple, Any


class MultiTaskDetectionModel(nn.Module):
    """Multi-task model combining detection and attribute classification."""
    
    def __init__(self, yolo_model_name: str = "yolo11m"):
        """Initialize multi-task model.
        
        Args:
            yolo_model_name: YOLO model size (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x)
        """
        super().__init__()
        
        # Load pretrained YOLO as backbone
        self.yolo_backbone = YOLO(f"{yolo_model_name}.pt")
        
        # Attribute classification heads
        # Assuming we'll extract features from YOLO backbone
        feature_dim = 512  # Adjust based on your backbone
        
        # Gender classifier (male, female, other)
        self.gender_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3),  # 3 classes
        )
        
        # Age group classifier (0-2, 3-5, 6-12, 13-18, 19-30, 31-45, 46-60, 60+)
        self.age_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 8),  # 8 classes
        )
        
        # Height range classifier (very_short, short, average, tall, very_tall)
        self.height_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 5),  # 5 classes
        )
        
        # BMI category classifier (underweight, normal, overweight, obese)
        self.bmi_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4),  # 4 classes
        )
        
        # Clothing type classifier (casual, formal, sports, etc.)
        # Expand this based on your needs
        self.clothing_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),  # 10 clothing types
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input images
        
        Returns:
            Dictionary with detection results and attribute predictions
        """
        # Detection via YOLO
        detection_results = self.yolo_backbone(x)
        
        # Extract features from detection head
        # Note: You'll need to modify this based on actual YOLO architecture
        # For now, this is a placeholder
        
        # Attribute predictions
        attributes = {
            "gender": self.gender_head(x),      # Placeholder
            "age_group": self.age_head(x),      # Placeholder
            "height_range": self.height_head(x), # Placeholder
            "bmi_category": self.bmi_head(x),   # Placeholder
            "clothing": self.clothing_head(x),   # Placeholder
        }
        
        return {
            "detections": detection_results,
            "attributes": attributes,
        }


class MultiTaskLoss(nn.Module):
    """Combined loss for detection and attribute classification."""
    
    def __init__(self, detection_weight: float = 0.7, attr_weight: float = 0.3):
        """Initialize loss function.
        
        Args:
            detection_weight: Weight for detection loss
            attr_weight: Weight for attribute classification loss
        """
        super().__init__()
        self.detection_weight = detection_weight
        self.attr_weight = attr_weight
        
        # Detection loss (handled by YOLO internally)
        # Attribute losses
        self.gender_loss = nn.CrossEntropyLoss()
        self.age_loss = nn.CrossEntropyLoss()
        self.height_loss = nn.CrossEntropyLoss()
        self.bmi_loss = nn.CrossEntropyLoss()
        self.clothing_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        attr_predictions: Dict[str, torch.Tensor],
        attr_targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Calculate losses.
        
        Args:
            attr_predictions: Predicted attributes
            attr_targets: Target attributes
        
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        
        # Calculate individual attribute losses
        losses["gender_loss"] = self.gender_loss(
            attr_predictions["gender"], attr_targets["gender"]
        )
        losses["age_loss"] = self.age_loss(
            attr_predictions["age_group"], attr_targets["age_group"]
        )
        losses["height_loss"] = self.height_loss(
            attr_predictions["height_range"], attr_targets["height_range"]
        )
        losses["bmi_loss"] = self.bmi_loss(
            attr_predictions["bmi_category"], attr_targets["bmi_category"]
        )
        losses["clothing_loss"] = self.clothing_loss(
            attr_predictions["clothing"], attr_targets["clothing"]
        )
        
        # Total attribute loss
        total_attr_loss = (
            losses["gender_loss"] +
            losses["age_loss"] +
            losses["height_loss"] +
            losses["bmi_loss"] +
            losses["clothing_loss"]
        ) / 5.0
        
        losses["total_attr_loss"] = total_attr_loss
        losses["total_loss"] = self.attr_weight * total_attr_loss
        
        return losses
