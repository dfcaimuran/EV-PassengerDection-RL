"""Reward functions for reinforcement learning passenger detection."""

import numpy as np
from typing import Dict, List, Any


class RewardCalculator:
    """Calculate rewards for detection quality."""

    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes.
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def calculate_detection_reward(
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        iou_threshold: float = 0.5,
    ) -> float:
        """Calculate reward based on detection quality.
        
        Args:
            predictions: List of predicted boxes
            ground_truth: List of ground truth boxes
            iou_threshold: Minimum IoU for positive match
        
        Returns:
            Reward score (0-1)
        """
        if not ground_truth:
            return 1.0 if not predictions else 0.0
        
        if not predictions:
            return 0.0
        
        matched = set()
        total_iou = 0.0
        
        for pred in predictions:
            best_iou = 0.0
            best_idx = -1
            
            for i, gt in enumerate(ground_truth):
                if i in matched:
                    continue
                
                iou = RewardCalculator.calculate_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_iou >= iou_threshold:
                matched.add(best_idx)
                total_iou += best_iou
        
        # Precision and recall
        precision = len(matched) / len(predictions) if predictions else 0.0
        recall = len(matched) / len(ground_truth) if ground_truth else 0.0
        
        # Weighted reward
        iou_reward = total_iou / len(matched) if matched else 0.0
        reward = 0.4 * precision + 0.4 * recall + 0.2 * iou_reward
        
        return float(reward)

    @staticmethod
    def calculate_efficiency_reward(
        predictions: List[Dict[str, Any]],
        inference_time: float,
        max_time: float = 0.1,
    ) -> float:
        """Calculate reward for inference efficiency.
        
        Args:
            predictions: List of predictions
            inference_time: Time taken for inference (seconds)
            max_time: Maximum acceptable time (seconds)
        
        Returns:
            Efficiency reward (0-1)
        """
        if inference_time <= max_time:
            return 1.0
        
        time_penalty = min(inference_time / max_time - 1.0, 1.0)
        return max(0.0, 1.0 - time_penalty * 0.5)

    @staticmethod
    def calculate_combined_reward(
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        inference_time: float,
        weights: Dict[str, float] = None,
    ) -> float:
        """Calculate combined reward.
        
        Args:
            predictions: List of predictions
            ground_truth: Ground truth annotations
            inference_time: Inference time
            weights: Weight for each component
        
        Returns:
            Combined reward score
        """
        if weights is None:
            weights = {"detection": 0.7, "efficiency": 0.3}
        
        detection_reward = RewardCalculator.calculate_detection_reward(
            predictions, ground_truth
        )
        efficiency_reward = RewardCalculator.calculate_efficiency_reward(
            predictions, inference_time
        )
        
        combined = (
            weights["detection"] * detection_reward +
            weights["efficiency"] * efficiency_reward
        )
        
        return float(combined)
