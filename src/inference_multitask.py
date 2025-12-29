"""Multi-task inference for passenger detection and attribute prediction."""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

from models.multitask_model import MultiTaskDetectionModel
from config import YOLO_CONFIG


class MultiTaskInferenceEngine:
    """Inference engine for multi-task passenger detection."""
    
    # Class mappings
    GENDER_CLASSES = ["male", "female", "other"]
    AGE_CLASSES = ["0-2", "3-5", "6-12", "13-18", "19-30", "31-45", "46-60", "60+"]
    HEIGHT_CLASSES = ["very_short", "short", "average", "tall", "very_tall"]
    BMI_CLASSES = ["underweight", "normal", "overweight", "obese"]
    CLOTHING_CLASSES = [
        "casual", "formal", "sports", "traditional", "work",
        "summer", "winter", "other", "unknown", "multiple"
    ]
    
    def __init__(self, checkpoint_path: str, device: str = "cuda:0"):
        """Initialize inference engine.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use
        """
        self.device = torch.device(device)
        self.model = MultiTaskDetectionModel().to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        
        print(f"✓ Model loaded from {checkpoint_path}")
    
    def infer(
        self,
        image_path: str,
        image_size: int = 640,
        confidence_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Run inference on image.
        
        Args:
            image_path: Path to image file
            image_size: Target image size
            confidence_threshold: Detection confidence threshold
        
        Returns:
            Dictionary with detections and attributes
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original_h, original_w = image.shape[:2]
        
        # Preprocess
        image_resized = cv2.resize(image, (image_size, image_size))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).float().unsqueeze(0) / 255.0
        image_tensor = image_tensor.permute(0, 3, 1, 2).to(self.device)  # (1, 3, H, W)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # Process outputs
        results = {
            "image_path": image_path,
            "original_size": (original_w, original_h),
            "detections": [],
        }
        
        # Extract detections
        detection_outputs = outputs["detection"].cpu().numpy()[0]  # (num_detections, 5)
        gender_logits = outputs["gender"].cpu()[0]  # (num_people, 3)
        age_logits = outputs["age"].cpu()[0]  # (num_people, 8)
        height_logits = outputs["height"].cpu()[0]  # (num_people, 5)
        bmi_logits = outputs["bmi"].cpu()[0]  # (num_people, 4)
        clothing_logits = outputs["clothing"].cpu()[0]  # (num_people, 10)
        
        # Process each detection
        num_detections = detection_outputs.shape[0]
        for i in range(num_detections):
            # Parse detection (x1, y1, x2, y2, confidence)
            x1, y1, x2, y2, conf = detection_outputs[i]
            
            if conf < confidence_threshold:
                continue
            
            # Scale coordinates back to original image
            scale_x = original_w / image_size
            scale_y = original_h / image_size
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            # Get attribute predictions
            if i < len(gender_logits):
                gender_idx = torch.argmax(gender_logits[i]).item()
                gender = self.GENDER_CLASSES[gender_idx]
                gender_conf = torch.softmax(gender_logits[i], dim=0)[gender_idx].item()
            else:
                gender = "unknown"
                gender_conf = 0.0
            
            if i < len(age_logits):
                age_idx = torch.argmax(age_logits[i]).item()
                age = self.AGE_CLASSES[age_idx]
                age_conf = torch.softmax(age_logits[i], dim=0)[age_idx].item()
            else:
                age = "unknown"
                age_conf = 0.0
            
            if i < len(height_logits):
                height_idx = torch.argmax(height_logits[i]).item()
                height = self.HEIGHT_CLASSES[height_idx]
                height_conf = torch.softmax(height_logits[i], dim=0)[height_idx].item()
            else:
                height = "unknown"
                height_conf = 0.0
            
            if i < len(bmi_logits):
                bmi_idx = torch.argmax(bmi_logits[i]).item()
                bmi = self.BMI_CLASSES[bmi_idx]
                bmi_conf = torch.softmax(bmi_logits[i], dim=0)[bmi_idx].item()
            else:
                bmi = "unknown"
                bmi_conf = 0.0
            
            if i < len(clothing_logits):
                clothing_idx = torch.argmax(clothing_logits[i]).item()
                clothing = self.CLOTHING_CLASSES[clothing_idx]
                clothing_conf = torch.softmax(clothing_logits[i], dim=0)[clothing_idx].item()
            else:
                clothing = "unknown"
                clothing_conf = 0.0
            
            detection = {
                "box": {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "width": x2 - x1, "height": y2 - y1,
                },
                "detection_confidence": float(conf),
                "attributes": {
                    "gender": {"class": gender, "confidence": float(gender_conf)},
                    "age_group": {"class": age, "confidence": float(age_conf)},
                    "height_range": {"class": height, "confidence": float(height_conf)},
                    "bmi_category": {"class": bmi, "confidence": float(bmi_conf)},
                    "clothing": {"class": clothing, "confidence": float(clothing_conf)},
                },
            }
            results["detections"].append(detection)
        
        return results
    
    def infer_batch(
        self,
        image_dir: str,
        output_file: str = "predictions.json",
        confidence_threshold: float = 0.5,
    ):
        """Run inference on batch of images.
        
        Args:
            image_dir: Directory containing images
            output_file: Output JSON file with results
            confidence_threshold: Detection confidence threshold
        """
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        all_results = {}
        
        print(f"\nRunning inference on {len(image_files)} images...")
        
        for idx, image_path in enumerate(image_files):
            try:
                results = self.infer(str(image_path), confidence_threshold=confidence_threshold)
                all_results[image_path.name] = results
                
                if (idx + 1) % 10 == 0:
                    print(f"  Processed {idx + 1}/{len(image_files)} images")
            except Exception as e:
                print(f"  Error processing {image_path.name}: {e}")
        
        # Save results
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"✓ Predictions saved to {output_file}")
        return all_results
    
    def visualize(
        self,
        image_path: str,
        predictions: Dict[str, Any],
        output_path: str = None,
        thickness: int = 2,
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """Visualize predictions on image.
        
        Args:
            image_path: Path to image file
            predictions: Predictions from infer()
            output_path: Save visualized image to this path (optional)
            thickness: Box line thickness
            font_scale: Font size
        
        Returns:
            Visualized image
        """
        image = cv2.imread(image_path)
        
        for detection in predictions["detections"]:
            box = detection["box"]
            attrs = detection["attributes"]
            
            # Draw bounding box
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            
            # Prepare text
            text_lines = [
                f"Det: {detection['detection_confidence']:.2f}",
                f"Gender: {attrs['gender']['class']} ({attrs['gender']['confidence']:.2f})",
                f"Age: {attrs['age_group']['class']}",
                f"Height: {attrs['height_range']['class']}",
                f"BMI: {attrs['bmi_category']['class']}",
                f"Clothing: {attrs['clothing']['class']}",
            ]
            
            # Draw text
            y_offset = y1 - 10
            for line in text_lines:
                cv2.putText(
                    image, line, (x1, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1
                )
                y_offset -= 20
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"✓ Visualization saved to {output_path}")
        
        return image


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-task inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--image-dir", type=str, help="Path to image directory")
    parser.add_argument("--output", type=str, help="Output file for predictions")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument("--device", type=str, default="cuda:0")
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = MultiTaskInferenceEngine(args.checkpoint, device=args.device)
    
    if args.image:
        # Single image
        results = engine.infer(args.image)
        print("\n" + "="*50)
        print("Inference Results")
        print("="*50)
        print(json.dumps(results, indent=2))
        
        if args.visualize:
            engine.visualize(args.image, results, output_path="output.jpg")
    
    elif args.image_dir:
        # Batch inference
        output_file = args.output or "predictions.json"
        engine.infer_batch(args.image_dir, output_file=output_file)


if __name__ == "__main__":
    main()
