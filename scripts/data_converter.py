"""Utility script to convert and validate passenger attribute data formats."""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any
import cv2


def validate_annotation_format(annotation_file: str) -> bool:
    """Validate annotation JSON format.
    
    Args:
        annotation_file: Path to annotation JSON file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # Check if it's a dictionary with image names as keys
        if not isinstance(data, dict):
            print("Error: Root must be a dictionary")
            return False
        
        # Validate each image entry
        for img_name, img_data in data.items():
            if not isinstance(img_data, dict):
                print(f"Error: {img_name} data must be a dictionary")
                return False
            
            required_keys = {"boxes", "genders", "age_groups", "height_ranges", 
                           "bmi_categories", "clothings"}
            if not required_keys.issubset(img_data.keys()):
                print(f"Error: {img_name} missing required keys")
                return False
            
            # Validate boxes
            boxes = img_data.get("boxes", [])
            if not isinstance(boxes, list):
                print(f"Error: {img_name} boxes must be a list")
                return False
            
            for box in boxes:
                if not isinstance(box, list) or len(box) != 4:
                    print(f"Error: {img_name} boxes must be [[x1, y1, x2, y2], ...]")
                    return False
            
            # Validate attribute lists length match
            num_people = len(boxes)
            genders = img_data.get("genders", [])
            ages = img_data.get("age_groups", [])
            heights = img_data.get("height_ranges", [])
            bmis = img_data.get("bmi_categories", [])
            clothings = img_data.get("clothings", [])
            
            if len(genders) != num_people:
                print(f"Error: {img_name} genders count mismatch")
                return False
            if len(ages) != num_people:
                print(f"Error: {img_name} age_groups count mismatch")
                return False
            if len(heights) != num_people:
                print(f"Error: {img_name} height_ranges count mismatch")
                return False
            if len(bmis) != num_people:
                print(f"Error: {img_name} bmi_categories count mismatch")
                return False
            if len(clothings) != num_people:
                print(f"Error: {img_name} clothings count mismatch")
                return False
        
        print(f"✓ Annotation format is valid")
        return True
    
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def validate_image_existence(image_dir: str, annotation_file: str) -> bool:
    """Validate that all annotated images exist.
    
    Args:
        image_dir: Directory containing images
        annotation_file: Path to annotation JSON file
    
    Returns:
        True if all images exist, False otherwise
    """
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    image_dir = Path(image_dir)
    missing_images = []
    
    for img_name in data.keys():
        img_path = image_dir / img_name
        if not img_path.exists():
            missing_images.append(img_name)
    
    if missing_images:
        print(f"Error: {len(missing_images)} images not found:")
        for img in missing_images[:10]:  # Show first 10
            print(f"  - {img}")
        return False
    
    print(f"✓ All {len(data)} annotated images exist")
    return True


def validate_attribute_values(annotation_file: str) -> bool:
    """Validate that attribute values are from allowed classes.
    
    Args:
        annotation_file: Path to annotation JSON file
    
    Returns:
        True if all values are valid, False otherwise
    """
    valid_values = {
        "genders": {"male", "female", "other"},
        "age_groups": {"0-2", "3-5", "6-12", "13-18", "19-30", "31-45", "46-60", "60+"},
        "height_ranges": {"very_short", "short", "average", "tall", "very_tall"},
        "bmi_categories": {"underweight", "normal", "overweight", "obese"},
        "clothings": {
            "casual", "formal", "sports", "traditional", "work",
            "summer", "winter", "other", "unknown", "multiple"
        },
    }
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    errors = []
    
    for img_name, img_data in data.items():
        for attr_key, valid_set in valid_values.items():
            attr_values = img_data.get(attr_key, [])
            for i, value in enumerate(attr_values):
                if value not in valid_set:
                    errors.append(f"{img_name}[{i}] {attr_key}: '{value}' not in {valid_set}")
    
    if errors:
        print(f"Error: {len(errors)} invalid attribute values:")
        for err in errors[:10]:  # Show first 10
            print(f"  - {err}")
        return False
    
    print(f"✓ All attribute values are valid")
    return True


def generate_summary(annotation_file: str, image_dir: str = None) -> Dict[str, Any]:
    """Generate summary statistics of dataset.
    
    Args:
        annotation_file: Path to annotation JSON file
        image_dir: Optional image directory to check image sizes
    
    Returns:
        Summary dictionary
    """
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    summary = {
        "total_images": len(data),
        "total_people": 0,
        "attribute_distribution": {
            "genders": {},
            "age_groups": {},
            "height_ranges": {},
            "bmi_categories": {},
            "clothings": {},
        }
    }
    
    for img_data in data.values():
        num_people = len(img_data.get("boxes", []))
        summary["total_people"] += num_people
        
        # Count attributes
        for gender in img_data.get("genders", []):
            summary["attribute_distribution"]["genders"][gender] = \
                summary["attribute_distribution"]["genders"].get(gender, 0) + 1
        
        for age in img_data.get("age_groups", []):
            summary["attribute_distribution"]["age_groups"][age] = \
                summary["attribute_distribution"]["age_groups"].get(age, 0) + 1
        
        for height in img_data.get("height_ranges", []):
            summary["attribute_distribution"]["height_ranges"][height] = \
                summary["attribute_distribution"]["height_ranges"].get(height, 0) + 1
        
        for bmi in img_data.get("bmi_categories", []):
            summary["attribute_distribution"]["bmi_categories"][bmi] = \
                summary["attribute_distribution"]["bmi_categories"].get(bmi, 0) + 1
        
        for clothing in img_data.get("clothings", []):
            summary["attribute_distribution"]["clothings"][clothing] = \
                summary["attribute_distribution"]["clothings"].get(clothing, 0) + 1
    
    return summary


def convert_from_yolo_format(yolo_labels_dir: str, output_file: str) -> None:
    """Convert from YOLO format (txt files) to multitask JSON format.
    
    YOLO format: Each image has a .txt file with:
    class_id x_center y_center width height (normalized 0-1)
    
    Note: YOLO format doesn't include attributes, so you'll need to add them manually.
    
    Args:
        yolo_labels_dir: Directory containing YOLO .txt label files
        output_file: Output JSON file
    """
    annotations = {}
    
    for txt_file in Path(yolo_labels_dir).glob("*.txt"):
        img_name = txt_file.stem  # Remove .txt extension
        
        boxes = []
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # Convert YOLO format to pixel coordinates
                    # For now, we'll store normalized coordinates
                    x_center, y_center, width, height = map(float, parts[1:5])
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    boxes.append([x1, y1, x2, y2])
        
        if boxes:
            num_people = len(boxes)
            annotations[f"{img_name}.jpg"] = {
                "boxes": boxes,
                "genders": ["other"] * num_people,  # Fill with defaults
                "age_groups": ["19-30"] * num_people,
                "height_ranges": ["average"] * num_people,
                "bmi_categories": ["normal"] * num_people,
                "clothings": ["unknown"] * num_people,
            }
    
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"✓ Converted {len(annotations)} images from YOLO format")
    print(f"  Note: Attributes are set to defaults. Update them manually with actual values.")
    print(f"  Output: {output_file}")


def main():
    """Main CLI utility."""
    parser = argparse.ArgumentParser(
        description="Utility for managing passenger attribute data"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate annotation format")
    validate_parser.add_argument("--annotations", type=str, required=True,
                                help="Path to annotation JSON file")
    validate_parser.add_argument("--images", type=str, help="Path to images directory")
    
    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Generate dataset summary")
    summary_parser.add_argument("--annotations", type=str, required=True,
                               help="Path to annotation JSON file")
    summary_parser.add_argument("--images", type=str, help="Path to images directory")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert from other formats")
    convert_parser.add_argument("--from-format", type=str, required=True,
                               choices=["yolo"],
                               help="Source format")
    convert_parser.add_argument("--input", type=str, required=True,
                               help="Input directory or file")
    convert_parser.add_argument("--output", type=str, required=True,
                               help="Output annotation JSON file")
    
    args = parser.parse_args()
    
    if args.command == "validate":
        print("Validating annotation format...")
        if not validate_annotation_format(args.annotations):
            return
        
        if args.images:
            print("Validating image existence...")
            if not validate_image_existence(args.images, args.annotations):
                return
        
        print("Validating attribute values...")
        if not validate_attribute_values(args.annotations):
            return
        
        print("\n✓ All validations passed!")
    
    elif args.command == "summary":
        print("Generating summary...")
        summary = generate_summary(args.annotations, args.images)
        
        print("\nDataset Summary:")
        print(f"  Total Images: {summary['total_images']}")
        print(f"  Total People: {summary['total_people']}")
        print(f"  Average People/Image: {summary['total_people'] / max(1, summary['total_images']):.2f}")
        
        print("\nAttribute Distribution:")
        for attr_type, dist in summary['attribute_distribution'].items():
            print(f"\n  {attr_type}:")
            for value, count in sorted(dist.items(), key=lambda x: -x[1]):
                pct = 100 * count / summary['total_people']
                print(f"    {value}: {count} ({pct:.1f}%)")
    
    elif args.command == "convert":
        if args.from_format == "yolo":
            print(f"Converting from YOLO format...")
            convert_from_yolo_format(args.input, args.output)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
