"""
Download and convert COCO dataset to YOLO format for person detection.

Usage:
    python scripts/download_coco.py --output data/coco --split val2017
"""

import os
import json
import argparse
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
import shutil


class COCODownloader:
    """Download and convert COCO dataset."""
    
    # COCO 2017 download URLs
    URLS = {
        "images_train2017": "http://images.cocodataset.org/zips/train2017.zip",
        "images_val2017": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations_train2017": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "annotations_val2017": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    }
    
    PERSON_CLASS_ID = 1  # Person is class 1 in COCO
    
    def __init__(self, output_dir: str = "data/coco"):
        """Initialize downloader.
        
        Args:
            output_dir: Directory to save COCO dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, url: str, output_path: Path, verbose: bool = True):
        """Download file with progress bar.
        
        Args:
            url: URL to download
            output_path: Where to save file
            verbose: Print progress
        """
        if output_path.exists():
            print(f"File already exists: {output_path}")
            return
        
        print(f"Downloading {url}...")
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"✓ Downloaded to {output_path}")
        except Exception as e:
            print(f"✗ Failed to download: {e}")
            raise
    
    def extract_zip(self, zip_path: Path, extract_to: Path):
        """Extract ZIP file.
        
        Args:
            zip_path: Path to ZIP file
            extract_to: Where to extract
        """
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ Extracted to {extract_to}")
    
    def download_coco(self, split: str = "val2017"):
        """Download COCO dataset split.
        
        Args:
            split: "train2017" or "val2017"
        """
        print(f"Downloading COCO {split}...")
        
        # Download images
        images_url = self.URLS[f"images_{split}"]
        images_zip = self.output_dir / f"{split}.zip"
        self.download_file(images_url, images_zip)
        self.extract_zip(images_zip, self.output_dir)
        
        # Download annotations (only once)
        anno_zip = self.output_dir / "annotations_trainval2017.zip"
        if not anno_zip.exists():
            anno_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            self.download_file(anno_url, anno_zip)
            self.extract_zip(anno_zip, self.output_dir)
    
    def coco_to_yolo(self, split: str = "val2017"):
        """Convert COCO annotations to YOLO format.
        
        Args:
            split: "train2017" or "val2017"
        """
        print(f"Converting {split} to YOLO format...")
        
        # Load COCO annotations
        anno_file = self.output_dir / "annotations" / f"instances_{split}.json"
        if not anno_file.exists():
            print(f"✗ Annotation file not found: {anno_file}")
            return
        
        with open(anno_file) as f:
            coco_data = json.load(f)
        
        # Build lookup tables
        images = {img['id']: img for img in coco_data['images']}
        annotations = [ann for ann in coco_data['annotations'] 
                      if ann['category_id'] == self.PERSON_CLASS_ID]
        
        print(f"Found {len(annotations)} person annotations")
        
        # Create YOLO label directory
        label_dir = self.output_dir / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert annotations
        converted = 0
        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in images:
                continue
            
            img = images[img_id]
            img_name = Path(img['file_name']).stem
            
            # YOLO format: <class> <x_center> <y_center> <width> <height>
            x, y, w, h = ann['bbox']
            x_center = (x + w / 2) / img['width']
            y_center = (y + h / 2) / img['height']
            w_norm = w / img['width']
            h_norm = h / img['height']
            
            # Clamp values to [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))
            
            # Write to YOLO format
            label_file = label_dir / f"{img_name}.txt"
            with open(label_file, 'a') as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            
            converted += 1
        
        print(f"✓ Converted {converted} annotations to YOLO format")
    
    def create_dataset_yaml(self, splits: List[str] = None):
        """Create dataset.yaml for YOLO training.
        
        Args:
            splits: List of splits (e.g., ["train2017", "val2017"])
        """
        if splits is None:
            splits = ["val2017"]
        
        # Determine paths
        train_path = None
        val_path = None
        
        if "train2017" in splits:
            train_path = "images/train2017"
        if "val2017" in splits:
            val_path = "images/val2017"
        
        # Create YAML content
        yaml_content = f"""# COCO 2017 dataset for person detection
path: {self.output_dir.absolute()}
train: {train_path if train_path else 'images/train2017'}
val: {val_path if val_path else 'images/val2017'}

# Classes
nc: 1
names:
  - person
"""
        
        yaml_file = self.output_dir / "dataset.yaml"
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        print(f"✓ Created {yaml_file}")
        return yaml_file
    
    def setup_complete(self):
        """Check if setup is complete."""
        yaml_file = self.output_dir / "dataset.yaml"
        return yaml_file.exists()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and convert COCO dataset to YOLO format"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/coco",
        help="Output directory (default: data/coco)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val2017",
        choices=["train2017", "val2017"],
        help="Dataset split to download (default: val2017, ~1GB)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download if files already exist"
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert existing annotations"
    )
    
    args = parser.parse_args()
    
    downloader = COCODownloader(args.output)
    
    try:
        if not args.convert_only:
            downloader.download_coco(args.split)
        
        downloader.coco_to_yolo(args.split)
        downloader.create_dataset_yaml([args.split])
        
        print("\n" + "="*60)
        print("✓ Setup complete!")
        print("="*60)
        print(f"\nDataset path: {downloader.output_dir.absolute()}")
        print(f"Dataset YAML: {downloader.output_dir / 'dataset.yaml'}")
        print("\nNext step:")
        print(f"   python -m src.train_rl --data {args.output}/dataset.yaml --iterations 5")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
