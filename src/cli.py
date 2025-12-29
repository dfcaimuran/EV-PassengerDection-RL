"""Simple CLI for YOLOv11 passenger detection."""

import argparse
from pathlib import Path
from src.inference import predict_image, predict_directory
from src.utils.data_utils import save_results
from src.utils.visualization import plot_detections


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="YOLOv11 Passenger Detection"
    )

    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--input-dir", type=str, help="Directory with images")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--weights", type=str, default="yolo11m.pt", help="Path to YOLO weights")
    parser.add_argument("--save-json", action="store_true", help="Save results as JSON")
    parser.add_argument("--visualize", action="store_true", help="Save visualization images")

    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    # Process single image
    if args.image:
        print(f"Processing image: {args.image}")
        results = predict_image(args.image, args.weights)
        print(f"Found {len(results)} detections")
        
        if args.save_json:
            output_path = Path(args.output_dir) / "results.json"
            save_results(results, str(output_path))
            print(f"Results saved to {output_path}")
        
        if args.visualize:
            output_path = Path(args.output_dir) / "visualization.jpg"
            plot_detections(args.image, results, str(output_path))
            print(f"Visualization saved to {output_path}")

    # Process directory
    elif args.input_dir:
        print(f"Processing directory: {args.input_dir}")
        results = predict_directory(args.input_dir, args.weights)
        print(f"Processed {len(results)} images")
        
        if args.save_json:
            output_path = Path(args.output_dir) / "batch_results.json"
            save_results(results, str(output_path))
            print(f"Results saved to {output_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

