#!/usr/bin/env python3
"""
Standalone Inference Script
---------------------------
Run parking detection on images and output results.

Supports two backends:
1. Local YOLOv8 (--backend local)
2. Roboflow API (--backend roboflow)

Usage:
    # Using Roboflow (cloud - no GPU needed)
    python run_inference.py --image parking_lot.jpg --config configs/lot_config.yaml --backend roboflow
    
    # Using local YOLOv8
    python run_inference.py --image parking_lot.jpg --config configs/lot_config.yaml --backend local
    
    # Directory of images
    python run_inference.py --image-dir demo_images/ --config configs/lot_config.yaml
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import yaml

from utils.slot_matcher import SlotMatcher
from utils.visualization import ParkingVisualizer


def get_detector(backend: str, model_path: str, confidence: float, device: str, api_key: str = None):
    """Get the appropriate detector based on backend choice."""
    if backend == "roboflow":
        from utils.roboflow_detector import RoboflowDetector
        return RoboflowDetector(
            api_key=api_key or os.getenv("ROBOFLOW_API_KEY"),
            confidence_threshold=confidence
        )
    else:
        from utils.detector import ParkingDetector
        return ParkingDetector(
            model_path=model_path,
            confidence_threshold=confidence,
            device=device
        )


def run_inference(
    image_path: str,
    config_path: str,
    model_path: str = "yolov8m.pt",
    output_dir: str = "outputs",
    confidence: float = 0.25,
    device: str = "auto",
    backend: str = "roboflow",
    api_key: str = None,
    save_annotated: bool = True,
    save_schematic: bool = True,
    save_json: bool = True
) -> dict:
    """
    Run full inference pipeline on a single image.
    
    Args:
        image_path: Path to input image
        config_path: Path to lot configuration YAML
        model_path: Path to YOLOv8 model weights (for local backend)
        output_dir: Directory for outputs
        confidence: Detection confidence threshold
        device: Device to use for local inference (cpu/cuda/auto)
        backend: Detection backend ("roboflow" or "local")
        api_key: Roboflow API key (for roboflow backend)
        save_annotated: Save annotated image
        save_schematic: Save schematic map
        save_json: Save JSON results
        
    Returns:
        Dictionary with results summary
    """
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_path = Path(image_path)
    image_name = image_path.stem
    
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"Backend: {backend.upper()}")
    print(f"{'='*60}")
    
    # Initialize detector
    print("\n[1/4] Loading detector...")
    detector = get_detector(backend, model_path, confidence, device, api_key)
    
    print("[2/4] Loading lot configuration...")
    matcher = SlotMatcher(config_path=config_path)
    visualizer = ParkingVisualizer(config_path=config_path)
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"       Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Run detection
    print("[3/4] Running detection...")
    detections = detector.detect(image_path)
    print(f"       Found {len(detections)} vehicles")
    
    # Match to slots
    print("[4/4] Matching to parking slots...")
    result = matcher.match(detections)
    
    # Print summary
    print(f"\n{'─'*40}")
    print(f"RESULTS SUMMARY")
    print(f"{'─'*40}")
    print(f"  Total slots:    {result.total_slots}")
    print(f"  Empty slots:    {result.empty_slots}")
    print(f"  Occupied slots: {result.occupied_slots}")
    print(f"  Occupancy rate: {result.occupancy_rate:.1%}")
    print(f"{'─'*40}")
    
    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if save_annotated:
        annotated = visualizer.draw_full_overlay(image, detections, result)
        annotated_path = output_dir / f"{image_name}_annotated_{timestamp}.jpg"
        cv2.imwrite(str(annotated_path), annotated)
        print(f"\n✓ Annotated image: {annotated_path}")
    
    if save_schematic:
        lot_name = matcher.lot_name if hasattr(matcher, 'lot_name') else "Parking Lot"
        schematic = visualizer.create_schematic_map(
            result.slots, 
            title=f"{lot_name} - {datetime.now().strftime('%H:%M:%S')}"
        )
        schematic_path = output_dir / f"{image_name}_schematic_{timestamp}.jpg"
        cv2.imwrite(str(schematic_path), schematic)
        print(f"✓ Schematic map:   {schematic_path}")
    
    if save_json:
        json_result = {
            "image": str(image_path),
            "timestamp": timestamp,
            "config": str(config_path),
            "model": model_path,
            "detections": [
                {
                    "bbox": det.bbox,
                    "confidence": det.confidence,
                    "class": det.class_name
                }
                for det in detections
            ],
            "occupancy": result.to_dict()
        }
        json_path = output_dir / f"{image_name}_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(json_result, f, indent=2)
        print(f"✓ JSON results:    {json_path}")
    
    return result.to_dict()


def run_batch_inference(
    image_dir: str,
    config_path: str,
    model_path: str = "yolov8m.pt",
    output_dir: str = "outputs",
    confidence: float = 0.25,
    device: str = "auto",
    backend: str = "roboflow",
    api_key: str = None
) -> list:
    """
    Run inference on all images in a directory.
    
    Args:
        image_dir: Directory containing images
        config_path: Path to lot configuration
        model_path: Path to model weights (for local backend)
        output_dir: Directory for outputs
        confidence: Detection confidence threshold
        device: Device to use (for local backend)
        backend: Detection backend ("roboflow" or "local")
        api_key: Roboflow API key
        
    Returns:
        List of result dictionaries
    """
    image_dir = Path(image_dir)
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = [p for p in image_dir.iterdir() 
              if p.suffix.lower() in image_extensions]
    
    if not images:
        print(f"No images found in {image_dir}")
        return []
    
    print(f"\nFound {len(images)} images to process")
    print(f"Using {backend.upper()} backend")
    
    results = []
    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] ", end="")
        try:
            result = run_inference(
                image_path=str(image_path),
                config_path=config_path,
                model_path=model_path,
                output_dir=output_dir,
                confidence=confidence,
                device=device,
                backend=backend,
                api_key=api_key
            )
            results.append({"image": str(image_path), "status": "success", "result": result})
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({"image": str(image_path), "status": "error", "error": str(e)})
    
    # Print batch summary
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total images: {len(images)}")
    print(f"  Successful:   {sum(1 for r in results if r['status'] == 'success')}")
    print(f"  Failed:       {sum(1 for r in results if r['status'] == 'error')}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Smart Parking Detection - Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using Roboflow cloud API (no GPU needed)
    python run_inference.py --image lot.jpg --config configs/lot_config.yaml --backend roboflow
    
    # Using local YOLOv8
    python run_inference.py --image lot.jpg --config configs/lot_config.yaml --backend local
    
    # Batch processing with Roboflow
    python run_inference.py --image-dir demo_images/ --config configs/lot_config.yaml --backend roboflow
    
    # Custom model (local only)
    python run_inference.py --image lot.jpg --model best.pt --backend local --output results/
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", "-i",
        type=str,
        help="Path to single input image"
    )
    input_group.add_argument(
        "--image-dir", "-d",
        type=str,
        help="Directory of images to process"
    )
    
    # Backend selection
    parser.add_argument(
        "--backend", "-b",
        type=str,
        default="roboflow",
        choices=["roboflow", "local"],
        help="Detection backend: 'roboflow' (cloud API) or 'local' (YOLOv8) (default: roboflow)"
    )
    
    # Roboflow settings
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)"
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/lot_config.yaml",
        help="Path to lot configuration YAML (default: configs/lot_config.yaml)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolov8m.pt",
        help="Path to YOLOv8 model weights for local backend (default: yolov8m.pt)"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs",
        help="Output directory (default: outputs/)"
    )
    parser.add_argument(
        "--no-annotated",
        action="store_true",
        help="Don't save annotated images"
    )
    parser.add_argument(
        "--no-schematic",
        action="store_true",
        help="Don't save schematic maps"
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Don't save JSON results"
    )
    
    # Detection settings
    parser.add_argument(
        "--confidence", "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"],
        help="Device for local inference (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Validate config exists
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        print("Create a lot configuration file or use --config to specify path")
        sys.exit(1)
    
    # Run inference
    if args.image:
        if not Path(args.image).exists():
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)
        
        run_inference(
            image_path=args.image,
            config_path=args.config,
            model_path=args.model,
            output_dir=args.output,
            confidence=args.confidence,
            device=args.device,
            backend=args.backend,
            api_key=args.api_key,
            save_annotated=not args.no_annotated,
            save_schematic=not args.no_schematic,
            save_json=not args.no_json
        )
    else:
        if not Path(args.image_dir).is_dir():
            print(f"Error: Directory not found: {args.image_dir}")
            sys.exit(1)
        
        run_batch_inference(
            image_dir=args.image_dir,
            config_path=args.config,
            model_path=args.model,
            output_dir=args.output,
            confidence=args.confidence,
            device=args.device,
            backend=args.backend,
            api_key=args.api_key
        )


if __name__ == "__main__":
    main()
