#!/usr/bin/env python3
"""
YOLOv8 Fine-Tuning Script for Parking Detection
------------------------------------------------
Fine-tune a pretrained YOLOv8 model on custom top-down parking lot images.

Usage:
    python train_parking_model.py --data configs/clemson_parking.yaml --epochs 50
    
    # Full options
    python train_parking_model.py \
        --data configs/clemson_parking.yaml \
        --model yolov8m.pt \
        --epochs 50 \
        --imgsz 640 \
        --batch-size 16 \
        --device auto \
        --project runs/detect \
        --name parking_model

Prerequisites:
    1. Annotated dataset in YOLO format
    2. Dataset config YAML file
    3. GPU recommended (CPU training is very slow)

Dataset Format:
    datasets/clemson_parking/
    ├── images/
    │   ├── train/
    │   │   ├── lot_001.jpg
    │   │   └── ...
    │   └── val/
    │       ├── lot_050.jpg
    │       └── ...
    └── labels/
        ├── train/
        │   ├── lot_001.txt  # YOLO format annotations
        │   └── ...
        └── val/
            └── ...

YOLO Label Format (each .txt file):
    class_id x_center y_center width height
    0 0.5 0.3 0.1 0.15
    (all values normalized 0-1)
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
    import torch
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install ultralytics torch")
    sys.exit(1)


def check_dataset(data_yaml: str) -> bool:
    """Verify dataset exists and is properly formatted."""
    import yaml
    
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"Error: Dataset config not found: {data_yaml}")
        return False
    
    with open(data_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get dataset root
    root = Path(config.get('path', data_path.parent))
    if not root.is_absolute():
        root = data_path.parent / root
    
    train_images = root / config.get('train', 'images/train')
    val_images = root / config.get('val', 'images/val')
    
    issues = []
    
    if not train_images.exists():
        issues.append(f"Training images not found: {train_images}")
    else:
        train_count = len(list(train_images.glob('*.[jJ][pP][gG]')) + 
                         list(train_images.glob('*.[pP][nN][gG]')))
        if train_count == 0:
            issues.append(f"No training images found in: {train_images}")
        else:
            print(f"✓ Found {train_count} training images")
    
    if not val_images.exists():
        issues.append(f"Validation images not found: {val_images}")
    else:
        val_count = len(list(val_images.glob('*.[jJ][pP][gG]')) + 
                       list(val_images.glob('*.[pP][nN][gG]')))
        if val_count == 0:
            issues.append(f"No validation images found in: {val_images}")
        else:
            print(f"✓ Found {val_count} validation images")
    
    # Check labels
    train_labels = train_images.parent.parent / 'labels' / 'train'
    val_labels = val_images.parent.parent / 'labels' / 'val'
    
    if train_labels.exists():
        label_count = len(list(train_labels.glob('*.txt')))
        print(f"✓ Found {label_count} training labels")
    else:
        issues.append(f"Training labels not found: {train_labels}")
    
    if val_labels.exists():
        label_count = len(list(val_labels.glob('*.txt')))
        print(f"✓ Found {label_count} validation labels")
    else:
        issues.append(f"Validation labels not found: {val_labels}")
    
    if issues:
        print("\n⚠ Dataset Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    return True


def get_device(device_arg: str) -> str:
    """Determine the best available device."""
    if device_arg != "auto":
        return device_arg
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU available: {gpu_name} ({gpu_mem:.1f} GB)")
        return "0"  # CUDA device index
    else:
        print("⚠ No GPU available, using CPU (training will be slow)")
        return "cpu"


def train_model(
    data_yaml: str,
    model: str = "yolov8m.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "auto",
    project: str = "runs/detect",
    name: str = "parking_model",
    resume: bool = False,
    patience: int = 20,
    workers: int = 8,
    optimizer: str = "auto",
    lr0: float = 0.01,
    augment: bool = True,
    val: bool = True,
    save: bool = True,
    save_period: int = -1,
    cache: bool = False,
    verbose: bool = True
):
    """
    Fine-tune YOLOv8 on parking dataset.
    
    Args:
        data_yaml: Path to dataset configuration YAML
        model: Pretrained model to fine-tune
        epochs: Number of training epochs
        imgsz: Input image size
        batch_size: Batch size (-1 for auto)
        device: Training device (auto/cpu/0/1/etc)
        project: Project directory for outputs
        name: Run name
        resume: Resume from last checkpoint
        patience: Early stopping patience
        workers: Number of data loader workers
        optimizer: Optimizer (SGD/Adam/auto)
        lr0: Initial learning rate
        augment: Use data augmentation
        val: Run validation after each epoch
        save: Save checkpoints
        save_period: Save checkpoint every N epochs (-1 for final only)
        cache: Cache images in RAM
        verbose: Verbose output
    """
    print("\n" + "="*60)
    print("YOLOv8 PARKING DETECTION TRAINING")
    print("="*60)
    
    # Check dataset
    print("\n[1/4] Checking dataset...")
    if not check_dataset(data_yaml):
        print("\n⚠ Please fix dataset issues before training")
        print("See README.md for dataset format instructions")
        return None
    
    # Determine device
    print("\n[2/4] Setting up device...")
    device = get_device(device)
    
    # Load model
    print(f"\n[3/4] Loading pretrained model: {model}")
    yolo_model = YOLO(model)
    
    # Training configuration
    print(f"\n[4/4] Starting training...")
    print(f"  Dataset:    {data_yaml}")
    print(f"  Model:      {model}")
    print(f"  Epochs:     {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device:     {device}")
    print(f"  Output:     {project}/{name}")
    print()
    
    # Train
    results = yolo_model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        resume=resume,
        patience=patience,
        workers=workers,
        optimizer=optimizer,
        lr0=lr0,
        augment=augment,
        val=val,
        save=save,
        save_period=save_period,
        cache=cache,
        verbose=verbose,
        # Parking-specific augmentation settings
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,      # No rotation for parking lots
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,       # No vertical flip for top-down
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
    )
    
    # Print results
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    weights_path = Path(project) / name / "weights" / "best.pt"
    if weights_path.exists():
        print(f"\n✓ Best weights saved to: {weights_path}")
        print(f"\nTo use your fine-tuned model:")
        print(f"  python scripts/run_inference.py --model {weights_path} --image your_image.jpg")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 for parking detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python train_parking_model.py --data configs/clemson_parking.yaml --epochs 50
    
    # With custom settings
    python train_parking_model.py \\
        --data configs/clemson_parking.yaml \\
        --model yolov8m.pt \\
        --epochs 100 \\
        --imgsz 640 \\
        --batch-size 16 \\
        --device 0
    
    # Resume interrupted training
    python train_parking_model.py --data configs/clemson_parking.yaml --resume

Dataset Structure:
    datasets/clemson_parking/
    ├── images/
    │   ├── train/       # Training images
    │   └── val/         # Validation images
    └── labels/
        ├── train/       # YOLO format .txt files
        └── val/

Label Format (each .txt file, one line per car):
    class_id x_center y_center width height
    0 0.5 0.3 0.1 0.15
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to dataset YAML config (e.g., configs/clemson_parking.yaml)"
    )
    
    # Model settings
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolov8m.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="Pretrained model to fine-tune (default: yolov8m.pt)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--imgsz", "--img-size",
        type=int,
        default=640,
        help="Input image size (default: 640)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Batch size (default: 16, use -1 for auto)"
    )
    
    # Device settings
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, 0, 1, etc. (default: auto)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of data loader workers (default: 8)"
    )
    
    # Output settings
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Project directory for outputs (default: runs/detect)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="parking_model",
        help="Run name (default: parking_model)"
    )
    
    # Training options
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20)"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        choices=["auto", "SGD", "Adam", "AdamW"],
        help="Optimizer (default: auto)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation"
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache images in RAM (faster but uses more memory)"
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Save checkpoint every N epochs (-1 for final only)"
    )
    
    args = parser.parse_args()
    
    # Run training
    train_model(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        patience=args.patience,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr,
        augment=not args.no_augment,
        cache=args.cache,
        save_period=args.save_period
    )


if __name__ == "__main__":
    main()
