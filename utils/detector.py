"""
Parking Detector Module
-----------------------
YOLOv8-based car detection for top-down parking lot images.

Usage:
    from utils.detector import ParkingDetector
    
    detector = ParkingDetector(model_path="yolov8m.pt")
    detections = detector.detect("parking_lot.jpg")
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Please install ultralytics: pip install ultralytics==8.2.0")


@dataclass
class Detection:
    """Single detection result."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 (pixel coordinates)
    confidence: float
    class_id: int
    class_name: str
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def area(self) -> int:
        """Get area of bounding box in pixels."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def to_polygon(self) -> List[Tuple[int, int]]:
        """Convert bbox to polygon (4 corners)."""
        x1, y1, x2, y2 = self.bbox
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


class ParkingDetector:
    """
    YOLOv8-based detector for cars in parking lot images.
    
    Supports:
    - Pretrained COCO models (yolov8n/s/m/l/x.pt)
    - Custom fine-tuned models
    - CPU and GPU inference
    
    Args:
        model_path: Path to model weights or model name
        confidence_threshold: Minimum detection confidence (0-1)
        device: 'cpu', 'cuda', 'cuda:0', or 'auto'
        classes: List of class IDs to detect (default: [2] for cars in COCO)
    """
    
    # COCO class IDs we care about for parking detection
    VEHICLE_CLASSES = {
        2: 'car',
        5: 'bus',
        7: 'truck',
        3: 'motorcycle'
    }
    
    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        confidence_threshold: float = 0.25,
        device: str = "auto",
        classes: Optional[List[int]] = None
    ):
        self.confidence_threshold = confidence_threshold
        self.device = self._resolve_device(device)
        self.classes = classes if classes is not None else [2]  # Default: cars only
        
        # Load model
        print(f"Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        
        # Check if custom model (different class mapping)
        self.is_custom_model = self._check_custom_model(model_path)
        
        print(f"Model loaded. Device: {self.device}, Classes: {self.classes}")
    
    def _resolve_device(self, device: str) -> str:
        """Determine the best available device."""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device
    
    def _check_custom_model(self, model_path: str) -> bool:
        """Check if this is a custom fine-tuned model."""
        # If model has custom names, it's likely fine-tuned
        if hasattr(self.model, 'names'):
            # COCO has 80 classes, custom models typically have fewer
            return len(self.model.names) < 80
        return False
    
    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        save_visualization: bool = False,
        output_path: Optional[str] = None
    ) -> List[Detection]:
        """
        Run detection on an image.
        
        Args:
            image: Path to image file or numpy array (BGR format)
            save_visualization: Whether to save annotated image
            output_path: Path to save visualization
            
        Returns:
            List of Detection objects
        """
        # Run inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            device=self.device,
            classes=None if self.is_custom_model else self.classes,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            if result.boxes is None:
                continue
                
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get bounding box
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Get confidence
                conf = float(boxes.conf[i].cpu().numpy())
                
                # Get class
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                # Get class name
                if self.is_custom_model:
                    cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                else:
                    cls_name = self.VEHICLE_CLASSES.get(cls_id, f"class_{cls_id}")
                
                # Filter by class for COCO models
                if not self.is_custom_model and cls_id not in self.classes:
                    continue
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name
                ))
        
        # Optionally save visualization
        if save_visualization and output_path:
            self._save_visualization(image, detections, output_path)
        
        return detections
    
    def detect_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        batch_size: int = 8
    ) -> List[List[Detection]]:
        """
        Run detection on multiple images.
        
        Args:
            images: List of image paths or arrays
            batch_size: Number of images to process at once
            
        Returns:
            List of detection lists (one per image)
        """
        all_detections = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Run batch inference
            results = self.model(
                batch,
                conf=self.confidence_threshold,
                device=self.device,
                classes=None if self.is_custom_model else self.classes,
                verbose=False
            )
            
            for result in results:
                detections = []
                
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for j in range(len(boxes)):
                        xyxy = boxes.xyxy[j].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        conf = float(boxes.conf[j].cpu().numpy())
                        cls_id = int(boxes.cls[j].cpu().numpy())
                        
                        if self.is_custom_model:
                            cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                        else:
                            cls_name = self.VEHICLE_CLASSES.get(cls_id, f"class_{cls_id}")
                            if cls_id not in self.classes:
                                continue
                        
                        detections.append(Detection(
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            class_id=cls_id,
                            class_name=cls_name
                        ))
                
                all_detections.append(detections)
        
        return all_detections
    
    def _save_visualization(
        self,
        image: Union[str, Path, np.ndarray],
        detections: List[Detection],
        output_path: str
    ):
        """Save image with detection boxes drawn."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        else:
            img = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.imwrite(output_path, img)
        print(f"Visualization saved to: {output_path}")


def main():
    """Test the detector on a sample image."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test parking detector")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--model", default="yolov8m.pt", help="Model path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default="auto", help="Device (cpu/cuda/auto)")
    args = parser.parse_args()
    
    detector = ParkingDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        device=args.device
    )
    
    detections = detector.detect(args.image, save_visualization=True, 
                                  output_path="detection_output.jpg")
    
    print(f"\nFound {len(detections)} vehicles:")
    for det in detections:
        print(f"  - {det.class_name} at {det.bbox} (conf: {det.confidence:.2f})")


if __name__ == "__main__":
    main()
