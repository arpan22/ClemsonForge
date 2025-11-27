"""
Roboflow Detector Module
------------------------
Car detection using Roboflow's hosted API for top-down parking lot images.

This is an alternative to the local YOLOv8 detector that uses Roboflow's
serverless API, eliminating the need to run models locally.

Usage:
    from utils.roboflow_detector import RoboflowDetector
    
    detector = RoboflowDetector(api_key="your_api_key")
    detections = detector.detect("parking_lot.jpg")

Advantages:
- No local GPU required
- No model download/setup
- Pre-trained on parking/car detection
- Serverless - scales automatically
"""

import os
import base64
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Dict, Any
import json

try:
    import requests
except ImportError:
    raise ImportError("Please install requests: pip install requests")

try:
    import numpy as np
    import cv2
except ImportError:
    raise ImportError("Please install opencv-python and numpy")


@dataclass
class Detection:
    """Single detection result (compatible with local detector)."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 (pixel coordinates)
    confidence: float
    class_id: int
    class_name: str
    detection_id: Optional[str] = None
    
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


class RoboflowDetector:
    """
    Roboflow-based detector for cars in parking lot images.
    
    Uses Roboflow's serverless API for inference, eliminating the need
    for local GPU or model management.
    
    Args:
        api_key: Roboflow API key (or set ROBOFLOW_API_KEY env var)
        workspace: Roboflow workspace name
        workflow_id: Workflow ID for car detection
        confidence_threshold: Minimum detection confidence (0-1)
        api_url: Roboflow API base URL
    """
    
    # Default configuration for parking detection
    DEFAULT_WORKSPACE = "parkpark-zclps"
    DEFAULT_WORKFLOW_ID = "find-cars-2"
    DEFAULT_API_URL = "https://serverless.roboflow.com"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        workspace: str = DEFAULT_WORKSPACE,
        workflow_id: str = DEFAULT_WORKFLOW_ID,
        confidence_threshold: float = 0.25,
        api_url: str = DEFAULT_API_URL
    ):
        # Get API key from param or environment
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Roboflow API key required. Provide via api_key parameter "
                "or set ROBOFLOW_API_KEY environment variable."
            )
        
        self.workspace = workspace
        self.workflow_id = workflow_id
        self.confidence_threshold = confidence_threshold
        self.api_url = api_url
        
        # Build endpoint URL
        self.endpoint = f"{self.api_url}/{self.workspace}/workflows/{self.workflow_id}"
        
        print(f"Roboflow Detector initialized")
        print(f"  Workspace: {self.workspace}")
        print(f"  Workflow: {self.workflow_id}")
        print(f"  Confidence threshold: {self.confidence_threshold}")
    
    def _encode_image(self, image: Union[str, Path, np.ndarray]) -> Tuple[str, int, int]:
        """
        Encode image to base64 for API request.
        
        Returns:
            Tuple of (base64_string, width, height)
        """
        if isinstance(image, (str, Path)):
            # Load from file
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image: {image}")
        else:
            # Already numpy array
            img = image
        
        height, width = img.shape[:2]
        
        # Encode to JPEG then base64
        _, buffer = cv2.imencode('.jpg', img)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        
        return base64_str, width, height
    
    def _parse_predictions(
        self, 
        response_data: Dict[str, Any],
        image_width: int,
        image_height: int
    ) -> List[Detection]:
        """
        Parse Roboflow API response into Detection objects.
        
        Roboflow returns predictions with x, y (center), width, height format.
        We convert to x1, y1, x2, y2 (corner) format.
        """
        detections = []
        
        try:
            outputs = response_data.get("outputs", [])
            if not outputs:
                return detections
            
            predictions_data = outputs[0].get("predictions", {})
            predictions = predictions_data.get("predictions", [])
            
            for pred in predictions:
                # Skip low confidence
                confidence = pred.get("confidence", 0)
                if confidence < self.confidence_threshold:
                    continue
                
                # Extract bounding box (Roboflow uses center x, y, width, height)
                cx = pred.get("x", 0)
                cy = pred.get("y", 0)
                w = pred.get("width", 0)
                h = pred.get("height", 0)
                
                # Convert to corner format (x1, y1, x2, y2)
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                x2 = int(cx + w / 2)
                y2 = int(cy + h / 2)
                
                # Clamp to image bounds
                x1 = max(0, min(x1, image_width))
                y1 = max(0, min(y1, image_height))
                x2 = max(0, min(x2, image_width))
                y2 = max(0, min(y2, image_height))
                
                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=pred.get("class_id", 0),
                    class_name=pred.get("class", "car"),
                    detection_id=pred.get("detection_id")
                )
                detections.append(detection)
        
        except (KeyError, IndexError, TypeError) as e:
            print(f"Warning: Error parsing predictions: {e}")
        
        return detections
    
    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        return_visualization: bool = False
    ) -> Union[List[Detection], Tuple[List[Detection], Optional[np.ndarray]]]:
        """
        Run detection on an image using Roboflow API.
        
        Args:
            image: Path to image file or numpy array (BGR format)
            return_visualization: If True, also return Roboflow's visualization
            
        Returns:
            List of Detection objects, or tuple (detections, visualization_image)
        """
        # Encode image
        base64_image, width, height = self._encode_image(image)
        
        # Prepare API request
        payload = {
            "api_key": self.api_key,
            "inputs": {
                "image": {
                    "type": "base64",
                    "value": base64_image
                }
            }
        }
        
        # Make request
        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if return_visualization:
                return [], None
            return []
        
        # Parse detections
        detections = self._parse_predictions(result, width, height)
        
        # Optionally get visualization
        visualization = None
        if return_visualization:
            try:
                viz_data = result.get("outputs", [{}])[0].get("visualization", {})
                if viz_data.get("type") == "base64":
                    viz_bytes = base64.b64decode(viz_data.get("value", ""))
                    viz_array = np.frombuffer(viz_bytes, dtype=np.uint8)
                    visualization = cv2.imdecode(viz_array, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"Warning: Could not decode visualization: {e}")
        
        if return_visualization:
            return detections, visualization
        return detections
    
    def detect_from_url(
        self,
        image_url: str,
        return_visualization: bool = False
    ) -> Union[List[Detection], Tuple[List[Detection], Optional[np.ndarray]]]:
        """
        Run detection on an image URL (more efficient for remote images).
        
        Args:
            image_url: URL of the image to analyze
            return_visualization: If True, also return visualization
            
        Returns:
            List of Detection objects
        """
        # First, get image dimensions
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            height, width = img.shape[:2]
        except Exception as e:
            print(f"Warning: Could not get image dimensions from URL: {e}")
            width, height = 1920, 1080  # Default fallback
        
        # Prepare API request with URL
        payload = {
            "api_key": self.api_key,
            "inputs": {
                "image": {
                    "type": "url",
                    "value": image_url
                }
            }
        }
        
        # Make request
        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if return_visualization:
                return [], None
            return []
        
        # Parse detections
        detections = self._parse_predictions(result, width, height)
        
        # Handle visualization
        visualization = None
        if return_visualization:
            try:
                viz_data = result.get("outputs", [{}])[0].get("visualization", {})
                if viz_data.get("type") == "base64":
                    viz_bytes = base64.b64decode(viz_data.get("value", ""))
                    viz_array = np.frombuffer(viz_bytes, dtype=np.uint8)
                    visualization = cv2.imdecode(viz_array, cv2.IMREAD_COLOR)
            except Exception:
                pass
        
        if return_visualization:
            return detections, visualization
        return detections
    
    def detect_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        show_progress: bool = True
    ) -> List[List[Detection]]:
        """
        Run detection on multiple images.
        
        Note: Roboflow API processes images sequentially. For true batch
        processing, consider their batch inference endpoints.
        
        Args:
            images: List of image paths or arrays
            show_progress: Show progress indicator
            
        Returns:
            List of detection lists (one per image)
        """
        all_detections = []
        
        for i, image in enumerate(images):
            if show_progress:
                print(f"Processing image {i+1}/{len(images)}...", end="\r")
            
            detections = self.detect(image)
            all_detections.append(detections)
        
        if show_progress:
            print(f"Processed {len(images)} images" + " " * 20)
        
        return all_detections


# Optional: SDK-based implementation
class RoboflowSDKDetector(RoboflowDetector):
    """
    Roboflow detector using the official inference-sdk.
    
    Requires: pip install inference-sdk
    
    Advantages over base class:
    - Automatic caching of workflow definitions
    - More efficient for repeated calls
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        workspace: str = RoboflowDetector.DEFAULT_WORKSPACE,
        workflow_id: str = RoboflowDetector.DEFAULT_WORKFLOW_ID,
        confidence_threshold: float = 0.25,
        use_cache: bool = True
    ):
        # Store params without calling parent __init__
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("Roboflow API key required")
        
        self.workspace = workspace
        self.workflow_id = workflow_id
        self.confidence_threshold = confidence_threshold
        self.use_cache = use_cache
        
        try:
            from inference_sdk import InferenceHTTPClient
            self.client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=self.api_key
            )
            print(f"Roboflow SDK Detector initialized (cache={'enabled' if use_cache else 'disabled'})")
        except ImportError:
            raise ImportError(
                "inference-sdk not installed. Install with: pip install inference-sdk\n"
                "Or use RoboflowDetector (requests-based) instead."
            )
    
    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        return_visualization: bool = False
    ) -> Union[List[Detection], Tuple[List[Detection], Optional[np.ndarray]]]:
        """Run detection using SDK."""
        # Handle numpy array - save to temp file
        if isinstance(image, np.ndarray):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                cv2.imwrite(f.name, image)
                image_path = f.name
        else:
            image_path = str(image)
        
        # Get image dimensions
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        # Run workflow
        try:
            result = self.client.run_workflow(
                workspace_name=self.workspace,
                workflow_id=self.workflow_id,
                images={"image": image_path},
                use_cache=self.use_cache
            )
        except Exception as e:
            print(f"SDK inference failed: {e}")
            if return_visualization:
                return [], None
            return []
        
        # Parse results (SDK returns list)
        if isinstance(result, list) and result:
            result = {"outputs": result}
        
        detections = self._parse_predictions(result, width, height)
        
        if return_visualization:
            return detections, None  # SDK visualization handling would go here
        return detections


def main():
    """Test the Roboflow detector."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Roboflow parking detector")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--api-key", help="Roboflow API key (or set ROBOFLOW_API_KEY)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--use-sdk", action="store_true", help="Use inference-sdk")
    args = parser.parse_args()
    
    # Create detector
    if args.use_sdk:
        detector = RoboflowSDKDetector(
            api_key=args.api_key,
            confidence_threshold=args.conf
        )
    else:
        detector = RoboflowDetector(
            api_key=args.api_key,
            confidence_threshold=args.conf
        )
    
    # Run detection
    print(f"\nAnalyzing: {args.image}")
    detections, viz = detector.detect(args.image, return_visualization=True)
    
    print(f"\nFound {len(detections)} vehicles:")
    for det in detections:
        print(f"  - {det.class_name} at {det.bbox} (conf: {det.confidence:.2f})")
    
    # Save visualization if available
    if viz is not None:
        output_path = "roboflow_detection.jpg"
        cv2.imwrite(output_path, viz)
        print(f"\nVisualization saved to: {output_path}")


if __name__ == "__main__":
    main()
