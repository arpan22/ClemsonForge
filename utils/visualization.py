"""
Visualization Module
--------------------
Drawing utilities for parking lot visualization.

Provides:
- Detection overlays on images
- Slot status visualization (colored polygons)
- Schematic map views
- Combined display generation

Usage:
    from utils.visualization import ParkingVisualizer
    
    viz = ParkingVisualizer(config_path="configs/lot_config.yaml")
    annotated_img = viz.draw_detections(image, detections)
    map_img = viz.draw_occupancy_map(result)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import yaml

from .detector import Detection
from .slot_matcher import SlotMatcher, OccupancyResult, SlotStatus, ParkingSlot


class ParkingVisualizer:
    """
    Visualization utilities for parking detection results.
    
    Args:
        config_path: Path to lot configuration (for visualization settings)
        empty_color: BGR color for empty slots (default: green)
        occupied_color: BGR color for occupied slots (default: red)
        unknown_color: BGR color for unknown slots (default: gray)
        alpha: Transparency for slot overlays (0-1)
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        empty_color: Tuple[int, int, int] = (0, 255, 0),      # Green
        occupied_color: Tuple[int, int, int] = (0, 0, 255),   # Red
        unknown_color: Tuple[int, int, int] = (128, 128, 128), # Gray
        alpha: float = 0.4
    ):
        self.empty_color = empty_color
        self.occupied_color = occupied_color
        self.unknown_color = unknown_color
        self.alpha = alpha
        self.border_thickness = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        
        if config_path:
            self._load_viz_config(config_path)
    
    def _load_viz_config(self, config_path: str):
        """Load visualization settings from config file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        viz_config = config.get('visualization', {})
        
        if 'empty_color' in viz_config:
            self.empty_color = tuple(viz_config['empty_color'])
        if 'occupied_color' in viz_config:
            self.occupied_color = tuple(viz_config['occupied_color'])
        if 'unknown_color' in viz_config:
            self.unknown_color = tuple(viz_config['unknown_color'])
        if 'slot_alpha' in viz_config:
            self.alpha = viz_config['slot_alpha']
        if 'border_thickness' in viz_config:
            self.border_thickness = viz_config['border_thickness']
        if 'font_scale' in viz_config:
            self.font_scale = viz_config['font_scale']
    
    def get_status_color(self, status: SlotStatus) -> Tuple[int, int, int]:
        """Get color for a given slot status."""
        if status == SlotStatus.EMPTY:
            return self.empty_color
        elif status == SlotStatus.OCCUPIED:
            return self.occupied_color
        else:
            return self.unknown_color
    
    def draw_detections(
        self,
        image: Union[str, Path, np.ndarray],
        detections: List[Detection],
        show_labels: bool = True,
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Draw detection bounding boxes on image.
        
        Args:
            image: Input image (path or array)
            detections: List of Detection objects
            show_labels: Whether to show class labels
            show_confidence: Whether to show confidence scores
            
        Returns:
            Annotated image as numpy array
        """
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        else:
            img = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 255, 255)  # Yellow for detections
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            if show_labels or show_confidence:
                # Build label text
                parts = []
                if show_labels:
                    parts.append(det.class_name)
                if show_confidence:
                    parts.append(f"{det.confidence:.0%}")
                label = " ".join(parts)
                
                # Draw label background
                (w, h), _ = cv2.getTextSize(label, self.font, self.font_scale, 1)
                cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w + 5, y1), color, -1)
                
                # Draw label text
                cv2.putText(img, label, (x1 + 2, y1 - 5),
                           self.font, self.font_scale, (0, 0, 0), 1)
        
        return img
    
    def draw_slots(
        self,
        image: Union[str, Path, np.ndarray],
        slots: Dict[str, ParkingSlot],
        show_ids: bool = True,
        fill: bool = True
    ) -> np.ndarray:
        """
        Draw parking slots with status colors on image.
        
        Args:
            image: Input image (path or array)
            slots: Dictionary of slot_id -> ParkingSlot
            show_ids: Whether to show slot IDs
            fill: Whether to fill slots with transparent color
            
        Returns:
            Annotated image as numpy array
        """
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        else:
            img = image.copy()
        
        # Create overlay for transparent fills
        overlay = img.copy()
        
        for slot_id, slot in slots.items():
            color = self.get_status_color(slot.status)
            pts = np.array(slot.polygon, np.int32).reshape((-1, 1, 2))
            
            if fill:
                # Fill polygon
                cv2.fillPoly(overlay, [pts], color)
            
            # Draw border
            cv2.polylines(img, [pts], True, color, self.border_thickness)
            
            if show_ids:
                # Draw slot ID at center
                cx, cy = slot.center
                (w, h), _ = cv2.getTextSize(slot_id, self.font, self.font_scale, 1)
                cv2.putText(img, slot_id, (cx - w // 2, cy + h // 2),
                           self.font, self.font_scale, (255, 255, 255), 2)
                cv2.putText(img, slot_id, (cx - w // 2, cy + h // 2),
                           self.font, self.font_scale, (0, 0, 0), 1)
        
        # Blend overlay
        if fill:
            img = cv2.addWeighted(overlay, self.alpha, img, 1 - self.alpha, 0)
        
        return img
    
    def draw_full_overlay(
        self,
        image: Union[str, Path, np.ndarray],
        detections: List[Detection],
        occupancy_result: OccupancyResult,
        show_detections: bool = True,
        show_slot_ids: bool = True
    ) -> np.ndarray:
        """
        Draw complete visualization with detections and slot statuses.
        
        Args:
            image: Input image
            detections: List of Detection objects
            occupancy_result: Result from SlotMatcher
            show_detections: Whether to show detection boxes
            show_slot_ids: Whether to show slot IDs
            
        Returns:
            Fully annotated image
        """
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        else:
            img = image.copy()
        
        # Draw slots first (underneath)
        img = self.draw_slots(img, occupancy_result.slots, 
                             show_ids=show_slot_ids, fill=True)
        
        # Draw detections on top
        if show_detections:
            img = self.draw_detections(img, detections,
                                       show_labels=True, show_confidence=True)
        
        # Add summary text
        img = self._add_summary(img, occupancy_result)
        
        return img
    
    def _add_summary(
        self,
        image: np.ndarray,
        result: OccupancyResult,
        position: str = "top"
    ) -> np.ndarray:
        """Add summary statistics to image."""
        img = image.copy()
        
        # Summary text
        text = (f"Total: {result.total_slots} | "
                f"Empty: {result.empty_slots} | "
                f"Occupied: {result.occupied_slots} | "
                f"Rate: {result.occupancy_rate:.0%}")
        
        # Calculate position
        (w, h), _ = cv2.getTextSize(text, self.font, 0.7, 2)
        
        if position == "top":
            y = 30
        else:
            y = img.shape[0] - 20
        
        x = 10
        
        # Draw background
        cv2.rectangle(img, (x - 5, y - h - 10), (x + w + 10, y + 10), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(img, text, (x, y), self.font, 0.7, (255, 255, 255), 2)
        
        return img
    
    def create_schematic_map(
        self,
        slots: Dict[str, ParkingSlot],
        width: int = 800,
        height: int = 600,
        padding: int = 50,
        show_ids: bool = True,
        title: Optional[str] = None
    ) -> np.ndarray:
        """
        Create a clean schematic map view of the parking lot.
        
        This creates a simplified bird's-eye view without the actual image,
        showing just the slot layout and occupancy status.
        
        Args:
            slots: Dictionary of parking slots
            width, height: Output image dimensions
            padding: Padding around the map
            show_ids: Whether to show slot IDs
            title: Optional title to display
            
        Returns:
            Schematic map as numpy array
        """
        # Create white background
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        if not slots:
            cv2.putText(img, "No slots configured", (width // 4, height // 2),
                       self.font, 1, (128, 128, 128), 2)
            return img
        
        # Find bounding box of all slots
        all_x = []
        all_y = []
        for slot in slots.values():
            for x, y in slot.polygon:
                all_x.append(x)
                all_y.append(y)
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # Calculate scale to fit in image
        slot_width = max_x - min_x
        slot_height = max_y - min_y
        
        available_width = width - 2 * padding
        available_height = height - 2 * padding - (40 if title else 0)
        
        scale_x = available_width / slot_width if slot_width > 0 else 1
        scale_y = available_height / slot_height if slot_height > 0 else 1
        scale = min(scale_x, scale_y)
        
        # Offset to center
        offset_x = padding + (available_width - slot_width * scale) / 2 - min_x * scale
        offset_y = padding + (available_height - slot_height * scale) / 2 - min_y * scale
        if title:
            offset_y += 40
        
        # Draw title
        if title:
            cv2.putText(img, title, (padding, 30), self.font, 0.8, (0, 0, 0), 2)
        
        # Draw slots
        for slot_id, slot in slots.items():
            # Transform polygon coordinates
            transformed = []
            for x, y in slot.polygon:
                tx = int(x * scale + offset_x)
                ty = int(y * scale + offset_y)
                transformed.append([tx, ty])
            
            pts = np.array(transformed, np.int32).reshape((-1, 1, 2))
            color = self.get_status_color(slot.status)
            
            # Fill
            cv2.fillPoly(img, [pts], color)
            
            # Border
            cv2.polylines(img, [pts], True, (0, 0, 0), 2)
            
            # ID label
            if show_ids:
                center_x = int(sum(p[0] for p in transformed) / len(transformed))
                center_y = int(sum(p[1] for p in transformed) / len(transformed))
                
                (w, h), _ = cv2.getTextSize(slot_id, self.font, 0.4, 1)
                cv2.putText(img, slot_id, (center_x - w // 2, center_y + h // 2),
                           self.font, 0.4, (255, 255, 255), 2)
                cv2.putText(img, slot_id, (center_x - w // 2, center_y + h // 2),
                           self.font, 0.4, (0, 0, 0), 1)
        
        # Add legend
        self._draw_legend(img, width, height)
        
        return img
    
    def _draw_legend(self, img: np.ndarray, width: int, height: int):
        """Draw color legend on image."""
        legend_items = [
            ("Empty", self.empty_color),
            ("Occupied", self.occupied_color),
        ]
        
        x = width - 120
        y = height - 60
        
        for i, (label, color) in enumerate(legend_items):
            ly = y + i * 25
            
            # Color box
            cv2.rectangle(img, (x, ly), (x + 15, ly + 15), color, -1)
            cv2.rectangle(img, (x, ly), (x + 15, ly + 15), (0, 0, 0), 1)
            
            # Label
            cv2.putText(img, label, (x + 20, ly + 12),
                       self.font, 0.45, (0, 0, 0), 1)
    
    def create_side_by_side(
        self,
        original: np.ndarray,
        annotated: np.ndarray,
        schematic: np.ndarray
    ) -> np.ndarray:
        """
        Create a side-by-side comparison view.
        
        Args:
            original: Original image
            annotated: Image with detection overlays
            schematic: Schematic map
            
        Returns:
            Combined image
        """
        # Resize all to same height
        target_height = 600
        
        def resize_to_height(img, h):
            aspect = img.shape[1] / img.shape[0]
            new_w = int(h * aspect)
            return cv2.resize(img, (new_w, h))
        
        original = resize_to_height(original, target_height)
        annotated = resize_to_height(annotated, target_height)
        schematic = resize_to_height(schematic, target_height)
        
        # Stack horizontally
        combined = np.hstack([annotated, schematic])
        
        return combined


def main():
    """Test visualization utilities."""
    from .slot_matcher import create_grid_slots, SlotMatcher
    from .detector import Detection
    
    # Create test slots
    slots_data = create_grid_slots(
        start_x=100, start_y=100,
        slot_width=80, slot_height=100,
        rows=3, cols=5
    )
    
    matcher = SlotMatcher()
    matcher.set_slots_from_list(slots_data)
    
    # Create fake detections
    detections = [
        Detection(bbox=(110, 110, 180, 200), confidence=0.9, class_id=0, class_name="car"),
        Detection(bbox=(200, 110, 270, 200), confidence=0.85, class_id=0, class_name="car"),
        Detection(bbox=(110, 230, 180, 320), confidence=0.95, class_id=0, class_name="car"),
    ]
    
    # Run matching
    result = matcher.match(detections)
    
    # Create visualizer
    viz = ParkingVisualizer()
    
    # Create schematic map
    schematic = viz.create_schematic_map(
        result.slots,
        width=800,
        height=600,
        title="Test Parking Lot"
    )
    
    # Save test output
    cv2.imwrite("test_schematic.jpg", schematic)
    print("Saved test_schematic.jpg")
    
    print(f"\nVisualization test complete!")
    print(f"Slots: {len(result.slots)}")
    print(f"Empty: {result.empty_slots}")
    print(f"Occupied: {result.occupied_slots}")


if __name__ == "__main__":
    main()
