"""
Slot Matcher Module
-------------------
Maps car detections to predefined parking slot polygons.

Uses IoU (Intersection over Union) to determine slot occupancy.

Usage:
    from utils.slot_matcher import SlotMatcher
    
    matcher = SlotMatcher(config_path="configs/lot_config.yaml")
    results = matcher.match(detections)
"""

import yaml
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union

try:
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
except ImportError:
    raise ImportError("Please install shapely: pip install shapely==2.0.3")

from .detector import Detection


class SlotStatus(Enum):
    """Parking slot occupancy status."""
    EMPTY = "empty"
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"


@dataclass
class ParkingSlot:
    """Represents a single parking slot."""
    id: str
    polygon: List[Tuple[int, int]]
    slot_type: str = "regular"
    status: SlotStatus = SlotStatus.UNKNOWN
    occupancy_confidence: float = 0.0
    matched_detection: Optional[Detection] = None
    
    @property
    def shapely_polygon(self) -> Polygon:
        """Convert to Shapely polygon for geometry operations."""
        return Polygon(self.polygon)
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of slot."""
        poly = self.shapely_polygon
        return (int(poly.centroid.x), int(poly.centroid.y))
    
    @property
    def area(self) -> float:
        """Get area of slot in pixels."""
        return self.shapely_polygon.area


@dataclass
class OccupancyResult:
    """Results of occupancy detection for entire lot."""
    slots: Dict[str, ParkingSlot] = field(default_factory=dict)
    total_slots: int = 0
    occupied_slots: int = 0
    empty_slots: int = 0
    unknown_slots: int = 0
    occupancy_rate: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "total": self.total_slots,
                "occupied": self.occupied_slots,
                "empty": self.empty_slots,
                "unknown": self.unknown_slots,
                "occupancy_rate": round(self.occupancy_rate * 100, 1)
            },
            "slots": {
                slot_id: {
                    "status": slot.status.value,
                    "type": slot.slot_type,
                    "confidence": round(slot.occupancy_confidence, 2),
                    "center": slot.center
                }
                for slot_id, slot in self.slots.items()
            }
        }


class SlotMatcher:
    """
    Matches car detections to parking slot polygons.
    
    Uses IoU (Intersection over Union) between detection bounding boxes
    and slot polygons to determine occupancy.
    
    Args:
        config_path: Path to lot configuration YAML file
        iou_threshold: Minimum IoU to consider a slot occupied (default: 0.3)
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[dict] = None,
        iou_threshold: float = 0.3
    ):
        self.iou_threshold = iou_threshold
        self.slots: Dict[str, ParkingSlot] = {}
        self.config = {}
        
        if config_path:
            self.load_config(config_path)
        elif config_dict:
            self._parse_config(config_dict)
    
    def load_config(self, config_path: Union[str, Path]):
        """Load lot configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self._parse_config(config)
    
    def _parse_config(self, config: dict):
        """Parse configuration dictionary."""
        self.config = config
        self.lot_name = config.get('lot_name', 'Unknown Lot')
        self.image_width = config.get('image_width', 1920)
        self.image_height = config.get('image_height', 1080)
        
        # Parse detection settings
        detection_config = config.get('detection', {})
        self.iou_threshold = detection_config.get('iou_threshold', self.iou_threshold)
        
        # Parse slots
        self.slots = {}
        for slot_data in config.get('slots', []):
            slot = ParkingSlot(
                id=slot_data['id'],
                polygon=[(p[0], p[1]) for p in slot_data['polygon']],
                slot_type=slot_data.get('type', 'regular')
            )
            self.slots[slot.id] = slot
        
        print(f"Loaded {len(self.slots)} parking slots for '{self.lot_name}'")
    
    def set_slots_from_list(self, slots_data: List[dict]):
        """
        Set slots directly from a list of slot definitions.
        
        Args:
            slots_data: List of dicts with 'id', 'polygon', and optional 'type'
        """
        self.slots = {}
        for slot_data in slots_data:
            slot = ParkingSlot(
                id=slot_data['id'],
                polygon=[(p[0], p[1]) for p in slot_data['polygon']],
                slot_type=slot_data.get('type', 'regular')
            )
            self.slots[slot.id] = slot
    
    def match(self, detections: List[Detection]) -> OccupancyResult:
        """
        Match detections to parking slots and determine occupancy.
        
        Args:
            detections: List of Detection objects from ParkingDetector
            
        Returns:
            OccupancyResult with slot statuses and summary statistics
        """
        # Reset all slots to empty
        for slot in self.slots.values():
            slot.status = SlotStatus.EMPTY
            slot.occupancy_confidence = 0.0
            slot.matched_detection = None
        
        # Convert detections to Shapely boxes
        detection_boxes = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            det_box = box(x1, y1, x2, y2)
            detection_boxes.append((det, det_box))
        
        # Match each detection to slots
        for det, det_box in detection_boxes:
            best_iou = 0.0
            best_slot = None
            
            for slot_id, slot in self.slots.items():
                slot_poly = slot.shapely_polygon
                
                # Calculate IoU
                try:
                    intersection = det_box.intersection(slot_poly).area
                    union = det_box.union(slot_poly).area
                    iou = intersection / union if union > 0 else 0
                except Exception:
                    iou = 0
                
                # Also check if detection center is within slot
                det_center = det.center
                center_in_slot = slot_poly.contains(
                    Polygon([(det_center[0]-1, det_center[1]-1),
                            (det_center[0]+1, det_center[1]-1),
                            (det_center[0]+1, det_center[1]+1),
                            (det_center[0]-1, det_center[1]+1)])
                )
                
                # Boost IoU if center is in slot
                if center_in_slot:
                    iou = max(iou, 0.5)
                
                if iou > best_iou:
                    best_iou = iou
                    best_slot = slot
            
            # Assign detection to best matching slot
            if best_slot and best_iou >= self.iou_threshold:
                # Only update if this detection has higher confidence
                if best_iou > best_slot.occupancy_confidence:
                    best_slot.status = SlotStatus.OCCUPIED
                    best_slot.occupancy_confidence = best_iou
                    best_slot.matched_detection = det
        
        # Calculate statistics
        occupied = sum(1 for s in self.slots.values() if s.status == SlotStatus.OCCUPIED)
        empty = sum(1 for s in self.slots.values() if s.status == SlotStatus.EMPTY)
        unknown = sum(1 for s in self.slots.values() if s.status == SlotStatus.UNKNOWN)
        total = len(self.slots)
        
        return OccupancyResult(
            slots=self.slots.copy(),
            total_slots=total,
            occupied_slots=occupied,
            empty_slots=empty,
            unknown_slots=unknown,
            occupancy_rate=occupied / total if total > 0 else 0
        )
    
    def get_slot_ids_by_status(self, status: SlotStatus) -> List[str]:
        """Get list of slot IDs with given status."""
        return [sid for sid, slot in self.slots.items() if slot.status == status]
    
    def get_empty_slots(self) -> List[str]:
        """Get list of empty slot IDs."""
        return self.get_slot_ids_by_status(SlotStatus.EMPTY)
    
    def get_occupied_slots(self) -> List[str]:
        """Get list of occupied slot IDs."""
        return self.get_slot_ids_by_status(SlotStatus.OCCUPIED)


def create_grid_slots(
    start_x: int,
    start_y: int,
    slot_width: int,
    slot_height: int,
    rows: int,
    cols: int,
    h_spacing: int = 10,
    v_spacing: int = 20,
    row_prefix: str = ""
) -> List[dict]:
    """
    Generate a grid of parking slots.
    
    Args:
        start_x, start_y: Top-left corner of grid
        slot_width, slot_height: Size of each slot
        rows, cols: Number of rows and columns
        h_spacing, v_spacing: Horizontal and vertical gaps between slots
        row_prefix: Prefix for slot IDs (e.g., "A" -> "A1", "A2", ...)
        
    Returns:
        List of slot definitions suitable for SlotMatcher
    """
    slots = []
    
    for row in range(rows):
        row_letter = chr(ord('A') + row) if not row_prefix else row_prefix
        y = start_y + row * (slot_height + v_spacing)
        
        for col in range(cols):
            x = start_x + col * (slot_width + h_spacing)
            
            slot = {
                'id': f"{row_letter}{col + 1}",
                'polygon': [
                    [x, y],
                    [x + slot_width, y],
                    [x + slot_width, y + slot_height],
                    [x, y + slot_height]
                ],
                'type': 'regular'
            }
            slots.append(slot)
    
    return slots


def main():
    """Test slot matcher with sample data."""
    # Create sample slots
    slots = create_grid_slots(
        start_x=100, start_y=100,
        slot_width=80, slot_height=100,
        rows=3, cols=5,
        h_spacing=10, v_spacing=20
    )
    
    print(f"Created {len(slots)} slots")
    
    # Create matcher
    matcher = SlotMatcher()
    matcher.set_slots_from_list(slots)
    
    # Create fake detections for testing
    fake_detections = [
        Detection(bbox=(110, 110, 180, 200), confidence=0.9, class_id=0, class_name="car"),
        Detection(bbox=(300, 110, 370, 200), confidence=0.85, class_id=0, class_name="car"),
        Detection(bbox=(110, 230, 180, 320), confidence=0.95, class_id=0, class_name="car"),
    ]
    
    # Run matching
    result = matcher.match(fake_detections)
    
    print(f"\nOccupancy Result:")
    print(f"  Total slots: {result.total_slots}")
    print(f"  Occupied: {result.occupied_slots}")
    print(f"  Empty: {result.empty_slots}")
    print(f"  Occupancy rate: {result.occupancy_rate:.1%}")
    
    print(f"\nEmpty slots: {matcher.get_empty_slots()}")
    print(f"Occupied slots: {matcher.get_occupied_slots()}")


if __name__ == "__main__":
    main()
