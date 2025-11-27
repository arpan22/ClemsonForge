"""
Smart Parking Detection System - Utilities Package

Provides two detector backends:
1. ParkingDetector - Local YOLOv8 inference (requires ultralytics)
2. RoboflowDetector - Cloud API inference (requires only requests)
"""

from .detector import ParkingDetector, Detection
from .slot_matcher import SlotMatcher, SlotStatus, OccupancyResult
from .visualization import ParkingVisualizer

# Roboflow detector (optional - cloud-based)
try:
    from .roboflow_detector import RoboflowDetector, RoboflowSDKDetector
except ImportError:
    RoboflowDetector = None
    RoboflowSDKDetector = None

__all__ = [
    # Local detector
    'ParkingDetector',
    'Detection',
    # Cloud detector  
    'RoboflowDetector',
    'RoboflowSDKDetector',
    # Slot matching
    'SlotMatcher',
    'SlotStatus',
    'OccupancyResult',
    # Visualization
    'ParkingVisualizer'
]
