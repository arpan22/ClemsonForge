"""
Automatic Parking Slot Detection
---------------------------------
Automatically detects parking spaces in any uploaded image.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DetectedSlot:
    """Represents a detected parking slot."""
    id: str
    polygon: List[List[int]]
    type: str = 'regular'


class AutoSlotDetector:
    """
    Automatically detects parking slots from any parking lot image.
    Uses computer vision to find parking lines and generate slot configurations.
    """

    def __init__(self, min_slot_area: int = 5000, max_slot_area: int = 50000):
        """
        Initialize auto-detector.

        Args:
            min_slot_area: Minimum area for a valid parking slot (pixels²)
            max_slot_area: Maximum area for a valid parking slot (pixels²)
        """
        self.min_slot_area = min_slot_area
        self.max_slot_area = max_slot_area

    def detect_parking_lines(self, image: np.ndarray) -> Tuple[List, List]:
        """
        Detect horizontal and vertical parking lines.

        Returns:
            Tuple of (horizontal_lines, vertical_lines)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while keeping edges sharp
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)

        # Adaptive threshold to handle varying lighting
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Edge detection
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

        # Morphological operations to connect broken lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=80,
            minLineLength=50,
            maxLineGap=20
        )

        if lines is None:
            return [], []

        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Classify as horizontal or vertical
            if angle < 15 or angle > 165:  # Horizontal (±15°)
                horizontal_lines.append(line[0])
            elif 75 < angle < 105:  # Vertical (90° ±15°)
                vertical_lines.append(line[0])

        return horizontal_lines, vertical_lines

    def cluster_lines(self, lines: List, is_horizontal: bool, tolerance: int = 20) -> List[int]:
        """
        Cluster parallel lines and return representative positions.

        Args:
            lines: List of line coordinates
            is_horizontal: True for horizontal lines, False for vertical
            tolerance: Distance tolerance for clustering

        Returns:
            List of line positions (y for horizontal, x for vertical)
        """
        if not lines:
            return []

        # Extract positions
        positions = []
        for line in lines:
            x1, y1, x2, y2 = line
            if is_horizontal:
                positions.append((y1 + y2) // 2)
            else:
                positions.append((x1 + x2) // 2)

        positions.sort()

        # Cluster nearby positions
        clusters = []
        current_cluster = [positions[0]]

        for pos in positions[1:]:
            if pos - current_cluster[-1] <= tolerance:
                current_cluster.append(pos)
            else:
                clusters.append(int(np.mean(current_cluster)))
                current_cluster = [pos]

        clusters.append(int(np.mean(current_cluster)))

        return clusters

    def estimate_grid_from_cars(self, image: np.ndarray) -> Dict:
        """
        Estimate parking grid by detecting cars in the image.
        Fallback method when line detection fails.
        """
        height, width = image.shape[:2]

        # Use simple blob detection for cars
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours (potential cars)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by area
        car_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_slot_area < area < self.max_slot_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / h if h > 0 else 0

                # Cars typically have aspect ratio between 0.4 and 2.5
                if 0.4 < aspect_ratio < 2.5:
                    car_boxes.append((x, y, w, h))

        if not car_boxes:
            # No cars detected, use default grid estimation
            return self.default_grid_estimate(width, height)

        # Analyze car positions to estimate grid
        car_boxes.sort(key=lambda b: (b[1], b[0]))  # Sort by y, then x

        # Calculate average car size
        avg_width = int(np.mean([b[2] for b in car_boxes]))
        avg_height = int(np.mean([b[3] for b in car_boxes]))

        # Estimate spacing from gaps between cars
        x_positions = sorted([b[0] for b in car_boxes])
        y_positions = sorted([b[1] for b in car_boxes])

        x_gaps = np.diff(x_positions)
        y_gaps = np.diff(y_positions)

        h_spacing = int(np.median(x_gaps)) if len(x_gaps) > 0 else 20
        v_spacing = int(np.median(y_gaps)) if len(y_gaps) > 0 else 25

        # Estimate grid parameters
        return {
            'slot_width': avg_width,
            'slot_height': avg_height,
            'h_spacing': max(10, h_spacing // 10),
            'v_spacing': max(10, v_spacing // 10),
            'start_x': min(x_positions) if x_positions else int(width * 0.05),
            'start_y': min(y_positions) if y_positions else int(height * 0.05)
        }

    def default_grid_estimate(self, width: int, height: int) -> Dict:
        """
        Default grid estimation based on standard parking lot proportions.
        """
        # Standard parking space proportions
        parking_area_width = width * 0.90
        parking_area_height = height * 0.85

        # Estimate grid size based on image dimensions
        # Typical parking slot: 2.5m × 5m in real world
        # Aspect ratio: ~0.5

        # Estimate columns and rows
        cols = max(3, min(10, int(width / 150)))  # Adjust based on width
        rows = max(2, min(6, int(height / 200)))   # Adjust based on height

        slot_width = int((parking_area_width * 0.92) / cols)
        slot_height = int((parking_area_height * 0.92) / rows)

        h_spacing = int((parking_area_width * 0.08) / (cols + 1))
        v_spacing = int((parking_area_height * 0.08) / (rows + 1))

        start_x = int((width - parking_area_width) / 2 + h_spacing)
        start_y = int((height - parking_area_height) / 2 + v_spacing)

        return {
            'slot_width': slot_width,
            'slot_height': slot_height,
            'h_spacing': h_spacing,
            'v_spacing': v_spacing,
            'start_x': start_x,
            'start_y': start_y,
            'rows': rows,
            'cols': cols
        }

    def detect_slots(self, image: np.ndarray,
                     force_grid: bool = False) -> Tuple[List[DetectedSlot], Dict]:
        """
        Automatically detect parking slots in an image.

        Args:
            image: Input parking lot image (BGR)
            force_grid: If True, always use grid estimation

        Returns:
            Tuple of (list of DetectedSlot objects, grid parameters dict)
        """
        height, width = image.shape[:2]

        # Try line detection first
        if not force_grid:
            h_lines, v_lines = self.detect_parking_lines(image)

            h_positions = self.cluster_lines(h_lines, is_horizontal=True)
            v_positions = self.cluster_lines(v_lines, is_horizontal=False)

            # Check if we have enough lines to form a grid
            if len(h_positions) >= 2 and len(v_positions) >= 2:
                return self.create_grid_from_lines(
                    h_positions, v_positions, width, height
                )

        # Fallback: estimate from cars or use defaults
        grid_params = self.estimate_grid_from_cars(image)

        if 'rows' not in grid_params or 'cols' not in grid_params:
            # Add default rows/cols if not detected
            grid_params['rows'] = max(2, min(6, int(height / 200)))
            grid_params['cols'] = max(3, min(10, int(width / 150)))

        return self.create_regular_grid(grid_params, width, height), grid_params

    def create_grid_from_lines(self, h_positions: List[int], v_positions: List[int],
                               width: int, height: int) -> Tuple[List[DetectedSlot], Dict]:
        """Create parking slots from detected lines."""
        slots = []
        slot_counter = 0

        rows = len(h_positions) - 1
        cols = len(v_positions) - 1

        for row_idx in range(rows):
            row_letter = chr(ord('A') + row_idx)
            y1 = h_positions[row_idx]
            y2 = h_positions[row_idx + 1]

            for col_idx in range(cols):
                x1 = v_positions[col_idx]
                x2 = v_positions[col_idx + 1]

                slot = DetectedSlot(
                    id=f"{row_letter}{col_idx + 1}",
                    polygon=[
                        [int(x1), int(y1)],
                        [int(x2), int(y1)],
                        [int(x2), int(y2)],
                        [int(x1), int(y2)]
                    ]
                )
                slots.append(slot)
                slot_counter += 1

        grid_params = {
            'rows': rows,
            'cols': cols,
            'slot_width': int(np.mean([v_positions[i+1] - v_positions[i] for i in range(cols)])),
            'slot_height': int(np.mean([h_positions[i+1] - h_positions[i] for i in range(rows)])),
            'start_x': v_positions[0],
            'start_y': h_positions[0]
        }

        return slots, grid_params

    def create_regular_grid(self, params: Dict, width: int, height: int) -> List[DetectedSlot]:
        """Create a regular grid of parking slots."""
        slots = []

        rows = params.get('rows', 3)
        cols = params.get('cols', 6)

        for row in range(rows):
            row_letter = chr(ord('A') + row)
            y = params['start_y'] + row * (params['slot_height'] + params['v_spacing'])

            for col in range(cols):
                x = params['start_x'] + col * (params['slot_width'] + params['h_spacing'])

                # Check bounds
                if (x + params['slot_width'] <= width and
                    y + params['slot_height'] <= height):

                    slot = DetectedSlot(
                        id=f"{row_letter}{col + 1}",
                        polygon=[
                            [int(x), int(y)],
                            [int(x + params['slot_width']), int(y)],
                            [int(x + params['slot_width']), int(y + params['slot_height'])],
                            [int(x), int(y + params['slot_height'])]
                        ]
                    )
                    slots.append(slot)

        return slots

    def slots_to_dict_list(self, slots: List[DetectedSlot]) -> List[Dict]:
        """Convert DetectedSlot objects to dictionary format."""
        return [
            {
                'id': slot.id,
                'polygon': slot.polygon,
                'type': slot.type
            }
            for slot in slots
        ]
