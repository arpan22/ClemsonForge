"""
Simple and Reliable Parking Grid Detector
-----------------------------------------
Creates accurate parking grids based on image analysis.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple


class SimpleGridDetector:
    """
    Simple, reliable parking grid detector.
    Analyzes image and creates accurate grid matching parking spaces.
    """

    def __init__(self):
        pass

    def analyze_image(self, image: np.ndarray) -> Dict:
        """
        Analyze parking lot image to determine optimal grid parameters.

        Args:
            image: BGR image of parking lot

        Returns:
            Dictionary with grid parameters
        """
        height, width = image.shape[:2]

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Detect white/light colored parking lines
        # White in HSV: low saturation, high value
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Detect orange/yellow lines (common in parking lots)
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Combine masks
        line_mask = cv2.bitwise_or(white_mask, orange_mask)

        # Clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, kernel)

        # Find contours to detect parking spaces
        contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Analyze detected features
        if len(contours) > 10:
            # Likely detected individual parking spaces or lines
            return self._analyze_from_contours(contours, width, height)
        else:
            # Fall back to smart estimation
            return self._smart_estimation(width, height)

    def _analyze_from_contours(self, contours: List, width: int, height: int) -> Dict:
        """Analyze parking spaces from detected contours."""

        # Get bounding boxes of potential parking spaces
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            # Filter by area (parking spaces are typically 5000-50000 px²)
            if 2000 < area < 80000:
                aspect_ratio = w / h if h > 0 else 0
                # Parking spaces typically have aspect ratio between 0.3 and 3.0
                if 0.2 < aspect_ratio < 4.0:
                    boxes.append((x, y, w, h))

        if not boxes:
            return self._smart_estimation(width, height)

        # Sort boxes by position (top to bottom, left to right)
        boxes.sort(key=lambda b: (b[1], b[0]))

        # Calculate average dimensions
        avg_width = int(np.median([b[2] for b in boxes]))
        avg_height = int(np.median([b[3] for b in boxes]))

        # Estimate rows and columns
        y_positions = sorted(set([b[1] for b in boxes]))
        x_positions = sorted(set([b[0] for b in boxes]))

        # Cluster positions to find rows/columns
        y_clusters = self._cluster_positions(y_positions, tolerance=avg_height // 2)
        x_clusters = self._cluster_positions(x_positions, tolerance=avg_width // 2)

        rows = len(y_clusters)
        cols = len(x_clusters)

        # Ensure reasonable grid size
        rows = max(2, min(6, rows))
        cols = max(3, min(15, cols))

        # Calculate spacing
        if len(x_clusters) > 1:
            h_spacing = int(np.median(np.diff(x_clusters))) - avg_width
        else:
            h_spacing = int(avg_width * 0.15)

        if len(y_clusters) > 1:
            v_spacing = int(np.median(np.diff(y_clusters))) - avg_height
        else:
            v_spacing = int(avg_height * 0.15)

        h_spacing = max(5, min(50, h_spacing))
        v_spacing = max(5, min(50, v_spacing))

        return {
            'rows': rows,
            'cols': cols,
            'slot_width': avg_width,
            'slot_height': avg_height,
            'start_x': min([b[0] for b in boxes]) if boxes else int(width * 0.05),
            'start_y': min([b[1] for b in boxes]) if boxes else int(height * 0.05),
            'h_spacing': h_spacing,
            'v_spacing': v_spacing
        }

    def _cluster_positions(self, positions: List[int], tolerance: int) -> List[int]:
        """Cluster nearby positions together."""
        if not positions:
            return []

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

    def _smart_estimation(self, width: int, height: int) -> Dict:
        """
        Smart grid estimation based on image dimensions.
        Uses parking lot standards and proportions.
        """
        # Standard parking space: 2.4m × 4.8m (aspect ratio ~0.5)
        # In pixels, estimate based on image size

        # Assume parking area uses 85-90% of image
        parking_width = width * 0.88
        parking_height = height * 0.85

        # Estimate number of columns based on width
        # Typical slot width in overhead view: 80-150 pixels for 1200px wide image
        estimated_slot_width = width / 10  # Rough estimate: 10 slots across
        cols = max(3, min(15, int(parking_width / estimated_slot_width)))

        # Estimate rows based on height
        # Typical slot height: 100-200 pixels for 800px tall image
        estimated_slot_height = height / 4  # Rough estimate: 4 rows down
        rows = max(2, min(6, int(parking_height / estimated_slot_height)))

        # Recalculate slot dimensions based on grid
        slot_width = int((parking_width * 0.92) / cols)
        slot_height = int((parking_height * 0.92) / rows)

        # Calculate spacing
        h_spacing = int((parking_width * 0.08) / (cols + 1))
        v_spacing = int((parking_height * 0.08) / (rows + 1))

        # Calculate starting position
        start_x = int((width - parking_width) / 2 + h_spacing)
        start_y = int((height - parking_height) / 2 + v_spacing)

        return {
            'rows': rows,
            'cols': cols,
            'slot_width': slot_width,
            'slot_height': slot_height,
            'start_x': start_x,
            'start_y': start_y,
            'h_spacing': h_spacing,
            'v_spacing': v_spacing
        }

    def create_grid(self, image: np.ndarray) -> Tuple[List[Dict], Dict]:
        """
        Create parking grid for any image.

        Args:
            image: BGR image of parking lot

        Returns:
            Tuple of (slots list, parameters dict)
        """
        height, width = image.shape[:2]

        # Analyze image to get grid parameters
        params = self.analyze_image(image)

        # Generate grid slots
        slots = []
        for row in range(params['rows']):
            row_letter = chr(ord('A') + row)
            y = params['start_y'] + row * (params['slot_height'] + params['v_spacing'])

            for col in range(params['cols']):
                x = params['start_x'] + col * (params['slot_width'] + params['h_spacing'])

                # Check bounds
                if (x + params['slot_width'] <= width and
                    y + params['slot_height'] <= height and
                    x >= 0 and y >= 0):

                    slot = {
                        'id': f"{row_letter}{col + 1}",
                        'polygon': [
                            [int(x), int(y)],
                            [int(x + params['slot_width']), int(y)],
                            [int(x + params['slot_width']), int(y + params['slot_height'])],
                            [int(x), int(y + params['slot_height'])]
                        ],
                        'type': 'regular'
                    }
                    slots.append(slot)

        return slots, params
