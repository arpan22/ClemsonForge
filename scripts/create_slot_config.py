#!/usr/bin/env python3
"""
Parking Slot Configuration Tool
-------------------------------
Interactive tool to define parking slot polygons on an image.

Usage:
    python create_slot_config.py --image parking_lot.jpg --output configs/my_lot.yaml

Controls:
    - Left click: Add polygon vertex
    - Right click: Complete current polygon
    - 'u': Undo last vertex
    - 'd': Delete last completed slot
    - 's': Save configuration
    - 'g': Generate grid of slots (auto mode)
    - 'q': Quit
    - 'r': Reset all
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple
import yaml

try:
    import cv2
    import numpy as np
except ImportError:
    print("Please install opencv-python: pip install opencv-python")
    sys.exit(1)


class SlotConfigTool:
    """Interactive tool for creating parking slot configurations."""
    
    def __init__(self, image_path: str, output_path: str, lot_name: str = "Parking Lot"):
        self.image_path = Path(image_path)
        self.output_path = Path(output_path)
        self.lot_name = lot_name
        
        # Load image
        self.original_image = cv2.imread(str(self.image_path))
        if self.original_image is None:
            raise ValueError(f"Could not load image: {self.image_path}")
        
        self.image_height, self.image_width = self.original_image.shape[:2]
        self.display_image = self.original_image.copy()
        
        # Slot data
        self.slots: List[dict] = []
        self.current_polygon: List[Tuple[int, int]] = []
        self.slot_counter = 0
        self.current_row = 'A'
        
        # Colors
        self.color_complete = (0, 255, 0)    # Green
        self.color_current = (0, 255, 255)   # Yellow
        self.color_vertex = (255, 0, 0)      # Blue
        
        # Window name
        self.window_name = "Parking Slot Configuration Tool"
    
    def run(self):
        """Start the interactive tool."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, min(1400, self.image_width), 
                        min(900, self.image_height))
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        self._print_instructions()
        
        while True:
            self._update_display()
            cv2.imshow(self.window_name, self.display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_config()
            elif key == ord('u'):
                self._undo_vertex()
            elif key == ord('d'):
                self._delete_last_slot()
            elif key == ord('r'):
                self._reset_all()
            elif key == ord('g'):
                self._generate_grid()
            elif key == ord('h'):
                self._print_instructions()
        
        cv2.destroyAllWindows()
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add vertex
            self.current_polygon.append((x, y))
            print(f"Added vertex at ({x}, {y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Complete polygon
            if len(self.current_polygon) >= 3:
                self._complete_slot()
            else:
                print("Need at least 3 vertices to complete a slot")
    
    def _complete_slot(self):
        """Complete current polygon as a slot."""
        self.slot_counter += 1
        slot_id = f"{self.current_row}{self.slot_counter}"
        
        slot = {
            'id': slot_id,
            'polygon': [list(p) for p in self.current_polygon],
            'type': 'regular'
        }
        
        self.slots.append(slot)
        self.current_polygon = []
        
        print(f"✓ Completed slot: {slot_id}")
        
        # Increment row letter after every 10 slots (or customize)
        if self.slot_counter >= 10:
            self.slot_counter = 0
            self.current_row = chr(ord(self.current_row) + 1)
    
    def _undo_vertex(self):
        """Remove last vertex from current polygon."""
        if self.current_polygon:
            removed = self.current_polygon.pop()
            print(f"Removed vertex at {removed}")
        else:
            print("No vertices to undo")
    
    def _delete_last_slot(self):
        """Delete the last completed slot."""
        if self.slots:
            removed = self.slots.pop()
            print(f"Deleted slot: {removed['id']}")
        else:
            print("No slots to delete")
    
    def _reset_all(self):
        """Reset all slots and current polygon."""
        self.slots = []
        self.current_polygon = []
        self.slot_counter = 0
        self.current_row = 'A'
        print("Reset all slots")
    
    def _generate_grid(self):
        """Generate a grid of slots automatically."""
        print("\n--- Grid Generation Mode ---")
        print("Enter grid parameters (press Enter for defaults):")
        
        try:
            start_x = int(input(f"Start X [{100}]: ") or "100")
            start_y = int(input(f"Start Y [{100}]: ") or "100")
            slot_width = int(input(f"Slot width [{80}]: ") or "80")
            slot_height = int(input(f"Slot height [{100}]: ") or "100")
            rows = int(input(f"Number of rows [{3}]: ") or "3")
            cols = int(input(f"Number of columns [{5}]: ") or "5")
            h_spacing = int(input(f"Horizontal spacing [{10}]: ") or "10")
            v_spacing = int(input(f"Vertical spacing [{20}]: ") or "20")
        except ValueError:
            print("Invalid input, using defaults")
            start_x, start_y = 100, 100
            slot_width, slot_height = 80, 100
            rows, cols = 3, 5
            h_spacing, v_spacing = 10, 20
        
        # Generate grid
        for row in range(rows):
            row_letter = chr(ord('A') + row)
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
                self.slots.append(slot)
        
        print(f"✓ Generated {rows * cols} slots in a {rows}x{cols} grid")
    
    def _update_display(self):
        """Update the display image with current state."""
        self.display_image = self.original_image.copy()
        
        # Draw completed slots
        for slot in self.slots:
            pts = np.array(slot['polygon'], np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(self.display_image, [pts], self.color_complete + (50,))
            cv2.polylines(self.display_image, [pts], True, self.color_complete, 2)
            
            # Draw slot ID
            center = np.mean(slot['polygon'], axis=0).astype(int)
            cv2.putText(self.display_image, slot['id'], tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(self.display_image, slot['id'], tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw current polygon
        if self.current_polygon:
            for point in self.current_polygon:
                cv2.circle(self.display_image, point, 5, self.color_vertex, -1)
            
            if len(self.current_polygon) > 1:
                pts = np.array(self.current_polygon, np.int32).reshape((-1, 1, 2))
                cv2.polylines(self.display_image, [pts], False, self.color_current, 2)
        
        # Draw info bar
        info = f"Slots: {len(self.slots)} | Current: {len(self.current_polygon)} vertices | Press 'h' for help"
        cv2.rectangle(self.display_image, (0, 0), (self.image_width, 30), (0, 0, 0), -1)
        cv2.putText(self.display_image, info, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _save_config(self):
        """Save the configuration to YAML file."""
        if not self.slots:
            print("No slots to save")
            return
        
        config = {
            'lot_name': self.lot_name,
            'description': f'Generated from {self.image_path.name}',
            'image_width': self.image_width,
            'image_height': self.image_height,
            'detection': {
                'confidence_threshold': 0.25,
                'iou_threshold': 0.3,
                'model': 'yolov8m.pt'
            },
            'slots': self.slots,
            'visualization': {
                'empty_color': [0, 255, 0],
                'occupied_color': [0, 0, 255],
                'unknown_color': [128, 128, 128],
                'slot_alpha': 0.4,
                'border_thickness': 2,
                'font_scale': 0.5
            }
        }
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=None, sort_keys=False)
        
        print(f"\n✓ Saved configuration to: {self.output_path}")
        print(f"  Total slots: {len(self.slots)}")
    
    def _print_instructions(self):
        """Print usage instructions."""
        print("\n" + "="*50)
        print("PARKING SLOT CONFIGURATION TOOL")
        print("="*50)
        print("\nControls:")
        print("  Left click  - Add polygon vertex")
        print("  Right click - Complete current polygon")
        print("  'u'         - Undo last vertex")
        print("  'd'         - Delete last completed slot")
        print("  's'         - Save configuration")
        print("  'g'         - Generate grid of slots")
        print("  'r'         - Reset all")
        print("  'h'         - Show this help")
        print("  'q'         - Quit")
        print("\nTip: Draw slots clockwise or counter-clockwise")
        print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive tool to create parking slot configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python create_slot_config.py --image lot.jpg --output configs/lot.yaml
    
    # With custom lot name
    python create_slot_config.py --image lot.jpg --output configs/lot.yaml --name "Main Campus Lot"
        """
    )
    
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to parking lot image"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output path for YAML config"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default="Parking Lot",
        help="Name of the parking lot"
    )
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    tool = SlotConfigTool(
        image_path=args.image,
        output_path=args.output,
        lot_name=args.name
    )
    tool.run()


if __name__ == "__main__":
    main()
