#!/usr/bin/env python3
"""
Quick Parking Slot Configuration Generator
------------------------------------------
Automatically generate parking slot configurations for demo images.

Usage:
    python scripts/quick_config.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import yaml
import numpy as np


def create_config_for_image(image_path, output_path, rows=4, cols=6):
    """Create a parking slot configuration for an image."""

    # Load image to get dimensions
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load {image_path}")
        return False

    img_height, img_width = img.shape[:2]

    # Calculate slot dimensions based on image size
    # Use 85% of width and 80% of height for parking area
    available_width = int(img_width * 0.85)
    available_height = int(img_height * 0.80)

    slot_width = int(available_width / cols) - 15
    slot_height = int(available_height / rows) - 25

    # Starting position (centered)
    start_x = int(img_width * 0.075)
    start_y = int(img_height * 0.10)

    # Generate slots
    slots = []
    for row in range(rows):
        row_letter = chr(ord('A') + row)
        y = start_y + row * (slot_height + 25)

        for col in range(cols):
            x = start_x + col * (slot_width + 15)

            # Check bounds
            if (x + slot_width <= img_width and y + slot_height <= img_height):
                slot = {
                    'id': f"{row_letter}{col + 1}",
                    'polygon': [
                        [int(x), int(y)],
                        [int(x + slot_width), int(y)],
                        [int(x + slot_width), int(y + slot_height)],
                        [int(x), int(y + slot_height)]
                    ],
                    'type': 'regular'
                }
                slots.append(slot)

    # Create config
    config = {
        'lot_name': f"{image_path.stem}",
        'description': f'Auto-generated configuration for {image_path.name}',
        'image_width': img_width,
        'image_height': img_height,
        'detection': {
            'confidence_threshold': 0.25,
            'iou_threshold': 0.3,
            'model': 'yolov8m.pt'
        },
        'slots': slots,
        'visualization': {
            'empty_color': [0, 255, 0],
            'occupied_color': [0, 0, 255],
            'unknown_color': [128, 128, 128],
            'slot_alpha': 0.4,
            'border_thickness': 2,
            'font_scale': 0.5
        }
    }

    # Save config
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=None, sort_keys=False)

    print(f"✓ Created config: {output_path.name} ({len(slots)} slots)")

    # Create visualization
    vis_img = img.copy()
    for slot in slots:
        pts = np.array(slot['polygon'], np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_img, [pts], True, (0, 255, 0), 2)

        # Draw slot ID
        center = np.mean(slot['polygon'], axis=0).astype(int)
        cv2.putText(vis_img, slot['id'], tuple(center),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Save visualization
    vis_path = output_path.parent / f"{output_path.stem}_preview.jpg"
    cv2.imwrite(str(vis_path), vis_img)
    print(f"  Preview: {vis_path.name}")

    return True


def main():
    """Generate configs for all demo images."""
    project_root = Path(__file__).parent.parent
    demo_dir = project_root / "demo_images"
    config_dir = project_root / "configs"

    if not demo_dir.exists():
        print(f"Error: Demo images directory not found: {demo_dir}")
        return

    # Find all demo images
    demo_images = list(demo_dir.glob("*.jpg")) + list(demo_dir.glob("*.png"))

    if not demo_images:
        print("No demo images found")
        return

    print(f"\nGenerating configurations for {len(demo_images)} demo images...\n")

    for img_path in demo_images:
        config_path = config_dir / f"{img_path.stem}_config.yaml"
        create_config_for_image(img_path, config_path, rows=4, cols=6)

    print(f"\n✓ Complete! Generated {len(demo_images)} configurations")
    print(f"  Location: {config_dir}/")
    print(f"\nTo use these configs:")
    print(f"  1. Go to the Streamlit app")
    print(f"  2. Upload one of the generated YAML files")
    print(f"  3. Select the corresponding demo image")
    print(f"  4. Click 'Detect Parking Spaces'\n")


if __name__ == "__main__":
    main()
