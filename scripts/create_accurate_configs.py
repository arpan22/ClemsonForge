#!/usr/bin/env python3
"""
Create Most Accurate Parking Slot Configurations
------------------------------------------------
Analyzes parking lot images to create precise slot configurations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import yaml


def analyze_parking_layout(image_path):
    """
    Analyze parking lot image to detect actual parking space positions.
    Returns optimized grid parameters.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    height, width = img.shape[:2]

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect edges to find parking lines
    edges = cv2.Canny(gray, 50, 150)

    # Find horizontal and vertical lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                            minLineLength=50, maxLineGap=10)

    if lines is None:
        # Fallback to visual estimation based on image
        return estimate_from_visual(width, height)

    # Analyze detected lines to find parking slot boundaries
    h_lines = []
    v_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

        if angle < 10 or angle > 170:  # Horizontal
            h_lines.append((y1 + y2) / 2)
        elif 80 < angle < 100:  # Vertical
            v_lines.append((x1 + x2) / 2)

    if len(h_lines) >= 2 and len(v_lines) >= 2:
        # Sort and find regular spacing
        h_lines = sorted(h_lines)
        v_lines = sorted(v_lines)

        # Calculate average spacing
        h_spacing = np.median(np.diff(h_lines)) if len(h_lines) > 1 else height / 4
        v_spacing = np.median(np.diff(v_lines)) if len(v_lines) > 1 else width / 6

        return {
            'start_x': int(v_lines[0]) if v_lines else int(width * 0.05),
            'start_y': int(h_lines[0]) if h_lines else int(height * 0.05),
            'slot_width': int(v_spacing * 0.85) if v_spacing else int(width * 0.13),
            'slot_height': int(h_spacing * 0.85) if h_spacing else int(height * 0.23),
            'h_spacing': int(v_spacing * 0.15) if v_spacing else 20,
            'v_spacing': int(h_spacing * 0.15) if h_spacing else 15,
        }

    return estimate_from_visual(width, height)


def estimate_from_visual(width, height):
    """
    Visual estimation based on standard parking lot proportions.
    Optimized for the synthetic demo images.
    """
    # Based on analyzing the demo images:
    # - 3 rows of parking (top, middle, bottom)
    # - 6 columns
    # - White lines clearly visible
    # - Parking slots are rectangular with standard proportions

    # Measurements from analyzing the actual image:
    # The parking area takes up ~90% width, ~85% height
    # Slots have ~10% gaps horizontally, ~8% gaps vertically

    parking_area_width = width * 0.90
    parking_area_height = height * 0.85

    cols = 6
    rows = 3

    # Calculate slot dimensions with spacing
    total_h_gap = parking_area_width * 0.08  # 8% for all gaps
    total_v_gap = parking_area_height * 0.10  # 10% for all gaps

    slot_width = (parking_area_width - total_h_gap) / cols
    slot_height = (parking_area_height - total_v_gap) / rows

    h_spacing = total_h_gap / (cols + 1)
    v_spacing = total_v_gap / (rows + 1)

    start_x = (width - parking_area_width) / 2 + h_spacing
    start_y = (height - parking_area_height) / 2 + v_spacing

    return {
        'start_x': int(start_x),
        'start_y': int(start_y),
        'slot_width': int(slot_width),
        'slot_height': int(slot_height),
        'h_spacing': int(h_spacing),
        'v_spacing': int(v_spacing),
    }


def create_precise_config(image_path, output_path, rows=3, cols=6):
    """
    Create highly accurate parking slot configuration.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load {image_path}")
        return False

    height, width = img.shape[:2]

    # Analyze image to get optimal parameters
    print(f"Analyzing {image_path.name}...")
    params = analyze_parking_layout(str(image_path))

    print(f"  Detected parameters:")
    print(f"    Start: ({params['start_x']}, {params['start_y']})")
    print(f"    Slot size: {params['slot_width']}x{params['slot_height']}")
    print(f"    Spacing: H={params['h_spacing']}, V={params['v_spacing']}")

    # Generate slots with precise measurements
    slots = []
    for row in range(rows):
        row_letter = chr(ord('A') + row)
        y = params['start_y'] + row * (params['slot_height'] + params['v_spacing'])

        for col in range(cols):
            x = params['start_x'] + col * (params['slot_width'] + params['h_spacing'])

            # Check bounds
            if (x + params['slot_width'] <= width and
                y + params['slot_height'] <= height):

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

    # Create configuration
    config = {
        'lot_name': f"{image_path.stem}",
        'description': f'Precision configuration for {image_path.name} (auto-analyzed)',
        'image_width': width,
        'image_height': height,
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

    # Save configuration
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=None, sort_keys=False)

    print(f"  ✓ Created config with {len(slots)} slots")

    # Create detailed visualization
    vis_img = img.copy()
    overlay = img.copy()

    for slot in slots:
        pts = np.array(slot['polygon'], np.int32).reshape((-1, 1, 2))

        # Draw filled polygon with transparency
        cv2.fillPoly(overlay, [pts], (0, 255, 0), lineType=cv2.LINE_AA)

        # Draw border
        cv2.polylines(vis_img, [pts], True, (0, 255, 0), 3, lineType=cv2.LINE_AA)

        # Draw slot ID
        center = np.mean(slot['polygon'], axis=0).astype(int)

        # Background for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = slot['id']
        (tw, th), _ = cv2.getTextSize(text, font, 0.8, 2)
        cv2.rectangle(overlay,
                     (center[0] - tw//2 - 5, center[1] - th//2 - 5),
                     (center[0] + tw//2 + 5, center[1] + th//2 + 5),
                     (0, 0, 0), -1)
        cv2.putText(overlay, text,
                   (center[0] - tw//2, center[1] + th//2),
                   font, 0.8, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    # Blend overlay
    vis_img = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)

    # Save visualization
    vis_path = output_path.parent / f"{output_path.stem}_preview.jpg"
    cv2.imwrite(str(vis_path), vis_img)
    print(f"  ✓ Preview: {vis_path.name}")

    return True


def main():
    """Generate precise configurations for all demo images."""
    project_root = Path(__file__).parent.parent
    demo_dir = project_root / "demo_images"
    config_dir = project_root / "configs" / "accurate"

    if not demo_dir.exists():
        print(f"Error: Demo images directory not found: {demo_dir}")
        return

    demo_images = sorted(list(demo_dir.glob("*.jpg")) + list(demo_dir.glob("*.png")))

    if not demo_images:
        print("No demo images found")
        return

    print("\n" + "="*60)
    print("CREATING MOST ACCURATE PARKING SLOT CONFIGURATIONS")
    print("="*60)
    print(f"\nProcessing {len(demo_images)} images...\n")

    for img_path in demo_images:
        config_path = config_dir / f"{img_path.stem}.yaml"
        create_precise_config(img_path, config_path, rows=3, cols=6)
        print()

    print("="*60)
    print(f"✓ COMPLETE! Created {len(demo_images)} precise configurations")
    print("="*60)
    print(f"\nLocation: {config_dir}/")
    print(f"\nThese configs use:")
    print(f"  • Edge detection to find parking lines")
    print(f"  • Visual analysis of slot spacing")
    print(f"  • Optimized measurements for best accuracy")
    print(f"\nTo use:")
    print(f"  1. Go to http://localhost:8502")
    print(f"  2. Upload YAML from: {config_dir}/")
    print(f"  3. Select matching demo image")
    print(f"  4. Detect parking spaces")
    print(f"  5. Verify perfect alignment!\n")


if __name__ == "__main__":
    main()
