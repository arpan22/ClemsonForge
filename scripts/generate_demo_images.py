#!/usr/bin/env python3
"""
Generate Synthetic Demo Images
------------------------------
Creates synthetic top-down parking lot images for testing.

Usage:
    python scripts/generate_demo_images.py --output demo_images/
"""

import argparse
from pathlib import Path
import random

try:
    import cv2
    import numpy as np
except ImportError:
    print("Please install opencv-python and numpy")
    exit(1)


def generate_parking_lot_image(
    width: int = 1200,
    height: int = 800,
    rows: int = 3,
    cols: int = 6,
    occupancy_rate: float = 0.5,
    seed: int = None
) -> np.ndarray:
    """
    Generate a synthetic top-down parking lot image.
    
    Args:
        width, height: Image dimensions
        rows, cols: Grid of parking slots
        occupancy_rate: Fraction of slots to fill with cars
        seed: Random seed for reproducibility
        
    Returns:
        numpy array (BGR image)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create gray asphalt background
    image = np.ones((height, width, 3), dtype=np.uint8) * 80
    
    # Add some texture/noise
    noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Calculate slot dimensions
    margin_x = int(width * 0.1)
    margin_y = int(height * 0.1)
    
    available_width = width - 2 * margin_x
    available_height = height - 2 * margin_y
    
    slot_width = int(available_width / cols * 0.85)
    slot_height = int(available_height / rows * 0.85)
    
    spacing_x = (available_width - cols * slot_width) // (cols + 1)
    spacing_y = (available_height - rows * slot_height) // (rows + 1)
    
    # Draw parking lines and potentially cars
    car_colors = [
        (40, 40, 40),      # Black
        (200, 200, 200),   # White/Silver
        (50, 50, 150),     # Red
        (150, 50, 50),     # Blue
        (50, 100, 50),     # Green
        (80, 80, 120),     # Brown/Maroon
    ]
    
    slots_info = []
    
    for row in range(rows):
        for col in range(cols):
            # Calculate slot position
            x = margin_x + spacing_x + col * (slot_width + spacing_x)
            y = margin_y + spacing_y + row * (slot_height + spacing_y)
            
            # Draw parking slot lines (white)
            cv2.rectangle(image, (x, y), (x + slot_width, y + slot_height), 
                         (255, 255, 255), 2)
            
            # Randomly place car
            is_occupied = random.random() < occupancy_rate
            
            if is_occupied:
                # Car dimensions (slightly smaller than slot)
                car_margin = 8
                car_x = x + car_margin
                car_y = y + car_margin
                car_w = slot_width - 2 * car_margin
                car_h = slot_height - 2 * car_margin
                
                # Random car color
                car_color = random.choice(car_colors)
                
                # Draw car body (rectangle with rounded appearance)
                cv2.rectangle(image, (car_x, car_y), 
                             (car_x + car_w, car_y + car_h), car_color, -1)
                
                # Draw windshield (darker rectangle at top)
                windshield_h = int(car_h * 0.25)
                windshield_color = tuple(max(0, c - 40) for c in car_color)
                cv2.rectangle(image, (car_x + 5, car_y + 5),
                             (car_x + car_w - 5, car_y + windshield_h),
                             windshield_color, -1)
                
                # Draw rear window
                cv2.rectangle(image, (car_x + 5, car_y + car_h - windshield_h),
                             (car_x + car_w - 5, car_y + car_h - 5),
                             windshield_color, -1)
                
                # Add slight shadow
                shadow_offset = 3
                cv2.rectangle(image, 
                             (car_x + shadow_offset, car_y + car_h),
                             (car_x + car_w + shadow_offset, car_y + car_h + shadow_offset),
                             (50, 50, 50), -1)
            
            slots_info.append({
                'row': row,
                'col': col,
                'x': x,
                'y': y,
                'width': slot_width,
                'height': slot_height,
                'occupied': is_occupied
            })
    
    # Add lane markings
    lane_y = height - margin_y // 2
    cv2.line(image, (margin_x, lane_y), (width - margin_x, lane_y), 
            (255, 255, 255), 2, cv2.LINE_AA)
    
    # Add arrows
    arrow_x = width // 2
    cv2.arrowedLine(image, (arrow_x - 50, lane_y), (arrow_x + 50, lane_y),
                   (255, 255, 255), 3, cv2.LINE_AA, tipLength=0.3)
    
    return image, slots_info


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic parking lot images")
    parser.add_argument("--output", "-o", type=str, default="demo_images",
                       help="Output directory")
    parser.add_argument("--count", "-n", type=int, default=5,
                       help="Number of images to generate")
    parser.add_argument("--width", type=int, default=1200, help="Image width")
    parser.add_argument("--height", type=int, default=800, help="Image height")
    parser.add_argument("--rows", type=int, default=3, help="Number of parking rows")
    parser.add_argument("--cols", type=int, default=6, help="Number of parking columns")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.count} synthetic parking lot images...")
    
    occupancy_rates = [0.2, 0.4, 0.5, 0.7, 0.9]
    
    for i in range(args.count):
        occupancy = occupancy_rates[i % len(occupancy_rates)]
        
        image, slots_info = generate_parking_lot_image(
            width=args.width,
            height=args.height,
            rows=args.rows,
            cols=args.cols,
            occupancy_rate=occupancy,
            seed=42 + i
        )
        
        # Save image
        filename = f"synthetic_lot_{i+1:02d}_{int(occupancy*100)}pct.jpg"
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), image)
        
        occupied = sum(1 for s in slots_info if s['occupied'])
        total = len(slots_info)
        
        print(f"  ✓ {filename} ({occupied}/{total} occupied)")
    
    print(f"\nGenerated {args.count} images in {output_dir}/")
    print("\nNote: These are synthetic images for testing.")
    print("For best results, use real top-down parking lot images.")


if __name__ == "__main__":
    main()
