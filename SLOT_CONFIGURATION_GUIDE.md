# Parking Slot Configuration Guide

## Problem: Misaligned Grid Boxes

The default grid layout doesn't match real parking lot layouts, causing poor detection accuracy. Each parking lot has a unique layout that requires custom slot definitions.

## Solution: 3 Ways to Create Custom Slot Configurations

---

## Method 1: Web-Based Slot Annotator (Easiest) ⭐

### Launch the Annotator

```bash
source venv/bin/activate
streamlit run app/slot_annotator.py
```

This will open a web interface at `http://localhost:8502`

### Steps:

1. **Upload Image**: Choose your parking lot image
2. **Select Mode**:
   - **Auto Grid**: Quick presets (3x5, 4x6, 5x8, 6x10)
   - **Custom Grid**: Adjust all parameters (position, size, spacing)
3. **Generate Slots**: Click "Generate Slots" button
4. **Preview**: See slots overlaid on your image
5. **Adjust**: Fine-tune parameters if needed
6. **Save**: Download YAML configuration file

### Features:

- Real-time preview
- Auto-calculated slot sizes based on image dimensions
- Multiple presets for common layouts
- Download YAML directly
- Works on any parking lot image

---

## Method 2: Quick Config Script (For Demo Images)

### Generate Configs for All Demo Images

```bash
source venv/bin/activate
python scripts/quick_config.py
```

This automatically creates configurations for all images in `demo_images/` folder.

### Output:

- YAML config files in `configs/` directory
- Preview images showing slot layout
- Ready to use in the main app

### Generated Files:

```
configs/
├── synthetic_lot_01_20pct_config.yaml
├── synthetic_lot_01_20pct_config_preview.jpg
├── synthetic_lot_02_40pct_config.yaml
├── synthetic_lot_02_40pct_config_preview.jpg
... etc
```

---

## Method 3: Manual OpenCV Tool (Most Flexible)

### For Complex or Irregular Layouts

```bash
source venv/bin/activate
python scripts/create_slot_config.py --image your_lot.jpg --output configs/your_lot.yaml
```

### Controls:

- **Left Click**: Add polygon vertex
- **Right Click**: Complete current polygon
- **'u'**: Undo last vertex
- **'d'**: Delete last slot
- **'s'**: Save configuration
- **'g'**: Generate grid automatically
- **'r'**: Reset all
- **'q'**: Quit

### Use Cases:

- Angled parking spaces
- Irregular layouts
- Mixed orientations (perpendicular + parallel parking)
- Custom polygon shapes

---

## Using Your Configuration in the Main App

### Option A: Upload YAML File

1. Go to main app: `http://localhost:8502`
2. In sidebar → "Parking Slot Config"
3. Upload your `.yaml` file
4. Uncheck "Use default grid layout"
5. Select your image
6. Click "Detect Parking Spaces"

### Option B: Set as Default

Edit `configs/lot_config.yaml` to make it the default configuration.

---

## Configuration File Format

```yaml
lot_name: "My Parking Lot"
description: "Custom configuration"

image_width: 1920
image_height: 1080

slots:
  - id: "A1"
    polygon: [[100, 100], [180, 100], [180, 200], [100, 200]]
    type: "regular"

  - id: "A2"
    polygon: [[190, 100], [270, 100], [270, 200], [190, 200]]
    type: "handicap"

  # ... more slots
```

### Slot Types:
- `regular`: Standard parking space
- `handicap`: Accessible parking
- `compact`: Compact car space
- `ev`: Electric vehicle charging
- `reserved`: Reserved parking

---

## Tips for Best Results

### 1. **Image Quality**
- Use high-resolution images (1920x1080 or higher)
- Top-down/overhead view works best
- Good lighting conditions
- Minimal obstructions

### 2. **Slot Sizing**
- Standard parking space: ~80-100 pixels wide × 100-150 pixels tall (for 1920x1080 images)
- Adjust based on your image resolution
- Account for lane spacing between slots

### 3. **Alignment**
- Start with Auto Grid to get close
- Use Custom Grid to fine-tune
- Check preview images before saving
- Test with detection to verify

### 4. **Spacing**
- Horizontal spacing: 10-20 pixels (between columns)
- Vertical spacing: 20-30 pixels (between rows)
- Adjust based on actual lane widths in image

### 5. **Testing**
- Always test configuration with actual detection
- Adjust IoU threshold if needed (default: 0.3)
- Check edge cases (partially visible cars)

---

## Troubleshooting

### Slots Don't Match Parking Spaces

**Solution**: Use Custom Grid mode or Manual tool
- Adjust start_x, start_y to shift entire grid
- Modify slot_width, slot_height to resize
- Change spacing to account for lanes

### Detection Misses Some Cars

**Solution**:
- Lower confidence threshold (try 0.20 instead of 0.25)
- Check if slots are too small/large
- Verify car is fully visible in image

### Too Many False Positives

**Solution**:
- Increase confidence threshold (try 0.35)
- Increase IoU threshold (try 0.4-0.5)
- Ensure slots don't overlap driving lanes

### Slots Overlap

**Solution**:
- Increase horizontal/vertical spacing
- Reduce number of rows/columns
- Use Manual tool to draw non-overlapping polygons

---

## Advanced: Custom Polygon Slots

For complex layouts (angled parking, curved lots, etc.), use the Manual tool:

```bash
python scripts/create_slot_config.py --image complex_lot.jpg --output configs/complex.yaml
```

### Example Angled Parking:

```yaml
slots:
  - id: "A1"
    polygon: [[100, 100], [160, 80], [180, 140], [120, 160]]  # 4 corners, angled
    type: "regular"
```

The polygon can have any number of vertices (minimum 3 for triangle).

---

## Quick Reference: All Tools

| Tool | Use Case | Command |
|------|----------|---------|
| **Slot Annotator** | Interactive web-based | `streamlit run app/slot_annotator.py` |
| **Quick Config** | Batch generate for demos | `python scripts/quick_config.py` |
| **Manual Tool** | Complex/custom layouts | `python scripts/create_slot_config.py -i image.jpg -o config.yaml` |
| **Main App** | Run detection | `streamlit run app/streamlit_app.py` |

---

## Next Steps

1. ✅ Create configuration for your parking lot
2. ✅ Test with demo images first
3. ✅ Upload your config in main app
4. ✅ Run detection and verify accuracy
5. ✅ Adjust parameters if needed
6. ✅ Save final configuration

---

## Support

- Check preview images to verify alignment
- Start with Auto Grid for quick setup
- Use Custom Grid for fine-tuning
- Use Manual tool only for complex layouts
- Test configurations before deployment

For more help, see:
- Main README: `README.md`
- Training guide: `docs/TRAINING_GUIDE.md`
- Example configs: `configs/` directory
