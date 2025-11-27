# Quick Start: Fix Misaligned Grid Boxes

## The Problem

The default grid doesn't match your parking lot layout:

![Misaligned Grid](https://i.imgur.com/example.png)

Green/red boxes don't align with actual parking spaces = poor accuracy!

---

## The Solution (3 Minutes)

### Step 1: Launch the Slot Annotator

```bash
source venv/bin/activate
streamlit run app/slot_annotator.py
```

Opens at: **http://localhost:8503**

### Step 2: Configure Your Parking Lot

1. **Upload your image** (or select a demo image)
2. **Choose "Auto Grid"** mode
3. **Select a preset**:
   - 3x5 (15 spots) - Small lot
   - 4x6 (24 spots) - Medium lot ⭐ Best for demos
   - 5x8 (40 spots) - Large lot
   - 6x10 (60 spots) - XL lot

4. **Click "Generate Slots"**
5. **Preview** - boxes should now align better!
6. **Save Configuration** - download the YAML file

### Step 3: Use in Main App

1. Go to main app: **http://localhost:8502**
2. Sidebar → "Parking Slot Config" → Upload your YAML file
3. Uncheck "Use default grid layout"
4. Select the same image
5. Click "Detect Parking Spaces"

### Result: ✅ Aligned Slots!

Green/red boxes now match actual parking spaces = accurate detection!

---

## Fine-Tuning (Optional)

If Auto Grid isn't perfect:

### Use Custom Grid Mode

Adjust these parameters:

- **Start X/Y**: Shift entire grid left/right/up/down
- **Slot Width**: Make boxes wider/narrower
- **Slot Height**: Make boxes taller/shorter
- **Spacing**: Adjust gaps between slots

Watch the preview update in real-time!

---

## Already Have Demo Images?

Quick generate configs for all demos:

```bash
python scripts/quick_config.py
```

This creates:
- ✅ 5 YAML config files in `configs/`
- ✅ 5 preview images showing slot layouts
- ✅ Ready to use immediately

Then just upload the YAML in the main app!

---

## Tools Available

| Tool | URL | Purpose |
|------|-----|---------|
| **Main App** | http://localhost:8502 | Run parking detection |
| **Slot Annotator** | http://localhost:8503 | Create slot configs |

---

## Tips

1. **Start with Auto Grid** - fastest way
2. **Check preview** - verify alignment before saving
3. **Test in main app** - run actual detection to confirm
4. **Adjust if needed** - use Custom Grid to fine-tune
5. **Save config** - reuse for similar parking lots

---

## For Different Parking Lots

Each parking lot needs its own config because:

- Different camera angles
- Different slot sizes/orientations
- Different number of spots
- Different layouts (straight, angled, etc.)

**Solution**:
- Create one config per parking lot
- Takes 2-3 minutes each
- Save all configs in `configs/` folder
- Upload the right one when needed

---

## Need Help?

See full guide: [SLOT_CONFIGURATION_GUIDE.md](SLOT_CONFIGURATION_GUIDE.md)

Or main docs: [README.md](README.md)
