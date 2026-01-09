# Smart Parking Detection System - MVP

A complete MVP for detecting parking space occupancy from top-down/overhead images. Supports both **local YOLOv8** inference and **Roboflow cloud API** for maximum flexibility.

## 🚀 Two Detection Backends

| Feature | Roboflow API (Cloud) | Local YOLOv8 |
|---------|---------------------|--------------|
| GPU Required | ❌ No | ✅ Recommended |
| Setup Complexity | Very Easy | Moderate |
| Inference Speed | ~1-2 sec/image | ~0.1-0.5 sec |
| Cost | Free tier available | Free |
| Fine-tuning | Via Roboflow UI | Local training |
| Offline Usage | ❌ No | ✅ Yes |

**Recommendation:** Start with Roboflow for quick demos, switch to local YOLOv8 for production or offline needs.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SMART PARKING DETECTION MVP                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────────────┐ │
│  │              │    │  Detection       │    │                            │ │
│  │  Input Image │───▶│  ┌────────────┐  │───▶│  Detection Results         │ │
│  │  (Top-down)  │    │  │ Roboflow   │  │    │  (Bounding Boxes + Conf)   │ │
│  │              │    │  │ OR YOLOv8  │  │    │                            │ │
│  └──────────────┘    │  └────────────┘  │    └─────────────┬──────────────┘ │
│                      └──────────────────┘                  │                 │
│                                                            ▼                 │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────────────┐ │
│  │              │    │                  │    │                            │ │
│  │  Slot Config │───▶│  Occupancy Logic │◀───│  IoU Matching              │ │
│  │  (JSON/YAML) │    │  (Slot Matcher)  │    │  (Detection → Slot)        │ │
│  │              │    │                  │    │                            │ │
│  └──────────────┘    └──────────────────┘    └─────────────┬──────────────┘ │
│                                                            │                 │
│                                                            ▼                 │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        Streamlit Web Interface                         │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────────────┐  │  │
│  │  │ Original Image  │  │ Annotated Image │  │ Occupancy Map/Grid    │  │  │
│  │  │ with Detections │  │ with Slot Status│  │ (Green=Empty/Red=Full)│  │  │
│  │  └─────────────────┘  └─────────────────┘  └───────────────────────┘  │  │
│  │                                                                        │  │
│  │  Summary: Total: 50 | Empty: 23 | Occupied: 27 | Occupancy: 54%       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Option A: Roboflow Cloud (Easiest - No GPU needed)

```bash
cd smart-parking-mvp

# Install lightweight dependencies
pip install -r requirements-roboflow.txt

# Set API key (already configured with demo key)
export ROBOFLOW_API_KEY="YOUR_API_KEY"

# Run demo app
streamlit run app/streamlit_app.py
```

### Option B: Local YOLOv8 (Full setup)

```bash
cd smart-parking-mvp

# Create virtual environment (Python 3.10 recommended)
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install full dependencies
pip install -r requirements.txt

# Run demo app
streamlit run app/streamlit_app.py
```

### Using the Demo App

1. Open browser to `http://localhost:8501`
2. Select detection backend (Roboflow or Local YOLOv8)
3. Upload a parking lot image or use demo images
4. Click "Detect Parking Spaces"
5. View results: annotated image, schematic map, and statistics

## Roboflow Integration

The project includes a pre-configured Roboflow workflow for car detection:

- **Workspace:** `parkpark-zclps`
- **Workflow:** `find-cars-2`
- **API Key:** Provided in the demo (replace with your own for production)

### Using Roboflow Programmatically

```python
from utils.roboflow_detector import RoboflowDetector

# Initialize detector
detector = RoboflowDetector(
    api_key="your_api_key",  # or set ROBOFLOW_API_KEY env var
    confidence_threshold=0.25
)

# Run detection
detections = detector.detect("parking_lot.jpg")

for det in detections:
    print(f"Found {det.class_name} at {det.bbox} (conf: {det.confidence:.2f})")
```

### Using with inference-sdk (More efficient)

```python
from utils.roboflow_detector import RoboflowSDKDetector

detector = RoboflowSDKDetector(
    api_key="your_api_key",
    use_cache=True  # Caches workflow for faster repeated calls
)
detections = detector.detect("parking_lot.jpg")
```

## Project Structure

```
smart-parking-mvp/
├── README.md                      # This file
├── requirements.txt               # Python dependencies (pinned versions)
├── environment.yml                # Conda environment (alternative)
├── app/
│   └── streamlit_app.py          # Main demo application
├── configs/
│   ├── lot_config.yaml           # Parking lot slot definitions
│   └── clemson_parking.yaml      # YOLOv8 training config
├── models/
│   └── .gitkeep                  # Model weights stored here
├── utils/
│   ├── __init__.py
│   ├── detector.py               # YOLOv8 inference wrapper
│   ├── slot_matcher.py           # Detection-to-slot mapping
│   └── visualization.py          # Drawing/overlay utilities
├── scripts/
│   ├── run_inference.py          # Standalone inference script
│   ├── train_parking_model.py    # Fine-tuning script
│   └── create_slot_config.py     # Tool to define slot polygons
├── demo_images/                   # Sample parking lot images
├── datasets/
│   └── clemson_parking/          # Your annotated dataset
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
└── outputs/                       # Inference outputs
```

## Configuration

### Parking Lot Slot Configuration (configs/lot_config.yaml)

Define your parking spots as polygons:

```yaml
lot_name: "Clemson Main Lot"
image_width: 1920
image_height: 1080
slots:
  - id: "A1"
    polygon: [[100, 100], [200, 100], [200, 180], [100, 180]]
  - id: "A2"
    polygon: [[210, 100], [310, 100], [310, 180], [210, 180]]
  # ... more slots
```

Use the slot configuration tool to interactively define slots:
```bash
python scripts/create_slot_config.py --image your_lot_image.jpg --output configs/your_lot.yaml
```

## Fine-Tuning Guide

If the pretrained YOLOv8 model doesn't perform well on your top-down Clemson images, fine-tune it:

### 1. Prepare Your Dataset

Annotate your images in YOLO format:
- Use [LabelImg](https://github.com/HumanSignal/labelImg) or [CVAT](https://cvat.ai/)
- Export in YOLO format (one .txt file per image)
- Each line: `class_id x_center y_center width height` (normalized 0-1)
- For car detection, use class_id = 0

Directory structure:
```
datasets/clemson_parking/
├── images/
│   ├── train/
│   │   ├── lot_001.jpg
│   │   ├── lot_002.jpg
│   │   └── ...
│   └── val/
│       ├── lot_050.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── lot_001.txt
    │   ├── lot_002.txt
    │   └── ...
    └── val/
        ├── lot_050.txt
        └── ...
```

### 2. Run Training

```bash
python scripts/train_parking_model.py \
    --data configs/clemson_parking.yaml \
    --epochs 50 \
    --imgsz 640 \
    --batch-size 16 \
    --model yolov8m.pt
```

### 3. Use Your Fine-Tuned Model

```bash
# Update the model path in the app or inference script
python scripts/run_inference.py \
    --image demo_images/test.jpg \
    --model runs/detect/parking_model/weights/best.pt
```

## GPU Training on Free Platforms

### Google Colab (Recommended - Free)

1. Upload your dataset to Google Drive
2. Open a new Colab notebook
3. Run:

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install ultralytics==8.2.0

# Copy dataset
!cp -r /content/drive/MyDrive/clemson_parking /content/datasets/

# Train
!python /content/drive/MyDrive/smart-parking-mvp/scripts/train_parking_model.py \
    --data /content/datasets/clemson_parking.yaml \
    --epochs 50 \
    --batch-size 16

# Download weights
from google.colab import files
files.download('runs/detect/parking_model/weights/best.pt')
```

### Kaggle Notebooks (Free - 30 GPU hours/week)

1. Create new notebook, enable GPU
2. Upload dataset as Kaggle Dataset
3. Similar process to Colab

See `docs/TRAINING_GUIDE.md` for detailed instructions.

## Technical Specifications

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10.x | 3.11 also works; avoid 3.12 |
| PyTorch | 2.1.0 | CUDA 11.8 compatible |
| Ultralytics | 8.2.0 | YOLOv8 implementation |
| Streamlit | 1.32.0 | Web interface |
| OpenCV | 4.9.0 | Image processing |
| NumPy | 1.26.0 | Array operations |

## API Reference

### ParkingDetector Class

```python
from utils.detector import ParkingDetector

# Initialize
detector = ParkingDetector(
    model_path="yolov8m.pt",  # or your fine-tuned model
    confidence_threshold=0.25,
    device="cpu"  # or "cuda"
)

# Run detection
detections = detector.detect(image_path="lot.jpg")
# Returns: List[Detection] with bbox, confidence, class_id
```

### SlotMatcher Class

```python
from utils.slot_matcher import SlotMatcher

# Initialize with config
matcher = SlotMatcher(config_path="configs/lot_config.yaml")

# Match detections to slots
results = matcher.match(detections)
# Returns: Dict[slot_id, OccupancyStatus]
```

## Troubleshooting

### Common Issues

1. **"CUDA out of memory"**
   - Reduce batch size: `--batch-size 8`
   - Use smaller model: `yolov8s.pt` instead of `yolov8m.pt`

2. **"No module named 'ultralytics'"**
   - Ensure venv is activated
   - Run: `pip install ultralytics==8.2.0`

3. **Slow inference on CPU**
   - Expected: 2-5 seconds per image on CPU
   - For faster inference, use GPU or reduce image size

4. **Poor detection accuracy**
   - Fine-tune on your specific dataset
   - Ensure annotations are accurate
   - Use more training data

## License

MIT License - Free for commercial and non-commercial use.

## Contributing

1. Fork the repository
2. Create feature branch
3. Submit pull request

## Support

- Issues: Open a GitHub issue
- Email: [your-email]
