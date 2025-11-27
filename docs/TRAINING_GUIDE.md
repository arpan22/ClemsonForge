# GPU Training Guide

This guide covers how to train/fine-tune the parking detection model on various platforms.

## Table of Contents

1. [Local GPU Training](#local-gpu-training)
2. [Google Colab (Free)](#google-colab-free)
3. [Kaggle Notebooks (Free)](#kaggle-notebooks-free)
4. [Cloud GPU Options](#cloud-gpu-options)

---

## Local GPU Training

If you have a NVIDIA GPU with at least 4GB VRAM:

### Prerequisites

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA support (example for CUDA 11.8)
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### Training Command

```bash
python scripts/train_parking_model.py \
    --data configs/clemson_parking.yaml \
    --model yolov8m.pt \
    --epochs 50 \
    --batch-size 16 \
    --imgsz 640
```

### Memory Issues?

```bash
# Reduce batch size
--batch-size 8

# Use smaller model
--model yolov8s.pt

# Use smaller image size
--imgsz 416
```

---

## Google Colab (Free)

Google Colab offers free GPU access (T4 GPU, ~15GB VRAM).

### Step 1: Prepare Your Data

1. Upload your dataset to Google Drive:
   ```
   MyDrive/
   └── clemson_parking/
       ├── images/
       │   ├── train/
       │   └── val/
       └── labels/
           ├── train/
           └── val/
   ```

2. Also upload the project files:
   ```
   MyDrive/
   └── smart-parking-mvp/
       ├── scripts/
       ├── configs/
       └── ...
   ```

### Step 2: Create Colab Notebook

Create a new notebook at [colab.research.google.com](https://colab.research.google.com)

**Enable GPU:** Runtime → Change runtime type → GPU (T4)

### Step 3: Run Training

Copy and run these cells:

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Cell 2: Install dependencies
!pip install ultralytics==8.2.0
```

```python
# Cell 3: Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

```python
# Cell 4: Create dataset config for Colab paths
import yaml

config = {
    'path': '/content/drive/MyDrive/clemson_parking',
    'train': 'images/train',
    'val': 'images/val',
    'nc': 1,
    'names': {0: 'car'}
}

with open('/content/dataset.yaml', 'w') as f:
    yaml.dump(config, f)
```

```python
# Cell 5: Train the model
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8m.pt')

# Train
results = model.train(
    data='/content/dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    project='/content/drive/MyDrive/parking_training',
    name='run1',
    device=0,
    patience=20,
)
```

```python
# Cell 6: Evaluate
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

```python
# Cell 7: Download weights
from google.colab import files
files.download('/content/drive/MyDrive/parking_training/run1/weights/best.pt')
```

### Tips for Colab

- **Session limits:** Free tier has ~12 hour session limits
- **Save checkpoints to Drive:** Results are automatically saved to Drive
- **Resume training:** If disconnected, you can resume:
  ```python
  model = YOLO('/content/drive/MyDrive/parking_training/run1/weights/last.pt')
  model.train(resume=True)
  ```

---

## Kaggle Notebooks (Free)

Kaggle offers 30 GPU hours per week with P100 GPUs.

### Step 1: Create Dataset

1. Go to [kaggle.com/datasets](https://kaggle.com/datasets)
2. Click "New Dataset"
3. Upload your annotated images and labels
4. Structure:
   ```
   clemson-parking-dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/
       └── val/
   ```

### Step 2: Create Notebook

1. Go to [kaggle.com/code](https://kaggle.com/code)
2. Click "New Notebook"
3. Settings → Accelerator → GPU P100

### Step 3: Add Dataset

- Click "Add data" → Search for your dataset → Add

### Step 4: Training Code

```python
# Cell 1: Install
!pip install -q ultralytics==8.2.0
```

```python
# Cell 2: Create config
import yaml

config = {
    'path': '/kaggle/input/clemson-parking-dataset',
    'train': 'images/train',
    'val': 'images/val',
    'nc': 1,
    'names': {0: 'car'}
}

with open('/kaggle/working/dataset.yaml', 'w') as f:
    yaml.dump(config, f)
```

```python
# Cell 3: Train
from ultralytics import YOLO

model = YOLO('yolov8m.pt')

results = model.train(
    data='/kaggle/working/dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    project='/kaggle/working/runs',
    name='parking',
    device=0,
)
```

```python
# Cell 4: Save output
# Kaggle automatically saves /kaggle/working/ as output
import shutil
shutil.copy('/kaggle/working/runs/parking/weights/best.pt', '/kaggle/working/best.pt')
```

### Download Results

After training, go to "Output" tab and download `best.pt`.

---

## Cloud GPU Options

For longer training or larger datasets:

### Option 1: Lambda Labs ($0.50/hr for A10)

1. Sign up at [lambdalabs.com](https://lambdalabs.com)
2. Launch instance with A10 or A100 GPU
3. SSH in and clone your project
4. Run training script

### Option 2: Vast.ai ($0.10-0.50/hr)

1. Sign up at [vast.ai](https://vast.ai)
2. Search for RTX 3090 or A100 instances
3. Use their Jupyter interface or SSH

### Option 3: RunPod ($0.20-0.80/hr)

1. Sign up at [runpod.io](https://runpod.io)
2. Deploy a GPU pod
3. Access via SSH or Jupyter

### Example Cloud Training Script

```bash
#!/bin/bash
# cloud_train.sh

# Clone project
git clone https://github.com/your-username/smart-parking-mvp.git
cd smart-parking-mvp

# Install dependencies
pip install -r requirements.txt

# Download your dataset (from Google Drive, S3, etc.)
# gdown --id YOUR_GDRIVE_FILE_ID -O dataset.zip
# unzip dataset.zip -d datasets/

# Train
python scripts/train_parking_model.py \
    --data configs/clemson_parking.yaml \
    --epochs 100 \
    --batch-size 32 \
    --imgsz 640 \
    --model yolov8m.pt

# Upload results
# aws s3 cp runs/detect/parking_model/weights/best.pt s3://your-bucket/
```

---

## Training Tips

### Data Augmentation

YOLOv8 applies augmentation automatically. For parking lots, consider:

```python
# In training call
model.train(
    ...,
    degrees=0.0,      # No rotation (parking lots are oriented)
    flipud=0.0,       # No vertical flip
    fliplr=0.5,       # Horizontal flip OK
    mosaic=1.0,       # Mosaic augmentation helps
)
```

### Hyperparameter Tuning

```python
# For better results with more compute
model.train(
    ...,
    epochs=100,       # More epochs
    patience=50,      # More patience
    lr0=0.001,        # Lower initial LR
    optimizer='AdamW', # AdamW often works well
)
```

### Monitoring Training

Training creates a `results.csv` file. Plot it:

```python
import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv('runs/detect/parking_model/results.csv')
results[['train/box_loss', 'val/box_loss']].plot()
plt.savefig('loss_curve.png')
```

### When to Stop Training

- Watch validation loss - if it stops decreasing for 20+ epochs, you're done
- mAP50 > 0.8 is usually good for parking detection
- mAP50-95 > 0.5 is excellent

---

## Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size
--batch-size 8  # or 4

# Clear cache
import torch
torch.cuda.empty_cache()
```

### "No labels found"
- Check label files exist in `labels/train/` and `labels/val/`
- Ensure filenames match: `image001.jpg` → `image001.txt`

### "Model not improving"
- Add more training data
- Check annotation quality
- Try different augmentation settings
- Increase epochs

### Colab Disconnection
```python
# Add keep-alive JavaScript in Colab
function KeepClicking(){
    console.log("Clicking");
    document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(KeepClicking, 60000)
```

---

## Expected Training Times

| Platform | GPU | Dataset Size | Epochs | Time |
|----------|-----|--------------|--------|------|
| Colab Free | T4 | 500 images | 50 | ~2 hours |
| Kaggle | P100 | 500 images | 50 | ~1.5 hours |
| RTX 3090 | Local | 500 images | 50 | ~45 min |
| A100 | Cloud | 500 images | 50 | ~20 min |

---

## After Training

1. **Evaluate:** Run validation on held-out test images
2. **Test:** Try inference on new images
3. **Deploy:** Update `model_path` in your app to use `best.pt`

```bash
# Test your fine-tuned model
python scripts/run_inference.py \
    --image test_image.jpg \
    --model runs/detect/parking_model/weights/best.pt \
    --config configs/lot_config.yaml
```
