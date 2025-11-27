#!/bin/bash
# Smart Parking Detection System - Setup Script
# Run with: bash setup.sh

set -e

echo "========================================"
echo "Smart Parking Detection System Setup"
echo "========================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "Detected Python version: $PYTHON_VERSION"

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 9 ]); then
    echo "Error: Python 3.9+ is required. Please install Python 3.10 or 3.11."
    exit 1
fi

if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 12 ]; then
    echo "Warning: Python 3.12+ may have compatibility issues with some packages."
    echo "Python 3.10 or 3.11 is recommended."
fi

# Create virtual environment
echo ""
echo "[1/4] Creating virtual environment..."

if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    echo "✓ Created virtual environment"
fi

# Activate virtual environment
source venv/bin/activate
echo "✓ Activated virtual environment"

# Upgrade pip
echo ""
echo "[2/4] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "[3/4] Installing dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "[4/4] Verifying installation..."

python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
python3 -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# Check for GPU
echo ""
echo "Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Create necessary directories
mkdir -p outputs demo_images models

# Success message
echo ""
echo "========================================"
echo "✓ Setup complete!"
echo "========================================"
echo ""
echo "To activate the environment in the future:"
echo "    source venv/bin/activate"
echo ""
echo "To run the demo app:"
echo "    streamlit run app/streamlit_app.py"
echo ""
echo "To run inference on an image:"
echo "    python scripts/run_inference.py --image your_image.jpg --config configs/lot_config.yaml"
echo ""
echo "For more information, see README.md"
echo ""
