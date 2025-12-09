# Quick Start Guide - Aerial Threat Detection System

This guide provides quick commands to get you started with training and using the Aerial Threat Detection System.

## Table of Contents

1. [Initial Setup](#initial-setup)
2. [Dataset Preparation](#dataset-preparation)
3. [Model Training](#model-training)
4. [Testing the System](#testing-the-system)
5. [Running the Application](#running-the-application)

---

## Initial Setup

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies for Electron app
cd electron-app
npm install
cd ..
```

### 2. Verify Installation

```bash
# Check Python and packages
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from ultralytics import YOLO; print('YOLOv8 Ready')"

# Run validation tests
python tests/test_validation.py
```

---

## Dataset Preparation

### Quick Dataset Setup

```bash
# 1. List available datasets
python scripts/prepare_dataset.py --list

# 2. Get download instructions for a dataset
python scripts/prepare_dataset.py --download uav-person

# 3. After downloading manually from Roboflow:
#    - Visit the URL provided
#    - Sign up/login to Roboflow
#    - Download in YOLOv8 format
#    - Extract to datasets/uav-person/

# 4. Verify the dataset structure
python scripts/prepare_dataset.py --verify datasets/uav-person

# 5. Create/update data.yaml configuration
python scripts/prepare_dataset.py --create-yaml datasets/uav-person --classes soldier civilian
```

### Example: Complete Dataset Setup

```bash
# Step-by-step example
mkdir -p datasets

# Download from Roboflow (manual step - follow website instructions)
# Extract to: datasets/my-aerial-dataset/

# Verify structure
python scripts/prepare_dataset.py --verify datasets/my-aerial-dataset

# Create configuration
python scripts/prepare_dataset.py \
    --create-yaml datasets/my-aerial-dataset \
    --classes soldier civilian
```

---

## Model Training

### Quick Training

```bash
# Basic training with YOLOv8s (recommended)
python scripts/train_model.py \
    --data datasets/my-aerial-dataset/data.yaml \
    --model yolov8s.pt \
    --epochs 100 \
    --batch 16

# The trained model will be saved to:
# runs/train/aerial_threat_detector/weights/best.pt
```

### Advanced Training Options

```bash
# Training with validation and export
python scripts/train_model.py \
    --data datasets/my-aerial-dataset/data.yaml \
    --model yolov8m.pt \
    --epochs 200 \
    --batch 32 \
    --imgsz 640 \
    --device 0 \
    --workers 8 \
    --patience 50 \
    --validate \
    --export onnx \
    --test-image test_data/sample.jpg

# CPU-only training (slower)
python scripts/train_model.py \
    --data datasets/my-aerial-dataset/data.yaml \
    --model yolov8n.pt \
    --epochs 50 \
    --batch 8 \
    --device cpu

# Fast training with smaller model
python scripts/train_model.py \
    --data datasets/my-aerial-dataset/data.yaml \
    --model yolov8n.pt \
    --epochs 50 \
    --batch 8
```

### Validation Only

```bash
# Validate an existing model
python scripts/train_model.py \
    --data datasets/my-aerial-dataset/data.yaml \
    --validate-only runs/train/aerial_threat_detector/weights/best.pt
```

### Deploy Trained Model

```bash
# Copy trained model to project root for deployment
cp runs/train/aerial_threat_detector/weights/best.pt ./best.pt

# Or specify a custom name
cp runs/train/aerial_threat_detector/weights/best.pt ./my_model_v1.pt
```

---

## Testing the System

### Run Tests

```bash
# Run validation tests (no GPU required)
python tests/test_validation.py

# Run full test suite (requires all dependencies)
python tests/test_system.py
```

### Manual Testing

```bash
# Test detection on single image
python src/detection_server.py \
    --source-type image \
    --source-path path/to/test_image.jpg \
    --model-path best.pt

# Test detection on video
python src/detection_server.py \
    --source-type video \
    --source-path path/to/test_video.mp4 \
    --model-path best.pt

# Test with webcam
python src/detection_server.py \
    --source-type webcam \
    --model-path best.pt
```

---

## Running the Application

### Option 1: Automated Start (Windows)

```bash
# Double-click or run:
start_app.bat
```

### Option 2: Manual Start

```bash
# Terminal 1: Start detection server
python src/detection_server.py --model-path best.pt

# Terminal 2: Start Electron app
cd electron-app
npm start
```

### Option 3: Individual Components

```bash
# Start only the detection server
python src/detection_server.py \
    --model-path best.pt \
    --host localhost \
    --port 5000

# Start with debug mode
python src/detection_server.py \
    --model-path best.pt \
    --debug

# Start Electron app independently
cd electron-app
npm start
```

---

## Common Commands Summary

### Setup
```bash
pip install -r requirements.txt                    # Install Python deps
cd electron-app && npm install && cd ..            # Install Node deps
python tests/test_validation.py                    # Verify setup
```

### Dataset
```bash
python scripts/prepare_dataset.py --list           # List datasets
python scripts/prepare_dataset.py --download uav-person  # Get instructions
python scripts/prepare_dataset.py --verify datasets/my-dataset  # Verify
python scripts/prepare_dataset.py --create-yaml datasets/my-dataset --classes soldier civilian  # Create config
```

### Training
```bash
# Quick start
python scripts/train_model.py --data datasets/my-dataset/data.yaml --model yolov8s.pt

# Production training
python scripts/train_model.py \
    --data datasets/my-dataset/data.yaml \
    --model yolov8m.pt \
    --epochs 200 \
    --batch 32 \
    --validate \
    --export onnx

# Deploy model
cp runs/train/aerial_threat_detector/weights/best.pt ./best.pt
```

### Testing
```bash
python tests/test_validation.py                    # Quick tests
python src/detection_server.py --source-type image --source-path test.jpg  # Test image
python src/detection_server.py --source-type webcam  # Test webcam
```

### Running
```bash
start_app.bat                                      # Windows quick start
# OR
python src/detection_server.py                     # Terminal 1
cd electron-app && npm start                       # Terminal 2
```

---

## Troubleshooting Quick Fixes

### GPU Not Detected
```bash
# Check CUDA installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Use CPU if needed
python scripts/train_model.py --data data.yaml --device cpu
```

### Out of Memory
```bash
# Reduce batch size
python scripts/train_model.py --data data.yaml --batch 8

# Use smaller model
python scripts/train_model.py --data data.yaml --model yolov8n.pt
```

### Model Not Found
```bash
# Check if model exists
ls -lh best.pt

# Download base model if needed (YOLOv8 will auto-download)
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"

# Specify model path explicitly
python src/detection_server.py --model-path ./runs/train/aerial_threat_detector/weights/best.pt
```

### Port Already in Use
```bash
# Use different port
python src/detection_server.py --port 5001

# Or kill existing process (Windows)
taskkill /F /IM python.exe

# Or kill existing process (Linux/Mac)
pkill -f detection_server
```

---

## Performance Tips

### For Faster Training
- Use smaller model: `yolov8n.pt` instead of `yolov8s.pt`
- Reduce batch size: `--batch 8`
- Reduce image size: `--imgsz 416`
- Use fewer epochs for testing: `--epochs 20`

### For Better Accuracy
- Use larger model: `yolov8m.pt` or `yolov8l.pt`
- Train longer: `--epochs 200`
- Use larger images: `--imgsz 1280`
- Increase batch size (if GPU allows): `--batch 32`

### For Real-Time Performance
- Use YOLOv8n or YOLOv8s
- Enable GPU acceleration
- Reduce confidence threshold if needed
- Use half precision (FP16) - enabled by default on GPU

---

## Next Steps

1. **Read the Documentation**:
   - [README.md](../README.md) - Project overview
   - [TRAINING_GUIDE.md](../docs/TRAINING_GUIDE.md) - Detailed training instructions
   - [FINAL_PRESENTATION.md](../docs/FINAL_PRESENTATION.md) - Complete system documentation
   - [ETHICAL_CONSIDERATIONS.md](../docs/ETHICAL_CONSIDERATIONS.md) - Ethical guidelines

2. **Prepare Your Dataset**:
   - Download from Roboflow
   - Verify structure
   - Create data.yaml

3. **Train Your Model**:
   - Start with YOLOv8s
   - Train for 100 epochs
   - Validate results

4. **Deploy and Test**:
   - Copy best.pt to root
   - Start the application
   - Test with sample data

5. **Iterate and Improve**:
   - Analyze results
   - Collect more data
   - Retrain with improvements

---

## Support

For detailed information:
- **Training Issues**: See [TRAINING_GUIDE.md](../docs/TRAINING_GUIDE.md)
- **Ethical Questions**: See [ETHICAL_CONSIDERATIONS.md](../docs/ETHICAL_CONSIDERATIONS.md)
- **System Documentation**: See [FINAL_PRESENTATION.md](../docs/FINAL_PRESENTATION.md)
- **General Help**: See [README.md](../README.md)

---

**Happy Detecting! 🚀**
