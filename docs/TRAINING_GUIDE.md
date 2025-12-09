# Training Guide for Aerial Threat Detection Model

This comprehensive guide walks you through the complete process of training a YOLOv8 model for soldier and civilian classification from aerial imagery.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Dataset Preparation](#dataset-preparation)
3. [Model Training](#model-training)
4. [Evaluation and Testing](#evaluation-and-testing)
5. [Model Deployment](#model-deployment)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements

**Minimum Requirements:**
- CPU: Intel Core i5 or AMD Ryzen 5
- RAM: 16GB
- Storage: 50GB free space
- GPU: Not required but training will be slow

**Recommended Requirements:**
- CPU: Intel Core i7/i9 or AMD Ryzen 7/9
- RAM: 32GB or more
- Storage: 100GB+ SSD
- GPU: NVIDIA RTX 3060 or better with 8GB+ VRAM
- CUDA: Version 11.0 or higher

### Software Requirements

1. **Python 3.8 or higher**
   ```bash
   python --version
   ```

2. **Required Python packages** (install via requirements.txt):
   ```bash
   pip install -r requirements.txt
   ```

3. **Additional packages for training**:
   ```bash
   pip install roboflow  # For dataset management (optional)
   ```

4. **CUDA and cuDNN** (for GPU acceleration):
   - Download from NVIDIA website
   - Ensure CUDA version matches PyTorch requirements

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('Ultralytics installed successfully')"
```

## Dataset Preparation

### Step 1: List Available Datasets

```bash
python scripts/prepare_dataset.py --list
```

This will display available datasets from Roboflow:
- UAV Person Detection Dataset
- Combatant Detection Dataset
- Soldiers Detection Dataset
- Look Down Folks Dataset

### Step 2: Download Dataset

#### Option A: Manual Download (Recommended)

1. Visit one of the dataset URLs (e.g., https://universe.roboflow.com/militarypersons/uav-person-3)
2. Sign up or log in to Roboflow
3. Click "Download Dataset"
4. Select "YOLOv8" format
5. Download and extract to `datasets/` directory

#### Option B: Using Roboflow API

```bash
# Get instructions for API download
python scripts/prepare_dataset.py --download uav-person --api-key YOUR_API_KEY
```

### Step 3: Verify Dataset Structure

After downloading, your dataset should have this structure:

```
datasets/
└── your-dataset/
    ├── train/
    │   ├── images/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   └── labels/
    │       ├── image1.txt
    │       ├── image2.txt
    │       └── ...
    ├── val/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── data.yaml
```

Verify the structure:
```bash
python scripts/prepare_dataset.py --verify datasets/your-dataset
```

### Step 4: Create/Update data.yaml

If data.yaml doesn't exist or needs updating:

```bash
python scripts/prepare_dataset.py --create-yaml datasets/your-dataset --classes soldier civilian
```

Example data.yaml:
```yaml
path: /path/to/datasets/your-dataset
train: train/images
val: val/images
test: test/images

nc: 2  # number of classes
names: ['soldier', 'civilian']
```

### Step 5: Data Augmentation (Optional but Recommended)

Data augmentation improves model generalization. Roboflow provides built-in augmentation, or you can use Python libraries:

**Recommended augmentations:**
- Rotation: ±15 degrees
- Horizontal flip: 50% probability
- Brightness adjustment: ±20%
- Zoom: 0.8x to 1.2x
- Gaussian noise
- Blur

## Model Training

### Step 1: Choose Base Model

YOLOv8 comes in 5 sizes:

| Model | Size | Speed | mAP | Use Case |
|-------|------|-------|-----|----------|
| YOLOv8n | Nano | Fastest | Lowest | Real-time on edge devices |
| YOLOv8s | Small | Fast | Good | Real-time on standard hardware |
| YOLOv8m | Medium | Moderate | Better | Balanced performance |
| YOLOv8l | Large | Slow | High | High accuracy priority |
| YOLOv8x | XLarge | Slowest | Highest | Maximum accuracy |

**Recommendation**: Start with YOLOv8s for a good balance, upgrade to YOLOv8m or YOLOv8l if you have a powerful GPU.

### Step 2: Start Training

#### Basic Training Command

```bash
python scripts/train_model.py \
    --data datasets/your-dataset/data.yaml \
    --model yolov8s.pt \
    --epochs 100 \
    --batch 16 \
    --imgsz 640
```

#### Advanced Training with Custom Parameters

```bash
python scripts/train_model.py \
    --data datasets/your-dataset/data.yaml \
    --model yolov8m.pt \
    --epochs 200 \
    --batch 32 \
    --imgsz 640 \
    --device 0 \
    --workers 8 \
    --patience 50 \
    --validate \
    --export onnx
```

#### Training Parameters Explained

- `--data`: Path to data.yaml configuration
- `--model`: Base model (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
- `--epochs`: Number of training epochs (100-300 typical)
- `--batch`: Batch size (adjust based on GPU memory: 8-64)
- `--imgsz`: Input image size (640 standard, 1280 for higher resolution)
- `--device`: GPU device (0, 0,1, cpu, auto)
- `--workers`: Data loading threads (4-8 typical)
- `--patience`: Early stopping patience

### Step 3: Monitor Training

Training progress is saved in `runs/train/aerial_threat_detector/`:

```
runs/train/aerial_threat_detector/
├── weights/
│   ├── best.pt          # Best model checkpoint
│   └── last.pt          # Last epoch checkpoint
├── results.png          # Training curves
├── confusion_matrix.png # Confusion matrix
├── labels.jpg           # Ground truth visualization
├── train_batch0.jpg     # Training batch samples
└── val_batch0_pred.jpg  # Validation predictions
```

**View training results:**
```bash
# View results image
xdg-open runs/train/aerial_threat_detector/results.png

# Or use Python
python -c "from PIL import Image; Image.open('runs/train/aerial_threat_detector/results.png').show()"
```

### Step 4: Training Tips

#### If Training is Too Slow:
- Reduce batch size: `--batch 8`
- Use smaller model: `--model yolov8n.pt`
- Reduce image size: `--imgsz 416`
- Use fewer workers: `--workers 4`

#### If Running Out of GPU Memory:
- Reduce batch size
- Use smaller model
- Reduce image size
- Enable gradient accumulation (not directly supported in ultralytics CLI)

#### If Model is Underfitting:
- Train for more epochs
- Use larger model (yolov8m, yolov8l)
- Reduce data augmentation
- Check dataset quality

#### If Model is Overfitting:
- Add more training data
- Increase data augmentation
- Use early stopping (--patience)
- Add dropout (modify model architecture)

## Evaluation and Testing

### Step 1: Validate Trained Model

```bash
python scripts/train_model.py \
    --data datasets/your-dataset/data.yaml \
    --validate-only runs/train/aerial_threat_detector/weights/best.pt
```

Key metrics to evaluate:
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5 (target: > 0.80)
- **mAP@0.5:0.95**: mAP across IoU thresholds 0.5 to 0.95 (target: > 0.60)
- **Precision**: Accuracy of positive predictions (target: > 0.85)
- **Recall**: Coverage of actual positives (target: > 0.80)

### Step 2: Test on Sample Images/Videos

```bash
# Test on image
python src/aerial_threat_detector.py

# Then select option 1 and provide image path

# Or use detection server
python src/detection_server.py --source-type image --source-path test_image.jpg
```

### Step 3: Analyze Results

**Confusion Matrix Analysis:**
- Check for class imbalance
- Identify frequently confused classes
- Assess false positive/negative rates

**Error Analysis:**
- Review misclassified examples
- Identify patterns in errors (lighting, altitude, occlusion)
- Consider targeted data augmentation

### Step 4: Test Under Various Conditions

Test the model with:
- Different lighting conditions (day, night, dusk)
- Various altitudes (50m, 100m, 200m, 500m)
- Different weather conditions
- Varying image quality and resolution
- Different camera angles

## Model Deployment

### Step 1: Copy Best Model

```bash
# Copy best model to project root
cp runs/train/aerial_threat_detector/weights/best.pt ./best.pt
```

### Step 2: Test Deployment

```bash
# Test with detection server
python src/detection_server.py --model-path best.pt

# In another terminal, start Electron app
cd electron-app
npm start
```

### Step 3: Export for Production (Optional)

```bash
# Export to ONNX for cross-platform deployment
python scripts/train_model.py \
    --data datasets/your-dataset/data.yaml \
    --export onnx
```

Export formats available:
- ONNX: Cross-platform inference
- TorchScript: PyTorch deployment
- CoreML: iOS/macOS deployment
- TFLite: Mobile/edge devices

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory
**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Reduce batch size
python scripts/train_model.py --data datasets/your-dataset/data.yaml --batch 8

# Or use CPU (much slower)
python scripts/train_model.py --data datasets/your-dataset/data.yaml --device cpu
```

#### 2. No Module Named 'ultralytics'
**Error**: `ModuleNotFoundError: No module named 'ultralytics'`

**Solution**:
```bash
pip install ultralytics
```

#### 3. Dataset Not Found
**Error**: `FileNotFoundError: Dataset not found`

**Solution**:
- Verify data.yaml path is correct
- Check dataset directory structure
- Use absolute paths in data.yaml

#### 4. Poor Model Performance

**Symptoms**: Low mAP, poor detection accuracy

**Solutions**:
1. **Check dataset quality**:
   - Verify label accuracy
   - Check for corrupted images
   - Ensure balanced class distribution

2. **Increase training**:
   - Train for more epochs
   - Reduce early stopping patience

3. **Adjust hyperparameters**:
   - Try different learning rates
   - Adjust augmentation parameters

4. **Use larger model**:
   - Switch from yolov8n to yolov8s or yolov8m

#### 5. Model Overfitting

**Symptoms**: High training accuracy, low validation accuracy

**Solutions**:
- Add more training data
- Increase data augmentation
- Use early stopping
- Reduce model complexity

#### 6. Training Crashes

**Possible causes**:
- Insufficient memory
- Corrupted data
- Invalid labels

**Solutions**:
```bash
# Verify dataset
python scripts/prepare_dataset.py --verify datasets/your-dataset

# Check for corrupted images
python -c "
import os
from PIL import Image
for root, dirs, files in os.walk('datasets/your-dataset'):
    for file in files:
        if file.endswith(('.jpg', '.png')):
            try:
                img = Image.open(os.path.join(root, file))
                img.verify()
            except:
                print(f'Corrupted: {file}')
"
```

## Performance Benchmarks

### Expected Training Times (YOLOv8s, 100 epochs)

| Dataset Size | GPU | Time |
|--------------|-----|------|
| 1,000 images | RTX 3060 | ~1 hour |
| 5,000 images | RTX 3060 | ~4-5 hours |
| 10,000 images | RTX 3060 | ~8-10 hours |
| 1,000 images | CPU only | ~12-15 hours |

### Expected Performance Metrics

| Model | mAP@0.5 | Precision | Recall | FPS (RTX 3060) |
|-------|---------|-----------|--------|----------------|
| YOLOv8n | 0.75-0.80 | 0.80-0.85 | 0.75-0.80 | ~200 |
| YOLOv8s | 0.80-0.85 | 0.85-0.88 | 0.80-0.85 | ~150 |
| YOLOv8m | 0.85-0.90 | 0.88-0.92 | 0.85-0.88 | ~100 |

*Note: Actual performance depends on dataset quality, quantity, and diversity*

## Best Practices

1. **Start Small**: Begin with a subset of data to verify pipeline
2. **Iterative Improvement**: Train, evaluate, improve data, repeat
3. **Version Control**: Keep track of model versions and training configs
4. **Regular Backups**: Save model checkpoints frequently
5. **Document Everything**: Record training parameters and results
6. **Ethical Considerations**: Review ethical implications before deployment

## Additional Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow Documentation](https://docs.roboflow.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Computer Vision Best Practices](https://github.com/microsoft/computervision-recipes)

## Support

For questions or issues:
1. Check this guide and troubleshooting section
2. Review project documentation in `docs/`
3. Consult YOLOv8 and Roboflow documentation
4. Create an issue in the project repository

---

**Happy Training! 🚀**
