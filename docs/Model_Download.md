# Model Download and Setup Instructions

## Overview

This guide explains how to obtain a trained YOLO model for the Aerial Threat Detection System.

## Option 1: Train Your Own Model (Recommended)

### Why Train Your Own?

- **Best Performance:** Tailored to your specific use case
- **Full Control:** Adjust training parameters as needed
- **Understanding:** Learn the complete ML pipeline
- **Customization:** Add custom classes or scenarios

### Steps

1. **Get Roboflow API Key**
   - Sign up at [Roboflow](https://roboflow.com)
   - Navigate to Settings → API
   - Copy your private API key

2. **Download Dataset**
   ```bash
   python download_dataset.py download \
     --api-key YOUR_API_KEY \
     --workspace militarypersons \
     --project uav-person-3 \
     --version 1 \
     --output dataset
   ```

3. **Train Model**
   ```bash
   python train_model.py \
     --dataset dataset \
     --model yolov8s \
     --epochs 100 \
     --batch 16
   ```

4. **Wait for Training**
   - Training time: 2-6 hours (depends on hardware)
   - Monitor progress in terminal
   - Model saved as `best.pt`

**Detailed instructions:** See [Training Guide](Training_Guide.md)

## Option 2: Download Pre-trained Model

### From Roboflow

If someone has already trained a model and shared it on Roboflow:

1. Visit the project on Roboflow Universe
2. Go to the "Deploy" tab
3. Select "Download Model"
4. Choose "YOLOv8" format
5. Download and extract
6. Copy `best.pt` to project root

### From Google Drive / Cloud Storage

If you have access to a pre-trained model:

1. Download the `.pt` file
2. Verify it's a valid YOLO model:
   ```bash
   python -c "from ultralytics import YOLO; model = YOLO('downloaded_model.pt'); print('Valid model')"
   ```
3. Rename to `best.pt`:
   ```bash
   mv downloaded_model.pt best.pt
   ```
4. Place in project root directory

## Option 3: Use YOLOv8 Base Model (For Testing Only)

### Quick Test Setup

If you just want to test the system interface without specific soldier/civilian detection:

```bash
# Download YOLOv8 nano model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Copy to expected location
cp yolov8n.pt best.pt
```

**⚠️ Note:** This base model is trained on COCO dataset (80 general classes) and will not specifically detect soldiers vs civilians. It will detect "person" class only.

### Other Base Models

Choose based on your hardware:

| Model | Size | Speed | Accuracy | GPU Memory |
|-------|------|-------|----------|------------|
| YOLOv8n | 3MB | Fastest | Lower | 2GB |
| YOLOv8s | 11MB | Fast | Good | 4GB |
| YOLOv8m | 26MB | Medium | Better | 6GB |
| YOLOv8l | 44MB | Slow | High | 8GB |
| YOLOv8x | 68MB | Slowest | Highest | 10GB+ |

```bash
# Download any base model
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
cp yolov8s.pt best.pt
```

## Verify Model Installation

### Check Model File

```bash
# On Linux/Mac
ls -lh best.pt

# On Windows
dir best.pt
```

Expected output:
- File exists
- Size: 3MB - 70MB (depending on model)
- File extension: `.pt`

### Test Model Loading

```bash
python -c "
from src.aerial_threat_detector import AerialThreatDetector
detector = AerialThreatDetector('best.pt', confidence_threshold=0.5)
if detector.model is not None:
    print('✅ Model loaded successfully!')
    print(f'Classes: {detector.class_names}')
else:
    print('❌ Failed to load model')
"
```

Expected output:
```
Using device: cuda  # or 'cpu'
Loading model from: best.pt
Model loaded successfully on cuda. Classes: ['person', 'soldier', 'civilian']
✅ Model loaded successfully!
Classes: ['person', 'soldier', 'civilian']
```

## Model File Location

### Correct Location

Place your `best.pt` file in the project root:

```
final-project_CSC126/
├── best.pt              ← Model file here
├── README.md
├── requirements.txt
├── train_model.py
└── ...
```

### Why This Location?

- Detection server looks for `best.pt` by default
- Electron app expects it in root directory
- Training script saves it here automatically

### Custom Location

If you want to use a different location:

```bash
# Specify model path when running
python src/detection_server.py --model-path /path/to/your/model.pt
```

## Model Requirements

### File Format

- **Format:** PyTorch (`.pt` or `.pth`)
- **Framework:** YOLOv8 (Ultralytics)
- **Classes:** Should include person/civilian/soldier classes

### Minimum Performance

For acceptable detection:
- mAP@0.5 > 0.70
- Precision > 0.75
- Recall > 0.70

### Compatibility

Ensure your model is compatible with:
- Ultralytics YOLOv8 (version 8.0+)
- PyTorch 1.12+
- Python 3.8+

## Troubleshooting

### Issue: "Model file not found"

**Solution:**
```bash
# Check if file exists
ls -la | grep best.pt

# If not, download or train model
python train_model.py --dataset dataset
```

### Issue: "Failed to load model"

**Solutions:**

1. **Check file integrity:**
   ```bash
   python -c "from ultralytics import YOLO; YOLO('best.pt')"
   ```

2. **Verify it's a YOLOv8 model:**
   - File should be from Ultralytics YOLOv8
   - Not YOLOv5, YOLOv7, or other versions

3. **Re-download or re-train:**
   - Corrupted file? Download again
   - Training failed? Check error logs

### Issue: "Wrong class names"

If model detects wrong classes (e.g., "car", "dog" instead of "soldier", "civilian"):

**Solution:**
- You're using a base COCO model
- Need to train on soldier/civilian dataset
- See [Training Guide](Training_Guide.md)

### Issue: "CUDA out of memory"

**Solution:**
```bash
# Use CPU instead
python src/detection_server.py --device cpu
```

### Issue: "Model is too slow"

**Solutions:**

1. **Use smaller model:**
   ```bash
   python train_model.py --model yolov8n  # Fastest
   ```

2. **Use GPU acceleration:**
   - Install CUDA
   - Install PyTorch with CUDA support

3. **Reduce input size:**
   - Resize images/videos before processing
   - Use lower resolution camera

## Model Information

### Recommended Models

For this project, we recommend:

**YOLOv8s (Small):**
- Good balance of speed and accuracy
- Works on most GPUs
- Suitable for real-time detection

**YOLOv8m (Medium):**
- Better accuracy
- Slightly slower
- Requires decent GPU (6GB+ VRAM)

### Training Time Estimates

| Hardware | Model Size | Dataset Size | Time |
|----------|------------|--------------|------|
| CPU only | YOLOv8n | 2000 images | 12-24 hours |
| RTX 3060 | YOLOv8s | 2000 images | 2-4 hours |
| RTX 3080 | YOLOv8m | 2000 images | 3-5 hours |
| RTX 4090 | YOLOv8l | 5000 images | 4-6 hours |

## Where to Get Help

### Documentation
- [Training Guide](Training_Guide.md) - Complete training instructions
- [Quick Start](Quick_Start.md) - Fast setup guide
- [Dataset Information](Dataset_Information.md) - Dataset details

### Common Resources
- [Ultralytics Docs](https://docs.ultralytics.com/)
- [Roboflow Universe](https://universe.roboflow.com/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)

### Support
- Check existing issues on GitHub
- Create new issue with details
- Include error messages and system info

## Next Steps

After obtaining your model:

1. ✅ Verify model loads correctly
2. ✅ Test on sample image/video
3. ✅ Evaluate performance metrics
4. ✅ Adjust confidence threshold
5. ✅ Run full application

**Ready to use? See [Quick Start Guide](Quick_Start.md)**

---

**Last Updated:** December 2024  
**Maintained by:** CSC Final Project Team
