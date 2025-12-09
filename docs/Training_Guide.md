# Model Training Guide

## Overview

This guide provides step-by-step instructions for training your own YOLOv8 model for aerial threat detection (soldier and civilian classification).

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Dataset Preparation](#dataset-preparation)
3. [Dataset Download from Roboflow](#dataset-download-from-roboflow)
4. [Training the Model](#training-the-model)
5. [Model Evaluation](#model-evaluation)
6. [Using the Trained Model](#using-the-trained-model)

## Prerequisites

### Hardware Requirements

**Minimum:**
- CPU: Intel i5 or AMD Ryzen 5
- RAM: 16GB
- Storage: 20GB free space
- GPU: Optional but strongly recommended

**Recommended:**
- CPU: Intel i7/i9 or AMD Ryzen 7/9
- RAM: 32GB
- Storage: 50GB+ free SSD space
- GPU: NVIDIA RTX 3060 or better (8GB+ VRAM)
- CUDA: 11.0 or higher

### Software Requirements

```bash
# Install Python dependencies
pip install -r requirements.txt

# Additional training dependencies
pip install roboflow
```

## Dataset Preparation

### Dataset Structure

Your dataset should follow the YOLO format with this structure:

```
dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
├── test/  (optional)
│   ├── images/
│   └── labels/
└── data.yaml
```

### Label Format

Each label file (`.txt`) contains one line per object:
```
class_id center_x center_y width height
```

Where:
- `class_id`: Class index (0 for civilian, 1 for soldier)
- `center_x, center_y`: Normalized center coordinates (0-1)
- `width, height`: Normalized box dimensions (0-1)

Example:
```
0 0.5 0.5 0.2 0.3
1 0.7 0.3 0.15 0.25
```

### Dataset Configuration (data.yaml)

Create a `data.yaml` file in your dataset directory:

```yaml
path: /path/to/dataset  # Absolute path to dataset
train: train/images
val: valid/images
test: test/images  # Optional

names:
  0: civilian
  1: soldier

nc: 2  # Number of classes
```

## Dataset Download from Roboflow

### Step 1: Get Roboflow API Key

1. Create an account at [Roboflow](https://roboflow.com)
2. Navigate to your workspace settings
3. Copy your API key

### Step 2: Download Recommended Datasets

We recommend using one or more of these datasets:

#### Option 1: UAV Person Dataset
```bash
python download_dataset.py download \
  --api-key YOUR_API_KEY \
  --workspace militarypersons \
  --project uav-person-3 \
  --version 1 \
  --output dataset_uav
```

#### Option 2: Combatant Dataset
```bash
python download_dataset.py download \
  --api-key YOUR_API_KEY \
  --workspace minwoo \
  --project combatant-dataset \
  --version 1 \
  --output dataset_combatant
```

#### Option 3: Soldiers Detection
```bash
python download_dataset.py download \
  --api-key YOUR_API_KEY \
  --workspace xphoenixua-nlncq \
  --project soldiers-detection-spf \
  --version 1 \
  --output dataset_soldiers
```

#### Option 4: Look Down Folks
```bash
python download_dataset.py download \
  --api-key YOUR_API_KEY \
  --workspace folks \
  --project look-down-folks \
  --version 1 \
  --output dataset_folks
```

### Step 3: Combine Multiple Datasets (Optional)

If you downloaded multiple datasets, combine them for better model performance:

```bash
python download_dataset.py combine \
  dataset_uav dataset_combatant dataset_soldiers \
  --output combined_dataset
```

### Step 4: Verify Dataset

```bash
python download_dataset.py verify combined_dataset
```

## Training the Model

### Basic Training

Start with default settings (recommended for beginners):

```bash
python train_model.py --dataset combined_dataset
```

### Advanced Training Options

#### Small Model (Faster, Less Accurate)
```bash
python train_model.py \
  --dataset combined_dataset \
  --model yolov8n \
  --epochs 100 \
  --batch 16
```

#### Medium Model (Balanced)
```bash
python train_model.py \
  --dataset combined_dataset \
  --model yolov8m \
  --epochs 150 \
  --batch 8 \
  --imgsz 640
```

#### Large Model (Most Accurate, Slowest)
```bash
python train_model.py \
  --dataset combined_dataset \
  --model yolov8x \
  --epochs 200 \
  --batch 4 \
  --imgsz 640
```

### Training Parameters Explained

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `--model` | Model size (n/s/m/l/x) | yolov8n | yolov8s or yolov8m |
| `--epochs` | Training epochs | 100 | 100-200 |
| `--batch` | Batch size | 16 | 4-32 (adjust for GPU) |
| `--imgsz` | Input image size | 640 | 416-1280 |
| `--lr0` | Initial learning rate | 0.01 | 0.001-0.1 |
| `--patience` | Early stopping | 50 | 30-100 |

### GPU Memory Considerations

| Model Size | Batch Size | GPU Memory Required |
|------------|------------|---------------------|
| YOLOv8n | 32 | 4GB |
| YOLOv8s | 16 | 6GB |
| YOLOv8m | 8 | 8GB |
| YOLOv8l | 4 | 10GB |
| YOLOv8x | 2 | 12GB+ |

If you encounter "CUDA out of memory" errors, reduce the batch size.

### Training on CPU

While not recommended, you can train on CPU:

```bash
python train_model.py \
  --dataset combined_dataset \
  --device cpu \
  --batch 2 \
  --epochs 50
```

**Note:** CPU training is 10-50x slower than GPU training.

## Understanding Training Output

### Training Metrics

Monitor these metrics during training:

- **Loss**: Should decrease over time (lower is better)
  - `box_loss`: Bounding box regression loss
  - `cls_loss`: Classification loss
  - `dfl_loss`: Distribution focal loss

- **mAP@0.5**: Mean Average Precision at IoU 0.5 (higher is better)
- **mAP@0.5:0.95**: Mean AP over IoU thresholds 0.5-0.95 (higher is better)
- **Precision**: Accuracy of positive predictions (higher is better)
- **Recall**: Coverage of actual positives (higher is better)

### Good Training Indicators

✅ **Signs of good training:**
- Steadily decreasing loss
- mAP@0.5 > 0.75
- mAP@0.5:0.95 > 0.50
- Precision and Recall > 0.70

❌ **Warning signs:**
- Loss plateaus early
- mAP < 0.50
- Large gap between train and validation metrics (overfitting)

## Model Evaluation

### Evaluate on Test Set

After training, evaluate your model:

```bash
python examples/evaluation_example.py \
  --model best.pt \
  --dataset-yaml combined_dataset/data.yaml \
  --test-images test_images/ \
  --output evaluation_results
```

This will generate:
- Performance metrics (mAP, Precision, Recall, F1-Score)
- Confusion matrix
- Prediction visualizations
- JSON report with detailed statistics

### Expected Performance

Target metrics for a good model:

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| mAP@0.5 | 0.70 | 0.85 | 0.95+ |
| Precision | 0.75 | 0.85 | 0.95+ |
| Recall | 0.70 | 0.80 | 0.90+ |
| F1-Score | 0.72 | 0.82 | 0.92+ |

## Using the Trained Model

### Model Location

After training, your best model is saved to:
```
runs/detect/aerial_threat_model/weights/best.pt
```

And copied to the project root as:
```
best.pt
```

### Test the Model

#### Single Image
```bash
python src/detection_server.py \
  --source-type image \
  --source-path test_image.jpg
```

#### Video File
```bash
python src/detection_server.py \
  --source-type video \
  --source-path test_video.mp4
```

#### Webcam
```bash
python src/detection_server.py \
  --source-type webcam
```

### Use with Electron App

1. Ensure `best.pt` is in the project root
2. Start the application:
   ```bash
   python src/detection_server.py
   ```
3. In another terminal:
   ```bash
   cd electron-app
   npm start
   ```

## Troubleshooting

### Common Issues

#### Issue: "CUDA out of memory"
**Solution:** Reduce batch size or use smaller model

#### Issue: "No module named 'roboflow'"
**Solution:** Install roboflow: `pip install roboflow`

#### Issue: Low mAP scores
**Solutions:**
- Train for more epochs
- Use larger model
- Add more training data
- Increase image size
- Adjust augmentation parameters

#### Issue: Overfitting (high train, low val performance)
**Solutions:**
- Add more training data
- Increase data augmentation
- Reduce model size
- Enable early stopping

## Best Practices

1. **Data Quality Over Quantity**
   - Ensure proper labeling
   - Diverse training samples
   - Balance class distribution

2. **Start Small**
   - Begin with YOLOv8s or YOLOv8n
   - Use default hyperparameters
   - Train on smaller subset first

3. **Monitor Training**
   - Check loss curves
   - Validate on held-out set
   - Stop if no improvement

4. **Iterative Improvement**
   - Analyze failures
   - Add challenging examples
   - Retrain and evaluate

## Additional Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow Documentation](https://docs.roboflow.com/)
- [YOLO Training Tips](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)

## Support

For questions or issues:
1. Check the documentation
2. Review error messages carefully
3. Search for similar issues
4. Create an issue in the repository with:
   - Error message
   - System specifications
   - Steps to reproduce
