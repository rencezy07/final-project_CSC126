# Quick Start Guide

## Getting Started in 5 Minutes

This guide will help you get the Aerial Threat Detection System up and running quickly.

## Prerequisites

- Python 3.8 or higher
- Node.js 14.0 or higher
- 8GB RAM minimum (16GB recommended)
- (Optional) NVIDIA GPU with CUDA for training

## Step 1: Clone the Repository

```bash
git clone https://github.com/rencezy07/final-project_CSC126.git
cd final-project_CSC126
```

## Step 2: Install Dependencies

### Option A: Using the Quick Start Script (Windows)

Double-click `start_app.bat` - this will:
- Install Python dependencies
- Install Node.js dependencies
- Start the detection server
- Launch the application

### Option B: Manual Installation

**Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**Install Node.js dependencies:**
```bash
cd electron-app
npm install
cd ..
```

## Step 3: Get a Trained Model

### Option A: Download Pre-trained Model (Recommended)

1. Download a pre-trained YOLOv8 model from:
   - [Roboflow Universe](https://universe.roboflow.com/)
   - Or train your own (see Step 4)

2. Place the model file as `best.pt` in the project root

### Option B: Use YOLOv8 Base Model (For Testing)

```bash
# Download YOLOv8 base model (no custom training)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**Note:** This won't detect soldiers/civilians specifically, but allows you to test the system.

## Step 4: Train Your Own Model (Optional)

### Quick Training Guide

1. **Get a Roboflow API Key:**
   - Sign up at [Roboflow](https://roboflow.com)
   - Get your API key from workspace settings

2. **Download Dataset:**
   ```bash
   python download_dataset.py download \
     --api-key YOUR_API_KEY \
     --workspace militarypersons \
     --project uav-person-3 \
     --version 1 \
     --output dataset
   ```

3. **Train Model:**
   ```bash
   python train_model.py --dataset dataset --epochs 100
   ```

4. **Wait for Training:**
   - Training takes 2-6 hours depending on hardware
   - Model will be saved as `best.pt`

**For detailed training instructions, see `docs/Training_Guide.md`**

## Step 5: Run the Application

### Method 1: Using Start Script (Easiest)

Windows:
```bash
start_app.bat
```

Linux/Mac:
```bash
python src/detection_server.py &
cd electron-app && npm start
```

### Method 2: Manual Start

**Terminal 1 - Start Detection Server:**
```bash
python src/detection_server.py
```

**Terminal 2 - Start GUI Application:**
```bash
cd electron-app
npm start
```

## Step 6: Test the System

### Using the GUI

1. **Select Input:**
   - Click "Select Image" for image detection
   - Click "Select Video" for video detection
   - Click "Start Webcam" for live detection

2. **Start Detection:**
   - Click "Start Detection"
   - View results in real-time

3. **Adjust Settings:**
   - Use confidence slider (0.1-1.0)
   - Toggle labels and confidence display

4. **Export Results:**
   - Click "Save Results" to export JSON

### Using Command Line

**Test on image:**
```bash
python src/detection_server.py --source-type image --source-path test.jpg
```

**Test on video:**
```bash
python src/detection_server.py --source-type video --source-path test.mp4
```

**Test webcam:**
```bash
python src/detection_server.py --source-type webcam
```

## Step 7: Run Examples

### Basic Usage Examples

```bash
python examples/basic_usage.py
```

Select from menu:
1. Single Image Detection
2. Video Detection
3. Real-time Webcam Detection
4. Custom Confidence Threshold
5. Batch Image Processing
6. Frame-by-Frame Processing

### Batch Processing

Process multiple images:
```bash
python examples/batch_processing.py \
  --model best.pt \
  --input test_images/ \
  --output results/ \
  --type images
```

Process multiple videos:
```bash
python examples/batch_processing.py \
  --model best.pt \
  --input test_videos/ \
  --output results/ \
  --type videos
```

### Model Evaluation

```bash
python examples/evaluation_example.py \
  --model best.pt \
  --test-images test_images/ \
  --output evaluation_results
```

## Troubleshooting

### Issue: "Model file not found"

**Solution:**
- Ensure `best.pt` is in the project root
- Or download a base YOLOv8 model:
  ```bash
  python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
  mv yolov8n.pt best.pt
  ```

### Issue: "CUDA out of memory"

**Solution:**
- Use CPU mode: Add `--device cpu` to commands
- Or reduce batch size during training

### Issue: "Port 5000 already in use"

**Solution:**
```bash
# Change port
python src/detection_server.py --port 5001
```

### Issue: "npm install fails"

**Solution:**
```bash
cd electron-app
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

### Issue: "Module not found"

**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

## Next Steps

### Learn More

- **Full Documentation:** See `README.md`
- **Training Guide:** See `docs/Training_Guide.md`
- **Dataset Info:** See `docs/Dataset_Information.md`
- **Technical Details:** See `docs/Technical_Report.md`

### Customize the System

1. **Adjust Detection Parameters:**
   - Edit confidence threshold in GUI
   - Modify `src/aerial_threat_detector.py` for advanced changes

2. **Customize UI:**
   - Edit `electron-app/index.html` for layout
   - Edit `electron-app/styles.css` for styling
   - Edit `electron-app/renderer.js` for functionality

3. **Add New Features:**
   - See example scripts for reference
   - Follow existing code structure

### Train Better Models

1. **Collect More Data:**
   - Use multiple Roboflow datasets
   - Combine datasets for diversity

2. **Optimize Training:**
   - Adjust hyperparameters
   - Use larger model (yolov8m, yolov8l)
   - Train for more epochs

3. **Evaluate and Iterate:**
   - Run evaluation script
   - Analyze failure cases
   - Add challenging examples

## Support and Resources

### Documentation
- README: `README.md`
- Training Guide: `docs/Training_Guide.md`
- Technical Report: `docs/Technical_Report.md`
- Ethical Guidelines: `docs/Ethical_Considerations.md`

### Code Examples
- Basic Usage: `examples/basic_usage.py`
- Batch Processing: `examples/batch_processing.py`
- Evaluation: `examples/evaluation_example.py`

### External Resources
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow Tutorials](https://roboflow.com/tutorials)
- [OpenCV Documentation](https://docs.opencv.org/)

### Getting Help

1. Check documentation in `docs/` folder
2. Review example scripts
3. Search for similar issues online
4. Create an issue on GitHub with:
   - Error message
   - Steps to reproduce
   - System information

## Summary

You now have a working Aerial Threat Detection System! 

**Basic workflow:**
1. ✅ Install dependencies
2. ✅ Get or train a model
3. ✅ Run the application
4. ✅ Test detection
5. ✅ Review results

**For production use:**
- Read ethical guidelines
- Ensure legal compliance
- Maintain human oversight
- Document all operations

**Have fun and use responsibly!** 🚁🎯
