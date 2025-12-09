# Aerial Threat Detection System

## Project Overview

The Aerial Threat Detection System is a comprehensive computer vision solution designed to classify soldiers and civilians in aerial imagery captured by drones. This system combines state-of-the-art deep learning techniques with a user-friendly interface to support reconnaissance and humanitarian operations.

## 🎯 Project Objectives

- **Primary Goal**: Build an image classification model to distinguish soldiers from civilians in aerial images
- **Real-time Processing**: Integrate the trained model with video streams for real-time detection
- **User Interface**: Provide an intuitive Electron-based application for easy interaction
- **Evaluation**: Assess model performance using standard computer vision metrics

## 🏗️ System Architecture

### Components

1. **YOLO Detection Engine** (`src/aerial_threat_detector.py`)
   - Core detection functionality using trained YOLO model
   - Support for images, videos, and real-time webcam input
   - Configurable confidence thresholds and detection parameters

2. **Real-time Server** (`src/detection_server.py`)
   - Flask-SocketIO server for real-time communication
   - WebSocket-based data streaming
   - Multi-threaded processing for optimal performance

3. **Electron GUI Application** (`electron-app/`)
   - Modern, responsive user interface
   - Drag-and-drop file support
   - Real-time visualization of detection results
   - Statistics dashboard and export functionality

4. **Evaluation Tools** (`src/utils/evaluation_utils.py`)
   - Performance metrics calculation (Precision, Recall, F1-Score)
   - Visualization tools for model assessment
   - Video processing utilities

### Data Flow

```
Input Source → YOLO Detector → Detection Server → Electron App → User Interface
     ↓              ↓               ↓                ↓             ↓
  Image/Video → Bounding Boxes → WebSocket → Canvas Display → Visual Results
```

## 🚀 Quick Start

**📖 New to the project? See the [Quick Start Guide](docs/Quick_Start.md) for a 5-minute setup!**

### Easy Setup (Recommended)
1. **Double-click `start_app.bat`** - This will automatically:
   - Install all required dependencies
   - Start the detection server
   - Launch the Electron application

### Get a Trained Model

**Option 1: Train Your Own Model (Recommended)**
```bash
# 1. Download dataset from Roboflow
python download_dataset.py download \
  --api-key YOUR_API_KEY \
  --workspace militarypersons \
  --project uav-person-3 \
  --version 1 \
  --output dataset

# 2. Train the model
python train_model.py --dataset dataset --epochs 100

# Model will be saved as 'best.pt'
```

**Option 2: Use Pre-trained Base Model (For Testing)**
```bash
# Download YOLOv8 base model (not specifically trained for soldiers/civilians)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mv yolov8n.pt best.pt
```

### Manual Setup
If you prefer manual setup or the batch file doesn't work:

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Node.js Dependencies**:
   ```bash
   cd electron-app
   npm install
   cd ..
   ```

3. **Start Detection Server** (in one terminal):
   ```bash
   python src/detection_server.py
   ```

4. **Start Electron App** (in another terminal):
   ```bash
   cd electron-app
   npm start
   ```

## 📱 How to Use

1. **Upload a Video**: 
   - Click "Select Video" button or drag & drop a video file
   - Supported formats: MP4, AVI, MOV, MKV, WMV

2. **Start Detection**:
   - Click the "Start Detection" button
   - The system will process the video and show results in real-time

3. **View Results**:
   - Watch detected soldiers (red boxes) and civilians (green boxes) 
   - Check detection statistics in the right panel
   - View detailed results in the bottom panel

4. **Export Results**:
   - Save detection results as JSON file
   - Export processed video (coming soon)

## 🚀 Features

### Core Functionality
- ✅ Real-time soldier/civilian detection
- ✅ Support for multiple input sources (images, videos, webcam)
- ✅ Adjustable confidence thresholds
- ✅ Bounding box visualization with class labels
- ✅ Detection statistics and analytics

### User Interface
- ✅ Modern, responsive design
- ✅ Drag-and-drop file input
- ✅ Real-time detection results display
- ✅ Configurable detection parameters
- ✅ Export functionality for results
- ✅ Dark theme optimized for surveillance applications

### Technical Features
- ✅ Multi-threaded processing
- ✅ WebSocket-based real-time communication
- ✅ GPU acceleration support (CUDA)
- ✅ Comprehensive error handling
- ✅ Performance monitoring and statistics

## 📋 Requirements

### System Requirements
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: 3.8 or higher
- **Node.js**: 14.0 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 2GB free space
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### Python Dependencies
```
torch>=1.12.0
torchvision>=0.13.0
ultralytics>=8.0.0
opencv-python>=4.6.0
pillow>=8.3.2
numpy>=1.21.0
matplotlib>=3.5.0
flask>=2.0.0
flask-socketio>=5.0.0
```

### Node.js Dependencies
```
electron>=26.0.0
socket.io-client>=4.7.0
```

## 🛠️ Installation and Setup

### 1. Clone and Setup Python Environment

```bash
# Navigate to your project directory
cd "c:\Users\lilir\OneDrive\Desktop\csc-final"

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Setup Electron Application

```bash
# Navigate to electron app directory
cd electron-app

# Install Node.js dependencies
npm install
```

### 3. Verify Model File

Ensure your trained YOLO model (`best.pt`) is in the root directory:
```
csc-final/
├── best.pt                 # Your trained YOLO model
├── src/
├── electron-app/
└── ...
```

## 🎮 Usage Guide

### Method 1: Using the Electron GUI Application

1. **Start the Application**
   ```bash
   cd electron-app
   npm start
   ```

2. **Select Input Source**
   - Click "Select Image" for single image detection
   - Click "Select Video" for video file processing
   - Click "Start Webcam" for real-time camera input

3. **Configure Detection Settings**
   - Adjust confidence threshold using the slider
   - Toggle label and confidence display options
   - Access advanced settings through the Settings menu

4. **Run Detection**
   - Click "Start Detection" to begin processing
   - View real-time results in the main display area
   - Monitor statistics in the sidebar panel

5. **Export Results**
   - Use "Save Results" to export detection data as JSON
   - Use "Export Video" for processed video files

### Method 2: Command Line Interface

For direct detection without the GUI:

```bash
# Image detection
python src/detection_server.py --source-type image --source-path "path/to/image.jpg"

# Video detection
python src/detection_server.py --source-type video --source-path "path/to/video.mp4"

# Webcam detection
python src/detection_server.py --source-type webcam

# Custom confidence threshold
python src/detection_server.py --source-type image --source-path "image.jpg" --confidence 0.7
```

### Method 3: Python API

```python
from src.aerial_threat_detector import AerialThreatDetector

# Initialize detector
detector = AerialThreatDetector("best.pt", confidence_threshold=0.5)

# Process single image
annotated_image, detections = detector.detect_image("path/to/image.jpg")

# Process video
detector.detect_video("path/to/video.mp4", "output_video.mp4")

# Real-time webcam
detector.detect_webcam()
```

## 📊 Model Performance

### Target Metrics

The system provides comprehensive evaluation tools to assess model performance:

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| **mAP@0.5** | 0.70 | 0.85 | 0.95+ |
| **Precision** | 0.75 | 0.85 | 0.95+ |
| **Recall** | 0.70 | 0.80 | 0.90+ |
| **F1-Score** | 0.72 | 0.82 | 0.92+ |

### Model Training

**📚 For complete training instructions, see [Training Guide](docs/Training_Guide.md)**

#### Quick Training

```bash
# 1. Download dataset
python download_dataset.py download \
  --api-key YOUR_ROBOFLOW_API_KEY \
  --workspace militarypersons \
  --project uav-person-3 \
  --version 1 \
  --output dataset

# 2. Train model
python train_model.py \
  --dataset dataset \
  --model yolov8s \
  --epochs 100 \
  --batch 16
```

#### Available Datasets

See [Dataset Information](docs/Dataset_Information.md) for complete details on:
- UAV Person Detection Dataset
- Combatant Detection Dataset
- Soldiers Detection Dataset
- Look Down Folks Dataset

All datasets are available on [Roboflow Universe](https://universe.roboflow.com/)

### Evaluation

Evaluate your trained model:

```bash
python examples/evaluation_example.py \
  --model best.pt \
  --dataset-yaml dataset/data.yaml \
  --test-images test_images/ \
  --output evaluation_results
```

This generates:
- Precision, Recall, F1-Score metrics
- mAP@0.5 and mAP@0.5:0.95
- Confusion matrix
- Per-class performance analysis
- Visualization plots

### Real-world Performance Considerations

- **Lighting Conditions**: System tested under various lighting scenarios
- **Altitude Variations**: Effective detection from 50-500 meters altitude
- **Weather Conditions**: Performance may vary in adverse weather
- **Resolution Requirements**: Minimum 640x480 input resolution recommended

## 🔧 Configuration Options

### Detection Parameters

| Parameter | Default | Range | Description |
|-----------|---------|--------|-------------|
| `confidence_threshold` | 0.5 | 0.1-1.0 | Minimum confidence for detections |
| `iou_threshold` | 0.5 | 0.1-1.0 | Non-maximum suppression threshold |
| `max_detections` | 100 | 1-1000 | Maximum detections per frame |
| `box_thickness` | 2 | 1-10 | Bounding box line thickness |
| `font_scale` | 0.6 | 0.3-2.0 | Label text size |

### System Configuration

```python
# GPU/CPU selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model optimization
model.half()  # Use FP16 for faster inference on GPU
```

## 📈 Performance Optimization

### Hardware Recommendations

1. **GPU Acceleration**
   - NVIDIA RTX series for optimal performance
   - Minimum 4GB VRAM for real-time processing
   - CUDA 11.0 or higher

2. **CPU Processing**
   - Intel i7 or AMD Ryzen 7 (8 cores minimum)
   - 16GB RAM for video processing
   - SSD storage for faster I/O

### Software Optimizations

1. **Model Optimization**
   ```python
   # Enable half precision (FP16) for faster inference
   model.half()
   
   # Use TensorRT for NVIDIA GPUs
   model = torch.jit.script(model)
   ```

2. **Multi-threading**
   ```python
   # Parallel frame processing
   from concurrent.futures import ThreadPoolExecutor
   
   with ThreadPoolExecutor(max_workers=4) as executor:
       futures = [executor.submit(process_frame, frame) for frame in frames]
   ```

## 🐛 Troubleshooting

### Common Issues and Solutions

1. **Model Loading Errors**
   ```
   Error: Could not load model file
   Solution: Verify best.pt exists and is a valid YOLO model file
   ```

2. **CUDA Out of Memory**
   ```
   Error: CUDA out of memory
   Solution: Reduce batch size or use CPU mode
   ```

3. **Webcam Access Issues**
   ```
   Error: Could not access webcam
   Solution: Check camera permissions and ensure no other applications are using the camera
   ```

4. **Electron App Won't Start**
   ```
   Error: Application failed to start
   Solution: Run 'npm install' in electron-app directory
   ```

### Debug Mode

Enable debug output for detailed information:

```bash
# Python debug mode
python src/detection_server.py --debug

# Electron debug mode
npm run dev
```

## 📁 Project Structure

```
final-project_CSC126/
├── best.pt                          # Trained YOLO model (not in repo, see training guide)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── train_model.py                   # Model training script
├── download_dataset.py              # Dataset download/preparation utility
│
├── src/                             # Core Python modules
│   ├── aerial_threat_detector.py    # Main detection class
│   ├── detection_server.py          # Flask-SocketIO server
│   └── utils/
│       ├── evaluation_utils.py      # Performance evaluation tools
│       └── __init__.py
│
├── electron-app/                    # Electron GUI application
│   ├── package.json                 # Node.js dependencies
│   ├── main.js                      # Electron main process
│   ├── index.html                   # Main application window
│   ├── styles.css                   # Application styling
│   └── renderer.js                  # Frontend JavaScript
│
├── docs/                            # Comprehensive documentation
│   ├── Quick_Start.md               # 5-minute quick start guide
│   ├── Training_Guide.md            # Complete model training guide
│   ├── Dataset_Information.md       # Dataset sources and info
│   ├── Ethical_Considerations.md    # Ethical guidelines and compliance
│   ├── Presentation_Template.md     # Project presentation template
│   └── Technical_Report.md          # Technical implementation details
│
└── examples/                        # Usage examples and scripts
    ├── basic_usage.py               # Simple detection examples
    ├── batch_processing.py          # Batch processing script
    └── evaluation_example.py        # Model evaluation script
```

## 🚀 Advanced Features

### Example Scripts

The `examples/` directory contains ready-to-use scripts:

#### 1. Basic Usage Examples
```bash
python examples/basic_usage.py
```
Interactive menu with:
- Single image detection
- Video file processing
- Real-time webcam detection
- Custom confidence thresholds
- Batch image processing
- Frame-by-frame processing

#### 2. Batch Processing
```bash
# Process multiple images
python examples/batch_processing.py \
  --model best.pt \
  --input test_images/ \
  --output results/ \
  --type images

# Process multiple videos
python examples/batch_processing.py \
  --model best.pt \
  --input test_videos/ \
  --output results/ \
  --type videos
```

#### 3. Model Evaluation
```bash
python examples/evaluation_example.py \
  --model best.pt \
  --dataset-yaml dataset/data.yaml \
  --test-images test_images/ \
  --output evaluation_results
```

### Custom Model Integration

Use your own trained models:

```python
# Load custom model
detector = AerialThreatDetector("path/to/custom_model.pt")

# Verify model compatibility
if detector.model is not None:
    print(f"Model loaded with {len(detector.class_names)} classes")
```

### API Integration

Expose detection as REST API:

```python
from flask import Flask, request, jsonify
from src.aerial_threat_detector import AerialThreatDetector

app = Flask(__name__)
detector = AerialThreatDetector("best.pt")

@app.route('/detect', methods=['POST'])
def detect():
    # Handle image upload and return detections
    pass
```

## 📚 Documentation

### Complete Guides

- **[Quick Start Guide](docs/Quick_Start.md)** - Get started in 5 minutes
- **[Training Guide](docs/Training_Guide.md)** - Complete model training instructions
- **[Dataset Information](docs/Dataset_Information.md)** - Dataset sources and preparation
- **[Ethical Considerations](docs/Ethical_Considerations.md)** - Ethical guidelines and compliance
- **[Technical Report](docs/Technical_Report.md)** - Technical implementation details
- **[Presentation Template](docs/Presentation_Template.md)** - Project presentation guide

### Key Topics

#### Getting Started
1. Install dependencies
2. Download or train a model
3. Run the application
4. Test detection capabilities

#### Model Training
1. Get Roboflow API key
2. Download datasets
3. Configure training parameters
4. Train and evaluate model

#### Deployment
1. Review ethical guidelines
2. Ensure legal compliance
3. Implement human oversight
4. Monitor performance

### External Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow Tutorials](https://roboflow.com/tutorials)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Electron Documentation](https://www.electronjs.org/docs)

## ⚠️ Ethical Notice

### Educational Purpose

This system is designed for:
- ✅ Educational demonstration
- ✅ Research and development
- ✅ Proof of concept
- ✅ Academic learning

### Not Intended For

- ❌ Production military applications without oversight
- ❌ Autonomous targeting systems
- ❌ Privacy-invasive surveillance
- ❌ Discriminatory applications

### Important Principles

1. **Human Oversight Required:** All critical decisions must have human approval
2. **Privacy Protection:** Comply with data protection laws (GDPR, CCPA, etc.)
3. **Civilian Safety:** Prioritize protection of non-combatants
4. **Transparency:** Document all system use and limitations
5. **Accountability:** Maintain clear responsibility chains

**📖 Read the complete [Ethical Considerations Guide](docs/Ethical_Considerations.md)**

## 🤝 Contributing

### Development Guidelines

1. **Code Style**: Follow PEP 8 for Python code
2. **Documentation**: Document all public functions and classes
3. **Testing**: Add tests for new features
4. **Error Handling**: Implement comprehensive error handling

### Adding New Features

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is developed for educational purposes as part of a Computer Science final project. All rights reserved.

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for the YOLO framework
- **Datasets**: Roboflow community for aerial surveillance datasets
- **Libraries**: PyTorch, OpenCV, Flask, and Electron communities
- **Inspiration**: Defense and humanitarian surveillance applications

## 📞 Support

For questions or support:
- Create an issue in the project repository
- Contact the development team
- Check the documentation in the `docs/` directory

---

**Note**: This system is designed for educational and research purposes. Always ensure compliance with local laws and regulations when using surveillance technology.