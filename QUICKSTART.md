# Quick Start Guide - Aerial Threat Detection System

## 🚀 Getting Started in 5 Minutes

### Prerequisites Check
- ✅ Python 3.8+ installed
- ✅ Node.js 16+ installed  
- ✅ Your trained `yolo11s.pt` model file

### Automated Setup

**Windows:**
```cmd
setup.bat
```

**Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

### Manual Setup

1. **Backend Setup:**
```bash
cd backend
pip install -r requirements.txt
```

2. **Frontend Setup:**
```bash
cd frontend
npm install
```

3. **Start System:**
```bash
# Terminal 1 (Backend)
cd backend && python api_server.py

# Terminal 2 (Frontend)
cd frontend && npm start
```

## 🎯 Quick Usage

### Image Detection
1. Open application → **Image Detection** tab
2. Drag & drop image or click **Load Image**
3. View results with soldier/civilian classifications

### Video Processing
1. Go to **Video Processing** tab
2. Upload video file → Click **Process Video**
3. Monitor progress and view detailed statistics

### Live Camera Feed  
1. Select **Live Feed** tab
2. Choose camera → Adjust confidence threshold
3. Click **Start Live Feed** for real-time detection

### Export Results
- Go to **Statistics** tab
- Click **Export Data** for JSON/CSV reports
- View comprehensive analytics and charts

## 🔧 Configuration

### Basic Settings
- **Confidence Threshold:** 0.1-1.0 (default: 0.5)
- **API Endpoint:** http://localhost:5000/api (configurable)
- **Camera Source:** Auto-detect available cameras

### Advanced Settings  
Access via menu or `Ctrl+,`:
- Detection parameters
- Display options
- Performance tuning
- Alert thresholds

## ⚠️ Troubleshooting

**Connection Issues:**
- Ensure backend server is running on port 5000
- Check status indicator (top-right corner)

**Model Loading Error:**
- Verify `yolo11s.pt` is in project root directory
- Check file permissions and integrity

**Camera Access:**
- Grant camera permissions
- Try different camera indices (0, 1, 2...)

**Performance Issues:**
- Close unnecessary applications
- Lower video resolution/confidence threshold
- Enable GPU acceleration if available

## 📞 Quick Support

- **Health Check:** http://localhost:5000/api/health
- **Help Menu:** F1 in application
- **Full Documentation:** README.md

## 🎮 Keyboard Shortcuts

- `Ctrl+O` - Load Image
- `Ctrl+Shift+O` - Load Video  
- `Ctrl+L` - Toggle Live Feed
- `Ctrl+E` - Export Results
- `Ctrl+,` - Settings
- `F11` - Fullscreen
- `F1` - Help

---

**Ready to detect aerial threats! 🚁**