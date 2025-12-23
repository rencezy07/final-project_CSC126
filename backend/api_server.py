from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
import base64
import json
from io import BytesIO
from PIL import Image
import threading
import time
from drone_detector import DroneDetector
import os

app = Flask(__name__)
CORS(app)

# Global variables
detector = None
video_stream = None
detection_stats = {'Soldier': 0, 'Civilian': 0}
current_video_path = None

def initialize_detector():
    """Initialize the YOLO detector"""
    global detector
    model_path = "../yolo11s.pt"
    
    if os.path.exists(model_path):
        try:
            detector = DroneDetector(model_path)
            print("✅ Detector initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Error initializing detector: {str(e)}")
            return False
    else:
        print(f"❌ Model file not found: {model_path}")
        return False

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - serves basic info"""
    return jsonify({
        'message': 'Aerial Threat Detection API',
        'version': '1.0.0',
        'status': 'running',
        'detector_loaded': detector is not None,
        'frontend_url': 'Use the desktop app or access API endpoints directly'
    })

@app.route('/api/status', methods=['GET'])
def api_status():
    """API status endpoint"""
    return jsonify({
        'status': 'online',
        'detector_ready': detector is not None,
        'available_endpoints': [
            '/api/health',
            '/api/status', 
            '/api/detect/image',
            '/api/detect/video/upload',
            '/api/stream/start',
            '/api/stream/stop',
            '/api/stream/frame',
            '/api/stats'
        ]
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'detector_loaded': detector is not None,
        'timestamp': time.time()
    })

@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    """Detect objects in uploaded image"""
    if detector is None:
        return jsonify({'error': 'Detector not initialized'}), 500
    
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and process image
        image = Image.open(file.stream)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Perform detection
        confidence_threshold = float(request.form.get('confidence', 0.5))
        annotated_frame, detections = detector.detect_frame(frame, confidence_threshold)
        
        # Convert result back to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Count detections by class
        detection_counts = {'Soldier': 0, 'Civilian': 0}
        for detection in detections:
            class_name = detection['class']
            # Handle case-insensitive matching
            if class_name.lower() == 'soldier':
                detection_counts['Soldier'] += 1
            elif class_name.lower() == 'civilian':
                detection_counts['Civilian'] += 1
        
        return jsonify({
            'success': True,
            'detections': detections,
            'detection_counts': detection_counts,
            'annotated_image': f"data:image/jpeg;base64,{img_base64}",
            'total_detections': len(detections)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect/video/upload', methods=['POST'])
def process_video():
    """Process uploaded video file with optimized batch processing"""
    if detector is None:
        return jsonify({'error': 'Detector not initialized'}), 500
    
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No video selected'}), 400
        
        # Save uploaded video temporarily
        import time
        timestamp = int(time.time())
        temp_video_path = f"temp_{timestamp}_{file.filename}"
        file.save(temp_video_path)
        
        # Process video with optimized batch processing
        confidence_threshold = float(request.form.get('confidence', 0.5))
        skip_frames = int(request.form.get('skip_frames', 2))  # Process every nth frame for speed
        
        # Process video and get comprehensive statistics
        result = detector.process_video_optimized(temp_video_path, confidence_threshold, skip_frames)
        
        # Cleanup
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        return jsonify({
            'success': True,
            'statistics': result['statistics'],
            'processing_info': result['processing_info'],
            'detections_timeline': result['detections_timeline'],
            'message': 'Video processed successfully'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect/video/stream', methods=['POST'])
def process_video_stream():
    """Process video with frame-by-frame streaming"""
    if detector is None:
        return jsonify({'error': 'Detector not initialized'}), 500
    
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No video selected'}), 400
        
        # Save uploaded video temporarily
        temp_video_path = f"temp_{file.filename}"
        file.save(temp_video_path)
        
        # Get video info
        cap = cv2.VideoCapture(temp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Store video path for frame streaming
        global current_video_path
        current_video_path = temp_video_path
        
        return jsonify({
            'success': True,
            'video_id': temp_video_path,
            'total_frames': total_frames,
            'fps': fps,
            'message': 'Video ready for streaming'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect/video/frame/<int:frame_number>')
def get_video_frame(frame_number):
    """Get specific frame from video with detections"""
    if detector is None:
        return jsonify({'error': 'Detector not initialized'}), 500
    
    try:
        global current_video_path
        if not hasattr(get_video_frame, 'current_video_path') and 'current_video_path' not in globals():
            return jsonify({'error': 'No video loaded'}), 400
        
        cap = cv2.VideoCapture(current_video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return jsonify({'error': 'Could not read frame'}), 400
        
        # Perform detection
        confidence_threshold = float(request.args.get('confidence', 0.5))
        annotated_frame, detections = detector.detect_frame(frame, confidence_threshold)
        
        # Convert result to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Count detections by class
        detection_counts = {'Soldier': 0, 'Civilian': 0}
        for detection in detections:
            class_name = detection['class']
            # Handle case-insensitive matching
            if class_name.lower() == 'soldier':
                detection_counts['Soldier'] += 1
            elif class_name.lower() == 'civilian':
                detection_counts['Civilian'] += 1
                
        # Update global stats
        global detection_stats
        detection_stats['Soldier'] += detection_counts['Soldier']
        detection_stats['Civilian'] += detection_counts['Civilian']
        
        cap.release()
        
        return jsonify({
            'success': True,
            'frame_number': frame_number,
            'detections': detections,
            'detection_counts': detection_counts,
            'annotated_frame': f"data:image/jpeg;base64,{img_base64}",
            'total_detections': len(detections)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect/video/cleanup', methods=['POST'])
def cleanup_video():
    """Clean up temporary video files"""
    try:
        global current_video_path
        if 'current_video_path' in globals() and os.path.exists(current_video_path):
            os.remove(current_video_path)
            del current_video_path
        return jsonify({'success': True, 'message': 'Video cleaned up'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream/start', methods=['POST'])
def start_stream():
    """Start webcam stream for real-time detection"""
    global video_stream
    
    if detector is None:
        return jsonify({'error': 'Detector not initialized'}), 500
    
    if video_stream is not None:
        return jsonify({'error': 'Stream already running'}), 400
    
    try:
        camera_index = int(request.json.get('camera_index', 0))
        confidence_threshold = float(request.json.get('confidence', 0.5))
        
        video_stream = VideoStream(detector, camera_index, confidence_threshold)
        video_stream.start()
        
        return jsonify({
            'success': True,
            'message': 'Video stream started'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream/stop', methods=['POST'])
def stop_stream():
    """Stop webcam stream"""
    global video_stream
    
    if video_stream is None:
        return jsonify({'error': 'No active stream'}), 400
    
    video_stream.stop()
    video_stream = None
    
    return jsonify({
        'success': True,
        'message': 'Video stream stopped'
    })

@app.route('/api/stream/frame')
def get_stream_frame():
    """Get current frame from video stream"""
    if video_stream is None or not video_stream.is_running:
        return jsonify({'error': 'No active stream'}), 400
    
    frame_data = video_stream.get_current_frame()
    if frame_data is None:
        return jsonify({'error': 'No frame available'}), 404
    
    return jsonify({
        'success': True,
        'frame': frame_data['frame'],
        'detections': frame_data['detections'],
        'detection_counts': frame_data['detection_counts'],
        'timestamp': frame_data['timestamp']
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get current detection statistics"""
    global detection_stats
    
    if video_stream:
        detection_stats = video_stream.get_stats()
    
    return jsonify({
        'success': True,
        'statistics': detection_stats,
        'stream_active': video_stream is not None and video_stream.is_running
    })

class VideoStream:
    """Class to handle video streaming"""
    
    def __init__(self, detector, camera_index=0, confidence_threshold=0.5):
        self.detector = detector
        self.camera_index = camera_index
        self.confidence_threshold = confidence_threshold
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.stats = {'Soldier': 0, 'Civilian': 0}
        self.thread = None
        self.lock = threading.Lock()
    
    def start(self):
        """Start the video stream"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        self.is_running = True
        self.thread = threading.Thread(target=self._update_frame)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the video stream"""
        self.is_running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
    
    def _update_frame(self):
        """Update frame in background thread"""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Perform detection
            annotated_frame, detections = self.detector.detect_frame(
                frame, self.confidence_threshold
            )
            
            # Update statistics
            frame_stats = {'Soldier': 0, 'Civilian': 0}
            for detection in detections:
                class_name = detection['class']
                # Handle case-insensitive matching
                if class_name.lower() == 'soldier':
                    frame_stats['Soldier'] += 1
                    self.stats['Soldier'] += 1
                elif class_name.lower() == 'civilian':
                    frame_stats['Civilian'] += 1
                    self.stats['Civilian'] += 1
            
            # Convert frame to base64
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Update current frame data
            with self.lock:
                self.current_frame = {
                    'frame': f"data:image/jpeg;base64,{img_base64}",
                    'detections': detections,
                    'detection_counts': frame_stats,
                    'timestamp': time.time()
                }
            
            time.sleep(0.1)  # Limit to ~10 FPS for API
    
    def get_current_frame(self):
        """Get the current frame data"""
        with self.lock:
            return self.current_frame
    
    def get_stats(self):
        """Get current statistics"""
        return self.stats.copy()

if __name__ == '__main__':
    print("🚁 Aerial Threat Detection API Server")
    print("=" * 50)
    
    # Initialize detector
    if initialize_detector():
        print("🌐 Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        print("❌ Failed to initialize detector. Please check your model file.")