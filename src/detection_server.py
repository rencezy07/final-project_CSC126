"""
Detection Server - Flask/SocketIO server for real-time communication
between the Electron app and the YOLO detection system
"""

import argparse
import base64
import cv2
import json
import time
from flask import Flask, request
from flask_socketio import SocketIO, emit
import numpy as np
from threading import Thread
import sys
import os

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from aerial_threat_detector import AerialThreatDetector


class DetectionServer:
    """Real-time detection server using Flask-SocketIO"""
    
    def __init__(self, model_path, confidence=0.5):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'aerial_threat_detection_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Initialize detector
        self.detector = AerialThreatDetector(model_path, confidence)
        
        # Server state
        self.is_running = False
        self.current_source = None
        self.detection_thread = None
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = None
        
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask-SocketIO event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            print(f"Client connected: {request.sid}")
            emit('status', {'message': 'Connected to detection server', 'status': 'connected'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"Client disconnected: {request.sid}")
            self.stop_detection()
        
        @self.socketio.on('start_detection')
        def handle_start_detection(data):
            """Start detection process"""
            source_type = data.get('source_type', 'webcam')
            source_path = data.get('source_path', None)
            
            print(f"Received start_detection request: type={source_type}, path={source_path}")
            
            if self.is_running:
                emit('error', {'message': 'Detection already running'})
                return
            
            try:
                # Update confidence if provided
                if 'confidence' in data:
                    self.detector.confidence_threshold = float(data['confidence'])
                    print(f"Updated confidence threshold to {self.detector.confidence_threshold}")
                
                self.start_detection(source_type, source_path)
                emit('status', {'message': 'Detection started successfully', 'status': 'running'})
            except Exception as e:
                error_msg = f'Failed to start detection: {str(e)}'
                print(f"Error: {error_msg}")
                emit('error', {'message': error_msg})
        
        @self.socketio.on('stop_detection')
        def handle_stop_detection():
            """Stop detection process"""
            self.stop_detection()
            emit('status', {'message': 'Detection stopped', 'status': 'stopped'})
        
        @self.socketio.on('update_settings')
        def handle_update_settings(data):
            """Update detection settings"""
            try:
                if 'confidence' in data:
                    self.detector.confidence_threshold = float(data['confidence'])
                emit('status', {'message': 'Settings updated', 'status': 'updated'})
            except Exception as e:
                emit('error', {'message': f'Failed to update settings: {str(e)}'})
    
    def start_detection(self, source_type, source_path=None):
        """Start detection in a separate thread"""
        if self.is_running:
            return False
        
        self.is_running = True
        self.current_source = (source_type, source_path)
        self.start_time = time.time()
        self.frame_count = 0
        self.detection_count = 0
        
        # Start detection thread
        self.detection_thread = Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        return True
    
    def stop_detection(self):
        """Stop detection process"""
        self.is_running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=2)
            self.detection_thread = None
    
    def _detection_loop(self):
        """Main detection loop"""
        source_type, source_path = self.current_source
        
        try:
            if source_type == 'webcam':
                self._process_webcam()
            elif source_type == 'image':
                self._process_image(source_path)
            elif source_type == 'video':
                self._process_video(source_path)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
                
        except Exception as e:
            print(f"Detection error: {str(e)}")
            self.socketio.emit('error', {'message': f'Detection error: {str(e)}'})
        finally:
            self.is_running = False
            self.socketio.emit('detection_complete')
    
    def _process_webcam(self):
        """Process webcam feed"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
        print("Starting webcam detection... (will run for 30 seconds or until stopped)")
        start_time = time.time()
        max_duration = 30  # 30 seconds max for demo
        
        try:
            while self.is_running:
                # Check time limit
                if time.time() - start_time > max_duration:
                    print(f"Webcam detection completed (reached {max_duration}s time limit)")
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    break
                
                self._process_frame(frame)
                
                # Emit time remaining
                elapsed = time.time() - start_time
                remaining = max_duration - elapsed
                if self.frame_count % 30 == 0:  # Every second
                    self.socketio.emit('detection_progress', {
                        'time_remaining': remaining,
                        'elapsed_time': elapsed
                    })
                
                # Control frame rate
                time.sleep(0.033)  # ~30 FPS
                
        finally:
            cap.release()
            print("Webcam detection completed")
            self.socketio.emit('detection_complete')
    
    def _process_image(self, image_path):
        """Process single image"""
        # Handle different path formats and normalize path
        normalized_path = os.path.normpath(image_path)
        
        if not os.path.exists(normalized_path):
            # Try alternative path formats
            alt_paths = [
                image_path,
                os.path.abspath(image_path),
                os.path.join(os.getcwd(), image_path)
            ]
            
            found_path = None
            for path in alt_paths:
                if os.path.exists(path):
                    found_path = path
                    break
            
            if not found_path:
                error_msg = f"Image not found: {image_path}. Tried paths: {alt_paths}"
                print(error_msg)
                raise FileNotFoundError(error_msg)
            
            normalized_path = found_path
        
        print(f"Processing image file: {normalized_path}")
        frame = cv2.imread(normalized_path)
        if frame is None:
            raise ValueError(f"Could not read image: {normalized_path}")
        
        print(f"Processing image: {image_path}")
        self._process_frame(frame)
        
        # Image processing is complete after one frame
        print("Image detection completed")
        self.socketio.emit('detection_complete')
    
    def _process_video(self, video_path):
        """Process video file with optimized performance"""
        # Handle different path formats and normalize path
        normalized_path = os.path.normpath(video_path)
        
        if not os.path.exists(normalized_path):
            # Try alternative path formats
            alt_paths = [
                video_path,
                os.path.abspath(video_path),
                os.path.join(os.getcwd(), video_path)
            ]
            
            found_path = None
            for path in alt_paths:
                if os.path.exists(path):
                    found_path = path
                    break
            
            if not found_path:
                error_msg = f"Video not found: {video_path}. Tried paths: {alt_paths}"
                print(error_msg)
                raise FileNotFoundError(error_msg)
            
            normalized_path = found_path
        
        print(f"Processing video file: {normalized_path}")
        cap = cv2.VideoCapture(normalized_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Validate video properties
        if fps <= 0:
            fps = 25.0  # Default fallback FPS
            print(f"Warning: Invalid FPS detected, using default: {fps}")
        
        if total_frames <= 0:
            print("Warning: Could not determine frame count, processing until end of video")
            total_frames = float('inf')  # Process until end
        
        # Performance optimization: Skip frames for faster processing
        skip_frames = max(1, int(fps / 10))  # Process ~10 frames per second max
        if total_frames > 1000:  # For very long videos, skip even more
            skip_frames = max(skip_frames, int(fps / 5))  # Process ~5 frames per second
        
        print(f"Processing video: {video_path} ({fps} FPS, {total_frames} frames, skipping every {skip_frames} frames)")
        
        processed_frames = 0
        frame_number = 0
        frames_read = 0
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video reached at frame {frames_read}")
                    break
                
                frames_read += 1
                
                # Only process every Nth frame for performance
                if frame_number % skip_frames == 0:
                    self._process_frame(frame, send_frame_data=(processed_frames % 3 == 0))  # Send frame data every 3rd processed frame
                    processed_frames += 1
                    
                    # Emit progress update
                    if total_frames != float('inf'):
                        progress = (frames_read / total_frames) * 100
                        expected_processed = total_frames // skip_frames
                    else:
                        progress = 0  # Unknown progress
                        expected_processed = frames_read // skip_frames
                    
                    self.socketio.emit('detection_progress', {
                        'progress': progress,
                        'processed_frames': processed_frames,
                        'total_frames': max(1, expected_processed),
                        'actual_frame': frames_read
                    })
                
                frame_number += 1
                
                # Small delay to prevent overwhelming the system, but much faster than before
                if processed_frames % 5 == 0:
                    time.sleep(0.01)  # Minimal delay every 5 processed frames
                
                # Safety check to prevent infinite loops (only when we know total frames)
                if total_frames != float('inf') and frames_read >= total_frames:
                    print(f"Reached expected total frames: {frames_read}/{total_frames}")
                    break
                
        finally:
            cap.release()
            if total_frames != float('inf'):
                print(f"Video processing completed: {processed_frames} processed frames from {frames_read} frames read (expected {total_frames})")
            else:
                print(f"Video processing completed: {processed_frames} processed frames from {frames_read} frames read")
            self.socketio.emit('detection_complete')
    
    def _process_frame(self, frame, send_frame_data=True):
        """Process a single frame and emit results with performance optimization"""
        try:
            # Perform detection
            annotated_frame, detections = self.detector.detect_frame(frame)
            
            # Update counters
            self.frame_count += 1
            self.detection_count += len(detections)
            
            # Calculate statistics
            stats = self._calculate_stats(detections)
            
            # Always emit detection results (lightweight)
            self.socketio.emit('detection_result', {
                'frame_id': self.frame_count,
                'detections': detections,
                'stats': stats,
                'timestamp': time.time()
            })
            
            # Only encode and send frame data when requested (reduce overhead)
            if send_frame_data:
                # Resize frame for faster encoding and transmission
                height, width = annotated_frame.shape[:2]
                if width > 800:  # Resize large frames
                    scale = 800 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    resized_frame = cv2.resize(annotated_frame, (new_width, new_height))
                else:
                    resized_frame = annotated_frame
                
                # Encode with lower quality for faster processing
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
                _, buffer = cv2.imencode('.jpg', resized_frame, encode_params)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                self.socketio.emit('frame_update', {
                    'frame': frame_b64,
                    'detections': detections,
                    'image_width': resized_frame.shape[1],
                    'image_height': resized_frame.shape[0]
                })
            
            # Print progress
            if self.frame_count % 10 == 0:  # Every 10 frames (more frequent updates)
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                print(f"Processed {self.frame_count} frames, {self.detection_count} total detections, {fps:.1f} FPS")
            
        except Exception as e:
            print(f"Frame processing error: {str(e)}")
    
    def _calculate_stats(self, detections):
        """Calculate detection statistics"""
        if not detections:
            return {
                'total_detections': 0,
                'class_counts': {},
                'avg_confidence': 0.0,
                'high_confidence_count': 0
            }
        
        class_counts = {}
        confidences = []
        high_conf_count = 0
        
        for det in detections:
            class_name = det['class_name']
            confidence = det['confidence']
            
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidences.append(confidence)
            
            if confidence > 0.8:
                high_conf_count += 1
        
        return {
            'total_detections': len(detections),
            'class_counts': class_counts,
            'avg_confidence': sum(confidences) / len(confidences),
            'high_confidence_count': high_conf_count
        }
    
    def run(self, host='localhost', port=5000, debug=False):
        """Run the detection server"""
        print(f"Starting Aerial Threat Detection Server on {host}:{port}")
        print(f"Model loaded: {self.detector.model_path}")
        print(f"Available classes: {self.detector.class_names}")
        
        self.socketio.run(self.app, host=host, port=port, debug=debug)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Aerial Threat Detection Server')
    parser.add_argument('--model-path', default='best.pt', 
                       help='Path to YOLO model file')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--host', default='localhost',
                       help='Server host address')
    parser.add_argument('--port', type=int, default=5000,
                       help='Server port')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    
    # Command line arguments for direct detection (without server)
    parser.add_argument('--source-type', choices=['image', 'video', 'webcam'],
                       help='Source type for direct detection')
    parser.add_argument('--source-path', help='Path to source file')
    
    args = parser.parse_args()
    
    # Check if model file exists - try multiple locations
    model_paths_to_try = [
        args.model_path,
        os.path.join('..', args.model_path),
        os.path.join(os.path.dirname(__file__), '..', args.model_path),
        os.path.join(os.getcwd(), args.model_path)
    ]
    
    model_path = None
    for path in model_paths_to_try:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print(f"Error: Model file not found. Tried locations:")
        for path in model_paths_to_try:
            print(f"  - {os.path.abspath(path)}")
        sys.exit(1)
    
    args.model_path = model_path
    
    # If direct detection arguments are provided, run detection directly
    if args.source_type is not None:
        print("Running direct detection mode...")
        detector = AerialThreatDetector(args.model_path, args.confidence)
        
        if detector.model is None:
            print("Failed to load model. Exiting.")
            sys.exit(1)
        
        if args.source_type == 'image' and args.source_path:
            annotated_image, detections = detector.detect_image(args.source_path)
            print(f"Found {len(detections)} detections")
            
            # Save result
            output_path = args.source_path.replace('.', '_detected.')
            cv2.imwrite(output_path, annotated_image)
            print(f"Result saved to: {output_path}")
            
        elif args.source_type == 'video' and args.source_path:
            output_path = args.source_path.replace('.', '_detected.')
            detector.detect_video(args.source_path, output_path)
            
        elif args.source_type == 'webcam':
            detector.detect_webcam()
            
        else:
            print("Error: source-path required for image and video sources")
            sys.exit(1)
    
    else:
        # Run server mode
        print("Running server mode...")
        server = DetectionServer(args.model_path, args.confidence)
        
        if server.detector.model is None:
            print("Failed to load model. Exiting.")
            sys.exit(1)
        
        try:
            print(f"Starting server on {args.host}:{args.port}...")
            server.run(host=args.host, port=args.port, debug=args.debug)
        except KeyboardInterrupt:
            print("\nShutting down server...")
        except Exception as e:
            print(f"Server error: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()