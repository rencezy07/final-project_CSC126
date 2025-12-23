import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path

class DroneDetector:
    def __init__(self, model_path):
        """
        Initialize the drone detector with YOLOv11 model
        
        Args:
            model_path (str): Path to the trained YOLO model (.pt file)
        """
        self.model_path = model_path
        self.model = None
        self.class_names = ['Soldier', 'Civilian']  # Adjust based on your training
        self.colors = {
            'Soldier': (0, 0, 255),    # Red for soldiers
            'Civilian': (0, 255, 0)    # Green for civilians
        }
        self.load_model()
    
    def load_model(self):
        """Load the YOLO model from the specified path"""
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
            
            # Print model info
            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
                print(f"📋 Detected classes: {self.class_names}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    def detect_frame(self, frame, confidence_threshold=0.5):
        """
        Perform detection on a single frame with optimized processing
        
        Args:
            frame: Input frame (numpy array)
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            tuple: (annotated_frame, detections_list)
        """
        if self.model is None:
            raise Exception("Model not loaded")
        
        try:
            # Run inference with optimized settings
            results = self.model.predict(
                source=frame, 
                conf=confidence_threshold,
                device='cpu',  # Use GPU if available, else CPU
                verbose=False,  # Reduce logging for speed
                half=False,  # Use FP16 for speed if supported
                imgsz=640  # Optimal input size for YOLOv11
            )
            
            # Get the first result (single image inference)
            result = results[0]
            
            # Extract detections
            detections = []
            annotated_frame = frame.copy()
            
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Get class name
                    if class_id < len(self.class_names):
                        class_name = self.class_names[class_id]
                    else:
                        class_name = f"Class_{class_id}"
                    
                    # Filter by confidence
                    if confidence >= confidence_threshold:
                        detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2]
                        })
                        
                        # Draw bounding box and label
                        color = self.colors.get(class_name, (255, 255, 255))
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add label
                        label = f"{class_name}: {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return annotated_frame, detections
            
        except Exception as e:
            print(f"❌ Detection error: {str(e)}")
            return frame, []
    
    def process_video(self, video_path, output_path=None, display_live=True):
        annotated_frame = frame.copy()
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if confidence >= confidence_threshold:
                        # Get class name
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
                        
                        # Store detection info
                        detection = {
                            'class': class_name,
                            'confidence': float(confidence),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        }
                        detections.append(detection)
                        
                        # Draw bounding box
                        color = self.colors.get(class_name, (255, 255, 255))
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Draw label with confidence
                        label = f"{class_name}: {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, (int(x1), int(y1) - label_size[1] - 10), 
                                    (int(x1) + label_size[0], int(y1)), color, -1)
                        cv2.putText(annotated_frame, label, (int(x1), int(y1) - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame, detections
    
    def process_video(self, video_path, output_path=None, display_live=True):
        """
        Process a video file for detection
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            display_live: Whether to display results in real-time
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = {'Soldier': 0, 'Civilian': 0}
        
        print(f"🎥 Processing video: {video_path}")
        print(f"📐 Resolution: {width}x{height} @ {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects in frame
            annotated_frame, detections = self.detect_frame(frame)
            
            # Count detections
            for detection in detections:
                class_name = detection['class']
                # Handle case-insensitive matching
                if class_name.lower() == 'soldier':
                    total_detections['Soldier'] += 1
                elif class_name.lower() == 'civilian':
                    total_detections['Civilian'] += 1
            
            # Add frame counter and statistics
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            stats_text = f"Soldiers: {total_detections['Soldier']} | Civilians: {total_detections['Civilian']}"
            cv2.putText(annotated_frame, stats_text, (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save frame if writer is available
            if writer:
                writer.write(annotated_frame)
            
            # Display frame if requested
            if display_live:
                cv2.imshow('Aerial Threat Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"⚡ Processed {frame_count} frames...")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Video processing complete!")
        print(f"📊 Final Statistics:")
        print(f"   🔴 Soldiers detected: {total_detections['Soldier']}")
        print(f"   🟢 Civilians detected: {total_detections['Civilian']}")
        print(f"   🎬 Total frames processed: {frame_count}")
        
        return total_detections
    
    def process_video_optimized(self, video_path, confidence_threshold=0.5, skip_frames=2):
        """
        Process video with optimized performance for faster analysis
        
        Args:
            video_path: Path to input video
            confidence_threshold: Minimum confidence for detections
            skip_frames: Process every nth frame for speed (1=every frame, 2=every other frame)
            
        Returns:
            dict: Comprehensive processing results
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        frame_count = 0
        processed_frame_count = 0
        total_detections = {'Soldier': 0, 'Civilian': 0}
        detections_timeline = []
        confidence_scores = {'Soldier': [], 'Civilian': []}
        processing_times = []
        
        print(f"🎥 Processing video: {video_path}")
        print(f"📐 Resolution: {width}x{height} @ {fps} FPS")
        print(f"⏱️ Duration: {duration:.2f}s ({total_frames} frames)")
        print(f"⚡ Processing every {skip_frames} frames for speed")
        
        import time
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for faster processing
            if frame_count % skip_frames != 0:
                frame_count += 1
                continue
            
            frame_start = time.time()
            
            # Detect objects in frame
            annotated_frame, detections = self.detect_frame(frame, confidence_threshold)
            
            frame_processing_time = time.time() - frame_start
            processing_times.append(frame_processing_time)
            
            # Count and store detections
            frame_soldiers = 0
            frame_civilians = 0
            frame_detections = []
            
            for detection in detections:
                class_name = detection['class']
                confidence = detection['confidence']
                
                # Handle case-insensitive matching
                if class_name.lower() == 'soldier':
                    total_detections['Soldier'] += 1
                    frame_soldiers += 1
                    confidence_scores['Soldier'].append(confidence)
                elif class_name.lower() == 'civilian':
                    total_detections['Civilian'] += 1
                    frame_civilians += 1
                    confidence_scores['Civilian'].append(confidence)
                
                frame_detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': detection['bbox']
                })
            
            # Store timeline data
            detections_timeline.append({
                'frame_number': frame_count,
                'timestamp': frame_count / fps,
                'soldiers': frame_soldiers,
                'civilians': frame_civilians,
                'total': frame_soldiers + frame_civilians,
                'detections': frame_detections
            })
            
            processed_frame_count += 1
            frame_count += 1
            
            # Progress indicator
            if processed_frame_count % 50 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"⚡ Progress: {progress:.1f}% ({processed_frame_count} frames processed)")
        
        # Cleanup
        cap.release()
        
        processing_time = time.time() - start_time
        avg_frame_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Calculate statistics
        processing_info = {
            'total_frames': total_frames,
            'processed_frames': processed_frame_count,
            'skip_factor': skip_frames,
            'processing_time': processing_time,
            'avg_frame_processing_time': avg_frame_time,
            'fps_achieved': processed_frame_count / processing_time if processing_time > 0 else 0,
            'original_fps': fps,
            'duration': duration,
            'resolution': f"{width}x{height}"
        }
        
        statistics = {
            'Soldier': total_detections['Soldier'],
            'Civilian': total_detections['Civilian'],
            'total': total_detections['Soldier'] + total_detections['Civilian'],
            'confidence_avg': {
                'Soldier': sum(confidence_scores['Soldier']) / len(confidence_scores['Soldier']) if confidence_scores['Soldier'] else 0,
                'Civilian': sum(confidence_scores['Civilian']) / len(confidence_scores['Civilian']) if confidence_scores['Civilian'] else 0
            },
            'detection_rate': {
                'soldiers_per_second': total_detections['Soldier'] / duration if duration > 0 else 0,
                'civilians_per_second': total_detections['Civilian'] / duration if duration > 0 else 0
            }
        }
        
        print(f"\n✅ Video processing complete!")
        print(f"📊 Final Statistics:")
        print(f"   🔴 Soldiers detected: {total_detections['Soldier']}")
        print(f"   🟢 Civilians detected: {total_detections['Civilian']}")
        print(f"   🎬 Frames processed: {processed_frame_count}/{total_frames}")
        print(f"   ⏱️ Processing time: {processing_time:.2f}s")
        print(f"   ⚡ Average FPS: {processing_info['fps_achieved']:.2f}")
        
        return {
            'statistics': statistics,
            'processing_info': processing_info,
            'detections_timeline': detections_timeline
        }
    
    def process_webcam(self, camera_index=0):
        """
        Process webcam feed for real-time detection
        
        Args:
            camera_index: Index of the camera to use (0 for default)
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("📹 Starting webcam detection. Press 'q' to quit...")
        print("🎯 Use this for simulating drone feed!")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects in frame
            annotated_frame, detections = self.detect_frame(frame)
            
            # Add overlay information
            cv2.putText(annotated_frame, f"Live Feed - Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display current detections count
            current_soldiers = sum(1 for d in detections if d['class'].lower() == 'soldier')
            current_civilians = sum(1 for d in detections if d['class'].lower() == 'civilian')
            
            status_text = f"Current: Soldiers: {current_soldiers}, Civilians: {current_civilians}"
            cv2.putText(annotated_frame, status_text, (10, annotated_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Aerial Threat Detection - Live Feed', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("📹 Webcam detection stopped")

def main():
    """Main function for testing the detector"""
    # Path to your trained model
    model_path = "../yolo11s.pt"  # Adjust path as needed
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure your yolo11s.pt file is in the correct location.")
        return
    
    # Initialize detector
    try:
        detector = DroneDetector(model_path)
        
        print("\n🚁 Aerial Threat Detection System")
        print("=" * 50)
        print("1. Process video file")
        print("2. Use webcam (simulate drone feed)")
        print("3. Process image")
        print("=" * 50)
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            video_path = input("Enter video file path: ").strip()
            if os.path.exists(video_path):
                output_path = input("Enter output path (or press Enter to skip): ").strip()
                if not output_path:
                    output_path = None
                detector.process_video(video_path, output_path)
            else:
                print("❌ Video file not found!")
        
        elif choice == '2':
            detector.process_webcam()
        
        elif choice == '3':
            image_path = input("Enter image file path: ").strip()
            if os.path.exists(image_path):
                # Load and process single image
                frame = cv2.imread(image_path)
                annotated_frame, detections = detector.detect_frame(frame)
                
                print(f"🎯 Detections found: {len(detections)}")
                for detection in detections:
                    print(f"  - {detection['class']}: {detection['confidence']:.2f}")
                
                # Display result
                cv2.imshow('Detection Result', annotated_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                # Save result
                output_path = f"detected_{os.path.basename(image_path)}"
                cv2.imwrite(output_path, annotated_frame)
                print(f"💾 Result saved as: {output_path}")
            else:
                print("❌ Image file not found!")
        
        else:
            print("❌ Invalid choice!")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()