import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time
from typing import List, Dict, Tuple, Optional


class AerialThreatDetector:
    """
    Aerial Threat Detection System using YOLO for Soldier and Civilian Classification
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the threat detector with a trained YOLO model
        
        Args:
            model_path: Path to the trained YOLO model (.pt file)
            confidence_threshold: Minimum confidence score for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = []
        self.colors = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Using device: {self.device}")
        
        # Load the model
        self.load_model()
        self._setup_colors()
    
    def load_model(self) -> bool:
        """Load the YOLO model from the specified path"""
        try:
            print(f"Loading model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Move model to appropriate device
            self.model.to(self.device)
            
            # Optimize model for inference
            if self.device == 'cuda':
                # Enable half precision on GPU for faster inference
                self.model.half()
                print("Enabled half-precision inference on GPU")
            
            # Get class names from the model
            if hasattr(self.model.model, 'names'):
                self.class_names = list(self.model.model.names.values())
            else:
                # Default class names if not available in model
                self.class_names = ['person', 'soldier', 'civilian']
            
            print(f"Model loaded successfully on {self.device}. Classes: {self.class_names}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def _setup_colors(self):
        """Setup colors for different classes"""
        colors = [
            (0, 255, 0),    # Green for civilians
            (0, 0, 255),    # Red for soldiers
            (255, 0, 0),    # Blue for general person
            (255, 255, 0),  # Cyan for additional classes
            (255, 0, 255),  # Magenta for additional classes
        ]
        
        for i, class_name in enumerate(self.class_names):
            if 'civilian' in class_name.lower():
                self.colors[class_name] = (0, 255, 0)  # Green
            elif 'soldier' in class_name.lower() or 'military' in class_name.lower():
                self.colors[class_name] = (0, 0, 255)  # Red
            else:
                self.colors[class_name] = colors[i % len(colors)]
    
    def detect_image(self, image_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Perform detection on a single image
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (annotated_image, detections_list)
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")
            
            return self.detect_frame(image)
            
        except Exception as e:
            print(f"Error detecting image: {str(e)}")
            return image, []
    
    def detect_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Perform detection on a single frame
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Tuple of (annotated_frame, detections_list)
        """
        if self.model is None:
            print("Model not loaded")
            return frame, []
        
        try:
            # Run inference with optimized parameters
            results = self.model(frame, 
                               conf=self.confidence_threshold,
                               iou=0.4,  # Lower IoU threshold for better performance
                               max_det=100,  # Limit max detections for better performance
                               verbose=False)  # Disable verbose output
            
            detections = []
            annotated_frame = frame.copy()
            
            # Process results
            for result in results:
                boxes = result.boxes
                
                if boxes is not None:
                    for i in range(len(boxes)):
                        # Get box coordinates
                        box = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Get confidence and class
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        if class_id < len(self.class_names):
                            class_name = self.class_names[class_id]
                        else:
                            class_name = f"Class_{class_id}"
                        
                        # Store detection info
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_name': class_name,
                            'class_id': class_id
                        }
                        detections.append(detection)
                        
                        # Draw bounding box and label
                        color = self.colors.get(class_name, (255, 255, 255))
                        
                        # Draw rectangle
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Prepare label
                        label = f"{class_name}: {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        
                        # Draw label background
                        cv2.rectangle(annotated_frame, 
                                    (x1, y1 - label_size[1] - 10),
                                    (x1 + label_size[0], y1), 
                                    color, -1)
                        
                        # Draw label text
                        cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return annotated_frame, detections
            
        except Exception as e:
            print(f"Error in frame detection: {str(e)}")
            return frame, []
    
    def detect_video(self, video_path: str, output_path: Optional[str] = None) -> bool:
        """
        Perform detection on a video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            
        Returns:
            Success status
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Error opening video file: {video_path}")
                return False
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
            
            # Setup video writer if output path is provided
            out = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Perform detection
                annotated_frame, detections = self.detect_frame(frame)
                
                # Display frame info
                info_text = f"Frame: {frame_count}/{total_frames} | Detections: {len(detections)}"
                cv2.putText(annotated_frame, info_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Save frame if output writer is available
                if out:
                    out.write(annotated_frame)
                
                # Display frame (optional, comment out for headless processing)
                cv2.imshow('Aerial Threat Detection', annotated_frame)
                
                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
                
                # Print progress every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_current = frame_count / elapsed if elapsed > 0 else 0
                    print(f"Processed {frame_count}/{total_frames} frames ({fps_current:.1f} fps)")
            
            # Cleanup
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            print(f"Video processing completed. Total frames: {frame_count}")
            return True
            
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return False
    
    def detect_webcam(self, camera_index: int = 0):
        """
        Perform real-time detection on webcam feed
        
        Args:
            camera_index: Camera index (usually 0 for default camera)
        """
        try:
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                print(f"Error opening camera {camera_index}")
                return False
            
            print("Starting real-time detection. Press 'q' to quit.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Perform detection
                annotated_frame, detections = self.detect_frame(frame)
                
                # Add FPS counter
                fps_text = f"Detections: {len(detections)}"
                cv2.putText(annotated_frame, fps_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Display frame
                cv2.imshow('Aerial Threat Detection - Live', annotated_frame)
                
                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            return True
            
        except Exception as e:
            print(f"Error in webcam detection: {str(e)}")
            return False
    
    def get_detection_stats(self, detections: List[Dict]) -> Dict:
        """
        Calculate detection statistics
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_detections': len(detections),
            'class_counts': {},
            'avg_confidence': 0.0,
            'high_confidence_count': 0
        }
        
        if not detections:
            return stats
        
        confidences = []
        for det in detections:
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Count classes
            stats['class_counts'][class_name] = stats['class_counts'].get(class_name, 0) + 1
            
            # Collect confidences
            confidences.append(confidence)
            
            # Count high confidence detections (>0.8)
            if confidence > 0.8:
                stats['high_confidence_count'] += 1
        
        # Calculate average confidence
        stats['avg_confidence'] = sum(confidences) / len(confidences)
        
        return stats


def main():
    """Main function for testing the detector"""
    # Path to your trained model
    model_path = "../best.pt"
    
    # Initialize detector
    detector = AerialThreatDetector(model_path, confidence_threshold=0.5)
    
    if detector.model is None:
        print("Failed to load model. Exiting.")
        return
    
    print("Aerial Threat Detection System Initialized")
    print(f"Available classes: {detector.class_names}")
    
    # Example usage
    while True:
        print("\nSelect an option:")
        print("1. Detect on image")
        print("2. Detect on video")
        print("3. Real-time webcam detection")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip()
            try:
                annotated_image, detections = detector.detect_image(image_path)
                stats = detector.get_detection_stats(detections)
                print(f"Detection results: {stats}")
                
                # Display image
                cv2.imshow('Detection Result', annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '2':
            video_path = input("Enter video path: ").strip()
            output_path = input("Enter output path (or press Enter to skip): ").strip()
            if not output_path:
                output_path = None
            
            detector.detect_video(video_path, output_path)
        
        elif choice == '3':
            detector.detect_webcam()
        
        elif choice == '4':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()