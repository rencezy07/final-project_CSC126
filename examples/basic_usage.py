"""
Basic Usage Examples for Aerial Threat Detection System
Demonstrates core functionality and common use cases
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aerial_threat_detector import AerialThreatDetector
import cv2


def example_1_image_detection():
    """Example 1: Detect soldiers and civilians in a single image"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Image Detection")
    print("="*80)
    
    # Initialize detector
    model_path = "../best.pt"  # Path to trained model
    detector = AerialThreatDetector(model_path, confidence_threshold=0.5)
    
    if detector.model is None:
        print("ERROR: Could not load model. Make sure best.pt exists.")
        return
    
    # Process image
    image_path = "test_image.jpg"  # Replace with your image path
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path")
        return
    
    print(f"Processing image: {image_path}")
    annotated_image, detections = detector.detect_image(image_path)
    
    # Display results
    print(f"\nFound {len(detections)} detections:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class_name']}: {det['confidence']:.2f} at {det['bbox']}")
    
    # Get statistics
    stats = detector.get_detection_stats(detections)
    print(f"\nStatistics:")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Average confidence: {stats['avg_confidence']:.2f}")
    print(f"  High confidence (>0.8): {stats['high_confidence_count']}")
    print(f"  Class counts: {stats['class_counts']}")
    
    # Save result
    output_path = "result_" + os.path.basename(image_path)
    cv2.imwrite(output_path, annotated_image)
    print(f"\nResult saved to: {output_path}")
    
    # Display result
    cv2.imshow("Detection Result", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def example_2_video_detection():
    """Example 2: Detect soldiers and civilians in a video"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Video Detection")
    print("="*80)
    
    # Initialize detector
    model_path = "../best.pt"
    detector = AerialThreatDetector(model_path, confidence_threshold=0.5)
    
    if detector.model is None:
        print("ERROR: Could not load model. Make sure best.pt exists.")
        return
    
    # Process video
    video_path = "test_video.mp4"  # Replace with your video path
    output_path = "result_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        print("Please provide a valid video path")
        return
    
    print(f"Processing video: {video_path}")
    print("This may take a while depending on video length...")
    print("Press 'q' to stop processing")
    
    success = detector.detect_video(video_path, output_path)
    
    if success:
        print(f"\nVideo processing completed!")
        print(f"Output saved to: {output_path}")
    else:
        print("\nVideo processing failed or was interrupted")


def example_3_webcam_detection():
    """Example 3: Real-time detection using webcam"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Real-time Webcam Detection")
    print("="*80)
    
    # Initialize detector
    model_path = "../best.pt"
    detector = AerialThreatDetector(model_path, confidence_threshold=0.5)
    
    if detector.model is None:
        print("ERROR: Could not load model. Make sure best.pt exists.")
        return
    
    print("Starting webcam detection...")
    print("Press 'q' to quit")
    
    detector.detect_webcam(camera_index=0)


def example_4_custom_confidence():
    """Example 4: Using custom confidence threshold"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Confidence Threshold")
    print("="*80)
    
    image_path = "test_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Compare different confidence thresholds
    thresholds = [0.3, 0.5, 0.7, 0.9]
    
    for threshold in thresholds:
        print(f"\n--- Confidence Threshold: {threshold} ---")
        
        detector = AerialThreatDetector("../best.pt", confidence_threshold=threshold)
        
        if detector.model is None:
            print("ERROR: Could not load model")
            return
        
        annotated_image, detections = detector.detect_image(image_path)
        
        print(f"Detections found: {len(detections)}")
        for det in detections:
            print(f"  {det['class_name']}: {det['confidence']:.2f}")


def example_5_batch_processing():
    """Example 5: Batch process multiple images"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Batch Image Processing")
    print("="*80)
    
    # Initialize detector once for efficiency
    detector = AerialThreatDetector("../best.pt", confidence_threshold=0.5)
    
    if detector.model is None:
        print("ERROR: Could not load model")
        return
    
    # List of images to process
    image_dir = "test_images"  # Directory containing images
    
    if not os.path.exists(image_dir):
        print(f"Directory not found: {image_dir}")
        print("Create a 'test_images' directory with some images")
        return
    
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    # Create output directory
    output_dir = "batch_results"
    os.makedirs(output_dir, exist_ok=True)
    
    total_detections = 0
    all_stats = []
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        print(f"\n[{i+1}/{len(image_files)}] Processing: {image_file}")
        
        # Process image
        annotated_image, detections = detector.detect_image(image_path)
        
        # Save result
        output_path = os.path.join(output_dir, f"result_{image_file}")
        cv2.imwrite(output_path, annotated_image)
        
        # Collect statistics
        stats = detector.get_detection_stats(detections)
        all_stats.append(stats)
        total_detections += len(detections)
        
        print(f"  Detections: {len(detections)}")
        print(f"  Classes: {stats['class_counts']}")
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    print(f"Images processed: {len(image_files)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections / len(image_files):.1f}")
    
    # Aggregate class counts
    total_class_counts = {}
    for stats in all_stats:
        for class_name, count in stats['class_counts'].items():
            total_class_counts[class_name] = total_class_counts.get(class_name, 0) + count
    
    print(f"Total class counts: {total_class_counts}")
    print(f"Results saved to: {output_dir}/")


def example_6_frame_processing():
    """Example 6: Process individual frames from OpenCV capture"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Frame-by-Frame Processing")
    print("="*80)
    
    detector = AerialThreatDetector("../best.pt", confidence_threshold=0.5)
    
    if detector.model is None:
        print("ERROR: Could not load model")
        return
    
    # Open video or webcam
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video path
    
    if not cap.isOpened():
        print("ERROR: Could not open video source")
        return
    
    print("Processing frames... Press 'q' to quit")
    
    frame_count = 0
    detection_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        annotated_frame, detections = detector.detect_frame(frame)
        
        frame_count += 1
        detection_count += len(detections)
        
        # Add info to frame
        info_text = f"Frame: {frame_count} | Detections: {len(detections)} | Total: {detection_count}"
        cv2.putText(annotated_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display
        cv2.imshow('Frame Processing', annotated_frame)
        
        # Break on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nProcessed {frame_count} frames with {detection_count} total detections")


def main():
    """Main function to run examples"""
    print("="*80)
    print("AERIAL THREAT DETECTION - USAGE EXAMPLES")
    print("="*80)
    print("\nAvailable examples:")
    print("1. Single Image Detection")
    print("2. Video Detection")
    print("3. Real-time Webcam Detection")
    print("4. Custom Confidence Threshold")
    print("5. Batch Image Processing")
    print("6. Frame-by-Frame Processing")
    print("0. Exit")
    
    while True:
        choice = input("\nSelect example (0-6): ").strip()
        
        if choice == '1':
            example_1_image_detection()
        elif choice == '2':
            example_2_video_detection()
        elif choice == '3':
            example_3_webcam_detection()
        elif choice == '4':
            example_4_custom_confidence()
        elif choice == '5':
            example_5_batch_processing()
        elif choice == '6':
            example_6_frame_processing()
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please select 0-6.")


if __name__ == '__main__':
    main()
