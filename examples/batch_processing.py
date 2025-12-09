"""
Batch Processing Script
Process multiple videos or images in batch mode
"""

import sys
import os
import argparse
from pathlib import Path
import time
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aerial_threat_detector import AerialThreatDetector
import cv2


class BatchProcessor:
    """Batch processing for multiple files"""
    
    def __init__(self, model_path, confidence=0.5, save_videos=True, save_images=True):
        """
        Initialize batch processor
        
        Args:
            model_path: Path to trained YOLO model
            confidence: Confidence threshold
            save_videos: Save annotated videos
            save_images: Save annotated images
        """
        self.detector = AerialThreatDetector(model_path, confidence)
        self.save_videos = save_videos
        self.save_images = save_images
        
        if self.detector.model is None:
            raise RuntimeError("Failed to load model")
    
    def process_images(self, input_dir, output_dir):
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results
        
        Returns:
            Processing results dictionary
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f'*{ext}'))
            image_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return None
        
        print(f"\nFound {len(image_files)} images to process")
        print("="*80)
        
        results = {
            'total_processed': 0,
            'total_detections': 0,
            'files': [],
            'class_counts': {},
            'processing_time': 0
        }
        
        start_time = time.time()
        
        for i, image_file in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}] Processing: {image_file.name}")
            
            try:
                # Process image
                annotated_image, detections = self.detector.detect_image(str(image_file))
                
                # Save result if enabled
                if self.save_images:
                    output_path = output_dir / f"result_{image_file.name}"
                    cv2.imwrite(str(output_path), annotated_image)
                
                # Update statistics
                file_result = {
                    'filename': image_file.name,
                    'detections': len(detections),
                    'classes': {}
                }
                
                for det in detections:
                    class_name = det['class_name']
                    file_result['classes'][class_name] = file_result['classes'].get(class_name, 0) + 1
                    results['class_counts'][class_name] = results['class_counts'].get(class_name, 0) + 1
                
                results['files'].append(file_result)
                results['total_detections'] += len(detections)
                results['total_processed'] += 1
                
                print(f"  Detections: {len(detections)}")
                print(f"  Classes: {file_result['classes']}")
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
        
        results['processing_time'] = time.time() - start_time
        
        # Save results JSON
        results_path = output_dir / 'batch_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def process_videos(self, input_dir, output_dir):
        """
        Process all videos in a directory
        
        Args:
            input_dir: Directory containing input videos
            output_dir: Directory to save results
        
        Returns:
            Processing results dictionary
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_dir.glob(f'*{ext}'))
            video_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        if not video_files:
            print(f"No videos found in {input_dir}")
            return None
        
        print(f"\nFound {len(video_files)} videos to process")
        print("="*80)
        
        results = {
            'total_processed': 0,
            'total_detections': 0,
            'total_frames': 0,
            'files': [],
            'class_counts': {},
            'processing_time': 0
        }
        
        start_time = time.time()
        
        for i, video_file in enumerate(video_files):
            print(f"\n[{i+1}/{len(video_files)}] Processing: {video_file.name}")
            
            try:
                output_path = None
                if self.save_videos:
                    output_path = str(output_dir / f"result_{video_file.name}")
                
                # Process video with frame counting
                frame_count, detections_count, class_counts = self._process_video_with_stats(
                    str(video_file), output_path
                )
                
                # Update statistics
                file_result = {
                    'filename': video_file.name,
                    'frames': frame_count,
                    'detections': detections_count,
                    'classes': class_counts
                }
                
                results['files'].append(file_result)
                results['total_detections'] += detections_count
                results['total_frames'] += frame_count
                results['total_processed'] += 1
                
                # Update global class counts
                for class_name, count in class_counts.items():
                    results['class_counts'][class_name] = results['class_counts'].get(class_name, 0) + count
                
                print(f"  Frames: {frame_count}")
                print(f"  Total detections: {detections_count}")
                print(f"  Classes: {class_counts}")
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
        
        results['processing_time'] = time.time() - start_time
        
        # Save results JSON
        results_path = output_dir / 'batch_video_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _process_video_with_stats(self, video_path, output_path=None):
        """Process video and collect statistics"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output is requested
        out = None
        if output_path and self.save_videos:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detections_count = 0
        class_counts = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, detections = self.detector.detect_frame(frame)
            
            frame_count += 1
            detections_count += len(detections)
            
            # Count classes
            for det in detections:
                class_name = det['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Write frame if output is enabled
            if out:
                out.write(annotated_frame)
        
        cap.release()
        if out:
            out.release()
        
        return frame_count, detections_count, class_counts
    
    def _print_summary(self, results):
        """Print processing summary"""
        print("\n" + "="*80)
        print("BATCH PROCESSING SUMMARY")
        print("="*80)
        print(f"Files processed: {results['total_processed']}")
        print(f"Total detections: {results['total_detections']}")
        
        if 'total_frames' in results:
            print(f"Total frames: {results['total_frames']}")
            if results['total_frames'] > 0:
                print(f"Average detections per frame: {results['total_detections'] / results['total_frames']:.2f}")
        else:
            print(f"Average detections per file: {results['total_detections'] / results['total_processed']:.2f}")
        
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        print(f"\nClass distribution:")
        for class_name, count in results['class_counts'].items():
            percentage = (count / results['total_detections']) * 100 if results['total_detections'] > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        print("="*80)


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description='Batch process images or videos')
    
    parser.add_argument('--model', required=True, help='Path to trained model (.pt)')
    parser.add_argument('--input', required=True, help='Input directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--type', choices=['images', 'videos'], required=True,
                       help='Type of files to process')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--no-save-videos', action='store_true',
                       help='Do not save annotated videos')
    parser.add_argument('--no-save-images', action='store_true',
                       help='Do not save annotated images')
    
    args = parser.parse_args()
    
    # Verify model exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        return
    
    # Verify input directory exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input directory not found: {args.input}")
        return
    
    # Create processor
    try:
        processor = BatchProcessor(
            model_path=args.model,
            confidence=args.confidence,
            save_videos=not args.no_save_videos,
            save_images=not args.no_save_images
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize processor: {e}")
        return
    
    # Process files
    if args.type == 'images':
        processor.process_images(args.input, args.output)
    else:
        processor.process_videos(args.input, args.output)


if __name__ == '__main__':
    main()
