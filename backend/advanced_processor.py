import cv2
import numpy as np
from drone_detector import DroneDetector
import os
import argparse
from datetime import datetime
import json

class AdvancedVideoProcessor:
    """Advanced video processor with additional features for evaluation"""
    
    def __init__(self, model_path):
        self.detector = DroneDetector(model_path)
        self.results_log = []
        
    def process_video_with_analysis(self, video_path, output_dir="output", confidence_threshold=0.5):
        """
        Process video with detailed analysis and multiple outputs
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Output video with detections
        output_video_path = os.path.join(output_dir, f"{video_name}_detected_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Analysis data
        frame_analysis = []
        total_stats = {'Soldier': 0, 'Civilian': 0}
        confidence_scores = {'Soldier': [], 'Civilian': []}
        
        print(f"🎥 Processing: {video_path}")
        print(f"📊 Total frames: {total_frames}")
        print(f"📐 Resolution: {width}x{height} @ {fps} FPS")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            annotated_frame, detections = self.detector.detect_frame(frame, confidence_threshold)
            
            # Collect frame statistics
            frame_data = {
                'frame_number': frame_count,
                'timestamp': frame_count / fps,
                'detections': len(detections),
                'soldier_count': 0,
                'civilian_count': 0,
                'detection_details': []
            }
            
            # Process detections
            for detection in detections:
                class_name = detection['class']
                confidence = detection['confidence']
                
                # Handle case-insensitive matching for stats
                normalized_class = class_name.lower()
                if normalized_class == 'soldier':
                    total_stats['Soldier'] += 1
                    confidence_scores['Soldier'].append(confidence)
                    frame_data['soldier_count'] += 1
                elif normalized_class == 'civilian':
                    total_stats['Civilian'] += 1
                    confidence_scores['Civilian'].append(confidence)
                    frame_data['civilian_count'] += 1
                
                frame_data['detection_details'].append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': detection['bbox']
                })
            
            frame_analysis.append(frame_data)
            
            # Add enhanced annotations
            self.add_enhanced_annotations(annotated_frame, frame_count, total_stats, 
                                        len(detections), fps, total_frames)
            
            # Write frame
            writer.write(annotated_frame)
            
            # Progress update
            if frame_count % (total_frames // 20) == 0 or frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"⚡ Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            frame_count += 1
        
        # Cleanup video processing
        cap.release()
        writer.release()
        
        # Generate analysis report
        analysis_results = self.generate_analysis_report(
            frame_analysis, total_stats, confidence_scores, video_path, total_frames, fps
        )
        
        # Save analysis to JSON
        analysis_file = os.path.join(output_dir, f"{video_name}_analysis_{timestamp}.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Generate visualization plots
        self.generate_plots(frame_analysis, output_dir, video_name, timestamp)
        
        print(f"\n✅ Processing complete!")
        print(f"📹 Output video: {output_video_path}")
        print(f"📊 Analysis report: {analysis_file}")
        print(f"📈 Visualization plots saved in: {output_dir}")
        
        return analysis_results
    
    def add_enhanced_annotations(self, frame, frame_count, total_stats, current_detections, fps, total_frames):
        """Add enhanced annotations to frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay for info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Frame info
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Time info
        time_sec = frame_count / fps
        time_min = int(time_sec // 60)
        time_sec_remainder = time_sec % 60
        cv2.putText(frame, f"Time: {time_min:02d}:{time_sec_remainder:05.2f}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Detection stats
        cv2.putText(frame, f"Total Soldiers: {total_stats['Soldier']}", (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Total Civilians: {total_stats['Civilian']}", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Current frame detections
        cv2.putText(frame, f"Current Detections: {current_detections}", (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def generate_analysis_report(self, frame_analysis, total_stats, confidence_scores, video_path, total_frames, fps):
        """Generate comprehensive analysis report"""
        
        # Calculate statistics
        soldier_confidences = confidence_scores['Soldier']
        civilian_confidences = confidence_scores['Civilian']
        
        report = {
            'video_info': {
                'path': video_path,
                'total_frames': total_frames,
                'fps': fps,
                'duration_seconds': total_frames / fps
            },
            'detection_summary': {
                'total_soldiers': total_stats['Soldier'],
                'total_civilians': total_stats['Civilian'],
                'total_detections': sum(total_stats.values()),
                'frames_with_detections': len([f for f in frame_analysis if f['detections'] > 0]),
                'detection_rate': len([f for f in frame_analysis if f['detections'] > 0]) / total_frames
            },
            'confidence_analysis': {
                'soldier_stats': {
                    'count': len(soldier_confidences),
                    'avg_confidence': np.mean(soldier_confidences) if soldier_confidences else 0,
                    'min_confidence': np.min(soldier_confidences) if soldier_confidences else 0,
                    'max_confidence': np.max(soldier_confidences) if soldier_confidences else 0,
                    'std_confidence': np.std(soldier_confidences) if soldier_confidences else 0
                },
                'civilian_stats': {
                    'count': len(civilian_confidences),
                    'avg_confidence': np.mean(civilian_confidences) if civilian_confidences else 0,
                    'min_confidence': np.min(civilian_confidences) if civilian_confidences else 0,
                    'max_confidence': np.max(civilian_confidences) if civilian_confidences else 0,
                    'std_confidence': np.std(civilian_confidences) if civilian_confidences else 0
                }
            },
            'temporal_analysis': {
                'peak_activity_frame': max(frame_analysis, key=lambda x: x['detections'])['frame_number'] if frame_analysis else 0,
                'avg_detections_per_frame': np.mean([f['detections'] for f in frame_analysis]) if frame_analysis else 0
            },
            'frame_details': frame_analysis[:100]  # First 100 frames for detailed review
        }
        
        return report
    
    def generate_plots(self, frame_analysis, output_dir, video_name, timestamp):
        """Generate visualization plots (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            
            # Detection timeline
            frames = [f['frame_number'] for f in frame_analysis]
            soldier_counts = [f['soldier_count'] for f in frame_analysis]
            civilian_counts = [f['civilian_count'] for f in frame_analysis]
            
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Detection timeline
            plt.subplot(2, 2, 1)
            plt.plot(frames, soldier_counts, 'r-', label='Soldiers', alpha=0.7)
            plt.plot(frames, civilian_counts, 'g-', label='Civilians', alpha=0.7)
            plt.xlabel('Frame Number')
            plt.ylabel('Detections per Frame')
            plt.title('Detection Timeline')
            plt.legend()
            plt.grid(True)
            
            # Plot 2: Cumulative detections
            plt.subplot(2, 2, 2)
            cumsum_soldiers = np.cumsum(soldier_counts)
            cumsum_civilians = np.cumsum(civilian_counts)
            plt.plot(frames, cumsum_soldiers, 'r-', label='Soldiers')
            plt.plot(frames, cumsum_civilians, 'g-', label='Civilians')
            plt.xlabel('Frame Number')
            plt.ylabel('Cumulative Detections')
            plt.title('Cumulative Detections Over Time')
            plt.legend()
            plt.grid(True)
            
            # Plot 3: Detection heatmap (grouped by time intervals)
            plt.subplot(2, 2, 3)
            interval_size = max(1, len(frames) // 20)  # 20 time intervals
            intervals = []
            soldier_intervals = []
            civilian_intervals = []
            
            for i in range(0, len(frames), interval_size):
                intervals.append(i)
                soldier_intervals.append(sum(soldier_counts[i:i+interval_size]))
                civilian_intervals.append(sum(civilian_counts[i:i+interval_size]))
            
            x = np.arange(len(intervals))
            width = 0.35
            plt.bar(x - width/2, soldier_intervals, width, label='Soldiers', color='red', alpha=0.7)
            plt.bar(x + width/2, civilian_intervals, width, label='Civilians', color='green', alpha=0.7)
            plt.xlabel('Time Interval')
            plt.ylabel('Total Detections')
            plt.title('Detection Distribution by Time Intervals')
            plt.legend()
            
            # Plot 4: Activity density
            plt.subplot(2, 2, 4)
            total_detections = [s + c for s, c in zip(soldier_counts, civilian_counts)]
            plt.hist(total_detections, bins=20, alpha=0.7, color='blue')
            plt.xlabel('Detections per Frame')
            plt.ylabel('Frequency')
            plt.title('Detection Density Distribution')
            plt.grid(True)
            
            plt.tight_layout()
            
            plot_file = os.path.join(output_dir, f"{video_name}_analysis_plots_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Analysis plots saved: {plot_file}")
            
        except ImportError:
            print("⚠️ Matplotlib not installed. Skipping plot generation.")
            print("Install with: pip install matplotlib")

def main():
    parser = argparse.ArgumentParser(description='Advanced Video Processing for Aerial Threat Detection')
    parser.add_argument('--video', '-v', required=True, help='Path to input video file')
    parser.add_argument('--model', '-m', default='../yolo11s.pt', help='Path to YOLO model file')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Confidence threshold')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return
    
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    processor = AdvancedVideoProcessor(args.model)
    results = processor.process_video_with_analysis(
        args.video, args.output, args.confidence
    )
    
    # Print summary
    print("\n📊 ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"🎥 Video: {results['video_info']['path']}")
    print(f"⏱️  Duration: {results['video_info']['duration_seconds']:.1f} seconds")
    print(f"🔴 Total Soldiers: {results['detection_summary']['total_soldiers']}")
    print(f"🟢 Total Civilians: {results['detection_summary']['total_civilians']}")
    print(f"📈 Detection Rate: {results['detection_summary']['detection_rate']:.1%}")
    print(f"🎯 Avg Soldier Confidence: {results['confidence_analysis']['soldier_stats']['avg_confidence']:.2f}")
    print(f"🎯 Avg Civilian Confidence: {results['confidence_analysis']['civilian_stats']['avg_confidence']:.2f}")

if __name__ == "__main__":
    main()