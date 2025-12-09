import cv2
import numpy as np
import os
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class EvaluationMetrics:
    """
    Class for calculating and visualizing model performance metrics
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.true_positives = defaultdict(int)
        self.false_positives = defaultdict(int)
        self.false_negatives = defaultdict(int)
        self.confidences = defaultdict(list)
        self.all_classes = set()
    
    def add_detection(self, predicted_class: str, actual_class: str, confidence: float, iou: float = 0.5):
        """
        Add a detection result for evaluation
        
        Args:
            predicted_class: Predicted class name
            actual_class: Ground truth class name
            confidence: Prediction confidence
            iou: Intersection over Union score
        """
        self.all_classes.add(predicted_class)
        self.all_classes.add(actual_class)
        
        if predicted_class == actual_class and iou >= 0.5:
            self.true_positives[predicted_class] += 1
        else:
            self.false_positives[predicted_class] += 1
            if actual_class:  # If there was a ground truth
                self.false_negatives[actual_class] += 1
        
        self.confidences[predicted_class].append(confidence)
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate precision, recall, and F1 scores
        
        Returns:
            Dictionary containing metrics for each class
        """
        metrics = {}
        
        for class_name in self.all_classes:
            tp = self.true_positives[class_name]
            fp = self.false_positives[class_name]
            fn = self.false_negatives[class_name]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            avg_confidence = np.mean(self.confidences[class_name]) if self.confidences[class_name] else 0
            
            metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'avg_confidence': avg_confidence,
                'num_predictions': len(self.confidences[class_name])
            }
        
        return metrics
    
    def print_metrics(self):
        """Print formatted metrics to console"""
        metrics = self.calculate_metrics()
        
        print("\n" + "="*80)
        print("MODEL EVALUATION METRICS")
        print("="*80)
        
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'TP':<5} {'FP':<5} {'FN':<5}")
        print("-"*80)
        
        total_tp, total_fp, total_fn = 0, 0, 0
        
        for class_name, class_metrics in metrics.items():
            print(f"{class_name:<15} {class_metrics['precision']:<10.3f} {class_metrics['recall']:<10.3f} "
                  f"{class_metrics['f1_score']:<10.3f} {class_metrics['true_positives']:<5} "
                  f"{class_metrics['false_positives']:<5} {class_metrics['false_negatives']:<5}")
            
            total_tp += class_metrics['true_positives']
            total_fp += class_metrics['false_positives']
            total_fn += class_metrics['false_negatives']
        
        # Calculate overall metrics
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        print("-"*80)
        print(f"{'OVERALL':<15} {overall_precision:<10.3f} {overall_recall:<10.3f} {overall_f1:<10.3f} {total_tp:<5} {total_fp:<5} {total_fn:<5}")
        print("="*80)
    
    def save_metrics_plot(self, output_path: str = "metrics_plot.png"):
        """
        Save a visualization of the metrics
        
        Args:
            output_path: Path to save the plot
        """
        metrics = self.calculate_metrics()
        
        if not metrics:
            print("No metrics to plot")
            return
        
        classes = list(metrics.keys())
        precision_scores = [metrics[c]['precision'] for c in classes]
        recall_scores = [metrics[c]['recall'] for c in classes]
        f1_scores = [metrics[c]['f1_score'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars1 = ax.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Scores')
        ax.set_title('Model Performance Metrics by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Metrics plot saved to: {output_path}")


class VideoProcessor:
    """
    Utility class for processing video files and streams
    """
    
    @staticmethod
    def extract_frames(video_path: str, output_dir: str, interval: int = 30):
        """
        Extract frames from video at specified intervals
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save extracted frames
            interval: Extract every nth frame
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return False
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % interval == 0:
                frame_filename = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {saved_count} frames from {frame_count} total frames")
        return True
    
    @staticmethod
    def create_video_from_frames(frames_dir: str, output_path: str, fps: int = 30):
        """
        Create video from directory of frames
        
        Args:
            frames_dir: Directory containing frame images
            output_path: Path for output video
            fps: Frames per second for output video
        """
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
        
        if not frame_files:
            print("No frame files found")
            return False
        
        # Read first frame to get dimensions
        first_frame_path = os.path.join(frames_dir, frame_files[0])
        first_frame = cv2.imread(first_frame_path)
        height, width, _ = first_frame.shape
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            out.write(frame)
        
        out.release()
        print(f"Video created: {output_path}")
        return True


class ImageUtils:
    """
    Utility functions for image processing and visualization
    """
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: Input image
            target_size: (width, height) target size
            
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        # Calculate scaling factor
        scale = min(target_width / width, target_height / height)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height))
        
        # Create padded image
        padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate padding
        pad_x = (target_width - new_width) // 2
        pad_y = (target_height - new_height) // 2
        
        # Place resized image in center
        padded[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
        
        return padded
    
    @staticmethod
    def create_detection_grid(images: List[np.ndarray], grid_size: Tuple[int, int] = (2, 2)) -> np.ndarray:
        """
        Create a grid of images for visualization
        
        Args:
            images: List of images
            grid_size: (rows, cols) for the grid
            
        Returns:
            Grid image
        """
        rows, cols = grid_size
        if len(images) > rows * cols:
            images = images[:rows * cols]
        
        # Resize all images to same size
        target_size = (400, 300)
        resized_images = [ImageUtils.resize_image(img, target_size) for img in images]
        
        # Pad with blank images if necessary
        while len(resized_images) < rows * cols:
            blank = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            resized_images.append(blank)
        
        # Create grid
        grid_rows = []
        for i in range(rows):
            row_images = resized_images[i * cols:(i + 1) * cols]
            grid_row = np.hstack(row_images)
            grid_rows.append(grid_row)
        
        grid = np.vstack(grid_rows)
        return grid
    
    @staticmethod
    def add_info_panel(image: np.ndarray, info_text: List[str]) -> np.ndarray:
        """
        Add information panel to image
        
        Args:
            image: Input image
            info_text: List of text lines to display
            
        Returns:
            Image with info panel
        """
        height, width = image.shape[:2]
        panel_width = 300
        
        # Create extended image
        extended_image = np.zeros((height, width + panel_width, 3), dtype=np.uint8)
        extended_image[:, :width] = image
        extended_image[:, width:] = (50, 50, 50)  # Dark gray panel
        
        # Add text to panel
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(extended_image, text, (width + 10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return extended_image


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
        
    Returns:
        IoU score
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if boxes intersect
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # Calculate areas
    intersection_area = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


def non_max_suppression(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections
    
    Args:
        detections: List of detection dictionaries
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Filtered detections
    """
    if not detections:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while detections:
        # Take the detection with highest confidence
        current = detections.pop(0)
        keep.append(current)
        
        # Remove detections with high IoU
        remaining = []
        for det in detections:
            iou = calculate_iou(current['bbox'], det['bbox'])
            if iou < iou_threshold or current['class_name'] != det['class_name']:
                remaining.append(det)
        
        detections = remaining
    
    return keep


if __name__ == "__main__":
    # Example usage of utility functions
    print("Aerial Threat Detection - Utility Functions")
    print("This module provides evaluation metrics and image processing utilities.")
    
    # Example: Create sample metrics
    evaluator = EvaluationMetrics()
    
    # Add some sample detection results
    evaluator.add_detection('soldier', 'soldier', 0.85, 0.7)
    evaluator.add_detection('civilian', 'civilian', 0.92, 0.8)
    evaluator.add_detection('soldier', 'civilian', 0.75, 0.6)
    evaluator.add_detection('civilian', 'soldier', 0.65, 0.5)
    
    # Print metrics
    evaluator.print_metrics()
    
    # Save plot
    evaluator.save_metrics_plot("sample_metrics.png")