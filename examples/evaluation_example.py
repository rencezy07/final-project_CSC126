"""
Model Evaluation Script
Evaluates trained YOLO model on test dataset and generates comprehensive metrics
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aerial_threat_detector import AerialThreatDetector
import cv2
import json
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two boxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU score
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou


def evaluate_yolo_model(model_path, dataset_yaml, save_dir='evaluation_results'):
    """
    Evaluate YOLO model using built-in validation
    
    Args:
        model_path: Path to trained model
        dataset_yaml: Path to dataset configuration
        save_dir: Directory to save results
    
    Returns:
        Validation metrics
    """
    print("\n" + "="*80)
    print("YOLO MODEL EVALUATION")
    print("="*80)
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)
    
    # Run validation
    print(f"Running validation on dataset: {dataset_yaml}")
    metrics = model.val(
        data=dataset_yaml,
        split='test',
        save_json=True,
        save_hybrid=True,
        conf=0.001,  # Low confidence for better recall calculation
        iou=0.6,
        max_det=300,
        plots=True,
        project=str(save_dir),
        name='validation'
    )
    
    # Print metrics
    print("\n" + "="*80)
    print("VALIDATION METRICS")
    print("="*80)
    
    print(f"\nmAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    # Per-class metrics
    if hasattr(metrics.box, 'maps'):
        print("\nPer-class mAP@0.5:")
        class_names = model.names
        for i, map_val in enumerate(metrics.box.maps):
            if i < len(class_names):
                print(f"  {class_names[i]}: {map_val:.4f}")
    
    print(f"\nResults saved to: {save_dir}")
    print("="*80)
    
    return metrics


def evaluate_on_test_images(model_path, test_images_dir, confidence=0.5):
    """
    Evaluate model on a directory of test images
    
    Args:
        model_path: Path to trained model
        test_images_dir: Directory containing test images
        confidence: Confidence threshold
    
    Returns:
        Dictionary with evaluation results
    """
    print("\n" + "="*80)
    print("TEST IMAGES EVALUATION")
    print("="*80)
    
    detector = AerialThreatDetector(model_path, confidence_threshold=confidence)
    
    if detector.model is None:
        print("ERROR: Could not load model")
        return None
    
    test_images_dir = Path(test_images_dir)
    
    if not test_images_dir.exists():
        print(f"ERROR: Test images directory not found: {test_images_dir}")
        return None
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(test_images_dir.glob(ext))
    
    if not image_files:
        print(f"No images found in {test_images_dir}")
        return None
    
    print(f"Found {len(image_files)} test images")
    
    # Process images
    results = {
        'total_images': len(image_files),
        'total_detections': 0,
        'class_counts': {},
        'images_with_detections': 0,
        'avg_confidence': [],
        'detections_per_image': []
    }
    
    for i, image_file in enumerate(image_files):
        print(f"Processing [{i+1}/{len(image_files)}]: {image_file.name}")
        
        annotated_image, detections = detector.detect_image(str(image_file))
        
        if detections:
            results['images_with_detections'] += 1
            results['total_detections'] += len(detections)
            results['detections_per_image'].append(len(detections))
            
            for det in detections:
                class_name = det['class_name']
                results['class_counts'][class_name] = results['class_counts'].get(class_name, 0) + 1
                results['avg_confidence'].append(det['confidence'])
    
    # Calculate statistics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Total images: {results['total_images']}")
    print(f"Images with detections: {results['images_with_detections']}")
    print(f"Total detections: {results['total_detections']}")
    print(f"Average detections per image: {np.mean(results['detections_per_image']):.2f}")
    print(f"Average confidence: {np.mean(results['avg_confidence']):.3f}")
    print(f"\nClass distribution:")
    for class_name, count in results['class_counts'].items():
        percentage = (count / results['total_detections']) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    print("="*80)
    
    return results


def plot_evaluation_results(results, save_path='evaluation_plot.png'):
    """
    Create visualization of evaluation results
    
    Args:
        results: Results dictionary from evaluate_on_test_images
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Class Distribution
    ax1 = axes[0, 0]
    if results['class_counts']:
        classes = list(results['class_counts'].keys())
        counts = list(results['class_counts'].values())
        colors = ['green' if 'civilian' in c.lower() else 'red' for c in classes]
        ax1.bar(classes, counts, color=colors, alpha=0.7)
        ax1.set_title('Class Distribution')
        ax1.set_ylabel('Count')
        ax1.set_xlabel('Class')
    
    # Plot 2: Detection Statistics
    ax2 = axes[0, 1]
    stats_labels = ['Total\nImages', 'Images with\nDetections', 'Total\nDetections']
    stats_values = [
        results['total_images'],
        results['images_with_detections'],
        results['total_detections']
    ]
    ax2.bar(stats_labels, stats_values, color=['blue', 'green', 'orange'], alpha=0.7)
    ax2.set_title('Detection Statistics')
    ax2.set_ylabel('Count')
    
    # Plot 3: Detections per Image Distribution
    ax3 = axes[1, 0]
    if results['detections_per_image']:
        ax3.hist(results['detections_per_image'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax3.set_title('Detections per Image Distribution')
        ax3.set_xlabel('Number of Detections')
        ax3.set_ylabel('Frequency')
        ax3.axvline(np.mean(results['detections_per_image']), color='red', 
                   linestyle='--', linewidth=2, label='Mean')
        ax3.legend()
    
    # Plot 4: Confidence Distribution
    ax4 = axes[1, 1]
    if results['avg_confidence']:
        ax4.hist(results['avg_confidence'], bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
        ax4.set_title('Confidence Score Distribution')
        ax4.set_xlabel('Confidence')
        ax4.set_ylabel('Frequency')
        ax4.axvline(np.mean(results['avg_confidence']), color='red', 
                   linestyle='--', linewidth=2, label='Mean')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nEvaluation plot saved to: {save_path}")
    plt.close()


def generate_evaluation_report(model_path, dataset_yaml=None, test_images_dir=None, 
                               output_dir='evaluation_results'):
    """
    Generate comprehensive evaluation report
    
    Args:
        model_path: Path to trained model
        dataset_yaml: Path to dataset configuration (for YOLO validation)
        test_images_dir: Directory with test images
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        'model_path': model_path,
        'yolo_metrics': None,
        'test_results': None
    }
    
    # 1. Run YOLO validation if dataset YAML is provided
    if dataset_yaml and os.path.exists(dataset_yaml):
        print("\n[1/2] Running YOLO validation...")
        try:
            metrics = evaluate_yolo_model(model_path, dataset_yaml, 
                                         save_dir=str(output_dir / 'yolo_validation'))
            report['yolo_metrics'] = {
                'map50': float(metrics.box.map50),
                'map': float(metrics.box.map),
                'precision': float(metrics.box.mp),
                'recall': float(metrics.box.mr)
            }
        except Exception as e:
            print(f"Error in YOLO validation: {e}")
    
    # 2. Evaluate on test images if directory is provided
    if test_images_dir and os.path.exists(test_images_dir):
        print("\n[2/2] Evaluating on test images...")
        try:
            results = evaluate_on_test_images(model_path, test_images_dir)
            if results:
                report['test_results'] = {
                    'total_images': results['total_images'],
                    'images_with_detections': results['images_with_detections'],
                    'total_detections': results['total_detections'],
                    'avg_detections_per_image': float(np.mean(results['detections_per_image'])),
                    'avg_confidence': float(np.mean(results['avg_confidence'])),
                    'class_counts': results['class_counts']
                }
                
                # Create visualization
                plot_evaluation_results(results, 
                                       save_path=str(output_dir / 'evaluation_plot.png'))
        except Exception as e:
            print(f"Error in test images evaluation: {e}")
    
    # Save report
    report_path = output_dir / 'evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n" + "="*80)
    print(f"Evaluation report saved to: {report_path}")
    print("="*80)
    
    return report


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Aerial Threat Detection Model')
    
    parser.add_argument('--model', required=True, help='Path to trained model (.pt)')
    parser.add_argument('--dataset-yaml', help='Path to dataset YAML for YOLO validation')
    parser.add_argument('--test-images', help='Directory containing test images')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='Confidence threshold')
    parser.add_argument('--output', default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Verify model exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        return
    
    # Run evaluation
    report = generate_evaluation_report(
        model_path=args.model,
        dataset_yaml=args.dataset_yaml,
        test_images_dir=args.test_images,
        output_dir=args.output
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.output}/")
    print("="*80)


if __name__ == '__main__':
    main()
