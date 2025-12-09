"""
Model Training Script for Aerial Threat Detection
Trains YOLOv8 model on soldier/civilian classification dataset
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO


class ModelTrainer:
    """Train YOLO model for aerial threat detection"""
    
    def __init__(self, data_yaml, model='yolov8n.pt', epochs=100, imgsz=640, batch=16):
        """
        Initialize trainer
        
        Args:
            data_yaml: Path to data.yaml configuration file
            model: Base model to use (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
        """
        self.data_yaml = Path(data_yaml)
        self.model_name = model
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        
        # Setup output directory
        self.output_dir = Path('runs/train')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify dataset
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Data config not found: {self.data_yaml}")
        
        # Load data config
        with open(self.data_yaml, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        print(f"Dataset configuration loaded:")
        print(f"  Classes: {self.data_config.get('names', [])}")
        print(f"  Number of classes: {self.data_config.get('nc', 0)}")
    
    def train(self, device='auto', workers=8, patience=50, save_period=10):
        """
        Train the model
        
        Args:
            device: Device to use ('auto', 'cpu', '0', '0,1', etc.)
            workers: Number of worker threads for data loading
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
        """
        print("\n" + "=" * 80)
        print("STARTING MODEL TRAINING")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"Epochs: {self.epochs}")
        print(f"Image size: {self.imgsz}")
        print(f"Batch size: {self.batch}")
        print(f"Device: {device}")
        print("=" * 80 + "\n")
        
        # Initialize model
        model = YOLO(self.model_name)
        
        # Train the model
        results = model.train(
            data=str(self.data_yaml),
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            device=device,
            workers=workers,
            patience=patience,
            save=True,
            save_period=save_period,
            project='runs/train',
            name='aerial_threat_detector',
            exist_ok=True,
            pretrained=True,
            optimizer='auto',
            verbose=True,
            seed=0,
            deterministic=True,
            single_cls=False,
            rect=False,
            cos_lr=False,
            close_mosaic=10,
            resume=False,
            amp=True,
            fraction=1.0,
            profile=False,
            # Data augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0
        )
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print("=" * 80)
        print(f"Results saved to: {results.save_dir}")
        print(f"Best model: {results.save_dir}/weights/best.pt")
        print(f"Last model: {results.save_dir}/weights/last.pt")
        print("=" * 80 + "\n")
        
        return results
    
    def validate(self, model_path='runs/train/aerial_threat_detector/weights/best.pt'):
        """
        Validate the trained model
        
        Args:
            model_path: Path to trained model
        """
        print("\n" + "=" * 80)
        print("VALIDATING MODEL")
        print("=" * 80)
        
        model = YOLO(model_path)
        
        # Validate on validation set
        metrics = model.val(
            data=str(self.data_yaml),
            imgsz=self.imgsz,
            batch=self.batch,
            conf=0.001,
            iou=0.6,
            device='auto',
            plots=True
        )
        
        print("\n" + "=" * 80)
        print("VALIDATION RESULTS")
        print("=" * 80)
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        print("=" * 80 + "\n")
        
        return metrics
    
    def export_model(self, model_path='runs/train/aerial_threat_detector/weights/best.pt', 
                    format='onnx'):
        """
        Export model to different formats
        
        Args:
            model_path: Path to trained model
            format: Export format ('onnx', 'torchscript', 'coreml', 'tflite', etc.)
        """
        print(f"\nExporting model to {format} format...")
        
        model = YOLO(model_path)
        model.export(format=format)
        
        print(f"✓ Model exported successfully")
    
    def test_prediction(self, model_path='runs/train/aerial_threat_detector/weights/best.pt',
                       test_image=None):
        """
        Test model on a sample image
        
        Args:
            model_path: Path to trained model
            test_image: Path to test image (optional)
        """
        print("\nTesting model prediction...")
        
        model = YOLO(model_path)
        
        if test_image and os.path.exists(test_image):
            results = model.predict(
                source=test_image,
                conf=0.5,
                save=True,
                project='runs/predict',
                name='test'
            )
            print(f"✓ Prediction saved to runs/predict/test")
        else:
            print("No test image provided or file not found")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Train YOLO model for Aerial Threat Detection'
    )
    
    # Required arguments
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to data.yaml configuration file'
    )
    
    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
        help='Base model to use (n=nano, s=small, m=medium, l=large, x=xlarge)'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (adjust based on GPU memory)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for training'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto, cpu, 0, 0,1, etc.)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of worker threads for data loading'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience'
    )
    
    # Actions
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation after training'
    )
    parser.add_argument(
        '--export',
        type=str,
        choices=['onnx', 'torchscript', 'coreml', 'tflite'],
        help='Export model to specified format after training'
    )
    parser.add_argument(
        '--test-image',
        type=str,
        help='Test image path for prediction after training'
    )
    
    # Validation only mode
    parser.add_argument(
        '--validate-only',
        type=str,
        help='Only validate an existing model (provide model path)'
    )
    
    args = parser.parse_args()
    
    # Print system information
    print("\n" + "=" * 80)
    print("SYSTEM INFORMATION")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80 + "\n")
    
    # Validation only mode
    if args.validate_only:
        trainer = ModelTrainer(
            data_yaml=args.data,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch
        )
        trainer.validate(args.validate_only)
        return
    
    # Initialize trainer
    trainer = ModelTrainer(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch
    )
    
    # Train the model
    results = trainer.train(
        device=args.device,
        workers=args.workers,
        patience=args.patience
    )
    
    # Validate if requested
    if args.validate:
        trainer.validate()
    
    # Export if requested
    if args.export:
        trainer.export_model(format=args.export)
    
    # Test prediction if requested
    if args.test_image:
        trainer.test_prediction(test_image=args.test_image)
    
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Check training results in runs/train/aerial_threat_detector/")
    print("2. Copy best.pt to project root for deployment")
    print("3. Test the model with: python src/aerial_threat_detector.py")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
