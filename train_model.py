"""
Model Training Script for Aerial Threat Detection
Trains YOLOv8 model on soldier and civilian classification dataset
"""

import os
import argparse
from ultralytics import YOLO
import torch
import yaml
from pathlib import Path


def setup_dataset_config(dataset_path, output_path='dataset.yaml'):
    """
    Create dataset configuration file for YOLO training
    
    Args:
        dataset_path: Path to the dataset directory
        output_path: Path to save the configuration file
    """
    config = {
        'path': str(Path(dataset_path).absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'names': {
            0: 'civilian',
            1: 'soldier'
        },
        'nc': 2  # number of classes
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Dataset configuration saved to: {output_path}")
    return output_path


def train_model(
    model_size='yolov8n',
    dataset_config='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch_size=16,
    device='',
    project='runs/detect',
    name='aerial_threat_model',
    pretrained=True,
    optimizer='auto',
    lr0=0.01,
    patience=50
):
    """
    Train YOLOv8 model for aerial threat detection
    
    Args:
        model_size: Model size (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        dataset_config: Path to dataset configuration YAML
        epochs: Number of training epochs
        imgsz: Input image size
        batch_size: Batch size for training
        device: Device to use ('' for auto, 'cpu', '0' for GPU 0, etc.)
        project: Project directory for saving results
        name: Experiment name
        pretrained: Use pretrained weights
        optimizer: Optimizer type (auto, SGD, Adam, AdamW)
        lr0: Initial learning rate
        patience: Early stopping patience (epochs)
    
    Returns:
        Trained model object
    """
    print("="*80)
    print("AERIAL THREAT DETECTION - MODEL TRAINING")
    print("="*80)
    print(f"Model: {model_size}")
    print(f"Dataset: {dataset_config}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {imgsz}")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {device if device else 'auto'}")
    print("="*80)
    
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("GPU Not Available - Training will use CPU (slower)")
    
    # Load model
    model_name = f"{model_size}.pt" if pretrained else f"{model_size}.yaml"
    print(f"\nLoading model: {model_name}")
    model = YOLO(model_name)
    
    # Train model
    print("\nStarting training...")
    results = model.train(
        data=dataset_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        optimizer=optimizer,
        lr0=lr0,
        patience=patience,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        plots=True,
        verbose=True,
        val=True,
        workers=8,
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    
    # Validate model
    print("\nValidating model on test set...")
    metrics = model.val()
    
    print("\n" + "="*80)
    print("VALIDATION METRICS")
    print("="*80)
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print("="*80)
    
    # Save best model to root directory
    best_model_path = Path(project) / name / 'weights' / 'best.pt'
    if best_model_path.exists():
        import shutil
        output_path = 'best.pt'
        shutil.copy(best_model_path, output_path)
        print(f"\nBest model copied to: {output_path}")
        print(f"You can now use this model with the detection system!")
    
    return model


def main():
    """Main training function with command-line arguments"""
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for aerial threat detection')
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--model', type=str, default='yolov8n',
                       choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                       help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--device', type=str, default='',
                       help='Device (empty for auto, cpu, 0, 0,1,2,3, etc.)')
    parser.add_argument('--project', type=str, default='runs/detect',
                       help='Project directory')
    parser.add_argument('--name', type=str, default='aerial_threat_model',
                       help='Experiment name')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Train from scratch without pretrained weights')
    parser.add_argument('--optimizer', type=str, default='auto',
                       choices=['auto', 'SGD', 'Adam', 'AdamW'],
                       help='Optimizer type')
    parser.add_argument('--lr0', type=float, default=0.01,
                       help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Setup dataset configuration
    dataset_config = setup_dataset_config(args.dataset)
    
    # Verify dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset path does not exist: {args.dataset}")
        return
    
    # Check for required directories
    required_dirs = ['train/images', 'valid/images', 'train/labels', 'valid/labels']
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"ERROR: Missing required directories: {missing_dirs}")
        print(f"Expected structure:")
        print(f"  {args.dataset}/")
        print(f"    ├── train/")
        print(f"    │   ├── images/")
        print(f"    │   └── labels/")
        print(f"    ├── valid/")
        print(f"    │   ├── images/")
        print(f"    │   └── labels/")
        print(f"    └── test/ (optional)")
        print(f"        ├── images/")
        print(f"        └── labels/")
        return
    
    # Train model
    model = train_model(
        model_size=args.model,
        dataset_config=dataset_config,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=not args.no_pretrained,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience
    )
    
    print("\n" + "="*80)
    print("All done! Your model is ready for deployment.")
    print("="*80)


if __name__ == '__main__':
    main()
