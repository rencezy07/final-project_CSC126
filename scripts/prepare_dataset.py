"""
Dataset Preparation Script for Aerial Threat Detection
Downloads and prepares datasets from Roboflow for training
"""

import os
import sys
import argparse
import requests
import zipfile
from pathlib import Path


class DatasetPreparer:
    """Prepare datasets from Roboflow for YOLO training"""
    
    def __init__(self, output_dir="datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Available datasets from problem statement
        self.available_datasets = {
            'uav-person': {
                'url': 'https://universe.roboflow.com/militarypersons/uav-person-3',
                'description': 'UAV Person Detection Dataset',
                'type': 'person'
            },
            'combatant': {
                'url': 'https://universe.roboflow.com/minwoo/combatant-dataset',
                'description': 'Combatant Detection Dataset',
                'type': 'soldier'
            },
            'soldiers': {
                'url': 'https://universe.roboflow.com/xphoenixua-nlncq/soldiers-detection-spf',
                'description': 'Soldiers Detection Dataset',
                'type': 'soldier'
            },
            'look-down-folks': {
                'url': 'https://universe.roboflow.com/folks/look-down-folks',
                'description': 'Look Down Folks Dataset',
                'type': 'civilian'
            }
        }
    
    def list_datasets(self):
        """List available datasets"""
        print("\nAvailable Datasets:")
        print("=" * 80)
        for key, info in self.available_datasets.items():
            print(f"\nDataset: {key}")
            print(f"  Description: {info['description']}")
            print(f"  Type: {info['type']}")
            print(f"  URL: {info['url']}")
        print("=" * 80)
    
    def download_from_roboflow(self, dataset_key, api_key=None):
        """
        Download dataset from Roboflow
        
        Args:
            dataset_key: Key from available_datasets
            api_key: Roboflow API key (optional)
        
        Note: This is a template. Users need to:
        1. Sign up at roboflow.com
        2. Get their API key
        3. Use Roboflow Python package for actual download
        """
        if dataset_key not in self.available_datasets:
            print(f"Error: Dataset '{dataset_key}' not found")
            return False
        
        dataset_info = self.available_datasets[dataset_key]
        print(f"\nTo download '{dataset_info['description']}':")
        print("1. Visit:", dataset_info['url'])
        print("2. Sign up/Login to Roboflow")
        print("3. Click 'Download Dataset' and select 'YOLOv8' format")
        print("4. Follow the download instructions provided")
        print("5. Extract the dataset to:", self.output_dir / dataset_key)
        
        if api_key:
            print("\nWith API key, you can use:")
            print("from roboflow import Roboflow")
            print(f"rf = Roboflow(api_key='{api_key[:10]}...')")
            print("# Then follow Roboflow's Python API documentation")
        else:
            print("\nFor automated downloads, install roboflow:")
            print("  pip install roboflow")
            print("  Then use your API key from roboflow.com")
        
        return True
    
    def prepare_dataset_structure(self, dataset_dir):
        """
        Prepare YOLO dataset structure
        
        Expected structure:
        dataset/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── val/
        │   ├── images/
        │   └── labels/
        ├── test/
        │   ├── images/
        │   └── labels/
        └── data.yaml
        """
        dataset_path = Path(dataset_dir)
        
        if not dataset_path.exists():
            print(f"Error: Dataset directory not found: {dataset_dir}")
            return False
        
        # Check for required directories
        required_dirs = [
            'train/images', 'train/labels',
            'val/images', 'val/labels'
        ]
        
        all_present = True
        for dir_path in required_dirs:
            full_path = dataset_path / dir_path
            if not full_path.exists():
                print(f"Warning: Missing directory: {full_path}")
                all_present = False
        
        if all_present:
            print(f"✓ Dataset structure verified: {dataset_dir}")
            return True
        else:
            print(f"✗ Dataset structure incomplete: {dataset_dir}")
            return False
    
    def create_data_yaml(self, dataset_dir, class_names, dataset_name="aerial_threat"):
        """
        Create data.yaml file for YOLO training
        
        Args:
            dataset_dir: Path to dataset directory
            class_names: List of class names (e.g., ['soldier', 'civilian'])
            dataset_name: Name of the dataset
        """
        dataset_path = Path(dataset_dir)
        yaml_path = dataset_path / 'data.yaml'
        
        yaml_content = f"""# Aerial Threat Detection Dataset Configuration
# Generated for YOLOv8 training

path: {dataset_path.absolute()}
train: train/images
val: val/images
test: test/images  # optional

# Classes
nc: {len(class_names)}  # number of classes
names: {class_names}  # class names

# Dataset info
dataset_name: {dataset_name}
description: Soldier and civilian classification from aerial imagery
"""
        
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"✓ Created data.yaml: {yaml_path}")
        return yaml_path
    
    def verify_dataset(self, dataset_dir):
        """Verify dataset integrity and print statistics"""
        dataset_path = Path(dataset_dir)
        
        if not dataset_path.exists():
            print(f"Error: Dataset not found: {dataset_dir}")
            return False
        
        print(f"\nVerifying dataset: {dataset_dir}")
        print("=" * 80)
        
        for split in ['train', 'val', 'test']:
            images_dir = dataset_path / split / 'images'
            labels_dir = dataset_path / split / 'labels'
            
            if images_dir.exists():
                image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
                label_files = list(labels_dir.glob('*.txt')) if labels_dir.exists() else []
                
                print(f"\n{split.upper()} Split:")
                print(f"  Images: {len(image_files)}")
                print(f"  Labels: {len(label_files)}")
                
                if len(image_files) != len(label_files):
                    print(f"  ⚠ Warning: Mismatch between images and labels")
        
        print("=" * 80)
        return True
    
    def augment_dataset(self, dataset_dir, output_dir, augmentations=['rotate', 'flip', 'scale']):
        """
        Apply data augmentation to the dataset
        
        Args:
            dataset_dir: Source dataset directory
            output_dir: Output directory for augmented dataset
            augmentations: List of augmentation types to apply
        """
        print(f"\nData augmentation is recommended for better model generalization.")
        print(f"Augmentations to apply: {', '.join(augmentations)}")
        print("\nFor augmentation, you can use:")
        print("1. Roboflow's built-in augmentation (recommended)")
        print("2. Albumentations library: pip install albumentations")
        print("3. imgaug library: pip install imgaug")
        print("\nExample augmentations:")
        print("  - Rotation: ±15 degrees")
        print("  - Horizontal flip")
        print("  - Brightness adjustment")
        print("  - Zoom: 0.8x to 1.2x")
        
        return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Prepare datasets for Aerial Threat Detection training'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets'
    )
    parser.add_argument(
        '--download',
        type=str,
        help='Download dataset by key (use --list to see available datasets)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Roboflow API key for automated downloads'
    )
    parser.add_argument(
        '--verify',
        type=str,
        help='Verify dataset structure and integrity'
    )
    parser.add_argument(
        '--create-yaml',
        type=str,
        help='Create data.yaml for dataset'
    )
    parser.add_argument(
        '--classes',
        type=str,
        nargs='+',
        default=['soldier', 'civilian'],
        help='Class names for data.yaml'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='datasets',
        help='Output directory for datasets'
    )
    
    args = parser.parse_args()
    
    preparer = DatasetPreparer(output_dir=args.output_dir)
    
    if args.list:
        preparer.list_datasets()
    
    if args.download:
        preparer.download_from_roboflow(args.download, args.api_key)
    
    if args.verify:
        preparer.verify_dataset(args.verify)
    
    if args.create_yaml:
        preparer.create_data_yaml(args.create_yaml, args.classes)
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n" + "=" * 80)
        print("QUICK START GUIDE")
        print("=" * 80)
        print("\n1. List available datasets:")
        print("   python prepare_dataset.py --list")
        print("\n2. Download instructions for a dataset:")
        print("   python prepare_dataset.py --download uav-person")
        print("\n3. After downloading, verify the dataset:")
        print("   python prepare_dataset.py --verify datasets/uav-person")
        print("\n4. Create data.yaml configuration:")
        print("   python prepare_dataset.py --create-yaml datasets/uav-person --classes soldier civilian")
        print("=" * 80)


if __name__ == "__main__":
    main()
