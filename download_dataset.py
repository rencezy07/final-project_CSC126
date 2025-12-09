"""
Dataset Download and Preparation Script
Downloads aerial surveillance datasets from Roboflow and prepares them for training
"""

import os
import argparse
import requests
from pathlib import Path
import zipfile
import shutil


def download_roboflow_dataset(api_key, workspace, project, version, format='yolov8', location='dataset'):
    """
    Download dataset from Roboflow
    
    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace name
        project: Project name
        version: Dataset version
        format: Export format (yolov8, yolov5, coco, etc.)
        location: Directory to save dataset
    
    Returns:
        Path to downloaded dataset
    """
    try:
        from roboflow import Roboflow
        
        print(f"Downloading dataset from Roboflow...")
        print(f"Workspace: {workspace}")
        print(f"Project: {project}")
        print(f"Version: {version}")
        
        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download(format, location=location)
        
        print(f"Dataset downloaded successfully to: {dataset.location}")
        return dataset.location
        
    except ImportError:
        print("ERROR: roboflow package not installed")
        print("Install it with: pip install roboflow")
        return None
    except Exception as e:
        print(f"ERROR downloading dataset: {str(e)}")
        return None


def prepare_dataset_structure(dataset_path):
    """
    Verify and prepare dataset structure for YOLO training
    
    Args:
        dataset_path: Path to dataset directory
    
    Returns:
        True if structure is valid, False otherwise
    """
    dataset_path = Path(dataset_path)
    
    print("\nVerifying dataset structure...")
    
    # Check required directories
    required_dirs = {
        'train': ['images', 'labels'],
        'valid': ['images', 'labels']
    }
    
    all_valid = True
    for split, subdirs in required_dirs.items():
        split_path = dataset_path / split
        if not split_path.exists():
            print(f"ERROR: Missing {split} directory")
            all_valid = False
            continue
        
        for subdir in subdirs:
            subdir_path = split_path / subdir
            if not subdir_path.exists():
                print(f"ERROR: Missing {split}/{subdir} directory")
                all_valid = False
            else:
                # Count files
                files = list(subdir_path.glob('*'))
                print(f"✓ {split}/{subdir}: {len(files)} files")
    
    # Check test directory (optional)
    test_path = dataset_path / 'test'
    if test_path.exists():
        for subdir in ['images', 'labels']:
            subdir_path = test_path / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob('*'))
                print(f"✓ test/{subdir}: {len(files)} files")
    
    return all_valid


def combine_datasets(dataset_paths, output_path='combined_dataset'):
    """
    Combine multiple datasets into a single dataset
    
    Args:
        dataset_paths: List of paths to datasets
        output_path: Path to save combined dataset
    
    Returns:
        Path to combined dataset
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCombining {len(dataset_paths)} datasets into: {output_path}")
    
    # Create directory structure
    for split in ['train', 'valid', 'test']:
        for subdir in ['images', 'labels']:
            (output_path / split / subdir).mkdir(parents=True, exist_ok=True)
    
    # Copy files from each dataset
    for idx, dataset_path in enumerate(dataset_paths):
        dataset_path = Path(dataset_path)
        print(f"\nProcessing dataset {idx + 1}/{len(dataset_paths)}: {dataset_path}")
        
        for split in ['train', 'valid', 'test']:
            split_path = dataset_path / split
            if not split_path.exists():
                continue
            
            for subdir in ['images', 'labels']:
                src_dir = split_path / subdir
                if not src_dir.exists():
                    continue
                
                dst_dir = output_path / split / subdir
                
                # Copy files with prefix to avoid name conflicts
                for file_path in src_dir.glob('*'):
                    if file_path.is_file():
                        new_name = f"ds{idx}_{file_path.name}"
                        dst_path = dst_dir / new_name
                        shutil.copy2(file_path, dst_path)
                
                files_count = len(list(src_dir.glob('*')))
                print(f"  Copied {files_count} files from {split}/{subdir}")
    
    print(f"\nCombined dataset created at: {output_path}")
    return str(output_path)


def display_dataset_info(dataset_path):
    """
    Display information about the dataset
    
    Args:
        dataset_path: Path to dataset directory
    """
    dataset_path = Path(dataset_path)
    
    print("\n" + "="*80)
    print("DATASET INFORMATION")
    print("="*80)
    
    # Check for data.yaml or dataset.yaml
    yaml_files = list(dataset_path.glob('*.yaml'))
    if yaml_files:
        print(f"Configuration file: {yaml_files[0].name}")
        
        import yaml
        try:
            with open(yaml_files[0], 'r') as f:
                config = yaml.safe_load(f)
                if 'names' in config:
                    print(f"Classes: {config['names']}")
                if 'nc' in config:
                    print(f"Number of classes: {config['nc']}")
        except:
            pass
    
    # Count images and labels
    for split in ['train', 'valid', 'test']:
        split_path = dataset_path / split
        if split_path.exists():
            images = list((split_path / 'images').glob('*')) if (split_path / 'images').exists() else []
            labels = list((split_path / 'labels').glob('*')) if (split_path / 'labels').exists() else []
            print(f"\n{split.upper()}:")
            print(f"  Images: {len(images)}")
            print(f"  Labels: {len(labels)}")
    
    print("="*80)


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description='Download and prepare aerial surveillance dataset')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download dataset from Roboflow')
    download_parser.add_argument('--api-key', required=True, help='Roboflow API key')
    download_parser.add_argument('--workspace', required=True, help='Roboflow workspace')
    download_parser.add_argument('--project', required=True, help='Project name')
    download_parser.add_argument('--version', type=int, required=True, help='Dataset version')
    download_parser.add_argument('--format', default='yolov8', help='Export format')
    download_parser.add_argument('--output', default='dataset', help='Output directory')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify dataset structure')
    verify_parser.add_argument('dataset', help='Path to dataset directory')
    
    # Combine command
    combine_parser = subparsers.add_parser('combine', help='Combine multiple datasets')
    combine_parser.add_argument('datasets', nargs='+', help='Paths to datasets to combine')
    combine_parser.add_argument('--output', default='combined_dataset', help='Output directory')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Display dataset information')
    info_parser.add_argument('dataset', help='Path to dataset directory')
    
    args = parser.parse_args()
    
    if args.command == 'download':
        dataset_path = download_roboflow_dataset(
            api_key=args.api_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            format=args.format,
            location=args.output
        )
        
        if dataset_path:
            prepare_dataset_structure(dataset_path)
            display_dataset_info(dataset_path)
            
            print("\n" + "="*80)
            print("Next steps:")
            print(f"1. Review the dataset at: {dataset_path}")
            print(f"2. Train the model with: python train_model.py --dataset {dataset_path}")
            print("="*80)
    
    elif args.command == 'verify':
        is_valid = prepare_dataset_structure(args.dataset)
        if is_valid:
            print("\n✓ Dataset structure is valid!")
            display_dataset_info(args.dataset)
        else:
            print("\n✗ Dataset structure has errors. Please fix them before training.")
    
    elif args.command == 'combine':
        combined_path = combine_datasets(args.datasets, args.output)
        prepare_dataset_structure(combined_path)
        display_dataset_info(combined_path)
    
    elif args.command == 'info':
        display_dataset_info(args.dataset)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
