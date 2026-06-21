"""
Brain Tumor Detection - Data Preprocessing Script
Reorganizes data into YOLO format and creates train/val/test splits
"""

import os
import shutil
import yaml
from pathlib import Path
from collections import defaultdict
import random
from PIL import Image
import numpy as np


class BrainTumorDataPreprocessor:
    """Preprocesses brain tumor data into YOLO format"""
    
    def __init__(self, 
                 source_dir: str,
                 output_dir: str = None,
                 test_split_ratio: float = 0.2,
                 random_seed: int = 42):
        """
        Initialize preprocessor
        
        Args:
            source_dir: Path to source data directory (contains Train/Val folders)
            output_dir: Path to output directory (default: parent of source_dir/processed)
            test_split_ratio: Ratio for train->test split (0.2 = 20% of train becomes test)
            random_seed: Seed for reproducibility
        """
        self.source_dir = Path(source_dir)
        
        # Set output directory
        if output_dir is None:
            self.output_dir = self.source_dir.parent / "processed"
        else:
            self.output_dir = Path(output_dir)
        
        self.test_split_ratio = test_split_ratio
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Class mapping
        self.class_names = {
            'Glioma': 0,
            'Meningioma': 1,
            'NoTumor': 2,
            'Pituitary': 3
        }
        
        # Reverse mapping for reference
        self.class_id_to_name = {v: k for k, v in self.class_names.items()}
        
        # Statistics
        self.stats = {
            'train': defaultdict(int),
            'val': defaultdict(int),
            'test': defaultdict(int),
        }
        
        self.image_stats = {
            'train': [],
            'val': [],
            'test': [],
        }
        
        self.errors = {
            'missing_images': [],
            'missing_labels': [],
            'invalid_images': [],
            'invalid_labels': [],
        }
    
    def run(self):
        """Run complete preprocessing pipeline"""
        print("BRAIN TUMOR DATA PREPROCESSING")
        
        # Step 1: Validate source structure
        print("Step 1: Validating source data structure...")
        if not self._validate_source_structure():
            print(" Source structure validation failed!")
            return False
        print(" Source structure validated\n")
        
        # Step 2: Create output directory structure
        print("Step 2: Creating output directory structure...")
        self._create_output_structure()
        print(" Output directory structure created\n")
        
        # Step 3: Process training data
        print("Step 3: Processing training data...")
        train_files = self._process_split('Train', 'train', create_test=True)
        print(f" Training data processed\n")
        
        # Step 4: Process validation data
        print("Step 4: Processing validation data...")
        self._process_split('Val', 'val', create_test=False)
        print(f" Validation data processed\n")
        
        # Step 5: Create data.yaml
        print("Step 5: Creating data.yaml configuration...")
        self._create_data_yaml()
        print(" data.yaml created\n")
        
        # Step 6: Print statistics
        print("Step 6: Generating statistics report...")
        self._print_statistics()
        
        # Step 7: Validate output
        print("\nStep 7: Validating output structure...")
        self._validate_output_structure()

        
        print(f"Output directory: {self.output_dir}")
        
        return True
    
    def _validate_source_structure(self) -> bool:
        """Validate that source directory has expected structure"""
        expected_structures = [
            self.source_dir / 'Train',
            self.source_dir / 'Val',
        ]
        
        for path in expected_structures:
            if not path.exists():
                print(f" Missing: {path}")
                return False
        
        # Check for class folders
        for split_name in ['Train', 'Val']:
            split_dir = self.source_dir / split_name
            for class_name in self.class_names.keys():
                class_dir = split_dir / class_name
                if not class_dir.exists():
                    print(f" Missing class folder: {class_dir}")
                    return False
                
                # Check for images and labels subfolders
                images_dir = class_dir / 'images'
                labels_dir = class_dir / 'labels'
                
                if not images_dir.exists():
                    print(f" Missing images folder: {images_dir}")
                    return False
                if not labels_dir.exists():
                    print(f" Missing labels folder: {labels_dir}")
                    return False
        
        return True
    
    def _create_output_structure(self):
        """Create output directory structure"""
        splits = ['train', 'val', 'test']
        
        for split in splits:
            images_dir = self.output_dir / 'images' / split
            labels_dir = self.output_dir / 'labels' / split
            
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
    
    def _process_split(self, source_split: str, output_split: str, create_test: bool = False):
        """
        Process a data split
        
        Args:
            source_split: Source split name (Train/Val)
            output_split: Output split name (train/val/test)
            create_test: If True, split this data into train/test
        """
        split_dir = self.source_dir / source_split
        
        # Collect all files for this split
        all_files = []
        
        for class_name in self.class_names.keys():
            class_id = self.class_names[class_name]
            class_dir = split_dir / class_name
            images_dir = class_dir / 'images'
            labels_dir = class_dir / 'labels'
            
            # Find all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = [f for f in images_dir.iterdir() 
                          if f.suffix.lower() in image_extensions]
            
            print(f"  {source_split}/{class_name}: {len(image_files)} images")
            
            for image_file in image_files:
                label_file = labels_dir / (image_file.stem + '.txt')
                
                all_files.append({
                    'image': image_file,
                    'label': label_file,
                    'class_name': class_name,
                    'class_id': class_id,
                })
        
        # If creating test split, split the data
        if create_test:
            num_test = int(len(all_files) * self.test_split_ratio)
            test_indices = set(random.sample(range(len(all_files)), num_test))
            
            train_files = [f for i, f in enumerate(all_files) if i not in test_indices]
            test_files = [f for i, f in enumerate(all_files) if i in test_indices]
            
            print(f"\n  Train/Test split: {len(train_files)} train, {len(test_files)} test")
            
            # Process train files
            self._copy_files(train_files, 'train')
            
            # Process test files
            self._copy_files(test_files, 'test')
        else:
            # Process all files as validation
            self._copy_files(all_files, output_split)
    
    def _copy_files(self, files: list, output_split: str):
        """Copy files to output directory and validate"""
        images_output_dir = self.output_dir / 'images' / output_split
        labels_output_dir = self.output_dir / 'labels' / output_split
        
        for file_info in files:
            image_path = file_info['image']
            label_path = file_info['label']
            class_name = file_info['class_name']
            class_id = file_info['class_id']
            
            # Validate image exists and is readable
            if not image_path.exists():
                self.errors['missing_images'].append(str(image_path))
                continue
            
            try:
                img = Image.open(image_path)
                img_size = img.size
                img.verify()
            except Exception as e:
                self.errors['invalid_images'].append({
                    'path': str(image_path),
                    'error': str(e)
                })
                continue
            
            # Validate label exists and is readable
            if not label_path.exists():
                self.errors['missing_labels'].append(str(label_path))
                continue
            
            try:
                with open(label_path, 'r') as f:
                    label_content = f.read().strip()
                
                # Validate label format
                if not label_content:
                    self.errors['invalid_labels'].append({
                        'path': str(label_path),
                        'error': 'Empty label file'
                    })
                    continue
                
                # Check each line is valid YOLO format
                for line in label_content.split('\n'):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        raise ValueError(f"Invalid format: {line}")
                    
                    # Verify values are valid
                    label_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                           0 <= width <= 1 and 0 <= height <= 1):
                        raise ValueError(f"Coordinates out of range: {line}")
            
            except Exception as e:
                self.errors['invalid_labels'].append({
                    'path': str(label_path),
                    'error': str(e)
                })
                continue
            
            # Copy files to output
            try:
                output_image_path = images_output_dir / image_path.name
                output_label_path = labels_output_dir / label_path.name
                
                shutil.copy2(image_path, output_image_path)
                shutil.copy2(label_path, output_label_path)
                
                # Update statistics
                self.stats[output_split][class_name] += 1
                self.image_stats[output_split].append({
                    'class': class_name,
                    'size': img_size,
                    'path': output_image_path.name
                })
            
            except Exception as e:
                print(f" Error copying files: {e}")
                continue
    
    def _create_data_yaml(self):
        """Create data.yaml for YOLO"""
        data_yaml = {
            'path': '.',
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.class_names),
            'names': {v: k for k, v in self.class_names.items()}
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
        
        print(f"  Created: {yaml_path}")
    
    def _print_statistics(self):
        """Print detailed statistics"""
        print("DATA STATISTICS")
        
        # Per-split statistics
        for split in ['train', 'val', 'test']:
            if not self.stats[split]:
                continue
            
            print(f"{split.upper()} SET:")
            print("-" * 40)
            
            total_images = sum(self.stats[split].values())
            print(f"Total images: {total_images}")
            print("\nPer-class breakdown:")
            
            for class_name in sorted(self.class_names.keys()):
                count = self.stats[split][class_name]
                class_id = self.class_names[class_name]
                percentage = (count / total_images * 100) if total_images > 0 else 0
                bar = "#" * int(percentage / 2)
                print(f"  {class_name:12} ({class_id}): {count:5} ({percentage:5.1f}%) {bar}")
            
            print()
        
        # Overall statistics
        total_all = sum(sum(split_stats.values()) for split_stats in self.stats.values())
        
        print("\nOVERALL STATISTICS:")
        print(f"Total images: {total_all}")
        print(f"  Train: {sum(self.stats['train'].values())}")
        print(f"  Val:   {sum(self.stats['val'].values())}")
        print(f"  Test:  {sum(self.stats['test'].values())}")
        
        # Class distribution
        print("\nClass Distribution (All splits):")
        class_totals = defaultdict(int)
        for split in self.stats:
            for class_name, count in self.stats[split].items():
                class_totals[class_name] += count
        
        for class_name in sorted(self.class_names.keys()):
            count = class_totals[class_name]
            class_id = self.class_names[class_name]
            percentage = (count / total_all * 100) if total_all > 0 else 0
            print(f"  {class_name:12} ({class_id}): {count:5} ({percentage:5.1f}%)")
        
        # Error report
        if any(self.errors.values()):
            print("\n  ERROR REPORT:")
            
            if self.errors['missing_images']:
                print(f"Missing images: {len(self.errors['missing_images'])}")
                for path in self.errors['missing_images'][:3]:
                    print(f"  - {path}")
                if len(self.errors['missing_images']) > 3:
                    print(f"  ... and {len(self.errors['missing_images']) - 3} more")
            
            if self.errors['missing_labels']:
                print(f"Missing labels: {len(self.errors['missing_labels'])}")
                for path in self.errors['missing_labels'][:3]:
                    print(f"  - {path}")
                if len(self.errors['missing_labels']) > 3:
                    print(f"  ... and {len(self.errors['missing_labels']) - 3} more")
            
            if self.errors['invalid_images']:
                print(f"Invalid images: {len(self.errors['invalid_images'])}")
                for err in self.errors['invalid_images'][:3]:
                    print(f"  - {err['path']}: {err['error']}")
                if len(self.errors['invalid_images']) > 3:
                    print(f"  ... and {len(self.errors['invalid_images']) - 3} more")
            
            if self.errors['invalid_labels']:
                print(f"Invalid labels: {len(self.errors['invalid_labels'])}")
                for err in self.errors['invalid_labels'][:3]:
                    print(f"  - {err['path']}: {err['error']}")
                if len(self.errors['invalid_labels']) > 3:
                    print(f"  ... and {len(self.errors['invalid_labels']) - 3} more")
        else:
            print("\n NO ERRORS - All data validated successfully!")
    
    def _validate_output_structure(self):
        """Validate the output structure"""
        print("\nValidating output directory structure...")
        
        expected_dirs = [
            self.output_dir / 'images' / 'train',
            self.output_dir / 'images' / 'val',
            self.output_dir / 'images' / 'test',
            self.output_dir / 'labels' / 'train',
            self.output_dir / 'labels' / 'val',
            self.output_dir / 'labels' / 'test',
        ]
        
        for dir_path in expected_dirs:
            if not dir_path.exists():
                print(f" Missing: {dir_path}")
                return False
            
            # Check that directory is not empty
            files = list(dir_path.glob('*'))
            if not files:
                print(f"  Empty: {dir_path}")
        
        # Check data.yaml exists
        yaml_path = self.output_dir / 'data.yaml'
        if not yaml_path.exists():
            print(f" Missing: {yaml_path}")
            return False
        
        print(" Output structure validated!")
        return True


def main():
    """Main execution"""
    
    # Define paths
    source_dir = "./data"

    # Create preprocessor
    preprocessor = BrainTumorDataPreprocessor(
        source_dir=source_dir,
        test_split_ratio=0.2,  # 20% of training data becomes test set
        random_seed=42
    )
    
    # Run preprocessing
    success = preprocessor.run()
    
    if success:
        print("\n Preprocessing succeeded!")
    else:
        print("\n Preprocessing failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
