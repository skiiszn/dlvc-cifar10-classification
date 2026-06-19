"""
YOLOv8 Training Module for Brain Tumor Detection

Provides training and validation logic for YOLOv8 model with
metrics logging and model checkpointing.

Usage:
    from model_yolov8 import YOLOv8
    from trainer_yolov8 import YOLOv8Trainer
    
    model = YOLOv8(model_size='m')
    trainer = YOLOv8Trainer(model=model, data_yaml='data.yaml')
    trainer.train(epochs=100, batch_size=16)
"""

from pathlib import Path
from typing import Tuple, Optional, Dict
from abc import ABCMeta, abstractmethod
import torch
from tqdm import tqdm

from models.model_yolov8 import YOLOv8


class BaseYOLOTrainer(metaclass=ABCMeta):
    """
    Base class for YOLO trainers.
    Defines the interface all YOLO trainers must implement.
    """
    
    @abstractmethod
    def train(self) -> None:
        """
        Holds training logic.
        """
        pass
    
    
    @abstractmethod
    def validate(self) -> Dict:
        """
        Holds validation logic.
        """
        pass


class YOLOv8Trainer(BaseYOLOTrainer):
    """
    Trainer for YOLOv8 object detection model.
    """
    
    def __init__(
        self,
        model: YOLOv8,
        data_yaml: str,
        training_save_dir: Path = Path('./checkpoints'),
        device: str = None,
        val_frequency: int = 5,
    ) -> None:
        """
        Initialize YOLOv8 trainer.
        
        Args:
            model (YOLOv8): YOLOv8 model instance to train
            data_yaml (str): Path to data.yaml configuration file
            training_save_dir (Path): Directory to save checkpoints. Default: './checkpoints'
            device (str): Device for training ('cpu', 'cuda'). Default: None (uses model device)
            val_frequency (int): Validation frequency (every N epochs). Default: 5
        """
        
        self.model = model
        self.data_yaml = data_yaml
        self.training_save_dir = Path(training_save_dir)
        self.device = device or model.device
        self.val_frequency = val_frequency
        
        # Create checkpoint directory
        self.training_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Best metrics tracking
        self.best_map50 = -1.0
        self.best_map = -1.0
        self.best_epoch = -1
        
        # Metrics history
        self.metrics_history = {
            'train': [],
            'val': [],
        }
        
        print(f"YOLOv8Trainer initialized")
        print(f"  Model: {self.model.get_model_info()['model_type']}-{model.model_size}")
        print(f"  Save dir: {self.training_save_dir}")
        print(f"  Device: {self.device}")
        print(f"  Data: {self.data_yaml}")
    
    
    def train(
        self,
        epochs: int = 100,
        batch_size: int = 16,
        imgsz: int = 640,
        patience: int = 20,
        lr0: float = 0.001,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        **kwargs
    ) -> Dict:
        """
        Train the YOLOv8 model.
        
        Args:
            epochs (int): Number of training epochs. Default: 100
            batch_size (int): Batch size for training. Default: 16
            imgsz (int): Input image size. Default: 640
            patience (int): Early stopping patience. Default: 20
            lr0 (float): Initial learning rate. Default: 0.001
            momentum (float): Optimizer momentum. Default: 0.937
            weight_decay (float): Weight decay regularization. Default: 0.0005
            **kwargs: Additional arguments passed to model.train()
            
        Returns:
            Dict containing training results and metrics
        """

        
        training_config = {
            'epochs': epochs,
            'batch_size': batch_size,
            'imgsz': imgsz,
            'patience': patience,
            'lr0': lr0,
            'momentum': momentum,
            'weight_decay': weight_decay,
        }
        
        print("Training Configuration:")
        for key, value in training_config.items():
            print(f"  {key}: {value}")
        print()

        # Train using model's built-in training
        model_name = f'yolov8{self.model.model_size}'
        results = self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=imgsz,
            patience=patience,
            lr0=lr0,
            momentum=momentum,
            weight_decay=weight_decay,
            device=self.device,
            project=str(self.training_save_dir),
            name=model_name,
            **kwargs
        )

        
        return {
            'results': results,
            'config': training_config,
            'best_epoch': self.best_epoch,
            'best_map50': self.best_map50,
            'best_map': self.best_map,
        }
    
    
    def validate(self) -> Dict:
        """
        Validate the model on the validation set.
        
        Returns:
            Dict containing validation metrics
        """

        
        val_results = self.model.validate(
            data=self.data_yaml,
            imgsz=640,
            batch_size=16,
            device=self.device,
        )
        
        metrics = self._extract_metrics(val_results)
        self.metrics_history['val'].append(metrics)
        
        return metrics
    
    
    def _extract_metrics(self, results: Dict) -> Dict:
        """
        Extract key metrics from validation/training results.
        
        Args:
            results: Results dict from model.validate() or model.train()
            
        Returns:
            Dict with extracted metrics
        """
        
        try:
            result_obj = results.get('results')
            
            if result_obj is None:
                return {}
            
            # Extract metrics from ultralytics Results object
            metrics = {
                'map50': getattr(result_obj, 'box.map50', -1),
                'map': getattr(result_obj, 'box.map', -1),
                'precision': getattr(result_obj, 'box.p', -1),
                'recall': getattr(result_obj, 'box.r', -1),
            }
            
            return metrics
        
        except Exception as e:
            print(f"Warning: Could not extract metrics: {e}")
            return {}
    
    
    def save_checkpoint(self, suffix: Optional[str] = None) -> None:
        """
        Save model checkpoint.
        
        Args:
            suffix (str): Suffix for checkpoint filename. Default: None
        """
        
        self.model.save(self.training_save_dir, suffix=suffix)
    
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path (Path): Path to checkpoint file
        """
        
        self.model.load(checkpoint_path)
        print(f"Loaded checkpoint: {checkpoint_path}")
    
    
    def get_training_summary(self) -> Dict:
        """
        Get summary of training process.
        
        Returns:
            Dict containing training summary statistics
        """
        
        return {
            'best_map50': self.best_map50,
            'best_map': self.best_map,
            'best_epoch': self.best_epoch,
            'total_val_runs': len(self.metrics_history['val']),
            'metrics_history': self.metrics_history,
        }
