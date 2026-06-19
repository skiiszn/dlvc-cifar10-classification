"""
YOLOv8 Model Wrapper for Brain Tumor Detection

Provides a consistent interface for YOLOv8 object detection model
with save/load functionality and training integration.

Usage:
    from model_yolov8 import YOLOv8
    
    model = YOLOv8(model_size='m')  # nano, small, medium, large, xlarge
    results = model.detect(image_path)
    model.save(save_dir)
    model.load(model_path)
"""

from pathlib import Path
from typing import Union, List, Dict, Optional
import torch
from ultralytics import YOLO


class YOLOv8:
    """
    YOLOv8 Object Detection Model wrapper.
    
    Wraps ultralytics YOLOv8 model with additional utilities for
    saving, loading, and integration with training pipeline.
    """
    
    def __init__(self, model_size: str = 'm', device: str = None):
        """
        Initialize YOLOv8 model.
        
        Args:
            model_size (str): Model size - 'n' (nano), 's' (small), 'm' (medium),
                            'l' (large), 'x' (xlarge). Default: 'm'
        
        Raises:
            ValueError: If model_size is not valid
        """
        
        valid_sizes = ['n', 's', 'm', 'l', 'x']
        if model_size not in valid_sizes:
            raise ValueError(f"model_size must be one of {valid_sizes}, got {model_size}")
        
        self.model_size = model_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pretrained YOLOv8 model
        model_name = f'yolov8{model_size}'
        self.model = YOLO(f'{model_name}.pt')
        self.model.to(self.device)
        
        print(f"YOLOv8-{model_size.upper()} model loaded on {self.device}")
    
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor or image
            
        Returns:
            Model predictions
        """
        return self.model(x)
    
    
    def detect(self, source: Union[str, Path, torch.Tensor], 
               conf: float = 0.45, iou: float = 0.45) -> Dict:
        """
        Run inference on an image or batch of images.
        
        Args:
            source: Image path, list of paths, or tensor
            conf (float): Confidence threshold (0-1). Default: 0.45
            iou (float): IOU threshold for NMS (0-1). Default: 0.45
            
        Returns:
            Dict containing detection results with keys:
                - 'results': ultralytics Results object
                - 'boxes': Detected bounding boxes
                - 'confidences': Confidence scores
                - 'class_ids': Class IDs
        """
        
        results = self.model(source, conf=conf, iou=iou, device=self.device)
        
        detection_data = {
            'results': results,
            'boxes': None,
            'confidences': None,
            'class_ids': None,
        }
        
        if len(results) > 0 and results[0].boxes is not None:
            detection_data['boxes'] = results[0].boxes.xyxy.cpu().numpy()
            detection_data['confidences'] = results[0].boxes.conf.cpu().numpy()
            detection_data['class_ids'] = results[0].boxes.cls.cpu().numpy().astype(int)
        
        return detection_data
    
    
    def train(self, data: Union[str, Path], 
              epochs: int = 100, batch_size: int = 16,
              imgsz: int = 640, patience: int = 20,
              lr0: float = 0.001, momentum: float = 0.937,
              weight_decay: float = 0.0005,
              device: str = None, **kwargs) -> Dict:
        """
        Train the YOLOv8 model.
        
        Args:
            data (str/Path): Path to data.yaml file
            epochs (int): Number of training epochs. Default: 100
            batch_size (int): Batch size for training. Default: 16
            imgsz (int): Input image size. Default: 640
            patience (int): Early stopping patience. Default: 20
            lr0 (float): Initial learning rate. Default: 0.001
            momentum (float): Optimizer momentum. Default: 0.937
            weight_decay (float): Weight decay regularization. Default: 0.0005
            device (str): Device for training. Default: None (uses self.device)
            **kwargs: Additional arguments passed to model.train()
            
        Returns:
            Dict containing training results
        """
        
        device = device or self.device
        
        results = self.model.train(
            data=str(data),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            patience=patience,
            device=device,
            lr0=lr0,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs
        )
        
        return {
            'results': results,
            'epochs': epochs,
            'batch_size': batch_size,
            'imgsz': imgsz,
        }
    
    
    def validate(self, data: Union[str, Path] = None,
                imgsz: int = 640, batch_size: int = 16,
                device: str = None, **kwargs) -> Dict:
        """
        Validate the model on a dataset.
        
        Args:
            data (str/Path): Path to data.yaml file
            imgsz (int): Input image size. Default: 640
            batch_size (int): Batch size for validation. Default: 16
            device (str): Device for validation. Default: None (uses self.device)
            **kwargs: Additional arguments passed to model.val()
            
        Returns:
            Dict containing validation metrics
        """
        
        device = device or self.device
        
        results = self.model.val(
            data=str(data) if data else None,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            **kwargs
        )
        
        return {
            'results': results,
            'imgsz': imgsz,
            'batch_size': batch_size,
        }
    
    
    def save(self, save_dir: Path, suffix: Optional[str] = None) -> None:
        """
        Save the model weights.
        
        Args:
            save_dir (Path): Directory to save model
            suffix (str): Suffix to add to filename. Default: None
            
        Example:
            model.save(Path('checkpoints/'))
            # Saves to: checkpoints/best_model.pt
            
            model.save(Path('checkpoints/'), suffix='epoch_50')
            # Saves to: checkpoints/best_model_epoch_50.pt
        """


        save_dir = Path(save_dir)
        # Create model-specific subdirectory
        model_name = f'yolov8{self.model_size}'
        save_dir = save_dir / model_name

        save_dir.mkdir(parents=True, exist_ok=True)

        # Include model name in filename for clarity
        model_name = f'yolov8{self.model_size}'
        filename = f"best_model_{model_name}.pt" if suffix is None \
            else f"best_model_{model_name}_{suffix}.pt"
        filepath = save_dir / filename
        
        self.model.save(str(filepath))
        print(f"YOLOv8 model saved to {filepath}")
    
    
    def load(self, model_path: Union[str, Path]) -> None:
        """
        Load model weights from a saved checkpoint.
        
        Args:
            model_path (str/Path): Path to model weights file (.pt)
            
        Example:
            model.load('checkpoints/best_model.pt')
        """
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = YOLO(str(model_path))
        self.model.to(self.device)
        print(f"YOLOv8 model loaded from {model_path}")
    
    
    def get_model_info(self) -> Dict:
        """
        Get information about the current model.
        
        Returns:
            Dict with model metadata
        """
        
        return {
            'size': self.model_size,
            'device': self.device,
            'model_type': 'YOLOv8',
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
        }
    
    
    def to(self, device: str) -> 'YOLOv8':
        """
        Move model to specified device.
        
        Args:
            device (str): Device name ('cpu', 'cuda', etc.)
            
        Returns:
            Self for method chaining
        """
        self.device = device
        self.model.to(device)
        return self
    
    
    def eval(self) -> 'YOLOv8':
        """
        Set model to evaluation mode.
        
        Returns:
            Self for method chaining
        """
        self.model.eval()
        return self
    
    
    def train_mode(self) -> 'YOLOv8':
        """
        Set model to training mode.
        
        Returns:
            Self for method chaining
        """
        self.model.train()
        return self
