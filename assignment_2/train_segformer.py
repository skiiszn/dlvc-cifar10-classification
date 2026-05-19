import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from typing import Tuple

from assignment_2_code.models.segformer import SegFormer
from assignment_2_code.dataset.cityscapes import CityscapesCustom
from assignment_2_code.dataset.oxfordpets import OxfordPetsCustom
from assignment_2_code.metrics import SegMetrics
from assignment_2_code.trainer import ImgSemSegTrainer
from config import VIZ_DIR


class HistoryTrackingTrainer(ImgSemSegTrainer):
    """
    Wraps ImgSemSegTrainer to record every (epoch, loss, mIoU) tuple
    produced by _train_epoch and _val_epoch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_history = {"epoch": [], "loss": [], "mIoU": []}
        self.val_history = {"epoch": [], "loss": [], "mIoU": []}

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        loss, mIoU = super()._train_epoch(epoch_idx)
        self.train_history["epoch"].append(epoch_idx)
        self.train_history["loss"].append(float(loss))
        self.train_history["mIoU"].append(float(mIoU))
        return loss, mIoU

    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        loss, mIoU = super()._val_epoch(epoch_idx)
        self.val_history["epoch"].append(epoch_idx)
        self.val_history["loss"].append(float(loss))
        self.val_history["mIoU"].append(float(mIoU))
        return loss, mIoU


def train(args):

    train_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])

    train_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])
    
    val_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    val_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])

    if args.dataset == "oxford":
        train_data = OxfordPetsCustom(root=args.data_path,
                                split="trainval",
                                target_types='segmentation', 
                                transform=train_transform,
                                target_transform=train_transform2,
                                download=True)

        val_data = OxfordPetsCustom(root=args.data_path, 
                                split="test",
                                target_types='segmentation', 
                                transform=val_transform,
                                target_transform=val_transform2,
                                download=True)
    if args.dataset == "city":
        train_data = CityscapesCustom(root=args.data_path, 
                                split="train",
                                mode="fine",
                                target_type='semantic', 
                                transform=train_transform,
                                target_transform=train_transform2)
        val_data = CityscapesCustom(root=args.data_path, 
                                split="val",
                                mode="fine",
                                target_type='semantic', 
                                transform=val_transform,
                                target_transform=val_transform2)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.dataset == "city":
        model = SegFormer(num_classes=19)  # cityscapes has 19 classes
    else:
        model = SegFormer(num_classes=3)  # oxford has 3 classes
    
    # fine-tuning phase:
    if args.pretrained_path and args.pretrained_path != 'None':
        pretrained_dict = torch.load(args.pretrained_path, map_location='cpu')
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and 'encoder' in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

    if args.freeze_encoder:
        print(f"\n{'='*50}")
        print("FREEZING ENCODER - Only training decoder")
        print(f"{'='*50}")
        for param in model.encoder.parameters():
            param.requires_grad = False
    
    model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, amsgrad=True)
    
    if args.dataset == "city":
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    else:  # oxford
        loss_fn = torch.nn.CrossEntropyLoss()

    if isinstance(train_data.classes_seg, (list, tuple)):
        num_classes = len(train_data.classes_seg)
    else:
        num_classes = train_data.classes_seg
    
    print(f"Number of classes: {num_classes}")
    train_metric = SegMetrics(classes=num_classes)
    val_metric = SegMetrics(classes=num_classes)
    val_frequency = 2
    
    exp_suffix = ""
    if args.pretrained_path and args.pretrained_path != 'None':
        exp_suffix += "_pretrained"
    if args.freeze_encoder:
        exp_suffix += "_frozen"
    if args.lr != 0.001:
        exp_suffix += f"_lr{args.lr}"
    if not args.pretrained_path and not args.freeze_encoder and args.lr == 0.001 and args.dataset == "oxford":
        exp_suffix = "_scratch"
    
    model_save_dir = Path(f"saved_models/segformer_{args.dataset}{exp_suffix}")
    model_save_dir.mkdir(parents=True, exist_ok=True)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    trainer = HistoryTrackingTrainer(model, 
                    optimizer,
                    loss_fn,
                    lr_scheduler,
                    train_metric,
                    val_metric,
                    train_data,
                    val_data,
                    device,
                    args.num_epochs, 
                    model_save_dir,
                    batch_size=args.batch_size,
                    val_frequency=val_frequency)
    trainer.train()
    
    tag = f"segformer_{args.dataset}{exp_suffix}"
    history_dir = Path(VIZ_DIR).parent / "train_metrics_data" / tag
    history_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = history_dir / "history.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'phase', 'loss', 'mIoU'])
        
        for i in range(len(trainer.train_history['epoch'])):
            writer.writerow([
                trainer.train_history['epoch'][i],
                'train',
                trainer.train_history['loss'][i],
                trainer.train_history['mIoU'][i]
            ])
        
        for i in range(len(trainer.val_history['epoch'])):
            writer.writerow([
                trainer.val_history['epoch'][i],
                'val',
                trainer.val_history['loss'][i],
                trainer.val_history['mIoU'][i]
            ])
    
    viz_path = Path(VIZ_DIR) / tag / "training_metrics_curves.png"
    viz_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    train_epochs = trainer.train_history['epoch']
    train_losses = trainer.train_history['loss']
    val_epochs = trainer.val_history['epoch']
    val_losses = trainer.val_history['loss']
    
    ax1.plot(train_epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(val_epochs, val_losses, 'r-o', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    train_mious = trainer.train_history['mIoU']
    val_mious = trainer.val_history['mIoU']
    
    ax2.plot(train_epochs, train_mious, 'b-', label='Train mIoU', linewidth=2)
    ax2.plot(val_epochs, val_mious, 'r-o', label='Val mIoU', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mIoU')
    ax2.set_title('Training and Validation mIoU')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_path, dpi=150)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training SegFormer')
    parser.add_argument('-d', '--gpu_id', default='0', type=str,
                      help='index of which GPU to use')
    parser.add_argument('--data_path', default='path_to_dataset', type=str,
                      help='path to the dataset')
    parser.add_argument('--num_epochs', default=30, type=int,
                      help='number of epochs')
    parser.add_argument('--batch_size', default=64, type=int,
                      help='batch size')
    parser.add_argument('--dataset', default='oxford', type=str,
                      choices=['oxford', 'city'],
                      help='dataset to use')
    parser.add_argument('--pretrained_path', default=None, type=str,
                      help='path to pretrained model for fine-tuning (Part 6)')
    parser.add_argument('--freeze_encoder', action='store_true',
                      help='Freeze encoder during fine-tuning (Part 6c)')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate configuration')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    train(args)