
import argparse
import os
import csv
import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt

from torchvision.models.segmentation import fcn_resnet50
from torchvision.models import ResNet50_Weights

from assignment_2.assignment_2_code.models.segment_model import DeepSegmenter
from assignment_2.assignment_2_code.dataset.oxfordpets import OxfordPetsCustom
from assignment_2.assignment_2_code.metrics import SegMetrics
from assignment_2.assignment_2_code.trainer import ImgSemSegTrainer

from config import OXFORD_PETS_DIR, MODEL_SAVE_DIR, LOG_DIR


class HistoryTrackingTrainer(ImgSemSegTrainer):
    """
    Wraps ``ImgSemSegTrainer`` to record every (epoch, loss, mIoU) tuple
    produced by ``_train_epoch`` / ``_val_epoch``.
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


def _save_run_artifacts(trainer: HistoryTrackingTrainer, tag: str) -> None:

    history_dir = Path(LOG_DIR) / "train_metrics_data" / tag
    history_dir.mkdir(parents=True, exist_ok=True)

    csv_path = history_dir / "history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "epoch", "loss", "mIoU"])
        for i in range(len(trainer.train_history["epoch"])):
            writer.writerow([
                "train",
                trainer.train_history["epoch"][i],
                trainer.train_history["loss"][i],
                trainer.train_history["mIoU"][i],
            ])
        for i in range(len(trainer.val_history["epoch"])):
            writer.writerow([
                "val",
                trainer.val_history["epoch"][i],
                trainer.val_history["loss"][i],
                trainer.val_history["mIoU"][i],
            ])
    print(f"Saved per-epoch metrics to {csv_path}")

    viz_path = Path(LOG_DIR) / "viz" / tag / "training_metrics_curves.png"
    fig = _plot_run_curves(
        train_history=trainer.train_history,
        val_history=trainer.val_history,
        save_path=viz_path,
        label=tag,
    )
    plt.close(fig)
    print(f"Saved training curves to {viz_path}")


def _plot_run_curves(train_history: dict, val_history: dict,
                     save_path: Path, label: str):
    """
    Render a 2x2 viz (val/loss, val/mIoU, train/loss, train/mIoU) for a
    single run. Combining multiple runs into one comparison figure (as in
    image_FCN.png) is left for a separate plotting script that reads the CSVs.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (ax_vl, ax_vm), (ax_tl, ax_tm) = axes

    ax_vl.plot(val_history["epoch"], val_history["loss"], marker="o", label=label)
    ax_vl.set_title("val/loss");
    ax_vl.set_xlabel("epoch");
    ax_vl.set_ylabel("loss");
    ax_vl.legend()

    ax_vm.plot(val_history["epoch"], val_history["mIoU"], marker="o", label=label)
    ax_vm.set_title("val/mIoU");
    ax_vm.set_xlabel("epoch");
    ax_vm.set_ylabel("mIoU");
    ax_vm.legend()

    ax_tl.plot(train_history["epoch"], train_history["loss"], marker="o", label=label)
    ax_tl.set_title("train/loss");
    ax_tl.set_xlabel("epoch");
    ax_tl.set_ylabel("loss");
    ax_tl.legend()

    ax_tm.plot(train_history["epoch"], train_history["mIoU"], marker="o", label=label)
    ax_tm.set_title("train/mIoU");
    ax_tm.set_xlabel("epoch");
    ax_tm.set_ylabel("mIoU");
    ax_tm.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    return fig






def train(args):

    train_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    train_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])#,
    
    val_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    val_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])

    train_data = OxfordPetsCustom(root=str(OXFORD_PETS_DIR),
                            split="trainval",
                            target_types='segmentation', 
                            transform=train_transform,
                            target_transform=train_transform2,
                            download=True)

    val_data = OxfordPetsCustom(root=str(OXFORD_PETS_DIR),
                            split="test",
                            target_types='segmentation', 
                            transform=val_transform,
                            target_transform=val_transform2,
                            download=True)



    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    num_classes = len(train_data.classes_seg)  # pet, background, border -> 3

    # Pre-trained backbone (Part 4, second run) vs. fully from scratch (first run).
    # We do NOT use `weights=...` because the prediction head has a different
    # number of output classes than the dataset the FCN was pretrained on.
    backbone_weights = ResNet50_Weights.IMAGENET1K_V1 if args.pretrained else None


    fcn = fcn_resnet50(
        weights=None,
        weights_backbone=backbone_weights,
        num_classes=num_classes,
        aux_loss=False,
    )

    tag = "fcn_pretrained" if args.pretrained else "fcn_scratch"
    print(f"\nConfiguration: pretrained={args.pretrained}  (tag: {tag})")

    model = DeepSegmenter(fcn).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, amsgrad=True)
    loss_fn = nn.CrossEntropyLoss()
    
    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = 2

    model_save_dir = Path(MODEL_SAVE_DIR) / tag
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
                    MODEL_SAVE_DIR,
                    batch_size=64,
                    val_frequency = val_frequency)
    trainer.train()
    _save_run_artifacts(trainer, tag)

    # see Reference implementation of ImgSemSegTrainer
    # just comment if not used
    #trainer.dispose()

if __name__ == "__main__":
    #args = argparse.ArgumentParser(description='Training')
    #args.add_argument('-d', '--gpu_id', default='0', type=str, help='index of which GPU to use')

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-d', '--gpu_id', default='0', type=str,
                        help='index of which GPU to use')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use ImageNet-pretrained ResNet50 backbone')
    
    #if not isinstance(args, tuple):
    #    args = args.parse_args()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 31


    train(args)