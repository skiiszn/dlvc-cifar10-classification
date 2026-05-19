
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

from assignment_2_code.models.segment_model import DeepSegmenter
from assignment_2_code.dataset.oxfordpets import OxfordPetsCustom
from assignment_2_code.metrics import SegMetrics
from assignment_2_code.trainer import ImgSemSegTrainer

from config import OXFORD_PETS_DIR, MODEL_SAVE_DIR, VIZ_DIR


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



_RUN_STYLE = {
    "fcn_pretrained": {"label": "FCN",              "color": "purple"},  # purple
    "fcn_scratch":    {"label": "FCN_from-scratch", "color": "green"},  # green
}


def plot_training_curves(runs: list, save_path) -> None:
    """
    Create a 2x2 training-curves visualization.
    Args:
        runs: list of run dicts, each with keys
              "tag", "train_history", "val_history".
              Pass a single-element list for an individual plot, or
              multiple elements to overlay them in one comparison figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (axis, panel title, phase, metric key) — order matches image_FCN.png.
    panels = [
        (axes[0, 0], "val/loss", "val_history", "loss"),
        (axes[0, 1], "val/mIoU", "val_history", "mIoU"),
        (axes[1, 0], "train/mIoU", "train_history", "mIoU"),
        (axes[1, 1], "train/loss", "train_history", "loss"),
    ]

    for ax, title, phase_key, metric in panels:
        for run in runs:
            style = _RUN_STYLE.get(
                run["tag"], {"label": run["tag"], "color": None}
            )
            hist = run[phase_key]
            ax.plot(
                hist["epoch"], hist[metric],
                color=style["color"], linewidth=2.0, label=style["label"],
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper center", ncol=len(runs), frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves to {save_path}")


def _save_history_csv(run: dict) -> None:
    """Persist one run's per-epoch history to CSV."""
    tag = run["tag"]
    history_dir = Path(VIZ_DIR) / "train_metrics_data" / tag
    history_dir.mkdir(parents=True, exist_ok=True)

    csv_path = history_dir / "history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "epoch", "loss", "mIoU"])
        for phase, key in (("train", "train_history"), ("val", "val_history")):
            hist = run[key]
            for i in range(len(hist["epoch"])):
                writer.writerow([
                    phase, hist["epoch"][i], hist["loss"][i], hist["mIoU"][i],
                ])
    print(f"Saved per-epoch metrics to {csv_path}")


def _save_run_artifacts(trainer: HistoryTrackingTrainer, tag: str) -> dict:
    run = {
        "tag": tag,
        "train_history": trainer.train_history,
        "val_history": trainer.val_history,
    }
    _save_history_csv(run)

    viz_path = Path(VIZ_DIR) / "viz" / tag / "training_metrics_curves.png"
    plot_training_curves([run], viz_path)
    return run







def train(args, pretrained: bool):

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

    num_classes = len(train_data.classes_seg)

    # Pre-trained backbone (second run) vs. from scratch (first run).
    backbone_weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None


    fcn = fcn_resnet50(
        weights=None,
        weights_backbone=backbone_weights,
        num_classes=num_classes,
        aux_loss=False,
    )

    tag = "FCN_pretrained" if pretrained else "FCN_from_scratch"
    print(f"\nConfiguration: pretrained={pretrained}  (tag: {tag})")

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
                    model_save_dir,
                    batch_size=64,
                    val_frequency = val_frequency)

    trainer.train()

    run = _save_run_artifacts(trainer, tag)

    # see Reference implementation of ImgSemSegTrainer
    # just comment if not used
    #trainer.dispose()

    return run

if __name__ == "__main__":
    #args = argparse.ArgumentParser(description='Training')
    #args.add_argument('-d', '--gpu_id', default='0', type=str, help='index of which GPU to use')

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-d', '--gpu_id', default='0', type=str,
                        help='index of which GPU to use')
    #parser.add_argument('--pretrained', action='store_true',
    #                    help='Use ImageNet-pretrained ResNet50 backbone')
    parser.add_argument('--mode', choices=['scratch', 'pretrained', 'both'],
                        default='both',
                        help="which run(s) to do: 'scratch', 'pretrained', "
                             "or 'both' (scratch first, then pretrained) ---"
                             "Use ImageNet-pretrained ResNet50 backbone")
    
    #if not isinstance(args, tuple):
    #    args = args.parse_args()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 31

    if args.mode == 'scratch':
        runs = [False]
    elif args.mode == 'pretrained':
        runs = [True]
    else:  # 'both'
        runs = [False, True]

    completed_runs = []
    for pretrained in runs:
        completed_runs.append(train(args, pretrained=pretrained))

    if len(completed_runs) > 1:
        combined_path = Path(VIZ_DIR) / "viz" / "fcn_comparison.png"
        plot_training_curves(completed_runs, combined_path)
