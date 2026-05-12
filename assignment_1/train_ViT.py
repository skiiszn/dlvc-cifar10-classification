# Feel free to change the imports according to your implementation and needs
import argparse
import csv
import os
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torchvision.transforms.v2 as v2

from assignment_1_code.models.class_model import DeepClassifier
from assignment_1_code.models.vit import YourViT
from assignment_1_code.metrics import Accuracy
from assignment_1_code.trainer import ImgClassificationTrainer
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from assignment_1_code.viz import plot_training_curves
from config import DATA_DIR, MODEL_SAVE_DIR


# ImageNet normalization stats
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Weight-decay value used when regularization is enabled
WEIGHT_DECAY_ON = 5e-4
WEIGHT_DECAY_OFF = 0.0


# Optimizer
#   * AdamW: lr=1e-3, amsgrad=True               (reference)
#   * Adam : lr=1e-3                             (same lr -> isolates the AdamW-vs-Adam difference)
#   * SGD  : lr=1e-2, momentum=0.9               (SGD typically needs a ~10x larger lr than Adam-style methods)
#
OPTIMIZER_FACTORIES = {
    "adamw": lambda params, wd: torch.optim.AdamW(
        params, lr=1e-3, amsgrad=True, weight_decay=wd
    ),
    "adam": lambda params, wd: torch.optim.Adam(
        params, lr=1e-3, weight_decay=wd
    ),
    "sgd": lambda params, wd: torch.optim.SGD(
        params, lr=1e-2, momentum=0.9, weight_decay=wd
    ),
}



# LR scheduler
#   * cosine     : T_max=num_epochs               (default for ViT; one full cosine half-cycle from lr to 0 over the whole run)
#   * exponential: gamma=0.9                      (kept for parity with the ResNet pipeline)
#   * step       : step_size=10, gamma=0.1        (with 30 epochs total this gives drops at epochs 10 and 20)
#
SCHEDULER_FACTORIES = {
    "exponential": lambda opt, n: torch.optim.lr_scheduler.ExponentialLR(
        opt, gamma=0.9
    ),
    "step": lambda opt, n: torch.optim.lr_scheduler.StepLR(
        opt, step_size=10, gamma=0.1
    ),
    "cosine": lambda opt, n: torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n
    ),
}








# Short abbreviations used inside config tags / save folder names so paths
_OPT_ABBREV = {"adamw": "adamw", "adam": "adam", "sgd": "sgd"}
_SCHED_ABBREV = {"exponential": "exp", "step": "step", "cosine": "cos"}



def config_tag(
    use_augmentation: bool,
    use_regularization: bool,
    optimizer_name: str,
    scheduler_name: str,
) -> str:
    """
    Short identifier for a (aug, reg, optimizer, scheduler) configuration.
    """
    aug = "aug" if use_augmentation else "noaug"
    reg = "reg" if use_regularization else "noreg"
    return f"{aug}_{reg}_{_OPT_ABBREV[optimizer_name]}_{_SCHED_ABBREV[scheduler_name]}"





class HistoryTrackingTrainer(ImgClassificationTrainer):
    """
    Wraps ``ImgClassificationTrainer`` to record every (epoch, loss, mAcc,
    mPCAcc) tuple returned by ``_train_epoch`` / ``_val_epoch``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_history = {"epoch": [], "loss": [], "mAcc": [], "mPCAcc": []}
        self.val_history = {"epoch": [], "loss": [], "mAcc": [], "mPCAcc": []}

    def _train_epoch(self, epoch_idx):
        loss, mAcc, mPCAcc = super()._train_epoch(epoch_idx)
        self.train_history["epoch"].append(epoch_idx)
        self.train_history["loss"].append(loss)
        self.train_history["mAcc"].append(mAcc)
        self.train_history["mPCAcc"].append(mPCAcc)
        return loss, mAcc, mPCAcc

    def _val_epoch(self, epoch_idx):
        loss, mAcc, mPCAcc = super()._val_epoch(epoch_idx)
        self.val_history["epoch"].append(epoch_idx)
        self.val_history["loss"].append(loss)
        self.val_history["mAcc"].append(mAcc)
        self.val_history["mPCAcc"].append(mPCAcc)
        return loss, mAcc, mPCAcc


def _save_run_artifacts(trainer: HistoryTrackingTrainer, tag: str) -> None:
    """
    After a finished training run, persist the per-epoch history to csv and
    render the training-curves plot.
    """

    history_dir = Path("results") / "train_metrics_data_ViT" / tag
    history_dir.mkdir(parents=True, exist_ok=True)
    csv_path = history_dir / "history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "epoch", "loss", "mAcc", "mPCAcc"])
        for i in range(len(trainer.train_history["epoch"])):
            writer.writerow(
                [
                    "train",
                    trainer.train_history["epoch"][i],
                    trainer.train_history["loss"][i],
                    trainer.train_history["mAcc"][i],
                    trainer.train_history["mPCAcc"][i],
                ]
            )
        for i in range(len(trainer.val_history["epoch"])):
            writer.writerow(
                [
                    "val",
                    trainer.val_history["epoch"][i],
                    trainer.val_history["loss"][i],
                    trainer.val_history["mAcc"][i],
                    trainer.val_history["mPCAcc"][i],
                ]
            )
    print(f"Saved per-epoch metrics to {csv_path}")

    viz_path = Path("results") / "viz_ViT" / tag / "training_metrics_curves.png"
    fig = plot_training_curves(
        train_history=trainer.train_history,
        val_history=trainer.val_history,
        save_path=viz_path,
        label="ViT",
    )
    plt.close(fig)
    print(f"Saved training curves to {viz_path}")









def build_train_transform(use_augmentation: bool) -> v2.Compose:
    """
    Build the training-time transform pipeline.

    With ``use_augmentation=True`` we add the two augmentations:
        * RandomCrop(32, padding=4)         -- random crops
        * RandomHorizontalFlip(p=0.5)       -- left/right mirroring

    With ``use_augmentation=False`` only the deterministic preprocessing is
    applied (type-cast to float32 in [0, 1] + ImageNet normalization)
    """
    transforms = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    if use_augmentation:
        transforms.extend(
            [
                v2.RandomCrop(size=32, padding=4),
                v2.RandomHorizontalFlip(p=0.5),
            ]
        )
    transforms.append(v2.Normalize(mean=NORM_MEAN, std=NORM_STD))
    return v2.Compose(transforms)


def build_eval_transform() -> v2.Compose:
    """Validation/test transform: deterministic, no augmentation."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]
    )








def train(
    args,
    use_augmentation: bool,
    use_regularization: bool,
    optimizer_name: str = "adamw",
    scheduler_name: str = "cosine",
) -> None:
    """
    Run a single training configuration.

    Args:
        args: parsed CLI args (uses ``args.num_epochs``).
        use_augmentation: if True, apply RandomCrop + HorizontalFlip on the
            training set.
        use_regularization: if True, set the optimizer's ``weight_decay`` to
            ``WEIGHT_DECAY_ON``; otherwise to ``WEIGHT_DECAY_OFF``.
        optimizer_name: key into ``OPTIMIZER_FACTORIES``.
        scheduler_name: key into ``SCHEDULER_FACTORIES``.
    """

    tag = config_tag(
        use_augmentation, use_regularization, optimizer_name, scheduler_name
    )
    print(
        f"Configuration: aug={use_augmentation}, reg={use_regularization}, "
        f"optimizer={optimizer_name}, scheduler={scheduler_name}  (tag: {tag})"
    )

    train_transform = build_train_transform(use_augmentation)
    val_transform = build_eval_transform()

    # Use config.py for all machine-dependent paths, e.g. DATA_DIR.
    train_data = CIFAR10Dataset(DATA_DIR, subset=Subset.TRAINING, transform=train_transform)

    # Validation data: batch 5
    val_data = CIFAR10Dataset(DATA_DIR, subset=Subset.VALIDATION, transform=val_transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model: ViT defaults: image_size=32, patch_size=4, dim=256, depth=6, heads=8, mlp_dim=512
    model = DeepClassifier(YourViT(num_classes=10))
    model.to(device)
    model_name = type(model.net).__name__

    # Regularization
    weight_decay = WEIGHT_DECAY_ON if use_regularization else WEIGHT_DECAY_OFF
    # different optimizers
    optimizer = OPTIMIZER_FACTORIES[optimizer_name](
        model.parameters(), weight_decay
    )
    print(
        f"Optimizer: {type(optimizer).__name__} "
        f"(lr={optimizer.param_groups[0]['lr']}, "
        f"weight_decay={optimizer.param_groups[0]['weight_decay']})"
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)
    val_frequency = 5

    model_save_dir = Path(MODEL_SAVE_DIR) / "checkpoints_ViT" /tag
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # different learning rate schedulers
    lr_scheduler = SCHEDULER_FACTORIES[scheduler_name](optimizer, args.num_epochs)
    print(f"LR scheduler: {type(lr_scheduler).__name__}")

    trainer = HistoryTrackingTrainer(  # ImgClassificationTrainer(
        model,
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
        batch_size=args.batch_size,  # feel free to change
        val_frequency=val_frequency,
    )
    trainer.train()

    _save_run_artifacts(trainer, tag)


def aug_reg_pipeline_configs():
    """The four (aug, reg) combinations on the default optimizer+scheduler."""
    return [
        (False, False, "adamw", "cosine"),  # 1. baseline
        (True,  False, "adamw", "cosine"),  # 2. augmentation only
        (False, True,  "adamw", "cosine"),  # 3. regularization only
        (True,  True,  "adamw", "cosine"),  # 4. both
    ]


def optimizer_sweep_configs():
    """Optimizer comparison on the baseline (no aug, no reg)."""
    return [(False, False, opt, "cosine")
            for opt in ["adamw", "adam", "sgd"]]


def scheduler_sweep_configs():
    """Scheduler comparison on the baseline (no aug, no reg)."""
    return [(False, False, "adamw", sched)
            for sched in ["cosine", "exponential", "step"]]


def run_configs(args, configs) -> None:
    """Run a list of (use_aug, use_reg, optimizer, scheduler) configs."""
    seen = set()
    for cfg in configs:
        if cfg in seen:
            continue  # skip duplicates when modes are unioned
        seen.add(cfg)
        train(args, *cfg)


# Mapping for the --combo CLI option.
_COMBO_MAP = {
    "baseline": (False, False),
    "aug":      (True,  False),
    "reg":      (False, True),
    "both":     (True,  True),
}


if __name__ == "__main__":
    # Feel free to change this part - you do not have to use this argparse and gpu handling
    parser = argparse.ArgumentParser(description="Training (Vision Transformer)")
    parser.add_argument(
        "-d", "--gpu_id", default="0", type=str, help="index of which GPU to use"
    )
    parser.add_argument(
        "--mode",
        choices=["aug_reg", "optimizer", "scheduler", "all"],
        default="aug_reg",
        help=(
            "Which pipeline to run. "
            "'aug_reg' = 4-combo augmentation/regularization grid (default); "
            "'optimizer' = optimizer change to baseline; "
            "'scheduler' = scheduler change to baseline; "
            "'all' = union of the above (deduped); "
        ),
    )
    parser.add_argument(
        "--num_epochs",
        default=30,
        type=int,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="batch size"
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.gpu_id = 0

    if args.mode == "aug_reg":
        run_configs(args, aug_reg_pipeline_configs())
    elif args.mode == "optimizer":
        run_configs(args, optimizer_sweep_configs())
    elif args.mode == "scheduler":
        run_configs(args, scheduler_sweep_configs())
    elif args.mode == "all":
        run_configs(
            args,
            aug_reg_pipeline_configs()
            + optimizer_sweep_configs()
            + scheduler_sweep_configs(),
        )
