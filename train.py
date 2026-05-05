# Feel free to change the imports according to your implementation and needs
import argparse
import os
import csv
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import matplotlib.pyplot as plt
import os

from assignment_1_code.models.class_model import (
    DeepClassifier,
)  # etc. change to your model
from assignment_1_code.metrics import Accuracy
from assignment_1_code.trainer import ImgClassificationTrainer
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from assignment_1_code.viz import plot_training_curves
from config import DATA_DIR, MODEL_SAVE_DIR

from torchvision.models import resnet18


# ImageNet normalization stats
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

# Weight-decay value used when regularization is enabled
WEIGHT_DECAY_ON  = 5e-4
WEIGHT_DECAY_OFF = 0.0

# Early-stopping patience
EARLY_STOPPING_PATIENCE = 2


# Optimizer
#   * AdamW: lr=1e-3, amsgrad=True               (the reference recipe)
#   * Adam : lr=1e-3                             (same lr -> isolates the
#                                                 AdamW-vs-Adam difference)
#   * SGD  : lr=1e-2, momentum=0.9               (SGD typically needs a ~10x
#                                                 larger lr than Adam-style
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
#   * exponential: gamma=0.9                 (the reference recipe; lr decays
#                                             smoothly each epoch)
#   * step       : step_size=10, gamma=0.1   (with 30 epochs total this gives
#                                             drops at epochs 10 and 20)
#   * cosine     : T_max=num_epochs          (one full cosine half-cycle from
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


# Short abbreviations used inside config tags / save folder names
_OPT_ABBREV   = {"adamw": "adamw", "adam": "adam", "sgd": "sgd"}
_SCHED_ABBREV = {"exponential": "exp", "step": "step", "cosine": "cos"}


def config_tag(
    use_augmentation:    bool,
    use_regularization:  bool,
    optimizer_name:      str,
    scheduler_name:      str,
    use_early_stopping:  bool,
) -> str:
    """
    Short identifier for one (aug, reg, optimizer, scheduler, es) configuration.
    Used as the sub-folder name under MODEL_SAVE_DIR so pipeline runs do not
    overwrite each other's checkpoints.
    """
    aug  = "aug"  if use_augmentation   else "noaug"
    reg  = "reg"  if use_regularization else "noreg"
    es   = "es"   if use_early_stopping else "noes"
    return f"{aug}_{reg}_{_OPT_ABBREV[optimizer_name]}_{_SCHED_ABBREV[scheduler_name]}_{es}"








class HistoryTrackingTrainer(ImgClassificationTrainer):
    """
    Wraps ``ImgClassificationTrainer`` to record every (epoch, loss, mAcc,
    mPCAcc) tuple returned by ``_train_epoch`` / ``_val_epoch``.
    """

    def __init__(self, *args, use_early_stopping: bool = False,
                 patience: int = EARLY_STOPPING_PATIENCE, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_history = {"epoch": [], "loss": [], "mAcc": [], "mPCAcc": []}
        self.val_history   = {"epoch": [], "loss": [], "mAcc": [], "mPCAcc": []}
        # Early-stopping state
        self.use_early_stopping = use_early_stopping
        self.patience           = patience
        self._es_counter        = 0          # consecutive val checks w/o improvement
        self._es_best_mPCAcc    = -1.0       # best val mPCAcc seen so far
        self.stopped_early      = False      # flag readable after training


    def _train_epoch(self, epoch_idx: int):
        loss, mAcc, mPCAcc = super()._train_epoch(epoch_idx)
        self.train_history["epoch"].append(epoch_idx)
        self.train_history["loss"].append(loss)
        self.train_history["mAcc"].append(mAcc)
        self.train_history["mPCAcc"].append(mPCAcc)
        return loss, mAcc, mPCAcc

    def _val_epoch(self, epoch_idx: int):
        loss, mAcc, mPCAcc = super()._val_epoch(epoch_idx)
        self.val_history["epoch"].append(epoch_idx)
        self.val_history["loss"].append(loss)
        self.val_history["mAcc"].append(mAcc)
        self.val_history["mPCAcc"].append(mPCAcc)
        return loss, mAcc, mPCAcc

    def train(self) -> None:
        if not self.use_early_stopping:
            # Delegate entirely to the base class — no overhead.
            super().train()
            return

        best_val_mPCAcc = -1.0

        for epoch_idx in range(self.num_epochs):
            self._train_epoch(epoch_idx)

            if (epoch_idx + 1) % self.val_frequency == 0:
                _, _, val_mPCAcc = self._val_epoch(epoch_idx)

                # Mirror the base-trainer checkpoint logic
                if val_mPCAcc > best_val_mPCAcc:
                    best_val_mPCAcc = val_mPCAcc
                    self.model.save(self.training_save_dir)

                # Early-stopping check
                if val_mPCAcc > self._es_best_mPCAcc:
                    self._es_best_mPCAcc = val_mPCAcc
                    self._es_counter = 0
                else:
                    self._es_counter += 1
                    print(
                        f"[EarlyStopping] No improvement for {self._es_counter}/"
                        f"{self.patience} validation checks "
                        f"(best val mPCAcc: {self._es_best_mPCAcc:.4f})"
                    )
                    if self._es_counter >= self.patience:
                        print(
                            f"[EarlyStopping] Stopping at epoch {epoch_idx} — "
                            f"no val mPCAcc improvement for {self.patience} "
                            f"consecutive validation checks."
                        )
                        self.stopped_early = True
                        break

            # Step the LR scheduler once per epoch (same as base class)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()



def _save_run_artifacts(trainer: HistoryTrackingTrainer, tag: str) -> None:
    """
    After a finished training run, persist the per-epoch history to CSV and
    render the training-curves dashboard.
    """
    history_dir = Path("results") / "train_metrics_data" / tag
    history_dir.mkdir(parents=True, exist_ok=True)
    csv_path = history_dir / "history.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "epoch", "loss", "mAcc", "mPCAcc"])
        for i in range(len(trainer.train_history["epoch"])):
            writer.writerow([
                "train",
                trainer.train_history["epoch"][i],
                trainer.train_history["loss"][i],
                trainer.train_history["mAcc"][i],
                trainer.train_history["mPCAcc"][i],
            ])
        for i in range(len(trainer.val_history["epoch"])):
            writer.writerow([
                "val",
                trainer.val_history["epoch"][i],
                trainer.val_history["loss"][i],
                trainer.val_history["mAcc"][i],
                trainer.val_history["mPCAcc"][i],
            ])

    print(f"Saved per-epoch metrics to {csv_path}")

    viz_path = Path("results") / "viz" / tag / "training_metrics_curves.png"
    fig = plot_training_curves(
        train_history=trainer.train_history,
        val_history=trainer.val_history,
        save_path=viz_path,
        label="ResNet",
    )
    plt.close(fig)
    print(f"Saved training curves to {viz_path}")







def build_train_transform(use_augmentation: bool) -> v2.Compose:
    """
    Build the training-time transform pipeline.

    With ``use_augmentation=True`` we add:
        * RandomCrop(32, padding=4)         -- random crops
        * RandomHorizontalFlip(p=0.5)       -- left/right mirroring

    With ``use_augmentation=False`` only the deterministic preprocessing is
    applied (type-cast to float32 in [0, 1] + ImageNet normalization).
    """
    transforms = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    if use_augmentation:
        transforms.extend([
            v2.RandomCrop(size=32, padding=4),
            v2.RandomHorizontalFlip(p=0.5),
        ])
    transforms.append(v2.Normalize(mean=NORM_MEAN, std=NORM_STD))
    return v2.Compose(transforms)


def build_eval_transform() -> v2.Compose:
    """Validation/test transform: deterministic, no augmentation."""
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])



def train(
    args,
    use_augmentation:   bool,
    use_regularization: bool,
    optimizer_name:     str  = "adamw",
    scheduler_name:     str  = "exponential",
    use_early_stopping: bool = False,
) -> None:

    tag = config_tag(
        use_augmentation, use_regularization,
        optimizer_name, scheduler_name, use_early_stopping,
    )
    print(
        f"\nConfiguration: aug={use_augmentation}, reg={use_regularization}, "
        f"optimizer={optimizer_name}, scheduler={scheduler_name}, "
        f"early_stopping={use_early_stopping}  (tag: {tag})"
    )

    train_transform = build_train_transform(use_augmentation)
    val_transform   = build_eval_transform()

    # Use config.py for all machine-dependent paths, e.g. DATA_DIR.
    train_data = CIFAR10Dataset(DATA_DIR, subset=Subset.TRAINING, transform=train_transform)

    # Validation data: batch 5
    val_data = CIFAR10Dataset(DATA_DIR, subset=Subset.VALIDATION, transform=val_transform)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DeepClassifier(resnet18(num_classes=10))
    model.to(device)

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

    model_save_dir = Path(MODEL_SAVE_DIR) / tag         # maybe needs additional path; be aware!!!!
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # different learning rate schedulers
    lr_scheduler = SCHEDULER_FACTORIES[scheduler_name](optimizer, args.num_epochs)
    print(f"LR scheduler: {type(lr_scheduler).__name__}")

    # early stopping
    if use_early_stopping:
        print(
            f"Early stopping: on  (patience={EARLY_STOPPING_PATIENCE} val checks "
            f"= {EARLY_STOPPING_PATIENCE * val_frequency} epochs)"
        )
    else:
        print("Early stopping: off")

    trainer = HistoryTrackingTrainer(
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
        batch_size=args.batch_size,
        val_frequency=val_frequency,
        use_early_stopping=use_early_stopping,
        patience=EARLY_STOPPING_PATIENCE,
    )
    trainer.train()

    if trainer.stopped_early:
        print(
            f"[EarlyStopping] Training ended at epoch "
            f"{trainer.train_history['epoch'][-1]} out of {args.num_epochs}."
        )

    _save_run_artifacts(trainer, tag)



def aug_reg_pipeline_configs(use_early_stopping: bool = False):
    """
    The (aug x reg) 4-combo pipeline with the reference optimizer + scheduler and early stopping off.
    """
    es = use_early_stopping
    return [
        (False, False, "adamw", "exponential", es),   # baseline
        (True,  False, "adamw", "exponential", es),   # aug only
        (False, True,  "adamw", "exponential", es),   # reg only
        (True,  True,  "adamw", "exponential", es),   # both
    ]


def optimizer_change_configs(use_early_stopping: bool = False):
    """
    Sweeps optimizers (AdamW, Adam, SGD) on baseline settings.
    """
    es = use_early_stopping
    return [
        (False, False, opt_name, "exponential", es)
        for opt_name in OPTIMIZER_FACTORIES.keys()
    ]


def scheduler_change_configs(use_early_stopping: bool = False):
    """
    Sweeps LR schedulers (Exponential, Step, Cosine) on baseline settings.
    """
    es = use_early_stopping
    return [
        (False, False, "adamw", sched_name, es)
        for sched_name in SCHEDULER_FACTORIES.keys()
    ]


def early_stopping_configs():
    """
    Compares early stopping ON vs OFF on baseline settings.
    """
    return [
        (False, False, "adamw", "exponential", False),   # no early stopping
        (False, False, "adamw", "exponential", True),    # with early stopping
    ]


def run_configs(args, configs) -> None:
    """
    Run each unique config tuple exactly once (deduplication via a set).
    """
    seen = set()
    for cfg in configs:
        if cfg in seen:
            continue
        seen.add(cfg)
        train(args, *cfg)







if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "-d", "--gpu_id", default="0", type=str,
        help="index of which GPU to use",
    )
    parser.add_argument(
        "--mode",
        choices=["aug_reg", "optimizer", "scheduler", "early_stopping", "all"],
        default="aug_reg",
        help=(
            "Which experiment to run:\n"
            "  'aug_reg'       = 4-combo aug/reg pipeline (AdamW + ExponentialLR);\n"
            "  'optimizer'     = change optimizers on baseline settings;\n"
            "  'scheduler'     = change schedulers on baseline settings;\n"
            "  'early_stopping'= compare ES-on vs ES-off on baseline settings;\n"
            "  'all'           = all of the above, deduplicated."
        ),
    )
    parser.add_argument(
        "--early_stopping", action="store_true",
        help=(
            "early_stopping (default: False)"
        ),
    )
    parser.add_argument(
        "--num_epochs", default=30, type=int,
        help="number of epochs",
    )
    parser.add_argument(
        "--batch_size", default=128, type=int,
        help="batch size",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.gpu_id = 0

    es = args.early_stopping

    if args.mode == "aug_reg":
        configs = aug_reg_pipeline_configs(use_early_stopping=False)
        if es:
            configs += aug_reg_pipeline_configs(use_early_stopping=True)
        run_configs(args, configs)

    elif args.mode == "optimizer":
        configs = optimizer_change_configs(use_early_stopping=False)
        if es:
            configs += optimizer_change_configs(use_early_stopping=True)
        run_configs(args, configs)

    elif args.mode == "scheduler":
        configs = scheduler_change_configs(use_early_stopping=False)
        if es:
            configs += scheduler_change_configs(use_early_stopping=True)
        run_configs(args, configs)

    elif args.mode == "early_stopping":
        run_configs(args, early_stopping_configs())

    elif args.mode == "all":
        configs = (
            aug_reg_pipeline_configs(use_early_stopping=False)
            + optimizer_change_configs(use_early_stopping=False)
            + scheduler_change_configs(use_early_stopping=False)
        )
        if es:
            configs += (
                aug_reg_pipeline_configs(use_early_stopping=True)
                + optimizer_change_configs(use_early_stopping=True)
                + scheduler_change_configs(use_early_stopping=True)
            )
        configs += early_stopping_configs()
        run_configs(args, configs)
