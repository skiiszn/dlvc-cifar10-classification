import torch
import torchvision
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
import numpy as np

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence


from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from config import DATA_DIR, VIZ_DIR


def imshow(img, filename="train_imgs.png"):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    VIZ_DIR.mkdir(parents=True, exist_ok=True) # ensure the visualization output directory exists
    save_path = VIZ_DIR/filename
    plt.imsave(save_path, np.transpose(npimg, (1, 2, 0)))
    # plt.imsave("test_1.png", np.transpose(npimg, (1, 2, 0)))


_TRAIN_PANELS = [
    ("train/loss", "loss"),
    ("train/mClassAcc", "mPCAcc"),
    ("train/mAcc", "mAcc"),
]
_VAL_PANELS = [
    ("val/mAcc", "mAcc"),
    ("val/mClassAcc", "mPCAcc"),
    ("val/loss", "loss"),
]


def plot_training_curves(
        train_history: Dict[str, Sequence],
        val_history: Dict[str, Sequence],
        save_path: Optional[Path] = None,
        label: str = "ResNet",
        color: str = "tab:orange",
        show: bool = False,
):
    """
    Builds training-curve dashboard.

    Args:
        train_history: dict with keys ``epoch``, ``loss``, ``mAcc``,
            ``mPCAcc``. Each value is a sequence with one entry per training
            epoch.
        val_history:   dict with the same keys, but with one entry per
            validation epoch (typically every ``val_frequency`` epochs).
        save_path: if given, save the figure here. Parent directories are
            created automatically. Should end in ``.png`` (or any other
            extension matplotlib understands).
        label: legend entry (default ``"ResNet"``). Override to e.g.
            ``"YourCNN"`` or ``"ViT"`` for the other models.
        color: line color. Default ``"tab:orange"`` matches wandb's defaults.
        show: if True, call ``plt.show()`` after saving.

    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    def _draw(ax, x, y, title):
        # Marker on every point
        ax.plot(
            x, y,
            label=label,
            color=color,
            marker="o",
            markersize=3,
            linewidth=1.5,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Step")
        ax.legend(loc="upper center", frameon=False)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.02, y=0.05)

    for ax, (title, key) in zip(axes[0], _TRAIN_PANELS):
        _draw(ax, train_history["epoch"], train_history[key], title)
    for ax, (title, key) in zip(axes[1], _VAL_PANELS):
        _draw(ax, val_history["epoch"], val_history[key], title)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def load_history_csv(path) -> Dict[str, Dict[str, List]]:
    """
    Load a per-epoch history CSV produced by ``train.py``.
    """
    train = {"epoch": [], "loss": [], "mAcc": [], "mPCAcc": []}
    val = {"epoch": [], "loss": [], "mAcc": [], "mPCAcc": []}

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bucket = train if row["phase"] == "train" else val
            bucket["epoch"].append(int(row["epoch"]))
            bucket["loss"].append(float(row["loss"]))
            bucket["mAcc"].append(float(row["mAcc"]))
            bucket["mPCAcc"].append(float(row["mPCAcc"]))

    return {"train": train, "val": val}


def replot_all_histories(
        train_data_root: Path = Path("results/train_data"),
        viz_root: Path = Path("results/viz"),
        label: str = "ResNet",
) -> None:

    if not train_data_root.is_dir():
        return
    for csv_path in sorted(train_data_root.glob("*/history.csv")):
        tag = csv_path.parent.name
        history = load_history_csv(csv_path)
        out_path = viz_root / tag / "training_curves.png"
        fig = plot_training_curves(
            history["train"], history["val"], save_path=out_path, label=label
        )
        plt.close(fig)
        print(f"plotted {tag} -> {out_path}")






if __name__ == "__main__":
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    train_data = CIFAR10Dataset(
        fdir="../data/cifar-10-batches-py", subset=Subset.TRAINING, transform=transform
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_data, batch_size=8, shuffle=False, num_workers=2
    )

    # get some random training images
    dataiter = iter(train_data_loader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(" ".join(f"{classes[labels[j]]:5s}" for j in range(8)))

    replot_all_histories()
