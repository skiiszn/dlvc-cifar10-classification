"""
Assignment 3 - Brain Tumor Detection
train.py: trains YOLOv5m, YOLOv8m, YOLOv11m sequentially.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
CODE_DIR    = BASE_DIR / "assignment_3_code"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(BASE_DIR))

from models.model_yolov5  import YOLOv5
from models.model_yolov8  import YOLOv8
from models.model_yolov11 import YOLOv11

from trainer_yolov5  import YOLOv5Trainer
from trainer_yolov8  import YOLOv8Trainer
from trainer_yolov11 import YOLOv11Trainer

CHECKPOINTS = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Resolve data.yaml always absolute so ultralytics can find it
YAML_PATH     = (BASE_DIR.parent / "processed" / "data.yaml").resolve()
YAML_PATH_STR = YAML_PATH.as_posix()

import contextlib
import yaml as _yaml

@contextlib.contextmanager
def patched_yaml(yaml_path: Path):
    """Temporarily write absolute path into data.yaml, restore original after."""
    raw = yaml_path.read_text(encoding="utf-8")
    data = _yaml.safe_load(raw)
    data["path"] = str(yaml_path.parent.resolve()).replace("\\", "/")
    yaml_path.write_text(_yaml.dump(data, sort_keys=False), encoding="utf-8")
    try:
        yield
    finally:
        yaml_path.write_text(raw, encoding="utf-8")

MODEL_REGISTRY = {
    "yolov5":  {"cls": YOLOv5,  "trainer_cls": YOLOv5Trainer,
                "size": "m", "label": "YOLOv5m",  "color": "#1f77b4"},
    "yolov8":  {"cls": YOLOv8,  "trainer_cls": YOLOv8Trainer,
                "size": "m", "label": "YOLOv8m",  "color": "#ff7f0e"},
    "yolov11": {"cls": YOLOv11, "trainer_cls": YOLOv11Trainer,
                "size": "m", "label": "YOLOv11m", "color": "#2ca02c"},
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.color": "#cccccc", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
})


def _parse_results_csv(csv_path: Path) -> Tuple[dict, dict]:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    def safe(col):
        return df[col].tolist() if col in df.columns else [0.0] * len(df)
    epochs = df["epoch"].tolist()
    train_history = {
        "epoch": epochs,
        "loss":  [a + b for a, b in zip(safe("train/box_loss"), safe("train/cls_loss"))],
        "mAP50": safe("metrics/mAP50(B)"),
    }
    val_history = {
        "epoch": epochs,
        "loss":  [a + b for a, b in zip(safe("val/box_loss"), safe("val/cls_loss"))],
        "mAP50": safe("metrics/mAP50(B)"),
    }
    return train_history, val_history


def plot_training_curves(runs: list, save_path: Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = [
        (axes[0, 0], "val/loss",    "val_history",   "loss"),
        (axes[0, 1], "val/mAP50",   "val_history",   "mAP50"),
        (axes[1, 0], "train/mAP50", "train_history", "mAP50"),
        (axes[1, 1], "train/loss",  "train_history", "loss"),
    ]
    for ax, title, phase_key, metric in panels:
        for run in runs:
            hist = run[phase_key]
            ax.plot(hist["epoch"], hist[metric], color=run["color"],
                    linewidth=2.0, marker="o", markersize=3, label=run["label"])
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylim(bottom=0)
        ax.legend(loc="best", frameon=True, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


def _save_history_csv(run: dict) -> None:
    csv_dir = RESULTS_DIR / "train_metrics_data" / run["tag"]
    csv_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_dir / "history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "epoch", "loss", "mAP50"])
        for phase, key in (("train", "train_history"), ("val", "val_history")):
            hist = run[key]
            for i in range(len(hist["epoch"])):
                writer.writerow([phase, hist["epoch"][i], hist["loss"][i], hist["mAP50"][i]])
    print(f"[saved] {csv_dir / 'history.csv'}")


def _save_run_artifacts(run: dict) -> None:
    _save_history_csv(run)
    plot_training_curves([run], RESULTS_DIR / "viz" / run["tag"] / "training_curves.png")


def train_model(model_key: str, num_epochs: int, batch_size: int) -> dict:
    cfg = MODEL_REGISTRY[model_key]
    print(f"\n{'='*60}\n  Training {cfg['label']}  ({num_epochs} epochs, batch={batch_size})\n{'='*60}")

    model    = cfg["cls"](model_size=cfg["size"])
    save_dir = CHECKPOINTS / model_key
    trainer  = cfg["trainer_cls"](model=model, data_yaml=YAML_PATH_STR,
                                   training_save_dir=save_dir)
    with patched_yaml(YAML_PATH):
        trainer.train(epochs=num_epochs, batch_size=batch_size, workers=0)

    # find results.csv (YOLO writes it inside the run folder)
    candidates = list(save_dir.rglob("results.csv"))
    if not candidates:
        print(f"[warn] results.csv not found for {model_key}, skipping history.")
        return {}
    results_csv = candidates[0]

    train_hist, val_hist = _parse_results_csv(results_csv)
    run = {"tag": model_key, "label": cfg["label"], "color": cfg["color"],
           "train_history": train_hist, "val_history": val_hist}
    _save_run_artifacts(run)
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 3 — YOLO training")
    parser.add_argument("--model", default="all",
                        choices=list(MODEL_REGISTRY.keys()) + ["all"])
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch",  default=16,  type=int)
    args = parser.parse_args()

    keys = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    completed_runs = []
    for key in keys:
        run = train_model(key, num_epochs=args.epochs, batch_size=args.batch)
        if run:
            completed_runs.append(run)

    if len(completed_runs) > 1:
        comparison_path = RESULTS_DIR / "viz" / "model_comparison.png"
        plot_training_curves(completed_runs, comparison_path)
        print(f"\nComparison plot -> {comparison_path}")