"""
Assignment 3 - Brain Tumor Detection
comparison.py: generates the two final output figures.

Outputs saved to results/:
    predictions_<model>.png      — rows: Input Image | Ground Truth | Prediction
    final_comparison_plot.png    — single plot, one line per model, val mAP@50

Usage:
    python comparison.py                      # all three models
    python comparison.py --model yolov8       # predictions for one model only
    python comparison.py --num_samples 4
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
CODE_DIR = BASE_DIR / "assignment_3_code"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(BASE_DIR))

# ── Paths ────────────────────────────────────────────────────────────────────
PROCESSED   = BASE_DIR.parent / "processed" 
CHECKPOINTS = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES  = {0: "Glioma", 1: "Meningioma", 2: "NoTumor", 3: "Pituitary"}
CLASS_COLORS = {0: "#e74c3c", 1: "#3498db",   2: "#2ecc71", 3: "#f39c12"}

MODEL_REGISTRY = {
    "yolov5":  {"label": "YOLOv5m",  "color": "#1f77b4"},
    "yolov8":  {"label": "YOLOv8m",  "color": "#ff7f0e"},
    "yolov11": {"label": "YOLOv11m", "color": "#2ca02c"},
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.color": "#cccccc", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
})


def _find_best_weights(model_key: str) -> Path | None:
    """Pick the most recently modified best.pt — handles multiple run folders."""
    candidates = list((CHECKPOINTS / model_key).rglob("best.pt"))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _draw_boxes(ax, boxes_norm, img_w, img_h, linewidth=2):
    """Draw YOLO-normalised (cls, xc, yc, w, h) boxes on ax."""
    for cls_id, xc, yc, bw, bh in boxes_norm:
        x1    = (xc - bw / 2) * img_w
        y1    = (yc - bh / 2) * img_h
        color = CLASS_COLORS[int(cls_id)]
        rect  = mpatches.Rectangle(
            (x1, y1), bw * img_w, bh * img_h,
            linewidth=linewidth, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, CLASS_NAMES[int(cls_id)],
                color="white", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor=color, alpha=0.85, linewidth=0))


def visualize_predictions(model_key: str, num_samples: int = 4,
                           conf: float = 0.25) -> None:
    weights = _find_best_weights(model_key)
    if weights is None:
        print(f"[skip] {model_key}: no best.pt found.")
        return

    print(f"\nGenerating predictions — {MODEL_REGISTRY[model_key]['label']} ...")

    # Use ultralytics YOLO directly — avoids issues with the detect() wrapper
    model = YOLO(str(weights))

    test_img_dir = PROCESSED / "images" / "test"
    label_dir    = PROCESSED / "labels"  / "test"
    exts         = {".jpg", ".jpeg", ".png", ".bmp"}
    all_imgs     = [f for f in test_img_dir.glob("*") if f.suffix.lower() in exts]
    sample       = random.sample(all_imgs, min(num_samples, len(all_imgs)))

    # Run inference on all samples at once
    results = model.predict(source=sample, imgsz=640, conf=conf, verbose=False)

    fig, axes = plt.subplots(num_samples, 3,
                             figsize=(12, 4 * num_samples),
                             facecolor="white")
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input Image", "Ground Truth", "Prediction"]

    for row, (img_path, result) in enumerate(zip(sample, results)):
        img  = Image.open(img_path).convert("RGB")
        w, h = img.size

        for col, title in enumerate(col_titles):
            axes[row, col].set_title(title, fontsize=11, fontweight="bold", pad=6)

        # col 0 — raw input image
        axes[row, 0].imshow(img)
        axes[row, 0].axis("off")

        # col 1 — ground truth boxes from label file
        axes[row, 1].imshow(img)
        axes[row, 1].axis("off")
        gt_label = label_dir / (img_path.stem + ".txt")
        if gt_label.exists():
            gt_boxes = []
            with open(gt_label) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        gt_boxes.append([int(parts[0])] + list(map(float, parts[1:])))
            _draw_boxes(axes[row, 1], gt_boxes, w, h)

        # col 2 — model predictions via ultralytics result object
        axes[row, 2].imshow(img)
        axes[row, 2].axis("off")
        if result.boxes is not None and len(result.boxes):
            pred_boxes = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf_v = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                xc = (x1 + x2) / 2 / w
                yc = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                pred_boxes.append([cls_id, xc, yc, bw, bh])
                axes[row, 2].text(
                    x1, y1 - 5,
                    f"{CLASS_NAMES[cls_id]} {conf_v:.2f}",
                    color="white", fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor=CLASS_COLORS[cls_id],
                              alpha=0.85, linewidth=0)
                )
            _draw_boxes(axes[row, 2], pred_boxes, w, h)
        else:
            axes[row, 2].text(0.5, 0.5, "no detection",
                              ha="center", va="center",
                              transform=axes[row, 2].transAxes,
                              color="red", fontsize=10)

    legend_handles = [
        mpatches.Patch(facecolor=CLASS_COLORS[i], label=CLASS_NAMES[i])
        for i in CLASS_NAMES
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), fontsize=10,
               framealpha=0.8, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    out_path = RESULTS_DIR / f"predictions_{model_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def _load_history(model_key: str) -> dict | None:
    csv_path = RESULTS_DIR / "train_metrics_data" / model_key / "history.csv"
    if not csv_path.exists():
        print(f"[skip] {model_key}: history CSV not found at {csv_path}")
        return None

    val_hist = {"epoch": [], "mAP50": []}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["phase"] == "val":
                val_hist["epoch"].append(int(row["epoch"]))
                val_hist["mAP50"].append(float(row["mAP50"]))

    cfg = MODEL_REGISTRY[model_key]
    return {"tag": model_key, "label": cfg["label"],
            "color": cfg["color"], "val_history": val_hist}


def plot_final_comparison() -> None:
    runs = [_load_history(k) for k in MODEL_REGISTRY]
    runs = [r for r in runs if r is not None]
    if not runs:
        print("[skip] No history CSVs found — run train.py --rebuild first.")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for run in runs:
        vh = run["val_history"]
        ax.plot(vh["epoch"], vh["mAP50"],
                color=run["color"], linewidth=2, marker="o", markersize=5,
                label=run["label"])

    ax.set_title("YOLO Model Comparison — Validation mAP@50",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation mAP@50")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", frameon=True, framealpha=0.85, fontsize=11)

    plt.tight_layout()
    out_path = RESULTS_DIR / "final_comparison_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 3 — predictions & comparison")
    parser.add_argument("--model", default="all",
                        choices=list(MODEL_REGISTRY.keys()) + ["all"])
    parser.add_argument("--num_samples", default=4,    type=int)
    parser.add_argument("--conf",        default=0.25, type=float)
    args = parser.parse_args()

    keys = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    for key in keys:
        visualize_predictions(key, num_samples=args.num_samples, conf=args.conf)
    plot_final_comparison()