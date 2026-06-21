"""
Assignment 3 - Brain Tumor Detection
test.py: loads trained model weights, evaluates on the held-out test split,
         prints a per-class AP table.

Usage:
    python test.py                  # evaluate all three models
    python test.py --model yolov8
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CODE_DIR = BASE_DIR / "assignment_3_code"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(BASE_DIR))

from models.model_yolov5 import YOLOv5
from models.model_yolov8 import YOLOv8
from models.model_yolov11 import YOLOv11

# ── Paths ────────────────────────────────────────────────────────────────────
CHECKPOINTS   = BASE_DIR / "checkpoints"
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

CLASS_NAMES  = {0: "Glioma", 1: "Meningioma", 2: "NoTumor", 3: "Pituitary"}

MODEL_REGISTRY = {
    "yolov5":  {"cls": YOLOv5,  "size": "m", "label": "YOLOv5m"},
    "yolov8":  {"cls": YOLOv8,  "size": "m", "label": "YOLOv8m"},
    "yolov11": {"cls": YOLOv11, "size": "m", "label": "YOLOv11m"},
}


def _find_best_weights(model_key: str, size: str) -> Path | None:
    """
    Locate best.pt under checkpoints/<model_key>/.
    If multiple run folders exist (e.g. yolov8m, yolov8m-2), picks the
    most recently modified one so the latest completed run is always used.
    """
    candidates = list((CHECKPOINTS / model_key).rglob("best.pt"))
    if not candidates:
        return None
    # sort by modification time, newest first
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def test_model(model_key: str) -> dict:
    """Load best weights and run evaluation on the test split."""
    cfg     = MODEL_REGISTRY[model_key]
    weights = _find_best_weights(model_key, cfg["size"])

    if weights is None:
        print(f"[skip] {cfg['label']}: no best.pt found under {CHECKPOINTS / model_key}")
        return {}

    print(f"\n{'='*60}")
    print(f"  Evaluating {cfg['label']}")
    print(f"  Weights: {weights}")
    print(f"{'='*60}")

    model = cfg["cls"](model_size=cfg["size"])
    model.load(weights)

    
    with patched_yaml(YAML_PATH):
        val_output = model.validate(data=YAML_PATH_STR, imgsz=640, batch_size=16, split="test")
    result_obj = val_output.get("results")

    if result_obj is None:
        print(f"[warn] No results returned for {cfg['label']}")
        return {}

    map50        = float(result_obj.box.map50)
    map5095      = float(result_obj.box.map)
    precision    = float(result_obj.box.mp)
    recall       = float(result_obj.box.mr)
    ap50_per_cls = result_obj.box.ap50   # numpy array

    print(f"\n  Results — {cfg['label']}:")
    print(f"  {'mAP@50':<18}: {map50:.4f}")
    print(f"  {'mAP@50-95':<18}: {map5095:.4f}")
    print(f"  {'Precision':<18}: {precision:.4f}")
    print(f"  {'Recall':<18}: {recall:.4f}")
    print(f"\n  Per-class AP@50:")
    for i, ap in enumerate(ap50_per_cls):
        print(f"    {CLASS_NAMES[i]:<14}: {float(ap):.4f}")

    return {
        "model":     cfg["label"],
        "mAP50":     map50,
        "mAP5095":   map5095,
        "precision": precision,
        "recall":    recall,
        "ap50_per_class": {CLASS_NAMES[i]: float(ap)
                           for i, ap in enumerate(ap50_per_cls)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 3 — test evaluation")
    parser.add_argument(
        "--model", default="all",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
    )
    args = parser.parse_args()

    keys = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    all_results = []
    for key in keys:
        r = test_model(key)
        if r:
            all_results.append(r)

    # summary table
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  SUMMARY — Test Set")
        print(f"{'='*60}")
        print(f"  {'Model':<12} {'mAP@50':>10} {'mAP@50-95':>12} {'Precision':>11} {'Recall':>9}")
        print(f"  {'-'*58}")
        for r in all_results:
            print(f"  {r['model']:<12} {r['mAP50']:>10.4f} {r['mAP5095']:>12.4f} "
                  f"{r['precision']:>11.4f} {r['recall']:>9.4f}")