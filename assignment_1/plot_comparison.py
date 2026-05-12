

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple



_STYLE_BASELINE = [{"color": "black",  "linestyle": "-"}]
_STYLE_AUG      = [{"color": "blue",   "linestyle": "--", "dashes": (5, 5)}]
_STYLE_REG      = [{"color": "orange", "linestyle": "--"}]
_STYLE_BOTH     = [{"color": "orange", "linestyle": "--"},
                   {"color": "blue",   "linestyle": "--", "dashes": (5, 5)}]

_LEGEND_LABELS_BASE = {
    "noaug_noreg": "no aug, no reg",
    "aug_noreg":   "aug only",
    "noaug_reg":   "reg only",
    "aug_reg":     "aug + reg",
}

# Metrics to plot
METRICS = [
    ("loss",   "loss"),
    ("mAcc",   "mean accuracy"),
    ("mPCAcc", "mean per-class accuracy"),
]



MODELS = [
    {
        "name":       "CNN",
        "scheduler":  "exponential scheduler",
        "data_root":  Path("results/train_metrics_data_CNN"),
        "out_root":   Path("results/viz_CNN_compare"),
        "out_prefix": "cnn_compare",
        "configs": [
            "CNN_noaug_noreg_adamw_exp",
            "CNN_aug_noreg_adamw_exp",
            "CNN_noaug_reg_adamw_exp",
            "CNN_aug_reg_adamw_exp",
        ],
        "styles": {
            "CNN_noaug_noreg_adamw_exp": _STYLE_BASELINE,
            "CNN_aug_noreg_adamw_exp":   _STYLE_AUG,
            "CNN_noaug_reg_adamw_exp":   _STYLE_REG,
            "CNN_aug_reg_adamw_exp":     _STYLE_BOTH,
        },
        "legend_labels": {
            "CNN_noaug_noreg_adamw_exp": _LEGEND_LABELS_BASE["noaug_noreg"],
            "CNN_aug_noreg_adamw_exp":   _LEGEND_LABELS_BASE["aug_noreg"],
            "CNN_noaug_reg_adamw_exp":   _LEGEND_LABELS_BASE["noaug_reg"],
            "CNN_aug_reg_adamw_exp":     _LEGEND_LABELS_BASE["aug_reg"],
        },
    },
    {
        "name":       "ResNet18",
        "scheduler":  "exponential scheduler",
        "data_root":  Path("results/train_metrics_data_ResNet18"),
        "out_root":   Path("results/viz_ResNet18_compare"),
        "out_prefix": "resnet18_compare",
        "configs": [
            "ResNet18_noaug_noreg_adamw_exp",
            "ResNet18_aug_noreg_adamw_exp",
            "ResNet18_noaug_reg_adamw_exp",
            "ResNet18_aug_reg_adamw_exp",
        ],
        "styles": {
            "ResNet18_noaug_noreg_adamw_exp": _STYLE_BASELINE,
            "ResNet18_aug_noreg_adamw_exp":   _STYLE_AUG,
            "ResNet18_noaug_reg_adamw_exp":   _STYLE_REG,
            "ResNet18_aug_reg_adamw_exp":     _STYLE_BOTH,
        },
        "legend_labels": {
            "ResNet18_noaug_noreg_adamw_exp": _LEGEND_LABELS_BASE["noaug_noreg"],
            "ResNet18_aug_noreg_adamw_exp":   _LEGEND_LABELS_BASE["aug_noreg"],
            "ResNet18_noaug_reg_adamw_exp":   _LEGEND_LABELS_BASE["noaug_reg"],
            "ResNet18_aug_reg_adamw_exp":     _LEGEND_LABELS_BASE["aug_reg"],
        },
    },
    {
        "name":       "ViT",
        "scheduler":  "cosine scheduler",
        "data_root":  Path("results/train_metrics_data_ViT"),
        "out_root":   Path("results/viz_ViT_compare"),
        "out_prefix": "vit_compare",
        "configs": [
            "noaug_noreg_adamw_cos",
            "aug_noreg_adamw_cos",
            "noaug_reg_adamw_cos",
            "aug_reg_adamw_cos",
        ],
        "styles": {
            "noaug_noreg_adamw_cos": _STYLE_BASELINE,
            "aug_noreg_adamw_cos":   _STYLE_AUG,
            "noaug_reg_adamw_cos":   _STYLE_REG,
            "aug_reg_adamw_cos":     _STYLE_BOTH,
        },
        "legend_labels": {
            "noaug_noreg_adamw_cos": _LEGEND_LABELS_BASE["noaug_noreg"],
            "aug_noreg_adamw_cos":   _LEGEND_LABELS_BASE["aug_noreg"],
            "noaug_reg_adamw_cos":   _LEGEND_LABELS_BASE["noaug_reg"],
            "aug_reg_adamw_cos":     _LEGEND_LABELS_BASE["aug_reg"],
        },
    },
]


def load_history(csv_path: Path) -> dict:
    out = {
        "train": {"epoch": [], "loss": [], "mAcc": [], "mPCAcc": []},
        "val":   {"epoch": [], "loss": [], "mAcc": [], "mPCAcc": []},
    }
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            phase = row["phase"]
            if phase not in out:
                continue
            out[phase]["epoch"].append(int(row["epoch"]))
            out[phase]["loss"].append(float(row["loss"]))
            out[phase]["mAcc"].append(float(row["mAcc"]))
            out[phase]["mPCAcc"].append(float(row["mPCAcc"]))
    return out



def plot_metric(
    model_cfg: dict,
    histories: dict,
    metric_key: str,
    metric_label: str,
    out_path: Path,
) -> None:
    configs       = model_cfg["configs"]
    styles        = model_cfg["styles"]
    legend_labels = model_cfg["legend_labels"]
    model_name    = model_cfg["name"]
    scheduler     = model_cfg["scheduler"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for phase, ax in zip(["train", "val"], axes):
        for tag in configs:
            if tag not in histories:
                continue
            x = histories[tag][phase]["epoch"]
            y = histories[tag][phase][metric_key]
            for style in styles[tag]:
                ax.plot(x, y, **style)
        ax.set_title(phase)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(metric_label)

    # Build legend handles (tuple of Line2Ds for the "both" config)
    handles, labels = [], []
    for tag in configs:
        tag_styles = styles[tag]
        if len(tag_styles) == 1:
            h = Line2D([0], [0], **tag_styles[0])
        else:
            h = tuple(Line2D([0], [0], **s) for s in tag_styles)
        handles.append(h)
        labels.append(legend_labels[tag])

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(configs),
        bbox_to_anchor=(0.5, 1.02),
        handler_map={tuple: HandlerTuple(ndivide=None)},
        frameon=False,
    )
    fig.suptitle(
        f"{model_name}: {metric_label}",
        y=1.08,
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {out_path}")



def process_model(model_cfg: dict) -> None:
    """Load all CSVs for one model and produce its 3 metric figures."""
    name      = model_cfg["name"]
    data_root = model_cfg["data_root"]
    out_root  = model_cfg["out_root"]
    prefix    = model_cfg["out_prefix"]
    configs   = model_cfg["configs"]


    # Load available histories
    histories = {}
    for tag in configs:
        csv_path = data_root / tag / "history.csv"
        if not csv_path.is_file():
            print(f"  warning: {csv_path} not found -- skipping")
            continue
        histories[tag] = load_history(csv_path)

    if not histories:
        print(f"  no history files found under {data_root} -- skipping model")
        return

    # Produce one figure per metric
    for metric_key, metric_label in METRICS:
        plot_metric(
            model_cfg=model_cfg,
            histories=histories,
            metric_key=metric_key,
            metric_label=metric_label,
            out_path=out_root / f"{prefix}_{metric_key}.png",
        )



def main() -> None:
    """Iterate over all three models and generate all 9 figures."""
    for model_cfg in MODELS:
        process_model(model_cfg)



if __name__ == "__main__":
    main()
