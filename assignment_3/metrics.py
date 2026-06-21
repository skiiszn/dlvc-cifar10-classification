"""
Assignment 3 - Brain Tumor Detection
Tracks per-class precision, recall, and AP@50 across batches of YOLO predictions.
Compute is deferred to the end of an epoch.
"""

from abc import ABCMeta, abstractmethod
from collections import defaultdict
import torch
import numpy as np


class PerformanceMeasure(metaclass=ABCMeta):
    """A performance measure."""

    @abstractmethod
    def reset(self):
        """Resets internal state."""
        pass

    @abstractmethod
    def update(self, predictions: list, targets: list):
        """
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data is unsupported.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of the performance."""
        pass


class DetectionMetrics(PerformanceMeasure):
    """
    Per-class detection metrics: Precision, Recall, and AP@50.

    update() accepts lists of dicts matching the YOLO ultralytics result format:
        predictions: list of dicts, one per image
            {
                'boxes':  Tensor[N, 4]  (x1, y1, x2, y2, normalised 0-1),
                'scores': Tensor[N],
                'labels': Tensor[N],    (int class id)
            }
        targets: list of dicts, one per image
            {
                'boxes':  Tensor[M, 4],
                'labels': Tensor[M],
            }
    """

    def __init__(self, num_classes: int, class_names: dict, iou_threshold: float = 0.5):
        """
        Args:
            num_classes:   number of detection classes (4 for brain tumor dataset)
            class_names:   dict mapping int id -> str name
                           e.g. {0:'Glioma', 1:'Meningioma', 2:'NoTumor', 3:'Pituitary'}
            iou_threshold: IoU threshold for a prediction to count as TP (default 0.5 = AP@50)
        """
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        self.num_classes   = num_classes
        self.class_names   = class_names
        self.iou_threshold = iou_threshold
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all accumulators (called at the start of every epoch)."""
        # per-class lists of (score, is_tp) tuples — filled by update()
        self._scores = defaultdict(list)   # class_id -> [float, ...]
        self._tp     = defaultdict(list)   # class_id -> [0/1, ...]
        self._n_gt   = defaultdict(int)    # class_id -> int  (ground-truth count)

    # ------------------------------------------------------------------
    def update(self, predictions: list, targets: list) -> None:
        """
        Accumulate one batch of predictions vs. targets.

        Args:
            predictions: list of dicts (one per image), keys: boxes, scores, labels
            targets:     list of dicts (one per image), keys: boxes, labels
        Raises:
            ValueError: if inputs are not lists or dicts are missing required keys
        """
        if not isinstance(predictions, list) or not isinstance(targets, list):
            raise ValueError("predictions and targets must be lists of dicts")
        if len(predictions) != len(targets):
            raise ValueError(
                f"predictions ({len(predictions)}) and targets ({len(targets)}) "
                "must have the same length"
            )

        for pred, tgt in zip(predictions, targets):
            for key in ("boxes", "scores", "labels"):
                if key not in pred:
                    raise ValueError(f"prediction dict missing key '{key}'")
            for key in ("boxes", "labels"):
                if key not in tgt:
                    raise ValueError(f"target dict missing key '{key}'")

            tgt_boxes  = tgt["boxes"]    # (M, 4)
            tgt_labels = tgt["labels"]   # (M,)

            # count ground-truth per class
            for cls in tgt_labels.tolist():
                self._n_gt[int(cls)] += 1

            pred_boxes  = pred["boxes"]   # (N, 4)
            pred_scores = pred["scores"]  # (N,)
            pred_labels = pred["labels"]  # (N,)

            # track which GT boxes have already been matched this image
            matched_gt = set()

            # sort predictions by descending confidence
            order = torch.argsort(pred_scores, descending=True)

            for idx in order.tolist():
                cls   = int(pred_labels[idx].item())
                score = float(pred_scores[idx].item())
                box   = pred_boxes[idx]   # (4,)

                self._scores[cls].append(score)

                # find GT boxes of the same class
                gt_mask = (tgt_labels == cls).nonzero(as_tuple=False).squeeze(1)

                best_iou  = 0.0
                best_gt   = -1
                for g in gt_mask.tolist():
                    if g in matched_gt:
                        continue
                    iou = self._box_iou(box, tgt_boxes[g])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt  = g

                if best_iou >= self.iou_threshold and best_gt >= 0:
                    self._tp[cls].append(1)
                    matched_gt.add(best_gt)
                else:
                    self._tp[cls].append(0)

    # ------------------------------------------------------------------
    def __str__(self) -> str:
        """Return a compact summary string, e.g. 'mAP@50: 0.72'."""
        return f"mAP@50: {self.mAP():.4f}"

    # ------------------------------------------------------------------
    def mAP(self) -> float:
        """Compute and return mean AP@50 across all classes (float 0-1)."""
        aps = [self._average_precision(c) for c in range(self.num_classes)]
        return float(np.mean(aps))

    def precision_recall_per_class(self) -> dict:
        """
        Return per-class precision and recall at the operating point
        (all predictions, sorted by score).

        Returns:
            dict: class_id -> {'precision': float, 'recall': float, 'ap': float,
                               'name': str, 'n_gt': int}
        """
        result = {}
        for cls in range(self.num_classes):
            n_gt = self._n_gt[cls]
            tp   = self._tp[cls]
            n_pred = len(tp)

            if n_pred == 0 or n_gt == 0:
                result[cls] = dict(precision=0.0, recall=0.0,
                                   ap=0.0, name=self.class_names.get(cls, str(cls)),
                                   n_gt=n_gt)
                continue

            tp_cum = np.cumsum(tp)
            precision = tp_cum[-1] / n_pred
            recall    = tp_cum[-1] / n_gt
            ap        = self._average_precision(cls)
            result[cls] = dict(precision=float(precision), recall=float(recall),
                               ap=float(ap), name=self.class_names.get(cls, str(cls)),
                               n_gt=n_gt)
        return result

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _average_precision(self, cls: int) -> float:
        """Compute AP for a single class using the 11-point interpolation."""
        n_gt = self._n_gt[cls]
        if n_gt == 0 or not self._tp[cls]:
            return 0.0

        # sort by score descending (already inserted in order, but be safe)
        paired = sorted(zip(self._scores[cls], self._tp[cls]),
                        key=lambda x: x[0], reverse=True)
        tp_arr = np.array([p[1] for p in paired], dtype=np.float32)

        tp_cum = np.cumsum(tp_arr)
        prec   = tp_cum / (np.arange(len(tp_arr)) + 1)
        rec    = tp_cum / n_gt

        # 11-point interpolation
        ap = 0.0
        for thr in np.linspace(0, 1, 11):
            mask = rec >= thr
            ap  += (prec[mask].max() if mask.any() else 0.0)
        return ap / 11.0

    @staticmethod
    def _box_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
        """IoU between two boxes in (x1,y1,x2,y2) format."""
        x1 = max(box_a[0].item(), box_b[0].item())
        y1 = max(box_a[1].item(), box_b[1].item())
        x2 = min(box_a[2].item(), box_b[2].item())
        y2 = min(box_a[3].item(), box_b[3].item())

        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = (box_a[2] - box_a[0]).item() * (box_a[3] - box_a[1]).item()
        area_b = (box_b[2] - box_b[0]).item() * (box_b[3] - box_b[1]).item()
        union  = area_a + area_b - inter
        return inter / union if union > 0 else 0.0