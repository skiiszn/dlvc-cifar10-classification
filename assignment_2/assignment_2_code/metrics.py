from abc import ABCMeta, abstractmethod
import torch

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass


class SegMetrics(PerformanceMeasure):
    '''
    Mean Intersection over Union.
    '''

    IGNORE_INDEX = 255

    def __init__(self, classes):
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        # Confusion matrix: rows = ground-truth class, cols = predicted class.
        # CM[i, j] = number of pixels with true label i predicted as class j.
        self.confusion_matrix = torch.zeros(
            (self.classes, self.classes), dtype=torch.int64
        )



    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (b,c,h,w) where b=batchsize, c=num_classes, h=height, w=width.
        target must have shape (b,h,w) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        Make sure to not include pixels of value 255 in the calculation since those are to be ignored. 
        '''

        # type checks
        if not isinstance(prediction, torch.Tensor):
            raise ValueError("prediction must be a torch.Tensor")
        if not isinstance(target, torch.Tensor):
            raise ValueError("target must be a torch.Tensor")

        # shape checks
        if prediction.dim() != 4:
            raise ValueError(
                f"prediction must have shape (b,c,h,w), "
                f"got tensor with {prediction.dim()} dims: {tuple(prediction.shape)}"
            )
        if target.dim() != 3:
            raise ValueError(
                f"target must have shape (b,h,w), "
                f"got tensor with {target.dim()} dims: {tuple(target.shape)}"
            )

        b, c, h, w = prediction.shape
        if c != self.classes:
            raise ValueError(
                f"prediction has {c} classes in dim=1, expected {self.classes}"
            )
        if tuple(target.shape) != (b, h, w):
            raise ValueError(
                f"target shape {tuple(target.shape)} is incompatible with "
                f"prediction shape {tuple(prediction.shape)} (expected ({b},{h},{w}))"
            )

        # value checks
        target_long = target.long()
        valid_mask = target_long != self.IGNORE_INDEX  # exclude pixels labeled 255

        if valid_mask.any():
            valid_targets = target_long[valid_mask]
            tmin = int(valid_targets.min().item())
            tmax = int(valid_targets.max().item())
            if tmin < 0 or tmax >= self.classes:
                raise ValueError(
                    f"target values (excluding the ignore label {self.IGNORE_INDEX}) "
                    f"must be in [0, {self.classes - 1}], got min={tmin}, max={tmax}"
                )
        else:
            # Every pixel is ignored — nothing to accumulate.
            return

        # Reduce prediction logits/probs to class labels
        # prediction (b, c, h, w) -> pred_labels (b, h, w)
        pred_labels = prediction.argmax(dim=1).long()

        # Update the confusion matrix
        t_flat = target_long[valid_mask]
        p_flat = pred_labels[valid_mask]

        # bincount trick: encode (true, pred) as a single index
        # index = true * C + pred  ->  reshape to (C, C)
        idx = t_flat * self.classes + p_flat
        cm_update = torch.bincount(
            idx, minlength=self.classes * self.classes
        ).reshape(self.classes, self.classes)

        self.confusion_matrix += cm_update.to(self.confusion_matrix.device).to(
            self.confusion_matrix.dtype
        )
   

    def __str__(self):
        '''
        Return a string representation of the performance, mean IoU.
        e.g. "mIou: 0.54"
        '''
        return f"mIou: {self.mIoU():.2f}"
          

    
    def mIoU(self) -> float:
        '''
        Compute and return the mean IoU as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        If the denominator for IoU calculation for one of the classes is 0,
        use 0 as IoU for this class.
        '''
        cm = self.confusion_matrix.to(torch.float64)

        if cm.sum() == 0:
            return 0.0

        # For each class k:
        # TP_k = CM[k, k]
        # FP_k = sum(CM[:, k]) - TP_k   (predicted k but truly something else)
        # FN_k = sum(CM[k, :]) - TP_k   (truly k but predicted something else)
        # IoU_k = TP_k / (TP_k + FP_k + FN_k) = CM[k,k] / (row_sum_k + col_sum_k - CM[k,k])
        tp = torch.diag(cm)
        row_sum = cm.sum(dim=1)  # ground-truth pixels per class (TP + FN)
        col_sum = cm.sum(dim=0)  # predicted pixels per class    (TP + FP)
        denom = row_sum + col_sum - tp

        # Per-class IoU: where denom == 0, IoU is defined as 0.
        iou = torch.where(
            denom > 0,
            tp / denom.clamp(min=1),  # clamp avoids div-by-zero in the false branch
            torch.zeros_like(tp),
        )

        return float(iou.mean().item())





