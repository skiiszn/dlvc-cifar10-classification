from abc import ABCMeta, abstractmethod
import torch
from typing import Dict


class PerformanceMeasure(metaclass=ABCMeta):
    """
    A performance measure.
    """

    @abstractmethod
    def reset(self):
        """
        Resets internal state.
        """

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        """

        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the performance.
        """

        pass


class Accuracy(PerformanceMeasure):
    """
    Average classification accuracy.
    """

    def __init__(self, classes) -> None:
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state.
        """
        self.correct_pred = {classname: 0 for classname in self.classes}
        self.total_pred = {classname: 0 for classname in self.classes}
        self.n_matching = 0  # number of correct predictions
        self.n_total = 0
        self.per_class_accuracies = {classname: 0.0 for classname in self.classes} # dict mapping class name to accuracy

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (batchsize,n_classes) with each row being a class-score vector.
        target must have shape (batchsize,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        [len(prediction.shape) should be equal to 2, and len(target.shape) should be equal to 1.]
        """


        # type checks
        if not isinstance(prediction, torch.Tensor):
            raise ValueError(
                f"prediction must be a torch.Tensor, got {type(prediction)}"
            )
        if not isinstance(target, torch.Tensor):
            raise ValueError(f"target must be a torch.Tensor, got {type(target)}")



        # prediction must have shape (batchsize,n_classes)
        if len(prediction.shape) != 2:
            raise ValueError(
                f"prediction must have shape (batchsize, n_classes), got {tuple(prediction.shape)}"
            )

        # target must have shape (batchsize,)
        if len(target.shape) != 1:
            raise ValueError(
                f"target must have shape (batchsize,), got {tuple(target.shape)}"
            )

        if prediction.shape[0] != target.shape[0]:
            raise ValueError(
                f"prediction and target must have the same batch size, "
                f"got {prediction.shape[0]} and {target.shape[0]}"
            )


        # prediction with each row being a class-score vector
        n_classes = len(self.classes)
        if prediction.shape[1] != n_classes:
            raise ValueError(
                f"prediction must have {n_classes} classes (second dim), "
                f"got {prediction.shape[1]}"
            )


        # values between 0 and c-1 (true class labels)
        if target.numel() > 0:
            t_min = int(target.min().item())
            t_max = int(target.max().item())
            if t_min < 0 or t_max >= n_classes:
                raise ValueError(
                    f"target values must be in [0, {n_classes - 1}], "
                    f"got min={t_min}, max={t_max}"
                )


        # predicted class is argmax over class-score dim
        pred_classes = prediction.argmax(dim=1)

        # move to cpu to prevent holding references and therefore safer iteration/comparison
        pred_classes_cpu = pred_classes.detach().cpu()
        target_cpu = target.detach().cpu()

        # overall accuracy update
        correct_mask = pred_classes_cpu == target_cpu  # comparing ground truth
        self.n_matching += int(correct_mask.sum().item())
        self.n_total += int(target_cpu.shape[0])

        # per-class update
        for label, pred in zip(target_cpu.tolist(), pred_classes_cpu.tolist()):
            classname = self.classes[label]
            self.total_pred[classname] += 1
            if label == pred:
                self.correct_pred[classname] += 1

    def __str__(self):
        """
        Return a string representation of the performance including:
        - overall accuracy
        - mean per-class accuracy
        - individual per-class accuracies for all classes
        """


        # make sure self.per_class_accuracies is up to date
        mean_pc_acc = self.per_class_accuracy()
        acc = self.accuracy()

        lines = [
            f"overall accuracy: {acc:.4f}",
            f"mean per-class accuracy: {mean_pc_acc:.4f}",
        ]
        for classname in self.classes:
            class_acc = self.per_class_accuracies.get(classname, 0.0)
            lines.append(f"individual per-class accuracy for class: {classname:5s} is {class_acc:.2f} ")
        return "\n".join(lines)

    def accuracy(self) -> float:
        """
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """


        if self.n_total == 0:
            return 0.0
        return self.n_matching / self.n_total

    def per_class_accuracy(self) -> float:
        """
        Compute and return the mean per-class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        Saves the individual per-class accuracies in self.per_class_accuracies as a dict mapping class name to accuracy.
        """

        if self.n_total == 0:
            # keep per-class dict consistent (zeroed) and return 0
            self.per_class_accuracies = {classname: 0.0 for classname in self.classes}
            return 0.0

        per_class_accs = []
        for classname in self.classes:
            total = self.total_pred[classname]
            if total == 0:
                # no samples seen for this class
                self.per_class_accuracies[classname] = 0.0
            else:
                self.per_class_accuracies[classname] = (
                        self.correct_pred[classname] / total
                )
            per_class_accs.append(self.per_class_accuracies[classname])

        # mean over all classes
        return sum(per_class_accs) / len(per_class_accs)
