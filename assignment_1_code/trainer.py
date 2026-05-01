import torch
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm

# for wandb users:
from assignment_1_code.wandb_logger import WandBLogger


class BaseTrainer(metaclass=ABCMeta):
    """
    Base class of all Trainers.
    """

    @abstractmethod
    def train(self) -> None:
        """
        Holds training logic.
        """

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float, float]:
        """
        Holds validation logic for one epoch.
        """

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float, float]:
        """
        Holds training logic for one epoch.
        """

        pass


class ImgClassificationTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        lr_scheduler,
        train_metric,
        val_metric,
        train_data,
        val_data,
        device,
        num_epochs: int,
        training_save_dir: Path,
        batch_size: int = 4,
        val_frequency: int = 5,
    ) -> None:
        """
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of training set
            val_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of validation set
            train_data (dlvc.datasets.cifar10.CIFAR10Dataset): Train dataset
            val_data (dlvc.datasets.cifar10.CIFAR10Dataset): Validation dataset
            device (torch.device): cuda or cpu - device used to train the network
            num_epochs (int): number of epochs to train the network
            training_save_dir (Path): the path to the folder where the best model is stored
            batch_size (int): number of samples in one batch
            val_frequency (int): how often validation is conducted during training (if it is 5 then every 5th
                                epoch we evaluate model on validation set)

        What does it do:
            - Stores given variables as instance variables for use in other class methods e.g. self.model = model.
            - Creates data loaders for the train and validation datasets
            - Optionally use weights & biases for tracking metrics and loss: initializer W&B logger

        """

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = Path(training_save_dir)
        self.batch_size = batch_size
        self.val_frequency = val_frequency

        # check directory exists
        self.training_save_dir.mkdir(parents=True, exist_ok=True)


        # best mean per-class accuracy seen on the validation set so far
        # Used to decide when to overwrite the best-model checkpoint
        self.best_val_pcacc = -1.0

        # data loaders
        self.train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )

        # optional: weights&biases logger
        # self.wandb_logger = WandBLogger(
        #     enabled=True,
        #     model=self.model,
        #     run_name=None,  # falls back to wandb run id
        # )









    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Training logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch.

        epoch_idx (int): Current epoch number
        """

        self.model.train()
        self.train_metric.reset()

        running_loss = 0.0
        n_batches = 0

        for inputs, targets in tqdm(
                self.train_loader,
                desc=f"train epoch {epoch_idx}",
                leave=False,
        ):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

            # backward pass + parameter update
            loss.backward()
            self.optimizer.step()

            # update: detach() so the metric does not hold on to the graph
            running_loss += loss.item()
            n_batches += 1
            self.train_metric.update(outputs.detach(), targets)

        epoch_loss = running_loss / max(n_batches, 1)
        epoch_acc = self.train_metric.accuracy()
        epoch_pcacc = self.train_metric.per_class_accuracy()

        # print results
        print(f"______epoch {epoch_idx}\n")
        print("[train]")
        print(f"loss: {epoch_loss}")
        print(self.train_metric)

        return epoch_loss, epoch_acc, epoch_pcacc

    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Validation logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """

        self.model.eval()
        self.val_metric.reset()

        running_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                running_loss += loss.item()
                n_batches += 1
                self.val_metric.update(outputs, targets)

        epoch_loss = running_loss / max(n_batches, 1)
        epoch_acc = self.val_metric.accuracy()
        epoch_pcacc = self.val_metric.per_class_accuracy()

        # print results
        print("[val]")
        print(f"loss: {epoch_loss}")
        print(self.val_metric)

        return epoch_loss, epoch_acc, epoch_pcacc

    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per class accuracy on validation data set is higher
        than currently saved best mean per class accuracy.
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """

        for epoch_idx in range(self.num_epochs):
            train_loss, train_acc, train_pcacc = self._train_epoch(epoch_idx)

            # weights&biases per-epoch train logging
            # self.wandb_logger.log(
            #     {
            #         "train/loss": train_loss,
            #         "train/accuracy": train_acc,
            #         "train/per_class_accuracy": train_pcacc,
            #         "epoch": epoch_idx,
            #     },
            #     commit=False,  # commit together with val metrics if val runs
            # )

            # validation pass + best-model checkpointing
            do_val = ((epoch_idx + 1) % self.val_frequency == 0) or (
                    epoch_idx == self.num_epochs - 1
            )
            if do_val:
                val_loss, val_acc, val_pcacc = self._val_epoch(epoch_idx)

                # weights&biases per-epoch val logging
                # self.wandb_logger.log(
                #     {
                #         "val/loss": val_loss,
                #         "val/accuracy": val_acc,
                #         "val/per_class_accuracy": val_pcacc,
                #         "epoch": epoch_idx,
                #     },
                #     commit=True,
                # )

                # save the model if val mean per-class accuracy improved
                if val_pcacc > self.best_val_pcacc:
                    self.best_val_pcacc = val_pcacc
                    self.model.save(self.training_save_dir, suffix="best")
                    print(
                        f"  -> new best model saved (val per-class acc={val_pcacc:.4f})"
                    )
            # else:
            #     # No val improvement this epoch - still commit the train-only wandb
            #     # self.wandb_logger.log({}, commit=True)
            #     pass


            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # self.wandb_logger.finish()
