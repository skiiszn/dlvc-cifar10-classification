import collections
import torch
from typing import  Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm

#from assignment_2_code.wandb_logger import WandBLogger

from assignment_2_code.dataset.oxfordpets import OxfordPetsCustom


class BaseTrainer(metaclass=ABCMeta):
    '''
    Base class of all Trainers.
    '''

    @abstractmethod
    def train(self) -> None:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float]:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float]:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

class ImgSemSegTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """
    def __init__(self, 
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
                 val_frequency: int = 5):
        '''
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.SegMetrics): SegMetrics class to get mIoU of training set
            val_metric (dlvc.metrics.SegMetrics): SegMetrics class to get mIoU of validation set
            train_data (dlvc.datasets...): Train dataset
            val_data (dlvc.datasets...): Validation dataset
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

        '''
        

    
        ##TODO implement
        # recycle your code from assignment 1 or use/adapt reference implementation
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

        # OxfordIIITPet labels are (1, 2, 3) so we need to subtract 1 to get 0-based
        # class indices for CrossEntropyLoss.
        # Cityscapes labels are already 0-based (with 255 as the ignore label).
        self.subtract_one = "OxfordIIITPet" in {c.__name__ for c in type(train_data).__mro__}
        #self.subtract_one = isinstance(train_data, OxfordPetsCustom)

        # make sure the checkpoint directory exists
        self.training_save_dir.mkdir(parents=True, exist_ok=True)

        # best mean IoU seen on the validation set so far
        # used to decide when to overwrite the best-model checkpoint
        self.best_val_mIoU = -1.0

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
            num_workers=1,
        )

        self.num_train_data = len(train_data)
        self.num_val_data = len(val_data)



        

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        """
        Training logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean IoU for this epoch.

        epoch_idx (int): Current epoch number
        """
        ##TODO implement
        # recycle your code from assignment 1 or use/adapt reference implementation
        self.model.train()
        self.train_metric.reset()

        running_loss = 0.0

        for inputs, labels in tqdm(
                self.train_loader,
                desc=f"train epoch {epoch_idx}",
                leave=False,
        ):
            # segmentation targets come with a channel dim: (B, 1, H, W) -> (B, H, W)
            # subtract 1 for Oxford so labels are in {0, 1, 2}
            labels = labels.squeeze(1) - int(self.subtract_one)

            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            batch_size = inputs.shape[0]

            # forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            # torchvision's fcn_resnet50 returns an OrderedDict with key 'out'
            if isinstance(outputs, collections.OrderedDict):
                outputs = outputs["out"]

            loss = self.loss_fn(outputs, labels)

            # backward pass + parameter update
            loss.backward()
            self.optimizer.step()

            # weight the loss by batch size so partial last batches are handled correctly
            running_loss += loss.item() * batch_size
            # detach() so the metric does not hold on to the graph
            self.train_metric.update(outputs.detach().cpu(), labels.detach().cpu())

        # step lr scheduler at end of each training epoch
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        epoch_loss = running_loss / max(self.num_train_data, 1)
        epoch_mIoU = self.train_metric.mIoU()

        # print results
        print(f"\n______epoch {epoch_idx}\n")
        print("[train]")
        print(f"loss: {epoch_loss}")
        print(self.train_metric)

        return epoch_loss, epoch_mIoU


    def _val_epoch(self, epoch_idx:int) -> Tuple[float, float]:
        """
        Validation logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean IoU for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        ##TODO implement
        # recycle your code from assignment 1 or use/adapt reference implementation
        self.model.eval()
        self.val_metric.reset()

        running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in tqdm(
                    self.val_loader,
                    desc=f"val epoch {epoch_idx}",
                    leave=False,
            ):
                labels = labels.squeeze(1) - int(self.subtract_one)

                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                batch_size = inputs.shape[0]

                outputs = self.model(inputs)
                if isinstance(outputs, collections.OrderedDict):
                    outputs = outputs["out"]

                loss = self.loss_fn(outputs, labels)

                running_loss += loss.item() * batch_size
                self.val_metric.update(outputs.cpu(), labels.cpu())

        epoch_loss = running_loss / max(self.num_val_data, 1)
        epoch_mIoU = self.val_metric.mIoU()

        # print results
        print("[val]")
        print(f"loss: {epoch_loss}")
        print(self.val_metric)

        return epoch_loss, epoch_mIoU

    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean IoU on validation data set is higher
        than currently saved best mean IoU or if it is end of training. 
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """
        ##TODO implement
        # recycle your code from assignment 1 or use/adapt reference implementation
        for epoch_idx in range(self.num_epochs):
            train_loss, train_mIoU = self._train_epoch(epoch_idx)

            # validation pass + best-model checkpointing
            do_val = (epoch_idx % self.val_frequency == 0) or (
                    epoch_idx == self.num_epochs - 1
            )
            if do_val:
                val_loss, val_mIoU = self._val_epoch(epoch_idx)

                # save the model if val mIoU improved
                if val_mIoU >= self.best_val_mIoU:
                    self.best_val_mIoU = val_mIoU
                    print(f" best mIoU: {val_mIoU}")
                    print(f" saving model to {self.training_save_dir}")
                    self.model.save(self.training_save_dir, suffix="best")

                # always save the last checkpoint at the end of training
                if epoch_idx == self.num_epochs - 1:
                    self.model.save(self.training_save_dir, suffix="last")



                





            
            


