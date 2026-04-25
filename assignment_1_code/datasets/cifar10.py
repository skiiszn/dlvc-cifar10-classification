import pickle
import os
from typing import Tuple
import numpy as np
from assignment_1_code.datasets.dataset import Subset, ClassificationDataset


from assignment_1_code.datasets.dataset import Subset, ClassificationDataset


class CIFAR10Dataset(ClassificationDataset):
    """
    Custom CIFAR-10 Dataset.
    """

    def __init__(self, fdir: str, subset: Subset, transform=None):
        """
        Initializes the CIFAR-10 dataset.
        """
        self.classes = (
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

        self.fdir = fdir
        self.subset = subset
        self.transform = transform

        self.images, self.labels = self.load_cifar()

    def load_cifar(self) -> Tuple:
        """
        Loads the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all images from "data_batch_5".
          - The test set contains all images from "test_batch".

        Depending on which subset is selected, the corresponding images and labels are returned.

        Images are loaded in the order they appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in RGB channel order.
        Labels should be returned either as a Python list of ints or as a
        numpy array with dtype int64.
        """
        if not os.path.isdir(self.fdir):
            raise ValueError(f"{self.fdir} is not a directory")
        
        if self.subset == Subset.TRAINING:
            batch_files = [f"data_batch_{i}" for i in range(1, 5)]  # batches 1-4
        elif self.subset == Subset.VALIDATION:
            batch_files = ["data_batch_5"]  # batch 5
        elif self.subset == Subset.TEST:
            batch_files = ["test_batch"]
        else:
            raise ValueError(f"Unknown subset: {self.subset}")
        
        images_list = []
        labels_list = []
        
        for batch_file in batch_files:
            file_path = os.path.join(self.fdir, batch_file)
            if not os.path.isfile(file_path):
                raise ValueError(f"Missing file: {file_path}")
            
            with open(file_path, 'rb') as f:
                batch_data = pickle.load(f, encoding='bytes')
            
            images_list.append(batch_data[b'data']) # pixel values
            labels_list.extend(batch_data[b'labels']) # class numbers 0-9
        
        images = np.concatenate(images_list, axis=0) # stack batches
        labels = np.array(labels_list, dtype=np.int64)
        
        # reshape images - (N, 3072) -> (N, 32, 32, 3) in RGB order
        images = images.reshape(-1, 3, 32, 32)
        images = images.transpose(0, 2, 3, 1)  # -> (N, 32, 32, 3) RGB
        images = images.astype(np.uint8)
        
        return images, labels
        

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns the idx-th sample in the dataset, which is a tuple,
        consisting of the image and labels.
        Applies transforms if not None.
        Raises IndexError if the index is out of bounds.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")
        
        image = self.images[idx] # one image
        label = self.labels[idx] # one label
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def num_classes(self) -> int:
        """
        Returns the number of classes.
        """
        return len(self.classes)
