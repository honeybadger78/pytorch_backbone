import cv2
import numpy as np

from glob import glob
from natsort import natsorted

import torch
from torch.utils.data import Dataset


class TinyImagenetDataset(Dataset):
    def __init__(
        self,
        path: str,
        augmentation: None,
        is_train: bool = True,
    ):
        super().__init__()
        self.path = path
        self.augmentation = augmentation
        self.is_train = is_train

        if self.is_train:
            self.image = natsorted(glob(path + "train/*/*.jpg"))

        else:
            self.image = natsorted(glob(path + "val/*/*.jpg"))

        self.label = np.array([int(path.split("/")[-2]) for path in self.image])
        self.label = torch.from_numpy(self.label)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        assert len(self.image) == len(self.label), "image and label must be same length"

        image = cv2.imread(self.image[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.label[index]

        if self.augmentation is not None:
            image = self.augmentation(image=image)["image"]

        return image, label
