import os
from glob import glob

import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchdata.datapipes.iter import FileLister, FileOpener
from torchdata.datapipes.map import MapDataPipe
from torchvision import transforms
import zipfile
import numpy as np
from PIL import Image


class CovarianceDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        clutter_files = glob(f'{root_dir}/clutter_*.cov')
        self.data = np.concatenate([np.fromfile(c, dtype=np.float32).reshape((-1, 32, 32, 2)) for c in clutter_files])
        self.transform = transform

    def __getitem__(self, idx):
        img = self.data[idx, ...]
        if self.transform is not None:
            img = self.transform(img)
        return img, img

    def __len__(self):
        return self.data.shape[0]


class DataModule(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            patch_size: Union[int, Sequence[int]] = (256, 256),
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.val_dataset = None
        self.train_dataset = None
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:

        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.034028642, 0.04619637), 0.4151423),
            ]
        )

        val_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.034028642, 0.04619637), 0.4151423),
            ]
        )

        self.train_dataset = CovarianceDataset(self.data_dir, transform=train_transforms)

        self.val_dataset = CovarianceDataset(self.data_dir, transform=val_transforms)

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )


if __name__ == '__main__':
    dpipe1 = FileLister('/home/jeff/repo/apache/data', 'clutter_*.tfrecords')
    dpipe2 = FileOpener(dpipe1, mode='b')
    test = dpipe2.load_from_tfrecord()
    for example in test:
        print(example)