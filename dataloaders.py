from glob import glob
from typing import List, Optional, Sequence, Union
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
from torchdata.datapipes.iter import FileLister, FileOpener
from torchvision import transforms
from pathlib import Path
import numpy as np

from models import BaseVAE


class CovarianceDataset(Dataset):

    def __init__(self, root_dir, transform=None, split=1., single_example=False, mu=0., var=1.):
        if Path(root_dir).is_dir():
            clutter_files = glob(f'{root_dir}/clutter_*.cov')
            self.data = np.concatenate([np.fromfile(c,
                                                    dtype=np.float32).reshape((-1, 32, 32, 2)) for c in clutter_files])
        else:
            self.data = np.fromfile(root_dir, dtype=np.float32).reshape((-1, 32, 32, 2))
        # Do split
        if split < 1:
            self.data = self.data[np.random.choice(np.arange(self.data.shape[0]),
                                                   int(self.data.shape[0] * split)), ...]
        if single_example:
            self.data[1:, ...] = self.data[0, ...]
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mu, var),
                ]
            )

    def __getitem__(self, idx):
        img = self.data[idx, ...]
        if self.transform is not None:
            img = self.transform(img)
        return img, img

    def __len__(self):
        return self.data.shape[0]


class PulseDataset(Dataset):
    def __init__(self, root_dir, transform=None, split=1.):
        if Path(root_dir).is_dir():
            clutter_files = glob(f'{root_dir}/clutter_*.spec')
            self.data = np.concatenate([np.fromfile(c,
                                                    dtype=np.float32).reshape((-1, 6554, 1)) for c in clutter_files])
        else:
            self.data = np.fromfile(root_dir, dtype=np.float32).reshape((-1, 6554, 1))
        # Do split
        if split < 1:
            self.data = self.data[np.random.choice(np.arange(self.data.shape[0]), int(self.data.shape[0] * split)), ...]
        self.transform = transform

    def __getitem__(self, idx):
        img = self.data[idx, ...]
        if self.transform is not None:
            img = self.transform(img)
        return img, img

    def __len__(self):
        return self.data.shape[0]


class WaveDataset(Dataset):
    def __init__(self, root_dir, vae_model, transform=None, spec_transform=None, split=.1, single_example=False,
                 device='cpu', mu=0., var=1.):
        assert Path(root_dir).is_dir()

        clutter_cov_files = glob(f'{root_dir}/clutter_*.cov')
        clutter_spec_files = glob(f'{root_dir}/clutter_*.spec')
        target_cov_files = glob(f'{root_dir}/targets.cov')
        target_spec_files = glob(f'{root_dir}/targets.spec')
        self.ccdata = np.concatenate([np.fromfile(c, dtype=np.float32).reshape(
            (-1, 32, 32, 2)) for c in clutter_cov_files]).swapaxes(1, 3) / var
        self.ccdata = vae_model(torch.tensor(self.ccdata))[2]
        self.tcdata = np.concatenate([np.fromfile(c, dtype=np.float32).reshape(
            (-1, 32, 32, 2)) for c in target_cov_files]).swapaxes(1, 3) / var
        self.csdata = np.concatenate([np.fromfile(c, dtype=np.float32).reshape(
            (-1, 6554, 2)) for c in clutter_spec_files])
        self.tsdata = np.concatenate([np.fromfile(c, dtype=np.float32).reshape(
            (-1, 6554, 2)) for c in target_spec_files])

        # Do split
        if split < 1:
            rch = np.random.choice(np.arange(self.csdata.shape[0]), int(self.csdata.shape[0] * split))
            self.ccdata = self.ccdata[rch, ...]
            self.csdata = self.csdata[rch, ...]
        if single_example:
            self.ccdata[1:, ...] = self.ccdata[0, ...]
            self.tcdata[1:, ...] = self.tcdata[0, ...]
            self.csdata[1:, ...] = self.csdata[0, ...]
            self.tsdata[1:, ...] = self.tsdata[0, ...]

        self.spec_sz = self.tcdata.shape[0]

        # Run through the VAE model
        vae_model.to(device)
        self.ccdata = vae_model(self.ccdata)
        self.tcdata = vae_model(self.tcdata)

        if spec_transform is not None:
            self.spec_transform = spec_transform
        else:
            self.spec_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

        self.transform = transform

    def __getitem__(self, idx):
        if self.transform is not None:
            ccd = self.transform(self.ccdata[idx, ...])
            tcd = self.transform(self.tcdata[idx, ...])
        if self.spec_sz < self.csdata.shape[0]:
            csd = self.spec_transform(self.csdata[idx % self.spec_sz, ...])
            tsd = self.spec_transform(self.tsdata[idx % self.spec_sz, ...])
        else:
            csd = self.spec_transform(self.csdata[idx, ...])
            tsd = self.spec_transform(self.tsdata[idx, ...])
        return ccd, tcd, csd, tsd

    def __len__(self):
        return self.csdata.shape[0]


class CovDataModule(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            patch_size: Union[int, Sequence[int]] = (256, 256),
            num_workers: int = 0,
            pin_memory: bool = False,
            train_split: float = .7,
            val_split: float = .3,
            single_example: bool = False,
            mu: float = 0.,
            var: float = 1.,
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
        self.train_split = train_split
        self.val_split = val_split
        self.single_example = single_example
        self.mu = mu
        self.var = var

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = CovarianceDataset(self.data_dir, split=self.train_split,
                                               single_example=self.single_example, mu=self.mu, var=self.var)

        self.val_dataset = CovarianceDataset(self.data_dir, split=self.val_split,
                                             single_example=self.single_example, mu=self.mu, var=self.var)

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


class WaveDataModule(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            vae_model: BaseVAE,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            num_workers: int = 0,
            pin_memory: bool = False,
            train_split: float = .7,
            val_split: float = .3,
            single_example: bool = False,
            mu: float = 0.,
            var: float = 1.,
            **kwargs,
    ):
        super().__init__()

        self.val_dataset = None
        self.vae_model = vae_model
        self.train_dataset = None
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_split = train_split
        self.val_split = val_split
        self.single_example = single_example
        self.mu = mu
        self.var = var

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = WaveDataset(self.data_dir, self.vae_model, split=self.train_split,
                                         single_example=self.single_example, mu=self.mu, var=self.var)

        self.val_dataset = WaveDataset(self.data_dir, self.vae_model, split=self.val_split,
                                       single_example=self.single_example, mu=self.mu, var=self.var)

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
