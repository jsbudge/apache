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

    def __init__(self, root_dir, transform=None, split=1., single_example=False, mu=0., var=1., noise_level=0.):
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

        self.noise_level = noise_level

    def __getitem__(self, idx):
        img = self.data[idx, ...]
        if self.transform is not None:
            img = self.transform(img)
        if self.noise_level > 0:
            return img + torch.randn_like(img) * self.noise_level, img
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
        ccdata = np.concatenate([np.fromfile(c, dtype=np.float32).reshape(
            (-1, 32, 32, 2)) for c in clutter_cov_files]).swapaxes(1, 3) / var
        tcdata = np.concatenate([np.fromfile(c, dtype=np.float32).reshape(
            (-1, 32, 32, 2)) for c in target_cov_files]).swapaxes(1, 3) / var
        csdata = np.concatenate([np.fromfile(c, dtype=np.float32).reshape(
            (-1, 6554, 2)) for c in clutter_spec_files])
        tsdata = np.concatenate([np.fromfile(c, dtype=np.float32).reshape(
            (-1, 6554, 2)) for c in target_spec_files])

        # Do split
        if split < 1:
            rch = np.random.choice(np.arange(csdata.shape[0]), int(csdata.shape[0] * split))
            ccdata = ccdata[rch, ...]
            csdata = csdata[rch, ...]
        if single_example:
            ccdata[1:, ...] = ccdata[0, ...]
            tcdata[1:, ...] = tcdata[0, ...]
            csdata[1:, ...] = csdata[0, ...]
            tsdata[1:, ...] = tsdata[0, ...]

        self.spec_sz = tcdata.shape[0]

        # Run through the VAE model
        vae_model.to(device)

        self.ccdata = torch.vstack([vae_model.encode(torch.tensor(ccdata[i:i+64]).to(device)).detach().cpu() for i in range(0, ccdata.shape[0], 64)])
        self.tcdata = torch.vstack([vae_model.encode(torch.tensor(tcdata[i:i+64]).to(device)).detach().cpu() for i in range(0, tcdata.shape[0], 64)])
        self.csdata = torch.tensor(csdata)
        self.tsdata = torch.tensor(tsdata)

        del ccdata
        del tcdata

        self.spec_transform = spec_transform
        self.transform = transform

    def __getitem__(self, idx):
        ccd = self.ccdata[idx, ...]
        csd = self.csdata[idx, ...]
        if self.spec_sz < self.csdata.shape[0]:
            tsd = self.tsdata[idx % self.spec_sz, ...]
            tcd = self.tcdata[idx % self.spec_sz, ...]
        else:
            tsd = self.tsdata[idx, ...]
            tcd = self.tcdata[idx, ...]
        if self.transform is not None:
            ccd = self.transform(ccd)
            tcd = self.transform(tcd)
        if self.spec_transform is not None:
            csd = self.spec_transform(csd)
            tsd = self.spec_transform(tsd)

        return ccd, tcd, csd, tsd

    def __len__(self):
        return self.csdata.shape[0]


class CovDataModule(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            num_workers: int = 0,
            pin_memory: bool = False,
            train_split: float = .7,
            val_split: float = .3,
            single_example: bool = False,
            mu: float = 0.,
            var: float = 1.,
            noise_level: float = 0.,
            **kwargs,
    ):
        super().__init__()

        self.val_dataset = None
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
        self.noise_level = noise_level

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = CovarianceDataset(self.data_dir, split=self.train_split,
                                               single_example=self.single_example, mu=self.mu, var=self.var,
                                               noise_level=self.noise_level)

        self.val_dataset = CovarianceDataset(self.data_dir, split=self.val_split,
                                             single_example=self.single_example, mu=self.mu, var=self.var,
                                             noise_level=self.noise_level)

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
            device: str = 'cpu',
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
        self.device = device

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = WaveDataset(self.data_dir, self.vae_model, split=self.train_split,
                                         single_example=self.single_example, mu=self.mu, var=self.var,
                                         device=self.device)

        self.val_dataset = WaveDataset(self.data_dir, self.vae_model, split=self.val_split,
                                       single_example=self.single_example, mu=self.mu, var=self.var, device=self.device)

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
