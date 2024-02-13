from glob import glob
from typing import List, Optional, Sequence, Union

import yaml
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
from torchdata.datapipes.iter import FileLister, FileOpener
from torchvision import transforms
from pathlib import Path
import numpy as np
from multiprocessing import cpu_count

from models import BaseVAE


def transformCovData(vae_model, device, root_dir, mu, var):
    assert Path(root_dir).is_dir()

    clutter_cov_files = glob(f'{root_dir}/clutter_*.cov')
    target_cov_files = glob(f'{root_dir}/targets.cov')
    ccdata = np.fromfile(clutter_cov_files[0], dtype=np.float32).reshape(
        (-1, 32, 32, 2)).swapaxes(1, 3)
    tcdata = np.fromfile(target_cov_files[0], dtype=np.float32).reshape(
        (-1, 32, 32, 2)).swapaxes(1, 3)
    ccdata = (ccdata - mu) / var
    tcdata = (tcdata - mu) / var
    # Run through the VAE model
    vae_model.to(device)

    with open(f'{root_dir}/clutter.enc', 'ab') as f:
        for i in range(0, ccdata.shape[0], 64):
            ed = vae_model.encode(torch.tensor(ccdata[i:i + 64]).to(device)).detach().cpu().data.numpy().astype(np.float32)
            ed.tofile(f)
    with open(f'{root_dir}/target.enc', 'ab') as f:
        for i in range(0, tcdata.shape[0], 64):
            ed = vae_model.encode(torch.tensor(tcdata[i:i + 64]).to(device)).detach().cpu().data.numpy().astype(np.float32)
            ed.tofile(f)


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


class RCSDataset(Dataset):
    def __init__(self, datapath='./data', split=None):
        # Load in data
        optical_data = np.fromfile('/home/jeff/repo/apache/data/ge_img.dat',
                                   dtype=np.float32).reshape((-1, 256, 256, 3)).swapaxes(1, 3)
        sar_data = 10 * np.log(np.fromfile('/home/jeff/repo/apache/data/sar_img.dat',
                               dtype=np.float32).reshape((-1, 1, 256, 256)))
        param_data = np.fromfile('/home/jeff/repo/apache/data/params.dat', dtype=np.float32).reshape((-1, 7))
        self.optical_data = torch.tensor(optical_data)
        self.sar_data = torch.tensor(sar_data / sar_data.min())
        self.param_data = torch.tensor(param_data)

        if split:
            idxes = np.random.choice(np.arange(self.optical_data.shape[0]), split)
            self.optical_data = self.optical_data[idxes, ...]
            self.sar_data = self.sar_data[idxes, ...]
            self.param_data = self.param_data[idxes, ...]

    def __getitem__(self, idx):
        return self.optical_data[idx, :], self.sar_data[idx, :], self.param_data[idx, :]

    def __len__(self):
        return self.optical_data.shape[0]


class OldWaveDataset(Dataset):
    def __init__(self, root_dir: str, vae_model, fft_sz: int, transform=None, spec_transform=None, split: int = 32,
                 single_example: bool = False, device: str = 'cpu', min_pulse_length: int = 1,
                 max_pulse_length: int = 2, mu: float = 0., var: float = 1.):
        assert Path(root_dir).is_dir()

        clutter_cov_files = glob(f'{root_dir}/clutter_*.cov')
        clutter_spec_files = glob(f'{root_dir}/clutter_*.spec')
        target_cov_files = glob(f'{root_dir}/targets.cov')
        target_spec_files = glob(f'{root_dir}/targets.spec')
        ccdata = np.fromfile(clutter_cov_files[0], dtype=np.float32).reshape(
            (-1, 32, 32, 2))
        tcdata = np.fromfile(target_cov_files[0], dtype=np.float32).reshape(
            (-1, 32, 32, 2))
        csdata = np.fromfile(clutter_spec_files[0], dtype=np.float32).reshape(
            (-1, fft_sz, 2))
        tsdata = np.fromfile(target_spec_files[0], dtype=np.float32).reshape(
            (-1, fft_sz, 2))
        per_file = int(split // len(clutter_cov_files))
        for i in range(1, len(clutter_cov_files)):
            tmp_cdata = np.fromfile(clutter_cov_files[i], dtype=np.float32).reshape(
                (-1, 32, 32, 2))
            ccs = np.random.choice(np.arange(tmp_cdata.shape[0]), per_file)
            ccdata = np.concatenate([ccdata, tmp_cdata[ccs]])
            csdata = np.concatenate([csdata, np.fromfile(clutter_spec_files[i], dtype=np.float32).reshape(
                (-1, fft_sz, 2))[ccs]])

        # Scale the data to get unit variance (this preserves phase, not magnitude)
        ccdata = ccdata[:split, ...]
        ccdata = (ccdata - mu) / var
        tcdata = tcdata[:split, ...]
        tcdata = (tcdata - mu) / var

        # Spectrum doesn't need normalization as that occurs during training
        csdata = csdata[:split, ...]
        tsdata = tsdata[:split, ...]
        ccdata = ccdata.swapaxes(1, 3)
        tcdata = tcdata.swapaxes(1, 3)
        if single_example:
            ccdata[1:, ...] = ccdata[0, ...]
            tcdata[1:, ...] = tcdata[0, ...]
            csdata[1:, ...] = csdata[0, ...]
            tsdata[1:, ...] = tsdata[0, ...]

        self.spec_sz = tcdata.shape[0]

        # Run through the VAE model
        vae_model.to(device)

        self.ccdata = torch.vstack([vae_model.encode(torch.tensor(ccdata[i:i + 64]).to(device)).detach().cpu()
                                    for i in range(0, ccdata.shape[0], 64)])
        self.tcdata = torch.vstack([vae_model.encode(torch.tensor(tcdata[i:i + 64]).to(device)).detach().cpu()
                                    for i in range(0, tcdata.shape[0], 64)])
        self.csdata = torch.tensor(csdata)
        self.tsdata = torch.tensor(tsdata)

        del ccdata
        del tcdata

        self.spec_transform = spec_transform
        self.transform = transform
        self.min_pulse_length = min_pulse_length
        self.max_pulse_length = max_pulse_length

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

        return ccd, tcd, csd, tsd, np.random.randint(self.min_pulse_length, self.max_pulse_length)

    def __len__(self):
        return self.csdata.shape[0]


class WaveDataset(Dataset):
    def __init__(self, root_dir: str, latent_dim: int = 50, fft_sz: int = 4096, transform=None, spec_transform=None,
                 split: int = 32,
                 single_example: bool = False, device: str = 'cpu', min_pulse_length: int = 1,
                 max_pulse_length: int = 2):
        assert Path(root_dir).is_dir()

        clutter_spec_files = glob(f'{root_dir}/clutter_*.spec')
        target_spec_files = glob(f'{root_dir}/targets.spec')
        ccdata = np.fromfile(f'{root_dir}/clutter.enc', dtype=np.float32).reshape(
            (-1, latent_dim))
        tcdata = np.fromfile(f'{root_dir}/target.enc', dtype=np.float32).reshape(
            (-1, latent_dim))
        csdata = np.fromfile(clutter_spec_files[0], dtype=np.float32).reshape(
            (-1, fft_sz, 2))
        tsdata = np.fromfile(target_spec_files[0], dtype=np.float32).reshape(
            (-1, fft_sz, 2))
        per_file = int(split // len(clutter_spec_files))
        curr_i = csdata.shape[0]
        total_idx = np.arange(csdata.shape[0])
        for i in range(1, len(clutter_spec_files)):
            cs_tmp = np.fromfile(clutter_spec_files[i], dtype=np.float32).reshape(
                (-1, fft_sz, 2))
            ccs = np.random.choice(cs_tmp.shape[0], per_file)
            total_idx = np.concatenate((total_idx, ccs + curr_i))
            curr_i += cs_tmp.shape[0]
            csdata = np.concatenate([csdata, cs_tmp[ccs]])

        ccdata = ccdata[:split, ...]
        tcdata = tcdata[:split, ...]

        # Spectrum doesn't need normalization as that occurs during training
        csdata = csdata[:split, ...]
        tsdata = tsdata[:split, ...]
        if single_example:
            ccdata[1:, ...] = ccdata[0, ...]
            tcdata[1:, ...] = tcdata[0, ...]
            csdata[1:, ...] = csdata[0, ...]
            tsdata[1:, ...] = tsdata[0, ...]

        self.spec_sz = tcdata.shape[0]

        self.ccdata = torch.tensor(ccdata)
        self.tcdata = torch.tensor(tcdata)
        self.csdata = torch.tensor(csdata)
        self.tsdata = torch.tensor(tsdata)

        self.spec_transform = spec_transform
        self.transform = transform
        self.min_pulse_length = min_pulse_length
        self.max_pulse_length = max_pulse_length

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

        return ccd, tcd, csd, tsd, np.random.randint(self.min_pulse_length, self.max_pulse_length)

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
            latent_dim: int = 50,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            num_workers: int = 0,
            pin_memory: bool = False,
            train_split: int = 1024,
            val_split: int = 32,
            single_example: bool = False,
            mu: float = 0.,
            var: float = 1.,
            device: str = 'cpu',
            fft_sz: int = 4096,
            min_pulse_length: int = 1,
            max_pulse_length: int = 2,
            **kwargs,
    ):
        super().__init__()

        self.val_dataset = None
        self.latent_dim = latent_dim
        self.train_dataset = None
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = cpu_count() // 2
        self.pin_memory = pin_memory
        self.train_split = train_split
        self.val_split = val_split
        self.single_example = single_example
        self.min_pulse_length = min_pulse_length
        self.max_pulse_length = max_pulse_length
        self.device = device
        self.fft_sz = fft_sz
        self.mu = mu
        self.var = var

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = WaveDataset(self.data_dir, self.latent_dim, self.fft_sz, split=self.train_split,
                                         single_example=self.single_example, min_pulse_length=self.min_pulse_length,
                                         max_pulse_length=self.max_pulse_length,
                                         device=self.device)

        self.val_dataset = WaveDataset(self.data_dir, self.latent_dim, self.fft_sz, split=self.val_split,
                                       single_example=self.single_example, min_pulse_length=self.min_pulse_length,
                                       max_pulse_length=self.max_pulse_length, device=self.device)

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


class RCSModule(LightningDataModule):
    def __init__(
            self,
            dataset_size: int = 256,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            num_workers: int = 0,
            pin_memory: bool = False,
            train_split: int = 1024,
            val_split: int = 32,
            single_example: bool = False,
            device: str = 'cpu',
            **kwargs,
    ):
        super().__init__()

        self.val_dataset = None
        self.train_dataset = None
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = cpu_count() // 2
        self.pin_memory = pin_memory
        self.train_split = train_split
        self.val_split = val_split
        self.single_example = single_example
        self.device = device
        self.dataset_size = dataset_size

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = RCSDataset(split=self.train_split)

        self.val_dataset = RCSDataset(split=self.val_split)

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
    with open('./vae_config.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    data = WaveDataset(**config['dataset_params'])
    data.setup()
